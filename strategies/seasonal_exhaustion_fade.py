"""
strategies/seasonal_exhaustion_fade.py
========================================
Three-layer seasonal exhaustion fade.

LAYER 1 — Seasonality Regime Filter
  Monthly score = mean(total return for same calendar month over 1-, 2-, 3-year lags).
  January Effect: score is discounted 30% in January to avoid overcounting the
  well-known January bias.
  Score >  +2% → Bullish regime → long fades only
  Score <  -2% → Bearish regime → short fades only
  Score  ±2%   → Neutral       → no trade (skip)
  Recalculated at the start of each calendar month.

LAYER 2 — Exhaustion Signal
  Bollinger Bands (20-period, 2.5σ) + ATR(14) expansion gate.
  Bullish trigger : close < lower_band  AND  ATR > 1.5 × 20-bar avg ATR
  Bearish trigger : close > upper_band  AND  ATR > 1.5 × 20-bar avg ATR
  Both conditions must hold simultaneously.

LAYER 3 — Positioning Confirmation
  RSI(14) as a crowding proxy (replaces funding rate used in crypto).
  Bullish fade : RSI < 30 (oversold — crowd is too short)
  Bearish fade : RSI > 70 (overbought — crowd is too long)

TREND GATE (hard override, checked before layers)
  ADX(14) > 30 → skip all fade signals for that ticker that day.

ENTRY
  All 3 layers + ADX gate → signal on day T → executed at open of T+1
  (T+1 shift enforced by BaseStrategy.run).
  Maximum 4 simultaneous positions.
  When > 4 signals fire: rank by RSI extremity, take top 4.

EXIT  (condition detected at close of bar T → executed at open of T+1)
  TP        : close reverts to (or crosses) middle Bollinger Band (20-SMA)
  SL        : close breaches 1.5×ATR beyond breach-candle high (short)
              or breach-candle low (long)
  Time stop : 12 bars after entry signal

TRANSACTION COST MODEL
  1. Commission     : 1.1 bps per leg (entry + exit = 2.2 bps round trip).
                      Configure via strategy params "commission_bps".
  2. Short borrow   : BORROW_COST_BPS dict (annualised).  60 bps/yr for
                      single stocks, 35 bps/yr for SPY / QQQ.
                      Call borrow_cost_adjustment() for the daily cost series.
  3. Market impact  : MI = 0.1 × σ₂₀ × sqrt(trade_vol / avg_daily_vol).
                      Approximated by global SLIPPAGE_BP in engine settings.
  4. Financing cost : Not applied — strategy is long-only by default and
                      no leverage is used.

POSITION SIZING
  Risk 1% of portfolio equity per trade.
  Signal strength ∝ RSI extremity (how far RSI is from the ob/os threshold).
  Use portfolio type "signal_weighted" for closest-to-spec allocation.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from strategies.exhaustion_fade import _wilder_adx, _wilder_rsi
from config.settings import STRATEGY_PARAMS


# ── ATR helper ─────────────────────────────────────────────────────────────────

def _wilder_atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int
) -> pd.DataFrame:
    """Wilder-smoothed ATR per column."""
    ew  = dict(alpha=1 / period, min_periods=period, adjust=False)
    out = {}
    for col in close.columns:
        h, l, c = high[col], low[col], close[col]
        tr = pd.concat(
            [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
        ).max(axis=1)
        out[col] = tr.ewm(**ew).mean()
    return pd.DataFrame(out)


# ── Strategy ───────────────────────────────────────────────────────────────────

class SeasonalExhaustionFadeStrategy(BaseStrategy):
    """
    Fade exhaustion moves only when the seasonal regime supports the direction.
    All three layers (seasonality + exhaustion + RSI crowding) plus ADX ≤ 30
    must simultaneously align before a signal fires.
    """

    # Annualised short borrow costs in bps (Interactive Brokers general-collateral rates).
    BORROW_COST_BPS: dict = {
        "AAPL": 60, "MSFT": 60, "GOOGL": 60, "AMZN": 60, "NVDA": 60,
        "META": 60, "JPM": 60,  "JNJ": 60,  "XOM": 60,  "UNH": 60,
        "SPY":  35, "QQQ":  35,
    }
    DEFAULT_BORROW_BPS: float = 60

    def __init__(self):
        super().__init__(
            "seasonal_exhaustion_fade",
            STRATEGY_PARAMS.get("seasonal_exhaustion_fade", {}),
        )
        p = self.params
        self.bb_window        = p.get("bb_window",        20)
        self.bb_std           = p.get("bb_std",           2.5)
        self.atr_period       = p.get("atr_period",       14)
        self.atr_avg_window   = p.get("atr_avg_window",   20)
        self.atr_multiple     = p.get("atr_multiple",     1.5)
        self.rsi_period       = p.get("rsi_period",       14)
        self.rsi_ob           = p.get("rsi_ob",           70)
        self.rsi_os           = p.get("rsi_os",           30)
        self.adx_period       = p.get("adx_period",       14)
        self.adx_threshold    = p.get("adx_threshold",    30)
        self.season_threshold = p.get("season_threshold", 0.02)   # 2% monthly
        self.jan_discount     = p.get("jan_discount",     0.30)   # 30% January reduction
        self.max_positions    = p.get("max_positions",    4)
        self.time_stop_bars   = p.get("time_stop_bars",   12)
        self.commission_bps   = p.get("commission_bps",   1.1)    # per leg
        self.risk_per_trade   = p.get("risk_per_trade",   0.01)   # 1% equity

    # ── Seasonality ───────────────────────────────────────────────────────────

    def _build_season_cache(
        self, returns: pd.DataFrame, dates: pd.DatetimeIndex,
    ) -> dict:
        """
        Pre-compute seasonality score for each (year, month) present in dates.

        Score = mean monthly total return across 1-, 2-, 3-year same-month lags.
        Monthly total return ≈ sum of daily returns within that month.
        January: score discounted by jan_discount (30%) to reduce January bias.
        Missing lags (insufficient history) are skipped rather than zero-filled.
        """
        cache: dict = {}
        for dt in dates:
            key = (dt.year, dt.month)
            if key in cache:
                continue

            yr, mo       = dt.year, dt.month
            lag_returns  = []
            for lag in (1, 2, 3):
                mask = (
                    (returns.index.year  == yr - lag) &
                    (returns.index.month == mo)
                )
                r_lag = returns.loc[mask]
                if not r_lag.empty:
                    lag_returns.append(r_lag.sum())   # total return for that month

            if not lag_returns:
                cache[key] = pd.Series(0.0, index=returns.columns)
                continue

            score = pd.concat(lag_returns, axis=1).mean(axis=1)
            if mo == 1:
                score = score * (1.0 - self.jan_discount)
            cache[key] = score

        return cache

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        high:    pd.DataFrame = None,
        low:     pd.DataFrame = None,
        volume:  pd.DataFrame = None,
        **_kwargs,
    ) -> pd.DataFrame:
        if high is None or low is None:
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

        tickers = close.columns.tolist()
        dates   = close.index

        # ── Indicators (vectorised across tickers) ─────────────────────────
        bb_mid   = close.rolling(self.bb_window).mean()
        bb_std_r = close.rolling(self.bb_window).std()
        bb_upper = bb_mid + self.bb_std * bb_std_r
        bb_lower = bb_mid - self.bb_std * bb_std_r

        atr      = _wilder_atr(high, low, close, self.atr_period)
        atr_avg  = atr.rolling(self.atr_avg_window).mean()
        adx      = _wilder_adx(high, low, close, self.adx_period)
        rsi      = _wilder_rsi(close, self.rsi_period)

        # ── Seasonality cache (one score per calendar month) ───────────────
        season_cache = self._build_season_cache(returns, dates)

        # ── Position state ─────────────────────────────────────────────────
        # Tracks per-ticker: whether we are in a trade, direction, entry bar
        # index, stop-loss price level, and signal strength.
        pos: dict = {
            t: {"active": False, "dir": 0, "entry_idx": 0, "sl": 0.0, "strength": 0.0}
            for t in tickers
        }
        n_open  = 0
        signals = pd.DataFrame(0.0, index=dates, columns=close.columns)

        for i, dt in enumerate(dates):
            season = season_cache.get(
                (dt.year, dt.month), pd.Series(0.0, index=tickers)
            )

            # ── Step 1: process exits ──────────────────────────────────────
            for t in tickers:
                p = pos[t]
                if not p["active"]:
                    continue

                px      = close.at[dt, t]
                bars_in = i - p["entry_idx"]
                mid     = bb_mid.at[dt, t]
                d       = p["dir"]

                tp_hit   = (d ==  1 and px >= mid) or (d == -1 and px <= mid)
                sl_hit   = (d ==  1 and px <= p["sl"]) or (d == -1 and px >= p["sl"])
                time_hit = bars_in >= self.time_stop_bars

                if tp_hit or sl_hit or time_hit:
                    p["active"] = False
                    p["dir"]    = 0
                    n_open     -= 1
                    # signal remains 0 for this bar (position closes at next open)
                else:
                    signals.at[dt, t] = d * p["strength"]

            # ── Step 2: collect entry candidates ───────────────────────────
            if n_open >= self.max_positions:
                continue

            candidates: list = []  # (rsi_extremity, ticker, direction, strength, sl)
            for t in tickers:
                if pos[t]["active"]:
                    continue

                s_score = float(season.get(t, 0.0))

                # Hard gate: neutral seasonality or trending → skip
                if abs(s_score) < self.season_threshold:
                    continue

                atr_v   = atr.at[dt, t]
                atr_a   = atr_avg.at[dt, t]
                adx_v   = adx.at[dt, t]
                rsi_v   = rsi.at[dt, t]
                px      = close.at[dt, t]
                bbl     = bb_lower.at[dt, t]
                bbu     = bb_upper.at[dt, t]

                if any(np.isnan(v) for v in (atr_v, atr_a, adx_v, rsi_v, bbl, bbu)):
                    continue

                atr_ok = atr_v > self.atr_multiple * atr_a
                adx_ok = adx_v <= self.adx_threshold

                if not (atr_ok and adx_ok):
                    continue

                # ── Long fade: bullish season, price < lower BB, RSI < 30 ──
                if s_score > self.season_threshold and px < bbl and rsi_v < self.rsi_os:
                    sl       = low.at[dt, t] - self.atr_multiple * atr_v
                    # Strength: how far RSI is below oversold threshold (0 at 30, 1 at 0)
                    strength = np.clip((self.rsi_os - rsi_v) / self.rsi_os, 0.10, 1.0)
                    extremity = self.rsi_os - rsi_v
                    candidates.append((extremity, t, 1, strength, sl))

                # ── Short fade: bearish season, price > upper BB, RSI > 70 ─
                elif s_score < -self.season_threshold and px > bbu and rsi_v > self.rsi_ob:
                    sl       = high.at[dt, t] + self.atr_multiple * atr_v
                    strength = np.clip((rsi_v - self.rsi_ob) / (100 - self.rsi_ob), 0.10, 1.0)
                    extremity = rsi_v - self.rsi_ob
                    candidates.append((extremity, t, -1, strength, sl))

            # Rank by RSI extremity (most crowded first), fill available slots
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, t, d, strength, sl in candidates[: self.max_positions - n_open]:
                pos[t] = {
                    "active": True, "dir": d, "entry_idx": i,
                    "sl": sl, "strength": strength,
                }
                signals.at[dt, t] = d * strength
                n_open += 1

        return signals

    # ── Borrow cost utility ────────────────────────────────────────────────────

    def borrow_cost_adjustment(
        self,
        signals:      pd.DataFrame,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Daily short borrow cost as a fraction of the position value.

        Returns a Series (index = signals.index) where each value is the
        total daily cost across all short positions.  Subtract this from
        portfolio returns to apply borrow costs.

        Example
        -------
        port_returns -= strat.borrow_cost_adjustment(signals)
        """
        cost = pd.Series(0.0, index=signals.index)
        for t in signals.columns:
            annual_bps  = self.BORROW_COST_BPS.get(t, self.DEFAULT_BORROW_BPS)
            daily_rate  = annual_bps / 10_000 / trading_days
            short_mask  = signals[t] < 0
            cost       += short_mask.astype(float) * daily_rate * signals[t].abs()
        return cost
