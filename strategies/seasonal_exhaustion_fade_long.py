"""
strategies/seasonal_exhaustion_fade_long.py
============================================
Long-only variant of SeasonalExhaustionFade.

Key differences from the bidirectional version
-----------------------------------------------
* Long entries only — no short fades, no borrow costs.
* Season gate relaxed:
    Bullish  (score >  +threshold) → full-strength long entry allowed
    Neutral  (|score| <= threshold) → long entry allowed at 50% strength
    Bearish  (score <  -threshold) → skip (no buying into a weak season)
  The original strategy required a bullish regime for longs AND a bearish
  regime for shorts.  Removing shorts frees us to trade in neutral months at
  reduced conviction, giving more opportunities without fighting the calendar.
* RSI oversold threshold raised to 35 (from 30) — with longs only we can
  afford slightly earlier entries; combined with the ATR expansion gate this
  keeps the quality bar high.
* Bollinger Band width kept at 2.5σ (same tightness as original).

Everything else — ADX hard gate, ATR expansion, TP at mid-band, SL at
1.5×ATR below entry low, 12-bar time stop, max 4 positions — is unchanged.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from strategies.exhaustion_fade import _wilder_adx, _wilder_rsi
from strategies.seasonal_exhaustion_fade import _wilder_atr
from config.settings import STRATEGY_PARAMS


class SeasonalExhaustionFadeLongStrategy(BaseStrategy):
    """
    Long-only seasonal exhaustion fade.

    Enters oversold reversals when:
      1. Seasonal score is NOT bearish (bullish OR neutral month).
      2. Price closes below the lower Bollinger Band (2.5σ).
      3. ATR is expanded beyond its 20-bar average (exhaustion confirmed).
      4. RSI(14) < 35 (crowd is oversold / too short).
      5. ADX(14) ≤ 30 (not in a strong trend — fade conditions only).

    Signal strength in bullish months = RSI extremity score.
    Signal strength in neutral months = 50% of above.
    """

    def __init__(self):
        super().__init__(
            "seasonal_exhaustion_fade_long",
            STRATEGY_PARAMS.get("seasonal_exhaustion_fade_long", {}),
        )
        p = self.params
        self.bb_window        = p.get("bb_window",        20)
        self.bb_std           = p.get("bb_std",           2.5)
        self.atr_period       = p.get("atr_period",       14)
        self.atr_avg_window   = p.get("atr_avg_window",   20)
        self.atr_multiple     = p.get("atr_multiple",     1.5)
        self.rsi_period       = p.get("rsi_period",       14)
        self.rsi_os           = p.get("rsi_os",           35)   # wider than 30 for long-only
        self.adx_period       = p.get("adx_period",       14)
        self.adx_threshold    = p.get("adx_threshold",    30)
        self.season_threshold = p.get("season_threshold", 0.02)
        self.jan_discount     = p.get("jan_discount",     0.30)
        self.neutral_discount = p.get("neutral_discount", 0.50)  # scale for neutral months
        self.max_positions    = p.get("max_positions",    4)
        self.time_stop_bars   = p.get("time_stop_bars",   12)

    # ── Seasonality ───────────────────────────────────────────────────────────

    def _build_season_cache(
        self, returns: pd.DataFrame, dates: pd.DatetimeIndex,
    ) -> dict:
        """
        Pre-compute seasonality score for each (year, month) in dates.
        Score = mean monthly total return across 1-, 2-, 3-year same-month lags.
        January score is discounted by jan_discount.
        """
        cache: dict = {}
        for dt in dates:
            key = (dt.year, dt.month)
            if key in cache:
                continue

            yr, mo      = dt.year, dt.month
            lag_returns = []
            for lag in (1, 2, 3):
                mask = (
                    (returns.index.year  == yr - lag) &
                    (returns.index.month == mo)
                )
                r_lag = returns.loc[mask]
                if not r_lag.empty:
                    lag_returns.append(r_lag.sum())

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

        # ── Indicators ────────────────────────────────────────────────────────
        bb_mid   = close.rolling(self.bb_window).mean()
        bb_std_r = close.rolling(self.bb_window).std()
        bb_lower = bb_mid - self.bb_std * bb_std_r

        atr      = _wilder_atr(high, low, close, self.atr_period)
        atr_avg  = atr.rolling(self.atr_avg_window).mean()
        adx      = _wilder_adx(high, low, close, self.adx_period)
        rsi      = _wilder_rsi(close, self.rsi_period)

        season_cache = self._build_season_cache(returns, dates)

        # ── Per-ticker position state ─────────────────────────────────────────
        pos: dict = {
            t: {"active": False, "entry_idx": 0, "sl": 0.0, "strength": 0.0}
            for t in tickers
        }
        n_open  = 0
        signals = pd.DataFrame(0.0, index=dates, columns=close.columns)

        for i, dt in enumerate(dates):
            season = season_cache.get(
                (dt.year, dt.month), pd.Series(0.0, index=tickers)
            )

            # ── Step 1: process exits ─────────────────────────────────────────
            for t in tickers:
                p = pos[t]
                if not p["active"]:
                    continue

                px      = close.at[dt, t]
                bars_in = i - p["entry_idx"]
                mid     = bb_mid.at[dt, t]

                tp_hit   = px >= mid
                sl_hit   = px <= p["sl"]
                time_hit = bars_in >= self.time_stop_bars

                if tp_hit or sl_hit or time_hit:
                    p["active"] = False
                    n_open     -= 1
                else:
                    signals.at[dt, t] = p["strength"]

            # ── Step 2: collect long entry candidates ─────────────────────────
            if n_open >= self.max_positions:
                continue

            candidates: list = []  # (rsi_extremity, ticker, strength, sl)
            for t in tickers:
                if pos[t]["active"]:
                    continue

                s_score = float(season.get(t, 0.0))

                # Skip if seasonality is explicitly bearish
                if s_score < -self.season_threshold:
                    continue

                atr_v = atr.at[dt, t]
                atr_a = atr_avg.at[dt, t]
                adx_v = adx.at[dt, t]
                rsi_v = rsi.at[dt, t]
                px    = close.at[dt, t]
                bbl   = bb_lower.at[dt, t]

                if any(np.isnan(v) for v in (atr_v, atr_a, adx_v, rsi_v, bbl)):
                    continue

                atr_ok = atr_v > self.atr_multiple * atr_a
                adx_ok = adx_v <= self.adx_threshold

                if not (atr_ok and adx_ok):
                    continue

                # Long entry: price below lower BB, RSI oversold
                if px < bbl and rsi_v < self.rsi_os:
                    sl        = low.at[dt, t] - self.atr_multiple * atr_v
                    base_str  = np.clip((self.rsi_os - rsi_v) / self.rsi_os, 0.10, 1.0)
                    # Discount signal strength in neutral months
                    if abs(s_score) <= self.season_threshold:
                        base_str *= self.neutral_discount
                    extremity = self.rsi_os - rsi_v
                    candidates.append((extremity, t, base_str, sl))

            # Rank by RSI extremity (most oversold first), fill available slots
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, t, strength, sl in candidates[: self.max_positions - n_open]:
                pos[t] = {
                    "active": True, "entry_idx": i,
                    "sl": sl, "strength": strength,
                }
                signals.at[dt, t] = strength
                n_open += 1

        return signals
