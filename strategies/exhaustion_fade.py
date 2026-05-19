"""
strategies/exhaustion_fade.py
==============================
Exhaustion Fade Strategy — ported from crypto_algo and re-tuned for equities.

Core idea (unchanged from crypto)
----------------------------------
Fade overextended price moves when three gates fire simultaneously:
  G1 — Bollinger Band breach on the prior bar, close back inside today
  G2 — Crowding signal confirms the market is one-sided
  G3 — Volume climax (high-volume reversals are more durable than low-volume ones)
  G4 — ADX ranging filter: only fade in non-trending regimes

Key adaptations for stocks
---------------------------
  Funding rate → RSI(14)
      Crypto perpetuals publish a real-time crowding signal (funding rate).
      Stocks don't have one, so we use RSI as the closest liquid proxy:
        RSI > rsi_ob (70)  → longs overcrowded → fade short (-1)
        RSI < rsi_os (30)  → shorts overcrowded → fade long  (+1)
      This mirrors the funding-rate logic exactly: extreme RSI = crowded side.

  bb_std 2.5 → 2.0
      Equities are less volatile than crypto; 2.0σ bands produce a similar
      breach frequency without requiring outsized moves to trigger the signal.

  Volume climax gate
      An additional equity-specific filter. A BB breach on average volume is
      often sector rotation or drift; a breach on 1.5×+ average volume is a
      real blow-off top or capitulation worthy of fading.
      Fallback: if volume is not passed, uses intraday range expansion
      (high - low) as a structural proxy for climax behaviour.

  Long-only mode (LONG_SHORT=False)
      Only +1 (oversold fade long) signals are used by the optimizer in
      long-only mode. Short signals are still generated so this strategy
      works in long-short contexts too.

Signal strength
---------------
  strength = clip(0.4 + 0.3 * adx_factor + 0.3 * rsi_factor, 0.3, 1.0)
    adx_factor = (adx_threshold - ADX) / adx_threshold        [0→1 as ADX drops]
    rsi_factor = |RSI - 50| / (50 - rsi_os)                   [0→1 as RSI extremes]
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


# ── Indicators ─────────────────────────────────────────────────────────────────

def _wilder_adx(
    high:  pd.DataFrame,
    low:   pd.DataFrame,
    close: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """Wilder-smoothed ADX per column, matching the crypto_algo implementation."""
    ew = dict(alpha=1 / period, min_periods=period, adjust=False)
    out = {}
    for col in close.columns:
        h, l, c = high[col], low[col], close[col]

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                       axis=1).max(axis=1)

        up, dn  = h.diff(), -l.diff()
        dm_p    = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
        dm_m    = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)

        atr     = tr.ewm(**ew).mean()
        di_p    = 100 * dm_p.ewm(**ew).mean() / atr.clip(lower=1e-8)
        di_m    = 100 * dm_m.ewm(**ew).mean() / atr.clip(lower=1e-8)

        dx      = 100 * (di_p - di_m).abs() / (di_p + di_m).clip(lower=1e-8)
        out[col] = dx.ewm(**ew).mean()

    return pd.DataFrame(out)


def _wilder_rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder RSI per column."""
    ew   = dict(alpha=1 / period, min_periods=period, adjust=False)
    d    = close.diff()
    gain = d.clip(lower=0).ewm(**ew).mean()
    loss = (-d.clip(upper=0)).ewm(**ew).mean()
    return 100 - 100 / (1 + gain / loss.clip(lower=1e-8))


# ── Strategy ───────────────────────────────────────────────────────────────────

class ExhaustionFadeStrategy(BaseStrategy):
    """
    Fade blow-off tops and capitulation lows in ranging equities.

    Signal is strongest when:
      • ADX is well below the threshold (market firmly ranging)
      • RSI is deeply overbought / oversold (crowded positioning)
      • Volume spike confirms a real climax, not routine drift
    """

    def __init__(self, params: dict = None):
        defaults = STRATEGY_PARAMS.get("exhaustion_fade", {})
        super().__init__("exhaustion_fade", {**defaults, **(params or {})})

    def generate_signals(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        high:    pd.DataFrame = None,
        low:     pd.DataFrame = None,
        volume:  pd.DataFrame = None,
        **kwargs,
    ) -> pd.DataFrame:
        if high is None or low is None:
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

        p          = self.params
        bb_window  = p.get("bb_window",     20)
        bb_std     = p.get("bb_std",         2.0)
        adx_period = p.get("adx_period",    14)
        adx_thresh = p.get("adx_threshold", 30)
        rsi_period = p.get("rsi_period",    14)
        rsi_ob     = p.get("rsi_ob",        70)
        rsi_os     = p.get("rsi_os",        30)
        vol_mult   = p.get("vol_multiple",   1.5)
        vol_lb     = p.get("vol_lookback",  20)

        # ── G1: Bollinger Band breach + close-back-inside ──────────────────────
        ma       = close.rolling(bb_window).mean()
        band     = bb_std * close.rolling(bb_window).std()
        upper    = ma + band
        lower    = ma - band

        g1_short = (close.shift(1) > upper.shift(1)) & (close <= upper)  # above → inside
        g1_long  = (close.shift(1) < lower.shift(1)) & (close >= lower)  # below → inside

        # ── G2: RSI crowding gate ──────────────────────────────────────────────
        rsi      = _wilder_rsi(close, rsi_period)
        g2_short = rsi > rsi_ob
        g2_long  = rsi < rsi_os

        # RSI factor: how far past the threshold (0 at threshold, 1 at extreme)
        rsi_f_short = ((rsi - rsi_ob) / (100 - rsi_ob)).clip(0, 1)
        rsi_f_long  = ((rsi_os - rsi) / rsi_os).clip(0, 1)

        # ── G3: Volume climax ──────────────────────────────────────────────────
        if volume is not None and not volume.empty:
            g3 = volume > (vol_mult * volume.rolling(vol_lb).mean())
        else:
            # Range expansion as structural climax proxy
            rng = (high - low) / close.shift(1).clip(lower=1e-8)
            g3  = rng > (vol_mult * rng.rolling(vol_lb).mean())

        # ── G4: ADX ranging filter ─────────────────────────────────────────────
        adx = _wilder_adx(high, low, close, adx_period)
        g4  = adx < adx_thresh

        # ADX factor: further below threshold = stronger ranging conviction
        adx_factor = ((adx_thresh - adx.clip(upper=adx_thresh)) / adx_thresh).clip(0, 1)

        # ── Signal strength ────────────────────────────────────────────────────
        strength_short = (0.4 + 0.3 * adx_factor + 0.3 * rsi_f_short).clip(0.3, 1.0)
        strength_long  = (0.4 + 0.3 * adx_factor + 0.3 * rsi_f_long ).clip(0.3, 1.0)

        # ── Combine gates ──────────────────────────────────────────────────────
        short_mask = g1_short & g2_short & g3 & g4
        long_mask  = g1_long  & g2_long  & g3 & g4

        signals  =  strength_long.where(long_mask,  0.0)
        signals -= strength_short.where(short_mask, 0.0)

        return signals
