"""
strategies/macro_regime.py
==========================
Macro regime strategy — Alpaca data only, no external feeds.

All five indicators are derived exclusively from the same Alpaca OHLCV
universe already used by every other strategy, so the data source is
fully consistent ("apples to apples").

Indicators
----------
1. VIX proxy        — SPY 20-day realised vol × √252 × 100 (annualised %)
                      < 20  → risk-on (+1)   ≥ 30 → risk-off (−1)
2. Equity momentum  — SPY price vs 200-day MA
                      above → risk-on (+1)   below → risk-off (−1)
3. Market breadth   — fraction of universe tickers with close > 50-day MA
                      > 60% → risk-on (+1)   < 40% → risk-off (−1)
4. Vol regime       — 10-day SPY vol vs 30-day SPY vol (vol expansion = stress)
                      contracting → risk-on (+1)   expanding → risk-off (−1)
5. Cross-momentum   — average universe momentum score
                      (12-1 month return, same as MomentumStrategy but averaged)
                      positive → risk-on (+1)   negative → risk-off (−1)

Composite
---------
Raw composite = mean of all five indicators, clipped to [−1, +1].
Output signal = (composite + 1) / 2, mapped to [0, 1]:
  −1 (full risk-off) → 0   (go to cash)
   0 (neutral)       → 0.5 (half-size position)
  +1 (full risk-on)  → 1.0 (full position)

Applied uniformly to every ticker so the optimizer sees equal expected
returns and allocates to the minimum-variance portfolio at the scaled
size.  Lower-risk modes reduce exposure rather than going to 0%.
"""

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class MacroRegimeStrategy(BaseStrategy):
    """
    Uniform macro regime signal derived entirely from Alpaca universe data.
    All tickers receive the same composite score each day.
    """

    def __init__(self):
        super().__init__("macro_regime", STRATEGY_PARAMS.get("macro_regime", {}))
        p = self.params
        self.vix_proxy_window  = p.get("vix_proxy_window",  20)   # SPY vol window
        self.vix_risk_on       = p.get("vix_risk_on",       20)   # annualised % threshold
        self.vix_risk_off      = p.get("vix_risk_off",      30)
        self.ma_long           = p.get("ma_long",           200)  # SPY trend MA
        self.breadth_ma        = p.get("breadth_ma",        50)   # breadth MA
        self.breadth_on        = p.get("breadth_on",        0.60) # > 60% → risk-on
        self.breadth_off       = p.get("breadth_off",       0.40) # < 40% → risk-off
        self.vol_short         = p.get("vol_short",         10)   # short vol window
        self.vol_long          = p.get("vol_long",          30)   # long vol window
        self.mom_long          = p.get("mom_long",          252)  # 12-month lookback
        self.mom_skip          = p.get("mom_skip",          21)   # skip last month

    # ── Individual indicators ─────────────────────────────────────────────────

    def _vix_proxy(self, spy_ret: pd.Series) -> pd.Series:
        """SPY realised vol scaled to approximate VIX units."""
        vol = spy_ret.rolling(self.vix_proxy_window).std() * np.sqrt(252) * 100
        sig = pd.Series(0.0, index=spy_ret.index)
        sig[vol < self.vix_risk_on]  =  1.0
        sig[vol >= self.vix_risk_off] = -1.0
        return sig.rename("vix_proxy")

    def _equity_momentum(self, spy: pd.Series) -> pd.Series:
        """SPY above/below 200-day MA."""
        ma  = spy.rolling(self.ma_long, min_periods=self.ma_long // 2).mean()
        sig = np.sign(spy - ma).rename("equity_mom")
        return sig

    def _market_breadth(self, close: pd.DataFrame) -> pd.Series:
        """Fraction of universe tickers trading above their 50-day MA."""
        ma      = close.rolling(self.breadth_ma, min_periods=self.breadth_ma // 2).mean()
        above   = (close > ma).astype(float)
        breadth = above.mean(axis=1)
        sig     = pd.Series(0.0, index=close.index)
        sig[breadth >= self.breadth_on]  =  1.0
        sig[breadth <= self.breadth_off] = -1.0
        return sig.rename("breadth")

    def _vol_regime(self, spy_ret: pd.Series) -> pd.Series:
        """Short-term vol expanding above long-term vol → stress."""
        v_short = spy_ret.rolling(self.vol_short).std()
        v_long  = spy_ret.rolling(self.vol_long).std()
        sig     = np.sign(v_long - v_short).rename("vol_regime")  # +1 contracting
        return sig

    def _cross_momentum(self, close: pd.DataFrame) -> pd.Series:
        """Average 12-1 month momentum across universe (standard formula)."""
        # Return from 12 months ago to 1 month ago — skips reversal-prone last month
        score = close.shift(self.mom_skip) / close.shift(self.mom_long) - 1
        avg   = score.mean(axis=1)
        sig   = np.sign(avg).rename("cross_mom")
        return sig

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        **_kwargs,
    ) -> pd.DataFrame:
        if "SPY" not in close.columns:
            logger.warning("MacroRegimeStrategy: SPY not in universe — returning flat")
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

        spy     = close["SPY"]
        spy_ret = returns["SPY"]

        components = [
            self._vix_proxy(spy_ret),
            self._equity_momentum(spy),
            self._market_breadth(close),
            self._vol_regime(spy_ret),
            self._cross_momentum(close),
        ]

        composite = (
            pd.concat(components, axis=1)
            .mean(axis=1)
            .clip(-1.0, 1.0)
            .reindex(close.index)
            .fillna(0.0)
        )

        # Map composite [-1, +1] → signal [0, 1] for long-only portfolios.
        # Full risk-off (−1) → 0 (cash), neutral (0) → 0.5, full risk-on (+1) → 1.
        # This prevents the optimizer from receiving all-negative μ and allocating 0%
        # during risk-off periods, which would make the strategy appear to do nothing.
        scale = (composite + 1.0) / 2.0

        risk_on_frac = (composite > 0).mean()
        logger.info(
            f"MacroRegimeStrategy: composite [{composite.min():.2f}, {composite.max():.2f}] "
            f"| risk-on {risk_on_frac:.0%} of days | scale [{scale.min():.2f}, {scale.max():.2f}]"
        )

        signals = pd.DataFrame(
            np.tile(scale.values[:, None], (1, len(close.columns))),
            index=close.index,
            columns=close.columns,
        )
        return signals
