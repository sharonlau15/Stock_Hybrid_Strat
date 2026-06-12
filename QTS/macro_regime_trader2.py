"""
macro_regime_trader.py  (v2 — standalone)
==========================================
Macro Regime-only trading machine. Zero imports from the rest of the project.
All config, strategy logic, portfolio construction, and execution are in this file.

Changes from v1
---------------
- Fully standalone: no imports from config/, strategies/, portfolio/, data/
- SIGNAL_UNIVERSE: 22 independent stocks across all 11 GICS sectors (no ETFs)
  so breadth/cross-momentum are genuine cross-sectional signals, not an
  echo chamber of 10 tech-heavy names + 2 indices that mechanically track them
- Portfolio: risk parity (inverse vol) — no matrix inversion, stable weights
  max-Sharpe on a uniform-signal, highly-correlated 12-name universe was
  latching onto covariance noise and producing unstable concentrated bets
- No stop-loss / no trailing stop — macro composite handles de-risking
  position-level stops fought the strategy's own signal (double de-risking)
  kept: 15% hard stop as a circuit breaker for catastrophic single-stock events
- Dead-code fix: backtest and live now use the same risk_parity_weights()
  v1 computed MaxSharpe weights then immediately overwrote with _signal_to_weights
- MAX_POSITION_SIZE: 0.15 (down from 0.20) for better concentration control
- NEW: compute_composite refactored into compute_components() (returns the
  5 raw {-1,0,+1} signals as a DataFrame) + compute_composite() (their mean).
  No behaviour change to backtest/live — composite output is identical.
- NEW: --mode analyze — decomposes the 5-indicator composite to check for
  redundant components (pairwise correlation), how often each component
  agrees with the overall composite sign, each component's standalone
  Sharpe if it alone drove the risk-parity sizing, and each component's
  correlation with next-day SPY return.

Usage
-----
  python macro_regime_trader2.py --mode backtest
  python macro_regime_trader2.py --mode analyze
  python macro_regime_trader2.py --mode live
  python macro_regime_trader2.py --mode full --run-now   # backtest → live if Sharpe ≥ 0.5
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).resolve().parent

for _env in [
    ROOT_DIR / "Alpaca.env",
    ROOT_DIR / ".env",
    ROOT_DIR.parent / "Alpaca.env",   # env file lives one level up
    ROOT_DIR.parent / ".env",
]:
    if _env.exists():
        load_dotenv(_env)
        break
    
# Dedicated MR paper-trading Alpaca account
API_KEY    = os.getenv("ALPACA_PAPER_API_KEY_MR2")    
API_SECRET = os.getenv("ALPACA_PAPER_API_SECRET_MR2")
PAPER      = True   # flip to False only for real money

# Paths — own cache directory so it doesn't collide with the main engine
DATA_DIR   = ROOT_DIR / "data" / "mr_cache"
LOG_DIR    = ROOT_DIR / "logs"
RESULT_DIR = ROOT_DIR / "results"
STATE_FILE = RESULT_DIR / "macro_live_state.json"
LOG_FILE   = LOG_DIR   / "macro_engine.log"

for _d in [DATA_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universes ─────────────────────────────────────────────────────────────────

# Tickers we actually trade (12 liquid names including ETFs as hedges)
TRADE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "JPM",  "JNJ",   "XOM",  "UNH",
    "SPY",  "QQQ",
]

# Broader cross-section used ONLY for computing breadth and cross-momentum.
# No ETF indices — we want independent stock observations across all sectors.
# With 22 names covering all 11 GICS sectors the breadth signal is genuine;
# using the 10-12 name trading universe (half of which are correlated tech)
# produces an echo-chamber reading that isn't real breadth.
SIGNAL_UNIVERSE = [
    # Technology (6)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    # Financials (3)
    "JPM",  "BAC",  "GS",
    # Healthcare (3)
    "JNJ",  "UNH",  "PFE",
    # Energy (2)
    "XOM",  "CVX",
    # Consumer Discretionary (2)
    "HD",   "MCD",
    # Consumer Staples (2)
    "WMT",  "PG",
    # Industrials (2)
    "CAT",  "HON",
    # Utilities / Materials (2)
    "NEE",  "LIN",
]

MARKET_PROXY = "SPY"   # used for VIX proxy, trend MA, vol regime

# All symbols to fetch (union of trade + signal, deduplicated)
ALL_SYMBOLS = sorted(set(TRADE_UNIVERSE) | set(SIGNAL_UNIVERSE))

# ── Data ─────────────────────────────────────────────────────────────────────
BACKTEST_START     = "2022-01-01"
SUBPERIOD_SPLIT    = "2023-01-01"   # 2022 = bear, 2023+ = bull
CACHE_EXPIRY_HOURS = 6

# ── Macro signal parameters ───────────────────────────────────────────────────
VIX_PROXY_WINDOW = 20    # SPY rolling vol window (days)
VIX_RISK_ON      = 20    # annualised vol % < this → risk-on  (+1)
VIX_RISK_OFF     = 30    # annualised vol % ≥ this → risk-off (−1)
MA_LONG          = 200   # SPY trend MA period
BREADTH_MA       = 50    # per-stock MA for breadth calc
BREADTH_ON       = 0.60  # > 60% of SIGNAL_UNIVERSE above MA → risk-on
BREADTH_OFF      = 0.40  # < 40% → risk-off
VOL_SHORT        = 10    # short vol window (stress detection)
VOL_LONG         = 30    # long vol window
MOM_LONG         = 252   # 12-month momentum lookback
MOM_SKIP         = 21    # skip most-recent month (reversal avoidance)

# ── Portfolio ─────────────────────────────────────────────────────────────────
TRADING_DAYS       = 252
RISK_LOOKBACK_DAYS = 126    # 6-month rolling vol estimate
VOL_FLOOR          = 0.03   # 3% annualised vol floor (prevents extreme weights)
MAX_POSITION_SIZE  = 0.15   # 15% single-name cap (down from 20%)
MAX_LIVE_POSITIONS = 8

# ── Execution ─────────────────────────────────────────────────────────────────
PORTFOLIO_USD       = 100_000   # matches MR paper account starting equity
MIN_ORDER_USD       = 1.0
REBALANCE_THRESHOLD = 0.03

# ── Risk management ───────────────────────────────────────────────────────────
# Stop-loss and trailing stop are intentionally disabled.
# The macro composite itself handles de-risking: when the regime flips to
# risk-off the rebalance job targets zero exposure. Layering a 5%/7% stop
# on top causes double de-risking and whipsaw re-entries.
# A 15% hard stop remains as a circuit breaker for catastrophic single events.
HARD_STOP_PCT      = 0.15
USE_TRAILING_STOP  = False
PRICE_MONITOR_SECS = 60


# ══════════════════════════════════════════════════════════════════════════════
# ALPACA CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

def _trading_client() -> TradingClient:
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MR2 / ALPACA_PAPER_API_SECRET_MR2 not set in env"
        )
    return TradingClient(API_KEY, API_SECRET, paper=PAPER)


def _data_client() -> StockHistoricalDataClient:
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MR2 / ALPACA_PAPER_API_SECRET_MR2 not set in env"
        )
    return StockHistoricalDataClient(API_KEY, API_SECRET)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING — own cache under data/mr_cache/
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}_1Day.parquet"


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) / 3600 < CACHE_EXPIRY_HOURS


def _fetch_one(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    path = _cache_path(symbol)
    if use_cache and _cache_fresh(path):
        logger.debug(f"Cache hit: {symbol}")
        return pd.read_parquet(path)

    logger.info(f"Fetching {symbol}…")
    end_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=BACKTEST_START,
        end=end_dt,
        adjustment="all",
    )
    bars = _data_client().get_stock_bars(req)
    df   = bars.df
    if df.empty:
        logger.warning(f"No data for {symbol}")
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(path)
    return df


def fetch_all(use_cache: bool = True) -> dict:
    """Fetch TRADE_UNIVERSE ∪ SIGNAL_UNIVERSE."""
    data = {}
    for sym in ALL_SYMBOLS:
        try:
            df = _fetch_one(sym, use_cache=use_cache)
            if not df.empty:
                data[sym] = df
        except Exception as e:
            logger.error(f"Failed {sym}: {e}")
        time.sleep(0.05)
    logger.info(f"Fetched {len(data)}/{len(ALL_SYMBOLS)} symbols")
    return data


def _close_matrix(data: dict) -> pd.DataFrame:
    return (
        pd.DataFrame({s: data[s]["close"] for s in data})
        .sort_index()
        .ffill()
        .dropna(how="all")
    )


def _ret_matrix(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close / close.shift(1)).dropna(how="all")


# ══════════════════════════════════════════════════════════════════════════════
# MACRO REGIME SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def _vix_proxy(spy_ret: pd.Series) -> pd.Series:
    """SPY realised vol scaled to approximate VIX units."""
    vol = spy_ret.rolling(VIX_PROXY_WINDOW).std() * np.sqrt(252) * 100
    sig = pd.Series(0.0, index=spy_ret.index)
    sig[vol < VIX_RISK_ON]   =  1.0
    sig[vol >= VIX_RISK_OFF] = -1.0
    return sig.rename("vix_proxy")


def _equity_momentum(spy: pd.Series) -> pd.Series:
    """SPY above/below 200-day MA."""
    ma = spy.rolling(MA_LONG, min_periods=MA_LONG // 2).mean()
    return np.sign(spy - ma).rename("equity_mom")


def _market_breadth(signal_close: pd.DataFrame) -> pd.Series:
    """
    Fraction of SIGNAL_UNIVERSE stocks above their 50-day MA.
    Uses 22 independent names across all sectors — no ETFs — so this
    is a genuine breadth reading rather than an echo of the trading universe.
    """
    ma      = signal_close.rolling(BREADTH_MA, min_periods=BREADTH_MA // 2).mean()
    above   = (signal_close > ma).astype(float)
    breadth = above.mean(axis=1)
    sig     = pd.Series(0.0, index=signal_close.index)
    sig[breadth >= BREADTH_ON]  =  1.0
    sig[breadth <= BREADTH_OFF] = -1.0
    return sig.rename("breadth")


def _vol_regime(spy_ret: pd.Series) -> pd.Series:
    """Short-term vol expanding above long-term → stress signal."""
    v_short = spy_ret.rolling(VOL_SHORT).std()
    v_long  = spy_ret.rolling(VOL_LONG).std()
    return np.sign(v_long - v_short).rename("vol_regime")   # +1 = contracting = good


def _cross_momentum(signal_close: pd.DataFrame) -> pd.Series:
    """
    Average 12-1 month momentum across SIGNAL_UNIVERSE (no ETFs).
    Positive average → broad momentum tailwind.
    """
    score = signal_close.shift(MOM_SKIP) / signal_close.shift(MOM_LONG) - 1
    avg   = score.mean(axis=1)
    return np.sign(avg).rename("cross_mom")


def compute_components(close: pd.DataFrame) -> pd.DataFrame:
    """
    Return the five raw component signal series as columns, each in {-1, 0, +1}.

    SPY is used only for VIX proxy, trend MA, and vol regime.
    Breadth and cross-momentum use SIGNAL_UNIVERSE (22 independent stocks).
    """
    if MARKET_PROXY not in close.columns:
        raise RuntimeError(
            f"Market proxy '{MARKET_PROXY}' missing from fetched data — "
            "check that SPY downloaded successfully"
        )
    spy     = close[MARKET_PROXY]
    spy_ret = np.log(spy / spy.shift(1))

    sig_cols     = [s for s in SIGNAL_UNIVERSE if s in close.columns]
    signal_close = close[sig_cols]

    components = [
        _vix_proxy(spy_ret),
        _equity_momentum(spy),
        _market_breadth(signal_close),
        _vol_regime(spy_ret),
        _cross_momentum(signal_close),
    ]
    return pd.concat(components, axis=1).reindex(close.index).fillna(0.0)


def compute_composite(close: pd.DataFrame) -> pd.Series:
    """
    5-indicator macro composite over time.
    Returns series in [-1, +1] — the mean of the five component signals.
    """
    comp_df   = compute_components(close)
    composite = comp_df.mean(axis=1).clip(-1.0, 1.0)
    return composite


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO — risk parity (inverse vol)
# ══════════════════════════════════════════════════════════════════════════════

def risk_parity_weights(ret_window: pd.DataFrame, symbols: list) -> pd.Series:
    """
    Inverse-volatility risk parity, capped at MAX_POSITION_SIZE.

    Chosen over max-Sharpe because:
    - No matrix inversion → no instability from near-singular covariance
    - Macro composite gives uniform expected returns per ticker, so
      max-Sharpe degenerates to minimum-variance via noisy matrix inverse
    - Inverse-vol achieves similar risk diversification more robustly
    """
    # Require ≥2 non-NaN observations so std() is valid; otherwise the symbol
    # gets NaN vol, which clip() does NOT replace, poisoning the capping loop.
    avail = [
        s for s in symbols
        if s in ret_window.columns and ret_window[s].notna().sum() >= 2
    ]
    vol   = ret_window[avail].std() * np.sqrt(TRADING_DAYS)
    vol   = vol.clip(lower=VOL_FLOOR)
    w     = 1.0 / vol
    w     = w / w.sum()

    for _ in range(20):
        over = w > MAX_POSITION_SIZE
        if not over.any():
            break
        w[over] = MAX_POSITION_SIZE
        remaining = 1.0 - w[over].sum()
        under     = ~over
        if under.any() and w[under].sum() > 0:
            w[under] = w[under] / w[under].sum() * remaining

    return w.reindex(symbols, fill_value=0.0)


def _subperiod_metrics(port_r: pd.Series, scale: pd.Series) -> dict:
    """
    Split port_r and scale at SUBPERIOD_SPLIT and return Sharpe + scale_mean
    for the 2022 bear (<split) and 2023+ bull (>=split) sub-periods.
    """
    split = pd.Timestamp(SUBPERIOD_SPLIT, tz="UTC")

    def _stats(r: pd.Series, s: pd.Series) -> dict:
        if len(r) == 0:
            return {"sharpe": 0.0, "scale_mean": round(float(s.mean()), 3) if len(s) else 0.0, "n_days": 0}
        ann_ret = float(r.mean() * TRADING_DAYS)
        ann_vol = float(r.std()  * np.sqrt(TRADING_DAYS))
        return {
            "sharpe":     round(ann_ret / ann_vol if ann_vol > 0 else 0.0, 3),
            "scale_mean": round(float(s.mean()), 3),
            "n_days":     len(r),
        }

    return {
        "bear": _stats(port_r[port_r.index <  split], scale[scale.index <  split]),
        "bull": _stats(port_r[port_r.index >= split], scale[scale.index >= split]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(use_cache: bool = True, fixed_scale: float | None = None) -> dict:
    """
    Run the walk-forward backtest.

    fixed_scale : if provided, bypass the macro composite entirely and hold
                  that constant scale (e.g. 1.0 = always fully invested pure
                  risk-parity baseline).  None (default) uses the 5-component
                  macro composite to derive a time-varying scale in [0, 1].
    """
    label = f"BASELINE (scale≡{fixed_scale}, pure risk-parity)" if fixed_scale is not None else "MACRO OVERLAY"
    logger.info("=" * 60)
    logger.info(f"MACRO REGIME TRADER v2 — BACKTEST  [{label}]")
    logger.info("=" * 60)

    data    = fetch_all(use_cache=use_cache)
    close   = _close_matrix(data)
    returns = _ret_matrix(close)

    if fixed_scale is not None:
        scale = pd.Series(float(fixed_scale), index=close.index).clip(0.0, 1.0)
        logger.info(f"Scale fixed at {fixed_scale} — macro composite bypassed")
    else:
        logger.info("Computing macro composite…")
        composite = compute_composite(close)
        # Map [-1, +1] → [0, 1]: full risk-off = cash, neutral = half, full risk-on = full
        scale = ((composite + 1.0) / 2.0).clip(0.0, 1.0)
        risk_on_frac = (composite > 0).mean()
        logger.info(
            f"Composite [{composite.min():.2f}, {composite.max():.2f}] "
            f"| risk-on {risk_on_frac:.0%} of days"
        )

    tradeable = [s for s in TRADE_UNIVERSE if s in close.columns]
    dates     = close.index
    wh        = pd.DataFrame(0.0, index=dates, columns=tradeable)

    logger.info("Running walk-forward backtest…")
    for i in range(RISK_LOOKBACK_DAYS, len(dates)):
        sc = float(scale.iloc[i])
        if sc < 0.05:
            continue
        ret_window = returns[tradeable].iloc[i - RISK_LOOKBACK_DAYS : i]
        base_w     = risk_parity_weights(ret_window, tradeable)
        wh.iloc[i] = base_w * sc

    fwd    = returns[tradeable].shift(-1)
    port_r = (wh * fwd).sum(axis=1).iloc[RISK_LOOKBACK_DAYS:-1].dropna()

    n         = len(port_r)
    ann_ret   = float(port_r.mean() * TRADING_DAYS)
    ann_vol   = float(port_r.std()  * np.sqrt(TRADING_DAYS))
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum       = (1 + port_r).cumprod()
    total_ret = float(cum.iloc[-1] - 1) if len(cum) else 0.0
    roll_max  = cum.cummax()
    max_dd    = float(((cum - roll_max) / roll_max).min())
    win_rate  = float((port_r > 0).sum() / n) if n else 0.0
    years     = n / TRADING_DAYS
    cagr      = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    calmar     = cagr / abs(max_dd) if max_dd < 0 else 0.0
    final_cap  = PORTFOLIO_USD * (1 + total_ret)
    subperiods = _subperiod_metrics(port_r, scale)

    metrics = {
        "sharpe":        round(sharpe,    3),
        "calmar":        round(calmar,    3),
        "cagr":          round(cagr,      4),
        "total_return":  round(total_ret, 4),
        "max_drawdown":  round(max_dd,    4),
        "annual_vol":    round(ann_vol,   4),
        "win_rate":      round(win_rate,  4),
        "final_capital": round(final_cap, 2),
        "n_days":        n,
        "bear":          subperiods["bear"],
        "bull":          subperiods["bull"],
    }

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS — Macro Regime v2 (risk parity portfolio)")
    logger.info(f"  Period        : {port_r.index[0].date()} → {port_r.index[-1].date()}")
    logger.info(f"  Trading days  : {n}")
    logger.info(f"  Sharpe        : {sharpe:.3f}")
    logger.info(f"  Calmar        : {calmar:.3f}")
    logger.info(f"  CAGR          : {cagr*100:.2f}%")
    logger.info(f"  Total Return  : {total_ret*100:.2f}%")
    logger.info(f"  Max Drawdown  : {max_dd*100:.2f}%")
    logger.info(f"  Annual Vol    : {ann_vol*100:.2f}%")
    logger.info(f"  Win Rate      : {win_rate*100:.1f}%")
    logger.info(f"  Final Capital : ${final_cap:,.2f}  (start ${PORTFOLIO_USD:,})")
    b, u = subperiods["bear"], subperiods["bull"]
    logger.info(f"  ── Sub-periods ──────────────────────────────────────")
    logger.info(f"  2022 bear  (n={b['n_days']:>4d})  Sharpe={b['sharpe']:+.3f}  scale_mean={b['scale_mean']:.3f}")
    logger.info(f"  2023+ bull (n={u['n_days']:>4d})  Sharpe={u['sharpe']:+.3f}  scale_mean={u['scale_mean']:.3f}")
    logger.info("=" * 60)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — correlation between the 5 composite components
# ══════════════════════════════════════════════════════════════════════════════

def _component_standalone_backtest(
    comp_sig: pd.Series,
    returns: pd.DataFrame,
    tradeable: list,
) -> dict:
    """
    Run the same risk-parity backtest as run_backtest, but using ONE
    component's signal (mapped {-1,0,+1} -> scale [0,1]) instead of the
    5-way composite. Lets us see each component's standalone Sharpe.
    """
    scale = ((comp_sig + 1.0) / 2.0).clip(0.0, 1.0)
    dates = comp_sig.index
    wh    = pd.DataFrame(0.0, index=dates, columns=tradeable)

    for i in range(RISK_LOOKBACK_DAYS, len(dates)):
        sc = float(scale.iloc[i])
        if sc < 0.05:
            continue
        ret_window = returns[tradeable].iloc[i - RISK_LOOKBACK_DAYS : i]
        base_w     = risk_parity_weights(ret_window, tradeable)
        wh.iloc[i] = base_w * sc

    fwd    = returns[tradeable].shift(-1)
    port_r = (wh * fwd).sum(axis=1).iloc[RISK_LOOKBACK_DAYS:-1].dropna()

    if len(port_r) == 0:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "scale_mean": float(scale.mean()),
                "bear": {"sharpe": 0.0, "scale_mean": 0.0, "n_days": 0},
                "bull": {"sharpe": 0.0, "scale_mean": 0.0, "n_days": 0}}

    ann_ret    = float(port_r.mean() * TRADING_DAYS)
    ann_vol    = float(port_r.std()  * np.sqrt(TRADING_DAYS))
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0
    subperiods = _subperiod_metrics(port_r, scale)

    return {
        "sharpe":     round(sharpe, 3),
        "ann_ret":    round(ann_ret, 4),
        "ann_vol":    round(ann_vol, 4),
        "scale_mean": round(float(scale.mean()), 3),
        "bear":       subperiods["bear"],
        "bull":       subperiods["bull"],
    }


def analyze_components(use_cache: bool = True) -> dict:
    """
    Decompose the 5-indicator macro composite and check:
      1. Pairwise correlation between the 5 raw component signals
      2. How often each component agrees with the overall composite sign
      3. Each component's standalone Sharpe if it alone drove the
         risk-parity book (same scale mapping as the full composite)
      4. Each component's correlation with next-day SPY return (does
         the signal actually point the right way?)

    Purpose: identify redundant components (high pairwise correlation =
    effectively double-counted in the 5-way average) vs. components that
    add genuinely independent information.
    """
    logger.info("=" * 60)
    logger.info("MACRO REGIME TRADER v2 — COMPONENT CORRELATION ANALYSIS")
    logger.info("=" * 60)

    data    = fetch_all(use_cache=use_cache)
    close   = _close_matrix(data)
    returns = _ret_matrix(close)

    comp_df   = compute_components(close)
    composite = comp_df.mean(axis=1).clip(-1.0, 1.0)

    # 1. Pairwise correlation matrix of raw component signals
    corr = comp_df.corr()
    logger.info("Pairwise correlation matrix of the 5 component signals:")
    logger.info("\n" + corr.round(2).to_string())

    # Flag near-duplicate pairs (|corr| >= 0.6, excluding diagonal)
    logger.info("-" * 60)
    logger.info("Highly correlated pairs (|corr| >= 0.6):")
    flagged = []
    cols = comp_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= 0.6:
                flagged.append((cols[i], cols[j], round(float(c), 3)))
                logger.info(f"  {cols[i]:12s} <-> {cols[j]:12s}  corr={c:+.3f}")
    if not flagged:
        logger.info("  (none)")

    # 2. Agreement with composite sign
    logger.info("-" * 60)
    logger.info("Agreement with composite sign (fraction of days same sign):")
    comp_sign = np.sign(composite)
    agreement = {}
    for col in cols:
        same = (np.sign(comp_df[col]) == comp_sign) & (comp_sign != 0)
        valid = (comp_sign != 0)
        frac = float(same.sum() / valid.sum()) if valid.sum() else 0.0
        agreement[col] = round(frac, 3)
        logger.info(f"  {col:12s} : {frac:.1%}")

    # 3. Standalone Sharpe per component
    tradeable = [s for s in TRADE_UNIVERSE if s in close.columns]
    logger.info("-" * 60)
    logger.info("Standalone backtest if each component alone drove sizing:")
    def _log_subperiods(res: dict) -> None:
        b, u = res["bear"], res["bull"]
        logger.info(
            f"    bear 2022  (n={b['n_days']:>4d})  Sharpe={b['sharpe']:+.3f}  scale_mean={b['scale_mean']:.3f}"
            f"  |  bull 2023+ (n={u['n_days']:>4d})  Sharpe={u['sharpe']:+.3f}  scale_mean={u['scale_mean']:.3f}"
        )

    standalone = {}
    for col in cols:
        res = _component_standalone_backtest(comp_df[col], returns, tradeable)
        standalone[col] = res
        logger.info(
            f"  {col:12s} : Sharpe={res['sharpe']:+.3f}  "
            f"ann_ret={res['ann_ret']*100:6.2f}%  "
            f"ann_vol={res['ann_vol']*100:5.2f}%  "
            f"scale_mean={res['scale_mean']:.3f}"
        )
        _log_subperiods(res)

    # Composite itself, for comparison
    comp_res = _component_standalone_backtest(composite, returns, tradeable)
    logger.info(f"  {'composite (5-avg)':20s} : Sharpe={comp_res['sharpe']:+.3f}  "
                 f"ann_ret={comp_res['ann_ret']*100:6.2f}%  "
                 f"ann_vol={comp_res['ann_vol']*100:5.2f}%  "
                 f"scale_mean={comp_res['scale_mean']:.3f}")
    _log_subperiods(comp_res)

    # Scale≡1 baseline (pure risk-parity, no macro overlay) — reference row
    baseline_sig = pd.Series(1.0, index=composite.index)
    baseline_res = _component_standalone_backtest(baseline_sig, returns, tradeable)
    logger.info(f"  {'baseline (scale≡1)':20s} : Sharpe={baseline_res['sharpe']:+.3f}  "
                 f"ann_ret={baseline_res['ann_ret']*100:6.2f}%  "
                 f"ann_vol={baseline_res['ann_vol']*100:5.2f}%  "
                 f"scale_mean={baseline_res['scale_mean']:.3f}")
    _log_subperiods(baseline_res)

    # 4. Correlation with next-day SPY return
    logger.info("-" * 60)
    logger.info("Correlation with next-day SPY return (does signal point the right way?):")
    spy_fwd_ret = returns[MARKET_PROXY].shift(-1)
    spy_corr = {}
    for col in cols:
        valid = comp_df[col].notna() & spy_fwd_ret.notna()
        c = float(np.corrcoef(comp_df[col][valid], spy_fwd_ret[valid])[0, 1])
        spy_corr[col] = round(c, 4)
        logger.info(f"  {col:12s} : corr={c:+.4f}")

    logger.info("=" * 60)

    return {
        "correlation_matrix": corr.round(4).to_dict(),
        "flagged_pairs":       flagged,
        "agreement_with_composite": agreement,
        "standalone_backtest": standalone,
        "composite_backtest":  comp_res,
        "baseline_backtest":   baseline_res,
        "spy_fwd_corr":        spy_corr,
    }




def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "positions":        {sym: 0.0 for sym in TRADE_UNIVERSE},
        "cash_usd":         float(PORTFOLIO_USD),
        "current_weights":  {},
        "last_run":         None,
        "nav_history":      [],
        "trade_log":        [],
        "position_entries": {},
    }


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE — helpers
# ══════════════════════════════════════════════════════════════════════════════

def _market_is_open() -> bool:
    try:
        return _trading_client().get_clock().is_open
    except Exception:
        return False


def _fetch_live_prices() -> dict:
    try:
        trades = _data_client().get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=TRADE_UNIVERSE)
        )
        return {sym: float(t.price) for sym, t in trades.items()}
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        return {}


def _compute_nav(state: dict, prices: dict) -> float:
    nav = state["cash_usd"]
    for sym, qty in state["positions"].items():
        if qty and sym in prices:
            nav += qty * prices[sym]
    return nav


def _check_hard_stops(state: dict, prices: dict) -> dict:
    """15% hard stop only — circuit breaker for catastrophic single-stock events."""
    exits   = {}
    entries = state.get("position_entries", {})
    for sym, qty in state["positions"].items():
        if qty == 0 or sym not in entries or sym not in prices:
            continue
        ep  = entries[sym]["entry_price"]
        pct = (prices[sym] - ep) / ep
        if pct <= -HARD_STOP_PCT:
            exits[sym] = prices[sym]
            logger.warning(f"[HARD STOP] {sym}  {pct*100:.2f}%  (15% circuit breaker)")
    return exits


def _execute_exits(exits: dict, state: dict) -> dict:
    client = _trading_client()
    for sym, price in exits.items():
        qty = state["positions"].get(sym, 0)
        if qty == 0:
            continue
        try:
            # close_position closes exactly what Alpaca holds — avoids
            # fractional qty mismatch between local state and Alpaca's fill record
            order = client.close_position(sym)
            state["positions"][sym] = 0.0
            state["cash_usd"]      += qty * price
            state["position_entries"].pop(sym, None)
            state.setdefault("trade_log", []).append({
                "time": str(datetime.now(timezone.utc)),
                "symbol": sym, "side": "SELL",
                "qty": qty, "price": price,
                "reason": "hard_stop", "order_id": str(order.id),
            })
        except Exception as e:
            logger.error(f"Exit failed {sym}: {e}")
    return state


def _execute_rebalance(target_w: dict, state: dict, nav: float, prices: dict) -> dict:
    if not _market_is_open():
        logger.warning("Market closed — skipping rebalance")
        return state

    client   = _trading_client()
    cur_vals = {s: state["positions"].get(s, 0) * prices.get(s, 0) for s in TRADE_UNIVERSE}
    cur_w    = {s: v / nav for s, v in cur_vals.items()} if nav > 0 else {}

    sells, buys = [], []
    for sym in TRADE_UNIVERSE:
        delta = target_w.get(sym, 0) - cur_w.get(sym, 0)
        usd   = delta * nav
        if abs(usd) < MIN_ORDER_USD:
            continue
        (sells if usd < 0 else buys).append((sym, usd))

    for sym, usd in sells + buys:
        price = prices.get(sym)
        if not price:
            continue
        qty  = abs(usd) / price
        side = OrderSide.SELL if usd < 0 else OrderSide.BUY
        try:
            order = client.submit_order(MarketOrderRequest(
                symbol=sym, notional=round(abs(usd), 2),
                side=side, time_in_force=TimeInForce.DAY,
            ))
            if side == OrderSide.BUY:
                state["positions"][sym] = state["positions"].get(sym, 0) + qty
                state["cash_usd"]      -= qty * price
                if sym not in state.get("position_entries", {}):
                    state.setdefault("position_entries", {})[sym] = {
                        "entry_price": price,
                        "entry_date":  str(datetime.now(timezone.utc)),
                    }
            else:
                new_qty = max(state["positions"].get(sym, 0) - qty, 0)
                state["positions"][sym] = new_qty
                state["cash_usd"]      += qty * price
                if new_qty == 0:
                    state.setdefault("position_entries", {}).pop(sym, None)
            state.setdefault("trade_log", []).append({
                "time": str(datetime.now(timezone.utc)),
                "symbol": sym, "side": side.value,
                "qty": round(qty, 4), "price": price,
                "reason": "macro_rebalance", "order_id": str(order.id),
            })
            logger.info(f"  {side.value:4s} {sym:5s}  qty={qty:.3f}  @ ${price:.2f}")
        except Exception as e:
            logger.error(f"Order failed {sym}: {e}")

    return state


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULED JOBS
# ══════════════════════════════════════════════════════════════════════════════

def price_monitor_job():
    """Check 15% hard stop every PRICE_MONITOR_SECS seconds."""
    state    = _load_state()
    open_pos = {s: q for s, q in state["positions"].items() if q != 0}
    if not open_pos:
        return
    prices = _fetch_live_prices()
    exits  = _check_hard_stops(state, prices)
    if exits:
        state = _execute_exits(exits, state)
        nav   = _compute_nav(state, prices)
        state["nav_history"].append({
            "date": str(datetime.now(timezone.utc)),
            "nav":  round(nav, 2), "event": "hard_stop",
        })
        _save_state(state)


def signal_rebalance_job():
    """Recompute macro signal and rebalance if weight delta exceeds threshold."""
    logger.info("-" * 55)
    logger.info(f"Macro signal recompute — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    try:
        data    = fetch_all(use_cache=False)
        close   = _close_matrix(data)
        returns = _ret_matrix(close)
    except Exception as e:
        logger.error(f"Data fetch failed — skipping: {e}")
        return

    composite = compute_composite(close)
    last_comp = float(composite.iloc[-1])
    scale     = float(((last_comp + 1.0) / 2.0))
    scale     = max(0.0, min(1.0, scale))
    logger.info(f"Composite: {last_comp:.3f}  →  scale: {scale:.3f}")

    tradeable  = [s for s in TRADE_UNIVERSE if s in close.columns]
    ret_window = returns[tradeable].iloc[-RISK_LOOKBACK_DAYS:]

    if scale < 0.05:
        target_w = {s: 0.0 for s in tradeable}
        logger.info("Risk-off: targeting full cash (composite ≤ -0.9)")
    else:
        base_w   = risk_parity_weights(ret_window, tradeable)
        target_w = (base_w * scale).to_dict()

    state  = _load_state()
    prices = _fetch_live_prices()
    nav    = _compute_nav(state, prices)

    if nav <= 0:
        logger.warning("NAV is zero — skipping rebalance")
        return

    prev_w = {
        sym: (state["positions"].get(sym, 0) * prices.get(sym, 0)) / nav
        for sym in tradeable
    }

    delta = sum(abs(target_w.get(s, 0) - prev_w.get(s, 0)) for s in tradeable)
    logger.info(f"Weight delta: {delta:.4f}  (threshold={REBALANCE_THRESHOLD})")

    if delta < REBALANCE_THRESHOLD:
        logger.info("Signal unchanged — holding.")
        return

    state = _execute_rebalance(target_w, state, nav, prices)
    state["current_weights"] = target_w
    state["last_run"]        = str(datetime.now(timezone.utc))
    nav = _compute_nav(state, prices)
    state["nav_history"].append({
        "date": str(datetime.now(timezone.utc)),
        "nav":  round(nav, 2), "event": "rebalance",
    })
    _save_state(state)
    logger.success(f"Rebalance complete. NAV=${nav:,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_live(run_now: bool = False):
    mode = "PAPER" if PAPER else "LIVE ⚠️"
    logger.info("=" * 60)
    logger.info(f"MACRO REGIME TRADER v2 — LIVE ENGINE [{mode}]")
    logger.info(f"  Trade universe   : {len(TRADE_UNIVERSE)} names")
    logger.info(f"  Signal universe  : {len(SIGNAL_UNIVERSE)} stocks (no ETFs)")
    logger.info(f"  Portfolio        : risk parity (inverse vol, cap {MAX_POSITION_SIZE:.0%})")
    logger.info(f"  Hard stop        : {HARD_STOP_PCT:.0%}  |  Trailing stop: disabled")
    logger.info(f"  Price monitor    : every {PRICE_MONITOR_SECS}s")
    logger.info(f"  Rebalance        : 09:35 ET + 15:50 ET, Mon–Fri")
    logger.info(f"  State file       : {STATE_FILE}")
    logger.info("=" * 60)

    try:
        acct = _trading_client().get_account()
        logger.info(f"Alpaca (MR paper) connected — equity=${float(acct.equity):,.2f}")
    except Exception as e:
        logger.error(f"Alpaca connection failed: {e}")
        sys.exit(1)

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        func=price_monitor_job, trigger="interval",
        seconds=PRICE_MONITOR_SECS, id="mr_price_monitor",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        func=signal_rebalance_job, trigger="cron",
        day_of_week="mon-fri", hour=9, minute=35,
        timezone="America/New_York", id="mr_signal_open",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        func=signal_rebalance_job, trigger="cron",
        day_of_week="mon-fri", hour=15, minute=50,
        timezone="America/New_York", id="mr_signal_close",
        max_instances=1, coalesce=True,
    )
    scheduler.start()

    if run_now:
        logger.info("--run-now: firing initial rebalance…")
        signal_rebalance_job()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.success("Macro engine stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logger():
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        LOG_FILE, level="DEBUG", rotation="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="Macro Regime Standalone Trader v2")
    parser.add_argument(
        "--mode", choices=["backtest", "baseline", "live", "full", "analyze"], default="backtest",
        help=(
            "backtest — run macro-overlay backtest, then print a side-by-side "
            "comparison against the scale≡1 pure risk-parity baseline. "
            "baseline — run only the scale≡1 pure risk-parity baseline (no macro overlay). "
            "live     — start live engine immediately. "
            "full     — run backtest first; if Sharpe ≥ min-sharpe proceed to live. "
            "analyze  — decompose the 5 composite components: pairwise "
            "correlation, agreement with composite sign, standalone "
            "Sharpe per component, and correlation with next-day SPY return."
        ),
    )
    parser.add_argument("--no-cache",   action="store_true", help="Force re-download of market data")
    parser.add_argument("--run-now",    action="store_true", help="Fire one rebalance immediately on live start")
    parser.add_argument("--min-sharpe", type=float, default=0.5, help="Minimum backtest Sharpe to proceed to live")
    args = parser.parse_args()

    if args.mode == "backtest":
        macro_m    = run_backtest(use_cache=not args.no_cache)
        baseline_m = run_backtest(use_cache=True, fixed_scale=1.0)
        logger.info("=" * 60)
        logger.info("COMPARISON — Macro Overlay vs Pure Risk-Parity Baseline (scale≡1)")
        logger.info(f"  {'':10s}  {'Sharpe':>7}  {'Calmar':>7}  {'CAGR':>7}  {'MaxDD':>7}  {'AnnVol':>7}  {'WinRate':>8}")
        for name, m in [("Macro", macro_m), ("Baseline", baseline_m)]:
            logger.info(
                f"  {name:10s}  {m['sharpe']:>7.3f}  {m['calmar']:>7.3f}  "
                f"{m['cagr']*100:>6.2f}%  {m['max_drawdown']*100:>6.2f}%  "
                f"{m['annual_vol']*100:>6.2f}%  {m['win_rate']*100:>7.1f}%"
            )
        logger.info("=" * 60)

    elif args.mode == "baseline":
        run_backtest(use_cache=not args.no_cache, fixed_scale=1.0)

    elif args.mode == "analyze":
        analyze_components(use_cache=not args.no_cache)

    elif args.mode == "live":
        run_live(run_now=args.run_now)

    elif args.mode == "full":
        metrics = run_backtest(use_cache=not args.no_cache)
        sharpe  = metrics["sharpe"]
        logger.info(f"Backtest Sharpe: {sharpe:.3f}  (minimum required: {args.min_sharpe})")
        if sharpe >= args.min_sharpe:
            logger.success("Sharpe threshold passed — starting live engine.")
            run_live(run_now=args.run_now)
        else:
            logger.error(
                f"Sharpe {sharpe:.3f} below threshold {args.min_sharpe} — "
                "live trading NOT started. Review strategy before going live."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
