"""
macro_regime_trader.py  (v3 — standalone)
==========================================
Macro Regime-only trading machine. Zero imports from the rest of the project.
All config, strategy logic, portfolio construction, and execution are in this file.

Changes from v2
---------------
- Final strategy candidate is now the Defensive 4-Signal Macro Regime Overlay:
  VIX proxy, SPY trend, market breadth, and volatility regime.
- 12-1 month cross momentum is excluded from the default base model after
  ablation/OOS testing showed it can dilute drawdown control.
- Adds a SPY volume-confirmation layer: volume breakout, Follow-Through Day,
  Distribution Day, and Heavy Distribution Day.
- Keeps base_price_only as the control model and compares it directly against
  price_plus_spy_volume_confirmation.
- Adds explicit execution-lag diagnostics proving that same-day close/volume
  signals only affect next-period returns.
- Adds benchmark comparison, transaction-cost sensitivity, out-of-sample
  metrics, parameter sensitivity, and a markdown research summary.
- Final implementation reduces turnover by using scale-change-only rebalancing
  for the defensive overlay; daily and weekly policies are retained as
  research comparisons, not optimization candidates.

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

Usage
-----
  python macro_regime_trader.py --mode backtest
  python macro_regime_trader.py --mode backtest --price-only
  python macro_regime_trader.py --mode live
  python macro_regime_trader.py --mode full --run-now   # backtest → live if Sharpe ≥ 0.5
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

try:
    from loguru import logger
except ImportError:
    class _FallbackLogger:
        def __getattr__(self, name):
            def _log(message="", *args, **kwargs):
                text = f"{name.upper()}: {message}"
                print(text.encode("ascii", "replace").decode("ascii"))
            return _log
        def remove(self): pass
        def add(self, *args, **kwargs): pass
    logger = _FallbackLogger()

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    TradingClient = None
    MarketOrderRequest = None
    OrderSide = None
    TimeInForce = None
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockLatestTradeRequest = None
    TimeFrame = None
    TimeFrameUnit = None


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

# Dedicated MR paper-trading Alpaca account. Keep PAPER=True for this research
# project; the live engine is intended for Alpaca paper trading only.
API_KEY    = os.getenv("ALPACA_PAPER_API_KEY_MRKENGO")
API_SECRET = os.getenv("ALPACA_PAPER_API_SECRET_MRKENGO")
PAPER      = True

# Paths — kengo-specific filenames to avoid clashing with macro_regime_trader2.py
# which runs from the same QTS/ directory and would otherwise share state + log.
DATA_DIR        = ROOT_DIR / "data" / "kengo_cache"
SOURCE_DATA_DIR = ROOT_DIR / "data" / "source_csv"
LOG_DIR         = ROOT_DIR / "logs"
RESULT_DIR      = ROOT_DIR / "results"
STATE_FILE      = RESULT_DIR / "kengo_live_state.json"
LOG_FILE        = LOG_DIR   / "kengo_engine.log"

LEGACY_CLASS_DATA_DIR = Path(
    r"C:\Users\kutsu\OneDrive\デスクトップ\KENGO\SMU\Class\QF-621Quantitative Trading Strategies\Apr-18"
)

for _d in [DATA_DIR, SOURCE_DATA_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universes ─────────────────────────────────────────────────────────────────

# Tickers we actually trade — 10 independent stocks only.
# SPY and QQQ are excluded: risk parity assigns them the highest weights
# (lowest vol → highest 1/vol) while they mechanically hold the same AAPL/
# MSFT/NVDA names already in the book, creating silent double-exposure that
# risk parity cannot see because it works per-symbol, not look-through.
# SPY is kept as MARKET_PROXY for signal computation (fetched but not traded).
TRADE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "JPM",  "JNJ",   "XOM",  "UNH",
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

# All symbols to fetch: trade + signal + market proxy (SPY is not in either list now)
ALL_SYMBOLS = sorted(set(TRADE_UNIVERSE) | set(SIGNAL_UNIVERSE) | {MARKET_PROXY})

# ── Data ─────────────────────────────────────────────────────────────────────
BACKTEST_START     = "2022-01-01"
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
USE_CROSS_MOMENTUM_IN_BASE = False  # final defensive 4-signal specification

# ── SPY volume-confirmation layer ────────────────────────────────────────────
# These are research defaults, not optimized values. Sensitivity analysis below
# tests nearby values to evaluate robustness rather than tune a leaderboard.
USE_SPY_VOLUME_CONFIRMATION = True
VOLUME_CONFIRMATION_WEIGHT  = 0.15
SPY_VOLUME_AVG_WINDOW       = 50
SPY_VOLUME_BREAKOUT         = 1.10
SPY_VOLUME_STRONG_BREAKOUT  = 1.20
SPY_FTD_MIN_RETURN          = 0.0125
SPY_FTD_MIN_DAY             = 4
SPY_DISTRIBUTION_MIN_DROP   = -0.0025
SPY_HEAVY_DIST_MIN_DROP     = -0.0100
SPY_CORRECTION_DRAWDOWN     = -0.08
SPY_RALLY_MA_WINDOW         = 50
SPY_DISTRIBUTION_WINDOW     = 25
SPY_DISTRIBUTION_CLUSTER    = 4

SPY_FTD_MIN_RETURN_GRID = [0.0100, 0.0125, 0.0150]
SPY_VOLUME_STRONG_BREAKOUT_GRID = [1.10, 1.20, 1.30]
VOLUME_CONFIRMATION_WEIGHT_GRID = [0.10, 0.15, 0.20]
COST_SCENARIOS_BPS = [0.0, 5.0, 10.0, 20.0]

BASE_MA_LONG_GRID = [150, 200, 250]
BASE_BREADTH_MA_GRID = [40, 50, 60]
BASE_VOL_SHORT_GRID = [10, 15]
BASE_VOL_LONG_GRID = [30, 60]

# ── Portfolio ─────────────────────────────────────────────────────────────────
TRADING_DAYS       = 252
RISK_LOOKBACK_DAYS = 126    # 6-month rolling vol estimate
VOL_FLOOR          = 0.03   # 3% annualised vol floor (prevents extreme weights)
MAX_POSITION_SIZE  = 0.15   # 15% single-name cap (down from 20%)
MAX_LIVE_POSITIONS = 8

# ── Backtest frictions ───────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 5.0

# Rebalancing policy for the final defensive overlay. The strategy's economic
# hypothesis is about market-level exposure timing, not daily micro-adjustment
# of stock weights, so the final candidate only trades when the regime scale
# changes. Daily and weekly variants are tested separately for robustness.
FINAL_REBALANCE_POLICY = "scale_change_only"
REBALANCE_POLICY_GRID = ["daily", "weekly", "scale_change_only"]
MARKET_IMPACT_COEFF  = 0.10
ADV_LOOKBACK_DAYS    = 20

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

_RUSSELL_SYMBOL_CACHE: dict[str, pd.DataFrame] = {}
_RUSSELL_CACHE_PATH: Path | None = None
_RISK_PARITY_ZERO_WARNED = False


# ══════════════════════════════════════════════════════════════════════════════
# ALPACA CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

def _trading_client() -> TradingClient:
    if TradingClient is None:
        raise RuntimeError("alpaca-py is not installed; live trading is unavailable.")
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MRKENGO / ALPACA_PAPER_API_SECRET_MRKENGO not set in env"
        )
    return TradingClient(API_KEY, API_SECRET, paper=PAPER)


def _data_client() -> StockHistoricalDataClient:
    if StockHistoricalDataClient is None:
        raise RuntimeError("alpaca-py is not installed; Alpaca data download is unavailable.")
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MRKENGO / ALPACA_PAPER_API_SECRET_MRKENGO not set in env"
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


def _local_csv_dirs() -> list[Path]:
    dirs = [SOURCE_DATA_DIR]
    env_dir = os.getenv("MR_LOCAL_DATA_DIR")
    if env_dir:
        dirs.insert(0, Path(env_dir))
    if LEGACY_CLASS_DATA_DIR.exists():
        dirs.append(LEGACY_CLASS_DATA_DIR)
    return dirs


def _read_spy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", utc=True)
    df = df.rename(columns={"adj_close": "adj_close"})
    return (
        df.set_index("date")[["open", "high", "low", "close", "volume"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(subset=["close", "volume"])
        .sort_index()
    )


def _load_russell_symbol_cache(path: Path):
    global _RUSSELL_SYMBOL_CACHE, _RUSSELL_CACHE_PATH
    if _RUSSELL_CACHE_PATH == path and _RUSSELL_SYMBOL_CACHE:
        return
    logger.info(f"Loading Russell source CSV once: {path}")
    wanted = set(ALL_SYMBOLS)
    chunks = []
    usecols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, na_values=["null"]):
        local = chunk[chunk["ticker"].isin(wanted)].copy()
        if not local.empty:
            chunks.append(local)
    if not chunks:
        _RUSSELL_SYMBOL_CACHE = {}
        _RUSSELL_CACHE_PATH = path
        return
    all_df = pd.concat(chunks, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"].astype(str), format="%Y%m%d", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    _RUSSELL_SYMBOL_CACHE = {}
    for ticker, df in all_df.groupby("ticker"):
        _RUSSELL_SYMBOL_CACHE[ticker] = (
            df.set_index("date")[["open", "high", "low", "close", "volume"]]
            .dropna(subset=["close", "volume"])
            .sort_index()
        )
    _RUSSELL_CACHE_PATH = path


def _read_russell_symbol_csv(path: Path, symbol: str) -> pd.DataFrame:
    _load_russell_symbol_cache(path)
    df = _RUSSELL_SYMBOL_CACHE.get(symbol)
    if df is None:
        return pd.DataFrame()
    return (
        df.copy()
    )


def _fetch_one_local_csv(symbol: str) -> pd.DataFrame:
    for data_dir in _local_csv_dirs():
        if symbol == MARKET_PROXY:
            spy_path = data_dir / "SPY.csv"
            if spy_path.exists():
                logger.info(f"Loading {symbol} from local CSV: {spy_path}")
                return _read_spy_csv(spy_path)

        russell_path = data_dir / "russell1000pvdata.csv"
        if russell_path.exists():
            df = _read_russell_symbol_csv(russell_path, symbol)
            if not df.empty:
                logger.info(f"Loading {symbol} from local Russell CSV")
                return df
    return pd.DataFrame()


def _fetch_one(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    path = _cache_path(symbol)
    if use_cache and _cache_fresh(path):
        logger.debug(f"Cache hit: {symbol}")
        return pd.read_parquet(path)
    if use_cache and path.exists() and (not API_KEY or not API_SECRET):
        logger.warning(f"Using stale cache for {symbol}; Alpaca credentials are not set.")
        return pd.read_parquet(path)
    local_df = _fetch_one_local_csv(symbol)
    if not local_df.empty:
        return local_df
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            f"No cached data for {symbol} and Alpaca credentials are not set. "
            "Offline analysis needs existing parquet cache files."
        )

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
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.warning(f"Could not write parquet cache for {symbol}: {e}")
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


def _dollar_volume_matrix(data: dict, close: pd.DataFrame) -> pd.DataFrame:
    volume = (
        pd.DataFrame({s: data[s]["volume"] for s in data if "volume" in data[s]})
        .sort_index()
        .reindex(close.index)
        .ffill()
    )
    return (volume * close).replace([np.inf, -np.inf], np.nan)


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


def compute_price_composite(close: pd.DataFrame) -> pd.Series:
    """
    Final price-based macro composite over time.

    SPY is used only for VIX proxy, trend MA, and vol regime.
    Breadth uses SIGNAL_UNIVERSE (22 independent stocks).
    Cross momentum is excluded from the final defensive 4-signal specification
    because ablation/OOS tests showed it can dilute de-risking during drawdowns.
    Returns series in [-1, +1].
    """
    components = compute_price_components(close)
    exclude = [] if USE_CROSS_MOMENTUM_IN_BASE else ["cross_mom"]
    return _composite_from_components(components, exclude).reindex(close.index).fillna(0.0)


def compute_price_components(close: pd.DataFrame) -> pd.DataFrame:
    """Return the five base price-regime components for ablation diagnostics."""
    spy = close[MARKET_PROXY]
    spy_ret = np.log(spy / spy.shift(1))
    sig_cols = [s for s in SIGNAL_UNIVERSE if s in close.columns]
    signal_close = close[sig_cols]
    return pd.concat(
        [
            _vix_proxy(spy_ret),
            _equity_momentum(spy),
            _market_breadth(signal_close),
            _vol_regime(spy_ret),
            _cross_momentum(signal_close),
        ],
        axis=1,
    ).reindex(close.index).fillna(0.0)


def _composite_from_components(components: pd.DataFrame, exclude: list[str] | None = None) -> pd.Series:
    cols = [c for c in components.columns if c not in set(exclude or [])]
    return components[cols].mean(axis=1).clip(-1.0, 1.0).rename("price_composite")


def _scale_from_composite(composite: pd.Series) -> pd.Series:
    return ((composite + 1.0) / 2.0).clip(0.0, 1.0)


def compute_price_composite_with_params(
    close: pd.DataFrame,
    ma_long: int = MA_LONG,
    breadth_ma: int = BREADTH_MA,
    vol_short: int = VOL_SHORT,
    vol_long: int = VOL_LONG,
    include_cross_mom: bool = USE_CROSS_MOMENTUM_IN_BASE,
) -> pd.Series:
    """Parameterized base composite for robustness checks, not optimization."""
    spy = close[MARKET_PROXY]
    spy_ret = np.log(spy / spy.shift(1))
    sig_cols = [s for s in SIGNAL_UNIVERSE if s in close.columns]
    signal_close = close[sig_cols]

    vol = spy_ret.rolling(VIX_PROXY_WINDOW).std() * np.sqrt(252) * 100
    vix_sig = pd.Series(0.0, index=spy_ret.index, name="vix_proxy")
    vix_sig[vol < VIX_RISK_ON] = 1.0
    vix_sig[vol >= VIX_RISK_OFF] = -1.0

    ma = spy.rolling(ma_long, min_periods=ma_long // 2).mean()
    trend_sig = np.sign(spy - ma).rename("equity_mom")

    breadth_ma_px = signal_close.rolling(breadth_ma, min_periods=breadth_ma // 2).mean()
    breadth = (signal_close > breadth_ma_px).astype(float).mean(axis=1)
    breadth_sig = pd.Series(0.0, index=signal_close.index, name="breadth")
    breadth_sig[breadth >= BREADTH_ON] = 1.0
    breadth_sig[breadth <= BREADTH_OFF] = -1.0

    v_short = spy_ret.rolling(vol_short).std()
    v_long = spy_ret.rolling(vol_long).std()
    vol_regime_sig = np.sign(v_long - v_short).rename("vol_regime")

    component_list = [vix_sig, trend_sig, breadth_sig, vol_regime_sig]
    if include_cross_mom:
        component_list.append(_cross_momentum(signal_close))

    return (
        pd.concat(component_list, axis=1)
        .mean(axis=1)
        .clip(-1.0, 1.0)
        .reindex(close.index)
        .fillna(0.0)
        .rename("price_composite")
    )


def _volume_params(
    ftd_min_return: float = SPY_FTD_MIN_RETURN,
    strong_breakout: float = SPY_VOLUME_STRONG_BREAKOUT,
    volume_weight: float = VOLUME_CONFIRMATION_WEIGHT,
) -> dict:
    return {
        "ftd_min_return": ftd_min_return,
        "strong_breakout": strong_breakout,
        "volume_weight": volume_weight,
    }


def compute_spy_volume_confirmation(spy_ohlcv: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Compute SPY volume-confirmation signals using only same-day observable data.

    The returned score is still a signal-date score. Backtest execution applies
    it to the next tradable period via forward returns, never same-day returns.
    """
    params = params or _volume_params()
    spy = spy_ohlcv[["close", "low", "volume"]].copy().sort_index()
    close = spy["close"]
    low = spy["low"]
    volume = spy["volume"]
    simple_ret = close.pct_change()
    volume_ma = volume.rolling(
        SPY_VOLUME_AVG_WINDOW, min_periods=SPY_VOLUME_AVG_WINDOW // 2
    ).mean()
    volume_ratio = (volume / volume_ma).replace([np.inf, -np.inf], np.nan)
    ma = close.rolling(SPY_RALLY_MA_WINDOW, min_periods=SPY_RALLY_MA_WINDOW // 2).mean()
    drawdown = close / close.rolling(63, min_periods=21).max() - 1.0

    volume_breakout = (simple_ret > 0) & (volume_ratio >= SPY_VOLUME_BREAKOUT)
    distribution_day = (
        (simple_ret <= SPY_DISTRIBUTION_MIN_DROP)
        & (volume > volume.shift(1))
    )
    heavy_distribution_day = (
        (simple_ret <= SPY_HEAVY_DIST_MIN_DROP)
        & (volume_ratio >= params["strong_breakout"])
    )

    follow_through_day = pd.Series(False, index=spy.index)
    failed_rally = pd.Series(False, index=spy.index)
    rally_low = np.nan
    rally_age = 0
    in_rally_attempt = False

    for i, dt in enumerate(spy.index):
        if i == 0:
            continue

        prior_rally_low = rally_low
        in_correction = (close.iloc[i] < ma.iloc[i]) or (drawdown.iloc[i] <= SPY_CORRECTION_DRAWDOWN)

        # Order matters: compare today's low with the rally low known before
        # today's bar is incorporated. Updating rally_low first would make the
        # undercut comparison self-referential and can prevent failed_rally
        # from ever triggering.
        undercut_prior_low = (
            in_rally_attempt
            and not np.isnan(prior_rally_low)
            and low.iloc[i] < prior_rally_low
        )
        if undercut_prior_low:
            failed_rally.iloc[i] = True
            in_rally_attempt = False
            rally_age = 0
            rally_low = np.nan

        starts_rally = (not in_rally_attempt) and in_correction and (simple_ret.iloc[i] > 0)
        if starts_rally:
            in_rally_attempt = True
            rally_age = 1
            rally_low = low.iloc[i]
        elif in_rally_attempt:
            rally_age += 1
            # Update only after the undercut check above. This preserves the
            # prior rally-low test and avoids look-ahead within the same bar.
            rally_low = min(prior_rally_low, low.iloc[i]) if not np.isnan(prior_rally_low) else low.iloc[i]

        is_ftd = (
            in_rally_attempt
            and rally_age >= SPY_FTD_MIN_DAY
            and simple_ret.iloc[i] >= params["ftd_min_return"]
            and volume.iloc[i] > volume.iloc[i - 1]
        )
        if is_ftd:
            follow_through_day.iloc[i] = True
            in_rally_attempt = False
            rally_age = 0
            rally_low = np.nan

    recent_distribution_count = distribution_day.rolling(
        SPY_DISTRIBUTION_WINDOW, min_periods=1
    ).sum()

    score = pd.Series(0.0, index=spy.index)
    score = score + volume_breakout.astype(float) * 0.25
    score = score + follow_through_day.astype(float) * 1.00
    score = score - distribution_day.astype(float) * 0.35
    score = score - heavy_distribution_day.astype(float) * 0.75
    score = score - (recent_distribution_count >= SPY_DISTRIBUTION_CLUSTER).astype(float) * 0.25
    score = score.clip(-1.0, 1.0).rename("volume_confirmation_score")

    return pd.DataFrame({
        "spy_return": simple_ret,
        "spy_volume_ratio": volume_ratio,
        "volume_breakout": volume_breakout.fillna(False),
        "follow_through_day": follow_through_day,
        "distribution_day": distribution_day.fillna(False),
        "heavy_distribution_day": heavy_distribution_day.fillna(False),
        "failed_rally": failed_rally,
        "recent_distribution_count": recent_distribution_count,
        "volume_confirmation_score": score,
    })


def compute_regime(
    close: pd.DataFrame,
    data: dict | None = None,
    use_volume_confirmation: bool = USE_SPY_VOLUME_CONFIRMATION,
    volume_params: dict | None = None,
) -> pd.DataFrame:
    price_composite = compute_price_composite(close)
    base_scale = ((price_composite + 1.0) / 2.0).clip(0.0, 1.0).rename("base_scale")

    if use_volume_confirmation and data and MARKET_PROXY in data:
        volume = compute_spy_volume_confirmation(data[MARKET_PROXY], volume_params).reindex(close.index)
    else:
        volume = pd.DataFrame(index=close.index)
        volume["volume_confirmation_score"] = 0.0
        for col in [
            "volume_breakout", "follow_through_day", "distribution_day",
            "heavy_distribution_day", "failed_rally",
        ]:
            volume[col] = False
        volume["spy_return"] = np.nan
        volume["spy_volume_ratio"] = np.nan
        volume["recent_distribution_count"] = 0.0

    params = volume_params or _volume_params()
    weight = params["volume_weight"] if use_volume_confirmation else 0.0
    final_composite = (
        (1.0 - weight) * price_composite
        + weight * volume["volume_confirmation_score"].fillna(0.0)
    ).clip(-1.0, 1.0).rename("final_composite")
    final_scale = ((final_composite + 1.0) / 2.0).clip(0.0, 1.0).rename("final_scale")

    return pd.concat(
        [
            price_composite,
            base_scale,
            volume,
            final_composite,
            final_scale,
        ],
        axis=1,
    )


def compute_composite(close: pd.DataFrame, data: dict | None = None) -> pd.Series:
    """Backward-compatible helper returning the v3 final composite in [-1, +1]."""
    return compute_regime(close, data)["final_composite"]


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
    avail = [s for s in symbols if s in ret_window.columns]
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

    # Final renormalization: capping loop may not fully converge when many
    # names hit the cap simultaneously. Without this, weights silently sum
    # to < 1 causing unintended cash drag with no log entry.
    total = w.sum()
    if total > 1e-9:
        w = w / total
    else:
        global _RISK_PARITY_ZERO_WARNED
        if not _RISK_PARITY_ZERO_WARNED:
            logger.warning("risk_parity_weights: all weights zero — returning equal weight")
            _RISK_PARITY_ZERO_WARNED = True
        w[:] = 1.0 / len(avail)

    return w.reindex(symbols, fill_value=0.0)


def _walk_forward_weights(
    returns: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    scale: pd.Series,
) -> pd.DataFrame:
    base_weights = _walk_forward_base_weights(returns, tradeable, dates)
    return base_weights.mul(scale.reindex(dates).fillna(0.0), axis=0)


def _walk_forward_base_weights(
    returns: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    wh = pd.DataFrame(0.0, index=dates, columns=tradeable)
    for i in range(RISK_LOOKBACK_DAYS, len(dates)):
        ret_window = returns[tradeable].iloc[i - RISK_LOOKBACK_DAYS : i]
        wh.iloc[i] = risk_parity_weights(ret_window, tradeable)
    return wh


def _market_impact_cost(wh: pd.DataFrame, dollar_volume: pd.DataFrame) -> pd.Series:
    delta_w = wh.diff().abs().fillna(wh.abs())
    adv = dollar_volume.reindex(columns=wh.columns).rolling(
        ADV_LOOKBACK_DAYS, min_periods=ADV_LOOKBACK_DAYS // 2
    ).mean()
    participation = (delta_w * PORTFOLIO_USD) / adv.replace(0.0, np.nan)
    impact_rate = MARKET_IMPACT_COEFF * np.sqrt(participation.clip(lower=0.0))
    return (delta_w * impact_rate).sum(axis=1).fillna(0.0).rename("market_impact")


def _portfolio_returns(
    wh: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    apply_costs: bool = True,
    cost_bps: float = TRANSACTION_COST_BPS,
    include_market_impact: bool = False,
) -> pd.DataFrame:
    # Target weights indexed by signal_date are applied to the next return
    # period. This explicit shift is the execution-lag control: a signal
    # observed at today's close/volume cannot earn today's close-to-close
    # return.
    #
    # The loop is self-financing. If the target row is unchanged, the portfolio
    # holds existing positions and allows weights to drift with price moves;
    # turnover is therefore zero. If the target changes, we trade from the
    # drifted current weights to the new target weights.
    fwd = returns[wh.columns].shift(-1).reindex(wh.index).fillna(0.0)
    target_weights = wh.fillna(0.0)
    target_values = target_weights.to_numpy(dtype=float)
    return_values = fwd.to_numpy(dtype=float)
    current = np.zeros(target_values.shape[1], dtype=float)
    previous_target = None
    gross_values = np.zeros(len(target_weights), dtype=float)
    turnover_values = np.zeros(len(target_weights), dtype=float)

    for i in range(len(target_weights)):
        target = target_values[i]
        target_changed = (
            previous_target is None
            or np.max(np.abs(target - previous_target)) > 1e-12
        )
        if target_changed:
            turnover_values[i] = np.abs(target - current).sum()
            post_trade = target.copy()
            previous_target = target.copy()
        else:
            post_trade = current.copy()

        asset_return = return_values[i]
        gross_today = float((post_trade * asset_return).sum())
        gross_values[i] = gross_today
        denominator = 1.0 + gross_today
        if denominator > 1e-12:
            current = post_trade * (1.0 + asset_return) / denominator
        else:
            current = np.zeros_like(current)

    gross = pd.Series(gross_values, index=target_weights.index, name="gross_return")
    turnover = pd.Series(turnover_values, index=target_weights.index, name="turnover")
    transaction_cost = (turnover * cost_bps / 10_000).rename("transaction_cost")
    if include_market_impact:
        market_impact = _market_impact_cost(wh, dollar_volume)
    else:
        market_impact = pd.Series(0.0, index=wh.index, name="market_impact")
    total_cost = (transaction_cost + market_impact).rename("total_cost")
    net = gross - total_cost if apply_costs else gross
    return pd.concat(
        [
            gross.rename("gross_return"),
            net.rename("net_return"),
            turnover,
            transaction_cost,
            market_impact,
            total_cost,
        ],
        axis=1,
    )


def _performance_metrics(port_r: pd.Series, turnover: pd.Series | None = None) -> dict:
    port_r = port_r.dropna()
    avg_turnover = float(turnover.reindex(port_r.index).mean()) if turnover is not None and len(port_r) else 0.0
    n = len(port_r)
    if n == 0:
        return {
            "sharpe": 0.0, "calmar": 0.0, "cagr": 0.0, "total_return": 0.0,
            "max_drawdown": 0.0, "annual_vol": 0.0, "win_rate": 0.0,
            "avg_daily_turnover": 0.0, "final_capital": PORTFOLIO_USD, "n_days": 0,
        }

    ann_ret = float(port_r.mean() * TRADING_DAYS)
    ann_vol = float(port_r.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + port_r).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    roll_max = cum.cummax()
    max_dd = float(((cum - roll_max) / roll_max).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    win_rate = float((port_r > 0).sum() / n)
    years = n / TRADING_DAYS
    cagr = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    final_cap = PORTFOLIO_USD * (1 + total_ret)

    return {
        "sharpe":        round(sharpe,    3),
        "calmar":        round(calmar,    3),
        "cagr":          round(cagr,      4),
        "total_return":  round(total_ret, 4),
        "max_drawdown":  round(max_dd,    4),
        "annual_vol":    round(ann_vol,   4),
        "win_rate":      round(win_rate,  4),
        "avg_daily_turnover": round(avg_turnover, 4),
        "final_capital": round(final_cap, 2),
        "n_days":        n,
    }


def _equal_weight_weights(dates: pd.DatetimeIndex, tradeable: list) -> pd.DataFrame:
    wh = pd.DataFrame(0.0, index=dates, columns=tradeable)
    if tradeable:
        wh.iloc[RISK_LOOKBACK_DAYS:] = 1.0 / len(tradeable)
    return wh


def _apply_rebalance_policy(
    target_weights: pd.DataFrame,
    scale: pd.Series,
    policy: str = FINAL_REBALANCE_POLICY,
) -> pd.DataFrame:
    """
    Convert daily target weights into executable weights under a rebalancing rule.

    Signals remain lagged exactly as before: the row indexed by signal_date is
    still applied only to the next return period inside _portfolio_returns().
    This function only controls how often the target weight is refreshed.
    """
    if policy not in REBALANCE_POLICY_GRID:
        raise ValueError(f"Unknown rebalance policy: {policy}")
    target = target_weights.fillna(0.0)
    if policy == "daily":
        return target

    scale = scale.reindex(target.index).fillna(0.0)
    held = pd.DataFrame(0.0, index=target.index, columns=target.columns)
    current = pd.Series(0.0, index=target.columns)
    previous_week = None
    previous_scale = None

    for dt in target.index:
        row = target.loc[dt]
        scale_today = float(scale.loc[dt])
        if hasattr(dt, "isocalendar"):
            iso = dt.isocalendar()
            week_key = (int(iso.year), int(iso.week))
        else:
            week_key = pd.Timestamp(dt).to_period("W")

        has_live_target = float(row.abs().sum()) > 1e-12
        needs_initial_trade = has_live_target and float(current.abs().sum()) <= 1e-12

        if policy == "weekly":
            should_rebalance = needs_initial_trade or week_key != previous_week
        else:
            scale_changed = (
                previous_scale is not None
                and abs(scale_today - previous_scale) > 1e-12
            )
            should_rebalance = needs_initial_trade or scale_changed

        if should_rebalance:
            current = row.copy()

        held.loc[dt] = current
        previous_week = week_key
        previous_scale = scale_today

    return held


def _strategy_weight_map(
    returns: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    regime: pd.DataFrame,
    base_rp_weights: pd.DataFrame | None = None,
    rebalance_policy: str = FINAL_REBALANCE_POLICY,
) -> dict:
    base_rp_weights = base_rp_weights if base_rp_weights is not None else _walk_forward_base_weights(
        returns, tradeable, dates
    )
    base_target = base_rp_weights.mul(regime["base_scale"].reindex(dates), axis=0)
    v3_target = base_rp_weights.mul(regime["final_scale"].reindex(dates), axis=0)
    return {
        "equal_weight_10_stock": _equal_weight_weights(dates, tradeable),
        "risk_parity_no_overlay": base_rp_weights,
        "base_price_only": _apply_rebalance_policy(
            base_target, regime["base_scale"], rebalance_policy
        ),
        "price_plus_spy_volume_confirmation": _apply_rebalance_policy(
            v3_target, regime["final_scale"], rebalance_policy
        ),
    }


def _strategy_return_map(
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    weights: dict,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> dict:
    frames = {}
    for name, wh in weights.items():
        frame = _portfolio_returns(
            wh,
            returns,
            dollar_volume,
            apply_costs=True,
            cost_bps=cost_bps,
            include_market_impact=False,
        )
        frames[name] = frame.iloc[RISK_LOOKBACK_DAYS:-1].dropna()
    return frames


def _metrics_table(
    strategy_returns: dict,
    spy_returns: pd.Series,
    sample_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    rows = []
    if sample_index is not None:
        sample_index = pd.DatetimeIndex(sample_index)

    spy = spy_returns.copy()
    if sample_index is not None:
        spy = spy.reindex(sample_index).dropna()
    spy_metrics = _performance_metrics(spy)
    rows.append({"strategy": "spy_buy_hold", **spy_metrics})

    for name, frame in strategy_returns.items():
        local = frame if sample_index is None else frame.reindex(sample_index).dropna()
        rows.append({
            "strategy": name,
            **_performance_metrics(local["net_return"], local["turnover"]),
        })
    return pd.DataFrame(rows)


def _signal_frequency_stats(regime: pd.DataFrame) -> dict:
    n_days = max(len(regime), 1)
    years = n_days / TRADING_DAYS
    stats = {}
    for col in ["follow_through_day", "distribution_day", "heavy_distribution_day"]:
        count = int(regime[col].fillna(False).sum())
        stats[f"{col}_count"] = count
        stats[f"{col}_annualized"] = round(count / years, 2) if years > 0 else 0.0
    stats["volume_confirmation_mean"] = float(regime["volume_confirmation_score"].mean())
    stats["volume_confirmation_std"] = float(regime["volume_confirmation_score"].std())
    stats["volume_confirmation_min"] = float(regime["volume_confirmation_score"].min())
    stats["volume_confirmation_max"] = float(regime["volume_confirmation_score"].max())
    stats["final_scale_mean"] = float(regime["final_scale"].mean())
    stats["final_scale_std"] = float(regime["final_scale"].std())
    stats["final_scale_min"] = float(regime["final_scale"].min())
    stats["final_scale_max"] = float(regime["final_scale"].max())
    stats["final_scale_below_50pct"] = float((regime["final_scale"] < 0.50).mean())
    return stats


def _log_signal_warnings(signal_stats: dict):
    ftd_freq = signal_stats["follow_through_day_annualized"]
    if ftd_freq < 1:
        logger.warning("Follow-Through Days are almost zero; the rule may be too strict.")
    if ftd_freq > 20:
        logger.warning("Follow-Through Days are very frequent; the rule may be too loose.")
    if signal_stats["final_scale_below_50pct"] < 0.05:
        logger.warning("final_scale rarely de-risks below 50%; drawdown control may be weak.")


def _build_diagnostics(
    regime: pd.DataFrame,
    base_frame: pd.DataFrame,
    v3_frame: pd.DataFrame,
    opportunity_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    idx = base_frame.index.intersection(v3_frame.index)
    next_dates = pd.Series(idx, index=idx).shift(-1)
    diagnostics = pd.DataFrame(index=idx)
    diagnostics["signal_date"] = idx
    diagnostics["next_return_date"] = next_dates
    diagnostics["execution_lag_used"] = diagnostics["next_return_date"].notna()
    diagnostics["base_scale"] = regime["base_scale"].reindex(idx)
    diagnostics["final_scale"] = regime["final_scale"].reindex(idx)
    diagnostics["base_strategy_return"] = base_frame["net_return"].reindex(idx)
    diagnostics["v3_strategy_return"] = v3_frame["net_return"].reindex(idx)
    if opportunity_detail is not None:
        for col in [
            "risk_parity_reference_return", "derisk_amount", "add_risk_amount",
            "opportunity_cost", "loss_avoided", "upside_captured", "extra_loss",
            "net_defensive_timing_benefit", "net_total_timing_benefit",
        ]:
            diagnostics[col] = opportunity_detail[col].reindex(idx)
    for col in [
        "volume_confirmation_score", "volume_breakout", "follow_through_day",
        "distribution_day", "heavy_distribution_day", "failed_rally",
        "recent_distribution_count", "spy_volume_ratio",
    ]:
        diagnostics[col] = regime[col].reindex(idx)
    return diagnostics.dropna(subset=["next_return_date"])


def _opportunity_cost_detail(
    regime: pd.DataFrame,
    strategy_returns: dict,
    strategy_name: str = "price_plus_spy_volume_confirmation",
    strategy_scale_col: str = "final_scale",
    baseline_name: str = "base_price_only",
    baseline_scale_col: str = "base_scale",
) -> pd.DataFrame:
    """
    Decompose v3-vs-base exposure changes into missed upside and avoided loss.

    The reference return is the next-period gross return of the risk-parity
    portfolio before any regime overlay. This isolates the timing effect of
    changing exposure scale rather than mixing in individual-stock selection.
    """
    idx = (
        strategy_returns[baseline_name].index
        .intersection(strategy_returns[strategy_name].index)
        .intersection(strategy_returns["risk_parity_no_overlay"].index)
    )
    base_scale = regime[baseline_scale_col].reindex(idx)
    final_scale = regime[strategy_scale_col].reindex(idx)
    rp_return = strategy_returns["risk_parity_no_overlay"]["gross_return"].reindex(idx)

    derisk_amount = (base_scale - final_scale).clip(lower=0.0)
    add_risk_amount = (final_scale - base_scale).clip(lower=0.0)
    positive_ref = rp_return.clip(lower=0.0)
    negative_ref = (-rp_return).clip(lower=0.0)

    detail = pd.DataFrame(index=idx)
    detail["signal_date"] = idx
    detail["base_scale"] = base_scale
    detail["final_scale"] = final_scale
    detail["scale_delta_v3_minus_base"] = final_scale - base_scale
    detail["risk_parity_reference_return"] = rp_return
    detail["derisk_amount"] = derisk_amount
    detail["add_risk_amount"] = add_risk_amount
    detail["opportunity_cost"] = derisk_amount * positive_ref
    detail["loss_avoided"] = derisk_amount * negative_ref
    detail["upside_captured"] = add_risk_amount * positive_ref
    detail["extra_loss"] = add_risk_amount * negative_ref
    detail["net_defensive_timing_benefit"] = detail["loss_avoided"] - detail["opportunity_cost"]
    detail["net_total_timing_benefit"] = (
        detail["loss_avoided"]
        - detail["opportunity_cost"]
        + detail["upside_captured"]
        - detail["extra_loss"]
    )
    detail["strategy"] = strategy_name
    detail["baseline_strategy"] = baseline_name
    detail["actual_strategy_minus_base_return"] = (
        strategy_returns[strategy_name]["net_return"].reindex(idx)
        - strategy_returns[baseline_name]["net_return"].reindex(idx)
    )
    detail["strategy_derisked_vs_base"] = derisk_amount > 0
    detail["strategy_added_risk_vs_base"] = add_risk_amount > 0
    return detail


def _summarize_opportunity_cost(detail: pd.DataFrame) -> pd.DataFrame:
    idx = detail.index
    split = int(len(idx) * 0.70)
    samples = {
        "full_sample": idx,
        "in_sample": idx[:split],
        "out_of_sample": idx[split:],
    }
    rows = []
    for sample, sample_idx in samples.items():
        local = detail.reindex(sample_idx).dropna()
        derisked = local[local["strategy_derisked_vs_base"]]
        added = local[local["strategy_added_risk_vs_base"]]
        total_oc = float(local["opportunity_cost"].sum())
        total_la = float(local["loss_avoided"].sum())
        total_upside = float(local["upside_captured"].sum())
        total_extra_loss = float(local["extra_loss"].sum())
        rows.append({
            "sample": sample,
            "strategy": local["strategy"].iloc[0] if len(local) else "",
            "n_days": len(local),
            "derisked_day_count": int(local["strategy_derisked_vs_base"].sum()),
            "added_risk_day_count": int(local["strategy_added_risk_vs_base"].sum()),
            "total_opportunity_cost": round(total_oc, 4),
            "total_loss_avoided": round(total_la, 4),
            "net_defensive_timing_benefit": round(total_la - total_oc, 4),
            "opportunity_cost_ratio": round(total_oc / (total_la + 1e-12), 4),
            "total_upside_captured": round(total_upside, 4),
            "total_extra_loss": round(total_extra_loss, 4),
            "net_total_timing_benefit": round(
                total_la - total_oc + total_upside - total_extra_loss, 4
            ),
            "actual_strategy_minus_base_return": round(float(local["actual_strategy_minus_base_return"].sum()), 4),
            "derisked_positive_return_hit_rate": round(
                float((derisked["risk_parity_reference_return"] > 0).mean())
                if len(derisked) else 0.0,
                4,
            ),
            "derisked_negative_return_hit_rate": round(
                float((derisked["risk_parity_reference_return"] < 0).mean())
                if len(derisked) else 0.0,
                4,
            ),
            "added_risk_positive_return_hit_rate": round(
                float((added["risk_parity_reference_return"] > 0).mean())
                if len(added) else 0.0,
                4,
            ),
            "added_risk_negative_return_hit_rate": round(
                float((added["risk_parity_reference_return"] < 0).mean())
                if len(added) else 0.0,
                4,
            ),
        })
    return pd.DataFrame(rows)


def _cost_sensitivity_table(
    weights: dict,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    spy_returns: pd.Series,
) -> pd.DataFrame:
    rows = []
    for cost_bps in COST_SCENARIOS_BPS:
        frames = _strategy_return_map(returns, dollar_volume, weights, cost_bps=cost_bps)
        table = _metrics_table(frames, spy_returns)
        table["cost_bps"] = cost_bps
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _rebalance_policy_tables(
    base_rp_weights: pd.DataFrame,
    regime: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    base_target = base_rp_weights.mul(regime["base_scale"].reindex(dates), axis=0)
    rows = []
    cost_rows = []
    oos_rows = []
    subperiod_rows = []
    turnover_rows = []
    drawdown_rows = []
    annual_turnover_rows = []
    trade_attribution_rows = []
    scale = regime["base_scale"].reindex(dates).fillna(0.0)
    scale_changes = int((scale.diff().abs() > 1e-12).sum())
    policy_frames = {}
    policy_weights = {}

    for policy in REBALANCE_POLICY_GRID:
        wh = _apply_rebalance_policy(base_target, scale, policy)
        policy_weights[policy] = wh
        for cost_bps in COST_SCENARIOS_BPS:
            frame = _portfolio_returns(
                wh,
                returns,
                dollar_volume,
                apply_costs=True,
                cost_bps=cost_bps,
                include_market_impact=False,
            ).iloc[RISK_LOOKBACK_DAYS:-1].dropna()
            metrics = _performance_metrics(frame["net_return"], frame["turnover"])
            trade_days = int((frame["turnover"] > 1e-12).sum())
            row = {
                "rebalance_policy": policy,
                "cost_bps": cost_bps,
                **metrics,
                "annualized_turnover": round(metrics["avg_daily_turnover"] * TRADING_DAYS, 4),
                "trade_day_count": trade_days,
                "trade_days_per_year": round(trade_days / (len(frame) / TRADING_DAYS), 2)
                if len(frame) else 0.0,
                "scale_change_count": scale_changes,
            }
            cost_rows.append(row)
            if cost_bps == TRANSACTION_COST_BPS:
                rows.append(row)
                policy_frames[policy] = frame

    full_index = next(iter(policy_frames.values())).index if policy_frames else pd.DatetimeIndex([])
    split = int(len(full_index) * 0.70)
    samples = {
        "full_sample": full_index,
        "in_sample": full_index[:split],
        "out_of_sample": full_index[split:],
    }
    periods = {
        "2010_2016": ("2010-01-01", "2016-12-31"),
        "2017_2020": ("2017-01-01", "2020-12-31"),
        "2021_plus": ("2021-01-01", None),
    }

    for policy, frame in policy_frames.items():
        for sample, idx in samples.items():
            local = frame.reindex(idx).dropna()
            oos_rows.append({
                "rebalance_policy": policy,
                "sample": sample,
                **_performance_metrics(local["net_return"], local["turnover"]),
            })

        for period, (start, end) in periods.items():
            idx = frame.index[frame.index >= pd.Timestamp(start, tz="UTC")]
            if end:
                idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
            local = frame.reindex(idx).dropna()
            if local.empty:
                continue
            subperiod_rows.append({
                "rebalance_policy": policy,
                "period": period,
                **_performance_metrics(local["net_return"], local["turnover"]),
            })

        turnover = frame["turnover"].dropna()
        trade_turnover = turnover[turnover > 1e-12]
        n_years = len(frame) / TRADING_DAYS if len(frame) else 0.0
        target_exposure = policy_weights[policy].abs().sum(axis=1).reindex(frame.index)
        turnover_rows.append({
            "rebalance_policy": policy,
            "n_days": len(frame),
            "trade_day_count": int((turnover > 1e-12).sum()),
            "trade_days_per_year": round(int((turnover > 1e-12).sum()) / n_years, 2)
            if n_years else 0.0,
            "scale_change_count": scale_changes,
            "avg_daily_turnover": round(float(turnover.mean()), 4),
            "annualized_turnover": round(float(turnover.mean()) * TRADING_DAYS, 4),
            "median_turnover": round(float(turnover.median()), 4),
            "avg_turnover_on_trade_days": round(float(trade_turnover.mean()), 4)
            if len(trade_turnover) else 0.0,
            "p75_turnover": round(float(turnover.quantile(0.75)), 4),
            "p90_turnover": round(float(turnover.quantile(0.90)), 4),
            "p95_turnover": round(float(turnover.quantile(0.95)), 4),
            "p99_turnover": round(float(turnover.quantile(0.99)), 4),
            "max_turnover": round(float(turnover.max()), 4),
            "share_turnover_gt_1pct": round(float((turnover > 0.01).mean()), 4),
            "share_turnover_gt_5pct": round(float((turnover > 0.05).mean()), 4),
            "share_turnover_gt_10pct": round(float((turnover > 0.10).mean()), 4),
            "share_turnover_gt_20pct": round(float((turnover > 0.20).mean()), 4),
            "avg_target_exposure": round(float(target_exposure.mean()), 4),
        })

        annual = pd.DataFrame({"turnover": turnover})
        annual["year"] = annual.index.year
        for year, year_data in annual.groupby("year"):
            annual_turnover_rows.append({
                "rebalance_policy": policy,
                "year": int(year),
                "n_days": int(len(year_data)),
                "trade_day_count": int((year_data["turnover"] > 1e-12).sum()),
                "avg_daily_turnover": round(float(year_data["turnover"].mean()), 4),
                "annualized_turnover": round(float(year_data["turnover"].mean()) * TRADING_DAYS, 4),
                "total_turnover": round(float(year_data["turnover"].sum()), 4),
                "max_daily_turnover": round(float(year_data["turnover"].max()), 4),
            })

        prev_target = None
        prev_scale = None
        prev_rp = None
        attribution = {
            "initial_allocation": {"count": 0, "turnover": 0.0},
            "scale_only": {"count": 0, "turnover": 0.0},
            "risk_parity_only": {"count": 0, "turnover": 0.0},
            "scale_and_risk_parity": {"count": 0, "turnover": 0.0},
        }
        for dt in frame.index:
            turnover_today = float(frame.loc[dt, "turnover"])
            if turnover_today <= 1e-12:
                continue
            target_today = policy_weights[policy].loc[dt]
            rp_today = base_rp_weights.loc[dt]
            scale_today = float(scale.loc[dt])
            if prev_target is None or float(prev_target.abs().sum()) <= 1e-12:
                reason = "initial_allocation"
            else:
                scale_changed = abs(scale_today - prev_scale) > 1e-12
                rp_changed = float((rp_today - prev_rp).abs().max()) > 1e-12
                if scale_changed and rp_changed:
                    reason = "scale_and_risk_parity"
                elif scale_changed:
                    reason = "scale_only"
                elif rp_changed:
                    reason = "risk_parity_only"
                else:
                    # The target can be unchanged while actual weights drift;
                    # this bucket is intentionally mapped to risk parity only
                    # because the executed target was not triggered by scale.
                    reason = "risk_parity_only"
            attribution[reason]["count"] += 1
            attribution[reason]["turnover"] += turnover_today
            prev_target = target_today.copy()
            prev_scale = scale_today
            prev_rp = rp_today.copy()

        total_attributed_turnover = sum(v["turnover"] for v in attribution.values())
        for reason, values in attribution.items():
            trade_attribution_rows.append({
                "rebalance_policy": policy,
                "trade_reason": reason,
                "trade_day_count": values["count"],
                "total_turnover": round(values["turnover"], 4),
                "share_of_turnover": round(
                    values["turnover"] / (total_attributed_turnover + 1e-12), 4
                ),
            })

    spy = close[MARKET_PROXY].dropna()
    spy_dd = spy / spy.cummax() - 1.0
    in_dd = spy_dd <= -0.10
    episodes = []
    start = None
    prev_dt = None
    for dt, active in in_dd.items():
        if active and start is None:
            start = dt
        elif not active and start is not None:
            episodes.append((start, prev_dt))
            start = None
        prev_dt = dt
    if start is not None:
        episodes.append((start, spy_dd.index[-1]))

    for episode_no, (start_dt, end_dt) in enumerate(episodes, start=1):
        for policy, frame in policy_frames.items():
            idx = frame.index[(frame.index >= start_dt) & (frame.index <= end_dt)]
            if len(idx) == 0:
                continue
            local = frame.reindex(idx).dropna()
            local_spy_dd = spy_dd.reindex(idx).dropna()
            exposure = policy_weights[policy].abs().sum(axis=1).reindex(idx)
            prior_idx = policy_weights[policy].index[policy_weights[policy].index < start_dt]
            prior_exposure = (
                float(policy_weights[policy].abs().sum(axis=1).reindex(prior_idx).iloc[-1])
                if len(prior_idx) else np.nan
            )
            exposure_values = exposure.dropna()
            first_derisk_date = pd.NaT
            days_to_first_derisk = np.nan
            min_exposure_date = pd.NaT
            days_to_min_exposure = np.nan
            first_rerisk_date = pd.NaT
            days_to_first_rerisk_after_min = np.nan
            if len(exposure_values):
                if not np.isnan(prior_exposure):
                    derisk = exposure_values < prior_exposure - 0.05
                    if derisk.any():
                        first_derisk_date = derisk[derisk].index[0]
                        days_to_first_derisk = int(exposure_values.index.get_loc(first_derisk_date))
                min_exposure_date = exposure_values.idxmin()
                days_to_min_exposure = int(exposure_values.index.get_loc(min_exposure_date))
                after_min = exposure_values.loc[min_exposure_date:]
                rerisk = after_min > float(exposure_values.loc[min_exposure_date]) + 0.05
                if rerisk.any():
                    first_rerisk_date = rerisk[rerisk].index[0]
                    days_to_first_rerisk_after_min = int(
                        exposure_values.index.get_loc(first_rerisk_date)
                        - exposure_values.index.get_loc(min_exposure_date)
                    )
            drawdown_rows.append({
                "episode": episode_no,
                "rebalance_policy": policy,
                "start_date": start_dt,
                "end_date": end_dt,
                "n_days": len(idx),
                "spy_min_drawdown": round(float(local_spy_dd.min()), 4)
                if len(local_spy_dd) else 0.0,
                "strategy_episode_return": round(float((1.0 + local["net_return"]).prod() - 1.0), 4)
                if len(local) else 0.0,
                "strategy_episode_max_drawdown": round(_max_drawdown_from_returns(local["net_return"]), 4),
                "avg_target_exposure": round(float(exposure.mean()), 4),
                "min_target_exposure": round(float(exposure.min()), 4),
                "max_target_exposure": round(float(exposure.max()), 4),
                "prior_target_exposure": round(prior_exposure, 4)
                if not np.isnan(prior_exposure) else np.nan,
                "first_derisk_date": first_derisk_date,
                "days_to_first_derisk": days_to_first_derisk,
                "min_exposure_date": min_exposure_date,
                "days_to_min_exposure": days_to_min_exposure,
                "first_rerisk_date": first_rerisk_date,
                "days_to_first_rerisk_after_min": days_to_first_rerisk_after_min,
                "total_turnover": round(float(local["turnover"].sum()), 4),
                "trade_day_count": int((local["turnover"] > 1e-12).sum()),
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(cost_rows),
        pd.DataFrame(oos_rows),
        pd.DataFrame(subperiod_rows),
        pd.DataFrame(turnover_rows),
        pd.DataFrame(drawdown_rows),
        pd.DataFrame(annual_turnover_rows),
        pd.DataFrame(trade_attribution_rows),
    )


def _oos_metrics_table(strategy_returns: dict, spy_returns: pd.Series) -> pd.DataFrame:
    common_index = strategy_returns["base_price_only"].index
    split = int(len(common_index) * 0.70)
    samples = {
        "full_sample": common_index,
        "in_sample": common_index[:split],
        "out_of_sample": common_index[split:],
    }
    rows = []
    for sample, idx in samples.items():
        table = _metrics_table(strategy_returns, spy_returns, idx)
        table["sample"] = sample
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _parameter_sensitivity_table(
    close: pd.DataFrame,
    data: dict,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    base_rp_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ftd_min_return in SPY_FTD_MIN_RETURN_GRID:
        for strong_breakout in SPY_VOLUME_STRONG_BREAKOUT_GRID:
            for volume_weight in VOLUME_CONFIRMATION_WEIGHT_GRID:
                params = _volume_params(ftd_min_return, strong_breakout, volume_weight)
                regime = compute_regime(close, data, True, params)
                target = base_rp_weights.mul(regime["final_scale"].reindex(dates), axis=0)
                wh = _apply_rebalance_policy(target, regime["final_scale"])
                frame = _portfolio_returns(
                    wh, returns, dollar_volume, True, TRANSACTION_COST_BPS, False
                ).iloc[RISK_LOOKBACK_DAYS:-1].dropna()
                metrics = _performance_metrics(frame["net_return"], frame["turnover"])
                rows.append({
                    "SPY_FTD_MIN_RETURN": ftd_min_return,
                    "SPY_VOLUME_STRONG_BREAKOUT": strong_breakout,
                    "VOLUME_CONFIRMATION_WEIGHT": volume_weight,
                    "sharpe": metrics["sharpe"],
                    "calmar": metrics["calmar"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_return": metrics["total_return"],
                    "follow_through_day_count": int(regime["follow_through_day"].sum()),
                    "distribution_day_count": int(regime["distribution_day"].sum()),
                    "average_final_scale": round(float(regime["final_scale"].mean()), 4),
                })
    return pd.DataFrame(rows)


def _base_ablation_table(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    base_rp_weights: pd.DataFrame,
) -> pd.DataFrame:
    components = compute_price_components(close)
    rows = []
    final_cols = ["vix_proxy", "equity_mom", "breadth", "vol_regime"]
    specs = {"defensive_4_signal_base": final_cols}
    specs.update({f"without_{col}": [c for c in final_cols if c != col] for col in final_cols})
    specs["legacy_5_signal_with_cross_mom"] = list(components.columns)
    for spec_name, cols in specs.items():
        composite = components[cols].mean(axis=1).clip(-1.0, 1.0).rename("price_composite")
        scale = _scale_from_composite(composite)
        target = base_rp_weights.mul(scale.reindex(dates), axis=0)
        wh = _apply_rebalance_policy(target, scale)
        frame = _portfolio_returns(
            wh, returns, dollar_volume, True, TRANSACTION_COST_BPS, False
        ).iloc[RISK_LOOKBACK_DAYS:-1].dropna()
        rows.append({
            "specification": spec_name,
            "included_signals": ",".join(cols),
            **_performance_metrics(frame["net_return"], frame["turnover"]),
            "average_scale": round(float(scale.reindex(frame.index).mean()), 4),
        })
    return pd.DataFrame(rows)


def _base_parameter_sensitivity_table(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    base_rp_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ma_long in BASE_MA_LONG_GRID:
        for breadth_ma in BASE_BREADTH_MA_GRID:
            for vol_short in BASE_VOL_SHORT_GRID:
                for vol_long in BASE_VOL_LONG_GRID:
                    if vol_short >= vol_long:
                        continue
                    composite = compute_price_composite_with_params(
                        close,
                        ma_long=ma_long,
                        breadth_ma=breadth_ma,
                        vol_short=vol_short,
                        vol_long=vol_long,
                    )
                    scale = _scale_from_composite(composite)
                    target = base_rp_weights.mul(scale.reindex(dates), axis=0)
                    wh = _apply_rebalance_policy(target, scale)
                    frame = _portfolio_returns(
                        wh, returns, dollar_volume, True, TRANSACTION_COST_BPS, False
                    ).iloc[RISK_LOOKBACK_DAYS:-1].dropna()
                    metrics = _performance_metrics(frame["net_return"], frame["turnover"])
                    rows.append({
                        "MA_LONG": ma_long,
                        "BREADTH_MA": breadth_ma,
                        "VOL_SHORT": vol_short,
                        "VOL_LONG": vol_long,
                        "sharpe": metrics["sharpe"],
                        "calmar": metrics["calmar"],
                        "cagr": metrics["cagr"],
                        "total_return": metrics["total_return"],
                        "max_drawdown": metrics["max_drawdown"],
                        "annual_vol": metrics["annual_vol"],
                        "avg_daily_turnover": metrics["avg_daily_turnover"],
                        "average_scale": round(float(scale.reindex(frame.index).mean()), 4),
                    })
    return pd.DataFrame(rows)


def _base_subperiod_metrics_table(strategy_returns: dict, spy_returns: pd.Series) -> pd.DataFrame:
    base_index = strategy_returns["base_price_only"].index
    periods = {
        "2010_2016": ("2010-01-01", "2016-12-31"),
        "2017_2020": ("2017-01-01", "2020-12-31"),
        "2021_plus": ("2021-01-01", None),
    }
    rows = []
    for period, (start, end) in periods.items():
        idx = base_index[base_index >= pd.Timestamp(start, tz="UTC")]
        if end:
            idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
        if len(idx) == 0:
            continue
        table = _metrics_table(strategy_returns, spy_returns, idx)
        table = table[table["strategy"].isin([
            "spy_buy_hold",
            "risk_parity_no_overlay",
            "base_price_only",
        ])]
        table["period"] = period
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _base_opportunity_cost_detail(
    regime: pd.DataFrame,
    strategy_returns: dict,
) -> pd.DataFrame:
    idx = (
        strategy_returns["base_price_only"].index
        .intersection(strategy_returns["risk_parity_no_overlay"].index)
    )
    base_scale = regime["base_scale"].reindex(idx)
    rp_return = strategy_returns["risk_parity_no_overlay"]["gross_return"].reindex(idx)
    derisk_amount = (1.0 - base_scale).clip(lower=0.0)
    positive_ref = rp_return.clip(lower=0.0)
    negative_ref = (-rp_return).clip(lower=0.0)

    detail = pd.DataFrame(index=idx)
    detail["signal_date"] = idx
    detail["base_scale"] = base_scale
    detail["risk_parity_reference_return"] = rp_return
    detail["base_derisk_amount"] = derisk_amount
    detail["opportunity_cost"] = derisk_amount * positive_ref
    detail["loss_avoided"] = derisk_amount * negative_ref
    detail["net_defensive_timing_benefit"] = detail["loss_avoided"] - detail["opportunity_cost"]
    detail["actual_base_minus_no_overlay_return"] = (
        strategy_returns["base_price_only"]["net_return"].reindex(idx)
        - strategy_returns["risk_parity_no_overlay"]["net_return"].reindex(idx)
    )
    detail["base_derisked_vs_no_overlay"] = derisk_amount > 0
    return detail


def _summarize_base_opportunity_cost(detail: pd.DataFrame) -> pd.DataFrame:
    idx = detail.index
    split = int(len(idx) * 0.70)
    samples = {
        "full_sample": idx,
        "in_sample": idx[:split],
        "out_of_sample": idx[split:],
    }
    rows = []
    for sample, sample_idx in samples.items():
        local = detail.reindex(sample_idx).dropna()
        derisked = local[local["base_derisked_vs_no_overlay"]]
        total_oc = float(local["opportunity_cost"].sum())
        total_la = float(local["loss_avoided"].sum())
        rows.append({
            "sample": sample,
            "n_days": len(local),
            "derisked_day_count": int(local["base_derisked_vs_no_overlay"].sum()),
            "total_opportunity_cost": round(total_oc, 4),
            "total_loss_avoided": round(total_la, 4),
            "net_defensive_timing_benefit": round(total_la - total_oc, 4),
            "opportunity_cost_ratio": round(total_oc / (total_la + 1e-12), 4),
            "actual_base_minus_no_overlay_return": round(
                float(local["actual_base_minus_no_overlay_return"].sum()), 4
            ),
            "derisked_positive_return_hit_rate": round(
                float((derisked["risk_parity_reference_return"] > 0).mean())
                if len(derisked) else 0.0,
                4,
            ),
            "derisked_negative_return_hit_rate": round(
                float((derisked["risk_parity_reference_return"] < 0).mean())
                if len(derisked) else 0.0,
                4,
            ),
        })
    return pd.DataFrame(rows)


def _max_drawdown_from_returns(ret: pd.Series) -> float:
    ret = ret.dropna()
    if ret.empty:
        return 0.0
    cum = (1.0 + ret).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def _cross_mom_research_tables(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    tradeable: list,
    dates: pd.DatetimeIndex,
    regime: pd.DataFrame,
    base_rp_weights: pd.DataFrame,
    strategy_returns: dict,
    spy_fwd_returns: pd.Series,
) -> dict:
    components = compute_price_components(close)
    legacy_composite = _composite_from_components(components, []).rename(
        "price_composite_with_cross_mom"
    )
    legacy_scale = _scale_from_composite(legacy_composite).rename("scale_with_cross_mom")
    final_composite = regime["price_composite"].rename("price_composite_without_cross_mom")
    final_scale = regime["base_scale"].rename("scale_without_cross_mom")

    legacy_target = base_rp_weights.mul(legacy_scale.reindex(dates), axis=0)
    legacy_weights = _apply_rebalance_policy(legacy_target, legacy_scale)
    legacy_returns = _portfolio_returns(
        legacy_weights,
        returns,
        dollar_volume,
        True,
        TRANSACTION_COST_BPS,
        False,
    ).iloc[RISK_LOOKBACK_DAYS:-1].dropna()

    cross_returns = {
        "base_price_only": strategy_returns["base_price_only"],
        "legacy_5_signal_with_cross_mom": legacy_returns,
    }
    comparison = _metrics_table(cross_returns, spy_fwd_returns)
    comparison = comparison[comparison["strategy"].isin([
        "base_price_only",
        "legacy_5_signal_with_cross_mom",
    ])]
    oos = _oos_metrics_table(cross_returns, spy_fwd_returns)
    oos = oos[oos["strategy"].isin(["base_price_only", "legacy_5_signal_with_cross_mom"])]
    subperiod = _cross_mom_subperiod_metrics(cross_returns, spy_fwd_returns)

    cross_regime = regime.copy()
    cross_regime["scale_with_cross_mom"] = legacy_scale
    extended_returns = dict(strategy_returns)
    extended_returns["legacy_5_signal_with_cross_mom"] = legacy_returns
    opportunity_detail = _opportunity_cost_detail(
        cross_regime,
        extended_returns,
        strategy_name="base_price_only",
        strategy_scale_col="base_scale",
        baseline_name="legacy_5_signal_with_cross_mom",
        baseline_scale_col="scale_with_cross_mom",
    )
    opportunity_summary = _summarize_opportunity_cost(opportunity_detail)

    drawdown_episodes, timing_diagnostics = _cross_mom_drawdown_analysis(
        close,
        cross_returns,
        legacy_composite,
        final_composite,
        legacy_scale,
        final_scale,
    )

    return {
        "comparison": comparison,
        "oos": oos,
        "subperiod": subperiod,
        "opportunity_detail": opportunity_detail,
        "opportunity_summary": opportunity_summary,
        "drawdown_episodes": drawdown_episodes,
        "timing_diagnostics": timing_diagnostics,
    }


def _cross_mom_subperiod_metrics(cross_returns: dict, spy_returns: pd.Series) -> pd.DataFrame:
    base_index = cross_returns["base_price_only"].index
    periods = {
        "2010_2016": ("2010-01-01", "2016-12-31"),
        "2017_2020": ("2017-01-01", "2020-12-31"),
        "2021_plus": ("2021-01-01", None),
    }
    rows = []
    for period, (start, end) in periods.items():
        idx = base_index[base_index >= pd.Timestamp(start, tz="UTC")]
        if end:
            idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
        if len(idx) == 0:
            continue
        table = _metrics_table(cross_returns, spy_returns, idx)
        table = table[table["strategy"].isin(["base_price_only", "legacy_5_signal_with_cross_mom"])]
        table["period"] = period
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cross_mom_drawdown_analysis(
    close: pd.DataFrame,
    cross_returns: dict,
    full_composite: pd.Series,
    no_cross_composite: pd.Series,
    full_scale: pd.Series,
    no_cross_scale: pd.Series,
    threshold: float = -0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = close[MARKET_PROXY].dropna()
    spy_dd = spy / spy.cummax() - 1.0
    in_dd = spy_dd <= threshold

    episodes = []
    start = None
    for dt, active in in_dd.items():
        if active and start is None:
            start = dt
        elif not active and start is not None:
            episodes.append((start, prev_dt))
            start = None
        prev_dt = dt
    if start is not None:
        episodes.append((start, spy_dd.index[-1]))

    episode_rows = []
    timing_rows = []
    for n, (start_dt, end_dt) in enumerate(episodes, start=1):
        idx = full_scale.index[(full_scale.index >= start_dt) & (full_scale.index <= end_dt)]
        if len(idx) == 0:
            continue
        base_ret = cross_returns["base_price_only"]["net_return"].reindex(idx).dropna()
        no_cross_ret = cross_returns["base_price_only"]["net_return"].reindex(idx).dropna()
        legacy_ret = cross_returns["legacy_5_signal_with_cross_mom"]["net_return"].reindex(idx).dropna()
        local_spy_dd = spy_dd.reindex(idx).dropna()
        scale_diff = no_cross_scale.reindex(idx) - full_scale.reindex(idx)
        more_defensive = scale_diff < -0.05
        less_defensive = scale_diff > 0.05
        first_more_defensive = scale_diff[more_defensive].index.min() if more_defensive.any() else pd.NaT

        episode_rows.append({
            "episode": n,
            "start_date": start_dt,
            "end_date": end_dt,
            "n_days": len(idx),
            "spy_min_drawdown": round(float(local_spy_dd.min()), 4) if len(local_spy_dd) else 0.0,
            "avg_scale_with_cross_mom": round(float(full_scale.reindex(idx).mean()), 4),
            "avg_scale_without_cross_mom": round(float(no_cross_scale.reindex(idx).mean()), 4),
            "avg_scale_diff_without_minus_with": round(float(scale_diff.mean()), 4),
            "days_without_cross_more_defensive": int(more_defensive.sum()),
            "days_without_cross_less_defensive": int(less_defensive.sum()),
            "first_more_defensive_date": first_more_defensive,
            "legacy_with_cross_episode_return": round(float((1.0 + legacy_ret).prod() - 1.0), 4) if len(legacy_ret) else 0.0,
            "without_cross_episode_return": round(float((1.0 + no_cross_ret).prod() - 1.0), 4) if len(no_cross_ret) else 0.0,
            "legacy_with_cross_episode_max_drawdown": round(_max_drawdown_from_returns(legacy_ret), 4),
            "without_cross_episode_max_drawdown": round(_max_drawdown_from_returns(no_cross_ret), 4),
        })

        local = pd.DataFrame(index=idx)
        local["episode"] = n
        local["signal_date"] = idx
        local["spy_drawdown"] = spy_dd.reindex(idx)
        local["price_composite_with_cross_mom"] = full_composite.reindex(idx)
        local["price_composite_without_cross_mom"] = no_cross_composite.reindex(idx)
        local["scale_with_cross_mom"] = full_scale.reindex(idx)
        local["scale_without_cross_mom"] = no_cross_scale.reindex(idx)
        local["scale_diff_without_minus_with"] = scale_diff.reindex(idx)
        local["legacy_with_cross_return"] = cross_returns["legacy_5_signal_with_cross_mom"]["net_return"].reindex(idx)
        local["without_cross_return"] = cross_returns["base_price_only"]["net_return"].reindex(idx)
        timing_rows.append(local.reset_index(drop=True))

    timing = pd.concat(timing_rows, ignore_index=True) if timing_rows else pd.DataFrame()
    return pd.DataFrame(episode_rows), timing


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    rendered = df.copy()
    rendered.columns = [str(c) for c in rendered.columns]
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in rendered.astype(str).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _write_research_summary(
    metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    rebalance_policy_comparison: pd.DataFrame,
    rebalance_policy_cost: pd.DataFrame,
    rebalance_policy_oos: pd.DataFrame,
    rebalance_policy_subperiod: pd.DataFrame,
    rebalance_policy_turnover: pd.DataFrame,
    rebalance_policy_drawdown: pd.DataFrame,
    rebalance_policy_annual_turnover: pd.DataFrame,
    rebalance_policy_trade_attribution: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    parameter_sensitivity: pd.DataFrame,
    opportunity_summary: pd.DataFrame,
    base_ablation: pd.DataFrame,
    base_parameter_sensitivity: pd.DataFrame,
    base_subperiod_metrics: pd.DataFrame,
    base_opportunity_summary: pd.DataFrame,
    cross_mom_comparison: pd.DataFrame,
    cross_mom_oos: pd.DataFrame,
    cross_mom_subperiod: pd.DataFrame,
    cross_mom_opportunity_summary: pd.DataFrame,
    cross_mom_drawdown_episodes: pd.DataFrame,
    signal_stats: dict,
):
    metric_lookup = metrics.set_index("strategy")
    base = metric_lookup.loc["base_price_only"]
    v3 = metric_lookup.loc["price_plus_spy_volume_confirmation"]
    calmar_improved = v3["calmar"] > base["calmar"]
    drawdown_improved = v3["max_drawdown"] > base["max_drawdown"]
    conclusion = (
        "The SPY volume-confirmation layer adds useful risk-management information "
        "under this tested design because it improves Calmar and reduces Max Drawdown "
        "relative to the base price-only model."
        if calmar_improved and drawdown_improved
        else
        "The SPY volume-confirmation layer did not add clear incremental value beyond "
        "the base price regime model under this tested design."
    )

    lines = [
        "# Macro Regime Trader v3 Research Summary",
        "",
        "## 1. Strategy objective",
        "Test whether a long-only equity macro regime overlay can improve exposure timing, with drawdown control and Calmar ratio as the main KPIs.",
        "",
        "## Research hypothesis",
        "SPY volume-confirmed Follow-Through Days identify durable risk-on transitions.",
        "",
        "## Economic mechanism",
        "Institutional buying after a correction should appear as index-level price gains on expanding volume. The volume-confirmation layer is therefore designed to test whether accumulation and distribution evidence improves exposure scaling beyond the base price-only macro regime model.",
        "",
        "## Expected improvement",
        "The v3 model should improve Max Drawdown and Calmar versus the base price-only model.",
        "",
        "## Failure condition",
        "If v3 does not improve drawdown metrics, or if the results disappear under out-of-sample and sensitivity tests, the Follow-Through Day layer is not robust.",
        "",
        "## Hypothesis evolution",
        "The v3 model tested whether FTD and SPY volume confirmation improve risk-on timing. Empirically, v3 slightly reduced volatility and Max Drawdown, but worsened Calmar because opportunity cost exceeded loss avoided. This suggests that unconditional daily volume penalties are too noisy and too costly. Therefore, the final strategy candidate returns to the simpler `base_price_only` macro regime overlay, and robustness tests focus on whether the base model is stable across signal ablations, parameter choices, subperiods, and opportunity-cost diagnostics.",
        "",
        "## 2. Economic rationale",
        "The final base model estimates broad equity risk appetite from SPY volatility, SPY trend, market breadth, and the volatility regime. Cross-sectional momentum and SPY volume confirmation were tested as extensions, but are not part of the final defensive specification.",
        "",
        "## 3. Signal design",
        "The control strategy is `base_price_only`, now defined as the Defensive 4-Signal Macro Regime Overlay. The tested v3 extension is `price_plus_spy_volume_confirmation`, which blends the base price composite with volume breakout, Follow-Through Day, Distribution Day, and Heavy Distribution Day evidence. A Follow-Through Day occurs after a rally attempt when SPY rises at least 1.25% on higher volume without undercutting the rally low.",
        "",
        "## Rule-based mathematical definitions",
        "",
        "Let \\(P_t\\), \\(L_t\\), and \\(V_t\\) denote SPY close, low, and volume on trading date \\(t\\). Let \\(w_t\\) denote portfolio weights set after observing date \\(t\\), and applied to returns from \\(t\\) to \\(t+1\\). Parameters are hypothesis definitions, not optimized values: \\(\\theta_{FTD}=1.25\\%\\), \\(\\theta_{DD}=-0.25\\%\\), \\(\\theta_{HDD}=-1.00\\%\\), \\(\\theta_C=-8\\%\\), and \\(\\lambda=0.15\\). Sensitivity tests evaluate nearby values separately.",
        "",
        "SPY daily return: \\(r^{SPY}_t=P_t/P_{t-1}-1\\).",
        "",
        "50-day drawdown: \\(DD^{50}_t=P_t / \\max(P_{t-49},...,P_t)-1\\).",
        "",
        "Correction indicator: \\(C_t=1\\{DD^{50}_t \\le \\theta_C\\}\\), optionally also requiring price weakness versus the 50-day moving average. Otherwise \\(C_t=0\\).",
        "",
        "Rally attempt state: a rally attempt begins on date \\(t\\) if \\(C_t=1\\), \\(r^{SPY}_t>0\\), and no rally attempt is already active. The rally attempt day count \\(A_t\\) is set to 1 on the first rally attempt day and increments by one while the rally remains active.",
        "",
        "Rally low: \\(RL_t\\) is the minimum SPY low observed during the active rally attempt through date \\(t\\). To avoid a self-referential undercut test, the failed-rally check uses the prior value \\(RL_{t-1}\\) before updating \\(RL_t=\\min(RL_{t-1},L_t)\\).",
        "",
        "Failed rally indicator: \\(FR_t=1\\{A_{t-1}>0 \\text{ and } L_t < RL_{t-1}\\}\\). If \\(FR_t=1\\), the rally attempt is reset before any Follow-Through Day can be confirmed.",
        "",
        "Follow-Through Day indicator: \\(FTD_t=1\\{A_t \\ge 4,\\ r^{SPY}_t \\ge \\theta_{FTD},\\ V_t>V_{t-1},\\ FR_t=0\\}\\). Otherwise \\(FTD_t=0\\).",
        "",
        "Distribution Day indicator: \\(DIST_t=1\\{r^{SPY}_t \\le \\theta_{DD},\\ V_t>V_{t-1}\\}\\).",
        "",
        "Heavy Distribution Day indicator: \\(HDIST_t=1\\{r^{SPY}_t \\le \\theta_{HDD},\\ V_t/MA_{50}(V)_t \\ge 1.20\\}\\).",
        "",
        "Volume confirmation score: \\(S^{VOL}_t=clip(0.25\\,BRK_t+1.00\\,FTD_t-0.35\\,DIST_t-0.75\\,HDIST_t-0.25\\,CLUST_t,-1,1)\\), where \\(BRK_t=1\\{r^{SPY}_t>0, V_t/MA_{50}(V)_t \\ge 1.10\\}\\) and \\(CLUST_t=1\\{\\sum_{i=0}^{24}DIST_{t-i}\\ge4\\}\\).",
        "",
        "Base regime score: \\(S^{BASE}_t\\) is the final price-only composite in \\([-1,1]\\), averaging the VIX proxy, SPY trend, market breadth, and volatility-regime indicators. The 12-1 month cross-momentum indicator is reported as an ablation/extension, but excluded from the final specification.",
        "",
        "Final v3 regime score: \\(S^{V3}_t=clip((1-\\lambda)S^{BASE}_t+\\lambda S^{VOL}_t,-1,1)\\).",
        "",
        "Final exposure scale: \\(x^{BASE}_t=clip((S^{BASE}_t+1)/2,0,1)\\) and \\(x^{V3}_t=clip((S^{V3}_t+1)/2,0,1)\\).",
        "",
        "Execution lag and portfolio weights: all indicators using \\(P_t\\), \\(L_t\\), or \\(V_t\\) are only applied to weights for the next return period. Thus \\(w_t=f(x_t,\\mathcal{I}_t)\\) is computed after date \\(t\\) information is observable, and earns asset returns \\(R_{t+1}\\), not \\(R_t\\).",
        "",
        "Portfolio return: \\(R^{p}_{t+1}=w_t^\\top R_{t+1}-TC_{t+1}\\), where transaction cost scenarios use \\(TC_{t+1}=c\\sum_i |w_{i,t}-w_{i,t-1}|\\) for \\(c\\in\\{0,5,10,20\\}\\) basis points.",
        "",
        "Opportunity cost: because the FTD layer is low frequency, it is evaluated not only by drawdown reduction but also by missed upside. When v3 is more defensive than base, \\(OC_{t+1}=\\max(x^{BASE}_t-x^{V3}_t,0)\\max(R^{RP}_{t+1},0)\\) and \\(LA_{t+1}=\\max(x^{BASE}_t-x^{V3}_t,0)\\max(-R^{RP}_{t+1},0)\\). The net defensive timing benefit is \\(\\sum LA-\\sum OC\\).",
        "",
        "When v3 takes more risk than base, \\(UC_{t+1}=\\max(x^{V3}_t-x^{BASE}_t,0)\\max(R^{RP}_{t+1},0)\\) and \\(EL_{t+1}=\\max(x^{V3}_t-x^{BASE}_t,0)\\max(-R^{RP}_{t+1},0)\\). These terms separate additional upside captured from extra downside loss.",
        "",
        "Base vs v3 evaluation metrics: compare `base_price_only` and `price_plus_spy_volume_confirmation` on Sharpe, Calmar, CAGR, total return, annual volatility, win rate, and Max Drawdown. The primary validation criteria are Max Drawdown and Calmar, including full-sample, in-sample, out-of-sample, transaction-cost sensitivity, and parameter-sensitivity results.",
        "",
        "## 4. Timing and look-ahead bias control",
        "Signals are computed on `signal_date` from close and volume data observable at that date. Portfolio returns use the next tradable period, recorded as `next_return_date` in `macro_v3_diagnostics.csv`; same-day signals are never applied to same-day returns.",
        "",
        "## 5. Base vs v3 comparison",
        _df_to_markdown(metrics[metrics["strategy"].isin(["base_price_only", "price_plus_spy_volume_confirmation"])]),
        "",
        "## 6. Benchmark comparison",
        _df_to_markdown(metrics),
        "",
        "## 7. Transaction cost sensitivity",
        _df_to_markdown(cost_sensitivity),
        "",
        "## 7b. Rebalancing policy and turnover control",
        "The final defensive overlay uses `scale_change_only` rebalancing because the research hypothesis is about market-level exposure timing, not daily stock-level micro-rebalancing. Daily and weekly variants are reported as implementation robustness checks.",
        "",
        _df_to_markdown(rebalance_policy_comparison),
        "",
        "### Rebalancing policy cost sensitivity",
        _df_to_markdown(rebalance_policy_cost),
        "",
        "### Rebalancing policy OOS robustness",
        _df_to_markdown(rebalance_policy_oos),
        "",
        "### Rebalancing policy subperiod robustness",
        _df_to_markdown(rebalance_policy_subperiod),
        "",
        "### Rebalancing policy turnover distribution",
        _df_to_markdown(rebalance_policy_turnover),
        "",
        "### Rebalancing policy annual turnover",
        _df_to_markdown(rebalance_policy_annual_turnover),
        "",
        "### Rebalancing policy trade attribution",
        "This table separates trades driven by scale changes from trades driven by risk-parity weight refreshes. For the final `scale_change_only` policy, trades are only triggered when the regime scale changes, although the refreshed target can also incorporate updated risk-parity weights.",
        "",
        _df_to_markdown(rebalance_policy_trade_attribution),
        "",
        "### Rebalancing policy drawdown episodes",
        _df_to_markdown(rebalance_policy_drawdown),
        "",
        "## 8. Out-of-sample results",
        _df_to_markdown(oos_metrics),
        "",
        "## 8b. Opportunity cost and timing benefit",
        "This section tests whether defensive exposure reductions create excessive missed upside. A useful low-frequency confirmation layer should avoid more loss than the upside it misses, especially out-of-sample.",
        "",
        _df_to_markdown(opportunity_summary),
        "",
        "## 9. Parameter sensitivity",
        _df_to_markdown(parameter_sensitivity),
        "",
        "## 9b. Base model robustness",
        "Because the volume layer did not add robust incremental value, the final strategy candidate is the simpler price-only macro regime overlay. The following tests evaluate whether the base model itself is robust.",
        "",
        "### Base signal ablation",
        _df_to_markdown(base_ablation),
        "",
        "### Base parameter sensitivity",
        _df_to_markdown(base_parameter_sensitivity),
        "",
        "### Base subperiod metrics",
        _df_to_markdown(base_subperiod_metrics),
        "",
        "### Base opportunity cost versus no-overlay risk parity",
        _df_to_markdown(base_opportunity_summary),
        "",
        "## 9c. Cross-momentum exclusion test",
        "Ablation suggested that the 12-1 month cross-momentum component may be too slow for a drawdown-control overlay. This section formally compares the full base model against a defensive four-signal version that excludes cross momentum.",
        "",
        "### Full-sample comparison",
        _df_to_markdown(cross_mom_comparison),
        "",
        "### OOS comparison",
        _df_to_markdown(cross_mom_oos),
        "",
        "### Subperiod comparison",
        _df_to_markdown(cross_mom_subperiod),
        "",
        "### Drawdown episode analysis",
        _df_to_markdown(cross_mom_drawdown_episodes),
        "",
        "### Opportunity cost of excluding cross momentum",
        _df_to_markdown(cross_mom_opportunity_summary),
        "",
        "## Signal frequency and scale diagnostics",
        _df_to_markdown(pd.DataFrame([signal_stats])),
        "",
        "## 10. Conclusion",
        conclusion,
        "",
        "The final submitted strategy candidate is therefore the Defensive 4-Signal Macro Regime Overlay with scale-change-only rebalancing. This keeps the implementation aligned with the economic hypothesis: trade when market-level risk exposure changes, not when small daily risk-parity estimates drift.",
    ]
    (RESULT_DIR / "macro_v3_research_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    use_cache: bool = True,
    use_volume_confirmation: bool = USE_SPY_VOLUME_CONFIRMATION,
) -> dict:
    logger.info("=" * 60)
    logger.info("MACRO REGIME TRADER v3 — RESEARCH BACKTEST")
    logger.info("=" * 60)

    data          = fetch_all(use_cache=use_cache)
    if MARKET_PROXY not in data:
        raise RuntimeError(
            f"{MARKET_PROXY} data is required for the macro regime and SPY volume layer. "
            "Provide cached parquet data or valid Alpaca paper-data credentials."
        )
    missing_trade = [s for s in TRADE_UNIVERSE if s not in data]
    if len(missing_trade) == len(TRADE_UNIVERSE):
        raise RuntimeError(
            "No trade-universe data is available. Provide cached parquet data or "
            "valid Alpaca paper-data credentials for offline research."
        )
    close         = _close_matrix(data)
    returns       = _ret_matrix(close)
    dollar_volume = _dollar_volume_matrix(data, close)

    logger.info("Computing base price regime and SPY volume-confirmation regime…")
    regime = compute_regime(close, data, use_volume_confirmation)
    composite = regime["final_composite"]
    scale = regime["final_scale"]
    signal_stats = _signal_frequency_stats(regime)
    _log_signal_warnings(signal_stats)

    risk_on_frac = (composite > 0).mean()
    flat_days    = int((scale < 0.05).sum())
    logger.info(
        f"Final composite [{composite.min():.2f}, {composite.max():.2f}] "
        f"| risk-on {risk_on_frac:.0%} of days"
    )
    logger.info(
        f"Volume score mean={signal_stats['volume_confirmation_mean']:.3f}  "
        f"std={signal_stats['volume_confirmation_std']:.3f}  "
        f"| FTD/year={signal_stats['follow_through_day_annualized']:.2f}  "
        f"| Distribution/year={signal_stats['distribution_day_annualized']:.2f}"
    )
    logger.info(
        f"Scale  min={scale.min():.3f}  max={scale.max():.3f}  mean={scale.mean():.3f}"
        f"  |  flat days (sc<0.05): {flat_days}/{len(scale)} ({flat_days/len(scale):.1%})"
    )
    if scale.min() > 0.30:
        logger.warning(
            f"Scale never drops below {scale.min():.2f} — the macro overlay is acting "
            f"as a constant ~{scale.mean():.0%} multiplier, not a true risk-on/off gate. "
            "Consider whether the composite thresholds need recalibration."
        )

    tradeable = [s for s in TRADE_UNIVERSE if s in close.columns]
    dates     = close.index

    logger.info("Running walk-forward research comparison…")
    base_rp_weights = _walk_forward_base_weights(returns, tradeable, dates)
    weights = _strategy_weight_map(returns, tradeable, dates, regime, base_rp_weights)
    strategy_returns = _strategy_return_map(
        returns, dollar_volume, weights, cost_bps=TRANSACTION_COST_BPS
    )
    # Align SPY buy-and-hold to the same signal-date convention used by the
    # strategies: signal_date earns the next close-to-close SPY return.
    spy_fwd_returns = returns[MARKET_PROXY].shift(-1).iloc[RISK_LOOKBACK_DAYS:-1].dropna()

    metrics_table = _metrics_table(strategy_returns, spy_fwd_returns)
    cost_sensitivity = _cost_sensitivity_table(weights, returns, dollar_volume, spy_fwd_returns)
    (
        rebalance_policy_comparison,
        rebalance_policy_cost,
        rebalance_policy_oos,
        rebalance_policy_subperiod,
        rebalance_policy_turnover,
        rebalance_policy_drawdown,
        rebalance_policy_annual_turnover,
        rebalance_policy_trade_attribution,
    ) = _rebalance_policy_tables(
        base_rp_weights, regime, returns, dollar_volume, dates, close
    )
    oos_metrics = _oos_metrics_table(strategy_returns, spy_fwd_returns)
    parameter_sensitivity = _parameter_sensitivity_table(
        close, data, returns, dollar_volume, tradeable, dates, base_rp_weights
    )
    opportunity_detail = _opportunity_cost_detail(regime, strategy_returns)
    opportunity_summary = _summarize_opportunity_cost(opportunity_detail)
    base_ablation = _base_ablation_table(
        close, returns, dollar_volume, tradeable, dates, base_rp_weights
    )
    base_parameter_sensitivity = _base_parameter_sensitivity_table(
        close, returns, dollar_volume, tradeable, dates, base_rp_weights
    )
    base_subperiod_metrics = _base_subperiod_metrics_table(strategy_returns, spy_fwd_returns)
    base_opportunity_detail = _base_opportunity_cost_detail(regime, strategy_returns)
    base_opportunity_summary = _summarize_base_opportunity_cost(base_opportunity_detail)
    cross_mom_tables = _cross_mom_research_tables(
        close,
        returns,
        dollar_volume,
        tradeable,
        dates,
        regime,
        base_rp_weights,
        strategy_returns,
        spy_fwd_returns,
    )
    diagnostics = _build_diagnostics(
        regime,
        strategy_returns["base_price_only"],
        strategy_returns["price_plus_spy_volume_confirmation"],
        opportunity_detail,
    )

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS — Macro Regime v3 Research Comparison")
    base_row = metrics_table.set_index("strategy").loc["base_price_only"]
    v3_row = metrics_table.set_index("strategy").loc["price_plus_spy_volume_confirmation"]
    logger.info(f"  Base Calmar={base_row['calmar']:.3f}  MaxDD={base_row['max_drawdown']*100:.2f}%")
    logger.info(f"  V3   Calmar={v3_row['calmar']:.3f}  MaxDD={v3_row['max_drawdown']*100:.2f}%")
    logger.info("-" * 60)
    for _, bm in metrics_table.iterrows():
        logger.info(
            f"  {bm['strategy']:36s} Sharpe={bm['sharpe']:.3f}  "
            f"CAGR={bm['cagr']*100:.2f}%  MaxDD={bm['max_drawdown']*100:.2f}%"
        )
    logger.info("=" * 60)

    metrics_table.to_csv(RESULT_DIR / "macro_v3_metrics_comparison.csv", index=False)
    diagnostics.to_csv(RESULT_DIR / "macro_v3_diagnostics.csv", index=False)
    cost_sensitivity.to_csv(RESULT_DIR / "macro_v3_cost_sensitivity.csv", index=False)
    rebalance_policy_comparison.to_csv(
        RESULT_DIR / "macro_rebalance_policy_comparison.csv", index=False
    )
    rebalance_policy_cost.to_csv(
        RESULT_DIR / "macro_rebalance_policy_cost_sensitivity.csv", index=False
    )
    rebalance_policy_oos.to_csv(
        RESULT_DIR / "macro_rebalance_policy_oos_metrics.csv", index=False
    )
    rebalance_policy_subperiod.to_csv(
        RESULT_DIR / "macro_rebalance_policy_subperiod_metrics.csv", index=False
    )
    rebalance_policy_turnover.to_csv(
        RESULT_DIR / "macro_rebalance_policy_turnover_breakdown.csv", index=False
    )
    rebalance_policy_drawdown.to_csv(
        RESULT_DIR / "macro_rebalance_policy_drawdown_episodes.csv", index=False
    )
    rebalance_policy_annual_turnover.to_csv(
        RESULT_DIR / "macro_rebalance_policy_annual_turnover.csv", index=False
    )
    rebalance_policy_trade_attribution.to_csv(
        RESULT_DIR / "macro_rebalance_policy_trade_attribution.csv", index=False
    )
    oos_metrics.to_csv(RESULT_DIR / "macro_v3_oos_metrics.csv", index=False)
    opportunity_detail.to_csv(RESULT_DIR / "macro_v3_opportunity_cost.csv", index=False)
    parameter_sensitivity.to_csv(
        RESULT_DIR / "macro_v3_parameter_sensitivity.csv", index=False
    )
    base_ablation.to_csv(RESULT_DIR / "macro_base_ablation.csv", index=False)
    base_parameter_sensitivity.to_csv(
        RESULT_DIR / "macro_base_parameter_sensitivity.csv", index=False
    )
    base_subperiod_metrics.to_csv(RESULT_DIR / "macro_base_subperiod_metrics.csv", index=False)
    base_opportunity_detail.to_csv(RESULT_DIR / "macro_base_opportunity_cost.csv", index=False)
    base_opportunity_summary.to_csv(
        RESULT_DIR / "macro_base_opportunity_summary.csv", index=False
    )
    cross_mom_tables["comparison"].to_csv(
        RESULT_DIR / "macro_cross_mom_comparison.csv", index=False
    )
    cross_mom_tables["oos"].to_csv(
        RESULT_DIR / "macro_cross_mom_oos_metrics.csv", index=False
    )
    cross_mom_tables["subperiod"].to_csv(
        RESULT_DIR / "macro_cross_mom_subperiod_metrics.csv", index=False
    )
    cross_mom_tables["drawdown_episodes"].to_csv(
        RESULT_DIR / "macro_cross_mom_drawdown_episodes.csv", index=False
    )
    cross_mom_tables["timing_diagnostics"].to_csv(
        RESULT_DIR / "macro_cross_mom_timing_diagnostics.csv", index=False
    )
    cross_mom_tables["opportunity_detail"].to_csv(
        RESULT_DIR / "macro_cross_mom_opportunity_cost.csv", index=False
    )
    cross_mom_tables["opportunity_summary"].to_csv(
        RESULT_DIR / "macro_cross_mom_opportunity_summary.csv", index=False
    )
    _write_research_summary(
        metrics_table,
        cost_sensitivity,
        rebalance_policy_comparison,
        rebalance_policy_cost,
        rebalance_policy_oos,
        rebalance_policy_subperiod,
        rebalance_policy_turnover,
        rebalance_policy_drawdown,
        rebalance_policy_annual_turnover,
        rebalance_policy_trade_attribution,
        oos_metrics,
        parameter_sensitivity,
        opportunity_summary,
        base_ablation,
        base_parameter_sensitivity,
        base_subperiod_metrics,
        base_opportunity_summary,
        cross_mom_tables["comparison"],
        cross_mom_tables["oos"],
        cross_mom_tables["subperiod"],
        cross_mom_tables["opportunity_summary"],
        cross_mom_tables["drawdown_episodes"],
        signal_stats,
    )
    logger.info(f"Saved v3 research outputs under {RESULT_DIR}")

    return v3_row.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STATE — JSON-backed, independent from main engine
# ══════════════════════════════════════════════════════════════════════════════

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
        _signal_rebalance_job_inner()
    except Exception as e:
        # APScheduler swallows unhandled exceptions into its own logger (not loguru),
        # so crashes would disappear silently. Catch everything here to ensure errors
        # always appear in kengo_engine.log.
        logger.exception(f"signal_rebalance_job crashed: {e}")


def _signal_rebalance_job_inner():
    try:
        data    = fetch_all(use_cache=False)
        close   = _close_matrix(data)
        returns = _ret_matrix(close)
    except Exception as e:
        logger.error(f"Data fetch failed — skipping: {e}")
        return

    regime = compute_regime(close, data)
    last = regime.iloc[-1]
    last_comp = float(last["final_composite"])
    scale = float(last["final_scale"])
    logger.info(
        f"Composite: {last_comp:.3f}  →  scale: {scale:.3f} "
        f"(price={last['price_composite']:.3f}, volume={last['volume_confirmation_score']:.3f})"
    )

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

    # If any tradeable symbol has no price, prev_w would treat its position as
    # zero — triggering a ghost buy on top of an existing holding. Abort instead.
    missing_prices = [s for s in tradeable if s not in prices]
    if missing_prices:
        logger.error(
            f"Price fetch incomplete — missing {missing_prices}. "
            "Skipping rebalance to avoid ghost buys on stale weights."
        )
        return

    nav = _compute_nav(state, prices)

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
    parser = argparse.ArgumentParser(description="Macro Regime Standalone Trader v3")
    parser.add_argument(
        "--mode", choices=["backtest", "live", "full"], default="backtest",
        help=(
            "backtest — run backtest and print results. "
            "live     — start live engine immediately. "
            "full     — run backtest first; if Sharpe ≥ min-sharpe proceed to live."
        ),
    )
    parser.add_argument("--no-cache",   action="store_true", help="Force re-download of market data")
    parser.add_argument("--price-only", action="store_true", help="Disable SPY volume-confirmation layer")
    parser.add_argument("--run-now",    action="store_true", help="Fire one rebalance immediately on live start")
    parser.add_argument("--min-sharpe", type=float, default=0.5, help="Minimum backtest Sharpe to proceed to live")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(
            use_cache=not args.no_cache,
            use_volume_confirmation=not args.price_only,
        )

    elif args.mode == "live":
        run_live(run_now=args.run_now)

    elif args.mode == "full":
        metrics = run_backtest(
            use_cache=not args.no_cache,
            use_volume_confirmation=not args.price_only,
        )
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
