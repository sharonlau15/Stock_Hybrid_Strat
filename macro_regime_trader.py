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

Usage
-----
  python macro_regime_trader.py --mode backtest
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

for _env in [ROOT_DIR / "Alpaca.env", ROOT_DIR / ".env"]:
    if _env.exists():
        load_dotenv(_env)
        break

# Dedicated MR paper-trading Alpaca account
API_KEY    = os.getenv("ALPACA_PAPER_API_KEY_MR")
API_SECRET = os.getenv("ALPACA_PAPER_API_SECRET_MR")
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
            "ALPACA_PAPER_API_KEY_MR / ALPACA_PAPER_API_SECRET_MR not set in env"
        )
    return TradingClient(API_KEY, API_SECRET, paper=PAPER)


def _data_client() -> StockHistoricalDataClient:
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MR / ALPACA_PAPER_API_SECRET_MR not set in env"
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


def compute_composite(close: pd.DataFrame) -> pd.Series:
    """
    5-indicator macro composite over time.

    SPY is used only for VIX proxy, trend MA, and vol regime.
    Breadth and cross-momentum use SIGNAL_UNIVERSE (22 independent stocks).
    Returns series in [-1, +1].
    """
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
    composite = (
        pd.concat(components, axis=1)
        .mean(axis=1)
        .clip(-1.0, 1.0)
        .reindex(close.index)
        .fillna(0.0)
    )
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
        logger.warning("risk_parity_weights: all weights zero — returning equal weight")
        w[:] = 1.0 / len(avail)

    return w.reindex(symbols, fill_value=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(use_cache: bool = True) -> dict:
    logger.info("=" * 60)
    logger.info("MACRO REGIME TRADER v2 — BACKTEST")
    logger.info("=" * 60)

    data    = fetch_all(use_cache=use_cache)
    close   = _close_matrix(data)
    returns = _ret_matrix(close)

    logger.info("Computing macro composite…")
    composite = compute_composite(close)
    # Map [-1, +1] → [0, 1]: full risk-off = cash, neutral = half, full risk-on = full
    scale = ((composite + 1.0) / 2.0).clip(0.0, 1.0)

    risk_on_frac = (composite > 0).mean()
    flat_days    = int((scale < 0.05).sum())
    logger.info(
        f"Composite [{composite.min():.2f}, {composite.max():.2f}] "
        f"| risk-on {risk_on_frac:.0%} of days"
    )
    logger.info(
        f"Scale  min={scale.min():.3f}  max={scale.max():.3f}  mean={scale.mean():.3f}"
        f"  |  flat days (sc<0.05): {flat_days}/{len(scale)} ({flat_days/len(scale):.1%})"
    )
    if scale.min() > 0.30:
        logger.warning(
            f"Scale never drops below {scale.min():.2f} — the macro overlay is acting "
            "as a constant ~{scale.mean():.0%} multiplier, not a true risk-on/off gate. "
            "Consider whether the composite thresholds need recalibration."
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
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    win_rate  = float((port_r > 0).sum() / n) if n else 0.0
    years     = n / TRADING_DAYS
    cagr      = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    final_cap = PORTFOLIO_USD * (1 + total_ret)

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
    logger.info("=" * 60)

    return metrics


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
    parser = argparse.ArgumentParser(description="Macro Regime Standalone Trader v2")
    parser.add_argument(
        "--mode", choices=["backtest", "live", "full"], default="backtest",
        help=(
            "backtest — run backtest and print results. "
            "live     — start live engine immediately. "
            "full     — run backtest first; if Sharpe ≥ min-sharpe proceed to live."
        ),
    )
    parser.add_argument("--no-cache",   action="store_true", help="Force re-download of market data")
    parser.add_argument("--run-now",    action="store_true", help="Fire one rebalance immediately on live start")
    parser.add_argument("--min-sharpe", type=float, default=0.5, help="Minimum backtest Sharpe to proceed to live")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(use_cache=not args.no_cache)

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
