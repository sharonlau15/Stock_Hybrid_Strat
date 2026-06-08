"""
macro_regime_trader.py
======================
Standalone Macro Regime-only trading machine.

Completely independent from the main multi-strategy engine.
Uses its own state file, log file, and scheduler.
Shares Alpaca credentials and universe from config/.

Usage
-----
  python macro_regime_trader.py --mode backtest   # backtest only, print results
  python macro_regime_trader.py --mode live       # live trading only
  python macro_regime_trader.py --mode full       # backtest → if Sharpe > 0.5 → go live
"""

import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

import os
from config.settings import (
    UNIVERSE, PORTFOLIO_USD, MIN_ORDER_USD,
    PAPER_TRADING, RESULT_DIR, LOG_DIR,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT, USE_TRAILING_STOP,
    PRICE_MONITOR_SECS, REBALANCE_THRESHOLD,
    MAX_LIVE_POSITIONS, MAX_POSITION_SIZE,
    BACKTEST_START, RISK_LOOKBACK_DAYS, TRADING_DAYS,
)
from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
from strategies.macro_regime import MacroRegimeStrategy
from portfolio.max_sharpe import MaxSharpePortfolio
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── Paths ──────────────────────────────────────────────────────────────────────

STATE_FILE = RESULT_DIR / "macro_live_state.json"
LOG_FILE   = LOG_DIR   / "macro_engine.log"

# ── Logger ─────────────────────────────────────────────────────────────────────

def _setup_logger():
    logger.remove()
    logger.add(sys.stderr,  level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(LOG_FILE,    level="DEBUG", rotation="1 week",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(use_cache: bool = True) -> dict:
    """
    Run a walk-forward backtest using only MacroRegimeStrategy.
    Returns a dict of key performance metrics and prints a summary.
    """
    logger.info("=" * 60)
    logger.info("MACRO REGIME TRADER — BACKTEST")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading universe data…")
    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close   = build_close_matrix(universe_data)
    returns = build_return_matrix(close)
    high_df = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
    low_df  = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})

    # ── Signals ───────────────────────────────────────────────────────────────
    logger.info("Generating MacroRegime signals…")
    strategy = MacroRegimeStrategy()
    signals  = strategy.run(close=close, returns=returns,
                             high=high_df, low=low_df)

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    logger.info("Running walk-forward backtest…")
    portfolio   = MaxSharpePortfolio()
    dates       = signals.index
    wh          = pd.DataFrame(0.0, index=dates, columns=UNIVERSE)

    for i, dt in enumerate(dates):
        if i < RISK_LOOKBACK_DAYS:
            continue
        cov = (
            returns.loc[:dt]
            .iloc[-RISK_LOOKBACK_DAYS:]
            .cov() * TRADING_DAYS
        )
        sig_row = signals.loc[dt]
        try:
            w = portfolio.compute_weights(sig_row, cov, long_short=False)
            for sym, weight in w.items():
                wh.loc[dt, sym] = weight
        except Exception:
            wh.loc[dt] = wh.iloc[i - 1]

    fwd     = returns.shift(-1)
    port_r  = (wh * fwd).sum(axis=1).iloc[RISK_LOOKBACK_DAYS:-1]
    port_r  = port_r.dropna()

    # ── Metrics ───────────────────────────────────────────────────────────────
    n          = len(port_r)
    ann_ret    = float(port_r.mean() * TRADING_DAYS)
    ann_vol    = float(port_r.std()  * np.sqrt(TRADING_DAYS))
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum        = (1 + port_r).cumprod()
    total_ret  = float(cum.iloc[-1] - 1) if len(cum) else 0.0
    roll_max   = cum.cummax()
    drawdown   = (cum - roll_max) / roll_max
    max_dd     = float(drawdown.min())
    calmar     = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    wins       = (port_r > 0).sum()
    win_rate   = float(wins / n) if n else 0.0
    final_cap  = PORTFOLIO_USD * (1 + total_ret)

    years      = n / TRADING_DAYS
    cagr       = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0

    metrics = {
        "sharpe":      round(sharpe,    3),
        "calmar":      round(calmar,    3),
        "cagr":        round(cagr,      4),
        "total_return":round(total_ret, 4),
        "max_drawdown":round(max_dd,    4),
        "annual_vol":  round(ann_vol,   4),
        "win_rate":    round(win_rate,  4),
        "final_capital":round(final_cap, 2),
        "n_days":      n,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS — Macro Regime (MaxSharpe portfolio)")
    logger.info(f"  Period       : {port_r.index[0].date()} → {port_r.index[-1].date()}")
    logger.info(f"  Trading days : {n}")
    logger.info(f"  Sharpe       : {sharpe:.3f}")
    logger.info(f"  Calmar       : {calmar:.3f}")
    logger.info(f"  CAGR         : {cagr*100:.2f}%")
    logger.info(f"  Total Return : {total_ret*100:.2f}%")
    logger.info(f"  Max Drawdown : {max_dd*100:.2f}%")
    logger.info(f"  Annual Vol   : {ann_vol*100:.2f}%")
    logger.info(f"  Win Rate     : {win_rate*100:.1f}%")
    logger.info(f"  Final Capital: ${final_cap:,.2f}  (start ${PORTFOLIO_USD:,})")
    logger.info("=" * 60)

    # Save latest signals for live engine warm-up
    (RESULT_DIR / "macro_signals_latest.csv").open("w").write(signals.tail(5).to_csv())

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STATE  (JSON-backed, independent from main engine's DB state)
# ══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "positions":       {sym: 0.0 for sym in UNIVERSE},
        "cash_usd":        float(PORTFOLIO_USD),
        "current_weights": {},
        "last_run":        None,
        "nav_history":     [],
        "trade_log":       [],
        "position_entries":{},
    }


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE — helpers (mirror of live_engine.py but macro-only)
# ══════════════════════════════════════════════════════════════════════════════

def _mr_keys() -> tuple[str, str]:
    """Return the Macro Regime-specific paper API key and secret."""
    key    = os.getenv("ALPACA_PAPER_API_KEY_MR")
    secret = os.getenv("ALPACA_PAPER_API_SECRET_MR")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_PAPER_API_KEY_MR / ALPACA_PAPER_API_SECRET_MR not set in env"
        )
    return key, secret


def _get_mr_trading_client():
    from alpaca.trading.client import TradingClient
    key, secret = _mr_keys()
    return TradingClient(key, secret, paper=True)


def _market_is_open() -> bool:
    try:
        return _get_mr_trading_client().get_clock().is_open
    except Exception:
        return False


def _fetch_prices() -> dict:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    key, secret = _mr_keys()
    client = StockHistoricalDataClient(key, secret)
    try:
        trades = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=UNIVERSE)
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


def _check_exits(state: dict, prices: dict) -> dict:
    exits   = {}
    entries = state.get("position_entries", {})
    for sym, qty in state["positions"].items():
        if qty == 0 or sym not in entries or sym not in prices:
            continue
        ep   = entries[sym]["entry_price"]
        cur  = prices[sym]
        pct  = (cur - ep) / ep
        if pct <= -STOP_LOSS_PCT:
            exits[sym] = cur
            logger.warning(f"[STOP LOSS]    {sym}  {pct*100:.2f}%")
        elif pct >= TAKE_PROFIT_PCT:
            exits[sym] = cur
            logger.success(f"[TAKE PROFIT]  {sym} +{pct*100:.2f}%")
        elif USE_TRAILING_STOP:
            peak = entries[sym].get("peak_price", ep)
            if cur > peak:
                entries[sym]["peak_price"] = cur
                peak = cur
            dd = (peak - cur) / peak
            if dd >= TRAILING_STOP_PCT:
                exits[sym] = cur
                logger.warning(f"[TRAILING STOP]{sym} -{dd*100:.2f}% from peak")
    return exits


def _execute_exits(exits: dict, state: dict) -> dict:
    client = _get_mr_trading_client()
    for sym, price in exits.items():
        qty = state["positions"].get(sym, 0)
        if qty == 0:
            continue
        try:
            # Use close_position so Alpaca closes exactly what it holds —
            # avoids fractional qty mismatch between local state and Alpaca's fill record.
            order = client.close_position(sym)
            state["positions"][sym] = 0.0
            state["cash_usd"]      += qty * price
            state["position_entries"].pop(sym, None)
            state.setdefault("trade_log", []).append({
                "time": str(datetime.now(timezone.utc)),
                "symbol": sym, "side": "SELL",
                "qty": qty, "price": price,
                "reason": "stop_tp", "order_id": str(order.id),
            })
        except Exception as e:
            logger.error(f"Exit failed {sym}: {e}")
    return state


def _signal_to_weights(sig: pd.Series) -> pd.Series:
    w   = pd.Series(0.0, index=UNIVERSE)
    pos = sig[sig > 0].nlargest(MAX_LIVE_POSITIONS)
    if pos.empty:
        return w
    raw = pos / pos.sum()
    for _ in range(20):
        over = raw > MAX_POSITION_SIZE
        if not over.any():
            break
        raw[over] = MAX_POSITION_SIZE
        left = 1.0 - raw[over].sum()
        unc  = ~over
        if unc.any() and raw[unc].sum() > 0:
            raw[unc] = raw[unc] / raw[unc].sum() * left
    w[pos.index] = raw.clip(upper=MAX_POSITION_SIZE)
    return w.reindex(UNIVERSE, fill_value=0)


def _execute_rebalance(target_w: pd.Series, state: dict,
                        nav: float, prices: dict) -> dict:
    if not _market_is_open():
        logger.warning("Market closed — skipping rebalance")
        return state

    client   = _get_mr_trading_client()
    cur_vals = {s: state["positions"].get(s, 0) * prices.get(s, 0) for s in UNIVERSE}
    cur_w    = {s: v / nav for s, v in cur_vals.items()} if nav > 0 else {}

    sells, buys = [], []
    for sym in UNIVERSE:
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
            filled_qty = qty
            if side == OrderSide.BUY:
                state["positions"][sym] = state["positions"].get(sym, 0) + filled_qty
                state["cash_usd"]      -= filled_qty * price
                if state["positions"].get(sym, 0) == filled_qty:
                    state.setdefault("position_entries", {})[sym] = {
                        "entry_price": price,
                        "entry_date":  str(datetime.now(timezone.utc)),
                        "peak_price":  price,
                    }
            else:
                new_qty = max(state["positions"].get(sym, 0) - filled_qty, 0)
                state["positions"][sym] = new_qty
                state["cash_usd"]      += filled_qty * price
                if new_qty == 0:
                    state.setdefault("position_entries", {}).pop(sym, None)
            state.setdefault("trade_log", []).append({
                "time": str(datetime.now(timezone.utc)),
                "symbol": sym, "side": side.value,
                "qty": round(filled_qty, 4), "price": price,
                "reason": "macro_rebalance", "order_id": str(order.id),
            })
            logger.info(f"  {side.value:4s} {sym:5s}  qty={filled_qty:.3f}  @ ${price:.2f}")
        except Exception as e:
            logger.error(f"Order failed {sym}: {e}")

    return state


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE — scheduled jobs
# ══════════════════════════════════════════════════════════════════════════════

def price_monitor_job():
    """Check stop loss / trailing stop / take profit every PRICE_MONITOR_SECS."""
    state = _load_state()
    open_pos = {s: q for s, q in state["positions"].items() if q != 0}
    if not open_pos:
        return

    prices = _fetch_prices()
    exits  = _check_exits(state, prices)
    if exits:
        state = _execute_exits(exits, state)
        nav   = _compute_nav(state, prices)
        state["nav_history"].append({
            "date": str(datetime.now(timezone.utc)),
            "nav":  round(nav, 2), "event": "stop_tp",
        })
        _save_state(state)


def signal_rebalance_job():
    """Recompute MacroRegime signals and rebalance if weight delta exceeds threshold."""
    logger.info("-" * 55)
    logger.info(f"Macro signal recompute — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    try:
        universe_data = get_universe_ohlcv(use_cache=False)
        close   = build_close_matrix(universe_data)
        returns = build_return_matrix(close)
        high_df = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
        low_df  = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})
    except Exception as e:
        logger.error(f"Data fetch failed — skipping: {e}")
        return

    strategy = MacroRegimeStrategy()
    signals  = strategy.run(close=close, returns=returns,
                             high=high_df, low=low_df)
    sig_row  = signals.iloc[-1].reindex(UNIVERSE, fill_value=0)

    cov = (
        returns.iloc[-RISK_LOOKBACK_DAYS:]
        .cov() * TRADING_DAYS
    )
    portfolio   = MaxSharpePortfolio()
    new_weights = pd.Series(
        portfolio.compute_weights(sig_row, cov, long_short=False)
    ).reindex(UNIVERSE, fill_value=0)
    new_weights = _signal_to_weights(sig_row)  # cap + top-N filter

    state  = _load_state()
    prices = _fetch_prices()
    nav    = _compute_nav(state, prices)

    prev_w = pd.Series({
        sym: (state["positions"].get(sym, 0) * prices.get(sym, 0)) / nav
        for sym in UNIVERSE
    }, dtype=float).fillna(0)

    delta = (new_weights - prev_w).abs().sum()
    logger.info(f"Weight delta: {delta:.4f}  (threshold={REBALANCE_THRESHOLD})")

    if delta < REBALANCE_THRESHOLD:
        logger.info("Signal unchanged — holding.")
        return

    state = _execute_rebalance(new_weights, state, nav, prices)
    state["current_weights"] = new_weights.to_dict()
    state["last_run"]        = str(datetime.now(timezone.utc))
    nav = _compute_nav(state, prices)
    state["nav_history"].append({
        "date": str(datetime.now(timezone.utc)),
        "nav":  round(nav, 2), "event": "rebalance",
    })
    _save_state(state)
    logger.success(f"Rebalance complete. NAV=${nav:,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE  —  start scheduler
# ══════════════════════════════════════════════════════════════════════════════

def run_live(run_now: bool = False):
    mode = "PAPER" if PAPER_TRADING else "LIVE ⚠️"
    logger.info("=" * 60)
    logger.info(f"MACRO REGIME TRADER — LIVE ENGINE [{mode}]")
    logger.info(f"  Price monitor : every {PRICE_MONITOR_SECS}s")
    logger.info(f"  Rebalance     : 09:35 ET + 15:50 ET, Mon–Fri")
    logger.info(f"  State file    : {STATE_FILE}")
    logger.info("=" * 60)

    # Check connectivity using MR-specific keys
    try:
        acct = _get_mr_trading_client().get_account()
        logger.info(f"Alpaca (MR paper) connected — equity=${float(acct.equity):,.2f}")
    except Exception as e:
        logger.error(f"Alpaca (MR paper) connection failed: {e}")
        sys.exit(1)

    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        func=price_monitor_job, trigger="interval",
        seconds=PRICE_MONITOR_SECS, id="macro_price_monitor",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        func=signal_rebalance_job, trigger="cron",
        day_of_week="mon-fri", hour=9, minute=35,
        timezone="America/New_York", id="macro_signal_open",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        func=signal_rebalance_job, trigger="cron",
        day_of_week="mon-fri", hour=15, minute=50,
        timezone="America/New_York", id="macro_signal_close",
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

def main():
    _setup_logger()

    parser = argparse.ArgumentParser(description="Macro Regime Standalone Trader")
    parser.add_argument(
        "--mode", choices=["backtest", "live", "full"], default="backtest",
        help=(
            "backtest — run backtest and print results. "
            "live     — start live engine immediately. "
            "full     — run backtest first; if Sharpe ≥ 0.5 proceed to live."
        ),
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download of market data")
    parser.add_argument("--run-now", action="store_true",
                        help="Fire one rebalance immediately on live start")
    parser.add_argument("--min-sharpe", type=float, default=0.5,
                        help="Minimum backtest Sharpe to proceed to live in full mode")
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
