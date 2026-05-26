"""
execution/live_engine.py
========================
Real-time trading engine using Alpaca.

Two concurrent loops
--------------------
 FAST  (every PRICE_MONITOR_SECS, default 60s)
   → Fetch current prices for all open positions
   → Check stop loss / trailing stop / take profit
   → Close breached positions immediately

 SIGNAL (cron, twice daily — US/Eastern, Mon–Fri)
   09:35 ET  market-open recompute  → trade on opening prices once settled
   15:50 ET  market-close recompute → position into next day's close
   → Rebalance only if total |Δweight| > REBALANCE_THRESHOLD

Market-hours guard: orders are only placed when the US market is open.
Both paper trading and live trading use the same code path — controlled
by PAPER_TRADING in settings.py.
"""

import time
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

from config.client import get_trading_client
from config.settings import (
    UNIVERSE, PORTFOLIO_USD, MIN_ORDER_USD, PAPER_TRADING,
    RESULT_DIR, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT, USE_TRAILING_STOP,
    PRICE_MONITOR_SECS,
    REBALANCE_THRESHOLD, MAX_LIVE_POSITIONS, MAX_POSITION_SIZE,
)
from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
from utils.logger import setup_logger
from db.state import (
    load_state,
    save_state,
    load_nav_history_for_report,
    load_trade_log_for_report,
)
from db.engine_controls import load_engine_controls, save_engine_controls

from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ── Market hours ───────────────────────────────────────────────────────────────
def market_is_open() -> bool:
    """Return True if the US stock market is currently open via Alpaca's clock."""
    try:
        client = get_trading_client()
        clock  = client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.warning(f"Could not fetch market clock: {e}")
        return False


# ── State management ───────────────────────────────────────────────────────────
# ── Portfolio NAV ──────────────────────────────────────────────────────────────
def compute_nav(state: dict, current_prices: dict) -> float:
    nav = state["cash_usd"]
    for sym, qty in state["positions"].items():
        if qty != 0 and sym in current_prices:
            nav += qty * current_prices[sym]
    return nav


# ── Fetch current prices ───────────────────────────────────────────────────────
def fetch_current_prices() -> dict:
    """Fetch latest trade prices for all universe symbols via Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from config.settings import PAPER_TRADING, PAPER_API_KEY, PAPER_API_SECRET, LIVE_API_KEY, LIVE_API_SECRET

    key    = PAPER_API_KEY if PAPER_TRADING else LIVE_API_KEY
    secret = PAPER_API_SECRET if PAPER_TRADING else LIVE_API_SECRET
    client = StockHistoricalDataClient(key, secret)

    prices = {}
    try:
        request = StockLatestTradeRequest(symbol_or_symbols=UNIVERSE)
        trades  = client.get_stock_latest_trade(request)
        for sym, trade in trades.items():
            prices[sym] = float(trade.price)
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
    return prices


# ── Stop loss / Take profit / Trailing stop ────────────────────────────────────
def check_position_exits(state: dict, current_prices: dict) -> dict:
    """Return {symbol: exit_price} for positions that breached a risk limit."""
    exits = {}
    entries = state.setdefault("position_entries", {})

    for sym, qty in state["positions"].items():
        if qty == 0 or sym not in entries:
            continue

        entry_price = entries[sym]["entry_price"]
        current     = current_prices.get(sym)
        if current is None:
            continue

        pct = (current - entry_price) / entry_price

        if pct <= -STOP_LOSS_PCT:
            exits[sym] = current
            logger.warning(f"[STOP LOSS] {sym} {pct*100:.2f}%")
            continue

        if pct >= TAKE_PROFIT_PCT:
            exits[sym] = current
            logger.success(f"[TAKE PROFIT] {sym} +{pct*100:.2f}%")
            continue

        if USE_TRAILING_STOP and TRAILING_STOP_PCT > 0:
            peak = entries[sym].get("peak_price", entry_price)
            if current > peak:
                entries[sym]["peak_price"] = current
                peak = current
            dd = (peak - current) / peak
            if dd >= TRAILING_STOP_PCT:
                exits[sym] = current
                logger.warning(f"[TRAILING STOP] {sym} -{dd*100:.2f}% from peak")

    return exits


# ── Execute a single exit ──────────────────────────────────────────────────────
def execute_exits(forced_exits: dict, state: dict) -> dict:
    """Close specific positions immediately."""
    client = get_trading_client()

    for sym, exit_price in forced_exits.items():
        qty = state["positions"].get(sym, 0)
        if qty == 0:
            continue
        try:
            order = client.submit_order(MarketOrderRequest(
                symbol        = sym,
                qty           = qty,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.DAY,
            ))
            proceeds = qty * exit_price
            state["positions"][sym] = 0.0
            state["cash_usd"]      += proceeds
            state["position_entries"].pop(sym, None)

            state.setdefault("trade_log", []).append({
                "time":     str(datetime.now(timezone.utc)),
                "symbol":   sym,
                "side":     "SELL",
                "qty":      qty,
                "price":    exit_price,
                "reason":   "stop_tp",
                "order_id": str(order.id),
            })
            logger.info(f"[EXIT] SELL {qty} {sym} @ ${exit_price:.2f}")
        except Exception as e:
            logger.error(f"Exit order failed {sym}: {e}")

    return state


# ── Signal → weights conversion ────────────────────────────────────────────────
def signal_to_weights(signal_series: pd.Series) -> pd.Series:
    """Convert raw signals to long-only position weights, capped at MAX_POSITION_SIZE."""
    weights  = pd.Series(0.0, index=UNIVERSE)
    pos_sigs = signal_series[signal_series > 0].nlargest(MAX_LIVE_POSITIONS)

    if pos_sigs.empty:
        return weights

    raw = pos_sigs / pos_sigs.sum()
    for _ in range(20):
        over = raw > MAX_POSITION_SIZE
        if not over.any():
            break
        raw[over] = MAX_POSITION_SIZE
        leftover  = 1.0 - raw[over].sum()
        uncapped  = ~over
        if uncapped.any() and raw[uncapped].sum() > 0:
            raw[uncapped] = raw[uncapped] / raw[uncapped].sum() * leftover

    weights[pos_sigs.index] = raw.clip(upper=MAX_POSITION_SIZE)
    return weights.reindex(UNIVERSE, fill_value=0)


# ── Full rebalance ─────────────────────────────────────────────────────────────
def execute_rebalance(target_weights: pd.Series, state: dict, nav: float,
                      current_prices: dict) -> dict:
    """Place orders to shift current weights towards target weights."""
    if not market_is_open():
        logger.warning("Market is closed — skipping rebalance orders")
        return state

    client = get_trading_client()

    current_values = {sym: state["positions"].get(sym, 0) * current_prices.get(sym, 0)
                      for sym in UNIVERSE}
    current_w = {sym: v / nav for sym, v in current_values.items()} if nav > 0 else {}

    # Sells first (free up cash), then buys
    orders = []
    for sym in UNIVERSE:
        target_w  = float(target_weights.get(sym, 0.0))
        current_w_ = current_w.get(sym, 0.0)
        delta_usdt = (target_w - current_w_) * nav
        if abs(delta_usdt) < MIN_ORDER_USD:
            continue
        orders.append((sym, delta_usdt))

    for sym, delta_usd in sorted(orders, key=lambda x: x[1]):  # sells (negative) first
        price = current_prices.get(sym)
        if not price:
            continue
        side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
        qty  = round(abs(delta_usd) / price, 6)
        if qty <= 0:
            continue

        try:
            order = client.submit_order(MarketOrderRequest(
                symbol        = sym,
                qty           = qty,
                side          = side,
                time_in_force = TimeInForce.DAY,
            ))
            cost = qty * price
            mode = "PAPER" if PAPER_TRADING else "LIVE"
            logger.success(
                f"[{mode}] {side.value} {qty:.4f} {sym} @ ${price:.2f} "
                f"| Δ=${delta_usd:+.2f} | orderId={order.id}"
            )

            if side == OrderSide.BUY:
                state["positions"][sym] = state["positions"].get(sym, 0) + qty
                state["cash_usd"]      -= cost
                if state["positions"].get(sym, 0) == qty:
                    state["position_entries"][sym] = {
                        "entry_price": price,
                        "entry_date":  str(datetime.now(timezone.utc)),
                        "peak_price":  price,
                    }
                    logger.info(
                        f"   SL=${price*(1-STOP_LOSS_PCT):.2f} "
                        f"| TP=${price*(1+TAKE_PROFIT_PCT):.2f}"
                    )
            else:
                new_qty = max(state["positions"].get(sym, 0) - qty, 0)
                state["positions"][sym] = new_qty
                state["cash_usd"]      += cost
                if new_qty == 0:
                    state["position_entries"].pop(sym, None)

            state.setdefault("trade_log", []).append({
                "time":     str(datetime.now(timezone.utc)),
                "symbol":   sym,
                "side":     side.value,
                "qty":      qty,
                "price":    price,
                "reason":   "signal_rebalance",
                "order_id": str(order.id),
            })
        except Exception as e:
            logger.error(f"Order failed {sym}: {e}")

    return state


# ══════════════════════════════════════════════════════════════════════════════
# JOB 1 — PRICE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
def price_monitor_job():
    """Poll prices and immediately exit any position that breaches a risk limit."""
    controls = load_engine_controls()

    # Close-all: market-sell everything, then stay halted
    if controls.get("close_all_triggered"):
        logger.warning("Kill switch close-all triggered — liquidating all positions")
        state  = load_state()
        prices = fetch_current_prices()
        all_exits = {
            sym: prices[sym]
            for sym, qty in state["positions"].items()
            if qty != 0 and sym in prices
        }
        if all_exits:
            state = execute_exits(all_exits, state)
            nav   = compute_nav(state, prices)
            state["nav_history"].append({
                "date":  str(datetime.now(timezone.utc)),
                "nav":   round(nav, 2),
                "event": "kill_switch_close_all",
            })
            save_state(state)
        save_engine_controls(
            kill_switch=True, kill_mode="halt",
            close_all_triggered=False,
            note=controls.get("note", ""),
        )
        return

    state = load_state()
    open_positions = {sym: qty for sym, qty in state["positions"].items() if qty != 0}
    if not open_positions:
        return

    prices       = fetch_current_prices()
    forced_exits = check_position_exits(state, prices)

    if forced_exits:
        state = execute_exits(forced_exits, state)
        nav   = compute_nav(state, prices)
        state["nav_history"].append({
            "date":  str(datetime.now(timezone.utc)),
            "nav":   round(nav, 2),
            "event": "stop_tp_exit",
        })
        save_state(state)


# ══════════════════════════════════════════════════════════════════════════════
# JOB 2 — SIGNAL RECOMPUTE + CONDITIONAL REBALANCE
# ══════════════════════════════════════════════════════════════════════════════
def signal_rebalance_job(strategies: list, signals_dict: dict):
    """Recompute signals and rebalance only if weight delta exceeds threshold."""
    controls = load_engine_controls()
    if controls.get("kill_switch"):
        logger.warning("Kill switch ACTIVE — skipping signal recompute")
        return

    logger.info("=" * 55)
    logger.info(f"Signal recompute — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    state = load_state()

    # 1. Fresh data
    try:
        universe_data = get_universe_ohlcv(use_cache=False)
        close   = build_close_matrix(universe_data)
        returns = build_return_matrix(close)
        high_df = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
        low_df  = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})
    except Exception as e:
        logger.error(f"Data fetch failed — skipping cycle: {e}")
        return

    # 2. Recompute signals for all strategies
    for strategy in strategies:
        try:
            signals_dict[strategy.name] = strategy.run(
                close=close, returns=returns, high=high_df, low=low_df,
            )
        except Exception as e:
            logger.error(f"Signal failed for {strategy.name}: {e}")

    # 3. Simple strategy selection: pick the one with best recent Sharpe
    #    (placeholder — replace with a proper SeasonalityAnalyzer if needed)
    best_name   = max(signals_dict, key=lambda n: _recent_sharpe(signals_dict[n]))
    blended_sig = signals_dict[best_name].iloc[-1].reindex(UNIVERSE, fill_value=0)
    state["active_strategy"] = best_name
    logger.info(f"Active strategy: {best_name}")

    # 4. Compute new target weights
    new_weights = signal_to_weights(blended_sig)

    # 5. Compare to actual current weights
    prices = fetch_current_prices()
    nav    = compute_nav(state, prices)
    prev_w = pd.Series({
        sym: (state["positions"].get(sym, 0) * prices.get(sym, 0)) / nav
        for sym in UNIVERSE
    }, dtype=float).fillna(0)

    weight_delta = (new_weights.reindex(UNIVERSE, fill_value=0) - prev_w).abs().sum()
    logger.info(
        f"Total weight delta: {weight_delta:.4f} "
        f"(threshold={REBALANCE_THRESHOLD}) "
        f"→ {'⚡ REBALANCE' if weight_delta >= REBALANCE_THRESHOLD else '✋ HOLD'}"
    )

    if weight_delta < REBALANCE_THRESHOLD:
        logger.info("Signal unchanged — holding.")
        return

    # 6. Rebalance
    state = execute_rebalance(new_weights, state, nav, prices)
    state["current_weights"] = new_weights.to_dict()
    state["last_run"]        = str(datetime.now(timezone.utc))
    state["nav_history"].append({
        "date":  str(datetime.now(timezone.utc)),
        "nav":   round(nav, 2),
        "event": "signal_rebalance",
    })
    save_state(state)
    logger.success(f"Rebalance complete. NAV=${nav:,.2f}")


def _recent_sharpe(sig_df: pd.DataFrame, window: int = 20) -> float:
    """Quick Sharpe estimate from a signal DataFrame's recent rows."""
    try:
        recent = sig_df.iloc[-window:].mean(axis=1)
        return float(recent.mean() / recent.std()) if recent.std() > 0 else 0.0
    except Exception:
        return 0.0


# ── Scheduler ─────────────────────────────────────────────────────────────────
def start_scheduler(strategies: list, signals_dict: dict, run_now: bool = False):
    """Start the two concurrent APScheduler jobs."""
    setup_logger()
    mode = "PAPER" if PAPER_TRADING else "LIVE"

    scheduler = BackgroundScheduler(timezone="UTC")

    # Job 1 — fast price / risk monitor (interval)
    scheduler.add_job(
        func          = price_monitor_job,
        trigger       = "interval",
        seconds       = PRICE_MONITOR_SECS,
        id            = "price_monitor",
        max_instances = 1,
        coalesce      = True,
    )

    # Job 2a — market-open recompute (09:35 ET)
    scheduler.add_job(
        func          = signal_rebalance_job,
        trigger       = "cron",
        day_of_week   = "mon-fri",
        hour          = 9,
        minute        = 35,
        timezone      = "America/New_York",
        args          = [strategies, signals_dict],
        id            = "signal_open",
        name          = "Signal recompute — market open (09:35 ET)",
        max_instances = 1,
        coalesce      = True,
    )

    # Job 2b — market-close recompute (15:50 ET)
    scheduler.add_job(
        func          = signal_rebalance_job,
        trigger       = "cron",
        day_of_week   = "mon-fri",
        hour          = 15,
        minute        = 50,
        timezone      = "America/New_York",
        args          = [strategies, signals_dict],
        id            = "signal_close",
        name          = "Signal recompute — market close (15:50 ET)",
        max_instances = 1,
        coalesce      = True,
    )

    scheduler.start()

    logger.info("=" * 55)
    logger.info(f"🚀  STOCK ALGO ENGINE — ALPACA [{mode}]")
    logger.info("=" * 55)
    logger.info(f"  Signal recompute : 09:35 ET (open) + 15:50 ET (close), Mon–Fri")
    logger.info(f"  Stop/TP monitor  : every {PRICE_MONITOR_SECS}s")
    logger.info(f"  Rebalance trigger: |Δweight| > {REBALANCE_THRESHOLD:.0%}")
    logger.info(f"  Risk             : SL={STOP_LOSS_PCT*100:.0f}%  TP={TAKE_PROFIT_PCT*100:.0f}%  Trail={TRAILING_STOP_PCT*100:.0f}%")
    logger.info(f"  Universe         : {', '.join(UNIVERSE)}")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 55)

    if run_now:
        signal_rebalance_job(strategies, signals_dict)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        _final_report()
        logger.success("Engine stopped.")


def _final_report():
    try:
        nav_hist = load_nav_history_for_report()
        if nav_hist:
            start = nav_hist[0]["nav"]
            end   = nav_hist[-1]["nav"]
            logger.info(f"Session return: {(end-start)/start*100:+.2f}%  (${start:,.2f} → ${end:,.2f})")
        trades = pd.DataFrame(load_trade_log_for_report())
        if not trades.empty:
            trades.to_csv(RESULT_DIR / "trade_log.csv", index=False)
            logger.success(f"Trade log saved → results/trade_log.csv")
    except Exception as e:
        logger.error(f"Final report error: {e}")
