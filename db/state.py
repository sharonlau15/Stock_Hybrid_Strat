"""
db/state.py — PostgreSQL-backed replacement for live_state.json (stock_hybrid).

Mirrors the same offset-tracking pattern as crypto_algo/db/state.py but
without hypothetical portfolios and using cash_usd (not cash_usdt).
"""

import json
from contextlib import contextmanager
from loguru import logger

from db.connection import get_conn, put_conn
from config.settings import UNIVERSE, PORTFOLIO_USD


@contextmanager
def _db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


def load_state() -> dict:
    """
    Load state from PostgreSQL. Returns the empty default state when no row
    exists yet (first run) — stock_hybrid has no Alpaca bootstrap equivalent.
    """
    try:
        with _db() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT positions, cash_usd, current_weights, active_strategy, last_run
                FROM live_state WHERE id = 1
            """)
            row = cur.fetchone()

            if row is None:
                return _empty_state()

            positions, cash_usd, current_weights, active_strategy, last_run = row
            state: dict = {
                "positions":        positions or {sym: 0.0 for sym in UNIVERSE},
                "cash_usd":         float(cash_usd or PORTFOLIO_USD),
                "current_weights":  current_weights or {},
                "active_strategy":  active_strategy,
                "last_run":         str(last_run) if last_run else None,
                "position_entries": {},
                "nav_history":      [],
                "trade_log":        [],
            }

            # Position entries
            cur.execute("""
                SELECT symbol, entry_price, entry_date, peak_price
                FROM position_entries
            """)
            for sym, ep, ed, pp in cur.fetchall():
                state["position_entries"][sym] = {
                    "entry_price": float(ep),
                    "entry_date":  str(ed),
                    "peak_price":  float(pp),
                }

            # Recent nav_history
            cur.execute("""
                SELECT recorded_at, nav, event
                FROM nav_history
                ORDER BY recorded_at DESC LIMIT 2880
            """)
            rows = cur.fetchall()
            state["nav_history"] = [
                {"date": str(r[0]), "nav": float(r[1]), "event": r[2]}
                for r in reversed(rows)
            ]

            # Recent trade_log
            cur.execute("""
                SELECT executed_at, symbol, side, qty, price, reason, order_id
                FROM trade_log
                ORDER BY executed_at DESC LIMIT 500
            """)
            rows = cur.fetchall()
            state["trade_log"] = [
                {
                    "time":     str(r[0]),
                    "symbol":   r[1],
                    "side":     r[2],
                    "qty":      float(r[3]),
                    "price":    float(r[4]),
                    "reason":   r[5],
                    "order_id": r[6],
                }
                for r in reversed(rows)
            ]

            # Watermarks for incremental saves
            state["_nav_db_count"]   = len(state["nav_history"])
            state["_trade_db_count"] = len(state["trade_log"])

        return state

    except Exception as e:
        logger.error(f"DB load_state failed: {e}")
        return _empty_state()


def save_state(state: dict):
    """Persist live trading state to PostgreSQL."""
    try:
        with _db() as conn:
            cur = conn.cursor()

            # Upsert core live_state row
            cur.execute("""
                INSERT INTO live_state
                    (id, positions, cash_usd, current_weights, active_strategy, last_run)
                VALUES (1, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    positions       = EXCLUDED.positions,
                    cash_usd        = EXCLUDED.cash_usd,
                    current_weights = EXCLUDED.current_weights,
                    active_strategy = EXCLUDED.active_strategy,
                    last_run        = EXCLUDED.last_run
            """, (
                json.dumps(state.get("positions", {})),
                state.get("cash_usd", PORTFOLIO_USD),
                json.dumps(state.get("current_weights", {})),
                state.get("active_strategy"),
                state.get("last_run"),
            ))

            # Sync position_entries
            cur.execute("DELETE FROM position_entries")
            for sym, entry in state.get("position_entries", {}).items():
                cur.execute("""
                    INSERT INTO position_entries (symbol, entry_price, entry_date, peak_price)
                    VALUES (%s, %s, %s, %s)
                """, (
                    sym,
                    entry["entry_price"],
                    entry["entry_date"],
                    entry["peak_price"],
                ))

            # Append only new nav_history entries
            nav_offset = state.get("_nav_db_count", 0)
            for row in state.get("nav_history", [])[nav_offset:]:
                cur.execute("""
                    INSERT INTO nav_history (recorded_at, nav, event)
                    VALUES (%s, %s, %s)
                """, (row.get("date"), row.get("nav"), row.get("event")))

            # Append only new trade_log entries
            trade_offset = state.get("_trade_db_count", 0)
            for row in state.get("trade_log", [])[trade_offset:]:
                cur.execute("""
                    INSERT INTO trade_log
                        (executed_at, symbol, side, qty, price, reason, order_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    row.get("time"),
                    row.get("symbol"),
                    row.get("side"),
                    row.get("qty"),
                    row.get("price"),
                    row.get("reason"),
                    row.get("order_id"),
                ))

    except Exception as e:
        logger.error(f"DB save_state failed: {e}")
        raise


def load_nav_history_for_report() -> list[dict]:
    """Used by _final_report() to get full nav history from DB."""
    with _db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT recorded_at, nav, event FROM nav_history ORDER BY recorded_at")
        return [{"date": str(r[0]), "nav": float(r[1]), "event": r[2]} for r in cur.fetchall()]


def load_trade_log_for_report() -> list[dict]:
    """Used by _final_report() to get full trade log from DB."""
    with _db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT executed_at, symbol, side, qty, price, reason, order_id
            FROM trade_log ORDER BY executed_at
        """)
        return [
            {
                "time":     str(r[0]),
                "symbol":   r[1],
                "side":     r[2],
                "qty":      float(r[3]),
                "price":    float(r[4]),
                "reason":   r[5],
                "order_id": r[6],
            }
            for r in cur.fetchall()
        ]


def _empty_state() -> dict:
    return {
        "positions":        {sym: 0.0 for sym in UNIVERSE},
        "cash_usd":         float(PORTFOLIO_USD),
        "current_weights":  {},
        "active_strategy":  None,
        "last_run":         None,
        "position_entries": {},
        "nav_history":      [],
        "trade_log":        [],
        "_nav_db_count":    0,
        "_trade_db_count":  0,
    }
