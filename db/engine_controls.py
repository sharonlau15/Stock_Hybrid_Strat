"""
db/engine_controls.py — Kill switch and engine control state.
"""

from contextlib import contextmanager
from loguru import logger

from db.connection import get_conn, put_conn


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


def _ensure_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS engine_controls (
            id                  INTEGER PRIMARY KEY DEFAULT 1,
            kill_switch         BOOLEAN NOT NULL DEFAULT FALSE,
            kill_mode           TEXT    NOT NULL DEFAULT 'halt',
            close_all_triggered BOOLEAN NOT NULL DEFAULT FALSE,
            note                TEXT,
            updated_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        INSERT INTO engine_controls (id) VALUES (1)
        ON CONFLICT (id) DO NOTHING
    """)


def load_engine_controls() -> dict:
    """Load kill switch state. Returns safe defaults if DB unavailable."""
    try:
        with _db() as conn:
            _ensure_table(conn)
            cur = conn.cursor()
            cur.execute("""
                SELECT kill_switch, kill_mode, close_all_triggered, note
                FROM engine_controls WHERE id = 1
            """)
            row = cur.fetchone()
            if row is None:
                return _default_controls()
            return {
                "kill_switch":         bool(row[0]),
                "kill_mode":           row[1] or "halt",
                "close_all_triggered": bool(row[2]),
                "note":                row[3] or "",
            }
    except Exception as e:
        logger.warning(f"load_engine_controls failed (using defaults): {e}")
        return _default_controls()


def save_engine_controls(
    kill_switch:         bool = False,
    kill_mode:           str  = "halt",
    close_all_triggered: bool = False,
    note:                str  = "",
) -> None:
    """Persist engine control state to PostgreSQL."""
    try:
        with _db() as conn:
            _ensure_table(conn)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO engine_controls
                    (id, kill_switch, kill_mode, close_all_triggered, note, updated_at)
                VALUES (1, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    kill_switch         = EXCLUDED.kill_switch,
                    kill_mode           = EXCLUDED.kill_mode,
                    close_all_triggered = EXCLUDED.close_all_triggered,
                    note                = EXCLUDED.note,
                    updated_at          = EXCLUDED.updated_at
            """, (kill_switch, kill_mode, close_all_triggered, note))
    except Exception as e:
        logger.error(f"save_engine_controls failed: {e}")
        raise


def _default_controls() -> dict:
    return {
        "kill_switch":         False,
        "kill_mode":           "halt",
        "close_all_triggered": False,
        "note":                "",
    }
