"""
db/connection.py — Thread-safe PostgreSQL connection pool.
DB_URL is read from the environment (set in Alpaca.env).
"""

import os
from loguru import logger

_pool = None


def _get_pool():
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool

    try:
        from psycopg2 import pool as pg_pool
    except ImportError:
        raise RuntimeError(
            "psycopg2 not installed — run: pip install psycopg2-binary"
        )

    url = os.environ.get("DB_URL")
    if not url:
        raise RuntimeError("DB_URL not set — skipping PostgreSQL")

    _pool = pg_pool.ThreadedConnectionPool(1, 5, dsn=url)
    logger.info("PostgreSQL connection pool initialised (stock_hybrid)")
    return _pool


def get_conn():
    return _get_pool().getconn()


def put_conn(conn):
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass
