"""
db/connection.py — Thread-safe PostgreSQL connection pool.
DB_URL is read from the environment (set in Alpaca.env).
"""

import os
from psycopg2 import pool as pg_pool
from loguru import logger

_pool: pg_pool.ThreadedConnectionPool | None = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        url = os.environ.get(
            "DB_URL",
            "postgresql://***REDACTED***",
        )
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
