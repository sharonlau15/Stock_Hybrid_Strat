"""
data/ingestion.py
=================
Fetch OHLCV bars from Alpaca's market data API.
Results are cached as Parquet files to avoid redundant API calls.
Cache is invalidated after CACHE_EXPIRY_HOURS.

Public API
----------
    get_ohlcv(symbol)              → pd.DataFrame
    get_universe_ohlcv()           → dict[str, pd.DataFrame]
    build_close_matrix(data)       → pd.DataFrame
    build_return_matrix(close)     → pd.DataFrame
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.client import get_data_client
from config.settings import (
    UNIVERSE, BACKTEST_START, BACKTEST_END,
    DATA_DIR, CACHE_EXPIRY_HOURS,
)


def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}_1Day.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_EXPIRY_HOURS


def get_ohlcv(
    symbol: str,
    start: str = BACKTEST_START,
    end: str | None = BACKTEST_END,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single symbol from Alpaca.

    Returns
    -------
    pd.DataFrame — index=date (UTC), columns: open, high, low, close, volume
    """
    path = _cache_path(symbol)

    if use_cache and _cache_is_fresh(path):
        logger.debug(f"Cache hit: {symbol}")
        return pd.read_parquet(path)

    logger.info(f"Fetching {symbol} from {start}")

    client = get_data_client()
    end_dt = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start,
        end=end_dt,
        adjustment="all",   # split + dividend adjusted
    )

    bars = client.get_stock_bars(request)
    df   = bars.df

    if df.empty:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()

    # Alpaca returns a MultiIndex (symbol, timestamp) — drop symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df.to_parquet(path)
    logger.success(f"Cached {symbol}: {len(df)} bars")
    return df


def get_universe_ohlcv(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for every symbol in UNIVERSE."""
    universe_data = {}
    for sym in tqdm(UNIVERSE, desc="Fetching universe OHLCV"):
        try:
            df = get_ohlcv(sym, use_cache=use_cache)
            if not df.empty:
                universe_data[sym] = df
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")
        time.sleep(0.1)

    logger.info(f"Universe loaded: {len(universe_data)}/{len(UNIVERSE)} symbols")
    return universe_data


def build_close_matrix(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack close prices into a wide DataFrame (dates × symbols)."""
    closes = {sym: df["close"] for sym, df in universe_data.items()}
    matrix = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
    return matrix


def build_return_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from close matrix."""
    return np.log(close / close.shift(1)).dropna(how="all")
