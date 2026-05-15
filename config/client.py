"""
config/client.py
================
Alpaca client factory.

Paper trading  → paper=True  (Alpaca paper environment)
Live trading   → paper=False (real money — guard with PAPER_TRADING flag)
"""

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from loguru import logger

from config.settings import (
    PAPER_TRADING,
    PAPER_API_KEY, PAPER_API_SECRET,
    LIVE_API_KEY,  LIVE_API_SECRET,
)


def get_trading_client() -> TradingClient:
    """Return an Alpaca TradingClient for paper or live trading."""
    if PAPER_TRADING:
        logger.debug("Using Alpaca PAPER trading client")
        return TradingClient(PAPER_API_KEY, PAPER_API_SECRET, paper=True)
    else:
        logger.warning("Using Alpaca LIVE trading client — real money at risk")
        return TradingClient(LIVE_API_KEY, LIVE_API_SECRET, paper=False)


def get_data_client() -> StockHistoricalDataClient:
    """Return an Alpaca data client (same for paper and live)."""
    if PAPER_TRADING:
        return StockHistoricalDataClient(PAPER_API_KEY, PAPER_API_SECRET)
    return StockHistoricalDataClient(LIVE_API_KEY, LIVE_API_SECRET)


def check_connectivity():
    """Verify API keys are set and Alpaca is reachable."""
    try:
        client = get_trading_client()
        account = client.get_account()
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        logger.info(
            f"Alpaca connected [{mode}] | "
            f"Equity: ${float(account.equity):,.2f} | "
            f"Buying power: ${float(account.buying_power):,.2f} | "
            f"Status: {account.status}"
        )
    except Exception as e:
        logger.error(f"Alpaca connectivity check failed: {e}")
        raise
