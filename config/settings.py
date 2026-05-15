"""
config/settings.py
==================
Central configuration for the Stock Hybrid Strategy system.
All tuneable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data" / "cache"
LOG_DIR    = ROOT_DIR / "logs"
RESULT_DIR = ROOT_DIR / "results"

for _d in [DATA_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Environment ────────────────────────────────────────────────────────────────
for _env_path in [ROOT_DIR / "Alpaca.env", ROOT_DIR / ".env"]:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

PAPER_API_KEY    = os.getenv("ALPACA_PAPER_API_KEY")
PAPER_API_SECRET = os.getenv("ALPACA_PAPER_API_SECRET")
LIVE_API_KEY     = os.getenv("ALPACA_LIVE_API_KEY")
LIVE_API_SECRET  = os.getenv("ALPACA_LIVE_API_SECRET")

# ── Mode ───────────────────────────────────────────────────────────────────────
# True  → paper trading (Alpaca paper environment, fake money)
# False → live trading  (real money — only flip when ready)
PAPER_TRADING = True

# ── Universe ───────────────────────────────────────────────────────────────────
# 12 liquid large-cap US stocks + ETFs across diverse sectors
UNIVERSE = [
    "AAPL",   # Apple         — Technology
    "MSFT",   # Microsoft     — Technology
    "GOOGL",  # Alphabet      — Technology
    "AMZN",   # Amazon        — Consumer / Cloud
    "NVDA",   # NVIDIA        — Semiconductors
    "META",   # Meta          — Social Media
    "JPM",    # JPMorgan      — Financials
    "JNJ",    # J&J           — Healthcare
    "XOM",    # Exxon         — Energy
    "UNH",    # UnitedHealth  — Healthcare
    "SPY",    # S&P 500 ETF   — Benchmark / Macro hedge
    "QQQ",    # Nasdaq ETF    — Tech benchmark
]

# ── Data ───────────────────────────────────────────────────────────────────────
TIMEFRAME        = "1Day"          # Alpaca timeframe string
BACKTEST_START   = "2022-01-01"    # Start date for historical data
BACKTEST_END     = None            # None = today
CACHE_EXPIRY_HOURS = 6             # Re-fetch if cached data is older than this

# ── Risk & Portfolio ───────────────────────────────────────────────────────────
TRADING_DAYS     = 252             # US stock market (not 365 like crypto)
MIN_ANNUALIZED_VOL  = 0.03         # 3% vol floor
MAX_WEIGHT_SUM      = 1.00         # |w|_1 ≤ 100%
LONG_SHORT          = False        # Long-only by default (no margin/shorting in paper)
RISK_LOOKBACK_DAYS  = 126          # 6-month covariance (~126 trading days)
MAX_POSITION_SIZE   = 0.20         # Single stock cap at 20%
TRANSACTION_COST_BP = 0            # Alpaca charges 0 commission on stocks
SLIPPAGE_BP         = 3            # Assumed slippage in backtest

# ── Position-Level Risk Management ─────────────────────────────────────────────
STOP_LOSS_PCT     = 0.05           # 5% hard stop loss
TAKE_PROFIT_PCT   = 0.10           # 10% take profit
TRAILING_STOP_PCT = 0.07           # 7% trailing stop
USE_TRAILING_STOP = True

# ── Execution ──────────────────────────────────────────────────────────────────
PORTFOLIO_USD        = 10_000      # Starting capital in USD
MIN_ORDER_USD        = 1.0         # Alpaca supports fractional shares — $1 minimum
REBALANCE_THRESHOLD  = 0.03        # Min total |Δweight| to trigger a rebalance
MAX_LIVE_POSITIONS   = 8           # Max simultaneous long positions

# ── Live trading schedule ──────────────────────────────────────────────────────
PRICE_MONITOR_SECS    = 60         # Check stop/TP every 60 seconds
# Signal recompute: 09:35 ET (market open) + 15:50 ET (market close), Mon–Fri

# ── Strategy Params ────────────────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "momentum": {
        "lookback_long":  252,     # 12-month return
        "lookback_short": 21,      # Skip last month
        "top_n":          4,
        "bottom_n":       0,       # Long-only: no shorts
    },
    "mean_reversion": {
        "zscore_window":  20,
        "entry_z":        2.0,
        "exit_z":         0.5,
    },
    "risk_parity": {
        "lookback":       63,      # 3-month vol
    },
    "cross_sectional_momentum": {
        "lookback":       20,
        "rank_method":    "min",
    },
    "vol_breakout": {
        "atr_period":     14,
        "atr_multiplier": 2.0,
    },
    "ml_signal": {
        "feature_lookbacks": [1, 3, 5, 10, 21],
        "train_window":      126,   # bars of history used for each training window
        "retrain_every":     21,    # retrain every ~1 month (no future data ever used)
        "n_estimators":      100,
        "max_depth":         4,
        "learning_rate":     0.05,
    },
}

# ── Portfolio Construction ─────────────────────────────────────────────────────
# Name of the default portfolio to use in single-portfolio backtest mode.
# Options: "max_sharpe" | "equal_weight" | "min_variance" | "risk_parity" | "signal_weighted"
PORTFOLIO_TYPE = "max_sharpe"

# Optional per-portfolio parameter overrides.  Keys must match portfolio names.
# Leave empty to use each portfolio's built-in defaults.
PORTFOLIO_PARAMS: dict = {}

# ── Seasonality ────────────────────────────────────────────────────────────────
SEASONALITY_MIN_PERIODS = 20
REGIME_MA_WINDOW        = 200      # SPY 200-day MA for bull/bear regime
STRATEGY_SELECT_METRIC  = "sharpe"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL    = "INFO"
LOG_ROTATION = "1 week"
