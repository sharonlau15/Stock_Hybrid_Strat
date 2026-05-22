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

DB_URL           = os.getenv("DB_URL")  # Set in .env or environment — never hardcode credentials

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
    "macro_regime": {
        "vix_proxy_window": 20,   # SPY rolling vol window (days)
        "vix_risk_on":      20,   # annualised vol % < this → risk-on  (+1)
        "vix_risk_off":     30,   # annualised vol % ≥ this → risk-off (−1)
        "ma_long":          200,  # SPY trend MA period
        "breadth_ma":       50,   # breadth MA period
        "breadth_on":       0.60, # > 60% above MA → risk-on
        "breadth_off":      0.40, # < 40% above MA → risk-off
        "vol_short":        10,   # short vol window
        "vol_long":         30,   # long vol window
        "mom_long":         252,  # 12-month momentum lookback
        "mom_skip":         21,   # skip last month
    },
    "seasonal_exhaustion_fade": {
        "bb_window":        20,    # Bollinger Band lookback
        "bb_std":           2.0,   # 2.0σ (lowered from 2.5 — breach too rare on large caps)
        "atr_period":       14,    # Wilder ATR period
        "atr_avg_window":   20,    # Rolling window for average ATR
        "atr_multiple":     1.2,   # ATR must exceed 1.2× avg ATR (lowered from 1.5)
        "rsi_period":       14,    # RSI lookback
        "rsi_ob":           65,    # Overbought threshold (loosened from 70)
        "rsi_os":           35,    # Oversold threshold  (loosened from 30)
        "adx_period":       14,    # ADX smoothing period
        "adx_threshold":    30,    # ADX ≤ 30 = ranging regime (hard gate)
        "season_threshold": 0.02,  # ±2% monthly return → bullish/bearish regime
        "jan_discount":     0.30,  # January Effect: reduce score weight 30%
        "max_positions":    4,     # Maximum simultaneous open positions
        "time_stop_bars":   12,    # Exit flat after 12 bars if no TP/SL hit
        "commission_bps":   1.1,   # Per leg (IB benchmark rate)
        "risk_per_trade":   0.01,  # Risk 1% of portfolio equity per trade
    },
    "seasonal_exhaustion_fade_long": {
        "bb_window":        20,    # Bollinger Band lookback
        "bb_std":           2.0,   # 2.0σ (lowered for signal frequency on large caps)
        "atr_period":       14,    # Wilder ATR period
        "atr_avg_window":   20,    # Rolling window for average ATR
        "atr_multiple":     1.2,   # ATR must exceed 1.2× avg ATR (lowered from 1.5)
        "rsi_period":       14,    # RSI lookback
        "rsi_os":           35,    # Oversold threshold
        "adx_period":       14,    # ADX smoothing period
        "adx_threshold":    30,    # ADX ≤ 30 = ranging regime (hard gate)
        "season_threshold": 0.02,  # ±2% monthly return → bullish/bearish regime
        "jan_discount":     0.30,  # January Effect: reduce score weight 30%
        "neutral_discount": 0.50,  # Scale signal strength in neutral months
        "max_positions":    4,     # Maximum simultaneous open positions
        "time_stop_bars":   12,    # Exit flat after 12 bars if no TP/SL hit
    },
    "sma_brownian": {
        "fast_window":  20,   # fast SMA bars (~1 month)
        "slow_window":  60,   # slow SMA bars (~1 quarter)
        "drift_window": 63,   # GBM drift estimation window (~1 quarter)
        "min_signal":   0.10, # ignore signals weaker than this
    },
    "exhaustion_fade": {
        "bb_window":      20,    # Bollinger Band lookback
        "bb_std":          2.0,  # 2.0σ (tighter than crypto 2.5σ — stocks less volatile)
        "adx_period":     14,    # Wilder ADX smoothing period
        "adx_threshold":  30,    # ADX < 30 = ranging regime (same as crypto)
        "rsi_period":     14,    # RSI lookback (replaces crypto funding rate)
        "rsi_ob":         70,    # Overbought threshold → fade short
        "rsi_os":         30,    # Oversold threshold  → fade long
        "vol_multiple":    1.5,  # Volume must exceed 1.5× rolling avg
        "vol_lookback":   20,    # Rolling avg window for volume climax gate
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
