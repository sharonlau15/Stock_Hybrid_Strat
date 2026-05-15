# Stock Hybrid Strategy System

A modular, multi-strategy algorithmic trading system for US equities, connected to Alpaca Markets. Supports full historical backtesting, walk-forward portfolio optimization, and live or paper trading via the same codebase.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Configuration](#2-configuration)
3. [Broker Connection — Alpaca](#3-broker-connection--alpaca)
4. [Data Pipeline](#4-data-pipeline)
5. [Strategy System](#5-strategy-system)
6. [Backtesting Engine](#6-backtesting-engine)
7. [Portfolio Optimizer](#7-portfolio-optimizer)
8. [Risk Management](#8-risk-management)
9. [Live Execution Engine](#9-live-execution-engine)
10. [Operating Modes](#10-operating-modes)
11. [Outputs & Persistence](#11-outputs--persistence)
12. [Adding a New Strategy](#12-adding-a-new-strategy)
13. [Key Numbers at a Glance](#13-key-numbers-at-a-glance)

---

## 1. Project Structure

```
Stock_Hybrid_Strat/
│
├── main.py                      Entry point — all four operating modes
├── Alpaca.env                   API keys (gitignored)
├── requirements.txt
│
├── config/
│   ├── settings.py              All tuneable parameters — single source of truth
│   └── client.py                Alpaca client factory (paper / live toggle)
│
├── data/
│   ├── ingestion.py             Fetch OHLCV from Alpaca, cache to Parquet
│   └── cache/                   Auto-created — one .parquet file per symbol
│
├── strategies/
│   ├── __init__.py              REGISTRY — the only file to edit when adding a strategy
│   ├── base.py                  BaseStrategy abstract class
│   ├── momentum.py              Strategy 1
│   ├── mean_reversion.py        Strategy 2
│   ├── risk_parity.py           Strategy 3
│   ├── cross_sectional.py       Strategy 4
│   ├── vol_breakout.py          Strategy 5
│   └── ml_signal.py             Strategy 6
│
├── backtest/
│   └── engine.py                Walk-forward backtester + performance metrics
│
├── portfolio/
│   └── optimizer.py             Max-Sharpe optimizer (scipy SLSQP)
│
├── execution/
│   └── live_engine.py           Two-loop live scheduler (APScheduler)
│
├── utils/
│   ├── logger.py                Loguru setup
│   └── reporting.py             Print tables + save CSVs
│
├── logs/                        Auto-created — daily rotating log files
└── results/                     Auto-created — CSVs, JSONs, trade log
```

---

## 2. Configuration

**File:** `config/settings.py`

Everything tuneable lives here. You never hardcode values in strategy or engine files.

### Trading Universe

12 liquid large-cap US stocks and ETFs, selected for sector diversity and liquidity:

| Ticker | Name            | Sector              |
|--------|-----------------|---------------------|
| AAPL   | Apple           | Technology          |
| MSFT   | Microsoft       | Technology          |
| GOOGL  | Alphabet        | Technology          |
| AMZN   | Amazon          | Consumer / Cloud    |
| NVDA   | NVIDIA          | Semiconductors      |
| META   | Meta            | Social Media        |
| JPM    | JPMorgan Chase  | Financials          |
| JNJ    | Johnson & Johnson | Healthcare        |
| XOM    | ExxonMobil      | Energy              |
| UNH    | UnitedHealth    | Healthcare          |
| SPY    | S&P 500 ETF     | Broad market        |
| QQQ    | Nasdaq-100 ETF  | Tech benchmark      |

Add stocks under config/settings.py

### Paper vs Live Toggle

```python
PAPER_TRADING = True   # → Alpaca paper account (fake money, safe)
PAPER_TRADING = False  # → Alpaca live account  (real money — flip only when ready)
```

This single flag controls which API keys are loaded and which Alpaca environment receives orders. The rest of the code is identical. Change under config/settings.py -> mode

### Risk Parameters

| Parameter              | Value   | Meaning                                        |
|------------------------|---------|------------------------------------------------|
| `STOP_LOSS_PCT`        | 5%      | Hard stop: close position if down 5% from entry |
| `TAKE_PROFIT_PCT`      | 10%     | Take profit: close position if up 10%          |
| `TRAILING_STOP_PCT`    | 7%      | Trailing stop: close if 7% below peak price    |
| `USE_TRAILING_STOP`    | True    | Trailing stop is active                        |
| `MAX_POSITION_SIZE`    | 20%     | No single stock can exceed 20% of portfolio    |
| `MAX_LIVE_POSITIONS`   | 8       | At most 8 simultaneous open long positions     |
| `REBALANCE_THRESHOLD`  | 3%      | Only rebalance if total weight change > 3%     |
| `PORTFOLIO_USD`        | $10,000 | Starting capital                               |
| `MIN_ORDER_USD`        | $1      | Minimum notional per order (fractional shares) |

### Portfolio Construction Parameters

| Parameter              | Value      | Meaning                                       |
|------------------------|------------|-----------------------------------------------|
| `TRADING_DAYS`         | 252        | US market days per year (not 365 like crypto) |
| `RISK_LOOKBACK_DAYS`   | 126        | ~6 months of daily returns for covariance     |
| `LONG_SHORT`           | False      | Long-only (no margin/shorting in paper mode)  |
| `MAX_WEIGHT_SUM`       | 1.00       | Gross exposure capped at 100%                 |
| `MIN_ANNUALIZED_VOL`   | 3%         | Portfolio vol floor (optimizer constraint)    |
| `TRANSACTION_COST_BP`  | 0 bps      | Alpaca charges zero commission                |
| `SLIPPAGE_BP`          | 3 bps      | Assumed slippage modelled in backtest         |

---

## 3. Broker Connection — Alpaca

**File:** `config/client.py`

### Two clients, same API shape

Alpaca uses the same REST API structure for paper and live trading. The only difference is the base URL, which `alpaca-py` handles internally via the `paper=True/False` flag.

```
Paper trading  → https://paper-api.alpaca.markets
Live trading   → https://api.alpaca.markets
Data API       → https://data.alpaca.markets  (same for both)
```

### Client factory functions

```python
get_trading_client()   # TradingClient — submits orders, queries account/positions
get_data_client()      # StockHistoricalDataClient — fetches OHLCV bars
```

Both read from `Alpaca.env`:

```
ALPACA_PAPER_API_KEY=...
ALPACA_PAPER_API_SECRET=...
ALPACA_LIVE_API_KEY=...
ALPACA_LIVE_API_SECRET=...
```

### Connectivity check

On startup, `check_connectivity()` calls `client.get_account()` and logs the account's equity, buying power, and status. If the keys are wrong or Alpaca is unreachable, the system raises immediately before touching any strategy or data code.

### Order mechanics

- **Order type:** Market orders (`MarketOrderRequest`)
- **Time in force:** `DAY` — unfilled orders expire at market close
- **Fractional shares:** Supported — minimum order is $1
- **Lot size rounding:** Quantities rounded to 6 decimal places
- **Sell before buy:** Within each rebalance, sell orders are placed before buy orders to free cash first

---

## 4. Data Pipeline

**File:** `data/ingestion.py`

### What data is fetched

| Data            | Source              | Endpoint                              | Frequency  |
|-----------------|---------------------|---------------------------------------|------------|
| OHLCV bars      | Alpaca Data API     | `StockBarsRequest` (1Day timeframe)   | Daily bars |
| Latest prices   | Alpaca Data API     | `StockLatestTradeRequest`             | Real-time  |

No external data sources beyond Alpaca (no sentiment, no funding rates — those are crypto-specific).

### OHLCV fetch

- **Backtest start date:** 2022-01-01 (configurable in settings)
- **Adjustment:** `adjustment="all"` — prices are split-adjusted and dividend-adjusted automatically by Alpaca. This is critical for backtesting accuracy.
- **Columns returned:** `open`, `high`, `low`, `close`, `volume`
- **Index:** UTC-normalized DatetimeIndex (daily)

### Caching

Every symbol's OHLCV is cached to `data/cache/{SYMBOL}_1Day.parquet` after the first fetch. On subsequent runs:

- If the cache file is **younger than 6 hours** (`CACHE_EXPIRY_HOURS`): loaded from disk, no API call
- If the cache is **older than 6 hours** or missing: re-fetched from Alpaca, cache overwritten

The live engine always bypasses cache (`use_cache=False`) to guarantee it has the most recent bar.

### Matrix construction

After fetching all symbols, two matrices are built:

```
close    — shape (n_days × 12 symbols), values = adjusted close prices
returns  — shape (n_days × 12 symbols), values = log(close_t / close_t-1)
```

Both are forward-filled for any missing dates (e.g. halted stocks), then rows with all-NaN are dropped.

`high_df` and `low_df` matrices are built the same way and passed to strategies that need intraday range (the ATR breakout strategy).

---

## 5. Strategy System

**Files:** `strategies/`

### Architecture

All strategies share a common interface through `BaseStrategy` (`strategies/base.py`):

```
generate_signals(close, returns, **kwargs) → signal DataFrame
```

The signal DataFrame has:
- **Same shape as close:** `n_days × n_symbols`
- **Values in `[-1, +1]`:** positive = long conviction, negative = short conviction, 0 = flat
- **No look-ahead:** `BaseStrategy.run()` automatically applies `signal.shift(1)` before returning — signals computed on day T are only usable from day T+1

### Registry

`strategies/__init__.py` contains the `REGISTRY` list. Adding a new strategy = create a file + add two lines there. Nothing else in the codebase changes.

---

### Strategy 1 — Momentum (`strategies/momentum.py`)

**What it does:** Ranks all stocks by their 12-month cumulative return, skipping the most recent month (the skip-month filter avoids the short-term reversal effect). The top 4 ranked stocks get a long signal of +1.

**Signal computation:**
```
score = (close / close[252 days ago]) - (close / close[21 days ago])
```
Rank the score each day across all 12 stocks. Top 4 → signal = +1. All others → 0.

**Key parameters:**
- `lookback_long = 252` (12-month window)
- `lookback_short = 21` (1-month skip)
- `top_n = 4`
- `bottom_n = 0` (long-only — no shorts)

**Why it works:** Persistent winner/loser effect — stocks that have outperformed over the past year tend to continue outperforming over the next 1-3 months. The 1-month skip avoids the short-term reversal contaminating the signal.

---

### Strategy 2 — Mean Reversion (`strategies/mean_reversion.py`)

**What it does:** Computes a rolling 20-day z-score of price vs its own moving average. When a stock is statistically cheap relative to its recent history (low z-score), it generates a long signal. When expensive (high z-score), it generates a short signal.

**Signal computation:**
```
z = (close - rolling_mean_20d) / rolling_std_20d
signal = -z / (entry_z × 3),  clipped to [-1, +1]
```
Negative z → positive signal (buy the dip). The signal is continuous — a larger deviation from the mean produces a stronger signal.

**Key parameters:**
- `zscore_window = 20` (trading days)
- `entry_z = 2.0` (±2σ = max signal strength)

**Why it works:** Short-term crypto overreaction to news creates exploitable reversion windows. In stocks, earnings surprises and macro news often cause temporary dislocations that revert over 3–20 days.

---

### Strategy 3 — Risk Parity (`strategies/risk_parity.py`)

**What it does:** Weights each stock inversely proportional to its recent volatility, so every position contributes the same amount of risk to the portfolio. Long-only, always fully invested.

**Signal computation:**
```
weight_i = (1 / rolling_vol_63d_i) / sum(1 / rolling_vol_63d_j for all j)
```
The signal is the weight itself — a positive number for every stock, summing to 1.

**Key parameters:**
- `lookback = 63` (3-month volatility window)

**Why it works:** Diversifies risk rather than capital. In environments where return dispersion is high, risk-parity outperforms equal-weight because it avoids concentrating capital in the most volatile names at the wrong time.

---

### Strategy 4 — Cross-Sectional Momentum (`strategies/cross_sectional.py`)

**What it does:** Every day, ranks all 12 stocks by their 20-day cumulative return. Converts the rank to a continuous signal in `[-1, +1]` — top-ranked gets +1, bottom-ranked gets -1, middle stocks get intermediate values.

**Signal computation:**
```
cum_ret  = close / close[20 days ago] - 1
rank     = cum_ret.rank(axis=1)             # cross-sectional rank each day
signal   = (rank - 1) / (n_stocks - 1) × 2 - 1
```

**Key parameters:**
- `lookback = 20`
- `rank_method = "min"` (ties get the lowest rank)

**Why it works:** Relative momentum (who is winning vs losing within the same asset class) removes the systematic market beta that time-series momentum carries. It's a market-neutral signal — in a down market, it still picks the least-bad performers.

---

### Strategy 5 — Volatility Breakout (`strategies/vol_breakout.py`)

**What it does:** Uses the 14-period Average True Range (ATR) to define a volatility band around each stock. A large daily move relative to the ATR produces a directional signal — a big up-move generates a long signal, a big down-move generates a short signal.

**Signal computation:**
```
ATR = rolling_14d_mean(TrueRange)
TrueRange = max(high-low, |high-prev_close|, |low-prev_close|)

half_band = atr_multiplier × ATR   (multiplier = 2.0)
signal    = (close - prev_close) / half_band,  clipped to [-1, +1]
```

**Key parameters:**
- `atr_period = 14`
- `atr_multiplier = 2.0`

**Why it works:** Volatility expansion after a period of compression signals the start of a new directional move. A move that exceeds 2× ATR is statistically unusual and tends to have follow-through (trend continuation).

---

### Strategy 6 — ML Signal (`strategies/ml_signal.py`)

**What it does:** Trains a LightGBM gradient-boosted classifier per symbol on lagged return features. The target is the sign of next-day return (1 = up, 0 = down). The output probability is converted to a continuous signal.

**Features used per symbol:**
| Feature          | Description                              |
|------------------|------------------------------------------|
| `ret_1d`         | Yesterday's return (lag 1)               |
| `ret_3d`         | Return 3 days ago                        |
| `ret_5d`         | Return 5 days ago (weekly)               |
| `ret_10d`        | Return 10 days ago (2-week)              |
| `ret_21d`        | Return 21 days ago (monthly)             |
| `vol_10d`        | 10-day rolling standard deviation        |
| `vol_21d`        | 21-day rolling standard deviation        |
| `skew_21d`       | 21-day rolling skewness                  |
| `vol_ratio`      | `vol_10d / vol_21d` (vol regime signal)  |

**Signal computation:**
```
signal = P(next_day_return > 0) × 2 - 1
```
A probability of 0.8 → signal of +0.6 (moderate long). A probability of 0.3 → signal of -0.4 (moderate short).

**No look-ahead guarantee:** The model is trained on `[t - train_window, t-1]` and predicts `t`. The `BaseStrategy.run()` shift then delays application to `t+1`.

**Key parameters:**
- `feature_lookbacks = [1, 3, 5, 10, 21]`
- `train_window = 126` (6 months of trading days)
- `n_estimators = 25` (reduced for speed: `n_estimators // 4`)
- `max_depth = 4`
- `learning_rate = 0.05`

---

## 6. Backtesting Engine

**File:** `backtest/engine.py`

### Execution timing (T+1 rule)

```
Day T close  →  signal computed
Day T+1      →  portfolio rebalanced at this price
Day T+1→T+2  →  return attributed to the new weights
```

The `shift(1)` in `BaseStrategy.run()` implements this. No look-ahead bias is possible.

### Walk-forward process

For each trading day from `RISK_LOOKBACK_DAYS` (day 126) onward:

1. **Covariance estimation:** Take the 126-day rolling window of daily returns ending at day T. Multiply by 252 to annualize.
2. **Signal:** Read the pre-computed (and already shifted) signal row for day T.
3. **Optimize:** Call the Max-Sharpe optimizer with the signal and covariance → new weights.
4. **Record weights:** Store the target weight vector for day T.
5. **Forward-fill:** Between rebalance dates, weights carry forward unchanged.

At the end:
```
portfolio_return_t = sum(weight_i_t × return_i_{t+1})
```
After computing the full return series, transaction costs are deducted.

### Transaction cost model

```
daily_cost = sum(|Δweight_i|) × (cost_bps + slippage_bps) / 10,000
```
Alpaca is zero-commission, so `cost_bps = 0` and only `slippage_bps = 3` applies.

### Performance metrics computed

| Metric          | Formula                                                  |
|-----------------|----------------------------------------------------------|
| Sharpe          | `(mean_return × 252) / (std × √252)`                    |
| Sortino         | `(mean_return × 252) / (downside_std × √252)`           |
| Calmar          | `CAGR / |max_drawdown|`                                  |
| CAGR            | `(1 + total_return)^(1/n_years) - 1`                    |
| Max Drawdown    | `min((cum_return - rolling_max) / rolling_max)`          |
| Win Rate        | `fraction of days with positive return`                  |
| Profit Factor   | `avg_win / |avg_loss|`                                   |
| Final Capital   | `$10,000 × (1 + total_return)`                           |

All annualized using **252 trading days** (not 365 — this is a stock system).

---

## 7. Portfolio Optimizer

**File:** `portfolio/optimizer.py`

### Method

**Maximum Sharpe Ratio** via `scipy.optimize.minimize` with the SLSQP solver.

The strategy signal is used as a proxy for expected returns:
```
μ_i = signal_i
```
This means the optimizer does not require a separate return forecast — the signal's direction and magnitude directly inform the expected return input.

### Objective

```
maximize:   Sharpe = μᵀw / √(wᵀΣw)
```
Implemented as minimizing negative Sharpe.

### Constraints

| Constraint              | Value  | Purpose                                          |
|-------------------------|--------|--------------------------------------------------|
| `Σ|w_i| ≤ 1.0`         | 100%   | Total gross exposure cap                         |
| `√(wᵀΣw) ≥ 0.03`       | 3%     | Annualized vol floor (prevents degenerate flat)  |
| `-0.2 ≤ w_i ≤ 0.2`     | ±20%   | Per-stock position cap                           |
| `w_i ≥ 0` (long-only)  | —      | When `LONG_SHORT = False`                        |

### Fallback

If the optimizer fails to converge (e.g. singular covariance, no valid signal), it falls back to **equal-weight** across all stocks with a positive signal. The portfolio is never left flat due to an optimizer error.

### Regularization

A small identity matrix `1e-6 × I` is added to the covariance to prevent singularity when some stocks have identical returns over the lookback window.

---

## 8. Risk Management

**File:** `portfolio/risk_manager.py` (dataclass), `execution/live_engine.py` (live checks)

Risk is enforced at two levels:

### Position-level stops (live trading only)

Each time a new BUY is executed, three prices are recorded:
- `entry_price` — the execution price
- `stop_loss_price` = `entry_price × (1 - 0.05)`
- `take_profit_price` = `entry_price × (1 + 0.10)`
- `peak_price` — updated every 60 seconds to the current price if it's a new high

Every 60 seconds the price monitor checks all open positions:

| Trigger          | Condition                                    | Action            |
|------------------|----------------------------------------------|-------------------|
| Hard stop loss   | `current_price ≤ entry × 0.95`              | Sell immediately  |
| Take profit      | `current_price ≥ entry × 1.10`              | Sell immediately  |
| Trailing stop    | `(peak - current) / peak ≥ 0.07`            | Sell immediately  |

Priority order: Take Profit > Trailing Stop > Hard Stop.

The trailing stop only activates once the position is in profit (i.e., `current > entry`). Before that, only the hard stop applies.

### Portfolio-level gate (rebalance threshold)

The live engine computes the sum of absolute weight changes before placing any order:
```
Σ|new_weight_i - current_weight_i| < 0.03  →  do nothing
```
This prevents micro-rebalances from noise in the signal and keeps turnover low.

---

## 9. Live Execution Engine

**File:** `execution/live_engine.py`

### Two concurrent loops (APScheduler `BackgroundScheduler`)

Both loops run in background threads. They share state via `live_state.json` and do not block each other.

---

#### Loop 1 — Price Monitor (every 60 seconds)

```
load state
↓
fetch current prices for all open positions
↓
check each position against stop loss / take profit / trailing stop
↓
if any breach → execute SELL immediately (market order)
              → update cash, remove position entry, log trade
              → save state
```

This loop runs **all day** regardless of whether the market is in its normal session. It will also fire on the open/close rebalance times and catch any intraday stops that were triggered between signal recomputes.

---

#### Loop 2 — Signal Recompute + Conditional Rebalance (cron, twice daily)

**Schedule (Mon–Fri, US Eastern, DST-aware):**
- `09:35 ET` — 5 minutes after market open (prices settled from opening auction)
- `15:50 ET` — 10 minutes before market close (final shape of the day's price action)

```
fetch fresh OHLCV from Alpaca (cache bypassed)
↓
recompute signals for all 6 strategies
↓
select best strategy by recent 20-day Sharpe
↓
convert signal to target weights (long-only, max 8 positions, 20% cap each)
↓
fetch current prices
↓
compute actual current weights from live position values / NAV
↓
Σ|new_weight - current_weight| < 3%?  →  HOLD (no orders placed)
                                  ≥ 3%?  →  REBALANCE
                                             • market_is_open() check
                                             • sell orders first
                                             • buy orders second
                                             • update cash / positions / entries
↓
save state
```

#### Market hours guard

Before placing any order, `market_is_open()` calls Alpaca's `/v2/clock` endpoint. If the market is closed (pre-market, after-hours, weekend, holiday), orders are skipped and a warning is logged. The position monitor still runs — it uses Alpaca's existing fills, not new orders.

---

### State file — `results/live_state.json`

The state file is the engine's memory across restarts. It stores:

| Key                  | Type          | Contents                                              |
|----------------------|---------------|-------------------------------------------------------|
| `positions`          | dict          | `{symbol: quantity}` for all 12 symbols               |
| `cash_usd`           | float         | Remaining uninvested cash                             |
| `current_weights`    | dict          | Last-known target weights                             |
| `position_entries`   | dict          | Per-symbol: `entry_price`, `entry_date`, `peak_price` |
| `trade_log`          | list          | Every order placed: time, symbol, side, qty, price    |
| `nav_history`        | list          | `{date, nav, event}` entries over time                |
| `active_strategy`    | str           | Name of the currently selected strategy               |
| `last_run`           | str           | ISO timestamp of last signal recompute                |

Writes are **atomic**: data is written to `live_state.tmp` then renamed to `live_state.json`. A crash mid-write never corrupts the state file.

---

### Signal → Weight conversion (live)

```python
signal_to_weights(signal_series):
    take top MAX_LIVE_POSITIONS (8) stocks with positive signal
    raw_weight = signal / sum(signals)          # proportional to signal strength
    iteratively cap any weight > MAX_POSITION_SIZE (20%)
    redistribute excess to uncapped positions
```

This always produces long-only weights summing to ≤ 100%, with no single position exceeding 20%.

---

## 10. Operating Modes

**File:** `main.py`

Run from the project root:

```bash
python main.py --mode backtest         # Offline backtest only, saves results
python main.py --mode live             # Backtest first, then start live engine
python main.py --mode full             # Same as live but forces fresh data
python main.py --mode report           # Print current live state and NAV
python main.py --mode live --run-now   # Fire one immediate rebalance on startup
python main.py --mode backtest --no-cache  # Force re-download of all data
```

### Backtest pipeline steps

```
1. Load OHLCV for all 12 symbols (cache or Alpaca)
2. Build close matrix and log-return matrix
3. Run all 6 strategies → 6 signal DataFrames
4. Walk-forward backtest each strategy → 6 BacktestResult objects
5. Print summary table (Sharpe, CAGR, drawdown, final capital)
6. Save to results/: strategy_metrics.json, portfolio_returns.csv, pnl_summary.csv
```

### Live pipeline steps

```
1. Run full backtest pipeline (to establish signals and rankings)
2. Start APScheduler with two jobs (price monitor + signal recompute)
3. Block on sleep loop — Ctrl+C triggers graceful shutdown
4. On shutdown: save trade_log.csv, print session return
```

### Report mode

Reads `results/live_state.json` and prints:
- Active strategy name
- Cash remaining
- Open positions and quantities
- NAV history (start vs current, % return)

---

## 11. Outputs & Persistence

### `results/` directory

| File                    | Written by      | Contents                                        |
|-------------------------|-----------------|-------------------------------------------------|
| `strategy_metrics.json` | reporting.py    | Sharpe, CAGR, drawdown etc. for all strategies  |
| `portfolio_returns.csv` | reporting.py    | Daily return series for each strategy           |
| `pnl_summary.csv`       | reporting.py    | P&L summary: start/end capital, total P&L       |
| `trade_log.csv`         | live_engine.py  | Every live order: time, symbol, side, qty, P&L  |
| `live_state.json`       | live_engine.py  | Full live portfolio state (see above)           |

### `logs/` directory

Daily rotating log files: `logs/algo_YYYY-MM-DD.log`

- Retained for 4 weeks then deleted automatically
- Mirrors everything printed to the console
- Level: INFO (configurable via `LOG_LEVEL` in settings)

---

## 12. Adding a New Strategy

**Two steps only:**

**Step 1** — Create `strategies/your_strategy.py`:

```python
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS

class YourStrategy(BaseStrategy):
    def __init__(self):
        # Add "your_strategy": {...} to STRATEGY_PARAMS in config/settings.py
        super().__init__("your_strategy", STRATEGY_PARAMS["your_strategy"])

    def generate_signals(self, close, returns, **kwargs):
        # close   : pd.DataFrame (n_days × n_symbols), adjusted close prices
        # returns : pd.DataFrame (n_days × n_symbols), daily log returns
        # kwargs  : may contain "high", "low" DataFrames if you need them
        #
        # Must return a pd.DataFrame of same shape as close,
        # values in [-1, +1]. Positive = long, negative = short, 0 = flat.
        signals = ...
        return signals
```

**Step 2** — Register it in `strategies/__init__.py`:

```python
from strategies.your_strategy import YourStrategy   # add this line

REGISTRY: list[type] = [
    ...
    YourStrategy,                                   # add this line
]
```

The strategy will automatically appear in backtests, the optimizer, and the live engine — no other files change.

---

## 13. Key Numbers at a Glance

| Topic                    | Value / Setting                                   |
|--------------------------|---------------------------------------------------|
| Broker                   | Alpaca Markets (`alpaca-py`)                      |
| Universe                 | 12 US stocks + ETFs                               |
| Data source              | Alpaca Data API (split/dividend adjusted)         |
| Backtest start           | 2022-01-01                                        |
| Bar size                 | 1 Day                                             |
| Annualization            | 252 trading days                                  |
| Covariance window        | 126 days (6 months)                               |
| Commission               | $0 (Alpaca)                                       |
| Slippage assumption      | 3 bps (backtest only)                             |
| Optimizer                | Max-Sharpe (scipy SLSQP)                          |
| Max position size        | 20%                                               |
| Max live positions       | 8                                                 |
| Stop loss                | 5% below entry                                    |
| Take profit              | 10% above entry                                   |
| Trailing stop            | 7% below peak                                     |
| Price monitor cadence    | Every 60 seconds                                  |
| Signal recompute times   | 09:35 ET + 15:50 ET, Mon–Fri                      |
| Rebalance threshold      | 3% total weight change                            |
| Starting capital         | $10,000                                           |
| Paper/live toggle        | `PAPER_TRADING` in `config/settings.py`           |
| Number of strategies     | 6 (easily extensible via registry)                |
| State persistence        | `results/live_state.json` (atomic writes)         |
