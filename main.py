"""
main.py
=======
Master entrypoint for the Stock Hybrid Strategy system.

Usage
-----
  # Backtest with default portfolio (PORTFOLIO_TYPE in settings.py)
  python main.py --mode backtest

  # Backtest with a specific portfolio constructor
  python main.py --mode backtest --portfolio min_variance

  # Compare all portfolio types across all strategies
  python main.py --mode backtest --portfolio all

  # Run live trading via Alpaca (paper by default)
  python main.py --mode live

  # Backtest then immediately go live
  python main.py --mode full

  # Print latest live state / performance
  python main.py --mode report

Available portfolio types
-------------------------
  max_sharpe      Maximize Sharpe ratio (SLSQP optimizer)
  equal_weight    Equal weight across positive-signal stocks
  min_variance    Minimize portfolio variance (SLSQP optimizer)
  risk_parity     Equal Risk Contribution (ERC, SLSQP optimizer)
  signal_weighted Weight proportional to signal strength (no optimizer)
  all             Run all of the above and compare

Modes
-----
  PAPER_TRADING = True  in config/settings.py → Alpaca paper account
  PAPER_TRADING = False in config/settings.py → Alpaca live account (real money)
"""

import argparse
import sys
import json
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

import pandas as pd
from loguru import logger

from config.settings import RESULT_DIR, PAPER_TRADING, PORTFOLIO_TYPE, PORTFOLIO_PARAMS
from config.client import check_connectivity
from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
from strategies import get_all_strategies
from backtest.engine import (
    run_all_backtests, run_portfolio_comparison, run_all_backtests_with_splits,
    run_constant_weight_backtests, run_constant_strategy_backtests,
)
from portfolio import get_portfolio, get_all_portfolios
from execution.live_engine import start_scheduler
from utils.logger import setup_logger
from utils.reporting import (
    print_summary_table, save_results,
    print_split_summary, save_split_results,
)


# ── Portfolio comparison printer ───────────────────────────────────────────────

def print_portfolio_comparison(results: dict[str, dict]) -> None:
    """
    Print a cross-tabulation of strategy × portfolio Sharpe ratios,
    then save results to portfolio_comparison_*.* files (separate from
    the single-portfolio results so the two don't overwrite each other).
    """
    # Collect all portfolio names in registration order
    all_ports = []
    for port_dict in results.values():
        for p in port_dict:
            if p not in all_ports:
                all_ports.append(p)

    # ── Sharpe heat-table ──────────────────────────────────────────────────────
    col_w = 16
    header = f"{'Strategy':<28}" + "".join(f"{p:>{col_w}}" for p in all_ports)
    logger.info("")
    logger.info("Portfolio Comparison — Sharpe Ratio")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    best_sharpe = -999.0
    best_label  = ""
    for strat, port_dict in results.items():
        row = f"{strat:<28}"
        for p in all_ports:
            val = port_dict.get(p)
            s   = val.metrics.get("sharpe", float("nan")) if val else float("nan")
            row += f"{s:>{col_w}.3f}"
            if s > best_sharpe:
                best_sharpe = s
                best_label  = f"{strat}/{p}"
        logger.info(row)
    logger.info("=" * len(header))
    logger.info(f"Best combination: {best_label}  (Sharpe={best_sharpe:.3f})")
    logger.info("")

    # ── Flat dict: "strategy/portfolio" → BacktestResult ──────────────────────
    flat = {
        f"{s}/{p}": r
        for s, port_dict in results.items()
        for p, r in port_dict.items()
    }
    metrics_summary = {label: result.metrics for label, result in flat.items()}
    print_summary_table(metrics_summary)

    # Save to dedicated comparison files (does not touch strategy_metrics.json)
    import json
    metrics_out = {label: r.metrics for label, r in flat.items()}
    with open(RESULT_DIR / "portfolio_comparison_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    logger.success("Saved portfolio_comparison_metrics.json")

    returns_df = pd.DataFrame({label: r.portfolio_returns for label, r in flat.items()})
    returns_df.to_csv(RESULT_DIR / "portfolio_comparison_returns.csv")
    logger.success("Saved portfolio_comparison_returns.csv")


# ── Backtest pipeline ──────────────────────────────────────────────────────────

def run_backtest_pipeline(
    use_cache:      bool = True,
    portfolio_name: str  = None,
    run_split:      bool = False,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
    bt_mode:        str  = "walk_forward",
) -> dict:
    """
    Full offline backtest pipeline.

    Parameters
    ----------
    portfolio_name : "all" compares every portfolio type; None uses PORTFOLIO_TYPE.
    run_split      : when True, splits returns into Train/Val/Test and reports
                     each period separately (70 / 15 / 15 by default).
    train_frac     : fraction of bars in the training period.
    val_frac       : fraction of bars in the validation period.
    bt_mode        : "walk_forward"     — standard dynamic rebalancing (default)
                     "constant_weight"  — weights frozen at first rebalance date
                     "constant_strategy"— signal frozen at first rebalance date,
                                         optimizer still adapts to fresh covariance

    Returns artifacts dict for the live engine.
    """
    logger.info("=" * 60)
    logger.info("Stock Hybrid Strategy — Backtest Pipeline")
    mode = "PAPER" if PAPER_TRADING else "LIVE"
    logger.info(f"Mode: {mode}")
    logger.info("=" * 60)

    portfolio_name = portfolio_name or PORTFOLIO_TYPE

    # ── 1. Data ────────────────────────────────────────────────────────────────
    logger.info("Step 1/4: Loading universe data")
    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close         = build_close_matrix(universe_data)
    returns       = build_return_matrix(close)
    high_df       = pd.DataFrame({s: universe_data[s]["high"]   for s in universe_data})
    low_df        = pd.DataFrame({s: universe_data[s]["low"]    for s in universe_data})
    volume_df     = pd.DataFrame({s: universe_data[s]["volume"] for s in universe_data})

    logger.info(f"Universe: {list(universe_data.keys())}")
    logger.info(f"Date range: {close.index[0].date()} → {close.index[-1].date()}")
    logger.info(f"Total bars: {len(close)}")

    # ── 2. Generate all signals ────────────────────────────────────────────────
    logger.info("Step 2/4: Generating strategy signals")
    strategies   = get_all_strategies()
    signals_dict = {}

    for strategy in strategies:
        logger.info(f"  → {strategy.name}")
        signals_dict[strategy.name] = strategy.run(
            close   = close,
            returns = returns,
            high    = high_df,
            low     = low_df,
            volume  = volume_df,
        )

    # ── 3. Backtest ────────────────────────────────────────────────────────────
    logger.info("Step 3/4: Running walk-forward backtests")

    if portfolio_name == "all":
        logger.info("Portfolio mode: comparing all portfolio types")
        portfolios       = get_all_portfolios(params_override=PORTFOLIO_PARAMS)
        comparison       = run_portfolio_comparison(
            strategies   = strategies,
            signals_dict = signals_dict,
            close        = close,
            returns      = returns,
            portfolios   = portfolios,
        )
        logger.info("Step 4/4: Saving results")
        print_portfolio_comparison(comparison)

        # Flatten for live engine: pick max_sharpe results as the "main" results
        backtest_results = {
            name: port_dict.get("max_sharpe") or next(iter(port_dict.values()))
            for name, port_dict in comparison.items()
        }
    else:
        logger.info(f"Portfolio mode: {portfolio_name} | Backtest mode: {bt_mode}")
        portfolio = get_portfolio(portfolio_name, params=PORTFOLIO_PARAMS.get(portfolio_name))
        logger.info("Step 4/4: Saving results")

        if run_split:
            logger.info(
                f"Split mode: Train {train_frac*100:.0f}% / "
                f"Val {val_frac*100:.0f}% / "
                f"Test {(1-train_frac-val_frac)*100:.0f}%"
            )
            split_results = run_all_backtests_with_splits(
                strategies   = strategies,
                signals_dict = signals_dict,
                close        = close,
                returns      = returns,
                optimizer    = portfolio,
                train_frac   = train_frac,
                val_frac     = val_frac,
            )
            print_split_summary(split_results)
            save_split_results(split_results)
            backtest_results = {
                name: sr.train for name, sr in split_results.items()
            }

        elif bt_mode == "constant_weight":
            backtest_results = run_constant_weight_backtests(
                strategies   = strategies,
                signals_dict = signals_dict,
                close        = close,
                returns      = returns,
                optimizer    = portfolio,
            )
            metrics_summary = {name: r.metrics for name, r in backtest_results.items()}
            print_summary_table(metrics_summary)
            save_results(backtest_results, prefix="constant_weight")

        elif bt_mode == "constant_strategy":
            backtest_results = run_constant_strategy_backtests(
                strategies   = strategies,
                signals_dict = signals_dict,
                close        = close,
                returns      = returns,
                optimizer    = portfolio,
            )
            metrics_summary = {name: r.metrics for name, r in backtest_results.items()}
            print_summary_table(metrics_summary)
            save_results(backtest_results, prefix="constant_strategy")

        else:  # walk_forward (default)
            backtest_results = run_all_backtests(
                strategies   = strategies,
                signals_dict = signals_dict,
                close        = close,
                returns      = returns,
                optimizer    = portfolio,
            )
            metrics_summary = {name: r.metrics for name, r in backtest_results.items()}
            print_summary_table(metrics_summary)
            save_results(backtest_results)

    return {
        "universe_data":    universe_data,
        "close":            close,
        "returns":          returns,
        "strategies":       strategies,
        "signals_dict":     signals_dict,
        "backtest_results": backtest_results,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    setup_logger()

    all_portfolio_names = ["max_sharpe", "equal_weight", "min_variance", "risk_parity", "signal_weighted", "all"]

    parser = argparse.ArgumentParser(description="Stock Hybrid Strategy Trader")
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "full", "report"],
        default="backtest",
    )
    parser.add_argument(
        "--portfolio",
        choices=all_portfolio_names,
        default=None,
        help=(
            "Portfolio construction method. "
            f"Options: {', '.join(all_portfolio_names)}. "
            f"Default: {PORTFOLIO_TYPE} (from config/settings.py)"
        ),
    )
    parser.add_argument("--no-cache",  action="store_true", help="Force re-download of data")
    parser.add_argument("--run-now",   action="store_true", help="Fire one rebalance immediately on startup")
    parser.add_argument("--split",     action="store_true", help="Split returns into Train/Val/Test (70/15/15) and report each period")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Training period fraction (default 0.70)")
    parser.add_argument("--val-frac",   type=float, default=0.15, help="Validation period fraction (default 0.15)")
    parser.add_argument(
        "--bt-mode",
        choices=["walk_forward", "constant_weight", "constant_strategy"],
        default="walk_forward",
        help=(
            "Backtesting mode. "
            "walk_forward: dynamic rebalancing (default). "
            "constant_weight: weights frozen at first rebalance, no further rebalancing. "
            "constant_strategy: signal frozen at first rebalance, optimizer still adapts."
        ),
    )
    args = parser.parse_args()

    mode = "PAPER" if PAPER_TRADING else "LIVE ⚠️  (real money)"
    logger.info(f"Trading mode: {mode}")
    logger.info("Checking Alpaca connectivity...")
    check_connectivity()

    try:
        if args.mode == "backtest":
            run_backtest_pipeline(
                use_cache      = not args.no_cache,
                portfolio_name = args.portfolio,
                run_split      = args.split,
                train_frac     = args.train_frac,
                val_frac       = args.val_frac,
                bt_mode        = args.bt_mode,
            )

        elif args.mode == "live":
            artifacts = run_backtest_pipeline(
                use_cache      = True,
                portfolio_name = args.portfolio,
            )
            start_scheduler(
                strategies   = artifacts["strategies"],
                signals_dict = artifacts["signals_dict"],
                run_now      = args.run_now,
            )

        elif args.mode == "full":
            artifacts = run_backtest_pipeline(
                use_cache      = not args.no_cache,
                portfolio_name = args.portfolio,
            )
            logger.info("Backtest complete. Launching live engine...")
            start_scheduler(
                strategies   = artifacts["strategies"],
                signals_dict = artifacts["signals_dict"],
                run_now      = args.run_now,
            )

        elif args.mode == "report":
            state_file = RESULT_DIR / "live_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                logger.info(f"Active strategy : {state.get('active_strategy')}")
                logger.info(f"Cash            : ${state['cash_usd']:,.2f}")
                open_pos = {s: q for s, q in state["positions"].items() if q != 0}
                logger.info(f"Open positions  : {open_pos}")
                nav_hist = state.get("nav_history", [])
                if nav_hist:
                    start_nav = nav_hist[0]["nav"]
                    latest    = nav_hist[-1]["nav"]
                    logger.info(
                        f"NAV             : ${latest:,.2f} "
                        f"({(latest-start_nav)/start_nav*100:+.2f}% vs start)"
                    )
            else:
                logger.warning("No live_state.json found. Run --mode live first.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Session complete.")


if __name__ == "__main__":
    main()
