"""utils/reporting.py — Print summary tables and save results to CSV."""

import json
import pandas as pd
from loguru import logger
from config.settings import RESULT_DIR


def print_summary_table(metrics_summary: dict):
    """Print a formatted strategy comparison table."""
    rows = []
    for name, m in metrics_summary.items():
        if "error" in m:
            continue
        rows.append({
            "Strategy":     name,
            "Sharpe":       m.get("sharpe"),
            "CAGR":         f"{m.get('cagr', 0)*100:.1f}%",
            "Max DD":       f"{m.get('max_drawdown', 0)*100:.1f}%",
            "Ann Vol":      f"{m.get('annual_vol', 0)*100:.1f}%",
            "Win Rate":     f"{m.get('win_rate', 0)*100:.1f}%",
            "Final $":      f"${m.get('final_capital', 0):,.0f}",
        })

    if not rows:
        logger.warning("No metrics to display.")
        return

    df = pd.DataFrame(rows).set_index("Strategy")
    logger.info("\n" + df.to_string())


def save_results(backtest_results: dict, output_dir=RESULT_DIR):
    """Save backtest results to CSV and JSON."""
    output_dir = output_dir or RESULT_DIR

    # Strategy metrics JSON
    metrics = {name: r.metrics for name, r in backtest_results.items()}
    with open(output_dir / "strategy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.success("Saved strategy_metrics.json")

    # Portfolio returns CSV
    returns_df = pd.DataFrame({
        name: r.portfolio_returns for name, r in backtest_results.items()
    })
    returns_df.to_csv(output_dir / "portfolio_returns.csv")
    logger.success("Saved portfolio_returns.csv")

    # P&L summary CSV
    pnl_rows = []
    for name, r in backtest_results.items():
        m = r.metrics
        if "error" not in m:
            pnl_rows.append({
                "strategy":      name,
                "initial":       m["initial_capital"],
                "final":         m["final_capital"],
                "total_pnl":     m["total_pnl"],
                "cagr":          m["cagr"],
                "sharpe":        m["sharpe"],
                "max_drawdown":  m["max_drawdown"],
            })
    pd.DataFrame(pnl_rows).to_csv(output_dir / "pnl_summary.csv", index=False)
    logger.success("Saved pnl_summary.csv")
