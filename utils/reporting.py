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


def save_results(backtest_results: dict, output_dir=RESULT_DIR, prefix: str = ""):
    """Save backtest results to CSV and JSON.

    prefix — when non-empty, files are named  {prefix}_metrics.json /
             {prefix}_returns.csv instead of the default names.
             Used by constant_weight and constant_strategy modes so they
             don't overwrite the standard walk-forward results.
    """
    output_dir = output_dir or RESULT_DIR
    p = f"{prefix}_" if prefix else ""

    metrics_file = f"{p}strategy_metrics.json" if prefix else "strategy_metrics.json"
    returns_file  = f"{p}portfolio_returns.csv"  if prefix else "portfolio_returns.csv"

    # Strategy metrics JSON
    metrics = {name: r.metrics for name, r in backtest_results.items()}
    with open(output_dir / metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.success(f"Saved {metrics_file}")

    # Portfolio returns CSV
    returns_df = pd.DataFrame({
        name: r.portfolio_returns for name, r in backtest_results.items()
    })
    returns_df.to_csv(output_dir / returns_file)
    logger.success(f"Saved {returns_file}")

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
    pnl_file = f"{p}pnl_summary.csv" if prefix else "pnl_summary.csv"
    pd.DataFrame(pnl_rows).to_csv(output_dir / pnl_file, index=False)
    logger.success(f"Saved {pnl_file}")

    # Also persist strategy metrics to PostgreSQL
    try:
        from db.connection import get_conn, put_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            for name, r in backtest_results.items():
                m = r.metrics
                cur.execute("""
                    INSERT INTO strategy_metrics
                        (strategy, sharpe, sortino, calmar, cagr, max_drawdown,
                         total_return, win_rate, profit_factor, recorded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (strategy) DO UPDATE SET
                        sharpe        = EXCLUDED.sharpe,
                        sortino       = EXCLUDED.sortino,
                        calmar        = EXCLUDED.calmar,
                        cagr          = EXCLUDED.cagr,
                        max_drawdown  = EXCLUDED.max_drawdown,
                        total_return  = EXCLUDED.total_return,
                        win_rate      = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        recorded_at   = EXCLUDED.recorded_at
                """, (
                    name,
                    m.get("sharpe"),
                    m.get("sortino"),
                    m.get("calmar"),
                    m.get("cagr"),
                    m.get("max_drawdown"),
                    m.get("total_return"),
                    m.get("win_rate"),
                    m.get("profit_factor"),
                ))
            conn.commit()
            logger.success("Strategy metrics written to PostgreSQL")
        finally:
            put_conn(conn)
    except Exception as e:
        logger.warning(f"DB write for strategy_metrics skipped: {e}")


def print_split_summary(split_results: dict) -> None:
    """
    Print a strategy × split-period Sharpe/CAGR table.

    Columns: Strategy | Train Sharpe | Val Sharpe | Test Sharpe
                      | Train CAGR  | Val CAGR   | Test CAGR
                      | Consistent?

    'Consistent' = val AND test Sharpe are both positive AND within 0.5 of
    the training Sharpe — a simple overfitting check.
    """
    rows = []
    for name, sr in split_results.items():
        def _s(result, key):
            v = result.metrics.get(key)
            return v if isinstance(v, float) else float("nan")

        tr_sharpe = _s(sr.train, "sharpe")
        va_sharpe = _s(sr.val,   "sharpe")
        te_sharpe = _s(sr.test,  "sharpe")
        consistent = (
            va_sharpe > 0 and te_sharpe > 0 and
            abs(tr_sharpe - va_sharpe) < 0.5 and
            abs(tr_sharpe - te_sharpe) < 0.5
        )
        rows.append({
            "Strategy":     name,
            "Train Sharpe": f"{tr_sharpe:.3f}",
            "Val Sharpe":   f"{va_sharpe:.3f}",
            "Test Sharpe":  f"{te_sharpe:.3f}",
            "Train CAGR":   f"{_s(sr.train,'cagr')*100:.1f}%",
            "Val CAGR":     f"{_s(sr.val,  'cagr')*100:.1f}%",
            "Test CAGR":    f"{_s(sr.test, 'cagr')*100:.1f}%",
            "Consistent":   "✓" if consistent else "✗",
        })

    if not rows:
        logger.warning("No split results to display.")
        return

    first = list(split_results.values())[0]
    logger.info("")
    logger.info("=" * 80)
    logger.info("Train / Validation / Test Split Results")
    logger.info(
        f"  Train   : start → {first.train_end.date()}"
        f"  ({len(first.train.portfolio_returns)} days)"
    )
    logger.info(
        f"  Val     : {first.train_end.date()} → {first.val_end.date()}"
        f"  ({len(first.val.portfolio_returns)} days)"
    )
    logger.info(
        f"  Test    : {first.val_end.date()} → end"
        f"  ({len(first.test.portfolio_returns)} days)"
    )
    logger.info("=" * 80)
    df = pd.DataFrame(rows).set_index("Strategy")
    logger.info("\n" + df.to_string())
    logger.info("=" * 80)


def save_split_results(split_results: dict, output_dir=RESULT_DIR) -> None:
    """Save split backtest results to split_metrics.json and split_returns.csv."""
    output_dir = output_dir or RESULT_DIR

    # Flat metrics: {"strategy/period": metrics_dict}
    flat_metrics = {}
    for name, sr in split_results.items():
        for period, result in [("train", sr.train), ("val", sr.val), ("test", sr.test)]:
            flat_metrics[f"{name}/{period}"] = result.metrics

    with open(output_dir / "split_metrics.json", "w") as f:
        json.dump(flat_metrics, f, indent=2, default=str)
    logger.success("Saved split_metrics.json")

    # Returns time series: columns = "strategy/period"
    returns_df = pd.DataFrame({
        f"{name}/{period}": getattr(sr, period).portfolio_returns
        for name, sr in split_results.items()
        for period in ("train", "val", "test")
    })
    returns_df.to_csv(output_dir / "split_returns.csv")
    logger.success("Saved split_returns.csv")

    # Summary table CSV (one row per strategy, columns = periods)
    summary_rows = []
    for name, sr in split_results.items():
        row = {"strategy": name}
        for period in ("train", "val", "test"):
            m = getattr(sr, period).metrics
            for key in ("sharpe", "cagr", "max_drawdown", "annual_vol"):
                row[f"{period}_{key}"] = m.get(key)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False)
    logger.success("Saved split_summary.csv")
