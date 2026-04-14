"""
Report Builder
───────────────
Orchestrates the full reporting pipeline:
  1. Run backtest (or load cached results)
  2. Compute analytics for each commodity + combined
  3. Generate all charts (PNG files)
  4. Export all data files (JSON, CSV)
  5. Write the composite report.png

Usage:
    from src.reporting.report_builder import build_reports
    build_reports(strategy, live_result, output_dir="output/")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.strategy import LoneStarStrategy, StrategyResult
from src.data.market_data import load_all_data
from src.backtest.engine import run_backtest, BacktestResult
from src.reporting.analytics import compute_all_metrics, PerformanceMetrics
from src.reporting.charts import (
    plot_equity_curve,
    plot_rolling_sharpe,
    plot_monthly_heatmap,
    plot_report,
)
from src.reporting.exporter import (
    ensure_output_dirs,
    export_analytics,
    export_monthly_returns,
    export_positions,
    export_execution_files,
)


def build_reports(
    config_path: str = "config/config.yaml",
    output_dir: str = "output",
    run_live: bool = True,
    verbose: bool = True,
) -> None:
    """
    Full pipeline: backtest → analytics → charts → exports.

    Args:
        config_path : Path to YAML config
        output_dir  : Root output directory
        run_live    : Also run today's live signal scan
        verbose     : Print progress
    """
    base = Path(output_dir)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Load config ────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    strategy = LoneStarStrategy(cfg)
    commodity_cfg = cfg["commodities"]
    data_cfg = cfg["data"]
    lookback = data_cfg.get("lookback_days", 504)

    log("\n" + "═" * 70)
    log("  ★  LONE STAR — Report Builder")
    log("═" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    log("  [1/5] Loading market data...")
    stock_tickers = strategy._get_all_stock_tickers()
    stock_metadata = strategy._get_stock_metadata()

    data = load_all_data(
        stock_tickers=stock_tickers,
        commodity_config=commodity_cfg,
        lookback_days=lookback,
        progress=False,
    )
    stock_returns    = data["stock_returns"]
    stock_prices     = data["stock_prices"]
    stock_volumes    = data.get("stock_volumes")      # may be None if download failed
    commodity_returns = data["commodity_returns"]
    market_returns   = data["market_returns"]

    log(f"     {len(stock_returns.columns)} stocks × {len(commodity_returns.columns)} commodities loaded")

    # ── Run backtest ───────────────────────────────────────────────────────
    log("  [2/5] Running walk-forward backtest...")
    bt_result = run_backtest(
        stock_returns=stock_returns,
        stock_prices=stock_prices,
        commodity_returns=commodity_returns,
        market_returns=market_returns,
        stock_metadata=stock_metadata,
        cfg=cfg,
        step_days=1,
        stock_volumes=stock_volumes,
    )

    commodities = bt_result.commodities
    ensure_output_dirs(base, commodities)

    # ── Compute analytics ──────────────────────────────────────────────────
    log("  [3/5] Computing analytics...")
    metrics = compute_all_metrics(bt_result)

    # ── Generate charts ────────────────────────────────────────────────────
    log("  [4/5] Generating charts...")

    # Root-level combined charts
    plot_equity_curve(
        bt_result.daily_pnl,
        title="Equity Curve — All Commodities + Combined",
        save_path=base / "equity_curve.png",
    )
    log("        equity_curve.png ✓")

    plot_rolling_sharpe(
        bt_result.daily_pnl,
        save_path=base / "rolling_sharpe.png",
    )
    log("        rolling_sharpe.png ✓")

    plot_monthly_heatmap(
        bt_result.monthly_returns,
        column="combined",
        title="Combined Monthly Returns Heatmap (Sharpe-Weighted)",
        save_path=base / "monthly_heatmap.png",
    )
    log("        monthly_heatmap.png ✓")

    # Composite report
    plot_report(
        daily_pnl=bt_result.daily_pnl,
        monthly_returns=bt_result.monthly_returns,
        metrics=metrics,
        nav_weights=bt_result.nav_weights,
        strategy_name="Lone Star Strategy",
        save_path=base / "report.png",
    )
    log("        report.png ✓")

    # Per-commodity charts
    for commodity in commodities:
        c_dir = base / commodity
        pnl_col = bt_result.daily_pnl[[commodity]].rename(columns={commodity: commodity})

        plot_equity_curve(
            pnl_col,
            title=f"Equity Curve — {commodity.replace('_', ' ').title()}",
            save_path=c_dir / "equity_curve.png",
        )
        plot_rolling_sharpe(
            pnl_col,
            title=f"Rolling Sharpe — {commodity.replace('_', ' ').title()}",
            save_path=c_dir / "rolling_sharpe.png",
        )
        plot_monthly_heatmap(
            bt_result.monthly_returns[[commodity]].rename(columns={commodity: commodity}),
            column=commodity,
            title=f"Monthly Returns — {commodity.replace('_', ' ').title()}",
            save_path=c_dir / "monthly_heatmap.png",
        )

    log(f"        Per-commodity charts ({len(commodities)} folders) ✓")

    # ── Export data files ──────────────────────────────────────────────────
    log("  [5/5] Exporting data files...")
    export_analytics(metrics, base)
    export_monthly_returns(bt_result.monthly_returns, base)
    export_positions(bt_result.trades, base)

    # Live signals
    live_result: Optional[StrategyResult] = None
    if run_live:
        log("        Running live signal scan...")
        live_result = strategy.run(verbose=False)
        export_execution_files(
            live_signals=live_result.scored_signals,
            base=base,
            run_date=date.today(),
        )
        log(f"        execution_*_{date.today()}.json ✓ ({len(live_result.scored_signals)} signals)")

    # ── Summary ────────────────────────────────────────────────────────────
    log("\n" + "─" * 70)
    log("  REPORT SUMMARY")
    log("─" * 70)

    comb = metrics.get("combined")
    if comb:
        log(f"  Period   : {comb.start_date} → {comb.end_date}")
        log(f"  Return   : {comb.fmt_return()}")
        log(f"  Annual   : {comb.fmt_annual()}")
        log(f"  Sharpe   : {comb.fmt_sharpe()}")
        log(f"  Max DD   : {comb.fmt_maxdd()}")
        log(f"  Trades   : {comb.n_trades}")

    log(f"\n  Output   : {base.resolve()}/")
    log("  Files    : report.png | equity_curve.png | rolling_sharpe.png")
    log("             monthly_heatmap.png | combined_analytics.json")
    log("             combined_monthly_returns.csv | positions.csv")
    log(f"             + {len(commodities)} commodity sub-folders")
    log("═" * 70 + "\n")
