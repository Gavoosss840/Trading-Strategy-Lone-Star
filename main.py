#!/usr/bin/env python3
"""
Lone Star — Entry Point
────────────────────────
Run the Lone Star commodity-exposure alpha strategy.

Usage:
    python main.py                          # Run live signals only
    python main.py --report                 # Full backtest + charts + report.png
    python main.py --report --output out/   # Custom output directory
    python main.py --config path/to/cfg.yaml
    python main.py --report --start 2022-01-01 --lookback 1500  # Backtest from 2022 with 6yr data
    python main.py --top 20                 # Show top 20 live signals
    python main.py --commodity crude_oil    # Filter to one commodity
    python main.py --zone europe            # Filter to one geographic zone
    python main.py --export signals.csv     # Export live signals to CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lone Star — Commodity Exposure Alpha Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top signals to display (default: 10)",
    )
    parser.add_argument(
        "--commodity",
        default=None,
        help="Filter signals to a specific commodity (e.g. crude_oil, gold)",
    )
    parser.add_argument(
        "--zone",
        default=None,
        choices=["americas", "europe", "middle_east", "asia_pacific"],
        help="Filter signals to a geographic zone",
    )
    parser.add_argument(
        "--direction",
        choices=["LONG", "SHORT"],
        default=None,
        help="Filter signals by direction",
    )
    parser.add_argument(
        "--export",
        default=None,
        metavar="FILE",
        help="Export signals to a CSV file",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Run full backtest and generate report.png + all output files",
    )
    parser.add_argument(
        "--output",
        default="output",
        metavar="DIR",
        help="Output directory for reports (default: output/)",
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Backtest start date, e.g. 2022-01-01 (default: earliest available)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Backtest end date, e.g. 2025-12-31 (default: latest available)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        metavar="DAYS",
        help="Override lookback_days from config (e.g. 1500 for ~6 years of data)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip live signal scan when running --report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pipeline progress output",
    )
    parser.add_argument(
        "--show-betas",
        action="store_true",
        help="Print the exposure map (commodity betas) after running",
    )
    parser.add_argument(
        "--show-shocks",
        action="store_true",
        help="Print detected commodity shocks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        from src.strategy import LoneStarStrategy
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print("Make sure you installed requirements: pip install -r requirements.txt", file=sys.stderr)
        return 1

    # ── Report mode (backtest + full output) ──────────────────────────────
    if args.report:
        try:
            from src.reporting.report_builder import build_reports
        except ImportError as exc:
            print(f"Import error (reporting): {exc}", file=sys.stderr)
            return 1
        build_reports(
            config_path=str(config_path),
            output_dir=args.output,
            run_live=not args.no_live,
            verbose=not args.quiet,
            start_date=args.start,
            end_date=args.end,
            lookback=args.lookback,
        )
        return 0

    # ── Load and run strategy ──────────────────────────────────────────────────
    strategy = LoneStarStrategy.from_config(str(config_path))
    result = strategy.run(verbose=not args.quiet)

    # ── Optional outputs ───────────────────────────────────────────────────────
    if args.show_betas and result.exposure_map is not None:
        from src.pipeline.exposure_mapping import summarize_exposures
        from tabulate import tabulate
        print("\n📐 Commodity Beta Exposure Map (significant pairs):")
        df = summarize_exposures(result.exposure_map).reset_index()
        # Enrich with zone and company name
        if not result.stock_metadata.empty:
            meta = result.stock_metadata[["name", "zone"]].reset_index()
            df = df.merge(meta, left_on="ticker", right_on="ticker", how="left")
            df.insert(1, "zone", df.pop("zone"))
            df.insert(2, "name", df.pop("name"))
        if not df.empty:
            print(tabulate(df.head(40), headers="keys", tablefmt="rounded_outline", showindex=False))
        else:
            print("  No significant exposures found.")

    if args.show_shocks and not result.active_shocks_df.empty:
        print("\n⚡ Detected Commodity Shocks:")
        print(result.active_shocks_df.to_string())

    # ── Filter signals ─────────────────────────────────────────────────────────
    signals = result.scored_signals
    meta = result.stock_metadata

    if args.commodity:
        signals = [s for s in signals if s.commodity == args.commodity]
    if args.direction:
        signals = [s for s in signals if s.direction == args.direction]
    if args.zone and not meta.empty:
        tickers_in_zone = meta[meta["zone"] == args.zone].index.tolist()
        signals = [s for s in signals if s.ticker in tickers_in_zone]

    # ── Print results ──────────────────────────────────────────────────────────
    result.scored_signals = signals
    result.print_signals(top_n=args.top)

    # ── Export ─────────────────────────────────────────────────────────────────
    if args.export:
        df = result.signals_dataframe()
        out_path = Path(args.export)
        df.to_csv(out_path, index=False)
        print(f"  Signals exported to: {out_path}")

    # ── Warnings ───────────────────────────────────────────────────────────────
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  ⚠  {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
