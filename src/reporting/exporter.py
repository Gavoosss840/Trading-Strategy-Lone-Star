"""
Exporter
─────────
Writes all output files to the output/ directory structure.

Output layout:
  output/
  ├── {commodity}/
  │   ├── analytics.json
  │   ├── equity_curve.png
  │   ├── monthly_heatmap.png
  │   ├── monthly_returns.csv
  │   ├── positions.csv
  │   └── rolling_sharpe.png
  ├── live/
  │   └── (same 6 files — today's live signals slice)
  ├── analytics.json               ← per-commodity summary
  ├── combined_analytics.json
  ├── combined_monthly_returns.csv
  ├── equity_curve.png
  ├── execution_{commodity}_{date}.json
  ├── monthly_heatmap.png
  ├── positions.csv
  ├── report.png
  └── rolling_sharpe.png
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.engine import BacktestResult, Trade
from src.reporting.analytics import PerformanceMetrics


# ── Directory setup ───────────────────────────────────────────────────────────

def ensure_output_dirs(base: Path, commodities: List[str]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for c in commodities + ["live"]:
        (base / c).mkdir(exist_ok=True)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _safe_json(obj):
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    return obj


def write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(_safe_json(data), f, indent=2, default=str)


# ── analytics.json ────────────────────────────────────────────────────────────

def export_analytics(
    metrics: Dict[str, PerformanceMetrics],
    base: Path,
) -> None:
    """Write analytics.json (root) and per-commodity analytics.json."""
    # Root summary
    summary = {k: v.to_dict() for k, v in metrics.items() if k != "combined"}
    write_json(summary, base / "analytics.json")

    # Combined
    if "combined" in metrics:
        write_json(metrics["combined"].to_dict(), base / "combined_analytics.json")

    # Per commodity
    for key, m in metrics.items():
        commodity_dir = base / key
        if commodity_dir.exists():
            write_json(m.to_dict(), commodity_dir / "analytics.json")


# ── monthly_returns.csv ───────────────────────────────────────────────────────

def export_monthly_returns(
    monthly_returns: pd.DataFrame,
    base: Path,
) -> None:
    """Write combined_monthly_returns.csv and per-commodity monthly_returns.csv."""
    mr = monthly_returns.copy()
    mr.index = mr.index.astype(str)

    mr.to_csv(base / "combined_monthly_returns.csv")

    for col in mr.columns:
        commodity_dir = base / col
        if commodity_dir.exists():
            mr[[col]].rename(columns={col: "monthly_return"}).to_csv(
                commodity_dir / "monthly_returns.csv"
            )


# ── positions.csv ─────────────────────────────────────────────────────────────

def _trade_to_row(t: Trade) -> dict:
    return {
        "ticker": t.ticker,
        "commodity": t.commodity,
        "role": t.role,
        "direction": t.direction,
        "entry_date": str(t.entry_date.date()) if t.entry_date else "",
        "exit_date": str(t.exit_date.date()) if t.exit_date else "open",
        "entry_price": round(t.entry_price, 4),
        "exit_price": round(t.exit_price, 4) if t.exit_price else None,
        "size_pct": round(t.size * 100, 2),
        "pnl_pct": round(t.pnl_pct * 100, 4),
        "holding_days": t.holding_days,
        "exit_reason": t.exit_reason,
    }


def export_positions(
    trades: List[Trade],
    base: Path,
) -> None:
    """Write positions.csv (root) and per-commodity positions.csv."""
    if not trades:
        return

    rows = [_trade_to_row(t) for t in trades]
    df_all = pd.DataFrame(rows)
    df_all.to_csv(base / "positions.csv", index=False)

    for commodity in df_all["commodity"].unique():
        commodity_dir = base / commodity
        if commodity_dir.exists():
            df_all[df_all["commodity"] == commodity].to_csv(
                commodity_dir / "positions.csv", index=False
            )


# ── execution_{commodity}_{date}.json ─────────────────────────────────────────

def export_execution_files(
    live_signals,               # List[ScoredSignal] from live run
    base: Path,
    run_date: Optional[date] = None,
) -> None:
    """Write one execution JSON file per commodity with today's signals."""
    if not live_signals:
        return

    run_date = run_date or date.today()
    date_str = run_date.strftime("%Y-%m-%d")

    # Group by commodity
    by_commodity: Dict[str, list] = {}
    for s in live_signals:
        by_commodity.setdefault(s.commodity, []).append(s)

    for commodity, signals in by_commodity.items():
        payload = {
            "strategy": "Lone Star",
            "commodity": commodity,
            "run_date": date_str,
            "n_signals": len(signals),
            "signals": [
                {
                    "ticker": s.ticker,
                    "direction": s.direction,
                    "alpha_pct": round(s.alpha * 100, 3),
                    "score": round(s.total_score, 4),
                    "strength": s.signal_strength,
                    "size_pct": round(s.position_size_pct * 100, 2),
                    "commodity": s.commodity,
                    "role": s.signal.role,
                    "beta_adjusted": round(s.signal.beta_adjusted, 4),
                    "shock_return_pct": round(s.signal.shock_return * 100, 3),
                    "shock_score": round(s.signal.shock_score, 4),
                    "shock_age_days": s.signal.shock_age_days,
                }
                for s in signals
            ],
        }
        fname = f"execution_{commodity}_{date_str}.json"
        write_json(payload, base / fname)

        # Also write to live/ subdir
        live_dir = base / "live"
        live_dir.mkdir(exist_ok=True)
        write_json(payload, live_dir / fname)


# ── Master export ─────────────────────────────────────────────────────────────

def export_all(
    result: BacktestResult,
    metrics: Dict[str, PerformanceMetrics],
    live_signals,
    base: Path,
) -> None:
    """Run all exporters in one call."""
    ensure_output_dirs(base, result.commodities)
    export_analytics(metrics, base)
    export_monthly_returns(result.monthly_returns, base)
    export_positions(result.trades, base)
    export_execution_files(live_signals, base)
    print(f"  [export] All files written to {base}/")
