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
  │   ├── execution_{commodity}_{date}.json   ← IBKR-ready (whole shares only)
  │   ├── manual_orders_{date}.json           ← ALL signals w/ bracket levels
  │   └── manual_orders_{date}.csv            ← same data in tabular form
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
import math
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
        "take_profit_price": round(t.take_profit_price, 4) if t.take_profit_price else None,
        "stop_loss_price": round(t.stop_loss_price, 4) if t.stop_loss_price else None,
        "alpha_at_entry_pct": round(t.alpha_at_entry * 100, 3),
        "size_pct": round(t.size * 100, 2),
        "pnl_pct": round(t.pnl_pct * 100, 4),
        "holding_days": t.holding_days,
        "exit_reason": t.exit_reason,
        "rebalance_count": t.rebalance_count,
        "last_rebalance_date": str(t.last_rebalance_date.date()) if t.last_rebalance_date else None,
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

def _signal_to_execution_record(s, nav_total: float) -> dict:
    """
    Convert a ScoredSignal to an execution order record with TP/SL levels.

    Take Profit price = entry fair-value when α = 0 (shock fully priced):
      LONG  → any reference price × (1 + |alpha|)
      SHORT → any reference price × (1 − |alpha|)

    Note: live signals have no entry price yet (not yet traded), so TP/SL
    are expressed as |alpha|% distances from the execution price — the
    broker/OMS should apply them at fill.

    Returns (record_dict, whole_shares) where whole_shares = floor(target_notional / price).
    Callers decide whether to route to execution file or manual report.
    """
    abs_alpha = abs(s.alpha)
    target_dollars = nav_total * s.position_size_pct
    # Whole-share quantity for IBKR (no fractional shares via API)
    price = getattr(s, "last_price", None)
    if price and price > 0:
        whole_shares = math.floor(target_dollars / price)
    else:
        whole_shares = 0

    return {
        "ticker": s.ticker,
        "direction": s.direction,
        "alpha_pct": round(s.alpha * 100, 3),
        "score": round(s.total_score, 4),
        "strength": s.signal_strength,
        "size_pct": round(s.position_size_pct * 100, 2),
        "target_notional_usd": round(target_dollars, 2),
        "whole_shares": whole_shares,
        "commodity": s.commodity,
        "role": s.signal.role,
        "beta_adjusted": round(s.signal.beta_adjusted, 4),
        "shock_return_pct": round(s.signal.shock_return * 100, 3),
        "shock_score": round(s.signal.shock_score, 4),
        "shock_age_days": s.signal.shock_age_days,
        # Exit levels (as % distances from fill price — apply at execution)
        "take_profit_distance_pct": round(abs_alpha * 100, 3),   # α=0 fair value
        "stop_loss_distance_pct": None,   # filled from config at OMS level
        "exit_logic": {
            "take_profit": f"LONG: fill × (1 + {abs_alpha:.3%}) | SHORT: fill × (1 - {abs_alpha:.3%})",
            "stop_loss":   "fill × (1 ± stop_loss_pct) — set from risk config",
        },
    }


def export_execution_files(
    live_signals,               # List[ScoredSignal] from live run
    base: Path,
    nav_total: float = 10_000.0,
    min_order_notional_usd: float = 100.0,
    run_date: Optional[date] = None,
) -> None:
    """
    Write one execution JSON file per commodity with today's IBKR-ready signals.

    Signals that cannot be expressed as at least 1 whole share (because NAV is
    too small or stock price is high) are excluded from the execution files —
    they appear instead in the manual_orders report alongside all other signals.
    """
    if not live_signals:
        return

    run_date = run_date or date.today()
    date_str = run_date.strftime("%Y-%m-%d")

    # Group by commodity; split into IBKR-executable vs manual
    by_commodity: Dict[str, list] = {}
    for s in live_signals:
        by_commodity.setdefault(s.commodity, []).append(s)

    for commodity, signals in by_commodity.items():
        ibkr_records = []
        for s in signals:
            rec = _signal_to_execution_record(s, nav_total)
            target_dollars = nav_total * s.position_size_pct
            # Route to execution file only if ≥ 1 whole share AND notional ≥ min
            if rec["whole_shares"] >= 1 and target_dollars >= min_order_notional_usd:
                ibkr_records.append(rec)

        if not ibkr_records:
            continue

        payload = {
            "strategy": "Lone Star",
            "commodity": commodity,
            "run_date": date_str,
            "nav_total_usd": nav_total,
            "note": "whole-share quantities only — fractional signals go to manual_orders report",
            "n_signals": len(ibkr_records),
            "signals": ibkr_records,
        }
        fname = f"execution_{commodity}_{date_str}.json"
        write_json(payload, base / fname)

        # Also write to live/ subdir
        live_dir = base / "live"
        live_dir.mkdir(exist_ok=True)
        write_json(payload, live_dir / fname)


# ── manual_orders_{date}.json / .csv ─────────────────────────────────────────

def _bracket_order(s, nav_total: float, stop_loss_pct: float) -> dict:
    """
    Build a complete IBKR bracket-order record for manual placement.

    Structure:
      Parent LMT  — entry at last known price (user adjusts before placing)
      Take Profit LMT leg — at α=0 fair value
      Stop Loss   STP leg — at entry × (1 ∓ stop_loss_pct)

    Because we don't have a guaranteed fill price, all prices are computed from
    `last_price` (last market close).  The user should review / adjust before
    sending to IBKR.
    """
    abs_alpha = abs(s.alpha)
    direction = s.direction.upper()          # "LONG" | "SHORT"
    price = getattr(s, "last_price", None) or 0.0

    target_dollars = nav_total * s.position_size_pct
    whole_shares = math.floor(target_dollars / price) if price > 0 else 0
    # Even if whole_shares == 0 we still output the record so the user can
    # decide whether to round up to 1 share manually.

    if direction == "LONG":
        action = "BUY"
        tp_price = round(price * (1 + abs_alpha), 4) if price else None
        sl_price = round(price * (1 - stop_loss_pct), 4) if price else None
        tp_action = "SELL"
        sl_action = "SELL"
    else:  # SHORT
        action = "SELL SHORT"
        tp_price = round(price * (1 - abs_alpha), 4) if price else None
        sl_price = round(price * (1 + stop_loss_pct), 4) if price else None
        tp_action = "BUY"
        sl_action = "BUY"

    return {
        "ticker": s.ticker,
        "commodity": s.commodity,
        "direction": direction,
        "alpha_pct": round(s.alpha * 100, 3),
        "score": round(s.total_score, 4),
        "strength": s.signal_strength,
        "size_pct": round(s.position_size_pct * 100, 2),
        "target_notional_usd": round(target_dollars, 2),
        "last_price": round(price, 4) if price else None,
        "whole_shares": whole_shares,
        "fractional_shares_needed": whole_shares < 1,
        # ── IBKR bracket order legs ──────────────────────────────────────────
        "parent_order": {
            "action": action,
            "quantity": max(whole_shares, 1),   # floor at 1 for user reference
            "order_type": "LMT",
            "limit_price": round(price, 4) if price else None,
            "note": "Adjust limit_price to desired entry before placing",
        },
        "take_profit_leg": {
            "action": tp_action,
            "quantity": max(whole_shares, 1),
            "order_type": "LMT",
            "limit_price": tp_price,
            "distance_pct": round(abs_alpha * 100, 3),
        },
        "stop_loss_leg": {
            "action": sl_action,
            "quantity": max(whole_shares, 1),
            "order_type": "STP",
            "stop_price": sl_price,
            "distance_pct": round(stop_loss_pct * 100, 3),
        },
    }


def export_manual_orders(
    live_signals,               # List[ScoredSignal]
    base: Path,
    nav_total: float = 10_000.0,
    stop_loss_pct: float = 0.05,
    run_date: Optional[date] = None,
) -> None:
    """
    Write manual_orders_{date}.json and manual_orders_{date}.csv to output/live/.

    Contains ALL live signals with complete IBKR bracket-order details
    (parent LMT entry + take-profit LMT + stop-loss STP).  Signals where
    whole_shares < 1 are flagged with fractional_shares_needed=True.

    Use this report to place orders manually when:
      • NAV is too low to buy ≥1 whole share at the target notional
      • You prefer full manual control over bracket orders
    """
    if not live_signals:
        return

    run_date = run_date or date.today()
    date_str = run_date.strftime("%Y-%m-%d")
    live_dir = base / "live"
    live_dir.mkdir(exist_ok=True)

    orders = [_bracket_order(s, nav_total, stop_loss_pct) for s in live_signals]

    # ── JSON ─────────────────────────────────────────────────────────────────
    payload = {
        "strategy": "Lone Star",
        "run_date": date_str,
        "nav_total_usd": nav_total,
        "stop_loss_pct": stop_loss_pct,
        "n_orders": len(orders),
        "instructions": (
            "Place as IBKR bracket orders: Parent LMT → attach TP LMT + SL STP legs. "
            "Review limit_price before submission. "
            "Signals with fractional_shares_needed=true require rounding up to 1 share "
            "or adjusting position size manually."
        ),
        "orders": orders,
    }
    write_json(payload, live_dir / f"manual_orders_{date_str}.json")

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = []
    for o in orders:
        rows.append({
            "ticker": o["ticker"],
            "commodity": o["commodity"],
            "direction": o["direction"],
            "alpha_pct": o["alpha_pct"],
            "score": o["score"],
            "strength": o["strength"],
            "size_pct": o["size_pct"],
            "target_notional_usd": o["target_notional_usd"],
            "last_price": o["last_price"],
            "whole_shares": o["whole_shares"],
            "fractional_shares_needed": o["fractional_shares_needed"],
            # Parent order
            "action": o["parent_order"]["action"],
            "quantity": o["parent_order"]["quantity"],
            "entry_limit_price": o["parent_order"]["limit_price"],
            # TP leg
            "tp_action": o["take_profit_leg"]["action"],
            "tp_limit_price": o["take_profit_leg"]["limit_price"],
            "tp_distance_pct": o["take_profit_leg"]["distance_pct"],
            # SL leg
            "sl_action": o["stop_loss_leg"]["action"],
            "sl_stop_price": o["stop_loss_leg"]["stop_price"],
            "sl_distance_pct": o["stop_loss_leg"]["distance_pct"],
        })

    pd.DataFrame(rows).to_csv(
        live_dir / f"manual_orders_{date_str}.csv", index=False
    )

    n_fractional = sum(1 for o in orders if o["fractional_shares_needed"])
    print(
        f"  [export] manual_orders_{date_str}: {len(orders)} signals "
        f"({n_fractional} require whole-share rounding)"
    )


# ── Master export ─────────────────────────────────────────────────────────────

def export_all(
    result: BacktestResult,
    metrics: Dict[str, PerformanceMetrics],
    live_signals,
    base: Path,
    nav_total: float = 10_000.0,
    stop_loss_pct: float = 0.05,
    min_order_notional_usd: float = 100.0,
) -> None:
    """Run all exporters in one call."""
    ensure_output_dirs(base, result.commodities)
    export_analytics(metrics, base)
    export_monthly_returns(result.monthly_returns, base)
    export_positions(result.trades, base)
    export_execution_files(
        live_signals, base,
        nav_total=nav_total,
        min_order_notional_usd=min_order_notional_usd,
    )
    export_manual_orders(
        live_signals, base,
        nav_total=nav_total,
        stop_loss_pct=stop_loss_pct,
    )
    print(f"  [export] All files written to {base}/")
