"""
Rebalancer
──────────
Computes DELTA orders by comparing live signals (target positions) against
a user-provided open-positions file (current state in the broker).

Instead of close + reopen with a different size, produces three lists:

  new_entries  — signals with no existing position  → full bracket order
  adjustments  — signals matching an existing position:
                   target > current  → BUY delta shares
                   target < current  → SELL delta shares
                   (unchanged)       → no order, just confirm TP/SL
  closes       — existing positions with no corresponding live signal
                 → market/limit SELL or BUY to close

Input file formats (CSV or JSON):
  CSV columns : ticker, commodity, direction, shares, entry_price, [entry_date]
  JSON        : list of dicts with the same keys

Usage (CLI):
    python main.py --positions output/live/my_positions.csv
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.reporting.exporter import _bracket_order, _vol_adjusted_sl, write_json


# ── Position loader ───────────────────────────────────────────────────────────

def load_positions(path: str | Path) -> List[dict]:
    """
    Load open positions from CSV or JSON file.

    Returns a list of dicts with keys:
      ticker, commodity, direction, shares (int), entry_price (float), entry_date (str)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Positions file not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "positions" in data:
            data = data["positions"]
        return [_normalise_position(r) for r in data]

    # CSV
    rows = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_normalise_position(row))
    return rows


def _normalise_position(r: dict) -> dict:
    return {
        "ticker":     str(r.get("ticker", "")).strip().upper(),
        "commodity":  str(r.get("commodity", "")).strip().lower(),
        "direction":  str(r.get("direction", "LONG")).strip().upper(),
        "shares":     int(float(r.get("shares", 0))),
        "entry_price": float(r.get("entry_price", 0.0)),
        "entry_date":  str(r.get("entry_date", "")),
    }


# ── Template generator ────────────────────────────────────────────────────────

def write_positions_template(base: Path) -> Path:
    """
    Write output/live/positions_template.csv so the user knows the format.
    They fill in their current IBKR holdings and pass it with --positions.
    """
    live_dir = base / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    out = live_dir / "positions_template.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "commodity", "direction", "shares", "entry_price", "entry_date"])
        writer.writerow(["FCX", "copper", "LONG", 10, 45.20, "2026-05-20"])
        writer.writerow(["SCCO", "copper", "LONG", 5, 82.10, "2026-05-18"])
        writer.writerow(["AG", "silver", "LONG", 20, 8.35, "2026-05-15"])
    return out


# ── Delta order computation ───────────────────────────────────────────────────

def compute_rebalance_orders(
    live_signals,                          # List[ScoredSignal]
    open_positions: List[dict],
    nav_total: float,
    stop_loss_pct: float = 0.03,
    atr_multiplier: float = 2.0,
    max_atr_stop: float = 0.12,
    tp_alpha_ratio: float = 1.0,
    min_order_notional_usd: float = 50.0,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Compare live signals (target) with open positions (current).

    Returns
    -------
    new_entries  : full bracket orders for positions that don't exist yet
    adjustments  : delta orders for positions that need resizing
    closes       : close orders for positions no longer in live signals
    """
    # Build lookup keyed by (ticker, commodity)
    pos_map: Dict[Tuple[str, str], dict] = {
        (p["ticker"], p["commodity"]): p
        for p in open_positions
    }
    sig_map: Dict[Tuple[str, str], object] = {
        (s.ticker, s.commodity): s
        for s in live_signals
    }

    new_entries: List[dict]  = []
    adjustments: List[dict]  = []
    closes: List[dict]       = []

    # ── Process signals → compare against open positions ─────────────────
    for (ticker, commodity), s in sig_map.items():
        price = getattr(s, "last_price", None) or 0.0
        if price <= 0:
            continue

        target_dollars = nav_total * s.position_size_pct
        target_shares  = math.floor(target_dollars / price) if price > 0 else 0

        if (ticker, commodity) not in pos_map:
            # ── New position — full bracket order ─────────────────────────
            order = _bracket_order(
                s, nav_total, stop_loss_pct,
                atr_multiplier, max_atr_stop, tp_alpha_ratio,
            )
            order["order_category"] = "new_entry"
            new_entries.append(order)

        else:
            # ── Existing position — compute delta ─────────────────────────
            pos          = pos_map[(ticker, commodity)]
            current_shares = pos["shares"]
            delta          = target_shares - current_shares

            if abs(delta) < 1:
                continue  # no meaningful change

            abs_tp_dist  = abs(s.alpha) * tp_alpha_ratio
            eff_stop     = _vol_adjusted_sl(s, stop_loss_pct, atr_multiplier, max_atr_stop)
            order_value  = abs(delta) * price

            if order_value < min_order_notional_usd:
                continue  # too small to bother

            if delta > 0:
                action = "BUY" if pos["direction"] == "LONG" else "BUY_TO_COVER"
                adj_type = "INCREASE"
            else:
                action = "SELL" if pos["direction"] == "LONG" else "SELL_SHORT"
                adj_type = "DECREASE"

            # Updated TP/SL levels from current price (for reference)
            if pos["direction"] == "LONG":
                new_tp = round(price * (1.0 + abs_tp_dist), 4)
                new_sl = round(price * (1.0 - eff_stop), 4)
            else:
                new_tp = round(price * (1.0 - abs_tp_dist), 4)
                new_sl = round(price * (1.0 + eff_stop), 4)

            adjustments.append({
                "order_category":     "adjustment",
                "type":               adj_type,
                "ticker":             ticker,
                "commodity":          commodity,
                "direction":          pos["direction"],
                "action":             action,
                "current_shares":     current_shares,
                "target_shares":      target_shares,
                "delta_shares":       abs(delta),
                "current_exposure_pct": round(current_shares * price / nav_total * 100, 2),
                "target_exposure_pct":  round(target_shares  * price / nav_total * 100, 2),
                "last_price":         round(price, 4),
                "order_value_usd":    round(order_value, 2),
                "order_type":         "LMT",
                "limit_price":        round(price, 4),
                # Updated TP/SL for reference — update existing bracket legs if needed
                "updated_tp_price":   new_tp,
                "updated_sl_price":   new_sl,
                "tp_distance_pct":    round(abs_tp_dist * 100, 3),
                "sl_distance_pct":    round(eff_stop * 100, 3),
                "note": (
                    f"Adjust existing {pos['direction']} position: "
                    f"{current_shares} → {target_shares} shares. "
                    f"Update TP/SL on existing bracket to new levels above."
                ),
            })

    # ── Positions no longer in live signals → close ───────────────────────
    for (ticker, commodity), pos in pos_map.items():
        if (ticker, commodity) not in sig_map:
            price = 0.0  # market price unknown without live data here
            action = "SELL" if pos["direction"] == "LONG" else "BUY_TO_COVER"
            closes.append({
                "order_category": "close",
                "ticker":         ticker,
                "commodity":      commodity,
                "direction":      pos["direction"],
                "action":         action,
                "shares":         pos["shares"],
                "order_type":     "MKT",
                "reason":         "signal_expired_or_commodity_suppressed",
                "note":           "Cancel existing TP/SL bracket legs before closing.",
            })

    return new_entries, adjustments, closes


# ── Export ────────────────────────────────────────────────────────────────────

def export_rebalance_orders(
    live_signals,
    open_positions_path: str | Path,
    base: Path,
    nav_total: float = 10_000.0,
    stop_loss_pct: float = 0.03,
    atr_multiplier: float = 2.0,
    max_atr_stop: float = 0.12,
    tp_alpha_ratio: float = 1.0,
    min_order_notional_usd: float = 50.0,
    run_date: Optional[date] = None,
) -> None:
    """
    Load open positions, compute delta orders, write rebalance_orders_{date}.json/.csv.
    """
    run_date = run_date or date.today()
    date_str = run_date.strftime("%Y-%m-%d")

    open_positions = load_positions(open_positions_path)

    new_entries, adjustments, closes = compute_rebalance_orders(
        live_signals, open_positions, nav_total,
        stop_loss_pct, atr_multiplier, max_atr_stop, tp_alpha_ratio,
        min_order_notional_usd,
    )

    total = len(new_entries) + len(adjustments) + len(closes)

    payload = {
        "strategy":           "Lone Star",
        "run_date":           date_str,
        "nav_total_usd":      nav_total,
        "open_positions_file": str(open_positions_path),
        "summary": {
            "new_entries":   len(new_entries),
            "adjustments":   len(adjustments),
            "closes":        len(closes),
            "total_orders":  total,
        },
        "instructions": (
            "new_entries: place as bracket orders (parent LMT + TP LMT + SL STP). "
            "adjustments: single BUY or SELL limit order — DO NOT close and reopen. "
            "Also update existing bracket TP/SL legs to updated_tp_price / updated_sl_price. "
            "closes: cancel existing bracket legs first, then close with MKT order."
        ),
        "new_entries":  new_entries,
        "adjustments":  adjustments,
        "closes":       closes,
    }

    live_dir = base / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    fname = f"rebalance_orders_{date_str}"
    write_json(payload, live_dir / f"{fname}.json")

    # ── CSV (flat, all order types in one file) ────────────────────────────
    import pandas as pd

    rows = []
    for o in new_entries:
        rows.append({
            "category":       "new_entry",
            "ticker":         o["ticker"],
            "commodity":      o["commodity"],
            "direction":      o["direction"],
            "action":         o["parent_order"]["action"],
            "delta_shares":   o["parent_order"]["quantity"],
            "current_shares": 0,
            "target_shares":  o["parent_order"]["quantity"],
            "order_value_usd": o["target_notional_usd"],
            "order_type":     "LMT",
            "limit_price":    o["parent_order"]["limit_price"],
            "tp_price":       o["take_profit_leg"]["limit_price"],
            "sl_price":       o["stop_loss_leg"]["stop_price"],
        })
    for o in adjustments:
        rows.append({
            "category":       "adjustment",
            "ticker":         o["ticker"],
            "commodity":      o["commodity"],
            "direction":      o["direction"],
            "action":         o["action"],
            "delta_shares":   o["delta_shares"],
            "current_shares": o["current_shares"],
            "target_shares":  o["target_shares"],
            "order_value_usd": o["order_value_usd"],
            "order_type":     o["order_type"],
            "limit_price":    o["limit_price"],
            "tp_price":       o["updated_tp_price"],
            "sl_price":       o["updated_sl_price"],
        })
    for o in closes:
        rows.append({
            "category":       "close",
            "ticker":         o["ticker"],
            "commodity":      o["commodity"],
            "direction":      o["direction"],
            "action":         o["action"],
            "delta_shares":   o["shares"],
            "current_shares": o["shares"],
            "target_shares":  0,
            "order_value_usd": None,
            "order_type":     "MKT",
            "limit_price":    None,
            "tp_price":       None,
            "sl_price":       None,
        })

    if rows:
        pd.DataFrame(rows).to_csv(live_dir / f"{fname}.csv", index=False)

    print(
        f"  [rebalance] {fname}: "
        f"{len(new_entries)} new | {len(adjustments)} adjust | {len(closes)} close"
    )
