"""
Order Router
─────────────
Routes live Lone Star signals to:
  1. IBKR TWS API — bracket orders (parent LMT + TP LMT + SL STP) for
     signals where floor(target_dollars / price) >= 1 whole share AND
     buying power is sufficient.
  2. Manual list  — CSV + JSON for signals where:
       • whole shares < 1   (fractional — IBKR API doesn't support it)
       • target notional < min_order_notional_usd (too small)
       • buying power < target notional (insufficient funds at this moment)
       • any API error during placement

Actual account NAV is fetched from IBKR and used for position sizing,
overriding the config's nav_total so orders are always scaled to reality.

Usage:
    from src.execution.order_router import route_signals
    route_signals(live_signals, cfg["ibkr"], cfg["risk"], cfg["execution"])
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.execution.ibkr_client import IBKRClient, IBKRConnectionError
from src.reporting.exporter import _vol_adjusted_sl, _bracket_order, write_json


def route_signals(
    live_signals,
    ibkr_cfg: dict,
    risk_cfg: dict,
    exec_cfg: dict,
    output_dir: str | Path = "output",
    run_date: Optional[date] = None,
    ibkr_mode_override: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Connect to IBKR, query real account NAV, route signals to API or manual file.

    Parameters
    ----------
    live_signals         : List[ScoredSignal] from LoneStarStrategy.run()
    ibkr_cfg             : cfg["ibkr"] section from config.yaml
    risk_cfg             : cfg["risk"] section  (TP/SL params)
    exec_cfg             : cfg["execution"] section
    output_dir           : root output directory (manual_orders go to output/live/)
    run_date             : date to stamp on output files (default: today)
    ibkr_mode_override   : "paper" or "live" — overrides ibkr_cfg["mode"]
    dry_run              : compute orders and write manual file WITHOUT connecting to IBKR
    """
    run_date = run_date or date.today()
    base     = Path(output_dir)

    # ── Risk / execution params ───────────────────────────────────────────────
    stop_loss_pct  = float(risk_cfg.get("stop_loss_pct", 0.03))
    atr_multiplier = float(risk_cfg.get("atr_stop_multiplier", 2.0))
    max_atr_stop   = float(risk_cfg.get("max_atr_stop_pct", 0.12))
    tp_alpha_ratio = float(risk_cfg.get("tp_alpha_ratio", 1.0))
    min_notional   = float(exec_cfg.get("min_order_notional_usd", 100.0))
    config_nav     = float(exec_cfg.get("nav_total", 10_000.0))

    # ── Account data (real or simulated) ─────────────────────────────────────
    if dry_run:
        print("  [order_router] DRY RUN — no connection to IBKR, using config nav_total")
        nav          = config_nav
        buying_power = config_nav
        client       = None
    else:
        client = IBKRClient.from_config(ibkr_cfg, mode_override=ibkr_mode_override)
        client.connect()
        mode_label = ibkr_mode_override or ibkr_cfg.get("mode", "paper")
        print(f"  [IBKR] Connected ({mode_label}) → {client.host}:{client.port}")
        try:
            nav          = client.get_nav()
            buying_power = client.get_buying_power()
            print(f"  [IBKR] NAV = ${nav:,.2f}  |  Buying power = ${buying_power:,.2f}")
        except Exception as exc:
            print(f"  [IBKR] Warning: could not read account values ({exc}). Using config nav_total.")
            nav          = config_nav
            buying_power = config_nav

    # ── Route each signal ─────────────────────────────────────────────────────
    ibkr_placed: List[dict]   = []
    manual_orders: List[dict] = []
    skipped: List[dict]       = []

    for s in live_signals:
        price = getattr(s, "last_price", None) or 0.0
        if price <= 0:
            skipped.append({"ticker": s.ticker, "reason": "no_price_data"})
            continue

        target_dollars = nav * s.position_size_pct
        whole_shares   = math.floor(target_dollars / price)
        bracket        = _bracket_order(
            s, nav, stop_loss_pct, atr_multiplier, max_atr_stop, tp_alpha_ratio
        )

        # ── Routing decision ──────────────────────────────────────────────────
        if whole_shares < 1:
            bracket["route"]  = "manual"
            bracket["reason"] = "fractional_share"
            manual_orders.append(bracket)
            continue
        if target_dollars < min_notional:
            bracket["route"]  = "manual"
            bracket["reason"] = "below_min_notional"
            manual_orders.append(bracket)
            continue
        if target_dollars > buying_power:
            bracket["route"]  = "manual"
            bracket["reason"] = "insufficient_funds"
            manual_orders.append(bracket)
            continue

        # ── Route to IBKR API ─────────────────────────────────────────────────
        eff_stop  = _vol_adjusted_sl(s, stop_loss_pct, atr_multiplier, max_atr_stop)
        abs_tp    = abs(s.alpha) * tp_alpha_ratio
        direction = s.direction.upper()
        if direction == "LONG":
            tp_price = round(price * (1 + abs_tp), 4)
            sl_price = round(price * (1 - eff_stop), 4)
        else:
            tp_price = round(price * (1 - abs_tp), 4)
            sl_price = round(price * (1 + eff_stop), 4)

        bracket["route"]      = "ibkr_api"
        bracket["quantity"]   = whole_shares
        bracket["tp_price"]   = tp_price
        bracket["sl_price"]   = sl_price

        if dry_run:
            ibkr_placed.append(bracket)
            print(
                f"  [DRY RUN] Would place: {s.ticker} {direction} {whole_shares}sh "
                f"@ ${price:.2f}  TP=${tp_price:.2f}  SL=${sl_price:.2f}"
            )
        else:
            try:
                trades = client.place_bracket_order(
                    ticker            = s.ticker,
                    direction         = direction,
                    quantity          = whole_shares,
                    entry_limit_price = round(price, 4),
                    tp_price          = tp_price,
                    sl_price          = sl_price,
                )
                bracket["order_id"] = trades[0].order.orderId if trades else None
                ibkr_placed.append(bracket)
                print(
                    f"  [IBKR ✓] {s.ticker} {direction} {whole_shares}sh "
                    f"@ ${price:.2f}  TP=${tp_price:.2f}  SL=${sl_price:.2f}  "
                    f"(orderId={bracket['order_id']})"
                )
            except Exception as exc:
                print(f"  [IBKR ✗] Failed to place {s.ticker}: {exc}")
                bracket["route"]  = "manual"
                bracket["reason"] = f"api_error: {exc}"
                manual_orders.append(bracket)

    if client and not dry_run:
        client.disconnect()

    # ── Write output files ────────────────────────────────────────────────────
    _write_order_summary(ibkr_placed, manual_orders, base, run_date, nav)

    # ── Summary line ──────────────────────────────────────────────────────────
    tag = "(DRY RUN) " if dry_run else ""
    print(
        f"  [order_router] {tag}"
        f"{len(ibkr_placed)} orders → IBKR API | "
        f"{len(manual_orders)} → manual list | "
        f"{len(skipped)} skipped"
    )


# ── Output writers ────────────────────────────────────────────────────────────

def _write_order_summary(
    ibkr_placed: list,
    manual_orders: list,
    base: Path,
    run_date: date,
    nav: float,
) -> None:
    """Write manual_orders and ibkr_placed JSON/CSV to output/live/."""
    date_str = run_date.strftime("%Y-%m-%d")
    live_dir = base / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    # Manual orders (fractional / insufficient funds / API errors)
    if manual_orders:
        payload = {
            "strategy":      "Lone Star",
            "run_date":      date_str,
            "nav_total_usd": nav,
            "n_orders":      len(manual_orders),
            "instructions": (
                "These orders require manual placement in TWS. "
                "Place as bracket orders: Parent LMT → attach TP LMT + SL STP. "
                "'fractional_share': position < 1 whole share — round up to 1 or skip. "
                "'insufficient_funds': buying power below target — reduce size manually. "
                "'api_error': order was rejected by IBKR — review and place manually."
            ),
            "orders": manual_orders,
        }
        write_json(payload, live_dir / f"manual_orders_{date_str}.json")
        _orders_to_csv(manual_orders, live_dir / f"manual_orders_{date_str}.csv")
        print(f"  [order_router] manual_orders_{date_str}.json/.csv  ({len(manual_orders)} orders)")

    # IBKR-placed orders log (for record-keeping)
    if ibkr_placed:
        payload = {
            "strategy":      "Lone Star",
            "run_date":      date_str,
            "nav_total_usd": nav,
            "n_orders":      len(ibkr_placed),
            "note":          "These orders were placed via IBKR API. Check TWS for status.",
            "orders":        ibkr_placed,
        }
        write_json(payload, live_dir / f"ibkr_placed_{date_str}.json")
        _orders_to_csv(ibkr_placed, live_dir / f"ibkr_placed_{date_str}.csv")
        print(f"  [order_router] ibkr_placed_{date_str}.json/.csv  ({len(ibkr_placed)} orders)")


def _orders_to_csv(orders: list, path: Path) -> None:
    rows = []
    for o in orders:
        rows.append({
            "route":                   o.get("route", ""),
            "reason":                  o.get("reason", ""),
            "ticker":                  o.get("ticker", ""),
            "commodity":               o.get("commodity", ""),
            "direction":               o.get("direction", ""),
            "alpha_pct":               o.get("alpha_pct", ""),
            "score":                   o.get("score", ""),
            "size_pct":                o.get("size_pct", ""),
            "target_notional_usd":     o.get("target_notional_usd", ""),
            "last_price":              o.get("last_price", ""),
            "whole_shares":            o.get("whole_shares", ""),
            "fractional_shares_needed": o.get("fractional_shares_needed", ""),
            "action":                  o["parent_order"]["action"],
            "quantity":                o["parent_order"]["quantity"],
            "entry_limit_price":       o["parent_order"]["limit_price"],
            "tp_limit_price":          o["take_profit_leg"]["limit_price"],
            "tp_distance_pct":         o["take_profit_leg"]["distance_pct"],
            "sl_stop_price":           o["stop_loss_leg"]["stop_price"],
            "sl_distance_pct":         o["stop_loss_leg"]["distance_pct"],
            "sl_vol_adjusted":         o["stop_loss_leg"]["vol_adjusted"],
            "order_id":                o.get("order_id", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
