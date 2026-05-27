"""
Analytics
──────────
Compute all performance metrics from a daily P&L series.

Metrics:
  total_return       : (final_equity / initial_equity) - 1
  annual_return      : CAGR over the period
  sharpe_ratio       : annualised (risk-free = 0 for simplicity)
  max_drawdown       : peak-to-trough max decline
  win_rate           : % of months with positive return
  n_trades           : total number of closed trades
  avg_holding_days   : average days per trade
  profit_factor      : gross_profit / abs(gross_loss)
  calmar_ratio       : annual_return / abs(max_drawdown)
  volatility         : annualised daily return std
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, Trade


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    universe: str
    start_date: str
    end_date: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    win_rate_monthly: float
    n_trades: int
    avg_holding_days: float
    profit_factor: float
    best_month: float
    worst_month: float

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round floats for JSON
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d

    def fmt_return(self) -> str:
        return f"{self.total_return * 100:+.1f}%"

    def fmt_annual(self) -> str:
        return f"{self.annual_return * 100:+.1f}%"

    def fmt_sharpe(self) -> str:
        return f"{self.sharpe_ratio:.2f}"

    def fmt_maxdd(self) -> str:
        return f"{self.max_drawdown * 100:.1f}%"


# ── Core computation ──────────────────────────────────────────────────────────

def _years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((end - start).days / 365.25, 1 / 252)


def compute_metrics(
    daily_pnl: pd.Series,
    universe: str,
    trades: Optional[List[Trade]] = None,
    monthly_returns: Optional[pd.Series] = None,
    risk_free_annual: float = 0.0,
) -> PerformanceMetrics:
    """Compute all performance metrics from a daily P&L series."""
    pnl = daily_pnl.dropna()
    if pnl.empty:
        return _empty_metrics(universe)

    equity = (1 + pnl).cumprod()
    total_ret = float(equity.iloc[-1] - 1)

    start = pnl.index[0]
    end = pnl.index[-1]
    n_years = _years(start, end)

    annual_ret = float((1 + total_ret) ** (1 / n_years) - 1)
    vol = float(pnl.std() * np.sqrt(252))
    sharpe = (annual_ret - risk_free_annual) / vol if vol > 0 else 0.0

    # Max drawdown
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = float(dd.min())

    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Monthly win rate
    if monthly_returns is not None and not monthly_returns.empty:
        mr = monthly_returns.dropna()
        win_rate = float((mr > 0).mean())
        best_month = float(mr.max())
        worst_month = float(mr.min())
    else:
        monthly = (1 + pnl).resample("ME").prod() - 1
        win_rate = float((monthly > 0).mean())
        best_month = float(monthly.max()) if not monthly.empty else 0.0
        worst_month = float(monthly.min()) if not monthly.empty else 0.0

    # Trade stats
    closed = [t for t in (trades or []) if not t.is_open]
    n_trades = len(closed)
    avg_hold = float(np.mean([t.holding_days for t in closed])) if closed else 0.0
    profits = [t.pnl_pct for t in closed if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in closed if t.pnl_pct <= 0]
    gross_profit = sum(profits)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    if profit_factor == float("inf"):
        profit_factor = 99.0

    return PerformanceMetrics(
        universe=universe,
        start_date=str(start.date()),
        end_date=str(end.date()),
        total_return=total_ret,
        annual_return=annual_ret,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        volatility=vol,
        calmar_ratio=calmar,
        win_rate_monthly=win_rate,
        n_trades=n_trades,
        avg_holding_days=avg_hold,
        profit_factor=profit_factor,
        best_month=best_month,
        worst_month=worst_month,
    )


def _empty_metrics(universe: str) -> PerformanceMetrics:
    today = str(date.today())
    return PerformanceMetrics(
        universe=universe, start_date=today, end_date=today,
        total_return=0.0, annual_return=0.0, sharpe_ratio=0.0,
        max_drawdown=0.0, volatility=0.0, calmar_ratio=0.0,
        win_rate_monthly=0.0, n_trades=0, avg_holding_days=0.0,
        profit_factor=0.0, best_month=0.0, worst_month=0.0,
    )


# ── From BacktestResult ───────────────────────────────────────────────────────

def compute_all_metrics(
    result: BacktestResult,
    risk_free_annual: float = 0.0,
) -> Dict[str, PerformanceMetrics]:
    """
    Compute metrics for every commodity column + combined.

    Returns dict: {commodity_name: PerformanceMetrics}
    """
    metrics: Dict[str, PerformanceMetrics] = {}

    commodity_trades: Dict[str, List[Trade]] = {}
    for t in result.trades:
        commodity_trades.setdefault(t.commodity, []).append(t)

    for col in result.daily_pnl.columns:
        pnl_series = result.daily_pnl[col]
        monthly_series = result.monthly_returns[col] if col in result.monthly_returns.columns else None
        trades_col = commodity_trades.get(col, result.trades if col == "combined" else [])
        metrics[col] = compute_metrics(
            daily_pnl=pnl_series,
            universe=col,
            trades=trades_col,
            monthly_returns=monthly_series,
            risk_free_annual=risk_free_annual,
        )

    return metrics


# ── Rolling Sharpe ────────────────────────────────────────────────────────────

def rolling_sharpe(daily_pnl: pd.DataFrame, window_days: int = 252) -> pd.DataFrame:
    """Compute rolling annualised Sharpe ratio for each column."""
    roll_mean = daily_pnl.rolling(window_days).mean() * 252
    roll_std = daily_pnl.rolling(window_days).std() * np.sqrt(252)
    return (roll_mean / roll_std.replace(0, np.nan)).dropna(how="all")
