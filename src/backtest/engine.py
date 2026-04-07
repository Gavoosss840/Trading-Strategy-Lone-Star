"""
Walk-Forward Backtest Engine
─────────────────────────────
Simulates the Lone Star pipeline historically using a vectorized
rolling-window approach.

For each trading day t (from day `lookback` onward):
  1. Rolling commodity beta  : β = cov(R_stock, R_commod) / var(R_commod)
                               computed over [t-lookback : t]
  2. Shock detection         : scan commodity returns [t-max_age : t]
  3. Alpha signal            : Expected = β × shock_return
                               Alpha    = Expected − cumulative residual
  4. Portfolio management    : open/close positions, mark-to-market daily

Output:
  BacktestResult with daily P&L per commodity, closed trades list,
  and NAV allocation weights.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    commodity: str
    direction: str          # "LONG" | "SHORT"
    role: str               # "producer" | "consumer"
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 1.0       # fraction of portfolio (e.g. 0.03)
    pnl_pct: float = 0.0
    exit_reason: str = ""   # "take_profit" | "stop_loss" | "expired" | "open"

    # Alpha-linked exit levels (set at entry)
    alpha_at_entry: float = 0.0              # |alpha| used to compute fair-value TP
    take_profit_price: Optional[float] = None  # price at which alpha = 0 (fully priced)
    stop_loss_price: Optional[float] = None    # price at max drawdown % from entry

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def close(self, exit_date: pd.Timestamp, exit_price: float, reason: str) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        raw = (exit_price / self.entry_price - 1)
        self.pnl_pct = raw if self.direction == "LONG" else -raw

    @property
    def holding_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


# ── Backtest result ────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    daily_pnl: pd.DataFrame            # index=date, cols=commodity + "combined"
    monthly_returns: pd.DataFrame      # index=YearMonth, cols=commodity + "combined"
    trades: List[Trade]
    nav_weights: Dict[str, float]      # commodity → % NAV allocated
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @property
    def commodities(self) -> List[str]:
        return [c for c in self.daily_pnl.columns if c != "combined"]

    def equity_curve(self) -> pd.DataFrame:
        """Cumulative return from 1.0 base for each column."""
        return (1 + self.daily_pnl).cumprod()

    def drawdown(self) -> pd.DataFrame:
        eq = self.equity_curve()
        rolling_max = eq.cummax()
        return (eq - rolling_max) / rolling_max


# ── Rolling beta helper ────────────────────────────────────────────────────────

def _rolling_beta(
    stock_ret: pd.Series,
    commod_ret: pd.Series,
    window: int,
) -> pd.Series:
    """
    Vectorized rolling beta: β = cov(Y, X) / var(X).
    Much faster than rolling OLS for backtesting.
    """
    aligned = pd.concat([stock_ret, commod_ret], axis=1).dropna()
    if aligned.empty or len(aligned) < window:
        return pd.Series(dtype=float)
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    roll_cov = y.rolling(window).cov(x)
    roll_var = x.rolling(window).var()
    return (roll_cov / roll_var.replace(0, np.nan)).rename(
        f"beta_{stock_ret.name}_{commod_ret.name}"
    )


def _capm_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling CAPM residuals for noise cleaning."""
    aligned = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if aligned.empty:
        return stock_ret
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    roll_beta = y.rolling(window).cov(x) / x.rolling(window).var().replace(0, np.nan)
    roll_alpha = y.rolling(window).mean() - roll_beta * x.rolling(window).mean()
    fitted = roll_alpha + roll_beta * x
    return (y - fitted).rename(stock_ret.name)


# ── Portfolio ──────────────────────────────────────────────────────────────────

class Portfolio:
    """Tracks open positions and computes daily mark-to-market P&L."""

    def __init__(self, stop_loss_pct: float, max_holding: int) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.max_holding = max_holding
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

    def open_position(self, trade: Trade) -> None:
        # One position per (ticker, commodity) at a time
        existing = [t for t in self.open_trades
                    if t.ticker == trade.ticker and t.commodity == trade.commodity]
        if not existing:
            self.open_trades.append(trade)

    def update(
        self,
        current_date: pd.Timestamp,
        prices: pd.Series,       # ticker → current price
    ) -> Dict[str, float]:
        """
        Mark-to-market all open positions.

        Exit logic:
          - Take Profit : price reaches α=0 fair-value (trade.take_profit_price)
          - Stop Loss   : price crosses max-drawdown threshold (trade.stop_loss_price)
          - Expiry      : held for max_holding days

        Returns dict of commodity → daily P&L contribution.
        """
        daily_pnl: Dict[str, float] = {}
        still_open = []

        for trade in self.open_trades:
            if trade.ticker not in prices or np.isnan(prices[trade.ticker]):
                still_open.append(trade)
                continue

            current_price = prices[trade.ticker]
            days_held = (current_date - trade.entry_date).days

            # Check exit conditions using absolute price levels
            hit_stop = False
            hit_tp = False

            if trade.stop_loss_price is not None:
                if trade.direction == "LONG":
                    hit_stop = current_price <= trade.stop_loss_price
                else:
                    hit_stop = current_price >= trade.stop_loss_price

            if trade.take_profit_price is not None:
                if trade.direction == "LONG":
                    hit_tp = current_price >= trade.take_profit_price
                else:
                    hit_tp = current_price <= trade.take_profit_price

            if hit_stop:
                trade.close(current_date, current_price, "stop_loss")
                self.closed_trades.append(trade)
            elif hit_tp:
                trade.close(current_date, current_price, "take_profit")
                self.closed_trades.append(trade)
            elif days_held >= self.max_holding:
                trade.close(current_date, current_price, "expired")
                self.closed_trades.append(trade)
            else:
                still_open.append(trade)
                # Daily contribution: size × signed daily stock return
                raw_ret = current_price / trade.entry_price - 1
                signed_ret = raw_ret if trade.direction == "LONG" else -raw_ret
                c = trade.commodity
                daily_pnl[c] = daily_pnl.get(c, 0.0) + trade.size * signed_ret

        self.open_trades = still_open
        return daily_pnl

    def force_close_all(self, current_date: pd.Timestamp, prices: pd.Series) -> None:
        for trade in self.open_trades:
            if trade.ticker in prices and not np.isnan(prices[trade.ticker]):
                trade.close(current_date, prices[trade.ticker], "end_of_backtest")
                self.closed_trades.append(trade)
        self.open_trades = []

    @property
    def all_trades(self) -> List[Trade]:
        return self.closed_trades + self.open_trades


# ── Main backtest runner ───────────────────────────────────────────────────────

def run_backtest(
    stock_returns: pd.DataFrame,
    stock_prices: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    market_returns: pd.Series,
    stock_metadata: pd.DataFrame,
    cfg: dict,
    step_days: int = 1,
) -> BacktestResult:
    """
    Run walk-forward backtest of the Lone Star strategy.

    Args:
        stock_returns   : DataFrame [dates × tickers]
        stock_prices    : DataFrame [dates × tickers] (for P&L)
        commodity_returns: DataFrame [dates × commodities]
        market_returns  : Series of market returns (for CAPM residuals)
        stock_metadata  : DataFrame(ticker → zone, primary_commodity, role)
        cfg             : Full strategy config dict
        step_days       : Re-generate signals every N days (1=daily, 5=weekly)
    """
    beta_cfg   = cfg["beta"]
    shock_cfg  = cfg["shock"]
    alpha_cfg  = cfg["alpha"]
    risk_cfg   = cfg["risk"]

    lookback   = beta_cfg.get("rolling_window", 252)
    min_r2_thr = beta_cfg.get("min_r_squared", 0.05)
    max_age    = shock_cfg.get("max_age_days", 5)
    min_amp    = shock_cfg.get("min_amplitude_pct", 1.5) / 100
    min_z      = shock_cfg.get("min_zscore", 1.5)
    vol_window = shock_cfg.get("lookback_vol_days", 63)
    min_alpha  = alpha_cfg.get("min_alpha_pct", 0.5) / 100
    max_alpha  = alpha_cfg.get("max_alpha_pct", 15.0) / 100
    stop_loss  = risk_cfg.get("stop_loss_pct", 0.05)
    max_hold   = risk_cfg.get("max_holding_days", 10)
    max_pos    = risk_cfg.get("max_position_pct", 0.05)
    ar_ratio   = alpha_cfg.get("already_priced_ratio", 0.85)

    commodities = list(commodity_returns.columns)

    # ── Pre-compute rolling betas (vectorized) ─────────────────────────────
    print("  [BT] Pre-computing rolling betas...", end=" ", flush=True)
    betas: Dict[Tuple[str, str], pd.Series] = {}
    residuals: Dict[str, pd.Series] = {}

    stocks_available = [t for t in stock_returns.columns
                        if t in stock_prices.columns]

    for ticker in stocks_available:
        s_ret = stock_returns[ticker].dropna()
        # CAPM residuals
        resid = _capm_residuals(s_ret, market_returns.reindex(s_ret.index).fillna(0), lookback)
        residuals[ticker] = resid
        # Commodity betas
        for commod in commodities:
            c_ret = commodity_returns[commod].dropna()
            b = _rolling_beta(s_ret, c_ret, lookback)
            betas[(ticker, commod)] = b

    print(f"✓ ({len(betas)} pairs)")

    # ── Pre-compute rolling commodity vol (for shock z-score) ─────────────
    commod_vol = commodity_returns.rolling(vol_window).std()

    # ── Simulation dates ───────────────────────────────────────────────────
    common_dates = stock_returns.index.intersection(commodity_returns.index)
    sim_dates = common_dates[lookback:]

    portfolio = Portfolio(stop_loss, max_hold)

    # daily P&L per commodity
    all_cols = commodities + ["combined"]
    daily_pnl_rows = []

    print(f"  [BT] Simulating {len(sim_dates)} days...", end=" ", flush=True)

    for i, t in enumerate(sim_dates):
        prices_today = stock_prices.loc[t] if t in stock_prices.index else pd.Series(dtype=float)

        # Mark-to-market existing positions
        pnl_today = portfolio.update(t, prices_today)

        # Generate new signals every `step_days`
        if i % step_days == 0:
            # Detect shocks in last max_age days
            recent_commod = commodity_returns.loc[:t].iloc[-(max_age + 1):]

            for commod in commodities:
                c_series = recent_commod[commod].dropna()
                if len(c_series) < 2:
                    continue

                # Most recent shock
                shock_ret = c_series.iloc[-1]
                shock_amp = abs(shock_ret)
                if shock_amp < min_amp:
                    continue

                vol_today = commod_vol.loc[t, commod] if t in commod_vol.index else np.nan
                if np.isnan(vol_today) or vol_today == 0:
                    continue
                shock_z = shock_amp / vol_today
                if shock_z < min_z:
                    continue

                shock_score = float(np.clip(
                    (shock_amp / 0.05) / 2 * np.tanh(shock_z / 2), 0, 1
                ))

                # For each stock with primary_commodity == commod
                if stock_metadata.empty:
                    continue
                relevant = stock_metadata[
                    stock_metadata.get("primary_commodity", pd.Series()) == commod
                ] if "primary_commodity" in stock_metadata.columns else pd.DataFrame()

                for ticker in relevant.index:
                    if ticker not in stocks_available:
                        continue

                    # Get rolling beta at time t
                    beta_series = betas.get((ticker, commod))
                    if beta_series is None or t not in beta_series.index:
                        continue
                    beta_val = beta_series.loc[t]
                    if np.isnan(beta_val) or beta_val == 0:
                        continue

                    # Cumulative residual (last max_age days)
                    resid_series = residuals.get(ticker)
                    if resid_series is None:
                        continue
                    recent_resid = resid_series.loc[:t].iloc[-max_age:]
                    actual_reaction = float(recent_resid.sum())

                    # Alpha
                    expected = beta_val * shock_ret
                    alpha = expected - actual_reaction

                    # Filters
                    if abs(alpha) < min_alpha or abs(alpha) > max_alpha:
                        continue
                    if (expected != 0 and
                            actual_reaction / expected >= ar_ratio):
                        continue

                    direction = "LONG" if alpha > 0 else "SHORT"

                    # ── CML + risk-parity sizing ─────────────────────────────
                    # σ_résiduel: rolling 63-day std of CAPM residuals (annualised)
                    resid_s = residuals.get(ticker)
                    if resid_s is not None:
                        recent_r = resid_s.loc[:t].iloc[-63:]
                        sigma_annual = float(recent_r.std() * np.sqrt(252)) if len(recent_r) > 5 else 0.25
                    else:
                        sigma_annual = 0.25  # 25% default if no data

                    sigma_annual = max(sigma_annual, 0.05)  # floor at 5%

                    # σ over holding period
                    sigma_hold = sigma_annual * np.sqrt(max_age / 252)
                    sigma_hold = max(sigma_hold, 1e-4)

                    # Sharpe of the signal
                    sharpe_signal = abs(alpha) / sigma_hold

                    # Base size: target 1% annual idiosyncratic risk per position
                    target_risk = risk_cfg.get("target_position_risk_pct", 0.01)
                    base_size = float(np.clip(target_risk / sigma_annual, 0.0, max_pos))

                    # CML boost: signal quality vs reference SR
                    ref_sr   = risk_cfg.get("reference_sharpe", 0.50)
                    max_boost = risk_cfg.get("max_signal_boost", 2.0)
                    boost = float(np.clip(sharpe_signal / ref_sr, 0.0, max_boost))

                    # Shock quality score [0, 1] as composite gate
                    gate = float(np.clip(shock_score, 0.0, 1.0))

                    size = float(np.clip(base_size * boost * gate, 0.005, max_pos))

                    if t not in stock_prices.index or ticker not in stock_prices.columns:
                        continue
                    entry_price = stock_prices.loc[t, ticker]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue

                    role = stock_metadata.loc[ticker, "role"] if ticker in stock_metadata.index else ""

                    # ── Alpha-linked exit levels ─────────────────────────────
                    # Take Profit: price at which the gap (α=0) is fully priced
                    #   LONG  → expect price to rise by |alpha|
                    #   SHORT → expect price to fall by |alpha|
                    abs_alpha = abs(alpha)
                    if direction == "LONG":
                        tp_price = entry_price * (1.0 + abs_alpha)
                        sl_price = entry_price * (1.0 - stop_loss)
                    else:
                        tp_price = entry_price * (1.0 - abs_alpha)
                        sl_price = entry_price * (1.0 + stop_loss)

                    trade = Trade(
                        ticker=ticker,
                        commodity=commod,
                        direction=direction,
                        role=role,
                        entry_date=t,
                        entry_price=entry_price,
                        size=size,
                        alpha_at_entry=abs_alpha,
                        take_profit_price=tp_price,
                        stop_loss_price=sl_price,
                    )
                    portfolio.open_position(trade)

        # Record daily P&L
        row = {c: pnl_today.get(c, 0.0) for c in commodities}
        row["combined"] = sum(row.values())
        row["date"] = t
        daily_pnl_rows.append(row)

    # Force-close remaining positions at end
    if sim_dates.size > 0:
        portfolio.force_close_all(sim_dates[-1], stock_prices.loc[sim_dates[-1]])

    print("✓")

    # ── Assemble results ───────────────────────────────────────────────────
    if not daily_pnl_rows:
        empty_df = pd.DataFrame(columns=all_cols)
        return BacktestResult(
            daily_pnl=empty_df,
            monthly_returns=empty_df,
            trades=[],
            nav_weights={},
            start_date=sim_dates[0] if len(sim_dates) else pd.Timestamp.today(),
            end_date=sim_dates[-1] if len(sim_dates) else pd.Timestamp.today(),
        )

    pnl_df = pd.DataFrame(daily_pnl_rows).set_index("date")[all_cols]

    # Monthly returns
    monthly = (1 + pnl_df).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    # NAV weights: proportion of total trades by commodity
    trade_counts = {}
    for t in portfolio.all_trades:
        trade_counts[t.commodity] = trade_counts.get(t.commodity, 0) + 1
    total_trades = max(sum(trade_counts.values()), 1)
    nav_weights = {c: trade_counts.get(c, 0) / total_trades for c in commodities}

    return BacktestResult(
        daily_pnl=pnl_df,
        monthly_returns=monthly,
        trades=portfolio.all_trades,
        nav_weights=nav_weights,
        start_date=sim_dates[0],
        end_date=sim_dates[-1],
    )
