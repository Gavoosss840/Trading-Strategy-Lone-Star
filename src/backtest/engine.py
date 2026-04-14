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
  5. Rebalancing (every N days):
       - Re-evaluate every open position's alpha signal
       - Adjust size up/down in-place (NO close + reopen)
       - Cancel and replace TP/SL from current price + residual alpha
       - Close if: signal gone, alpha < threshold, or direction reversed

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
    exit_reason: str = ""   # "take_profit" | "stop_loss" | "expired" | "rebalance_*" | "open"

    # Alpha-linked exit levels — cancelled and replaced at each rebalancing
    alpha_at_entry: float = 0.0              # |alpha| at last sizing event
    take_profit_price: Optional[float] = None  # α=0 fair-value target (current price basis)
    stop_loss_price: Optional[float] = None    # trailing max-drawdown guard

    # Rebalancing tracking
    rebalance_count: int = 0
    last_rebalance_date: Optional[pd.Timestamp] = None

    # Daily P&L tracking — updated each mark-to-market step.
    # Initialised to entry_price (slippage-adjusted) so day-1 P&L correctly
    # reflects the entry spread cost.
    last_price: Optional[float] = None

    # Original shock preserved across rebalancings so the expected-alpha
    # calculation always uses the shock that triggered the position, not
    # the most recent commodity return.
    original_shock_return: float = 0.0
    shock_date: Optional[pd.Timestamp] = None

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
    EWMA beta: β = ewm_cov(Y, X) / ewm_var(X).

    Exponentially-weighted covariance/variance gives more weight to recent
    observations, making the beta estimate more responsive to regime changes
    than a flat rolling window.  span=window keeps the effective memory
    comparable to a simple rolling window of the same length.
    """
    aligned = pd.concat([stock_ret, commod_ret], axis=1).dropna()
    min_periods = max(window // 4, 30)
    if aligned.empty or len(aligned) < min_periods:
        return pd.Series(dtype=float)
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    roll_cov = y.ewm(span=window, min_periods=min_periods).cov(x)
    roll_var = x.ewm(span=window, min_periods=min_periods).var()
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


# ── Rebalancing ────────────────────────────────────────────────────────────────

def _rebalance_portfolio(
    portfolio: "Portfolio",
    current_date: pd.Timestamp,
    prices: pd.Series,
    betas: Dict[Tuple[str, str], pd.Series],
    residuals: Dict[str, pd.Series],
    commodity_returns: pd.DataFrame,
    commod_vol: pd.DataFrame,
    risk_cfg: dict,
    min_alpha: float,
    max_alpha: float,
    min_amp: float,
    min_z: float,
    stop_loss_pct: float,
    max_age: int,
    ar_ratio: float,
) -> None:
    """
    Re-evaluate every open position and adjust in-place.

    Rules (applied in order):
      1. No price data              → skip (keep position unchanged)
      2. Signal reversed direction  → close  (reason: rebalance_direction_reversed)
      3. Shock faded (amp < min)    → close  (reason: rebalance_shock_faded)
      4. Alpha below threshold      → close  (reason: rebalance_alpha_expired)
      5. Already fully priced       → close  (reason: rebalance_already_priced)
      6. Otherwise                  → adjust size up/down IN-PLACE
                                      cancel + replace TP/SL from current price

    TP/SL replacement logic:
      - TP = current_price × (1 ± residual_alpha)   [remaining gap to fair value]
      - SL = current_price × (1 ∓ stop_loss_pct)    [trailing stop protects gains]

    IMPORTANT: positions are NEVER closed then re-opened — only size is mutated.
    """
    max_pos    = risk_cfg.get("max_position_pct", 0.05)
    target_risk = risk_cfg.get("target_position_risk_pct", 0.01)
    ref_sr     = risk_cfg.get("reference_sharpe", 0.50)
    max_boost  = risk_cfg.get("max_signal_boost", 2.0)

    slippage_pct = portfolio.slippage_pct
    still_open: List[Trade] = []

    for trade in portfolio.open_trades:
        # ── 1. No price → keep as-is ────────────────────────────────────────
        if trade.ticker not in prices or np.isnan(prices[trade.ticker]):
            still_open.append(trade)
            continue

        current_price = prices[trade.ticker]
        commod = trade.commodity

        # ── Re-compute beta at current date ─────────────────────────────────
        beta_series = betas.get((trade.ticker, commod))
        if beta_series is None or current_date not in beta_series.index:
            still_open.append(trade)
            continue
        beta_val = beta_series.loc[current_date]
        if np.isnan(beta_val) or beta_val == 0:
            still_open.append(trade)
            continue

        # ── Use ORIGINAL shock — not the most recent commodity return ────────
        # The trade was opened because of a specific shock event.  Using the
        # latest daily commodity return as "shock_ret" during rebalancing would
        # randomly flip direction and corrupt the alpha calculation.
        shock_ret = trade.original_shock_return
        shock_amp = abs(shock_ret)

        # Age check: close if original shock is too stale (max_age × 2 days)
        days_since_shock = (
            (current_date - trade.shock_date).days
            if trade.shock_date is not None
            else max_age * 2 + 1
        )
        if days_since_shock > max_age * 2:
            eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
            trade.close(current_date, eff, "rebalance_shock_faded")
            portfolio.closed_trades.append(trade)
            continue

        # ── 3. Original shock too small → close ─────────────────────────────
        if shock_amp < min_amp:
            eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
            trade.close(current_date, eff, "rebalance_shock_faded")
            portfolio.closed_trades.append(trade)
            continue

        # ── Re-compute alpha: cumulative residual since position entry ────────
        resid_series = residuals.get(trade.ticker)
        if resid_series is None:
            still_open.append(trade)
            continue

        # Residuals accumulated since entry (the "actual reaction" since shock)
        resid_since_entry = resid_series.loc[
            (resid_series.index >= trade.entry_date) &
            (resid_series.index <= current_date)
        ]
        actual_reaction = float(resid_since_entry.sum()) if len(resid_since_entry) > 0 else 0.0

        expected   = beta_val * shock_ret
        new_alpha  = expected - actual_reaction
        abs_alpha  = abs(new_alpha)
        new_dir    = "LONG" if new_alpha > 0 else "SHORT"

        # ── 2. Direction reversed → close ────────────────────────────────────
        if new_dir != trade.direction:
            eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
            trade.close(current_date, eff, "rebalance_direction_reversed")
            portfolio.closed_trades.append(trade)
            continue

        # ── 4. Alpha below threshold → close ─────────────────────────────────
        if abs_alpha < min_alpha:
            eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
            trade.close(current_date, eff, "rebalance_alpha_expired")
            portfolio.closed_trades.append(trade)
            continue

        # ── 5. Already fully priced → close ──────────────────────────────────
        if expected != 0 and actual_reaction / expected >= ar_ratio:
            eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
            trade.close(current_date, eff, "rebalance_already_priced")
            portfolio.closed_trades.append(trade)
            continue

        # ── 6. Adjust size + cancel/replace TP/SL ────────────────────────────

        # New sizing (same CML formula as entry)
        resid_s = residuals.get(trade.ticker)
        if resid_s is not None:
            recent_r = resid_s.loc[:current_date].iloc[-63:]
            sigma_annual = float(recent_r.std() * np.sqrt(252)) if len(recent_r) > 5 else 0.25
        else:
            sigma_annual = 0.25
        sigma_annual = max(sigma_annual, 0.05)
        sigma_hold   = sigma_annual * np.sqrt(max_age / 252)
        sigma_hold   = max(sigma_hold, 1e-4)

        sharpe_signal = abs_alpha / sigma_hold
        base_size     = float(np.clip(target_risk / sigma_annual, 0.0, max_pos))
        boost         = float(np.clip(sharpe_signal / ref_sr, 0.0, max_boost))

        vol_today = commod_vol.loc[current_date, commod] if current_date in commod_vol.index else np.nan
        if np.isnan(vol_today) or vol_today == 0:
            gate = 0.5
        else:
            shock_z = shock_amp / vol_today
            if shock_z < min_z:
                # z-score too low → close
                eff = current_price * (1.0 - slippage_pct) if trade.direction == "LONG" else current_price * (1.0 + slippage_pct)
                trade.close(current_date, eff, "rebalance_shock_faded")
                portfolio.closed_trades.append(trade)
                continue
            gate = float(np.clip((shock_amp / 0.05) / 2 * np.tanh(shock_z / 2), 0, 1))

        new_size = float(np.clip(base_size * boost * gate, 0.005, max_pos))

        # ── Cancel old TP/SL and replace from current price ──────────────────
        # TP  = current_price × (1 ± residual_alpha)   remaining gap to fair value
        # SL  = current_price × (1 ∓ stop_loss_pct)    trailing stop (protects gains)
        if trade.direction == "LONG":
            new_tp = current_price * (1.0 + abs_alpha)
            new_sl = current_price * (1.0 - stop_loss_pct)
        else:
            new_tp = current_price * (1.0 - abs_alpha)
            new_sl = current_price * (1.0 + stop_loss_pct)

        # Mutate in-place — no close/reopen
        trade.size               = new_size
        trade.alpha_at_entry     = abs_alpha
        trade.take_profit_price  = new_tp
        trade.stop_loss_price    = new_sl
        trade.rebalance_count   += 1
        trade.last_rebalance_date = current_date

        still_open.append(trade)

    portfolio.open_trades = still_open


# ── Portfolio ──────────────────────────────────────────────────────────────────

class Portfolio:
    """Tracks open positions and computes daily mark-to-market P&L."""

    def __init__(
        self,
        stop_loss_pct: float,
        max_holding: int,
        slippage_pct: float = 0.001,
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.max_holding = max_holding
        self.slippage_pct = slippage_pct
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
        prices: pd.Series,
    ) -> Dict[str, float]:
        """
        Mark-to-market all open positions.

        Exit logic:
          - Take Profit : price reaches α=0 fair-value (trade.take_profit_price)
          - Stop Loss   : price crosses trailing max-drawdown threshold
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

            # Basis for today's incremental P&L.
            # last_price is set to the slippage-adjusted entry price on day 1,
            # so the entry spread cost is captured in the first day's return.
            prev_price = trade.last_price if trade.last_price is not None else trade.entry_price

            hit_stop = False
            hit_tp   = False

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

            if hit_stop or hit_tp or days_held >= self.max_holding:
                # Apply exit slippage: sell below market (LONG) / buy above market (SHORT)
                if trade.direction == "LONG":
                    exit_price = current_price * (1.0 - self.slippage_pct)
                else:
                    exit_price = current_price * (1.0 + self.slippage_pct)

                # Record daily P&L from last mark-to-market to slippage-adjusted exit
                raw_daily = exit_price / prev_price - 1
                signed_daily = raw_daily if trade.direction == "LONG" else -raw_daily
                c = trade.commodity
                daily_pnl[c] = daily_pnl.get(c, 0.0) + trade.size * signed_daily

                reason = "stop_loss" if hit_stop else ("take_profit" if hit_tp else "expired")
                trade.close(current_date, exit_price, reason)
                self.closed_trades.append(trade)
            else:
                # Still open — daily return from last_price to today's close (no slippage)
                raw_daily = current_price / prev_price - 1
                signed_daily = raw_daily if trade.direction == "LONG" else -raw_daily
                c = trade.commodity
                daily_pnl[c] = daily_pnl.get(c, 0.0) + trade.size * signed_daily

                trade.last_price = current_price
                still_open.append(trade)

        self.open_trades = still_open
        return daily_pnl

    def force_close_all(self, current_date: pd.Timestamp, prices: pd.Series) -> None:
        for trade in self.open_trades:
            if trade.ticker in prices and not np.isnan(prices[trade.ticker]):
                mkt = prices[trade.ticker]
                if trade.direction == "LONG":
                    exit_price = mkt * (1.0 - self.slippage_pct)
                else:
                    exit_price = mkt * (1.0 + self.slippage_pct)
                trade.close(current_date, exit_price, "end_of_backtest")
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
    stock_volumes: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Run walk-forward backtest of the Lone Star strategy.

    Args:
        stock_returns    : DataFrame [dates × tickers]
        stock_prices     : DataFrame [dates × tickers] (for P&L)
        commodity_returns: DataFrame [dates × commodities]
        market_returns   : Series of market returns (for CAPM residuals)
        stock_metadata   : DataFrame(ticker → zone, primary_commodity, role)
        cfg              : Full strategy config dict
        step_days        : Re-generate NEW signals every N days (1=daily)
    """
    beta_cfg      = cfg["beta"]
    shock_cfg     = cfg["shock"]
    alpha_cfg     = cfg["alpha"]
    risk_cfg      = cfg["risk"]
    execution_cfg = cfg.get("execution", {})

    lookback    = beta_cfg.get("rolling_window", 252)
    max_age     = shock_cfg.get("max_age_days", 5)
    min_amp     = shock_cfg.get("min_amplitude_pct", 1.5) / 100
    min_z       = shock_cfg.get("min_zscore", 1.5)
    vol_window  = shock_cfg.get("lookback_vol_days", 63)
    min_alpha   = alpha_cfg.get("min_alpha_pct", 0.5) / 100
    max_alpha   = alpha_cfg.get("max_alpha_pct", 15.0) / 100
    stop_loss   = risk_cfg.get("stop_loss_pct", 0.05)
    max_hold    = risk_cfg.get("max_holding_days", 10)
    max_pos     = risk_cfg.get("max_position_pct", 0.05)
    ar_ratio    = alpha_cfg.get("already_priced_ratio", 0.85)
    rebal_freq  = risk_cfg.get("rebalancing_frequency_days", 2)
    max_commod_exp = risk_cfg.get("max_commodity_exposure", 0.20)

    slippage_pct  = execution_cfg.get("slippage_pct", 0.001)
    nav_total     = execution_cfg.get("nav_total", 10_000)
    adv_filter_pct = execution_cfg.get("adv_filter_pct", 0.05)

    commodities = list(commodity_returns.columns)

    # ── Pre-compute rolling betas (vectorized) ─────────────────────────────
    print("  [BT] Pre-computing rolling betas...", end=" ", flush=True)
    betas: Dict[Tuple[str, str], pd.Series] = {}
    residuals: Dict[str, pd.Series] = {}

    stocks_available = [t for t in stock_returns.columns
                        if t in stock_prices.columns]

    for ticker in stocks_available:
        s_ret = stock_returns[ticker].dropna()
        resid = _capm_residuals(s_ret, market_returns.reindex(s_ret.index).fillna(0), lookback)
        residuals[ticker] = resid
        for commod in commodities:
            c_ret = commodity_returns[commod].dropna()
            b = _rolling_beta(s_ret, c_ret, lookback)
            betas[(ticker, commod)] = b

    print(f"✓ ({len(betas)} pairs)")

    # ── Pre-compute rolling commodity vol (for shock z-score) ─────────────
    commod_vol = commodity_returns.rolling(vol_window).std()

    # ── Simulation dates ───────────────────────────────────────────────────
    common_dates = stock_returns.index.intersection(commodity_returns.index)
    sim_dates    = common_dates[lookback:]

    portfolio = Portfolio(stop_loss, max_hold, slippage_pct)

    all_cols       = commodities + ["combined"]
    daily_pnl_rows = []

    print(f"  [BT] Simulating {len(sim_dates)} days "
          f"(rebalance every {rebal_freq}d)...", end=" ", flush=True)

    for i, t in enumerate(sim_dates):
        prices_today = stock_prices.loc[t] if t in stock_prices.index else pd.Series(dtype=float)

        # ── 1. Rebalance open positions (every rebal_freq days) ───────────
        if i > 0 and i % rebal_freq == 0 and portfolio.open_trades:
            _rebalance_portfolio(
                portfolio       = portfolio,
                current_date    = t,
                prices          = prices_today,
                betas           = betas,
                residuals       = residuals,
                commodity_returns = commodity_returns,
                commod_vol      = commod_vol,
                risk_cfg        = risk_cfg,
                min_alpha       = min_alpha,
                max_alpha       = max_alpha,
                min_amp         = min_amp,
                min_z           = min_z,
                stop_loss_pct   = stop_loss,
                max_age         = max_age,
                ar_ratio        = ar_ratio,
            )

        # ── 2. Mark-to-market (TP/SL/expiry checks) ──────────────────────
        pnl_today = portfolio.update(t, prices_today)

        # ── 3. Generate NEW signals every step_days ───────────────────────
        if i % step_days == 0:
            recent_commod = commodity_returns.loc[:t].iloc[-(max_age + 1):]

            for commod in commodities:
                c_series = recent_commod[commod].dropna()
                if len(c_series) < 2:
                    continue

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

                # ── Cross-commodity: consider ALL stocks, not just primary ones ──
                # The beta dict already covers every (ticker, commodity) pair with
                # sufficient history.  Filtering by primary_commodity would miss
                # e.g. a nat-gas producer that also carries significant crude beta.
                for ticker in stocks_available:
                    beta_series = betas.get((ticker, commod))
                    if beta_series is None or t not in beta_series.index:
                        continue
                    beta_val = beta_series.loc[t]
                    if np.isnan(beta_val) or beta_val == 0:
                        continue

                    resid_series = residuals.get(ticker)
                    if resid_series is None:
                        continue
                    recent_resid    = resid_series.loc[:t].iloc[-max_age:]
                    actual_reaction = float(recent_resid.sum())

                    expected = beta_val * shock_ret
                    alpha    = expected - actual_reaction

                    if abs(alpha) < min_alpha or abs(alpha) > max_alpha:
                        continue
                    if expected != 0 and actual_reaction / expected >= ar_ratio:
                        continue

                    direction = "LONG" if alpha > 0 else "SHORT"
                    abs_alpha = abs(alpha)

                    # CML sizing
                    resid_s = residuals.get(ticker)
                    if resid_s is not None:
                        recent_r     = resid_s.loc[:t].iloc[-63:]
                        sigma_annual = float(recent_r.std() * np.sqrt(252)) if len(recent_r) > 5 else 0.25
                    else:
                        sigma_annual = 0.25

                    sigma_annual  = max(sigma_annual, 0.05)
                    sigma_hold    = sigma_annual * np.sqrt(max_age / 252)
                    sigma_hold    = max(sigma_hold, 1e-4)
                    sharpe_signal = abs_alpha / sigma_hold

                    target_risk = risk_cfg.get("target_position_risk_pct", 0.01)
                    base_size   = float(np.clip(target_risk / sigma_annual, 0.0, max_pos))
                    ref_sr      = risk_cfg.get("reference_sharpe", 0.50)
                    max_boost   = risk_cfg.get("max_signal_boost", 2.0)
                    boost       = float(np.clip(sharpe_signal / ref_sr, 0.0, max_boost))
                    gate        = float(np.clip(shock_score, 0.0, 1.0))
                    size        = float(np.clip(base_size * boost * gate, 0.005, max_pos))

                    if t not in stock_prices.index or ticker not in stock_prices.columns:
                        continue
                    entry_price = stock_prices.loc[t, ticker]
                    if np.isnan(entry_price) or entry_price <= 0:
                        continue

                    # ── ADV liquidity filter ─────────────────────────────────
                    if stock_volumes is not None and ticker in stock_volumes.columns:
                        try:
                            t_pos = stock_volumes.index.get_loc(t)
                        except KeyError:
                            t_pos = -1
                        if t_pos >= 20:
                            avg_vol = float(
                                stock_volumes.iloc[max(0, t_pos - 20): t_pos][ticker].mean()
                            )
                            adv_usd = avg_vol * entry_price
                            if adv_usd > 0:
                                position_notional = size * nav_total
                                if position_notional > adv_filter_pct * adv_usd:
                                    continue  # Position too large vs ADV

                    # ── Commodity exposure cap ───────────────────────────────
                    # Prevent over-concentration in a single commodity.
                    current_commod_exp = sum(
                        tr.size for tr in portfolio.open_trades if tr.commodity == commod
                    )
                    if current_commod_exp + size > max_commod_exp:
                        available = max_commod_exp - current_commod_exp
                        if available < risk_cfg.get("min_position_pct", 0.005):
                            continue  # No room even for the minimum position
                        size = available  # Trim to fit under cap

                    # ── Entry slippage ───────────────────────────────────────
                    # The effective price paid includes half-spread cost.
                    if direction == "LONG":
                        effective_entry = entry_price * (1.0 + slippage_pct)
                    else:
                        effective_entry = entry_price * (1.0 - slippage_pct)

                    # TP / SL anchored on effective entry price
                    if direction == "LONG":
                        tp_price = effective_entry * (1.0 + abs_alpha)
                        sl_price = effective_entry * (1.0 - stop_loss)
                    else:
                        tp_price = effective_entry * (1.0 - abs_alpha)
                        sl_price = effective_entry * (1.0 + stop_loss)

                    # Role determined by beta sign (correct for cross-commodity)
                    role = "producer" if beta_val > 0 else "consumer"

                    trade = Trade(
                        ticker               = ticker,
                        commodity            = commod,
                        direction            = direction,
                        role                 = role,
                        entry_date           = t,
                        entry_price          = effective_entry,
                        size                 = size,
                        alpha_at_entry       = abs_alpha,
                        take_profit_price    = tp_price,
                        stop_loss_price      = sl_price,
                        last_price           = effective_entry,   # daily P&L basis
                        original_shock_return = shock_ret,        # preserved for rebalancing
                        shock_date           = t,
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
            daily_pnl      = empty_df,
            monthly_returns = empty_df,
            trades         = [],
            nav_weights    = {},
            start_date     = sim_dates[0] if len(sim_dates) else pd.Timestamp.today(),
            end_date       = sim_dates[-1] if len(sim_dates) else pd.Timestamp.today(),
        )

    pnl_df = pd.DataFrame(daily_pnl_rows).set_index("date")[all_cols]

    monthly = (1 + pnl_df).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    trade_counts = {}
    for tr in portfolio.all_trades:
        trade_counts[tr.commodity] = trade_counts.get(tr.commodity, 0) + 1
    total_trades = max(sum(trade_counts.values()), 1)
    nav_weights  = {c: trade_counts.get(c, 0) / total_trades for c in commodities}

    return BacktestResult(
        daily_pnl       = pnl_df,
        monthly_returns = monthly,
        trades          = portfolio.all_trades,
        nav_weights     = nav_weights,
        start_date      = sim_dates[0],
        end_date        = sim_dates[-1],
    )
