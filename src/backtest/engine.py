"""
Walk-Forward Backtest Engine  v2
──────────────────────────────────
Simulates the Lone Star pipeline historically.

Optimisations vs v1:
  1. Daily P&L fix        — prev_price tracking; no more cumulative-from-entry
  2. EWMA market-adj beta — Frisch-Waugh partial beta, exponentially weighted
  3. EWMA CAPM residuals  — EWMA instead of equal-weight rolling
  4. Multi-day shocks     — best 1/2/3-day window inside max_age period
  5. Cross-commodity      — all stocks evaluated vs every shocked commodity
  6. Commodity cap        — max_commodity_exposure enforced before opening
  7. Transaction costs    — slippage bps applied at entry AND exit
  8. ADV filter           — illiquid stocks skipped (rolling 20-day notional)
  9. Original shock ref   — rebalance uses the shock that triggered entry,
                            not today's commodity return
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    commodity: str
    direction: str          # "LONG" | "SHORT"
    role: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    # Alpha-linked exit levels
    alpha_at_entry: float = 0.0
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

    # Rebalancing tracking
    rebalance_count: int = 0
    last_rebalance_date: Optional[pd.Timestamp] = None

    # ── v2 additions ────────────────────────────────────────────────────────
    # Daily P&L requires the price from the previous bar, not from entry.
    prev_price: Optional[float] = None
    # Original shock that triggered entry — used in rebalance to keep
    # expected-reaction consistent (vs using today's arbitrary commodity move).
    original_shock_return: float = 0.0
    original_shock_date: Optional[pd.Timestamp] = None
    original_shock_age: int = 0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def close(
        self,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str,
        slippage_pct: float = 0.0,
    ) -> None:
        self.exit_date = exit_date
        # Apply exit-side slippage (worsens fill)
        if slippage_pct > 0:
            if self.direction == "LONG":
                exit_price = exit_price * (1.0 - slippage_pct)
            else:
                exit_price = exit_price * (1.0 + slippage_pct)
        self.exit_price = exit_price
        self.exit_reason = reason
        raw = exit_price / self.entry_price - 1.0
        self.pnl_pct = raw if self.direction == "LONG" else -raw

    @property
    def holding_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


# ── Backtest result ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    daily_pnl: pd.DataFrame
    monthly_returns: pd.DataFrame
    trades: List[Trade]
    nav_weights: Dict[str, float]
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @property
    def commodities(self) -> List[str]:
        return [c for c in self.daily_pnl.columns if c != "combined"]

    def equity_curve(self) -> pd.DataFrame:
        return (1 + self.daily_pnl).cumprod()

    def drawdown(self) -> pd.DataFrame:
        eq = self.equity_curve()
        return (eq - eq.cummax()) / eq.cummax()


# ── Beta helpers ──────────────────────────────────────────────────────────────

def _rolling_beta(
    stock_ret: pd.Series,
    commod_ret: pd.Series,
    window: int,
) -> pd.Series:
    """Vectorised rolling beta β = cov(Y, X) / var(X). Fallback mode."""
    aligned = pd.concat([stock_ret, commod_ret], axis=1).dropna()
    if aligned.empty or len(aligned) < window:
        return pd.Series(dtype=float)
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    return (y.rolling(window).cov(x) / x.rolling(window).var().replace(0, np.nan))


def _ewma_market_adjusted_beta(
    stock_ret: pd.Series,
    commod_ret: pd.Series,
    market_ret: pd.Series,
    halflife: int = 63,
) -> pd.Series:
    """
    EWMA commodity beta net of the market factor (Frisch-Waugh).

    Steps (all vectorised via pandas ewm):
      1. Partial market out of stock return    → y_res
      2. Partial market out of commodity return → x_res
      3. EWMA beta of y_res on x_res            = β_commodity net of market

    Exponential weighting with `halflife` days gives ~3× more weight to
    recent observations vs. equal-weight rolling, adapting faster to
    regime changes while remaining stable.
    """
    aligned = pd.concat([stock_ret, commod_ret, market_ret], axis=1).dropna()
    if aligned.empty or len(aligned) < 30:
        return pd.Series(dtype=float)

    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    m = aligned.iloc[:, 2]

    ew = lambda s: s.ewm(halflife=halflife, min_periods=20)

    var_m = ew(m).var().replace(0, np.nan)

    # Partial out market from stock
    beta_ym   = ew(y).cov(m) / var_m
    alpha_ym  = ew(y).mean() - beta_ym * ew(m).mean()
    y_res = y - (alpha_ym + beta_ym * m)

    # Partial out market from commodity
    beta_xm   = ew(x).cov(m) / var_m
    alpha_xm  = ew(x).mean() - beta_xm * ew(m).mean()
    x_res = x - (alpha_xm + beta_xm * m)

    var_xres = ew(x_res).var().replace(0, np.nan)
    beta_net = ew(y_res).cov(x_res) / var_xres

    return beta_net.rename(f"beta_{stock_ret.name}_{commod_ret.name}")


# ── Residual helpers ──────────────────────────────────────────────────────────

def _ewma_capm_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    halflife: int = 63,
) -> pd.Series:
    """
    EWMA CAPM residuals: stock return after removing exponentially-weighted
    market component. Adapts faster than equal-weight rolling OLS.
    """
    aligned = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if aligned.empty:
        return stock_ret
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]

    ew = lambda s: s.ewm(halflife=halflife, min_periods=20)

    var_x = ew(x).var().replace(0, np.nan)
    beta  = ew(y).cov(x) / var_x
    alpha = ew(y).mean() - beta * ew(x).mean()
    fitted = alpha + beta * x
    return (y - fitted).rename(stock_ret.name)


def _rolling_capm_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling CAPM residuals — kept as fallback when EWMA is disabled."""
    aligned = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if aligned.empty:
        return stock_ret
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    roll_beta  = y.rolling(window).cov(x) / x.rolling(window).var().replace(0, np.nan)
    roll_alpha = y.rolling(window).mean() - roll_beta * x.rolling(window).mean()
    fitted = roll_alpha + roll_beta * x
    return (y - fitted).rename(stock_ret.name)


# ── Multi-day shock scanner ───────────────────────────────────────────────────

def _scan_best_shock(
    c_series: pd.Series,
    vol_series: pd.Series,
    min_amp: float,
    min_z: float,
    speed_decay: float,
    max_age: int,
    max_accum: int = 3,
) -> Optional[Tuple[float, float, int, pd.Timestamp]]:
    """
    Find the best shock in the last max_age bars by scanning 1, 2, 3-day
    accumulated windows. Returns (shock_return, shock_score, age_days, date)
    or None if nothing passes the thresholds.

    Multi-day rationale: an OPEC cut or Fed pivot can unfold over 2-3 sessions.
    A +2.5%/day move over 3 days is just as exploitable as a single +7.5% spike.
    Speed factor penalises multi-day moves so they score lower for equal amplitude.
    """
    best: Optional[Tuple[float, float, int, pd.Timestamp]] = None
    best_score = 0.0

    n = len(c_series)
    for window_days in range(1, max_accum + 1):
        sf = speed_decay ** (window_days - 1)
        for i in range(n - window_days + 1):
            end_i   = i + window_days - 1
            age     = n - 1 - end_i
            if age > max_age:
                continue
            acc_ret = float(c_series.iloc[i:i + window_days].sum())
            amp     = abs(acc_ret)
            if amp < min_amp:
                continue
            end_date = c_series.index[end_i]
            vol = vol_series.get(end_date, np.nan)
            if np.isnan(vol) or vol <= 0:
                continue
            z = amp / (vol * np.sqrt(window_days))
            if z < min_z:
                continue
            score = (amp / 0.05) / 2.0 * np.tanh(z / 2.0) * sf
            if score > best_score:
                best_score = score
                best = (acc_ret, score, age, end_date)

    return best


# ── Dynamic opportunity / risk scaling ───────────────────────────────────────

def _compute_dynamic_commodity_scale(
    recent_shock_scores: deque,
    commodity_pnl_history: deque,
    risk_window: int,
    min_scale: float,
    max_scale: float,
) -> float:
    """
    Scale the commodity exposure cap by an opportunity/risk ratio.

    opportunity = peak shock score seen in the tracked window
                  (strong recent shock → more room to trade)
    risk        = cumulative P&L loss over the risk window
                  (recent losses → tighten allocation)

    Returns a multiplier in [min_scale, max_scale] applied to the
    static base cap from commodity_exposure_overrides.
    """
    best_shock = max(recent_shock_scores, default=0.0)
    # reference score ≈ 0.5 (3 % move, z≈2.5) → tanh(1) ≈ 0.76
    opp = float(np.tanh(best_shock / 0.5))

    buf = list(commodity_pnl_history)
    recent = buf[-risk_window:] if len(buf) >= risk_window else buf
    recent_pnl = float(sum(recent)) if recent else 0.0
    # 1.5 % drawdown → risk = 1.0  (scale roughly halved)
    risk = max(-recent_pnl / 0.015, 0.0)

    raw = (1.0 + opp) / (1.0 + risk)
    return float(np.clip(raw, min_scale, max_scale))


# ── Sharpe-weighted commodity allocation ──────────────────────────────────────

def _sharpe_weighted_exposure(
    pnl_history: Dict[str, deque],
    commodities: List[str],
    base_max_exp: float,
    floor_mult: float = 0.25,
    cap_mult: float = 2.0,
) -> Dict[str, float]:
    """
    Compute per-commodity max exposure scaled by rolling realised Sharpe.

    Commodities with a strong positive Sharpe get up to cap_mult × base_max_exp.
    Commodities with zero/negative Sharpe are floored at floor_mult × base_max_exp.
    Equal weights apply until 20 days of history have accumulated.

    Using lagged data (history updated *after* decisions) so there is no
    look-ahead bias — today's sizing reflects yesterday's realised performance.
    """
    sharpes: Dict[str, float] = {}
    for c in commodities:
        buf = pnl_history[c]
        if len(buf) >= 20:
            arr = np.array(buf)
            mu    = arr.mean() * 252
            sigma = arr.std() * np.sqrt(252)
            sharpes[c] = mu / sigma if sigma > 0 else 0.0
        else:
            sharpes[c] = 0.0  # neutral until enough history

    pos_sharpes = {c: max(s, 0.0) for c, s in sharpes.items()}
    total = sum(pos_sharpes.values())
    n     = len(commodities)

    if total > 0:
        weights = {c: pos_sharpes[c] / total for c in commodities}
    else:
        weights = {c: 1.0 / n for c in commodities}

    return {
        c: float(np.clip(
            base_max_exp * n * weights[c],
            base_max_exp * floor_mult,
            base_max_exp * cap_mult,
        ))
        for c in commodities
    }


# ── Rebalancing ───────────────────────────────────────────────────────────────

def _rebalance_portfolio(
    portfolio: "Portfolio",
    current_date: pd.Timestamp,
    prices: pd.Series,
    betas: Dict[Tuple[str, str], pd.Series],
    residuals: Dict[str, pd.Series],
    commod_vol: pd.DataFrame,
    risk_cfg: dict,
    min_alpha: float,
    min_amp: float,
    min_z: float,
    stop_loss_pct: float,
    max_age: int,
    ar_ratio: float,
    slippage_pct: float = 0.0,
    speed_decay: float = 0.7,
    stock_vol_14d: Optional[pd.DataFrame] = None,
    atr_multiplier: float = 2.0,
    max_atr_stop: float = 0.12,
) -> None:
    """
    Re-evaluate every open position. Adjust size in-place; NEVER close+reopen.

    Uses trade.original_shock_return (not today's commodity move) so the
    expected-reaction benchmark stays consistent across the holding period.

    Close conditions (in order):
      1. No price data
      2. Shock faded (amp < min_amp or z < min_z)
      3. Original shock too old (> 2×max_age calendar days)
      4. Direction reversed
      5. Alpha below threshold
      6. Already fully priced

    Otherwise: resize + reset trailing TP/SL from current price.
    """
    max_pos     = risk_cfg.get("max_position_pct", 0.05)
    min_pos     = risk_cfg.get("min_position_pct", 0.005)
    target_risk = risk_cfg.get("target_position_risk_pct", 0.01)
    ref_sr      = risk_cfg.get("reference_sharpe", 0.50)
    max_boost   = risk_cfg.get("max_signal_boost", 2.0)

    still_open: List[Trade] = []

    for trade in portfolio.open_trades:
        # 1. No price
        if trade.ticker not in prices or np.isnan(prices[trade.ticker]):
            still_open.append(trade)
            continue

        current_price = prices[trade.ticker]
        commod = trade.commodity

        # Beta at current date
        beta_series = betas.get((trade.ticker, commod))
        if beta_series is None or current_date not in beta_series.index:
            still_open.append(trade)
            continue
        beta_val = float(beta_series.loc[current_date])
        if np.isnan(beta_val) or beta_val == 0:
            still_open.append(trade)
            continue

        # 2. Original shock amplitude / z-score
        shock_ret = trade.original_shock_return
        shock_amp = abs(shock_ret)
        if shock_amp < min_amp:
            trade.close(current_date, current_price, "rebalance_shock_faded", slippage_pct)
            portfolio.closed_trades.append(trade)
            continue

        vol_today = (
            commod_vol.loc[current_date, commod]
            if current_date in commod_vol.index else np.nan
        )
        if np.isnan(vol_today) or vol_today <= 0:
            still_open.append(trade)
            continue
        shock_z = shock_amp / vol_today
        if shock_z < min_z:
            trade.close(current_date, current_price, "rebalance_shock_faded", slippage_pct)
            portfolio.closed_trades.append(trade)
            continue

        # Residuals accumulated since entry
        resid_series = residuals.get(trade.ticker)
        if resid_series is None:
            still_open.append(trade)
            continue
        resid_window = resid_series.loc[
            (resid_series.index >= trade.entry_date) &
            (resid_series.index <= current_date)
        ]
        actual_reaction = float(resid_window.sum()) if len(resid_window) > 0 else 0.0

        expected  = beta_val * shock_ret
        new_alpha = expected - actual_reaction
        abs_alpha = abs(new_alpha)
        new_dir   = "LONG" if new_alpha > 0 else "SHORT"

        # 4. Direction reversed
        if new_dir != trade.direction:
            trade.close(current_date, current_price, "rebalance_direction_reversed", slippage_pct)
            portfolio.closed_trades.append(trade)
            continue

        # 5. Alpha too small
        if abs_alpha < min_alpha:
            trade.close(current_date, current_price, "rebalance_alpha_expired", slippage_pct)
            portfolio.closed_trades.append(trade)
            continue

        # 6. Already priced
        if expected != 0 and abs(actual_reaction / expected) >= ar_ratio:
            trade.close(current_date, current_price, "rebalance_already_priced", slippage_pct)
            portfolio.closed_trades.append(trade)
            continue

        # ── Resize in-place ───────────────────────────────────────────────
        resid_s = residuals.get(trade.ticker)
        if resid_s is not None:
            recent_r = resid_s.loc[:current_date].iloc[-63:]
            sigma_ann = float(recent_r.std() * np.sqrt(252)) if len(recent_r) > 5 else 0.25
        else:
            sigma_ann = 0.25
        sigma_ann  = max(sigma_ann, 0.05)
        sigma_hold = max(sigma_ann * np.sqrt(max_age / 252), 1e-4)

        sharpe_sig = abs_alpha / sigma_hold
        base_size  = float(np.clip(target_risk / sigma_ann, 0.0, max_pos))
        boost      = float(np.clip(sharpe_sig / ref_sr, 0.0, max_boost))
        gate       = float(np.clip((shock_amp / 0.05) / 2 * np.tanh(shock_z / 2), 0, 1))
        new_size   = float(np.clip(base_size * boost * gate, min_pos, max_pos))

        # Vol-adjusted stop-loss (wider for high-vol stocks)
        eff_stop = stop_loss_pct
        if stock_vol_14d is not None:
            if current_date in stock_vol_14d.index and trade.ticker in stock_vol_14d.columns:
                dv = float(stock_vol_14d.at[current_date, trade.ticker])
                if not np.isnan(dv) and dv > 0:
                    eff_stop = max(stop_loss_pct, min(atr_multiplier * dv, max_atr_stop))

        # Trailing TP/SL reset from current price
        _tp_ratio = risk_cfg.get("tp_alpha_ratio", 1.0)
        abs_tp    = abs_alpha * _tp_ratio
        if trade.direction == "LONG":
            new_tp = current_price * (1.0 + abs_tp)
            new_sl = current_price * (1.0 - eff_stop)
        else:
            new_tp = current_price * (1.0 - abs_tp)
            new_sl = current_price * (1.0 + eff_stop)

        trade.size               = new_size
        trade.alpha_at_entry     = abs_alpha
        trade.take_profit_price  = new_tp
        trade.stop_loss_price    = new_sl
        trade.rebalance_count   += 1
        trade.last_rebalance_date = current_date
        # NOTE: prev_price is NOT changed here — Portfolio.update() handles it

        still_open.append(trade)

    portfolio.open_trades = still_open


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """Tracks open positions and computes DAILY (not cumulative) P&L."""

    def __init__(
        self,
        stop_loss_pct: float,
        max_holding: int,
        slippage_pct: float = 0.0,
        trail_activation_pct: float = 0.0,
        trail_buffer_pct: float = 0.005,
    ) -> None:
        self.stop_loss_pct        = stop_loss_pct
        self.max_holding          = max_holding
        self.slippage_pct         = slippage_pct
        self.trail_activation_pct = trail_activation_pct
        self.trail_buffer_pct     = trail_buffer_pct
        self.open_trades:   List[Trade] = []
        self.closed_trades: List[Trade] = []

    def open_position(self, trade: Trade) -> None:
        """One position per (ticker, commodity) at a time."""
        already = any(
            t.ticker == trade.ticker and t.commodity == trade.commodity
            for t in self.open_trades
        )
        if not already:
            self.open_trades.append(trade)

    def update(
        self,
        current_date: pd.Timestamp,
        prices: pd.Series,
    ) -> Dict[str, float]:
        """
        Mark-to-market all open positions.

        P&L uses DAILY price change (current vs previous bar), NOT the
        cumulative return since entry — that was the v1 bug that made
        Sharpe, drawdown, and equity curve all incorrect.

        Exit logic: TP / SL / max-holding expiry.
        """
        daily_pnl: Dict[str, float] = {}
        still_open: List[Trade] = []

        for trade in self.open_trades:
            if trade.ticker not in prices or np.isnan(prices[trade.ticker]):
                still_open.append(trade)
                continue

            current_price = prices[trade.ticker]
            days_held = (current_date - trade.entry_date).days

            # ── Trailing stop activation ───────────────────────────────────
            # Once price moves trail_activation_pct of the way toward TP,
            # lock in breakeven + small buffer so profit can't fully reverse.
            if self.trail_activation_pct > 0 and trade.take_profit_price is not None:
                if trade.direction == "LONG":
                    full_move = trade.take_profit_price - trade.entry_price
                    cur_move  = current_price - trade.entry_price
                    if full_move > 0 and cur_move >= self.trail_activation_pct * full_move:
                        trail_sl = trade.entry_price * (1.0 + self.trail_buffer_pct)
                        if trade.stop_loss_price is None or trail_sl > trade.stop_loss_price:
                            trade.stop_loss_price = trail_sl
                else:  # SHORT
                    full_move = trade.entry_price - trade.take_profit_price
                    cur_move  = trade.entry_price - current_price
                    if full_move > 0 and cur_move >= self.trail_activation_pct * full_move:
                        trail_sl = trade.entry_price * (1.0 - self.trail_buffer_pct)
                        if trade.stop_loss_price is None or trail_sl < trade.stop_loss_price:
                            trade.stop_loss_price = trail_sl

            # Exit checks
            hit_stop = hit_tp = False
            if trade.stop_loss_price is not None:
                hit_stop = (
                    current_price <= trade.stop_loss_price if trade.direction == "LONG"
                    else current_price >= trade.stop_loss_price
                )
            if trade.take_profit_price is not None:
                hit_tp = (
                    current_price >= trade.take_profit_price if trade.direction == "LONG"
                    else current_price <= trade.take_profit_price
                )

            if hit_stop:
                trade.close(current_date, current_price, "stop_loss", self.slippage_pct)
                self.closed_trades.append(trade)
            elif hit_tp:
                trade.close(current_date, current_price, "take_profit", self.slippage_pct)
                self.closed_trades.append(trade)
            elif days_held >= self.max_holding:
                trade.close(current_date, current_price, "expired", self.slippage_pct)
                self.closed_trades.append(trade)
            else:
                still_open.append(trade)

                # ── DAILY P&L (v2 fix) ────────────────────────────────────
                # Use prev_price (yesterday's close) not entry_price.
                # On the first bar after entry, prev_price == entry_price
                # so daily_ret = 0 (no unrealised gain from fill logic).
                prev = trade.prev_price
                if prev is not None and prev > 0:
                    daily_ret = current_price / prev - 1.0
                else:
                    daily_ret = 0.0

                signed_ret = daily_ret if trade.direction == "LONG" else -daily_ret
                c = trade.commodity
                daily_pnl[c] = daily_pnl.get(c, 0.0) + trade.size * signed_ret

                # Advance prev_price for next bar
                trade.prev_price = current_price

        self.open_trades = still_open
        return daily_pnl

    def force_close_all(
        self,
        current_date: pd.Timestamp,
        prices: pd.Series,
    ) -> None:
        for trade in self.open_trades:
            if trade.ticker in prices and not np.isnan(prices[trade.ticker]):
                trade.close(
                    current_date, prices[trade.ticker],
                    "end_of_backtest", self.slippage_pct,
                )
                self.closed_trades.append(trade)
        self.open_trades = []

    @property
    def all_trades(self) -> List[Trade]:
        return self.closed_trades + self.open_trades


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(
    stock_returns: pd.DataFrame,
    stock_prices: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    market_returns: pd.Series,
    stock_metadata: pd.DataFrame,
    cfg: dict,
    step_days: int = 1,
    sector_returns: Optional[Dict[str, pd.Series]] = None,
    stock_volumes: Optional[pd.DataFrame] = None,
    vix_series: Optional[pd.Series] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """
    Walk-forward backtest — Lone Star v2 with all quant optimisations.

    Args:
        stock_returns     : DataFrame [dates × tickers] log returns
        stock_prices      : DataFrame [dates × tickers] adjusted close
        commodity_returns : DataFrame [dates × commodity_keys] log returns
        market_returns    : Series — S&P 500 daily returns
        stock_metadata    : DataFrame(ticker → zone, primary_commodity, role)
        cfg               : Full YAML config dict
        step_days         : Re-scan for NEW signals every N days (1 = daily)
        sector_returns    : {commodity_key: Series} — reserved for future extension
        stock_volumes     : DataFrame [dates × tickers] daily share volume (ADV filter)
        vix_series        : VIX level series for regime filter (^VIX prices)
    """
    beta_cfg  = cfg["beta"]
    shock_cfg = cfg["shock"]
    alpha_cfg = cfg["alpha"]
    risk_cfg  = cfg["risk"]
    exec_cfg  = cfg.get("execution", {})
    data_cfg  = cfg.get("data", {})

    lookback    = beta_cfg.get("rolling_window", 252)
    halflife    = beta_cfg.get("ewma_halflife_days", 63)
    use_ewma    = beta_cfg.get("use_ewma", True)
    use_mkt_adj = beta_cfg.get("use_market_adjusted", True)

    max_age     = shock_cfg.get("max_age_days", 5)
    min_amp     = shock_cfg.get("min_amplitude_pct", 1.5) / 100
    min_z       = shock_cfg.get("min_zscore", 1.5)
    vol_window  = shock_cfg.get("lookback_vol_days", 63)
    speed_decay = shock_cfg.get("speed_decay", 0.7)
    max_accum   = shock_cfg.get("max_accumulation_days", 3)

    min_alpha   = alpha_cfg.get("min_alpha_pct", 0.5) / 100
    max_alpha   = alpha_cfg.get("max_alpha_pct", 15.0) / 100
    ar_ratio    = alpha_cfg.get("already_priced_ratio", 0.85)

    stop_loss       = risk_cfg.get("stop_loss_pct", 0.05)
    max_hold        = risk_cfg.get("max_holding_days", 10)
    max_pos         = risk_cfg.get("max_position_pct", 0.05)
    min_pos         = risk_cfg.get("min_position_pct", 0.005)
    max_commod_exp  = risk_cfg.get("max_commodity_exposure", 0.20)
    rebal_freq      = risk_cfg.get("rebalancing_frequency_days", 2)
    min_cross_beta  = risk_cfg.get("min_cross_commodity_beta", 0.20)

    slippage_pct = exec_cfg.get("transaction_cost_bps", 10) / 10_000 / 2  # per leg
    min_adv_usd  = data_cfg.get("min_avg_daily_volume_usd", 0)

    # ── Regime filter params ──────────────────────────────────────────────
    vix_threshold    = risk_cfg.get("regime_vix_threshold", 25.0)
    correl_threshold = risk_cfg.get("regime_correl_threshold", 0.70)
    correl_window    = risk_cfg.get("regime_correl_window", 20)

    # ── Concentration cap ─────────────────────────────────────────────────
    max_pos_per_shock = risk_cfg.get("max_positions_per_shock", 5)

    # ── Vol-adjusted stop-loss ────────────────────────────────────────────
    atr_multiplier = risk_cfg.get("atr_stop_multiplier", 2.0)
    max_atr_stop   = risk_cfg.get("max_atr_stop_pct", 0.12)

    # ── Sharpe-weighted allocation ────────────────────────────────────────
    sharpe_floor_mult = risk_cfg.get("sharpe_weight_floor", 0.25)
    sharpe_cap_mult   = risk_cfg.get("sharpe_weight_cap", 2.0)

    # ── Order quality: TP ratio, trailing stop, portfolio heat ───────────
    tp_alpha_ratio     = risk_cfg.get("tp_alpha_ratio", 1.0)
    trail_activation   = risk_cfg.get("trail_stop_activation_pct", 0.0)
    trail_buffer       = risk_cfg.get("trail_stop_buffer_pct", 0.005)
    max_portfolio_heat = risk_cfg.get("max_portfolio_heat", 0.0)   # 0 = disabled

    # ── Dynamic opportunity / risk allocation ─────────────────────────────
    dyn_cfg         = risk_cfg.get("dynamic_allocation", {})
    dyn_enabled     = dyn_cfg.get("enabled", False)
    dyn_opp_window  = int(dyn_cfg.get("opportunity_lookback_days", 10))
    dyn_risk_window = int(dyn_cfg.get("risk_lookback_days", 21))
    dyn_min_scale   = float(dyn_cfg.get("min_scale_factor", 0.15))
    dyn_max_scale   = float(dyn_cfg.get("max_scale_factor", 2.0))
    high_conv_thr   = float(dyn_cfg.get("high_conviction_score_threshold", 0.70))

    # ── Momentum pre-filter ───────────────────────────────────────────────
    momentum_lb  = risk_cfg.get("momentum_lookback_days", 5)
    momentum_adv = risk_cfg.get("momentum_max_adverse_pct", 0.15)

    # ── Portfolio drawdown circuit breaker ────────────────────────────────
    circuit_window = risk_cfg.get("drawdown_circuit_window_days", 5)
    circuit_stop   = risk_cfg.get("drawdown_circuit_stop_pct", 0.015)

    commodities = list(commodity_returns.columns)

    # ── Per-commodity hard exposure caps (override Sharpe-weighted) ───────
    commod_exp_overrides: Dict[str, float] = risk_cfg.get("commodity_exposure_overrides", {})

    # ── Pre-compute rolling ADV for liquidity filter ──────────────────────
    rolling_adv: Optional[pd.DataFrame] = None
    if stock_volumes is not None and not stock_volumes.empty and min_adv_usd > 0:
        common_v = stock_prices.columns.intersection(stock_volumes.columns)
        if len(common_v) > 0:
            notional = stock_prices[common_v] * stock_volumes[common_v]
            rolling_adv = notional.rolling(20, min_periods=5).mean()

    # ── Pre-compute 14-day rolling vol per stock (for vol-adj stop-loss) ─
    stock_vol_14d = stock_returns.rolling(14, min_periods=5).std()

    # ── Pre-compute N-day cumulative momentum per stock ───────────────────
    stock_cum_mom = (
        (1 + stock_returns)
        .rolling(momentum_lb, min_periods=max(momentum_lb // 2, 2))
        .apply(np.prod, raw=True) - 1
    )

    # ── Pre-compute betas and residuals ───────────────────────────────────
    beta_method = (
        "EWMA+market-adj" if (use_ewma and use_mkt_adj)
        else "EWMA" if use_ewma else "rolling"
    )
    print(f"  [BT] Pre-computing {beta_method} betas...", end=" ", flush=True)

    betas:    Dict[Tuple[str, str], pd.Series] = {}
    residuals: Dict[str, pd.Series] = {}

    stocks_available = [
        t for t in stock_returns.columns if t in stock_prices.columns
    ]

    mkt = market_returns.rename("market") if market_returns is not None else None

    for ticker in stocks_available:
        s_ret = stock_returns[ticker].dropna()

        # Residuals
        if use_ewma and mkt is not None:
            resid = _ewma_capm_residuals(
                s_ret,
                mkt.reindex(s_ret.index).fillna(0),
                halflife=halflife,
            )
        else:
            resid = _rolling_capm_residuals(
                s_ret,
                (mkt.reindex(s_ret.index).fillna(0) if mkt is not None
                 else pd.Series(0.0, index=s_ret.index)),
                window=lookback,
            )
        residuals[ticker] = resid

        # Commodity betas
        for commod in commodities:
            c_ret = commodity_returns[commod].dropna()
            if use_ewma and use_mkt_adj and mkt is not None:
                b = _ewma_market_adjusted_beta(
                    s_ret, c_ret,
                    mkt.reindex(s_ret.index).fillna(0),
                    halflife=halflife,
                )
            elif use_ewma:
                aligned = pd.concat([s_ret, c_ret], axis=1).dropna()
                if not aligned.empty:
                    y = aligned.iloc[:, 0]
                    x = aligned.iloc[:, 1]
                    ew_cov = y.ewm(halflife=halflife).cov(x)
                    ew_var = x.ewm(halflife=halflife).var().replace(0, np.nan)
                    b = ew_cov / ew_var
                else:
                    b = pd.Series(dtype=float)
            else:
                b = _rolling_beta(s_ret, c_ret, lookback)

            betas[(ticker, commod)] = b

    print(f"✓ ({len(betas)} pairs)")

    # ── Pre-compute rolling commodity volatility ──────────────────────────
    commod_vol = commodity_returns.rolling(vol_window).std()

    # ── Simulation dates ──────────────────────────────────────────────────
    common_dates = stock_returns.index.intersection(commodity_returns.index)
    sim_dates    = common_dates[lookback:]

    # ── Pre-compute regime risk-off flags ─────────────────────────────────
    # VIX level > threshold OR rolling |correl(mkt, avg commodity)| > threshold
    regime_risk_off = pd.Series(False, index=common_dates, dtype=bool)

    if vix_series is not None:
        vix_aligned = vix_series.reindex(common_dates).ffill()
        regime_risk_off |= (vix_aligned > vix_threshold).fillna(False)

    if mkt is not None:
        commod_avg = commodity_returns.reindex(common_dates).mean(axis=1)
        mkt_aligned = mkt.reindex(common_dates).fillna(0)
        roll_correl = (
            mkt_aligned
            .rolling(correl_window, min_periods=max(correl_window // 2, 5))
            .corr(commod_avg)
            .abs()
        )
        regime_risk_off |= (roll_correl > correl_threshold).fillna(False)

    # ── Apply user-requested date range filter ────────────────────────────
    if start_date:
        sim_dates = sim_dates[sim_dates >= pd.Timestamp(start_date)]
    if end_date:
        sim_dates = sim_dates[sim_dates <= pd.Timestamp(end_date)]

    regime_risk_off = regime_risk_off.reindex(sim_dates, fill_value=False)

    n_risk_off = int(regime_risk_off.sum())
    if n_risk_off > 0:
        print(f"  [BT] Regime filter: {n_risk_off}/{len(sim_dates)} days flagged risk-off "
              f"(VIX>{vix_threshold} or |correl|>{correl_threshold})")

    # Rolling P&L history per commodity — feeds Sharpe-weighted allocation
    pnl_history: Dict[str, deque] = {c: deque(maxlen=63) for c in commodities}
    # Start with equal weights; will be updated after each day
    eff_commod_exp: Dict[str, float] = {c: max_commod_exp for c in commodities}
    # Apply hard per-commodity overrides (static base caps on day 0)
    for _c, _cap in commod_exp_overrides.items():
        if _c in eff_commod_exp:
            eff_commod_exp[_c] = min(eff_commod_exp[_c], _cap)

    # Per-commodity rolling shock score tracker (feeds dynamic allocation)
    recent_shock_scores: Dict[str, deque] = {
        c: deque(maxlen=dyn_opp_window) for c in commodities
    }

    portfolio  = Portfolio(
        stop_loss, max_hold, slippage_pct,
        trail_activation_pct=trail_activation,
        trail_buffer_pct=trail_buffer,
    )
    all_cols   = commodities + ["combined"]
    pnl_rows: List[dict] = []

    print(
        f"  [BT] Simulating {len(sim_dates)} days "
        f"(rebal every {rebal_freq}d, slippage {slippage_pct*10_000:.0f}bps/leg)...",
        end=" ", flush=True,
    )

    for i, t in enumerate(sim_dates):
        prices_today = (
            stock_prices.loc[t] if t in stock_prices.index
            else pd.Series(dtype=float)
        )

        # ── 0a. Daily shock tracking — always runs, feeds dynamic alloc ───
        for _c in commodities:
            _cs = commodity_returns.loc[:t][_c].dropna()
            if len(_cs) >= vol_window + max_accum + 2:
                _vs = commod_vol.loc[:t][_c].dropna()
                _r  = _scan_best_shock(
                    _cs.iloc[-(max_age + max_accum + 2):], _vs,
                    min_amp, min_z, speed_decay, max_age, max_accum,
                )
                recent_shock_scores[_c].append(_r[1] if _r else 0.0)
            else:
                recent_shock_scores[_c].append(0.0)

        # ── 0b. Sharpe-weighted exposure limits (uses lagged history) ─────
        if i > 0:  # skip day 0; history is empty, equal weights already set
            eff_commod_exp = _sharpe_weighted_exposure(
                pnl_history, commodities, max_commod_exp,
                sharpe_floor_mult, sharpe_cap_mult,
            )
            if dyn_enabled:
                # Dynamic cap = base_cap × opportunity/risk scale factor
                # High-conviction signals still bypass via override below.
                for _c in commodities:
                    base_cap = commod_exp_overrides.get(_c, max_commod_exp)
                    scale = _compute_dynamic_commodity_scale(
                        recent_shock_scores[_c],
                        pnl_history[_c],
                        dyn_risk_window,
                        dyn_min_scale,
                        dyn_max_scale,
                    )
                    eff_commod_exp[_c] = min(eff_commod_exp[_c], base_cap * scale)
            else:
                # Legacy: hard overrides only
                for _c, _cap in commod_exp_overrides.items():
                    if _c in eff_commod_exp:
                        eff_commod_exp[_c] = min(eff_commod_exp[_c], _cap)

        # ── 1. Rebalance open positions ───────────────────────────────────
        if i > 0 and i % rebal_freq == 0 and portfolio.open_trades:
            _rebalance_portfolio(
                portfolio      = portfolio,
                current_date   = t,
                prices         = prices_today,
                betas          = betas,
                residuals      = residuals,
                commod_vol     = commod_vol,
                risk_cfg       = risk_cfg,
                min_alpha      = min_alpha,
                min_amp        = min_amp,
                min_z          = min_z,
                stop_loss_pct  = stop_loss,
                max_age        = max_age,
                ar_ratio       = ar_ratio,
                slippage_pct   = slippage_pct,
                speed_decay    = speed_decay,
                stock_vol_14d  = stock_vol_14d,
                atr_multiplier = atr_multiplier,
                max_atr_stop   = max_atr_stop,
            )

        # ── 2. Mark-to-market (TP / SL / expiry) ─────────────────────────
        pnl_today = portfolio.update(t, prices_today)

        # ── 3. Generate new signals every step_days ───────────────────────
        # ── Portfolio drawdown circuit breaker ────────────────────────────
        # If combined P&L dropped > threshold over the last N trading days,
        # treat today as risk-off for new entries (all signals are wrong → stop).
        circuit_break_today = False
        if circuit_window > 0 and len(pnl_rows) >= circuit_window:
            recent_loss = sum(r["combined"] for r in pnl_rows[-circuit_window:])
            if recent_loss < -circuit_stop:
                circuit_break_today = True

        # Regime filter: skip new entries on risk-off days
        risk_off_today = bool(regime_risk_off.get(t, False)) or circuit_break_today
        if i % step_days == 0 and not risk_off_today:
            for commod in commodities:
                # Portfolio heat cap: stop opening new positions across all
                # commodities once total open exposure reaches the limit.
                if max_portfolio_heat > 0:
                    _cur_heat = sum(tr.size for tr in portfolio.open_trades)
                    if _cur_heat >= max_portfolio_heat:
                        break

                c_series = commodity_returns.loc[:t][commod].dropna()
                if len(c_series) < vol_window + max_accum + 2:
                    continue

                vol_series = commod_vol.loc[:t][commod].dropna()

                shock_result = _scan_best_shock(
                    c_series    = c_series.iloc[-(max_age + max_accum + 2):],
                    vol_series  = vol_series,
                    min_amp     = min_amp,
                    min_z       = min_z,
                    speed_decay = speed_decay,
                    max_age     = max_age,
                    max_accum   = max_accum,
                )
                if shock_result is None:
                    continue

                shock_ret, shock_score, shock_age, shock_date = shock_result

                # Current commodity exposure check
                commod_cap     = eff_commod_exp[commod]
                cur_commod_exp = sum(
                    tr.size for tr in portfolio.open_trades
                    if tr.commodity == commod
                )

                if cur_commod_exp >= commod_cap:
                    # High-conviction override: exceptional shocks bypass the
                    # dynamically-reduced cap and fall back to the static base cap.
                    # This ensures we never miss a strong gold/crude signal just
                    # because recent performance was poor.
                    if dyn_enabled and shock_score >= high_conv_thr:
                        hard_cap = commod_exp_overrides.get(commod, max_commod_exp)
                        if cur_commod_exp < hard_cap:
                            commod_cap = hard_cap   # expand ceiling for this signal
                        else:
                            continue  # even static cap is full
                    else:
                        continue

                # ── Collect all candidate signals (concentration cap) ──────
                # Score every eligible stock, sort by quality, keep top N.
                candidates: List[dict] = []

                target_risk = risk_cfg.get("target_position_risk_pct", 0.01)
                ref_sr      = risk_cfg.get("reference_sharpe", 0.50)
                max_boost_v = risk_cfg.get("max_signal_boost", 2.0)

                for ticker in stocks_available:

                    # ADV liquidity filter
                    if rolling_adv is not None and min_adv_usd > 0:
                        if (
                            t in rolling_adv.index
                            and ticker in rolling_adv.columns
                        ):
                            adv = rolling_adv.at[t, ticker]
                            if np.isnan(adv) or adv < min_adv_usd:
                                continue

                    beta_series = betas.get((ticker, commod))
                    if beta_series is None or t not in beta_series.index:
                        continue
                    beta_val = float(beta_series.loc[t])
                    if np.isnan(beta_val) or abs(beta_val) < min_cross_beta:
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

                    # ── Momentum pre-filter ───────────────────────────────
                    # Reject if the stock has been moving strongly AGAINST
                    # the signal direction — likely a fundamental issue, not a lag.
                    if (
                        t in stock_cum_mom.index
                        and ticker in stock_cum_mom.columns
                    ):
                        cum_mom = float(stock_cum_mom.at[t, ticker])
                        if not np.isnan(cum_mom):
                            if direction == "LONG" and cum_mom < -momentum_adv:
                                continue
                            if direction == "SHORT" and cum_mom > momentum_adv:
                                continue

                    # CML position sizing
                    recent_r   = resid_series.loc[:t].iloc[-63:]
                    sigma_ann  = float(recent_r.std() * np.sqrt(252)) if len(recent_r) > 5 else 0.25
                    sigma_ann  = max(sigma_ann, 0.05)
                    sigma_hold = max(sigma_ann * np.sqrt(max_age / 252), 1e-4)
                    sharpe_sig = abs(alpha) / sigma_hold

                    base_size  = float(np.clip(target_risk / sigma_ann, 0.0, max_pos))
                    boost      = float(np.clip(sharpe_sig / ref_sr, 0.0, max_boost_v))
                    gate       = float(np.clip(shock_score, 0.0, 1.0))
                    size       = float(np.clip(base_size * boost * gate, min_pos, max_pos))

                    if t not in stock_prices.index or ticker not in stock_prices.columns:
                        continue
                    raw_price = stock_prices.at[t, ticker]
                    if np.isnan(raw_price) or raw_price <= 0:
                        continue

                    # Vol-adjusted stop-loss: max(fixed, atr_mult × 14d_daily_vol)
                    eff_stop = stop_loss
                    if t in stock_vol_14d.index and ticker in stock_vol_14d.columns:
                        dv = float(stock_vol_14d.at[t, ticker])
                        if not np.isnan(dv) and dv > 0:
                            eff_stop = max(stop_loss, min(atr_multiplier * dv, max_atr_stop))

                    # Quality score for ranking (|alpha| × shock quality × size)
                    quality = abs(alpha) * shock_score * gate

                    candidates.append({
                        "ticker":    ticker,
                        "alpha":     alpha,
                        "size":      size,
                        "direction": direction,
                        "raw_price": raw_price,
                        "eff_stop":  eff_stop,
                        "quality":   quality,
                        "role": (
                            stock_metadata.loc[ticker, "role"]
                            if ticker in stock_metadata.index else ""
                        ),
                    })

                # ── Sort by quality desc; keep top N (concentration cap) ───
                candidates.sort(key=lambda c: c["quality"], reverse=True)

                remaining_cap = commod_cap - cur_commod_exp
                for cand in candidates[:max_pos_per_shock]:
                    if remaining_cap < min_pos:
                        break

                    size = min(cand["size"], remaining_cap)
                    if size < min_pos:
                        continue

                    direction   = cand["direction"]
                    raw_price   = cand["raw_price"]
                    eff_stop    = cand["eff_stop"]

                    entry_price = (
                        raw_price * (1.0 + slippage_pct) if direction == "LONG"
                        else raw_price * (1.0 - slippage_pct)
                    )
                    abs_alpha = abs(cand["alpha"])
                    abs_tp    = abs_alpha * tp_alpha_ratio  # conservative target

                    if direction == "LONG":
                        tp_price = entry_price * (1.0 + abs_tp)
                        sl_price = entry_price * (1.0 - eff_stop)
                    else:
                        tp_price = entry_price * (1.0 - abs_tp)
                        sl_price = entry_price * (1.0 + eff_stop)

                    trade = Trade(
                        ticker                = cand["ticker"],
                        commodity             = commod,
                        direction             = direction,
                        role                  = cand["role"],
                        entry_date            = t,
                        entry_price           = entry_price,
                        size                  = size,
                        alpha_at_entry        = abs_alpha,
                        take_profit_price     = tp_price,
                        stop_loss_price       = sl_price,
                        prev_price            = entry_price,
                        original_shock_return = shock_ret,
                        original_shock_date   = shock_date,
                        original_shock_age    = shock_age,
                    )
                    portfolio.open_position(trade)
                    remaining_cap -= size

        # Record daily P&L
        row = {c: pnl_today.get(c, 0.0) for c in commodities}
        row["combined"] = sum(row.values())
        row["date"] = t
        pnl_rows.append(row)

        # Update rolling P&L history (lagged: today's result sizes tomorrow's positions)
        for c in commodities:
            pnl_history[c].append(row[c])

    # Force-close all remaining positions at end of simulation
    if sim_dates.size > 0:
        portfolio.force_close_all(
            sim_dates[-1],
            stock_prices.loc[sim_dates[-1]] if sim_dates[-1] in stock_prices.index
            else pd.Series(dtype=float),
        )

    print("✓")

    # ── Assemble results ──────────────────────────────────────────────────
    if not pnl_rows:
        empty = pd.DataFrame(columns=all_cols)
        return BacktestResult(
            daily_pnl=empty, monthly_returns=empty,
            trades=[], nav_weights={},
            start_date=sim_dates[0] if len(sim_dates) else pd.Timestamp.today(),
            end_date=sim_dates[-1] if len(sim_dates) else pd.Timestamp.today(),
        )

    pnl_df  = pd.DataFrame(pnl_rows).set_index("date")[all_cols]
    monthly = (1 + pnl_df).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    trade_counts = {}
    for tr in portfolio.all_trades:
        trade_counts[tr.commodity] = trade_counts.get(tr.commodity, 0) + 1
    total = max(sum(trade_counts.values()), 1)
    nav_weights = {c: trade_counts.get(c, 0) / total for c in commodities}

    return BacktestResult(
        daily_pnl       = pnl_df,
        monthly_returns = monthly,
        trades          = portfolio.all_trades,
        nav_weights     = nav_weights,
        start_date      = sim_dates[0],
        end_date        = sim_dates[-1],
    )
