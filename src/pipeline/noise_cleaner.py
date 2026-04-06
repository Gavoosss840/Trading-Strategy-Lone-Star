"""
Step 3 — Market Noise Cleaning
────────────────────────────────
Isolates the stock-specific ("idiosyncratic") return by removing the
market and factor-driven component.

Two models, in order of preference:
  1. Fama-French 5-Factor (if data available)
  2. CAPM (fallback)

Formula:
    Residual_t = R_stock_t − R̂_stock_t

Where R̂ is the return explained by systematic factors.

The residual is the "pure alpha" — movements not explained by the market
or standard risk factors. This is the signal we compare against the
commodity shock's expected impact.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── CAPM baseline ─────────────────────────────────────────────────────────────

def fit_capm(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    risk_free: Optional[pd.Series] = None,
) -> Tuple[float, float, pd.Series]:
    """
    Fit CAPM:  (R_stock - Rf) = α + β·(R_mkt - Rf)

    Returns:
        (alpha, beta_mkt, residuals)
    """
    # Align
    common = stock_returns.index.intersection(market_returns.index)
    rs = stock_returns.loc[common].dropna()
    rm = market_returns.loc[common].dropna()
    common2 = rs.index.intersection(rm.index)
    rs = rs.loc[common2]
    rm = rm.loc[common2]

    if risk_free is not None:
        rf = risk_free.reindex(common2).fillna(0)
        rs_excess = rs - rf
        rm_excess = rm - rf
    else:
        rs_excess = rs
        rm_excess = rm

    if len(rs_excess) < 30:
        return 0.0, 1.0, rs

    slope, intercept, r_val, p_val, std_err = stats.linregress(rm_excess.values, rs_excess.values)

    explained = intercept + slope * rm_excess
    residuals = rs_excess - explained

    return float(intercept), float(slope), residuals


# ── Fama-French 5-Factor ──────────────────────────────────────────────────────

def fit_fama_french(
    stock_returns: pd.Series,
    ff_factors: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Fit Fama-French 5-Factor model:
        (R - Rf) = α + β1·MktRF + β2·SMB + β3·HML + β4·RMW + β5·CMA + ε

    Args:
        stock_returns: daily log returns for a single stock
        ff_factors   : DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, RF]

    Returns:
        (factor_loadings dict, residuals Series)
    """
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available_factors = [f for f in factor_cols if f in ff_factors.columns]

    if not available_factors:
        warnings.warn("FF factors missing expected columns, falling back to CAPM.")
        return {}, stock_returns

    # Align dates
    common = stock_returns.index.intersection(ff_factors.index)
    rs = stock_returns.loc[common].dropna()
    ff = ff_factors.loc[common].dropna()
    common2 = rs.index.intersection(ff.index)
    rs = rs.loc[common2]
    ff = ff.loc[common2]

    if len(rs) < 60:
        return {}, rs

    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)
    rs_excess = rs - rf

    X = ff[available_factors].values
    y = rs_excess.values

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])

    try:
        coeffs, residuals_raw, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}, rs

    alpha_intercept = coeffs[0]
    factor_loadings = dict(zip(available_factors, coeffs[1:]))
    factor_loadings["alpha"] = alpha_intercept

    explained = X_with_const @ coeffs
    residuals = pd.Series(y - explained, index=rs_excess.index, name=rs.name)

    return factor_loadings, residuals


# ── Main noise cleaning pipeline ──────────────────────────────────────────────

def compute_residuals(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free: Optional[pd.Series] = None,
    ff_factors: Optional[pd.DataFrame] = None,
    rolling_window: int = 252,
) -> pd.DataFrame:
    """
    Remove market/factor noise from all stocks in the universe.

    Uses rolling estimation so factor loadings update over time.
    For the signal we care about recent residuals — the last 10 days.

    Args:
        stock_returns  : DataFrame [dates × tickers]
        market_returns : Series of market returns
        risk_free      : Series of daily risk-free rates (optional)
        ff_factors     : FF5 factor DataFrame (optional)
        rolling_window : Window for factor estimation

    Returns:
        DataFrame of residual returns, same shape as stock_returns.
    """
    use_ff = ff_factors is not None and not ff_factors.empty

    # Estimate factor loadings on the full available history
    # (rolling in production; full-sample here for efficiency)
    residuals_dict = {}

    for ticker in stock_returns.columns:
        series = stock_returns[ticker].dropna()

        if use_ff:
            # Align ff_factors to available returns
            ff_aligned = ff_factors.reindex(series.index).dropna()
            series_aligned = series.reindex(ff_aligned.index).dropna()
            common = ff_aligned.index.intersection(series_aligned.index)
            if len(common) >= 60:
                _, resids = fit_fama_french(series.loc[common], ff_factors.loc[common])
                residuals_dict[ticker] = resids
                continue

        # Fallback: CAPM
        mkt_aligned = market_returns.reindex(series.index)
        rf_aligned = risk_free.reindex(series.index) if risk_free is not None else None
        _, _, resids = fit_capm(series, mkt_aligned, rf_aligned)
        residuals_dict[ticker] = resids

    if not residuals_dict:
        return pd.DataFrame()

    residuals_df = pd.DataFrame(residuals_dict)
    return residuals_df


def get_recent_residuals(
    residuals_df: pd.DataFrame,
    n_days: int = 5,
) -> pd.DataFrame:
    """Return the last n_days of residuals for signal comparison."""
    return residuals_df.iloc[-n_days:]


def cumulative_residual(
    residuals_df: pd.DataFrame,
    n_days: int = 5,
) -> pd.Series:
    """
    Cumulative residual return over the last n_days.
    This represents the "actual idiosyncratic reaction" to compare against
    the expected reaction from the commodity shock.
    """
    recent = get_recent_residuals(residuals_df, n_days)
    # Compound returns: (1+r1)(1+r2)...-1 approximated as sum for small returns
    return recent.sum()


def residual_volatility(
    residuals_df: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """
    Annualised idiosyncratic volatility per ticker from CAPM residuals.

    This is σ_résiduel used in the CML+Kelly position sizing:
        sigma_annual = std(daily_residuals, window) × √252

    Uses only the last `window` trading days for a recent estimate.

    Returns:
        Series indexed by ticker, values = annualised vol (decimal, e.g. 0.25)
    """
    if residuals_df.empty:
        return pd.Series(dtype=float, name="residual_vol_annual")

    recent = residuals_df.iloc[-window:] if len(residuals_df) >= window else residuals_df
    daily_std = recent.std()
    return (daily_std * np.sqrt(252)).rename("residual_vol_annual")
