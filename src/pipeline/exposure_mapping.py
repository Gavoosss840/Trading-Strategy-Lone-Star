"""
Step 1 — Exposure Mapping
──────────────────────────
Identifies the commodity beta of each stock via OLS regression:

    R_stock = α + β_commodity · R_commodity + ε

Outputs:
  - β  (commodity sensitivity)
  - β > 0 → producer (benefits when commodity price rises)
  - β < 0 → consumer (hurt when commodity price rises)
  - R² and p-value for significance filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BetaResult:
    ticker: str
    commodity: str
    beta: float
    alpha_intercept: float
    r_squared: float
    p_value: float       # p-value for the beta coefficient
    std_error: float
    n_obs: int
    role: str            # "producer" | "consumer" | "neutral"

    def is_significant(self, min_r2: float = 0.05, max_pval: float = 0.10) -> bool:
        return self.r_squared >= min_r2 and self.p_value <= max_pval


@dataclass
class ExposureMap:
    """Collection of BetaResult for all (stock, commodity) pairs."""
    results: List[BetaResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = [
            {
                "ticker": r.ticker,
                "commodity": r.commodity,
                "beta": r.beta,
                "alpha_intercept": r.alpha_intercept,
                "r_squared": r.r_squared,
                "p_value": r.p_value,
                "std_error": r.std_error,
                "n_obs": r.n_obs,
                "role": r.role,
                "significant": r.is_significant(),
            }
            for r in self.results
        ]
        return pd.DataFrame(rows).set_index(["ticker", "commodity"])

    def get(self, ticker: str, commodity: str) -> Optional[BetaResult]:
        for r in self.results:
            if r.ticker == ticker and r.commodity == commodity:
                return r
        return None

    def betas_for_stock(self, ticker: str) -> pd.Series:
        """Return {commodity: beta} series for a given stock."""
        betas = {
            r.commodity: r.beta
            for r in self.results
            if r.ticker == ticker
        }
        return pd.Series(betas, name=ticker)

    def significant_pairs(self, min_r2: float = 0.05, max_pval: float = 0.10) -> List[BetaResult]:
        return [r for r in self.results if r.is_significant(min_r2, max_pval)]


# ── Regression engine ─────────────────────────────────────────────────────────

def _remove_outliers(x: np.ndarray, y: np.ndarray, z_thresh: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """Remove observations where either series has a |z-score| > z_thresh."""
    mask = (np.abs(stats.zscore(x)) < z_thresh) & (np.abs(stats.zscore(y)) < z_thresh)
    return x[mask], y[mask]


def _ols_beta(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float, float, float, int]:
    """
    OLS regression y = α + β·x.

    Returns (beta, alpha, r_squared, p_value, std_error, n_obs).
    """
    n = len(x)
    if n < 30:
        return 0.0, 0.0, 0.0, 1.0, np.nan, n

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return (
        float(slope),
        float(intercept),
        float(r_value ** 2),
        float(p_value),
        float(std_err),
        n,
    )


def _classify_role(beta: float, threshold: float = 0.05) -> str:
    if beta > threshold:
        return "producer"
    elif beta < -threshold:
        return "consumer"
    return "neutral"


# ── Main estimation function ──────────────────────────────────────────────────

def estimate_betas(
    stock_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    min_observations: int = 120,
    outlier_zscore: float = 3.5,
    rolling_window: Optional[int] = None,
) -> ExposureMap:
    """
    Estimate commodity betas for every (stock, commodity) pair.

    Args:
        stock_returns       : DataFrame [dates × tickers] of log returns
        commodity_returns   : DataFrame [dates × commodities] of log returns
        min_observations    : Minimum shared observations for regression
        outlier_zscore      : Z-score threshold for outlier removal
        rolling_window      : If set, use last N observations only

    Returns:
        ExposureMap with BetaResult for each (stock, commodity) pair.
    """
    exposure_map = ExposureMap()

    # Align dates
    common_dates = stock_returns.index.intersection(commodity_returns.index)
    stocks = stock_returns.loc[common_dates]
    commodities = commodity_returns.loc[common_dates]

    if rolling_window is not None:
        stocks = stocks.iloc[-rolling_window:]
        commodities = commodities.iloc[-rolling_window:]

    for ticker in stocks.columns:
        for commodity in commodities.columns:
            y_full = stocks[ticker].dropna()
            x_full = commodities[commodity].dropna()

            # Align
            common = y_full.index.intersection(x_full.index)
            if len(common) < min_observations:
                continue

            y = y_full.loc[common].values
            x = x_full.loc[common].values

            # Remove outliers
            x_clean, y_clean = _remove_outliers(x, y, z_thresh=outlier_zscore)

            if len(x_clean) < min_observations:
                continue

            beta, alpha_intercept, r2, pval, stderr, n = _ols_beta(x_clean, y_clean)

            result = BetaResult(
                ticker=ticker,
                commodity=commodity,
                beta=beta,
                alpha_intercept=alpha_intercept,
                r_squared=r2,
                p_value=pval,
                std_error=stderr,
                n_obs=n,
                role=_classify_role(beta),
            )
            exposure_map.results.append(result)

    return exposure_map


def get_latest_betas(
    stock_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    rolling_window: int = 252,
    min_observations: int = 120,
    outlier_zscore: float = 3.5,
) -> ExposureMap:
    """
    Convenience wrapper — estimate betas using the last `rolling_window` days.
    This is the primary entry point used by the strategy pipeline.
    """
    return estimate_betas(
        stock_returns=stock_returns,
        commodity_returns=commodity_returns,
        min_observations=min_observations,
        outlier_zscore=outlier_zscore,
        rolling_window=rolling_window,
    )


# ── Beta summary table ────────────────────────────────────────────────────────

def summarize_exposures(exposure_map: ExposureMap, min_r2: float = 0.05) -> pd.DataFrame:
    """
    Return a clean summary table of significant (stock, commodity) exposures,
    sorted by |beta| descending.
    """
    df = exposure_map.to_dataframe()
    if df.empty:
        return df
    sig = df[df["r_squared"] >= min_r2].copy()
    sig["abs_beta"] = sig["beta"].abs()
    return sig.sort_values("abs_beta", ascending=False).drop(columns="abs_beta")
