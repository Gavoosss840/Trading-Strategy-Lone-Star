"""
Step 2 — Fundamental Beta Adjustment
──────────────────────────────────────
Adjusts the raw statistical commodity beta for a stock's financial structure
using the Modigliani-Miller framework and operating ratio modifiers.

Key idea:
  The raw beta from regression captures total equity sensitivity, but part of
  that sensitivity comes from financial leverage, not fundamental operations.
  We "unlever" to get the asset-level commodity sensitivity, then optionally
  re-adjust for quality factors.

Formula (MM unlevering):
    β_asset = β_equity / [1 + (1 − t) × D/E]

Quality adjustments:
  - High leverage amplifies both gains AND losses from commodity shocks
    → leverage_factor adjusts the effective sensitivity
  - Thin margins indicate limited buffer against commodity cost increases
    → margin_factor adjusts how directly the stock absorbs the shock
  - Low interest coverage suggests balance-sheet stress amplification
    → coverage_factor adds a distress premium

Final adjusted beta:
    β_adj = β_asset × (1 + leverage_weight × leverage_factor)
                     × (1 + margin_weight  × margin_factor)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.pipeline.exposure_mapping import BetaResult, ExposureMap


# ── Core Modigliani-Miller unlevering ─────────────────────────────────────────

def unlever_beta(
    beta_equity: float,
    debt_to_equity: float,
    tax_rate: float = 0.21,
) -> float:
    """
    Unlever equity beta to get asset (unlevered) beta.

        β_asset = β_equity / [1 + (1 − t) × D/E]

    Args:
        beta_equity  : Raw statistical beta from regression
        debt_to_equity: D/E ratio from balance sheet
        tax_rate     : Effective corporate tax rate

    Returns:
        Unlevered (asset-level) commodity beta
    """
    denominator = 1 + (1 - tax_rate) * max(debt_to_equity, 0.0)
    if denominator == 0:
        return beta_equity
    return beta_equity / denominator


def relever_beta(
    beta_asset: float,
    debt_to_equity: float,
    tax_rate: float = 0.21,
) -> float:
    """Re-lever an asset beta to equity level with given D/E."""
    return beta_asset * (1 + (1 - tax_rate) * max(debt_to_equity, 0.0))


# ── Adjustment factors ────────────────────────────────────────────────────────

def _leverage_factor(debt_to_equity: float) -> float:
    """
    Leverage amplification factor.

    High D/E amplifies commodity sensitivity (both positive and negative).
    Returns a value near 0 for low leverage, positive for high leverage.
    Capped to avoid extreme amplification.
    """
    # Normalised: D/E = 0 → 0, D/E = 2 → 0.5, D/E = 5 → 0.8
    return float(np.clip(debt_to_equity / (debt_to_equity + 2), 0.0, 1.0))


def _margin_factor(operating_margin: float) -> float:
    """
    Margin sensitivity factor.

    Low margins → commodity costs represent larger share of revenues
               → higher sensitivity (factor > 0)
    High margins → partial buffer   → lower pass-through (factor < 0)

    Returns value in [-0.3, +0.5].
    """
    baseline = 0.15  # Reference margin (15%)
    margin_clipped = float(np.clip(operating_margin, -0.20, 0.50))
    return float(np.clip((baseline - margin_clipped) * 2, -0.3, 0.5))


def _coverage_factor(interest_coverage: float) -> float:
    """
    Interest coverage sensitivity factor.

    Low coverage → financial distress amplifies commodity shock impact.
    Returns a value in [0, 0.3] (pure amplifier, never reduces).
    """
    # coverage = 1 → 0.3, coverage = 5 → 0.1, coverage = 20+ → 0
    return float(np.clip(0.3 / max(interest_coverage, 1.0), 0.0, 0.30))


# ── Adjusted beta computation ─────────────────────────────────────────────────

def compute_adjusted_beta(
    beta_raw: float,
    fundamentals: Dict[str, float],
    cfg: Dict,
) -> float:
    """
    Compute the fundamental-adjusted commodity beta.

    Pipeline:
      1. Unlever β_equity → β_asset  (remove leverage from regression beta)
      2. Apply operating modifier:
         β_adj = β_asset × (1 + Σ w_i × factor_i)

    Args:
        beta_raw     : Raw OLS commodity beta for this stock
        fundamentals : Dict with debt_to_equity, operating_margin,
                       interest_coverage, tax_rate
        cfg          : Config section (fundamental adjustment weights)

    Returns:
        β_adjusted (preserves sign of raw beta)
    """
    de = fundamentals.get("debt_to_equity", 1.0)
    om = fundamentals.get("operating_margin", 0.10)
    ic = fundamentals.get("interest_coverage", 5.0)
    tr = fundamentals.get("tax_rate", 0.21)

    w_lev = cfg.get("leverage_weight", 0.4)
    w_mar = cfg.get("margin_weight", 0.3)
    w_cov = cfg.get("coverage_weight", 0.3)

    # Step 1: Unlever
    beta_asset = unlever_beta(beta_raw, de, tr)

    # Step 2: Compute adjustment factors
    lev_f = _leverage_factor(de)
    mar_f = _margin_factor(om)
    cov_f = _coverage_factor(ic)

    # Step 3: Composite modifier
    # The modifier shifts beta_asset slightly based on financial structure
    modifier = 1.0 + w_lev * lev_f + w_mar * mar_f + w_cov * cov_f

    beta_adjusted = beta_asset * modifier

    # Sanity clamp: adjusted beta shouldn't exceed 3× raw beta
    max_beta = 3.0 * abs(beta_raw) if beta_raw != 0 else 3.0
    beta_adjusted = float(np.clip(beta_adjusted, -max_beta, max_beta))

    return beta_adjusted


# ── Batch adjustment for exposure map ────────────────────────────────────────

def adjust_exposure_map(
    exposure_map: ExposureMap,
    fundamentals_df: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    """
    Apply fundamental adjustment to all beta results in the exposure map.

    Args:
        exposure_map   : ExposureMap from Step 1
        fundamentals_df: DataFrame with ticker index and ratio columns
        cfg            : fundamental section from config

    Returns:
        DataFrame with columns:
            ticker, commodity, beta_raw, beta_adjusted, role,
            r_squared, p_value, debt_to_equity, operating_margin,
            interest_coverage, adjustment_factor
    """
    rows = []

    for result in exposure_map.results:
        ticker = result.ticker

        # Get fundamentals for this ticker (use defaults if missing)
        if ticker in fundamentals_df.index:
            fund = fundamentals_df.loc[ticker].to_dict()
        else:
            fund = {
                "debt_to_equity": 1.0,
                "operating_margin": 0.10,
                "interest_coverage": 5.0,
                "tax_rate": 0.21,
            }

        beta_adj = compute_adjusted_beta(result.beta, fund, cfg)
        adj_factor = beta_adj / result.beta if result.beta != 0 else 1.0

        rows.append({
            "ticker": ticker,
            "commodity": result.commodity,
            "beta_raw": result.beta,
            "beta_adjusted": beta_adj,
            "adjustment_factor": adj_factor,
            "role": result.role,
            "r_squared": result.r_squared,
            "p_value": result.p_value,
            "n_obs": result.n_obs,
            "debt_to_equity": fund.get("debt_to_equity"),
            "operating_margin": fund.get("operating_margin"),
            "interest_coverage": fund.get("interest_coverage"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(["ticker", "commodity"])
    return df
