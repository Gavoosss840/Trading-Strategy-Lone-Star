"""
Fundamental Data Loader
────────────────────────
Fetches financial ratios for stocks via yfinance:
  - Debt/Equity ratio (leverage)
  - Operating margin
  - Interest coverage ratio
  - Beta (market beta from yfinance as reference)

These are used in Step 2 (Fundamental Adjustment) to adjust the
raw commodity beta for financial structure.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ── Key ratio extraction ──────────────────────────────────────────────────────

def _safe_get(info: Dict, *keys, default: float = np.nan) -> float:
    """Try multiple keys in yfinance info dict, return first non-None value."""
    for key in keys:
        val = info.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    return default


def get_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Fetch key fundamental ratios for a single ticker.

    Returns a dict with:
        debt_to_equity    : D/E ratio (financial leverage)
        operating_margin  : operating income / revenue
        interest_coverage : EBIT / interest expense
        net_margin        : net income / revenue
        market_beta       : market beta (from yfinance)
        tax_rate          : effective tax rate
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # ── Leverage ──────────────────────────────────────────────────────────
        debt_to_equity = _safe_get(
            info,
            "debtToEquity",      # yfinance standard key
            "totalDebt",         # fallback raw value
            default=1.0,
        )
        # yfinance returns D/E as percentage in some cases → normalise
        if debt_to_equity > 20:
            debt_to_equity = debt_to_equity / 100

        # ── Margins ───────────────────────────────────────────────────────────
        operating_margin = _safe_get(
            info,
            "operatingMargins",
            "ebitdaMargins",
            default=0.10,
        )
        # Ensure margin is in [0, 1] decimal form
        if abs(operating_margin) > 1:
            operating_margin = operating_margin / 100

        net_margin = _safe_get(
            info,
            "profitMargins",
            default=0.05,
        )
        if abs(net_margin) > 1:
            net_margin = net_margin / 100

        # ── Interest coverage ─────────────────────────────────────────────────
        # Approximate from income statement if direct ratio not available
        ebit = _safe_get(info, "ebit", "operatingIncome", default=np.nan)
        interest_expense = _safe_get(
            info, "interestExpense", "totalInterestExpense", default=np.nan
        )
        if not np.isnan(ebit) and not np.isnan(interest_expense) and interest_expense != 0:
            interest_coverage = abs(ebit / interest_expense)
        else:
            # Fallback: infer from D/E and margins
            interest_coverage = max(1.0, 5.0 / max(debt_to_equity, 0.1))

        # ── Market beta ───────────────────────────────────────────────────────
        market_beta = _safe_get(info, "beta", default=1.0)

        # ── Effective tax rate ────────────────────────────────────────────────
        income_before_tax = _safe_get(info, "incomeBeforeTax", default=np.nan)
        income_tax = _safe_get(info, "incomeTaxExpense", default=np.nan)
        if (
            not np.isnan(income_before_tax)
            and not np.isnan(income_tax)
            and income_before_tax > 0
        ):
            tax_rate = income_tax / income_before_tax
            tax_rate = float(np.clip(tax_rate, 0.0, 0.40))
        else:
            tax_rate = 0.21  # US statutory rate

        return {
            "debt_to_equity": float(np.clip(debt_to_equity, 0.0, 20.0)),
            "operating_margin": float(np.clip(operating_margin, -1.0, 1.0)),
            "net_margin": float(np.clip(net_margin, -1.0, 1.0)),
            "interest_coverage": float(np.clip(interest_coverage, 0.1, 50.0)),
            "market_beta": float(np.clip(market_beta, -5.0, 5.0)),
            "tax_rate": tax_rate,
        }

    except Exception as exc:
        warnings.warn(f"Fundamentals fetch failed for {ticker}: {exc}. Using defaults.")
        return _default_fundamentals()


def _default_fundamentals() -> Dict[str, float]:
    return {
        "debt_to_equity": 1.0,
        "operating_margin": 0.10,
        "net_margin": 0.05,
        "interest_coverage": 5.0,
        "market_beta": 1.0,
        "tax_rate": 0.21,
    }


def get_fundamentals_bulk(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch fundamentals for multiple tickers.

    Returns a DataFrame with tickers as index and ratio names as columns.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = get_fundamentals(ticker)

    df = pd.DataFrame(results).T
    df.index.name = "ticker"
    return df


# ── Quality score helpers ─────────────────────────────────────────────────────

def leverage_quality_score(debt_to_equity: float) -> float:
    """
    Map D/E ratio to a quality score in [0, 1].

    Low leverage → high score (company can absorb commodity shock).
    High leverage → low score (amplified distress risk).
    """
    if np.isnan(debt_to_equity):
        return 0.5
    # Sigmoid-like: D/E = 0 → 1.0, D/E = 2 → 0.5, D/E = 5 → 0.15
    return float(1 / (1 + 0.5 * debt_to_equity))


def margin_quality_score(operating_margin: float) -> float:
    """
    Map operating margin to a quality score in [0, 1].

    High margins → more pricing power → can partially absorb shocks.
    Thin/negative margins → more vulnerable.
    """
    if np.isnan(operating_margin):
        return 0.5
    # Clip and normalise: 0% → 0.2, 20% → 0.8, 40%+ → 1.0
    clipped = float(np.clip(operating_margin, -0.20, 0.50))
    return float((clipped + 0.20) / 0.70)


def coverage_quality_score(interest_coverage: float) -> float:
    """
    Map interest coverage ratio to a quality score in [0, 1].

    High coverage → safer → better quality for our trade.
    """
    if np.isnan(interest_coverage):
        return 0.5
    # log-scaled: coverage=1 → 0.2, coverage=5 → 0.6, coverage=20+ → 1.0
    return float(np.clip(np.log1p(interest_coverage) / np.log1p(20), 0.0, 1.0))


def compute_fundamental_quality(fundamentals: pd.DataFrame, cfg: Dict) -> pd.Series:
    """
    Compute a composite fundamental quality score for each ticker.

    Score = w_leverage * leverage_score + w_margin * margin_score + w_coverage * coverage_score
    """
    w_lev = cfg.get("leverage_weight", 0.4)
    w_mar = cfg.get("margin_weight", 0.3)
    w_cov = cfg.get("coverage_weight", 0.3)

    scores = {}
    for ticker, row in fundamentals.iterrows():
        lev_score = leverage_quality_score(row.get("debt_to_equity", 1.0))
        mar_score = margin_quality_score(row.get("operating_margin", 0.10))
        cov_score = coverage_quality_score(row.get("interest_coverage", 5.0))
        scores[ticker] = w_lev * lev_score + w_mar * mar_score + w_cov * cov_score

    return pd.Series(scores, name="fundamental_quality")
