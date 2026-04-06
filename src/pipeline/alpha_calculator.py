"""
Step 5 — Alpha Calculation (Expected vs Actual)
────────────────────────────────────────────────
Core of the Lone Star strategy.

Given a commodity shock and the adjusted beta of a stock:
    Expected reaction = β_adjusted × Shock_return
    Actual reaction   = Residual return (from Step 3, last N days)
    Alpha             = Expected - Actual

Interpretation:
    Alpha > 0  →  Under-reaction  →  LONG  (market hasn't priced the full shock)
    Alpha < 0  →  Over-reaction   →  SHORT (market over-shot, mean reversion expected)
    Alpha ≈ 0  →  Fully priced    →  NO TRADE

The key insight: we know approximately what the stock SHOULD do based on its
commodity exposure. The gap between expected and actual is the exploitable alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.pipeline.shock_detector import CommodityShock


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class AlphaSignal:
    ticker: str
    commodity: str
    role: str                    # "producer" | "consumer"
    beta_adjusted: float
    shock_return: float          # Commodity shock return
    expected_reaction: float     # β_adj × shock_return
    actual_reaction: float       # Observed residual return
    alpha: float                 # expected - actual
    direction: str               # "LONG" | "SHORT" | "NO TRADE"
    shock_score: float
    shock_age_days: int
    confidence: float            # Signal confidence [0, 1]

    @property
    def is_tradeable(self) -> bool:
        return self.direction in ("LONG", "SHORT")

    def __repr__(self) -> str:
        sign = "+" if self.alpha > 0 else ""
        return (
            f"AlphaSignal({self.ticker} | {self.commodity} | {self.direction} | "
            f"α={sign}{self.alpha:.3%} | β={self.beta_adjusted:.2f} | "
            f"expected={self.expected_reaction:.3%} actual={self.actual_reaction:.3%})"
        )


# ── Alpha computation ─────────────────────────────────────────────────────────

def compute_alpha(
    expected_reaction: float,
    actual_reaction: float,
) -> float:
    """
    Alpha = Expected reaction − Actual reaction.

    Positive alpha: stock under-reacted (should be higher) → LONG
    Negative alpha: stock over-reacted (should be lower)   → SHORT
    """
    return expected_reaction - actual_reaction


def classify_direction(
    alpha: float,
    min_alpha: float = 0.005,
) -> str:
    """
    Classify trade direction from alpha.

    Args:
        alpha      : Expected - Actual return gap
        min_alpha  : Minimum |alpha| to generate a signal (default 0.5%)

    Returns:
        "LONG" | "SHORT" | "NO TRADE"
    """
    if alpha > min_alpha:
        return "LONG"
    elif alpha < -min_alpha:
        return "SHORT"
    return "NO TRADE"


def _signal_confidence(
    alpha: float,
    shock_score: float,
    r_squared: float,
    beta_std_error: float,
    beta_value: float,
) -> float:
    """
    Estimate signal confidence as a composite of:
      - Shock quality (shock_score)
      - Beta regression quality (r_squared)
      - Beta precision (|beta| / std_error → t-statistic proxy)
      - Alpha magnitude relative to beta uncertainty

    Returns value in [0, 1].
    """
    # Beta t-statistic proxy
    t_stat = abs(beta_value) / max(beta_std_error, 1e-6) if beta_std_error > 0 else 2.0
    t_score = np.clip(t_stat / 4.0, 0.0, 1.0)  # t=4 → 1.0

    # Alpha magnitude (normalise by 5% reference)
    alpha_score = np.clip(abs(alpha) / 0.05, 0.0, 1.0)

    # Regression quality
    r2_score = np.clip(r_squared / 0.30, 0.0, 1.0)  # R²=30% → 1.0

    # Composite (weighted average)
    confidence = 0.35 * shock_score + 0.25 * t_score + 0.25 * r2_score + 0.15 * alpha_score
    return float(np.clip(confidence, 0.0, 1.0))


# ── Main signal generator ─────────────────────────────────────────────────────

def generate_alpha_signals(
    shocks: List[CommodityShock],
    adjusted_betas: pd.DataFrame,
    cumulative_residuals: pd.Series,
    cfg: Dict,
    beta_std_errors: Optional[Dict] = None,
    r_squared_map: Optional[Dict] = None,
) -> List[AlphaSignal]:
    """
    Generate AlphaSignal for every (stock, commodity shock) pair.

    Args:
        shocks              : Active commodity shocks from Step 4
        adjusted_betas      : DataFrame with MultiIndex (ticker, commodity)
                              and column 'beta_adjusted', 'role'
        cumulative_residuals: Series indexed by ticker (Step 3 output)
        cfg                 : Alpha section from config
        beta_std_errors     : Optional {(ticker, commodity): std_error}
        r_squared_map       : Optional {(ticker, commodity): r_squared}

    Returns:
        List of AlphaSignal, sorted by |alpha| descending.
    """
    min_alpha = cfg.get("min_alpha_pct", 0.5) / 100
    max_alpha = cfg.get("max_alpha_pct", 15.0) / 100

    signals: List[AlphaSignal] = []

    for shock in shocks:
        commodity = shock.commodity

        # Get all stocks with a beta for this commodity
        if adjusted_betas.empty:
            continue
        if "commodity" in adjusted_betas.index.names:
            try:
                commodity_betas = adjusted_betas.xs(commodity, level="commodity")
            except KeyError:
                continue
        else:
            continue

        for ticker in commodity_betas.index:
            row = commodity_betas.loc[ticker]

            beta_adj = row.get("beta_adjusted", 0.0)
            role = row.get("role", "neutral")
            r2 = row.get("r_squared", 0.0)

            if beta_adj == 0.0:
                continue

            # Expected reaction: how much should this stock move given the shock?
            expected_reaction = beta_adj * shock.price_return

            # Actual reaction: idiosyncratic return since shock date
            actual_reaction = float(cumulative_residuals.get(ticker, 0.0))

            # Alpha gap
            alpha = compute_alpha(expected_reaction, actual_reaction)

            # Skip if alpha is outside reasonable bounds
            if abs(alpha) < min_alpha or abs(alpha) > max_alpha:
                continue

            # Trade direction
            direction = classify_direction(alpha, min_alpha)
            if direction == "NO TRADE":
                continue

            # Confidence
            std_err = (beta_std_errors or {}).get((ticker, commodity), abs(beta_adj) * 0.2)
            confidence = _signal_confidence(
                alpha=alpha,
                shock_score=shock.shock_score,
                r_squared=r2,
                beta_std_error=std_err,
                beta_value=beta_adj,
            )

            signal = AlphaSignal(
                ticker=ticker,
                commodity=commodity,
                role=role,
                beta_adjusted=beta_adj,
                shock_return=shock.price_return,
                expected_reaction=expected_reaction,
                actual_reaction=actual_reaction,
                alpha=alpha,
                direction=direction,
                shock_score=shock.shock_score,
                shock_age_days=shock.age_days,
                confidence=confidence,
            )
            signals.append(signal)

    # Sort by |alpha| descending
    signals.sort(key=lambda s: abs(s.alpha), reverse=True)
    return signals


def signals_to_dataframe(signals: List[AlphaSignal]) -> pd.DataFrame:
    """Convert signal list to a summary DataFrame."""
    if not signals:
        return pd.DataFrame()
    rows = [
        {
            "ticker": s.ticker,
            "commodity": s.commodity,
            "role": s.role,
            "direction": s.direction,
            "alpha_%": s.alpha * 100,
            "expected_%": s.expected_reaction * 100,
            "actual_%": s.actual_reaction * 100,
            "beta_adj": s.beta_adjusted,
            "shock_return_%": s.shock_return * 100,
            "shock_score": s.shock_score,
            "shock_age_days": s.shock_age_days,
            "confidence": s.confidence,
        }
        for s in signals
    ]
    return pd.DataFrame(rows)
