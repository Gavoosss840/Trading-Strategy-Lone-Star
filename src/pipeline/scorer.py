"""
Step 7 — Final Scoring
─────────────────────────
Computes a composite trade score for each filtered signal:

    Score = w_α × α_score
          + w_shock × shock_score
          + w_lev × leverage_quality
          + w_mar × margin_quality
          + w_tim × timing_score

Each component is normalised to [0, 1].
The final score drives position sizing and signal ranking.

Score interpretation:
  ≥ 0.70  → Strong signal (full position)
  0.50–0.70 → Moderate signal (half position)
  0.30–0.50 → Weak signal (quarter position, monitor only)
  < 0.30  → Discard
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.pipeline.alpha_calculator import AlphaSignal
from src.data.fundamental_data import (
    leverage_quality_score,
    margin_quality_score,
    coverage_quality_score,
)


# ── Scored signal ─────────────────────────────────────────────────────────────

@dataclass
class ScoredSignal:
    signal: AlphaSignal

    # Component scores
    alpha_score: float
    shock_score: float
    leverage_score: float
    margin_score: float
    timing_score: float

    # Composite
    total_score: float

    # Position guidance
    position_size_pct: float   # Recommended % of portfolio (0–5%)
    signal_strength: str       # "strong" | "moderate" | "weak" | "discard"

    @property
    def ticker(self) -> str:
        return self.signal.ticker

    @property
    def commodity(self) -> str:
        return self.signal.commodity

    @property
    def direction(self) -> str:
        return self.signal.direction

    @property
    def alpha(self) -> float:
        return self.signal.alpha

    def __repr__(self) -> str:
        return (
            f"ScoredSignal({self.ticker} | {self.direction} | "
            f"score={self.total_score:.3f} [{self.signal_strength}] | "
            f"α={self.alpha:.3%} | size={self.position_size_pct:.1%})"
        )


# ── Component score normalisation ─────────────────────────────────────────────

def _alpha_score(alpha: float, reference_alpha: float = 0.03) -> float:
    """
    Normalise alpha to [0, 1].
    reference_alpha = 3% is treated as a "full score" alpha.
    """
    return float(np.clip(abs(alpha) / reference_alpha, 0.0, 1.0))


def _timing_score(shock_age_days: int, max_age: int = 5) -> float:
    """
    Timing score decays with shock age.
    Age 0 = 1.0 (today's shock), age max_age = 0.1
    """
    if shock_age_days >= max_age:
        return 0.1
    return float(1.0 - (shock_age_days / max_age) * 0.9)


def _position_size(
    score: float,
    max_position_pct: float = 0.05,
    min_score: float = 0.30,
) -> float:
    """
    Map score to position size.
    Linear scaling from 0% at min_score to max_position_pct at 1.0.
    """
    if score < min_score:
        return 0.0
    size_ratio = (score - min_score) / (1.0 - min_score)
    return float(np.clip(size_ratio * max_position_pct, 0.0, max_position_pct))


def _signal_strength(score: float) -> str:
    if score >= 0.70:
        return "strong"
    elif score >= 0.50:
        return "moderate"
    elif score >= 0.30:
        return "weak"
    return "discard"


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_signal(
    signal: AlphaSignal,
    fundamentals: Optional[Dict[str, float]],
    cfg_scoring: Dict,
    cfg_risk: Dict,
    cfg_shock: Optional[Dict] = None,
) -> ScoredSignal:
    """
    Compute composite score for a single alpha signal.

    Args:
        signal        : AlphaSignal from Step 5
        fundamentals  : Dict with debt_to_equity, operating_margin,
                        interest_coverage for signal.ticker
        cfg_scoring   : scoring section from config
        cfg_risk      : risk section from config
        cfg_shock     : shock section from config (for max_age)
    """
    w_alpha = cfg_scoring.get("alpha_weight", 0.40)
    w_shock = cfg_scoring.get("shock_weight", 0.20)
    w_lev = cfg_scoring.get("leverage_quality_weight", 0.15)
    w_mar = cfg_scoring.get("margin_quality_weight", 0.15)
    w_tim = cfg_scoring.get("timing_weight", 0.10)
    max_pos = cfg_risk.get("max_position_pct", 0.05)
    min_score = cfg_scoring.get("min_score_to_trade", 0.45)
    max_age = (cfg_shock or {}).get("max_age_days", 5)

    # Component scores
    a_score = _alpha_score(signal.alpha)
    s_score = float(np.clip(signal.shock_score, 0.0, 1.0))

    fund = fundamentals or {}
    l_score = leverage_quality_score(fund.get("debt_to_equity", 1.0))
    m_score = margin_quality_score(fund.get("operating_margin", 0.10))
    t_score = _timing_score(signal.shock_age_days, max_age)

    # Composite
    total = (
        w_alpha * a_score
        + w_shock * s_score
        + w_lev * l_score
        + w_mar * m_score
        + w_tim * t_score
    )
    total = float(np.clip(total, 0.0, 1.0))

    pos_size = _position_size(total, max_pos, min_score)

    return ScoredSignal(
        signal=signal,
        alpha_score=a_score,
        shock_score=s_score,
        leverage_score=l_score,
        margin_score=m_score,
        timing_score=t_score,
        total_score=total,
        position_size_pct=pos_size,
        signal_strength=_signal_strength(total),
    )


def score_all_signals(
    signals: List[AlphaSignal],
    fundamentals_df: pd.DataFrame,
    cfg_scoring: Dict,
    cfg_risk: Dict,
    cfg_shock: Optional[Dict] = None,
    top_n: int = 20,
) -> List[ScoredSignal]:
    """
    Score all filtered signals and return top_n ranked by total_score.

    Args:
        signals        : List of filtered AlphaSignal
        fundamentals_df: DataFrame with ticker index (from Step 2 data)
        cfg_scoring    : scoring section from config
        cfg_risk       : risk section from config
        top_n          : Return top N signals only
    """
    min_score = cfg_scoring.get("min_score_to_trade", 0.45)
    scored = []

    for signal in signals:
        fund = (
            fundamentals_df.loc[signal.ticker].to_dict()
            if (not fundamentals_df.empty and signal.ticker in fundamentals_df.index)
            else None
        )
        ss = score_signal(signal, fund, cfg_scoring, cfg_risk, cfg_shock)
        if ss.total_score >= cfg_scoring.get("min_display_score", 0.30):
            scored.append(ss)

    # Sort by score
    scored.sort(key=lambda s: s.total_score, reverse=True)

    # Deduplicate: only the best signal per ticker
    seen_tickers = set()
    deduped = []
    for ss in scored:
        if ss.ticker not in seen_tickers:
            seen_tickers.add(ss.ticker)
            deduped.append(ss)

    return deduped[:top_n]


def scored_signals_to_dataframe(scored: List[ScoredSignal]) -> pd.DataFrame:
    """Convert scored signals to a display-ready DataFrame."""
    if not scored:
        return pd.DataFrame()
    rows = [
        {
            "Ticker": ss.ticker,
            "Commodity": ss.commodity,
            "Role": ss.signal.role,
            "Direction": ss.direction,
            "Alpha %": f"{ss.alpha * 100:+.2f}%",
            "Score": f"{ss.total_score:.3f}",
            "Strength": ss.signal_strength,
            "Size %": f"{ss.position_size_pct * 100:.1f}%",
            "Shock Score": f"{ss.shock_score:.3f}",
            "α Score": f"{ss.alpha_score:.3f}",
            "Lev. Score": f"{ss.leverage_score:.3f}",
            "Timing": f"{ss.timing_score:.3f}",
            "Age (d)": ss.signal.shock_age_days,
        }
        for ss in scored
    ]
    return pd.DataFrame(rows)
