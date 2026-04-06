"""
Step 7 — Final Scoring
─────────────────────────
Computes a composite trade score for each filtered signal, then derives
position size via the CML + Risk Parity framework.

── Composite score ────────────────────────────────────────────────────────
    Score = w_α × α_score
          + w_shock × shock_score
          + w_lev × leverage_quality
          + w_mar × margin_quality
          + w_tim × timing_score

── Position sizing (CML + inverse-vol) ────────────────────────────────────
    σ_hold    = σ_résiduel_annual × √(holding_days / 252)
    SR_signal = |alpha| / σ_hold
    base_size = target_risk / σ_résiduel_annual    ← risk-parity baseline
    boost     = min(SR_signal / SR_ref, max_boost)  ← CML quality lift
    w         = base_size × boost × composite_score
    w         = clip(w, min_pos, max_pos)

Economic interpretation:
  - base_size : inverse-vol allocation — each position targets the same
                idiosyncratic risk contribution (risk parity)
  - boost     : multiplier when the trade's SR exceeds a reference SR,
                positioning the portfolio on the Capital Market Line
  - score     : quality gate from the 7-step pipeline

Score interpretation:
  ≥ 0.70  → Strong    (full boost applied)
  0.50–0.70 → Moderate
  0.30–0.50 → Weak
  < 0.30  → Discard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.pipeline.alpha_calculator import AlphaSignal
from src.data.fundamental_data import (
    leverage_quality_score,
    margin_quality_score,
)


# ── Scored signal ─────────────────────────────────────────────────────────────

@dataclass
class ScoredSignal:
    signal: AlphaSignal

    # Component scores [0, 1]
    alpha_score: float
    shock_score: float
    leverage_score: float
    margin_score: float
    timing_score: float

    # Composite pipeline score
    total_score: float
    signal_strength: str        # "strong" | "moderate" | "weak" | "discard"

    # CML + risk-parity sizing breakdown
    residual_vol_annual: float  # σ_résiduel (annualised)
    sharpe_signal: float        # SR of the trade over the holding period
    base_size: float            # inverse-vol allocation (risk parity baseline)
    size_boost: float           # CML quality multiplier
    position_size_pct: float    # Final recommended position (fraction of portfolio)

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
            f"α={self.alpha:.3%} | SR={self.sharpe_signal:.2f} | "
            f"size={self.position_size_pct:.1%})"
        )


# ── CML + Risk Parity position sizing ────────────────────────────────────────

def cml_position_size(
    alpha: float,
    residual_vol_annual: float,
    composite_score: float,
    holding_days: int = 5,
    target_risk: float = 0.01,
    reference_sharpe: float = 0.50,
    max_boost: float = 2.0,
    max_pos: float = 0.05,
    min_pos: float = 0.005,
) -> Dict[str, float]:
    """
    CML-informed inverse-vol position sizing.

    Args:
        alpha               : Expected residual return over holding period (decimal)
        residual_vol_annual : Annualised CAPM-residual volatility of the stock
        composite_score     : Pipeline quality score [0, 1]
        holding_days        : Expected holding period (days)
        target_risk         : Target idiosyncratic risk per position (annual, decimal)
                              e.g. 0.01 = 1% annual vol contribution
        reference_sharpe    : Reference SR — positions with SR > this get a boost
        max_boost           : Cap on the signal quality multiplier
        max_pos             : Hard maximum position size
        min_pos             : Hard minimum position size

    Returns:
        Dict with sizing breakdown for transparency.
    """
    fallback = {
        "position_size_pct": float(np.clip(composite_score * max_pos, min_pos, max_pos)),
        "base_size": max_pos,
        "size_boost": 1.0,
        "sharpe_signal": 0.0,
        "residual_vol_annual": residual_vol_annual,
    }

    if residual_vol_annual <= 0 or np.isnan(residual_vol_annual):
        return fallback

    # σ over the holding period
    sigma_hold = residual_vol_annual * np.sqrt(max(holding_days, 1) / 252)
    sigma_hold = max(sigma_hold, 1e-4)

    # Sharpe of the signal over the holding period
    sharpe_signal = abs(alpha) / sigma_hold

    # Base allocation: inverse-vol (risk parity baseline)
    # Each position contributes `target_risk` of annualised idiosyncratic vol
    base_size = target_risk / residual_vol_annual
    base_size = float(np.clip(base_size, 0.0, max_pos))

    # CML boost: scale up when signal SR > reference SR
    boost = float(np.clip(sharpe_signal / reference_sharpe, 0.0, max_boost))

    # Final: risk-parity × signal quality × pipeline score
    w = base_size * boost * composite_score
    w = float(np.clip(w, min_pos, max_pos))

    return {
        "position_size_pct": w,
        "base_size": base_size,
        "size_boost": boost,
        "sharpe_signal": sharpe_signal,
        "residual_vol_annual": residual_vol_annual,
    }


# ── Score component helpers ───────────────────────────────────────────────────

def _alpha_score(alpha: float, reference_alpha: float = 0.03) -> float:
    return float(np.clip(abs(alpha) / reference_alpha, 0.0, 1.0))


def _timing_score(shock_age_days: int, max_age: int = 5) -> float:
    if shock_age_days >= max_age:
        return 0.1
    return float(1.0 - (shock_age_days / max_age) * 0.9)


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
    residual_vol_annual: float = 0.0,
) -> ScoredSignal:
    """
    Compute composite score and CML-informed position size for one signal.

    Args:
        signal               : AlphaSignal from Step 5
        fundamentals         : Dict with debt_to_equity, operating_margin for ticker
        cfg_scoring          : scoring section from config
        cfg_risk             : risk section from config
        cfg_shock            : shock section from config (for max_age, holding_days)
        residual_vol_annual  : Annualised CAPM-residual vol (from noise_cleaner)
    """
    w_alpha = cfg_scoring.get("alpha_weight", 0.40)
    w_shock = cfg_scoring.get("shock_weight", 0.20)
    w_lev   = cfg_scoring.get("leverage_quality_weight", 0.15)
    w_mar   = cfg_scoring.get("margin_quality_weight", 0.15)
    w_tim   = cfg_scoring.get("timing_weight", 0.10)

    max_age  = (cfg_shock or {}).get("max_age_days", 5)
    max_pos  = cfg_risk.get("max_position_pct", 0.05)
    min_pos  = cfg_risk.get("min_position_pct", 0.005)

    # CML sizing params
    target_risk      = cfg_risk.get("target_position_risk_pct", 0.01)
    ref_sharpe       = cfg_risk.get("reference_sharpe", 0.50)
    max_boost        = cfg_risk.get("max_signal_boost", 2.0)
    holding_days     = (cfg_shock or {}).get("max_age_days", 5)

    # ── Component scores ──────────────────────────────────────────────────
    a_score = _alpha_score(signal.alpha)
    s_score = float(np.clip(signal.shock_score, 0.0, 1.0))

    fund = fundamentals or {}
    l_score = leverage_quality_score(fund.get("debt_to_equity", 1.0))
    m_score = margin_quality_score(fund.get("operating_margin", 0.10))
    t_score = _timing_score(signal.shock_age_days, max_age)

    # ── Composite ─────────────────────────────────────────────────────────
    total = float(np.clip(
        w_alpha * a_score + w_shock * s_score + w_lev * l_score
        + w_mar * m_score + w_tim * t_score,
        0.0, 1.0,
    ))

    # ── CML + risk-parity sizing ──────────────────────────────────────────
    sizing = cml_position_size(
        alpha=signal.alpha,
        residual_vol_annual=residual_vol_annual,
        composite_score=total,
        holding_days=holding_days,
        target_risk=target_risk,
        reference_sharpe=ref_sharpe,
        max_boost=max_boost,
        max_pos=max_pos,
        min_pos=min_pos,
    )

    return ScoredSignal(
        signal=signal,
        alpha_score=a_score,
        shock_score=s_score,
        leverage_score=l_score,
        margin_score=m_score,
        timing_score=t_score,
        total_score=total,
        signal_strength=_signal_strength(total),
        residual_vol_annual=sizing["residual_vol_annual"],
        sharpe_signal=sizing["sharpe_signal"],
        base_size=sizing["base_size"],
        size_boost=sizing["size_boost"],
        position_size_pct=sizing["position_size_pct"],
    )


def score_all_signals(
    signals: List[AlphaSignal],
    fundamentals_df: pd.DataFrame,
    cfg_scoring: Dict,
    cfg_risk: Dict,
    cfg_shock: Optional[Dict] = None,
    residual_vol_map: Optional[Dict[str, float]] = None,
    top_n: int = 20,
) -> List[ScoredSignal]:
    """
    Score all filtered signals and return top_n ranked by total_score.

    Args:
        signals          : List of filtered AlphaSignal
        fundamentals_df  : DataFrame with ticker index
        cfg_scoring      : scoring section from config
        cfg_risk         : risk section from config
        residual_vol_map : {ticker: annualised_residual_vol} from noise_cleaner
        top_n            : Return top N signals only
    """
    vol_map = residual_vol_map or {}
    min_display = cfg_scoring.get("min_display_score", 0.30)
    scored = []

    for signal in signals:
        fund = (
            fundamentals_df.loc[signal.ticker].to_dict()
            if (not fundamentals_df.empty and signal.ticker in fundamentals_df.index)
            else None
        )
        resid_vol = vol_map.get(signal.ticker, 0.0)

        ss = score_signal(
            signal=signal,
            fundamentals=fund,
            cfg_scoring=cfg_scoring,
            cfg_risk=cfg_risk,
            cfg_shock=cfg_shock,
            residual_vol_annual=resid_vol,
        )
        if ss.total_score >= min_display:
            scored.append(ss)

    # Sort by composite score descending
    scored.sort(key=lambda s: s.total_score, reverse=True)

    # Keep only the best signal per ticker (deduplicate)
    seen: set = set()
    deduped = []
    for ss in scored:
        if ss.ticker not in seen:
            seen.add(ss.ticker)
            deduped.append(ss)

    return deduped[:top_n]


def scored_signals_to_dataframe(scored: List[ScoredSignal]) -> pd.DataFrame:
    """Convert scored signals to a display-ready DataFrame."""
    if not scored:
        return pd.DataFrame()
    rows = [
        {
            "Ticker":      ss.ticker,
            "Commodity":   ss.commodity,
            "Role":        ss.signal.role,
            "Direction":   ss.direction,
            "Alpha %":     f"{ss.alpha * 100:+.2f}%",
            "Score":       f"{ss.total_score:.3f}",
            "Strength":    ss.signal_strength,
            "σ resid":     f"{ss.residual_vol_annual * 100:.1f}%",
            "SR signal":   f"{ss.sharpe_signal:.2f}",
            "Base size":   f"{ss.base_size * 100:.1f}%",
            "Boost":       f"{ss.size_boost:.2f}×",
            "Size %":      f"{ss.position_size_pct * 100:.1f}%",
            "Shock Score": f"{ss.shock_score:.3f}",
            "Timing":      f"{ss.timing_score:.3f}",
            "Age (d)":     ss.signal.shock_age_days,
        }
        for ss in scored
    ]
    return pd.DataFrame(rows)
