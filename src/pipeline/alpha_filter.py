"""
Step 6 — Critical Alpha Filter: "Is There Still Alpha?"
─────────────────────────────────────────────────────────
Before scoring, we must verify the alpha is still exploitable.
We eliminate two categories of bad signals:

  1. Already-priced moves:
     The market has already absorbed the shock. Actual reaction ≥ threshold × expected.
     → α decaying, position entry is too late.

  2. Exaggerated reactions:
     The stock moved far MORE than expected — possible news specific to the stock,
     unrelated corporate event, or thin liquidity distortion.
     → α may be "noise", not commodity-driven.

  3. Stale shocks:
     Shock is too old — the market has had time to price it in.
     → Reduced probability of mean reversion.

  4. Weak statistical basis:
     Beta R² is too low — the historical relationship is not reliable enough
     to generate a tradeable signal.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.pipeline.alpha_calculator import AlphaSignal


# ── Filter functions ───────────────────────────────────────────────────────────

def _is_already_priced(signal: AlphaSignal, already_priced_ratio: float = 0.85) -> bool:
    """
    True if the stock has already reacted by more than `already_priced_ratio`
    of the expected move. Not enough residual alpha left to trade.

    Example: expected +5%, actual +4.5% → 90% priced → skip if ratio=0.85
    """
    if signal.expected_reaction == 0:
        return True
    reaction_ratio = signal.actual_reaction / signal.expected_reaction
    return reaction_ratio >= already_priced_ratio


def _is_exaggerated(signal: AlphaSignal, max_alpha_pct: float = 0.15) -> bool:
    """
    True if the stock moved FAR more than expected in the OPPOSITE direction
    of the shock (over-shoot beyond max_alpha).

    This protects against cases where an unrelated event (earnings, M&A news)
    is distorting the stock, making the commodity-driven alpha measurement noisy.
    """
    return abs(signal.alpha) > max_alpha_pct


def _is_stale(signal: AlphaSignal, max_age: int = 5, decay_halflife: int = 3) -> Tuple[bool, float]:
    """
    Assess staleness of the shock.

    Returns (is_too_stale, decay_multiplier).
    Decay multiplier reduces alpha estimate based on shock age.
    """
    if signal.shock_age_days > max_age:
        return True, 0.0

    # Exponential decay: at age=0, multiplier=1.0; at age=halflife, multiplier=0.5
    decay = 0.5 ** (signal.shock_age_days / max(decay_halflife, 1))
    return False, float(decay)


def _has_sufficient_beta_quality(signal: AlphaSignal, min_shock_score: float = 0.1) -> bool:
    """Minimum shock quality check."""
    return signal.shock_score >= min_shock_score


# ── Main filter pipeline ──────────────────────────────────────────────────────

class FilterResult:
    """Wraps a filtered signal with metadata about why it passed/failed."""
    __slots__ = ("signal", "passed", "reason", "decay_multiplier")

    def __init__(
        self,
        signal: AlphaSignal,
        passed: bool,
        reason: str = "",
        decay_multiplier: float = 1.0,
    ) -> None:
        self.signal = signal
        self.passed = passed
        self.reason = reason
        self.decay_multiplier = decay_multiplier


def apply_filters(
    signals: List[AlphaSignal],
    cfg: Dict,
) -> Tuple[List[AlphaSignal], List[FilterResult]]:
    """
    Apply all alpha filters to the signal list.

    Returns:
        (passed_signals, all_filter_results)
    """
    already_priced_ratio = cfg.get("already_priced_ratio", 0.85)
    max_alpha_pct = cfg.get("max_alpha_pct", 15.0) / 100
    max_age = cfg.get("max_age_days", 5)  # from shock config
    decay_halflife = cfg.get("decay_halflife_days", 3)
    min_shock_score = 0.10

    results: List[FilterResult] = []
    passed: List[AlphaSignal] = []

    for signal in signals:
        # Filter 1: Shock quality
        if not _has_sufficient_beta_quality(signal, min_shock_score):
            results.append(FilterResult(signal, False, "weak_shock_score"))
            continue

        # Filter 2: Already priced
        if _is_already_priced(signal, already_priced_ratio):
            results.append(FilterResult(signal, False, "already_priced"))
            continue

        # Filter 3: Exaggerated reaction
        if _is_exaggerated(signal, max_alpha_pct):
            results.append(FilterResult(signal, False, "exaggerated_reaction"))
            continue

        # Filter 4: Stale shock
        is_stale, decay = _is_stale(signal, max_age, decay_halflife)
        if is_stale:
            results.append(FilterResult(signal, False, "stale_shock"))
            continue

        # Passed all filters — apply decay to alpha
        if decay < 1.0:
            object.__setattr__(signal, "alpha", signal.alpha * decay) if hasattr(signal, "__setattr__") else None
            # Dataclass is mutable, update in place
            signal.alpha = signal.alpha  # alpha already set — apply decay below

        results.append(FilterResult(signal, True, "passed", decay_multiplier=decay))
        passed.append(signal)

    return passed, results


def filter_signals(
    signals: List[AlphaSignal],
    cfg: Dict,
) -> List[AlphaSignal]:
    """
    Main entry point. Returns only signals that pass all filters.
    Applies shock-age decay to the alpha estimate.
    """
    already_priced_ratio = cfg.get("already_priced_ratio", 0.85)
    max_alpha_pct = cfg.get("max_alpha_pct", 15.0) / 100
    max_age = cfg.get("max_age_days", 5)
    decay_halflife = cfg.get("decay_halflife_days", 3)

    passed = []
    for signal in signals:
        # Shock quality gate
        if signal.shock_score < 0.10:
            continue

        # Already priced
        if (
            signal.expected_reaction != 0
            and signal.actual_reaction / signal.expected_reaction >= already_priced_ratio
        ):
            continue

        # Exaggerated
        if abs(signal.alpha) > max_alpha_pct:
            continue

        # Stale
        if signal.shock_age_days > max_age:
            continue

        # Apply age-decay to alpha
        decay = 0.5 ** (signal.shock_age_days / max(decay_halflife, 1))
        signal.alpha = signal.alpha * decay

        # Re-check minimum alpha after decay
        min_alpha = 0.005  # 0.5%
        if abs(signal.alpha) < min_alpha:
            continue

        passed.append(signal)

    return passed


def filter_summary(all_results: List[FilterResult]) -> Dict[str, int]:
    """Return counts by filter reason."""
    summary: Dict[str, int] = {}
    for r in all_results:
        reason = "passed" if r.passed else r.reason
        summary[reason] = summary.get(reason, 0) + 1
    return summary
