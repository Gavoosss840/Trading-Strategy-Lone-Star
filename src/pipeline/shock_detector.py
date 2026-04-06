"""
Step 4 — Commodity Shock Detection
─────────────────────────────────────
Identifies significant, exploitable commodity price moves.

A "shock" is characterised by three dimensions:
  1. Amplitude  : |daily move| vs minimum threshold
  2. Speed      : Single-day vs multi-day accumulation (faster = higher score)
  3. Surprise   : Z-score vs historical volatility baseline

Shock Score = amplitude × speed_factor × tanh(surprise_z / 2)
             (bounded to [0, 1] for cleaner composition downstream)

Only shocks within the last `max_age_days` window are considered actionable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class CommodityShock:
    commodity: str
    date: pd.Timestamp
    price_return: float      # Raw commodity return on shock date
    amplitude: float         # |price_return| as decimal
    speed_factor: float      # 1.0 = single-day, decays for multi-day
    surprise_z: float        # |return| / historical_vol (z-score)
    shock_score: float       # Composite score [0, 1]
    direction: str           # "up" or "down"
    age_days: int            # Days since shock (0 = today)

    @property
    def is_bullish(self) -> bool:
        return self.direction == "up"

    def __repr__(self) -> str:
        return (
            f"CommodityShock({self.commodity} | {self.date.date()} | "
            f"{'+' if self.is_bullish else ''}{self.price_return:.2%} | "
            f"score={self.shock_score:.3f} | age={self.age_days}d)"
        )


# ── Shock scoring engine ──────────────────────────────────────────────────────

def _speed_factor(n_days: int, decay: float = 0.7) -> float:
    """
    Decay speed factor for multi-day shocks.

    Single day (n=1): factor = 1.0
    Two days  (n=2): factor = 0.7
    Three days(n=3): factor = 0.49
    etc.
    """
    return float(decay ** (n_days - 1))


def _surprise_score(ret: float, hist_vol_daily: float) -> float:
    """
    Z-score of the return vs historical daily volatility.
    Returns z ≥ 0 (absolute surprise).
    """
    if hist_vol_daily <= 0 or np.isnan(hist_vol_daily):
        return 0.0
    return abs(ret) / hist_vol_daily


def _composite_shock_score(
    amplitude: float,
    speed_factor: float,
    surprise_z: float,
    min_amplitude: float = 0.015,
    min_zscore: float = 1.5,
) -> float:
    """
    Composite shock score in [0, 1].

    Score = amplitude_norm × speed_factor × tanh(surprise_z / 2)

    amplitude_norm: normalised amplitude relative to 5% reference
    tanh clamps the surprise contribution to (0, 1)
    """
    if amplitude < min_amplitude or surprise_z < min_zscore:
        return 0.0

    amplitude_norm = np.clip(amplitude / 0.05, 0.0, 2.0) / 2.0  # 5% = 0.5 score
    surprise_component = np.tanh(surprise_z / 2)

    score = amplitude_norm * speed_factor * surprise_component
    return float(np.clip(score, 0.0, 1.0))


# ── Shock detection ────────────────────────────────────────────────────────────

def detect_shocks(
    commodity_returns: pd.DataFrame,
    cfg: Dict,
    lookback_window: Optional[int] = None,
) -> List[CommodityShock]:
    """
    Scan commodity return series for recent shocks.

    Args:
        commodity_returns : DataFrame [dates × commodity_keys] of log returns
        cfg               : shock section from config.yaml
        lookback_window   : How many days of returns to check for shocks.
                            Defaults to max_age_days from config.

    Returns:
        List of CommodityShock objects, sorted by shock_score descending.
    """
    min_amplitude = cfg.get("min_amplitude_pct", 1.5) / 100
    min_zscore = cfg.get("min_zscore", 1.5)
    vol_window = cfg.get("lookback_vol_days", 63)
    speed_decay = cfg.get("speed_decay", 0.7)
    max_age = cfg.get("max_age_days", 5)

    if lookback_window is None:
        lookback_window = max_age

    shocks: List[CommodityShock] = []

    for commodity in commodity_returns.columns:
        series = commodity_returns[commodity].dropna()
        if len(series) < vol_window + 5:
            continue

        # Historical volatility baseline (daily std, NOT annualised)
        hist_vol = series.rolling(vol_window).std()

        # Scan the last `lookback_window` + buffer days
        scan_series = series.iloc[-(lookback_window + 5):]

        for i, (date, ret) in enumerate(scan_series.items()):
            age_days = len(scan_series) - 1 - i  # 0 = most recent date
            if age_days > max_age:
                continue

            amplitude = abs(ret)
            vol_today = hist_vol.get(date, np.nan)

            if np.isnan(vol_today) or vol_today == 0:
                continue

            surprise_z = _surprise_score(ret, vol_today)
            sf = _speed_factor(1, speed_decay)  # Single-day shock detection

            score = _composite_shock_score(
                amplitude=amplitude,
                speed_factor=sf,
                surprise_z=surprise_z,
                min_amplitude=min_amplitude,
                min_zscore=min_zscore,
            )

            if score > 0:
                shock = CommodityShock(
                    commodity=commodity,
                    date=date,
                    price_return=float(ret),
                    amplitude=amplitude,
                    speed_factor=sf,
                    surprise_z=surprise_z,
                    shock_score=score,
                    direction="up" if ret > 0 else "down",
                    age_days=age_days,
                )
                shocks.append(shock)

    # Sort by score descending, then by recency
    shocks.sort(key=lambda s: (s.shock_score, -s.age_days), reverse=True)

    # Deduplicate: keep best shock per commodity per window
    seen: Dict[str, CommodityShock] = {}
    deduped = []
    for shock in shocks:
        key = shock.commodity
        if key not in seen:
            seen[key] = shock
            deduped.append(shock)
        elif shock.shock_score > seen[key].shock_score:
            seen[key] = shock
            deduped = [s for s in deduped if s.commodity != key] + [shock]

    return deduped


def get_active_shocks(
    commodity_returns: pd.DataFrame,
    cfg: Dict,
) -> List[CommodityShock]:
    """
    Return shocks that are currently actionable (within max_age_days).
    Primary entry point for the strategy pipeline.
    """
    return detect_shocks(commodity_returns, cfg)


def shocks_to_dataframe(shocks: List[CommodityShock]) -> pd.DataFrame:
    """Convert shock list to a summary DataFrame."""
    if not shocks:
        return pd.DataFrame(
            columns=["commodity", "date", "price_return", "amplitude",
                     "surprise_z", "shock_score", "direction", "age_days"]
        )
    rows = [
        {
            "commodity": s.commodity,
            "date": s.date,
            "price_return": s.price_return,
            "amplitude_%": s.amplitude * 100,
            "surprise_z": s.surprise_z,
            "shock_score": s.shock_score,
            "direction": s.direction,
            "age_days": s.age_days,
        }
        for s in shocks
    ]
    return pd.DataFrame(rows).set_index("commodity")


# ── Multi-day shock aggregation ───────────────────────────────────────────────

def accumulate_shock(
    commodity_returns: pd.DataFrame,
    commodity: str,
    n_days: int,
    vol_window: int = 63,
    speed_decay: float = 0.7,
) -> Optional[float]:
    """
    Aggregate return over n_days and score as a multi-day shock.
    Used when a commodity has been drifting consistently in one direction.
    """
    series = commodity_returns[commodity].dropna()
    if len(series) < vol_window + n_days:
        return None

    hist_vol = series.rolling(vol_window).std().iloc[-1]
    recent_ret = series.iloc[-n_days:].sum()  # Approximate compounded return

    amplitude = abs(recent_ret)
    surprise_z = _surprise_score(recent_ret, hist_vol * np.sqrt(n_days))
    sf = _speed_factor(n_days, speed_decay)

    return _composite_shock_score(amplitude, sf, surprise_z)
