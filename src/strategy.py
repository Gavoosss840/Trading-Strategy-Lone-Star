"""
Lone Star Strategy — Main Orchestrator
────────────────────────────────────────
Chains all 7 pipeline steps into a single run() call.

Pipeline:
  Step 1  — Exposure Mapping      : β_commodity for each stock
  Step 2  — Fundamental Adjustment: β_adjusted via MM + financial ratios
  Step 3  — Noise Cleaning        : Residual = R_stock − R_explained (FF5/CAPM)
  Step 4  — Shock Detection       : Identify commodity shocks (amplitude/surprise)
  Step 5  — Alpha Calculation     : Alpha = Expected − Actual reaction
  Step 6  — Alpha Filter          : Remove already-priced / exaggerated moves
  Step 7  — Final Scoring         : Score = f(α, shock, leverage, margins, timing)

Usage:
    from src.strategy import LoneStarStrategy
    strategy = LoneStarStrategy.from_config("config/config.yaml")
    result = strategy.run()
    result.print_signals()
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data.market_data import load_all_data
from src.data.fundamental_data import get_fundamentals_bulk, compute_fundamental_quality
from src.pipeline.exposure_mapping import get_latest_betas, ExposureMap
from src.pipeline.fundamental_adjustment import adjust_exposure_map
from src.pipeline.noise_cleaner import compute_residuals, cumulative_residual, residual_volatility
from src.pipeline.shock_detector import get_active_shocks, shocks_to_dataframe
from src.pipeline.alpha_calculator import generate_alpha_signals, signals_to_dataframe
from src.pipeline.alpha_filter import filter_signals
from src.pipeline.scorer import score_all_signals, scored_signals_to_dataframe, ScoredSignal

warnings.filterwarnings("ignore")


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    scored_signals: List[ScoredSignal] = field(default_factory=list)
    exposure_map: Optional[ExposureMap] = None
    adjusted_betas: pd.DataFrame = field(default_factory=pd.DataFrame)
    active_shocks_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fundamentals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # Raw market data — stored so the backtest engine (report_builder) can
    # reuse the same download without fetching again.  Also makes market_returns
    # explicitly available for any downstream analytics step.
    stock_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_volumes: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_returns_data: pd.Series = field(default_factory=pd.Series)

    def signals_dataframe(self) -> pd.DataFrame:
        df = scored_signals_to_dataframe(self.scored_signals)
        if df.empty or self.stock_metadata.empty:
            return df
        # Inject zone and company name
        meta = self.stock_metadata[["name", "zone"]].rename(columns={"name": "Company"})
        df = df.merge(meta, left_on="Ticker", right_index=True, how="left")
        # Reorder: Ticker, Company, Zone first
        front = ["Ticker", "Company", "zone", "Direction", "Commodity", "Role"]
        rest = [c for c in df.columns if c not in front]
        return df[[c for c in front if c in df.columns] + rest]

    def print_signals(self, top_n: int = 10) -> None:
        """Pretty-print the top scored signals to stdout."""
        try:
            from tabulate import tabulate
            use_tabulate = True
        except ImportError:
            use_tabulate = False

        print("\n" + "═" * 80)
        print("  ★  LONE STAR — Trade Signals")
        print("═" * 80)

        if self.active_shocks_df.empty:
            print("  No active commodity shocks detected.")
            print("═" * 80)
            return

        print("\n📦 Active Commodity Shocks:")
        shock_display = self.active_shocks_df[
            ["price_return", "amplitude_%", "surprise_z", "shock_score", "direction", "age_days"]
        ].copy() if not self.active_shocks_df.empty else self.active_shocks_df

        if not shock_display.empty:
            shock_display["price_return"] = shock_display["price_return"].map("{:+.2%}".format)
            shock_display["amplitude_%"] = shock_display["amplitude_%"].map("{:.2f}%".format)
            shock_display["surprise_z"] = shock_display["surprise_z"].map("{:.2f}".format)
            shock_display["shock_score"] = shock_display["shock_score"].map("{:.3f}".format)
            if use_tabulate:
                print(tabulate(shock_display, headers="keys", tablefmt="rounded_outline"))
            else:
                print(shock_display.to_string())

        if not self.scored_signals:
            print("\n  No tradeable signals after filtering.")
            print("═" * 80)
            return

        print(f"\n📊 Top {min(top_n, len(self.scored_signals))} Trade Signals:")
        df = self.signals_dataframe().head(top_n)
        if use_tabulate:
            print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
        else:
            print(df.to_string(index=False))

        longs = sum(1 for s in self.scored_signals if s.direction == "LONG")
        shorts = sum(1 for s in self.scored_signals if s.direction == "SHORT")
        print(f"\n  Summary: {longs} LONG  |  {shorts} SHORT  |  {len(self.scored_signals)} total signals")

        # Zone breakdown
        if not self.stock_metadata.empty:
            df_full = self.signals_dataframe()
            if "zone" in df_full.columns:
                zone_counts = df_full["zone"].value_counts().to_dict()
                zone_str = "  |  ".join(f"{z}: {n}" for z, n in zone_counts.items())
                print(f"  By zone: {zone_str}")

        print(f"  Pipeline completed in {self.run_time_seconds:.1f}s")
        print("═" * 80 + "\n")


# ── Strategy class ─────────────────────────────────────────────────────────────

class LoneStarStrategy:
    """
    Lone Star commodity-exposure alpha strategy.

    Attributes:
        cfg: Parsed YAML configuration dict.
    """

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "LoneStarStrategy":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_all_stock_tickers(self) -> List[str]:
        """Flatten all stock tickers from the geographic config into a unique list."""
        tickers = set()
        stocks_cfg = self.cfg.get("stocks", {})
        for zone_list in stocks_cfg.values():
            for entry in zone_list:
                tickers.add(entry["ticker"])
        return sorted(tickers)

    def _get_stock_metadata(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns [ticker, name, zone,
        primary_commodity, role] for every stock in the universe.
        """
        rows = []
        stocks_cfg = self.cfg.get("stocks", {})
        for zone, entries in stocks_cfg.items():
            for entry in entries:
                rows.append({
                    "ticker": entry["ticker"],
                    "name": entry.get("name", entry["ticker"]),
                    "zone": zone,
                    "primary_commodity": entry.get("primary_commodity", ""),
                    "role": entry.get("role", ""),
                })
        df = pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")
        return df

    def _print_step(self, step: int, name: str) -> None:
        print(f"  [{step}/7] {name}...", end=" ", flush=True)

    def _done(self, n: int = 0) -> None:
        suffix = f"({n} results)" if n else "✓"
        print(suffix)

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> StrategyResult:
        """
        Execute the full 7-step Lone Star pipeline.

        Returns:
            StrategyResult with scored signals and intermediate outputs.
        """
        t0 = time.time()
        result = StrategyResult()
        warn_list: List[str] = []

        commodity_cfg = self.cfg["commodities"]
        data_cfg = self.cfg["data"]
        beta_cfg = self.cfg["beta"]
        fund_cfg = self.cfg["fundamental"]
        shock_cfg = self.cfg["shock"]
        alpha_cfg = self.cfg["alpha"]
        scoring_cfg = self.cfg["scoring"]
        risk_cfg = self.cfg["risk"]
        output_cfg = self.cfg["output"]

        lookback = data_cfg.get("lookback_days", 504)
        shock_lookback = data_cfg.get("shock_lookback_days", 63)
        rolling_window = beta_cfg.get("rolling_window", 252)
        min_obs = data_cfg.get("min_observations", 120)

        stock_tickers = self._get_all_stock_tickers()
        stock_metadata = self._get_stock_metadata()
        result.stock_metadata = stock_metadata

        if verbose:
            print("\n" + "─" * 60)
            print("  ★  LONE STAR — Running Pipeline")
            print("─" * 60)
            zones = self.cfg.get("stocks", {}).keys()
            zone_counts = {
                z: len(self.cfg["stocks"][z]) for z in zones
            }
            zone_str = "  |  ".join(f"{z}: {n}" for z, n in zone_counts.items())
            print(f"  Universe: {len(stock_tickers)} stocks × {len(commodity_cfg)} commodities")
            print(f"  Zones: {zone_str}")

        # ── Data load ──────────────────────────────────────────────────────────
        if verbose:
            print("  [0/7] Loading market data...", end=" ", flush=True)
        data = load_all_data(
            stock_tickers=stock_tickers,
            commodity_config=commodity_cfg,
            lookback_days=lookback,
            progress=False,
        )
        stock_returns    = data["stock_returns"]
        stock_prices     = data["stock_prices"]       # needed for backtest + ADV filter
        stock_volumes    = data.get("stock_volumes", pd.DataFrame())  # ADV liquidity filter
        commodity_returns = data["commodity_returns"]
        market_returns   = data["market_returns"]     # passed to noise cleaning (CAPM / FF5)
        risk_free        = data["risk_free_rate"]
        ff_factors       = data["ff_factors"]

        # Make market data available on the result for downstream reuse
        # (e.g. report_builder can call run_backtest without re-downloading)
        result.stock_prices       = stock_prices
        result.stock_volumes      = stock_volumes
        result.market_returns_data = market_returns

        if verbose:
            actual_stocks = len(stock_returns.columns)
            actual_commodities = len(commodity_returns.columns)
            print(f"✓ ({actual_stocks} stocks, {actual_commodities} commodities)")

        if stock_returns.empty:
            warn_list.append("No stock return data loaded.")
            result.warnings = warn_list
            result.run_time_seconds = time.time() - t0
            return result

        # ── Step 1: Exposure Mapping ───────────────────────────────────────────
        if verbose:
            self._print_step(1, "Exposure mapping (beta regression)")
        exposure_map = get_latest_betas(
            stock_returns=stock_returns,
            commodity_returns=commodity_returns,
            rolling_window=rolling_window,
            min_observations=min_obs,
            outlier_zscore=beta_cfg.get("outlier_zscore_threshold", 3.5),
            market_returns=(
                market_returns if beta_cfg.get("use_market_adjusted", True) else None
            ),
        )
        result.exposure_map = exposure_map
        if verbose:
            sig_pairs = len(exposure_map.significant_pairs(min_r2=beta_cfg.get("min_r_squared", 0.05)))
            self._done(sig_pairs)

        if not exposure_map.results:
            warn_list.append("No beta estimates computed — check data availability.")
            result.warnings = warn_list
            result.run_time_seconds = time.time() - t0
            return result

        # ── Step 2: Fundamental Adjustment ────────────────────────────────────
        if verbose:
            self._print_step(2, "Fundamental beta adjustment")
        fundamentals_df = get_fundamentals_bulk(stock_tickers)
        result.fundamentals_df = fundamentals_df

        adjusted_betas = adjust_exposure_map(exposure_map, fundamentals_df, fund_cfg)
        result.adjusted_betas = adjusted_betas
        if verbose:
            self._done(len(adjusted_betas))

        # ── Step 3: Noise Cleaning ─────────────────────────────────────────────
        if verbose:
            self._print_step(3, "Market noise cleaning (Fama-French / CAPM)")
        residuals_df = compute_residuals(
            stock_returns=stock_returns,
            market_returns=market_returns,
            risk_free=risk_free if isinstance(risk_free, pd.Series) else None,
            ff_factors=ff_factors,
            rolling_window=rolling_window,
        )
        # Cumulative residual over the shock window (last shock_lookback days capped at 10)
        n_residual_days = min(shock_cfg.get("max_age_days", 5), 10)
        cum_residuals = cumulative_residual(residuals_df, n_days=n_residual_days)

        # Annualised idiosyncratic vol per ticker (used for CML sizing in Step 7)
        resid_vol_map = residual_volatility(residuals_df, window=63).to_dict()

        if verbose:
            self._done(len(cum_residuals))

        # ── Step 4: Shock Detection ────────────────────────────────────────────
        if verbose:
            self._print_step(4, "Commodity shock detection")
        # Use only the shock_lookback window of commodity returns for shock detection
        recent_commodity = commodity_returns.iloc[-shock_lookback:]
        active_shocks = get_active_shocks(recent_commodity, shock_cfg)
        result.active_shocks_df = shocks_to_dataframe(active_shocks)
        if verbose:
            self._done(len(active_shocks))

        if not active_shocks:
            if verbose:
                print("  No active commodity shocks — no signals generated.")
            result.warnings.append("No commodity shocks detected in the recent window.")
            result.run_time_seconds = time.time() - t0
            return result

        # ── Step 5: Alpha Calculation ──────────────────────────────────────────
        if verbose:
            self._print_step(5, "Alpha calculation (expected vs actual)")

        # Build beta_std_errors and r_squared_map from adjusted_betas
        r_squared_map = {}
        if not adjusted_betas.empty and "r_squared" in adjusted_betas.columns:
            for (ticker, commodity), row in adjusted_betas.iterrows():
                r_squared_map[(ticker, commodity)] = row.get("r_squared", 0.0)

        raw_signals = generate_alpha_signals(
            shocks=active_shocks,
            adjusted_betas=adjusted_betas,
            cumulative_residuals=cum_residuals,
            cfg=alpha_cfg,
            r_squared_map=r_squared_map if r_squared_map else None,
        )
        result.all_signals_df = signals_to_dataframe(raw_signals)
        if verbose:
            self._done(len(raw_signals))

        # ── Step 6: Alpha Filter ───────────────────────────────────────────────
        if verbose:
            self._print_step(6, "Alpha filter (remove priced / exaggerated moves)")

        # Merge shock config into alpha config for filter
        filter_cfg = {**alpha_cfg, "max_age_days": shock_cfg.get("max_age_days", 5)}
        filtered_signals = filter_signals(raw_signals, filter_cfg)
        if verbose:
            self._done(len(filtered_signals))

        # ── Step 7: Final Scoring ──────────────────────────────────────────────
        if verbose:
            self._print_step(7, "Final scoring")
        scored = score_all_signals(
            signals=filtered_signals,
            fundamentals_df=fundamentals_df,
            cfg_scoring=scoring_cfg,
            cfg_risk=risk_cfg,
            cfg_shock=shock_cfg,
            residual_vol_map=resid_vol_map,
            top_n=output_cfg.get("top_n_signals", 10),
        )
        result.scored_signals = scored
        if verbose:
            self._done(len(scored))

        result.warnings = warn_list
        result.run_time_seconds = time.time() - t0
        return result
