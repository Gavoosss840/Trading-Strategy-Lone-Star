"""
Market Data Loader
──────────────────
Downloads and caches price data for stocks and commodity futures via yfinance.
Provides daily returns, rolling volatility, and Fama-French factor data.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _date_range(lookback_days: int) -> Tuple[str, str]:
    end = datetime.today()
    # Add buffer for weekends/holidays
    start = end - timedelta(days=int(lookback_days * 1.5))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill up to 5 days, then drop remaining NaNs."""
    return df.ffill(limit=5).dropna(how="all")


def _to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


# ── Core download functions ───────────────────────────────────────────────────

def download_prices(
    tickers: List[str],
    lookback_days: int = 504,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Returns a DataFrame with dates as index and tickers as columns.
    Missing tickers are silently dropped with a warning.
    """
    start, end = _date_range(lookback_days)
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=progress,
            threads=True,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    # yfinance returns MultiIndex when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = _clean_prices(prices)

    # Warn about tickers with no data
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        warnings.warn(f"No data retrieved for: {missing}")

    return prices.dropna(axis=1, how="all")


def get_returns(
    tickers: List[str],
    lookback_days: int = 504,
    progress: bool = False,
) -> pd.DataFrame:
    """Download prices and return daily log-return DataFrame."""
    prices = download_prices(tickers, lookback_days=lookback_days, progress=progress)
    return _to_returns(prices)


def get_commodity_returns(
    commodity_config: Dict,
    lookback_days: int = 504,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download commodity futures returns.

    Args:
        commodity_config: dict from config.yaml  commodities section
                          {name: {ticker: ..., name: ...}, ...}

    Returns:
        DataFrame with commodity names as columns.
    """
    ticker_map = {
        key: cfg["ticker"]
        for key, cfg in commodity_config.items()
    }
    tickers = list(ticker_map.values())
    returns = get_returns(tickers, lookback_days=lookback_days, progress=progress)

    # Rename columns from ticker to commodity key
    reverse_map = {v: k for k, v in ticker_map.items()}
    returns = returns.rename(columns=reverse_map)
    return returns


def get_risk_free_rate(lookback_days: int = 504) -> pd.Series:
    """
    Download 13-week T-Bill annualised yield and convert to daily rate.
    Falls back to 0 if unavailable.
    """
    try:
        prices = download_prices(["^IRX"], lookback_days=lookback_days)
        if "^IRX" not in prices.columns:
            raise ValueError("^IRX not in data")
        # ^IRX is annualised %, convert to daily decimal
        daily_rf = prices["^IRX"] / 100 / 252
        return daily_rf
    except Exception:
        warnings.warn("Could not download risk-free rate, defaulting to 0.")
        return pd.Series(0.0, dtype=float)


# ── Fama-French factors ───────────────────────────────────────────────────────

def get_fama_french_factors(lookback_days: int = 504) -> Optional[pd.DataFrame]:
    """
    Download Fama-French 5-factor daily data from Ken French's data library
    via pandas-datareader.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    Returns None if unavailable (graceful degradation to CAPM).
    """
    try:
        import pandas_datareader.data as web

        end = datetime.today()
        start = end - timedelta(days=int(lookback_days * 1.5))

        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=start,
            end=end,
        )[0]

        # Factors are in percentage — convert to decimals
        ff5 = ff5 / 100
        ff5.index = pd.to_datetime(ff5.index)
        ff5.index.name = "Date"
        return ff5

    except Exception as exc:
        warnings.warn(
            f"Fama-French data unavailable ({exc}). Will use CAPM-only noise cleaning."
        )
        return None


# ── Market index proxy ────────────────────────────────────────────────────────

def get_market_returns(lookback_days: int = 504) -> pd.Series:
    """S&P 500 daily returns as market proxy."""
    returns = get_returns(["^GSPC"], lookback_days=lookback_days)
    if "^GSPC" in returns.columns:
        return returns["^GSPC"].rename("market")
    return pd.Series(dtype=float, name="market")


# ── Convenience: full dataset bundle ─────────────────────────────────────────

def load_all_data(
    stock_tickers: List[str],
    commodity_config: Dict,
    lookback_days: int = 504,
    progress: bool = False,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    One-shot loader that returns a bundle with:
      - stock_prices
      - stock_returns
      - commodity_returns
      - market_returns
      - risk_free_rate
      - ff_factors  (may be None)
    """
    all_tickers = list(set(stock_tickers + ["^GSPC", "^IRX"]))
    commodity_tickers = [cfg["ticker"] for cfg in commodity_config.values()]
    all_tickers += commodity_tickers

    prices = download_prices(all_tickers, lookback_days=lookback_days, progress=progress)

    stock_prices = prices[[t for t in stock_tickers if t in prices.columns]]
    stock_returns = _to_returns(stock_prices)

    commodity_ticker_map = {cfg["ticker"]: key for key, cfg in commodity_config.items()}
    commodity_prices_raw = prices[
        [t for t in commodity_tickers if t in prices.columns]
    ]
    commodity_returns = _to_returns(commodity_prices_raw).rename(
        columns=commodity_ticker_map
    )

    market_returns = (
        _to_returns(prices[["^GSPC"]])["^GSPC"].rename("market")
        if "^GSPC" in prices.columns
        else pd.Series(dtype=float, name="market")
    )

    risk_free = (
        prices["^IRX"] / 100 / 252
        if "^IRX" in prices.columns
        else pd.Series(0.0, index=prices.index)
    )

    ff_factors = get_fama_french_factors(lookback_days=lookback_days)

    return {
        "stock_prices": stock_prices,
        "stock_returns": stock_returns,
        "commodity_returns": commodity_returns,
        "market_returns": market_returns,
        "risk_free_rate": risk_free,
        "ff_factors": ff_factors,
    }


# ── Rolling stats helpers ─────────────────────────────────────────────────────

def rolling_volatility(returns: pd.DataFrame | pd.Series, window: int = 63) -> pd.DataFrame | pd.Series:
    """Annualised rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score of a series."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)
