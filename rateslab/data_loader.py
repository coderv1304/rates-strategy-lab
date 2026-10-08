"""Data ingestion: FRED (cached to CSV), Treasury.gov fallback and a synthetic generator."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- FRED
def fetch_fred_series(series_id: str, start: str = C.START_DATE, refresh: bool = False) -> pd.Series:
    """Return one FRED series. Reads the CSV cache unless refresh=True."""
    C.ensure_dirs()
    path = C.DATA_RAW / f"{series_id}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(series_id)

    from dotenv import load_dotenv
    from fredapi import Fred

    load_dotenv(C.ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing. Copy .env.example to .env and add your key.")
    s = Fred(api_key=key).get_series(series_id, observation_start=start).rename(series_id)
    s.to_csv(path, header=True)
    return s


def align_macro(raw: dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Put macro series on the business-day index, respecting publication lags."""
    out = {}
    for name, s in raw.items():
        s = s.dropna().astype(float)
        if name == "CPI":
            s = s.pct_change(12) * 100.0  # year-on-year inflation, percent
            s = s.dropna()
        lag = C.RELEASE_LAG_DAYS.get(name, 0)
        if lag:
            s = s.copy()
            s.index = s.index + pd.Timedelta(days=lag)
        full = s.reindex(s.index.union(index)).ffill()
        out[name] = full.reindex(index)
    return pd.DataFrame(out)


def load_fred_dataset(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    yields = pd.DataFrame({k: fetch_fred_series(v, refresh=refresh) for k, v in C.TENOR_SERIES.items()})
    yields = yields.dropna(subset=["2Y", "10Y"])  # keep days where the core points exist
    yields = yields.ffill(limit=3).dropna()  # bridge short gaps only; never fill long gaps
    raw_macro = {k: fetch_fred_series(v, refresh=refresh) for k, v in C.MACRO_SERIES.items()}
    macro = align_macro(raw_macro, yields.index).dropna()
    yields = yields.loc[macro.index]
    return yields, macro


# ------------------------------------------------------------------- Treasury.gov (optional)
def fetch_treasury_year(year: int) -> pd.DataFrame:
    """Daily par yield curve for one year from Treasury.gov (percent). URL may change over time."""
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}&page&_format=csv"
    )
    df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
    rename = {"1 Mo": "1M", "3 Mo": "3M", "6 Mo": "6M", "1 Yr": "1Y", "2 Yr": "2Y", "3 Yr": "3Y",
              "5 Yr": "5Y", "7 Yr": "7Y", "10 Yr": "10Y", "20 Yr": "20Y", "30 Yr": "30Y"}
    return df.rename(columns=rename)[list(rename.values())]


# --------------------------------------------------------------------------- synthetic
def make_synthetic(years: int = 22, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three-factor Nelson-Siegel-style simulated curve + macro drivers.

    ONLY for demonstrating the pipeline without an API key. Never quote its results.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(C.START_DATE, periods=int(years * 261))
    n, dt = len(idx), 1 / C.TRADING_DAYS

    def ou(theta: float, mu: float, sigma: float, x0: float) -> np.ndarray:
        x = np.empty(n)
        x[0] = x0
        shocks = rng.standard_normal(n)
        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * shocks[i]
        return x

    level = ou(0.25, 3.6, 0.9, 4.0)
    slope = ou(1.2, -1.5, 0.7, -1.0)  # NS beta1: negative means upward-sloping curve
    curv = ou(1.5, -0.5, 1.0, 0.0)
    lam = 2.5
    cols = {}
    for tenor, tau in C.TENOR_YEARS.items():
        x = tau / lam
        f1 = (1 - np.exp(-x)) / x
        f2 = f1 - np.exp(-x)
        y = level + slope * f1 + curv * f2 + rng.normal(0, 0.01, n)
        cols[tenor] = np.clip(y, 0.02, None)
    yields = pd.DataFrame(cols, index=idx)

    ffr = np.clip(yields["3M"].values + rng.normal(0, 0.05, n), 0.05, None)
    bei = 2.0 + 0.25 * (level - 3.6) + ou(2.0, 0.0, 0.25, 0.0)
    unrate = np.clip(5.5 + ou(0.5, 0.0, 1.2, 0.0), 3.0, 12.0)
    cpi = np.clip(2.5 + ou(0.8, 0.0, 1.0, 0.0), -1.0, 9.0)
    macro = pd.DataFrame({"FFR": ffr, "BEI10": bei, "UNRATE": unrate, "CPI": cpi}, index=idx)
    return yields, macro


# ------------------------------------------------------------------------------ entry
def load_dataset(source: str = "fred", refresh: bool = False, save: bool = True):
    if source == "fred":
        yields, macro = load_fred_dataset(refresh=refresh)
    elif source == "synthetic":
        yields, macro = make_synthetic()
    else:
        raise ValueError("source must be 'fred' or 'synthetic'")
    if save:
        C.ensure_dirs()
        yields.to_csv(C.DATA_PROC / "yields.csv", index_label="Date")
        macro.to_csv(C.DATA_PROC / "macro.csv", index_label="Date")
    return yields, macro


def slope_bp(yields: pd.DataFrame, short: str = "2Y", long: str = "10Y") -> pd.Series:
    """Curve slope in basis points (long minus short)."""
    return ((yields[long] - yields[short]) * 100.0).rename(f"{short[:-1]}s{long[:-1]}s")
