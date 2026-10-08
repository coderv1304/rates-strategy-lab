"""Nelson-Siegel-Svensson curve fitting implemented directly with scipy (no extra package)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C

PARAMS = ["b0", "b1", "b2", "b3", "lam1", "lam2"]
LOWER = np.array([-1.0, -15.0, -30.0, -30.0, 0.2, 1.0])
UPPER = np.array([15.0, 15.0, 30.0, 30.0, 6.0, 20.0])


def nss_yield(tau, b0, b1, b2, b3, lam1, lam2):
    """Svensson (1994) zero/par yield curve in percent for maturities tau (years)."""
    tau = np.asarray(tau, dtype=float)
    x1, x2 = tau / lam1, tau / lam2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3


def fit_nss_day(taus: np.ndarray, ys: np.ndarray, x0: np.ndarray | None = None):
    """Least-squares NSS fit to one day's yields. Returns (params, rmse_bp)."""
    if x0 is None:
        x0 = np.array([ys[-1], ys[0] - ys[-1], 0.0, 0.0, 1.5, 6.0])
    x0 = np.clip(x0, LOWER + 1e-6, UPPER - 1e-6)
    res = least_squares(lambda p: nss_yield(taus, *p) - ys, x0, bounds=(LOWER, UPPER))
    return res.x, 100.0 * float(np.sqrt(np.mean(res.fun**2)))


def fit_nss_history(yields_pct: pd.DataFrame, step: int = 21) -> pd.DataFrame:
    """Fit every `step`-th day, warm-starting from the previous day's parameters."""
    taus = np.array([C.TENOR_YEARS[c] for c in yields_pct.columns])
    rows, x0 = [], None
    for date, row in yields_pct.iloc[::step].iterrows():
        params, rmse = fit_nss_day(taus, row.values.astype(float), x0)
        x0 = params
        rows.append({"date": date, **dict(zip(PARAMS, params)), "rmse_bp": rmse})
    return pd.DataFrame(rows).set_index("date")
