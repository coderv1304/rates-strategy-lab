"""Statistical tests: stationarity, macro regression with HAC errors, cointegration, half-life."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint, kpss


def stationarity_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) side by side."""
    rows = []
    for name, s in series.items():
        s = s.dropna()
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        if adf_p < 0.05 and kpss_p >= 0.05:
            verdict = "stationary"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            verdict = "unit root"
        else:
            verdict = "ambiguous"
        rows.append({"series": name, "ADF stat": adf_stat, "ADF p": adf_p,
                     "KPSS stat": kpss_stat, "KPSS p": kpss_p, "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


def monthly_slope_regression(slope_bp: pd.Series, macro: pd.DataFrame, maxlags: int = 3):
    """Regress monthly change in the slope on monthly changes in macro drivers (Newey-West SE)."""
    m = pd.concat([slope_bp.rename("slope"), macro], axis=1).resample("ME").last().diff().dropna()
    x = sm.add_constant(m.drop(columns="slope"))
    fit = sm.OLS(m["slope"], x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    lb = acorr_ljungbox(fit.resid, lags=[12], return_df=True)
    diag = {"durbin_watson": float(durbin_watson(fit.resid)),
            "ljung_box_p_lag12": float(lb["lb_pvalue"].iloc[0]),
            "r_squared": float(fit.rsquared), "n_obs": int(fit.nobs)}
    return fit, diag


def engle_granger(y10: pd.Series, y2: pd.Series) -> dict:
    stat, p, crit = coint(y10, y2, trend="c")
    return {"stat": float(stat), "p_value": float(p),
            "crit_1_5_10": [float(c) for c in crit]}


def ou_half_life(spread: pd.Series) -> dict:
    """AR(1) fit s_t = a + phi*s_{t-1} + e; half-life = -ln2 / ln(phi) in observations."""
    s = spread.dropna()
    x, y = s.shift(1).dropna().values, s.iloc[1:].values
    phi, a = np.polyfit(x, y, 1)
    half = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
    return {"phi": float(phi), "half_life_days": float(half),
            "long_run_mean": float(a / (1 - phi)) if phi != 1 else np.nan}
