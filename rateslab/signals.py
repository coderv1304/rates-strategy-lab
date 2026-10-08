"""Trading signals. Convention: every z-score is oriented so NEGATIVE means 'go steepener'.

All signals use only data available at the close of day t. The backtest then holds the position
from t+1 onward, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pca_factors import fit_pca


def macro_residual_z(slope_bp: pd.Series, drivers: pd.DataFrame,
                     window: int = 1260, refit: int = 63) -> pd.Series:
    """Z-score of the slope's residual from a regression on macro driver LEVELS.

    Coefficients are refit every `refit` days on the previous `window` days only, then frozen
    and applied to the next block (out-of-sample residuals). Low z = slope flatter than macro implies.
    """
    df = pd.concat([slope_bp.rename("y"), drivers], axis=1).dropna()
    y = df["y"].values
    x = np.column_stack([np.ones(len(df)), df[drivers.columns].values])
    z = pd.Series(np.nan, index=df.index)
    for t0 in range(window, len(df), refit):
        lo, t1 = t0 - window, min(t0 + refit, len(df))
        beta, *_ = np.linalg.lstsq(x[lo:t0], y[lo:t0], rcond=None)
        train_res = y[lo:t0] - x[lo:t0] @ beta
        z.iloc[t0:t1] = (y[t0:t1] - x[t0:t1] @ beta - train_res.mean()) / train_res.std(ddof=1)
    return z.reindex(slope_bp.index)


def pca_slope_z(changes_bp: pd.DataFrame, fit_window: int = 756, refit: int = 63,
                accum: int = 126, zwin: int = 504) -> pd.Series:
    """Z-score of the recent cumulative slope-factor move (mean-reversion view).

    Loadings come from PCA fit on past data only; they are applied to the next block of changes.
    """
    n = len(changes_bp)
    shocks = pd.Series(np.nan, index=changes_bp.index)
    for t0 in range(fit_window, n, refit):
        train = changes_bp.iloc[t0 - fit_window:t0]
        load = fit_pca(train).loadings["Slope"].values
        t1 = min(t0 + refit, n)
        shocks.iloc[t0:t1] = (changes_bp.iloc[t0:t1] - train.mean()).values @ load
    cum = shocks.rolling(accum).sum()
    return ((cum - cum.rolling(zwin).mean()) / cum.rolling(zwin).std()).rename("pca_slope_z")


def momentum_z(slope_bp: pd.Series, lookback: int = 60, zwin: int = 504) -> pd.Series:
    """Trend baseline. Rising slope -> positive momentum -> NEGATIVE signal (steepener)."""
    mom = slope_bp.diff(lookback)
    mz = mom / mom.rolling(zwin).std()
    return (-mz).rename("momentum_z")


def positions_from_z(z: pd.Series, entry: float = 1.0, exit_: float = 0.0,
                     allow_short: bool = False) -> pd.Series:
    """Hysteresis rule: enter steepener at z < -entry, leave when z > -exit_ (mirror for flattener)."""
    pos, state = np.zeros(len(z)), 0
    for i, v in enumerate(z.values):
        if np.isnan(v):
            state = 0
        elif state == 0:
            if v < -entry:
                state = 1
            elif allow_short and v > entry:
                state = -1
        elif state == 1 and v > -exit_:
            state = 0
        elif state == -1 and v < exit_:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="position")
