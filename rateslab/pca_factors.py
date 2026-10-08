"""PCA of yield-curve changes: level, slope, curvature (eigendecomposition of the covariance)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAMES = ["Level", "Slope", "Curvature"]


@dataclass
class PCAResult:
    loadings: pd.DataFrame   # tenors x components
    explained: pd.Series     # share of variance
    scores: pd.DataFrame     # daily factor realisations (bp-scaled)


def _orient(v: np.ndarray) -> np.ndarray:
    """Fix eigenvector signs so components are comparable across windows.

    Level: positive on average. Slope: positive = steepening (long end up vs short end).
    Curvature: positive = belly up relative to the wings.
    """
    v = v.copy()
    mid = v.shape[0] // 2
    if v[:, 0].sum() < 0:
        v[:, 0] *= -1
    if v.shape[1] > 1 and v[-1, 1] - v[0, 1] < 0:
        v[:, 1] *= -1
    if v.shape[1] > 2 and v[mid, 2] - 0.5 * (v[0, 2] + v[-1, 2]) < 0:
        v[:, 2] *= -1
    return v


def fit_pca(changes_bp: pd.DataFrame, n_components: int = 3) -> PCAResult:
    x = changes_bp - changes_bp.mean()
    cov = np.cov(x.values, rowvar=False)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order][:, :n_components]
    v = _orient(v)
    names = NAMES[:n_components]
    return PCAResult(
        loadings=pd.DataFrame(v, index=changes_bp.columns, columns=names),
        explained=pd.Series(w[:n_components] / w.sum(), index=names),
        scores=pd.DataFrame(x.values @ v, index=changes_bp.index, columns=names),
    )


def rolling_pca(changes_bp: pd.DataFrame, window: int = 756, step: int = 21):
    """Re-estimate PCA on trailing windows. Returns (explained, slope_loadings) by end date."""
    exp_rows, slope_rows, idx = [], [], []
    for end in range(window, len(changes_bp) + 1, step):
        res = fit_pca(changes_bp.iloc[end - window:end])
        exp_rows.append(res.explained.values)
        slope_rows.append(res.loadings["Slope"].values)
        idx.append(changes_bp.index[end - 1])
    explained = pd.DataFrame(exp_rows, index=idx, columns=NAMES)
    slope = pd.DataFrame(slope_rows, index=idx, columns=changes_bp.columns)
    return explained, slope
