import numpy as np
import pandas as pd

from rateslab.pca_factors import fit_pca


def _fake_changes(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    taus = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    level = rng.normal(0, 6, n)[:, None] * np.ones(len(taus))
    slope = rng.normal(0, 2, n)[:, None] * np.linspace(-1, 1, len(taus))
    noise = rng.normal(0, 0.2, (n, len(taus)))
    return pd.DataFrame(level + slope + noise, columns=[f"{int(t)}Y" for t in taus])


def test_pca_recovers_level_and_slope_with_fixed_signs():
    res = fit_pca(_fake_changes())
    assert res.explained["Level"] > 0.8
    assert res.explained.sum() <= 1.0 + 1e-9
    assert res.loadings["Level"].sum() > 0
    assert res.loadings["Slope"].iloc[-1] > res.loadings["Slope"].iloc[0]


def test_loadings_are_orthonormal():
    v = fit_pca(_fake_changes()).loadings.values
    assert np.allclose(v.T @ v, np.eye(3), atol=1e-8)
