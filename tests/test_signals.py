import numpy as np
import pandas as pd

from rateslab.signals import macro_residual_z, momentum_z


def test_macro_z_uses_only_past_data_for_coefficients():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", periods=900)
    x = pd.DataFrame({"a": rng.normal(size=len(idx)).cumsum()}, index=idx)
    y = pd.Series(2 * x["a"].values + rng.normal(size=len(idx)), index=idx)
    z1 = macro_residual_z(y, x, window=300, refit=50)
    y2 = y.copy()
    y2.iloc[-1] += 1000  # change only the very last observation
    z2 = macro_residual_z(y2, x, window=300, refit=50)
    assert z1.iloc[:-1].equals(z2.iloc[:-1]) or np.allclose(z1.iloc[:-1].dropna(), z2.iloc[:-1].dropna())


def test_momentum_z_sign_convention_negative_means_steepener():
    idx = pd.bdate_range("2015-01-01", periods=800)
    rising = pd.Series(np.linspace(0, 100, len(idx)) + np.random.default_rng(0).normal(0, 1, len(idx)), index=idx)
    assert momentum_z(rising).dropna().iloc[-1] < 0
