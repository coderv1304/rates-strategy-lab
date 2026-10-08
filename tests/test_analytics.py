import numpy as np
import pytest

from rateslab.analytics import (bond_metrics, par_mod_duration, price_change_pct, steepener_carry_bp_per_day,
                                steepener_hedge)


def test_par_bond_prices_at_100():
    for y, n in [(2.0, 2), (4.5, 10), (6.0, 30)]:
        assert bond_metrics(y, n).price == pytest.approx(100.0, abs=1e-9)


def test_closed_form_par_duration_matches_cashflow_duration():
    m = bond_metrics(4.0, 10)
    assert m.modified == pytest.approx(float(par_mod_duration(0.04, 10)), rel=1e-10)


def test_dv01_matches_finite_difference():
    y, n = 4.0, 10
    m = bond_metrics(y, n, coupon_pct=y)
    up = bond_metrics(y + 0.01, n, coupon_pct=y).price
    dn = bond_metrics(y - 0.01, n, coupon_pct=y).price
    assert (dn - up) / 2 == pytest.approx(m.dv01, rel=1e-3)


def test_duration_convexity_approximation_is_close_for_small_shocks():
    y, n = 4.0, 10
    m = bond_metrics(y, n)
    exact = bond_metrics(y + 0.5, n, coupon_pct=y).price / 100 - 1
    assert price_change_pct(m, 50) / 100 == pytest.approx(exact, abs=2e-4)


def test_steepener_is_dv01_neutral_and_10y_is_smaller_notional():
    h = steepener_hedge(4.0, 4.4, 100e6)
    assert h["net_dv01_usd"] == pytest.approx(0.0, abs=1e-6)
    assert h["notional_10y"] < 100e6


def test_carry_is_finite_and_scales_with_curve_shape():
    c = steepener_carry_bp_per_day(np.array([4.0, 4.0]), np.array([4.5, 3.5]), np.array([4.3, 4.3]))
    assert np.all(np.isfinite(c))
    assert c[0] != c[1]
