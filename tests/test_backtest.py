import numpy as np
import pandas as pd

from rateslab.backtest import circular_shift_test, daily_pnl, deflated_sharpe, performance, pnl_from_pos
from rateslab.signals import positions_from_z


def test_position_is_held_the_day_after_the_signal():
    pos = np.array([0, 1, 1, 0, 0], dtype=float)
    d = np.array([np.nan, 10, 20, 30, 40], dtype=float)
    pnl = pnl_from_pos(pos, d, np.zeros(5), 0.0)
    assert list(pnl) == [0, 0, 20, 30, 0]  # signal on day 1 earns day 2 and 3 moves only


def test_no_lookahead_changing_todays_move_cannot_change_todays_pnl_via_position():
    idx = pd.RangeIndex(6)
    pos = pd.Series([0, 1, 1, 1, 0, 0], dtype=float, index=idx)
    base = pd.Series([0, 1, -1, 2, 3, 1], dtype=float, index=idx)
    pnl_a = daily_pnl(pos, base, None, 0.0)
    shocked = base.copy()
    shocked.iloc[1] = 500  # the move on the signal day itself must not be earned
    assert daily_pnl(pos, shocked, None, 0.0).iloc[1] == pnl_a.iloc[1] == 0.0


def test_costs_charged_on_entry_and_exit():
    pos = np.array([0, 1, 0, 0], dtype=float)
    pnl = pnl_from_pos(pos, np.zeros(4), np.zeros(4), cost_per_leg=0.25)
    assert pnl.sum() == -(0.25 * 2 * 2)  # two legs, in and out


def test_hysteresis_rule():
    z = pd.Series([0, -1.2, -0.6, 0.1, 0.0, -2.0])
    pos = positions_from_z(z, entry=1.0, exit_=0.0)
    assert list(pos) == [0, 1, 1, 0, 0, 1]


def test_performance_and_robustness_functions_run():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    d = pd.Series(rng.normal(0, 3, len(idx)), index=idx)
    pos = pd.Series((rng.random(len(idx)) > 0.5).astype(float), index=idx)
    pnl = daily_pnl(pos, d, None, 0.25)
    perf = performance(pnl, pos)
    assert perf["max_drawdown_bp"] <= 0
    res = circular_shift_test(pos, d, pd.Series(0.0, index=idx), n_perm=50)
    assert 0 <= res["p_value"] <= 1
    assert 0 <= deflated_sharpe(pnl, np.array([0.1, 0.3, -0.2, 0.5])) <= 1
