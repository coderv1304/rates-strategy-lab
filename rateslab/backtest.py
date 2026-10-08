"""Walk-forward backtest engine for a DV01-neutral 2s10s steepener, plus robustness tests.

P&L unit: bp-equivalents, i.e. dollars per $1 of DV01 on each leg. A steepener earns +1 per bp
that 2s10s widens. Position decided at the close of t is held over t+1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from . import config as C
from .signals import positions_from_z

TD = C.TRADING_DAYS


def pnl_from_pos(pos: np.ndarray, dslope: np.ndarray, carry: np.ndarray, cost_per_leg: float):
    """Core numpy P&L. Costs: two legs, each pays `cost_per_leg` bp per unit of position change."""
    held = np.r_[0.0, pos[:-1]]
    carry_prev = np.r_[0.0, carry[:-1]]
    gross = held * (np.nan_to_num(dslope) + np.nan_to_num(carry_prev))
    cost = np.abs(np.diff(np.r_[0.0, held])) * 2.0 * cost_per_leg
    return gross - cost


def daily_pnl(pos: pd.Series, dslope: pd.Series, carry: pd.Series | None = None,
              cost_per_leg: float = C.COST_BP_PER_LEG) -> pd.Series:
    carry_arr = np.zeros(len(pos)) if carry is None else carry.reindex(pos.index).values
    out = pnl_from_pos(pos.values, dslope.reindex(pos.index).values, carry_arr, cost_per_leg)
    return pd.Series(out, index=pos.index, name="pnl_bp")


def sharpe(pnl) -> float:
    x = np.asarray(pnl, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return 0.0 if sd == 0 or np.isnan(sd) else float(x.mean() / sd * np.sqrt(TD))


def performance(pnl: pd.Series, pos: pd.Series | None = None) -> dict:
    x = pnl.dropna()
    equity = x.cumsum()
    max_dd = float((equity - equity.cummax()).min())
    ann_ret, ann_vol = float(x.mean() * TD), float(x.std(ddof=1) * np.sqrt(TD))
    downside = x[x < 0].std(ddof=1) * np.sqrt(TD) if (x < 0).sum() > 1 else np.nan
    active = x[x != 0]
    out = {
        "ann_return_bp": ann_ret, "ann_vol_bp": ann_vol, "sharpe": sharpe(x),
        "sortino": float(ann_ret / downside) if downside and not np.isnan(downside) else np.nan,
        "max_drawdown_bp": max_dd,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan,
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "total_pnl_bp": float(x.sum()), "years": len(x) / TD,
    }
    if pos is not None:
        p = pos.reindex(x.index).fillna(0)
        out["turnover_per_year"] = float(p.diff().abs().sum() / (len(x) / TD))
        out["time_in_market"] = float((p != 0).mean())
    return out


def grid_sharpe(z: pd.Series, dslope: pd.Series, carry: pd.Series, entries, exits,
                cost_per_leg: float = C.COST_BP_PER_LEG, allow_short: bool = False) -> pd.DataFrame:
    """Full-sample (in-sample) Sharpe for every (entry, exit) pair; NaN where exit >= entry."""
    grid = pd.DataFrame(np.nan, index=list(exits), columns=list(entries))
    for e in entries:
        for x in exits:
            if x < e:
                pos = positions_from_z(z, e, x, allow_short)
                grid.loc[x, e] = sharpe(daily_pnl(pos, dslope, carry, cost_per_leg))
    grid.index.name, grid.columns.name = "exit", "entry"
    return grid


@dataclass
class WalkForwardResult:
    oos_pnl: pd.Series
    oos_pos: pd.Series
    folds: pd.DataFrame
    trial_sharpes: np.ndarray


def walk_forward(z: pd.Series, slope_bp: pd.Series, carry: pd.Series, entries, exits,
                 train_days: int = 1260, test_days: int = 126,
                 cost_per_leg: float = C.COST_BP_PER_LEG,
                 allow_short: bool = False) -> WalkForwardResult:
    """Choose (entry, exit) by in-sample Sharpe on the trailing train window, trade the next block.

    The z-series itself is already causal (see signals.py). Each test block starts flat.
    """
    dslope = slope_bp.diff()
    idx = slope_bp.index
    start = idx.get_loc(z.first_valid_index()) + train_days
    pnl_parts, pos_parts, rows, trials = [], [], [], []
    for t in range(start, len(idx), test_days):
        tr, te = idx[t - train_days:t], idx[t:t + test_days]
        best = None
        for e in entries:
            for x in exits:
                if x >= e:
                    continue
                pos = positions_from_z(z.loc[tr], e, x, allow_short)
                sh = sharpe(daily_pnl(pos, dslope.loc[tr], carry.loc[tr], cost_per_leg))
                trials.append(sh)
                if best is None or sh > best[0]:
                    best = (sh, e, x)
        is_sh, e, x = best
        pos_te = positions_from_z(z.loc[te], e, x, allow_short)
        pnl_te = daily_pnl(pos_te, dslope.loc[te], carry.loc[te], cost_per_leg)
        pnl_parts.append(pnl_te)
        pos_parts.append(pos_te)
        rows.append({"test_start": te[0], "test_end": te[-1], "entry": e, "exit": x,
                     "is_sharpe": is_sh, "oos_sharpe": sharpe(pnl_te)})
    return WalkForwardResult(pd.concat(pnl_parts), pd.concat(pos_parts),
                             pd.DataFrame(rows).set_index("test_start"), np.array(trials))


def block_bootstrap_sharpe(pnl: pd.Series, n_boot: int = 2000, block: int = 20, seed: int = 0):
    """Circular block bootstrap of the Sharpe ratio (keeps short-term autocorrelation)."""
    x, n = pnl.values, len(pnl)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        out[b] = sharpe(x[idx])
    return out


def circular_shift_test(pos: pd.Series, dslope: pd.Series, carry: pd.Series,
                        cost_per_leg: float = C.COST_BP_PER_LEG, n_perm: int = 1000,
                        min_shift: int = 63, seed: int = 0) -> dict:
    """Does timing matter? Slide the position series against returns and compare Sharpes.

    p-value = share of shifted-position Sharpes at least as large as the real one.
    """
    p = pos.values
    d = dslope.reindex(pos.index).values
    c = carry.reindex(pos.index).values
    real = sharpe(pnl_from_pos(p, d, c, cost_per_leg))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, len(p) - min_shift, n_perm)
    null = np.array([sharpe(pnl_from_pos(np.roll(p, k), d, c, cost_per_leg)) for k in shifts])
    return {"real_sharpe": real, "p_value": float((null >= real).mean()),
            "null_mean": float(null.mean()), "null_95pct": float(np.quantile(null, 0.95))}


def deflated_sharpe(pnl: pd.Series, trial_sharpes_annual, n_trials: int | None = None) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): probability the true Sharpe > 0
    after accounting for the number of strategy variants you tried."""
    x = pnl.dropna().values
    t = len(x)
    sr = x.mean() / x.std(ddof=1)
    trials = np.asarray(trial_sharpes_annual, dtype=float) / np.sqrt(TD)
    n = n_trials or len(trials)
    g = 0.5772156649
    sr0 = trials.std(ddof=1) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    sk, ku = skew(x), kurtosis(x, fisher=False)
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
