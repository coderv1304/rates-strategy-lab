"""End-to-end run: data -> charts -> stats -> PCA -> NSS -> QuantLib -> signals -> backtest.

Examples
    python -m rateslab.pipeline --source synthetic          # offline demo, no API key
    python -m rateslab.pipeline --source fred               # real data (needs FRED_API_KEY)
    python -m rateslab.pipeline --source fred --only data   # just download and cache
    python -m rateslab.pipeline --source fred --skip quantlib,stats
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import report as R
from .analytics import steepener_carry_bp_per_day
from .backtest import (block_bootstrap_sharpe, circular_shift_test, daily_pnl, deflated_sharpe,
                       grid_sharpe, performance, sharpe, walk_forward)
from .curves import fit_nss_day, fit_nss_history, nss_yield
from .data_loader import load_dataset, slope_bp
from .pca_factors import fit_pca, rolling_pca
from .signals import macro_residual_z, momentum_z, pca_slope_z

ENTRIES = [0.5, 1.0, 1.5, 2.0]
EXITS = [0.0, 0.25, 0.5]


def step_charts(yields: pd.DataFrame, slope: pd.Series) -> None:
    R.plot_yield_history(yields)
    R.plot_slope_with_inversions(slope)
    first = str(yields.index[0].date())
    picks = [d for d in ("2007-06-29", "2019-08-30", "2021-12-31", "2023-07-03") if d >= first]
    R.plot_curve_snapshots(yields, picks + [str(yields.index[-1].date())])
    print("[charts] saved to reports/figures")


def step_stats(yields: pd.DataFrame, macro: pd.DataFrame, slope: pd.Series) -> None:
    from .stats_tests import (engle_granger, monthly_slope_regression, ou_half_life,
                              stationarity_table)

    tab = stationarity_table({"10Y level": yields["10Y"], "10Y daily change": yields["10Y"].diff(),
                              "2s10s level": slope, "2s10s daily change": slope.diff()})
    print("\n[stats] Stationarity (ADF H0 = unit root, KPSS H0 = stationary)\n", tab.round(4))
    tab.to_csv(C.REPORT_DIR / "stats_stationarity.csv")
    fit, diag = monthly_slope_regression(slope, macro)
    print("\n[stats] Monthly change in 2s10s on macro changes (Newey-West)\n", fit.summary().tables[1])
    print("diagnostics:", {k: round(v, 4) for k, v in diag.items()})
    print("[stats] Engle-Granger 10Y vs 2Y:", engle_granger(yields["10Y"], yields["2Y"]))
    print("[stats] OU half-life of 2s10s:", ou_half_life(slope))


def step_pca(yields: pd.DataFrame) -> pd.DataFrame:
    changes = yields[C.PCA_TENORS].diff().dropna() * 100.0  # bp
    res = fit_pca(changes)
    print("\n[pca] variance explained\n", res.explained.round(4))
    print("[pca] loadings\n", res.loadings.round(3))
    R.plot_pca_loadings(res.loadings, res.explained)
    explained, _ = rolling_pca(changes)
    R.plot_rolling_pca(explained)
    return changes


def step_nss(yields: pd.DataFrame) -> None:
    taus = np.array([C.TENOR_YEARS[c] for c in yields.columns])
    row = yields.iloc[-1]
    params, rmse = fit_nss_day(taus, row.values.astype(float))
    print(f"\n[nss] latest-day fit RMSE: {rmse:.2f} bp; params: {np.round(params, 3)}")
    R.plot_nss_fit(taus, row.values, lambda t: nss_yield(t, *params), str(row.name.date()))
    hist = fit_nss_history(yields, step=21)
    hist.to_csv(C.REPORT_DIR / "nss_params_history.csv")
    print(f"[nss] fitted {len(hist)} dates, median RMSE {hist['rmse_bp'].median():.2f} bp")


def step_backtest(yields: pd.DataFrame, macro: pd.DataFrame, slope: pd.Series,
                  changes: pd.DataFrame, source: str) -> None:
    carry = pd.Series(steepener_carry_bp_per_day(yields["2Y"], yields["10Y"], yields["3M"]),
                      index=yields.index)
    signals = {"macro_residual": macro_residual_z(slope, macro[["FFR", "BEI10", "UNRATE", "CPI"]]),
               "pca_slope": pca_slope_z(changes).reindex(yields.index),
               "momentum": momentum_z(slope)}
    dslope = slope.diff()
    rows, curves, results, all_trials = [], {}, {}, []
    for name, z in signals.items():
        grid = grid_sharpe(z, dslope, carry, ENTRIES, EXITS)
        R.plot_heatmap(grid, name)
        all_trials += list(grid.values[~np.isnan(grid.values)])
        wf = walk_forward(z, slope, carry, ENTRIES, EXITS)
        perf = performance(wf.oos_pnl, wf.oos_pos)
        perf_costly = performance(*_pnl_with_cost(wf, dslope, carry, 1.0))
        rows.append({"signal": name, "IS Sharpe (fold mean)": wf.folds["is_sharpe"].mean(),
                     "OOS Sharpe": perf["sharpe"], "OOS Sharpe @1bp/leg": perf_costly["sharpe"],
                     "OOS ann ret (bp)": perf["ann_return_bp"], "OOS ann vol (bp)": perf["ann_vol_bp"],
                     "Sortino": perf["sortino"], "Max DD (bp)": perf["max_drawdown_bp"],
                     "Calmar": perf["calmar"], "Hit rate": perf["hit_rate"],
                     "Turnover/yr": perf["turnover_per_year"], "Years": perf["years"]})
        curves[name] = wf.oos_pnl
        results[name] = wf
        wf.folds.to_csv(C.REPORT_DIR / f"folds_{name}.csv")
    table = pd.DataFrame(rows).set_index("signal")

    best = table["OOS Sharpe"].idxmax()
    wf = results[best]
    boot = block_bootstrap_sharpe(wf.oos_pnl)
    perm = circular_shift_test(wf.oos_pos, dslope, carry)
    dsr = deflated_sharpe(wf.oos_pnl, np.array(all_trials), n_trials=len(all_trials))
    R.plot_equity(curves)
    R.plot_rolling_sharpe(wf.oos_pnl)
    R.plot_bootstrap(boot, sharpe(wf.oos_pnl))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n[backtest] walk-forward out-of-sample results\n", table.round(3).T)
    print(f"\n[backtest] best OOS signal: {best}")
    print(f"  bootstrap 95% CI for Sharpe: [{lo:.2f}, {hi:.2f}]")
    print(f"  circular-shift test p-value: {perm['p_value']:.3f} (null 95th pct {perm['null_95pct']:.2f})")
    print(f"  deflated Sharpe probability: {dsr:.3f}  ({len(all_trials)} variants counted)")
    table.to_csv(C.REPORT_DIR / "backtest_results.csv")
    note = "SYNTHETIC DATA - pipeline demo only.\n\n" if source == "synthetic" else ""
    (C.REPORT_DIR / "results.md").write_text(
        note + table.round(3).to_markdown() + f"\n\nBest OOS signal: **{best}**. "
        f"Bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]. Circular-shift p = {perm['p_value']:.3f}. "
        f"Deflated Sharpe probability = {dsr:.3f} over {len(all_trials)} variants.\n")


def _pnl_with_cost(wf, dslope, carry, cost_per_leg):
    return daily_pnl(wf.oos_pos, dslope, carry, cost_per_leg), wf.oos_pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fred", choices=["fred", "synthetic"])
    ap.add_argument("--refresh", action="store_true", help="re-download from FRED")
    ap.add_argument("--only", default="", help="use 'data' to stop after the download")
    ap.add_argument("--skip", default="", help="comma list: charts,stats,pca,nss,quantlib,backtest")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    C.ensure_dirs()
    yields, macro = load_dataset(args.source, refresh=args.refresh)
    print(f"[data] {args.source}: {yields.shape[0]} days, {yields.index[0].date()} -> "
          f"{yields.index[-1].date()}")
    if args.only == "data":
        return
    slope = slope_bp(yields)

    if "charts" not in skip:
        step_charts(yields, slope)
    if "stats" not in skip:
        step_stats(yields, macro, slope)
    changes = step_pca(yields) if "pca" not in skip else yields[C.PCA_TENORS].diff().dropna() * 100
    if "nss" not in skip:
        step_nss(yields)
    if "quantlib" not in skip:
        from . import ql_demo

        ql_demo.run(yields)
    if "backtest" not in skip:
        step_backtest(yields, macro, slope, changes, args.source)
    print("\nDone. See reports/ for tables and reports/figures/ for charts.")


if __name__ == "__main__":
    main()
