"""Charts. Every function saves a PNG into reports/figures and returns the file path."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name: str):
    C.ensure_dirs()
    path = C.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_yield_history(yields: pd.DataFrame, cols=("2Y", "5Y", "10Y", "30Y")):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in cols:
        ax.plot(yields.index, yields[c], lw=1, label=c)
    ax.set(title="US Treasury par yields", ylabel="Yield (%)")
    ax.legend(ncol=4)
    return _save(fig, "01_yield_history")


def plot_slope_with_inversions(slope: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(slope.index, slope, lw=1, color="#1f4e79")
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(slope.index, slope, 0, where=slope < 0, color="#c0392b", alpha=0.35,
                    label="Inverted")
    ax.set(title="2s10s slope (10Y minus 2Y)", ylabel="bp")
    ax.legend()
    return _save(fig, "02_slope_inversions")


def plot_curve_snapshots(yields: pd.DataFrame, dates: list[str]):
    taus = [C.TENOR_YEARS[c] for c in yields.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in dates:
        row = yields.loc[:d].iloc[-1]
        ax.plot(taus, row.values, marker="o", ms=3, label=str(row.name.date()))
    ax.set(title="Curve snapshots", xlabel="Maturity (years)", ylabel="Yield (%)")
    ax.legend()
    return _save(fig, "03_curve_snapshots")


def plot_pca_loadings(loadings: pd.DataFrame, explained: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in loadings.columns:
        ax.plot(loadings.index, loadings[c], marker="o", label=f"{c} ({explained[c]:.1%})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="PCA loadings of daily yield changes", ylabel="Loading")
    ax.legend()
    return _save(fig, "04_pca_loadings")


def plot_rolling_pca(explained: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    explained.plot(ax=ax)
    ax.set(title="Rolling PCA: variance explained (3-year window)", ylabel="Share")
    return _save(fig, "05_rolling_pca")


def plot_nss_fit(taus, observed, fitted, label: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grid = np.linspace(min(taus), max(taus), 200)
    ax.scatter(taus, observed, color="k", zorder=3, label="Observed par yields")
    ax.plot(grid, fitted(grid), color="#1f77b4", label="Nelson-Siegel-Svensson fit")
    ax.set(title=f"NSS fit on {label}", xlabel="Maturity (years)", ylabel="Yield (%)")
    ax.legend()
    return _save(fig, "06_nss_fit")


def plot_equity(curves: dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, pnl in curves.items():
        ax.plot(pnl.index, pnl.cumsum(), lw=1.2, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Out-of-sample cumulative P&L (walk-forward, after costs)",
           ylabel="bp-equivalents")
    ax.legend()
    return _save(fig, "07_equity_curves")


def plot_heatmap(grid: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1, origin="lower")
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid.values[i, j]):
                ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(title=f"In-sample Sharpe by thresholds: {name}", xlabel="entry |z|", ylabel="exit z")
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    return _save(fig, f"08_heatmap_{name}")


def plot_rolling_sharpe(pnl: pd.Series, window: int = 252):
    roll = pnl.rolling(window).mean() / pnl.rolling(window).std() * np.sqrt(C.TRADING_DAYS)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(roll.index, roll, lw=1.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Rolling 1-year Sharpe (out-of-sample, best signal)", ylabel="Sharpe")
    return _save(fig, "09_rolling_sharpe")


def plot_bootstrap(boot: np.ndarray, real: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot, bins=40, color="#7f8c8d", alpha=0.8)
    ax.axvline(real, color="#c0392b", lw=2, label=f"Observed {real:.2f}")
    ax.axvline(0, color="k", lw=1)
    ax.set(title="Block-bootstrap distribution of out-of-sample Sharpe", xlabel="Sharpe")
    ax.legend()
    return _save(fig, "10_bootstrap")
