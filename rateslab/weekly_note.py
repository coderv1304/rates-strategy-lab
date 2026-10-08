"""Generate a draft weekly market note (markdown + chart) from the processed dataset.

You still write the 'Why' and 'View' sections yourself: the script only fills the numbers.
Run:  python -m rateslab.weekly_note
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as C
from . import report as R
from .data_loader import slope_bp
from .signals import macro_residual_z


def build_note() -> str:
    yields = pd.read_csv(C.DATA_PROC / "yields.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(C.DATA_PROC / "macro.csv", index_col=0, parse_dates=True)
    slope = slope_bp(yields)
    last = yields.index[-1]
    week_ago = yields.index[-6]
    chg = (yields.loc[last] - yields.loc[week_ago]) * 100
    pct = (slope.rank(pct=True).iloc[-1]) * 100
    z = macro_residual_z(slope, macro[["FFR", "BEI10", "UNRATE", "CPI"]]).dropna()
    z_last = z.iloc[-1] if len(z) else float("nan")

    R.plot_yield_history(yields.loc[last - pd.Timedelta(days=365):])
    chart = "../reports/figures/01_yield_history.png"
    state = ("below" if z_last < 0 else "above") if pd.notna(z_last) else "n/a"
    lines = [
        f"# Rates note: week ending {last.date()}",
        "",
        "## 1. What moved",
        f"- 2Y {yields.loc[last, '2Y']:.2f}% ({chg['2Y']:+.0f}bp w/w), "
        f"10Y {yields.loc[last, '10Y']:.2f}% ({chg['10Y']:+.0f}bp w/w).",
        f"- 2s10s at {slope.iloc[-1]:.0f}bp ({slope.iloc[-1] - slope.loc[week_ago]:+.0f}bp w/w), "
        f"{pct:.0f}th percentile of its history since {slope.index[0].year}.",
        f"- Macro-residual z-score: {z_last:+.2f} (slope is {state} the level implied by macro drivers).",
        "",
        "## 2. Why",
        "_Write this yourself: data releases, central-bank communication, supply, positioning._",
        "",
        "## 3. Chart",
        f"![Yield history]({chart})",
        "",
        "## 4. View and risk",
        "_Your view in one sentence, then what would prove you wrong._",
        "",
        f"<sub>Generated {dt.date.today()} by rateslab.weekly_note. Data: FRED.</sub>",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    C.ensure_dirs()
    text = build_note()
    out = C.NOTES_DIR / f"note_{dt.date.today():%Y_%m_%d}.md"
    out.write_text(text, encoding="utf-8")
    print(f"Draft written to {out}")
