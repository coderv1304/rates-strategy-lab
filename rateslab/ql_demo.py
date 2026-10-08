"""Runs the QuantLib section end to end and writes tables to reports/."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .analytics import bond_metrics
from .ql_tools import (bond_cross_check, bootstrap_ust_curve, steepener_scenarios, zero_table)


def run(yields: pd.DataFrame) -> None:
    C.ensure_dirs()
    date = yields.index[-1]
    row = yields.iloc[-1]
    print(f"\n[QuantLib] curve date: {date.date()}")

    rows = []
    for years in (2, 10):
        a, q = bond_metrics(float(row[f"{years}Y"]), years), bond_cross_check(float(row[f"{years}Y"]), years)
        rows.append({"bond": f"{years}Y par", "price (mine)": a.price, "price (QL)": q["price"],
                     "mod dur (mine)": a.modified, "mod dur (QL)": q["modified"],
                     "convexity (mine)": a.convexity, "convexity (QL)": q["convexity"]})
    check = pd.DataFrame(rows).set_index("bond")
    print(check.round(6).T)
    check.to_csv(C.REPORT_DIR / "ql_bond_crosscheck.csv")

    zt = zero_table(bootstrap_ust_curve(date, row))
    print("\nBootstrapped Treasury zero curve:\n", zt.round(4))
    zt.to_csv(C.REPORT_DIR / "ql_ust_zero_curve.csv")

    scen, info = steepener_scenarios(date, row)
    print("\nSOFR swap steepener (ILLUSTRATIVE quotes):", {k: round(v, 4) for k, v in info.items()})
    print(scen.round(0))
    scen.to_csv(C.REPORT_DIR / "ql_swap_scenarios.csv")
