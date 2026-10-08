"""QuantLib helpers: bond cross-checks, Treasury bootstrap, SOFR OIS curve and swap scenarios.

IMPORTANT: free historical SOFR swap quotes are not available, so the SOFR curve is built from
ILLUSTRATIVE quotes derived from Treasury par yields minus assumed swap spreads. Label it as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import QuantLib as ql

from . import config as C

# ILLUSTRATIVE swap-spread assumptions in bp (swap rate minus Treasury yield). Not market data.
ILLUSTRATIVE_SWAP_SPREAD_BP = {1: -2.0, 2: -5.0, 3: -8.0, 5: -14.0, 7: -20.0, 10: -28.0,
                               20: -45.0, 30: -55.0}
SWAP_TENORS = list(ILLUSTRATIVE_SWAP_SPREAD_BP)


def qdate(ts: pd.Timestamp) -> ql.Date:
    return ql.Date(ts.day, ts.month, ts.year)


# ------------------------------------------------------------------ bond cross-check
def bond_cross_check(yield_pct: float, years: int, freq: int = 2) -> dict:
    """Price/duration/convexity from QuantLib for a par bond priced on its issue date."""
    issue = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = issue
    maturity = issue + ql.Period(int(years), ql.Years)
    sched = ql.Schedule(issue, maturity, ql.Period(ql.Semiannual), ql.NullCalendar(),
                        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    dc = ql.Thirty360(ql.Thirty360.BondBasis)
    bond = ql.FixedRateBond(0, 100.0, sched, [yield_pct / 100.0], dc)
    rate = ql.InterestRate(yield_pct / 100.0, dc, ql.Compounded, ql.Semiannual)
    return {
        "price": bond.cleanPrice(rate.rate(), dc, ql.Compounded, ql.Semiannual),
        "macaulay": ql.BondFunctions.duration(bond, rate, ql.Duration.Macaulay),
        "modified": ql.BondFunctions.duration(bond, rate, ql.Duration.Modified),
        "convexity": ql.BondFunctions.convexity(bond, rate),
    }


# ------------------------------------------------------------- Treasury bootstrapping
def bootstrap_ust_curve(date: pd.Timestamp, par_yields_pct: pd.Series):
    """Bootstrap a discount curve: deposits for <1Y, par fixed-rate bonds (price 100) from 1Y."""
    today = qdate(date)
    ql.Settings.instance().evaluationDate = today
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    dc = ql.Thirty360(ql.Thirty360.BondBasis)
    helpers = []
    for tenor, y in par_yields_pct.items():
        yrs, r = C.TENOR_YEARS[tenor], float(y) / 100.0
        if yrs < 1.0:
            months = int(round(yrs * 12))
            helpers.append(ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(r)), ql.Period(months, ql.Months), 0, cal,
                ql.ModifiedFollowing, False, ql.Actual360()))
        else:
            maturity = cal.advance(today, ql.Period(int(round(yrs)), ql.Years))
            sched = ql.Schedule(today, maturity, ql.Period(ql.Semiannual), cal, ql.Following,
                                ql.Following, ql.DateGeneration.Backward, False)
            helpers.append(ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(100.0)), 0, 100.0, sched, [r], dc))
    curve = ql.PiecewiseLogLinearDiscount(today, helpers, dc)
    curve.enableExtrapolation()
    return curve


def zero_table(curve, maturities=(0.5, 1, 2, 3, 5, 7, 10, 20, 30)) -> pd.DataFrame:
    rows = [{"maturity_y": m, "zero_pct": 100 * curve.zeroRate(float(m), ql.Continuous).rate(),
             "discount": curve.discount(float(m))} for m in maturities]
    return pd.DataFrame(rows).set_index("maturity_y")


# ----------------------------------------------------------------------- SOFR OIS
def proxy_sofr_quotes(par_yields_pct: pd.Series) -> dict[int, float]:
    """Illustrative SOFR swap quotes (percent) = interpolated Treasury yield + assumed spread."""
    taus = [C.TENOR_YEARS[c] for c in par_yields_pct.index]
    out = {}
    for n, spread in ILLUSTRATIVE_SWAP_SPREAD_BP.items():
        out[n] = float(np.interp(n, taus, par_yields_pct.values)) + spread / 100.0
    return out


def build_sofr_curve(ref: ql.Date, quotes_pct: dict[int, float]):
    ql.Settings.instance().evaluationDate = ref
    helpers = [
        ql.OISRateHelper(2, ql.Period(n, ql.Years), ql.QuoteHandle(ql.SimpleQuote(q / 100.0)),
                         ql.Sofr())
        for n, q in quotes_pct.items()
    ]
    curve = ql.PiecewiseLogLinearDiscount(ref, helpers, ql.Actual365Fixed())
    curve.enableExtrapolation()
    return curve


def _ois_swap(ref: ql.Date, years: int, fixed_rate: float, curve, receiver: bool):
    handle = ql.YieldTermStructureHandle(curve)
    index = ql.Sofr(handle)
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    start = cal.advance(ref, 2, ql.Days)
    end = cal.advance(start, ql.Period(years, ql.Years))
    sched = ql.Schedule(start, end, ql.Period(ql.Annual), cal, ql.ModifiedFollowing,
                        ql.ModifiedFollowing, ql.DateGeneration.Backward, False)
    kind = ql.OvernightIndexedSwap.Receiver if receiver else ql.OvernightIndexedSwap.Payer
    swap = ql.OvernightIndexedSwap(kind, 1_000_000.0, sched, fixed_rate, ql.Actual360(), index)
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    return swap


def par_swap_rate(ref: ql.Date, years: int, curve) -> float:
    return _ois_swap(ref, years, 0.03, curve, receiver=False).fairRate()


def swap_npv(ref: ql.Date, years: int, fixed_rate: float, quotes_pct: dict[int, float],
             receiver: bool) -> float:
    """NPV per 1m notional of a swap struck at fixed_rate, on a curve rebuilt from the quotes."""
    curve = build_sofr_curve(ref, quotes_pct)
    return _ois_swap(ref, years, fixed_rate, curve, receiver).NPV()


def shock_quotes(quotes_pct: dict[int, float], s2_bp: float, s10_bp: float,
                 s5_bp: float | None = None) -> dict[int, float]:
    """Piecewise-linear shock across tenors: s2 at 2Y, s10 at 10Y (optionally s5 at 5Y)."""
    xs, ys = ([2, 10], [s2_bp, s10_bp]) if s5_bp is None else ([2, 5, 10], [s2_bp, s5_bp, s10_bp])
    return {n: q + float(np.interp(n, xs, ys)) / 100.0 for n, q in quotes_pct.items()}


def steepener_scenarios(ref_ts: pd.Timestamp, par_yields_pct: pd.Series,
                        notional_2y: float = 100e6) -> tuple[pd.DataFrame, dict]:
    """Receive 2Y / pay 10Y SOFR swaps, 10Y sized DV01-neutral. Revalue under curve shocks."""
    ref = qdate(ref_ts)
    ql.Settings.instance().evaluationDate = ref
    base = proxy_sofr_quotes(par_yields_pct)
    base_curve = build_sofr_curve(ref, base)
    k2, k10 = par_swap_rate(ref, 2, base_curve), par_swap_rate(ref, 10, base_curve)

    def dv01(years: int, k: float) -> float:  # receiver gain per 1bp fall, per 1m notional
        up = swap_npv(ref, years, k, shock_quotes(base, 1, 1), receiver=True)
        return swap_npv(ref, years, k, base, receiver=True) - up

    d2, d10 = dv01(2, k2), dv01(10, k10)
    n10 = notional_2y * d2 / d10
    scenarios = {
        "Parallel +25bp": (25, 25, None), "Parallel -25bp": (-25, -25, None),
        "Bear steepener (10Y +25)": (0, 25, None), "Bull steepener (2Y -25)": (-25, 0, None),
        "Twist steepener (2Y -12.5, 10Y +12.5)": (-12.5, 12.5, None),
        "Twist flattener (2Y +12.5, 10Y -12.5)": (12.5, -12.5, None),
        "Belly +10bp (5Y)": (0, 0, 10),
    }
    rows = []
    for name, (s2, s10, s5) in scenarios.items():
        q = shock_quotes(base, s2, s10, s5)
        pnl2 = swap_npv(ref, 2, k2, q, True) * notional_2y / 1e6
        pnl10 = swap_npv(ref, 10, k10, q, False) * n10 / 1e6
        rows.append({"scenario": name, "2Y receiver P&L": pnl2, "10Y payer P&L": pnl10,
                     "net P&L (USD)": pnl2 + pnl10})
    info = {"fixed_2y": k2, "fixed_10y": k10, "notional_10y": n10,
            "dv01_2y_usd": d2 * notional_2y / 1e6, "dv01_10y_usd": d10 * n10 / 1e6}
    return pd.DataFrame(rows).set_index("scenario"), info
