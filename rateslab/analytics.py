"""Closed-form fixed income analytics: price, duration, convexity, DV01, hedge ratios, carry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BondMetrics:
    price: float       # per 100 face
    macaulay: float    # years
    modified: float    # years
    convexity: float   # years^2
    dv01: float        # price change per 1bp, per 100 face


def bond_cashflows(coupon_pct: float, years: float, freq: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(years * freq))
    t = np.arange(1, n + 1) / freq
    cf = np.full(n, coupon_pct / freq)
    cf[-1] += 100.0
    return t, cf


def bond_metrics(yield_pct: float, years: float, coupon_pct: float | None = None,
                 freq: int = 2) -> BondMetrics:
    """Metrics for a bond priced on a coupon date. Coupon defaults to the yield (par bond)."""
    coupon_pct = yield_pct if coupon_pct is None else coupon_pct
    y = yield_pct / 100.0
    t, cf = bond_cashflows(coupon_pct, years, freq)
    disc = (1.0 + y / freq) ** (-t * freq)
    pv = cf * disc
    price = pv.sum()
    mac = (t * pv).sum() / price
    mod = mac / (1.0 + y / freq)
    conv = (t * (t + 1.0 / freq) * pv).sum() / (price * (1.0 + y / freq) ** 2)
    return BondMetrics(price, mac, mod, conv, mod * price * 1e-4)


def par_mod_duration(y_dec, years: float, freq: int = 2):
    """Vectorised modified duration of a par bond: (1 - (1+y/f)^(-n f)) / y."""
    y = np.maximum(np.asarray(y_dec, dtype=float), 1e-6)
    return (1.0 - (1.0 + y / freq) ** (-years * freq)) / y


def price_change_pct(m: BondMetrics, dy_bp: float) -> float:
    """Duration + convexity approximation of the percent price change for a yield shock."""
    dy = dy_bp * 1e-4
    return (-m.modified * dy + 0.5 * m.convexity * dy**2) * 100.0


def steepener_hedge(y2_pct: float, y10_pct: float, notional_2y: float = 100e6) -> dict:
    """DV01-neutral sizing of a 2s10s steepener: long 2Y, short 10Y."""
    m2, m10 = bond_metrics(y2_pct, 2), bond_metrics(y10_pct, 10)
    dv01_2 = notional_2y / 100.0 * m2.dv01
    notional_10y = notional_2y * m2.dv01 / m10.dv01
    dv01_10 = notional_10y / 100.0 * m10.dv01
    return {
        "dv01_2y_usd": dv01_2, "dv01_10y_usd": dv01_10,
        "notional_10y": notional_10y, "hedge_ratio_10_per_2": notional_10y / notional_2y,
        "net_dv01_usd": dv01_2 - dv01_10,
    }


def steepener_carry_bp_per_day(y2_pct, y10_pct, repo_pct):
    """Financing-adjusted carry of a DV01-neutral steepener, in bp-equivalents per day.

    Unit position = $1 of DV01 on each leg. Income per $1 DV01 on a leg is (y - repo) / (D * 1e-4)
    per year. Roll-down is ignored (documented limitation). Inputs may be arrays.
    """
    y2, y10, r = (np.asarray(v, dtype=float) / 100.0 for v in (y2_pct, y10_pct, repo_pct))
    d2, d10 = par_mod_duration(y2, 2), par_mod_duration(y10, 10)
    annual = (y2 - r) / (d2 * 1e-4) - (y10 - r) / (d10 * 1e-4)
    return annual / 252.0
