"""Central configuration: paths, tenor maps, FRED series ids and research parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
NOTES_DIR = ROOT / "notes"

START_DATE = "2003-01-01"  # T10YIE (breakevens) starts in 2003

# Treasury constant-maturity par yields (percent) on FRED
TENOR_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2",
    "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}
TENOR_YEARS = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}
# Short bills are policy-rate driven, so factor analysis uses the coupon part of the curve
PCA_TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# Macro drivers. Key = our name, value = FRED id
MACRO_SERIES = {"FFR": "DFF", "BEI10": "T10YIE", "UNRATE": "UNRATE", "CPI": "CPIAUCSL"}
# Days to shift each series forward so we only use it after it was published (no look-ahead)
RELEASE_LAG_DAYS = {"BEI10": 1, "UNRATE": 35, "CPI": 45}

TRADING_DAYS = 252
COST_BP_PER_LEG = 0.25  # assumed half bid/ask per leg, in yield bp


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROC, FIG_DIR, REPORT_DIR, NOTES_DIR):
        p.mkdir(parents=True, exist_ok=True)
