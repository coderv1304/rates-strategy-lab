import pytest

ql = pytest.importorskip("QuantLib")

from rateslab.analytics import bond_metrics  # noqa: E402
from rateslab.ql_tools import bond_cross_check  # noqa: E402


@pytest.mark.parametrize("y,years", [(3.5, 2), (4.2, 10), (5.0, 5)])
def test_my_analytics_match_quantlib(y, years):
    mine, theirs = bond_metrics(y, years), bond_cross_check(y, years)
    assert mine.price == pytest.approx(theirs["price"], abs=1e-6)
    assert mine.macaulay == pytest.approx(theirs["macaulay"], abs=1e-6)
    assert mine.modified == pytest.approx(theirs["modified"], abs=1e-6)
    assert mine.convexity == pytest.approx(theirs["convexity"], rel=1e-6)
