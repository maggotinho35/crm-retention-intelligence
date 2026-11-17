"""Fast checks that the building blocks behave. Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from crm_bi import tool_selection as ts  # noqa: E402
from crm_bi.business_impact import customer_value  # noqa: E402
from crm_bi.config import ASSUMPTIONS, PRODUCTS  # noqa: E402
from crm_bi.data_generation import generate  # noqa: E402


def test_generator_is_reproducible():
    a, _, _ = generate(seed=1, n=300)
    b, _, _ = generate(seed=1, n=300)
    assert a.equals(b)


def test_generator_matches_documented_churn_rate():
    c, comp, tx = generate()
    assert abs(c.churned.mean() - 0.246) < 0.01
    assert len(c) == 1200 and len(tx) == 1200
    assert set(comp.customer_id) <= set(c.customer_id)
    assert set(PRODUCTS) <= set(tx.columns)


def test_no_missing_values():
    c, comp, tx = generate(n=400)
    assert not c.isna().any().any() and not comp.isna().any().any() and not tx.isna().any().any()


def test_customer_value_positive_and_increasing_in_balance():
    c, _, _ = generate(n=200)
    v = customer_value(c, ASSUMPTIONS)
    assert (v > 0).all()
    hi = c.copy()
    hi["avg_balance"] = hi["avg_balance"] * 2
    assert (customer_value(hi, ASSUMPTIONS) > v).all()


def test_tool_scores_reproduce_portfolio():
    s = ts.weighted_scores()
    assert np.allclose(sorted(s.values), [3.35, 3.5, 4.2])
    assert s.idxmax().startswith("B")
