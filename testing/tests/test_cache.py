"""P5: semantic cache isolation. The leak test is red with naive keying, green per customer."""
from __future__ import annotations

from atlas.domain.cache import NaiveCache, PerCustomerCache


def test_naive_cache_leaks_across_customers():
    c = NaiveCache()
    c.put("cust_legacy_term", "what's my bill?", "Daniel's bill is GBP 39", generic=False)
    # Sarah asks the same question and receives Daniel's answer, the isolation bug.
    assert c.get("cust_current", "what's my bill?", generic=False) == "Daniel's bill is GBP 39"


def test_per_customer_cache_does_not_leak():
    c = PerCustomerCache()
    c.put("cust_legacy_term", "what's my bill?", "Daniel's bill is GBP 39", generic=False)
    assert c.get("cust_current", "what's my bill?", generic=False) is None
    assert c.get("cust_legacy_term", "what's my bill?", generic=False) == "Daniel's bill is GBP 39"


def test_generic_answers_are_shared():
    c = PerCustomerCache()
    c.put("cust_legacy_term", "what are your opening hours?", "9 to 5", generic=True)
    assert c.get("cust_current", "what are your opening hours?", generic=True) == "9 to 5"
