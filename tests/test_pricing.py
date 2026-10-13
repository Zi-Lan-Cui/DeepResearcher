from decimal import Decimal

from deepresearcher import pricing
from deepresearcher.pricing import ModelPrice, cost_cny, price_for_model


def test_cost_cny_splits_cached_and_uncached_input(monkeypatch):
    monkeypatch.setitem(
        pricing.MODEL_PRICES,
        "unit-model-x",
        ModelPrice(
            input_cny_per_million=1.0, cached_input_cny_per_million=0.5, output_cny_per_million=2.0
        ),
    )
    # 未命中 600×1 + 命中 400×0.5 + 输出 20×2 = 840 元/百万 → 1000 token 计 0.00084 元。
    assert cost_cny("unit-model-x", 1000, 20, 400) == Decimal("0.00084000")


def test_unlisted_model_costs_zero(monkeypatch):
    monkeypatch.setattr(pricing, "MODEL_PRICES", {})
    assert price_for_model("no-such-model") is None
    assert cost_cny("no-such-model", 1000, 100, 0) == Decimal("0")


def test_shipped_price_entries_are_positive():
    assert pricing.PRICE_TABLE_VERSION
    for model, price in pricing.MODEL_PRICES.items():
        assert model
        assert price.input_cny_per_million > 0
        assert price.output_cny_per_million > 0
        assert price.cached_input_cny_per_million >= 0
