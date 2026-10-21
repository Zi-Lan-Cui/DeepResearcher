"""LLM 单价表：计价币种人民币，单位 元/百万 token。

价格按厂商公开价目登记，未登记价目的模型不估算：成本记 0，并在
price_version 列写 unpriced，让缺价在数据中可见，而不是混进总额。
价目变动时同步更新 PRICE_TABLE_VERSION，重算入口按版本戳回填历史行。
"""

from dataclasses import dataclass
from decimal import Decimal

# 现行价目批次（小米 MiMo V2.6 系列，2026-09-22 发布，延续 V2.5 定价）。
PRICE_TABLE_VERSION = "2026-10-22-cny-v26"


@dataclass(frozen=True)
class ModelPrice:
    input_cny_per_million: float
    cached_input_cny_per_million: float
    output_cny_per_million: float


# 键为 provider 回执中的模型名（对应 run_usage.model 列）。
# V2.6 与 V2.5 定价相同；V2.5 系列已于 2026-10-21 下线，条目保留供重算历史用量。
MODEL_PRICES: dict[str, ModelPrice] = {
    # 小米 MiMo 国内价目。缓存命中按 99% 折扣计价；海外平台另按美元标价，本表不采。
    "mimo-v2.6-flash": ModelPrice(
        input_cny_per_million=1.0,
        cached_input_cny_per_million=0.02,
        output_cny_per_million=2.0,
    ),
    "mimo-v2.6-pro": ModelPrice(
        input_cny_per_million=3.0,
        cached_input_cny_per_million=0.025,
        output_cny_per_million=6.0,
    ),
    "mimo-v2.5": ModelPrice(
        input_cny_per_million=1.0,
        cached_input_cny_per_million=0.02,
        output_cny_per_million=2.0,
    ),
    "mimo-v2.5-pro": ModelPrice(
        input_cny_per_million=3.0,
        cached_input_cny_per_million=0.025,
        output_cny_per_million=6.0,
    ),
}


def price_for_model(model: str | None) -> ModelPrice | None:
    """按模型名查现行价目；未登记返回 None（记 unpriced）。"""

    if not model:
        return None
    return MODEL_PRICES.get(model)


def cost_cny(
    model: str | None, input_tokens: int, output_tokens: int, cached_input_tokens: int
) -> Decimal:
    """按现行价目计一次调用的人民币成本；缺价返回 0。"""

    price = price_for_model(model)
    if price is None:
        return Decimal("0")
    uncached = max(0, input_tokens - cached_input_tokens)
    total = (
        Decimal(uncached) * Decimal(str(price.input_cny_per_million))
        + Decimal(cached_input_tokens) * Decimal(str(price.cached_input_cny_per_million))
        + Decimal(output_tokens) * Decimal(str(price.output_cny_per_million))
    ) / Decimal(1_000_000)
    return total.quantize(Decimal("0.00000001"))
