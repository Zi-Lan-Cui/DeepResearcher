"""LLM 单价表：记账币种人民币，单位 元/百万 token。

价格按公开牌价登记，缺价模型不猜价——落库记 0 并保留 unpriced 版本戳，
对账时缺价是可见事实而非隐藏误差。牌价更动时同步更新 PRICE_TABLE_VERSION，
重算入口按版本戳幂等回填历史行。
"""

from dataclasses import dataclass
from decimal import Decimal

# 现行牌价批次（小米 MiMo 2026-05-27 永久调价公告生效日）。
PRICE_TABLE_VERSION = "2026-05-27-cny"


@dataclass(frozen=True)
class ModelPrice:
    input_cny_per_million: float
    cached_input_cny_per_million: float
    output_cny_per_million: float


# 键为 provider 回执的模型名（run_usage.model 列）。
MODEL_PRICES: dict[str, ModelPrice] = {
    # 小米 MiMo 国内牌价：缓存写入限时免费，记 0；海外平台按美元另有牌价，本表不掺。
    # 官方公告 2026-10-14 起 v2.5 自动路由至 v2.6 系列，届时回执模型名若不在表内即落 unpriced。
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
    """按模型名查现行牌价；未登记返回 None（记 unpriced）。"""

    if not model:
        return None
    return MODEL_PRICES.get(model)


def cost_cny(
    model: str | None, input_tokens: int, output_tokens: int, cached_input_tokens: int
) -> Decimal:
    """按现行牌价计一次调用的人民币成本；缺价返回 0。"""

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
