"""service.execution.telemetry 的全局 provider 装配:复用、幂等与非法值降级。"""

import logging

import pytest
from opentelemetry import trace as opentelemetry_trace
from opentelemetry.sdk.trace import TracerProvider

from deepresearcher.service.execution.telemetry import configure_tracer_provider
from fakes_service import service_config


def test_configure_creates_or_reuses_single_sdk_provider(tmp_path, caplog):
    # 全局 provider 每进程只能安装一次,本用例可能是首次创建也可能是复用
    # 先装的,故断言只收敛在:配置后全局是 SDK provider 且引用不变。
    first = service_config(tmp_path, otel_exporter="none")
    configure_tracer_provider(first)
    provider_after_first = opentelemetry_trace.get_tracer_provider()
    configure_tracer_provider(first)
    configure_tracer_provider(first)
    assert opentelemetry_trace.get_tracer_provider() is provider_after_first
    assert isinstance(provider_after_first, TracerProvider)


def test_unknown_exporter_value_warns_and_behaves_as_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="deepresearcher.service.execution.telemetry"):
        configure_tracer_provider(service_config(tmp_path, otel_exporter="jaeger-ish"))
    assert any("SERVICE_OTEL_EXPORTER" in record.getMessage() for record in caplog.records)
    assert isinstance(opentelemetry_trace.get_tracer_provider(), TracerProvider)


@pytest.mark.parametrize("exporter", ["none", "NONE", "Console", "jaeger"])
def test_configure_never_raises_for_any_string_value(tmp_path, exporter):
    # 任何字符串取值都不得让装配抛错,否则 worker 起不来。console 的正常路径
    # 由端到端用例 test_otel_console_exporter_completes_run 覆盖,此处避免把
    # ConsoleSpanProcessor 挂上全局 provider。
    configure_tracer_provider(service_config(tmp_path, otel_exporter=exporter))
