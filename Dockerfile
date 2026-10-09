# 应用镜像:API 与 Worker 共用,由 docker-compose 编排(CMD 起 API,worker 服务覆盖 command)。
# 容器内以 `python -m` 启动,与 README 开发路径同一入口;console script 供宿主机/uvx 使用。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# uv:按 uv.lock 做可复现锁定安装。PyPI 发行,不走 ghcr(部分网络不可达)。
RUN pip install --no-cache-dir uv

# 先装依赖再拷代码,依赖层在源码变更时保持缓存命中。
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# tiktoken 的 BPE 表在首次使用时按 URL 下载;构建期预取进缓存,运行时不再出网。
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

RUN useradd -m appuser \
    && mkdir -p /app/var/service/events /app/var/logs \
    && chown -R appuser:appuser /app "$TIKTOKEN_CACHE_DIR"
USER appuser

EXPOSE 8080

CMD ["python", "-m", "deepresearcher.service.main"]
