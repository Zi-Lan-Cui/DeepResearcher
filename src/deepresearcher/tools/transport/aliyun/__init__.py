"""阿里云网页搜索/抓取 SDK 的窄接口封装。"""

from deepresearcher.tools.transport.aliyun.client import (
    AliyunDtsApi,
    AliyunDtsClient,
    AliyunWebFetchBody,
    AliyunWebSearchBody,
    AliyunWebSearchItem,
    create_aliyun_dts_client,
)

__all__ = [
    "AliyunDtsApi",
    "AliyunDtsClient",
    "AliyunWebFetchBody",
    "AliyunWebSearchBody",
    "AliyunWebSearchItem",
    "create_aliyun_dts_client",
]
