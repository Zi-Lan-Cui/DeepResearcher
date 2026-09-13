"""阿里云 DTS AI SDK 的窄接口封装。"""

from deepresearcher.tools.web.aliyun.client import (
    AliyunDtsApi,
    AliyunDtsClient,
    create_aliyun_dts_client,
)

__all__ = ["AliyunDtsApi", "AliyunDtsClient", "create_aliyun_dts_client"]
