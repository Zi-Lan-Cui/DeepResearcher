"""内置网页抓取 Provider。"""

from deepresearcher.tools.web.fetch.providers.aliyun import AliyunFetchProvider
from deepresearcher.tools.web.fetch.providers.direct import DirectHttpFetchProvider

__all__ = ["AliyunFetchProvider", "DirectHttpFetchProvider"]
