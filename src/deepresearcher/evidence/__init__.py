"""Evidence 数据契约与确定性校验基础。

本包 __init__ 刻意不 re-export ``Evidence``：schemas.sections(顶层契约)import
evidence.models,而 evidence.models import schemas.sources——两个包互相引用,
急切 re-export 会让"先碰 evidence 的后碰 schemas"的导入顺序直接炸半初始化。
需要 Evidence 的模块请 ``from deepresearcher.evidence.models import Evidence``。
"""

from deepresearcher.evidence.tokens import get_token_estimator

__all__ = ["get_token_estimator"]
