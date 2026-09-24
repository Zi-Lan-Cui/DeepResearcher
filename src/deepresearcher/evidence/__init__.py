"""Evidence 数据契约与确定性校验基础。

本包 __init__ 刻意不 re-export ``Evidence``：schemas.sections(顶层契约)import
evidence.models,而 evidence.models import schemas.sources——两个包互相引用,
急切 re-export 会让"先导入 evidence、后导入 schemas"的顺序在半初始化的包上直接报错。
需要 Evidence 的模块请 ``from deepresearcher.evidence.models import Evidence``。
"""
