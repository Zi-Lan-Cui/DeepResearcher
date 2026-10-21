"""Evidence 数据契约与确定性校验基础。

本包 __init__ 不 re-export Evidence，避免与 schemas 的循环 import 在半初始化时报错；
需要时用 from deepresearcher.evidence.models import Evidence。
"""
