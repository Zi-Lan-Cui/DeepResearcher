"""编排层的可诊断错误。"""

from deepsearch_agent.errors import AgentError


class WriterError(AgentError):
    """研究报告无法生成可审阅、可追溯的段落草稿。"""

    code = "writer_error"


class WriterGenerationError(WriterError):
    """LLM 或结构化输出层无法生成报告草稿，不能由改稿流程安全修复。"""

    code = "writer_generation"
