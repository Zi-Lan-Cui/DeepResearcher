"""LLM 调用的基础设施重试策略。"""

from langchain_core.runnables import Runnable

from deepsearch_agent.config import LLMRetryConfig

TRANSIENT_ERRORS = (TimeoutError, ConnectionError)


def with_transport_retry(runnable: Runnable, policy: LLMRetryConfig) -> Runnable:
    return runnable.with_retry(
        retry_if_exception_type=TRANSIENT_ERRORS,
        stop_after_attempt=policy.transport_attempts,
        wait_exponential_jitter=True,
        exponential_jitter_params={
            "initial": policy.initial_seconds,
            "max": policy.max_seconds,
            "exp_base": policy.exp_base,
            "jitter": policy.jitter,
        },
    )
