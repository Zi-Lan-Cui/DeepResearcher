import logging


def get_logger(name: str = "deepsearch_agent") -> logging.Logger:
    return logging.getLogger(name)
