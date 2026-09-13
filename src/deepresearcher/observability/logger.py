import logging


def get_logger(name: str = "deepresearcher") -> logging.Logger:
    return logging.getLogger(name)
