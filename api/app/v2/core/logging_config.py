import logging


def build_log_config(level: str = "INFO") -> dict:
    """Uvicorn reads this dict and applies it via logging.config.dictConfig
    at startup, replacing its own default logger setup. v1 logger didn't
    override uvicorn's default logger"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "access": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "api": {"handlers": ["default"], "level": level, "propagate": False},
            "api_v2": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": level,
                "propagate": False,
            },
        },
    }


def init_logger(name: str = "api_v2") -> logging.Logger:
    """instantiate logger"""
    return logging.getLogger(name)
