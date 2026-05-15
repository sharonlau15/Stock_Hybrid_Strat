"""utils/logger.py — Loguru setup."""

import sys
from loguru import logger
from config.settings import LOG_DIR, LOG_LEVEL, LOG_ROTATION


def setup_logger():
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "algo_{time:YYYY-MM-DD}.log",
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention="4 weeks",
    )
