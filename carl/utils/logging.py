from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "carl") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("CARL_LOGLEVEL", "INFO").upper())
    logger.propagate = False
    return logger
