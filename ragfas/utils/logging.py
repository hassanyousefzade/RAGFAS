# -*- coding: utf-8 -*-
"""Console + file logging used by every entry point."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Tuple

from ragfas.utils.paths import check_dir


def setup_logger(name: str, log_dir: str, tag: str = "run") -> Tuple[logging.Logger, str]:
    """Create a logger writing to both stdout and ``<log_dir>/<tag>_<ts>.log``."""
    check_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{log_dir}/{tag}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger, log_path


def show_tqdm() -> bool:
    """Disable progress bars under nohup so the log file stays clean."""
    return sys.stderr.isatty()


__all__ = ["setup_logger", "show_tqdm"]