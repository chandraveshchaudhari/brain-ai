"""Logging and progress utilities for Brain-AI runs.

Sets up a file logger (full DEBUG) + stdout handler (INFO, visible in Jupyter)
so that progress prints appear under notebook cells without any notebook changes.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_run_logger(
    output_dir: Path,
    name: str = "brain_automl",
    level: str = "INFO",
) -> logging.Logger:
    """Return a logger that writes to *output_dir/logs/* and to stdout.

    The stdout handler shows messages in Jupyter notebook cell output.
    A unique timestamp suffix prevents handler accumulation across re-runs.
    """
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{name}_{timestamp}.log"

    # Use a timestamped logger name so re-running a cell doesn't accumulate handlers.
    logger_name = f"{name}.{timestamp}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — captures DEBUG and above for post-run inspection.
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler — INFO and above, visible in Jupyter cell output.
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Brain-AI run started")
    logger.info(f"Log file : {log_file}")
    logger.info(f"Output dir: {output_dir}")
    return logger
