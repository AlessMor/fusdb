"""Central logging helpers shared across fusdb classes."""
from __future__ import annotations

import logging


def make_logger(
    module_logger: logging.Logger,
    owner: str,
    *,
    verbose: bool,
) -> logging.Logger:
    """Return child logger configured for the requested verbosity."""
    log = module_logger.getChild(owner)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    return log


def set_log_verbosity(log: logging.Logger, *, verbose: bool) -> None:
    """Set logger level from boolean verbosity."""
    log.setLevel(logging.INFO if verbose else logging.WARNING)


def log_message(log: logging.Logger, level: int, msg: str, *args: object) -> None:
    """Emit one log message preserving the class call-site location."""
    log.log(level, msg, *args, stacklevel=2)
