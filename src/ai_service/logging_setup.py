"""
logging_setup.py - JSON logs with the request id on every line.

WHAT: configures `structlog`. Call `setup_logging()` once at start-up, then `log = structlog.get_logger()` anywhere.
WHY:  shared-contracts section 9: every log line must carry requestId / paymentId so one payment can be followed across services.

How to use it:
    log.info("l1 finished", checks=3)                       # a normal line
    structlog.contextvars.bind_contextvars(requestId=...)   # from now on, in this thread, every line carries requestId
Never log the message body or an account number: log ids, hashes and lengths.
"""
from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,   # adds the values bound with bind_contextvars
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
