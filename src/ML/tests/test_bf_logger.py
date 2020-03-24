import pytest
import logging
from bf_logging import bf_log

def test_logger_returns_same_logger():
    bf_log.setup_logger()
    logger_1 = logging.getLogger(__name__)

    bf_log.setup_logger()
    logger_2 = logging.getLogger(__name__)

    assert logger_1 is logger_2
