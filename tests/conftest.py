import logging
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())  # Prevents actual logging output
    return logger

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)
