import logging
import pytest
import shutil

# @pytest.fixture
# def dummy_logger():
#     logger = logging.getLogger("test_logger")
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(logging.NullHandler())  # Prevents actual logging output
#     return logger

# Create a dummy_logger with a real handler so our get_logger_directory function works
@pytest.fixture
def dummy_logger():
    import tempfile
    import os
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    temp_dir = tempfile.mkdtemp()
    temp_log_path = os.path.join(temp_dir, "test.log")

    handler = logging.FileHandler(temp_log_path)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    yield logger

    # Teardown: close handlers and remove temp dir
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    shutil.rmtree(temp_dir)


# see https://docs.pytest.org/en/stable/example/simple.html#control-skipping-of-tests-according-to-command-line-option
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)