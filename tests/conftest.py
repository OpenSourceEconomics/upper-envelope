import os
import sys
from pathlib import Path

import jax

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"

WEALTH_GRID_POINTS = 100

# Add the utils directory to the path so that we can import helper functions.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def pytest_sessionstart(session):  # noqa: ARG001
    jax.config.update("jax_enable_x64", val=True)


def pytest_configure(config):  # noqa: ARG001
    """Called after command line options have been parsed."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"


def pytest_unconfigure(config):  # noqa: ARG001
    """Called before test process is exited."""
    os.environ.pop("NUMBA_DISABLE_JIT", None)
