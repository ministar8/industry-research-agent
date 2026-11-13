import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test.

    Streamlit resolves its config directory from the user's home directory, so the
    OS-level variables it needs are preserved: clearing them makes the script run
    crash before it can report a result.
    """
    preserved = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "SYSTEMROOT", "PATH"}
    }
    with patch.dict(os.environ, preserved, clear=True):
        yield
