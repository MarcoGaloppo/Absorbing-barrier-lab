"""
Root conftest: puts the repository root on ``sys.path`` so ``tests/`` can import
``tennis_lab``, and defines the ``slow`` marker.

Slow tests are the exhaustive barrier-ceiling searches - hundreds of thousands
of simulated matches. They are skipped by default and run with ``--runslow``,
which is what CI does on the scheduled job.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the exhaustive Monte-Carlo tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: exhaustive Monte-Carlo test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
