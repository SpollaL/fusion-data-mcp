import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live network access. "
        "Run with --run-integration or RUN_INTEGRATION_TESTS=1.",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live network access.",
    )


def pytest_collection_modifyitems(config, items):
    run_integration = (
        config.getoption("--run-integration")
        or os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"
    )
    skip_integration = pytest.mark.skip(
        reason="Live network test. Use --run-integration or RUN_INTEGRATION_TESTS=1."
    )
    if not run_integration:
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_integration)
