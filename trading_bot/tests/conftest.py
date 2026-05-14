"""Pytest setup — use an isolated SQLite file per test session."""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _temp_db_env():
    """Force a temp database for the duration of the test session."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp.name}"
    os.environ["LIVE_TRADING"] = "false"
    os.environ["TRADING_MODE"] = "paper"
    os.environ["HUMAN_APPROVAL_MODE"] = "false"  # let tests exercise the broker
    yield
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
