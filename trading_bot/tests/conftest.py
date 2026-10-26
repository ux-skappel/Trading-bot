"""Pytest setup: an isolated SQLite file for the whole test session.

The environment is set at import time, before any test module imports
``trading_bot`` — the database engine is created from the settings on first
import, so setting it later (e.g. in a fixture) would be too late and the
tests would write to the real ./trading_bot.db.
"""

from __future__ import annotations

import atexit
import os
import tempfile

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp.name}"
os.environ["LIVE_TRADING"] = "false"
os.environ["TRADING_MODE"] = "paper"
os.environ["HUMAN_APPROVAL_MODE"] = "false"  # let tests exercise the broker


@atexit.register
def _cleanup() -> None:
    try:
        os.unlink(_tmp.name)
    except OSError:
        pass
