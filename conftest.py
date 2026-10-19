"""Keep pytest's disposable cache outside the synchronized checkout."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    root = Path(__file__).resolve().parent
    storage = root / "configs/storage.json"
    if storage.is_file():
        registered = json.loads(storage.read_text(encoding="utf-8-sig"))
        data = Path(registered["data_root"]).resolve()
    else:
        data = Path(tempfile.gettempdir()) / "RinkSpace_Data"
    # Explicit user overrides still follow pytest's normal configuration rules.
    config.inicfg.setdefault("cache_dir", str(data / "cache" / "pytest"))
