"""Keep pytest's disposable cache outside the synchronized checkout."""
from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rinkspace.storage import storage_cache_root


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not config.pluginmanager.hasplugin("cacheprovider"):
        return
    data = storage_cache_root(Path(__file__).resolve().parent)
    config.inicfg.setdefault("cache_dir", str(data / "cache" / "pytest"))
