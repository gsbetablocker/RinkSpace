"""Windows runtime directories are shared across native boundary instances."""
import os
from pathlib import Path

import pytest
from rinkspace.core import native

pytestmark = pytest.mark.skipif(
    os.name != "nt" or not os.environ.get("RINKSPACE_CORE_LIBRARY"),
    reason="explicit Windows native library required",
)


def test_repeated_native_loads_reuse_search_directory_registrations(monkeypatch):
    registrations = []
    original = os.add_dll_directory
    monkeypatch.setattr(native, "_DLL_DIRECTORY_HANDLES", {})

    def register(directory):
        registrations.append(Path(directory).resolve())
        return original(directory)

    monkeypatch.setattr(os, "add_dll_directory", register)
    cores = [native.NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"]) for _ in range(32)]
    assert all(core.status_message(0) for core in cores)
    assert registrations
    assert len(registrations) == len(set(registrations))
    assert all(core._dll_dirs == cores[0]._dll_dirs for core in cores)
