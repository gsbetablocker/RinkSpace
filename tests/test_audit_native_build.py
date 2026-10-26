"""Artifact discovery after a successful platform-specific native build."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(os.name == "nt", reason="POSIX platform-name selection")
@pytest.mark.parametrize("platform, filename", [
    ("darwin", "librinkspace_core.dylib"),
    ("linux", "librinkspace_core.so"),
])
def test_build_finds_the_library_produced_by_cmake(tmp_path, monkeypatch, platform, filename):
    script = Path(__file__).resolve().parents[1] / "scripts/build_core.py"
    spec = importlib.util.spec_from_file_location("audit_build_core", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "native-build"
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.sys, "argv", [str(script), "--out", str(output), "--cmake", "cmake"])

    def build(command, **kwargs):
        if "--build" in command:
            (output / filename).write_bytes(b"mock toolchain output")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", build)
    assert module.main() == 0
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    assert manifest["library"] == str((output / filename).resolve())
