"""Regression checks for native child access and cmd.exe quoting."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location("build_core", Path(__file__).resolve().parents[1]/"scripts/build_core.py")
build_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_core)


def test_failed_child_traversal_stops_before_cmake_and_preserves_records(tmp_path, monkeypatch, capsys):
    output = tmp_path/"build"
    output.mkdir()
    (output/"build.log").write_text("previous failure\n", encoding="utf-8")
    (output/"build.json").write_text("previous library identity\n", encoding="utf-8")
    calls = []

    def denied(body, env, cwd):
        calls.append(body)
        return subprocess.CompletedProcess(body, 1, "", "Access is denied.")

    def preflight(args, project, out):
        build_core.windows_preflight(project, out, {})
        pytest.fail("An inaccessible child directory must stop the build")

    def no_compiler(*args, **kwargs):
        pytest.fail("CMake/compiler must not start after a failed child access probe")

    monkeypatch.setattr(build_core, "windows_shell", denied)
    monkeypatch.setattr(build_core, "prepare_toolchain", preflight)
    monkeypatch.setattr(build_core.subprocess, "run", no_compiler)
    monkeypatch.setattr(sys, "argv", ["build_core.py", "--out", str(output), "--cuda"])
    assert build_core.main() == 2
    assert len(calls) == 1
    assert "cd /d" in calls[0]
    assert "sandbox_permissions=require_escalated" in capsys.readouterr().err
    assert (output/"build.log").read_text() == "previous failure\n"
    assert (output/"build.json").read_text() == "previous library identity\n"


def test_toolchain_check_does_not_compile_or_replace_build_identity(tmp_path, monkeypatch):
    (tmp_path/"build.json").write_text("existing library\n", encoding="utf-8")
    monkeypatch.setattr(build_core, "prepare_toolchain", lambda *a: ({}, {"cmake": "cmake"}))
    monkeypatch.setattr(build_core.subprocess, "run", lambda *a, **kw: pytest.fail("check-only must not compile"))
    monkeypatch.setattr(sys, "argv", ["build_core.py", "--out", str(tmp_path), "--check-toolchain"])
    assert build_core.main() == 0
    assert (tmp_path/"build.json").read_text() == "existing library\n"
    assert not (tmp_path/"build.log").exists()


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe integration")
def test_native_child_access_with_spaces_and_cmd_metacharacters(tmp_path):
    project = tmp_path/"project space & 中文 %literal% !"
    (project/"native").mkdir(parents=True)
    (project/"native/CMakeLists.txt").write_text("project(probe)\n", encoding="utf-8")
    output = tmp_path/"build space & 中文 %literal% !"
    output.mkdir()
    env = {k.upper(): v for k, v in os.environ.items()}
    build_core.windows_preflight(project, output, env)
    assert list(output.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe integration")
def test_vcvars_call_uses_cmd_quoting_and_preserves_environment_values(tmp_path):
    vcvars = tmp_path/"tools with spaces & chars"/"vcvars64.bat"
    vcvars.parent.mkdir()
    vcvars.write_text('@set "RINKSPACE_TEST_VALUE=spaces & value=42"\n', encoding="ascii")
    env = {k.upper(): v for k, v in os.environ.items()}
    result = build_core.windows_shell('call "%RINKSPACE_VCVARS%" >nul && set RINKSPACE_TEST_VALUE',
                                     {**env, "RINKSPACE_VCVARS": str(vcvars)}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "RINKSPACE_TEST_VALUE=spaces & value=42" in result.stdout
