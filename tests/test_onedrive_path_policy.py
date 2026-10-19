"""The sync checkout must remain usable by ordinary Windows path APIs."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from rinkspace import path_policy
from rinkspace.io import dump_json
from rinkspace.path_policy import (
    OneDrivePathError, check_project_path, guard_project_write,
    scan_project_paths, windows_path_units,
)


ROOT = Path(__file__).resolve().parents[1]


def _sized_path(root: Path, units: int) -> Path:
    prefix = root / "nested"
    remaining = units - windows_path_units(str(prefix)) - 1
    assert 0 < remaining <= 255
    return prefix / ("a" * remaining)


def _mock_reparse_directory(monkeypatch, directory: Path, tag: int):
    original_scandir = path_policy.os.scandir

    class Entry:
        def __init__(self, original):
            self.original = original
            self.path = original.path

        def stat(self, *, follow_symlinks):
            metadata = self.original.stat(follow_symlinks=follow_symlinks)
            if Path(self.path) == directory:
                return SimpleNamespace(st_mode=metadata.st_mode, st_reparse_tag=tag)
            return metadata

    class Entries:
        def __init__(self, path):
            self.original = original_scandir(path)

        def __enter__(self):
            return (Entry(item) for item in self.original)

        def __exit__(self, *args):
            self.original.close()

    monkeypatch.setattr(path_policy.os, "scandir", Entries)


def test_absolute_windows_boundary_uses_physical_path(tmp_path):
    project = tmp_path / "project"
    allowed = _sized_path(project, 259)
    rejected = _sized_path(project, 260)
    assert not check_project_path(allowed, project_root=project)
    assert check_project_path(rejected, project_root=project)[0].actual_units == 260
    with pytest.raises(OneDrivePathError, match="未创建文件"):
        guard_project_write(rejected, project_root=project)
    assert not project.exists()


@pytest.mark.skipif(os.name != "nt", reason="Extended drive paths are a Windows syntax")
def test_extended_path_prefix_cannot_bypass_project_guard(tmp_path):
    project = tmp_path / "project"
    rejected = _sized_path(project, 260)
    extended = "\\\\?\\" + str(rejected)
    assert check_project_path(extended, project_root=project)[0].actual_units == 260
    with pytest.raises(OneDrivePathError):
        guard_project_write(extended, project_root=project)
    assert not project.exists()


def test_utf16_surrogate_units_and_component_boundaries(tmp_path):
    project = tmp_path / "project"
    assert windows_path_units("甲😀") == 3
    path = _sized_path(project, 258)
    assert not check_project_path(path, project_root=project)
    assert check_project_path(path.with_name(path.name + "😀"), project_root=project)[0].actual_units == 260
    allowed_component = check_project_path(project / ("a" * 255), project_root=project)
    rejected_component = check_project_path(project / ("a" * 256), project_root=project)
    assert not any(row.reason == "component" for row in allowed_component)
    assert any(row.reason == "component" for row in rejected_component)


def test_data_long_paths_keep_existing_support(tmp_path):
    project = tmp_path / "OneDrive" / "project"
    data = tmp_path / "RinkSpace_Data" / ("a" * 100) / ("b" * 100) / ("c" * 100)
    assert windows_path_units(str(data)) > 260
    assert not check_project_path(data, project_root=project)
    assert guard_project_write(data, project_root=project) == data.resolve()


def test_dump_json_rejects_before_creating_parent(tmp_path, monkeypatch):
    project = tmp_path / "project"
    monkeypatch.setattr(path_policy, "PROJECT_ROOT", project.resolve())
    rejected = _sized_path(project, 260)
    with pytest.raises(OneDrivePathError):
        dump_json(rejected, {"result": "must not be written"})
    assert not project.exists()
    dump_json(project / "short.json", {"result": "saved"})
    assert (project / "short.json").read_text(encoding="utf-8").startswith("{")


def test_scan_includes_ignored_environment_and_cache_directories(tmp_path):
    project = tmp_path / "project"
    for directory in (".venv", ".cache", "__pycache__", ".pytest_cache", "node_modules"):
        target = _sized_path(project / directory, 260)
        target.parent.mkdir(parents=True)
        target.write_text("preserved", encoding="utf-8")
    (project / ".gitignore").write_text("*\n", encoding="utf-8")
    report = scan_project_paths(project)
    assert not report.ok
    assert not report.errors
    assert len(report.violations) == 5
    assert {Path(row.path).relative_to(project).parts[0] for row in report.violations} == {
        ".venv", ".cache", "__pycache__", ".pytest_cache", "node_modules",
    }


def test_only_name_surrogate_reparse_points_are_links():
    directory = stat.S_IFDIR | 0o755
    # SYMLINK and MOUNT_POINT redirect traversal; CLOUD tags do not.
    assert path_policy._directory_link(SimpleNamespace(st_mode=directory, st_reparse_tag=0xA000000C))
    assert path_policy._directory_link(SimpleNamespace(st_mode=directory, st_reparse_tag=0xA0000003))
    assert not path_policy._directory_link(SimpleNamespace(st_mode=directory, st_reparse_tag=0x9000001A))
    assert not path_policy._directory_link(SimpleNamespace(st_mode=directory, st_reparse_tag=0x9000301A))


def test_scan_cloud_placeholder_directory_is_not_skipped(tmp_path, monkeypatch):
    project = tmp_path / "project"
    cloud = project / "cloud"
    target = _sized_path(cloud, 260)
    target.parent.mkdir(parents=True)
    target.write_text("source", encoding="utf-8")
    _mock_reparse_directory(monkeypatch, cloud, 0x9000001A)
    report = scan_project_paths(project)
    assert not report.skipped_links
    assert not report.errors
    assert [row.path for row in report.violations] == [str(target)]


@pytest.mark.parametrize("tag", [0xA000000C, 0xA0000003])
def test_scan_does_not_descend_into_symlink_or_junction_tags(tmp_path, monkeypatch, tag):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "unscanned.txt").write_text("outside", encoding="utf-8")
    link = project / "linked"
    link.mkdir()
    (link / "must_not_be_visited.txt").write_text("link contents", encoding="utf-8")
    _mock_reparse_directory(monkeypatch, link, tag)
    report = scan_project_paths(project)
    assert report.ok
    assert report.checked_paths == 2
    assert report.skipped_links == (str(link),)


def test_scan_access_errors_fail_instead_of_claiming_safe(tmp_path, monkeypatch):
    def denied(path):
        raise PermissionError("test denied")

    monkeypatch.setattr(path_policy.os, "scandir", denied)
    report = scan_project_paths(tmp_path)
    assert not report.ok
    assert len(report.errors) == 1


def test_cli_checks_planned_paths_without_writing(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("check_onedrive_paths", ROOT / "scripts/check_onedrive_paths.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    path = _sized_path(project, 260)
    assert module.main(["--root", str(project), "--candidate-only", "--candidate", str(path), "--json"]) == 1
    assert '"ok": false' in capsys.readouterr().out
    assert not project.exists()
    assert module.main(["--root", str(project), "--candidate-only", "--candidate", "short.json"]) == 0


def test_checkout_contains_no_unsafe_windows_paths():
    report = scan_project_paths(ROOT)
    assert report.ok, report.as_dict()
