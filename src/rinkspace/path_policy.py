"""Check the physical OneDrive checkout before creating Windows paths.

The local Windows MAX_PATH limit includes its terminating NUL. This module
therefore rejects 260 or more UTF-16 code units in the absolute checkout path,
and components over 255 units. Registered data outside the checkout retains
its existing long-path support. No files are renamed or removed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import stat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_MAX_PATH = 260
WINDOWS_MAX_COMPONENT = 255
_NAME_SURROGATE = 0x20000000


def windows_path_units(value: str) -> int:
    """Windows counts UTF-16 code units, including both halves of an emoji."""
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _normal_windows_path(path: Path) -> str:
    value = str(path)
    # The extended prefix changes API access, not the path OneDrive syncs.
    if value.upper().startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _physical_path(path: str | Path) -> Path:
    # An extended-prefix spelling must not evade the checkout containment
    # check. resolve() alone retains that prefix on Windows.
    return Path(_normal_windows_path(Path(path).expanduser().resolve()))


@dataclass(frozen=True)
class PathViolation:
    path: str
    reason: str
    actual_units: int
    allowed_units: int
    component: str | None = None


@dataclass(frozen=True)
class PathScan:
    root: str
    checked_paths: int
    violations: tuple[PathViolation, ...]
    skipped_links: tuple[str, ...]
    errors: tuple[dict[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.violations and not self.errors

    def as_dict(self) -> dict:
        return {
            "root": self.root, "checked_paths": self.checked_paths,
            "violations": [asdict(row) for row in self.violations],
            "skipped_links": list(self.skipped_links), "errors": list(self.errors),
            "ok": self.ok,
        }


class OneDrivePathError(ValueError):
    """A write would create an unsafe path in the synchronized checkout."""


def _violations(path: Path) -> list[PathViolation]:
    absolute = _normal_windows_path(path)
    result = []
    units = windows_path_units(absolute)
    if units >= WINDOWS_MAX_PATH:
        result.append(PathViolation(absolute, "absolute_path", units, WINDOWS_MAX_PATH - 1))
    for component in path.parts:
        if component == path.anchor:
            continue
        units = windows_path_units(component)
        if units > WINDOWS_MAX_COMPONENT:
            result.append(PathViolation(absolute, "component", units, WINDOWS_MAX_COMPONENT, component))
    return result


def check_project_path(path: str | Path, *, project_root: str | Path | None = None) -> tuple[PathViolation, ...]:
    """Check a resolved physical path, including a path that does not exist."""
    root = _physical_path(project_root) if project_root is not None else PROJECT_ROOT
    physical = _physical_path(path)
    if not physical.is_relative_to(root):
        return ()
    return tuple(_violations(physical))


def guard_project_write(path: str | Path, *, project_root: str | Path | None = None) -> Path:
    """Validate before mkdir/write; return the physical destination."""
    physical = _physical_path(path)
    root = _physical_path(project_root) if project_root is not None else PROJECT_ROOT
    issues = _violations(physical) if physical.is_relative_to(root) else ()
    if issues:
        details = "; ".join(
            f"{row.reason}: {row.actual_units} UTF-16 units > {row.allowed_units}"
            for row in issues
        )
        raise OneDrivePathError(
            f"OneDrive 项目路径超过 Windows 限制，未创建文件：{physical} ({details})。"
            "请缩短项目内目录/文件名；运行数据使用已登记的 RinkSpace_Data 输出位置。"
        )
    return physical


def _directory_link(metadata: os.stat_result) -> bool:
    # Only name-surrogate reparse tags redirect path traversal. OneDrive cloud
    # placeholders also carry REPARSE_POINT, but are ordinary synced content.
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_reparse_tag", 0) & _NAME_SURROGATE
    )


def scan_project_paths(project_root: str | Path | None = None) -> PathScan:
    """Scan everything, including ignored files, without following links.

    Reads directory metadata only; cloud file content is not hydrated. Access
    errors fail the check instead of silently treating incomplete scans as safe.
    """
    root = _physical_path(project_root or PROJECT_ROOT)
    pending = [root]
    checked = 1
    violations = _violations(root)
    skipped_links = []
    errors = []
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    checked += 1
                    violations.extend(_violations(path))
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        if _directory_link(metadata):
                            skipped_links.append(str(path))
                        elif stat.S_ISDIR(metadata.st_mode):
                            pending.append(path)
                    except OSError as error:
                        errors.append({"path": str(path), "error": str(error)})
        except OSError as error:
            errors.append({"path": str(directory), "error": str(error)})
    return PathScan(
        str(root), checked,
        tuple(sorted(violations, key=lambda row: (row.path, row.reason))),
        tuple(sorted(skipped_links)),
        tuple(sorted(errors, key=lambda row: row["path"])),
    )
