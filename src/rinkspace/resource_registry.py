"""Opt-in resource resolution for a relocated checkout; no runtime hooks.

This module never rewrites source packets, installs import hooks, loads native
libraries, creates directories, or changes environment variables. Existing
solver and AssetStore entry points do not call it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .storage import load_project_path_aliases


def reference_key(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    if "\x00" in text:
        raise ValueError("Resource reference contains NUL")
    if PureWindowsPath(text).drive:
        return str(PureWindowsPath(text)).replace("\\", "/").casefold()
    return str(PurePosixPath(text))


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def under(root: Path, relative: str) -> Path:
    relative = relative.replace("\\", "/")
    path = PurePosixPath(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError(f"Resource path is not bounded and relative: {relative}")
    result = (root / Path(*path.parts)).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Resource path escapes root: {relative}")
    return result


@dataclass(frozen=True)
class Resource:
    path: Path
    root_kind: str
    relative_path: str
    expected_sha256: str | None = None
    alias_registered: bool = False

    def template_reference(self) -> str:
        return f"@{self.root_kind}/{self.relative_path}"


class ResourceRegistry:
    """Resolve declared aliases into project/Data roots, without opening assets."""

    def __init__(self, project_root: str | Path, data_root: str | Path):
        self.project = Path(project_root).resolve()
        self.data = Path(data_root).resolve()
        if self.project == self.data or self.project.is_relative_to(self.data) or self.data.is_relative_to(self.project):
            raise ValueError("Project and Data must be separate directory trees")
        self.aliases: dict[str, Resource] = {}
        self.prefixes: list[tuple[str, str, str]] = []
        self.path_aliases: dict[Path, tuple[Path, str]] = {}
        self.privacy_derivatives: dict[Path, tuple[str, str]] = {}

    def add_alias(self, old_reference: str, *, root_kind: str, relative_path: str,
                  expected_sha256: str | None = None) -> None:
        resource = self._resource(root_kind, relative_path, expected_sha256, True)
        keys = {reference_key(old_reference), reference_key(resource.template_reference()), reference_key(resource.path)}
        for key in keys:
            previous = self.aliases.get(key)
            if previous is not None and previous.path != resource.path:
                raise ValueError(f"Conflicting resource alias: {old_reference}")
        same_target = {key: value for key, value in self.aliases.items() if value.path == resource.path}
        known_hashes = {value.expected_sha256 for value in same_target.values() if value.expected_sha256 is not None}
        if resource.expected_sha256 is not None:
            known_hashes.add(resource.expected_sha256)
        if len(known_hashes) > 1:
            raise ValueError(f"Conflicting resource alias identity: {old_reference}")
        effective_hash = next(iter(known_hashes), None)
        resource = Resource(resource.path, root_kind, resource.relative_path, effective_hash, True)
        # Every alias to one target shares the strongest registered identity.
        # Updating only the canonical key would leave older aliases unhashed.
        for key in keys | same_target.keys():
            self.aliases[key] = resource

    def add_prefix(self, old_root: str, *, root_kind: str, relative_path: str = "") -> None:
        target = self._resource(root_kind, relative_path)
        key = reference_key(old_root).rstrip("/")
        for old_key, old_kind, old_relative in self.prefixes:
            if old_key != key:
                continue
            previous = self._resource(old_kind, old_relative)
            if previous.path != target.path or old_kind != root_kind:
                raise ValueError(f"Conflicting resource prefix: {old_root}")
            return
        self.prefixes.append((key, root_kind, target.relative_path))
        self.prefixes.sort(key=lambda row: len(row[0]), reverse=True)

    def _resource(self, root_kind: str, relative_path: str, expected_sha256: str | None = None,
                  alias_registered: bool = False) -> Resource:
        if root_kind not in {"project", "data"}:
            raise ValueError(f"Unknown resource root: {root_kind}")
        if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Invalid expected SHA-256")
        root = self.project if root_kind == "project" else self.data
        target = under(root, relative_path)
        if root_kind == "project":
            relocation = self.path_aliases.get(target.relative_to(self.project))
            if relocation is not None:
                new_relative, preserved_hash = relocation
                current_hash, _ = self._project_identity(new_relative, preserved_hash)
                if expected_sha256 is not None and expected_sha256 not in {preserved_hash, current_hash}:
                    raise ValueError(f"Conflicting resource alias identity: {relative_path}")
                target = under(self.project, new_relative.as_posix())
                expected_sha256 = preserved_hash
                alias_registered = True
            expected_sha256, derivative = self._project_identity(target.relative_to(self.project), expected_sha256)
            alias_registered = alias_registered or derivative
        return Resource(target, root_kind, target.relative_to(root).as_posix(), expected_sha256, alias_registered)

    def _project_identity(self, relative: Path, expected: str | None) -> tuple[str | None, bool]:
        """Translate only a registered original identity to its public derivative."""
        derivative = self.privacy_derivatives.get(relative)
        if derivative is None:
            return expected, False
        original, current = derivative
        if expected is not None and expected not in {original, current}:
            raise ValueError(f"Conflicting privacy derivative identity: {relative.as_posix()}")
        return current, True

    def _load_privacy_derivatives(self) -> None:
        path = self.project / "configs/privacy_derivatives.json"
        if not path.is_file():
            return
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("schema") != "rinkspace.privacy-derivatives.v1" or not isinstance(document.get("files"), list):
            raise ValueError("Unsupported privacy derivatives configuration")
        for row in document["files"]:
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str)
                                                  for key in ("path", "original_sha256", "public_sha256")):
                raise ValueError("Privacy derivative requires path, original_sha256 and public_sha256")
            target = under(self.project, row["path"]).relative_to(self.project)
            if target == Path("."):
                raise ValueError("Privacy derivative must name a file")
            original, current = row["original_sha256"], row["public_sha256"]
            if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (original, current)):
                raise ValueError("Invalid privacy derivative SHA-256")
            if target in self.privacy_derivatives:
                raise ValueError(f"Duplicate privacy derivative path: {row['path']}")
            self.privacy_derivatives[target] = (original, current)

    def resolve(self, reference: str | Path, *, must_exist: bool = False, verify: bool = False) -> Resource:
        value = str(reference)
        key = reference_key(value)
        if key in self.aliases:
            result = self.aliases[key]
        elif value.startswith("@project/") or value.startswith("@data/"):
            kind, relative = value[1:].split("/", 1)
            result = self._resource(kind, relative)
        elif value.startswith("@"):
            raise ValueError(f"Unknown named resource root: {value}")
        elif not Path(value).is_absolute() and not PureWindowsPath(value).drive:
            result = self._resource("project", value)
        else:
            result = None
            for prefix, kind, relative in self.prefixes:
                if key == prefix or key.startswith(prefix + "/"):
                    # Keep the input spelling after a Windows case-insensitive match.
                    normalized = str(PureWindowsPath(value)).replace("\\", "/") if PureWindowsPath(value).drive else str(PurePosixPath(value))
                    tail = normalized[len(prefix):].lstrip("/")
                    result = self._resource(kind, "/".join(part for part in (relative, tail) if part))
                    break
            if result is None:
                candidate = Path(value).resolve()
                if candidate.is_relative_to(self.project) and not PureWindowsPath(value).drive:
                    result = self._resource("project", candidate.relative_to(self.project).as_posix())
                elif candidate.is_relative_to(self.data) and not PureWindowsPath(value).drive:
                    result = self._resource("data", candidate.relative_to(self.data).as_posix())
                else:
                    raise ValueError(f"Unregistered external resource: {value}")
        if must_exist and not result.path.exists():
            raise FileNotFoundError(result.path)
        if verify:
            if result.expected_sha256 is None:
                raise ValueError(f"No expected SHA-256 registered: {value}")
            if sha256(result.path) != result.expected_sha256:
                raise ValueError(f"Resource identity differs: {value}")
        return result

    @classmethod
    def from_project(cls, project_root: str | Path, data_root: str | Path) -> "ResourceRegistry":
        registry = cls(project_root, data_root)
        from .storage import load_storage_config
        config = load_storage_config(registry.project)
        old_project = config.get("legacy_root", str(registry.project))
        registry._load_privacy_derivatives()
        registry.path_aliases = load_project_path_aliases(registry.project)
        for old_relative, (new_relative, expected) in registry.path_aliases.items():
            old_path = old_relative.as_posix()
            references = {old_path, "@project/" + old_path, str(registry.project / old_relative),
                          old_project.rstrip("/\\") + "/" + old_path}
            for reference in references:
                registry.add_alias(reference, root_kind="project", relative_path=new_relative.as_posix(),
                                   expected_sha256=expected)
        for prefix, target in config.get("legacy_paths", {}).items():
            registry.add_prefix(old_project.rstrip("/\\") + "/" + prefix, root_kind="data", relative_path=target)
        for prefix, target in config.get("project_paths", {}).items():
            registry.add_prefix(old_project.rstrip("/\\") + "/" + prefix, root_kind="project", relative_path=target)
        registry.add_prefix(config.get("data_root", str(registry.data)), root_kind="data")
        registry.add_prefix(old_project, root_kind="project")
        registry.add_prefix(str(registry.project), root_kind="project")
        registry.add_prefix(str(registry.data), root_kind="data")
        for manifest_path in sorted((registry.project / "events").rglob("asset_preservation_manifest*.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for row in manifest["assets"]:
                registry.add_alias(row["source_path"], root_kind="project", relative_path=row["project_path"],
                                   expected_sha256=row["sha256"])
        source_manifests = []
        for directory in ("events", "calibration", "environments"):
            source_manifests.extend((registry.project / directory).rglob("source_preservation_manifest.json"))
        for manifest_path in sorted(source_manifests):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for row in manifest["records"]:
                for original in {row["original_path"], row.get("original_resolved_path", row["original_path"])}:
                    registry.add_alias(original, root_kind="project", relative_path=row["preserved_path"],
                                       expected_sha256=row["sha256"])
        provenance = sorted((registry.project / "scripts").glob("*.provenance.json"))
        provenance += sorted((registry.project / "calibration").rglob("*.provenance.json"))
        for manifest_path in provenance:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if manifest.get("schema") == "rinkspace.recovered-source-provenance.v1":
                registry.add_alias(manifest["original_path"], root_kind="project", relative_path=manifest["preserved_path"],
                                   expected_sha256=manifest["preserved_sha256"])
        for manifest_path in sorted((registry.project / "events").rglob("external_resource_preservation.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for row in manifest["assets"]:
                registry.add_alias(row["source_path"], root_kind="data", relative_path=row["data_relative_path"],
                                   expected_sha256=row["sha256"])
        return registry


__all__ = ["Resource", "ResourceRegistry", "reference_key", "sha256", "under"]
