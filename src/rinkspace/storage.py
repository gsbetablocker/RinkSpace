"""Registered asset locations; the software root remains separate from data.

Legacy source manifests and model documents stay byte-identical. Their paths
are resolved here, while new runs and disposable caches use the data root.
"""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import re
from urllib.parse import quote

from .path_policy import guard_project_write


class AssetStore:
    def __init__(self, root: str | Path):
        self.project = Path(root).resolve()
        config_path = self.project / "configs/storage.json"
        self.configured = config_path.is_file()
        config = json.loads(config_path.read_text(encoding="utf-8-sig")) if self.configured else {}
        if self.configured and config.get("schema_version") != "rinkspace.storage.v1":
            raise ValueError("Unsupported storage configuration")
        self.data = Path(config["data_root"]).resolve() if self.configured else self.project
        if self.configured and not Path(config["data_root"]).is_absolute():
            raise ValueError("Storage data_root must be absolute")
        self.event = self._under(self.data, config.get("event_root", "."))
        self.legacy_root = Path(config.get("legacy_root", self.project)).resolve()
        self.path_aliases = load_project_path_aliases(self.project)
        self.project_paths = []
        for prefix, target in config.get("project_paths", {}).items():
            self.project_paths.append((self._relative(prefix), self._under(self.project, target)))
        self.project_paths.sort(key=lambda row: len(row[0].parts), reverse=True)
        self.legacy_paths = []
        for prefix, target in config.get("legacy_paths", {}).items():
            prefix_path = self._relative(prefix)
            self.legacy_paths.append((prefix_path, self._under(self.data, target)))
        self.legacy_paths.sort(key=lambda row: len(row[0].parts), reverse=True)
        self.fallback_paths = []
        for prefix, target in config.get("fallback_paths", {}).items():
            self.fallback_paths.append((self._relative(prefix), self._under(self.data, target)))
        self.fallback_paths.sort(key=lambda row: len(row[0].parts), reverse=True)
        self.public_project_paths = tuple(
            self._under(self.project, relative)
            for relative in config.get("public_project_paths", ("outputs", "evidence"))
        )

    @staticmethod
    def _relative(value: str | Path) -> Path:
        text = str(value).replace("\\", "/")
        path = Path(text)
        if (path.is_absolute() or PureWindowsPath(text).drive
                or ".." in path.parts or "\x00" in text):
            raise ValueError("Asset path must be a bounded relative path")
        return path

    @classmethod
    def _under(cls, base: Path, relative: str | Path) -> Path:
        path = (base / cls._relative(relative)).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Asset path escapes its registered root")
        return path

    def resolve(self, value: str | Path, *, base: str | Path | None = None,
                must_exist: bool = False) -> Path:
        """Resolve project/registered data assets without altering their records."""
        value = Path(str(value).replace("\\", "/"))
        path = (value if value.is_absolute() else Path(base or self.project) / value).resolve()
        path = self._project_path_alias(path)
        for project in (self.project, self.legacy_root):
            if not path.is_relative_to(project):
                continue
            relative = path.relative_to(project)
            for prefix, target in self.project_paths:
                if relative.is_relative_to(prefix):
                    alias = (target / relative.relative_to(prefix)).resolve()
                    if not alias.is_relative_to(self.project):
                        raise ValueError("Project alias escapes the code root")
                    if alias.exists() or not path.exists():
                        path = alias
                    break
            break
        path = self._project_path_alias(path)
        mapped = None
        for project in (self.project, self.legacy_root):
            if not path.is_relative_to(project):
                continue
            relative = path.relative_to(project)
            for prefix, target in self.legacy_paths:
                if relative.is_relative_to(prefix):
                    mapped = (target / relative.relative_to(prefix)).resolve()
                    if not mapped.is_relative_to(self.data):
                        raise ValueError("Mapped asset escapes the registered data root")
                    break
            if mapped is not None:
                break
        # Prefer the canonical copy after migration, while retained software
        # assets (such as .cache/bin) continue to work during staged migration.
        if mapped is not None and (mapped.exists() or not path.exists()):
            path = mapped
        if not path.exists():
            for project in (self.project, self.legacy_root):
                if not path.is_relative_to(project):
                    continue
                relative = path.relative_to(project)
                for prefix, target in self.fallback_paths:
                    if relative.is_relative_to(prefix):
                        fallback = (target / relative.relative_to(prefix)).resolve()
                        if not fallback.is_relative_to(self.data):
                            raise ValueError("Fallback asset escapes the registered data root")
                        if fallback.suffix.lower() == ".md" and not fallback.exists():
                            text = fallback.with_suffix(".txt")
                            if text.is_file():
                                fallback = text
                        path = fallback
                        break
                if path.is_relative_to(self.data):
                    break
        if not any(path.is_relative_to(root) for root in (self.project, self.data)):
            raise ValueError("Asset path outside registered project/data roots")
        if not path.exists():
            from .packed_assets import materialize_project_asset
            path = materialize_project_asset(path, project_root=self.project)
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def _project_path_alias(self, path: Path) -> Path:
        """Follow exact relocations while keeping immutable old references usable."""
        for project in (self.project, self.legacy_root):
            if path.is_relative_to(project):
                relative = path.relative_to(project)
                alias = self.path_aliases.get(relative)
                if alias is not None:
                    return self._under(self.project, alias[0])
                break
        return path

    def source(self, value: str | Path) -> Path:
        """Resolve a source already registered by the local source configuration."""
        value = Path(value).expanduser()
        try:
            return self.resolve(value)
        except ValueError:
            # The existing source-registration API also permits an explicitly
            # registered original at another location, without copying it.
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise
            return path.resolve()

    def output(self, relative: str | Path) -> Path:
        """New output location; never silently falls back into the code checkout."""
        return guard_project_write(
            self._under(self.event if self.configured else self.project / "outputs", relative),
            project_root=self.project,
        )

    def directory(self, legacy: str | Path, new: str | Path) -> Path:
        """Reuse an existing migrated artifact directory, otherwise use new data."""
        if not self.configured:
            return self.resolve(legacy)
        current = self.output(new)
        previous = self.resolve(legacy)
        if current.exists() or not previous.exists() or not previous.is_relative_to(self.data):
            return current
        return previous

    def _public(self, path: Path) -> bool:
        # Resolving here also rejects a symlink from an allowed subtree to an
        # unrelated file. The unconfigured mode retains the old bounded roots.
        path = path.resolve()
        if self.configured and path.is_relative_to(self.data):
            return True
        return any(path.is_relative_to(root) for root in self.public_project_paths)

    def asset_url(self, value: str | Path) -> str:
        path = self.resolve(value)
        if not self._public(path):
            raise ValueError("Asset is outside registered public asset roots")
        if self.configured and path.is_relative_to(self.data):
            relative = "data/" + path.relative_to(self.data).as_posix()
        else:
            relative = ("project/" if self.configured else "") + path.relative_to(self.project).as_posix()
        return "/assets/" + quote(relative, safe="/")

    def public_asset(self, relative: str) -> Path:
        """Resolve a Web asset route, including old /assets/outputs/... links."""
        path = self._relative(relative)
        if not path.parts:
            raise ValueError("Missing asset path")
        if path.parts[0] == "data":
            if not self.configured:
                raise ValueError("No registered data root")
            resolved = self._under(self.data, Path(*path.parts[1:]))
        else:
            if path.parts[0] == "project":
                path = Path(*path.parts[1:])
            resolved = self.resolve(path)
        if not self._public(resolved):
            raise ValueError("Asset is outside registered public asset roots")
        return resolved


def resolve_asset(root: str | Path, value: str | Path, **options) -> Path:
    return AssetStore(root).resolve(value, **options)


def load_project_path_aliases(project: Path) -> dict[Path, tuple[Path, str]]:
    """Read bounded, hash-preserving file relocations; absent config is opt-out."""
    config_path = project / "configs/path_aliases.json"
    if not config_path.is_file():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("schema") != "rinkspace.project-path-aliases.v1" or not isinstance(config.get("aliases"), list):
        raise ValueError("Unsupported project path aliases configuration")
    aliases: dict[Path, tuple[Path, str]] = {}
    identities: dict[Path, str] = {}
    for row in config["aliases"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("old_path", "new_path", "sha256")):
            raise ValueError("Project path alias requires old_path, new_path and SHA-256")
        expected = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid project path alias SHA-256")
        old = AssetStore._under(project, row["old_path"]).relative_to(project)
        new = AssetStore._under(project, row["new_path"]).relative_to(project)
        if old == Path(".") or new == Path("."):
            raise ValueError("Project path alias must name a file")
        target = (new, expected)
        if old in aliases and aliases[old] != target:
            raise ValueError(f"Conflicting project path alias: {row['old_path']}")
        if new in identities and identities[new] != expected:
            raise ValueError(f"Conflicting project path alias identity: {row['new_path']}")
        aliases[old] = target
        identities[new] = expected
    # File relocations are a single hop; disallow ambiguous cycles/chains.
    if any(new in aliases and new != old for old, (new, _) in aliases.items()):
        raise ValueError("Project path aliases cannot chain")
    return aliases
