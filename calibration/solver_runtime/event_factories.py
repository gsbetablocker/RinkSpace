"""Load the four authenticated current event operators from owned project source.

This is a separate, closed source registry so an existing runtime manifest can
be inspected while a solve is running. It never searches Data for Python code.
An explicitly supplied dynamic source filename can be authenticated against
the four original source digests, but only owned bytes are ever executed.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import sys

from .config import PACKAGE_ROOT

MANIFEST_PATH = PACKAGE_ROOT / "event_factories_public_20261026.json"
_original_spec_from_file_location = importlib.util.spec_from_file_location
_spec_hook_installed = False
_locations_unspecified = object()


@lru_cache(maxsize=1)
def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf8"))


def _path_key(path):
    # Resolve existing historical junctions, retaining the literal spelling in
    # the registry too so removal of those junctions cannot remove ownership.
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def _aliases(row):
    return {str(Path(row[key])).replace("\\", "/").casefold()
            for key in ("original_path", "original_resolved_path")} | {
        _path_key(row["original_path"]), _path_key(row["original_resolved_path"]),
        _path_key(PACKAGE_ROOT / row["path"])}


@lru_cache(maxsize=1)
def _source_alias_index():
    return {alias: row for row in _manifest()["modules"].values()
            for alias in _aliases(row)}


def source_record(module_name=None, source_file=None, *, authenticate_dynamic=False):
    """Return a declared record; never infer an operator from a filename alone.

    For explicit generated/pickle filenames only, authenticate_dynamic permits
    a byte digest match against this fixed four-source allowlist. It is not a
    module import fallback, and a differing source has no owned substitute.
    """
    modules = _manifest()["modules"]
    if source_file is None:
        return modules.get(module_name)
    literal = str(Path(source_file)).replace("\\", "/").casefold()
    aliases = _source_alias_index()
    row = aliases.get(literal)
    if row is not None:
        return row
    # Nonmatching basenames cannot be one of the four literal source aliases.
    # Avoid filesystem resolutions for every unrelated ordinary import.
    if Path(source_file).name in {Path(row["path"]).name for row in modules.values()}:
        row = aliases.get(_path_key(source_file))
        if row is not None:
            return row
    if authenticate_dynamic:
        candidate = Path(source_file)
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            for row in modules.values():
                if digest == row["original_sha256"]:
                    return row
    return None


def owned_source_path(module_name=None, source_file=None, *, authenticate_dynamic=False):
    row = source_record(module_name, source_file, authenticate_dynamic=authenticate_dynamic)
    if row is None:
        raise ImportError(f"Undeclared event source: {module_name}: {source_file}")
    path = PACKAGE_ROOT / row["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ImportError(f"Owned event source identity changed: {path}")
    return path


def _resource_bindings(module, row):
    """Retain historical data semantics without changing source or pickle names."""
    resource = Path(row["resource_directory"])
    filename = Path(row["path"]).name
    if filename == "main9_full_source_factory.py":
        module.HERE = resource
    elif filename == "run_main9_typed_core_source_rf.py":
        module.HERE = resource
        module.N = resource.parent
        module.SCENE = module.N / "three_installation_shared_F_column_keys_v2_guarded/shared_Fx_fixed_height/candidate_scene_concentric_DIAGNOSTIC_ONLY.json"
        module.PACKETS = [resource / "main9_ice_source/main9_R3_R4_external_boundary_packet_qualified.json",
                          resource / "main9_ice_source/main9_C_C0_paint_support_packet_qualified.json",
                          module.N / "main9_board_ice_key_source_v1/core_key_visible_board_edges_source.json"]
    elif filename == "secondary_full_source_factory.py":
        module.B = resource
    if hasattr(module, "ref"):
        original_ref = module.ref

        def owned_ref(path):
            source = source_record(source_file=path)
            return original_ref(owned_source_path(source_file=path) if source is not None else path)

        # FD manifests authenticate the unchanged typed adapter digest. They
        # must read owned source even though the neighboring JSON stays Data.
        module.ref = owned_ref
    module.__rinkspace_original_source__ = row["original_path"]
    module.__rinkspace_original_source_sha256__ = row["original_sha256"]
    module.__rinkspace_resource_directory__ = str(resource)


class EventSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, row):
        super().__init__(fullname, str(path))
        self.row = row

    def exec_module(self, module):
        # get_code can consult .pyc; compile authenticated source bytes directly.
        path = Path(self.path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.row["sha256"]:
            raise ImportError(f"Owned event source identity changed: {path}")
        exec(compile(raw, str(path), "exec"), module.__dict__)
        _resource_bindings(module, self.row)


def source_spec(fullname, source_file=None, *, authenticate_dynamic=False):
    """Build an exact owned spec with the caller's unchanged module identity."""
    row = source_record(fullname, source_file, authenticate_dynamic=authenticate_dynamic)
    if row is None:
        return None
    path = owned_source_path(fullname, source_file, authenticate_dynamic=authenticate_dynamic)
    return _original_spec_from_file_location(fullname, path,
        loader=EventSourceLoader(fullname, path, row))


def load_source(source_file, module_name=None, *, authenticate_dynamic=False):
    """Replace a declared module_at/spec_from_file_location operation."""
    name = module_name or Path(source_file).stem
    spec = source_spec(name, source_file, authenticate_dynamic=authenticate_dynamic)
    if spec is None:
        raise ImportError(f"Undeclared event source: {name}: {source_file}")
    existing = sys.modules.get(name)
    if existing is not None:
        if _path_key(getattr(existing, "__file__", "")) != _path_key(spec.origin):
            raise RuntimeError(f"Event source already loaded from another file: {name}")
        return existing
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class EventSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        return source_spec(fullname)


def install_finder():
    """Install only these exact event names; runtime.install can call this."""
    for finder in sys.meta_path:
        if isinstance(finder, EventSourceFinder):
            return finder
    finder = EventSourceFinder()
    sys.meta_path.insert(0, finder)
    return finder


def install_spec_hook():
    """Redirect only four declared source aliases in explicit legacy loaders.

    These loaders bypass sys.meta_path. Preserve their supplied module names
    and forward every other filename and option to the original API unchanged.
    """
    global _spec_hook_installed
    if _spec_hook_installed:
        return

    def exact_event_spec(name, location=None, *, loader=None, submodule_search_locations=_locations_unspecified):
        if location is not None and loader is None and source_record(source_file=location) is not None:
            return source_spec(name, location)
        if submodule_search_locations is _locations_unspecified:
            return _original_spec_from_file_location(name, location, loader=loader)
        return _original_spec_from_file_location(name, location, loader=loader,
            submodule_search_locations=submodule_search_locations)

    importlib.util.spec_from_file_location = exact_event_spec
    _spec_hook_installed = True


def verify_sources():
    """Authenticate owned event bytes, with no reads of historical Python."""
    return {name: dict(path=str(owned_source_path(name)), sha256=row["sha256"],
                      original_sha256=row["original_sha256"])
            for name, row in _manifest()["modules"].items()}
