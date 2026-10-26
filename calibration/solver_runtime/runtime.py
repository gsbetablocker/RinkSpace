"""Import only the reviewed runtime dependency closure from project source."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

from .config import PACKAGE_ROOT, PROJECT_ROOT, PROGRAM_SOURCE_ROOT, SOURCE_ROOT, data_root

_finder = None
MANIFEST_PATH = PACKAGE_ROOT / "source_manifest_public_20261026.json"


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf8"))


def source_path(module_name: str) -> Path:
    """Return the owned source file for an unchanged legacy pickle/import name."""
    return PACKAGE_ROOT / _manifest()["modules"][module_name]["path"]


class RuntimeSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, manifest):
        self.modules = manifest["modules"]
        self.namespaces = manifest["namespace_packages"]

    def find_spec(self, fullname, path=None, target=None):
        row = self.modules.get(fullname)
        if row is not None:
            candidate = PACKAGE_ROOT / row["path"]
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != row["sha256"]:
                raise ImportError(f"Solver source manifest differs: {fullname}: {candidate}")
            locations = [str(candidate.parent)] if candidate.name == "__init__.py" else None
            return importlib.util.spec_from_file_location(fullname, candidate, submodule_search_locations=locations)
        if fullname in self.namespaces:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(PACKAGE_ROOT / self.namespaces[fullname])]
            return spec
        return None


def install(*, resource_root: str | Path | None = None, verify: bool = True) -> Path:
    """Install once in the parent and each Windows spawned child.

    This does not prepare a graph, load observations, spawn workers, change the
    optimization target, or compile kernels. Legacy names remain stable for
    existing pickles. Call before any solver/candidate module import.
    """
    global _finder
    if resource_root is not None:
        requested = Path(resource_root).resolve()
        if _finder is not None and requested != data_root():
            raise RuntimeError("A running solver runtime cannot change its event resource root")
        os.environ["RINKSPACE_SOLVER_DATA_ROOT"] = str(requested)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if _finder is None:
        manifest = _manifest()
        for name, row in manifest["modules"].items():
            loaded = sys.modules.get(name)
            if loaded is None:
                continue
            actual = getattr(loaded, "__file__", None)
            expected = (PACKAGE_ROOT / row["path"]).resolve()
            if actual is not None and Path(actual).resolve() != expected:
                raise RuntimeError(f"Install solver_runtime before importing {name}: already loaded {actual}")
        if verify:
            status(verify_sources=True)
        _finder = RuntimeSourceFinder(manifest)
        sys.meta_path.insert(0, _finder)
        # Four byte-identical private event operators are owned too. Their
        # explicit legacy filename loaders bypass sys.meta_path, so redirect
        # only the registered four source aliases before graph/pickle loading.
        from .event_factories import install_finder, install_spec_hook
        install_finder()
        install_spec_hook()
        # Only project-owned code is a fallback; Data is a resource address,
        # never a general Python source-search directory.
        for root in reversed((PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "calibration/work",
                              PROJECT_ROOT / "calibration/shot_tracking/pipeline", PROJECT_ROOT / "calibration/program_tracking")):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
    return data_root()


def status(*, verify_sources: bool = True) -> dict:
    """Inspect reusable source identity without a solve or native compilation."""
    manifest = _manifest()
    invalid = []
    if verify_sources:
        for row in list(manifest["modules"].values()) + manifest["native_sources"]:
            path = PACKAGE_ROOT / row["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                invalid.append(str(path))
        if invalid:
            raise RuntimeError("Owned solver source identity changed: " + ", ".join(invalid))
    return dict(schema="rinkspace.reusable-solver-runtime.v1", package_root=str(PACKAGE_ROOT),
                data_root=str(data_root()), program_source_root=str(PROGRAM_SOURCE_ROOT),
                owned_module_count=len(manifest["modules"]), native_source_count=len(manifest["native_sources"]),
                source_manifest_sha256=hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
                solver_core="z190_joint_source_solver.solve_joint_source",
                event_adapter="run_current_named66_full_programme.main",
                source_identity_verified=verify_sources, runtime_modules_owned=True,
                runtime_version=manifest["runtime_version"], previous_manifest=manifest["previous_manifest"],
                legacy_event_source_manifest=str(PACKAGE_ROOT / "legacy_dependencies.json"),
                event_adapter_uses_explicit_external_private_factories=False,
                owned_event_factory_count=4,
                event_factory_manifest=str(PACKAGE_ROOT / "event_factories_public_20261026.json"),
                builds_during_install=False, optimization_or_accuracy_claim=False)


def verify_legacy_dependencies() -> dict:
    """Authenticate the owned private event operators before graph loading."""
    manifest_path = PACKAGE_ROOT / "legacy_dependencies.json"
    from .event_factories import verify_sources
    verified = verify_sources()
    return dict(manifest_path=str(manifest_path), verified_owned_event_sources=verified,
                verified_external_sources=[],
                inputs_or_source_formulas_changed=False)
