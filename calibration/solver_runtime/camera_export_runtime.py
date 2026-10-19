"""Camera-only scheduling for the unchanged complete camera exporter.

The original export function still creates/checks the immutable NPZ. A small
state/block view changes only the local-camera accessor for that invocation.
No source actor, motion graph, numerical producer or registered file changes.
The optional first-call comparison uses the caller's already-built graph.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import numpy as np


class UnsupportedCameraExport(TypeError):
    """A consumer or camera method is outside the audited original contract."""


def _sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _function_sha256(function):
    return hashlib.sha256(inspect.getsource(function).encode("utf8")).hexdigest()


def _audit_export_consumer(original):
    """Refuse an exporter which consumes anything beyond the reviewed camera."""
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("state", "core", "path"):
        raise UnsupportedCameraExport("Original state/core/path camera exporter required")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "export_camera":
        raise UnsupportedCameraExport("Original export_camera definition required")
    locals_ = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "b" and node.func.attr == "local"]
    if len(locals_) != 1 or ast.dump(locals_[0]) != ast.dump(ast.parse("b.local(core)").body[0].value):
        raise UnsupportedCameraExport("One original b.local(core) access required")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "b" and node.attr not in ("local", "bundle"):
                raise UnsupportedCameraExport("Exporter now consumes a non-camera block field")
            if node.value.id == "state" and node.attr not in ("original", "blocks"):
                raise UnsupportedCameraExport("Exporter now consumes an unaudited state field")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "b"
                and node.value.attr == "bundle" and node.attr not in ("current_calibration", "stamps", "view")):
            raise UnsupportedCameraExport("Exporter now consumes a non-camera bundle field")
    return _function_sha256(original)


class _CameraBlockView:
    def __init__(self, block, camera_local_authority=None):
        from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
        authority = StructuredProgrammeBlock.camera_local if camera_local_authority is None else camera_local_authority
        method = getattr(block, "camera_local", None)
        if (not isinstance(block, StructuredProgrammeBlock)
                or getattr(method, "__func__", None) is not authority):
            raise UnsupportedCameraExport("Original StructuredProgrammeBlock.camera_local required")
        self._block = block

    def local(self, core):
        # This original method validates the complete world and applies the
        # original pivot/mount/lens chart; programme motion sync stays lazy.
        return self._block.camera_local(core)

    def __getattr__(self, name):
        return getattr(self._block, name)


class _CameraStateView:
    def __init__(self, state, camera_local_authority=None):
        self._state = state
        self.blocks = tuple(_CameraBlockView(block, camera_local_authority) for block in state.blocks)

    def __getattr__(self, name):
        return getattr(self._state, name)


def camera_only_export(original, state, core, path, *, camera_local_authority=None):
    """Call the original exporter with camera-only access on owned blocks."""
    return original(_CameraStateView(state, camera_local_authority), core, path)


def compare_camera_archives(original_path, candidate_path):
    """Check every NPZ field byte-for-byte, including all manifest JSON bytes."""
    original_path, candidate_path = Path(original_path), Path(candidate_path)
    fields = []
    with np.load(original_path, allow_pickle=False) as original, np.load(candidate_path, allow_pickle=False) as candidate:
        names = original.files
        inventory_equal = (len(names) == len(set(names)) and len(candidate.files) == len(set(candidate.files))
                           and set(names) == set(candidate.files))
        for name in names:
            actual = original[name]
            trial = candidate[name] if name in candidate.files else None
            equal = (trial is not None and actual.dtype == trial.dtype and actual.shape == trial.shape
                     and actual.tobytes(order="C") == trial.tobytes(order="C"))
            fields.append(dict(name=name, dtype=str(actual.dtype), shape=list(actual.shape),
                               original_sha256=_sha256_array(actual),
                               candidate_sha256=None if trial is None else _sha256_array(trial),
                               byte_exact=bool(equal)))
        metadata = json.loads(str(original["metadata_json"].item()))
        candidate_metadata = json.loads(str(candidate["metadata_json"].item()))
        metadata_equal = metadata == candidate_metadata
        pts = original["pts"]
        actual_exposures = int(len(np.unique(original["exposure_representative_pts"])))
        report = dict(schema="rinkspace.same-q-complete-camera-export-comparison.v1",
                      passed=bool(inventory_equal and all(field["byte_exact"] for field in fields) and metadata_equal),
                      field_inventory_exact=bool(inventory_equal), fields=fields,
                      all_metadata_fields_exact=bool(metadata_equal),
                      all_calibration_manifests_exact=bool(metadata["calibrations"] == candidate_metadata["calibrations"]),
                      source_sha256=metadata["source_sha256"], source_time_base=metadata["source_time_base"],
                      exposure_policy=metadata["exposure_policy"], native_presentations=int(len(pts)),
                      actual_canonical_exposures=actual_exposures,
                      source_pts_first=int(pts[0]), source_pts_last=int(pts[-1]),
                      installation_ids=sorted(metadata["calibrations"]),
                      original_camera_path=str(original_path.resolve()),
                      candidate_camera_path=str(candidate_path.resolve()), accuracy_accepted=False)
    return report


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def install_camera_export_runtime(driver, diagnostic_directory, *, verify_first_export=True):
    """Install before the next review; never build or copy a complete graph.

    With verification enabled, the first candidate export is isolated from
    small camera-binding side effects. The original then runs on the actual
    caller state/path, and all seven archive fields are compared. Only PASS
    enables later camera-only calls; a failure retains the original exporter.
    The first callback returns the original canonical artifact in both cases.
    """
    existing = getattr(driver, "_rinkspace_camera_export_runtime", None)
    if existing is not None:
        return existing
    original = driver.export_camera
    original_sha = _audit_export_consumer(original)
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock
    from .staged_source_guard import preserve_programme_camera_bindings

    destination = Path(diagnostic_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve()
    receipt_path = destination / "camera_export_runtime_receipt.json"
    receipt = dict(schema="rinkspace.camera-only-export-execution.v1",
                   status="same_q_real_export_comparison_pending" if verify_first_export else "enabled_without_new_comparison",
                   original_export_function_sha256=original_sha,
                   original_camera_local_function_sha256=_function_sha256(StructuredProgrammeBlock.camera_local),
                   original_optics_local_function_sha256=_function_sha256(SecondaryOpticsFixedXYZRinkBlock.local),
                   helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                   verification_first_export=bool(verify_first_export),
                   original_function_body_and_immutable_export_path_retained=True,
                   original_source_sha_time_base_aliases_views_contracts_and_calibrations_retained=True,
                   complete_motion_graph_not_constructed_or_copied=True,
                   registered_source_files_unchanged=True, accuracy_accepted=False,
                   camera_only_calls=0, original_fallback_calls=0)
    verification_pending = bool(verify_first_export)
    candidate_enabled = not verification_pending
    verified_state = None
    camera_local_authority = StructuredProgrammeBlock.camera_local

    @functools.wraps(original)
    def export_camera(state, core, path):
        nonlocal verification_pending, candidate_enabled, verified_state
        if verification_pending:
            verification_pending = False
            token = _sha256_array(core)
            evidence = dict(core_q_sha256=token, core_q_shape=list(np.shape(core)),
                            complete_q_identity_not_inferred_from_core=True,
                            same_caller_graph=True, order="candidate_then_original",
                            motion_sync_cache_not_warmed_by_candidate=True)
            candidate_path = None
            begin = time.perf_counter()
            try:
                # Restore only calibration and the three-entry world-validation
                # cache. No motion scenes, arrays, role bindings or profiles
                # are read/copied/synchronized by this small context.
                with preserve_programme_camera_bindings(state.blocks):
                    candidate_path = camera_only_export(original, state, core,
                        destination / "camera_only_same_q_PROBE.npz", camera_local_authority=camera_local_authority)
            except Exception as exc:
                evidence["candidate_error"] = dict(type=type(exc).__name__, message=str(exc))
            evidence["candidate_export_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            original_path = original(state, core, path)
            evidence["original_export_seconds"] = time.perf_counter() - begin
            receipt["original_fallback_calls"] += 1
            try:
                if candidate_path is None:
                    evidence["passed"] = False
                else:
                    evidence.update(compare_camera_archives(original_path, candidate_path))
                unchanged = _sha256_array(core) == token
                evidence["core_q_unchanged"] = unchanged
                evidence["passed"] = bool(evidence.get("passed", False) and unchanged)
            except Exception as exc:
                evidence.update(passed=False, comparison_error=dict(type=type(exc).__name__, message=str(exc)))
            candidate_enabled = evidence["passed"]
            verified_state = state if candidate_enabled else None
            receipt["status"] = "same_q_real_export_PASS" if candidate_enabled else "comparison_FAILED_original_export_retained"
            comparison_path = destination / "actual_same_q_camera_export_comparison.json"
            _write_json(comparison_path, evidence)
            receipt["comparison_path"] = str(comparison_path)
            receipt["comparison_core_q_sha256"] = token
            _write_json(receipt_path, receipt)
            return original_path
        if candidate_enabled and (not verify_first_export or state is verified_state):
            begin = time.perf_counter()
            try:
                result = camera_only_export(original, state, core, path, camera_local_authority=camera_local_authority)
            except UnsupportedCameraExport:
                # A later incompatible graph stays on the unchanged exporter.
                receipt["original_fallback_calls"] += 1
                return original(state, core, path)
            receipt["camera_only_calls"] += 1
            receipt["last_camera_only_export"] = dict(core_q_sha256=_sha256_array(core),
                elapsed_seconds=time.perf_counter() - begin)
            _write_json(receipt_path, receipt)
            return result
        receipt["original_fallback_calls"] += 1
        return original(state, core, path)

    driver.export_camera = export_camera
    driver._rinkspace_camera_export_original = original
    driver._rinkspace_camera_export_runtime = receipt
    _write_json(receipt_path, receipt)
    return receipt
