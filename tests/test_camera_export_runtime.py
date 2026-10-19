"""Exercise the exact exporter and full-field proof hook on a tiny camera graph."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install

install()
from calibration.solver_runtime import camera_export_runtime as candidate
from z190_physical_camera import CanonicalView, PhysicalCameraModel, SharedCalibration, ViewState
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock


def qtoken(q):
    return hashlib.sha256(np.ascontiguousarray(q).view(np.uint8)).hexdigest()


@pytest.fixture
def camera_graph(monkeypatch):
    # Compile only this existing function definition. Imports/constructors of
    # the complete driver and complete motion graph never execute here.
    source = Path(__file__).resolve().parents[1] / "calibration/solver_runtime/legacy/run_current_named66_full_programme.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "export_camera")
    namespace = dict(np=np, replace=replace, qtoken=qtoken)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    original = namespace["export_camera"]
    calibration = SharedCalibration("existing", (10., -20., 4.), lever_xyz=(.1, -.6, .3), k1=(.2,))
    reference = np.array([0., .1, -.2, .3, np.log(1920.), -.1, .2, -.3, np.log(1800.)])
    bundle = SimpleNamespace(calibration=calibration, stamps=(100, 120), index={100: 0, 120: 1})
    bundle.current_calibration = lambda local: bundle.calibration
    bundle.view = lambda local, p: ViewState(tuple(local[1+4*bundle.index[p]:4+4*bundle.index[p]]),
                                            float(local[4+4*bundle.index[p]] - np.log(1920.)))
    block = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    block.bundle = bundle
    block.motion_sync_calls = 0
    def camera_local(self, core):
        bundle.calibration = replace(calibration, pivot_xyz=(10. + core[0], -20., 4.))
        self._native_camera_valid_worlds = {core.tobytes(): True}
        return core
    monkeypatch.setattr(StructuredProgrammeBlock, "camera_local", camera_local)
    def local(core):
        block.motion_sync_calls += 1
        return block.camera_local(core)
    block.local = local
    views = {p: CanonicalView("existing", bundle.view(reference, p)) for p in bundle.stamps}
    state = SimpleNamespace(blocks=[block], original=SimpleNamespace(model=PhysicalCameraModel({"existing": calibration}),
        views=views, aliases={100: 100, 110: 100, 120: 120}, native=[100, 110, 120]),
        current_user_mount_constraint_contract={"source": "actual user mount contract"},
        current_user_pivot_outside_constraint_contract={"source": "actual user pivot contract"})
    return SimpleNamespace(state=state, core=reference, block=block, calibration=calibration,
                           driver=SimpleNamespace(export_camera=original), original=original)


def test_same_q_hook_compares_all_archive_fields_then_omits_only_motion_sync(camera_graph, tmp_path):
    graph = camera_graph
    receipt = candidate.install_camera_export_runtime(graph.driver, tmp_path / "proof")
    result = graph.driver.export_camera(graph.state, graph.core, tmp_path / "canonical.npz")
    assert result == (tmp_path / f"canonical_{qtoken(graph.core)[:16]}.npz")
    assert graph.block.motion_sync_calls == 1
    assert receipt["status"] == "same_q_real_export_PASS"
    report = json.loads(Path(receipt["comparison_path"]).read_text(encoding="utf8"))
    assert report["passed"] and report["core_q_unchanged"]
    assert report["all_metadata_fields_exact"] and report["all_calibration_manifests_exact"]
    assert {field["name"] for field in report["fields"]} == {
        "pts", "camera25", "exposure_representative_pts", "installation_id", "view_state4",
        "effective_pp_offset_px", "metadata_json"}
    assert all(field["byte_exact"] for field in report["fields"])
    assert report["native_presentations"] == 3 and report["actual_canonical_exposures"] == 2
    next_core = graph.core.copy()
    next_core[0] += .01
    actual = graph.driver.export_camera(graph.state, next_core, tmp_path / "next.npz")
    assert graph.block.motion_sync_calls == 1
    # Same original immutable existing-artifact check still runs on reuse.
    assert graph.driver.export_camera(graph.state, next_core, tmp_path / "next.npz") == actual
    baseline = graph.original(graph.state, next_core, tmp_path / "next_original.npz")
    assert candidate.compare_camera_archives(baseline, actual)["passed"]
    assert candidate.install_camera_export_runtime(graph.driver, tmp_path / "another") is receipt


def test_same_camera_arrays_with_changed_source_clock_fail_full_comparison(camera_graph, tmp_path):
    graph = camera_graph
    original = graph.original(graph.state, graph.core, tmp_path / "original.npz")
    altered = tmp_path / "altered.npz"
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["source_time_base"] = "1/15361"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(altered, **arrays)
    report = candidate.compare_camera_archives(original, altered)
    assert not report["passed"] and not report["all_metadata_fields_exact"]
    assert next(field for field in report["fields"] if field["name"] == "camera25")["byte_exact"]
    assert not next(field for field in report["fields"] if field["name"] == "metadata_json")["byte_exact"]


def test_unknown_camera_method_keeps_original_exporter(camera_graph, tmp_path):
    graph = camera_graph
    known = graph.block.camera_local
    graph.block.camera_local = lambda core: known(core)
    receipt = candidate.install_camera_export_runtime(graph.driver, tmp_path / "proof")
    result = graph.driver.export_camera(graph.state, graph.core, tmp_path / "original.npz")
    assert result.exists() and graph.block.motion_sync_calls == 1
    assert receipt["status"] == "comparison_FAILED_original_export_retained"
    assert "candidate_error" in json.loads(Path(receipt["comparison_path"]).read_text(encoding="utf8"))
    next_core = graph.core.copy()
    next_core[0] += .01
    graph.driver.export_camera(graph.state, next_core, tmp_path / "next.npz")
    assert graph.block.motion_sync_calls == 2


def test_incompatible_export_consumer_is_not_installed(tmp_path):
    def export_camera(state, core, path):
        for b in state.blocks:
            local = b.local(core)
            b.bundle.cache(local, False)
        return path
    driver = SimpleNamespace(export_camera=export_camera)
    with pytest.raises(candidate.UnsupportedCameraExport, match="non-camera bundle"):
        candidate.install_camera_export_runtime(driver, tmp_path)
    assert driver.export_camera is export_camera
