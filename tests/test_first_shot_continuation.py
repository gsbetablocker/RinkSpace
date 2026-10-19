"""Authenticate a complete numerical seed without changing the 8a baseline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.program_tracking import first_shot_fit as fit


@pytest.fixture
def seed_contract():
    # A short mock chart still carries every one of the 48 shared layout slots.
    baseline = np.r_[np.arange(12, dtype=np.float64) / 100, np.zeros(48)]
    baseline.setflags(write=False)
    guard_calls = []
    state = SimpleNamespace(
        count=len(baseline), initial_q=baseline,
        pts=np.array([256256, 256512], dtype=np.int64),
        parent_q=np.arange(7, dtype=np.float64),
        parent_shared_slots=np.array([0, 2, 4], dtype=np.int64),
        binding_reference={"path": "authenticated_source.json", "sha256": "source"},
        proof={"source_clock": {"time_base": "1/12800", "fps": "50/1"},
               "active_parameter_count": len(baseline),
               "shared_print_layout_advertisement_ids": [2, 3, 4, 5, 6, 7],
               "shared_print_layout_global_columns_by_ad": {2: list(range(12, 20))},
               "world_parameter_indices": [0, 1, 12, 13]},
        guard=lambda q: guard_calls.append(q.copy()),
    )
    bundle = SimpleNamespace(state=state,
        scales=np.array([.5, 1.25, 2., 3.5], dtype=np.float64),
        quadratic=np.array([False, False, False, False], dtype=bool))
    association = {"path": "authenticated_association.json", "sha256": "association"}
    producer_names = ["first_shot_fit.py", "first_shot_physical_state.py",
        "first_shot_source_kernel.py", "first_shot_print_layout.py",
        "first_shot_alignment_monitor.py", "render_result.py", "ad_landmarks.py"]
    identity = {"schema": "rinkspace.first-shot-native50-physical-source-graph.v1",
        "input": state.binding_reference, "association": association,
        "physical_state": json.loads(json.dumps(state.proof)),
        "producer": [fit.reference(ROOT / "calibration/program_tracking" / name)
                     for name in producer_names]}
    q = baseline.copy()
    q[:12] += .01
    q[12:] = np.linspace(-.03, .04, 48)
    arrays = dict(schema=np.array("rinkspace.joint-source-LM-checkpoint.v1"),
        identity_json=np.array(json.dumps(identity)), q=q, initial_q=baseline.copy(),
        quadratic=bundle.quadratic.copy(), scales=bundle.scales.copy(),
        damping=np.array(.0003), next_line_search_fraction=np.array(1.),
        aux__native_pts=state.pts.copy(), aux__parent_q=state.parent_q.copy(),
        aux__parent_shared_slots=state.parent_shared_slots.copy())
    return SimpleNamespace(bundle=bundle, association=association,
        arrays=arrays, guard_calls=guard_calls)


def write_seed(tmp_path, arrays):
    path = tmp_path / "seed.npz"
    np.savez_compressed(path, **arrays)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def edit_identity(arrays, change):
    identity = json.loads(str(arrays["identity_json"].item()))
    change(identity)
    arrays["identity_json"] = np.array(json.dumps(identity))


def test_complete_seed_preserves_all_layout_slots_and_original_baseline(seed_contract, tmp_path):
    c = seed_contract
    before = c.bundle.state.initial_q.tobytes()
    path, digest = write_seed(tmp_path, c.arrays)
    original_bytes = path.read_bytes()
    q, proof = fit.load_seed_state(c.bundle, path, digest, c.association)
    np.testing.assert_array_equal(q, c.arrays["q"])
    np.testing.assert_array_equal(q[-48:], c.arrays["q"][-48:])
    assert np.count_nonzero(q[-48:]) == 48
    assert c.bundle.state.initial_q.tobytes() == before
    assert not np.shares_memory(q, c.bundle.state.initial_q)
    assert path.read_bytes() == original_bytes
    assert len(c.guard_calls) == 1
    assert proof["initial_damping"] == .0003
    assert proof["initial_line_search_fraction"] == 1.
    assert proof["old_history_and_source_reviews_used_as_control"] is False
    assert proof["old_J_H_IRLS_replay_or_profile_reused"] is False
    assert proof["original_8a_initial_q_sha256"] == hashlib.sha256(before).hexdigest()


def test_continuation_of_seeded_branch_uses_explicit_8a_aux_baseline(seed_contract, tmp_path):
    c = seed_contract
    c.arrays["aux__original_8a_initial_q"] = c.bundle.state.initial_q.copy()
    c.arrays["initial_q"] = c.arrays["q"].copy()  # prior branch's numerical start
    path, digest = write_seed(tmp_path, c.arrays)
    q, _ = fit.load_seed_state(c.bundle, path, digest, c.association)
    np.testing.assert_array_equal(q, c.arrays["q"])
    np.testing.assert_array_equal(c.bundle.state.initial_q, c.arrays["aux__original_8a_initial_q"])


def test_archive_hash_is_pinned_before_loading(seed_contract, tmp_path):
    path, _ = write_seed(tmp_path, seed_contract.arrays)
    with pytest.raises(ValueError, match="SHA256"):
        fit.load_seed_state(seed_contract.bundle, path, "0" * 64, seed_contract.association)
    assert seed_contract.guard_calls == []


def test_new_objective_initializes_only_parameters_and_resets_rebuilt_layout(seed_contract, tmp_path):
    c = seed_contract
    c.bundle.state.material_revision_proof = {"reset_layout_advertisement_ids": [2]}
    c.bundle.state.print_layout_global_indices_by_ad = {2: np.arange(12, 20)}
    c.arrays["scales"][:] = 99.  # Previous factors must not enter this objective.
    del c.arrays["identity_json"]  # Legacy archives cannot establish the material chart.
    path, digest = write_seed(tmp_path, c.arrays)
    before = path.read_bytes()
    q, proof = fit.load_parameter_initializer(c.bundle, path, digest)
    np.testing.assert_array_equal(q[:12], c.arrays["q"][:12])
    np.testing.assert_array_equal(q[12:20], np.zeros(8))
    np.testing.assert_array_equal(q[20:], c.arrays["q"][20:])
    assert proof["old_objective_or_history_or_J_H_IRLS_reused"] is False
    assert proof["initial_damping"] == 1e-3
    assert path.read_bytes() == before


def test_parameter_initializer_rejects_foreign_native_clock(seed_contract, tmp_path):
    c = seed_contract
    c.arrays["aux__native_pts"][1] += 1
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="native clock changed"):
        fit.load_parameter_initializer(c.bundle, path, digest)


@pytest.mark.parametrize("field", ["input", "association", "clock", "proof", "layout_order"])
def test_changed_source_or_parameter_identity_is_rejected(seed_contract, tmp_path, field):
    c = seed_contract
    def change(identity):
        if field in ("input", "association"):
            identity[field]["sha256"] = "different"
        elif field == "clock":
            identity["physical_state"]["source_clock"]["fps"] = "60/1"
        elif field == "proof":
            identity["physical_state"]["world_parameter_indices"] = [1, 0, 12, 13]
        else:
            identity["physical_state"]["shared_print_layout_advertisement_ids"] = [3, 2, 4, 5, 6, 7]
    edit_identity(c.arrays, change)
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="source, clock, model or physical parameter chart"):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


@pytest.mark.parametrize("field,mode", [
    ("scales", "value"), ("scales", "dtype"), ("scales", "shape"),
    ("quadratic", "value"), ("quadratic", "dtype"),
    ("aux__native_pts", "value"), ("aux__native_pts", "dtype"),
    ("aux__parent_q", "value"), ("aux__parent_shared_slots", "value"),
    ("initial_q", "value"), ("initial_q", "dtype"),
    ("aux__original_8a_initial_q", "value"),
])
def test_source_arrays_and_8a_baseline_require_exact_dtype_and_values(seed_contract, tmp_path, field, mode):
    c = seed_contract
    if field == "aux__original_8a_initial_q":
        c.arrays[field] = c.bundle.state.initial_q.copy()
    a = c.arrays[field].copy()
    if mode == "dtype":
        a = a.astype(np.uint8 if a.dtype.kind == "b" else np.float32 if a.dtype.kind == "f" else np.int32)
    elif mode == "shape":
        a = a[:-1]
    elif a.dtype.kind == "b":
        a[0] = not bool(a[0])
    else:
        a[0] += 1
    c.arrays[field] = a
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="exact source/initialization array"):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


@pytest.mark.parametrize("field", ["aux__native_pts", "aux__parent_q", "aux__parent_shared_slots"])
def test_missing_auxiliary_contract_is_rejected(seed_contract, tmp_path, field):
    c = seed_contract
    del c.arrays[field]
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises((ValueError, KeyError)):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


@pytest.mark.parametrize("mode", ["dtype", "shape", "nan", "inf"])
def test_full_q_requires_all_finite_float64_coordinates(seed_contract, tmp_path, mode):
    c = seed_contract
    if mode == "dtype":
        c.arrays["q"] = c.arrays["q"].astype(np.float32)
    elif mode == "shape":
        c.arrays["q"] = c.arrays["q"][:-1]
    else:
        c.arrays["q"][-1] = np.nan if mode == "nan" else np.inf
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="all finite float64"):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


@pytest.mark.parametrize("field,value", [
    ("damping", 0.), ("damping", -1.), ("damping", np.nan), ("damping", np.inf),
    ("next_line_search_fraction", 0.), ("next_line_search_fraction", 1.01),
    ("next_line_search_fraction", np.nan),
])
def test_numerical_seed_must_be_valid(seed_contract, tmp_path, field, value):
    c = seed_contract
    c.arrays[field] = np.array(value)
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="damping or line-search"):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


@pytest.mark.parametrize("field", ["schema", "producer_missing", "critical_hash"])
def test_checkpoint_and_source_producers_are_authenticated(seed_contract, tmp_path, field):
    c = seed_contract
    if field == "schema":
        c.arrays["schema"] = np.array("unsupported-checkpoint")
    elif field == "producer_missing":
        edit_identity(c.arrays, lambda i: i["producer"].pop(3))
    else:
        edit_identity(c.arrays, lambda i: i["producer"][1].update(sha256="0" * 64))
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError):
        fit.load_seed_state(c.bundle, path, digest, c.association)
    assert c.guard_calls == []


def snapshot_seed(c, tmp_path, *, change_source=False):
    source = Path(fit.__file__).read_text(encoding="utf8")
    source = source.replace("def main(argv=None):\n", "def main(argv=None):\n    snapshot_main_only = True\n", 1)
    if change_source:
        source = source.replace("def write_json(path, value):\n", "def write_json(path, value):\n    unexpected_source_change = True\n", 1)
    snapshot = tmp_path / "old_consumer.py"
    snapshot.write_text(source, encoding="utf8")
    digest = fit.reference(snapshot)["sha256"]
    edit_identity(c.arrays, lambda i: i["producer"][0].update(sha256=digest))
    path, archive_digest = write_seed(tmp_path, c.arrays)
    return path, archive_digest, snapshot


def test_authenticated_snapshot_allows_only_main_and_seed_loader_changes(seed_contract, tmp_path):
    c = seed_contract
    path, digest, snapshot = snapshot_seed(c, tmp_path)
    q, proof = fit.load_seed_state(c.bundle, path, digest, c.association, snapshot)
    np.testing.assert_array_equal(q, c.arrays["q"])
    assert proof["authenticated_consumer_snapshot"] == fit.reference(snapshot)


@pytest.mark.parametrize("mode", ["missing", "wrong_hash", "source_AST"])
def test_consumer_snapshot_rejects_unauthenticated_or_changed_source_AST(seed_contract, tmp_path, mode):
    c = seed_contract
    path, digest, snapshot = snapshot_seed(c, tmp_path, change_source=mode == "source_AST")
    if mode == "missing":
        snapshot = None
    elif mode == "wrong_hash":
        snapshot.write_text(snapshot.read_text(encoding="utf8") + "\n# changed bytes\n", encoding="utf8")
    with pytest.raises(ValueError, match="snapshot|consumer changed"):
        fit.load_seed_state(c.bundle, path, digest, c.association, snapshot)
    assert c.guard_calls == []


@pytest.mark.parametrize("mode", ["source_directory", "checkpoint_directory", "old_artifacts_directory"])
def test_cli_seed_preserves_existing_output_directories(tmp_path, monkeypatch, mode):
    from calibration import solver_runtime
    monkeypatch.setattr(solver_runtime, "install", lambda: None)
    monkeypatch.setattr(solver_runtime, "configure_native_runtime", lambda **kw: {})
    def forbidden(*args, **kwargs):
        pytest.fail("Existing branch must be rejected before source construction or optimization")
    monkeypatch.setattr(fit, "FirstShotSourceBundle", forbidden)
    monkeypatch.setattr(solver_runtime, "solve_joint_source", forbidden)
    source = tmp_path / "old_branch"
    source.mkdir()
    seed = source / "latest_accepted_state.npz"
    seed.write_bytes(b"original archive")
    output = source if mode == "source_directory" else tmp_path / "other_old_branch"
    output.mkdir(exist_ok=True)
    name = "latest_accepted_state.npz" if mode == "checkpoint_directory" else "current.json"
    marker = output / name
    marker.write_bytes(b"original archived output")
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(SystemExit) as failure:
        fit.main(["--binding", "unused", "--association", "unused", "--output", str(output),
                  "--seed-state", str(seed), "--seed-state-sha256", "0" * 64])
    assert failure.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("extra", [[], ["--seed-state-sha256", "0" * 64, "--resume"]])
def test_cli_requires_pinned_seed_and_disallows_resume(tmp_path, extra):
    with pytest.raises(SystemExit) as failure:
        fit.main(["--binding", "unused", "--association", "unused", "--output", str(tmp_path),
                  "--seed-state", str(tmp_path / "seed.npz"), *extra])
    assert failure.value.code == 2


@pytest.fixture
def material_initializer(seed_contract, tmp_path):
    c = seed_contract
    def document(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf8")
        return fit.reference(path)
    chart = {name: document(name + ".json", {"chart": name})
        for name in ("model", "material_template_revision", "print_layout")}
    chart.update(source={"sha256": "same-native-video"},
        source_clock={"fps": "50/1", "time_base": "1/12800"})
    previous = deepcopy(chart)
    previous["source_localization_noise_revision"] = {"sha256": "previous-noise"}
    current = deepcopy(chart)
    current["source_localization_noise_revision"] = {"sha256": "new-noise"}
    c.previous_binding = document("previous_binding.json", previous)
    c.bundle.state.binding_reference = document("current_binding.json", current)
    c.bundle.state.material_revision_proof = {"reset_layout_advertisement_ids": [2, 5]}
    c.bundle.state.print_layout_global_indices_by_ad = {
        n: np.arange(12 + 8*i, 20 + 8*i) for i, n in enumerate([2, 3, 4, 5, 6, 7])}
    c.current_binding = current
    c.document = document
    return c


@pytest.mark.parametrize("schema,key", [
    ("rinkspace.first-shot-native50-physical-source-graph.v1", "input"),
    ("rinkspace.same-q-partial-noise-one-public-step.v1", "binding"),
])
def test_parameter_continuation_same_chart_noise_change_preserves_all_layouts(
        material_initializer, tmp_path, schema, key):
    c = material_initializer
    c.arrays["identity_json"] = np.array(json.dumps({"schema": schema, key: c.previous_binding}))
    c.arrays["scales"][:] = 99.
    path, digest = write_seed(tmp_path, c.arrays)
    before = path.read_bytes()
    q, proof = fit.load_parameter_initializer(c.bundle, path, digest)
    assert q.tobytes() == c.arrays["q"].tobytes()
    assert np.count_nonzero(q[-48:]) == 48
    assert proof["reset_source_rebuilt_print_layouts"] == []
    assert proof["material_chart_continuation"]["compatible_material_chart"] is True
    assert proof["initial_damping"] == 1e-3
    assert proof["old_objective_or_history_or_J_H_IRLS_reused"] is False
    assert c.bundle.scales.tolist() == [.5, 1.25, 2., 3.5]
    assert len(c.guard_calls) == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["model", "material_template_revision", "print_layout",
    "source", "source_clock"])
def test_parameter_continuation_changed_material_chart_resets_only_rebuilt_layouts(
        material_initializer, tmp_path, field):
    c = material_initializer
    current = deepcopy(c.current_binding)
    current[field] = (c.document("changed_" + field + ".json", {"changed": True})
        if field not in ("source", "source_clock") else {"changed": True})
    c.bundle.state.binding_reference = c.document("changed_binding.json", current)
    c.arrays["identity_json"] = np.array(json.dumps({
        "schema": "rinkspace.first-shot-native50-physical-source-graph.v1",
        "input": c.previous_binding}))
    path, digest = write_seed(tmp_path, c.arrays)
    q, proof = fit.load_parameter_initializer(c.bundle, path, digest)
    expected = c.arrays["q"].copy()
    expected[np.r_[12:20, 36:44]] = 0.
    assert q.tobytes() == expected.tobytes()
    assert proof["reset_source_rebuilt_print_layouts"] == [2, 5]
    assert proof["material_chart_continuation"]["compatible_material_chart"] is False
    assert len(c.guard_calls) == 1


@pytest.mark.parametrize("mode", ["missing_identity", "unknown_schema", "missing_binding"])
def test_parameter_continuation_without_authenticated_chart_is_conservative(
        material_initializer, tmp_path, mode):
    c = material_initializer
    if mode == "missing_identity":
        del c.arrays["identity_json"]
    else:
        c.arrays["identity_json"] = np.array(json.dumps({
            "schema": "unknown" if mode == "unknown_schema" else
                "rinkspace.first-shot-native50-physical-source-graph.v1",
            "binding": c.previous_binding}))
    path, digest = write_seed(tmp_path, c.arrays)
    q, proof = fit.load_parameter_initializer(c.bundle, path, digest)
    assert proof["reset_source_rebuilt_print_layouts"] == [2, 5]
    assert np.count_nonzero(q[-48:]) == 32


@pytest.mark.parametrize("which", ["previous", "current"])
def test_parameter_continuation_rejects_forged_binding_hash(
        material_initializer, tmp_path, which):
    c = material_initializer
    previous = deepcopy(c.previous_binding)
    if which == "previous":
        previous["sha256"] = "0" * 64
    else:
        c.bundle.state.binding_reference = dict(c.bundle.state.binding_reference, sha256="0"*64)
    c.arrays["identity_json"] = np.array(json.dumps({
        "schema": "rinkspace.first-shot-native50-physical-source-graph.v1", "input": previous}))
    path, digest = write_seed(tmp_path, c.arrays)
    with pytest.raises(ValueError, match="Source reference changed"):
        fit.load_parameter_initializer(c.bundle, path, digest)
    assert c.guard_calls == []


def test_parameter_initializer_without_material_revision_does_not_require_identity(seed_contract, tmp_path):
    c = seed_contract
    c.arrays["identity_json"] = np.array("legacy diagnostic label, not a material identity")
    path, digest = write_seed(tmp_path, c.arrays)
    q, proof = fit.load_parameter_initializer(c.bundle, path, digest)
    assert q.tobytes() == c.arrays["q"].tobytes()
    assert proof["material_chart_continuation"] is None
