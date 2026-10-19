"""Frozen accepted evidence reaches the physical state and public source solver."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import pickle
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
from calibration.program_tracking import complete_official_fit_dataset as complete
from calibration.program_tracking import first_shot_fit as runtime
from calibration.program_tracking import framewise_model as model
from calibration.program_tracking import official_static_surface_dataset as source
from calibration.program_tracking.continuous_board_association import with_continuous_board_association
from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.first_shot_source_kernel import evaluate_projected_targets


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/fit.json"
REPORT = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/runtime_verification.json")
METRICS = dict(schema="rinkspace.first-shot-runtime-verification.v1", checks={})
GEOMETRY_COLUMNS = (12, 17, 32, 45, 48, 100, 109, 111)
POLICY = dict(schema="rinkspace.continuous-board-association.v1",
              candidate_parts=["upper_right", "upper_long_side", "upper_left"])


@pytest.fixture(scope="module", autouse=True)
def measurements():
    yield
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    METRICS["status"] = "measurements_written_pending_pytest_result"
    METRICS["implementation"] = [source.reference(ROOT / path) for path in (
        "calibration/program_tracking/first_shot_fit.py",
        "calibration/program_tracking/first_shot_source_kernel.py",
        "calibration/program_tracking/first_shot_physical_state.py",
        "tests/test_first_shot_fit_runtime.py")]
    REPORT.write_text(json.dumps(METRICS, ensure_ascii=False, indent=2), encoding="utf8")


@pytest.fixture(scope="module")
def state():
    result = FirstShotPhysicalState.from_binding(BINDING)
    assert result.count == 5119 and result.geometry_count == 118
    return result


def accepted_frame(state, index, *, continuous=False):
    _, data = complete._direct_dataset(state.binding_reference)
    record = next(frame for frame in data["appended_frames"] if frame["index"] == index)
    packet = source.read_reference(record["source_packet"])
    binding = source.read_reference(record["source_binding"])
    features = {feature["id"]: feature for feature in packet["features"]}
    rows = []
    for observation in binding["observations"]:
        segment = features[observation["source_feature_id"]]["segments"][observation["source_segment_index"]]
        uv = np.asarray(segment["uv"], float)[observation["source_vertex_indices"]]
        sigma = np.asarray([record["source_scales_by_point_id"][pid] for pid in observation["source_point_ids"]])
        target = deepcopy(state.registry["targets"][observation["target_id"]])
        if continuous and target["kind"] == "board_edge":
            target = with_continuous_board_association(target, POLICY)
        row = model.build_row(state.approved_scene, target, uv, sigma, state.templates,
                              row_id=observation["id"], point_correspondence=observation["point_correspondence"])
        row.update(_target_key=runtime._target_key(target), source_packet=record["source_packet"],
            source_point_ids=observation["source_point_ids"], source_feature_id=observation["source_feature_id"],
            source_segment_index=observation["source_segment_index"], original_vertex_indices=observation["source_vertex_indices"])
        rows.append(row)
    frame = complete._fit_frame(record, rows, state.approved_scene, state.templates)
    frame.update(index=index, source_packet=record["source_packet"])
    return frame, record


@pytest.mark.parametrize("index", [201, 1241])
def test_actual_frozen_whitening_matches_complete_source_objective(state, index):
    frame, record = accepted_frame(state, index)
    before_uv = [row["uv"].copy() for row in frame["rows"]]
    before_sigma = [row["sigma"].copy() for row in frame["rows"]]
    before_operators = [group["operator"].matrix.copy() for group in frame["whitening_groups"]]
    paths = {row["_target_key"]: row["world"] for row in frame["rows"]}
    geometry = {key: np.broadcast_to(np.eye(3), (len(path), 3, 3)) for key, path in paths.items()}
    camera = state.camera25(state.initial_q, index)
    raw, camera_jac, geometry_jac = evaluate_projected_targets(camera, frame["rows"], paths, geometry)
    whitening = runtime.FrozenWhitening(frame)
    got_r, got_j, got_g = whitening.apply(raw), whitening.apply(camera_jac), whitening.apply(geometry_jac)
    expected_r, expected_j = complete.evaluate_frame(camera, frame)
    np.testing.assert_allclose(got_r, expected_r, rtol=2e-12, atol=1e-9)
    # Source sigma division/reconstruction and batched versus separate BLAS
    # dot products may round differently on cancelling source-only rows.
    # Bound those operations using their actual absolute summands, instead
    # of relaxing a fixed tolerance for arbitrary objective differences.
    rounding = np.empty_like(got_j)
    for indices, matrix, destination in whitening.groups:
        rounding[destination] = 64*np.finfo(float).eps*(abs(matrix) @ abs(camera_jac[indices]))
    rounding[whitening.ordinary_start:] = 64*np.finfo(float).eps*abs(expected_j[whitening.ordinary_start:])
    assert np.all(abs(got_j-expected_j) <= rounding+1e-10)
    reference_g = whitening.apply(-camera_jac[:, :3]/100.)
    geometry_rounding = np.empty_like(got_g)
    for indices, matrix, destination in whitening.groups:
        summands = np.maximum(abs(geometry_jac[indices]), abs(camera_jac[indices, :3]/100.))
        geometry_rounding[destination] = 64*np.finfo(float).eps*(abs(matrix) @ summands)
    geometry_rounding[whitening.ordinary_start:] = 64*np.finfo(float).eps*abs(reference_g[whitening.ordinary_start:])
    assert np.all(abs(got_g-reference_g) <= geometry_rounding+1e-10)
    # Check the explicit raw reconstruction independently of batched W.
    normalized_r, normalized_j = model.evaluate_rows(camera, frame["rows"])
    original_sigma = np.repeat(np.concatenate(before_sigma), 2)
    np.testing.assert_allclose(raw, normalized_r*original_sigma, rtol=2e-12, atol=1e-9)
    np.testing.assert_allclose(camera_jac, normalized_j*original_sigma[:, None], rtol=2e-12, atol=2e-8)
    ids = [pid for row in frame["rows"] for pid in row["source_point_ids"]]
    assert len(ids) == len(set(ids)) == record["point_count"]
    assert not np.allclose(original_sigma, 1.)
    for group, before in zip(frame["whitening_groups"], before_operators, strict=True):
        np.testing.assert_array_equal(group["operator"].matrix, before)
        assert tuple(ids[i] for i in group["point_indices"]) == group["operator"].source_point_ids
    for row, uv, sigma in zip(frame["rows"], before_uv, before_sigma, strict=True):
        np.testing.assert_array_equal(row["uv"], uv)
        np.testing.assert_array_equal(row["sigma"], sigma)
    METRICS["checks"][f"frozen_whitening_{index}"] = dict(passed=True,
        source_points=len(ids), formal_rows=len(got_r),
        maximum_r_difference=float(np.max(abs(got_r-expected_r))),
        maximum_J_difference=float(np.max(abs(got_j-expected_j))),
        maximum_J_original_summand_roundoff_bound=float(np.max(rounding)),
        original_UV_ID_sigma_operator_matrix_preserved=True,
        scale_semantics="1/norm_W_row_is_numerical_pixel_equivalent_not_original_2N_pixels")


@pytest.fixture(scope="module")
def small_bundle(state):
    frames = [accepted_frame(state, index, continuous=True)[0] for index in (201, 1241)]
    bundle = runtime.FirstShotSourceBundle.__new__(runtime.FirstShotSourceBundle)
    bundle.state = state
    bundle.frames, bundle.whitening = frames, [runtime.FrozenWhitening(frame) for frame in frames]
    descriptors = OrderedDict((row["_target_key"], row["target"]) for frame in frames for row in frame["rows"])
    bundle.keys, bundle.target_descriptors = tuple(descriptors), tuple(descriptors.values())
    bundle.scales = np.concatenate([w.scales for w in bundle.whitening])
    bundle.quadratic = np.zeros(len(bundle.scales), bool)
    bundle.source_workers, bundle._source_pool, bundle.timings = 1, None, []
    bundle.ledgers = [dict(family="actual_accepted_runtime_small_check", programme_index=frame["index"],
        native_pts=frame["native_pts"], time_base="1/12800", source_points=frame["source_point_count"],
        source_packet=frame["source_packet"], original_source_point_ids=[pid for row in frame["rows"] for pid in row["source_point_ids"]])
        for frame in frames]
    cache = {}
    def world(self, q, jacobian=False):
        token = np.asarray(q[:state.geometry_count]).tobytes(), bool(jacobian)
        if token in cache:
            return cache[token]
        world = dict(zip(self.keys, state.world_paths(q, self.target_descriptors), strict=True))
        derivative = None
        if jacobian:
            derivative = {key: np.zeros(path.shape+(state.geometry_count,)) for key, path in world.items()}
            h = float(state.provider.fd_step)
            for column in GEOMETRY_COLUMNS:
                plus, minus = q.copy(), q.copy(); plus[column] += h; minus[column] -= h
                wp, wm = state.world_paths(plus, self.target_descriptors), state.world_paths(minus, self.target_descriptors)
                for key, p, m in zip(self.keys, wp, wm, strict=True):
                    derivative[key][:, :, column] = (p-m)/(2*h)
        cache[token] = world, derivative
        return world, derivative
    bundle.world = MethodType(world, bundle)
    return bundle


def test_actual_state_whitened_complete_directional_fd_and_csr_mapping(state, small_bundle):
    q = state.initial_q.copy()
    r, j = small_bundle.evaluate(q, True)
    assert j.shape == (len(r), 5119) and j.has_sorted_indices and j.has_canonical_format
    assert np.isfinite(j.data).all() and np.isfinite(r).all()
    assert not np.any(j[:, 127:927].toarray())  # No frame 1..200 motion coordinates.
    direction = np.zeros_like(q)
    direction[list(GEOMETRY_COLUMNS)] = [0.2, -.1, .15, -.12, .08, .13, -.09, .11]
    direction[118:127] = [.1, -.12, .05, .02, -.015, .03, .4, -.3, .2]
    direction[state.view_columns(201)] = [.004, -.003, .002, .008]
    direction[state.view_columns(1241)] = [-.003, .002, -.001, .006]
    # At 2e-5 two actual corner/curve points cross adjacent nearest segments.
    # The smaller probe stays on the local piecewise differentiable branch.
    step = 2e-6
    rp = small_bundle.evaluate(q + step*direction, False)
    rm = small_bundle.evaluate(q - step*direction, False)
    fd = (rp-rm)/(2*step)
    analytic = j @ direction
    worlds, derivatives = small_bundle.world(q, True)
    plus_worlds, _ = small_bundle.world(q+step*direction, False)
    minus_worlds, _ = small_bundle.world(q-step*direction, False)
    absolute_maps, raw_analytic, raw_fds, raw_plus, raw_minus = [], [], [], [], []
    for frame, whitening in zip(small_bundle.frames, small_bundle.whitening, strict=True):
        camera, columns, chain = state.camera15_mapping(q, frame["index"])
        raw, cam, geom = evaluate_projected_targets(camera, frame["rows"], worlds, derivatives)
        predicted = geom @ direction[:118] + cam @ chain @ direction[columns]
        rawp = evaluate_projected_targets(state.camera25(q+step*direction,frame["index"]),
            frame["rows"],plus_worlds,jacobian=False)[0]
        rawm = evaluate_projected_targets(state.camera25(q-step*direction,frame["index"]),
            frame["rows"],minus_worlds,jacobian=False)[0]
        measured = (rawp-rawm)/(2*step)
        np.testing.assert_allclose(predicted, measured, rtol=2e-6, atol=2e-6)
        absolute = np.zeros((whitening.count,len(raw)))
        for indices, matrix, destination in whitening.groups:
            absolute[destination,indices] = abs(matrix)
        ordinary_rows = whitening.ordinary_start+np.arange(len(whitening.ordinary))
        absolute[ordinary_rows,whitening.ordinary] = 1./whitening.ordinary_scales
        absolute_maps.append(absolute)
        raw_analytic.append(predicted); raw_fds.append(measured)
        raw_plus.append(rawp); raw_minus.append(rawm)
    # The saved source covariance has one cancelling group with |W| about
    # 9.45e6. A sub-micro-pixel FD rounding error cannot promise a uniform
    # 1e-4 bound after that matrix. Verify the raw derivative independently,
    # then bound its *actual* source-only propagation and dot-product rounding.
    bounds = []
    for w, predicted, measured, p, m in zip(absolute_maps,raw_analytic,raw_fds,raw_plus,raw_minus,strict=True):
        bounds.append(w @ abs(predicted-measured) + 64*np.finfo(float).eps*(
            w @ ((abs(p)+abs(m))/(2*step)+abs(predicted))))
    propagated_bound = np.concatenate(bounds)
    assert np.all(abs(analytic-fd) <= propagated_bound+1e-8)
    np.testing.assert_array_equal(small_bundle.evaluate(q, False), r)
    METRICS["checks"]["physical_full_directional_fd"] = dict(passed=True,
        actual_programme_indices=[201,1241], full_state_coordinates=5119,
        geometry_columns_tested=list(GEOMETRY_COLUMNS), camera_columns="all9shared plus actual RF4 per frame",
        maximum_absolute_error=float(np.max(abs(analytic-fd))),
        maximum_scaled_error=float(np.max(abs(analytic-fd)/np.maximum(1.,abs(fd)))),
        maximum_raw_pixel_directional_error=float(max(np.max(abs(a-f)) for a,f in zip(raw_analytic,raw_fds,strict=True))),
        maximum_source_whitening_absolute_coefficient=float(max(w.max() for w in absolute_maps)),
        maximum_propagated_FD_and_BLAS_roundoff_bound=float(propagated_bound.max()),
        whitened_FD_uniform_tolerance_claim=False,
        all118_geometry_derivatives_not_rebuilt=True, sparse_shape_and_original_column_mapping_valid=True)


def test_existing_print_and_circle_material_columns_have_real_world_influence(state):
    keys = ["Ad13.print.outer_circle_print_stroke", "Ad11.print.outermost_visible_circular_print_stroke"]
    descriptors = [state.registry["targets"][key] for key in keys]
    initial = state.world_paths(state.initial_q, descriptors)
    for column, target in ((100,0), (102,0), (109,1), (111,1)):
        shifted = state.initial_q.copy(); shifted[column] += 1e-4
        changed = state.world_paths(shifted, descriptors)
        assert changed[target].shape == initial[target].shape
        assert np.max(abs(changed[target]-initial[target])) > 1e-6
    METRICS["checks"]["existing_print_circle_columns"] = dict(passed=True,
        existing_material_columns=[100,102,109,111], synthetic_or_additional_FIT_points_created=False)


def test_actual_streaming_blocks_keep_native_clock_ledgers_and_frozen_source_contract(state, small_bundle):
    install()
    from z190_streaming_source_normal import accumulate_source_normal, replay_source_cost
    blocks = list(small_bundle.source_blocks(state.initial_q, True))
    assert len(blocks) == 1
    assert blocks[0].row_ledger["frames"] == small_bundle.ledgers
    for ledger in blocks[0].row_ledger["frames"]:
        assert len(ledger["original_source_point_ids"]) == ledger["source_points"]
        assert ledger["time_base"] == "1/12800"
    np.testing.assert_array_equal(blocks[0].scales, small_bundle.scales)
    assert not blocks[0].quadratic.any() and blocks[0].J.shape[1] == 5119
    accumulated = accumulate_source_normal(lambda: iter(blocks), parameter_count=5119,
        expected_row_count=len(small_bundle.scales), frozen_profile_token="stateless_actual_source_whitening",
        retain_native_replay=True, source_state=state.initial_q)
    trial_cost = replay_source_cost(lambda: small_bundle.source_blocks(state.initial_q, False),
        accumulated.contract, frozen_profile_token="stateless_actual_source_whitening")
    assert trial_cost == accumulated.cost
    METRICS["checks"]["actual_source_block_contract"] = dict(passed=True,
        source_blocks=1, actual_source_frames=2, original_source_points=sum(frame["source_point_count"] for frame in small_bundle.frames),
        formal_rows=len(small_bundle.scales), native50_clock_preserved=True,
        trial_and_linearization_source_cost=trial_cost)


def test_light_camera_state_and_actual_source_worker_spawn(state, small_bundle):
    camera_state = runtime._CameraState(state)
    encoded = pickle.dumps(camera_state, protocol=5)
    restored = pickle.loads(encoded)
    assert not hasattr(restored,"provider") and len(encoded) < 200000
    for index in (201,1241):
        np.testing.assert_array_equal(restored.camera25(state.initial_q,index), state.camera25(state.initial_q,index))
    frame, whitening = small_bundle.frames[0], small_bundle.whitening[0]
    worlds, _ = small_bundle.world(state.initial_q, False)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
            initializer=runtime._init_source, initargs=(camera_state,[frame],[whitening])) as executor:
        got_r, got_j, receipt = executor.submit(runtime._source_chunk,
            (state.initial_q,worlds,None,[0])).result()
    expected_r, _ = runtime._evaluate_frame(state,state.initial_q,frame,whitening,worlds,None)
    np.testing.assert_array_equal(got_r, expected_r)
    assert got_j is None and receipt["backend"]["stage"] == "first_shot_source_worker"
    assert receipt["backend"]["projection_hook"]["enabled"]
    for library in receipt["backend"]["actual_native_libraries"].values():
        assert library["source_and_library_match_manifest"]
    METRICS["checks"]["actual_source_worker_spawn"] = dict(passed=True,
        light_camera_state_pickle_bytes=len(encoded), geometry_factory_serialized=False,
        actual_source_rows=len(got_r), source_values_byte_identical=True,
        worker_pid=receipt["pid"], actual_backend=receipt["backend"])


def test_public_original_streaming_schur_solver_checkpoint_roundtrip(tmp_path):
    install()
    from calibration.solver_runtime import solve_joint_source
    from z190_streaming_source_normal import SourceRowBlock
    initial, target = np.array([2., 1.]), np.array([0., 2.])
    roles, scales = np.ones(2, bool), np.array([.7,1.3])
    identity = dict(schema="rinkspace.synthetic-runtime-roundtrip.v1", source="two_original_rows")
    ledger = dict(family="synthetic_original_source_contract", original_start=0, original_stop=2)
    jac = csr_matrix(np.eye(2))
    def blocks(q, derivative):
        yield SourceRowBlock(q-target, jac if derivative else None, roles, scales, ledger)
    def unused(*args):
        raise AssertionError("The public optimizer must use actual streaming source blocks")
    def checkpoint(q, event):
        assert not q.flags.writeable
        if event["checkpoint_kind"] == "accepted":
            return dict(stop_branch=True, reason="One synthetic real-source acceptance verifies runtime")
    kwargs = dict(source_block_factory=blocks, cache_native_source_replay=True,
        reuse_source_jacobian=True, global_schur_count=1, schur_backtracking=True,
        adaptive_source_backtracking=True, schur_view_lm_ratio=1.,
        schur_backtracking_aware_damping=True, checkpoint_identity=identity,
        checkpoint_arrays=dict(original_ids=np.array([1,2])), checkpoint_callback=checkpoint, verbose=False)
    result = solve_joint_source(unused, initial, roles, scales, tmp_path, lambda q: True, **kwargs)
    assert result.status == "stopped_by_source_review" and result.history[0]["accepted"]
    assert np.linalg.norm(result.q-target) < np.linalg.norm(initial-target)
    with np.load(result.checkpoint, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["q"], result.q)
        np.testing.assert_array_equal(saved["initial_q"], initial)
        np.testing.assert_array_equal(saved["scales"], scales)
        saved_damping, saved_fraction = float(saved["damping"]), float(saved["next_line_search_fraction"])
    resumed_start = []
    def resume_checkpoint(q, event):
        if event["checkpoint_kind"] == "resumed":
            resumed_start.append(q.copy())
            return dict(stop_branch=True, reason="Exact synthetic resume boundary verified")
    kwargs["checkpoint_callback"] = resume_checkpoint
    resumed = solve_joint_source(unused, initial, roles, scales, tmp_path, lambda q: True, resume=True, **kwargs)
    assert len(resumed_start) == 1
    np.testing.assert_array_equal(resumed_start[0], result.q)
    np.testing.assert_array_equal(resumed.q, result.q)
    assert resumed.damping == saved_damping
    with np.load(resumed.checkpoint, allow_pickle=False) as saved:
        assert float(saved["next_line_search_fraction"]) == saved_fraction
    METRICS["checks"]["public_solver_checkpoint_roundtrip"] = dict(passed=True,
        synthetic_parameter_count=2, synthetic_source_rows=2, exact_q_and_initial_q_preserved=True,
        damping_and_learned_fraction_preserved=True, actual_programme_optimizer_started=False)
