"""Small source-camera initializers on an authenticated complete state.

This prepares a candidate, not a solver result. Shared world, fixed pivots,
levers and optics stay unchanged; the complete existing source must evaluate
the candidate with fresh profiles/J before any numerical acceptance.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _q_sha(q):
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def unit_source_normals(normals, count):
    """Convert only float32 unit-vector roundoff for the strict old operator.

    Stored source values are untouched. A non-unit source direction is an
    input error, rather than a reason to silently rescale its pixel support.
    """
    original = np.asarray(normals, dtype=np.float64)
    if original.shape != (count, 2) or not np.isfinite(original).all():
        raise ValueError("One finite original source normal per source UV is required")
    lengths = np.linalg.norm(original, axis=1)
    error = float(np.max(np.abs(lengths - 1.)))
    if error > 1e-6:
        raise ValueError("Source normals must already be unit vectors within float32 roundoff")
    return original / lengths[:, None], error


def source_jacobian_in_camera25_units(jacobian):
    """Undo the old source operator's C/100, PP/1000 and skew/1000 chart.

    Its values consume physical camera25, while its derivative columns use
    the retained conditioned eleven-parameter chart plus four raw Brown
    coefficients. SharedCalibration maps physical camera25 components.
    """
    value = np.asarray(jacobian,dtype=float)
    if value.ndim != 2 or value.shape[1] != 15 or not np.isfinite(value).all():
        raise ValueError("Existing finite fifteen-column source derivative required")
    return value*np.array([.01,.01,.01,1.,1.,1.,1.,1.,.001,.001,.001,1.,1.,1.,1.])


def authenticated_view_slots(state_arrays, camera_arrays, native_pts, *, allow_reference=False):
    """Locate the actual RF4 slot; do not infer it from sorted native time."""
    positions = np.flatnonzero(camera_arrays["pts"] == native_pts)
    if len(positions) != 1:
        raise ValueError("One exact original native presentation is required")
    native_index = int(positions[0])
    actual_pts = int(camera_arrays["exposure_representative_pts"][native_index])
    canonical = np.asarray(state_arrays["aux__canonical_pts"])
    matches = np.flatnonzero(canonical == actual_pts)
    if len(matches) != 1:
        raise ValueError("One authenticated actual exposure must belong to the complete state")
    ordinal = int(matches[0])
    lengths = np.asarray(state_arrays["aux__block_lengths"])
    if np.any(lengths % 4):
        raise ValueError("Original block lengths must retain their actual four-scalar view chart")
    cumulative = np.r_[0, np.cumsum(lengths // 4)]
    if int(cumulative[-1]) != len(canonical):
        raise ValueError("Complete original exposure order differs from block lengths")
    block = int(np.searchsorted(cumulative[1:], ordinal, side="right"))
    if ordinal == int(cumulative[block]) and not allow_reference:
        raise ValueError("Reference exposure cannot be changed by a local RF4 initializer")
    first = int(state_arrays["aux__block_start"][block]) + 4 * (ordinal - int(cumulative[block]))
    indices = np.arange(first, first + 4, dtype=np.int64)
    original = state_arrays["q"][indices]
    view = camera_arrays["view_state4"][native_index]
    # The retained complete solver chart stores final log(f_px), whereas
    # ViewState exports log(f_domain_px/domain_width). Keep this original
    # translation explicit; writing the normalized value into q is incorrect.
    camera25 = camera_arrays["camera25"][native_index]
    if (not np.array_equal(original[:3], view[:3]) or
            original[3] != float(np.mean(camera25[6:8]))):
        raise ValueError("Complete q RF4 does not exactly reproduce the frozen native camera export")
    return indices, native_index, actual_pts


def frozen_scene_curve(scene, curve):
    """Use the same exported current material; pixels never construct XYZ."""
    features = {item["id"]: item for item in scene["ice_features"]}
    kind = curve["kind"]
    if kind == "house_ring":
        feature = features[f"{curve['side']}{curve['row']}"]
        ring = next(item for item in feature["rings"] if item["role"] == curve["ring"])
        angle = np.linspace(0., 2. * np.pi, 361)
        return np.asarray(ring["center_xyz"]) + ring["radius"] * np.c_[np.cos(angle), np.sin(angle), np.zeros(len(angle))]
    if kind == "current_ice_segment":
        identity, index = curve["feature_id"], curve["segment_index"]
        if identity.startswith("lane"):
            if index not in (0, 2):
                raise ValueError("Only original modeled Lane long sides are admitted")
            segment = features[identity]["paint_segments_xyz"][index]
        else:
            if identity not in ("B1", "B2") or index != 0:
                raise ValueError("An explicit modeled ice segment is required")
            segment = features[identity]["segments_xyz"][index]
        return np.linspace(*np.asarray(segment, dtype=float), 1025)
    if kind in ("board_top", "ice_board_contact"):
        field = "top_edge_xyz" if kind == "board_top" else "bottom_edge_xyz"
        world = np.asarray(scene["board_surface"][field], dtype=float)
        # Direction names describe the source view, never a material region.
        # The same closed finite perimeter is used by source and full export.
        if not np.array_equal(world[0],world[-1]):world=np.vstack((world,world[0]))
        return world
    if kind == "advertisement_boundary":
        advertisement = next(item for item in scene["advertisements"]
            if item["user_number"] == curve["advertisement_number"])
        scope = curve.get("extent_scope")
        if (scope not in ("external_border", "internal_graphic_only")
                or advertisement.get("extent_scope") != scope):
            raise ValueError("Source print scope must match the actual modeled finite material")
        field = {"top_edge": "top_edge_xyz", "bottom_edge": "bottom_edge_xyz",
                 "left_vertical": "left_vertical_xyz", "right_vertical": "right_vertical_xyz"}[curve["edge"]]
        return np.asarray(advertisement["fitted_boundary_curves_xyz"][field], dtype=float)
    if kind in ("advertisement_circle", "board_circle"):
        number = str(curve.get("advertisement_number", curve.get("advertisement_id")))
        role = curve.get("target_role", curve.get("extent_scope", curve.get("edge")))
        if role not in ("outermost_visible_circular_print_stroke", "visible outermost circular print stroke"):
            raise ValueError("A board-circle initializer requires the already confirmed outermost print role")
        item = scene["additional_board_circle_patterns"]["circles"][number]
        return np.asarray(item["circle_outer_stroke_xyz_scene_units"], dtype=float)
    raise ValueError("Unsupported material must use its existing verified world provider")


def forward_connected_perimeter_indices(mask):
    """Retain one original cyclic run; never join across behind-camera edges."""
    valid = np.asarray(mask)
    if valid.dtype != np.bool_ or valid.ndim != 1 or len(valid) < 3 or not valid.any():
        raise ValueError("A real central-forward perimeter visibility mask is required")
    if valid.all():
        return np.r_[np.arange(len(valid)), 0]
    starts = np.flatnonzero(valid & ~np.roll(valid, 1))
    if len(starts) != 1:
        raise ValueError("Multiple central-forward perimeter branches require explicit physical scope")
    indices, position = [], int(starts[0])
    while valid[position]:
        indices.append(position)
        position = (position + 1) % len(valid)
    if len(indices) < 2:
        raise ValueError("No complete original finite forward perimeter segment")
    return np.asarray(indices, dtype=np.int64)


def current_initializer_source_rows(camera25, rows):
    """Retain the complete physical curve and the existing nearest finite foot.

    Direction labels describe source context. They never select a smaller
    course face or a min-y scanline branch. The actual observation operator
    already validates each optical segment, without joining disjoint parts.
    """
    return list(rows)


def camera_with_pivot_height(calibration, view, original_height, candidate_height, scene_unit_working_m):
    """Change only the existing shared pivot Z in its explicit working chart."""
    from dataclasses import replace
    values = np.asarray([original_height, candidate_height, scene_unit_working_m], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.):
        raise ValueError("Original/candidate pivot heights and declared working scale must be positive")
    pivot = np.asarray(calibration.pivot_xyz).copy()
    pivot[2] += (candidate_height - original_height) / scene_unit_working_m
    return replace(calibration, pivot_xyz=tuple(pivot)).camera25_and_jacobians(view)


def current_complete_source_geometry(state_path, auxiliary_path):
    """Restore the actual material chart, including explicit free geometry tails.

    Historical states without the new initializer retain their original
    named66 constructor. No profiles or source Jacobian are loaded.
    """
    from calibration.solver_runtime import install
    install()
    from isolated_solver_bootstrap import install as install_isolated
    install_isolated()
    from simple_wide_perspective_reset_candidate.audit_structured_current_axis_reference import make_structured_geometry
    from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry
    from structured_programme_source_candidate.paired_location_programme_geometry import PairedLocationProgrammeGeometry
    from user_corrected_programme_candidate.initialize_latest9a8c_world66 import apply_latest_geometry_auxiliary
    initializer = None
    with np.load(state_path,allow_pickle=False) as arrays:
        slots=arrays['aux__current_shared_world_geometry_parameter_indices'].copy()
        complete_q=arrays['q'].copy()
        key='aux__current_physical_geometry_initializer_json'
        if key in arrays.files:
            value=arrays[key]
            if value.ndim!=0:
                raise ValueError('Scalar saved physical geometry initializer required')
            initializer=json.loads(str(value.item()))
            if not isinstance(initializer,dict):
                raise ValueError('Typed saved physical geometry initializer required')
    if (slots.ndim!=1 or slots.dtype.kind not in 'iu'
            or complete_q.ndim!=1 or not np.isfinite(complete_q).all()
            or np.any(slots<0) or np.any(slots>=len(complete_q))):
        raise ValueError('Explicit current complete-state geometry slots required')
    g62,q62,owner,*_=make_structured_geometry()
    original=owner.g
    while type(original).__name__!='SharedIceHockeyNumericGeometry':
        original=original.original
    kwargs={}
    if initializer is not None:
        locations=initializer.get('additional_independent_locations',[])
        templates=initializer.get('frozen_source_print_templates',[])
        lanes=initializer.get('independent_lane_right_extensions',False)
        strict_crease=initializer.get('strict_half_circle_crease',False)
        derived_center=initializer.get('derive_hockey_center_from_course',False)
        crease_placements=initializer.get('independent_goal_crease_transverse_placements',False)
        shared_axis=initializer.get('shared_hockey_curling_axis',False)
        if (not isinstance(locations,list) or not isinstance(templates,list)
                or not isinstance(lanes,bool) or not isinstance(strict_crease,bool)
                or not isinstance(derived_center,bool) or not isinstance(crease_placements,bool)
                or not isinstance(shared_axis,bool)):
            raise ValueError('Explicit location, frozen-print, lane, crease and centre initializer fields required')
        kwargs=dict(additional_independent_locations=locations,
                    source_print_templates=templates,
                    independent_lane_right_extensions=lanes,
                    strict_half_circle_crease=strict_crease,
                    derive_hockey_center_from_course=derived_center,
                    independent_goal_crease_transverse_placements=crease_placements,
                    shared_hockey_curling_axis=shared_axis)
    current=PairedLocationsSharedSizesGeometry(g62.parent,q62[:g62.parent_parameter_count],**kwargs)
    payload=json.loads(Path(auxiliary_path).read_text(encoding='utf-8-sig'))
    apply_latest_geometry_auxiliary(current,payload)
    world=complete_q[slots].copy()
    ng=int(current.parameter_count)
    if world.shape!=(ng,) or len(np.unique(slots))!=ng:
        raise ValueError("Actual complete current geometry slots required")
    provider=PairedLocationProgrammeGeometry(current,original)
    provider.validate_parameters(world)
    return provider,world,dict(auxiliary_path=str(Path(auxiliary_path).resolve()),auxiliary_sha256=_sha(auxiliary_path),
        actual_world_parameter_indices=slots.tolist(),actual_world_float64_sha256=_q_sha(world),
        actual_world_parameter_count=ng,explicit_saved_physical_geometry_initializer=initializer,
        strict_half_circle_crease=bool(kwargs.get('strict_half_circle_crease',False)),
        derive_hockey_center_from_course=bool(kwargs.get('derive_hockey_center_from_course',False)),
        independent_goal_crease_transverse_placements=bool(kwargs.get('independent_goal_crease_transverse_placements',False)),
        shared_hockey_curling_axis=bool(kwargs.get('shared_hockey_curling_axis',False)),
        existing_driver_material_constructor_reused=True,large_profiles_and_source_J_not_loaded=True)


def apply_absolute_camera_initializer(q, packet, *, delta_path=None):
    """Pure full-state transaction, suitable after complete checkpoint restore."""
    value = np.asarray(q)
    if value.dtype != np.dtype("float64") or value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("The complete original float64 parameter state is required")
    if (packet.get("schema") != "rinkspace.source-qualified-camera-pose-initializer.v1" or
            packet.get("role") != "initializer_only" or packet.get("accuracy_accepted") is not False or
            packet.get("initializer_is_not_accepted_fit") is not True or
            packet.get("shared_world_pivots_levers_and_optics_unchanged") is not True or
            packet.get("fresh_complete_source_profiles_and_J_required") is not True or
            packet.get("parent_q_sha256") != _q_sha(value)):
        raise ValueError("RF4 initializer is not bound to this exact complete parent state")
    path = Path(delta_path or packet["delta_npz"]["path"])
    if _sha(path) != packet["delta_npz"]["sha256"]:
        raise ValueError("Initializer typed delta changed")
    with np.load(path, allow_pickle=False) as arrays:
        indices, before, after = (arrays[name].copy() for name in ("parameter_indices", "original_values", "candidate_values"))
    permitted = np.asarray(packet["permitted_RF4_indices"], dtype=np.int64)
    if (indices.dtype.kind not in "iu" or indices.ndim != 1 or not len(indices) or
            len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(value)) or
            not np.array_equal(indices, permitted) or before.dtype != value.dtype or after.dtype != value.dtype or
            before.shape != indices.shape or after.shape != indices.shape or not np.isfinite(after).all() or
            not np.array_equal(value[indices], before)):
        raise ValueError("Initializer must preserve the original RF4 slot domain and values exactly")
    protected = np.asarray(packet["protected_shared_parameter_indices"], dtype=np.int64)
    if np.intersect1d(indices, protected).size:
        raise ValueError("An RF4 initializer cannot change shared physical or world parameters")
    candidate = value.copy()
    candidate[indices] = after
    return candidate


def prepare_absolute_camera_initializer(state_path, camera_path, scene_path, source_packet_path, output_directory):
    """Solve only tiny independent RF4 initializers from reviewed clear RGB.

    The source packet is an existing native ice/board packet or the explicit
    initializer-source schema. Each admitted row needs unchanged UV/sigma,
    exact source clock/provenance and original RGB review. No fallback sigma,
    foreground point, transferred neighboring UV or expected identity is used.
    """
    from scipy.optimize import least_squares
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
    from calibration.solver_runtime import install
    install()
    from z190_source_camera_diagnostic import absolute_value_jac15

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path, delta_path = output / "absolute_RF4_initializer_candidate.json", output / "absolute_RF4_initializer_delta.npz"
    if report_path.exists() or delta_path.exists():
        raise FileExistsError("An existing initializer and its source evidence must not be overwritten")
    with np.load(state_path, allow_pickle=False) as arrays:
        state = {name: arrays[name].copy() for name in ("q", "aux__canonical_pts", "aux__block_lengths", "aux__block_start",
            "aux__installation_xyz_slots", "aux__current_shared_mount_parameter_indices",
            "aux__current_all_installation_optics_parameter_indices", "aux__current_shared_world_geometry_parameter_indices")}
    with np.load(camera_path, allow_pickle=False) as arrays:
        camera = {name: arrays[name].copy() for name in ("pts", "exposure_representative_pts", "installation_id", "view_state4", "camera25")}
        metadata = json.loads(str(arrays["metadata_json"].item()))
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8-sig"))
    source = json.loads(Path(source_packet_path).read_text(encoding="utf-8-sig"))
    if (source.get("schema") not in ("rinkspace.native-absolute-camera-initialization-source.v1",
            "rinkspace.additional-native-ice-curves.v1", "rinkspace.additional-native-board-boundary-source.v1") or
            source.get("source_video_sha256") != metadata["source_sha256"] or
            source.get("time_base") != metadata["source_time_base"]):
        raise ValueError("Clear absolute source packet must belong to the same authenticated video and native clock")
    groups, controls = defaultdict(list), defaultdict(list)
    source_rows = [(row, True) for row in source["rows"]] + [(row, False) for row in source.get("evaluation_rows", ())]
    for source_row, fit_row in source_rows:
        if source_row.get("source_RGB_actual_measurement") is not True or not source_row.get("physical_identity_authority"):
            raise ValueError("Clear source role/identity and actual RGB measurement need explicit source review")
        if source_row.get("model_projection_used_to_generate_or_select_uv") is not False:
            raise ValueError("Projected or unspecified source UV cannot initialize a camera")
        native = int(source_row["pts"])
        indices, native_index, actual = authenticated_view_slots(state, camera, native)
        if source_row.get("actual_pts", actual) != actual:
            raise ValueError("Source exposure disagrees with the original proof")
        frame = source_row.get("source_raw_frame", source_row.get("source_path"))
        frame_sha = source_row.get("source_file_sha256", source_row.get("source_frame_sha256"))
        if frame is None or frame_sha is None or _sha(frame) != frame_sha:
            raise ValueError("Original clear RGB file identity is required")
        uv = np.asarray(source_row["uv"], dtype=np.float64)
        sigma = np.broadcast_to(np.asarray(source_row["sigma_px"], dtype=np.float64), (len(uv),)).copy()
        if (uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv) or not np.isfinite(uv).all() or
                np.any(uv < 0.) or np.any(uv >= [1920., 1080.]) or
                np.any(sigma <= 0) or not np.isfinite(sigma).all()):
            raise ValueError("Original finite native UV and explicit working sigma are required")
        row = dict(source_row, uv=uv, sigma=sigma, world=frozen_scene_curve(scene, source_row["curve"]))
        if "normal_interval_px" in source_row:
            row["normal_interval"] = np.asarray(source_row["normal_interval_px"], dtype=float)
            row["normal"], row["original_source_normal_unit_roundoff_max"] = unit_source_normals(source_row["normal"], len(uv))
        (groups if fit_row else controls)[(actual, native_index, tuple(indices))].append(row)
    if not groups:
        raise ValueError("No reviewed actual absolute source measurements")

    all_indices, all_before, all_after, diagnostics = [], [], [], []

    def current_source_rows(camera25, rows):
        return current_initializer_source_rows(camera25, rows)

    def native_pixel_diagnostic(camera25, row):
        # Unnormalized original optical residual and separately the existing
        # interval-censored residual. Neither source positions nor support is
        # moved toward the current projection.
        current_row = current_source_rows(camera25, [row])[0]
        raw_row = dict(current_row)
        raw_row.pop("normal_interval", None)
        sigma2 = row["sigma"][:, None]
        raw = np.asarray(absolute_value_jac15(camera25, [raw_row], False)).reshape(-1, 2) * sigma2
        outside = np.asarray(absolute_value_jac15(camera25, [current_row], False)).reshape(-1, 2) * sigma2
        lengths, outside_lengths = np.linalg.norm(raw, axis=1), np.linalg.norm(outside, axis=1)
        return dict(raw_native_pixel_error_mean=float(lengths.mean()), raw_native_pixel_error_max=float(lengths.max()),
            raw_native_pixel_error_per_original_point=lengths.tolist(),
            projected_minus_original_source_UV_native_px=raw.tolist(),
            outside_original_normal_interval_error_mean_px=float(outside_lengths.mean()),
            outside_original_normal_interval_error_max_px=float(outside_lengths.max()),
            outside_original_normal_interval_error_per_original_point_px=outside_lengths.tolist(),
            normal_interval_present="normal_interval" in row,
            original_source_normal_unit_roundoff_max=row.get("original_source_normal_unit_roundoff_max", 0.),
            working_normals_only_unit_roundoff_normalized_original_packet_unchanged=True,
            derived_forward_perimeter_indices=current_row.get("derived_current_forward_perimeter_indices", np.empty(0,dtype=int)).tolist(),
            derived_physical_forward_scope_is_not_source_measurement=bool("derived_current_forward_perimeter_indices" in current_row),
            accuracy_or_independent_holdout_not_claimed=True)

    for (actual, native_index, indices), rows in groups.items():
        identity = str(camera["installation_id"][native_index])
        calibration = SharedCalibration.from_manifest(metadata["calibrations"][identity])
        initial = camera["view_state4"][native_index].copy()
        if not np.array_equal(calibration.camera25(ViewState.from_state4(initial)), camera["camera25"][native_index]):
            raise ValueError("Physical camera initialization must reproduce the actual frozen calibration exactly")
        if np.any(camera["camera25"][native_index, 15:] != 0.):
            raise ValueError("Unlocking higher Brown/tilt terms requires its existing complete source operator")

        def evaluate(view, jacobian=False):
            current, _, mapping = calibration.camera25_and_jacobians(ViewState.from_state4(view))
            result = absolute_value_jac15(current, current_source_rows(current, rows), jacobian)
            return (result[0], source_jacobian_in_camera25_units(result[1]) @ mapping[:15]) if jacobian else result

        before = evaluate(initial)
        initial_jacobian = evaluate(initial, True)[1]
        initial_rank = int(np.linalg.matrix_rank(initial_jacobian))
        initial_singular = np.linalg.svd(initial_jacobian, compute_uv=False)
        if initial_rank < 4:
            raise ValueError("Reviewed absolute sources do not constrain all four actual view parameters")
        width = calibration.output_mode.domain_wh[0]
        solved = least_squares(lambda v: evaluate(v), initial, jac=lambda v: evaluate(v, True)[1],
            bounds=([-np.inf]*3 + [np.log(100./width)], [np.inf]*3 + [np.log(50000./width)]),
            x_scale="jac", max_nfev=80)
        after = evaluate(solved.x)
        original_q_RF4 = state["q"][np.asarray(indices)].copy()
        candidate_q_RF4 = solved.x.copy()
        candidate_q_RF4[3] += original_q_RF4[3] - initial[3]
        all_indices.extend(indices); all_before.extend(original_q_RF4); all_after.extend(candidate_q_RF4)
        per_group, cursor = [], 0
        current = calibration.camera25(ViewState.from_state4(solved.x))
        original_camera = camera["camera25"][native_index]
        for row in rows:
            count = 2 * len(row["uv"])
            per_group.append(dict(id=row["id"], source_pts=row["pts"], actual_pts=actual,
                feature_id=row.get("feature_id"), curve=row["curve"], source_UV_count=len(row["uv"]),
                original_source_UV_float64_sha256=_q_sha(row["uv"]),
                before_normalized_cost=float(before[cursor:cursor+count] @ before[cursor:cursor+count] / 2),
                after_normalized_cost=float(after[cursor:cursor+count] @ after[cursor:cursor+count] / 2),
                before=native_pixel_diagnostic(original_camera, row), after=native_pixel_diagnostic(current, row),
                source_sigma_and_normal_intervals_unchanged=True))
            cursor += count
        diagnostics.append(dict(actual_pts=actual, native_pts=int(camera["pts"][native_index]), installation_id=identity,
            RF4_indices=[int(index) for index in indices], before_RF4=initial.tolist(), candidate_RF4=solved.x.tolist(),
            before_original_q_RF4=original_q_RF4.tolist(), candidate_original_q_RF4=candidate_q_RF4.tolist(),
            exported_ViewState_logf_and_original_q_logf_translation=float(original_q_RF4[3]-initial[3]),
            before_cost=float(before @ before / 2), after_cost=float(after @ after / 2),
            initial_actual_source_J_rank=initial_rank,
            initial_actual_source_J_singular_values=initial_singular.tolist(),
            initial_actual_source_J_condition_number=float(initial_singular[0]/initial_singular[-1]),
            final_interval_clipped_source_J_rank=int(np.linalg.matrix_rank(evaluate(solved.x, True)[1])),
            actual_evaluations=int(solved.nfev), optimizer_status=int(solved.status),
            candidate_camera25=current.tolist(), source_groups=per_group,
            unused_tinyfit_same_frame_source_controls=[dict(id=row["id"], source_pts=row["pts"], curve=row["curve"],
                source_UV_count=len(row["uv"]), original_source_UV_float64_sha256=_q_sha(row["uv"]),
                before=native_pixel_diagnostic(original_camera,row),after=native_pixel_diagnostic(current,row),
                consumed_by_tiny_initializer=False) for row in controls.get((actual,native_index,indices), ())],
            shared_world_pivots_levers_PP_and_Brown_unchanged=True,
            unknown_or_occluded_new_measurements_created=False))
    indices = np.asarray(all_indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Repeated native presentations of one exposure must be grouped explicitly before initialization")
    np.savez_compressed(delta_path, parameter_indices=indices,
        original_values=np.asarray(all_before, dtype=np.float64), candidate_values=np.asarray(all_after, dtype=np.float64))
    protected = np.unique(np.concatenate([np.ravel(state[name]) for name in (
        "aux__installation_xyz_slots", "aux__current_shared_mount_parameter_indices",
        "aux__current_all_installation_optics_parameter_indices", "aux__current_shared_world_geometry_parameter_indices")]))
    evidence = {int(row["pts"]): dict(path=str(Path(row.get("source_raw_frame", row.get("source_path"))).resolve()),
        sha256=row.get("source_file_sha256", row.get("source_frame_sha256")), pts=int(row["pts"])) for row, _ in source_rows}
    report = dict(schema="rinkspace.source-qualified-camera-pose-initializer.v1", role="initializer_only",
        parent_q_sha256=_q_sha(state["q"]), parent_state=dict(path=str(Path(state_path).resolve()), sha256=_sha(state_path)),
        camera_export=dict(path=str(Path(camera_path).resolve()), sha256=_sha(camera_path)),
        scene=dict(path=str(Path(scene_path).resolve()), sha256=_sha(scene_path)),
        actual_source_packet=dict(path=str(Path(source_packet_path).resolve()), sha256=_sha(source_packet_path)),
        source_video_sha256=metadata["source_sha256"], time_base=metadata["source_time_base"],
        delta_npz=dict(path=str(delta_path.resolve()), sha256=_sha(delta_path)), permitted_RF4_indices=indices.tolist(),
        actual_pts=sorted({item["actual_pts"] for item in diagnostics}), source_evidence=list(evidence.values()),
        protected_shared_parameter_indices=protected.tolist(), source_frames=diagnostics,
        shared_world_pivots_levers_and_optics_unchanged=True, initializer_is_not_accepted_fit=True,
        fresh_complete_source_profiles_and_J_required=True, old_motion_and_absolute_rows_not_reweighted_or_retired=True,
        focal_100_to_50000_px_domain_is_implementation_working_range_not_user_or_hardware_calibration=True,
        complete_original_motion_and_all_camera_source_validation_pending=True,
        initializer_source_code=dict(path=str(Path(__file__).resolve()), sha256=_sha(__file__)),
        accuracy_accepted=False, full_solver_started=False, unmeasured_hardware_calibration_not_claimed=True)
    apply_absolute_camera_initializer(state["q"], report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def solve_source_qualified_installation_views(calibration, initial_views, original_height,
        scene_unit_working_m, source_evaluators, *, source_quadratic_masks, roll_bias_radians,
        optical_height_reference_index=None, with_height=True, max_nfev=80):
    """Initialize one installation from unchanged, explicitly selected FIT factors.

    Each evaluator accepts ``(actual_camera25, jacobian)`` and returns the
    existing normalized residual and its retained, conditioned J15. No source
    rows, sigma, normal intervals or FIT/HELD roles are generated here. The
    original Huber2 source loss, fixed existing roll bias and one installation
    reference optical-height prior are used. Complete motion is still deferred
    to the original solver.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix, vstack
    from calibration.program_tracking.z190_physical_camera import ViewState
    from calibration.solver_runtime import install
    install()
    from correlated_roll_prior_candidate.explicit_bias_source import theta_and_gradient, SIGMA

    views = np.asarray(initial_views, float)
    if views.ndim != 2 or views.shape[1] != 4 or not len(views) or not np.isfinite(views).all():
        raise ValueError("Authenticated finite RF4 views required")
    if len(source_evaluators) != len(views) or len(source_quadratic_masks) != len(views) or not np.isfinite(roll_bias_radians):
        raise ValueError("Each authenticated exposure needs its original source evaluator and bias")
    original_masks=[]
    for mask in source_quadratic_masks:
        value=np.asarray(mask)
        if value.ndim!=1 or value.dtype!=np.bool_:
            raise ValueError("Explicit original per-scalar quadratic masks required")
        original_masks.append(value.copy())
    if optical_height_reference_index is not None and not 0 <= optical_height_reference_index < len(views):
        raise ValueError("Original installation reference must name an authenticated exposure")
    offset = int(with_height)
    initial = np.r_[original_height, views.ravel()] if with_height else views.ravel().copy()
    fd_modes, comparisons = [], []

    def local(ordinal, view, height, jacobian):
        camera, _, mapping = camera_with_pivot_height(calibration, ViewState.from_state4(view),
            original_height, height, scene_unit_working_m)
        result = source_evaluators[ordinal](camera, jacobian)
        if not jacobian:
            return np.asarray(result, float)
        residual, legacy = result
        physical = source_jacobian_in_camera25_units(legacy)
        return np.asarray(residual, float), np.column_stack((physical[:, 2] / scene_unit_working_m,
            physical @ mapping[:15]))

    # Verify every real factor's height/RF chain against its actual value.
    # A legacy piecewise near/finite operator may require value FD locally.
    for ordinal, view in enumerate(views):
        residual, analytic = local(ordinal, view, original_height, True)
        if original_masks[ordinal].shape!=residual.shape:
            raise ValueError("Original factor loss mask does not match actual source residuals")
        columns = []
        for column, step in enumerate((1e-6, 1e-7, 1e-7, 1e-7, 1e-6)):
            p, m = view.copy(), view.copy()
            hp = hm = original_height
            if column == 0:
                hp += step; hm -= step
            else:
                p[column-1] += step; m[column-1] -= step
            columns.append((local(ordinal,p,hp,False)-local(ordinal,m,hm,False))/(2*step))
        numerical = np.stack(columns, axis=1)
        error = np.linalg.norm(analytic-numerical) / max(1.,np.linalg.norm(numerical))
        fd_modes.append(bool(error > 1e-5))
        comparisons.append(dict(exposure_index=ordinal, residual_scalar_count=len(residual),
            actual_value_centered_FD_relative_error=float(error),
            derivative_mode='actual_value_centered_FD' if fd_modes[-1] else 'verified_original_analytic'))

    def evaluate(parameters, jacobian=False):
        height = float(parameters[0]) if with_height else original_height
        current_views = parameters[offset:].reshape(-1,4)
        values, blocks, masks = [], [], []
        for ordinal, view in enumerate(current_views):
            if jacobian:
                residual, derivative = local(ordinal, view, height, True)
                if fd_modes[ordinal]:
                    columns=[]
                    for column,step in enumerate((1e-6,1e-7,1e-7,1e-7,1e-6)):
                        p,m=view.copy(),view.copy();hp=hm=height
                        if column==0:hp+=step;hm-=step
                        else:p[column-1]+=step;m[column-1]-=step
                        columns.append((local(ordinal,p,hp,False)-local(ordinal,m,hm,False))/(2*step))
                    derivative=np.stack(columns,axis=1)
            else:
                residual=local(ordinal,view,height,False)
            theta, gradient = theta_and_gradient(view[None,:3])
            tail=[float((theta[0]-roll_bias_radians)/SIGMA)]
            tail_derivative=[np.r_[0.,gradient[0]/SIGMA,0.]]
            if ordinal == optical_height_reference_index:
                camera,_,mapping=camera_with_pivot_height(calibration,ViewState.from_state4(view),
                    original_height,height,scene_unit_working_m)
                tail.append(float((camera[2]*scene_unit_working_m-1.6)/.18))
                tail_derivative.append(np.r_[1./.18,mapping[2]*scene_unit_working_m/.18])
            values.append(np.r_[residual,tail])
            if original_masks[ordinal].shape!=residual.shape:
                raise ValueError("Original source residual inventory changed")
            masks.append(np.r_[original_masks[ordinal],np.ones(len(tail),bool)])
            if jacobian:
                derivative=np.vstack((derivative,tail_derivative))
                rows,cols=np.nonzero(derivative[:,1:])
                global_cols=offset+4*ordinal+cols
                entries=derivative[:,1:][rows,cols]
                if with_height:
                    active=np.flatnonzero(derivative[:,0])
                    rows=np.r_[rows,active];global_cols=np.r_[global_cols,np.zeros(len(active),int)]
                    entries=np.r_[entries,derivative[active,0]]
                blocks.append(csr_matrix((entries,(rows,global_cols)),shape=(len(values[-1]),len(parameters))))
        residual=np.concatenate(values)
        return (residual,vstack(blocks,format='csr'),np.concatenate(masks)) if jacobian else residual

    quadratic=evaluate(initial,True)[2]
    def original_mixed_loss(z):
        # SciPy z=r**2; priors quadratic, source factors original Huber2.
        root=np.sqrt(np.maximum(z,1e-300));active=(~quadratic)&(z>4.)
        result=np.vstack((z.copy(),np.ones_like(z),np.zeros_like(z)))
        result[0,active]=4.*root[active]-4.
        result[1,active]=2./root[active]
        result[2,active]=-1./root[active]**3
        return result
    width=calibration.output_mode.domain_wh[0]
    lower=np.tile([-np.inf,-np.inf,-np.inf,np.log(100./width)],len(views))
    upper=np.tile([np.inf,np.inf,np.inf,np.log(50000./width)],len(views))
    if with_height:lower=np.r_[1e-6,lower];upper=np.r_[np.inf,upper]
    result=least_squares(evaluate,initial,jac=lambda p:evaluate(p,True)[1],loss=original_mixed_loss,
        bounds=(lower,upper),x_scale='jac',tr_solver='lsmr',max_nfev=max_nfev)
    before,after=evaluate(initial),evaluate(result.x)
    def cost(r):return float(original_mixed_loss(r*r)[0].sum()/2.)
    return dict(candidate_views=result.x[offset:].reshape(-1,4).copy(),
        candidate_pivot_Z=float(result.x[0]) if with_height else float(original_height),
        before_cost=cost(before),after_cost=cost(after),actual_evaluations=int(result.nfev),
        optimizer_status=int(result.status),initial_actual_value_derivative_checks=comparisons,
        original_Huber2_and_quadratic_priors_preserved=True,
        original_bias_fixed_radians=float(roll_bias_radians),
        original_optical_height_prior_count=int(optical_height_reference_index is not None),
        original_roll_prior_count=len(views),
        source_quadratic_scalar_count=sum(int(mask.sum()) for mask in original_masks),
        source_Huber2_scalar_count=sum(int((~mask).sum()) for mask in original_masks),
        complete_motion_not_in_partial_initializer=True)


def prepare_shared_pivot_height_initializer(state_path, camera_path, scene_path, source_packet_paths, output_directory,
                                          *, retained_source_control_ids=(), jacobian_mode='existing_analytic'):
    """Compare existing RF-only and shared-Pz+RF on clear source anchors.

    The shared-height result needs the root solver's explicitly extended
    adapter. It is a provisional initial state, not an accepted full-source
    solution. No shared world, Pxy, lever or optical parameter is admitted.
    """
    from scipy.optimize import least_squares
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState
    from calibration.solver_runtime import install
    install()
    from z190_source_camera_diagnostic import absolute_value_jac15
    if jacobian_mode not in ('existing_analytic','centered_actual_value_FD'):
        raise ValueError("Initializer Jacobian mode must name the original analytic or actual-value FD path")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path, delta_path = output/'shared_Pz_RF_initializer_diagnostic.json', output/'shared_Pz_RF_initializer_delta.npz'
    if report_path.exists() or delta_path.exists():
        raise FileExistsError("Existing source initialization diagnostics cannot be overwritten")
    state_names = ("q", "aux__canonical_pts", "aux__block_lengths", "aux__block_start", "aux__installation_xyz_slots",
        "aux__all_installation_optics_installation_order", "aux__current_camera_mount_working_constraints_json")
    with np.load(state_path, allow_pickle=False) as saved:
        state = {name:saved[name].copy() for name in state_names}
    with np.load(camera_path, allow_pickle=False) as saved:
        camera = {name:saved[name].copy() for name in ("pts","exposure_representative_pts","installation_id","view_state4","camera25")}
        metadata = json.loads(str(saved["metadata_json"].item()))
    scene = json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
    packets, groups, controls = [], defaultdict(list), defaultdict(list)
    retained_ids = set(retained_source_control_ids)
    if len(retained_ids) != len(retained_source_control_ids):
        raise ValueError("Retained original control IDs must be unique")
    found_retained = set()
    for path in source_packet_paths:
        source = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (source.get('schema') != 'rinkspace.native-absolute-camera-initialization-source.v1' or
                source.get('source_video_sha256') != metadata['source_sha256'] or source.get('time_base') != metadata['source_time_base']):
            raise ValueError("Every shared-height anchor requires exact original video and clock binding")
        packets.append(dict(path=str(Path(path).resolve()),sha256=_sha(path)))
        for section in ('rows','evaluation_rows'):
            for original in source.get(section, ()):
                if (original.get('source_RGB_actual_measurement') is not True or not original.get('physical_identity_authority') or
                        original.get('model_projection_used_to_generate_or_select_uv') is not False):
                    raise ValueError("Shared-height anchors require actual reviewed source RGB and identity")
                indices,native_index,actual = authenticated_view_slots(state,camera,int(original['pts']))
                if original.get('actual_pts') != actual or _sha(original['source_raw_frame']) != original['source_file_sha256']:
                    raise ValueError("Shared-height source RGB/exposure evidence changed")
                uv = np.asarray(original['uv'],dtype=np.float64)
                sigma = np.broadcast_to(np.asarray(original['sigma_px'],dtype=np.float64),(len(uv),)).copy()
                if (uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv) or not np.isfinite(uv).all() or
                        np.any(uv<0.) or np.any(uv >= [1920.,1080.]) or np.any(sigma<=0.) or not np.isfinite(sigma).all()):
                    raise ValueError("Unchanged actual finite source UV and declared working sigma required")
                row = dict(original,uv=uv,sigma=sigma,world=frozen_scene_curve(scene,original['curve']),
                    original_packet_section=section,retained_original_control=original['id'] in retained_ids)
                if 'normal_interval_px' in original:
                    row['normal_interval'] = np.asarray(original['normal_interval_px'],float)
                    row['normal'],row['original_source_normal_unit_roundoff_max'] = unit_source_normals(original['normal'],len(uv))
                if row['retained_original_control']:
                    if section != 'evaluation_rows' or original['id'] in found_retained:
                        raise ValueError("Retained control must uniquely name an original evaluation row")
                    found_retained.add(original['id'])
                (groups if section=='rows' or row['retained_original_control'] else controls)[(actual,native_index,tuple(indices))].append(row)
    if found_retained != retained_ids:
        raise ValueError("Every requested retained control must exist in the authenticated source packets")
    keys = list(groups)
    identities = {str(camera['installation_id'][key[1]]) for key in keys}
    if not keys or len(identities) != 1:
        raise ValueError("One shared physical installation and explicit source anchors required")
    identity = identities.pop()
    order = state['aux__all_installation_optics_installation_order'].tolist()
    installation_index = order.index(identity)
    pivot_indices = state['aux__installation_xyz_slots'][installation_index]
    pivot_z_index = int(pivot_indices[2])
    original_height = float(state['q'][pivot_z_index])
    working_contract = json.loads(str(state['aux__current_camera_mount_working_constraints_json'].item()))
    scene_unit_working_m = float(working_contract['scene_unit_working_m'])
    calibration = SharedCalibration.from_manifest(metadata['calibrations'][identity])
    if not np.allclose(state['q'][pivot_indices]/scene_unit_working_m,calibration.pivot_xyz,rtol=1e-14,atol=1e-14):
        raise ValueError("Original fixed pivot slots and declared scene-unit chart do not match camera export")
    initial_views = [camera['view_state4'][key[1]].copy() for key in keys]
    for view,key in zip(initial_views,keys):
        if not np.array_equal(calibration.camera25(ViewState.from_state4(view)),camera['camera25'][key[1]]):
            raise ValueError("Actual physical camera must reproduce each original source anchor exactly")
        if np.any(camera['camera25'][key[1],15:] != 0.):
            raise ValueError("Higher original Brown terms require their full existing source operator")

    def evaluate(parameters, with_height, jacobian=False):
        views = parameters[1:].reshape(-1,4) if with_height else parameters.reshape(-1,4)
        height = parameters[0] if with_height else original_height
        values,derivatives = [],[]
        for ordinal,(view,key) in enumerate(zip(views,keys)):
            current,_,mapping = camera_with_pivot_height(calibration,ViewState.from_state4(view),original_height,height,scene_unit_working_m)
            result = absolute_value_jac15(current,current_initializer_source_rows(current,groups[key]),jacobian)
            if jacobian:
                residual,raw_j = result
                raw_j = source_jacobian_in_camera25_units(raw_j)
                block = np.zeros((len(residual),len(parameters)))
                start = (1 if with_height else 0)+4*ordinal
                block[:,start:start+4] = raw_j@mapping[:15]
                if with_height:
                    block[:,0] = raw_j[:,2]/scene_unit_working_m
                derivatives.append(block)
            else:
                residual = result
            values.append(residual)
        residual = np.concatenate(values)
        return (residual,np.concatenate(derivatives)) if jacobian else residual

    initial_RF = np.concatenate(initial_views)
    width = calibration.output_mode.domain_wh[0]
    lower_RF = np.tile([-np.inf,-np.inf,-np.inf,np.log(100./width)],len(keys))
    upper_RF = np.tile([np.inf,np.inf,np.inf,np.log(50000./width)],len(keys))
    solutions = []
    for with_height in (False,True):
        initial = np.r_[original_height,initial_RF] if with_height else initial_RF.copy()
        lower = np.r_[1e-6,lower_RF] if with_height else lower_RF
        upper = np.r_[np.inf,upper_RF] if with_height else upper_RF
        initial_j = evaluate(initial,with_height,True)[1]
        singular = np.linalg.svd(initial_j,compute_uv=False)
        steps = np.tile([1e-7,1e-7,1e-7,1e-6],len(keys))
        if with_height:
            steps = np.r_[1e-6,steps]
        def actual_value_FD(parameters):
            columns=[]
            for column,step in enumerate(steps):
                plus,minus=parameters.copy(),parameters.copy()
                plus[column]+=step;minus[column]-=step
                columns.append((evaluate(plus,with_height)-evaluate(minus,with_height))/(2.*step))
            return np.stack(columns,axis=1)
        initial_fd = actual_value_FD(initial)
        derivative_comparison=[dict(column=column,step=float(step),
            relative_L2_error=float(np.linalg.norm(initial_j[:,column]-initial_fd[:,column])/max(1.,np.linalg.norm(initial_fd[:,column]))),
            maximum_absolute_error=float(np.max(np.abs(initial_j[:,column]-initial_fd[:,column])))) for column,step in enumerate(steps)]
        selected_jac = actual_value_FD if jacobian_mode=='centered_actual_value_FD' else lambda p:evaluate(p,with_height,True)[1]
        solved = least_squares(lambda p:evaluate(p,with_height),initial,jac=selected_jac,
            bounds=(lower,upper),x_scale='jac',max_nfev=100)
        height = float(solved.x[0]) if with_height else original_height
        views = solved.x[1:].reshape(-1,4) if with_height else solved.x.reshape(-1,4)
        diagnostics=[]
        for key,view in zip(keys,views):
            current = camera_with_pivot_height(calibration,ViewState.from_state4(view),original_height,height,scene_unit_working_m)[0]
            source_groups=[]
            for consumed,rows in ((True,groups[key]),(False,controls.get(key,()))):
                for row in rows:
                    measurements=[]
                    for value in (camera['camera25'][key[1]],current):
                        working_row = current_initializer_source_rows(value,[row])[0]
                        raw_row = dict(working_row);raw_row.pop('normal_interval',None)
                        raw = absolute_value_jac15(value,[raw_row],False).reshape(-1,2)*row['sigma'][:,None]
                        outside = absolute_value_jac15(value,[working_row],False).reshape(-1,2)*row['sigma'][:,None]
                        measurements.append(dict(raw_native_error_mean=float(np.linalg.norm(raw,axis=1).mean()),
                            raw_native_error_max=float(np.linalg.norm(raw,axis=1).max()),
                            outside_original_normal_interval_error_mean=float(np.linalg.norm(outside,axis=1).mean()),
                            raw_projected_minus_original_source_uv=raw.tolist()))
                    source_groups.append(dict(id=row['id'],curve=row['curve'],count=len(row['uv']),used_in_tiny_initializer=consumed,
                        original_packet_section=row['original_packet_section'],
                        retained_original_control_consumed_explicitly_not_independent_holdout=row['retained_original_control'],
                        original_source_UV_float64_sha256=_q_sha(row['uv']),before=measurements[0],after=measurements[1]))
            diagnostics.append(dict(native_pts=int(camera['pts'][key[1]]),actual_pts=key[0],RF4_indices=[int(v) for v in key[2]],
                original_ViewState_RF4=camera['view_state4'][key[1]].tolist(),candidate_ViewState_RF4=view.tolist(),
                candidate_camera25=current.tolist(),source_groups=source_groups))
        solutions.append(dict(scope='shared_Pz_plus_RF' if with_height else 'RF_only',
            original_pivot_Z_working_m=original_height,candidate_pivot_Z_working_m=height,
            initial_J_rank=int(np.linalg.matrix_rank(initial_j)),initial_J_singular_values=singular.tolist(),
            initial_J_condition_number=float(singular[0]/singular[-1]),
            actual_initial_J_centered_FD_comparison=derivative_comparison,optimizer_jacobian_mode=jacobian_mode,
            cost_before=float(evaluate(initial,with_height)@evaluate(initial,with_height)/2),
            cost_after=float(evaluate(solved.x,with_height)@evaluate(solved.x,with_height)/2),
            actual_evaluations=int(solved.nfev),optimizer_status=int(solved.status),source_frames=diagnostics))
    selected = solutions[-1]
    indices,before,after = [pivot_z_index],[original_height],[selected['candidate_pivot_Z_working_m']]
    for key,frame in zip(keys,selected['source_frames']):
        old = state['q'][list(key[2])]
        candidate = np.asarray(frame['candidate_ViewState_RF4']).copy()
        candidate[3] += old[3]-camera['view_state4'][key[1],3]
        indices.extend(key[2]);before.extend(old);after.extend(candidate)
    np.savez_compressed(delta_path,parameter_indices=np.asarray(indices,np.int64),
        original_values=np.asarray(before,np.float64),candidate_values=np.asarray(after,np.float64))
    report=dict(schema='rinkspace.source-qualified-camera-height-initializer-diagnostic.v1',role='initializer_only',
        parent_q_sha256=_q_sha(state['q']),parent_state=dict(path=str(Path(state_path).resolve()),sha256=_sha(state_path)),
        camera_export=dict(path=str(Path(camera_path).resolve()),sha256=_sha(camera_path)),scene=dict(path=str(Path(scene_path).resolve()),sha256=_sha(scene_path)),
        source_packets=packets,source_video_sha256=metadata['source_sha256'],time_base=metadata['source_time_base'],
        installation_id=identity,existing_shared_pivot_Z_parameter_index=pivot_z_index,
        original_installation_xyz_parameter_indices=pivot_indices.tolist(),scene_unit_working_m=scene_unit_working_m,
        working_scale_and_focal_bounds_are_not_measured_hardware_calibration=True,
        pivot_Pxy_lever_PP_Brown_and_shared_world_unchanged=True,shared_Pz_affects_all_same_installation_exposures=True,
        source_normals_only_roundoff_normalized_original_packet_UV_intervals_weights_unchanged=True,
        original_evaluation_ids_explicitly_retained_in_objective=sorted(retained_ids),
        existing_source_J_conditioned_camera_chart_converted_to_actual_camera25_units=True,
        delta_npz=dict(path=str(delta_path.resolve()),sha256=_sha(delta_path)),solutions=solutions,
        source_shot2_and_whole_shot6_counterfactual_validation_pending=True,accuracy_accepted=False,
        root_existing_RF_only_adapter_cannot_apply_this_shared_Pz_candidate=True,full_solver_started=False)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
