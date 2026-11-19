"""Import and verify a scoped repair as a new objective on the full state."""
import json
from pathlib import Path

import numpy as np

from .absolute_motion_curve import _log
from .absolute_motion_state import curve_from_packet, evaluate_curve
from .body_clock import time_base
from .rebuild_prepare import digest, load
from .sequence_skate_state import SequenceState, shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state
from .weighted_blocks import signature


def import_scoped_candidate(reference, camera, prepared, document, specification, parent_path):
    """Verify the old explicit geometry and its historical x before rebasing."""
    parent = load(parent_path)
    for key, path in (('parent_sha256', parent_path),
                      ('parent_evidence_sha256', specification['parent_evidence']),
                      ('model_sha256', specification['model_path'])):
        if digest(Path(path)) != specification[key]:
            raise ValueError('Scoped import changed ' + key)
    if specification['model_sha256'] != reference.hashes['model']:
        raise ValueError('Scoped import model differs from current prepared model')
    expected = reference.fixed_checkpoint_identity
    transition = specification.get('registered_camera_transition')
    if transition:
        old_camera = json.loads(str(parent['camera_binding_json']))
        new_camera = camera.info
        if (old_camera['state_sha256'] != transition['parent_camera_sha256']
                or new_camera['state_sha256'] != transition['camera_sha256']
                or new_camera.get('scene_values_sha256') != transition['scene_values_sha256']):
            raise ValueError('Scoped import camera transition differs from its registered bindings')
        for key in ('metres_per_scene_unit', 'solver_metres_per_scene_unit', 'length_export_ratio'):
            if not np.isclose(old_camera[key], new_camera[key], atol=1e-13, rtol=0.):
                raise ValueError('Scoped camera transition changed world units: '+key)
        expected = dict(expected, camera_binding_json=parent['camera_binding_json'])
    validate_checkpoint_identity(parent, expected)
    from .programme_skate_evidence import programme_observation_path
    parent_observations = programme_observation_path(prepared, specification['parent_evidence'])
    for key, path in (('prepared_input_sha256', prepared / 'input.npz'),
                      ('prepared_observations_sha256', parent_observations)):
        if str(parent[key]) != digest(path):
            raise ValueError('Scoped import changed ' + key)
    if str(parent['source_evidence_sha256']) != specification['parent_evidence_sha256']:
        raise ValueError('Parent checkpoint and declared source ledger differ')
    pts = reference.p['pts']
    np.testing.assert_array_equal(parent['action_objective_source_pts'], pts)
    if list(time_base(reference.p)) != specification['source_time_base']:
        raise ValueError('Scoped import source time base changed')
    if document['source_sha256'] != specification['source_video_sha256']:
        raise ValueError('Scoped import source video changed')
    if not getattr(reference, 'fixed_skate_mount', False):
        raise ValueError('Scoped import requires a fixed ankle mount')
    np.testing.assert_array_equal(parent['x'][:17], reference.binding.x)
    curve = curve_from_packet(parent)
    shoe = shoe_curve_from_packet(parent)
    np.testing.assert_array_equal(shoe.rotation_controls,
                                  np.broadcast_to(np.eye(3), shoe.rotation_controls.shape))
    reference.absolute_curve = curve_from_packet(parent, seed=True)
    reference.relative_skate_curve = shoe_curve_from_packet(parent, seed=True)
    state = SequenceState(reference, parent['seconds'])
    previous = state.current_curve(parent['x'][17:])
    previous_shoe = state.shoe_current_curve(parent['x'][17:])
    np.testing.assert_allclose(previous.position_controls, curve.position_controls, atol=2e-12, rtol=0)
    np.testing.assert_allclose(previous.rotation_controls, curve.rotation_controls, atol=2e-11, rtol=0)
    np.testing.assert_allclose(previous_shoe.rotation_controls, shoe.rotation_controls, atol=2e-11, rtol=0)
    times = reference.times
    n = len(times)
    rows = np.arange(n)
    tau = camera.readout_times(rows)
    shutter = times[:, None] + tau[:, None] * np.linspace(-.5, .5, 5)
    clocks = (times, shutter.ravel(), .5*(times[1:] + times[:-1]))
    maxima = dict(root_m=0., rotations=0., fk_m=0., skate_rotation=0., skate_origin_m=0.)
    for clock in clocks:
        for i in range(0, len(clock), 512):
            t = clock[i:i+512]
            a, b = evaluate_curve(curve, t), evaluate_curve(previous, t)
            for pose in (a, b):
                pose['skate_relative_R'] = evaluate_curve(shoe, t)['rotations']
            fa, fb = (fk_state(reference, pose, np.zeros((len(t), 69))) for pose in (a, b))
            for label, key, lhs, rhs, scale in (
                ('root_m','P',a,b,reference.binding.ratio),
                ('rotations','rotations',a,b,1.),
                ('fk_m','native_world',fa,fb,reference.binding.ratio),
                ('skate_rotation','R',fa,fb,1.),
                ('skate_origin_m','O',fa,fb,reference.binding.ratio),
            ):
                maxima[label] = max(maxima[label], float(np.max(np.abs(lhs[key]-rhs[key]))*scale))
    if max(maxima.values()) > 3e-10:
        raise ValueError(f'Scoped import is not a zero-update replay: {maxima}')
    preserve_chart = specification.get('retain_parent_objective_context', False)
    if preserve_chart:
        if transition or parent_observations == programme_observation_path(prepared, document):
            raise ValueError('An observation-only refit needs a new ledger and the original camera')
        x = parent['x'].copy()
    else:
        # A scoped geometry import normally uses the saved curve as the new chart.
        reference.absolute_curve = curve
        reference.relative_skate_curve = shoe
        x = np.r_[parent['x'][:17], np.zeros(state.size)]
    receipt = dict(schema='rinkspace.scoped-candidate-import.v1', parent=str(parent_path),
        parent_sha256=specification['parent_sha256'], historical_objective_reused=False,
        parent_tangent_chart_retained=preserve_chart,
        source_exposures=n, shutter_nodes=int(shutter.size), midpoint_samples=n-1,
        source_pts=pts[[0,-1]].tolist(), source_time_base=list(time_base(reference.p)),
        prepared_input_sha256=str(parent['prepared_input_sha256']),
        observations_sha256=str(parent['prepared_observations_sha256']),
        camera_sha256=camera.info['state_sha256'], model_sha256=reference.hashes['model'],
        source_video_sha256=document['source_sha256'], max_replay_difference=maxima)
    if transition:
        receipt['registered_camera_transition'] = transition
    return parent, x, receipt


def _import_projection_difference(problem, expected_result):
    expected_result = Path(expected_result)
    body = problem.body
    with np.load(expected_result, allow_pickle=False) as packet:
        np.testing.assert_array_equal(packet['source_pts'], problem.reference.p['pts'])
        expected = packet['COCO_uv']
        actual = body.cache['uv'][:, body.layout.body_landmarks]
        # Objective packets may omit inadmissible rows. Those rows are not
        # represented as zero-error comparisons.
        good = np.isfinite(actual).all(axis=-1) & np.isfinite(expected).all(axis=-1)
        if not good.any():
            raise ValueError('No shared native body projection to verify')
        return np.abs(actual[good]-expected[good])


def verify_import_projection(problem, receipt, expected_result):
    """Require identical source pixels when the camera was kept."""
    difference = _import_projection_difference(problem, expected_result)
    peak = float(difference.max())
    if peak > 3e-7:
        raise ValueError(f'Scoped import changed native body projection by {peak} px')
    receipt.update(compared_coco_rows=len(difference), max_coco_difference_px=peak)


def record_import_reprojection(problem, receipt, expected_result):
    """Measure the native pixel change from the registered camera alone."""
    d = _import_projection_difference(problem, expected_result)
    receipt.update(compared_coco_rows=len(d), max_coco_difference_px=float(d.max()),
                   camera_only_coco_pixel_delta_p50_p95_max=np.percentile(np.linalg.norm(d,axis=1),[50,95,100]).tolist())


def require_full_repair_rows(problem, value, *, jacobian=False):
    """Verify assembled rows and selected columns of the actual main objective."""
    expected = ('body_pixels', 'dynamic_foot_compatibility', 'cavity_AL', 'finite_ice',
                'surface_intersection', 'anatomical_joint_domain', 'manual_native_leg',
                'ordered_native_skate', 'reviewed_native_support', 'whole_skate_source',
                'directed_boot_source', 'support_shoe_phase', 'programme_parent_root_trust',
                'source_supported_grasp', 'actual_so3_angular_acceleration',
                'actual_model_com_acceleration', 'programme_root_curvature',
                'programme_component_ice')
    if problem.reference.source_document['scoped_candidate_import'].get('opening_lunge'):
        from .programme_repair_factors import ProgrammeOpeningLunge
        expected += (ProgrammeOpeningLunge.support_name, ProgrammeOpeningLunge.lead_name)
    blocks = {block.name:block for block in value['weighted_blocks']}
    missing = [name for name in expected if name not in blocks or not len(blocks[name].residual)]
    flights = [name for name in blocks if name.endswith('/flight_integrated_com')]
    dense = [name for name in blocks if name.endswith('/flight_dense_com_acceleration')]
    if problem.reference.source_document.get('com_motion_policy',{}).get('confirmed_flight_acceleration_scale_mps2') is not None:
        episodes = problem.reference.source_document['source_episode_review']['episodes']
        required = {'episode/'+e['episode_id']+'/flight_dense_com_acceleration'
                    for e in episodes if e['kind']=='flight_visible_candidate'}
        if set(dense) != required or any(not len(blocks[name].residual) for name in dense):
            raise ValueError('Dense flight rows do not cover the declared source flights')
    if missing or not flights:
        raise ValueError(f'Incomplete revised programme objective: {missing}, flights={flights}')
    selection = problem.active_parameter_indices
    out = {}
    for name, block in blocks.items():
        if name not in expected and name not in flights and name not in dense and not name.endswith('/flight_specific_momentum'):
            continue
        entry = dict(rows=len(block.residual), weighted_cost=float(block.residual@block.residual/2),
                     source_association=block.association)
        if jacobian and block.jacobian is not None:
            reduced = block.jacobian[:, selection].tocsc()
            entry['active_columns_with_derivatives'] = int(np.count_nonzero(np.diff(reduced.indptr)))
        out[name] = entry
    return out


def proposal_vector(problem, path, parent_sha256, parent_vector=None):
    """Convert a proposal's explicit geometry into the main stage's chart."""
    packet = load(path)
    parent_key = ('scoped_import_parent_sha256' if parent_vector is not None
                  else 'scoped_repair_parent_sha256')
    if str(packet[parent_key]) != parent_sha256:
        raise ValueError('Proposal was generated against a different parent')
    validate_checkpoint_identity(packet, problem.reference.fixed_checkpoint_identity)
    state = problem.body.state
    candidate = curve_from_packet(packet)
    np.testing.assert_array_equal(candidate.knots, state.knots)
    np.testing.assert_array_equal(packet['x'][:17], problem.binding.x)
    np.testing.assert_array_equal(shoe_curve_from_packet(packet).rotation_controls,
                                  state.shoe_curve.rotation_controls)
    if parent_vector is not None:
        np.testing.assert_array_equal(packet['scoped_parent_parameter_vector'],parent_vector)
        if str(packet['prepared_input_sha256']) != str(problem.checkpoint_extra['prepared_input_sha256']):
            raise ValueError('Same-chart proposal changed the prepared geometry')
        x = packet['x'].copy()
        if x.shape != parent_vector.shape or not np.isfinite(x).all():
            raise ValueError('Same-chart proposal has an invalid parameter layout')
        for current, expected in ((state.current_curve(x[17:]),candidate),
                                  (state.shoe_current_curve(x[17:]),shoe_curve_from_packet(packet))):
            np.testing.assert_allclose(current.position_controls,expected.position_controls,atol=2e-12,rtol=0)
            np.testing.assert_allclose(current.rotation_controls,expected.rotation_controls,atol=2e-11,rtol=0)
            np.testing.assert_allclose(current._check_log_chart(),expected._check_log_chart(),atol=3e-10,rtol=0)
        return x
    dr = _log(state.curve.rotation_controls.swapaxes(-1,-2) @ candidate.rotation_controls)
    unchanged = np.all(state.curve.rotation_controls==candidate.rotation_controls,axis=(-2,-1))
    dr[unchanged]=0.
    dp = candidate.position_controls-state.curve.position_controls
    local = np.c_[dp,dr.reshape(len(dp),-1)]
    x = np.r_[packet['x'][:17],local.ravel(),np.zeros(state.shoe_size)]
    got = state.current_curve(x[17:])
    np.testing.assert_allclose(got.rotation_controls,candidate.rotation_controls,atol=3e-11,rtol=0)
    # The central rotations alone do not validate a lifted edge branch.
    np.testing.assert_allclose(got._check_log_chart(),candidate._check_log_chart(),atol=3e-10,rtol=0)
    return x


def compare_objective_replay(old, registry, blocks, declared_changes):
    """Check same-pose rows by name across a declared objective revision."""
    offset = 0
    previous = {}
    for entry in registry:
        name, start, stop = (entry[k] for k in ('name', 'start', 'stop'))
        if name in previous or start != offset or not start <= stop <= len(old):
            raise ValueError('Invalid parent weighted row registry: '+name)
        previous[name] = old[start:stop]
        offset = stop
    if offset != len(old):
        raise ValueError('Parent registry does not cover the weighted residual')
    current = {entry.name:entry.residual for entry in blocks}
    if len(current) != len(blocks):
        raise ValueError('Duplicate current weighted row names')
    all_names = previous.keys() | current.keys()
    if declared_changes - all_names:
        raise ValueError('Declared revised rows were not consumed')
    changed, unchanged, maximum = {}, 0, 0.
    for name in sorted(all_names):
        if name in declared_changes:
            a = previous.get(name, np.empty(0)); b = current.get(name, np.empty(0))
            changed[name] = dict(before_rows=len(a),after_rows=len(b),
                                 before_cost=float(a@a/2),after_cost=float(b@b/2))
            continue
        if name not in current or name not in previous:
            raise ValueError('Undeclared weighted row population change: '+name)
        a, b = previous[name], current[name]
        if a.shape != b.shape or not np.allclose(a,b,rtol=0.,atol=2.e-8):
            raise ValueError('Undeclared same-pose weighted residual change: '+name)
        unchanged += len(a)
        maximum = max(maximum,float(np.max(abs(a-b),initial=0.)))
    return dict(changes=changed,unchanged_rows=unchanged,unchanged_maximum_abs_difference=maximum)


def weighted_residual_identity(value):
    """Persist the actual weighted row vector and a comparison digest."""
    r=np.ascontiguousarray(value['residual'])
    return dict(scoped_weighted_residual_sha256=np.array(signature(r)),
                scoped_weighted_residual=r.copy())
