"""Saved-direction, value-only certification of native local station charts.

No source objective/J, solve, frame decoding or image is constructed. The
original native54 source arrays, sampling and saved group budgets are retained.
"""
from pathlib import Path
import hashlib
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = [
    'calibration/solver_runtime/finite_rectangle_station_step_constraints.py',
    'calibration/solver_runtime/finite_rectangle_station_faces.py',
    'calibration/solver_runtime/finite_rectangle_projection_fd.py',
    'calibration/solver_runtime/native_truth_nominal_chart.py',
    'calibration/solver_runtime/legacy/structured_programme_source_candidate/source_step_curvature_tightening.py',
    'calibration/solver_runtime/legacy/run_current_named66_full_programme.py',
]
IMPLEMENTATION_BEFORE_IMPORT = [dict(path=str((_ROOT/p).resolve()),
    sha256=hashlib.sha256((_ROOT/p).read_bytes()).hexdigest()) for p in _IMPLEMENTATION_FILES]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_saved_curved_truth_carriers_20261011 as old
common = old.common
np = common.np
from calibration.solver_runtime import finite_rectangle_station_step_constraints as station
from calibration.solver_runtime import finite_rectangle_projection_fd as projection
from calibration.solver_runtime.finite_rectangle_station_faces import finite_rectangle_station_halfspaces

OUT = old.OUT.with_name('repaired_native54_station_nominal_value_audit_20261011.json')


def verify_implementation_unchanged():
    current = [common.ref(r['path']) for r in IMPLEMENTATION_BEFORE_IMPORT]
    if current != IMPLEMENTATION_BEFORE_IMPORT:
        raise RuntimeError('Runtime implementation changed during saved-direction validation')
    return current


def native_source_array_digest(human):
    digest = hashlib.sha256()
    for record in human.full_stroke_records:
        digest.update(record['row']['id'].encode('utf8'))
        for name in ('uv','normals','bounds','tangent_bounds'):
            array = np.ascontiguousarray(record[name])
            digest.update(name.encode('ascii'))
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.dtype.str.encode('ascii'))
            digest.update(array.tobytes())
    return digest.hexdigest()


def load_saved():
    options = common.read(old.CONFIG)['source_options']
    parent_dir = common.RUN / 'accepted_states' / old.PARENT
    identity = common.read(parent_dir / 'state_identity.json')
    path = parent_dir / 'complete_state_UNACCEPTED.npz'
    assert common.sha(path) == identity['state_npz_sha256']
    assert common.sha(identity['camera_path']) == identity['camera_sha256']
    keys = ('q', 'aux__canonical_pts', 'aux__block_lengths', 'aux__block_start', 'aux__block_names',
        'aux__installation_xyz_slots', 'aux__current_shared_mount_parameter_indices',
        'aux__current_all_installation_optics_parameter_indices', 'aux__correlated_roll_prior_contract_json',
        'aux__current_complete_parameter_layout_json')
    with np.load(path, allow_pickle=False) as archive:
        small = {k: archive[k].copy() for k in keys}
    provider, world, proof = common.current_complete_source_geometry(path, options['--world-init'])
    with np.load(identity['camera_path'], allow_pickle=False) as archive:
        cameras = {k: archive[k].copy() for k in archive.files}
    state, error = common.state_camera_adapter(small, cameras, provider)
    assert error == 0.
    manifest = common.read(old.RUN / 'input_manifest.json')
    state.current_user_pivot_outside_constraint_contract = manifest['camera_pivot_exterior_constraints']
    scope = manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope is not None:
        state.current_user_physical_course_scope_source = scope['path']
    board = common.NativeBoardEdgeFactor(state, options['--additional-board-source'], len(small['q']))
    ice = common.NativeIceCurveFactor(state, options['--additional-ice-source'], len(small['q']))
    human = common.HumanStrokeTruth(state, options['--direct-human-truth-authority'], board, ice,
                                   parameter_count=len(small['q']))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    files = list((old.RUN / 'actual_complete_source_directions').glob(old.PARENT + '_*.npz'))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as archive:
        q, step = archive['q'].copy(), archive['step'].copy()
    assert common.asha(q) == old.PARENT and step.shape == q.shape == (34340,)
    eq, ed = state.evaluation_q(q), state.evaluation_q(step)
    child_path = old.RUN / 'accepted_states' / old.CHILD / 'complete_state_UNACCEPTED.npz'
    child_identity = common.read(child_path.parent / 'state_identity.json')
    assert common.sha(child_path) == child_identity['state_npz_sha256']
    with np.load(child_path, allow_pickle=False) as archive:
        child = archive['q'].copy()
    accepted, retraction = common.retract(state, eq + .5 * ed)
    np.testing.assert_array_equal(state.canonical_parameter_layout.canonical_q(accepted), child)
    caps_paths = [parent_dir / 'truth_restoration_state.json', child_path.parent / 'truth_restoration_state.json']
    caps = []
    for cap_path in caps_paths:
        saved = common.read(cap_path)
        native = saved['families']['native']
        assert native['group_ids'] == [r['row']['id'] for r in human.full_stroke_records]
        caps.append(np.asarray(native['limits'], float))
    references = dict(parent_state=common.ref(path), parent_identity=common.ref(parent_dir/'state_identity.json'),
        parent_camera=common.ref(identity['camera_path']), child_state=common.ref(child_path),
        child_identity=common.ref(child_path.parent/'state_identity.json'), saved_direction=common.ref(files[0]),
        saved_budgets=[common.ref(p) for p in caps_paths], config=common.ref(old.CONFIG),
        sources=[common.ref(options[k]) for k in ('--additional-board-source','--additional-ice-source',
            '--direct-human-truth-authority','--direct-human-full-stroke-guard',
            '--direct-human-stroke-cap-support','--practical-human-location-envelope')])
    return human, eq, ed, accepted, caps, references


def original_query_maxima(queries):
    return np.array([float(abs(r['query']['residual']).max()) for r in queries])


def nominal_value_check(human, value, caps, label):
    queries = human.queries(value, full_stroke=True)
    contexts = station.nominal_station_contexts(human, value, full_queries=queries)
    projections = projection.projected_station_contexts(human, value, contexts, np.empty(0, int))
    assert projections['report']['real_FD_evaluations'] == 0
    count = len(contexts)
    s = projections['nominal_physical_station']
    cone = np.tile([-np.inf, np.inf], (count, 1))
    sides = [c.get('nominal_physical_endpoint_side') for c in contexts]
    if not any('nominal_physical_endpoint_side' in c for c in contexts):
        raise RuntimeError('Value audit requires category-certified true endpoint contexts')
    cone[np.array([side == 'lower' for side in sides]), 0] = 0.
    cone[np.array([side == 'upper' for side in sides]), 1] = 0.
    # An active original endpoint category can have an interior one-ulp t.
    # Its physical station coordinate must be the true global endpoint.
    s = s.copy()
    s[np.array([side == 'lower' for side in sides])] = 0.
    s[np.array([side == 'upper' for side in sides])] = 1.
    uv = np.stack([c['record']['uv'][c['source_vertex']] for c in contexts])
    normals = np.stack([c['record']['normals'][c['source_vertex']] for c in contexts])
    nb = np.stack([c['record']['bounds'][c['source_vertex']] for c in contexts])
    tb = np.stack([c['record']['tangent_bounds'][c['source_vertex']] for c in contexts])
    local_caps = caps[[c['group'] for c in contexts]]
    # Pure zero placeholder chart is not a derivative: nominal values depend
    # only on the original foot, tangent, support and tangent cone.
    model = finite_rectangle_station_halfspaces(projections['projected_foot_xy'], np.zeros((count,2,1)),
        projections['station_screen_tangent'], s, np.tile([0.,1.],(count,1)), uv, normals, nb, tb,
        local_caps, station_tangent_cone=cone)
    fixed = station.actual_station_model_carriers(human, value,
        dict(finite_station_contexts=contexts, original_native_carrier_count=sum(2*len(r['uv']) for r in human.full_stroke_records)))
    fixed = fixed[-2*count:].reshape(count,2)
    by_group = {}
    mismatch, allowance_excess, model_max = [], [], np.full(count, -np.inf)
    for face, provenance in enumerate(model['face_provenance']):
        point = provenance['source_point_index']
        context = contexts[point]
        actual = float(abs(queries[context['group']]['query']['residual'].reshape(-1,2)[context['source_vertex']]).max())
        if provenance['kind'] == 'nonnegative_epigraph':
            model_max[point] = max(model_max[point], 0.)
            continue
        coefficients, support = [], 0.
        parents = provenance['original_source_faces']
        weights = provenance.get('source_face_weights', [1.])
        terms = []
        for weight, parent in zip(weights, parents):
            axis = 0 if parent['axis'] == 'normal' else 1
            sign = -1. if parent['side'] == 'lower' else 1.
            bounds = nb if axis == 0 else tb
            terms.append(weight*sign*fixed[point,axis])
            support += weight*sign*bounds[point,0 if sign < 0 else 1]
        face_value = float(sum(terms)-support)
        bound = station._verify_nominal_face(face_value, actual, context=context, provenance=provenance,
            operands=np.r_[terms,support,local_caps[point],uv[point],projections['projected_foot_xy'][point]])
        mismatch.append(abs(face_value-provenance['local_epigraph_face_at_nominal']))
        allowance_excess.append(face_value-actual-bound)
        model_max[point] = max(model_max[point], face_value)
    for point, context in enumerate(contexts):
        g = context['group']
        raw = float(abs(queries[g]['query']['residual'].reshape(-1,2)[context['source_vertex']]).max())
        entry = by_group.setdefault(g, dict(group=g,id=queries[g]['id'],vertices=0,maximum_local_face_minus_original_vertex= -np.inf,
            original_query_maximum=0.,maximum_local_face_value=0.,true_lower_endpoint_vertices=0,true_upper_endpoint_vertices=0))
        entry['vertices'] += 1
        entry['maximum_local_face_minus_original_vertex'] = max(entry['maximum_local_face_minus_original_vertex'],float(model_max[point]-raw))
        entry['original_query_maximum'] = max(entry['original_query_maximum'],raw)
        entry['maximum_local_face_value'] = max(entry['maximum_local_face_value'],float(model_max[point]))
        entry['true_lower_endpoint_vertices'] += int(sides[point] == 'lower')
        entry['true_upper_endpoint_vertices'] += int(sides[point] == 'upper')
    actual_max = original_query_maxima(queries)
    return dict(label=label,native_groups=len(queries),certified_contexts=count,local_faces=len(model['lower']),
        converted_groups=len(by_group),real_projection_FD_evaluations=0,nominal_value_only=True,
        maximum_nominal_response_vs_model_formula_px=max(mismatch),
        maximum_face_minus_original_vertex_minus_arithmetic_allowance_px=max(allowance_excess),
        all_local_faces_verified_against_original_vertex_without_budget=True,
        original_dynamic_guard_excess_over_saved_budget=np.maximum(actual_max-caps,0.),
        local_group_inventory=list(by_group.values()),
        sampling_knots_have_no_station_cap=True,distant_endpoint_projection_never_replaces_nominal_chart=True)


def main():
    began = time.perf_counter()
    human, eq, ed, accepted, caps, refs = load_saved()
    original_arrays_sha = native_source_array_digest(human)
    nominal = [nominal_value_check(human, eq, caps[0], 'q276'),
               nominal_value_check(human, accepted, caps[1], 'q8a15')]
    original_guards = []
    for label, alpha, value in [('saved_full',1.,eq+ed),('saved_half_before_retraction',.5,eq+.5*ed),
                                ('accepted_half_after_retraction',.5,accepted)]:
        queries = human.queries(value,full_stroke=True)
        maximum = original_query_maxima(queries)
        original_guards.append(dict(label=label,alpha=alpha,original_dynamic_query_maxima=maximum,
            parent_saved_budgets=caps[0],excess_over_parent_saved_budgets=np.maximum(maximum-caps[0],0.),
            failing_groups=np.flatnonzero(maximum-caps[0]>1e-8)))
    verified_implementation = verify_implementation_unchanged()
    assert native_source_array_digest(human) == original_arrays_sha
    report = dict(schema='rinkspace.repaired-native54-nominal-station-value-audit.v1',
        script=common.ref(__file__),implementation=verified_implementation,
        implementation_captured_before_import=IMPLEMENTATION_BEFORE_IMPORT,
        implementation_import_to_report_SHA_unchanged=True,
        references=refs,previous_frozen_dynamic_evidence=common.ref(old.OUT),
        new_fit=False,source_or_truth_J_constructed=False,directional_FD_constructed=False,
        source_images_written=0,source_frames_decoded=0,original_support_budgets_sampling_unchanged=True,
        original_native_source_arrays_before_after_sha256=original_arrays_sha,
        original_native_source_arrays_before_after_bitexact=True,
        q276_half_direction_reproduces_saved_q8_bit_for_bit=True,nominal_checks=nominal,
        original_guard_controls=original_guards,
        limits=['Value-only nominal consistency and original guard replay; this report alone does not validate all chart Jacobians or driver SOC gating.',
            'A local tangent face at a trial is not required to equal the original dynamic guard; the original guard controls acceptance.'],
        seconds=time.perf_counter()-began)
    OUT.write_text(common.json.dumps(report,default=common.plain,ensure_ascii=False,indent=2),encoding='utf8')
    print(common.json.dumps(dict(path=str(OUT),sha256=common.sha(OUT),bytes=OUT.stat().st_size,seconds=report['seconds'])))


if __name__ == '__main__':
    main()
