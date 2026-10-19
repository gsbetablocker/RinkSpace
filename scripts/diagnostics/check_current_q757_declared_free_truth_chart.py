"""Actual current 86-group producer preflight; no full graph, source J or QP."""
from pathlib import Path
import os,sys,time,hashlib,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/diagnostics'),str(ROOT),str(ROOT/'src')]
import audit_current_user_ice_truth as a
import numpy as np
from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart
from rebuild_captured_QP_all_truth_faces import face_map
from calibration.solver_runtime.original_halfspace_primal import original_full_qp_kkt_verified

HERE=a.HERE
Q='75719b65139fa662366146c4517312319bdad9a88e6f2c25704966f550cb1eee'
BASE=a.RUNS/'okada_review86_joint_curve_integration_20261010_2342'
SEED=BASE/'accepted_states'/Q
CAP=a.RUNS/'required_QP_capture_q757_20261011_0100/required_reduced_QP'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_same757_board185_constant_axes_20261011.json'

def emit(**v):print(json.dumps(v,ensure_ascii=False,allow_nan=False),flush=True)
def signatures(factors):
    result=[]
    inventories=[('native_primary',factors['native'].records),('native_full',factors['native'].full_stroke_records),
                 ('wide',factors['wide'].records),('confirmed',factors['confirmed'].records)]
    for family,records in inventories:
        for ordinal,row in enumerate(records):
            result.append(dict(family=family,index=ordinal,fields={key:dict(shape=list(np.shape(row[key])),
                dtype=np.asarray(row[key]).dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(row[key]).tobytes()).hexdigest())
                for key in ('uv','normals','bounds','tangent_bounds','original_raster_normal_bounds',
                            'original_source_tangent_bounds','radius')if key in row and row[key]is not None}))
    return result

def main():
    started=time.perf_counter();options=a.read(CONFIG)['source_options']
    state,q,identity,camera_error=a.load_state(SEED,Path(options['--world-init']),a.read(BASE/'input_manifest.json'))
    assert camera_error==0.;identity['_accepted_directory']=str(SEED);q_before=q.tobytes()
    board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q))
    ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=a.bind_human(state,board,ice,options,len(q))
    wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    old71=a.read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle,material,placement=a.existing_LOW_source_owners(state,q,SEED,old71,a.read(SEED/'scene_UNACCEPTED.json'))
    confirmed=a.UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,
        parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    factors=dict(native=human,wide=wide,confirmed=confirmed);before=signatures(factors)
    scope=a.read(options['--truth-restoration-reviewed-manual-append-audit'])
    capvalues=a.read(scope['exact_caps_values']['path'])
    caps={f:np.asarray(capvalues['families'][f]['limits'],float)for f in factors}
    actual=a.values(human,wide,confirmed,q)
    for f in factors:
        np.testing.assert_array_equal(actual['buffers'][f],capvalues['families'][f]['violations_px'])
        assert np.max(actual['buffers'][f]-caps[f])<=1e-8
    with np.load(CAP/'all_truth_constant_axes_original_domain.npz',allow_pickle=False)as z:
        oldA=z['constraint_matrix'];oldb=z['original_constraint_lower'];oldown=z['row_group_index']
        requested=z['global_parameter_indices'];scales=z['global_parameter_scales'];captured_v=z['current_group_violations']
    np.testing.assert_array_equal(np.concatenate([actual['buffers'][f]for f in factors]),captured_v)
    fixed=np.array([2,34126,34127,34128,34129]);keep=~np.isin(requested,fixed)
    selected=requested[keep];layout=state.canonical_parameter_layout
    evalcols=layout.canonical_to_evaluation[selected];fixed_eval=layout.canonical_to_evaluation[fixed]
    prefix=int(np.flatnonzero(oldown>=0)[0]);cursor=prefix;offset=0;results=[]
    emit(stage='actual_current86_owner_loaded',seconds=time.perf_counter()-started,requested_columns=len(requested),effective_columns=len(selected))
    for family,factor in factors.items():
        t=time.perf_counter();kwargs=dict(group_limits=caps[family],return_step_carrier_response=True)
        if family=='native':kwargs.update(full_stroke=True,endpoint_inward_restoration=False)
        produced=constraints_on_declared_free_chart(factor,q,evalcols,fixed_eval,**kwargs)
        own,added=face_map(family,factor,produced['step_carrier_response'],offset,caps[family]);offset+=len(caps[family])
        stop=cursor+len(produced['lower']);A=produced['matrix']*scales[keep];b=produced['lower']+added
        np.testing.assert_array_equal(A,oldA[cursor:stop][:,keep])
        np.testing.assert_array_equal(b,oldb[cursor:stop]);np.testing.assert_array_equal(own,oldown[cursor:stop])
        pullback=produced['report'].get('declared_fixed_tangent_pullback')
        if pullback is not None:assert pullback['original_fixed_derivative_nonzero_count']==0
        results.append(dict(family=family,rows=len(b),columns=A.shape[1],seconds=time.perf_counter()-t,
            all_original_faces_targets_ownership_bitexact=True,declared_fixed_tangent_pullback=pullback))
        emit(stage='actual_current85_chart_producer_passed',**results[-1]);cursor=stop
        # Missing a genuine active nonfixed column must still fail BEFORE the
        # producer call, rather than silently treating a dependency as fixed.
        dependencies=np.intersect1d(factor.active,evalcols)
        assert len(dependencies)>0
        try:constraints_on_declared_free_chart(factor,q,evalcols[evalcols!=dependencies[0]],fixed_eval,**kwargs)
        except ValueError as error:assert 'nonfixed original truth dependency' in str(error)
        else:raise AssertionError('Unfixed original truth dependency was silently omitted')
    assert cursor==len(oldb) and signatures(factors)==before and q.tobytes()==q_before
    assert np.count_nonzero(oldA[:,~keep])==0
    # All production initial/SOC consumers share the same closure and hints
    # carry the changed 285-column shape. Reuse the already verified real QP,
    # rather than running a second solve to check its certificate consumers.
    replay=CAP/'constant_axes_restricted_trace_original_settings__declared_fixed_free_chart'
    kkt=a.read(replay/'verified_report.json')
    with np.load(replay/'verified_direction.npz',allow_pickle=False)as z:
        emit(stage='real_saved_QP_direction_fields',fields=z.files)
        direction=z['step'] if 'step'in z.files else z['full_step']
    assert original_full_qp_kkt_verified(kkt,direction)
    output=HERE/'current_q757_full86_declared_fixed285_actual_production_preflight_20261011.json'
    report=dict(schema='rinkspace.current-full86-declared-fixed-shared-chart-production-preflight.v1',passed=True,
        complete_q_sha256=Q,configuration=a.ref(CONFIG),source_manifest=a.ref(ROOT/'calibration/solver_runtime/source_manifest.json'),
        complete_state_reference=a.ref(SEED/'complete_state_UNACCEPTED.npz'),camera_archive_error=camera_error,
        current_protected_source_options={k:a.ref(options[k])for k in ('--additional-board-source','--additional-ice-source',
            '--direct-human-truth-authority','--direct-human-full-stroke-guard','--direct-human-stroke-cap-support',
            '--practical-human-location-envelope','--user-confirmed-source-truth-authority')},
        requested_canonical_columns=requested.tolist(),effective_canonical_columns=selected.tolist(),
        explicitly_fixed_omitted_canonical_columns=requested[~keep].tolist(),
        effective_evaluation_columns=evalcols.tolist(),full_parameters=len(q),truth_groups=offset,
        original_physical_prefix_rows=prefix,all_original_rows=len(oldb),family_producers=results,
        source_arrays_signatures=before,original_guard_values_supports_caps_and_arrays_bitexact=True,
        original_A_omitted_fixed_column_bitexact_zero=True,all_original_A_free_columns_targets_ownership_bitexact=True,
        missing_nonfixed_dependencies_rejected_for_all_three_families=True,
        actual_real_285_required_QP_full_certificate=a.ref(replay/'verified_report.json'),
        original_full_QP_KKT_consumer_accepts_verified_generic_certificate=True,
        physical_prefix_not_recomputed_here=True,
        no_full_source_graph_source_J_QP_or_candidate_produced=True,seconds=time.perf_counter()-started)
    a.save(output,report);emit(stage='actual_current_full86_fixed285_preflight_PASS',report=a.ref(output),seconds=report['seconds'])

if __name__=='__main__':main()
