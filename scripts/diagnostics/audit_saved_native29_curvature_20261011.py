"""Read-only original finite-support audit of already saved complete steps.

No solve, source graph, complete source J, source images or fitted state is
created. Small native29 carrier derivatives use the public guard producer.
"""
from pathlib import Path
import os, sys, json, hashlib, copy, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'scripts/diagnostics')]
sys.dont_write_bytecode=True
import numpy as np
from calibration.solver_runtime import install, configure_native_runtime
install()
configure_native_runtime(stage='readonly_saved_native29_curvature_audit',native_threads=1)
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,finite_stroke_queries
from structured_programme_source_candidate.native_camera_local import native_camera_local
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from structured_programme_source_candidate.human_constraint_dependency_fd import group_dependencies
from installation_mount_candidate.mount_graph import pivot_outside_domain
from calibration.solver_runtime.finite_rectangle_station_step_constraints import (
    finite_rectangle_station_constraints,actual_station_model_carriers,station_model_implementation_sources,
    _current_vertical_print_primitive)
from structured_programme_source_candidate.source_step_curvature_tightening import actual_carrier_step_response

RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_samea454_actual947_minimal_pivot_trial_resume_memory_sha')
FINAL='2766454eeb8282632e6bf1226d0305ffcda70b74fbe7d916b2614b8672f8b308'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_samea454_actual947_roles_SOC_sufficient_minimal_pivot_trial_20261011.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def carrier(query):return np.c_[query['signed_normal_px'],query['coverage_tangent_px']].ravel()
def retract(state,value):
    # Literal no-FD-reserve branch of the production reinitializer. The small
    # adapter has no installation optics graph; the shared course and original
    # pivot domain are identical. Accepted states must reproduce bit for bit.
    corrected=value.copy();before=pivot_outside_domain(state,value)
    changes=[]
    for row in before['installations']:
        if not row['valid']:
            slots=np.asarray(row['pivot_parameter_indices'][:2],int)
            corrected[slots]=row['nearest_course_xy_working_m']
            changes.extend(slots.tolist())
    after=pivot_outside_domain(state,corrected);assert after['valid']
    return corrected,dict(before=before,after=after,changed_columns=changes)

def main():
    began=time.perf_counter();directory=RUN/'accepted_states'/FINAL
    identity=read(directory/'state_identity.json');path=directory/'complete_state_UNACCEPTED.npz'
    assert sha(path)==identity['state_npz_sha256']
    assert sha(identity['camera_path'])==identity['camera_sha256']
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
          'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
          'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
          'aux__current_complete_parameter_layout_json')
    with np.load(path,allow_pickle=False)as z:small={k:z[k].copy()for k in keys}
    assert asha(small['q'])==identity['q_sha256']
    options=read(CONFIG)['source_options']
    provider,world,proof=current_complete_source_geometry(path,options['--world-init'])
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    state,error=state_camera_adapter(small,cameras,provider);assert error==0.,error
    manifest=read(RUN/'input_manifest.json')
    state.current_user_pivot_outside_constraint_contract=manifest['camera_pivot_exterior_constraints']
    scope=manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope is not None:state.current_user_physical_course_scope_source=scope['path']
    board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(small['q']))
    ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(small['q']))
    human=HumanStrokeTruth(state,options['--direct-human-truth-authority'],board,ice,parameter_count=len(small['q']))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    dispatch=[]
    for i,record in enumerate(human.full_stroke_records):
        primitive=_current_vertical_print_primitive(record,state.evaluation_q(small['q']))
        dispatch.append(dict(group=i,id=record['row']['id'],curve=record['row']['curve'],
            certified_whole_vertical_print=primitive is not None,
            world_samples=None if primitive is None else len(primitive),guard_vertices=len(record['uv'])))
    local=copy.copy(human);local.records=[human.records[29]];local.full_stroke_records=[human.full_stroke_records[29]]
    local.blocks={local.records[0]['row']['actual_pts']:human.blocks[local.records[0]['row']['actual_pts']]}
    local.active=group_dependencies(local,local.full_stroke_records)[0]
    assert local.records[0]['row']['id']=='user_native230024192_ad20_right_vertical_20261008'
    output=dict(schema='rinkspace.saved-native29-real-support-curvature-causal-audit.v1',
                new_fit_or_full_source_J=False,source_images_written=0,source_support_or_zero_caps_changed=False,
                script=ref(__file__),
                config=ref(CONFIG),final_state=ref(path),camera=ref(identity['camera_path']),
                repaired_station_model_sources=station_model_implementation_sources(),
                camera_archive_reproduction_maximum=error,world_proof=proof,
                source_references={k:ref(options[k])for k in ('--additional-board-source','--additional-ice-source',
                    '--direct-human-truth-authority','--direct-human-full-stroke-guard','--direct-human-stroke-cap-support',
                    '--practical-human-location-envelope')},
                native29_record=local.records[0]['record'],full_guard_support={k:local.full_stroke_records[0][k]
                    for k in ('uv','normals','bounds','tangent_bounds','original_raster_normal_bounds','original_source_tangent_bounds')
                    if k in local.full_stroke_records[0]},all_native_group_dispatch=dispatch,directions=[])
    history=read(RUN/'solver/iterations.json')
    parents=[read(CONFIG)['continuation']['state']['q_sha256']]+[
        read(p/'state_identity.json')['q_sha256']for p in sorted((RUN/'accepted_states').iterdir(),key=lambda p:p.stat().st_mtime)]
    for iteration,parent in enumerate(parents[:-1]):
        files=list((RUN/'actual_complete_source_directions').glob(parent+'_*.npz'));assert len(files)==1
        with np.load(files[0],allow_pickle=False)as z:q=z['q'].copy();step=z['step'].copy()
        assert asha(q)==parent;assert step.shape==(34340,)
        eq=state.evaluation_q(q);ed=state.evaluation_q(step)
        base=local.queries(eq,full_stroke=True)[0]['query']
        print(json.dumps(dict(stage='readonly_native29_parent',iteration=iteration,q_sha256=parent,
                            guard_max=float(abs(base['residual']).max()),query_keys=list(base))),flush=True)
        J=local.jacobian(eq,carriers=True,full_stroke=True)
        fixed_model=finite_rectangle_station_constraints(local,eq,local.active,full_stroke=True,
            group_limits=np.array([0.]),return_step_carrier_response=True)
        response=fixed_model['step_carrier_response'];A=fixed_model['matrix'];target=fixed_model['lower']
        face_FD=[]
        # Also retain smaller stencils to expose cancellation at 1e-6. The
        # 1e-3/1e-4 directional stencils stay on the actual nominal branch and
        # avoid dividing pixel projection roundoff by an excessively tiny h.
        for h in (1e-3,1e-4,1e-5,1e-6):
            plus=actual_carrier_step_response(response,actual_station_model_carriers(local,eq+h*ed,response))
            minus=actual_carrier_step_response(response,actual_station_model_carriers(local,eq-h*ed,response))
            estimate=(plus-minus)/(2*h);predicted=A@ed[local.active]
            face_FD.append(dict(h=h,maximum_error=float(abs(estimate-predicted).max()),
                worst_face=int(np.argmax(abs(estimate-predicted))),
                vertex0_cross_face_FD=float(estimate[4]),vertex0_cross_face_Jv=float(predicted[4])))
        columns=np.unique(J.indices);canonical_columns=state.canonical_parameter_layout.evaluation_to_canonical[columns]
        basecarrier=carrier(base);linear=J@ed
        full=local.full_stroke_records[0];bounds=np.c_[full['bounds'][:,0],full['tangent_bounds'][:,0]].ravel()
        upper=np.c_[full['bounds'][:,1],full['tangent_bounds'][:,1]].ravel()
        samples=[]
        block=local.blocks[local.records[0]['row']['actual_pts']]
        def frozen(value):
            cam=block.bundle.camera(native_camera_local(block,value[:state.structured_programme_coordinate_chart.core_count]),local.records[0]['row']['actual_pts'])
            return finite_stroke_queries(cam,board._world(value,local.records[0]['row']['curve']),
                full['uv'],full['normals'],full['bounds'],tangent_bounds=full['tangent_bounds'],
                frozen_segments=base['nominal_finite_segments'],
                frozen_segment_categories=base['nominal_finite_segment_categories'],
                frozen_support_faces=base['nominal_source_support_faces'],
                frozen_station_branches=base['nominal_source_station_branches'])
        cam=block.bundle.camera(native_camera_local(block,eq[:state.structured_programme_coordinate_chart.core_count]),local.records[0]['row']['actual_pts'])
        projected=project_central_valid_full_brown(cam,board._world(eq,local.records[0]['row']['curve']))[0]
        vertex0_segment=int(base['nominal_finite_segments'][0]);delta=projected[vertex0_segment+1]-projected[vertex0_segment]
        nt=np.stack((full['normals'][0],[-full['normals'][0,1],full['normals'][0,0]]))
        for k in range(12):
            alpha=2.**(-k);query=local.queries(eq+alpha*ed,full_stroke=True)[0]['query']
            recovered,recovery=retract(state,eq+alpha*ed)
            rq=local.queries(recovered,full_stroke=True)[0]['query']
            value=carrier(query);predicted=basecarrier+alpha*linear
            frozen_carrier=carrier(frozen(eq+alpha*ed))
            face_actual=actual_carrier_step_response(response,actual_station_model_carriers(local,eq+alpha*ed,response))
            face_linear=A@(alpha*ed[local.active]);new_slack=face_actual-target
            face_index=int(np.argmin(new_slack))
            violation=value-np.clip(value,bounds,upper)
            predicted_violation=predicted-np.clip(predicted,bounds,upper)
            worst=int(np.argmax(abs(violation)))
            samples.append(dict(k=k,alpha=alpha,maximum_actual_violation_px=float(abs(violation).max()),
                maximum_predicted_violation_px=float(abs(predicted_violation).max()),
                worst_vertex=worst//2,worst_axis='normal'if worst%2==0 else'tangent',
                carrier=value,linear_predicted_carrier=predicted,curvature_error=value-predicted,
                nominal_branch_nonlinear_carrier=frozen_carrier,
                nominal_branch_curvature_error=frozen_carrier-predicted,
                actual_violation=violation,predicted_violation=predicted_violation,
                branch_signature=query['source_support_branch_signature'],
                changed_branch_vertices=np.flatnonzero(np.any(query['source_support_branch_signature']!=base['source_support_branch_signature'],axis=1)),
                query_max_same_as_residual=float(abs(query['residual']).max()),
                actual_trial_pivot_recovery=recovery,recovered_carrier=carrier(rq),
                recovered_maximum_violation_px=float(abs(rq['residual']).max()),
                recovered_worst_vertex=int(np.argmax(abs(rq['residual'])))//2,
                recovered_branch_signature=rq['source_support_branch_signature'],
                repaired_station_face_actual_response=face_actual,
                repaired_station_face_linear_response=face_linear,
                repaired_station_maximum_actual_face_violation_px=float(np.maximum(-new_slack,0.).max()),
                repaired_station_worst_face=face_index,
                repaired_station_worst_face_curvature=float(face_linear[face_index]-face_actual[face_index])))
        alpha=history[iteration]['global_schur_trial_records'][-1]['whole_source_step_fraction']
        with np.load(RUN/'accepted_states'/parents[iteration+1]/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:child=z['q'].copy()
        difference=np.flatnonzero(child!=q+alpha*step)
        assert difference.tolist()==[11,12]
        recovered,recovery=retract(state,eq+alpha*ed)
        np.testing.assert_array_equal(state.canonical_parameter_layout.canonical_q(recovered),child)
        rf=np.concatenate([np.arange(int(start),int(start+length))for start,length in zip(small['aux__block_start'],small['aux__block_lengths'])])
        assert np.array_equal(child[rf],(q+alpha*step)[rf])
        soc=history[iteration]['global_schur_linear_report']['sequential_measured_shared_source_curvature_tightening']
        output['directions'].append(dict(iteration=iteration,parent_q_sha256=parent,saved_direction=ref(files[0]),
            step_sha256=asha(step),child_q_sha256=parents[iteration+1],accepted_alpha=alpha,
            pivot_retracted_columns=difference,all_RF4_exposures_shrunk_by_same_alpha=len(rf)//4,
            changed_RF4_exposures=int(np.count_nonzero(np.any(step[rf].reshape(-1,4)!=0,axis=1))),
            actual_native29_carrier_J_nonzero_canonical_columns=canonical_columns,
            native29_nominal_carrier=basecarrier,nominal_branch_signature=base['source_support_branch_signature'],
            vertex0_projected_segment_delta=delta,vertex0_segment_normal_tangent_slope=nt@delta,
            vertex0_balanced_to_old_tangent_response_weight=abs((nt@delta)[0])/np.sum(abs(nt@delta)),
            repaired_station_model_report=fixed_model['report'],
            repaired_station_original_dynamic_carrier_face_FD=face_FD,
            repaired_station_face_lower=target,
            native29_carrier_22column_contributions=J[:,columns].toarray()*ed[columns][None,:],
            native29_carrier_directional_first_order_px=linear,
            native29_carrier_J_by_column=J[:,columns].toarray(),
            SOC_original_evidence=soc,samples=samples))
        print(json.dumps(dict(stage='readonly_native29_direction_complete',iteration=iteration,
            nonzero_columns=canonical_columns.tolist(),actual_steps=[r['maximum_actual_violation_px']for r in samples],
            predicted_steps=[r['maximum_predicted_violation_px']for r in samples])),flush=True)
    output['seconds']=time.perf_counter()-began
    dest=Path(str(RUN)+'.bootstrap')/'native29_saved_direction_causal_audit.json'
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False,default=plain)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(dest),seconds=output['seconds'])),flush=True)

if __name__=='__main__':main()
