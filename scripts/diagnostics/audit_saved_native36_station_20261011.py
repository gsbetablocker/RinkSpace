"""Read only: native36 on the complete direction saved by the one-cycle trial.

No optimizer, source graph, full source J, source image or fitted state is
created. Small carrier derivatives use the original public truth producer.
"""
from pathlib import Path
import sys, copy, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_saved_native29_curvature_20261011 as common
import numpy as np
from calibration.solver_runtime.finite_rectangle_station_step_constraints import (
    _current_boundary_clipped_primitive, finite_rectangle_station_constraints,
    actual_station_model_carriers, station_model_implementation_sources)
from structured_programme_source_candidate.source_step_curvature_tightening import actual_carrier_step_response

ROOT=common.ROOT
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
PARENT=common.FINAL
CHILD='8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'

def main():
    started=time.perf_counter(); options=common.read(CONFIG)['source_options']
    original=common.RUN/'accepted_states'/PARENT
    identity=common.read(original/'state_identity.json'); path=original/'complete_state_UNACCEPTED.npz'
    assert common.sha(path)==identity['state_npz_sha256']
    assert common.sha(identity['camera_path'])==identity['camera_sha256']
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
          'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
          'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
          'aux__current_complete_parameter_layout_json')
    with np.load(path,allow_pickle=False)as z:small={k:z[k].copy()for k in keys}
    provider,world,proof=common.current_complete_source_geometry(path,options['--world-init'])
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    state,error=common.state_camera_adapter(small,cameras,provider);assert error==0.,error
    manifest=common.read(RUN/'input_manifest.json')
    state.current_user_pivot_outside_constraint_contract=manifest['camera_pivot_exterior_constraints']
    scope=manifest['additional_real_native_board_edges'].get('physical_course_scope_authority')
    if scope is not None:state.current_user_physical_course_scope_source=scope['path']
    board=common.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(small['q']))
    ice=common.NativeIceCurveFactor(state,options['--additional-ice-source'],len(small['q']))
    human=common.HumanStrokeTruth(state,options['--direct-human-truth-authority'],board,ice,parameter_count=len(small['q']))
    human.bind_full_stroke_guard(options['--direct-human-full-stroke-guard'])
    human.bind_stroke_cap_support(options['--direct-human-stroke-cap-support'])
    human.bind_practical_location_envelope(options['--practical-human-location-envelope'])
    local=copy.copy(human);local.records=[human.records[36]];local.full_stroke_records=[human.full_stroke_records[36]]
    local.blocks={local.records[0]['row']['actual_pts']:human.blocks[local.records[0]['row']['actual_pts']]}
    local.active=common.group_dependencies(local,local.full_stroke_records)[0]
    assert local.records[0]['row']['id']=='user_complete_yellow_C_centreline_native229246976'
    direction_paths=list((RUN/'actual_complete_source_directions').glob(PARENT+'_*.npz'));assert len(direction_paths)==1
    with np.load(direction_paths[0],allow_pickle=False)as z:q=z['q'].copy();step=z['step'].copy()
    assert common.asha(q)==PARENT;assert step.shape==(34340,)
    eq=state.evaluation_q(q);ed=state.evaluation_q(step);record=local.full_stroke_records[0]
    block=local.blocks[record['row']['actual_pts']]
    def project(value):
        camera=block.bundle.camera(common.native_camera_local(block,value[:state.structured_programme_coordinate_chart.core_count]),record['row']['actual_pts'])
        primitive=record['factor']._world(value,record['row']['curve'])
        pixels,depth,central=common.project_central_valid_full_brown(camera,primitive)
        return camera,primitive,pixels,depth,central
    base=local.queries(eq,full_stroke=True)[0]['query'];basecarrier=common.carrier(base)
    J=local.jacobian(eq,carriers=True,full_stroke=True)
    columns=np.unique(J.indices);canonical_columns=state.canonical_parameter_layout.evaluation_to_canonical[columns]
    linear=J@ed
    model=finite_rectangle_station_constraints(local,eq,local.active,full_stroke=True,
        group_limits=np.array([0.]),return_step_carrier_response=True)
    response=model['step_carrier_response'];A=model['matrix'];target=model['lower']
    primitive=_current_boundary_clipped_primitive(record,eq);assert primitive is not None
    camera,world,pixels,depth,central=project(eq)
    curve=record['row']['curve'];factor=record['factor']
    effective=factor.geometry.effective_world(eq[factor.world_indices])
    feature=factor._ice_material_view.features(effective)[curve['feature_id']]
    def frozen(value):
        cam,primitive,_,_,_=project(value)
        return common.finite_stroke_queries(cam,primitive,record['uv'],record['normals'],record['bounds'],
            tangent_bounds=record['tangent_bounds'],frozen_segments=base['nominal_finite_segments'],
            frozen_segment_categories=base['nominal_finite_segment_categories'],
            frozen_support_faces=base['nominal_source_support_faces'],
            frozen_station_branches=base['nominal_source_station_branches'])
    lower=np.c_[record['bounds'][:,0],record['tangent_bounds'][:,0]].ravel()
    upper=np.c_[record['bounds'][:,1],record['tangent_bounds'][:,1]].ravel()
    samples=[]
    for k in range(11):
        alpha=2.**(-k);value=eq+alpha*ed
        recovered,retraction=common.retract(state,value)
        query=local.queries(value,full_stroke=True)[0]['query']
        rq=local.queries(recovered,full_stroke=True)[0]['query']
        actual=common.carrier(query);predicted=basecarrier+alpha*linear
        fq=frozen(value);fc=common.carrier(fq)
        fv=actual_carrier_step_response(response,actual_station_model_carriers(local,value,response))
        flinear=A@(alpha*ed[local.active]);slack=fv-target
        true=actual-np.clip(actual,lower,upper);frozen_violation=fc-np.clip(fc,lower,upper)
        worst=int(np.argmax(abs(true)));vertex=worst//2
        segment=int(base['nominal_finite_segments'][vertex]);delta=pixels[segment+1]-pixels[segment]
        normal=record['normals'][vertex];tangent=np.array([-normal[1],normal[0]])
        slopes=np.stack((normal,tangent))@delta
        _current_boundary_clipped_primitive(record,value)
        samples.append(dict(k=k,alpha=alpha,carrier=actual,linear_predicted_carrier=predicted,
            curvature_error=actual-predicted,nominal_branch_nonlinear_carrier=fc,
            nominal_branch_curvature_error=fc-predicted,actual_violation=true,
            maximum_actual_violation_px=float(abs(true).max()),
            maximum_predicted_violation_px=float(abs(predicted-np.clip(predicted,lower,upper)).max()),
            maximum_frozen_branch_violation_px=float(abs(frozen_violation).max()),
            branch_signature=query['source_support_branch_signature'],
            changed_branch_vertices=np.flatnonzero(np.any(query['source_support_branch_signature']!=base['source_support_branch_signature'],axis=1)),
            worst_vertex=vertex,worst_axis='normal'if worst%2==0 else'tangent',
            worst_base_branch=base['source_support_branch_signature'][vertex],
            worst_trial_branch=query['source_support_branch_signature'][vertex],
            worst_nominal_source_uv=record['uv'][vertex],worst_original_normal=normal,
            worst_original_normal_bounds=record['bounds'][vertex],worst_original_tangent_bounds=record['tangent_bounds'][vertex],
            worst_nominal_segment_delta=pixels[segment+1]-pixels[segment],
            worst_nominal_segment_normal_tangent_slope=slopes,
            normal_fraction_of_balanced_Linf_response=float(abs(slopes[0])/np.sum(abs(slopes))),
            tangent_fraction_of_balanced_Linf_response=float(abs(slopes[1])/np.sum(abs(slopes))),
            source_query_maximum=float(abs(query['residual']).max()),
            actual_pivot_retraction=retraction,recovered_maximum_violation_px=float(abs(rq['residual']).max()),
            recovered_branch_signature=rq['source_support_branch_signature'],
            current_retained_face_actual_response=fv,current_retained_face_linear_response=flinear,
            current_retained_face_maximum_violation_px=float(np.maximum(-slack,0).max()),
            current_retained_face_worst_index=int(np.argmin(slack))))
    # These are original small public guard derivatives, never a full-source J.
    FD=[]
    for h in (1e-3,1e-4,1e-5,1e-6):
        actual_derivative=(common.carrier(local.queries(eq+h*ed,full_stroke=True)[0]['query'])-
                           common.carrier(local.queries(eq-h*ed,full_stroke=True)[0]['query']))/(2*h)
        frozen_derivative=(common.carrier(frozen(eq+h*ed))-common.carrier(frozen(eq-h*ed)))/(2*h)
        FD.append(dict(h=h,maximum_dynamic_derivative_difference=float(abs(actual_derivative-linear).max()),
                       maximum_frozen_derivative_difference=float(abs(frozen_derivative-linear).max()),
                       actual_derivative=actual_derivative,frozen_derivative=frozen_derivative))
    face_FD=[]
    for h in (1e-3,1e-4,1e-5,1e-6):
        plus=actual_carrier_step_response(response,actual_station_model_carriers(local,eq+h*ed,response))
        minus=actual_carrier_step_response(response,actual_station_model_carriers(local,eq-h*ed,response))
        estimate=(plus-minus)/(2*h);predicted=A@ed[local.active]
        face_FD.append(dict(h=h,maximum_error=float(abs(estimate-predicted).max()),
            worst_face=int(np.argmax(abs(estimate-predicted))),directional_FD=estimate,
            directional_model=predicted))
    childpath=RUN/'accepted_states'/CHILD/'complete_state_UNACCEPTED.npz'
    childidentity=common.read(childpath.parent/'state_identity.json')
    assert common.sha(childpath)==childidentity['state_npz_sha256']
    with np.load(childpath,allow_pickle=False)as z:child=z['q'].copy()
    recovered,retraction=common.retract(state,eq+.5*ed)
    np.testing.assert_array_equal(state.canonical_parameter_layout.canonical_q(recovered),child)
    bounds_check=dict(source_guard_vertices=len(record['uv']),physical_world_samples=len(world),
        source_guard_first_last_uv=record['uv'][[0,-1]],physical_world_endpoints=world[[0,-1]],
        projected_physical_endpoints=pixels[[0,-1]],physical_endpoint_depth=depth[[0,-1]],
        physical_endpoint_central=central[[0,-1]],
        nominal_selected_segments=base['nominal_finite_segments'],
        nominal_segment_station=base['nearest_finite_segment_station'],
        nominal_physical_station=base['finite_primitive_station'],
        nominal_source_support_branch_signature=base['source_support_branch_signature'],
        nominal_vertices_in_first_last_sampling_cells=np.flatnonzero((base['nominal_finite_segments']==0)|(base['nominal_finite_segments']==len(world)-2)),
        current_station_route_source_vertices=[c['source_vertex']for c in response.get('finite_station_contexts',[])],
        physical_certificate_feature=feature,
        producer_certificate='Exact sampled world equals current feature whole segment; true endpoints independently meet current fitted course boundary',
        sampling_cell_is_not_physical_cap=True,source_drawing_endpoint_is_not_physical_cap=True)
    report=dict(schema='rinkspace.saved-native36-station-coordinate-readonly-audit.v1',
        new_fit_or_full_source_J=False,source_images_written=0,source_support_or_zero_caps_changed=False,
        audit_wrote_solver_code=False,implementation_changed_since_before_audit=True,
        before_implementation_evidence=common.ref(Path(str(RUN)+'.bootstrap')/'native36_saved_direction_station_causal_audit.json'),
        original_ice_source_or_query_producer_changed=False,
        certified_boundary_station_local_route_extended=True,script=common.ref(__file__),config=common.ref(CONFIG),
        parent_state=common.ref(path),parent_identity=common.ref(original/'state_identity.json'),
        parent_camera=common.ref(identity['camera_path']),child_state=common.ref(childpath),
        child_identity=common.ref(childpath.parent/'state_identity.json'),saved_direction=common.ref(direction_paths[0]),
        step_sha256=common.asha(step),camera_archive_reproduction_maximum=error,
        current_station_implementation=station_model_implementation_sources(),world_proof=proof,
        source_references={k:common.ref(options[k])for k in ('--additional-ice-source','--direct-human-truth-authority',
            '--direct-human-full-stroke-guard','--direct-human-stroke-cap-support','--practical-human-location-envelope')},
        native36_record=local.records[0]['record'],native36_row={k:v for k,v in record['row'].items()if not k.startswith('_')},
        physical_cap_vs_computational_cell_evidence=bounds_check,
        original_support={k:record[k]for k in('uv','normals','bounds','tangent_bounds',
            'original_raster_normal_bounds','original_source_tangent_bounds')if k in record},
        original_carrier=basecarrier,carrier_J_nonzero_canonical_columns=canonical_columns,
        carrier_J_by_column=J[:,columns].toarray(),carrier_directional_first_order_px=linear,
        current_step_face_report=model['report'],current_retained_face_lower=target,
        current_step_face_count=len(target),current_finite_station_context_count=len(response.get('finite_station_contexts',[])),
        retained_original_face_mask=response.get('retained_face_mask'),FD=FD,station_face_FD=face_FD,samples=samples,
        accepted_half_direction_reproduced_bit_for_bit=True,accepted_pivot_retraction=retraction,
        seconds=time.perf_counter()-started)
    dest=Path(str(RUN)+'.bootstrap')/'native36_saved_direction_station_repaired_audit.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=common.plain)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(dest),sha256=common.sha(dest),seconds=report['seconds'],
        finite_contexts=report['current_finite_station_context_count'],vertices=len(record['uv']),world_samples=len(world),
        nonzero_columns=canonical_columns.tolist(),
        steps=[dict(alpha=s['alpha'],max=s['maximum_actual_violation_px'],frozen_max=s['maximum_frozen_branch_violation_px'],
          worst_vertex=s['worst_vertex'],base=s['worst_base_branch'].tolist(),trial=s['worst_trial_branch'].tolist(),
          slope=s['worst_nominal_segment_normal_tangent_slope'].tolist())for s in samples]),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
