"""Actual saved-q full truth and local finite station preflight, no programme graph."""
from pathlib import Path
import argparse,os,sys,time,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
from calibration.solver_runtime import install
install()
import audit_current_user_ice_truth as a
from calibration.solver_runtime.finite_rectangle_station_step_constraints import finite_rectangle_station_constraints,actual_station_model_carriers
from calibration.solver_runtime.finite_rectangle_projection_fd import projected_station_contexts
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart,actual_carrier_step_response
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();started=time.perf_counter();config=a.read(opt.config);options=config['source_options']
    directory=Path(options['--camera-seed']).parent;manifest=a.read(directory.parent.parent/'input_manifest.json')
    parent=a.read(directory/'truth_restoration_state.json')
    state,q,identity,error=a.load_state(directory,Path(options['--world-init']),manifest)
    assert error==0.;identity['_accepted_directory']=str(directory)
    board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=a.bind_human(state,board,ice,options,len(q));wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    circle,material,placement=a.existing_LOW_source_owners(state,q,directory,options,a.read(directory/'scene_UNACCEPTED.json'))
    confirmed=UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,parameter_count=len(q),
        circle_factor=circle,material_factor=material,placement_source=placement)
    before=a.values(human,wide,confirmed,q);loaded=time.perf_counter()
    print(json.dumps(dict(stage='actual86_loaded',seconds=loaded-started)),flush=True)
    direction=next((directory.parent.parent/'actual_complete_source_directions').glob('*.npz'))
    with np.load(direction,allow_pickle=False)as z:
        requested=z['requested_shared_indices'];fixed=z['fixed_parameter_indices'];mapping=z['canonical_to_evaluation']
    selected=mapping[requested[~np.isin(requested,fixed)]];fixed=mapping[fixed]
    limits=np.asarray(parent['families']['native']['limits'])
    original=constraints_on_declared_free_chart(human,q,selected,fixed,full_stroke=True,group_limits=limits,
        return_step_carrier_response=True)
    result=finite_rectangle_station_constraints(human,q,selected,fixed,full_stroke=True,group_limits=limits,
        return_step_carrier_response=True)
    response=result['step_carrier_response'];contexts=response.get('finite_station_contexts',[])
    assert contexts
    mask=original['step_carrier_response']['retained_face_mask'];C=len(original['step_carrier_response']['nominal_carrier'])
    columns=np.r_[np.arange(C),np.arange(C)][mask]
    replaced=[c['original_carrier_offset']+axis for c in contexts for axis in (0,1)]
    kept=~np.isin(columns,replaced);unchanged=int(np.count_nonzero(kept))
    assert np.array_equal(result['matrix'][:unchanged],original['matrix'][kept])
    assert np.array_equal(result['lower'][:unchanged],original['lower'][kept])
    assert response['row_group_index'].shape==result['lower'].shape
    assert np.array_equal(response['row_limit_expansion'],limits[response['row_group_index']])
    actual=actual_station_model_carriers(human,q,response)
    assert np.array_equal(actual,response['nominal_carrier'])
    assert np.all(actual_carrier_step_response(response,actual)==0.)
    assert np.max(result['lower'],initial=-np.inf)<=1e-8
    native_done=time.perf_counter();print(json.dumps(dict(stage='actual_native_finite_faces',seconds=native_done-loaded,
        contexts=len(contexts),rows=len(result['lower']),unchanged_rows=unchanged)),flush=True)
    for family,factor in (('wide',wide),('confirmed',confirmed)):
        domain=constraints_on_declared_free_chart(factor,q,selected,fixed,group_limits=np.asarray(parent['families'][family]['limits']),
            return_step_carrier_response=True)
        assert domain['matrix'].shape[1]==len(selected)
        assert np.max(domain['lower'],initial=-np.inf)<=1e-8
    after=a.values(human,wide,confirmed,q)
    for key in ('primary','full','wide','confirmed_carriers','confirmed_details','buffers'):
        assert a.exact(before[key],after[key]),key
    for family,values in after['buffers'].items():
        assert np.all(values<=np.asarray(parent['families'][family]['limits'])+1e-8)
    projections=projected_station_contexts(human,q,contexts,selected)
    artifact=opt.output.with_suffix('.npz')
    np.savez(artifact,selected_evaluation_columns=selected,canonical_to_evaluation=mapping,
        projected_foot_xy=projections['projected_foot_xy'],fixed_foot_jacobian=projections['fixed_foot_jacobian'],
        station_screen_tangent=projections['station_screen_tangent'],nominal_physical_station=projections['nominal_physical_station'],
        physical_endpoints_xy=projections['physical_endpoints_xy'],endpoint_jacobian=projections['endpoint_jacobian'],
        endpoint_valid=projections['endpoint_valid'],groups=np.array([c['group']for c in contexts]),
        vertices=np.array([c['source_vertex']for c in contexts]),segments=np.array([c['segment']for c in contexts]),
        source_uv=np.stack([c['record']['uv'][c['source_vertex']]for c in contexts]),
        source_normals=np.stack([c['record']['normals'][c['source_vertex']]for c in contexts]),
        normal_bounds=np.stack([c['record']['bounds'][c['source_vertex']]for c in contexts]),
        tangent_bounds=np.stack([c['record']['tangent_bounds'][c['source_vertex']]for c in contexts]),
        group_limits=limits[[c['group']for c in contexts]],matrix=result['matrix'][unchanged:],lower=result['lower'][unchanged:],
        original_q=q)
    a.save(opt.output,dict(schema='rinkspace.actual-current-full86-finite-station-step-route-preflight.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],config=a.ref(opt.config),complete_state=a.ref(directory/'complete_state_UNACCEPTED.npz'),
        original_source_count=manifest['source_rows'],full_q=len(q),effective_selected_count=len(selected),
        all86_query_carriers_original_support_and_limits_unchanged=True,source_split_and_UV_not_modified=True,
        native_original_step_rows=len(original['lower']),native_finite_step_rows=len(result['lower']),
        unchanged_original_native_faces_bitexact=unchanged,replaced_original_native_faces=int(np.count_nonzero(~kept)),
        local_model=result['report']['finite_rectangle_station_local_model'],local_actual_projection=projections['report'],
        compact_actual_FD_and_face_arrays=a.ref(artifact),phases=dict(load_and_original86_queries=loaded-started,
            original_plus_finite_native_constraints=native_done-loaded,wide_confirmed_and_identity=time.perf_counter()-native_done),
        seconds=time.perf_counter()-started,programme_graph_J_normal_or_solver_not_constructed=True,
        nonlinear_acceptance_and_visual_accuracy_not_claimed=True))
    print(json.dumps(dict(report=str(opt.output),seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
