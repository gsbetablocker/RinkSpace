"""One small actual c134 source check before the full-target continuation."""
from pathlib import Path
import os,sys,json,pickle,hashlib,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth,UserConfirmedSourceTruth
from structured_programme_source_candidate.current_wide_source import CurrentWideSource

def main():
    started=time.perf_counter()
    runroot=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    old=runroot/'okada_joint_compact_fastcamera_c134_20261008_104825_235'
    initial=old/'initial_states/965e58d4c3ac40833090d4d9db639dd2870dbb17558d449893d10b72405f4031'
    current=runroot/'okada_joint_continuous_perimeter_current16ef_fulltarget_20261008_173656_145'
    auth=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261008/c134_full_programme/registered_user_complete_stroke_source_append_20261008')
    output=old/'diagnostics/morning_baseline_source20_identity/actual_common_axis_and_confirmed_source_check.json'
    if output.exists():raise FileExistsError(output)
    with np.load(initial/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        arrays={k:z[k]for k in z.files if k=='q'or k.startswith('aux__')}
    identity=json.loads((initial/'state_identity.json').read_text(encoding='utf8'))
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k]for k in z.files}
    owner=json.loads((current/'actual_absolute_geometry_FD_pool_runtime.json').read_text(encoding='utf8'))['material_owner']
    install_mappingproxy_pickle();install_geometry_unpickle()
    with Path(owner['path']).open('rb')as stream:provider=pickle.load(stream)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    q=state.evaluation_q(arrays['q']);world=q[state.geometry_parameter_indices]
    before=provider.scene(world)
    provider.current.shared_hockey_curling_axis=True
    for key in ('_successful_world_validation','_value_cache','_scene_cache'):
        cache=getattr(provider,key,None)
        if cache is not None:cache.clear()
    after=provider.scene(world)
    features={r['id']:r for r in after['ice_features']}
    prior={r['id']:r for r in before['ice_features']}
    lane=np.asarray(features['lane1']['polygon_xyz'])
    lane_axis=lane[1,:2]-lane[0,:2]
    crosses=[]
    for name in ('B1','B2','C'):
        poly=np.asarray(features[name]['segments_xyz'][0]);axis=poly[-1,:2]-poly[0,:2]
        cosine=float(axis@lane_axis/np.linalg.norm(axis)/np.linalg.norm(lane_axis))
        assert abs(cosine)<1e-12,(name,cosine)
        crosses.append(dict(feature=name,world_cosine=cosine))
    kept=[]
    for name in ('lane1','lane2','lane3','lane4','F1','F2','F3','F4','R1','R2','R3','R4'):
        for key in ('polygon_xyz','center_xyz','radius','rings'):
            if key not in features[name]:continue
            assert json.dumps(features[name][key],sort_keys=True)==json.dumps(prior[name][key],sort_keys=True),(name,key)
        kept.append(name)
    state.current_user_physical_course_scope_source=auth/'user20_existing_lower_long_side_connected_lower_right_scope_authority_v1.json'
    board=NativeBoardEdgeFactor(state,auth/'board_parent77_plus_user_confirmed230501120_top_contact_v1.json',len(q))
    ice=NativeIceCurveFactor(state,auth/'ice_parent51_plus_user19_right_goal_baseline_registered_complete_stroke_FIT_CANDIDATE_v2_HUMAN_LABEL_SUPPORT.json',len(q))
    block=ice.block(q,True);column=int(state.geometry_parameter_indices[14]);step=1e-6
    direction=np.zeros(len(q));direction[column]=1
    fd=(ice.block(q+step*direction,False).r-ice.block(q-step*direction,False).r)/(2*step)
    je=float(abs(fd-block.J[:,column].toarray().ravel()).max())
    if je>=2e-3:
        order=np.argsort(abs(fd-block.J[:,column].toarray().ravel()))[-8:]
        offsets=np.cumsum([0]+[2*len(r['uv'])for r in ice.rows])
        print(json.dumps(dict(common_yaw_true_FD_error=je,rows=[dict(scalar=int(k),
            id=ice.rows[int(np.searchsorted(offsets,k,side='right')-1)]['id'],
            FD=float(fd[k]),J=float(block.J[k,column]))for k in order])),flush=True)
    from structured_programme_source_candidate.native_ice_curve_source import _curve_error_on_association
    actual_cameras=ice._cameras(q)
    associations=[_curve_error_on_association(actual_cameras[row['actual_pts']],ice._world(q,row['curve']),row['uv'])[1]for row in ice.rows]
    replay_fd=(ice._derivative_residual(q+step*direction,cameras=actual_cameras,associations=associations)
               -ice._derivative_residual(q-step*direction,cameras=actual_cameras,associations=associations))/(2*step)
    replay_error=float(abs(replay_fd-block.J[:,column].toarray().ravel()).max())
    assert replay_error<2e-3,(je,replay_error)
    assert block.J[:,int(state.geometry_parameter_indices[2])].nnz==0
    truth=HumanStrokeTruth(state,auth/'human_registered_hard_truth_authorities_v1.json',board,ice,parameter_count=len(q))
    truth.bind_full_stroke_guard(auth/'human_registered_full_stroke_hard_guard_v1.json')
    truth.bind_stroke_cap_support(auth/'human_registered_stroke_end_2D_source_footprint_support_v2.json')
    envelope=runroot/'human_hardtruth_native5760_wide733_caps_PhaseI_20261008_142016_806/fixed_practical_manual_location_envelope_v2_ENGINEERING_LOCALIZATION.json'
    truth.bind_practical_location_envelope(envelope)
    old_scene=json.loads((initial/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    exported=[r['camera25']for r in old_scene['frame_cameras']if r.get('source_pts')==56218880]
    assert len(exported)==1
    wauth=auth/'human_original_wide_top_contours_hard_authority_v1.json'
    wmeta=json.loads(wauth.read_text(encoding='utf8'))
    trace=json.loads(Path(wmeta['trace_packet_reference']['path']).read_text(encoding='utf8'))
    wide=CurrentWideSource.__new__(CurrentWideSource);wide.geometry=provider.current;wide.qualified_scene=provider.qualified_scene
    wide.geometry_parameter_indices=state.geometry_parameter_indices;wide.observations=[];wide.ad_packet={'rows':[]}
    wide.template_camera25=np.asarray(exported[0],float);wide.trace_packet=trace
    wide.trace_uv=[np.asarray(r['observed_centres_source_px'],float)for r in trace['contours']]
    layout=state.canonical_parameter_layout
    wi=layout.canonical_to_evaluation[np.asarray(arrays['aux__current_wide_camera_parameter_indices'],int)]
    assert np.array_equal(wide.camera(q[wi]),exported[0])
    wt=WideStrokeTruth(wide,wauth,len(q),wi)
    wt.bind_practical_location_envelope(envelope)
    confirmed=UserConfirmedSourceTruth(state,auth/'user_confirmed_native_source_priority_authorities_v1.json',board,ice,parameter_count=len(q))
    _,_,_,confirmed_groups=confirmed._carriers(q)
    report=dict(schema='rinkspace.actual-common-ice-axis-c134-native-check.v1',
        source_state=str(initial/'complete_state_UNACCEPTED.npz'),canonical_q_sha256=identity['q_sha256'],
        camera_reproduce_max_difference=camera_error,original_hockey_yaw=float(world[2]),
        common_yaw=float(world[14]),initial_rotation_degrees=float(np.rad2deg(world[14]-world[2])),
        world_perpendicularity=crosses,curling_features_bitexact_preserved=kept,
        native_ice_scalar_rows=len(block.r),native_common_yaw_true_value_FD_max_error=je,
        native_common_yaw_nominal_finite_association_J_max_error=replay_error,derived_yaw_J_column_empty=True,
        native_manual_location_report=truth.report(q),wide_manual_location_report=wt.report(q),
        confirmed_source_groups=confirmed_groups,
        full_motion_graph_or_normal_loaded=False,source_UV_sigma_roles_or_camera_changed=False,
        seconds=time.perf_counter()-started,accuracy_accepted=False)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report_path=str(output),seconds=report['seconds'],common_yaw_J_error=je,
        native=report['native_manual_location_report']['maximum_violation_px'],
        wide=report['wide_manual_location_report']['maximum_violation_px'],confirmed_groups=confirmed_groups)),flush=True)

if __name__=='__main__':main()
