"""Actual complete-state material/source verification without a motion graph."""
from pathlib import Path
import argparse,hashlib,json,os,sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout,PairedProgrammeCoordinateChart
from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def token(value):return hashlib.sha256(np.asarray(value,dtype=np.float64).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def numeric_json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--auxiliary',type=Path,required=True)
    ap.add_argument('--track-packet',type=Path,required=True);ap.add_argument('--old-track-packet',type=Path,required=True)
    ap.add_argument('--edge-packet',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--strict-half-circle-crease',action='store_true')
    ap.add_argument('--derive-hockey-center-from-course',action='store_true')
    ap.add_argument('--reuse-existing-initializer',action='store_true')
    ap.add_argument('--reuse-completed-world-checks',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    arrays=dict(np.load(args.state,allow_pickle=False));original_q=arrays['q'].copy();assert original_q.shape==(34306,)
    packet=read(args.track_packet);initializer=dict(additional_independent_locations=[15,16,17,18,19],
        frozen_source_print_templates=packet['frozen_source_print_templates'],independent_lane_right_extensions=True)
    if args.strict_half_circle_crease:initializer['strict_half_circle_crease']=True
    if args.derive_hockey_center_from_course:initializer['derive_hockey_center_from_course']=True
    layout=CompleteProgrammeParameterLayout(len(original_q),32);chart=PairedProgrammeCoordinateChart(32)
    old_world_indices=arrays['aux__current_shared_world_geometry_parameter_indices'].copy()
    canonical_q=np.r_[original_q,np.zeros(32)];arrays['q']=canonical_q
    arrays.update(aux__current_shared_world_geometry_parameter_indices=layout.canonical_indices(chart.world_indices),
        aux__current_complete_parameter_layout_json=np.asarray(json.dumps(layout.manifest())),
        aux__current_evaluation_to_canonical_parameter_indices=layout.evaluation_to_canonical,
        aux__current_physical_geometry_initializer_json=np.asarray(json.dumps(initializer)),
        aux__physical_geometry_parent_complete_q_sha256=np.asarray(token(original_q)),
        aux__physical_geometry_state_role=np.asarray('complete inherited parameter/profile initializer; new target source J/cost not evaluated'))
    state_path=args.output/'complete_state_initializer_UNACCEPTED.npz'
    if args.reuse_existing_initializer:
        with np.load(state_path,allow_pickle=False)as existing:
            assert set(existing.files)==set(arrays)
            assert all(np.array_equal(existing[k],v)for k,v in arrays.items())
    else:np.savez_compressed(state_path,**arrays)
    profiles=[k for k in arrays if k.startswith('profile__')]
    with np.load(state_path,allow_pickle=False)as saved:
        assert np.array_equal(saved['q'][:len(original_q)],original_q)
        assert all(np.array_equal(saved[k],arrays[k])for k in profiles)
    del arrays
    provider,world,proof=current_complete_source_geometry(state_path,args.auxiliary)
    old_provider,old_world,old_proof=current_complete_source_geometry(args.state,args.auxiliary)
    identity=read(args.state.parent/'state_identity.json');cameras=dict(np.load(identity['camera_path'],allow_pickle=False))
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names','aux__installation_xyz_slots',
        'aux__current_shared_mount_parameter_indices','aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json')
    with np.load(state_path,allow_pickle=False)as saved:small={k:saved[k].copy()for k in(*keys,'aux__current_complete_parameter_layout_json')}
    with np.load(args.state,allow_pickle=False)as saved:old_small={k:saved[k].copy()for k in keys}
    state,camera_error=state_camera_adapter(small,cameras,provider);old_state,old_camera_error=state_camera_adapter(old_small,cameras,old_provider)
    q=state.evaluation_q(canonical_q);oldq=old_small['q']
    new_factor=NativeBoardTrackFactor(state,args.track_packet,len(q));old_factor=NativeBoardTrackFactor(old_state,args.old_track_packet,len(oldq))
    new_block=new_factor.block(q,True);old_block=old_factor.block(oldq,True)
    def track_values(value):return np.concatenate([new_factor._evaluate(value,row)[0]for row in new_factor.rows+new_factor.anchor_rows])
    parent_ids={r['id']for r in read(args.old_track_packet)['rows'][:44]}
    parent_fit=[r for r in old_factor.rows if r['id']in parent_ids]
    assert [r['id']for r in new_factor.rows[:len(parent_fit)]]==[r['id']for r in parent_fit]
    parent_points=sum(len(r['uv0'])for r in parent_fit);parent_scalars=2*parent_points
    assert parent_points==515
    mapped=new_block.J[:,layout.canonical_to_evaluation].tocsr()
    prefix_J_difference=(mapped[:parent_scalars,:len(oldq)]-old_block.J[:parent_scalars]).tocsr()
    prefix_r_difference=float(np.max(abs(new_block.r[:parent_scalars]-old_block.r[:parent_scalars])))
    prefix_J_max=float(np.max(abs(prefix_J_difference.data)))if prefix_J_difference.nnz else 0.
    assert prefix_r_difference==0. and prefix_J_max==0.
    rng=np.random.default_rng(3433898);d=np.zeros(len(q));cols=np.unique(new_block.J.indices);d[cols]=rng.normal(size=len(cols))
    d/=max(float(abs(new_block.J@d).max()),1.);h=1e-4
    actual=(track_values(q+h*d)-track_values(q-h*d))/(2*h)
    fd_error=float(np.max(abs(actual-new_block.J@d)));assert fd_error<2e-3,fd_error
    world_direction=np.zeros(len(q));world_direction[state.geometry_parameter_indices]=rng.normal(size=provider.parameter_count)
    world_direction/=max(float(abs(new_block.J@world_direction).max()),1.)
    wactual=(track_values(q+h*world_direction)-track_values(q-h*world_direction))/(2*h)
    world_direction_error=float(abs(wactual-new_block.J@world_direction).max());assert world_direction_error<2e-3,world_direction_error
    edges=NativeBoardEdgeFactor(state,args.edge_packet,len(q));eb=edges.block(q,True)
    ed=np.zeros(len(q));ed[edges.active]=rng.normal(size=len(edges.active));ed/=max(float(abs(eb.J@ed).max()),1.)
    eh,branch=edges.local_support_directional_step(q,ed,1e-4)
    assert branch['same_support_branch_step_found']
    eactual=(edges._residual(q+eh*ed,edges.rows)-edges._residual(q-eh*ed,edges.rows))/(2*eh)
    edge_error=float(abs(eactual-eb.J@ed).max());edge_allowed=2e-3+2e-5*float(abs(eactual).max());assert edge_error<=edge_allowed,(edge_error,edge_allowed)
    all_world_checks=[]
    if args.reuse_completed_world_checks:
        prior=read(args.output/'all98_world_column_FD_progress.json')
        assert args.reuse_existing_initializer and prior['complete']
        all_world_checks=prior['columns']
        assert len(all_world_checks)==provider.parameter_count
        assert all(int(r['logical_world_column'])==i and r['passed']for i,r in enumerate(all_world_checks))
    else:
        for logical,column in enumerate(state.geometry_parameter_indices):
            direction=np.zeros(len(q));direction[column]=1.;wh=1e-6
            tp=track_values(q+wh*direction);tm=track_values(q-wh*direction)
            observed=(tp-tm)/(2*wh);predicted=new_block.J[:,column].toarray().ravel()
            terr=float(abs(observed-predicted).max());tallowed=2e-3+2e-5*float(abs(observed).max())
            local_h,local_branch=edges.local_support_directional_step(q,direction,wh)
            assert local_branch['same_support_branch_step_found']
            ep=edges._residual(q+local_h*direction,edges.rows);em=edges._residual(q-local_h*direction,edges.rows)
            observed_edge=(ep-em)/(2*local_h);predicted_edge=eb.J[:,column].toarray().ravel()
            eerr=float(abs(observed_edge-predicted_edge).max());eallowed=2e-3+2e-5*float(abs(observed_edge).max())
            record=dict(logical_world_column=logical,evaluation_column=int(column),canonical_column=int(layout.evaluation_to_canonical[column]),
                track_FD_max_error=terr,track_allowed=tallowed,track_J_max=float(abs(predicted).max()),track_FD_max=float(abs(observed).max()),
                edge_FD_max_error=eerr,edge_allowed=eallowed,edge_J_max=float(abs(predicted_edge).max()),edge_FD_max=float(abs(observed_edge).max()),
                edge_local_step=local_h,edge_support_branch_same=True,passed=terr<=tallowed and eerr<=eallowed)
            all_world_checks.append(record)
            (args.output/'all98_world_column_FD_progress.json').write_text(json.dumps(dict(columns=all_world_checks,complete=len(all_world_checks)==provider.parameter_count),indent=2)+'\n',encoding='utf8')
            assert record['passed'],record
    scene=provider.scene(world);scene=new_factor.export_print_scene(scene,q)
    scene['source_event_state_contract']=dict(q_sha256=token(canonical_q),evaluation_q_sha256=token(q),
        source_state_role='inherited complete initializer; small actual source verification only',accuracy_accepted=False)
    print_descriptor=scene['source_conditioned_board_prints'];print_descriptor['evaluation_q_sha256']=print_descriptor['q_sha256'];print_descriptor['q_sha256']=token(canonical_q)
    export_error=0.
    for item in print_descriptor['patterns']:
        n=item['advertisement_id']
        for contour in item['external_contours']:
            uv=np.asarray(contour['frozen_normalized_material_coordinates']);actual_world=provider.current.source_template_material(n,*uv.T,world)[0]
            export_error=max(export_error,float(abs(actual_world-np.asarray(contour['xyz_scene_units'])).max()))
    assert export_error==0.
    view=CurrentIceMaterialView(provider.current);features=view.features(world);scene_features={f['id']:f for f in scene['ice_features']};lane_checks=[]
    for name,k in provider.current.lane_right_extension_indices.items():
        before=np.asarray(features[name]['segments_xyz']);assert np.array_equal(before,np.asarray(scene_features[name]['segments_xyz']))
        changed=world.copy();changed[k]+=.01;after=np.asarray(view.features(changed)[name]['segments_xyz'])
        assert np.array_equal(before[:,0],after[:,0])and not np.array_equal(before[:,1],after[:,1])
        lane_checks.append(dict(id=name,world_index=k,canonical_index=int(layout.canonical_indices(chart.world_indices)[k]),
            endpoint_change_scene_units=float(np.linalg.norm(after[:,1]-before[:,1])),provider_render_same_endpoint=True,
            endpoint_FIT_identified=False,free_parameter_not_an_observation=True))
    q0_changes=[]
    old_rows={r['id']:r for r in old_factor.rows+old_factor.held}
    for row in new_factor.rows+new_factor.held:
        if row.get('initialization'):
            change=float(abs(new_factor._evaluate(q,row)[1]-old_factor._evaluate(oldq,old_rows[row['id']])[1]).max())
            q0_changes.append(dict(id=row['id'],split=row['split'],maximum_projection_change_px=change))
    new_columns=layout.canonical_indices(chart.world_indices)[66:]
    report=dict(schema='rinkspace.actual-complete-source-unlocked-physical-geometry-check.v1',
        state_path=str(state_path),state_npz_sha256=sha(state_path),q_sha256=token(canonical_q),old_q_sha256=token(original_q),
        original_complete_34306_parameters_bitexact=True,profile_arrays_preserved=len(profiles),
        prior_target_quadratic_scales_retained_as_inheritance_only=True,new_target_full_normal_or_cost_not_evaluated=True,
        physical_parameter_count=provider.parameter_count,complete_parameter_count=len(canonical_q),camera_archive_reproduce_max=camera_error,
        strict_half_circle_crease=bool(args.strict_half_circle_crease),
        derive_hockey_center_from_course=bool(args.derive_hockey_center_from_course),
        completed_world_checks_reused_after_report_only_failure=bool(args.reuse_completed_world_checks),
        old_camera_archive_reproduce_max=old_camera_error,coordinate_layout=layout.manifest(),
        parent44_source_residual_max_difference=prefix_r_difference,parent44_source_J_max_difference=prefix_J_max,
        track_source_FIT_scalar_rows=new_factor.row_count,track_J_shape=list(new_block.J.shape),track_coupled_FD_max_error=fd_error,
        track_all98_world_directional_FD_max_error=world_direction_error,all98_world_column_FD_checks=all_world_checks,
        actual_FD_values_bypass_declared_activity_and_factor_result_caches=True,
        edge_J_shape=list(eb.J.shape),edge_coupled_FD_max_error=edge_error,edge_FD_allowed=edge_allowed,edge_support_step=branch,
        new_geometry_columns=[dict(canonical_index=int(k),track_J_nnz=int(mapped[:,k].nnz),
            edge_J_nnz=int(eb.J[:,layout.canonical_to_evaluation[k]].nnz))for k in new_columns],
        q0_old_bad_material_operator_projection_changes=q0_changes,print_source_export_world_max_difference=export_error,
        lane_endpoint_checks=lane_checks,world_proof=proof,motion_graph_loaded=False,full_solver_started=False,accuracy_accepted=False)
    path=args.output/'actual_complete_q_unlocked_geometry_check.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=numeric_json)+'\n',encoding='utf8')
    (args.output/'scene_initializer_UNACCEPTED.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2,default=numeric_json)+'\n',encoding='utf8')
    (args.output/'small_actual_source_review_UNACCEPTED.json').write_text(json.dumps(dict(tracks=new_factor.review(q),edges=edges.review(q)),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(path),q_sha256=report['q_sha256'],track_FD=fd_error,edge_FD=edge_error,new_geometry_columns=report['new_geometry_columns'])),flush=True)

if __name__=='__main__':main()
