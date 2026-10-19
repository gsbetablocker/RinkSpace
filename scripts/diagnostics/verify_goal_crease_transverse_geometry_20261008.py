"""Verify the actual two-column crease placement extension on saved sources."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import numpy as np
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def token(value):return hashlib.sha256(np.asarray(value,dtype=np.float64).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','state','auxiliary','ice-packet','output'):
        ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--wide-check',action='store_true')
    a=ap.parse_args()
    keys=('q','aux__canonical_pts','aux__block_lengths','aux__block_start','aux__block_names',
          'aux__installation_xyz_slots','aux__current_shared_mount_parameter_indices',
          'aux__current_all_installation_optics_parameter_indices','aux__correlated_roll_prior_contract_json',
          'aux__current_complete_parameter_layout_json','aux__all_installation_optics_contract_json')
    with np.load(a.state,allow_pickle=False)as z:arrays={k:z[k].copy()for k in keys}
    with np.load(a.parent,allow_pickle=False)as z:old_arrays={k:z[k].copy()for k in keys}
    assert arrays['q'].shape==(34340,) and old_arrays['q'].shape==(34338,)
    assert np.array_equal(arrays['q'][:34338],old_arrays['q']) and np.array_equal(arrays['q'][-2:],np.zeros(2))
    identity=read(a.parent.parent/'state_identity.json')
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k].copy()for k in z.files}
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    old_provider,old_world,_=current_complete_source_geometry(a.parent,a.auxiliary)
    assert provider.parameter_count==100 and old_provider.parameter_count==98
    assert np.array_equal(world[:98],old_world) and np.array_equal(world[98:],np.zeros(2))
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    old_state,old_camera_error=state_camera_adapter(old_arrays,cameras,old_provider)
    q=state.evaluation_q(arrays['q']);old_q=old_state.evaluation_q(old_arrays['q'])
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    new=NativeIceCurveFactor(state,a.ice_packet,len(q));old=NativeIceCurveFactor(old_state,a.ice_packet,len(old_q))
    baseline=new._residual(q,new.rows);old_r=old._residual(old_q,old.rows)
    assert np.array_equal(baseline,old_r)
    for lhs,rhs in zip(new.rows+new.held,old.rows+old.held):
        assert lhs['id']==rhs['id'] and lhs['split']==rhs['split'] and lhs['actual_pts']==rhs['actual_pts']
        assert np.array_equal(lhs['uv'],rhs['uv']) and np.array_equal(lhs['sigma'],rhs['sigma'])
    scene=provider.scene(world);previous=old_provider.scene(old_world)
    features={f['id']:f for f in scene['ice_features']};old_features={f['id']:f for f in previous['ice_features']}
    assert set(features)==set(old_features)
    compared=0
    for identity,f in features.items():
        for field in ('center_xyz','radius','polygon_xyz','segments_xyz','paint_segments_xyz'):
            if field in f:
                assert np.array_equal(f[field],old_features[identity][field]),(identity,field)
                compared+=1
    view=CurrentIceMaterialView(provider.current)
    for side in ('left','right'):
        assert np.array_equal(view.features(world)['goal_crease_'+side]['polygon_xyz'],features['goal_crease_'+side]['polygon_xyz'])
    block=new.block(q,True);columns=[]
    for side,logical in provider.current.goal_crease_transverse_indices.items():
        col=int(state.geometry_parameter_indices[logical]);direction=np.zeros(len(q));direction[col]=1.
        h=1e-5;plus=new._residual(q+h*direction,new.rows);minus=new._residual(q-h*direction,new.rows)
        observed=(plus-minus)/(2*h);predicted=block.J[:,col].toarray().ravel()
        error=float(abs(observed-predicted).max());assert error<2e-3,(side,error)
        columns.append(dict(side=side,logical_world_column=int(logical),evaluation_column=col,
                            canonical_column=int(state.canonical_parameter_layout.evaluation_to_canonical[col]),
                            actual_native_FD_max_error=error,J_max=float(abs(predicted).max())))
    changes=[]
    for side,offset in (('left',-.173),('right',.05)):
        changed=world.copy();changed[provider.current.goal_crease_transverse_indices[side]]=offset
        full={f['id']:f for f in provider.scene(changed)['ice_features']};current=view.features(changed)
        target='goal_crease_'+side
        for fid in features:
            for field in ('center_xyz','radius','polygon_xyz','segments_xyz','paint_segments_xyz'):
                if field not in features[fid]:continue
                assert np.array_equal(full[fid][field],current[fid][field]),(fid,field,'source/export')
                if fid!=target:assert np.array_equal(full[fid][field],features[fid][field]),(fid,field,'unrelated')
        moved=np.asarray(full[target]['polygon_xyz'])-np.asarray(features[target]['polygon_xyz'])
        across=np.array([-np.sin(world[2]),np.cos(world[2]),0.])
        assert np.max(abs(moved-offset*across/provider.scene_unit_m))<1e-12
        candidate=q.copy();candidate[state.geometry_parameter_indices[provider.current.goal_crease_transverse_indices[side]]]=offset
        changes.append(dict(side=side,offset_working_m=offset,actual_source_rows=new.review(candidate)['rows']))
    wide_check=None
    if a.wide_check:
        from structured_programme_source_candidate.current_wide_source import CurrentWideSource
        from wide_anchor_candidate.wide_high_source_factor import WORLD_INDICES as OLD_WORLD_INDICES
        from wide_anchor_candidate.wide_high_free_optics_source_factor import WideHighFreeOpticsSourceFactor
        optics=json.loads(str(arrays['aux__all_installation_optics_contract_json'].item()))
        ids=json.loads(str(arrays['aux__correlated_roll_prior_contract_json'].item()))['installation_order']
        secondary=ids[1]
        original_state=SimpleNamespace(q=arrays['q'][:34225],ids=ids,slots={secondary:np.array([8,9,10])},
            geometry=SimpleNamespace(parameter_count=23,scene=lambda unused:provider.current.scene(world)),
            geometry_parameter_indices=OLD_WORLD_INDICES,
            installation_optics={secondary:SimpleNamespace(mount_indices=np.array([34198,34199,34200]))},optics_contract=optics)
        original=WideHighFreeOpticsSourceFactor(original_state)
        trace=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json')
        wide=CurrentWideSource(original,state,read(trace))
        core_count=state.structured_programme_coordinate_chart.core_count
        private=q[np.asarray(state.structured_programme_coordinate_chart.high_indices,int)]
        core=q[:core_count].copy()
        wr,wJ,_=wide.evaluate(core,private,True)
        saved=np.asarray(read(a.parent.parent/'source_review_UNACCEPTED.json')['wide']['source_normalized_residual'])
        assert np.array_equal(wr,saved) and len(wr)==5808
        wide_columns=[]
        for side,logical in provider.current.goal_crease_transverse_indices.items():
            col=int(state.geometry_parameter_indices[logical]);direction=np.zeros(len(core));direction[col]=1.
            h=1e-5
            observed=(wide.evaluate(core+h*direction,private,False)[0]-wide.evaluate(core-h*direction,private,False)[0])/(2*h)
            predicted=wJ[:,col].toarray().ravel();error=float(abs(observed-predicted).max())
            assert error<2e-3,(side,error)
            wide_columns.append(dict(side=side,logical_world_column=int(logical),actual_FD_max_error=error,J_max=float(abs(predicted).max())))
        assert wide_columns[0]['J_max']==0. and wide_columns[1]['J_max']>0.
        wide_check=dict(actual_unique_scalar_rows=5808,zero_new_coordinates_saved_source_r_bitexact=True,
                        new_column_FD_checks=wide_columns,only_observed_right_crease_has_wide_source_derivative=True)
    report=dict(schema='rinkspace.actual-saved-source-two-crease-placement-columns-check.v1',
        parent=dict(path=str(a.parent),sha256=sha(a.parent),q_sha256=token(old_arrays['q'])),
        initializer=dict(path=str(a.state),sha256=sha(a.state),q_sha256=token(arrays['q'])),
        old34338_complete_parameters_bitexact=True,added_canonical_columns=[34338,34339],
        old98_world_parameters_bitexact=True,zero_initialized_new_coordinates=True,
        physical_parameter_count=100,complete_parameter_count=34340,
        camera_archive_reproduce_max=camera_error,old_camera_archive_reproduce_max=old_camera_error,
        source_packet=dict(path=str(a.ice_packet),sha256=sha(a.ice_packet)),
        native412_residual_zero_extension_bitexact=True,old_UV_sigma_roles_and_clock_preserved=True,
        unchanged_numeric_ice_feature_fields=compared,new_column_actual_FD_checks=columns,
        source_export_same_crease_geometry=True,only_requested_crease_polygon_moved=True,
        common_radius_goal_trapezoids_faceoff_and_other_ice_geometry_unchanged=True,
        offset_source_reviews=changes,world_proof=proof,
        actual_wide_source_check=wide_check,
        source_motion_graph_not_loaded=True,full_normal_not_evaluated=True,whole_video_accuracy_accepted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(a.output),sha256=sha(a.output),new_columns=columns,
                         zero_native_residual_bitexact=True,export_source_bitexact=True)),flush=True)


if __name__=='__main__':main()
