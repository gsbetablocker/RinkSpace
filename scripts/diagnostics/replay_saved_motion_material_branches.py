"""Replay complete declared cuts' original finite/unknown motion, without J/fit."""
from pathlib import Path
import argparse,importlib.util,json,sys,time,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from audit_production_motion_coverage import read,ref,sha
import numpy as np

def stats(values):
    x=np.asarray(values,float).reshape(-1);finite=x[np.isfinite(x)]
    return dict(count=len(x),nonfinite=int(len(x)-len(finite)),p0_p10_p50_p90_p95_p100=np.quantile(finite,[0,.1,.5,.9,.95,1]).tolist()if len(finite)else None)
def summary(parts):
    out={}
    for key,values in parts.items():
        x=np.concatenate(values)if values else np.empty(0)
        out[key]=stats(x)
    return out
def record(parts,key,value):parts.setdefault(key,[]).append(np.asarray(value).reshape(-1))
def fields(parts,a,b,r,sigma,weights):
    flow=b-a;length=np.linalg.norm(flow,axis=1);good=length>1e-8
    along=np.divide(np.einsum('ij,ij->i',r,flow),length,out=np.full(len(length),np.nan),where=good)
    record(parts,'actual_source_flow_norm_native_px',length)
    record(parts,'actual_native_residual_norm_px',np.linalg.norm(r,axis=1))
    record(parts,'signed_residual_along_observed_motion_native_px',along)
    record(parts,'relative_signed_along_observed_motion',np.divide(along,length,out=np.full(len(length),np.nan),where=good))
    record(parts,'source_sigma_native_px_components',sigma)
    record(parts,'actual_saved_frozen_profile_weight_components',weights)
    record(parts,'sqrt_frozen_weight_sigma_normalized_residual_norm',np.linalg.norm(np.sqrt(weights)*r/sigma,axis=1))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state-directory',type=Path,required=True)
    ap.add_argument('--source-replay-module',type=Path,required=True)
    ap.add_argument('--coverage-report',type=Path,required=True)
    ap.add_argument('--shot',type=int,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();began=time.perf_counter();state=args.state_directory.resolve();run=state.parent.parent
    identity=read(state/'state_identity.json');manifest=read(run/'input_manifest.json');coverage=read(args.coverage_report)
    if coverage['q_sha256']!=identity['q_sha256']:raise ValueError('Coverage and residual audit states differ')
    for record0 in (coverage['saved_state'],coverage['complete_state'],coverage['actual_runtime_manifest']):
        if sha(record0['path'])!=record0['sha256']:raise ValueError('Saved source coverage reference changed')
    spec=importlib.util.spec_from_file_location('rs_original_source_replay',args.source_replay_module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.STATE=state;old.RUN=run
    cv2=old.cv2
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
    from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
    from types import SimpleNamespace
    binding_path=args.source_replay_module.parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in read(binding_path)['models']if r['model']=='699')['pairs']
    blocks,checks,authorities=old.reconstruct(manifest,read(run/'actual_original_motion_input_compaction.json'),bindings)
    with np.load(state/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        names=z['aux__block_names'].tolist();lengths=z['aux__block_lengths']//4;canonical=z['aux__canonical_pts'].copy()
        weights={name:z['profile__weights_'+str(i)].copy()for i,name in enumerate(names)}
        domain=json.loads(str(z['aux__current_depth_and_material_domain_contract_json'].item()))
    stamps={};cursor=0
    for name,length in zip(names,lengths):stamps[name]=canonical[cursor:cursor+length];cursor+=int(length)
    floors={r['block']:r['candidate_minimum_source_Z_m']/domain['scene_unit_metres']for r in domain['blocks']}
    with np.load(identity['camera_path'],allow_pickle=False)as z:camera_pts=z['pts'].copy();cameras=z['camera25'].copy()
    scene=read(state/'scene_UNACCEPTED.json');border=np.asarray(scene['ice_boundary']['points_xyz']);height=float(np.asarray(scene['board_surface']['top_edge_xyz'])[:,2].max())
    results=[]
    for shot in coverage['shots']:
        if shot['shot_index']not in set(args.shot):continue
        if shot['finite_other_surface_points']:raise ValueError('This explicit comparator requires only board finite rows in the selected whole cut')
        finite_parts={};unknown_parts={};perpair=[];finite_checks=[];zero_weight_count=0
        for block_record in shot['per_block']:
            name=block_record['block'];source=blocks[name];ss=stamps[name];ia,ib=source['ia'],source['ib'];pa,pb=ss[ia],ss[ib]
            indices=np.flatnonzero((pa>=shot['start_pts'])&(pa<shot['stop_pts_exclusive'])&(pb>=shot['start_pts'])&(pb<shot['stop_pts_exclusive']))
            known=source['finite_mask'][indices];fi=indices[known];ui=indices[~known]
            if len(fi)!=block_record['finite_board_points']or len(ui)!=block_record['unknown_depth_points']:raise ValueError('Whole-cut exact motion masks differ from coverage proof')
            sigma=source['motion_sigma'].reshape(-1,2);w=weights[name];zero_weight_count+=int(np.count_nonzero(w[indices]==0))
            cams=cameras[np.searchsorted(camera_pts,ss)];rot=np.asarray([cv2.Rodrigues(c[3:6])[0]for c in cams])
            if not np.array_equal(camera_pts[np.searchsorted(camera_pts,ss)],ss):raise ValueError('Camera export and production exposure indices differ')
            if not np.all(cams[:,8:]==cams[0,8:]):raise ValueError('Whole-cut optics are not shared as required by the public native kernel')
            def native(idx,planes=None,at_infinity=False):
                aa,bb=source['a'][idx],source['b'][idx];ca,cb=cams[ia[idx]],cams[ib[idx]]
                return old.native_motion(a=aa,b=bb,fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),ra=rot[ia[idx]],rb=rot[ib[idx]],
                    da=None,db=None,ca=ca[:,:3],cb=ca[:,:3]if at_infinity else cb[:,:3],
                    dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,pp=cams[0,8:10],k1=cams[0,11],
                    planes=np.zeros((len(idx),4))if planes is None else planes,jacobian=False,sigma=sigma[idx],profile_weight=w[idx],
                    minimum_source_depth_scene_units=floors[name],return_trace=True)
            # All retained unknown FIT rows, in bounded in-memory chunks only.
            for offset in range(0,len(ui),20000):
                idx=ui[offset:offset+20000];r,check,trace=native(idx);r=r.reshape(-1,2)
                ri,_,_=native(idx,at_infinity=True);ri=ri.reshape(-1,2)
                fields(unknown_parts,source['a'][idx],source['b'][idx],r,sigma[idx],w[idx])
                record(unknown_parts,'infinity_rotation_zoom_residual_norm_px',np.linalg.norm(ri,axis=1))
                record(unknown_parts,'free_depth_absorption_native_px',np.linalg.norm(r-ri,axis=1))
                flow=source['b'][idx]-source['a'][idx];length=np.linalg.norm(flow,axis=1)
                along=np.divide(np.einsum('ij,ij->i',ri,flow),length,out=np.full(len(idx),np.nan),where=length>1e-8)
                record(unknown_parts,'infinity_signed_residual_along_source_flow_px',along)
                depth=np.divide(1.-trace[:,1],trace[:,1],out=np.full(len(idx),np.inf),where=trace[:,1]>0)
                record(unknown_parts,'profiled_positive_depth_scene_units',depth)
                record(unknown_parts,'unknown_interior_profile_indicator',trace[:,9])
            # Every finite pair, using the original public current-facet selector.
            if len(fi):
                order=np.lexsort((ib[fi],ia[fi]));ordered=fi[order]
                ends=np.r_[0,np.flatnonzero((np.diff(ia[ordered])!=0)|(np.diff(ib[ordered])!=0))+1,len(ordered)]
                for left,right in zip(ends[:-1],ends[1:]):
                    idx=ordered[left:right];ca,cb=cams[ia[idx[0]]],cams[ib[idx[0]]];aa,bb=source['a'][idx],source['b'][idx]
                    course=SimpleNamespace(sample_perimeter=lambda q,with_jacobian=False:(border,))
                    selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(idx),source_row_groups={0:np.arange(len(idx))},
                        source_cameras={0:ca},source_uv=aa,exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
                    facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('saved_course',old.token(border)))
                    edge=border[:-1]if np.array_equal(border[0],border[-1])else border;zero=np.zeros((3,0))
                    planes=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
                    r,check,trace=native(idx,planes);r=r.reshape(-1,2)
                    if check['finite_rows']!=len(idx):raise ValueError('Public kernel did not dispatch all declared board points to finite branch')
                    xyz=old.current_far_board_points(ca,aa,border);pred,depth,valid=old.project_central_valid_full_brown(cb,xyz,with_jacobian=False)
                    difference=float(np.max(abs(r-(pred-bb)),initial=0.))
                    if difference>1e-6 or not valid.all():raise ValueError('Original finite kernel differs from saved current-board full-Brown transfer')
                    target_xyz=old.current_far_board_points(cb,bb,border)
                    fields(finite_parts,aa,bb,r,sigma[idx],w[idx])
                    record(finite_parts,'source_intersection_z_scene_units',xyz[:,2]);record(finite_parts,'target_intersection_z_scene_units',target_xyz[:,2])
                    record(finite_parts,'source_or_target_outside_current_board_height_indicator',((xyz[:,2]<0)|(xyz[:,2]>height)|(target_xyz[:,2]<0)|(target_xyz[:,2]>height)).astype(int))
                    record(finite_parts,'conditional_same_material_world_point_disagreement_scene_units',np.linalg.norm(xyz-target_xyz,axis=1))
                    flow=bb-aa;length=np.linalg.norm(flow,axis=1);along=np.divide(np.einsum('ij,ij->i',r,flow),length,out=np.full(len(idx),np.nan),where=length>1e-8)
                    perpair.append(dict(block=name,source_pts=int(pa[idx[0]]),target_pts=int(pb[idx[0]]),points=len(idx),
                        finite_residual_native_px=stats(np.linalg.norm(r,axis=1)),signed_along_observed_motion_px=stats(along),
                        relative_along_observed_motion=stats(np.divide(along,length,out=np.full(len(idx),np.nan),where=length>1e-8)),
                        source_sigma=stats(sigma[idx]),frozen_profile_weights=stats(w[idx]),
                        maximum_public_native_vs_current_board_difference_px=difference))
                    finite_checks.append(difference)
        results.append(dict(shot_index=shot['shot_index'],source_camera=shot['source_camera'],start_pts=shot['start_pts'],stop_pts_exclusive=shot['stop_pts_exclusive'],
            finite_board_points=shot['finite_board_points'],unknown_points=shot['unknown_depth_points'],finite_board_pairs=shot['finite_board_actual_pairs'],
            exact_production_sigma_profile_and_masks_preserved=True,finite=summary(finite_parts),unknown=summary(unknown_parts),
            finite_all_original_pairs=perpair,zero_saved_profile_components=zero_weight_count,
            maximum_native_vs_public_current_board_transfer_difference_px=max(finite_checks,default=0.),
            current_board_height_scene_units=height))
        print(json.dumps(dict(stage='complete_whole_cut_residual_replay',shot=shot['shot_index'],finite=shot['finite_board_points'],unknown=shot['unknown_depth_points'],seconds=time.perf_counter()-began)),flush=True)
    result=dict(schema='rinkspace.complete-cut-production-finite-and-unknown-motion-material-residual-replay-readonly.v1',
        q_sha256=identity['q_sha256'],state_identity=ref(state/'state_identity.json'),coverage_report=ref(args.coverage_report),
        source_reconstruction_module=ref(args.source_replay_module),script=ref(Path(__file__)),
        all_original_source_arrays_exact_reconstructed=checks,source_authorities=authorities,
        production_role_contract=manifest['original_motion_far_board_role_correction'],shots=results,
        rho0_rotation_zoom_is_counterfactual_decomposition_not_new_fit_or_truth=True,
        unknown_materials_not_reclassified_as_board=True,declared_board_material_identity_is_existing_source_certificate_not_new_visual_confirmation=True,
        finite_current_planes_are_geometry_not_measured_XYZ=True,
        full_source_graph_or_J_or_H_or_fitting_loaded=False,new_source_images_written=0,
        original_source_UV_sigma_clock_roles_profiles_configs_or_saved_states_changed=False,
        source_residual_or_profile_weight_statistics_do_not_prove_whole_cut_visual_accuracy=True,seconds=time.perf_counter()-began)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(args.output),seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
