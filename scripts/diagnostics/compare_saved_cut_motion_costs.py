"""Compare complete saved-cut material costs using unchanged production inputs.

No source graph, J, fitting, image decoding or image cache is constructed.
Both states use the saved direction's exact frozen profile and Huber2 loss.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from audit_production_motion_coverage import read,ref,sha
from replay_saved_motion_material_branches import stats
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('before-state','after-state','direction','source-replay-module','coverage-report','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--shot',type=int,required=True);ap.add_argument('--accepted-alpha',type=float,required=True)
    a=ap.parse_args();started=time.perf_counter();states=[a.before_state.resolve(),a.after_state.resolve()];run=states[1].parent.parent
    if a.output.exists():raise FileExistsError(a.output)
    identities=[read(s/'state_identity.json')for s in states]
    arrays=[];cams=[];borders=[]
    for s,ident in zip(states,identities):
        for path,expected in [(s/'complete_state_UNACCEPTED.npz',ident['state_npz_sha256']),(s/'scene_UNACCEPTED.json',ident['scene_sha256']),
                              (Path(ident['camera_path']),ident['camera_sha256']),(run/'input_manifest.json',ident['input_manifest_sha256'])]:
            if sha(path)!=expected:raise ValueError(('Changed complete state reference',str(path)))
        with np.load(s/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:arrays.append({k:z[k]for k in z})
        with np.load(ident['camera_path'],allow_pickle=False)as z:cams.append(dict(pts=z['pts'],camera=z['camera25']))
        borders.append(np.asarray(read(s/'scene_UNACCEPTED.json')['ice_boundary']['points_xyz']))
    with np.load(a.direction,allow_pickle=False)as z:direction={k:z[k]for k in z}
    if not np.array_equal(direction['q'],arrays[0]['q']):raise ValueError('Saved complete direction and before state differ')
    if str(direction['frozen_profile_token'])!=identities[0]['frozen_profile_token']or identities[0]['frozen_profile_token']!=identities[1]['frozen_profile_token']:
        raise ValueError('Saved direction and complete states have different frozen profiles')
    if str(direction['source_input_manifest_sha256'])!=sha(run/'input_manifest.json'):raise ValueError('Saved direction source manifest changed')
    coverage=read(a.coverage_report);shot=next(r for r in coverage['shots']if r['shot_index']==a.shot)
    if coverage['q_sha256']!=identities[0]['q_sha256']:raise ValueError('Exact production coverage state differs')
    if shot['finite_other_surface_points']or len(shot['per_block'])!=1:raise ValueError('Comparator needs one board-only finite chart')
    name=shot['per_block'][0]['block'];spec=importlib.util.spec_from_file_location('rs_original_source_replay',a.source_replay_module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.STATE=states[1];old.RUN=run
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
    from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
    from types import SimpleNamespace
    bp=a.source_replay_module.parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in read(bp)['models']if r['model']=='699')['pairs']
    manifest=read(run/'input_manifest.json');authentication=coverage['actual_runtime_manifest']
    if sha(authentication['path'])!=authentication['sha256']:raise ValueError('Prior exact original-array authentication changed')
    certified_manifest=read(authentication['path']);reconstruction_manifest=dict(manifest)
    if 'persistent_source_replay_identity'not in manifest:
        # This fresh cycle saves no actor input-array digests. Reuse the prior
        # authenticated raw array hashes only after checking the same original
        # compaction, role packet and actual exposure contracts. Do not label
        # this a new independent hash capture from the fresh actor process.
        for key in ('original_motion_far_board_role_correction','original_motion_edge_counts','original_motion_rows','actual_exposures','native_presentations'):
            if manifest[key]!=certified_manifest[key]:raise ValueError(('Original source reconstruction contract changed',key))
        # Compaction carries state-dependent R/J validation telemetry. Compare
        # its source authority and exact row layout, not those changing numbers.
        for key in ('schema','low_candidate','main_candidate','removed_motion_points','removed_motion_scalar_rows','original_retirement_scalar_remap'):
            if manifest['original_motion_input_compaction'][key]!=certified_manifest['original_motion_input_compaction'][key]:
                raise ValueError(('Original compaction source authority changed',key))
        for before_block,after_block in zip(certified_manifest['original_motion_input_compaction']['blocks'],manifest['original_motion_input_compaction']['blocks']):
            for key in ('block','before_points','after_points','removed_points','old_scalar_start','new_scalar_start','old_scalar_rows','new_scalar_rows','exact_prefix_fields'):
                if before_block[key]!=after_block[key]:raise ValueError(('Original compaction row layout changed',key))
        reconstruction_manifest['persistent_source_replay_identity']=certified_manifest['persistent_source_replay_identity']
    blocks,checks,authorities=old.reconstruct(reconstruction_manifest,read(run/'actual_original_motion_input_compaction.json'),bindings)
    src=blocks[name];names=arrays[0]['aux__block_names'].tolist();ordinal=names.index(name);lengths=arrays[0]['aux__block_lengths']//4
    cursor=sum(lengths[:ordinal]);stamps=arrays[0]['aux__canonical_pts'][cursor:cursor+lengths[ordinal]]
    for key in ('aux__block_names','aux__block_start','aux__block_lengths','aux__canonical_pts','aux__current_evaluation_to_canonical_parameter_indices','quadratic'):
        if not np.array_equal(arrays[0][key],arrays[1][key]):raise ValueError(('Complete chart/row layout changed',key))
    if not all(np.array_equal(arrays[0]['profile__weights_'+str(i)],arrays[1]['profile__weights_'+str(i)])for i in range(len(names))):raise ValueError('Frozen profile arrays changed')
    weight=arrays[0]['profile__weights_'+str(ordinal)];sigma=src['motion_sigma'].reshape(-1,2);pa,pb=stamps[src['ia']],stamps[src['ib']]
    ids=np.flatnonzero((pa>=shot['start_pts'])&(pa<shot['stop_pts_exclusive'])&(pb>=shot['start_pts'])&(pb<shot['stop_pts_exclusive']))
    fi=ids[src['finite_mask'][ids]];ui=ids[~src['finite_mask'][ids]]
    if len(fi)!=shot['finite_board_points']or len(ui)!=shot['unknown_depth_points']:raise ValueError('Exact cut motion role counts differ')
    log=read(run/'actual_trial_family_cost_pass_0001.json');ledger=next(r for r in log['families']if r['original_row_ledger']['family']=='programme:'+name)['original_row_ledger']
    rowbegin=ledger['original_start'];motion_rows=(rowbegin+2*ids[:,None]+np.arange(2)).ravel()
    if np.any(arrays[0]['quadratic'][motion_rows]):raise ValueError('This comparator requires original Huber2 motion rows')
    residuals=[];floors=[];rot=[];ordered_cams=[]
    for arr,camera in zip(arrays,cams):
        contract=json.loads(str(arr['aux__current_depth_and_material_domain_contract_json'].item()))
        floors.append(next(r['candidate_minimum_source_Z_m']for r in contract['blocks']if r['block']==name)/contract['scene_unit_metres'])
        index=np.searchsorted(camera['pts'],stamps)
        if not np.array_equal(camera['pts'][index],stamps):raise ValueError('Saved camera and exact production exposure order differ')
        chart=camera['camera'][index];ordered_cams.append(chart);rot.append(np.asarray([old.cv2.Rodrigues(c[3:6])[0]for c in chart]))
    if floors[0]!=floors[1]:raise ValueError('Original unknown depth floor changed')
    for state_index in range(2):
        chart=ordered_cams[state_index];rr=rot[state_index];border=borders[state_index];out=np.zeros((len(src['a']),2))
        def native(index,planes=None):
            ca,cb=chart[src['ia'][index]],chart[src['ib'][index]]
            r,_=old.native_motion(a=src['a'][index],b=src['b'][index],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
                ra=rr[src['ia'][index]],rb=rr[src['ib'][index]],da=None,db=None,ca=ca[:,:3],cb=cb[:,:3],
                dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,pp=chart[0,8:10],k1=chart[0,11],
                planes=np.zeros((len(index),4))if planes is None else planes,jacobian=False,sigma=sigma[index],
                profile_weight=weight[index],minimum_source_depth_scene_units=floors[state_index])
            return r.reshape(-1,2)
        for offset in range(0,len(ui),20000):index=ui[offset:offset+20000];out[index]=native(index)
        if len(fi):
            source_groups={int(view):fi[src['ia'][fi]==view]for view in np.unique(src['ia'][fi])}
            for view,index in source_groups.items():
                ca=chart[view];course=SimpleNamespace(sample_perimeter=lambda q,with_jacobian=False:(border,))
                selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(index),source_row_groups={0:np.arange(len(index))},
                    source_cameras={0:ca},source_uv=src['a'][index],exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
                facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('saved_course',old.token(border)))
                edge=border[:-1]if np.array_equal(border[0],border[-1])else border;zero=np.zeros((3,0))
                planes=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
                out[index]=native(index,planes)
        residuals.append(out)
    def cost(r,index):
        normalized=r[index]/sigma[index];absolute=abs(normalized)
        return float(np.where(absolute<=2.,.5*absolute*absolute,2.*absolute-2.).sum())
    material=[]
    for kind,index in [('finite_board',fi),('unknown_depth',ui),('all_cut_motion',ids)]:
        before,after=[cost(r,index)for r in residuals]
        pairwise=[]
        for r in residuals:
            flow=src['b'][index]-src['a'][index];energy=float(np.sum(flow*flow));dot=float(np.sum(r[index]*flow))
            pairwise.append(dict(raw_residual_norm_px=stats(np.linalg.norm(r[index],axis=1)),
                saved_frozen_IRLS_quadratic=float(.5*np.sum(weight[index]*(r[index]/sigma[index])**2)),
                residual_dot_source_flow_px2=dot,source_flow_energy_px2=energy,energy_weighted_relative_along_motion=dot/energy))
        material.append(dict(material_branch=kind,points=len(index),source_sigma=stats(sigma[index]),frozen_weights=stats(weight[index]),
            production_Huber2_cost_before=before,production_Huber2_cost_after=after,production_Huber2_delta=after-before,states=pairwise))
    pairs=[]
    for sa,sb in sorted(set(zip(pa[ids].tolist(),pb[ids].tolist()))):
        selected=ids[(pa[ids]==sa)&(pb[ids]==sb)];record=dict(source_pts=int(sa),target_pts=int(sb))
        for kind,mask in [('finite_board',src['finite_mask'][selected]),('unknown_depth',~src['finite_mask'][selected])]:
            index=selected[mask];before,after=[cost(r,index)for r in residuals]
            record[kind]=dict(points=len(index),before=before,after=after,delta=after-before)
        pairs.append(record)
    evaluation_map=arrays[0]['aux__current_evaluation_to_canonical_parameter_indices'];starts=arrays[0]['aux__block_start'];blengths=arrays[0]['aux__block_lengths'];RF={}
    expected=direction['q']+a.accepted_alpha*direction['step'];retracted=np.flatnonzero(arrays[1]['q']!=expected)
    for block,start,length in zip(names,starts,blengths):
        columns=evaluation_map[start:start+length].reshape(-1,4);RF[block]=dict(actual_exposures=len(columns),
            changed_exposures=int(np.any(arrays[1]['q'][columns]!=arrays[0]['q'][columns],axis=1).sum()),
            maximum_difference_from_same_alpha_saved_RF4=float(np.max(abs(arrays[1]['q'][columns]-expected[columns]))))
    cutviews=np.unique(np.r_[src['ia'][ids],src['ib'][ids]]);cutcolumns=evaluation_map[starts[ordinal]+np.ravel(4*cutviews[:,None]+np.arange(4))].reshape(-1,4)
    result=dict(schema='rinkspace.complete-saved-cut-original-material-cost-direction-comparison.v1',before=ref(states[0]/'state_identity.json'),after=ref(states[1]/'state_identity.json'),
        before_q=identities[0]['q_sha256'],after_q=identities[1]['q_sha256'],source_reconstruction_module=ref(a.source_replay_module),script=ref(Path(__file__)),
        saved_complete_direction=ref(a.direction),coverage=ref(a.coverage_report),trial_family_cost_log=ref(run/'actual_trial_family_cost_pass_0001.json'),
        inherited_exact_original_arrays=checks,prior_array_authentication_manifest=authentication,
        new_run_actor_input_array_hash_capture_saved='persistent_source_replay_identity'in manifest,
        original_source_authorities=authorities,shared_frozen_profile_token=identities[0]['frozen_profile_token'],
        all_frozen_profile_arrays_identical=True,original_Huber2_motion_rows_and_source_sigmas_preserved=True,shot_index=a.shot,materials=material,actual_pairs=pairs,
        pairs_finite_decreases_unknown_increases=sum(r['finite_board']['delta']<0 and r['unknown_depth']['delta']>0 for r in pairs),
        pairs_finite_increases_unknown_decreases=sum(r['finite_board']['delta']>0 and r['unknown_depth']['delta']<0 for r in pairs),
        accepted_shared_fraction=a.accepted_alpha,all_RF4_blocks=RF,non_RF4_retracted_canonical_columns=retracted.tolist(),
        cut_actual_exposures=len(cutviews),cut_changed_exposures=int(np.any(arrays[1]['q'][cutcolumns]!=arrays[0]['q'][cutcolumns],axis=1).sum()),
        cut_RF4_exactly_same_shared_fraction=bool(np.array_equal(arrays[1]['q'][cutcolumns],expected[cutcolumns])),
        local_cost_change_not_gradient_conflict_or_whole_cut_accuracy_certificate=True,new_fitting_or_J_H_or_source_graph=False,new_source_images_written=0,seconds=time.perf_counter()-started)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(a.output),materials=material,pairs=len(pairs),RF=RF,seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
