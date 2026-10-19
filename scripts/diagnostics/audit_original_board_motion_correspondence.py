"""Retest saved board FIT endpoints on the exact original native presentations.

Streams two native frames at a time. Deep LK/FB is diagnostic evidence only:
the source UV, FIT roles, budgets, profiles and saved state are never changed.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from audit_production_motion_coverage import read,ref,sha
from replay_saved_motion_material_branches import stats
import av,cv2,numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('state-directory','source-replay-module','coverage-report','production-replay-report','diagnostic-review','source','native-pair-bindings','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--shot',type=int,default=8)
    a=ap.parse_args();began=time.perf_counter();state=a.state_directory.resolve();run=state.parent.parent
    if a.output.exists():raise FileExistsError(a.output)
    identity=read(state/'state_identity.json');coverage=read(a.coverage_report);prior=read(a.production_replay_report);review=read(a.diagnostic_review)
    if not(identity['q_sha256']==coverage['q_sha256']==prior['q_sha256']==review['state_identity']['q_sha256']):raise ValueError('Saved diagnostic states differ')
    for p,wanted in [(state/'complete_state_UNACCEPTED.npz',identity['state_npz_sha256']),(state/'scene_UNACCEPTED.json',identity['scene_sha256']),
                     (Path(identity['camera_path']),identity['camera_sha256']),(run/'input_manifest.json',identity['input_manifest_sha256']),
                     (Path(review['source_tracks_path']),review['source_tracks_sha256'])]:
        if sha(p)!=wanted:raise ValueError(('Changed immutable input',str(p)))
    source_stat=a.source.stat();source_digest=sha(a.source)
    expected_source='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
    if source_digest!=expected_source:raise ValueError('Actual whole source file SHA mismatch')
    shot=next(r for r in coverage['shots']if r['shot_index']==a.shot)
    if shot['finite_other_surface_points']or len(shot['per_block'])!=1 or shot['per_block'][0]['block']!='low':raise ValueError('This diagnostic explicitly requires one low chart containing board-only finite rows')
    spec=importlib.util.spec_from_file_location('rs_original_source_replay',a.source_replay_module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.STATE=state;old.RUN=run
    bp=a.source_replay_module.parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in read(bp)['models']if r['model']=='699')['pairs']
    blocks,checks,authorities=old.reconstruct(read(run/'input_manifest.json'),read(run/'actual_original_motion_input_compaction.json'),bindings)
    source=blocks['low']
    with np.load(state/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        names=z['aux__block_names'].tolist();lengths=z['aux__block_lengths']//4;cursor=sum(lengths[:names.index('low')]);stamps=z['aux__canonical_pts'][cursor:cursor+lengths[names.index('low')]].copy()
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={int(p):c for p,c in zip(z['pts'],z['camera25'])}
    border=np.asarray(read(state/'scene_UNACCEPTED.json')['ice_boundary']['points_xyz'])
    pa,pb=stamps[source['ia']],stamps[source['ib']]
    chosen=np.flatnonzero(source['finite_mask']&(pa>=shot['start_pts'])&(pa<shot['stop_pts_exclusive'])&(pb>=shot['start_pts'])&(pb<shot['stop_pts_exclusive']))
    if len(chosen)!=shot['finite_board_points']:raise ValueError('Exact board FIT count changed')
    raw=source['active_to_raw'][chosen];native_bindings=[r for r in read(a.native_pair_bindings)if r['cut']==a.shot]
    groups=[];seen=[]
    for binding in native_bindings:
        hit=chosen[(raw>=binding['global_retained_start'])&(raw<binding['global_retained_end_exclusive'])]
        if not len(hit):continue
        native_a,native_b=map(int,binding['native_pair_pts']);actual_a,actual_b=map(int,binding['canonical_pair_pts'])
        if native_b-native_a!=256 or actual_a==actual_b:raise ValueError('Unexpected original native edge')
        if not np.all(pa[hit]==actual_a)or not np.all(pb[hit]==actual_b):raise ValueError('Native/actual source binding differs from exact runtime input')
        groups.append(dict(native_source_pts=native_a,native_target_pts=native_b,actual_source_pts=actual_a,actual_target_pts=actual_b,indices=hit))
        seen.extend(hit.tolist())
    if sorted(seen)!=chosen.tolist():raise ValueError('Original native presentations do not bind every exact board FIT row once')
    group_by_target={r['native_target_pts']:r for r in groups}
    if len(group_by_target)!=len(groups):raise ValueError('Original target native presentation is ambiguous')
    wanted={r[key]for r in groups for key in ('native_source_pts','native_target_pts')}
    receipt_path=Path(review['source_tracks_path']).parent/'source_extraction_receipt.json';deep_receipt=read(receipt_path)
    settings=dict(winSize=(int(deep_receipt['LK_window_native_px']),)*2,maxLevel=int(deep_receipt['LK_pyramid_levels']),
                  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,int(deep_receipt['LK_max_iterations']),float(deep_receipt['LK_epsilon'])))
    fb_gate=float(deep_receipt['FB_gate_native_px']);cv2.setNumThreads(1)
    original_pairs={tuple((r['source_pts'],r['target_pts'])):r for r in next(s for s in prior['shots']if s['shot_index']==a.shot)['finite_all_original_pairs']}
    independent_pairs={(r['source_pts'],r['target_pts']):r for r in review['pairs']if r['shot_index']==a.shot}
    if set(independent_pairs)!=set(original_pairs):raise ValueError('Independent v2 and original FIT actual-pair coverage differs')
    rows=[];parts={};processed=set();gray_cache={};decoded=[]
    def collect(key,value):parts.setdefault(key,[]).append(np.asarray(value).reshape(-1))
    with av.open(str(a.source))as container:
        stream=container.streams.video[0];stream.thread_count=2;container.seek(min(wanted)-15360,stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:continue
            if frame.pts>max(wanted):break
            if frame.pts not in wanted:continue
            if str(frame.time_base)!='1/15360':raise ValueError('Original native source clock changed')
            bgr=frame.to_ndarray(format='bgr24')
            if bgr.shape!=(1080,1920,3):raise ValueError('Original native source domain changed')
            gray_cache[frame.pts]=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);decoded.append(int(frame.pts))
            group=group_by_target.get(frame.pts)
            if group is not None:
                if group['native_source_pts']not in gray_cache:raise ValueError('Exact original native source frame not decoded')
                idx=group['indices'];aa=source['a'][idx];bb=source['b'][idx];ga=gray_cache[group['native_source_pts']];gb=gray_cache[group['native_target_pts']]
                start=aa.astype(np.float32).reshape(-1,1,2)
                target,status,_=cv2.calcOpticalFlowPyrLK(ga,gb,start,None,**settings)
                back,back_status,_=cv2.calcOpticalFlowPyrLK(gb,ga,target,None,**settings)
                target=target.reshape(-1,2);back=back.reshape(-1,2);fb=np.linalg.norm(back-aa,axis=1)
                good=(status.ravel()!=0)&(back_status.ravel()!=0)&np.isfinite(target).all(1)&np.isfinite(fb)&(fb<fb_gate)
                good&=(target[:,0]>1)&(target[:,0]<1918)&(target[:,1]>1)&(target[:,1]<1078)
                ca,cb=cameras[group['actual_source_pts']],cameras[group['actual_target_pts']]
                xyz=old.current_far_board_points(ca,aa,border);pred,_,valid=old.project_central_valid_full_brown(cb,xyz,with_jacobian=False)
                if not valid.all():raise ValueError('Existing board model source domain changed')
                old_r=pred-bb;new_r=pred-target;delta=target-bb;flow=bb-aa;energy=np.sum(flow*flow);dot=np.sum(old_r*flow)
                angle=np.divide(np.einsum('ij,ij->i',target-aa,flow),np.linalg.norm(target-aa,axis=1)*np.linalg.norm(flow,axis=1),out=np.full(len(idx),np.nan),where=(np.linalg.norm(target-aa,axis=1)*np.linalg.norm(flow,axis=1))>1e-8)
                collect('all_original_finite_residual_px',np.linalg.norm(old_r,axis=1))
                collect('deep_LK_FB_qualified_target_minus_original_target_px',np.linalg.norm(delta[good],axis=1))
                collect('deep_LK_FB_qualified_residual_px',np.linalg.norm(new_r[good],axis=1))
                collect('matched_qualified_original_residual_px',np.linalg.norm(old_r[good],axis=1))
                collect('deep_LK_FB_qualified_motion_direction_cosine',angle[good])
                collect('deep_LK_FB_qualified_FB_px',fb[good])
                collect('deep_LK_FB_status_all_rows_indicator',good.astype(int))
                key=(group['actual_source_pts'],group['actual_target_pts']);orig=original_pairs[key];ind=independent_pairs[key]
                if abs(np.median(np.linalg.norm(old_r,axis=1))-orig['finite_residual_native_px']['p0_p10_p50_p90_p95_p100'][2])>1e-6:raise ValueError('Old finite public transfer changed')
                rows.append(dict(**{k:v for k,v in group.items()if k!='indices'},points=len(idx),deep_source_FB_qualified_points=int(good.sum()),
                    source_UV_exact_original=True,source_native_alias_different=group['native_source_pts']!=group['actual_source_pts'],
                    target_native_alias_different=group['native_target_pts']!=group['actual_target_pts'],
                    original_finite_residual_px=stats(np.linalg.norm(old_r,axis=1)),
                    deep_matched_original_residual_px=stats(np.linalg.norm(old_r[good],axis=1)),
                    deep_matched_new_residual_px=stats(np.linalg.norm(new_r[good],axis=1)),
                    deep_target_minus_original_target_px=stats(np.linalg.norm(delta[good],axis=1)),
                    deep_FB_px=stats(fb[good]),original_observed_motion_median_xy_px=np.median(flow,axis=0).tolist(),
                    deep_observed_motion_median_xy_px=np.median((target-aa)[good],axis=0).tolist()if good.any()else None,
                    original_residual_median_xy_px=np.median(old_r,axis=0).tolist(),deep_residual_median_xy_px=np.median(new_r[good],axis=0).tolist()if good.any()else None,
                    original_motion_energy_px2=float(energy),original_residual_dot_motion_px2=float(dot),
                    original_energy_weighted_relative_along_motion=float(dot/energy)if energy else None,
                    deep_v2_independent_origin_pair=ind))
                processed.add(frame.pts)
            # Original source edges are adjacent native presentations. Preserve
            # the current source frame for the next edge, with no image writes.
            for stamp in list(gray_cache):
                if stamp<frame.pts:del gray_cache[stamp]
    if processed!=set(group_by_target)or set(decoded)!=wanted:raise ValueError('Exact native source-pair evidence is incomplete')
    # Reading can update access time; content identity uses inode, size and
    # modification time, matching the authenticated source-cache helper.
    stat_identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if stat_identity(a.source.stat())!=stat_identity(source_stat):raise ValueError('Actual source file changed during streaming audit')
    compare=[]
    for row in rows:
        original=row['original_finite_residual_px']['p0_p10_p50_p90_p95_p100'][2];ind=row['deep_v2_independent_origin_pair']['error_native_px']['median']
        compare.append((original,ind))
    values=np.asarray(compare);summary={key:stats(np.concatenate(value))for key,value in parts.items()}
    result=dict(schema='rinkspace.exact-original-native-FIT-board-UV-deep-LK-readonly.v1',q_sha256=identity['q_sha256'],shot_index=a.shot,
        state_identity=ref(state/'state_identity.json'),coverage_report=ref(a.coverage_report),production_replay_report=ref(a.production_replay_report),
        diagnostic_review=ref(a.diagnostic_review),diagnostic_source_receipt=ref(receipt_path),native_pair_bindings=ref(a.native_pair_bindings),
        original_source_arrays_authenticated=checks,source_authorities=authorities,source=dict(path=str(a.source),sha256=source_digest,bytes=source_stat.st_size,mtime_ns=source_stat.st_mtime_ns),
        script=ref(Path(__file__)),source_replay_module=ref(a.source_replay_module),LK_settings=dict(window_native_px=settings['winSize'][0],maxLevel=settings['maxLevel'],maxIterations=settings['criteria'][1],epsilon=settings['criteria'][2],FB_gate_native_px=fb_gate),
        all_original_actual_pairs=len(rows),all_original_finite_board_points=len(chosen),decoded_native_presentations=len(decoded),
        native_alias_different_pair_count=sum(r['source_native_alias_different']or r['target_native_alias_different']for r in rows),
        both_native_and_actual_exposure_identity_preserved=True,summaries=summary,
        pair_median_native_px_original=stats(values[:,0]),pair_median_native_px_v2_different_GFTT_origins=stats(values[:,1]),
        pair_median_original_v2_correlation=float(np.corrcoef(values.T)[0,1]),pair_median_v2_minus_original_px=stats(values[:,1]-values[:,0]),
        pair_count_original_below_half_px_v2_above_one_px=int(np.sum((values[:,0]<.5)&(values[:,1]>1.))),
        original_sigma_is_saved_pair_budget_not_calibrated_pixel_measurement_noise=True,
        deep_LK_FB_is_diagnostic_not_new_FIT_or_material_authority=True,independent_v2_different_GFTT_origins_not_same_pixel_identity=True,
        original_UV_clock_sigma_profile_material_or_configs_changed=False,new_source_images_written=0,full_J_H_graph_or_fitting_performed=False,
        rows=rows,seconds=time.perf_counter()-began)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(a.output),points=len(chosen),pairs=len(rows),summaries=summary,seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
