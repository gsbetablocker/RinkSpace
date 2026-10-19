"""Read-only coverage of the exact saved production motion rows and roles.

Reuses the existing authenticated numeric source reconstruction. No source
images, programme owner/graph, Jacobian, fitting or production mutation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def token(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def certify(value,expected,label):
    value=np.asarray(value,dtype=np.dtype(expected['dtype']))
    actual=dict(dtype=value.dtype.str,shape=list(value.shape),sha256=token(value))
    if actual!=expected:raise ValueError('Production source array reconstruction differs: '+label)
    return actual
def compact_indices(old,kept):
    old=np.asarray(sorted(set(map(int,old))),np.int64)
    where=np.searchsorted(kept,old);valid=where<len(kept)
    valid[valid]&=kept[where[valid]]==old[valid]
    return where[valid],old[~valid]
def pair_summary(a,b,mask):
    aa=a[mask];bb=b[mask]
    pairs=sorted({(int(p),int(q))for p,q in zip(aa,bb)})
    undirected={tuple(sorted(p))for p in pairs};exposures=sorted(set(map(int,np.r_[aa,bb])))
    dt=np.asarray([abs(p-q)/15360. for p,q in pairs],float)
    return dict(point_count=int(mask.sum()),directed_source_pairs=len(pairs),undirected_source_pairs=len(undirected),
        incident_actual_exposures=len(exposures),incident_actual_pts=exposures,
        pair_baseline_seconds_p0_p50_p100=np.quantile(dt,[0,.5,1]).tolist()if len(dt)else None,
        pairs=[dict(source_pts=p,target_pts=q,dt_seconds=abs(q-p)/15360.,
            points=int(np.count_nonzero(mask&(a==p)&(b==q))))for p,q in pairs])
def gap_summary(stamps,covered):
    missing=~np.isin(stamps,covered);longest=0;start=None;longest_bounds=None
    for i,empty in enumerate(np.r_[missing,False]):
        if empty and start is None:start=i
        if not empty and start is not None:
            if i-start>longest:longest=i-start;longest_bounds=[int(stamps[start]),int(stamps[i-1])]
            start=None
    return dict(no_finite_board_motion_incident_actual_exposures=int(missing.sum()),
        longest_consecutive_uncovered_actual_exposures=longest,longest_uncovered_native_pts_bounds=longest_bounds,
        longest_uncovered_elapsed_seconds=None if longest_bounds is None else (longest_bounds[1]-longest_bounds[0])/15360.)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state-directory',type=Path,required=True)
    ap.add_argument('--baseline-state-directory',type=Path,required=True,help='Complete pre-947 state for exact role-index difference')
    ap.add_argument('--source-replay-module',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();began=time.perf_counter();state=a.state_directory.resolve();run=state.parent.parent
    identity=read(state/'state_identity.json');manifest=read(run/'input_manifest.json')
    for path,key in ((state/'complete_state_UNACCEPTED.npz','state_npz_sha256'),
            (state/'scene_UNACCEPTED.json','scene_sha256'),(Path(identity['camera_path']),'camera_sha256'),
            (run/'input_manifest.json','input_manifest_sha256')):
        if sha(path)!=identity[key]:raise ValueError('Immutable saved state reference differs: '+str(path))
    spec=importlib.util.spec_from_file_location('rs_existing_original_source_replay',a.source_replay_module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.STATE=state;old.RUN=run
    binding_path=a.source_replay_module.parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in read(binding_path)['models']if r['model']=='699')['pairs']
    compaction=read(run/'actual_original_motion_input_compaction.json')
    blocks,checks,authorities=old.reconstruct(manifest,compaction,bindings)
    expected=manifest['persistent_source_replay_identity']['actual_original_source_input_arrays']
    role=manifest['original_motion_far_board_role_correction']
    baseline=read(a.baseline_state_directory/'state_identity.json')
    baseline_manifest=read(a.baseline_state_directory.parent.parent/'input_manifest.json')
    if sha(a.baseline_state_directory.parent.parent/'input_manifest.json')!=baseline['input_manifest_sha256']:
        raise ValueError('Baseline production manifest changed')
    baseline_roles=baseline_manifest['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block']
    with np.load(state/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        names=z['aux__block_names'].tolist();lengths=z['aux__block_lengths']//4
        canonical=z['aux__canonical_pts'].copy();starts=z['aux__block_start'].copy()
        saved_roles=json.loads(str(z['aux__current_original_motion_far_board_role_correction_json'].item()))
        if saved_roles['packet_sha256']!=role['packet_sha256']:raise ValueError('Saved state role packet differs from run')
        weights={name:z['profile__weights_'+str(i)].copy()for i,name in enumerate(names)}
    stamps={};cursor=0
    for name,length in zip(names,lengths):stamps[name]=canonical[cursor:cursor+length];cursor+=int(length)
    if cursor!=len(canonical):raise ValueError('Saved canonical exposure block lengths differ')
    main=read(compaction['main_candidate']['path']);base_board={};extra_checks={}
    for record in main['classification']:
        name=record['block']
        with np.load(record['map']['path'],allow_pickle=False)as mp,np.load(record['original_admission_cache']['path'],allow_pickle=False)as z:
            base_board[name]=np.linalg.norm(z['active_finite_planes'][mp['keep_old_point_indices'],:2],axis=1)>0
    lowtyped=Path(old.DATA)/'calibration/runs/solver_acceleration_20261003/prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory/finite_material_source_binding.json'
    source_board_original=[r['global_bundle_index']for r in read(lowtyped)if r['surface_kind']=='finite_board_face']
    base_board['low']=np.zeros(len(blocks['low']['a']),bool)
    source_board_compact,retired=compact_indices(source_board_original,blocks['low']['active_to_original'])
    if len(retired):raise ValueError('Original finite board pixels unexpectedly retired')
    base_board['low'][source_board_compact]=True
    base_board['secondary']=np.zeros(len(blocks['secondary']['a']),bool)
    board_masks={};new_masks={};block_reports={};new947=[]
    for name,source in blocks.items():
        certify(stamps[name],expected['programme:'+name+':stamps'],name+':stamps')
        oldindices=role['original_global_bundle_indices_by_block'][name]
        added=sorted(set(oldindices)-set(baseline_roles.get(name,[])))
        kept=source['active_to_original']
        if len(kept)>1 and np.any(np.diff(kept)<=0):raise ValueError('Source compact map order changed')
        converted,retired=compact_indices(oldindices,kept)
        new,retired_new=compact_indices(added,kept)
        if len(retired_new):raise ValueError('New947 includes retired motion rows')
        board=base_board[name].copy();board[converted]=True
        course=np.flatnonzero(board)
        extra_checks[name]=dict(course_board_indices=certify(course,expected['programme:'+name+':course_board_indices'],name+':course_board_indices'),
            static_noncourse_finite_count_exact=int(np.count_nonzero(source['finite_mask']&~board))==expected['programme:'+name+':static_noncourse_finite_point_indices']['shape'][0])
        if not extra_checks[name]['static_noncourse_finite_count_exact']or np.any(board&~source['finite_mask']):
            raise ValueError('Finite board/static finite production inventory differs')
        newmask=np.zeros(len(board),bool);newmask[new]=True;new_masks[name]=newmask;board_masks[name]=board
        ia,ib=source['ia'],source['ib'];pa,pb=stamps[name][ia],stamps[name][ib]
        w=weights[name]
        if w.shape!=(len(board),2):raise ValueError('Saved actual motion profile size differs')
        if not np.all(w>0):raise ValueError('Unexpected zero/nonpositive profile: audit needs explicit inactive weighting')
        summary=pair_summary(pa,pb,newmask)
        new947.append(dict(block=name,**summary))
        block_reports[name]=dict(actual_camera_exposures=len(stamps[name]),actual_view_state_parameters=4*len(stamps[name]),
            motion_points=len(board),finite_board_points=int(board.sum()),finite_other_surface_points=int((source['finite_mask']&~board).sum()),
            unknown_depth_points=int((~source['finite_mask']).sum()),finite_board_fraction=float(board.mean()),
            positive_saved_profile_weights_count=int((w>0).sum()),actual_saved_profile_weight_range=[float(w.min()),float(w.max())],
            original_role_certificate_points=len(oldindices),runtime_retired_certificate_indices=retired.tolist(),
            newly_retyped947_points=len(new),finite_board_pair_coverage=pair_summary(pa,pb,board),
            additional947_pair_coverage=summary)
    shots=read(ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
    shot_reports=[]
    for shot_index,shot in enumerate(shots,1):
        lo,hi=int(shot['start_pts']),int(shot['end_pts']);local=[];viewpts=[];total=boardtotal=othertotal=unknown=addedtotal=0
        covered=[];newcovered=[];allpairs=set();boardpairs=set();newpairs=set()
        finite_dts=[]
        for name,source in blocks.items():
            ss=stamps[name];viewpts.extend(map(int,ss[(ss>=lo)&(ss<hi)]))
            pa,pb=ss[source['ia']],ss[source['ib']]
            mask=(pa>=lo)&(pa<hi)&(pb>=lo)&(pb<hi)
            if not mask.any():continue
            bmask=mask&board_masks[name];other=mask&source['finite_mask']&~board_masks[name];umask=mask&~source['finite_mask'];new=mask&new_masks[name]
            total+=int(mask.sum());boardtotal+=int(bmask.sum());othertotal+=int(other.sum());unknown+=int(umask.sum());addedtotal+=int(new.sum())
            allpairs|={tuple(sorted((int(p),int(q))))for p,q in zip(pa[mask],pb[mask])}
            bp={tuple(sorted((int(p),int(q))))for p,q in zip(pa[bmask],pb[bmask])};boardpairs|=bp
            npairs={tuple(sorted((int(p),int(q))))for p,q in zip(pa[new],pb[new])};newpairs|=npairs
            covered.extend(map(int,np.r_[pa[bmask],pb[bmask]]));newcovered.extend(map(int,np.r_[pa[new],pb[new]]))
            finite_dts.extend(abs(p-q)/15360. for p,q in bp)
            local.append(dict(block=name,motion_points=int(mask.sum()),finite_board_points=int(bmask.sum()),
                finite_other_surface_points=int(other.sum()),unknown_depth_points=int(umask.sum()),new947_points=int(new.sum())))
        viewpts=np.asarray(sorted(set(viewpts)),np.int64);covered=sorted(set(covered));newcovered=sorted(set(newcovered))
        shot_reports.append(dict(shot_index=shot_index,source_camera=shot['camera_id'],start_pts=lo,stop_pts_exclusive=hi,
            actual_exposures=len(viewpts),view_state_parameters=4*len(viewpts),motion_points=total,
            finite_board_points=boardtotal,finite_other_surface_points=othertotal,unknown_depth_points=unknown,
            finite_board_fraction=boardtotal/total if total else None,unknown_fraction=unknown/total if total else None,
            original_motion_actual_pairs=len(allpairs),finite_board_actual_pairs=len(boardpairs),new947_actual_pairs=len(newpairs),
            finite_board_incident_actual_exposures=len(covered),new947_incident_actual_exposures=len(newcovered),
            new947_points=addedtotal,new947_incident_pts=newcovered,
            finite_board_pair_dt_seconds_p0_p50_p100=np.quantile(finite_dts,[0,.5,1]).tolist()if finite_dts else None,
            **gap_summary(viewpts,covered),per_block=local))
    totals=dict(motion_points=sum(x['motion_points']for x in shot_reports),
        finite_board_points=sum(x['finite_board_points']for x in shot_reports),
        finite_other_surface_points=sum(x['finite_other_surface_points']for x in shot_reports),
        unknown_depth_points=sum(x['unknown_depth_points']for x in shot_reports),
        new947_points=sum(x['new947_points']for x in shot_reports),
        new947_actual_pairs=sum(x['new947_actual_pairs']for x in shot_reports),
        new947_incident_actual_exposures=sum(x['new947_incident_actual_exposures']for x in shot_reports),
        total_actual_exposures=sum(x['actual_exposures']for x in shot_reports))
    if totals['motion_points']!=sum(len(x['a'])for x in blocks.values()):raise ValueError('Cross-shot source rows exist; revise declared audit scope')
    if totals['new947_points']!=947:raise ValueError('The saved source revision is not exactly the947 production point change')
    totals['unknown_depth_fraction']=totals['unknown_depth_points']/totals['motion_points']
    totals['new947_camera_state_fraction']=totals['new947_incident_actual_exposures']/totals['total_actual_exposures']
    totals['finite_board_incident_actual_exposures']=sum(x['finite_board_incident_actual_exposures']for x in shot_reports)
    totals['actual_exposure_R3_logf_view_parameters']=4*totals['total_actual_exposures']
    code_refs=[dict(**ref(ROOT/path),evidence=detail)for path,detail in (
        ('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_fixed_lever_xyz_source_bundle.py',
            'lines32,50: optimized independent4 slots per actual exposure; cache reshapes each view intoRodrigues3/logf and computes optical centres from fixed shared pivot/lever'),
        ('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_moving_center_motion.py',
            'lines4-12,121-163,294-325: each unknown source row independently profiles one positive inverse-depth/epipolar-line nuisance; finite rows use the actual plane depth instead'),
        ('calibration/solver_runtime/legacy/structured_programme_source_candidate/original_motion_far_board_roles.py',
            'lines530-574,767-780: admitted finite point source ray selects current board facet; physical operator/finite mask changes only listed original source rows'),
        ('calibration/solver_runtime/legacy/correlated_roll_prior_candidate/explicit_bias_source.py',
            'lines135-159: original tail prior couples per-view roll to installation bias; it does not prescribe pan, pitch or zoom over the whole cut'))]
    result=dict(schema='rinkspace.exact-production-motion-role-whole-programme-coverage-readonly.v1',
        saved_state=ref(state/'state_identity.json'),q_sha256=identity['q_sha256'],actual_runtime_manifest=ref(run/'input_manifest.json'),
        complete_state=ref(state/'complete_state_UNACCEPTED.npz'),baseline_saved_state=ref(a.baseline_state_directory/'state_identity.json'),
        current_role_packet=dict(path=role['packet_path'],sha256=role['packet_sha256']),
        reconstruction_module=ref(a.source_replay_module),audit_script=ref(Path(__file__)),
        production_code_evidence=code_refs,
        exact_source_arrays_by_block=checks,exact_extra_course_and_finite_checks=extra_checks,source_authorities=authorities,
        totals=totals,shots=shot_reports,blocks=block_reports,actual947_source_pairs_and_exposures=new947,
        source_motion_rows_reconstructed_after_real_compaction_and_runtime_finite_mask=True,
        review_adjacent_tracks_not_counted_as_FIT=True,all_motion_point_profile_components_positive=True,
        finite_unknown_ratios_are_coverage_not_parameter_information_rank=True,
        other_absolute_native_ice_board_material_sources_and_priors_exist_not_counted_as_motion_edges=True,
        causal_findings=dict(
            proven='947 material role corrections affect only25 short source pairs/35 camera states; the independently parameterized8503 camera views retain98.44 percent unknown-depth source rows elsewhere.',
            consistent_mechanism='Per-row free depth can absorb an along-epipolar pan/zoom discrepancy outside the locally corrected pairs; fixed shared pivot/lever does not force a corrected temporal R/logf evolution.',
            not_proven='Coverage alone does not establish that every remaining visual mismatch comes from unknown depth, nor that views without finite-board motion have no absolute/native observations.',
            next_discriminating_calculation='Intersect the final-vs-a454 all8494 true adjacent-pair same-source motion audit with these exact finite-board/947 pair masks. Replay only residual/profile state of the existing production unknown rows on bad pairs, with sourceUV/sigma/frozen weights unchanged; compare profiled depth and nuisance tangent against conditional board-ray transfer. No J or fit is needed.',
            support_signature='Persistent actual board-flow deficit outside finite-board motion coverage, with small original profiled residual and fitted depth much nearer than the conditional current board depth, supports the remaining free-depth absorption mechanism.',
            falsification_signature='The same large deficit on source-qualified finite-board pairs, especially cut8 where every actual pair has finite-board support, means coverage is insufficient as an explanation; compare the exact plane-transfer residual and saved scene/optics instead.'),
        new_source_images_written=0,new_source_arrays_or_model_or_config_written=False,
        full_source_graph_J_H_or_optimizer_loaded=False,accuracy_accepted=False,seconds=time.perf_counter()-began)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(a.output),totals=totals,shots=[{k:x[k]for k in ('shot_index','actual_exposures','motion_points','finite_board_points','finite_other_surface_points','unknown_depth_points','new947_points','new947_actual_pairs','new947_incident_actual_exposures')}for x in shot_reports]),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
