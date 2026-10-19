"""Read-only validation of the isolated board repair, with explicit limits."""
from pathlib import Path
from .repair_first_shot_board_patterns import *

def audit():
    registry=read(MODEL/'feature_registry.json')['targets'];manifest={r['index']:r for r in read(DATA/'frame_manifest.json')['frames']}
    errors=[];rows=[];counts=[];circle=[];nonprint_equal=0;source_masks=[]
    for i in range(201,1249):
        file=REPAIR/f'board_candidates/{i:06d}.json';p=read(file);old=read(DATA/f'board_candidates/{i:06d}.json')
        original_nonprint=[f for f in old['features']if'.print.'not in f['target_id']]
        repaired_nonprint=[f for f in p['features']if'.print.'not in f['target_id']]
        if original_nonprint!=repaired_nonprint:errors.append([i,'nonprint_features_changed'])
        else:nonprint_equal+=1
        mp=Path(current_mask_path(i,manifest[i]));mask=cv2.imread(str(mp),0)>0;mh=sha(mp)
        if Path(p['source']['mask_path'])!=mp or p['source']['mask_sha256']!=mh:errors.append([i,'current_mask_reference_mismatch'])
        if p['source']['png_sha256']!=manifest[i]['png_sha256']:errors.append([i,'original_native_png_reference_mismatch'])
        n=0;nc=0
        for f in p['features']:
            if f['target_id']not in registry:errors.append([i,f['target_id'],'unregistered_target'])
            if'.print.'in f['target_id']and f.get('label')!=registry[f['target_id']]['contour_key']:errors.append([i,f['target_id'],'contour_label_mismatch'])
            for s in f['segments']:
                uv=np.asarray(s['uv']);n+=len(uv)
                if not np.isfinite(uv).all()or((uv<0).any())or(uv[:,0]>1919).any()or(uv[:,1]>1079).any():errors.append([i,f['target_id'],'native_bounds'])
                else:
                    xy=np.rint(uv).astype(int)
                    if mask[xy[:,1],xy[:,0]].any():errors.append([i,f['target_id'],'point_in_person_mask'])
                for key in('point_bindings','transition_iqr_px','localization_sigma_px'):
                    if key in s and len(s[key])!=len(uv):errors.append([i,f['target_id'],'per_vertex_array_mismatch',key])
                if f['target_id']==TARGET:
                    nc+=len(uv);raw=np.asarray(s['source_native_readout_uv'])
                    if np.max(np.linalg.norm(uv[:,None]-raw[None],axis=2).min(1))>1.e-8:errors.append([i,'Ad11_observation_not_from_current_native_readout'])
                    if not print_inside_current_board_band(p['features'],uv).all():errors.append([i,'Ad11_outside_current_board_material'])
        if n!=p['point_count']:errors.append([i,'point_count_mismatch'])
        if n>500:errors.append([i,'board_only_exceeds_total_hard_cap'])
        counts.append(n)
        if nc:circle.append(dict(index=i,points=nc))
        source_masks.append(dict(index=i,path=str(mp),sha256=mh))
        rows.append(dict(index=i,sha256=sha(file),point_count=n,Ad11_points=nc,baseline_candidate_sha256=sha(DATA/f'board_candidates/{i:06d}.json')))
    baseline=read(REPAIR/'baseline_receipt.json');preserved=[]
    for record in baseline['files']:
        path=Path(record['path'])
        if '/source_datasets/'in path.as_posix()or '/models/'in path.as_posix():
            actual=sha(path)
            if actual!=record['sha256']:errors.append(['baseline_original_changed',str(path)])
            preserved.append(dict(path=str(path),sha256=actual))
    report=dict(schema='rinkspace.isolated-first-shot-board-repair-read-only-audit.v1',
        status='ready_for_root_visual_and_model_validation'if not errors else'errors_require_repair',
        frames=1048,range=[201,1248],original_nonprint_feature_packets_preserved_exactly=nonprint_equal,
        total_board_points=sum(counts),minimum_board_points=min(counts),maximum_board_points=max(counts),
        Ad11=dict(target_id=TARGET,observed_frames=len(circle),observed_index_range=[circle[0]['index'],circle[-1]['index']],
            old_observed_frames=1,old_points=2,new_points=sum(r['points']for r in circle),rows=circle),
        unchanged_original_dataset_and_model_files=len(preserved),original_dataset_and_model_files=preserved,
        source_masks=source_masks,rows=rows,errors=errors,
        geometry_and_templates_unchanged=True,user_manual_modifications_assumed=False,
        automatic_checks_are_not_physical_accuracy_acceptance=True,
        visual_review_scope='Ad11 native 797–912 complete sequential source review; current corrected masks 15 full native frames; clear new print components complete sequential 184-frame source review split with independent QA (pages 01–12 /root/prepare_first_shot_remaining; pages 13–23 /root/first_shot_board_ads); unchanged original upper/contact boundaries retain original full sequential review.',
        limitation='Remaining old print components have source diagnostic sample review, not a human per-pixel semantic proof. Root must assess current native overlays; no zero-error audit is substituted for physical validation.')
    write(REPAIR/'board_repair_final_audit.json',report)
    print(dict(frames=1048,total=sum(counts),minimum=min(counts),maximum=max(counts),circle_frames=len(circle),circle_points=sum(r['points']for r in circle),errors=len(errors)),flush=True)

if __name__=='__main__':audit()
