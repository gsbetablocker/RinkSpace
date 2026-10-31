"""Admit independently registered, source-supported material measurements.

This is machine qualification of observed source boundaries, not user visual
acceptance or an accuracy claim. Unresolved records stay separately pending.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
import cv2,numpy as np
from capture_independent_materials import source_exclusion,write
from build_independent_shot_packets import selected_shots


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shot-ids',help='Comma-separated bootstrap shot IDs; default: all');a=p.parse_args()
    source=json.loads((a.capture/'binding.json').read_text('utf8'));index=json.loads((a.root/'index.json').read_text('utf8'))
    if source['source_sha256']!=index['source']['actual_sha256'] or not source['independent_source_capture']:raise ValueError('Wrong independent source')
    refs=json.loads((a.root/'print_references/references.json').read_text('utf8'))['references'];lookup={(r['ad_id'],r['pts']):r for r in refs}
    a.output.mkdir(parents=True,exist_ok=False);summaries={}
    for shot in selected_shots(index,a.shot_ids):
        sid=shot['shot_id'];folder=a.output/sid;folder.mkdir();path=a.capture/sid/'source_rows.jsonl';records=[json.loads(x)for x in path.read_text('utf8').splitlines()];notes={r['pts']:r for r in map(json.loads,(a.capture/sid/'registration.jsonl').read_text('utf8').splitlines())}
        rows_count=0;qualified_frames=0;pending_count=0;empty=[]
        with (folder/'source_rows.jsonl').open('w',encoding='utf8')as output:
            for r in records:
                pt=r['pts'];mask,mp=source_exclusion(a.root,sid,pt);current_sha=sha(mp);good=[];pending=list(r.get('pending_rows',[]));nn=notes[pt]['prints']
                for row in r['rows']:
                    reasons=[];matched=[n for n in nn if n['ad']==row['entity'] and n.get('status')=='source_content_registered' and n['reference_pts']==row['source_reference_pts']]
                    if len(matched)!=1:reasons.append('source_identity_not_unique')
                    else:
                        n=matched[0];meta=lookup[(n['ad'],n['reference_pts'])];curve=meta['native_edge_path_ids'].index(row['path_id']);readout=n['source_readout'];selected=set(readout['selected_candidate_ids'])
                        exact=[c for c in readout['candidate_records']if c['candidate_id']in selected and c['curve']==curve and c['uv']==row['uv']]
                        if len(exact)!=1 or exact[0]['candidate_flags']!=0:reasons.append('no_unique_unflagged_actual_source_measurement')
                        if n['source_pattern_check'].get('supported')is not True:reasons.append('source_pattern_not_verified')
                    if row.get('model_UV_synthesized')is not False:reasons.append('measurement_not_explicit_actual_source')
                    uv=np.asarray(row['uv'],float);normal=np.asarray(row['normal'],float);qq=np.asarray(row['actual_source_quantiles'],float)
                    if not np.isfinite(uv).all() or qq.shape!=(2,3,2):reasons.append('invalid_actual_quantiles')
                    else:
                        offsets=(qq[:,[0,2]].reshape(-1,2)-uv)@normal;lo,hi=offsets.min()-3,offsets.max()+3;segment=uv+np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)*4))+1))[:,None]*normal
                        if np.any(segment<0)or np.any(segment>[1919,1079]):reasons.append('source_span_outside_image')
                        else:
                            lower,upper=np.floor(segment).astype(int),np.ceil(segment).astype(int)
                            if any(mask[yy[:,1],xx[:,0]].any()for yy in(lower,upper)for xx in(lower,upper)):reasons.append('current_fine_mask_or_source_overlay_intersection')
                    if reasons:pending.append(dict(row,qualification_reasons=reasons));pending_count+=1
                    else:good.append(row)
                r.update(rows=good,pending_rows=pending,fit_eligible=bool(good),review_state='machine_qualified_independent_print_materials'if good else'pending_no_supported_materials',qualification=dict(source_capture=str(path),source_capture_sha256=sha(path),current_person_mask_path=str(mp),current_person_mask_sha256=current_sha,actual_UV_unchanged=True,native_residual_used=False,identity_source='known-rig ordered physical print and verified original-image pattern correspondence',human_acceptance=False,accuracy_claim=False))
                output.write(json.dumps(r,ensure_ascii=False)+'\n');rows_count+=len(good);qualified_frames+=bool(good)
                if not good:empty.append(pt)
        shutil.copyfile(a.capture/sid/'poses.npz',folder/'poses.npz')
        summaries[sid]=dict(rows=rows_count,qualified_frames=qualified_frames,frames=len(records),new_pending_rows=pending_count,unsupported_pts=empty)
    write(a.output/'binding.json',dict(source,fit_eligible=True,source_capture=str(a.capture),source_capture_binding_sha256=sha(a.capture/'binding.json'),qualification_method='actual registered-pattern RGB q50 with full transition support and current fine person exclusion; source-line organization; no model residual gate',human_acceptance=False,qualification_summaries=summaries))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
