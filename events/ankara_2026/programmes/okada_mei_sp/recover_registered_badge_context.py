"""Recover only requested missing badge frames with broader source navigation.

Full available original-image features navigate H. The approved badge ROI alone
still passes the unchanged whole-pattern check; public RGB q50s supply all UV.
"""
from pathlib import Path
import argparse,hashlib,json
import cv2,numpy as np
import capture_independent_badge as badge
from capture_independent_badge_full import read_profile_proof
from capture_independent_prints import source_patch_check
from capture_independent_materials import source_exclusion,spaced_indices,plain
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from rinkspace.geometry import project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in('root','reference-root','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pts',type=int,nargs='+',required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);index=badge.read(root/'index.json');meta=badge.read(a.reference_root/'reference.json')
    image=cv2.imread(meta['source_image']);refex,_=source_exclusion(root,'s07',meta['pts']);features=descriptors(image,(~refex).astype('uint8')*255)
    if hashlib.sha256(image.tobytes()).hexdigest()!=meta['source_BGR_sha256']:raise ValueError('Original source reference changed')
    reference=dict(source_image=image,reference_mask=cv2.imread(str(a.reference_root/'reference_mask.png'),0)>0)
    shape=np.load(a.reference_root/'reference_geometry.npz',allow_pickle=False)['reference_search_uv'];original={}
    for line in(a.capture/'source_rows.jsonl').open(encoding='utf8'):
        record=json.loads(line);original[record['pts']]=record
    targets=[pt for pt in sorted(set(a.pts))if not original.get(pt,{}).get('fit_eligible')]
    frames=badge.exact_frames(index['source']['resolved_path'],targets);out.mkdir(parents=True);(out/'profiles').mkdir();summaries=[]
    with(out/'source_rows.jsonl').open('x',encoding='utf8')as rows_out:
        for pt in targets:
            current=frames[pt];rgb=cv2.cvtColor(current,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,'s07',pt);match=match_features(features,descriptors(current,(~ex).astype('uint8')*255),ReuseConfig(min_matches=6,ransac_px=3.))
            rows=[];pending=[];check=None;record=None;raw=[];h=None;profile=None;status='pending_no_public_RootSIFT_context_match'
            if match is not None:
                _,_,h=match;check=source_patch_check(current,ex,meta,reference,h)
                if check['supported']:
                    search=project(h,shape);public,record,raw,arrays=read_profile_proof(rgb,search,ex,dict(source_sha256=index['source']['actual_sha256'],source_pts=pt,time_base='1/12800',shot_id='s07',decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),material_signal=meta['material_signal'])
                    profile=out/'profiles'/f'{pt}.npz';np.savez_compressed(profile,**arrays)
                    for ci,q in enumerate(raw):pending.append(dict(q,pts=pt,path_id=meta['native_path_id'],entity=meta['ad_id'],candidate_order=ci,source_reference_pts=meta['pts'],source_profile_proof=str(profile),fit_eligible=False))
                    for ci in spaced_indices([q['uv']for q in public]):
                        q=public[ci];trace=next(qr for qr in pending if qr['uv']==q['uv']);rows.append(dict(trace,fit_eligible=True,review_state='approved_source_instance_and_unchanged_public_RGB_exterior_qualified',qualification=dict(known_source_instance=meta['ad_id'],source_pattern_check=check,existing_public_closed_outline_qualified=True,original_RGB_UV_unchanged=True,native_residual_used=False,shape_or_H_UV_used_as_observation=False,source_reference=str(a.reference_root/'reference.json'),source_reference_sha256=badge.sha(a.reference_root/'reference.json'))))
                    status='source_pattern_and_public_RGB_exterior_qualified'if rows else'pending_no_public_closed_outline'
                else:status='pending_original_shield_whole_pattern_unsupported'
            row_record=dict(pts=pt,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=hashlib.sha256(current.tobytes()).hexdigest(),mask_path=str(mp),mask_sha256=badge.sha(mp),rows=rows,pending_rows=pending,fit_eligible=bool(rows),review_state=status,source_reference=str(a.reference_root/'reference.json'),source_reference_sha256=badge.sha(a.reference_root/'reference.json'),source_profile_proof=None if profile is None else str(profile),source_transfer_basis='full_available_original_RGB_context_RootSIFT_for_navigation_only',status=status)
            rows_out.write(json.dumps(plain(row_record),ensure_ascii=False,allow_nan=False)+'\n')
            proof=dict(row_record,RootSIFT_matches=0 if match is None else len(match[0]),H_current_from_reference=None if h is None else h.tolist(),source_pattern_check=check,reader_record=record,all_actual_exterior_candidates=raw,public_thresholds_changed=False,original_shield_ROI_unchanged=True)
            badge.write(out/f'{pt}_proof.json',proof);summaries.append(dict(pts=pt,status=status,rows=len(rows),matches=proof['RootSIFT_matches'],pattern_check=check))
    badge.write(out/'report.json',dict(source_video_sha256=index['source']['actual_sha256'],source_reference=str(a.reference_root/'reference.json'),source_reference_sha256=badge.sha(a.reference_root/'reference.json'),reference_features=len(features[0]),navigation_descriptor_domain='full available source image, not additional physical observations',unchanged_pattern_ROI=meta['roi_xyxy_inclusive'],native_path_id=meta['native_path_id'],frames=summaries,total_rows=sum(q['rows']for q in summaries),implementation=str(Path(__file__).resolve()),implementation_sha256=badge.sha(__file__),fit_executed=False))
    print(json.dumps(dict(frames=len(summaries),qualified_frames=sum(bool(q['rows'])for q in summaries),rows=sum(q['rows']for q in summaries),output=str(out))),flush=True)


if __name__=='__main__':main()