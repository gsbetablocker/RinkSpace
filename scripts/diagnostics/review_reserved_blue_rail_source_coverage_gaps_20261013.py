"""Save original native source witnesses in gaps of independent rail identity.

Selection uses source-evidence absence only. No candidate model is read, and
all blue components remain unassigned until individual source points review.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import av
import cv2
import numpy as np
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference
BASE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/independent_reserved_native_evidence_20261013')
def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))
def main():
    cv2.setNumThreads(1)
    old=BASE/'reviewed_upper_blue_rail_source_identity_v1/reviewed_whole_native_upper_blue_rail_source_receipt.json'
    prior=json.loads(old.read_text(encoding='utf8'))
    rp=BASE/'native_blue_component_point_evidence_v1/whole_shot_source_candidate_receipt.json'
    receipt=json.loads(rp.read_text(encoding='utf8'))
    authenticate_source_reference(receipt['source_video_reference'])
    for key in ('source_numeric_evidence_reference','source_candidate_inventory_reference'):authenticate_source_reference(receipt[key])
    inventory=json.loads(Path(receipt['source_candidate_inventory_reference']['path']).read_text(encoding='utf8'))
    with np.load(receipt['source_numeric_evidence_reference']['path'],allow_pickle=False)as z:
        pts=z['native_pts'];hashes=z['decoded_BGR_sha256'];uv=z['candidate_observed_uv'];offsets=z['candidate_point_offsets']
    spans=[];start=None
    for i,row in enumerate(prior['native_frame_evidence']):
        absent=not row['qualified_upper_blue_rail_source_points']
        if absent and start is None:start=i
        if not absent and start is not None:spans.append((start,i));start=None
    if start is not None:spans.append((start,len(pts)))
    selected=[]
    for a,b in spans:
        for lo in range(a,b,180):selected.append((lo+min(lo+180,b)-1)//2)
    selected_pts={int(pts[i]):i for i in selected}
    candidate_by_pts={}
    for ci,c in enumerate(inventory['candidate_rows']):candidate_by_pts.setdefault(c['native_pts'],[]).append((ci,c))
    out=BASE/'native_upper_blue_rail_coverage_gap_original_review_v1'
    if out.exists():raise FileExistsError('Keep completed source witnesses')
    out.mkdir()
    records=[];strips=[]
    with av.open(receipt['source_video_reference']['path'])as container:
        stream=container.streams.video[0];stream.thread_count=1
        container.seek(min(selected_pts),stream=stream,backward=True)
        for decoded in container.decode(stream):
            native=decoded.pts
            if native>max(selected_pts):break
            if native not in selected_pts:continue
            i=selected_pts[native];raw=decoded.to_ndarray(format='bgr24');digest=hashlib.sha256(raw.tobytes()).hexdigest()
            if digest!=hashes[i].decode():raise ValueError('Exact source BGR changed')
            path=out/f'{native}_RAW_NATIVE.png';cv2.imwrite(str(path),raw)
            canvas=raw.copy();rows=[]
            for ci,c in candidate_by_pts[native]:
                lo,hi=map(int,offsets[ci:ci+2]);rank=c['candidate_rank']
                for local,point in enumerate(uv[lo:hi]):
                    q=tuple(np.rint(point).astype(int));cv2.circle(canvas,q,2,(0,255,255),1,cv2.LINE_AA)
                    if local%3==0:cv2.putText(canvas,f'{rank}:{local}',(q[0]+3,q[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,100,240),1,cv2.LINE_AA)
                rows.append(dict(candidate_rank=rank,exact_candidate_global_range=[lo,hi],component_bbox_xywh=c['component_bbox_xywh']))
            pointer=out/f'{native}_UNASSIGNED_BLUE_READOUT_POINTERS.png';cv2.imwrite(str(pointer),canvas)
            strip=canvas[270:650].copy();cv2.putText(strip,str(native),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2,cv2.LINE_AA);strips.append(strip)
            records.append(dict(native_pts=native,decoded_BGR_sha256=digest,raw_source_reference=ref(path),unassigned_source_pointer_reference=ref(pointer),candidate_rows=rows))
    if len(records)!=len(selected_pts):raise ValueError('A selected actual native frame was not decoded')
    pages=[]
    for page in range(0,len(strips),3):
        path=out/f'original_rail_context_page_{page//3+1:02}.png';cv2.imwrite(str(path),np.vstack(strips[page:page+3]));pages.append(ref(path))
    result=dict(schema='rinkspace.independent-native-rail-source-gap-review-navigation.v1',source_candidate_receipt_reference=ref(rp),prior_source_identity_receipt_reference=ref(old),native_frames=2556,selected_source_frames=len(records),selection_basis='Temporal original source witnesses inside unanchored point-continuity intervals',records=records,native_scale_original_context_pages=pages,all_candidates_still_unassigned=True,candidate_model_output_used=False,independent_shot_used_for_training_or_tuning=False,implementation_reference=ref(__file__))
    path=out/'source_gap_original_review_navigation.json';path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(reference=ref(path),selected_native_frames=list(selected_pts))),flush=True)
if __name__=='__main__':main()
