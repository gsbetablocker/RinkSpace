"""Save original-RGB witnesses for difficult source-only rail point links.

This audit does not change frozen evaluation ground truth or read predictions.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import av,cv2,numpy as np
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
BASE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/independent_reserved_native_evidence_20261013')
def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))
def main():
    cv2.setNumThreads(1)
    rp=BASE/'native_blue_component_point_evidence_v1/whole_shot_source_candidate_receipt.json';r=json.loads(rp.read_text(encoding='utf8'))
    ip=BASE/'reviewed_upper_blue_rail_source_identity_v2/upper_blue_rail_source_identity_exact_indices.npz'
    for key in ('source_numeric_evidence_reference','source_video_reference'):authenticate_source_reference(r[key])
    with np.load(r['source_numeric_evidence_reference']['path'],allow_pickle=False)as z:
        uv=z['candidate_observed_uv'];pts=z['native_pts'];hashes=z['decoded_BGR_sha256'];fo=z['frame_point_offsets'];links=z['directed_neighbor_candidate_point_indices'];metrics=z['source_only_flow_diagnostic_metrics']
    with np.load(ip,allow_pickle=False)as z:positive=z['source_link_supported_upper_rail_indices']
    mask=np.zeros(len(uv),bool);mask[positive]=True
    accepted=np.flatnonzero((links[:,1]>=0)&mask[links[:,0]])
    selected=[];target_frames=set()
    for column,name in ((0,'largest_forward_backward_error'),(1,'largest_source_normal_link_error')):
        temporal=[]
        for k in accepted[np.argsort(metrics[accepted,column])[::-1]]:
            a,b=links[k];i=int(np.searchsorted(fo,a,side='right')-1);j=int(np.searchsorted(fo,b,side='right')-1)
            native=int(pts[i])
            if any(abs(native-p)<15360 for p in temporal):continue
            temporal.append(native);target_frames.update([native,int(pts[j])])
            if j!=i+1 or pts[j]-pts[i]!=256:raise ValueError('Accepted source link crossed a real native boundary')
            selected.append(dict(selection_basis=name,source_link_row_index=int(k),source_native_pts=native,target_native_pts=int(pts[j]),
                source_global_point_index=int(a),target_global_point_index=int(b),source_UV=uv[a].tolist(),target_UV=uv[b].tolist(),
                forward_backward_error_px=float(metrics[k,0]),source_normal_error_px=float(metrics[k,1]),x_snap_error_px=float(metrics[k,2]),
                eligible_other_source_strip_margin_px=None if not np.isfinite(metrics[k,3])else float(metrics[k,3])))
            if len(temporal)==5:break
    out=BASE/'source_link_difficult_original_witnesses_v1'
    if out.exists():raise FileExistsError('Preserve completed original-source witness set')
    out.mkdir();records=[];strip_pages=[];all_strips=[]
    lookup={int(t):i for i,t in enumerate(pts)}
    with av.open(r['source_video_reference']['path'])as container:
        stream=container.streams.video[0];stream.thread_count=1;container.seek(min(target_frames),stream=stream,backward=True)
        for decoded in container.decode(stream):
            t=decoded.pts
            if t>max(target_frames):break
            if t not in target_frames:continue
            i=lookup[t];raw=decoded.to_ndarray(format='bgr24');digest=hashlib.sha256(raw.tobytes()).hexdigest()
            if digest!=hashes[i].decode():raise ValueError('Native original pixels differ from frozen complete evidence')
            raw_path=out/f'{t}_RAW_NATIVE.png';cv2.imwrite(str(raw_path),raw);canvas=raw.copy()
            for gi in np.flatnonzero(mask[fo[i]:fo[i+1]])+fo[i]:
                q=tuple(np.rint(uv[gi]).astype(int));cv2.circle(canvas,q,3,(0,255,0),1,cv2.LINE_AA)
            for edge in selected:
                for role in ('source','target'):
                    if edge[role+'_native_pts']!=t:continue
                    gi=edge[role+'_global_point_index'];q=tuple(np.rint(uv[gi]).astype(int));cv2.circle(canvas,q,7,(0,130,255),2,cv2.LINE_AA)
                    cv2.putText(canvas,str(gi),(q[0]+7,q[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)
            pointer=out/f'{t}_SOURCE_LINK_IDENTITY_QA_POINTERS.png';cv2.imwrite(str(pointer),canvas)
            strip=canvas[270:650].copy();cv2.putText(strip,str(t),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2,cv2.LINE_AA);all_strips.append(strip)
            records.append(dict(native_pts=int(t),decoded_BGR_sha256=digest,raw_source_reference=ref(raw_path),source_link_QA_pointer_reference=ref(pointer)))
    for i in range(0,len(all_strips),3):
        p=out/f'source_link_original_context_page_{i//3+1:02}.png';cv2.imwrite(str(p),np.vstack(all_strips[i:i+3]));strip_pages.append(ref(p))
    result=dict(schema='rinkspace.independent-reserved-source-link-original-QA-navigation.v1',complete_source_receipt_reference=ref(rp),source_link_identity_indices_reference=ref(ip),
        selected_actual_source_links=selected,raw_source_frame_witnesses=records,native_scale_original_context_pages=strip_pages,
        frozen_direct_evaluation_fact_unchanged=True,selected_link_points_not_added_to_direct_GT=True,candidate_model_output_used=False,
        full_shot_accuracy_accepted=False,implementation_reference=ref(__file__))
    p=out/'difficult_source_link_original_witness_navigation.json';p.write_text(json.dumps(result,separators=(',',':')),encoding='utf8');print(json.dumps(dict(reference=ref(p),original_frames=len(records),links=len(selected))))
if __name__=='__main__':main()
