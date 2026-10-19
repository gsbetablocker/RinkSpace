"""Freeze inspected rail seeds and conservative image-only source continuity.

The independent shot is never trained or tuned. Unreached candidates remain
unknown, and printed-blue exclusions refer only to this upper-rail identity.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import cv2
import numpy as np
from calibration.program_tracking.far_rail_source_identity_flow import split_measured_uv
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference

BASE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/independent_reserved_native_evidence_20261013')


def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))


def main():
    cv2.setNumThreads(1)
    source=BASE/'native_blue_component_point_evidence_v1'
    rp=source/'whole_shot_source_candidate_receipt.json';receipt=json.loads(rp.read_text(encoding='utf8'))
    for k in ('source_numeric_evidence_reference','source_candidate_inventory_reference'):authenticate_source_reference(receipt[k])
    metadata=json.loads(Path(receipt['source_candidate_inventory_reference']['path']).read_text(encoding='utf8'))
    with np.load(receipt['source_numeric_evidence_reference']['path'],allow_pickle=False)as z:
        uv=z['candidate_observed_uv'];intervals=z['vertical_10_to_90_colour_transition_interval_px']
        offsets=z['candidate_point_offsets'];frame_offsets=z['frame_point_offsets'];pts=z['native_pts']
        links=z['directed_neighbor_candidate_point_indices'];hashes=z['decoded_BGR_sha256']
    navigation=json.loads((BASE/'reserved_source_native_navigation.json').read_text(encoding='utf8'))
    raw_by_pts={r['native_pts']:r for r in navigation['records']}
    lookup={(c['native_pts'],c['candidate_rank']):i for i,c in enumerate(metadata['candidate_rows'])}
    # Coordinates below name individually inspected source points, not whole
    # blue components. Unlisted points receive no direct physical authority.
    positive={262580224:{0:[3,15,30],1:[4,14,20]},
        262901760:{0:[3,11,19],1:[3,11,19]},
        263234304:{0:[6,18,27],1:[2,8,14],2:[1,5,9]}}
    printed={262580224:{2:[0,3,6],3:[0,1,2]},262901760:{2:[0,1],3:[0,1]}}
    out=BASE/'reviewed_upper_blue_rail_source_identity_v1'
    if out.exists():raise FileExistsError('Keep completed independent source facts unchanged')
    out.mkdir()
    labels=np.zeros(len(uv),np.uint8);direct=[];marked=[]
    for native in raw_by_pts:
        original=Path(raw_by_pts[native]['path']);authenticate_source_reference(dict(path=str(original),sha256=raw_by_pts[native]['png_sha256']))
        raw=cv2.imread(str(original));canvas=raw.copy()
        for kind,selection,color in ((1,positive.get(native,{}),(0,255,0)),(2,printed.get(native,{}),(0,0,255))):
            for rank,local_indices in selection.items():
                ci=lookup[(native,rank)];lo,hi=map(int,offsets[ci:ci+2])
                for local in local_indices:
                    gi=lo+local
                    if not lo<=gi<hi:raise ValueError('Actual reviewed source point is absent')
                    labels[gi]=kind
                    q=tuple(np.rint(uv[gi]).astype(int));cv2.circle(canvas,q,4,color,1,cv2.LINE_AA)
                    cv2.putText(canvas,f'{rank}:{local}',(q[0]+4,q[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
                    direct.append(dict(native_pts=native,decoded_BGR_sha256=raw_by_pts[native]['decoded_BGR_sha256'],
                        candidate_rank=rank,observed_uv_index=local,source_global_point_index=gi,
                        literal_observed_uv=uv[gi].tolist(),source_vertical_10_to_90_transition_interval_px=intervals[gi].tolist(),
                        decision='actual_upper_blue_board_rail'if kind==1 else 'printed_blue_not_upper_board_rail',
                        original_source_reference=dict(path=str(original),sha256=raw_by_pts[native]['png_sha256']),
                        reviewer='assistant_direct_original_RGB_material_review',
                        physical_scope='board',whole_component_identity_assigned=False,
                        model_camera_or_model_output_used=False))
        path=out/f'{native}_ORIGINAL_REVIEWED_SEED_POINTERS.png';cv2.imwrite(str(path),canvas);marked.append(ref(path))
    fact=dict(schema='rinkspace.independent-exact-native-upper-blue-rail-source-seeds.v1',
        source_candidate_receipt_reference=ref(rp),source_video_reference=receipt['source_video_reference'],
        shot_id=receipt['shot_id'],native_time_base=receipt['source_time_base'],
        reserved_whole_shot=receipt['exact_reserved_native_interval'],records=direct,
        source_review_pointer_references=marked,
        specific_material_definition='Upper visible source blue cap boundary of the physical rink board; blue advertisements and clothing do not share this identity.',
        scope_of_printed_blue_rejection='Only upper_blue_board_rail; printed marks are not asserted outside the whole board model.',
        actual_source_material_seeds_reviewed_before_any_candidate_model_output=True,
        raw_UV_and_colour_transition_intervals_changed=False,
        source_interval_is_not_q25_q75_sigma_or_statistical_CI=True,
        independent_shot_ever_used_for_training_or_tuning=False,three_frames_are_not_whole_shot_accuracy_acceptance=True,
        implementation_reference=ref(__file__))
    fp=out/'reviewed_exact_upper_blue_rail_source_seeds.json';fp.write_text(json.dumps(fact,separators=(',',':')),encoding='utf8')
    # Exact UV links form point chains, rather than a connected-blue-component
    # identity. Two source-reviewed/linked anchors may bound a continuous thin
    # current measured strip. No extension beyond those bounds is permitted.
    parent=np.arange(len(uv),dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for a,b in links:
        if b>=0:
            ra,rb=find(int(a)),find(int(b))
            if ra!=rb:parent[rb]=ra
    roots=np.asarray([find(i)for i in range(len(uv))],np.int64)
    segments=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        lo,hi=int(lo),int(hi)
        for local in split_measured_uv(uv[lo:hi]):
            node=lo+local;width=intervals[node,1]-intervals[node,0]
            good=np.isfinite(width)&(width>0)&(width<=24.)
            cuts=np.flatnonzero(~good)
            for a,b in zip(np.r_[0,cuts+1],np.r_[cuts,len(node)]):
                part=node[a:b]
                if len(part):segments.append(part)
    steps=0;conflicts=[]
    while True:
        before=labels.copy();root_labels=np.zeros(len(uv),np.uint8)
        for gi in np.flatnonzero(labels):
            root=int(roots[gi]);value=int(labels[gi])
            if root_labels[root]and root_labels[root]!=value:conflicts.append(root)
            root_labels[root]=value
        if conflicts:raise ValueError('A point continuity chain conflicts with independently reviewed physical seeds')
        labels=np.maximum(labels,root_labels[roots])
        for segment in segments:
            for value in (1,2):
                selected=np.flatnonzero(labels[segment]==value)
                if len(selected)>=2:
                    part=segment[selected[0]:selected[-1]+1]
                    if np.any((labels[part]!=0)&(labels[part]!=value)):
                        raise ValueError('A measured strip between reviewed anchors contains another reviewed material')
                    labels[part]=value
        steps+=1
        if np.array_equal(before,labels):break
    positive_indices=np.flatnonzero(labels==1).astype(np.int64)
    printed_indices=np.flatnonzero(labels==2).astype(np.int64)
    unknown=np.flatnonzero(labels==0).astype(np.int64)
    covered=[];missing=[];per_frame=[]
    for i,native in enumerate(pts):
        lo,hi=map(int,frame_offsets[i:i+2]);pos=int(np.count_nonzero(labels[lo:hi]==1))
        nonrail=int(np.count_nonzero(labels[lo:hi]==2));pending=int(np.count_nonzero(labels[lo:hi]==0))
        (covered if pos else missing).append(int(native))
        per_frame.append(dict(native_pts=int(native),decoded_BGR_sha256=hashes[i].decode(),
            qualified_upper_blue_rail_source_points=pos,known_printed_blue_nonrail_points=nonrail,
            unassigned_blue_candidate_points=pending))
    ap=out/'qualified_upper_blue_rail_source_point_indices.npz'
    np.savez_compressed(ap,upper_blue_rail_source_point_indices=positive_indices,
        printed_blue_nonrail_source_point_indices=printed_indices,unassigned_source_point_indices=unknown,
        covered_native_pts=np.asarray(covered,np.int64),native_pts_without_qualified_rail_points=np.asarray(missing,np.int64))
    result=dict(schema='rinkspace.independent-whole-native-shot-reviewed-upper-blue-rail-source-evidence.v1',
        reviewed_physical_source_seed_reference=ref(fp),complete_native_candidate_evidence_reference=ref(rp),
        exact_source_numeric_reference=receipt['source_numeric_evidence_reference'],
        qualified_exact_source_indices_reference=ref(ap),native_frame_evidence=per_frame,
        native_frames=len(pts),native_frames_with_qualified_rail=len(covered),native_frames_without_qualified_rail=len(missing),
        qualified_rail_source_points=len(positive_indices),printed_blue_nonrail_source_points=len(printed_indices),
        blue_candidates_unassigned=len(unknown),source_continuity_closure_rounds=steps,
        single_seed_did_not_promote_a_whole_connected_component=True,
        current_actual_measured_points_between_at_least_two_anchors_only=True,
        source_colour_interval_semantics=receipt['interval_semantics'],
        unknown_or_missing_material_not_filled=True,source_UV_synthetic_or_interpolated=False,
        source_sigma_or_model_geometry_used=False,all_native_evidence_retained=True,
        candidate_model_output_used_for_ground_truth=False,model_prediction_comparison_performed=False,
        permanent_whole_shot_training_and_tuning_exclusion=True,whole_shot_accuracy_accepted=False,
        implementation_reference=ref(__file__))
    path=out/'reviewed_whole_native_upper_blue_rail_source_receipt.json';path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(path),seed_fact_reference=ref(fp),
        native_frames=len(pts),covered=len(covered),missing=len(missing),rail_points=len(positive_indices),
        nonrail_points=len(printed_indices),unassigned=len(unknown),closure_rounds=steps)),flush=True)


if __name__=='__main__':main()
