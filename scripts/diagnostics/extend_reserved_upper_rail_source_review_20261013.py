"""Freeze additional original-source upper-rail seeds, before model output.

Only individually inspected measured source points seed identity. Derived
source-link indices retain a separate role from direct material review.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import cv2
import numpy as np
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference
from calibration.program_tracking.far_rail_source_identity_flow import expand_between_reviewed_seeds
from calibration.program_tracking.independent_source_point_identity import source_point_identity_closure
BASE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/independent_reserved_native_evidence_20261013')
def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))
def main():
    cv2.setNumThreads(1)
    prior_path=BASE/'reviewed_upper_blue_rail_source_identity_v1/reviewed_exact_upper_blue_rail_source_seeds.json'
    prior=json.loads(prior_path.read_text(encoding='utf8'))
    gp=BASE/'native_upper_blue_rail_coverage_gap_original_review_v1/source_gap_original_review_navigation.json'
    gap=json.loads(gp.read_text(encoding='utf8'))
    rp=BASE/'native_blue_component_point_evidence_v1/whole_shot_source_candidate_receipt.json'
    receipt=json.loads(rp.read_text(encoding='utf8'))
    for key in ('source_numeric_evidence_reference','source_candidate_inventory_reference'):authenticate_source_reference(receipt[key])
    metadata=json.loads(Path(receipt['source_candidate_inventory_reference']['path']).read_text(encoding='utf8'))
    with np.load(receipt['source_numeric_evidence_reference']['path'],allow_pickle=False)as z:
        uv=z['candidate_observed_uv'];bands=z['vertical_10_to_90_colour_transition_interval_px'];peaks=z['colour_peak_score']
        offsets=z['candidate_point_offsets'];fo=z['frame_point_offsets'];pts=z['native_pts'];hashes=z['decoded_BGR_sha256'];links=z['directed_neighbor_candidate_point_indices']
    lookup={(c['native_pts'],c['candidate_rank']):i for i,c in enumerate(metadata['candidate_rows'])}
    by_pts={r['native_pts']:r for r in gap['records']}
    # Exact point indices inspected in the frozen original-RGB context pages.
    selected={
        262632448:{0:[3,12,21],1:[3,9,18]},262678528:{1:[3,12,24],0:[3,15,30]},
        262724608:{0:[3,12,21],1:[3,6,9]},262770688:{1:[3,9,15],0:[3,18,36]},
        262816768:{1:[3,6,9],0:[6,18,36]},262848256:{1:[1,4,7],0:[3,9,15]},
        262949888:{0:[3,12,24],1:[3,15,27]},262995968:{0:[3,15,30],1:[3,12,24]},
        263042048:{1:[3,6,9],0:[3,15,30]},263088128:{0:[3,12,21],1:[3,9,12]},
        263134208:{0:[3,9,15],1:[3,9,18]},263180288:{1:[3,6],0:[3,12,18]},
        263211776:{0:[3,15,30],1:[3,9,15]}}
    out=BASE/'reviewed_upper_blue_rail_source_identity_v2'
    if out.exists():raise FileExistsError('Keep frozen independent evidence unchanged')
    out.mkdir()
    labels=np.zeros(len(uv),np.uint8);records=[];corrections=[]
    for record in prior['records']:
        r=dict(record);gi=r['source_global_point_index']
        if not np.array_equal(np.asarray(r['literal_observed_uv']),uv[gi]):raise ValueError('Prior literal source UV changed')
        authenticate_source_reference(r['original_source_reference'])
        if (r['native_pts'],r['candidate_rank'],r['observed_uv_index'])==(262580224,3,2):
            r.update(decision='foreground_person_not_upper_board_rail',physical_scope='outside_board_body',
                source_category='foreground_person_occluding_board',original_source_material_review_correction=True)
            corrections.append(dict(source_global_point_index=gi,prior_seed_reference=ref(prior_path),
                corrected_specific_category='foreground_person_occluding_board',rail_exclusion_unchanged=True))
        labels[gi]=1 if r['decision']=='actual_upper_blue_board_rail'else 2
        records.append(r)
    for native,ranks in selected.items():
        original=by_pts[native]['raw_source_reference'];authenticate_source_reference(original)
        for rank,local_indices in ranks.items():
            ci=lookup[(native,rank)];lo,hi=map(int,offsets[ci:ci+2])
            for local in local_indices:
                gi=lo+local
                if not lo<=gi<hi:raise ValueError('Individually reviewed source point absent')
                labels[gi]=1
                records.append(dict(native_pts=native,decoded_BGR_sha256=by_pts[native]['decoded_BGR_sha256'],
                    candidate_rank=rank,observed_uv_index=local,source_global_point_index=gi,
                    literal_observed_uv=uv[gi].tolist(),source_vertical_10_to_90_transition_interval_px=bands[gi].tolist(),
                    decision='actual_upper_blue_board_rail',physical_scope='board',
                    original_source_reference=original,reviewer='assistant_direct_original_RGB_material_review',
                    whole_component_identity_assigned=False,model_camera_or_model_output_used=False))
    direct_indices=np.flatnonzero(labels).astype(np.int64);direct_labels=labels[direct_indices].copy()
    pointer_refs=[]
    for native in sorted({r['native_pts']for r in records}):
        rows=[r for r in records if r['native_pts']==native]
        raw=cv2.imread(rows[0]['original_source_reference']['path']);canvas=raw.copy()
        for r in rows:
            gi=r['source_global_point_index'];q=tuple(np.rint(uv[gi]).astype(int));color=(0,255,0)if labels[gi]==1 else(0,0,255)
            cv2.circle(canvas,q,4,color,1,cv2.LINE_AA)
            cv2.putText(canvas,f"{r['candidate_rank']}:{r['observed_uv_index']}",(q[0]+4,q[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        path=out/f'{native}_ORIGINAL_REVIEWED_SEED_POINTERS.png';cv2.imwrite(str(path),canvas);pointer_refs.append(ref(path))
    fact=dict(schema='rinkspace.independent-exact-native-upper-blue-rail-source-seeds.v2',records=records,
        source_candidate_receipt_reference=ref(rp),additional_original_navigation_reference=ref(gp),superseded_seed_fact_reference=ref(prior_path),
        source_category_corrections=corrections,source_review_pointer_references=pointer_refs,
        direct_original_reviewed_native_frames=len(pointer_refs),direct_original_reviewed_rail_points=int(np.count_nonzero(labels==1)),
        direct_original_reviewed_nonrail_points=int(np.count_nonzero(labels==2)),
        whole_blue_component_identity_assigned=False,candidate_model_output_used_for_ground_truth=False,
        permanent_whole_shot_training_and_tuning_exclusion=True,whole_shot_accuracy_accepted=False,
        source_interval_semantics=receipt['interval_semantics'],implementation_reference=ref(__file__))
    fp=out/'reviewed_exact_upper_blue_rail_source_seeds.json';fp.write_text(json.dumps(fact,separators=(',',':')),encoding='utf8')
    derived,rounds=source_point_identity_closure(uv,bands,peaks,offsets,links,labels,expand_between_reviewed_seeds)
    frames=[];covered=[];missing=[]
    for i,native in enumerate(pts):
        a,b=map(int,fo[i:i+2]);pos=int(np.count_nonzero(derived[a:b]==1));neg=int(np.count_nonzero(derived[a:b]==2))
        (covered if pos else missing).append(int(native))
        frames.append(dict(native_pts=int(native),decoded_BGR_sha256=hashes[i].decode(),source_link_supported_rail_points=pos,
            source_link_supported_nonrail_points=neg,unassigned_blue_points=int(b-a-pos-neg)))
    ap=out/'upper_blue_rail_source_identity_exact_indices.npz'
    np.savez_compressed(ap,direct_reviewed_source_point_indices=direct_indices,direct_reviewed_source_point_labels=direct_labels,
        source_link_supported_upper_rail_indices=np.flatnonzero(derived==1),source_link_supported_nonrail_indices=np.flatnonzero(derived==2),
        unassigned_source_point_indices=np.flatnonzero(derived==0),native_pts_without_source_link_supported_rail=np.asarray(missing,np.int64))
    result=dict(schema='rinkspace.independent-whole-native-shot-source-rail-evidence.v2',source_material_seed_reference=ref(fp),
        complete_native_candidate_evidence_reference=ref(rp),exact_source_indices_reference=ref(ap),native_frame_evidence=frames,
        actual_decoded_native_frames=len(pts),frames_with_source_link_supported_rail=len(covered),frames_without_source_link_supported_rail=len(missing),
        direct_reviewed_rail_points=int(np.count_nonzero(labels==1)),source_link_supported_rail_points=int(np.count_nonzero(derived==1)),
        source_link_supported_nonrail_points=int(np.count_nonzero(derived==2)),unassigned_points=int(np.count_nonzero(derived==0)),
        source_only_identity_closure_rounds=rounds,
        derived_source_link_points_are_not_direct_original_material_review=True,
        no_extrapolated_UV_or_filled_missing_points=True,
        independent_prediction_comparison_performed=False,whole_shot_accuracy_accepted=False,
        source_colour_transition_intervals_are_not_sigma_q25_q75_or_statistical_CI=True,
        permanent_whole_shot_training_and_tuning_exclusion=True,candidate_model_output_used_for_ground_truth=False,
        implementation_references=[ref(__file__),ref(ROOT/'calibration/program_tracking/independent_source_point_identity.py'),ref(ROOT/'calibration/program_tracking/far_rail_source_identity_flow.py')])
    path=out/'whole_native_source_rail_identity_receipt.json';path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(path),source_seed_fact=ref(fp),frames_with_rail=len(covered),frames_without_rail=len(missing),source_link_supported_rail_points=int(np.count_nonzero(derived==1)))),flush=True)
if __name__=='__main__':main()
