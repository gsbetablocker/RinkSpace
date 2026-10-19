"""Collect every native frame's image-only blue candidates and exact UV links.

The independent shot remains permanently outside training and model tuning.
Candidates and links are unassigned; physical rail seeds require source review.
Colour intervals retain the existing 10/50/90 transition semantics.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import av
import cv2
import numpy as np
from calibration.program_tracking.far_rail_source_components import blue_strip_candidates
from calibration.program_tracking.far_rail_source_identity_flow import match_sparse_bidirectional,PointwisePolicy
from calibration.program_tracking.strict_model_scope import file_sha,authenticate_source_reference

TRAIN=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training')
RESERVED=TRAIN/'continuous_review/independent_reserved_native_evidence_20261013'


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def main():
    started=time.perf_counter();cv2.setNumThreads(1)
    index_path=TRAIN/'supervision_index.json';index=json.loads(index_path.read_text(encoding='utf8'))
    navigation_path=RESERVED/'reserved_source_native_navigation.json';nav=json.loads(navigation_path.read_text(encoding='utf8'))
    shot=next(r for r in index['shot_plan']['shots']if r['id']=='258048000:R0440_021')
    if (shot['split']!='independent_evaluation'or shot['whole_shot']is not True
            or shot['propagation_across_cuts_allowed']is not False
            or [shot['start_pts'],shot['end_pts']]!=nav['reserved_whole_shot']):
        raise ValueError('This source interval must be an unchanged complete independent shot')
    for sample in index['samples']:
        pts=sample.get('source_pts',sample.get('pts',-1))
        if shot['start_pts']<=pts<shot['end_pts']:raise ValueError('Independent shot overlaps an existing training sample')
    source=dict(path=nav['source_path'],sha256=nav['source_sha256']);authenticate_source_reference(source)
    if shot['source_sha256']!=source['sha256']or shot['time_base']!='1/15360':raise ValueError('Reserved source or native time base changed')
    implementation=[ref(__file__),ref(ROOT/'calibration/program_tracking/far_rail_source_components.py'),
        ref(ROOT/'calibration/program_tracking/far_rail_source_identity_flow.py')]
    out=RESERVED/'native_blue_component_point_evidence_v1'
    if out.exists():raise FileExistsError('Preserve completed earlier independent evidence')
    out.mkdir()
    policy=PointwisePolicy()
    expected_known={r['native_pts']:r for r in nav['records']}
    pts_values=[];hash_values=[];candidate_counts=[];frame_offsets=[0]
    candidate_rows=[];candidate_point_offsets=[0];uv_values=[];interval_values=[];peak_values=[]
    links=[];reasons=[];reason_code={};counts=Counter();raw_duplicate_counts=Counter()
    previous=None;last_pts=None;previous_offsets=None;first_pts=None
    journal_path=out/'source_candidate_frame_inventory.jsonl'
    with journal_path.open('x',encoding='utf8')as journal,av.open(source['path'])as container:
        stream=container.streams.video[0];stream.thread_count=1
        if str(stream.time_base)!=shot['time_base']:raise ValueError('Actual decoder time base changed')
        container.seek(shot['start_pts'],stream=stream,backward=True)
        for decoded in container.decode(stream):
            pts=decoded.pts
            if pts is None or pts<shot['start_pts']:continue
            if pts>=shot['end_pts']:break
            if last_pts is not None and pts<=last_pts:raise ValueError('Native decoder clock is not increasing')
            raw=decoded.to_ndarray(format='bgr24')
            if raw.shape!=(1080,1920,3)or raw.dtype!=np.uint8:raise ValueError('Original native image domain changed')
            digest=hashlib.sha256(raw.tobytes()).hexdigest()
            if pts in expected_known and digest!=expected_known[pts]['decoded_BGR_sha256']:raise ValueError('Original navigation source pixels changed')
            candidates=blue_strip_candidates(raw,split_connected_columns=True)
            point_base=frame_offsets[-1];local_offsets=[0]
            for rank,c in enumerate(candidates):
                uv=np.asarray(c['observed_uv'],np.float64);bands=np.asarray(c['vertical_transition_interval_px'],np.float64)
                peaks=np.asarray(c['color_peak_score'],np.float64)
                if (uv.shape!=bands.shape or peaks.shape!=(len(uv),)or not np.isfinite(uv).all()
                        or not np.isfinite(bands).all()or np.any(bands[:,0]>0)or np.any(bands[:,1]<0)):
                    raise ValueError('Saved actual blue readout shape/transition differs')
                uv_values.append(uv);interval_values.append(bands);peak_values.append(peaks)
                local_offsets.append(local_offsets[-1]+len(uv));candidate_point_offsets.append(candidate_point_offsets[-1]+len(uv))
                candidate_rows.append(dict(native_pts=int(pts),candidate_rank=rank,
                    component_id=int(c['component_id']),component_bbox_xywh=c['component_bbox_xywh'],
                    component_area=int(c['component_area']),thin_column_fraction=c['thin_column_fraction'],
                    source_support_span_px=c['source_support_span_px'],source_ranking_score=c['source_ranking_score'],
                    thickness_p50_p95=c['thickness_p50_p95'],physical_identity='unassigned'))
            if previous is not None:
                ppts,praw,pcandidates,pbase=previous
                if pts-ppts==256:
                    seeds=[(r,i)for r,c in enumerate(pcandidates)for i in range(len(c['observed_uv']))]
                    linked=match_sparse_bidirectional(ppts,pts,praw,raw,pcandidates,candidates,seeds,policy=policy)
                    for row in linked['rows']:
                        src=pbase+previous_offsets[row['source_candidate_rank']]+row['source_uv_index']
                        accepted=row['status']=='bidirectional_source_match'
                        target=(point_base+local_offsets[row['target_candidate_rank']]+row['target_uv_index']if accepted else -1)
                        reason=row['reason'];code=reason_code.setdefault(reason,len(reason_code))
                        metrics=[row.get(k)if row.get(k)is not None else np.nan for k in ('fb_error_px','normal_error_px','x_snap_error_px','competitor_margin_px')]
                        links.append((src,target));reasons.append(code)
                        counts['bidirectional_candidate_links']+=int(accepted);counts['unlinked_source_candidate_points']+=int(not accepted)
                        counts['candidate_pair_attempts']+=1
                        # Source flow diagnostics never replace either measured UV.
                        row['_source_graph_metrics']=metrics
                    if linked['rows']:
                        metric_values.extend([r['_source_graph_metrics']for r in linked['rows']])
                else:
                    counts['native_clock_gaps_not_linked']+=1
            pts_values.append(int(pts));hash_values.append(digest);raw_duplicate_counts[digest]+=1
            candidate_counts.append(len(candidates));frame_offsets.append(point_base+local_offsets[-1])
            journal.write(json.dumps(dict(native_pts=int(pts),decoded_BGR_sha256=digest,
                candidate_count=len(candidates),measured_candidate_points=local_offsets[-1],
                candidate_global_point_range=[point_base,frame_offsets[-1]],
                all_material_identities_unassigned=True),separators=(',',':'))+'\n');journal.flush()
            previous=(pts,raw,candidates,point_base);previous_offsets=local_offsets;last_pts=pts
            if first_pts is None:first_pts=pts
            if len(pts_values)%50==0:
                print(json.dumps(dict(native_frames=len(pts_values),native_pts=int(pts),
                    measured_candidate_points=frame_offsets[-1],bidirectional_links=counts['bidirectional_candidate_links'],
                    elapsed_seconds=round(time.perf_counter()-started,3))),flush=True)
    if not pts_values or first_pts!=shot['start_pts']or last_pts+256!=shot['end_pts']:
        raise ValueError('Complete actual native source shot boundaries were not decoded')
    known=set(expected_known)
    if not known.issubset(pts_values):raise ValueError('An existing reserved navigation native frame was omitted')
    if len(links)!=len(metric_values):raise ValueError('Candidate link diagnostics lost rows')
    arrays_path=out/'native_blue_component_candidates_and_links.npz'
    np.savez_compressed(arrays_path,native_pts=np.asarray(pts_values,np.int64),
        decoded_BGR_sha256=np.asarray(hash_values,dtype='S64'),
        frame_point_offsets=np.asarray(frame_offsets,np.int64),frame_candidate_counts=np.asarray(candidate_counts,np.int64),
        candidate_point_offsets=np.asarray(candidate_point_offsets,np.int64),
        candidate_observed_uv=np.concatenate(uv_values)if uv_values else np.empty((0,2),np.float64),
        vertical_10_to_90_colour_transition_interval_px=np.concatenate(interval_values)if interval_values else np.empty((0,2),np.float64),
        colour_peak_score=np.concatenate(peak_values)if peak_values else np.empty((0,),np.float64),
        directed_neighbor_candidate_point_indices=np.asarray(links,np.int64).reshape(-1,2),
        source_only_flow_diagnostic_metrics=np.asarray(metric_values,np.float64).reshape(-1,4),
        candidate_link_reason_code=np.asarray(reasons,np.int16))
    metadata_path=out/'native_blue_component_candidate_inventory.json'
    metadata_path.write_text(json.dumps(dict(candidate_rows=candidate_rows,
        link_reason_vocabulary={str(v):k for k,v in reason_code.items()},
        source_flow_diagnostic_metric_columns=['fb_error_px','normal_error_px','x_snap_error_px','competitor_margin_px']),separators=(',',':')),encoding='utf8')
    for r in implementation:authenticate_source_reference(r)
    report=dict(schema='rinkspace.independent-whole-shot-native-blue-source-candidate-evidence.v1',
        supervision_index_reference=ref(index_path),reserved_navigation_reference=ref(navigation_path),
        source_video_reference=source,source_time_base=shot['time_base'],shot_id=shot['id'],
        exact_reserved_native_interval=[shot['start_pts'],shot['end_pts']],
        first_decoded_native_pts=first_pts,last_decoded_native_pts=last_pts,decoded_native_frames=len(pts_values),
        decoded_identical_BGR_presentations_observed=sum(n-1 for n in raw_duplicate_counts.values()),
        actual_exposure_aliases_inferred=False,frame_clock_subsampled=False,
        measured_candidate_points=frame_offsets[-1],source_candidate_components=len(candidate_rows),
        counts=dict(counts),pointwise_policy=asdict(policy),
        observed_uv_semantics='50 percent transition of the saved original-image blue-minus-red/green colour profile',
        interval_semantics='10-to-90 percent source colour transition, not q25/q75, sigma or a statistical confidence interval',
        source_numeric_evidence_reference=ref(arrays_path),source_candidate_inventory_reference=ref(metadata_path),
        native_clock_journal_reference=ref(journal_path),implementation_references=implementation,
        ground_truth_physical_identity_assigned=False,ground_truth_seeds_require_original_source_review=True,
        blue_component_identity_not_propagated_as_a_whole=True,
        independent_shot_ever_used_for_training_or_model_tuning=False,
        image_only_prediction_comparison_performed=False,whole_shot_accuracy_accepted=False,
        model_camera_projection_or_model_output_used=False,elapsed_seconds=time.perf_counter()-started)
    path=out/'whole_shot_source_candidate_receipt.json';path.write_text(json.dumps(report,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(path),native_frames=len(pts_values),points=frame_offsets[-1],counts=dict(counts))),flush=True)


if __name__=='__main__':
    metric_values=[]
    main()
