"""Prepare continuous source review, then export one exact B1 gap delta."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack, _selected_review_rows, _exact_binding)
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, file_sha
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import _slice_current_row
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers

WORK = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
FULL = WORK / 'current_B1_gap_full_034_actual_scope_RGB_20261013_v2'
SMALL = WORK / 'current_B1_gap_actual_ice_scope_RGB_vector_20261013_v3'
MIDDLE = WORK / 'current_B1_gap_middle_actual_scope_RGB_20261013_v1'
OUTPUT = WORK / 'current_B1_full_034_missing_reviewed_source_delta_20261013_v1'
DATASET = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def literal(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    raise TypeError(type(value).__name__)


def source_rows():
    full = read(FULL / 'actual_B1_gap_review.json')
    dataset = read(DATASET)
    cut = dataset['cuts'][0]
    headers, offsets = read_minimal_compact_source_headers(cut['source_pack']['path'])
    with np.load(cut['support_index']['path'], allow_pickle=False) as z:
        fit = set(map(int, z['fit_point_indices']))
    existing = set()
    for i, header in enumerate(headers):
        if header['physical_material_id'] == 'B1:segment0':
            if any(j in fit for j in range(int(offsets[i]), int(offsets[i+1]))):existing.add(header['actual_pts'])
    seen, rows = set(), []
    origins = [(FULL, full), (SMALL, read(SMALL / 'actual_B1_gap_review.json')),
        (MIDDLE, read(MIDDLE / 'actual_B1_gap_review.json'))]
    for directory, packet in origins:
        packet_ref = ref(directory / 'actual_B1_gap_review.json')
        for record in packet['records']:
            pts = record['source_binding']['actual_pts']
            if not 229459200 <= pts < 229497600:continue
            if pts in existing:raise ValueError('Only missing feature/exposure may enter this delta')
            components = {c['candidate_index']:c for c in record['source_vector_receipt']['actual_RGB_vector_components']}
            for original in record['normalized_rows']:
                row = deepcopy(original)
                for name in ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv', 'source_side_quantiles_uv'):
                    row[name] = np.asarray(row[name], np.float64)
                row['original_point_indices'] = np.asarray(row['original_point_indices'], np.int64)
                keep = []
                for i, uv in enumerate(row['uv']):
                    key = (pts, uv.tobytes())
                    if key not in seen:seen.add(key);keep.append(i)
                if not keep:continue
                if len(keep) != len(row['uv']):row = _slice_current_row(row, np.asarray(keep, np.int64))
                queries = [components[int(i)]['original_query_index'] for i in row['original_point_indices']]
                if len(queries) > 1 and np.any(np.diff(queries) <= 0):
                    raise ValueError('A source row contains ambiguous/repeated original source stations')
                cuts = np.r_[0, np.flatnonzero(np.diff(queries) != 1) + 1, len(queries)]
                row.update(source_visible_paths=[list(range(int(a),int(b))) for a,b in zip(cuts[:-1],cuts[1:])],
                    source_original_query_indices=queries, original_source_delta_receipt=packet_ref,
                    family='B1_missing_034_actual_ice_scope_RGB_vector_source_delta',
                    final_current_RGB_pixel_precision_checked=True)
                if (not np.isfinite(row['source_side_quantiles_uv']).all()
                        or not np.array_equal(row['source_sides_uv'],row['source_side_quantiles_uv'][:,:,1])
                        or not np.array_equal(row['uv'],row['source_sides_uv'].mean(1))
                        or not np.isfinite(row['sigma_px']).all() or np.any(row['sigma_px'] <= 0)):
                    raise ValueError('Only intact actual two-q50 centers with measured finite support may be exported')
                rows.append(row)
    rows.sort(key=lambda r:(r['actual_pts'],min(r['source_original_query_indices'])))
    milestone = read(WORK / 'current_B1_gap_reviewed_source_delta_20261013_v2/reviewed_B1_source_delta_reference.json')
    with np.load(milestone['source_pack']['path'],allow_pickle=False) as z:
        from calibration.program_tracking.compact_source_json import unpack_source_json
        mheaders = unpack_source_json(z['rows_json']); moffsets = z['row_offsets']; muv = z['uv']
        for i,header in enumerate(mheaders):
            if any((header['actual_pts'],u.tobytes()) not in seen for u in muv[moffsets[i]:moffsets[i+1]]):
                raise ValueError('Final delta must contain all 12 reviewed milestone UV exactly once')
    return full, dataset, rows, milestone


def prepare(full, rows, milestone):
    if OUTPUT.exists():raise FileExistsError('Keep the prior review plan and all source packs unchanged')
    OUTPUT.mkdir(parents=True)
    by_pts = {}
    for row in rows:by_pts.setdefault(row['actual_pts'], []).append(row)
    pages, coverage = [], []
    strips=[]
    inventory=read(WORK/'source_pixels/source_measurements.json')['inventory']
    b1=next(i for i,r in enumerate(inventory)if r['curve_id']=='B1:segment0')
    for record in full['records']:
        pts=record['source_binding']['actual_pts']
        if pts<229459200:continue
        raw=cv2.imread(record['raw_reference']['path'])
        if hashlib.sha256(raw.tobytes()).hexdigest()!=record['source_binding']['decoded_native_BGR_sha256']:
            raise ValueError('Exact raw source recovery image changed')
        with np.load(record['array_reference']['path'],allow_pickle=False)as z:
            qids=np.flatnonzero(z[f'{pts}_current_query_curve']==b1)
            guide=z[f'{pts}_current_query_uv'][qids]
            if len(guide):
                normal=np.median(z[f'{pts}_current_query_normal'][qids],axis=0);normal/=np.linalg.norm(normal)
                tangent=np.array([normal[1],-normal[0]])
                if tangent[0]<0:tangent=-tangent
                origin=np.median(guide,axis=0);along=(guide-origin)@tangent
                first,last=float(along.min()-20),float(along.max()+20)
                width=int(np.ceil(last-first))+1
                xx=np.arange(width,dtype=float)+first;yy=np.arange(-70,71,dtype=float)
                xy=origin+xx[None,:,None]*tangent+yy[:,None,None]*normal
                strip=cv2.remap(raw,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,borderValue=(245,245,245))
                # These display coordinates annotate existing source readouts.
                # They are never source observations or interface search bounds.
                for row in by_pts.get(pts,[]):
                    for uv,sides in zip(row['uv'],row['source_sides_uv']):
                        coord=np.rint([float((uv-origin)@tangent-first),float((uv-origin)@normal+70)]).astype(int)
                        cv2.circle(strip,tuple(coord),2,(0,0,255),1,cv2.LINE_AA)
                        for side in sides:
                            pos=np.rint([float((side-origin)@tangent-first),float((side-origin)@normal+70)]).astype(int)
                            cv2.circle(strip,tuple(pos),1,(0,255,255),-1,cv2.LINE_AA)
                # A display scale only: source UV/q arrays remain literal native pixels.
                if width>1200:strip=cv2.resize(strip,(1200,141),interpolation=cv2.INTER_AREA)
                panel=np.full((171,1200,3),245,np.uint8);panel[30:,:strip.shape[1]]=strip
            else:panel=np.full((171,1200,3),245,np.uint8)
        count=sum(len(r['uv'])for r in by_pts.get(pts,[]))
        cv2.putText(panel,f'PTS {pts} B1 reviewed candidates {count}; RAW RGB, yellow sides/red midpoint',
            (5,22),0,.55,(0,0,0),1,cv2.LINE_AA)
        strips.append(panel)
        pending=record['point_precision_pending']
        coverage.append(dict(actual_pts=pts, source_binding=record['source_binding'],
            actual_blue_candidates=record['actual_B1_RGB_vector_candidates'],
            complete_actual_ice_scope_candidates=record['actual_B1_scope_clear_candidates'],
            actual_two_interface_pixel_supported_candidates=record['actual_B1_pixel_supported_candidates'],
            final_review_candidate_points=count,
            same_query_competing_blue_bands_pending=record['multiple_complete_actual_blue_bands_at_same_query'],
            source_pixel_pending_reason_counts=dict(Counter(x.get('status','unspecified')for x in pending)),
            source_identity_decisions=record['B1_source_model_context_decisions'],
            current_array_reference=record['array_reference'], frame_review_reference=record['frame_review_reference'],
            raw_reference=record['raw_reference'], same_8a28_all_grid_reference=record['grid_reference']))
    for start in range(0,len(strips),8):
        path=OUTPUT/f'continuous_actual_RGB_strip_review_{start//8+1:02d}.png'
        cv2.imwrite(str(path),np.concatenate(strips[start:start+8],axis=0));pages.append(ref(path))
    result=dict(schema='rinkspace.B1-full-034-missing-source-review-plan.v1',
        original_dataset_reference=ref(DATASET), full_replay_reference=ref(FULL/'actual_B1_gap_review.json'),
        smaller_12_point_pack_milestone_only=milestone['source_pack'],
        small_12_point_pack_all_exact_UV_included_once_in_final_candidate_set=True,
        final_delivery_must_use_one_final_pack_only=True,
        local_middle_original_source_receipt=ref(MIDDLE/'actual_B1_gap_review.json'),
        review_pages=pages, coverage=coverage, final_source_rows=rows,
        total_final_candidate_points=sum(len(r['uv'])for r in rows),
        actual_exposures_to_process=len(coverage), actual_exposures_with_final_candidates=len(by_pts),
        candidate_source_arrays_not_subsampled=True, original_frozen_dataset_changed=False,
        source_review_complete=False, source_FIT_admitted=False)
    path=OUTPUT/'source_delta_review_plan.json'
    path.write_text(json.dumps(result,ensure_ascii=False,default=literal,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(review_plan=ref(path),points=result['total_final_candidate_points'],
        exposures=result['actual_exposures_with_final_candidates'],pages=len(pages))))


def export(full,dataset,rows,milestone):
    plan_path=OUTPUT/'source_delta_review_plan.json';plan=read(plan_path)
    if plan['full_replay_reference']!=ref(FULL/'actual_B1_gap_review.json'):raise ValueError('Source replay changed')
    for page in plan['review_pages']:
        if page!=ref(page['path']):raise ValueError('Actually reviewed native strip image changed')
    facts=dict(plan,schema='rinkspace.B1-full-034-missing-agent-reviewed-source-facts.v1',
        final_source_rows_reference=ref(plan_path),source_review_complete=True,
        reviewer='gray_review_resume_agent_original_source_review',
        review_note='顺序核实整段原RGB蓝带、人物遮挡、两实际q50接口及时间连续；无完整侧面/竞争/未确证身份保留pending。',
        original_native_full_frame_and_same8a28_all_grid_used=True,
        final_source_UV_is_two_actual_q50_midpoint=True,
        projected_UV_or_fitted_residual_used_as_source_pixel_truth=False,
        model_distance_or_expected_width_used_to_clip_interfaces=False,
        smaller_12_point_pack_is_only_prior_milestone_and_must_not_be_added_separately=True,
        prior_wrong_scope_exact_UV_review_reference=ref(WORK/'current_B1_gap_reviewed_source_delta_20261013_v2/exact_B1_source_review_facts.json'),
        prior_incomplete_full_replay=dict(output_root=str(WORK/'current_B1_gap_full_034_actual_scope_RGB_20261013_v1'),
            last_completed_actual_pts=229494784, first_empty_clear_candidate_failure_actual_pts=229495040,
            error='ValueError: cannot reshape array of size 0 into shape (0,newaxis)',
            old_bound_public_implementation_bytes_unchanged=True,
            independently_fixed_by_empty_guard_and_per_exposure_array_checkpoint=True))
    facts.pop('final_source_rows')
    facts_path=OUTPUT/'reviewed_source_delta_facts.json'
    if facts_path.exists():raise FileExistsError(facts_path)
    facts_path.write_text(json.dumps(facts,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    record_by_pts={r['source_binding']['actual_pts']:r for r in full['records']}
    endpoints=[]
    for row in rows:
        row['exact_original_source_agent_point_review']=ref(facts_path)
        record=record_by_pts[row['actual_pts']];binding=_exact_binding(record['source_binding'])
        generated=_selected_review_rows([row],binding,facts['reviewer'])
        for proof in generated.records.values():
            proof=deepcopy(proof);proof.update(review_authority='agent_original_source_review',
                original_source_image_actually_reviewed=True,review_note=facts['review_note'],
                source_image_evidence=dict(record['raw_reference'],kind='original_video_decoded_actual_native_PNG',
                    actual_pts=binding['actual_pts'],native_pts=binding['native_pts'],decoded_BGR_sha256=binding['decoded_BGR_sha256']),
                exact_original_source_agent_point_review=ref(facts_path))
            endpoints.append(proof)
    source=dict(video_sha256=dataset['source']['video_sha256'],time_base='1/15360',pixel_domain='native1920x1080')
    review_path=OUTPUT/'exact_source_scope_review.json'
    review_path.write_text(json.dumps(dict(schema=REVIEW_SCHEMA,source=source,reviewed_endpoints=endpoints),
        ensure_ascii=False,separators=(',',':')),encoding='utf8')
    review=ExactSourceScopeReview.load(review_path)
    refs=[ref(plan_path),ref(facts_path),ref(review_path),ref(Path(__file__)),ref(FULL/'actual_B1_gap_review.json'),
        ref(SMALL/'actual_B1_gap_review.json'),ref(MIDDLE/'actual_B1_gap_review.json'),ref(DATASET)]
    refs+=full['implementation_references']
    refs+=[full['model_reference'],full['source_cache_reference']]
    refs+=[record['array_reference']for record in full['records']]
    refs+=[milestone['source_pack'],milestone['source_review_facts'],milestone['source_scope_review']]
    writer=StrictAbsolutePackWriter(OUTPUT/'B1_full_034_missing_exposures_reviewed_source_delta.npz',source=source,
        expected_actual_pts=dataset['expected_actual_pts'],original_references=refs)
    for record in full['records']:writer.mark_exposure_processed(record['source_binding']['actual_pts'],source_binding=record['source_binding'])
    for row in rows:
        if writer.add_row(row,review)!=len(row['uv']):raise ValueError('Public writer rejected an exact reviewed pixel')
    pack=writer.finish();consumer=StrictAbsoluteSourcePack(pack['path'],expected_sha256=pack['sha256'])
    count=sum(len(r['uv'])for r in consumer.fit_rows)
    if count!=plan['total_final_candidate_points']or consumer.held_rows:raise ValueError('Public consumer changed reviewed delta')
    result=dict(schema='rinkspace.B1-full-034-missing-reviewed-source-delta-reference.v1',source_pack=pack,
        source_review_facts=ref(facts_path),source_scope_review=ref(review_path),source_review_plan=ref(plan_path),
        final_points=count,actual_exposures_with_new_B1=len({r['actual_pts']for r in consumer.fit_rows}),
        actual_missing_exposures_reviewed=plan['actual_exposures_to_process'],
        actual_missing_exposures_without_new_B1=plan['actual_exposures_to_process']-len({r['actual_pts']for r in consumer.fit_rows}),
        previous_12_point_pack_is_milestone_only=milestone['source_pack'],all_previous_12_exact_UV_included_once=True,
        consume_only_this_final_delta_pack=True,only_B1_missing_feature_exposures_added=True,
        original_dataset_and_admitted_rows_unchanged=True,public_strict_source_consumer_verified=True,
        full_shot_camera_accuracy_accepted=False,solver_started=False)
    path=OUTPUT/'reviewed_full_B1_source_delta_reference.json'
    path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(reference=ref(path),**result),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-after-agent-source-review',action='store_true');args=p.parse_args()
    full,dataset,rows,milestone=source_rows()
    if args.export_after_agent_source_review:export(full,dataset,rows,milestone)
    else:prepare(full,rows,milestone)


if __name__=='__main__':main()
