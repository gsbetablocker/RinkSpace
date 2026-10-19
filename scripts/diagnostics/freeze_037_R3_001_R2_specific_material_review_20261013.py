"""Freeze exact original-source layer review without editing source packs."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import (
    _pending_indices, current_row_source_binding)
from calibration.program_tracking.source_nested_material_identity_audit import (
    inherited_nested_material_review_problem)

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1')
EVID = BASE / '037_R3_001_R2_actual_specific_material_review_20261013_v1'

# These lists record direct review of the exact original RGB, source-native
# neighbors and visible material topology. No fit/model UV selects this list.
KEEP_R3_INNER = {47057, 47069, 47084, 47772, 47810, 47811, 47812, 47821, 47840}
KEEP_R2_MIDDLE = {23671, 23672, 23801, 23802, 23803, 23804, *range(24224, 24231)}
KEEP_R2_INNER = {
    24347, 24348, 25147, 25167, 25168, 25169, 25402, 25665, 25666, 25667,
    25975, 25976, *range(27647,27651), *range(28058,28063),
    *range(28257,28262), *range(28470,28475), *range(28933,28939), 28985,
    *range(29092,29097), *range(29159,29165), *range(29213,29220),
    *range(29309,29315), *range(29377,29383), *range(29435,29442),
    *range(29495,29502), *range(29558,29565), *range(29617,29623)}
KEEP_R2_CENTER = {
    *range(25943,25961), 25969, 25970, 26078, 26079, *range(26186,26189),
    *range(26310,26313), *range(26586,26589), *range(27874,27877),
    *range(28063,28066), *range(28171,28174)}
KEEP = {'037': KEEP_R3_INNER,
        '001': KEEP_R2_MIDDLE | KEEP_R2_INNER | KEEP_R2_CENTER}
TOPOLOGY = {
    'R3:inner': 'Exact q50 lies on the visible pink/red inner disc exterior against the pale annulus of the same source house; it is independent of the gray outer ring and the inherited outer annotation.',
    'R2:middle': 'Exact q50 lies on the interior edge of the visible pink outer paint annulus, against its pale inner annulus; distinct from its exterior edge and the gray-blue inner ring.',
    'R2:inner': 'Exact q50 lies on the exterior edge of the visible gray-blue inner ring against the pale annulus; the ring and its separate small pale center are visible in the exact RGB and neighboring frames.',
    'R2:center': 'Exact q50 lies on the border of the small pale center within the gray-blue inner ring; distinct from that ring exterior, the pink outer annulus, and ice lines.'}

def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return ref(path)

def main():
    context_path = EVID/'exact_target_source_rows_and_full_context.json'
    context = json.loads(context_path.read_text(encoding='utf-8'))
    evidence_ref = ref(EVID/'specific_inner_original_source_evidence.json')
    source_frames = json.loads(Path(evidence_ref['path']).read_text(encoding='utf-8'))['source_frames']
    records = context['records']
    all_keep, all_pending, groups, per_cut = [], [], [], []
    implementation = ref(__file__)
    for cut in ('037','001'):
        rows = [r for r in records if r['cut'] == cut]
        pack_ref = rows[0]['source_pack_reference']
        pack = StrictAbsoluteSourcePack(pack_ref['path'], expected_sha256=pack_ref['sha256'])
        entries, target, kept, pending, frame_images = [], set(), set(), set(), {}
        for r in rows:
            ri = r['original_source_row_index']
            header = pack.rows[ri]
            lo,hi = map(int,pack.values['row_offsets'][ri:ri+2])
            indices = r['original_pack_global_point_indices']
            assert all(lo <= i < hi for i in indices)
            assert np.array_equal(pack.values['uv'][indices], np.asarray(r['uv']))
            assert header['split'] == 'fit'
            assert header['physical_material_id'] == r['physical_material_id']
            assert header['native_pts'] == r['native_pts'] and header['actual_pts'] == r['actual_pts']
            binding = current_row_source_binding(pack,ri)
            native = r['native_pts']
            raw_path = EVID/f'{native}_EXACT_RAW_NATIVE.png'
            raw = frame_images.setdefault(native, cv2.imread(str(raw_path)))
            assert hashlib.sha256(raw.tobytes()).hexdigest() == r['decoded_BGR_sha256']
            target.update(indices)
            pending_here = []
            for i in indices:
                j = i-lo
                uv = pack.values['uv'][i].tolist()
                if i not in KEEP[cut]:
                    pending.add(i);pending_here.append(i);continue
                kept.add(i)
                q = pack.values['source_side_quantiles_uv'][i]
                # Original 037 locators were declared float32 while their saved
                # support quantiles are float64. Check that declared precision;
                # never replace or round either literal stored array.
                assert np.array_equal(pack.values['uv'][i].astype(np.float32),
                    q[:,1].mean(axis=0).astype(np.float32))
                assert np.isfinite(q).all() and np.isfinite(pack.values['sigma_px'][i])
                n=q[0,2]-q[0,0];n=n/np.linalg.norm(n)
                coords=np.asarray(uv)+np.arange(-10,11)[:,None]*n
                rgb=cv2.remap(raw,coords[:,0].astype('float32').reshape(1,-1),
                    coords[:,1].astype('float32').reshape(1,-1),cv2.INTER_LINEAR).reshape(-1,3)[:,::-1]
                proof = dict(original_source_image_actually_reviewed=True,decision='allow',
                    source_feature_physical_identity_confirmed=True,
                    physical_material_id=r['physical_material_id'],actual_pts=r['actual_pts'],
                    native_pts=native,decoded_BGR_sha256=r['decoded_BGR_sha256'],uv=uv,
                    model_projection_or_fit_residual_used_for_identity=False,
                    source_pack_reference=pack_ref,pack_row_index=ri,pack_point_index=i,
                    original_candidate_index=header.get('original_candidate_indices',[None]*(hi-lo))[j],
                    source_query_index=header.get('source_query_indices',[None]*(hi-lo))[j],
                    original_source_point_index=int(pack.values['original_point_indices'][i]),
                    scope='ice',specific_material_topology=TOPOLOGY[r['physical_material_id']],
                    original_raw_native_image_reference=ref(raw_path),
                    exact_frame_review_reference=ref(EVID/f'{native}_exact_source_frame_review.json'),
                    original_sigma_px=float(pack.values['sigma_px'][i]),
                    original_source_side_quantiles_uv=q.tolist(),
                    source_q50_equal_at_original_declared_float32_precision=True,
                    literal_normal_unit_uv=n.tolist(),literal_source_RGB_profile_offsets_px=list(range(-10,11)),
                    literal_source_RGB_profile=rgb.tolist(),
                    is_new_source_point=False,source_UV_sigma_clock_or_weight_changed=False,
                    keep_is_existing_original_point_not_delta=True)
                assert inherited_nested_material_review_problem(header,proof,point_uv=uv) is None
                all_keep.append(proof)
            if pending_here:
                local=np.asarray(pending_here)-lo
                entry=dict(pack_row_index=ri,physical_material_id=r['physical_material_id'],
                    source_binding=binding,pack_point_indices=pending_here,
                    exclude_from_final_named_FIT=True,status='source_material_measured_named_identity_pending',
                    reason='Exact original RGB, seed/native neighbors and same 8a28 full-grid context were reviewed. This precise existing point does not independently identify its claimed named inner layer: source locations include other ring interfaces, faint ring-end/ice texture, and later off-house ice lines. Alternating gradients and outer annotation alone are insufficient. Preserve literal measured readouts pending, with no positive/negative training label; outer authority remains valid.')
                for field in ('original_candidate_indices','source_query_indices'):
                    if field in header:entry[field]=np.asarray(header[field])[local].tolist()
                entries.append(entry)
                all_pending.extend(dict(source_pack_reference=pack_ref,pack_row_index=ri,
                    pack_point_index=i,physical_material_id=r['physical_material_id'],
                    actual_pts=r['actual_pts'],native_pts=native,decoded_BGR_sha256=r['decoded_BGR_sha256'],
                    uv=pack.values['uv'][i].tolist(),decision='pending_specific_material_identity',
                    physical_negative_asserted=False) for i in pending_here)
        assert kept == KEEP[cut] and kept.isdisjoint(pending) and kept|pending == target
        group=dict(source_pack_reference=pack_ref,pending_named_point_count=len(pending),entries=entries,
            source_review_witnesses=[evidence_ref,ref(context_path)])
        assert sum(map(len,_pending_indices(pack,group).values())) == len(pending)
        groups.append(group)
        per_cut.append(dict(cut=cut,target_points=len(target),independently_verified_kept=len(kept),
            exact_pending_points=len(pending),kept_by_material=dict(Counter(
                p['physical_material_id'] for p in all_keep if p['source_pack_reference']==pack_ref))))
    assert len(all_keep)+len(all_pending)==813
    keep_ref=save(BASE/'037_001_independently_verified_specific_inner_keepfacts.json',dict(
        schema='rinkspace.exact-specific-inner-original-source-independent-keepfacts.v1',
        inventory_reference=context['inventory_reference'],source_evidence_reference=evidence_ref,
        exact_context_reference=ref(context_path),implementation_reference=implementation,
        exact_existing_point_keep_count=len(all_keep),points=all_keep,
        guard_cleared_by_exact_original_RGB_material_topology=True,
        additional_FIT_points_created=0,manual_false_used_as_rejection=False,
        original_source_pack_or_dataset_changed=False,outer_identity_authority_retracted=False))
    facts_ref=save(BASE/'037_001_exact_specific_inner_material_review_facts.json',dict(
        schema='rinkspace.037-001-specific-inner-exact-original-RGB-review-facts.v1',
        source_evidence_reference=evidence_ref,exact_context_reference=ref(context_path),
        implementation_reference=implementation,independent_keepfacts_reference=keep_ref,
        exact_reviewed_points=813,per_cut=per_cut,exact_pending_point_count=len(all_pending),
        pending_points=all_pending,decoded_original_native_frames=len(source_frames),
        actual_target_exposures=len({r['actual_pts'] for r in records}),
        source_model_used_for_navigation_only=True,source_coordinates_quantiles_and_sigmas_changed=False,
        manual_review_false_not_used_as_rejection=True,all_ordinary_HELD_unchanged=True,
        scope_negative_asserted=False,confirmed_outer_material_retained=True))
    for g in groups:g['source_review_witnesses'] += [keep_ref,facts_ref]
    filter_ref=save(BASE/'037_001_exact_specific_inner_material_pending_filter.json',dict(
        schema='rinkspace.exact-source-named-identity-pending-filter.v1',
        retain_pending_source_readouts_full_literal_evidence=True,source_pack_groups=groups,
        independently_verified_keepfacts_reference=keep_ref,confirmed_outer_source_retained=True,
        unknown_specific_material_is_not_negative=True,source_UV_sigma_or_clock_modified=False,
        review_accuracy_accepted=False,source_layer_scope_not_certified_from_projection=True))
    receipt=save(BASE/'037_001_specific_inner_final_review_receipt.json',dict(
        schema='rinkspace.037-001-specific-inner-final-review-receipt.v1',
        pending_filter_reference=filter_ref,independent_keepfacts_reference=keep_ref,
        source_review_facts_reference=facts_ref,per_cut=per_cut,
        reviewed_points=813,kept_points=len(all_keep),pending_points=len(all_pending),
        exact_partition_verified=True,public_exact_pending_filter_validator_passed=True,
        all_exact_keep_points_clear_public_specific_inner_guard=True,
        source_FIT_HELD_packs_unchanged=True,kept_original_points_are_not_new_delta=True))
    print(json.dumps(dict(filter_reference=filter_ref,keepfacts_reference=keep_ref,
        facts_reference=facts_ref,receipt_reference=receipt,per_cut=per_cut),ensure_ascii=False,indent=2))

if __name__ == '__main__':main()
