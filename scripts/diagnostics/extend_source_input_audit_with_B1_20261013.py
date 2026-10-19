"""Extend the exact 30-source audit with the actual reviewed 12-point B1 pack.

Unaffected cuts retain their authenticated literal audit. The affected cut is
re-read and compared across all its actual input arrays, including the delta.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
import numpy as np
from audit_source_input_merge_operators_20261013 import audit_cut
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
JOINT=WORK/'current_source_pool_joint_qualified_20261013_v1'


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def main():
    started=time.perf_counter()
    prior_path=WORK/'original_terminal_plus_current_gray_board_merge_operator_audit_20261013_v1.json'
    parent=ref(prior_path)
    if parent['sha256']!='97efa745f770fe082317cb34210888eb69e935ad01431a1b72d2ce6b33c79713':raise ValueError('Independent 30-source audit bytes changed')
    previous=json.loads(prior_path.read_text(encoding='utf8'))
    dataset_path=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'
    dataset=json.loads(dataset_path.read_text(encoding='utf8'))
    source=ref(WORK/'current_B1_gap_reviewed_source_delta_20261013_v2/B1_missing_three_exposures_reviewed_source_delta.npz')
    if source['sha256']!='e365e1bc2eca8907200d35d690279f2094c6883ccdd368328bb431d0faa3a9b8':raise ValueError('Reviewed B1 source bytes changed')
    pack=StrictAbsoluteSourcePack(source['path'],expected_sha256=source['sha256'])
    if len(pack.values['uv'])!=12 or len(pack.rows)!=3:raise ValueError('Unexpected reviewed B1 quantities')
    base=next(c['source_pack']for c in dataset['cuts']if c['source_pack']['path'].endswith('223403520_R0402_034.npz'))
    authenticate_source_reference(base)
    old_headers,_=read_minimal_compact_source_headers(base['path'])
    with np.load(base['path'],allow_pickle=False)as z:
        base_metadata=json.loads(str(z['metadata_json'].item()))
        frames={int(p):(int(n),s.decode())for p,n,s in zip(z['source_frame_actual_pts'],z['source_frame_native_pts'],z['source_frame_BGR_sha256'])if n>=0 and s}
    if pack.metadata['source']!=base_metadata['source']:raise ValueError('B1 is not bound to the same exact original source')
    for pts,binding in pack.frame_bindings.items():
        if frames.get(pts)!=(binding['native_pts'],binding['decoded_BGR_sha256']):raise ValueError('B1 native clock or decoded original pixels differ')
    old_targets={json.dumps(r['model_target'],sort_keys=True)for r in old_headers if r['physical_material_id']=='B1:segment0'}
    roles=Counter()
    for row in pack.rows:
        if (row['shot_id']!='223403520:R0402_034' or row['scope']!='ice' or row['physical_material_id']!='B1:segment0'
                or row['observation_semantics']!='paint_centerline' or row['split']!='fit'
                or json.dumps(row['model_target'],sort_keys=True)not in old_targets):
            raise ValueError('B1 role, material or exact current-model target changed')
        roles[str(row.get('vision_training_role'))]+=row['point_count']
    if not any(r==dataset['frozen_model_reference']for r in pack.metadata['original_references']):raise ValueError('Reviewed B1 pack has no same frozen model binding')
    if not (np.isfinite(pack.values['sigma_px'])&(pack.values['sigma_px']>0)).all():raise ValueError('Reviewed B1 lacks finite measured support')
    if not np.array_equal(pack.values['uv'],pack.values['source_side_quantiles_uv'][:,:,1].mean(1)):raise ValueError('B1 center changed actual q50 source readout')
    prior_refs=[r['source_reference']for c in previous['records']for r in c['inputs']]
    current_refs=[c['source_pack']for c in dataset['cuts']]
    key=lambda r:(str(Path(r['path']).resolve()).casefold(),r['sha256'])
    if set(map(key,prior_refs))!=set(map(key,current_refs))or len(prior_refs)!=30:raise ValueError('Prior audit no longer binds the same 30 sources')
    for r in prior_refs:authenticate_source_reference(r)
    affected=next(c for c in previous['records']if c['cut']=='034')
    current=audit_cut('034',[r['source_reference']for r in affected['inputs']]+[source])
    records=[current if c['cut']=='034' else c for c in previous['records']]
    total=Counter()
    for c in records:total.update(c['counts'])
    additions=JOINT/'B1_source_delta_inputs_for_current_review_20261013_v2.json'
    addition=dict(schema='rinkspace.current-reviewed-source-delta-inputs.v1',records=[dict(cut='034',absolute_inputs=source)],
        frozen_model_reference=dataset['frozen_model_reference'],original_pack_references_unchanged=True,
        vision_training_roles_preserved=dict(roles),unlisted_role_not_promoted_to_train=True)
    if additions.exists():raise FileExistsError('Keep earlier audit input manifests')
    additions.write_text(json.dumps(addition,separators=(',',':')),encoding='utf8')
    result=dict(previous,records=records,counts=dict(total),points=sum(r['points']for r in records),
        rows=sum(r['rows']for r in records),unique_exact_pixel_target_count=sum(r['unique_exact_pixel_target_count']for r in records),
        input_manifest_references=[*previous['input_manifest_references'],ref(additions)],
        authenticated_prior_30_source_exact_audit_reference=parent,
        unaffected_cut_literal_audits_reused_after_authenticating_all_original_inputs=True,
        affected_cut_all_input_arrays_reread_and_exact_operator_checked=True,
        B1_reviewed_source_delta_reference=source,B1_literal_clock_and_model_target_check=True,
        B1_literal_q50_center_and_finite_sigma_check=True,B1_vision_training_roles_preserved=dict(roles),
        B1_unlisted_training_role_promoted=False,elapsed_seconds=time.perf_counter()-started,
        implementation_reference=ref(__file__),reused_cut_audit_implementation_reference=ref(ROOT/'scripts/diagnostics/audit_source_input_merge_operators_20261013.py'))
    path=JOINT/'original_terminal_plus_current_gray_board_B1_merge_operator_audit_20261013_v2.json'
    if path.exists():raise FileExistsError('Keep earlier exact audits')
    path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(audit_reference=ref(path),points=result['points'],rows=result['rows'],counts=dict(total),B1_roles=dict(roles))),flush=True)


if __name__=='__main__':main()
