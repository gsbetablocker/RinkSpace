"""Apply the exact source-verified S04 withdrawals without changing other rows.

The original S04 package and the independent fitting run remain read-only.
No name suffix, fitted residual, or whole-ad exclusion is an admission rule.
"""
from pathlib import Path
import collections
import gzip
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from rinkspace.vision_native import file_ref

DATA = Path('D:/CodexData/RinkSpace_Data')
OUT = DATA/'runs/native_ice_points_s04_20261027'
REPAIRS = DATA/'runs/native_joint_s01_s04_20261027/s04_source_repairs.jsonl.gz'
DEST = OUT/'points_repaired'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def apply():
    parent = OUT/'points'
    parent_ref = file_ref(parent/'point_observations.npz')
    repair_ref = file_ref(REPAIRS)
    receipt_file = DEST/'source_review_application.json'
    if receipt_file.exists():
        receipt = read(receipt_file)
        if receipt['parent_input'] != parent_ref or receipt['reviewed_source_records'] != repair_ref:
            raise ValueError('Source review parent changed; do not silently overwrite the previous revision')
        if receipt['output_input'] != file_ref(DEST/'point_observations.npz'):
            raise ValueError('Reviewed output changed after application')
        return DEST, receipt
    mapping = read(parent/'mapping.json')
    if parent_ref['sha256'] != mapping['point_input_sha256']:
        raise ValueError('Original delivery mapping/input mismatch')
    z = dict(np.load(parent/'point_observations.npz', allow_pickle=False))
    seeds = dict(np.load(parent/'initialized_points.npz', allow_pickle=False))
    seed_report = read(parent/'station_initialization.json')
    if set(seeds) != {'stations'} or len(seeds['stations']) != len(z['pts']):
        raise ValueError('Unexpected parent station layout')
    lookup = {str(v): i for i,v in enumerate(z['observation_ids'])}
    paths = np.asarray(mapping['path_ids'])[z['path_indices']]
    with gzip.open(REPAIRS,'rt',encoding='utf8') as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != 1323 or len({r['observation_id'] for r in records}) != 1323:
        raise ValueError('This source review must contain exactly 1323 distinct observed IDs')
    keep = np.ones(len(z['pts']), bool)
    families = collections.Counter()
    for r in records:
        i = lookup.get(r['observation_id'])
        if i is None or int(z['pts'][i]) != r['pts'] or not np.array_equal(z['uv'][i], r['uv']) or paths[i] != r['path_id']:
            raise ValueError('The independent source record does not match the original S04 row')
        if r['action'] != 'exclude_from_current_fit_keep_original_source':
            raise ValueError('Unsupported source review action')
        if 'Ad4.ANKARA.native440.boundary_' in paths[i] and 751616 <= z['pts'][i] <= 857600:
            families['Ad4_wrong_advertisement_identity'] += 1
        elif paths[i] == 'demo_initial/A1:outer' and 945408 <= z['pts'][i] <= 947456:
            families['A1_material_identity_without_source_support'] += 1
        else:
            raise ValueError('Reviewed ID outside its independently inspected source scope')
        keep[i] = False
    if dict(families) != {'Ad4_wrong_advertisement_identity':1267,'A1_material_identity_without_source_support':56}:
        raise ValueError('Unexpected reviewed scope')
    later = (z['pts'] >= 933120)&(z['pts'] <= 943360)&(np.char.find(paths,'Ad4.ANKARA.native440.boundary_') >= 0)
    if int(later.sum()) != 3929 or not keep[later].all():
        raise ValueError('The real later ANKARA observations must all remain')
    # Verify the current geometry assets still match the measured input binding.
    if file_ref(mapping['scene_path'])['sha256'] != mapping['scene_sha256']:
        raise ValueError('Current native model changed; this is not a geometry migration')
    DEST.mkdir(exist_ok=True)
    if any((DEST/n).exists() for n in ['point_observations.npz','mapping.json','initialized_points.npz']):
        raise FileExistsError('Partial source revision already exists; inspect before replacing it')
    selected = {k:v[keep] for k,v in z.items()}
    np.savez_compressed(DEST/'point_observations.npz', **selected)
    np.savez_compressed(DEST/'initialized_points.npz', stations=seeds['stations'][keep])
    out_ref = file_ref(DEST/'point_observations.npz')
    new_mapping = dict(mapping, point_input_sha256=out_ref['sha256'],
        parent_input=parent_ref, source_review_records=repair_ref,
        reviewed_withdrawals=1323, original_source_data_modified=False)
    write(DEST/'mapping.json',new_mapping)
    reindex = np.full(len(keep),-1,dtype=np.int64)
    reindex[keep] = np.arange(int(keep.sum()))
    remaining_unresolved = []
    for group in seed_report['unresolved_groups']:
        loc = [int(reindex[i]) for i in group['rows'] if keep[i]]
        if loc: remaining_unresolved.append(dict(group,rows=loc))
    seed_report = dict(seed_report, unresolved_groups=remaining_unresolved,
        curve_rows=int((selected['kind'] == 1).sum()),
        source_revision='Exact retained station subset; no fresh native evaluation or parameter update',
        parent_initialization_report=file_ref(parent/'station_initialization.json'))
    # Old error quantiles describe the pre-review initialization, not a new fit.
    seed_report['parent_seed_error_quantiles_px'] = seed_report.pop('seed_error_quantiles_px',None)
    write(DEST/'station_initialization.json',seed_report)
    check = dict(np.load(DEST/'point_observations.npz',allow_pickle=False))
    for k in z:
        if not np.array_equal(check[k],z[k][keep]):
            raise ValueError('An unreviewed observation field changed: '+k)
    if file_ref(parent/'point_observations.npz') != parent_ref or file_ref(REPAIRS) != repair_ref:
        raise ValueError('A source changed during application')
    ice_lo,ice_hi = mapping['source_row_namespace']['ice']
    ice = (check['source_rows'] >= ice_lo)&(check['source_rows'] < ice_hi)
    board = check['source_rows'] >= ice_hi
    receipt = dict(schema='rinkspace.exact_source_review_application.v1',parent_input=parent_ref,
        reviewed_source_records=repair_ref,output_input=out_ref,
        removed_by_source_reason=dict(families),before_rows=len(keep),after_rows=int(keep.sum()),
        retained_later_ANKARA_rows=3929,source_exposures=int(len(np.unique(check['pts']))),
        first_pts=int(check['pts'].min()),last_pts=int(check['pts'].max()),
        ice_points=int(ice.sum()),ad_points=int((~ice&~board).sum()),board_points=int(board.sum()),
        frames_with_ice=int(len(np.unique(check['pts'][ice]))),
        all_seven_fields_of_every_retained_row_identical=True,original_UV_PTS_ID_path_preserved=True,
        original_sources_modified=False,native_geometry_modified=False,peer_fit_modified=False,
        name_suffix_used_as_rejection_reason=False,whole_advertisement_exclusion=False,
        human_visual_acceptance=False,fresh_GPU_validation_run=False,
        source_visual_evidence=sorted({r['evidence'] for r in records}))
    write(receipt_file,receipt)
    return DEST,receipt


if __name__ == '__main__':
    path,result = apply()
    print(json.dumps(dict(directory=str(path),**result),ensure_ascii=False),flush=True)
