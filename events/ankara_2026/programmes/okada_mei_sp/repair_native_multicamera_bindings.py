"""Apply source-verified S02 identities without importing geometry or changing UV."""
from pathlib import Path
import json
import shutil
import numpy as np
import prepare_native_multicamera as prepared


def repair():
    root = prepared.RUN
    source = root / 'point_observations.npz'
    manifest_path = root / 'joint_manifest.json'
    manifest = prepared.read(manifest_path)
    mapping = prepared.read(root / 'mapping.json')
    before_hash = prepared.digest(source)
    if before_hash != manifest['point_input_sha256']:
        raise ValueError('The active input changed before the repair')
    review = root / 'binding_review' / 's02_trapezoid_repair.json'
    if review.exists():
        raise FileExistsError('This source repair already has a receipt')
    data = prepared.npz(source)
    if set(data) != prepared.FIELDS:
        raise ValueError('Expected the seven unweighted observation fields')
    old_id = 'd78_material_162/path0'
    new_id = 'demo_initial/goal_trapezoid_right_lower_outer:0'
    old_index = mapping['path_ids'].index(old_id)
    new_index = mapping['path_ids'].index(new_id)
    first, last = 625664, 651008  # 48.880--50.860, source 50 fps.
    rows = np.flatnonzero((data['path_indices'] == old_index)
                         & (data['pts'] >= first) & (data['pts'] <= last))
    if len(rows) != 472 or not np.array_equal(np.unique(data['pts'][rows]),
                                             np.arange(first, last + 1, 256)):
        raise ValueError('The reviewed 100-frame / 472-point group differs')
    original_paths = data['path_indices'].copy()
    receipt = dict(
        schema='rinkspace.source_identity_repair.v1',
        source_pts=[first, last], source_seconds=[first / 12800, last / 12800],
        frames=100, rows=472, old_path_id=old_id, new_path_id=new_id,
        reason='The re-entering red diagonal is the right lower goal trapezoid; '
               'the source tracker retained the earlier F2--R2 connection identity.',
        evidence=[str(root / 'binding_review' / f'{pts}_bindings.png')
                  for pts in (625664, 626432, 630784, 644608, 650240)],
        user_evidence='Source 50.360 s screenshot identifies the lower-right trapezoid; '
                      'the repair follows the continuous source run, not one frame.',
        before_sha256=before_hash, original_uv_preserved=True,
        weights_or_uncertainties_added=False, native_geometry_modified=False,
        observations=[dict(row=int(i), observation_id=str(data['observation_ids'][i]),
                           source_row=int(data['source_rows'][i]),
                           pts=int(data['pts'][i]), uv=data['uv'][i].tolist()) for i in rows])
    archive = root / 'point_observations_imported.npz'
    if archive.exists():
        if prepared.digest(archive) != before_hash:
            raise ValueError('Existing imported observation archive differs')
    else:
        shutil.copy2(source, archive)
    prepared.write(root / 'binding_review' / 'manifest_before_trapezoid_repair.json', manifest)
    data['path_indices'][rows] = new_index
    pending = root / 'point_observations.pending.npz'
    np.savez(pending, **data)
    after = prepared.npz(pending)
    original = prepared.npz(archive)
    for key in prepared.FIELDS - {'path_indices'}:
        np.testing.assert_array_equal(after[key], original[key])
    np.testing.assert_array_equal(np.flatnonzero(after['path_indices'] != original_paths), rows)
    after_hash = prepared.digest(pending)
    receipt['after_sha256'] = after_hash
    manifest['point_input_sha256'] = after_hash
    manifest.setdefault('source_repair_history', []).append(dict(
        receipt=str(review), before_sha256=before_hash, after_sha256=after_hash,
        rebound_observations=len(rows), original_uv_preserved=True))
    prepared.write(review, receipt)
    prepared.write(root / 'joint_manifest.pending.json', manifest)
    pending.replace(source)
    (root / 'joint_manifest.pending.json').replace(manifest_path)
    print(json.dumps({key: receipt[key] for key in ('frames', 'rows', 'source_seconds',
                     'old_path_id', 'new_path_id', 'after_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    repair()
