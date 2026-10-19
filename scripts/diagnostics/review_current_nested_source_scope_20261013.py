"""Bind exact current nested-paint questions to original pixels and same-model QA.

This does not change a pack, certify an inferred material, or run training.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.render_result import SceneRenderer
from rinkspace.programme_source_stream import FrozenProgrammeSource

WORK = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
DATASET = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'
OUT = WORK / 'current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1'


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def literal(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True, parents=True)
    dataset = json.loads(DATASET.read_text(encoding='utf8'))
    cut = next(c for c in dataset['cuts'] if c['source_pack']['path'].endswith('223403520_R0402_034.npz'))
    # This is an inventory, not a new FIT authority. Decode the authenticated
    # saved graph once without re-running every row's upstream qualification.
    if file_sha(cut['source_pack']['path']) != cut['source_pack']['sha256']:
        raise ValueError('Original source pack bytes changed')
    with np.load(cut['source_pack']['path'], allow_pickle=False) as z:
        values = {k:z[k] for k in ('row_offsets','uv','original_point_indices',
            'source_proof_template_index','source_frame_actual_pts',
            'source_frame_native_pts','source_frame_BGR_sha256')}
        rows = unpack_source_json(z['rows_json'])
        templates = unpack_source_json(z['source_proof_templates_json'])
        metadata = json.loads(str(z['metadata_json'].item()))
    pack = SimpleNamespace(values=values, rows=rows, metadata=metadata,
        frame_bindings={int(p):dict(native_pts=int(n),decoded_BGR_sha256=s.decode())
            for p,n,s in zip(values['source_frame_actual_pts'],values['source_frame_native_pts'],
                values['source_frame_BGR_sha256']) if n>=0 and s})
    with np.load(cut['support_index']['path'], allow_pickle=False) as z: fit = z['fit_point_indices']
    offsets, uv = pack.values['row_offsets'], pack.values['uv']
    targets = ('F3:middle', 'F3:inner', 'F3:center')
    records = []
    counts = Counter()
    for ri, row in enumerate(pack.rows):
        if row['physical_material_id'] not in targets: continue
        lo, hi = map(int, offsets[ri:ri+2])
        indices = fit[(fit >= lo) & (fit < hi)]
        if not len(indices): continue
        proofs = [dict(templates[int(p)],uv=u.tolist()) for p,u in
            zip(values['source_proof_template_index'][indices],uv[indices])]
        # Store full evidence for a small targeted exposure; inventory all others.
        rec = dict(original_source_row_index=ri, actual_pts=row['actual_pts'],
            physical_material_id=row['physical_material_id'], point_indices=indices,
            point_local_indices=indices-lo, uv=uv[indices],
            source_seed_authority=row.get('source_seed_authority'),
            temporal_evidence=row.get('temporal_evidence'),
            source_identity_reference=row.get('source_identity_reference'),
            source_scope_proofs=proofs,
            source_paint_identity_evidence=row.get('source_paint_identity_evidence'),
            original_source_point_indices=pack.values['original_point_indices'][indices])
        if row['actual_pts'] == 229333248:
            rec['complete_original_row_header'] = row
        records.append(rec)
        counts[row['physical_material_id']] += len(indices)
    result = dict(schema='rinkspace.current-nested-source-scope-point-review.v1',
        frozen_dataset_reference=ref(DATASET), source_pack_reference=cut['source_pack'],
        support_index_reference=cut['support_index'], frozen_model_reference=dataset['frozen_model_reference'],
        exact_original_FIT_points_by_material=dict(counts), records=records,
        fact='User-confirmed F3 outer carrier is retained; it does not by itself certify every inferred inner layer.',
        decision='targeted_original_RGB_review_in_progress', source_FIT_admitted=False,
        frozen_dataset_changed=False, confirmed_source_points_changed=False,
        reflection_or_other_negative_labels_created=False, training_started=False)
    inventory = OUT / '034_F3_nested_current_point_inventory.json'
    inventory.write_text(json.dumps(result, ensure_ascii=False, default=literal, separators=(',', ':')), encoding='utf8')
    print(json.dumps(dict(inventory=ref(inventory), counts=dict(counts), exposures=len(set(r['actual_pts'] for r in records)))), flush=True)
    source = Path(pack.metadata['source']['path']) if pack.metadata['source'].get('path') else Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
    programme = FrozenProgrammeSource(dataset['frozen_model_reference']['path'], source, ROOT / 'calibration/program_tracking/source_shots.json')
    renderer = SceneRenderer(programme.scene, draw_grid=True, draw_advertisements=True)
    selected = [r for r in records if r['actual_pts'] == 229333248]
    frames = []
    for frame in programme.frames(start_pts=229332992, stop_pts=229333505, cpu_threads=1):
        if frame['native_pts'] not in (229332992, 229333248, 229333504): continue
        pts = int(frame['native_pts']); raw = frame['raw_bgr']
        bgr_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        if pts in pack.frame_bindings and bgr_sha != pack.frame_bindings[pts]['decoded_BGR_sha256']:
            raise ValueError('Exact original native decode does not match frozen source binding')
        raw_path = OUT / f'{pts}_nested_review_EXACT_RAW.png'
        cv2.imwrite(str(raw_path), raw)
        canvas, _ = renderer.draw(raw, frame['camera25'], title=f'Same 8a28 ALL GRID; exact original nested-layer review; native {pts}')
        grid_path = OUT / f'{pts}_nested_review_same8a28_ALL_GRID.png'
        cv2.imwrite(str(grid_path), canvas)
        if pts == 229333248:
            labelled = raw[790:935, 50:470].copy()
            for row in selected:
                for global_i, p in zip(row['point_indices'], row['uv']):
                    q = tuple(np.rint(p-np.array([50,790])).astype(int))
                    color = (0,0,255) if row['physical_material_id'].endswith('center') else (0,255,255)
                    cv2.circle(labelled, q, 2, color, 1, cv2.LINE_AA)
                    cv2.putText(labelled, str(global_i), (q[0]+3,q[1]-3), cv2.FONT_HERSHEY_SIMPLEX,.28,color,1,cv2.LINE_AA)
            label_path = OUT / f'{pts}_nested_review_EXACT_RAW_POINT_IDS.png'
            cv2.imwrite(str(label_path), labelled)
        crop_path = OUT / f'{pts}_nested_review_EXACT_RAW_CROP.png'
        cv2.imwrite(str(crop_path), raw[790:935,50:470])
        frames.append(dict(source_binding=frame['source_binding'], decoded_BGR_sha256=bgr_sha,
            raw_reference=ref(raw_path), raw_crop_reference=ref(crop_path), same_model_QA_reference=ref(grid_path)))
    frame_path = OUT / '034_F3_nested_original_neighbor_frame_receipt.json'
    frame_path.write_text(json.dumps(dict(schema='rinkspace.exact-native-nested-review-QA.v1',frames=frames,
        same_model_is_navigation_only=True, pixel_label_is_review_pointer_only=True,
        material_or_FIT_decision_created=False, implementation_reference=ref(__file__)), ensure_ascii=False), encoding='utf8')
    print(json.dumps(dict(receipt=ref(frame_path), frames=len(frames))), flush=True)


if __name__ == '__main__': main()
