"""Render exact flagged R3/R2 source pixels and their actual native context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from calibration.program_tracking.render_result import SceneRenderer
from rinkspace.programme_source_stream import FrozenProgrammeSource

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
AUDIT=WORK/'current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1/terminal_nine_specific_inner_seed_chain_inventory.json'
OUT=AUDIT.parent/'037_R3_001_R2_actual_specific_material_review_20261013_v1'
MODEL=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012/accepted_states/8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891/state_identity.json')
SOURCE=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
COLORS={'middle':(0,170,255),'inner':(255,220,0),'center':(0,220,0)}


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def literal(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)


def main():
    if OUT.exists():raise FileExistsError('Keep previous source reviews unchanged')
    OUT.mkdir(parents=True)
    cv2.setNumThreads(1)
    inventory=json.loads(AUDIT.read_text(encoding='utf8'))
    targets=[r for r in inventory['specific_inner_source_review_needed_records']if r['cut']in('037','001')]
    by_native={};contexts=[]
    for cut in('037','001'):
        current=[r for r in targets if r['cut']==cut];reference=current[0]['source_pack_reference']
        authenticate_source_reference(reference)
        with np.load(reference['path'],allow_pickle=False)as z:
            rows=unpack_source_json(z['rows_json']);offsets=z['row_offsets'];uv=z['uv'];q=z['source_side_quantiles_uv'];sigma=z['sigma_px'];original=z['original_point_indices']
            for target in current:
                row=rows[target['original_source_row_index']]
                indices=np.asarray(target['additional_review_needed_indices'],np.int64)
                if (not np.array_equal(uv[indices],np.asarray(target['uv']))
                        or row['actual_pts']!=target['actual_pts']or row['native_pts']!=target['native_pts']):
                    raise ValueError('Source row or exact finite point inventory changed')
                entry=dict(target,exact_original_point_indices=original[indices],source_sigma_px=sigma[indices],
                    source_quantiles_uv=q[indices],full_source_row_context=row,
                    source_reference_unchanged=True,source_point_decision='awaiting_specific_original_material_review')
                contexts.append(entry);by_native.setdefault(row['native_pts'],[]).append(entry)
    context_path=OUT/'exact_target_source_rows_and_full_context.json'
    context_path.write_text(json.dumps(dict(schema='rinkspace.037-R3-001-R2-exact-inner-source-review-context.v1',
        inventory_reference=ref(AUDIT),records=contexts,source_model_navigation_reference=ref(MODEL),
        inherited_outer_authority_not_revoked=True,original_source_pixels_changed=False),
        ensure_ascii=False,default=literal,separators=(',',':')),encoding='utf8')
    programme=FrozenProgrammeSource(MODEL,SOURCE,ROOT/'calibration/program_tracking/source_shots.json')
    renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
    # The first seed and each distinct later reentry are read in their own source
    # neighbourhood. This does not replay other feature families or fit a scene.
    windows=[(229839104,229847809),(229858816,229859329),(229903872,229904385),
        (230686720,230713857)]
    seeds={229840384,230688512}
    selected=set(by_native)|seeds
    for p in list(selected):selected.update(x for x in(p-256,p+256)if x in programme.lookup)
    records=[];panels={'037':[],'001':[]}
    for start,stop in windows:
        for frame in programme.frames(start_pts=start,stop_pts=stop,cpu_threads=1):
            native=int(frame['native_pts'])
            if native not in selected:continue
            binding=dict(frame['source_binding'],native_pts=native)
            raw=frame['raw_bgr'];targets_here=by_native.get(native,[])
            for target in targets_here:
                if target['decoded_BGR_sha256']!=binding['decoded_native_BGR_sha256']:
                    raise ValueError('Target source point differs from decoded exact native BGR')
            exact=OUT/f'{native}_EXACT_RAW_NATIVE.png';cv2.imwrite(str(exact),raw)
            marked=raw.copy()
            for target in targets_here:
                role=target['physical_material_id'].rsplit(':',1)[-1];color=COLORS[role]
                for index,uv in zip(target['additional_review_needed_indices'],target['uv']):
                    point=tuple(np.rint(uv).astype(int));cv2.circle(marked,point,3,color,1,cv2.LINE_AA)
                    cv2.putText(marked,str(index),tuple(np.rint(np.asarray(uv)+[4,-4]).astype(int)),0,.3,color,1,cv2.LINE_AA)
            cut='037'if native<230000000 else'001'
            crop=(marked[700:1000,450:1800]if cut=='037'else marked[600:1010,:1600])
            crop_path=OUT/f'{native}_EXACT_SOURCE_MATERIAL_CROP.png';cv2.imwrite(str(crop_path),crop)
            canvas,_=renderer.draw(marked,frame['camera25'],title=f'8a28 ALL GRID; {cut} actual existing inner points; native {native}')
            grid=OUT/f'{native}_SAME8a28_ALL_GRID.png';cv2.imwrite(str(grid),canvas)
            record=dict(source_binding=binding,representative=bool(frame['representative']),
                decoded_native_RGB_sha256=hashlib.sha256(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
                raw_reference=ref(exact),source_point_crop_reference=ref(crop_path),all_grid_reference=ref(grid),
                target_source_row_indices=[r['original_source_row_index']for r in targets_here],
                target_global_point_indices=[i for r in targets_here for i in r['additional_review_needed_indices']],
                source_material_review_pending=True,source_scope_negative_asserted=False)
            path=OUT/f'{native}_exact_source_frame_review.json'
            path.write_text(json.dumps(record,ensure_ascii=False,separators=(',',':')),encoding='utf8')
            record['frame_review_reference']=ref(path);records.append(record)
            if targets_here or native in seeds:
                thumb=cv2.resize(crop,(1200,300 if cut=='037'else 308),interpolation=cv2.INTER_AREA)
                header=np.full((28,1200,3),245,np.uint8)
                cv2.putText(header,f'{cut} native {native}; middle=orange inner=cyan center=green; exact flagged points only',
                    (4,20),0,.5,(0,0,0),1,cv2.LINE_AA)
                panels[cut].append(np.concatenate([header,thumb]))
            print(json.dumps(dict(native_pts=native,cut=cut,flagged_points=len(record['target_global_point_indices'])),separators=(',',':')),flush=True)
    pages=[]
    for cut,items in panels.items():
        for start in range(0,len(items),5):
            path=OUT/f'{cut}_continuous_actual_source_layers_{start//5+1:02d}.png'
            cv2.imwrite(str(path),np.concatenate(items[start:start+5]));pages.append(ref(path))
    result=dict(schema='rinkspace.037-R3-001-R2-specific-inner-original-source-evidence.v1',
        target_inventory_reference=ref(AUDIT),exact_source_context_reference=ref(context_path),
        frozen_model_navigation_reference=ref(MODEL),implementation_reference=ref(Path(__file__)),
        original_seed_authorities=[r for r in inventory['authenticated_seed_authorities']],
        source_frames=records,continuous_source_review_pages=pages,
        exact_flagged_points=sum(len(r['additional_review_needed_indices'])for r in contexts),
        manual_false_used_as_rejection=False,outer_identity_authority_retracted=False,
        source_scope_negative_asserted=False,pending_filter_created=False,
        fitted_or_projected_UV_used_as_source_pixels=False,original_source_or_dataset_changed=False)
    path=OUT/'specific_inner_original_source_evidence.json'
    path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(reference=ref(path),source_frames=len(records),points=result['exact_flagged_points'],pages=len(pages))),flush=True)


if __name__=='__main__':main()
