"""Produce exact native crop evidence for every selected nested-layer exposure."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.render_result import SceneRenderer
from calibration.program_tracking.strict_model_scope import file_sha
from rinkspace.programme_source_stream import FrozenProgrammeSource

WORK = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1')


def ref(path): return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def main():
    cv2.setNumThreads(1)
    inventory_path=WORK/'034_F3_nested_current_point_inventory.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    grouped={}
    for row in inventory['records']: grouped.setdefault(row['actual_pts'],[]).append(row)
    out=WORK/'034_F3_all_37_exact_source_exposures'
    if out.exists(): raise FileExistsError('Keep completed earlier source reviews unchanged')
    out.mkdir()
    programme=FrozenProgrammeSource(inventory['frozen_model_reference']['path'],
        Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4'),
        ROOT/'calibration/program_tracking/source_shots.json')
    renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
    records=[]
    for frame in programme.frames(start_pts=min(grouped),stop_pts=max(grouped)+1,cpu_threads=1):
        pts=int(frame['native_pts'])
        if pts not in grouped:continue
        if not frame['representative']:raise ValueError('Selected nested point frame is not canonical actual source')
        rows=grouped[pts];uv=np.concatenate([np.asarray(r['uv'])for r in rows])
        x0,y0=np.maximum(np.floor(uv.min(0)).astype(int)-75,0)
        x1,y1=np.minimum(np.ceil(uv.max(0)).astype(int)+76,[1920,1080])
        raw=frame['raw_bgr'];crop=raw[y0:y1,x0:x1].copy()
        raw_path=out/f'{pts}_EXACT_RAW_CROP.png';cv2.imwrite(str(raw_path),crop)
        canvas,_=renderer.draw(raw,frame['camera25'],title=f'8a28 ALL GRID; nested source identity review; actual {pts}')
        grid_path=out/f'{pts}_same8a28_ALL_GRID.png';cv2.imwrite(str(grid_path),canvas)
        marked=crop.copy()
        for row in rows:
            color={'F3:middle':(0,255,255),'F3:inner':(0,255,0),'F3:center':(0,0,255)}[row['physical_material_id']]
            for pi,p in zip(row['point_indices'],row['uv']):
                q=tuple(np.rint(np.asarray(p)-[x0,y0]).astype(int))
                cv2.circle(marked,q,2,color,1,cv2.LINE_AA)
                cv2.putText(marked,str(pi),(q[0]+3,q[1]-3),cv2.FONT_HERSHEY_SIMPLEX,.26,color,1,cv2.LINE_AA)
        mark_path=out/f'{pts}_EXACT_SOURCE_POINT_IDS.png';cv2.imwrite(str(mark_path),marked)
        record=dict(actual_pts=pts,source_binding=frame['source_binding'],
            crop_native_xyxy=[int(x0),int(y0),int(x1),int(y1)],
            decoded_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            exact_raw_crop_reference=ref(raw_path),same_model_full_grid_reference=ref(grid_path),
            point_pointer_reference=ref(mark_path),exact_pending_candidate_indices=[i for r in rows for i in r['point_indices']])
        records.append(record)
        print(json.dumps(dict(actual_pts=pts,points=len(record['exact_pending_candidate_indices']))),flush=True)
    if len(records)!=len(grouped):raise ValueError('Exact source review exposure coverage incomplete')
    # Native-scale contact sheets use copies of original crop pixels. The
    # separate marked views and same-model full grid remain independently linked.
    sheets=[]
    for start in range(0,len(records),6):
        block=records[start:start+6]
        crops=[cv2.imread(r['exact_raw_crop_reference']['path'])for r in block]
        width=max(c.shape[1] for c in crops);height=max(c.shape[0]for c in crops)+25
        sheet=np.full((height*3,width*2,3),238,np.uint8)
        for i,(crop,r)in enumerate(zip(crops,block)):
            y=(i//2)*height;x=(i%2)*width
            sheet[y+25:y+25+crop.shape[0],x:x+crop.shape[1]]=crop
            cv2.putText(sheet,f"actual {r['actual_pts']} xy {r['crop_native_xyxy'][:2]}",(x+4,y+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        path=out/f'EXACT_NATIVE_PIXEL_CROPS_page_{start//6+1:02d}.png';cv2.imwrite(str(path),sheet);sheets.append(ref(path))
    result=dict(schema='rinkspace.exact-source-nested-material-whole-selected-sequence-review.v1',
        inventory_reference=ref(inventory_path),records=records,native_scale_raw_crop_contact_sheets=sheets,
        exposure_count=len(records),point_count=sum(len(r['exact_pending_candidate_indices'])for r in records),
        cropped_pixels_resized_or_interpolated=False,model_is_navigation_only=True,
        no_FIT_or_negative_decision_created=True,implementation_reference=ref(__file__))
    path=out/'exact_selected_source_sequence.json';path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(path),exposures=len(records),points=result['point_count'])),flush=True)


if __name__=='__main__':main()
