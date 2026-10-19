"""Replay exact current inherited inner points in cuts 038/039, without FIT."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.render_result import SceneRenderer
from calibration.program_tracking.strict_model_scope import file_sha
from rinkspace.programme_source_stream import FrozenProgrammeSource

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012/current_source_pool_joint_qualified_20261013_v1/source_scope_accuracy_review_v1')


def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))


def main():
    cv2.setNumThreads(1)
    inventory_path=WORK/'terminal_nine_specific_inner_seed_chain_inventory.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    records=[r for r in inventory['specific_inner_source_review_needed_records']if r['cut']in ('038','039')]
    by_pts={}
    for r in records:by_pts.setdefault(r['native_pts'],[]).append(r)
    dataset=json.loads(Path(inventory['frozen_finite_v1_dataset_reference']['path']).read_text(encoding='utf8'))
    programme=FrozenProgrammeSource(dataset['frozen_model_reference']['path'],
        Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4'),ROOT/'calibration/program_tracking/source_shots.json')
    renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
    out=WORK/'038_039_inherited_inner_exact_source_review'
    if out.exists():raise FileExistsError('Preserve earlier source point reviews')
    out.mkdir()
    frames=[]
    for frame in programme.frames(start_pts=min(by_pts),stop_pts=max(by_pts)+1,cpu_threads=1):
        pts=int(frame['native_pts'])
        if pts not in by_pts:continue
        rows=by_pts[pts];raw=frame['raw_bgr'];digest=frame['source_binding']['decoded_native_BGR_sha256']
        if any(r['decoded_BGR_sha256']!=digest for r in rows):raise ValueError('Frozen point exposure differs from exact original pixels')
        uv=np.concatenate([np.asarray(r['uv'])for r in rows])
        x0,y0=np.maximum(np.floor(uv.min(0)).astype(int)-65,0);x1,y1=np.minimum(np.ceil(uv.max(0)).astype(int)+66,[1920,1080])
        crop=raw[y0:y1,x0:x1].copy();path=out/f'{pts}_EXACT_RAW_CROP.png';cv2.imwrite(str(path),crop)
        canvas,_=renderer.draw(raw,frame['camera25'],title=f'Same 8a28 ALL GRID; inherited inner material review; native {pts}')
        grid_path=out/f'{pts}_same8a28_ALL_GRID.png';cv2.imwrite(str(grid_path),canvas)
        marked=crop.copy()
        for row in rows:
            color={'middle':(0,255,255),'inner':(0,255,0),'center':(0,0,255)}[row['physical_material_id'].rsplit(':',1)[-1]]
            for gi,p in zip(row['original_pack_global_point_indices'],row['uv']):
                q=tuple(np.rint(np.asarray(p)-[x0,y0]).astype(int));cv2.circle(marked,q,2,color,1,cv2.LINE_AA)
                cv2.putText(marked,str(gi),(q[0]+3,q[1]-3),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1,cv2.LINE_AA)
        mp=out/f'{pts}_EXACT_SOURCE_POINT_IDS.png';cv2.imwrite(str(mp),marked)
        frames.append(dict(native_pts=pts,actual_pts=frame['actual_pts'],cut=rows[0]['cut'],source_binding=frame['source_binding'],
            crop_native_xyxy=[int(x0),int(y0),int(x1),int(y1)],exact_raw_crop_reference=ref(path),
            same_model_full_grid_reference=ref(grid_path),point_pointer_reference=ref(mp),
            original_pack_global_point_indices=[i for r in rows for i in r['original_pack_global_point_indices']]))
    if len(frames)!=len(by_pts):raise ValueError('Current selected source point exposure replay incomplete')
    sheets=[]
    for start in range(0,len(frames),8):
        block=frames[start:start+8];crops=[cv2.imread(r['exact_raw_crop_reference']['path'])for r in block]
        width=max(c.shape[1]for c in crops);height=max(c.shape[0]for c in crops)+25
        sheet=np.full((height*4,width*2,3),238,np.uint8)
        for i,(c,r)in enumerate(zip(crops,block)):
            y=(i//2)*height;x=(i%2)*width;sheet[y+25:y+25+c.shape[0],x:x+c.shape[1]]=c
            cv2.putText(sheet,f"{r['cut']} native {r['native_pts']} xy {r['crop_native_xyxy'][:2]}",(x+3,y+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        sp=out/f'EXACT_NATIVE_PIXEL_CROPS_page_{start//8+1:02d}.png';cv2.imwrite(str(sp),sheet);sheets.append(ref(sp))
    result=dict(schema='rinkspace.exact-inherited-inner-selected-point-source-review-sequence.v1',
        inventory_reference=ref(inventory_path),frozen_model_reference=dataset['frozen_model_reference'],
        records=frames,native_scale_raw_crop_contact_sheets=sheets,decoded_selected_exposures=len(frames),
        exact_selected_points=sum(len(r['original_pack_global_point_indices'])for r in frames),
        material_scope_decision_not_created=True,model_is_navigation_only=True,
        cropped_source_pixels_resized_or_interpolated=False,implementation_reference=ref(__file__))
    rp=out/'exact_038_039_selected_inner_source_sequence.json';rp.write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(receipt=ref(rp),exposures=len(frames),points=result['exact_selected_points'])),flush=True)


if __name__=='__main__':main()
