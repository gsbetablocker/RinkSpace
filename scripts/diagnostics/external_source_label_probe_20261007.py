"""Audit frozen FIT contact samples against newly decoded native RGB only.

No solver, inferred world geometry, or network output is used to locate pixels.
The two frames are chosen by nearest source group to shot temporal midpoints.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle

import av
import cv2
import numpy as np

cv2.setNumThreads(1)
BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
PACKET = BASE/'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    packet=pickle.loads(PACKET.read_bytes())
    rows=packet['programme_absolute_rows']
    selected=[]
    for block,midpoint in [('main4',229824000),('low',230157312)]:
        candidates=[r for r in rows if r['collection_role']=='FIT' and r['block']==block and r['source_row']['id']=='far_contact']
        selected.append(min(candidates,key=lambda r:abs(r['group_actual_pts']-midpoint)))
    source=BASE/'media/originals/oxXu1avhrvs.mp4'
    results=[]
    for item in selected:
        pts=int(item['group_actual_pts']);row=item['source_row'];uv=np.asarray(row['uv'])
        with av.open(str(source)) as c:
            stream=c.streams.video[0];stream.thread_count=1
            c.seek(pts-15360,stream=stream,backward=True,any_frame=False)
            frame=next(f for f in c.decode(stream) if f.pts==pts)
            bgr=frame.to_ndarray(format='bgr24')
        overlay=bgr.copy();observations=[]
        for index,(x,y) in enumerate(uv):
            x,y=int(x),int(y)
            cv2.drawMarker(overlay,(x,y),(0,0,255),cv2.MARKER_CROSS,15,1)
            cv2.putText(overlay,str(index),(x+6,y-6),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)
            ylo=max(0,y-24);yhi=min(1080,y+25)
            profile=np.median(bgr[ylo:yhi,max(0,x-5):min(1920,x+6)].astype(float),axis=1)
            # Explicit raw signal; no automatic promotion of its maximum to truth.
            yellow=(profile[:,1]+profile[:,2])/2-profile[:,0]
            observations.append(dict(index=index,original_uv=[x,y],profile_y=list(range(ylo,yhi)),bgr_median_11columns=profile.tolist(),yellow_contrast=yellow.tolist()))
        y0=max(0,int(uv[:,1].min())-60);y1=min(1080,int(uv[:,1].max())+60)
        cv2.imwrite(str(out/f'contact_{pts}_raw.png'),bgr[y0:y1])
        cv2.imwrite(str(out/f'contact_{pts}_stored_FIT.png'),overlay[y0:y1])
        results.append(dict(block=item['block'],pts=pts,source_time_seconds=pts/15360,crop_xyxy=[0,y0,1920,y1],decoded_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),frozen_source_method=row.get('method'),observations=observations))
    report=dict(schema='rinkspace.external-source-label-probe.v1',source_video=str(source),packet=str(PACKET),packet_sha256=hashlib.sha256(PACKET.read_bytes()).hexdigest(),selection='nearest existing FIT far_contact group to shot 4 and shot 5 temporal midpoints; no residual selection',current_solver_admission_rechecked=False,automatic_rgb_argmax_is_not_ground_truth=True,results=results)
    (out/'source_label_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'output':str(out),'selected_pts':[r['pts']for r in results],'source_points':sum(len(r['observations'])for r in results)}))


if __name__=='__main__':main()
