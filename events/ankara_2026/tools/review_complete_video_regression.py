"""Decode exact shared source moments from existing complete videos."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2

def extract(video, output, pts):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with av.open(str(video)) as c:
        s=c.streams.video[0]
        for p in pts:
            index=(p-229114112)//256
            target=int(round(index/60/float(s.time_base)))
            c.seek(target,stream=s,backward=True,any_frame=False)
            found=None
            for f in c.decode(s):
                i=int(round(float(f.pts*f.time_base)*60))
                if i==index:
                    found=f
                    break
                if i>index:break
            if found is None:raise ValueError((p,index,target))
            path=output/f'video_native_source_{p}.png'
            if not cv2.imwrite(str(path),found.to_ndarray(format='bgr24')):raise OSError(path)
            rows.append(dict(source_pts=p,output_frame_index=index,path=str(path),decoded_pts=found.pts))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pts',required=True,type=int,nargs='+')
    a=p.parse_args()
    rows=extract(a.video,a.output,a.pts)
    (a.output/'checks.json').write_text(json.dumps(dict(schema='rinkspace.existing-video-exact-source-review.v1',video=str(a.video),rows=rows),indent=2),encoding='utf8')
    print(json.dumps(dict(images=len(rows),path=str(a.output))))

if __name__=='__main__':main()
