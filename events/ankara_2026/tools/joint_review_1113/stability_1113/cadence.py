"""Read visually repeated source presentations directly, without using fitted camera."""
from concurrent.futures import ThreadPoolExecutor
from track_source import source_gray
from camera_audit import HERE,setup,sj,vj
from time import monotonic
import cv2
import numpy as np


def read(d):
    previous=None
    result=[]
    for pic in source_gray(d):
        pic=cv2.resize(pic,(480,270),interpolation=cv2.INTER_AREA)[64:235,12:-12].astype('f4')
        if previous is not None:
            diff=np.abs(pic-previous)
            result.append([float(np.mean(diff)),float(np.quantile(diff[::3,::3],.95))])
        previous=pic
    return np.array(result)


def main():
    cv2.setNumThreads(1)
    m,c,base,z=setup()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={k:pool.submit(read,vj.shot_from_inputs(k,m,base)) for k in m['shots']}
        result={}
        rows={}
        for k,t in jobs.items():
            a=t.result()
            ratio=a[1:-1,0]/np.maximum((a[2:,0]+a[:-2,0])*.5,1.e-8)
            ix=np.flatnonzero((ratio<.25)&(a[1:-1,0]<1.25))+1
            pts=base[k+'_pts'][1:]
            result[k]=a
            rows[k]=[[float(pts[i])/12800.,*a[i].tolist(),float(ratio[i-1])] for i in ix]
            print(k,len(a),rows[k][:14],flush=True)
    sj.atomic_arrays(HERE/'cadence.npz',**result)
    sj.write_json(HERE/'cadence.json',rows)


if __name__=='__main__':
    main()
