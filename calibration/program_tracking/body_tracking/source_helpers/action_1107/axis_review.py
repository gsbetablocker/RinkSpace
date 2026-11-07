from pathlib import Path
import json, sys
import av, cv2, numpy as np
from scipy.spatial.transform import Rotation
BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PRE = BASE / 'prepared_physics'
OLD = BASE.parent / 'trajectory_continue_1107/temporal_joint_cached/final/fitted/result.npz'
MODEL = BASE / 'physics_guarded/final/fitted/result.npz'
EDGES = [(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
POINTS = [5,6,11,12,13,14,15,16]
def render(candidate=MODEL, name='physics'):
    prep = json.loads((PRE/'preparation.json').read_text('utf8'))
    p=np.load(PRE/'input.npz')
    sec=p['pts']/12800
    use=np.flatnonzero((sec>=77.62)&(sec<=78.36))
    stamps={p['pts'][i]:i for i in use}
    a=np.load(OLD); b=np.load(candidate); o=np.load(PRE/'observations.npz')
    for z in (a,b,o): np.testing.assert_array_equal(z['pts'],p['pts'])
    positions=(None,a['COCO_uv'],b['COCO_uv'],o['uv'])
    src=[]
    W,H=248,230
    bboxes=p['source_bbox']
    records=[]
    with av.open(prep['source_video']) as video:
        s=video.streams.video[0]
        assert (s.time_base.numerator,s.time_base.denominator)==(1,12800)
        s.codec_context.thread_count=2
        video.seek(int(min(stamps)), stream=s, backward=True)
        for f in video.decode(s):
            if f.pts>max(stamps):break
            if f.pts not in stamps:continue
            i=stamps[f.pts]
            image=f.to_ndarray(format='bgr24')
            c=bboxes[i]; center=(c[:2]+c[2:])/2
            extent=int(np.ceil(.66*np.max(c[2:]-c[:2])))
            x,y=np.maximum(center-extent,0).astype(int)
            x1,y1=np.minimum(center+extent,[image.shape[1],image.shape[0]]).astype(int)
            crop=image[y:y1,x:x1].copy()
            records.append(dict(pts=int(f.pts),row=int(i),crop_xyxy=[int(x),int(y),int(x1),int(y1)]))
            row=[]
            for uv,label in zip(positions,('source','old',name,'tracking')):
                tile=crop.copy()
                if uv is not None:
                    v=np.rint(uv[i]-[x,y]).astype(int)
                    for j,k in EDGES:
                        col=(255,205,45) if j in [5,7,11,13] else (45,150,255)
                        cv2.line(tile,tuple(v[j]),tuple(v[k]),col,2,cv2.LINE_AA)
                    for j in POINTS:cv2.circle(tile,tuple(v[j]),3,(245,245,245),-1,cv2.LINE_AA)
                    hip=(v[11]+v[12])//2; shoulder=(v[5]+v[6])//2
                    cv2.line(tile,tuple(hip),tuple(shoulder),(60,245,130),3,cv2.LINE_AA)
                factor=min(W/tile.shape[1],(H-19)/tile.shape[0])
                tile=cv2.resize(tile,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
                board=np.full((H,W,3),24,np.uint8)
                dx=(W-tile.shape[1])//2
                board[:tile.shape[0],dx:dx+tile.shape[1]]=tile
                cv2.putText(board,f'{f.pts/12800:.2f} | {label}',(7,H-5),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1,cv2.LINE_AA)
                row.append(board)
            src.append(np.hstack(row))
    assert len(src)==len(stamps)
    for i in range(0,len(src),5):
        cv2.imwrite(str(OUT/f'{name}_{i//5+1:02d}.jpg'),np.vstack(src[i:i+5]),[cv2.IMWRITE_JPEG_QUALITY,95])
    (OUT/'source_crops.json').write_text(json.dumps(records,indent=2),encoding='utf8')

if __name__ == '__main__':
    render(Path(sys.argv[1]),sys.argv[2]) if len(sys.argv)>1 else render()
