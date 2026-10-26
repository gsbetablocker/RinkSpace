"""Extend checked complete artwork identities with actual adjacent-image tracks."""
from pathlib import Path
import sys,json,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_tracks_s01 as t
r=t.reg;b=r.b;OUT=r.OUT
def run():
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];pts=sorted(r.VIS);index={p:i for i,p in enumerate(pts)};start=time.monotonic()
    for ai in [4,71,76,11,12,67]:
        rows=doc['records'][str(ai)];available={int(p):v for p,v in rows.items()};ref=r.reference(ats[ai]);seen=set()
        seeds=sorted(available,key=lambda p:-(available[p].get('full_pattern_score') or {}).get('pattern_correlation_median',0))
        for seed in seeds:
            if seed in seen:continue
            seen.add(seed)
            for direction in [-1,1]:
                H=np.asarray(available[seed]['H']);prev=seed
                for j in range(index[seed]+direction,len(pts) if direction>0 else -1,direction):
                    pt=pts[j]
                    if pt in seen:break
                    result,reason=t.step(ref,H,prev,pt)
                    if result is None:break
                    nextH=np.asarray(result['H'])
                    if pt in available:
                        q=ref['points'][ref['probe']];delta=np.linalg.norm(cv2.perspectiveTransform(q[None],nextH)[0]-cv2.perspectiveTransform(q[None],np.asarray(available[pt]['H']))[0],axis=1)
                        if np.median(delta)>5:break
                        nextH=np.asarray(available[pt]['H'])
                    else:rows[str(pt)]=dict(result,atlas_id=ai,pts=pt,method='original_artwork_identity_through_adjacent_actual_RGB',root_seed_pts=seed,previous_pts=prev,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_projection_used=False)
                    seen.add(pt);prev=pt;H=nextH
        print(json.dumps(dict(stage='complete_artwork_source_tracks',atlas=ai,frames=len(rows),seconds=round(time.monotonic()-start,1))),flush=True)
    b.write(OUT/'source_identity_charts.json',doc)
if __name__=='__main__':run()
