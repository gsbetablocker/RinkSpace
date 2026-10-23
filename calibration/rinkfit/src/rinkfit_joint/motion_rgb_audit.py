"""Re-decode every motion exposure from the authenticated official native video.

This checks the pixel/clock lineage, not the correctness of every finite world
instance. Diagnostic stills use measured UV only; no model projection is drawn.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,time
import cv2,numpy as np
from .observed_primal_restart import checked,reference,read,write_new
from .native_source import iter_scope_native_rgb


def audit(manifest,scope,roles,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    packet=read(manifest);expected={};byshot=defaultdict(list);seen=set();shots=Counter();samples={}
    for pair in packet['pairs']:
        raw=read(checked(pair['source']));byshot[raw['shot_id']].append((pair,raw))
        for pts,sha in [(raw['previous_native_pts'],raw['previous_source_RGB_sha256']),(raw['native_pts'],raw['current_source_RGB_sha256'])]:
            if pts in expected and expected[pts]!=sha:raise ValueError('Conflicting frame identity')
            expected[pts]=sha
    for sid,pairs in byshot.items():
        for i in sorted({0,len(pairs)//2,len(pairs)-1}):
            pair,raw=pairs[i];samples[raw['native_pts']]=(sid,np.asarray(raw['current_uv'])[pair['selected_indices']])
    source=read(scope)['source'];role=read(roles)
    if role['source_video_sha256']!=source['sha256']:raise ValueError('Role/video identity differs')
    start=time.monotonic();images=[]
    with (out/'frames.jsonl').open('x',encoding='utf8') as stream:
        for pts,rgb,receipt in iter_scope_native_rgb(scope,expected,roles_path=roles):
            if receipt['decoded_native_RGB_sha256']!=expected[pts]:raise ValueError('Actual RGB differs at '+str(pts))
            stream.write(json.dumps(receipt)+'\n');seen.add(pts);shots[receipt['shot_id']]+=1
            if pts in samples:
                sid,uv=samples[pts];bgr=rgb[:,:,::-1].copy()
                for u,v in uv:
                    x,y=int(round(u)),int(round(v));cv2.rectangle(bgr,(x-10,y-10),(x+10,y+10),(40,240,40),1)
                cv2.rectangle(bgr,(0,0),(1920,42),(15,15,15),-1)
                cv2.putText(bgr,f'{sid} PTS={pts} t={pts/12800:.3f}s | actual selected 21px patches', (14,29),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                file=out/f'{sid}_{pts}.png';cv2.imwrite(str(file),bgr);images.append((sid,pts,file))
            if len(seen)%250==0:print('verified RGB exposures',len(seen),flush=True)
    if seen!=set(expected):raise ValueError('Missing complete exposure denominator')
    thumbs=[cv2.resize(cv2.imread(str(f)),(640,360),interpolation=cv2.INTER_AREA)for sid,pts,f in sorted(images)]
    if len(thumbs)==6:cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(thumbs[:3]),np.hstack(thumbs[3:])]))
    result=dict(schema='rinkfit.actual_motion_RGB_redecode_audit.v1',motion_packet=reference(manifest),scope=reference(scope),roles=reference(roles),
        source=source,expected_native_exposures=len(expected),actual_verified_exposures=len(seen),by_shot=dict(shots),
        exact_RGB_hashes_match=True,native_PTS_and_roles_verified=True,frame_receipts=reference(out/'frames.jsonl'),
        source_UV_modified=False,world_instance_qualification_issued=False,fit_performed=False,seconds=time.monotonic()-start,
        diagnostic_stills=[reference(f)for sid,pts,f in images])
    write_new(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','scope','roles','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(audit(a.manifest,a.scope,a.roles,a.output)))
if __name__=='__main__':main()
