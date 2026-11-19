"""Recheck real red paint flanks, retaining source pixels and proposal history."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import re
import sys
import cv2
import numpy as np
from shared import DATA,EVENT,OUT,arrays,read,save,write
from review import camera_state,native_for,source_frames,ji

sys.path.append(str(EVENT/'programmes/chen_xi_jin_sp/tools'))
from ice_paint_profiles import read_paint_profiles

RED=re.compile(r'^(A[1-4](\..*)?|C0|goal_line_.*|trapezoid_.*)$')


def matte(shot,pts):
    root=DATA/'events/ankara_2026/programmes'
    if shot['program']=='chen':
        folder=root/'chen_xi_jin_sp/fit_1115/visibility_source_final'
        path=read(folder/'binding.json')['shots'][shot['shot']]['packet']
    elif shot['program']=='nayeon':
        path=root/'nayeon_ko_sp/fit_1116/shared4_rework/matte_review'/(shot['shot']+'.npz')
    else:return None
    a=arrays(path);np.testing.assert_array_equal(a['pts'],pts)
    assert a['source_sha256'].item()==shot['source_sha256']
    return a


def one(name,shot,base,tone='red'):
    a=arrays(OUT/'prepared'/(name+'.npz'));raw_packet=arrays(shot['observations'])
    material=(np.array([bool(RED.fullmatch(f)) for f in a['family']]) if tone=='red' else np.isin(a['family'],['B1','B2']))
    selected=np.flatnonzero(material&(a['fold']<5)&(a['origin_row']>=0))
    if not len(selected):return name,dict(rows=0)
    pts=base[name+'_pts'];c=base[name+'_controls'];r=base[name+'_rigs'];times=base[name+'_times']
    native=native_for('original',shot['program']);table=ji.native_paths(native,np.unique(a['path'][selected]),count=257)
    masks=matte(shot,pts)
    uv=a['uv'][selected];normal=np.zeros_like(uv);width=np.full(len(uv),12.)
    if 'source_bracket' in raw_packet:
        old=raw_packet['source_bracket'][a['origin_row'][selected]]
        delta=old[:,1]-old[:,0];length=np.linalg.norm(delta,axis=1)
        valid=np.isfinite(delta).all(1)&(length>2.)
        width[valid]=length[valid];normal[valid]=delta[valid]/length[valid,None]
    valid_source=np.zeros(len(uv),bool);proposed=np.full_like(uv,np.nan)
    edges=np.full((len(uv),2,2),np.nan);contrast=np.full((len(uv),2),np.nan)
    score=np.full(len(uv),np.nan);confirmed_old=np.zeros(len(uv),bool)
    hashes=[];frame_pts=[]
    for pt,raw in source_frames(shot['source'],np.unique(a['pts'][selected])):
        loc=np.flatnonzero(a['pts'][selected]==pt);ix=selected[loc];frame=int(a['idx'][ix[0]])
        mask=np.zeros(raw.shape[:2],np.uint8)
        if masks is not None and masks['valid'][frame]:
            mask=cv2.dilate(np.unpackbits(masks['packed'][frame],axis=1,count=1920),np.ones((7,7),np.uint8))
        # Broadcast graphics are not painted ice.
        mask[:158,:590]=1;mask[:292,:451]=1;mask[928:,1618:]=1
        missing=loc[np.linalg.norm(normal[loc],axis=1)<.9]
        for path in np.unique(a['path'][selected[missing]]):
            which=missing[a['path'][selected[missing]]==path]
            world,st=table[path]
            z=native.core.project_motion(r[frame],c,shot['first'],shot['dt'],1080,times[frame],world)
            poly=z['uv'];poly[z['status']!=0]=np.nan
            delta=np.gradient(poly,axis=0);length=np.linalg.norm(delta,axis=1)
            direction=np.c_[-delta[:,1],delta[:,0]]/np.maximum(length[:,None],1.e-12)
            u=a['station'][selected[which]]
            normal[which]=np.c_[np.interp(u,st,direction[:,0]),np.interp(u,st,direction[:,1])]
        normal[loc]/=np.maximum(np.linalg.norm(normal[loc],axis=1)[:,None],1.e-12)
        hits,old_ok=read_paint_profiles(raw,mask,uv[loc],normal[loc],width[loc],np.repeat(tone,len(loc)))
        if tone=='blue' and 'source_bracket' not in raw_packet:
            # Legacy blue readings may lie at a paint edge. Search for the
            # complete material band at additional source-only width scales.
            # A projection never supplies the resulting midpoint or width.
            for trial_width in (24.,48.,96.):
                todo=np.array([i for i,h in enumerate(hits) if h is None],int)
                if not len(todo):break
                extra,_=read_paint_profiles(raw,mask,uv[loc[todo]],normal[loc[todo]],
                    np.full(len(todo),trial_width),np.repeat('blue',len(todo)))
                for j,hit in zip(todo,extra):
                    if hit is not None and hit[1]<=4. and hit[2]>=-4.:hits[j]=hit
        for j,hit,old in zip(loc,hits,old_ok):
            confirmed_old[j]=old
            if hit is None:continue
            value,left,right,strength,slope,method=hit
            edges[j]=uv[j]+np.array([left,right])[:,None]*normal[j]
            proposed[j]=edges[j].mean(0);contrast[j]=strength;score[j]=value
            valid_source[j]=True
        frame_pts.append(pt);hashes.append(sha256(raw.tobytes()).hexdigest())
    save(OUT/(tone+'_paint_source')/(name+'.npz'),selected_row=selected,origin_row=a['origin_row'][selected],
        pts=a['pts'][selected],path=a['path'][selected],family=a['family'][selected],fold=a['fold'][selected],
        old_uv=uv,proposed_uv=proposed,normal=normal,source_bracket=edges,source_side_contrast=contrast,
        source_valid=valid_source,old_flanks_confirmed=confirmed_old,score=score,
        source_frame_pts=np.asarray(frame_pts),source_frame_BGR_sha256=np.asarray(hashes),
        source_sha256=shot['source_sha256'])
    result=dict(rows=len(uv),paired_flanks=int(valid_source.sum()),old_pair_confirmed=int(confirmed_old.sum()),
        frames=len(frame_pts),source=shot['source'],operator='existing ice_paint_profiles raw two-flank reader',
        geometry_role='orientation only for legacy rows lacking normals; no projection used as a measurement')
    print(name,result['rows'],result['paired_flanks'],result['old_pair_confirmed'],flush=True)
    return name,result


def run(programmes,tone='red',shots=None):
    cv2.setNumThreads(1)
    folder=OUT/(tone+'_paint_source');folder.mkdir(exist_ok=True)
    setup=read(OUT/'prepared/setup.json');base=camera_state('original')
    jobs=[(name,s,base,tone) for name,s in setup['shots'].items()
          if (not programmes or s['program'] in programmes) and (not shots or name in shots)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        result=dict(pool.map(lambda x:one(*x),jobs))
    previous=read(folder/'audit.json') if (folder/'audit.json').exists() else {}
    previous.update(result);write(folder/'audit.json',previous)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--programmes',nargs='+')
    ap.add_argument('--tone',choices=('red','blue'),default='red');ap.add_argument('--shots',nargs='+')
    args=ap.parse_args();run(args.programmes,args.tone,args.shots)
