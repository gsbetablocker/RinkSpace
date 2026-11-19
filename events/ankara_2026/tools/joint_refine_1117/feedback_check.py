"""Compare the same original-source measurements and camera tracks across revisions."""
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from feedback_source import (CASES, DEST, PARENT, contour_samples, foreground, matte, read_contour,
                             selected_paths, signals)
from feedback_fit import extreme
from shared import OUT, PREPARED, PROGRAMMES, arrays, prepared_packet_path, read, save, write
from review import camera_state, ji, native_for, source_frames
from source_bundle import relative
from solve import stats


def low_sweep(programme):
    """Keep visibility and actual ink evidence for every assigned end-camera exposure."""
    state=camera_state(PARENT);n=native_for(PARENT,programme)
    groups=selected_paths(n);groups={k:groups[k] for k in ('F4','goal_crease_left','R4')}
    keys=[k for g in groups.values() for k in g];tab=ji.native_paths(n,keys,count=193)
    world={k:v[0] for k,v in tab.items()};setup=read(PREPARED/'setup.json')['shots']
    cv2.setNumThreads(1)
    for name,s in setup.items():
        if s['program']!=programme or s['camera_index']!=2:continue
        out=DEST/'end_sweep'/(name+'.npz')
        if out.exists():continue
        pts=state[name+'_pts'];times=state[name+'_times'];packed=matte(s,pts)
        material=[];ledger=np.zeros((len(pts),3,3),np.int32)
        frames=iter(source_frames(s['source'],pts))
        for start in range(0,len(pts),32):
            inds=np.arange(start,min(len(pts),start+32))
            jobs=[(i,k,times[i]) for i in inds for k in keys]
            curves=ji.project_groups(n,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
            lookup={(i,k):v for (i,k,_),v in zip(jobs,curves)}
            for i in inds:
                pt,img=next(frames);assert pt==pts[i]
                mask=None;gray=None
                for j,(fam,paths) in enumerate(groups.items()):
                    parts=contour_samples({k:lookup[i,k] for k in paths},tab,paths,fam)
                    ledger[i,j,0]=sum(len(z[1]) for z in parts)
                    if ledger[i,j,0]<8:continue
                    if gray is None:
                        gray=signals(img);mask=foreground(s,packed,i,pt)
                        if mask is not None:
                            mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
                            mask[:270,:492]=255;mask[936:,1670:]=255
                    if mask is None:continue
                    hits=read_contour(gray,mask,parts,fam)
                    ledger[i,j,1]=len(hits)
                    # Preserve a spread of actual raw pixels on both contours,
                    # including their most horizontally informative portions.
                    for key in paths:
                        hh=[h for h in hits if h[0]==key]
                        if len(hh)<8:continue
                        normals=np.abs([h[4][0] for h in hh])
                        take=np.unique(np.r_[np.linspace(0,len(hh)-1,min(12,len(hh))).astype(int),
                                             np.argsort(normals)[-6:]])
                        ledger[i,j,2]+=len(take)
                        for k in take:
                            h=hh[k];material.append((i,keys.index(key),h[1],*h[2],*h[4],*h[6]))
        z=np.asarray(material,float).reshape(-1,9)
        save(out,pts=pts,ledger=ledger,families=list(groups),keys=keys,index=z[:,0].astype(int),
             path=z[:,1].astype(int),station=z[:,2],uv=z[:,3:5],normal=z[:,5:7],shift=z[:,7:9],
             source_sha256=s['source_sha256'])
        print(name,len(pts),'exposures',len(z),'source edges',flush=True)


def compare_features(candidate,shots=None,scope='reads'):
    from lane_review import pixel_errors
    state=camera_state(candidate);n=native_for(candidate,'jin')
    setup=read(PREPARED/'setup.json')['shots'];results={}
    for name,s in setup.items():
        if shots and name not in shots:continue
        p=DEST/scope/(name+'.npz')
        if not p.is_file():continue
        a=arrays(p)
        if scope in ('end_sweep','material_sweep'):
            a['idx']=a['index'];a['path']=a['keys'][a['path']]
            a['pts']=a['pts'][a['idx']]
            a['family']=np.array([n.paths[k]['label'][4:].split(':')[0] for k in a['path']])
        if not len(a.get('idx',[])):continue
        # Re-evaluate the same measured pixels on the new finite paths.
        mask=np.ones(len(a['idx']),bool)
        if scope=='reads':
            from feedback_fit import admit
            mask=admit(name,s,a,False)<9
        if not mask.any():continue
        e,st=pixel_errors(n,state,s,name,a,mask)
        date=DEST/'scores'/candidate.replace('/','_')/scope
        save(date/(name+'.npz'),error=e,station=st)
        results[name]=stats(e[mask])
    return results


def shot_cases(candidate,names=None):
    """Horizontal extents, print shifts and paint midlines on each reported frame."""
    state=camera_state(candidate);n=native_for(candidate,'jin');groups=selected_paths(n)
    from feedback_source import read_rail, read_stripe
    setup=read(PREPARED/'setup.json')['shots']
    tab=ji.native_paths(n,[k for paths in groups.values() for k in paths],count=257)
    world={k:v[0] for k,v in tab.items()};results={}
    for name,windows in CASES.items():
        if names and name not in names:continue
        s=setup[name];pts=state[name+'_pts'];times=state[name+'_times'];packed=matte(s,pts);rows=[]
        for sec,fs in windows:
            pt=round(sec*12800);i=int(np.searchsorted(pts,pt))
            _,img=next(source_frames(s['source'],[pt]));gray=signals(img)
            mask=foreground(s,packed,i,pt)
            if mask is None:continue
            mask=cv2.dilate(mask,np.ones((7,7),np.uint8));mask[:270,:492]=255;mask[936:,1670:]=255
            jobs=[(i,k,times[i]) for f in fs for k in groups[f]]
            uv=ji.project_groups(n,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
            poly={k:z for (_,k,_),z in zip(jobs,uv)}
            for f in fs:
                paths=groups[f];parts=contour_samples({k:poly[k] for k in paths},tab,paths,f)
                if not parts:continue
                if f in ('B1','C0'):hits=read_stripe(img,mask,parts,f)
                elif f=='board':hits=read_rail(img,mask,parts)
                else:hits=read_contour(gray,mask,parts,f)
                if f=='board':
                    hits=[h for h in hits if h[0].endswith('bottom') and (h[3][0]<300 if name=='nayeon_s03' else h[3][0]>1410)]
                if not hits:continue
                normal=np.array([h[4] for h in hits]);d=np.array([h[2]-h[3] for h in hits])
                rows.append(dict(seconds=sec,pts=pt,family=f,rows=len(hits),
                    source_registration_xy=np.round(hits[0][-1],2).tolist() if f not in ('B1','C0','board') else None,
                    source_normal_median=float(np.median(abs((normal*d).sum(1)))),
                    source_normal_p90=float(np.quantile(abs((normal*d).sum(1)),.9))))
        results[name]=rows
    filename='examples.json' if names is None else 'examples_'+'_'.join(names)+'.json'
    write(DEST/'scores'/candidate.replace('/','_')/filename,results)
    return results


def baseline_packet(name):
    from lane_review import pixel_errors,native
    link=read((OUT/PARENT)/'source_bindings.json')[name]
    if prepared_packet_path(name).resolve()==Path(link['packet']['path']).resolve():return
    b=camera_state(PARENT);n=native(PARENT)
    s=read(PREPARED/'setup.json')['shots'][name];a=arrays(prepared_packet_path(name))
    with np.load(link['matched_errors']['path']) as original:
        count=len(original['error'])
        select=np.arange(len(a['pts']))>=count
        e,st=pixel_errors(n,b,s,name,a,select)
        e[:count]=original['error'];st[:count]=original['station']
        values={k:original[k] for k in original.files if k not in ('error','station')}
    save(baseline_folder()/(name+'.npz'),error=e,station=st,**values)


def baseline_folder():
    # This revision changes only the admission role of already measured rows.
    name='material_inputs' if PREPARED.name=='crease_sources' else PREPARED.name
    return DEST/'scores'/('baseline' if name=='prepared' else name+'_baseline')


def simple_eval(candidate,names):
    from lane_review import evaluate
    folder=DEST/'scores'/candidate.replace('/','_')/'all'
    for name in names:evaluate(candidate,name,folder)


def scored_packet(folder,name):
    local=folder/(name+'.npz')
    if local.is_file():return local
    refs=folder/'references.json'
    return Path(read(refs)[name]['score']) if refs.is_file() else local


def common_summary(candidate):
    from shared import programme_state_sha
    from lane_review import pixel_errors
    from motion_delivery import camera_audit
    base=camera_state(PARENT);state=camera_state(candidate);native=native_for(candidate,'jin')
    shots=read(PREPARED/'setup.json')['shots'];report={};pooled=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    physical={};path=DEST/'scores'/candidate.replace('/','_')/'all'
    old=read((OUT/PARENT)/'source_bindings.json')
    for name,s in shots.items():
        a=arrays(prepared_packet_path(name))
        reference=baseline_folder()/(name+'.npz')
        if not reference.exists():reference=Path(old[name]['matched_errors']['path'])
        z=[arrays(reference),arrays(scored_packet(path,name))]
        if len(a['pts'])!=len(z[0]['error']) or len(a['pts'])!=len(z[1]['error']):raise ValueError(name)
        with np.load(old[name]['packet']['path']) as original:
            original_eligible=original['fold']<5
        count=len(original_eligible)
        new=np.arange(len(a['pts']))>=count;eligible=a['fold']<5
        layer=np.where(np.char.startswith(a['family'],'Ad'),'print',np.where(np.char.startswith(a['family'],'board.'),'board','ice'))
        prior=np.r_[original_eligible,np.zeros(len(new)-count,bool)]
        outline=np.isin(a['material'],['outline_xmin','outline_xmax'])
        masks={'prior':prior,'new_fit':new&eligible&~outline,'new_check':new&(a['fold']==6)&~outline,
               'prior_currently_admitted':prior&eligible,'fixed_anchors':a['station_fixed']&eligible,
               'lane_caps':np.isin(a['path'],[f'demo_initial/lane{i}:1' for i in range(1,5)])&eligible}
        masks.update({k:(prior&(layer==k)) for k in ('ice','board','print')})
        for fam in ('F4','R4','goal_crease_left','B1','C0','Ad1','Ad4','Ad7','Ad8','Ad9','Ad10','Ad11','Ad12','Ad24','Ad25','Ad26'):
            masks[fam+'_prior']=prior&(a['family']==fam)
            masks[fam+'_check']=new&(a['fold']==6)&~outline&(a['family']==fam)
        physical[name]=dict(before=camera_audit(native,base,name,s),after=camera_audit(native,state,name,s))
        groups={}
        for key,use in masks.items():
            val=[zz['error'][use] for zz in z];groups[key]=[stats(vv) for vv in val]
            for dst,vv in zip(pooled[s['program']][key],val):dst.append(vv)
        for key in ('lag1_rebound_held','lag8_rebound_held'):
            val=[zz[key] for zz in z]
            good=np.isfinite(val[0])&np.isfinite(val[1]);val=[v[good] for v in val]
            groups[key]=[stats(vv) for vv in val]
            for dst,vv in zip(pooled[s['program']][key],val):dst.append(vv)
        report[name]=groups
    summary={prog:{k:[stats(np.concatenate(v)) for v in pair] for k,pair in bins.items()} for prog,bins in pooled.items()}
    out=OUT/candidate
    write(out/'source_comparison.json',dict(baseline=PARENT,candidate=candidate,programmes=summary,shots=report))
    write(out/'physical_mounts.json',physical)
    return summary


if __name__=='__main__':
    from argparse import ArgumentParser
    ap=ArgumentParser(description=__doc__);ap.add_argument('task',choices=['end','features','cases'])
    ap.add_argument('--candidate',default=PARENT);ap.add_argument('--programme');ap.add_argument('--shots',nargs='+')
    ap.add_argument('--scope',default='reads')
    a=ap.parse_args()
    if a.task=='end':low_sweep(a.programme)
    elif a.task=='cases':shot_cases(a.candidate)
    else:compare_features(a.candidate,a.shots,a.scope)
