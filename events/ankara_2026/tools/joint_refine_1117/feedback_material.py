"""Verify paint and historic source points against the exposed physical surface."""
from collections import defaultdict
import shutil

import cv2
import numpy as np

from feedback_fit import measure_tip, relative
from feedback_source import (CASES, DEST, contour_samples, foreground, matte,
                             read_contour, sample_image, selected_paths, signals)
from flow_depths import rays
from review import camera_state, ji, native_for, source_frames
from shared import PREPARED, arrays, prepared_packet_path, read, save, write


FAMILIES=('F4','goal_crease_left','R4')
GUIDE=relative(DEST/'common')


def floor_mask(n,state,name,s,i):
    # Pixel-centred coarse cells. Evaluate the actual rolling camera on each
    # row and invert the measured lens before intersecting the ice plane.
    step=16
    xs=np.arange(step/2-.5,1920,step)
    ys=np.arange(step/2-.5,1080+step,step)
    xx,yy=np.meshgrid(xs,ys)
    uv=np.c_[xx.ravel(),yy.ravel()]
    rig=state[name+'_rigs'][i]
    tm=state[name+'_times'][i]+rig[37]*(ys/1079.-.5)
    c=n.core.evaluate_motion_camera(rig,state[name+'_controls'],s['first'],s['dt'],tm)
    c={k:np.repeat(v,len(xs),axis=0) for k,v in c.items()}
    ray=rays(uv,c)
    with np.errstate(invalid='ignore',divide='ignore'):
        depth=-c['centre'][:,2]/ray[:,2]
        plane=c['centre'][:,:2]+depth[:,None]*ray[:,:2]
    x,y,rx,ry,p,_=n.scene['board']
    delta=np.maximum(0.,abs(plane)-[x-rx,y-ry])/[rx,ry]
    level=1.-np.sum(np.minimum(delta,3.)**p,axis=1)
    level[~np.isfinite(level)|(depth<=0)]=-3.
    smooth=cv2.resize(level.reshape(xx.shape).astype('f4'),None,fx=step,fy=step,
                      interpolation=cv2.INTER_LINEAR)[:1080]
    return (smooth<0).astype(np.uint8)


def verified_hits(raw,parts,mask,fam):
    p=np.concatenate([z[1] for z in parts])
    vertical=np.ptp(p[:,1])
    hits=read_contour(signals(raw),mask,parts,fam,(78,min(19,max(6,.55*vertical))))
    if not hits:return []
    xy=np.array([h[2] for h in hits]);norm=np.array([h[4] for h in hits])
    grid=xy[:,None]+norm[:,None]*np.array([-8.,-5.,5.,8.])[None,:,None]
    rgb=sample_image(raw,grid).astype(float)
    inside=rgb[:,:2].mean(1);outside=rgb[:,2:].mean(1)
    if fam=='goal_crease_left':
        # Translucent blue paint reduces red relative to blue on its inside.
        blue=(inside[:,0]-inside[:,2])-(outside[:,0]-outside[:,2])
        good=(blue>1.4)&(outside.mean(1)>inside.mean(1)+2.)&(inside.min(1)>103)
    else:
        # Both layers belong to the same dark annulus and have opposite polarity.
        sign=np.array([h[0] for h in hits])
        polarity={z[0]:z[4] for z in parts}
        luma=(outside-inside)@np.array([.114,.587,.299])
        good=(luma*np.array([polarity[k] for k in sign])>2.)&(rgb.min((1,2))>107)
        # Printed advertisements and the yellow kickplate are more saturated
        # than this grey house paint on the original exposures.
        good&=(np.ptp(inside,axis=1)<40)&(np.ptp(outside,axis=1)<40)
    output=[]
    for path in dict.fromkeys(h[0] for h in hits):
        rr=np.array([j for j,h in enumerate(hits) if h[0]==path and good[j]],dtype=int)
        if len(rr)<5:continue
        xy0=xy[rr]
        if np.ptp(xy0,axis=0).max()<26:continue
        output.extend(hits[j] for j in rr)
    return output


def scan(programme):
    cv2.setNumThreads(1)
    b=camera_state(GUIDE);native=native_for(GUIDE,programme)
    paths=selected_paths(native);keys=[k for fam in FAMILIES for k in paths[fam]]
    tab=ji.native_paths(native,keys,count=257);world={k:v[0] for k,v in tab.items()}
    for name,s in read(PREPARED/'setup.json')['shots'].items():
        if s['program']!=programme or s['camera_index']!=2:continue
        out=DEST/'material_sweep'/(name+'.npz')
        if out.exists():continue
        a=arrays(prepared_packet_path(name))
        relevant=np.flatnonzero(np.isin(a['family'],FAMILIES)&(a['fold']<9))
        old_rows=defaultdict(list)
        for r in relevant:old_rows[a['idx'][r]].append(r)
        pts=b[name+'_pts'];times=b[name+'_times'];packed=matte(s,pts)
        ledger=np.zeros((len(pts),len(FAMILIES),3),np.int32)
        traces=[];tips=[];bad=[];why=[];frames=iter(source_frames(s['source'],pts))
        for start in range(0,len(pts),32):
            inds=np.arange(start,min(len(pts),start+32))
            jobs=[(i,k,times[i]) for i in inds for k in keys]
            curves=ji.project_groups(native,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
            lookup={(i,k):z for (i,k,_),z in zip(jobs,curves)}
            for i in inds:
                pt,img=next(frames)
                if pt!=pts[i]:raise ValueError((name,pt))
                groups=[contour_samples({k:lookup[i,k] for k in paths[f]},tab,paths[f],f) for f in FAMILIES]
                ledger[i,:,0]=[sum(len(v[1]) for v in g) for g in groups]
                rr=np.asarray(old_rows[i],dtype=int)
                if not len(rr) and np.max(ledger[i,:,0])<8:continue
                surface=floor_mask(native,b,name,s,i)
                person=foreground(s,packed,i,pt)
                if len(rr):
                    rxy=np.rint(a['uv'][rr]).astype(int)
                    ok=(rxy>=0).all(1)&(rxy<[1920,1080]).all(1)
                    rxy=rxy[ok];ids=rr[ok];x,y=rxy.T
                    # Deep inside the board or foreground matte is positive
                    # contrary evidence, including for inherited observations.
                    outside=cv2.erode(surface,np.ones((11,11),np.uint8))[y,x]!=0
                    human=(cv2.erode(person,np.ones((5,5),np.uint8))[y,x]!=0
                           if person is not None else np.zeros(len(x),bool))
                    for criterion,reason in ((outside,'outside_measured_ice'),(human&~outside,'source_foreground')):
                        take=ids[criterion&~a['station_fixed'][ids]]
                        bad.extend(take);why.extend([reason]*len(take))
                if person is None:continue
                mask=cv2.dilate((surface|(person!=0)).astype(np.uint8),np.ones((19,19),np.uint8))
                mask[:270,:492]=1;mask[936:,1670:]=1
                for j,(fam,g) in enumerate(zip(FAMILIES,groups)):
                    if ledger[i,j,0]<8:continue
                    hits=verified_hits(img,g,mask,fam)
                    ledger[i,j,1]=len(hits)
                    for key in paths[fam]:
                        hh=[h for h in hits if h[0]==key]
                        if len(hh)<6:continue
                        nx=np.abs([h[4][0] for h in hh])
                        keep=np.unique(np.r_[np.linspace(0,len(hh)-1,min(12,len(hh))).astype(int),np.argsort(nx)[-5:]])
                        ledger[i,j,2]+=len(keep)
                        for r in keep:
                            h=hh[r];traces.append((i,keys.index(key),h[1],*h[2],*h[4],*h[6],h[5]))
                        poly,status=lookup[i,key]
                        if (status!=0).any():continue
                        for side in (-1,1):
                            v=measure_tip(img,mask,poly,tab[key][1],hh[0][6],side,paths[fam][key])
                            if not v:continue
                            st,actual,guide,strength,pixels=v
                            # The fitted extremum must touch this same measured
                            # boundary. Clipping across the image edge is not a tip.
                            dd=np.array([h[2] for h in hh])-actual
                            if np.min(np.hypot(dd[:,0]*.42,dd[:,1]))>13.:continue
                            if fam=='goal_crease_left':
                                z=actual[None,None,:]+np.array([[[-8.*side,0.],[8.*side,0.]]])
                                col=sample_image(img,z).reshape(2,3).astype(float)
                                if (col[0,0]-col[0,2])-(col[1,0]-col[1,2])<1.:continue
                            tips.append(dict(idx=i,pts=pt,path=key,station=st,uv=actual,guide=guide,
                                normal=[side,0.],contrast=strength,shift=actual-guide,family=fam,
                                material='outline_xmin' if side<0 else 'outline_xmax',pixels=pixels))
        z=np.asarray(traces,float).reshape(-1,10)
        data={k:np.asarray([h[k] for h in tips]) for k in tips[0] if k!='pixels'} if tips else {}
        pix=[h['pixels'] for h in tips]
        save(DEST/'material_tips'/(name+'.npz'),**data,pixel_offsets=np.r_[0,np.cumsum([len(a) for a in pix])],
             edge_pixels=np.concatenate(pix) if pix else np.empty((0,2)),source_sha256=s['source_sha256'])
        save(out,pts=pts,ledger=ledger,families=FAMILIES,keys=keys,index=z[:,0].astype(int),path=z[:,1].astype(int),
             station=z[:,2],uv=z[:,3:5],normal=z[:,5:7],shift=z[:,7:9],contrast=z[:,9],
             blocked_rows=np.asarray(bad,dtype=int),blocked_reasons=np.asarray(why,dtype='U32'),source_sha256=s['source_sha256'])
        print(name,len(pts),len(traces),len(tips),len(bad),flush=True)


def prepare():
    from feedback_update import roles
    folder=DEST/'material_inputs';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    report={};prior_directory=str(PREPARED)
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);s['observations']=str(old)
        if s['camera_index']!=2:continue
        a=arrays(old);audit=arrays(DEST/'material_sweep'/(name+'.npz'))
        assert s['source_sha256']==audit['source_sha256'].item()
        a.setdefault('admission_reason',np.full(len(a['idx']),'inherited_source_role'))
        original=a['fold'].copy()
        prior_scan=np.isin(a['family'],FAMILIES)&np.isin(a['material'],
            ['outline_xmin','outline_xmax','review_ice_contour','review_ice_counterview'])&~a['station_fixed']
        a['fold'][prior_scan]=9
        a['admission_reason']=np.where(prior_scan,'superseded_by_surface_verification',a['admission_reason'])
        blocked=audit['blocked_rows'];a['fold'][blocked]=9
        a['admission_reason']=a['admission_reason'].astype('U90')
        a['admission_reason'][blocked]=audit['blocked_reasons']
        changed=np.flatnonzero(original!=a['fold'])
        register=folder/'role_changes'/(name+'.npz')
        save(register,row=changed,previous=original[changed],current=a['fold'][changed],
             label=a['admission_reason'][changed],pts=a['pts'][changed],origin_row=a['origin_row'][changed],
             source_sha256=s['source_sha256'],previous_packet=str(old))
        labels,ct=np.unique(a['admission_reason'][changed],return_counts=True)
        entry=dict(historic_packet=str(old),role_changes=str(register),reclassified=dict(zip(labels,map(int,ct))))
        for kind in ('material_sweep','material_tips'):
            f=DEST/kind/(name+'.npz');q=arrays(f)
            if kind=='material_sweep':
                ix=q['index'];q['path']=q['keys'][q['path']]
                path_to_family={k:fam for fam,kk in zip(FAMILIES,([0,1],[2],[3,4])) for k in q['keys'][kk]}
                q.update(idx=ix,pts=q['pts'][ix],family=np.array([path_to_family[k] for k in q['path']]),
                         material=np.full(len(ix),'surface_verified_boundary'))
            if not len(q.get('idx',[])):continue
            role=roles(name,q,'dense_tips' if kind=='material_tips' else kind)
            # Consecutive frames are highly correlated. Fit a spread in time
            # and reserve neighbouring exposures with independently read UV.
            recurrent=(q['idx']%(3 if kind=='material_tips' else 9))==0
            near=np.zeros(len(role),bool)
            for sec,fams in CASES.get(name,[]):
                near|=np.isin(q['family'],fams)&(abs(q['pts']/12800.-sec)<.049)
            role[(role==0)&~recurrent]=6
            role[(role<9)&near]=6
            rr=np.flatnonzero(role<9)
            if not len(rr):continue
            code=max(map(int,s['source_packets']))+1
            vals=dict(pts=q['pts'][rr],idx=q['idx'][rr],uv=q['uv'][rr],station=q['station'][rr],
                path=q['path'][rr],family=q['family'][rr],material=q['material'][rr],fold=role[rr],source_fold=role[rr],
                station_fixed=False,source_packet=code,source_row=rr,origin_row=-(110000000 if kind=='material_tips' else 100000000)-rr,
                baseline_display=False,admission_reason='unoccluded_ice_material',restored_admission_reason='unoccluded_ice_material')
            for key,col in a.items():
                v=np.broadcast_to(vals[key],(len(rr),)+col.shape[1:])
                a[key]=np.concatenate((col,v.astype(col.dtype if col.dtype.kind!='U' else str)))
            s['source_packets'][str(code)]=dict(path=str(f),source_row='original exposure measurement row',source_sha256=s['source_sha256'])
            entry[kind]=dict(fit=int(np.sum(role==0)),check=int(np.sum(role==6)),source=str(f))
        new=folder/(name+'.npz');save(new,**a);s['observations']=str(new);report[name]=entry
        print(name,entry['reclassified'],flush=True)
    setup['parent']=prior_directory;write(folder/'setup.json',setup);write(folder/'source_audit.json',report)
    weights=read(DEST/'balanced_weights.json')
    weights.update(surface_verified_boundary=4.,outline_xmin=24.,outline_xmax=24.)
    write(DEST/'material_weights.json',weights)


def refit(which):
    from source_bundle import native_run
    fam,coords={
        'f4':('F4',['F_column_position/3','house_ring_radius/14','house_ring_radius/15']),
        'r4':('R4',['R_column_position/3','FR_row_alignment/3','house_ring_radius/30','house_ring_radius/31']),
        'crease':('goal_crease_left',['left_crease_carrier_offset/0','crease_semiaxes/0','crease_semiaxes/1'])
    }[which]
    native_run(DEST/'material_fit'/which,GUIDE,['--mode','scene','--families',fam,
        '--scene-blocks','scene/ankara_2026.field/ice_tolerance','--ice-coordinates',*coords,
        '--unit-weights','--material-weights',DEST/'material_weights.json',
        '--station-seed-dir',DEST/'scores/review_fixes_1118_common/all'])


def preserve_lane_world(source,reference):
    """Re-express a lane's two measured straight sides at the updated house x anchors."""
    from shared import NativeSceneGPU, ParameterState, numeric_sha
    state=camera_state(relative(source));n=native_for(relative(source),'jin');old=native_for(relative(reference),'jin')
    block=next(v for k,v in n.parameter_state.blocks.items() if k.endswith('/ice_tolerance'))
    off=block['offset']-39;end=block['offset']+block['size']
    q=state['scene_values'][off:end]
    o=old.parameter_state.values[off:end]
    np.testing.assert_array_equal(q,np.asarray(n.scene['shared_layout']))
    for row in range(4):
        def anchors(a):
            x=np.array([a[0]+a[3]*a[39+4+row],a[1]+a[3]*a[39+8+row]])
            # The common base row_y is constant in these targeted refinements.
            y=np.array([0.,a[3]*a[39+12+row]])
            return x,y
        x,y=anchors(q);xo,yo=anchors(o)
        old_axis=yo+o[9]*o[[147+row,151+row]]
        old_width=o[9]*(1+o[[39+48+row,39+52+row]])
        u=(x-xo[0])/(xo[1]-xo[0])
        q[[147+row,151+row]]=(old_axis[0]+u*np.diff(old_axis)[0]-y)/q[9]
        q[[39+48+row,39+52+row]]=(old_width[0]+u*np.diff(old_width)[0])/q[9]-1
    new=NativeSceneGPU(n.scene_path,n.core,parameter_state=ParameterState(n.parameter_state.layout,state['scene_values']))
    paths=[k for k,v in old.paths.items() if v['label'].startswith('ice.lane')]
    for k in paths:
        np.testing.assert_allclose(new.paths[k]['parameters'],old.paths[k]['parameters'],rtol=0,atol=5e-14)
    state['scene_values_sha256']=numeric_sha(state['scene_values']);save(source/'state.npz',**state)
    note=read(source/'inputs.json')
    note.update(lane_world_preserved=relative(reference),checked_lane_paths=len(paths))
    write(source/'inputs.json',note)


def camera(name,parent,stage='material_camera',gain=5.,stride=7):
    from source_bundle import native_run
    opts=['--mode','camera','--shots',name,'--stride',stride,'--unit-weights',
          '--station-seed-dir',DEST/'scores/review_fixes_1118_common/all','--material-weights',DEST/'material_weights.json',
          '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
          '--texture-dir',DEST/'texture','--rebind-texture','--triplet-scale',5.,'--flow-gain',gain]
    if name.startswith('nayeon'):opts+=['--photo-bindings',DEST.parent/'repair/prepared_v2/nayeon_photo_bindings.npz']
    native_run(DEST/stage/name,relative(parent),opts)


def texture_axes(name):
    from motion_tracks import packet
    from shared import sha
    from feedback_source import PARENT
    s=read(PREPARED/'setup.json')['shots'][name];f=packet(name,s);q=arrays(f)
    pts=camera_state(PARENT)[name+'_pts'];m=matte(s,pts)
    ice=np.flatnonzero((q['family']==1)&(q['quality']<16.))
    requests=defaultdict(list)
    for row in ice:
        i=q['index'][row];requests[i].append((row,1,q['uv'][row]))
        requests[i-q['lag'][row]].append((row,0,q['left_uv'][row]))
    grids=np.zeros((len(q['index']),2,2,2));indices=sorted(requests)
    for i,(pt,img) in zip(indices,source_frames(s['source'],pts[indices]),strict=True):
        person=foreground(s,m,i,pt)
        if person is None:continue
        for row,side,xy in requests[i]:
            x,y=np.rint(xy).astype(int)
            if not (18<x<1901 and 18<y<1061) or person[y-8:y+9,x-8:x+9].any():continue
            tile=signals(img[y-17:y+18,x-17:x+18])
            smooth=cv2.GaussianBlur(tile,(0,0),1.4)
            gradients=np.stack([cv2.Sobel(smooth,-1,*axes,scale=.125)[4:-4,4:-4].ravel()
                                for axes in ((1,0),(0,1))])
            grids[row,side]=gradients@gradients.T/gradients.shape[1]
    metric=np.tile(np.eye(2),(len(q['index']),1,1));ratio=np.ones(len(metric));evidence=np.zeros(len(metric),bool)
    normal=np.zeros((len(metric),2))
    for row in ice:
        e,v=np.linalg.eigh(grids[row])
        if min(e[:,1])<.024 or abs(v[0,:,-1]@v[1,:,-1])<.85:continue
        ratio[row]=np.clip(np.mean(e[:,0]/e[:,1]),0.,1.)
        n=v[1,:,-1];normal[row]=n;proj=np.outer(n,n)
        metric[row]=3.6*(proj+np.sqrt(ratio[row])*(np.eye(2)-proj));evidence[row]=True
    save(DEST/'texture_axes'/(name+'.npz'),metric=metric,minor_over_major=ratio,normal=normal,evidence=evidence,
         source_sha256=s['source_sha256'],packet_sha256=sha(f),pts=q['pts'])


def opening_print():
    """Measure consecutive and eight-frame motion on the printed board in the opening pan."""
    from flow_depths import bind
    import cv2
    from collections import deque
    name='jin_s01';c=relative(DEST/'delivery');b=camera_state(c);n=native_for(c,'jin')
    s=read(PREPARED/'setup.json')['shots'][name];pts=b[name+'_pts'];packed=matte(s,pts)
    a=arrays(prepared_packet_path(name));ids=np.flatnonzero(np.char.startswith(a['family'],'Ad')&(a['fold']<5))
    old=read(DEST/'texture/sources.json')[name]
    from pathlib import Path
    original=arrays(Path(old)/(name+'.npz'));quality=arrays(Path(old)/'quality'/(name+'.npz'))
    raw=[];history=deque(maxlen=9)
    select=np.flatnonzero((pts/12800.>=32.5)&(pts/12800.<=33.2))
    for i,(pt,img) in zip(select,source_frames(s['source'],pts[select]),strict=True):
        im=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);person=foreground(s,packed,i,pt)
        mask=np.zeros_like(im)
        rr=ids[a['idx'][ids]==i]
        for k in np.unique(a['family'][rr]):
            xy=a['uv'][rr[a['family'][rr]==k]]
            if len(xy)>=8:cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(xy).astype('i4')),255)
        if person is None:mask[:]=0
        else:mask[person!=0]=0
        points=cv2.goodFeaturesToTrack(im,160,.008,4,mask=mask)
        history.append((i,im,mask,points))
        for lag in (1,8):
            if len(history)<=lag:continue
            left,previous,_,corners=history[-lag-1]
            if corners is None:continue
            uv,ok,_=cv2.calcOpticalFlowPyrLK(previous,im,corners,None,winSize=(21,21),maxLevel=4)
            back,ok2,_=cv2.calcOpticalFlowPyrLK(im,previous,uv,None,winSize=(21,21),maxLevel=4)
            end=uv[:,0];xy=corners[:,0];u,v=np.rint(end).astype(int).T
            good=(ok[:,0]!=0)&(ok2[:,0]!=0)&(np.linalg.norm(back[:,0]-xy,axis=1)<.48)
            good&=(u>4)&(u<1915)&(v>4)&(v<1075)&(mask[v.clip(0,1079),u.clip(0,1919)]!=0)
            rr=np.flatnonzero(good)
            if len(rr)<6:continue
            matrix,inside=cv2.estimateAffinePartial2D(xy[rr],end[rr],method=cv2.RANSAC,ransacReprojThreshold=1.15)
            if matrix is None:continue
            for j in rr[inside.ravel()!=0]:raw.append((i,lag,*xy[j],*end[j]))
    m=np.asarray(raw);index=m[:,0].astype(int);lag=m[:,1].astype(int);left=m[:,2:4];uv=m[:,4:6]
    world,_=bind(n,s,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],index-lag,left,np.zeros(len(m),int))
    valid=np.isfinite(world).all(1);width=((left[:,0]//12).astype(int)+3*(left[:,1]//12).astype(int))%5
    target=DEST/'opening_texture'
    ext=dict(index=index,lag=lag,left_uv=left,uv=uv,valid=valid,world=world,held=width==0)
    out={k:np.concatenate([original[k],ext[k]]) for k in ext}
    save(target/(name+'.npz'),**out,pts=pts,source_sha256=s['source_sha256'])
    save(target/'quality'/(name+'.npz'),keep=np.r_[quality['keep'],valid],pts=pts)
    save(target/'detail'/(name+'.npz'),transient=np.flatnonzero((pts/12800.>=32.74)&(pts/12800.<=32.96)))
    write(target/'source.json',dict(parent=old,shot=name,source=s['source'],source_sha256=s['source_sha256'],
           original_rows=len(original['index']),new_valid=int(valid.sum()),held_valid=int(np.sum(valid&(width==0)))))
