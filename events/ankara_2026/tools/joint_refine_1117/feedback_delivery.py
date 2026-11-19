"""Source images, repeated end views, and exact-clock checks for the reviewed field."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from feedback_check import baseline_folder, compare_features, scored_packet
from feedback_fit import PRIOR, extreme
from feedback_source import CASES, DEST, PARENT, contour_samples, selected_paths
from motion_delivery import camera_audit, ledger_check, media_check
from repair_motion import frame_median, stat
from review import camera_state, ji, native_for, sample, source_frames
from shared import (EVENT, LIBRARY, OUT, PREPARED, PROGRAMMES, arrays, numeric_sha, prepared_packet_path,
                    programme_state_sha, read, safe_path, save, sha, write)
from solve import stats
from source_bundle import relative


def print_registration(candidate):
    n=native_for(candidate,'jin');state=camera_state(candidate);setup=read(PREPARED/'setup.json')['shots'];result={}
    for name in ('jin_s04','nayeon_s03','nayeon_s04'):
        q=arrays(DEST/'print_reference'/(name+'.npz'));s=setup[name]
        table=ji.native_paths(n,np.unique(q['path']),count=513)
        group=defaultdict(list)
        for r,(i,k) in enumerate(zip(q['idx'],q['path'])):group[i,k].append(r)
        jobs=[(i,k,state[name+'_times'][i]) for i,k in group]
        poly=ji.project_groups(n,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,{k:v[0] for k,v in table.items()})
        records=[]
        for ((i,k),ids),(uv,status) in zip(group.items(),poly):
            if np.any(status!=0):continue
            rr=np.array(ids);st=q['station'][rr];pred=np.stack([np.interp(st,table[k][1],val) for val in uv.T],axis=1)
            delta=q['uv'][rr]-pred
            records.append(dict(pts=int(q['pts'][rr[0]]),family=str(q['family'][rr[0]]),
                source_minus_projection_xy=np.round(np.median(delta,axis=0),2).tolist(),
                source_registration_correlation=float(q['correlation'][rr[0]]),blur=float(q['blur_width'][rr[0]])))
        result[name]=records
    write(DEST/'scores'/candidate.replace('/','_')/'same_print.json',result)
    return result


def left_print_report(candidate):
    """Transfer the source registration of the same printed sign between camera views."""
    source=arrays(DEST/'print_reference/left_end_transfer.npz')
    name='nayeon_s03';s=read(PREPARED/'setup.json')['shots'][name]
    states=[camera_state(c) for c in (PARENT,candidate)]
    natives=[native_for(c,'nayeon') for c in (PARENT,candidate)]
    groups=selected_paths(natives[0]);data=[]
    keys=[k for f in source['families'] for k in groups[f]]
    tables=[ji.native_paths(n,keys,count=193) for n in natives]
    for pt,ad,dx,dy,corr,gap,blur in source['registrations']:
        if corr<.76 or gap<.035:continue
        i=int(np.searchsorted(states[0][name+'_pts'],int(pt)));fam='Ad'+str(int(ad))
        projection=[]
        for n,b,tab in zip(natives,states,tables):
            parts=[n.core.project_motion(b[name+'_rigs'][i],b[name+'_controls'],s['first'],s['dt'],1080,
                   b[name+'_times'][i],tab[k][0]) for k in groups[fam]]
            if not all(np.all(r['status']==0) for r in parts):break
            projection.append(np.concatenate([r['uv'] for r in parts]))
        if len(projection)!=2:continue
        d=projection[1]-projection[0]
        data.append(dict(pts=int(pt),family=fam,before=[dx,dy],after=([dx,dy]-np.median(d,axis=0)).tolist(),
            within_print_differential_px=float(np.max(np.linalg.norm(d-np.median(d,axis=0),axis=1))),
            correlation=corr,peak_gap=gap,blur_width=blur))
    write(DEST/'scores'/candidate.replace('/','_')/'left_print_transfer.json',data)
    return data


def extent_scores(candidate,source='dense_tips'):
    n=native_for(candidate,'jin');b=camera_state(candidate);setup=read(PREPARED/'setup.json')['shots']
    out=DEST/'scores'/candidate.replace('/','_')/('extents' if source=='dense_tips' else source)
    for f in (DEST/source).glob('*.npz'):
        name=f.stem;s=setup[name];q=arrays(f)
        if 'pts' not in q:continue
        paths=np.unique(q['path']);tab=ji.native_paths(n,paths,count=513);world={k:v[0] for k,v in tab.items()}
        groups=defaultdict(list);e=np.full(len(q['pts']),np.nan)
        for r,(i,k) in enumerate(zip(q['idx'],q['path'])):groups[i,k].append(r)
        jobs=[(i,k,b[name+'_times'][i]) for i,k in groups]
        proj=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for ((i,k),rr),(uv,status) in zip(groups.items(),proj):
            if np.any(status!=0):continue
            for r in rr:
                v=extreme(uv,tab[k][1],int(q['normal'][r,0]))
                if v:e[r]=q['uv'][r,0]-v[1][0]
        save(out/(name+'.npz'),error=e,pts=q['pts'],family=q['family'])


def end_report(candidate,scope='end_sweep',tip_scope='extents'):
    root=DEST/'scores';labels=[PARENT.replace('/','_'),candidate.replace('/','_')]
    report={};pool=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    n=native_for(candidate,'jin')
    for path in sorted((DEST/scope).glob('*.npz')):
        name=path.stem;a=arrays(path);program=name.split('_')[0]
        z=[np.load(root/label/scope/path.name) if (root/label/scope/path.name).exists() else None for label in labels]
        tips=[np.load(root/label/tip_scope/path.name) if (root/label/tip_scope/path.name).exists() else None for label in labels]
        key_family=np.array([n.paths[k]['label'][4:].split(':')[0] for k in a['keys']]);fams=key_family[a['path']]
        groups={}
        for j,fam in enumerate(a['families']):
            item=dict(exposures=len(a['pts']),visible_guide=int((a['ledger'][:,j,0]>0).sum()),
                      source_measured=int((a['ledger'][:,j,2]>0).sum()))
            if z[0] is not None and z[1] is not None:
                mask=fams==fam
                values=[v['error'][mask] for v in z]
                item['normal_distance']=[stats(v) for v in values]
                for target,val in zip(pool[program][fam+'_edge'],values):target.append(val)
            if tips[0] is not None and tips[1] is not None:
                values=[abs(v['error'][v['family']==fam]) for v in tips]
                item['horizontal_extent']=[stats(v) for v in values]
                for target,val in zip(pool[program][fam+'_extent'],values):target.append(val)
            groups[fam]=item
        for v in [*z,*tips]:
            if v is not None:v.close()
        report[name]=groups
    agg={p:{k:[stats(np.concatenate(a)) for a in pair] for k,pair in b.items()} for p,b in pool.items()}
    write(OUT/candidate/'end_views.json',dict(shots=report,programmes=agg,source_surface_packet=scope,extent_source=tip_scope,
          exposures=sum(next(iter(s.values()))['exposures'] for s in report.values()),
          definition='Actual outer and middle paint or blue crease arc, plus measured horizontal extrema where visible'))


# Crops specify unannotated source pixels. Some retain two separate features in
# the same image, so rotation and magnification are visible without a rescale.
CROPS={
 ('okada_s03',57.38):[(1130,790,1720,915),(10,780,470,892)],
 ('okada_s05',100.26):[(0,552,720,670),(0,845,570,1080)],
 ('okada_s08',139.70):[(95,526,938,622),(245,820,865,1065)],
 ('jin_s03',39.12):[(1300,276,1920,530)],
 ('jin_s04',42.18):[(1230,344,1920,527),(260,351,995,513)],
 ('jin_s04',47.04):[(0,554,885,679),(180,818,730,1080)],
 ('jin_s08',70.58):[(370,541,823,771)],
 ('jin_s10',83.90):[(112,897,860,984),(1150,903,1665,1019)],
 ('nayeon_s03',42.34):[(1660,456,1920,744)],
 ('nayeon_s03',50.8):[(0,397,350,879)],
 ('nayeon_s04',62.56):[(1390,281,1920,475),(696,281,1320,427)],
 ('nayeon_s04',63.56):[(1140,388,1810,602)],
 ('nayeon_s04',78.82):[(914,397,1320,530)]}


def cards(candidate,extra=()):
    candidates=[PARENT,candidate];states=[camera_state(c) for c in candidates]
    natives=[native_for(c,'jin') for c in candidates]
    setup=read(PREPARED/'setup.json')['shots'];paths=selected_paths(natives[0])
    table=[ji.native_paths(n,[k for fs in paths.values() for k in fs],count=513) for n in natives]
    folder=OUT/candidate/'inspection';folder.mkdir(exist_ok=True)
    tasks=[(name,sec,fs,CROPS[name,sec]) for name,ws in CASES.items() for sec,fs in ws]
    for name,sec,fs,crops in [*tasks,*extra]:
        s=setup[name];pt=round(sec*12800);i=int(np.searchsorted(states[0][name+'_pts'],pt));_,raw=next(source_frames(s['source'],[pt]))
        panels=[raw.copy() for _ in range(3)]
        for panel,b,n,tab in zip(panels[1:],states,natives,table):
            for f in fs:
                for k in paths[f]:
                    pred=n.core.project_motion(b[name+'_rigs'][i],b[name+'_controls'],s['first'],s['dt'],1080,b[name+'_times'][i],tab[k][0])
                    uv=pred['uv'];good=(pred['status']==0)&np.isfinite(uv).all(1)&(abs(uv)<4000).all(1)
                    runs=np.split(np.flatnonzero(good),np.flatnonzero(np.diff(np.flatnonzero(good))!=1)+1)
                    color=(211,151,13) if f=='board' else (42,193,48) if f.startswith('Ad') else (0,195,236)
                    for run in runs:
                        if len(run)>1:cv2.polylines(panel,[np.int32(np.rint(uv[run]*16))],False,color,1,cv2.LINE_AA,4)
        for ix,(x0,y0,x1,y1) in enumerate(crops):
            zoom=1 if x1-x0>610 else min(1.6,740/(x1-x0))
            rows=[]
            for text,panel in zip(('Source','Previous','Current'),panels):
                q=cv2.resize(panel[y0:y1,x0:x1],None,fx=zoom,fy=zoom,interpolation=cv2.INTER_LINEAR)
                q=cv2.copyMakeBorder(q,21,0,0,0,cv2.BORDER_CONSTANT,value=(27,27,27))
                cv2.putText(q,f'{text}   {name}   {sec:.3f} s', (6,15),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1,cv2.LINE_AA)
                rows.append(q)
            cv2.imwrite(str(folder/f'{name}_{pt}_{ix}.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])


def motion_report(candidate):
    base=camera_state(PARENT);state=camera_state(candidate);native=native_for(candidate,'jin')
    folder=OUT/candidate;old=read(PRIOR/'motion_all_surfaces.json')['shot_files']
    setup=read(PREPARED/'setup.json')['shots'];summary=read(folder/'repair_motion/summary.json')
    scores=DEST/'scores'/candidate.replace('/','_')/'all'
    physical={};files={};records={};pools=defaultdict(lambda:defaultdict(lambda:[[],[]]))
    peaks=defaultdict(dict)
    for name,s in setup.items():
        p=s['program'];physical[name]=dict(before=camera_audit(native,base,name,s),after=camera_audit(native,state,name,s))
        path=Path(summary[name].get('identical_motion_file',folder/'repair_motion'/(name+'.npz')))
        files[name]=str(path);z=[arrays(old[name]),arrays(path)]
        _,a,b=np.intersect1d(z[0]['source_row'],z[1]['source_row'],assume_unique=True,return_indices=True)
        for key in ('index','lag','family'):np.testing.assert_array_equal(z[0][key][a],z[1][key][b])
        pts=z[0]['pts'];np.testing.assert_array_equal(pts,z[1]['pts'])
        index=z[0]['index'][a];bins={}
        for label,code,minimum in [('board',0,4),('ice',1,3)]:
            for term,lag,vector in [('adjacent',1,'displacement'),('lag8',8,'displacement'),('jitter',1,'curvature')]:
                key=label+'_'+term
                take=(z[0]['family'][a]==code)&(z[0]['lag'][a]==lag)
                values=[frame_median(index,np.linalg.norm(tab[vector][ids],axis=1),take,len(pts),minimum) for tab,ids in zip(z,[a,b])]
                valid=np.isfinite(values[0])&np.isfinite(values[1]);ix=np.flatnonzero(valid)
                x,y=[v[valid] for v in values];d=y-x
                worst=np.argsort(d)[-3:][::-1]
                bins[key]=dict(before=stat(x),after=stat(y),largest_increases=[
                    dict(pts=int(pts[ix[j]]),previous_px=float(x[j]),current_px=float(y[j])) for j in worst])
                for target,val in zip(pools[p][key],[x,y]):target.append(val)
        records[name]=bins
        for k,peak in physical[name]['after']['trajectory'].items():
            if k in peaks[p] and peaks[p][k]['absolute']>=peak['absolute']:continue
            pt=peak['source_pts'];i=np.searchsorted(pts,pt);ids=(z[1]['index']==i)&(z[1]['lag']==1)
            with np.load(summary[name]['source_path']) as src:
                rows=z[1]['source_row'][ids];step=src['uv'][rows]-src['left_uv'][rows]
            err=z[1]['displacement'][ids];held=None
            if scored_packet(scores,name).is_file():
                with np.load(scored_packet(scores,name)) as f:
                    v=f['lag1_rebound_held'][i];held=float(v) if np.isfinite(v) else None
            peaks[p][k]=dict(**peak,shot=name,source_rows=len(step),
                measured_step_px=float(np.median(np.linalg.norm(step,axis=1))) if len(step) else None,
                fitted_displacement_error_px=float(np.median(np.linalg.norm(err,axis=1))) if len(step) else None,
                held_board_error_px=held)
    agg={p:{k:dict(before=stat(np.concatenate(x)),after=stat(np.concatenate(y))) for k,(x,y) in v.items()} for p,v in pools.items()}
    write(folder/'motion_all_surfaces.json',dict(programmes=agg,shot_files=files,shots=records,parent=PARENT))
    write(folder/'physical_mounts.json',physical);write(folder/'trajectory_extrema.json',peaks)


def lens_check(candidate):
    from flow_depths import rays
    n=native_for(candidate,'jin');states=[camera_state(c) for c in (PARENT,candidate)]
    report={};samples=np.array([[x,y] for y in (0,539.5,1079) for x in (0,959.5,1919)])
    for name,s in read(PREPARED/'setup.json')['shots'].items():
        results=[]
        for b in states:
            t=b[name+'_times'];rig,inv=np.unique(b[name+'_rigs'],axis=0,return_inverse=True)
            lower=np.inf;shift=0.;corner_error=0.;invalid=0
            for rid,r in enumerate(rig):
                ii=np.flatnonzero(inv==rid);uv=np.tile(samples,(len(ii),1))
                time=np.repeat(t[ii],len(samples))+r[37]*(uv[:,1]/1079.-.5)
                c=n.core.evaluate_motion_camera(r,b[name+'_controls'],s['first'],s['dt'],time)
                ray=np.einsum('nij,nj->ni',c['rotation'],rays(uv,c))
                x,y=(ray[:,:2]/ray[:,2:]).T;r2=x*x+y*y;d=c['k1']+2*c['k2']*r2;L=1+r2*(c['k1']+r2*c['k2'])
                p,q=c['p1'],c['p2']
                a=L+2*x*x*d+2*p*y+6*q*x;off=2*x*y*d+2*p*x+2*q*y;cc=L+2*y*y*d+6*p*y+2*q*x
                eig=(a+cc-np.hypot(a-cc,2*off))/2
                reproj=np.c_[c['cx']+c['fx']*(x*L+2*p*x*y+q*(r2+2*x*x)),c['cy']+c['fy']*(y*L+p*(r2+2*y*y)+2*q*x*y)]
                undistorted=np.c_[c['cx']+c['fx']*x,c['cy']+c['fy']*y]
                valid=(c['status']==0)&np.isfinite(eig)&(eig>0)&np.isfinite(reproj).all(1)
                invalid+=int((~valid).sum())
                if valid.any():
                    lower=min(lower,float(eig[valid].min()));shift=max(shift,float(np.linalg.norm(undistorted[valid]-uv[valid],axis=1).max()))
                    corner_error=max(corner_error,float(np.linalg.norm(reproj[valid]-uv[valid],axis=1).max()))
            results.append(dict(checked=len(t)*len(samples),invalid=invalid,
                minimum_sampled_jacobian=float(lower) if np.isfinite(lower) else None,
                maximum_image_distortion_px=shift,maximum_sensor_roundtrip_px=corner_error))
        report[name]=dict(before=results[0],after=results[1])
    write(OUT/candidate/'sensor_domain.json',report)
    return report


def reuse_render(parent,candidate,programme):
    """Bind an already encoded programme with identical exposure and pixel inputs."""
    a=camera_state(parent);b=camera_state(candidate)
    s=read(PREPARED/'setup.json')['shots']
    names=[name for name,v in s.items() if v['program']==programme]
    src=OUT/parent/'review'/programme;dst=OUT/candidate/'review'/programme
    record=read(src/'binding.json');coverage=read(src/'coverage.json')
    original=Path(coverage['video'])
    if record['mode']!='complete' or not coverage['validation']['native_PTS_equal']:raise ValueError(parent)
    same=programme_state_sha(a,names)
    assert record['state_sha256']==sha(OUT/parent/'state.npz')
    assert same==programme_state_sha(b,names)==record['programme_state_sha256']
    scene=native_for(candidate,programme)
    assert sha(scene.scene_path)==record['scene_sha256'] and record['source_sha256']==s[names[0]]['source_sha256']
    for name in names:assert sha(prepared_packet_path(name))==record['observation_packets'][name]
    dst.mkdir(parents=True,exist_ok=True)
    ledger=Path(record.get('source_ledger_directory',src))
    if not original.is_file() or not (ledger/'source_frames.jsonl').is_file():raise FileNotFoundError(src)
    coverage['video']=str(original)
    record.update(candidate=candidate,state=str(OUT/candidate/'state.npz'),state_sha256=sha(OUT/candidate/'state.npz'),
        source_ledger_directory=str(ledger),
        identical_programme_render=dict(original_binding=str(src/'binding.json'),original_candidate=parent,programme_sha256=same))
    write(dst/'binding.json',record)
    write(dst/'coverage.json',coverage)


def rapid_checks(candidate):
    from flow_depths import bind
    from repair_motion import project
    from feedback_source import matte,foreground
    b0=camera_state(PARENT);b1=camera_state(candidate)
    refs=read(DEST/'texture/sources.json');setup=read(PREPARED/'setup.json')['shots']
    events=[('jin_s07',57.86),('jin_s10',78.44),('okada_s02',45.50),
            ('okada_s07',111.44),('chen_s14',155.06),('nayeon_s04',55.02)]
    output={}
    for name,sec in events:
        s=setup[name];folder=Path(refs[name]);src=arrays(folder/(name+'.npz'))
        qual=arrays(folder/'quality'/(name+'.npz'));pt=round(sec*12800)
        i=np.searchsorted(src['pts'],pt);assert src['pts'][i]==pt
        rr=np.flatnonzero((src['index']==i)&src['held']&src['valid']&qual['keep'])
        if not len(rr):continue
        index=src['index'][rr];lag=src['lag'][rr];left=index-lag;errors=[]
        for b,c in [(b0,PARENT),(b1,candidate)]:
            n=native_for(c,s['program'])
            world,_=bind(n,s,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],left,src['left_uv'][rr],np.zeros(len(rr),int))
            valid=np.isfinite(world).all(1);e=np.full(len(rr),np.nan)
            delta=project(n,s,b,name,index[valid],world[valid])-project(n,s,b,name,left[valid],world[valid])
            e[valid]=np.linalg.norm(delta-(src['uv'][rr]-src['left_uv'][rr])[valid],axis=1);errors.append(e)
        output[name]=dict(source_pts=pt,source_packet=str(folder/(name+'.npz')),lags={})
        for step in (1,8):
            m=(lag==step)&np.isfinite(errors[0])&np.isfinite(errors[1])
            if m.any():output[name]['lags'][str(step)]=dict(n=int(m.sum()),
                source_step_px=float(np.median(np.linalg.norm((src['uv'][rr]-src['left_uv'][rr])[m],axis=1))),
                errors_px=[float(np.median(e[m])) for e in errors])
    name='jin_s01';s=setup[name];pts=b1[name+'_pts'];packed=matte(s,pts)
    a=arrays(prepared_packet_path(name));times=np.arange(420352-3*256,420352+4*256,256)
    frames=list(source_frames(s['source'],times));tracks=[];evidence=[]
    for (pt,raw),(target,img) in zip(frames,frames[1:]):
        i=np.searchsorted(pts,pt);j=i+1;assert pts[j]==target
        m=(a['pts']==pt)&(a['family']=='Ad25')&(a['fold']<5)
        if m.sum()<4:continue
        mask=np.zeros((1080,1920),np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(np.int32(np.rint(a['uv'][m]))),255)
        human=foreground(s,packed,i,pt);endhuman=foreground(s,packed,j,target)
        if human is None or endhuman is None:continue
        mask[human!=0]=0
        first,second=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in (raw,img)]
        corners=cv2.goodFeaturesToTrack(first,120,.008,4,mask=mask)
        if corners is None or len(corners)<5:continue
        uv,st,_=cv2.calcOpticalFlowPyrLK(first,second,corners,None,winSize=(21,21),maxLevel=4)
        back,st2,_=cv2.calcOpticalFlowPyrLK(second,first,uv,None,winSize=(21,21),maxLevel=4)
        uv=uv[:,0];xy=corners[:,0]
        xr,yr=np.rint(uv).astype(int).T
        good=(st.ravel()!=0)&(st2.ravel()!=0)&(np.linalg.norm(back[:,0]-xy,axis=1)<.6)
        good&=(xr>4)&(xr<1915)&(yr>4)&(yr<1075)&(endhuman[yr.clip(0,1079),xr.clip(0,1919)]==0)
        x,y=xy[good],uv[good]
        if len(x)<5:continue
        transform,inliers=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=1.2)
        if transform is None:continue
        x=x[inliers.ravel()!=0];y=y[inliers.ravel()!=0]
        measurement=y-x;errs=[];projected_scales=[]
        for b,c in [(b0,PARENT),(b1,candidate)]:
            n=native_for(c,'jin');left=np.full(len(x),i);end=left+1
            world,_=bind(n,s,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],left,x,np.zeros(len(x),int))
            sel=np.isfinite(world).all(1)
            if sel.sum()<4:break
            predicted=project(n,s,b,name,end[sel],world[sel])-project(n,s,b,name,left[sel],world[sel])
            errs.append(float(np.median(np.linalg.norm(predicted-measurement[sel],axis=1))))
            fitted,_=cv2.estimateAffinePartial2D(x[sel],x[sel]+predicted,method=cv2.LMEDS)
            projected_scales.append(float(np.linalg.norm(fitted[:,0])))
        if len(errs)==2:
            tracks.append(dict(source_pts=int(target),n=len(x),source_step_px=float(np.median(np.linalg.norm(measurement,axis=1))),
                source_apparent_scale=float(np.linalg.norm(transform[:,0])),projected_scales=projected_scales,errors_px=errs))
            evidence.extend([[pt,target,*v,*u] for v,u in zip(x,y)])
    save(OUT/candidate/'jin_s01_zoom_source.npz',source_pairs=np.asarray(evidence).reshape(-1,6),source_sha256=s['source_sha256'])
    write(OUT/candidate/'fast_pan_checks.json',dict(previous=PARENT,held_source_texture=output,jin_s01_printed_corners=tracks))


def reuse_unchanged_checks(parent,candidate,changed):
    src=OUT/parent;dst=OUT/candidate;old=camera_state(parent);new=camera_state(candidate)
    source=DEST/'scores'/parent.replace('/','_')/'all'
    target=DEST/'scores'/candidate.replace('/','_')/'all';refs={}
    oldflow=read(src/'repair_motion/summary.json');flow=read(dst/'repair_motion/summary.json')
    assert read(src/'repair_motion/binding.json')['state_sha256']==sha(src/'state.npz')
    for name in read(PREPARED/'setup.json')['shots']:
        if name in changed:continue
        digest=programme_state_sha(old,[name]);assert digest==programme_state_sha(new,[name])
        report=read(source/(name+'.json'))
        assert report['shot_state']==digest and report['packet_sha']==sha(prepared_packet_path(name))
        score=scored_packet(source,name);motion=Path(oldflow[name].get('identical_motion_file',src/'repair_motion'/(name+'.npz')))
        assert score.is_file() and motion.is_file()
        refs[name]=dict(score=str(score),verified_shot_state=digest,packet_sha256=report['packet_sha'],parent=parent)
        flow[name]=dict(oldflow[name],identical_motion_file=str(motion),verified_shot_state=digest)
    write(target/'references.json',refs);write(dst/'repair_motion/summary.json',flow)


def export(candidate,version='review_fixes_1118'):
    from deliver import camera_export,file_ref,geometry
    from shared import registration
    folder=OUT/candidate;state=camera_state(candidate);n=native_for(candidate,'jin')
    coords=n.parameter_state.values;digest=numeric_sha(coords);rev=n.parameter_state.layout['layout_revision']
    src=read(PREPARED/'setup.json')['shots'];prior=read(PRIOR/'source_bindings.json');binding={}
    scores=DEST/'scores'/candidate.replace('/','_')/'all'
    for name,s in src.items():
        path=prepared_packet_path(name);p=Path(prior[name]['packet']['path'])
        with np.load(path) as a,np.load(p) as b:
            size=len(b['pts'])
            for key in ('pts','idx','uv','station_fixed','origin_row','path','family'):
                np.testing.assert_array_equal(a[key][:size],b[key])
            changed=np.flatnonzero(a['fold'][:size]!=b['fold'])
            if len(changed):
                role_file=PREPARED/'role_changes'/(name+'.npz')
                if not role_file.exists():role_file=Path(read(PREPARED/'setup.json')['parent'])/'role_changes'/(name+'.npz')
                reg=arrays(role_file)
                early=reg['row']<size
                np.testing.assert_array_equal(reg['row'][early],changed)
                np.testing.assert_array_equal(reg['previous'][early],b['fold'][changed])
                np.testing.assert_array_equal(reg['current'][early],a['fold'][changed])
                assert np.all(a['fold'][changed]==9) and not np.any(a['station_fixed'][changed])
            if 'source_packets' not in s:raise ValueError(name)
            fixed=int(np.count_nonzero(a['station_fixed']&(a['fold']<5)))
        binding[name]=dict(packet=file_ref(path),matched_errors=file_ref(scored_packet(scores,name)),
            prior_source_rows_preserved=size,eligible_fixed_anchors=fixed,source=s['source'],source_sha256=s['source_sha256'])
    write(folder/'source_bindings.json',binding)
    snapshot=EVENT/'scene/field'/(version+'.npz');save(snapshot,scene_values=coords,scene_values_sha256=digest,layout_revision=rev)
    table=EVENT/'scene'/(version+'_coordinates.json');write(table,geometry(n))
    field=dict(version_id=version,status='source_review_candidate',snapshot=file_ref(snapshot),structure=file_ref(n.scene_path),
        value_sha256=digest,layout_revision=rev,native_library=file_ref(LIBRARY),scene_parameters_active=False,coordinate_scale='rink length = 1')
    reference=sample(n);records={};verified={}
    comparison=read(folder/'source_comparison.json')
    for program,directory in PROGRAMMES.items():
        names=[name for name,s in src.items() if s['program']==program];local=folder/'review'/program
        cover=read(local/'coverage.json');info=read(local/'binding.json');native=native_for(candidate,program)
        other=sample(native)
        np.testing.assert_array_equal(other['xyz'],reference['xyz'])
        assert [v['id'] for v in other['cuts']]==[v['id'] for v in reference['cuts']]
        assert info['mode']=='complete' and info['state_sha256']==sha(folder/'state.npz') and info['scene_sha256']==sha(n.scene_path)
        assert cover['validation']['decoded_frames']==cover['frames']==sum(len(state[key+'_pts']) for key in names)
        assert info['programme_state_sha256']==programme_state_sha(state,names) and info['source_sha256']==src[names[0]]['source_sha256']
        for name in names:assert info['observation_packets'][name]==binding[name]['packet']['sha256']
        media=media_check(cover['video'],cover);media['ledger']=ledger_check(Path(info.get('source_ledger_directory',local)),state,names)
        camera=camera_export(program,state,digest,rev,version=version,prepared=PREPARED)
        rec=dict(schema='rinkspace.programme_joint_refinement.v1',event_id='ankara_2026',programme_id=directory,
            version_id=version,status='source_review_candidate',user_visual_acceptance=False,
            previous=file_ref(EVENT/'programmes'/directory/'lane_motion_1118.json'),source=registration(program)['source'],
            geometry_binding=field,camera=file_ref(camera),shots=names,fit_roles=[0,1,2,3,4],fit_observations={key:binding[key]['packet'] for key in names},
            full_review=dict(video=file_ref(cover['video']),coverage=file_ref(local/'coverage.json'),render_state=file_ref(folder/'state.npz'),
                programme_state_sha256=info['programme_state_sha256'],source_seconds=cover['source_seconds'],frames=cover['frames'],
                native_time_base=[1,12800],fps=50,source_audio=cover['source_audio'],media_verification=media),
            comparison=comparison['programmes'][program],physical_mounts=file_ref(folder/'physical_mounts.json'))
        target=EVENT/'programmes'/directory/(version+'.json');write(target,rec);records[program]=file_ref(target);verified[program]=media
    write(folder/'delivery_verification.json',verified)
    prior_input=DEST/'prepared';res=dict(schema='rinkspace.event_joint_refinement.v1',event_id='ankara_2026',version_id=version,
        status='source_review_candidate',user_visual_acceptance=False,previous=file_ref(EVENT/'scene/lane_motion_1118.json'),
        geometry_binding=field,coordinates=file_ref(table),programmes=records,selected_run=candidate,checkpoint=file_ref(folder/'state.npz'),
        fit_setup=file_ref(PREPARED/'setup.json'),source_admission=[file_ref(p/'source_audit.json') for p in dict.fromkeys([prior_input,DEST/'verified',DEST/'balanced',DEST/'material_inputs',PREPARED])],
        source_bindings=file_ref(folder/'source_bindings.json'),comparison=file_ref(folder/'source_comparison.json'),
        physical_mounts=file_ref(folder/'physical_mounts.json'),surface_motion=file_ref(folder/'motion_all_surfaces.json'),
        trajectory_extrema=file_ref(folder/'trajectory_extrema.json'),end_views=file_ref(folder/'end_views.json'),
        optics=file_ref(folder/'sensor_domain.json'),visual_review=file_ref(folder/'visual_review.json'),
        fast_source_windows=file_ref(folder/'fast_pan_checks.json'),opening_source_flow=file_ref(DEST/'opening_texture/source.json'),
        lane_caps_sensitivity=file_ref(DEST/'source/cap_review.json'))
    write(EVENT/'scene'/(version+'.json'),res)
