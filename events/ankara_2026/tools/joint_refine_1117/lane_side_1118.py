"""Identify the fourth sheet's upper boundary from source paint and its junctions."""
from collections import Counter

import cv2
import numpy as np

from feedback_material import rays
from lane_end_evidence import foreground, matte
from review import camera_state, native_for, source_frames
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, write


BASE = 'review_fixes_1118/release'
WORK = OUT / 'lane4_side_1118'
SIDE = 'demo_initial/lane4:0'
CAP = 'demo_initial/lane4:1'
TRANSVERSE = 'ice/lane4:F_transverse'
# The survey spans both long sides, the curling houses, beyond the finite
# cap and to the end boards. Its pixel sampling does not use an assumed side.
X = np.linspace(-.452, .348, 641)
Y = np.linspace(.244, .120, 249)
GX, GY = np.meshgrid(X[::8], Y[::8])
GRID = np.stack((GX, GY, np.zeros_like(GX)), axis=-1).reshape(-1, 3)
MX, MY = np.meshgrid(np.arange(len(X), dtype='f4')/8, np.arange(len(Y), dtype='f4')/8)


def project(native, state, name, shot, index, xyz):
    return native.core.project_motion(state[name+'_rigs'][index], state[name+'_controls'],
        shot['first'], shot['dt'], 1080, state[name+'_times'][index], xyz)


def inverse(native, state, name, shot, index, uv):
    rig = state[name+'_rigs'][index]
    times = state[name+'_times'][index] + rig[37]*(uv[:,1]/1079. - .5)
    camera = native.core.evaluate_motion_camera(rig, state[name+'_controls'], shot['first'], shot['dt'], times)
    direction = rays(uv, camera)
    depth = -camera['centre'][:,2]/direction[:,2]
    xy = camera['centre'][:,:2] + direction[:,:2]*depth[:,None]
    xy[(depth <= 0) | (camera['status'] != 0)] = np.nan
    return xy


def source_survey(programme):
    cv2.setNumThreads(1)
    shots=read(PREPARED/'setup.json')['shots']
    native=native_for(BASE,programme); state=camera_state(BASE)
    for name,shot in shots.items():
        if shot['program'] != programme:
            continue
        target=WORK/'survey'/(name+'.npz')
        if target.exists():
            continue
        pts=state[name+'_pts']
        sample=np.unique(np.r_[np.arange(5,len(pts),31),len(pts)-1])
        coarse={}
        for i in sample:
            p=project(native,state,name,shot,i,GRID)
            uv=np.clip(p['uv'],-20000,20000).reshape(*GX.shape,2)
            uv[p['status'].reshape(GX.shape)!=0]=-20000
            uv=np.nan_to_num(uv,nan=-20000).astype('f4')
            if np.count_nonzero((uv[:,:,0]>0)&(uv[:,:,0]<1920)&(uv[:,:,1]>0)&(uv[:,:,1]<1080))>=7:
                coarse[i]=uv
        points=[];views=[]
        person=matte(shot,pts) if coarse else None
        frames=source_frames(shot['source'],pts[list(coarse)]) if coarse else []
        for (pt,img),(i,uv) in zip(frames,coarse.items(),strict=True):
            assert pt==pts[i]
            mask=foreground(shot,person,i,pt)
            if mask is None:
                continue
            mask=cv2.dilate(mask,np.ones((13,13),np.uint8))
            mask[:304,:505]=255;mask[918:,1658:]=255
            lookup=cv2.remap(uv,MX,MY,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            valid=(lookup[:,:,0]>5)&(lookup[:,:,0]<1914)&(lookup[:,:,1]>5)&(lookup[:,:,1]<1074)
            blocked=cv2.remap(mask,lookup,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            col=cv2.remap(img,lookup,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            light=cv2.cvtColor(col,cv2.COLOR_BGR2GRAY).astype('f4')
            # Store measured darkness relative to the background and the
            # red-minus-blue channel. Either signal can distinguish the paint.
            hi=cv2.GaussianBlur(light,(0,0),5,7)-light
            red=col[:,:,2].astype('f4')-col[:,:,0].astype('f4')
            hues=red-cv2.GaussianBlur(red,(0,0),5,7)
            slab=np.stack((hi,hues),axis=2)
            slab=np.clip(np.rint(slab*4),-126,126).astype('i1')
            slab[~valid|(blocked>0)|(light<117)]=-128
            if ((slab[:,:,0]!=-128).sum()<1200):
                continue
            points.append(i);views.append(slab)
        save(target,indices=np.array(points,dtype='i4'),pts=pts[points],survey=np.asarray(views,dtype='i1'),
            camera_index=shot['camera_index'],source_sha256=shot['source_sha256'])
        print(name,len(points),flush=True)


def historic():
    shots=read(PREPARED/'setup.json')['shots'];state=camera_state(BASE)
    native=native_for(BASE,'jin')
    samples=arrays(OUT/'review_fixes_1118/free_geometry/lane4/selected_rows.npz')
    info={};packed={}
    for name,shot in shots.items():
        with np.load(prepared_packet_path(name)) as a:
            path=a['path'];r=np.flatnonzero(path==SIDE)
            if not len(r):
                continue
            idx=a['idx'][r];uv=a['uv'][r];origin=a['origin_row'][r]
            fit=np.isin(origin,samples['origin_row'][samples['shot']==name])
            folds=a['fold'][r];fixed=a['station_fixed'][r];mat=a['material'][r]
            xy=np.zeros_like(uv)
            for i in np.unique(idx):
                sel=idx==i;xy[sel]=inverse(native,state,name,shot,i,uv[sel])
            packed[name]=dict(rows=r,idx=idx,pts=a['pts'][r],uv=uv,world=xy,used=fit,fold=folds,fixed=fixed,material=mat)
            info[name]=dict(total=len(r),used=int(fit.sum()),admitted=int((folds<5).sum()),
                right_half_used=int((fit&(xy[:,0]>0)).sum()),material=dict(Counter(mat[fit].tolist())))
    for name,z in packed.items():
        save(WORK/'historic'/(name+'.npz'),**z)
    write(WORK/'historic/summary.json',info)


def composite():
    import warnings
    for programme in ['jin','okada','chen','nayeon','all']:
        files=sorted((WORK/'survey').glob('*.npz' if programme=='all' else programme+'_*.npz'))
        # Each shot supplies one median image, so a long camerawork segment
        # cannot dominate the shared source appearance.
        images=[];support=np.zeros((len(Y),len(X)),int)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for file in files:
                z=arrays(file)
                if not len(z['pts']):
                    continue
                raw=z['survey'].astype('f4');valid=raw[:,:,:,0]!=-128
                support+=valid.sum(axis=0);raw[~valid]=np.nan
                images.append(np.nanmedian(raw,axis=0)/4.)
            if not images:continue
            med=np.nanmedian(images,axis=0)
        save(WORK/'charts'/(programme+'.npz'),signals=med,support=support)
        lum=np.clip(216.-np.nan_to_num(med[:,:,0])*10,10,251).astype('u1')
        result=cv2.cvtColor(lum,cv2.COLOR_GRAY2BGR)
        result[support==0]=[55,55,55]
        result=cv2.resize(result,None,fx=1.6,fy=1.4)
        for y in (.24,.23,.22,.21,.20,.19,.18,.17,.16,.15,.14,.13):
            pix=round((Y[0]-y)/.0005*1.4)
            cv2.putText(result,f'{y:.2f}',(2,pix),cv2.FONT_HERSHEY_SIMPLEX,.42,(50,40,155),1,cv2.LINE_AA)
        cv2.imwrite(str(WORK/'charts'/(programme+'.jpg')),result)


def diagnostic_crops():
    native=native_for(BASE,'jin');state=camera_state(BASE)
    old=native_for('lane_motion_1118/checked','jin')
    shots=read(PREPARED/'setup.json')['shots']
    output=WORK/'inspection';output.mkdir(exist_ok=True)
    cases=[('jin_s08',66.58,'R'),('nayeon_s04',67.96,'R'),('jin_s23',170.78,'R'),
           ('okada_s02',55.,'cap'),('okada_s03',57.04,'cap'),('chen_s12',117.72,'cap'),
           ('okada_s01',34.62,'left'),('jin_s11',92.36,'left'),('okada_s04',73.26,'mid')]
    for name,sec,which in cases:
        shot=shots[name];pt=round(sec*12800);i=int(np.searchsorted(state[name+'_pts'],pt))
        assert state[name+'_pts'][i]==pt
        _,raw=next(source_frames(shot['source'],[pt]))
        x={'R':.222,'cap':.306,'left':-.395,'mid':0.}[which]
        y=.195 if which=='R' else .213
        centre=project(native,state,name,shot,i,np.array([[x,y,0.]]))['uv'][0]
        if not np.all(np.isfinite(centre)):continue
        lx,ly=np.clip(np.rint(centre).astype(int)-[370,110],0,[1180,860]);ux,uy=lx+740,ly+220
        guide=raw.copy()
        for model,keys,color in [(old,[SIDE],(255,205,30)),(native,[SIDE,CAP,TRANSVERSE],(0,211,255))]:
            for k in keys:
                a,b=np.array(model.paths[k]['parameters']).reshape(2,3)
                world=np.linspace(a,b,201);p=project(native,state,name,shot,i,world)
                uv=p['uv'];sel=(p['status']==0)&np.isfinite(uv).all(1)&(abs(uv)<5000).all(1)
                for rr in np.split(np.flatnonzero(sel),np.where(np.diff(np.flatnonzero(sel))!=1)[0]+1):
                    if len(rr)>1:cv2.polylines(guide,[np.int32(np.rint(uv[rr]*16))],False,color,1,cv2.LINE_AA,4)
        mark=raw.copy()
        file=WORK/'historic'/(name+'.npz')
        if file.is_file():
            z=arrays(file);m=(z['pts']==pt)&z['used']
            for uv in z['uv'][m]:
                cv2.circle(mark,tuple(np.rint(uv).astype(int)),3,(25,20,234),-1,cv2.LINE_AA)
        def panel(img,label):
            crop=img[ly:uy,lx:ux];crop=cv2.copyMakeBorder(crop,22,0,0,0,cv2.BORDER_CONSTANT,value=(23,23,23))
            cv2.putText(crop,label,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
            return crop
        rows=[panel(raw,f'Original  {name}  {sec:.3f} s'),panel(mark,'Historical pixels admitted as upper side'),
            panel(guide,'Delivered geometry (yellow); earlier upper side in the same camera (cyan)')]
        cv2.imwrite(str(output/f'{name}_{pt}_{which}.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,97])


def trace_candidates():
    """Find straight, two-sided longitudinal paint in rectified original frames."""
    import warnings
    from scipy.signal import find_peaks
    native=native_for(BASE,'jin')
    centres=[native.paths[k]['parameters'][0] for k in ('demo_initial/F4:center','demo_initial/R4:center')]
    xs=np.arange(-.414,.315,.0125)
    # The right and left houses cast long apparent flat arcs in the end camera.
    # A boundary identity must be supported outside their entire x extents.
    xs=xs[(np.min(abs(xs[:,None]-np.asarray(centres)),axis=1)>.044)]
    lines=[]
    for file in sorted((WORK/'survey').glob('*.npz')):
        z=arrays(file);shot=file.stem;observations=[]
        for i,f in zip(z['indices'],z['survey']):
            response=f[:,:,0].astype('f4')/4
            response[f[:,:,0]==-128]=np.nan
            for x in xs:
                j=int(np.argmin(abs(X-x)));patch=response[:,j-6:j+7]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    profile=np.nanmedian(patch,axis=1)
                valid=np.isfinite(profile)&(np.isfinite(patch).sum(1)>=10)
                profile=np.nan_to_num(profile)
                peaks,_=find_peaks(profile,prominence=.25)
                for iy in peaks:
                    if not .183<Y[iy]<.231 or not valid[iy-8:iy+9].all():continue
                    neighbours=patch[iy-8:iy+9]
                    valid_columns=np.isfinite(neighbours).all(0)
                    if valid_columns.sum()<9:continue
                    neighbours=neighbours[:,valid_columns]
                    before=neighbours[:4].mean(axis=0);after=neighbours[-4:].mean(axis=0)
                    contrast=np.minimum(neighbours[7:10].mean(axis=0)-before,
                                        neighbours[7:10].mean(axis=0)-after)
                    strength=float(np.median(contrast))
                    if strength<.48 or np.sum(contrast>.35)<6:continue
                    observations.append((int(i),x,Y[iy],strength))
        lines.extend([(file.stem,*v) for v in observations])
        save(WORK/'trace_candidates'/file.name,observations=np.asarray(observations).reshape(-1,4))
    # Score independent x stations within each shot. A single heavily repeated
    # patch provides at most one vote per station and shot in this survey.
    yy=np.arange(.184,.2305,.0005)
    l,r=np.meshgrid(yy,yy,indexing='ij')
    groups={}
    for shot,i,x,y,weight in lines:
        key=(shot,x)
        if key not in groups:groups[key]=np.zeros_like(l)
        t=(x+.425)/.74
        dist=(l*(1-t)+r*t-y)/.0018
        groups[key]=np.maximum(groups[key],min(3.,weight)*np.exp(-.5*dist**2))
    best={}
    for case in ['jin','okada','chen','nayeon','all']:
        score=sum((v for (s,_),v in groups.items() if case=='all' or s.startswith(case+'_')),np.zeros_like(l))
        if not score.any():continue
        ix=np.unravel_index(np.argmax(score),score.shape)
        best[case]=dict(left=float(l[ix]),right=float(r[ix]),score=float(score[ix]))
    write(WORK/'trace_candidates/consensus.json',best)
    print('paint hypotheses',best,'source candidates',len(lines))
