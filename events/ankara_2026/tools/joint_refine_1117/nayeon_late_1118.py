"""Read the two later holdout regressions and refit compact fixed-rig trajectories."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, kron

from feedback_source import contour_samples, selected_paths
from jin_board_1118 import paths
from lane_end_evidence import foreground, matte
from lane_side_1118 import BASE, WORK
from nayeon_side_1118 import bar_transitions, contours_of_coloured_panel, rail_sections, rectangle_parts
from review import camera_state, ji, native_for, source_frames
from shared import EVENT, OUT, PREPARED, arrays, numeric_sha, read, save, sha, write
from source_bundle import relative

sys.path.insert(0,str(EVENT/'programmes/chen_xi_jin_sp/tools'))
from physical_reads import coherent, samples, source_profiles, stripe_candidates


SELECTION='lane4_side_1118/selection'
PREVIOUS_NATIVE='lane4_side_1118/nayeon_guard/native/partial'
DEST=WORK/'nayeon_late'
GOAL='demo_initial/goal_line_left:0'
CASES={'nayeon_s03':dict(exact=51.94, window=(51.48,52.38), knots=(51.72,51.94,52.16), radius=.24, protected=50.80),
       'nayeon_s06':dict(exact=125.54, window=(125.30,125.84), knots=(125.43,125.55,125.67), radius=.16, protected=125.12)}


def read_red(raw,mask,projected,table):
    uv,status=projected
    uv=uv.copy(); uv[status!=0]=np.nan
    result=samples(uv,table[GOAL][1],axis=1,step=14.)
    if result is None:
        return []
    p,no,tangent,station=result
    clear=(p[:,0]>55)&(p[:,0]<1500)&(p[:,1]>490)&(p[:,1]<1048)
    p,no,tangent,station=(v[clear] for v in (p,no,tangent,station))
    if len(p)<5:
        return []
    radius=37
    profiles,blocked,_=source_profiles(raw,mask,p,no,tangent,radius,'red')
    choices=[]
    for v,h in zip(profiles,blocked,strict=True):
        c=stripe_candidates(v,h,radius,expected_min=4,max_width=27,min_contrast=2.)
        choices.append([z for z in c if abs(z[0])<29])
    accepted=coherent(choices,p,stripe=True,minimum_span=94.)
    return [(GOAL,station[i],p[i]+r[0]*no[i],p[i],no[i],r[4],p[i]+np.asarray(r[2:4])[:,None]*no[i])
            for i,r in accepted]


def measure(name,selection):
    cv2.setNumThreads(1)
    s=read(PREPARED/'setup.json')['shots'][name]; case=CASES[name]
    b=camera_state(selection); n=native_for(selection,'nayeon')
    pts=b[name+'_pts']; t=pts/12800.; lo,hi=case['window']; pt0=round(case['exact']*12800)
    exposure=np.flatnonzero((t>=lo)&(t<=hi))
    boards=selected_paths(n)['board']
    panel=paths(n); families=('Ad6','Ad7') if name.endswith('03') else ('Ad6',)
    bars=[k for k in n.paths if k.startswith('print_instance/Ad5::') and 'native3976' in k]
    keys=[k for f in families for k in panel[f]]+bars
    tab=ji.native_paths(n,keys,count=129)
    tab.update(ji.native_paths(n,[GOAL],count=513))
    tab.update(ji.native_paths(n,boards,count=3073))
    mats=matte(s,pts); results=defaultdict(list); ledger=[]; out=DEST/name;out.mkdir(parents=True,exist_ok=True)
    world={k:v[0] for k,v in tab.items()}
    stream=source_frames(s['source'],pts[exposure])
    for block in np.array_split(exposure,max(1,(len(exposure)+7)//8)):
        jobs=[(i,k,b[name+'_times'][i]) for i in block for k in tab]
        pr=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        lookup={(i,k):uv for (i,k,_),uv in zip(jobs,pr,strict=True)}
        for i in block:
            pt,raw=next(stream); assert pt==pts[i]
            person=foreground(s,mats,i,pt)
            if person is None:
                ledger.append(dict(pts=pt,rows=0,reason='source_foreground_unknown'))
                continue
            mask=cv2.dilate(person,np.ones((9,9),np.uint8)); mask[:262,:551]=255; mask[942:,1680:]=255
            role=6 if abs(int(pt)-pt0)<=256 or i%6==4 else 0
            frame=[]
            def append(f,kind,h,weight):
                k,st,uv,p,no,contrast=h[:6]
                bracket=h[6] if len(h)==7 else np.repeat(np.nan,4).reshape(2,2)
                if not ((uv>1).all() and (uv<[1919,1079]).all()):
                    return
                for key,val in dict(pts=int(pt),idx=int(i),path=k,station=float(st),uv=uv,guide=p,normal=no,
                    contrast=float(contrast),material=kind,family=f,fold=role,group=f'{name}:{i//4}:{f}',
                    scalar_weight=weight,source_bracket=bracket).items():
                    results[key].append(val)
                frame.append((f,uv,bracket))
            for f in families:
                pp=rectangle_parts({k:lookup[i,k] for k in panel[f]},tab)
                for h in contours_of_coloured_panel(raw,mask,pp,'blue' if f=='Ad7' else 'red'):
                    strong=f=='Ad7' and 'right_vertical' in h[0]
                    append(f,'source_ad7_blue_end' if strong else 'plate_transition',h,5.3 if strong else 1.8)
            for h in bar_transitions(raw,mask,{k:lookup[i,k] for k in bars},tab):
                append('Ad5','plate_transition',h,1.6)
            curves={k:lookup[i,k] for k in boards}
            parts=contour_samples(curves,tab,boards,'board')
            rails,_=rail_sections(raw,mask,parts,False)
            for family,kind,*h in rails:
                append(family,kind,h,1.5 if family=='board.top' else 2.3)
            for h in read_red(raw,mask,lookup[i,GOAL],tab):
                append('goal_line_left','source_two_flanks_paint_centre',h,7.2 if name.endswith('06') else 2.6)
            ledger.append(dict(pts=int(pt),role=role,rows=len(frame)))
            if abs(int(pt)-pt0)<=1280 and (i%3==0 or pt==pt0):
                pic=raw.copy()
                for f,uv,ends in frame:
                    color=(36,220,247) if f=='goal_line_left' else (80,215,40) if f.startswith('Ad') else (230,160,55)
                    cv2.circle(pic,tuple(np.rint(2*uv).astype(int)),3,color,-1,cv2.LINE_AA,1)
                    if np.isfinite(ends).all():
                        for e in ends:cv2.circle(pic,tuple(np.rint(2*e).astype(int)),2,(220,105,235),-1,cv2.LINE_AA,1)
                cv2.imwrite(str(out/f'{pt}_source.jpg'),raw,[1,97]);cv2.imwrite(str(out/f'{pt}_reads.jpg'),pic,[1,97])
    q={k:np.asarray(v) for k,v in results.items()}
    save(out/'read.npz',**q,source_sha256=s['source_sha256'])
    write(out/'ledger.json',dict(exact=case['exact'],input=selection,rows=ledger,source=s['source'],source_sha256=s['source_sha256']))
    for key in np.unique(q['material']):
        sel=q['material']==key; k=sel&(q['pts']==pt0)
        print(name,key,int(sel.sum()),'exact',int(k.sum()),'median',np.median(np.sum((q['uv'][k]-q['guide'][k])*q['normal'][k],axis=1)).round(2) if k.any() else None,flush=True)


def use_points(name,pack,gain):
    chosen=[];weights=[]
    for pt in np.unique(pack['pts'][pack['fold']<5]):
        at=(pack['pts']==pt)&(pack['fold']<5)
        for material in np.unique(pack['material'][at]):
            for f in np.unique(pack['family'][at&(pack['material']==material)]):
                ix=np.flatnonzero(at&(pack['material']==material)&(pack['family']==f))
                ix=ix[np.linspace(0,len(ix)-1,min(20,len(ix))).round().astype(int)]
                mult=gain if (name.endswith('03') and material=='source_ad7_blue_end') or (name.endswith('06') and f=='goal_line_left') else 1.
                chosen.extend(ix);weights.extend(mult*pack['scalar_weight'][ix]/np.sqrt(len(ix)))
    rows=np.asarray(chosen,int); assert len(np.unique(rows))==len(rows)
    return {k:v[rows] for k,v in pack.items() if v.ndim>0},rows,np.asarray(weights)


def targeted_tips(name,n,b,s):
    """Read genuine prior extrema as x-only factors at their current stations."""
    from feedback_fit import extreme

    path=OUT/'review_fixes_1118'/'material_tips'/(name+'.npz'); source=arrays(path)
    t=source['pts']/12800.
    if name.endswith('03'):
        take=(t>=52.08)&(t<=52.46)&(abs(t-52.38)>.0201)&(source['path']=='d78_material_155/path0')
        take&=source['normal'][:,0]<0
    else:
        take=(source['pts']==round(125.20*12800))&(source['family']=='goal_crease_left')
    rows=np.flatnonzero(take)
    ids=np.unique(source['path'][rows]); tab=ji.native_paths(n,ids,count=1537)
    data=defaultdict(list); provenance=[]
    for j in rows:
        i=int(source['idx'][j]); k=source['path'][j]
        p=n.core.project_motion(b[name+'_rigs'][i],b[name+'_controls'],s['first'],s['dt'],1080,
                                b[name+'_times'][i],tab[k][0])
        uv=p['uv'].copy();uv[p['status']!=0]=np.nan
        ex=extreme(uv,tab[k][1],int(source['normal'][j,0]))
        if ex is None:
            continue
        for field in ('pts','idx','path','uv','normal','contrast','family'):
            data[field].append(source[field][j])
        data['station'].append(ex[0]);data['guide'].append(ex[1]);data['fold'].append(0)
        data['material'].append('outline_xmin' if source['normal'][j,0]<0 else 'outline_xmax')
        data['group'].append(f'{name}:source-extremum:{i}')
        data['scalar_weight'].append(1.)
        data['source_bracket'].append(np.full((2,2),np.nan))
        provenance.append(j)
    return {k:np.asarray(v) for k,v in data.items()},np.asarray(provenance),path


def rebase_motion(name,old,selected,s):
    """Express the previous trajectory in the selected fixed-rig camera chart."""
    from flow_depths import rays

    n=native_for(SELECTION,'nayeon'); key=name+'_controls'
    direction=np.array([[960.,530.],[320.,470.],[1580.,470.],[960.,330.],[960.,835.]])
    times=old[name+'_times']; sample=np.unique(np.r_[np.arange(0,len(times),5),len(times)-1])
    physical=old[name+'_rigs']; chosen=selected[name+'_rigs']; c=old[key]
    step=np.array([.00025,.00025,.00075]); offsets=[]
    for i in sample:
        t=np.full(len(direction),times[i])
        a=n.core.evaluate_motion_camera(physical[i],c,s['first'],s['dt'],t)
        def ray(r,shift):
            pose=n.core.evaluate_motion_camera(r,c+shift,s['first'],s['dt'],t)
            vec=rays(direction,pose)
            return (vec/np.linalg.norm(vec,axis=1)[:,None]).ravel()
        target=ray(physical[i],np.zeros(3)); delta=np.zeros(3)
        for iteration in range(2):
            now=ray(chosen[i],delta)
            derivatives=[(ray(chosen[i],delta+d)-now)/d[j] for j,d in enumerate(np.diag(step))]
            jac=np.column_stack(derivatives)
            if not np.isfinite(jac).all() or not np.isfinite(target-now).all() or (a['status']!=0).any():
                raise ValueError(('invalid camera rebase',name,int(i)))
            delta+=np.linalg.lstsq(jac,target-now,rcond=None)[0]
        offsets.append(delta)
    measured=np.asarray(offsets)
    knots=s['first']+s['dt']*np.arange(len(c))
    x=np.clip(knots,times[0],times[-1])
    correction=CubicSpline(times[sample],measured,bc_type='natural')(x)
    return c+correction


def return_seed(name,label,transport=False):
    selected=camera_state(SELECTION); old=camera_state(BASE)
    key=name+'_controls'; s=read(PREPARED/'setup.json')['shots'][name]
    if name.endswith('03'):
        hand=camera_state(PREVIOUS_NATIVE)
        np.testing.assert_array_equal(selected[name+'_rigs'],hand[name+'_rigs'])
        selected[key]=hand[key]
        visible=(49.12,49.44,51.17,51.54)
    else:
        visible=(124.01,124.26,125.17,125.39)
    np.testing.assert_array_equal(old[name+'_pts'],selected[name+'_pts'])
    count=len(selected[key]); t=selected[name+'_pts'][0]/12800.+s['first']+s['dt']*np.arange(count)
    def rise(x):
        x=np.clip(x,0.,1.)
        return x**3*(10.+x*(-15.+6.*x))
    w=rise((t-visible[0])/(visible[1]-visible[0]))*rise((visible[3]-t)/(visible[3]-visible[2]))
    outside=rebase_motion(name,old,selected,s) if transport else old[key]
    selected[key]=outside+w[:,None]*(selected[key]-outside)
    # Preserve bits of the source-supported intervals for reproducible views.
    candidate=(hand[key] if name.endswith('03') else camera_state(SELECTION)[key])
    selected[key][w==1]=candidate[w==1]
    np.testing.assert_array_equal(selected[key][w==0],outside[w==0])
    out=DEST/name/label
    save(out/'state.npz',**selected)
    write(out/'inputs.json',dict(selection=SELECTION,prior=BASE,support_seconds=visible,
        fixed_rig=True,geometry_fixed=True,reexpressed_old_optics=transport,
        policy='quintic control envelope tied to corrected source windows; physical orientation rays used solely to express original controls in the selected rig chart' if transport else 'quintic control envelope tied to corrected source windows',
        localized_native_s03=PREVIOUS_NATIVE if name.endswith('03') else None))


def previous_mount_seed(fraction=0.):
    """Use the registered pre-review installation for the entire S03 shot."""
    selected=camera_state(SELECTION); previous=camera_state(BASE)
    name='nayeon_s03'
    old_rig=previous[name+'_rigs']; newer=selected[name+'_rigs']
    label='previous_mount'
    if fraction:
        from scipy.spatial.transform import Rotation, Slerp

        assert len(np.unique(old_rig,axis=0))==len(np.unique(newer,axis=0))==1
        rig=old_rig[0]+fraction*(newer[0]-old_rig[0])
        r0=np.stack((old_rig[0,:9],newer[0,:9])).reshape(2,3,3)
        rig[:9]=Slerp([0.,1.],Rotation.from_matrix(r0))([fraction]).as_matrix()[0].ravel()
        rig[12:15]/=np.linalg.norm(rig[12:15])
        selected[name+'_rigs']=np.repeat(rig[None],len(old_rig),axis=0)
        s=read(PREPARED/'setup.json')['shots'][name]
        selected[name+'_controls']=rebase_motion(name,previous,selected,s)
        label=f'middle_mount_{round(fraction*100)}'
    else:
        selected[name+'_controls']=previous[name+'_controls']
        selected[name+'_rigs']=old_rig
    np.testing.assert_array_equal(selected[name+'_pts'],previous[name+'_pts'])
    out=DEST/name/label
    save(out/'state.npz',**selected)
    write(out/'inputs.json',dict(geometry=SELECTION,physical=BASE,shared_calibration_candidate=SELECTION,
        fraction_of_selected_calibration=fraction,constant_rig_rows=int(len(np.unique(selected[name+'_rigs'],axis=0))),
        policy='A single fixed rig for the entire shot; physical rotation interpolation and trajectory chart alignment for the midpoint'))


def fit(name,parent,label,gain,balance=False,tips=False,orthogonal=False,blue=False,early=False,normal_only=False,
        surface_protect=False):
    from solve import Journal,stats

    case=CASES[name]; out=DEST/name/label;out.mkdir(parents=True,exist_ok=True); log=Journal(out/'solve.jsonl')
    pack_file=DEST/name/'read.npz'; pack=arrays(pack_file)
    if blue or early:
        pack_file=WORK/'nayeon_source'/'verified_inputs'/(name+'.npz')
        raw=arrays(pack_file)
        bounds=(49.48,51.34) if early else (125.18,125.30)
        select=(raw['pts']>=round(bounds[0]*12800))&(raw['pts']<=round(bounds[1]*12800))
        select&=(raw['fold']<5)&np.isin(raw['material'],['plate_transition','review_rail_edge','front_rail_edge'])
        if early:
            select&=(abs(raw['pts']-round(49.98*12800))>256)&(abs(raw['pts']-round(50.8*12800))>256)
        origin=np.flatnonzero(select)
        pack={k:v[origin] for k,v in raw.items() if v.ndim>0 and len(v)==len(select)}
        pack['scalar_weight']=np.where(pack['family']=='board.top',1.5,2.)
        if early:
            pack['scalar_weight']=np.select([
                pack['material']=='front_rail_edge',
                (pack['family']=='board.bottom')&(pack['uv'][:,0]<310),
                pack['material']=='plate_transition'],[5.1,5.4,2.5],default=1.6)
        pack['source_bracket']=np.full((len(origin),2,2),np.nan)
        pack['source_sha256']=raw['source_sha256']
    q,source_rows,w=use_points(name,pack,gain)
    if blue or early:
        source_rows=origin[source_rows]
    if early:
        w[(q['material']=='plate_transition')&(abs(q['normal'][:,0])>.8)]*=1.7
    if balance and name.endswith('03'):
        w[q['family']=='goal_line_left']*=1.32 if tips else 2.55
        w[q['family']=='board.bottom']*=1.75
        w[q['family']=='board.top']*=1.35
        if tips:
            w[(q['family']=='Ad5')|(q['family']=='Ad6')]*=2.3
    if orthogonal and name.endswith('03'):
        plates=(q['family']=='Ad5')|(q['family']=='Ad6')
        if early:
            near=q['uv'][:,0]<310
            w[plates&(abs(q['normal'][:,0])>.8)]*=2.0
            w[(q['family']=='Ad7')&(abs(q['normal'][:,0])>.8)]*=1.8
            w[(q['family']=='board.bottom')&near]*=4.5
            w[q['material']=='front_rail_edge']*=2.2
            w[(q['family']=='board.bottom')&~near]*=.85
        else:
            w[plates&(abs(q['normal'][:,0])>.8)]*=3.6
            w[q['family']=='goal_line_left']*=1.95
            w[q['family']=='board.bottom']*=1.05
    if surface_protect:
        w[q['family']=='goal_line_left']*=1.45
        w[q['family']=='board.bottom']*=1.48
        w[q['family']=='board.top']*=1.35
    s=read(PREPARED/'setup.json')['shots'][name]; b=camera_state(parent); n=native_for(parent,'nayeon')
    if name.endswith('03') and parent==SELECTION:
        previously=camera_state(PREVIOUS_NATIVE)
        np.testing.assert_array_equal(b[name+'_rigs'],previously[name+'_rigs'])
        b[name+'_controls']=previously[name+'_controls']
    direct=len(source_rows); extrema_source=None
    if tips or blue:
        extra,provenance,extrema_source=targeted_tips(name,n,b,s)
        if len(provenance):
            q={k:np.concatenate((v,extra[k])) for k,v in q.items()}
            source_rows=np.r_[source_rows,provenance]
            strength=(11. if orthogonal else 3.8) if name.endswith('03') else 2.5*gain if blue else 4.8
            w=np.r_[w,np.repeat(strength,len(provenance))]
    if early:
        ext=WORK/'nayeon_guard'/'native'/'partial'/'used_source_uv.npz'; whole=arrays(ext)
        valid=whole['station_fixed']&(whole['material']=='whole_print_transfer'); extrema_source=ext
        tail={k:whole[k][valid] for k in q if k in whole}
        tail['guide']=np.full((valid.sum(),2),np.nan)
        tail['scalar_weight']=np.full(valid.sum(),1.)
        tail['source_bracket']=np.full((valid.sum(),2,2),np.nan)
        q={k:np.concatenate((v,tail[k])) for k,v in q.items()}
        source_rows=np.r_[source_rows,np.flatnonzero(valid)]
        w=np.r_[w,np.repeat(4.4,valid.sum())]
    pts=b[name+'_pts']; t=b[name+'_times']; controls=b[name+'_controls']; r=b[name+'_rigs']
    np.testing.assert_array_equal(pts[q['idx']],q['pts'])
    assert s['source_sha256']==pack['source_sha256'].item() and (q['fold']<5).all()
    seconds=pts[0]/12800.+s['first']+s['dt']*np.arange(len(controls))
    centers=np.asarray(case['knots'])
    if early:
        centers=np.array([49.62,49.91,50.20,50.50,50.78,51.06,51.24])
    if tips:
        centers=np.r_[centers, [52.34,52.49] if name.endswith('03') else [125.23]]
    radius=.30 if early else case['radius']
    shape=np.maximum(0.,1.-((seconds[:,None]-centers[None])/radius)**2)**3
    if tips and name.endswith('06'):
        shape[:, -1]=np.maximum(0.,1.-((seconds-125.23)/.056)**2)**3
        shape[seconds<125.18]=0.
    if blue:
        shape=(np.maximum(0.,1.-((seconds-125.225)/.087)**2)**3)[:,None]
        shape[seconds<125.18]=0.
    scales=np.array([[.002],[0.],[0.]]) if blue else np.diag([.002,.002,.004])
    matrix=kron(csr_matrix(shape),csr_matrix(scales),format='csr')
    active=shape.any(1); rigs,camera_ids=np.unique(r[q['idx']],axis=0,return_inverse=True)
    no=q['normal']; tangent=np.c_[-no[:,1],no[:,0]]
    whiten=w[:,None,None]*np.stack((no,.60*tangent),axis=1)
    ids=np.arange(len(source_rows),dtype=np.int32)
    fixed=ids>=direct
    hold_stations=fixed|normal_only
    ranks=np.where(hold_stations,1,2).astype(np.int32)
    local_ids=np.where(hold_stations,-1,np.cumsum(~hold_stations)-1).astype(np.int32)
    factors=dict(local_ids=local_ids,group_offsets=np.arange(len(ids)+1,dtype=np.int32),members=ids,
        ranks=ranks,weight_offsets=np.r_[0,np.cumsum(2*ranks,dtype=np.int32)],
        weights=whiten[np.column_stack((np.ones(len(ids),bool),~hold_stations))].ravel())
    mcam=dict(first_control_time=s['first'],control_dt=s['dt'],image_height=1080,reference_pts=0,
        time_base_num=1,time_base_den=12800)
    log('start',rows=len(ids),source=str(pack_file),source_sha256=sha(pack_file),parent=parent,
        binding='selection geometry, fixed installation, compact trajectory',dimensions=matrix.shape[1],
        altered_knots=[float(seconds[active].min()),float(seconds[active].max())],
        shot=name,target_gain=gain,balances_rail_and_goal=balance,source_extrema=int(fixed.sum()),
        pan_only_blue_guard=blue,refined_previous_mount=early,source_normal_residual_only=normal_only,
        surface_protection=surface_protect,
        extrema_source=str(extrema_source) if extrema_source else None,
        exact_source_pts=int(round(case['exact']*12800)),held_capture=True)
    with n.resident(camera=mcam,controls=controls,rigs=rigs,camera_ids=camera_ids.astype(np.int32),
                    path_ids=q['path'],stations=q['station'],pts=q['pts'],observed=q['uv'],motion_times=t[q['idx']]) as p:
        p.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.65+1e-6))
        support=p._path_supports[p._codes[:len(ids)]]
        window=np.where(np.char.startswith(q['path'],'board/continuous/'),.030,.20)
        lower=np.maximum(support[:,0],q['station']-window)[~hold_stations];upper=np.minimum(support[:,1],q['station']+window)[~hold_stations]
        plan=p.configure_linearization(directions=matrix,local_scales=np.full(np.count_nonzero(~hold_stations),.0015),huber_delta=1.65,**factors)
        gpu=p.configure_solver(lower=lower,upper=upper,device_budget_bytes=11*1024**3,
            host_budget_bytes=24*1024**3,initial_lambda_relative=.001,damping_metric='jacobi_image')
        before=p.evaluate_errors();assert np.all(np.isfinite(before))
        log('prepared',rows=len(ids),plan=plan,gpu=gpu,before=stats(before))
        def checkpoint():
            expo=p.export_state(); state=dict(b); state[name+'_controls']=expo['controls']
            state['scene_values_sha256']=numeric_sha(state['scene_values'])
            np.testing.assert_array_equal(state[name+'_controls'][~active],controls[~active])
            save(out/'state.npz',**state)
        def report(rec):
            if rec['summary']['accepted']:
                checkpoint()
            value=rec['summary']['trial_cost']
            event='trial' if value is not None and np.isfinite(value) else 'trial_domain_failure'
            log(event,**rec,checkpoint=str(out/'state.npz'))
            return not (out/'stop').exists()
        answer=p.solve(relative_cost_tolerance=1e-6,step_tolerance=1e-6,progress=report,
            numerical_progress=lambda rec:log('factorization_retry',**rec))
        checkpoint()
        after=p.evaluate_errors(); stations=p.export_state()['stations'][:len(ids)]
        save(out/'used_source_uv.npz',**q,source_row=source_rows,input_packet=fixed.astype('i1'),station_fixed=fixed,
            source_sha256=pack['source_sha256'],source_fold=q['fold'],weight=w)
        if not (blue or early):
            save(out/'new_render_uv.npz',**{k:v[~fixed] for k,v in q.items()},source_row=source_rows[~fixed],
                station_fixed=np.zeros(direct,bool),source_sha256=pack['source_sha256'],source_fold=q['fold'][~fixed])
        save(out/'fitted_stations.npz',stations=stations,source_row=source_rows,pts=q['pts'])
        save(out/'errors.npz',before=before,after=after,pts=q['pts'],family=q['family'])
        write(out/'report.json',dict(parent=parent,rows=len(ids),before=stats(before),after=stats(after),solver=answer,
            activated_knot_range=[float(seconds[active].min()),float(seconds[active].max())],source_sha256=pack['source_sha256'].item(),
            observations=dict(packet=str(pack_file),sha256=sha(pack_file),role='new original source measurements; fit roles 0; development check role 6',
                              archival_extrema=str(extrema_source) if extrema_source else None,archival_rows=int(fixed.sum()))))
        log('finish',rows=len(ids),before=stats(before),after=stats(after),solver=answer)


def compare(name,labels):
    from side_repair import positions
    import solve

    case=CASES[name];out=DEST/name; q=arrays(out/'read.npz'); sec=q['pts']/12800.
    models={'parent':BASE,'selection':SELECTION,**{v:relative(out/v) for v in labels}}
    stats={}; at=q['pts']==round(case['exact']*12800)
    for tag,path in models.items():
        p,_=positions(path,name,q); residual=np.sum((q['uv']-p)*q['normal'],axis=1)
        assert np.all(np.isfinite(residual))
        all_metrics={}
        for when,m in [('exact',at),('check',q['fold']==6),('supported',np.ones(len(sec),dtype=bool))]:
            metrics={}
            for f in np.unique(q['family'][m]):
                part=m&(q['family']==f)
                if f.startswith('Ad'):
                    part&=abs(q['normal'][:,0])>.8
                if f=='Ad7' and name.endswith('03'):
                    part&=q['material']=='source_ad7_blue_end'
                if part.any():
                    metrics[str(f)]=solve.stats(abs(residual[part]))
            all_metrics[when]=metrics
        stats[tag]=all_metrics
    write(out/'comparison.json',dict(models=models,exact=case['exact'],by_model=stats))
    for tag,report in stats.items():print(name,tag,{f:round(v['p90'],2) for f,v in report['exact'].items()},flush=True)


def tail_seed():
    """Retain the verified Ad7 response and rejoin the matched early-state motion."""
    name='nayeon_s03'; box=DEST/name
    early=camera_state(relative(box/'middle_early')); late=camera_state(relative(box/'middle_late'))
    s=read(PREPARED/'setup.json')['shots'][name]; key=name+'_controls'
    np.testing.assert_array_equal(early['scene_values'],late['scene_values'])
    np.testing.assert_array_equal(early[name+'_rigs'],late[name+'_rigs'])
    t=early[name+'_pts'][0]/12800.+s['first']+s['dt']*np.arange(len(early[key]))
    x=np.clip((52.29-t)/(52.29-52.05),0.,1.); w=x**3*(10+x*(-15+6*x))
    c=early[key]+w[:,None]*(late[key]-early[key]);c[w==1]=late[key][w==1]
    early[key]=c
    out=box/'tail_join_seed'
    save(out/'state.npz',**early)
    write(out/'inputs.json',dict(early=relative(box/'middle_early'),late=relative(box/'middle_late'),
        fade_knots=[52.05,52.29],physical_calibration_fixed=True))


def tail_fit(label,absolute=None,bridge=False,absolute_gain=1.,parent=None,tip_gain=6.2,lead=False):
    """Bridge the S03 tail with observed displacements and fixed installation."""
    import motion_tracks
    from solve import Journal, stats

    name='nayeon_s03'; parent=parent or 'lane4_side_1118/nayeon_late/nayeon_s03/middle_late'
    out=DEST/name/label; out.mkdir(parents=True,exist_ok=True); log=Journal(out/'solve.jsonl')
    s=read(PREPARED/'setup.json')['shots'][name]; b=camera_state(parent); n=native_for(parent,'nayeon')
    controls=b[name+'_controls']; times=b[name+'_times']; pts=b[name+'_pts']; rig=b[name+'_rigs']
    actual_ring,inv=np.unique(rig,axis=0,return_inverse=True)
    source=WORK/'motion_evidence'/'nayeon_s03_extended'/'mixed'
    shapes={'world':(0,3),'uv':(0,2,2),'times':(0,2),'pts':(0,2),'camera':(0,2),
        'weight':(0,),'shot':(0,),'source_row':(0,),'lag':(0,), 'triplet':(0,),
        'prev2_uv':(0,2),'prev2_times':(0,), 'prev2_pts':(0,), 'prev2_camera':(0,), 'triplet_scale':(0,)}
    types={'pts':np.int64,'camera':np.int32,'source_row':np.int64,'lag':np.int8,
        'prev2_pts':np.int64,'prev2_camera':np.int32,'triplet':bool,'shot':'U12'}
    tracks={k:np.empty(v,dtype=types.get(k,np.float64)) for k,v in shapes.items()}
    binding=motion_tracks.append_board_texture(tracks,source,{'shots':{name:s}},b,
        {name:(0,len(controls))},{name:dict(start=0,inverse=inv)},native=n)
    if len(tracks['world'])<500:
        raise ValueError('Incomplete tail displacement input')
    track_times=tracks['pts']/12800.
    keep=(track_times[:,0]>=(52.16 if bridge else 52.30))&(track_times[:,1]<=53.42)
    tracks={k:v[keep] for k,v in tracks.items()}
    rows=np.empty(0,int)
    if absolute:
        packet=arrays(absolute); active=(packet['fold']<5)&(packet['pts']>=round(52.44*12800))
        active&=packet['pts']<=round(53.18*12800)
        if 'role' in packet:
            active&=packet['role']=='repair_support'
        rows=np.flatnonzero(active)
        q={k:v[rows] for k,v in packet.items() if v.ndim>0 and len(v)==len(active)}
        np.testing.assert_array_equal(pts[q['idx']],q['pts'])
        assert packet['source_sha256'].item()==s['source_sha256']
        q.update(camera=inv[q['idx']],times=times[q['idx']]-s['first'])
        if 'normal' not in q:
            vertical=np.char.find(q['path'],'vertical')>=0
            q['normal']=np.c_[vertical,~vertical].astype(float)
        h=q['normal']; t=np.c_[-h[:,1],h[:,0]]
        group=np.column_stack((q['pts'], np.unique(q['family'],return_inverse=True)[1]))
        _,i,c=np.unique(group,axis=0,return_inverse=True,return_counts=True)
        gain=absolute_gain*np.where(np.char.startswith(q['family'],'Ad'),7.5,3.0)/np.sqrt(c[i])
        white=gain[:,None,None]*np.stack((h,.52*t),axis=1)
    else:
        q=dict(path=np.empty(0,'U32'),station=np.empty(0),pts=np.empty(0,np.int64),uv=np.empty((0,2)),
            camera=np.empty(0,np.int32),times=np.empty(0))
        white=np.empty((0,2,2))
    bridge_file=None; tip_data=None
    count=len(rows)
    if bridge:
        from feedback_fit import extreme

        bridge_file=OUT/'review_fixes_1118'/'material_tips'/(name+'.npz')
        a=arrays(bridge_file)
        k='d78_material_155/path0'; use=(a['path']==k)&(a['normal'][:,0]<0)
        use&=np.isin(a['pts'],np.rint(np.array([52.24,52.34,52.38,52.42])*12800).astype(np.int64))
        tab=ji.native_paths(n,[k],count=1537)[k]; data=defaultdict(list)
        for j in np.flatnonzero(use):
            i=int(a['idx'][j]); qx=n.core.project_motion(rig[i],controls,s['first'],s['dt'],1080,times[i],tab[0]); v=qx['uv']
            v[qx['status']!=0]=np.nan; ex=extreme(v,tab[1],-1)
            if ex is None:
                continue
            for f,item in dict(path=k,station=ex[0],pts=a['pts'][j],uv=a['uv'][j],camera=inv[i],
                               times=times[i]-s['first'],source_row=j,normal=a['normal'][j],idx=i,
                               family='F4',material='outline_xmin',fold=0).items():data[f].append(item)
        tip_data={k:np.asarray(v) for k,v in data.items()}
        if len(tip_data.get('pts',[]))<3:
            raise ValueError('Missing actual outer-circle tip support')
        count+=len(tip_data['pts'])
        w=np.zeros((len(tip_data['pts']),2,2)); w[:,0,0]=tip_gain
        white=np.concatenate((white,w))
    bound={f:np.r_[q[f],tip_data[f]] if tip_data else q[f] for f in ('path','station','pts','uv','camera','times')}
    pfields=motion_tracks.attach(n,bound,tracks)
    factors=motion_tracks.factors(count,white,tracks)
    if count>len(rows):
        tip_sel=np.arange(len(rows),count)
        factors['local_ids'][tip_sel]=-1; factors['ranks'][tip_sel]=1
        remove=(factors['weight_offsets'][tip_sel,None]+[2,3]).ravel()
        factors['weights']=np.delete(factors['weights'],remove)
        sizes=np.diff(factors['weight_offsets']); sizes[tip_sel]-=2
        factors['weight_offsets']=np.r_[0,np.cumsum(sizes,dtype=np.int32)]
    seconds=pts[0]/12800.+s['first']+s['dt']*np.arange(len(controls))
    width=.205; centres=np.array([52.53,52.69,52.86,53.02,53.13])
    if bridge:
        width=.25; centres=np.array([52.45,52.60,52.77,52.94,53.09])
    basis=np.maximum(0.,1.-((seconds[:,None]-centres[None])/width)**2)**3
    if lead:
        # The first certified pair starts at 52.16. A continuous direction
        # reaches across its endpoint without a coefficient step at 52.20.
        front=np.maximum(0.,1.-((seconds-52.26)/.22)**2)**3
        basis=np.column_stack((front,basis))
    if not bridge:
        basis[seconds<=52.425]=0.
    active=basis.any(1)
    dirs=kron(csr_matrix(basis),csr_matrix(np.diag([.002,.002,.004])),format='csr')
    log('start',rows=count,relative_pairs=len(tracks['weight']),source=binding,parent=parent,
        fixed_installation=True,physical_calibration_frozen=True,scene_frozen=True,columns=dirs.shape[1],
        local_knot_range=[float(seconds[active].min()),float(seconds[active].max())],
        actual_uv_source=str(absolute) if absolute else None,bridge_tip_source=str(bridge_file) if bridge_file else None,
        tip_gain=tip_gain if bridge else None,early_continuation=lead)
    mcam=dict(first_control_time=0.,control_dt=s['dt'],image_height=1080,reference_pts=0,time_base_num=1,time_base_den=12800)
    with n.resident(camera=mcam,controls=controls,rigs=actual_ring,camera_ids=pfields['camera'],
        path_ids=pfields['path'],stations=pfields['station'],pts=pfields['pts'],observed=pfields['uv'],
        motion_times=pfields['times']) as p:
        p.configure_projection_time_support(half_seconds=float(np.max(abs(actual_ring[:,37]))*.65+1e-6))
        support=p._path_supports[p._codes[:len(rows)]]
        lo=np.maximum(support[:,0],q['station']-.035); hi=np.minimum(support[:,1],q['station']+.035)
        plan=p.configure_linearization(directions=dirs,local_scales=np.full(len(rows),.0015),huber_delta=2.7,**factors)
        gpu=p.configure_solver(lower=lo,upper=hi,device_budget_bytes=11*1024**3,
            host_budget_bytes=24*1024**3,initial_lambda_relative=.001,damping_metric='jacobi_image')
        log('prepared',rows=count,relative_pairs=len(tracks['weight']),plan=plan,gpu=gpu)
        def checkpoint():
            export=p.export_state(); result=dict(b); result[name+'_controls']=export['controls']
            np.testing.assert_array_equal(result[name+'_controls'][~active],controls[~active])
            save(out/'state.npz',**result)
        def progress(rec):
            if rec['summary']['accepted']:
                checkpoint()
            v=rec['summary']['trial_cost']
            log('trial' if v is not None and np.isfinite(v) else 'trial_domain_failure',**rec,checkpoint=str(out/'state.npz'))
            return not (out/'stop').exists()
        answer=p.solve(relative_cost_tolerance=1e-6,step_tolerance=1e-6,progress=progress,
            numerical_progress=lambda rec:log('factorization_retry',**rec))
        checkpoint()
        if len(rows):
            fields={k:v for k,v in q.items() if k not in ('camera','times','station_fixed')}
            save(out/'used_source_uv.npz',**fields,anchor_row=rows,source_sha256=s['source_sha256'],
                station_fixed=np.zeros(len(rows),bool))
        if tip_data:
            save(out/'used_f4_uv.npz',**{k:v for k,v in tip_data.items() if k not in ('camera','times')},
                source_sha256=s['source_sha256'],station_fixed=np.ones(len(tip_data['pts']),bool))
        save(out/'used_tracks.npz',**tracks,source_sha256=s['source_sha256'])
        write(out/'report.json',dict(parent=parent,binding=binding,fit_direct_rows=count,phase='relative true print and ice displacement',
            absolute_source=str(absolute) if absolute else None,tip_source=str(bridge_file) if bridge_file else None,
            used_motion_pairs=len(tracks['weight']),solver=answer))
        log('finish',rows=count,relative_pairs=len(tracks['weight']),solver=answer)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('measure','fit','compare','return-seed','previous-mount','tail-fit','tail-seed'))
    ap.add_argument('shot',choices=tuple(CASES))
    ap.add_argument('--parent',default=SELECTION);ap.add_argument('--label',default='native');ap.add_argument('--gain',type=float,default=1.)
    ap.add_argument('--balance',action='store_true')
    ap.add_argument('--transport',action='store_true')
    ap.add_argument('--tips',action='store_true')
    ap.add_argument('--orthogonal',action='store_true')
    ap.add_argument('--blue',action='store_true')
    ap.add_argument('--early',action='store_true')
    ap.add_argument('--normal-only',action='store_true')
    ap.add_argument('--surface-protect',action='store_true')
    ap.add_argument('--mount-fraction',type=float,default=0.)
    ap.add_argument('--labels',nargs='+',default=['native'])
    ap.add_argument('--absolute',type=Path)
    ap.add_argument('--bridge',action='store_true')
    ap.add_argument('--tip-gain',type=float,default=6.2)
    ap.add_argument('--lead',action='store_true')
    args=ap.parse_args()
    if args.action=='measure':measure(args.shot,args.parent)
    if args.action=='fit':fit(args.shot,args.parent,args.label,args.gain,args.balance,args.tips,args.orthogonal,
                              args.blue,args.early,args.normal_only,args.surface_protect)
    if args.action=='compare':compare(args.shot,args.labels)
    if args.action=='return-seed':return_seed(args.shot,args.label,args.transport)
    if args.action=='previous-mount':previous_mount_seed(args.mount_fraction)
    if args.action=='tail-seed':tail_seed()
    if args.action=='tail-fit':tail_fit(args.label,args.absolute,args.bridge,args.gain,
                                      args.parent if args.parent!=SELECTION else None,args.tip_gain,args.lead)
