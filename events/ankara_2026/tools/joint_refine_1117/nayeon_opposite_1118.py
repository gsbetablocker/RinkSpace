"""Fit the S03 opposite-side true plate edges with real inter-exposure motion."""
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import csr_matrix, kron

import motion_tracks
from feedback_fit import extreme
from lane_side_1118 import BASE, WORK
from motion_delivery import camera_audit
from review import camera_state, ji, native_for
from shared import OUT, PREPARED, arrays, read, save, sha, write
from side_repair import positions
from solve import Journal, stats


NAME='nayeon_s03'
ROOT=WORK/'nayeon_late'/NAME
PARENT='lane4_side_1118/nayeon_late/nayeon_s03/tail_join_lead'
ONSET_PARENT='lane4_side_1118/nayeon_late/nayeon_s03/opposite_admitted'
RIGHT=WORK/'appearance_audit'/'final_risk'/'nayeon_s03_ad23_edges.npz'
LEFT=ROOT/'read.npz'
TIPS=OUT/'review_fixes_1118'/'material_tips'/(NAME+'.npz')
FLOW=ROOT/'opposite_plate_motion'/'mixed'


def pair_actual_rows(q,admitted):
    """Bind exact source coordinates and physical paths to the current admitted input."""
    eligible=np.flatnonzero(np.isin(admitted['pts'],np.unique(q['pts'])))
    def key(data,j):
        return int(data['pts'][j]),str(data['path'][j]),*map(float,data['uv'][j])
    index={}
    for j in eligible:
        index.setdefault(key(admitted,j),[]).append(j)
    matched=[]
    for j in range(len(q['pts'])):
        rows=index.get(key(q,j),[])
        if q['source_code'][j]==0:
            assert q['source_row'][j] in rows
            matched.append(q['source_row'][j])
        elif rows:
            same=[i for i in rows if admitted['station_fixed'][i]==q['station_fixed'][j]]
            matched.append(min(same or rows,key=lambda i:abs(admitted['station'][i]-q['station'][j])))
        else:
            matched.append(-1)
    return np.asarray(matched,dtype=np.int64)


def onset_edges(admitted):
    """The early true plate boundaries checked against the original frame crops."""
    seconds=admitted['pts']/12800.
    v=np.char.find(admitted['path'],'vertical')>=0
    select=(seconds>=51.66)&(seconds<=51.84)&v&(np.arange(len(seconds))<100000)
    select&=np.isin(admitted['family'],['Ad2','Ad3','Ad6','Ad7'])&(admitted['fold']<5)
    return np.flatnonzero(select)


def observations(n,b,s,onset=False):
    right=arrays(RIGHT)
    assert right['source_sha256'].item()==s['source_sha256']
    assert not np.isin(right['row'],[83708,83709,83710]).any()
    usable=np.flatnonzero((right['role']=='repair_support')&(right['fold']<5))
    r={k:right[k][usable] for k in ('pts','idx','uv','station','path','family','material')}
    r.update(normal=np.repeat([[1.,0.]],len(usable),axis=0),source_row=right['row'][usable],
        packet_row=usable,source_code=np.zeros(len(usable),'i1'),fold=right['fold'][usable],station_fixed=np.zeros(len(usable),bool))
    left=arrays(LEFT)
    assert left['source_sha256'].item()==s['source_sha256']
    t=left['pts']/12800.; p=left['family']; kind=left['material']
    use=(t>=51.70)&(t<=52.44)&(left['fold']<5)&(left['idx']%2==0)
    use&=np.isin(p,['Ad5','Ad6','Ad7','board.top','board.bottom','goal_line_left'])
    use&=((p!='goal_line_left')|np.isfinite(left['source_bracket']).all((1,2)))
    wanted=[]
    for pt in np.unique(left['pts'][use]):
        for fam in np.unique(p):
            mask=use&(left['pts']==pt)&(p==fam)
            categories=(mask&(kind=='source_ad7_blue_end'),mask&(kind!='source_ad7_blue_end')) if fam=='Ad7' else (mask,)
            for part in categories:
                ix=np.flatnonzero(part)
                if len(ix):wanted.extend(ix[np.linspace(0,len(ix)-1,min(22,len(ix))).round().astype(int)])
    wanted=np.asarray(wanted,int)
    l={k:left[k][wanted] for k in ('pts','idx','uv','station','path','family','material','normal','fold')}
    l.update(source_row=wanted,packet_row=wanted,source_code=np.ones(len(wanted),'i1'),station_fixed=np.zeros(len(wanted),bool))
    a=arrays(TIPS); pts=b[NAME+'_pts']; times=b[NAME+'_times']; rig=b[NAME+'_rigs']; c=b[NAME+'_controls']
    out=[]; k='d78_material_155/path0'; t=a['pts']/12800.
    rows=np.flatnonzero((t>=52.24)&(t<=52.42)&(a['path']==k)&(a['normal'][:,0]<0))
    table=ji.native_paths(n,[k],count=1537)[k]
    for j in rows:
        i=a['idx'][j]; p=n.core.project_motion(rig[i],c,s['first'],s['dt'],1080,times[i],table[0]); uv=p['uv']
        uv[p['status']!=0]=np.nan
        value=extreme(uv,table[1],-1)
        if value is not None:
            out.append(dict(pts=a['pts'][j],idx=i,uv=a['uv'][j],station=value[0],path=k,family='F4',material='outline_xmin',
                fold=0,station_fixed=True,normal=[-1.,0.],source_row=j,packet_row=j,source_code=2))
    if not out:
        raise ValueError('No original F4 endpoint support')
    ext={key:np.asarray([v[key] for v in out]) for key in out[0]}
    q={key:np.concatenate((r[key],l[key],ext[key])) for key in r}
    target=right['admitted_path'].item()
    assert sha(target)==right['admitted_sha256'].item()
    admitted=arrays(target)
    match=pair_actual_rows(q,admitted)
    if onset:
        rows=onset_edges(admitted)
        rows=rows[~np.isin(rows,match)]
        head={key:admitted[key][rows] for key in q if key in admitted}
        head.update(normal=np.repeat([[1.,0.]],len(rows),axis=0),source_row=rows,
            packet_row=rows,source_code=np.full(len(rows),3,'i1'))
        q={key:np.concatenate((q[key],head[key])) for key in q}
        match=np.r_[match,rows]
    keep=(match>=0)&(admitted['fold'][match]<5)
    if onset:
        leave=np.rint(np.array([51.70,51.86,51.94,52.32,52.38])*12800).astype(np.int64)
        keep&=(q['pts']<=round(52.26*12800))&~np.isin(q['pts'],leave)
    q={key:v[keep] for key,v in q.items()}
    q['admitted_row']=match[keep]
    q['fold']=admitted['fold'][match[keep]]
    np.testing.assert_array_equal(pts[q['idx']],q['pts'])
    return q


def track_inputs(n,b,s,inv,fit,onset=False):
    shapes={'world':(0,3),'uv':(0,2,2),'times':(0,2),'pts':(0,2),'camera':(0,2),
        'weight':(0,),'shot':(0,),'source_row':(0,),'lag':(0,), 'triplet':(0,),
        'prev2_uv':(0,2),'prev2_times':(0,), 'prev2_pts':(0,), 'prev2_camera':(0,), 'triplet_scale':(0,)}
    types={'pts':np.int64,'camera':np.int32,'source_row':np.int64,'lag':np.int8,
        'prev2_pts':np.int64,'prev2_camera':np.int32,'triplet':bool,'shot':'U12'}
    t={key:np.empty(v,dtype=types.get(key,np.float64)) for key,v in shapes.items()}
    report=motion_tracks.append_board_texture(t,FLOW,{'shots':{NAME:s}},b,
        {NAME:(0,len(b[NAME+'_controls']))},{NAME:dict(start=0,inverse=inv)},native=n)
    a=arrays(FLOW/(NAME+'.npz')); surfaces=a['family'][t['source_row']]
    begin,end=(51.58,52.38) if onset else (51.62,52.50)
    select=(t['pts'][:,0]>=round(begin*12800))&(t['pts'][:,1]<=round(end*12800))
    if fit:
        select&=(t['lag']==1)|(t['lag']==2)|(t['lag']==4)|(t['lag']==8)
    return {key:v[select] for key,v in t.items()},surfaces[select],report


def fit(label,right_gain,flow_gain,ad7_gain,tip_gain,tip_basis,onset,focus_flow_gain):
    out=ROOT/label;out.mkdir(parents=True,exist_ok=True); log=Journal(out/'solve.jsonl')
    parent=ONSET_PARENT if onset else PARENT
    n=native_for(parent,'nayeon'); b=camera_state(parent); s=read(PREPARED/'setup.json')['shots'][NAME]
    q=observations(n,b,s,onset); count=len(q['pts']); c=b[NAME+'_controls']; rig=b[NAME+'_rigs']; times=b[NAME+'_times']; pts=b[NAME+'_pts']
    rigs,inv=np.unique(rig,axis=0,return_inverse=True)
    t,surface,report=track_inputs(n,b,s,inv,True,onset)
    t['weight']*=flow_gain*np.where(surface==0,1.,.72)
    if onset:
        focus=(t['lag']==8)&(t['pts'][:,1]>=round(51.82*12800))&(t['pts'][:,1]<=round(51.98*12800))
        t['weight'][focus]*=focus_flow_gain
    world_rows={**q,'camera':inv[q['idx']],'times':times[q['idx']]-s['first']}
    packed=motion_tracks.attach(n,world_rows,t)
    family=q['family']; code=q['source_code']; material=q['material']
    group=np.rec.fromarrays((q['pts'],family,material))
    _,g,ng=np.unique(group,return_inverse=True,return_counts=True)
    w=np.select(((code==0)|((code==3)&np.isin(family,['Ad2','Ad3'])),code==2,
                 (material=='source_ad7_blue_end')|((code==3)&(family=='Ad7')),family=='goal_line_left',
                 family=='board.bottom',family=='board.top'),
                (right_gain,tip_gain,ad7_gain,5.8,4.1,3.8),default=3.8)/np.sqrt(ng[g])
    h=q['normal']; white=np.zeros((count,2,2));white[:,0]=h*w[:,None]
    factors=motion_tracks.factors(count,white,t)
    factors['local_ids'][:count]=-1;factors['ranks'][:count]=1
    remove=(factors['weight_offsets'][:count,None]+[2,3]).ravel()
    factors['weights']=np.delete(factors['weights'],remove)
    sizes=np.diff(factors['weight_offsets']);sizes[:count]-=2
    factors['weight_offsets']=np.r_[0,np.cumsum(sizes,dtype=np.int32)]
    sec=pts[0]/12800.+s['first']+s['dt']*np.arange(len(c))
    centres=np.array([51.66,51.77,51.88,51.99,52.09]) if onset else np.array([51.87,52.00,52.13,52.27])
    radius=.12 if onset else .162
    basis=np.maximum(0.,1.-((sec[:,None]-centres[None])/radius)**2)**3
    if tip_basis:
        basis=np.column_stack((basis,np.maximum(0.,1.-((sec-52.33)/.104)**2)**3))
    active=basis.any(1)
    lo,hi=(51.53,52.22) if onset else (51.68,52.44)
    assert sec[active].max()<hi and sec[active].min()>lo
    matrix=kron(csr_matrix(basis),csr_matrix(np.diag([.002,.002,.004])),format='csr')
    mcam=dict(first_control_time=0.,control_dt=s['dt'],image_height=1080,reference_pts=0,time_base_num=1,time_base_den=12800)
    log('start',rows=count,real_relative_pairs=len(t['weight']),source_sha256=s['source_sha256'],parent=parent,
        board_edge_source=str(RIGHT),paired_source=report,rig_fixed=True,scene_fixed=True,
        active_knots=[float(sec[active].min()),float(sec[active].max())],right_gain=right_gain,flow_gain=flow_gain,
        ad7_gain=ad7_gain,tip_gain=tip_gain,tip_basis=tip_basis,focus_flow_gain=focus_flow_gain)
    with n.resident(camera=mcam,controls=c,rigs=rigs,camera_ids=packed['camera'],path_ids=packed['path'],
        stations=packed['station'],pts=packed['pts'],observed=packed['uv'],motion_times=packed['times']) as p:
        p.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.65+1e-6))
        plan=p.configure_linearization(directions=matrix,local_scales=np.empty(0),huber_delta=2.5,**factors)
        gpu=p.configure_solver(lower=np.empty(0),upper=np.empty(0),device_budget_bytes=11*1024**3,
            host_budget_bytes=24*1024**3,initial_lambda_relative=.001,damping_metric='jacobi_image')
        before=p.evaluate_errors()[:count]
        if not np.isfinite(before).all():raise ValueError('invalid direct source projections at seed')
        log('prepared',rows=count,plan=plan,gpu=gpu,before=stats(before))
        def checkpoint():
            z=dict(b);z[NAME+'_controls']=p.export_state()['controls']
            np.testing.assert_array_equal(z[NAME+'_controls'][~active],c[~active]);save(out/'state.npz',**z)
        def update(rec):
            z=rec['summary'];val=z['trial_cost']
            if z['accepted']:checkpoint()
            log('trial' if val is not None and np.isfinite(val) else 'trial_domain_failure',**rec)
            return not (out/'stop').exists()
        end=p.solve(relative_cost_tolerance=1e-6,step_tolerance=1e-6,progress=update,
            numerical_progress=lambda rec:log('factorization_retry',**rec))
        checkpoint();after=p.evaluate_errors()[:count]
        save(out/'used_direct_uv.npz',**q,weight=w,source_sha256=s['source_sha256'])
        save(out/'used_displacement.npz',**t,surface=surface,source_sha256=s['source_sha256'])
        write(out/'report.json',dict(parent=parent,rows=count,source_packs=[str(p) for p in (RIGHT,LEFT,TIPS)],
            source_hashes=[sha(p) for p in (RIGHT,LEFT,TIPS)],relative=report,source_depths='conditional per actual left UV',
            before=stats(before),after=stats(after),solver=end))
        log('finish',rows=count,relative_pairs=len(t['weight']),before=stats(before),after=stats(after),solver=end)


def verify(label):
    from flow_depths import bind

    out=ROOT/label; candidate=f'lane4_side_1118/nayeon_late/{NAME}/{label}'
    s=read(PREPARED/'setup.json')['shots'][NAME]; before=camera_state(PARENT); after=camera_state(candidate)
    n=native_for(PARENT,'nayeon'); pts=before[NAME+'_pts']; times=before[NAME+'_times']; rig=before[NAME+'_rigs']
    assert np.array_equal(rig,after[NAME+'_rigs']) and np.array_equal(after['scene_values'],before['scene_values'])
    assert np.array_equal(pts,after[NAME+'_pts']) and len(np.unique(rig,axis=0))==1
    old=before[NAME+'_controls']; new=after[NAME+'_controls']; rig_one=rig[0]
    anchors=arrays(RIGHT); right_check=anchors['role']=='same_exposure_check'; left=arrays(LEFT)
    left_check=(left['pts']==round(51.94*12800))&np.isin(left['family'],['Ad5','Ad6','Ad7','goal_line_left','board.top','board.bottom'])
    focus=[]
    for q,use in ((anchors,right_check),(left,left_check)):
        q={k:v[use] for k,v in q.items() if v.ndim and len(v)==len(use)}
        if 'normal' not in q:q['normal']=np.repeat([[1.,0.]],len(q['pts']),axis=0)
        focus.append(q)
    a=arrays(TIPS); j=np.flatnonzero((a['pts']==round(52.38*12800))&(a['path']=='d78_material_155/path0')&(a['normal'][:,0]<0))[0]
    k=a['path'][j]; i=a['idx'][j]; tab=ji.native_paths(n,[k],count=1537)[k]
    flow=arrays(FLOW/(NAME+'.npz')); valid=flow['keep']&flow['valid']
    left_index=flow['index'][valid]-flow['lag'][valid];right_index=flow['index'][valid]
    pair=flow['uv'][valid]-flow['left_uv'][valid]; surface=flow['family'][valid]; held=flow['held'][valid]
    absolute_x=flow['left_uv'][valid,0]; focus_end=pts[right_index]/12800.; lag=flow['lag'][valid]
    world,_=bind(n,s,rig,old,times,left_index,flow['left_uv'][valid],surface)
    use=np.isfinite(world).all(1)
    held&=use
    view={}
    for tag,model,c in [('prior_release',BASE,camera_state(BASE)[NAME+'_controls']),('tail_join_lead',PARENT,old),('opposite_fix',candidate,new)]:
        m=camera_state(model); native=native_for(model,'nayeon'); use_rig=m[NAME+'_rigs'][0]
        ans={}
        for q in focus:
            uv,_=positions(model,NAME,q); delta=q['uv']-uv
            e=np.sum(delta*q['normal'],axis=1); normal=np.abs(e)
            for pt in np.unique(q['pts']):
                for fam in np.unique(q['family'][q['pts']==pt]):
                    cut=(q['pts']==pt)&(q['family']==fam)
                    if str(fam)=='Ad7':cut&=q['material']=='source_ad7_blue_end'
                    if str(fam) in ('Ad5','Ad6'):cut&=abs(q['normal'][:,0])>.8
                    if cut.any():
                        ans[f'{pt/12800.:.2f}/{fam}']=dict(rows=int(cut.sum()),p50=float(np.median(normal[cut])),
                            p90=float(np.quantile(normal[cut],.90)),signed=float(np.median(e[cut])))
        out_f4=native.core.project_motion(m[NAME+'_rigs'][i],c,s['first'],s['dt'],1080,times[i],tab[0]); uv=out_f4['uv']
        uv[out_f4['status']!=0]=np.nan; val=extreme(uv,tab[1],-1)
        ans['52.38/F4']=dict(rows=1,p50=float(abs(a['uv'][j,0]-val[1][0])),signed=float(a['uv'][j,0]-val[1][0]))
        projected=[native.core.project_motion(use_rig,c,s['first'],s['dt'],1080,times[index[use]],world[use])
                   for index in (left_index,right_index)]
        visible=(projected[0]['status']==0)&(projected[1]['status']==0)
        v=np.full_like(pair,np.nan); v[use]=projected[1]['uv']-projected[0]['uv']-pair[use]
        vv=np.zeros(len(use),bool);vv[use]=visible
        motion={}
        for bucket,scope in [('fitted_core',(focus_end>=51.70)&(focus_end<=52.44)),
                             ('entering',focus_end<51.70),('leaving',(focus_end>=52.44)&(focus_end<=52.54))]:
            result={}
            for key,mask in [('board_left',(surface==0)&(absolute_x<850)),('board_right',(surface==0)&(absolute_x>1150)),('ice',surface==1)]:
                column={}
                for span in (1,8):
                    g=scope&mask&vv&held&(lag==span)
                    if not g.any():continue
                    x=v[g,0]; y=v[g,1]; distance=np.hypot(x,y)
                    frames=pts[right_index[g]]
                    peak=max([np.quantile(distance[frames==pt],.95) for pt in np.unique(frames)])
                    column[str(span)]=dict(n=int(g.sum()),p95=float(np.quantile(distance,.95)),peak_exposure_p95=float(peak),
                        x_p95=float(np.quantile(abs(x),.95)),y_p95=float(np.quantile(abs(y),.95)),
                        x_signed=float(np.median(x)))
                if column:result[key]=column
            motion[bucket]=result
        halo=np.max(abs(use_rig[37]))/2
        maxima=native.core.motion_derivative_extrema(c,s['first'],s['dt'],[times[0]-halo,times[-1]+halo])['value']
        view[tag]=dict(absolute=ans,held_tracks=motion,maxima=maxima.tolist(),camera_domain=camera_audit(native,m,NAME,s)['valid'])
        print(tag,{key:round(item.get('p90',item['p50']),2) for key,item in ans.items()},flush=True)
    check={}
    world_check=np.array([[0.,0.,0.],[-.36,-.23,0.],[-.35,-.23,.018],[.01,.22,.018],[.21,-.19,0.]])
    for sec in (42.34,49.98,50.80,52.50,52.54,52.70,53.20):
        k=(np.searchsorted(pts,round(sec*12800)));a,b=[n.core.project_motion(rig[k],c,s['first'],s['dt'],1080,times[k],world_check) for c in (old,new)]
        np.testing.assert_array_equal(a['status'],b['status'])
        finite=(a['status']==0)&np.isfinite(a['uv']).all(1)&np.isfinite(b['uv']).all(1)
        check[str(sec)]=float(np.max(abs(a['uv'][finite]-b['uv'][finite]),initial=0.))
        assert check[str(sec)]<=1.e-10
    write(out/'comparison.json',dict(state=candidate,metrics=view,protected_projection_delta=check,
        source_sha256=s['source_sha256'],real_uv_checks=str(RIGHT),early_motion_certificate=str(FLOW/'quality'/(NAME+'.npz'))))


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--label',default='opposite_native')
    ap.add_argument('--right-gain',type=float,default=19.)
    ap.add_argument('--flow-gain',type=float,default=1.)
    ap.add_argument('--ad7-gain',type=float,default=10.)
    ap.add_argument('--tip-gain',type=float,default=10.)
    ap.add_argument('--tip-basis',action='store_true')
    ap.add_argument('--onset',action='store_true')
    ap.add_argument('--focus-flow-gain',type=float,default=1.)
    ap.add_argument('--verify',action='store_true')
    a=ap.parse_args()
    if a.verify:verify(a.label)
    else:fit(a.label,a.right_gain,a.flow_gain,a.ad7_gain,a.tip_gain,a.tip_basis,a.onset,a.focus_flow_gain)
