"""Bounded native camera corrections, preserving real irregular source motion."""
from argparse import ArgumentParser
from time import monotonic
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix,kron
from camera_audit import HERE, PREP, ji, sj, vj, setup, native_scene
from measure import project
from refit import track_selection,rig_seed
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples

# Relative to the original timecode. Window edges taper the correction to zero.
ZONES={
's01':[(32.70,33.02)],
's03':[(38.72,38.96),(39.64,40.02)],
's04':[(42.18,42.68),(43.04,43.45),(46.98,47.42)],
's06':[(52.68,57.46)],
's07':[(60.54,60.99),(62.85,63.37)],
's09':[(72.70,73.07),(76.68,78.24)],
's10':[(78.26,78.70),(83.24,85.36),(88.78,89.14)],
's12':[(104.35,105.05)],
's13':[(105.38,106.15),(112.16,113.74)],
's14':[(120.65,122.53),(122.62,123.17),(123.22,123.98)],
's15':[(124.40,124.94)],
's16':[(127.62,128.30),(131.80,132.12)],
's17':[(133.02,133.80),(135.08,135.51)],
's18':[(136.88,137.96),(140.15,140.56)],
's19':[(142.82,143.60)],
's21':[(147.02,148.20),(148.85,149.08)],
's22':[(149.14,150.04),(151.40,151.92),(152.94,155.08)],
's24':[(171.28,171.92),(172.16,178.68),(183.02,183.54),(194.48,195.24)],
}


def taper(shot,seconds,edge=.16):
    gain=np.zeros_like(seconds)
    begin,end=shot.pts[[0,-1]]/12800.
    for lo,hi in ZONES.get(shot.id,[]):
        a=np.ones_like(seconds) if lo<=begin+1.e-8 else np.clip((seconds-lo)/edge,0.,1.)
        b=np.ones_like(seconds) if hi>=end-1.e-8 else np.clip((hi-seconds)/edge,0.,1.)
        gain=np.maximum(gain,np.sin(np.pi/2*np.minimum(a,b))**2)
    return gain


def quadratic_halo(c,d):
    """Continue each exposure boundary with matched position, slope and curvature."""
    out=c.copy()
    a,b=np.rint((d.times[[0,-1]]-d.first)/d.dt).astype(int)
    for i in range(a-2,-1,-1):
        out[i]=3*out[i+1]-3*out[i+2]+out[i+3]
    for i in range(b+2,len(c)):
        out[i]=3*out[i-1]-3*out[i-2]+out[i-3]
    return out


def virtual_anchors(core,native,d,c,r):
    # Frozen-camera projected samples are an explicit trust prior only.
    board=native.scene['board']
    s=np.linspace(0.,core.board_perimeter(board),129)[:-1]
    pts=[]
    for frac in (.15,.83):
        pts.extend(core.board_points(board,s,np.full(len(s),board[5]*frac)))
    for x in np.linspace(-.44,.44,15):
        for y in np.linspace(-float(board[1])*.86,float(board[1])*.86,10):
            pts.append([x,y,0.])
    w=np.array(pts)
    ii=np.arange(len(d.pts))
    j=np.repeat(ii,len(w))
    ww=np.tile(w,(len(ii),1))
    uv,st=project(core,native,d,c,r,j,ww)
    inside=(st==0)&(uv[:,0]>32)&(uv[:,0]<1888)&(uv[:,1]>38)&(uv[:,1]<1042)
    keep=[]
    for i in ii:
        ib=np.flatnonzero(inside[i*len(w):(i+1)*len(w)])+i*len(w)
        if not len(ib):
            continue
        bins=(uv[ib,0]//240).astype(int)+(uv[ib,1]//150).astype(int)*8
        for b in np.unique(bins):
            q=ib[bins==b]
            if len(q)<=2:
                keep.extend(q)
            else:
                keep.extend(q[np.rint(np.linspace(0,len(q)-1,2)).astype(int)])
    keep=np.asarray(keep,dtype=np.int64)
    return j[keep],ww[keep],uv[keep]


def fit(d,native,origin,shared,mode,folder,log):
    begin=monotonic()
    key=d.name
    c0=origin[key+'_controls'];r0=origin[key+'_rig']
    r=rig_seed(d,r0,mode if mode!='halo' else 'motion',shared)
    path=folder/(key+'.npz')
    if mode=='halo':
        c=quadratic_halo(c0,d)
        sj.atomic_arrays(path,controls=c,rig=r,pts=d.pts,scene_values=origin['scene_values'])
        return
    a=vj.load_checkpoint(PREP/(key+'.npz'))
    t=vj.load_checkpoint(HERE/'source_tracks'/(key+'.npz'))
    base=vj.load_checkpoint(HERE/'baseline_motion'/(key+'.npz'))
    core=native.core
    sec=d.pts/12800.
    dt=d.dt
    knot_sec=d.first+dt*np.arange(len(c0))-d.times[0]+sec[0]
    mask=taper(d,knot_sec)
    by_frame=taper(d,sec)
    c=quadratic_halo(c0,d)
    if mode=='motion':
        # Remove excess control curvature only in the identified bands before
        # the original-pixel fit. Preserve verified timing elsewhere.
        smooth=gaussian_filter1d(c,1.3,axis=0,mode='nearest')
        c=quadratic_halo(c+.58*mask[:,None]*(smooth-c),d)
    if mode!='motion':
        mask=np.ones_like(mask)
    chart=UniformMotionChart.create(len(c),d.first,dt,2,d.times[0])
    raw_space=chart.refinement*mask[:,None]
    space=quadratic_halo(raw_space,d)
    ii,w=cubic_samples(len(c),d.first,dt,d.times)
    matrix=np.einsum('ni,nik->nk',w,space[ii])
    j,x,uv=virtual_anchors(core,native,d,c0,r0)
    if mode=='motion':
        keep=by_frame[j]>.001
        j,x,uv=j[keep],x[keep],uv[keep]
    all_named=a['fold']==0
    if mode=='motion':
        all_named&=by_frame[a['idx']]>.001
    ni=np.flatnonzero(all_named)
    # Leave all historical UV test frames alone. Retain a second spatial
    # fold independent of correction in regions without usable temporal folds.
    track,wt=track_selection(d,t,base,1.)
    hold_x=(t['anchor_uv'][track,0]//56).astype(int)%4==2
    tr=~hold_x
    if mode=='motion':
        tr&=(by_frame[t['index'][track]]+by_frame[t['index'][track]-t['lag'][track]])>.001
    track,wt=track[tr],wt[tr]
    zz=t['world'][track]
    tr_frames=np.c_[t['index'][track]-t['lag'][track],t['index'][track]].ravel()
    tr_uv=np.stack((t['left_uv'][track],t['uv'][track]),axis=1).reshape(-1,2)
    names=sorted(set(a['path'][ni]))
    paths=[native.paths[name] for name in names]
    ix={name:k for k,name in enumerate(names)}
    codes=np.r_[[ix[k] for k in a['path'][ni]],np.full(2*len(track)+len(j),len(paths))]
    stations=np.r_[a['station'][ni],np.repeat(np.arange(len(track)),2),len(track)+np.arange(len(j))]
    stations[len(ni):]/=max(len(track)+len(j)-1,1)
    paths.append(dict(kind=4,vertices=np.vstack((zz,x))))
    indices=np.r_[a['idx'][ni],tr_frames,j]
    observed=np.vstack((a['uv'][ni],tr_uv,uv))
    if not len(j):
        raise ValueError('No applicable native trust anchors')
    cam,cam_id=np.unique(r[indices],axis=0,return_inverse=True)
    D=np.flatnonzero(np.count_nonzero(abs(matrix[np.unique(indices)])>.008,axis=0)>=2)
    directions=kron(csr_matrix(space[:,D]),csr_matrix(np.diag([.0009,.0009,.002])),format='csr')
    named_w=a['weight'][ni]*.90
    # Keep both near ice and far rail in the prior. It suppresses drift that
    # image motion alone cannot constrain. It carries less mass at verified jumps.
    count=np.bincount(j,minlength=len(sec))
    pw=(3.2-1.8*by_frame[j])/np.sqrt(count[j]) if mode=='motion' else (4.1-1.5*by_frame[j])/np.sqrt(count[j])
    n=len(ni);nt=len(track);na=len(j)
    p=ji.ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
       options=native.scene['shared_geometry_options'],rig=cam,controls=c,first_control_time=d.first,control_dt=dt,
       image_height=1080,codes=codes,stations=stations,times=d.times[indices],observed=observed,camera_ids=cam_id)
    with p:
        p.configure_geometry(paths)
        p.configure_projection_time_support(half_seconds=max(abs(cam[:,37]))*.64+1.e-7)
        status=p.evaluate('cuda')['status']
        valid_named=status[:n]==0
        valid_track=(status[n:n+2*nt].reshape(-1,2)==0).all(1)
        valid_prior=status[n+2*nt:]==0
        # A candidate unable to project a saved real point gets no approval
        # by deletion. Solver rows of zero confidence are tracked separately.
        if not (valid_named.all() and valid_track.all() and valid_prior.all()):
            bad=np.flatnonzero(status)
            log('domain_failure',variant=key+'/'+mode,rows=len(status),indices=indices[bad].tolist(),
                native_status=status[bad].tolist(),named_failed=int((~valid_named).sum()),
                tracks_failed=int((~valid_track).sum()),trust_failed=int((~valid_prior).sum()),checkpoint=str(path))
            raise ValueError('Seed outside native projection domain')
        white=np.zeros((n,2,2));white[:,0,0]=white[:,1,1]=named_w
        tw=np.zeros((nt,2,4));tw[:,0,0]=-wt;tw[:,0,2]=wt;tw[:,1,1]=-wt;tw[:,1,3]=wt
        pr=np.zeros((na,2,2));pr[:,0,0]=pr[:,1,1]=pw
        group=np.r_[np.arange(n,dtype=np.int32),n+2*np.arange(nt,dtype=np.int32),n+2*nt+np.arange(na+1,dtype=np.int32)]
        weight_offset=np.r_[4*np.arange(n,dtype=np.int32),4*n+8*np.arange(nt,dtype=np.int32),
                           4*n+8*nt+4*np.arange(na+1,dtype=np.int32)]
        rows=np.arange(n+2*nt+na,dtype=np.int32)
        local=np.r_[np.arange(n,dtype=np.int32),np.full(2*nt+na,-1,np.int32)]
        plan=p.configure_linearization(directions=directions,local_ids=local,local_scales=np.full(n,.0015),
            group_offsets=group,members=rows,ranks=np.full(n+nt+na,2,np.int32),
            weight_offsets=weight_offset,weights=np.r_[white.ravel(),tw.ravel(),pr.ravel()],huber_delta=2.7)
        bounds=p._path_supports[p._codes[:n]]
        gpu=p.configure_solver(lower=bounds[:,0],upper=bounds[:,1],
            device_budget_bytes=8*1024**3,host_budget_bytes=8*1024**3,initial_lambda_relative=.001,
            damping_metric='jacobi_image')
        def summary():
            res=p.evaluate('cuda')
            uv=res['values'][:,9:11]
            a0=np.linalg.norm(uv[:n],axis=1)*named_w
            du=uv[n:n+2*nt].reshape(-1,2,2)
            b0=np.linalg.norm(du[:,1]-du[:,0],axis=1)*wt
            c0=np.linalg.norm(uv[n+2*nt:],axis=1)*pw
            def huber(y):
                return np.where(y<2.7,.5*y*y,2.7*(y-1.35))
            mat=np.where(np.char.startswith(a['family'][ni],'Ad'),'ad',
                np.where(np.char.startswith(a['family'][ni],'board'),'rail','ice'))
            return dict(**{x:float(huber(a0[mat==x]).sum()) for x in ('rail','ad','ice')},
                        board_motion=float(huber(b0[t['family'][track]==0]).sum()),
                        ice_motion=float(huber(b0[t['family'][track]==1]).sum()),
                        trust_prior=float(huber(c0).sum())),int(np.count_nonzero(res['status'])),res['timings']
        obj,bad,timing=summary()
        log('prepared',variant=key+'/'+mode,rows=len(rows),real_uv=n,track_pairs=nt,trust_rows=na,
            objective_by_layer=obj,domain_failures=bad,block_counts=plan,gpu=gpu,device_ms=timing,checkpoint=str(path))
        def save():
            s=p.export_state()
            sj.atomic_arrays(path,controls=s['controls'],rig=r,pts=d.pts,
                             scene_values=origin['scene_values'],fit_named_rows=ni,fit_track_rows=track)
        count=0
        def record(data):
            nonlocal count,obj,bad,timing
            if data['summary']['accepted']:
                count+=1
                save()
                obj,bad,timing=summary()
            log('trial',variant=key+'/'+mode,rows=len(rows),objective_by_layer=obj,domain_failures=bad,
                evaluation_ms=timing,checkpoint=str(path),**data)
            return not (folder/'stop.request').exists()
        result=p.solve(gradient_tolerance=3.e-5,step_tolerance=1.e-5,relative_cost_tolerance=6.e-5,progress=record,
                       numerical_progress=lambda rec:log('factorization_retry',variant=key+'/'+mode,**rec))
        if not count:
            save()
        log('finish',variant=key+'/'+mode,rows=len(rows),reason=result['stop_reason'],
            accepted=count,seconds=monotonic()-begin,objective_by_layer=obj,domain_failures=bad,solver=result,
            checkpoint=str(path))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('motion','rigid','shared','halo'))
    p.add_argument('--shots',nargs='+')
    p.add_argument('--name')
    a=p.parse_args()
    m,core,hist,state=setup()
    n=native_scene(m,core,state)
    shared={d['camera_index']:hist[k+'_rig'][0] for k,d in m['shots'].items() if k.startswith('okada')}
    out=HERE/a.name
    out.mkdir(exist_ok=False)
    log=sj.Journal(out/'solve.jsonl')
    shots=a.shots or (sorted(ZONES) if a.mode=='motion' else [f's{i:02d}' for i in range(1,25)])
    for sid in shots:
        k='jin_'+sid
        if (out/'stop.request').exists():break
        d=vj.shot_from_inputs(k,m,hist)
        log('start',variant=k+'/'+a.mode,rows=None,source=str(PREP/(k+'.npz')))
        try:
            fit(d,n,state,shared[d.camera_index],a.mode,out,log)
        except (ValueError,RuntimeError) as e:
            log('failure',variant=k+'/'+a.mode,reason=str(e),checkpoint=str(out/(k+'.npz')))


if __name__=='__main__':
    main()
