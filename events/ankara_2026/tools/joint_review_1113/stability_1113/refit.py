"""Refit Jin's native per-shot trajectory with direct UVs and measured frame transfer."""
from argparse import ArgumentParser
from time import monotonic
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix,kron
from scipy.sparse.linalg import lsqr

from camera_audit import HERE, PREP, ji, sj, vj, setup, native_scene
from measure import predictions
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos


def coefficient_space(d,c,stride):
    chart=UniformMotionChart.create(len(c),d.first,d.dt,stride,d.times[0])
    space=chart.refinement
    shape=space.shape
    tr=continue_exposure_halos(np.broadcast_to(space[:,None,:],(len(c),3,shape[1])),
                               d.first,d.dt,d.times[[0,-1]])[:,0]
    ii,w=cubic_samples(len(c),d.first,d.dt,d.times)
    A=np.einsum('ni,nik->nk',w,tr[ii])
    use=np.flatnonzero(np.max(abs(A),axis=0)>1.e-9)
    space=tr[:,use]
    B=A[:,use]
    y=np.einsum('ni,nic->nc',w,c[ii])
    # Fit only the original exposure centres. Unsupported control padding may
    # contain arbitrary gradients from the earlier per-frame solve.
    bend=np.diff(np.eye(len(use)),n=2,axis=0)
    regularizer=.055*bend
    mat=csr_matrix(np.vstack((B,regularizer)))
    rhs=np.vstack((y,np.zeros((len(bend),3))))
    init=np.stack([lsqr(mat,z,atol=2.e-12,btol=2.e-12,iter_lim=1000)[0] for z in rhs.T],axis=1)
    return space,B,space@init


def rig_seed(d,r,mode,shared):
    if mode=='motion':
        return r.copy()
    med=np.median(r,axis=0)
    scale=np.maximum(np.ptp(r,axis=0),1.e-8)
    representative=r[np.argmin(np.linalg.norm((r-med)/scale,axis=1))].copy()
    if mode=='shared':
        # Same fixed pivot and mechanical offsets as the corresponding Okada
        # installation. Keep Jin's shot gauge and lens branch.
        representative[15:24]=shared[15:24]
    return np.broadcast_to(representative,r.shape).copy()


def track_selection(d,tracks,p,trust):
    ii=tracks['index']
    ma=tracks['family']==0
    quality=tracks['quality']
    errors=np.linalg.norm(p['displacement'],axis=1)
    u=tracks['uv']
    eligible=(tracks['fold']==0)|(tracks['fold']==1)
    eligible&=(quality<16)&np.isfinite(errors)&(errors<40)
    use=np.flatnonzero(eligible)
    use=use[np.lexsort((quality[use],tracks['lag'][use],tracks['family'][use],ii[use]))]
    labels=np.column_stack((ii[use],tracks['family'][use],tracks['lag'][use]))
    rows=[]
    weights=[]
    for q in np.split(use,np.flatnonzero(np.any(np.diff(labels,axis=0)!=0,axis=1))+1):
        if not len(q):
            continue
        # Reject residuals discordant with the local source field. The median
        # shift itself is never excluded just for disagreeing with the prior.
        vec=p['displacement'][q]
        r=np.linalg.norm(vec-np.median(vec,axis=0),axis=1)
        ok=(r<=max(2.0,min(9.,3*np.median(r)+.5)))
        q=q[ok]
        if len(q)<(5 if ma[q[0]] else 3) if len(q) else True:
            continue
        # Sample across image columns so one advert cannot dominate.
        cap=2 if ma[q[0]] else 2
        chosen=[]
        for col in range(0,1920,320):
            z=q[(u[q,0]>=col)&(u[q,0]<col+320)]
            if len(z):
                chosen.extend(z[:cap])
        if len(chosen)<3:
            continue
        lag=tracks['lag'][chosen[0]]
        # At 50 fps, use both the direct increment and the 160 ms drift.
        # The 20 ms residual is smaller; it receives more leverage per px.
        mass=(2.7 if lag==1 else 1.45)*(1. if ma[chosen[0]] else .55)*trust
        weights.extend([mass/np.sqrt(len(chosen))]*len(chosen))
        rows.extend(chosen)
    return np.asarray(rows,np.int64),np.asarray(weights)


def build_problem(native,d,c,r,space,sampled,a,t,baseline,trust):
    track_ids,track_weights=track_selection(d,t,baseline,trust)
    w=t['world'][track_ids]
    ntrack=len(w)
    right=t['index'][track_ids]
    left=right-t['lag'][track_ids]
    xy=np.stack((t['left_uv'][track_ids],t['uv'][track_ids]),axis=1).reshape(-1,2)
    frames=np.stack((left,right),axis=1).ravel()
    held=(t['fold']==2)|(t['fold']==3)
    held_frames=np.zeros(len(d.pts),bool)
    for k in np.unique(t['index'][held]):
        held_frames[k]=True
    # Previously declared named-UV folds stay independent of these source tracks.
    use=(a['fold']==0)&np.isfinite(a['station'])&np.isfinite(a['uv']).all(1)
    if ntrack==0:
        raise ValueError('No independently measured training track')
    ids=np.flatnonzero(use)
    pathnames=sorted(set(a['path'][ids]))
    pathmap={name:i for i,name in enumerate(pathnames)}
    paths=[native.paths[name] for name in pathnames]+[dict(kind=4,vertices=w)]
    station=np.r_[a['station'][ids],np.repeat(np.arange(ntrack)/max(ntrack-1,1),2)]
    code=np.r_[[pathmap[k] for k in a['path'][ids]],np.full(2*ntrack,len(pathnames))]
    observed=np.vstack((a['uv'][ids],xy))
    indices=np.r_[a['idx'][ids],frames]
    family=np.r_[np.where(np.char.startswith(a['family'][ids],'Ad'),'advertisement',
                 np.where(np.char.startswith(a['family'][ids],'board'),'rail','ice')),
                 np.where(t['family'][track_ids]==0,'board_motion','ice_motion')]
    cam,ci=np.unique(r[indices],axis=0,return_inverse=True)
    # Only control directions with several nearby source factors participate.
    nb=np.sum(np.abs(sampled[np.unique(indices)])>.04,axis=0)
    active=nb>=3
    directions=kron(csr_matrix(space[:,active]),csr_matrix(np.diag([.0012,.0012,.003])),format='csr')
    if not directions.shape[1]:
        raise ValueError('No supported camera controls')
    p=ji.ResidentResidualProblem(native.core,board=native.scene['board'],layout=native.scene['shared_layout'],
       options=native.scene['shared_geometry_options'],rig=cam,controls=c,first_control_time=d.first,control_dt=d.dt,
       image_height=1080,codes=code,stations=station,times=d.times[indices],observed=observed,camera_ids=ci)
    p.configure_geometry(paths)
    p.configure_projection_time_support(half_seconds=max(abs(cam[:,37]))*.64+1.e-7)
    return p,directions,ids,family,track_ids,track_weights,indices


def objective(p,n,family,white_w,track_w):
    raw=p.evaluate('cuda')
    rr=raw['values'][:,9:11]
    bad=int(np.count_nonzero(raw['status']))
    e=np.r_[np.linalg.norm(rr[:n],axis=1)*white_w,
            np.linalg.norm(rr[n:].reshape(-1,2,2)[:,1]-rr[n:].reshape(-1,2,2)[:,0],axis=1)*track_w]
    loss=np.where(e<2.7,.5*e**2,2.7*(e-1.35))
    return {k:float(loss[family==k].sum()) for k in np.unique(family)},bad,raw['timings']


def fit(d,native,origin,shared,stride,mode,trust,folder,log):
    key=d.name
    a=vj.load_checkpoint(PREP/(key+'.npz'))
    t=vj.load_checkpoint(HERE/'source_tracks'/(key+'.npz'))
    b=vj.load_checkpoint(HERE/'baseline_motion'/(key+'.npz'))
    c0=origin[key+'_controls'];r0=origin[key+'_rig']
    path=folder/(key+'.npz')
    if path.exists():
        return
    space,B,c=coefficient_space(d,c0,stride)
    r=rig_seed(d,r0,mode,shared)
    p,directions,ids,family,track_ids,w,fi=build_problem(native,d,c,r,space,B,a,t,b,trust)
    with p:
        begin=monotonic()
        n=len(ids)
        nt=len(track_ids)
        error=p.evaluate('cuda')['status']
        active=np.flatnonzero(error[:n]==0)
        if len(active)!=n or np.any(error[n:]!=0):
            raise ValueError(('Initial projection domain',int(n-len(active)),int(np.count_nonzero(error[n:]))))
        white_w=a['weight'][ids]*1.0
        white=np.zeros((n,2,2));white[:,0,0]=white[:,1,1]=white_w
        track_white=np.zeros((nt,2,4))
        track_white[:,0,0]=-w;track_white[:,0,2]=w
        track_white[:,1,1]=-w;track_white[:,1,3]=w
        rows=np.arange(n+2*nt,dtype=np.int32)
        offsets=np.r_[np.arange(n,dtype=np.int32),n+2*np.arange(nt+1,dtype=np.int32)]
        weights=np.r_[white.ravel(),track_white.ravel()]
        weight_offsets=np.r_[4*np.arange(n,dtype=np.int32),4*n+8*np.arange(nt+1,dtype=np.int32)]
        local=np.r_[np.arange(n,dtype=np.int32),np.full(2*nt,-1,np.int32)]
        plan=p.configure_linearization(directions=directions,local_ids=local,
            local_scales=np.full(n,.0015),group_offsets=offsets,members=rows,
            ranks=np.full(n+nt,2,np.int32),weight_offsets=weight_offsets,
            weights=weights,huber_delta=2.7)
        bounds=p._path_supports[p._codes[:n]]
        gpu=p.configure_solver(lower=bounds[:,0],upper=bounds[:,1],
            device_budget_bytes=8*1024**3,host_budget_bytes=8*1024**3,
            initial_lambda_relative=.0008,damping_metric='jacobi_image')
        loss,bad,timing=objective(p,n,family,white_w,w)
        log('prepared',variant=key+'/'+mode,rows=len(rows),named_uv=n,static_track_pairs=nt,
            domain_failures=bad,blocks=plan,gpu=gpu,objective_by_layer=loss,timings=timing,checkpoint=str(path))
        accepted=0
        start=d.pts[0]/12800.
        def write_state():
            state=p.export_state()
            sj.atomic_arrays(path,controls=state['controls'],rig=r,pts=d.pts,scene_values=origin['scene_values'],
                fit_named_rows=ids,fit_track_rows=track_ids)
        def update(record):
            nonlocal accepted,loss,bad,timing
            if record['summary']['accepted']:
                accepted+=1
                write_state()
                loss,bad,timing=objective(p,n,family,white_w,w)
            log('trial',variant=key+'/'+mode,rows=len(rows),objective_by_layer=loss,
                domain_failures=bad,checkpoint=str(path),evaluation_ms=timing,**record)
            return not (folder/'stop.request').exists()
        result=p.solve(gradient_tolerance=1.e-5,step_tolerance=3.e-5,relative_cost_tolerance=1.5e-5,progress=update,
                       numerical_progress=lambda rec:log('factorization_retry',variant=key+'/'+mode,**rec))
        if not accepted:
            write_state()
        log('finish',variant=key+'/'+mode,rows=len(rows),reason=result['stop_reason'],
            accepted=accepted,seconds=monotonic()-begin,source_first_seconds=start,
            solver=result,checkpoint=str(path),objective_by_layer=loss,domain_failures=bad)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('motion','rigid','shared'))
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--trust',type=float,default=1.)
    p.add_argument('--shots',nargs='+')
    p.add_argument('--name')
    a=p.parse_args()
    m,core,hist,state=setup()
    n=native_scene(m,core,state)
    shared={d['camera_index']:hist[key+'_rig'][0] for key,d in m['shots'].items() if key.startswith('okada')}
    out=HERE/(a.name or (a.mode+str(a.stride)))
    out.mkdir(exist_ok=False)
    log=sj.Journal(out/'solve.jsonl')
    jobs=[k for k in m['shots'] if k.startswith('jin') and (a.shots is None or k[4:] in a.shots)]
    order=('jin_s22','jin_s14','jin_s24','jin_s10','jin_s19','jin_s13','jin_s04','jin_s07','jin_s17','jin_s01','jin_s06')
    jobs.sort(key=lambda k:(order.index(k) if k in order else len(order),k))
    for k in jobs:
        if (out/'stop.request').exists():
            break
        d=vj.shot_from_inputs(k,m,hist)
        log('start',variant=k+'/'+a.mode,rows=None,stride=a.stride,source=str(PREP/(k+'.npz')))
        try:
            fit(d,n,state,shared[d.camera_index],a.stride,a.mode,a.trust,out,log)
        except (ValueError,RuntimeError) as e:
            log('failure',variant=k+'/'+a.mode,reason=str(e),checkpoint=str(out/(k+'.npz')))


if __name__=='__main__':
    main()
