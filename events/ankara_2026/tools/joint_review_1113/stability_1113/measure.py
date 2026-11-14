"""Compare identical static source tracks against exact native rolling-row projection."""
from camera_audit import HERE, ji, sj, vj, setup, native_scene
import numpy as np


def project(core,native,shot,c,rig,idx,xyz):
    m=len(idx)
    uv=np.full((m,2),np.nan)
    status=np.empty(m,np.int32)
    for start in range(0,m,16000):
        sl=slice(start,start+16000)
        ii=idx[sl]
        x=xyz[sl]
        rigs,cam=np.unique(rig[ii],axis=0,return_inverse=True)
        n=len(ii)
        station=np.arange(n)/max(n-1,1)
        with ji.ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
            options=native.scene['shared_geometry_options'],rig=rigs,controls=c,first_control_time=shot.first,
            control_dt=shot.dt,image_height=1080,codes=np.zeros(n,np.int32),stations=station,
            times=shot.times[ii],observed=np.zeros((n,2)),camera_ids=cam) as r:
            r.configure_geometry([dict(kind=4,vertices=x)])
            r.configure_projection_time_support(half_seconds=max(abs(rigs[:,37]))*.64+1.e-7)
            a=r.evaluate('cuda')
        uv[sl]=a['values'][:,3:5]
        status[sl]=a['status']
    uv[status!=0]=np.nan
    return uv,status


def predictions(core,native,shot,c,rig,tracks):
    ii=tracks['index']
    left=ii-tracks['lag']
    triple=(tracks['lag']==1)&np.isfinite(tracks['prev2_uv']).all(1)
    xyz=tracks['world']
    idx=np.r_[ii,left,ii[triple]-2]
    p,st=project(core,native,shot,c,rig,idx,np.vstack((xyz,xyz,xyz[triple])))
    n=len(ii)
    a,b,g=p[:n],p[n:2*n],p[2*n:]
    observed=tracks['uv']-tracks['left_uv']
    err=(a-b)-observed
    accel=np.full_like(err,np.nan)
    accel[triple]=err[triple]-((b[triple]-g)-(tracks['left_uv'][triple]-tracks['prev2_uv'][triple]))
    return dict(displacement=err,jitter=accel,target=a,left=b,
                model_motion=a-b,source_motion=observed,failures=int(np.count_nonzero(st)))


def frame_errors(idx,err,choose,n,minimum=4):
    """One robust score per source exposure, independent of the dot density."""
    s=np.full(n,np.nan)
    ix=np.flatnonzero(choose&np.isfinite(err))
    if len(ix):
        ix=ix[np.argsort(idx[ix],kind='stable')]
        groups=np.split(ix,np.flatnonzero(np.diff(idx[ix]))+1)
        for g in groups:
            if len(g)>=minimum:
                s[idx[g[0]]]=np.median(err[g])
    return s


def err_norm(p,k):
    return np.linalg.norm(p[k],axis=1)


def stat(v):
    q=v[np.isfinite(v)]
    return {'frames':len(q), **dict(zip(('p50','p90','p95','max'),
            map(float,np.quantile(q,[.5,.9,.95,1.]))))} if len(q) else {'frames':0}


def summarize(shot,t,p):
    n=len(shot.pts)
    family=t['family']
    index=t['index']
    result={}
    per_frame={}
    for mat in ('board','ice'):
        for term,lag in (('adjacent',1),('lag8',8),('jitter',1)):
            val=err_norm(p,'jitter' if term=='jitter' else 'displacement')
            target=(family==int(mat=='ice'))&(t['lag']==lag)
            for fold,fold_mask in (('all',np.ones(len(index),bool)),('held',t['fold']==3)):
                q=frame_errors(index,val,target&fold_mask,n,minimum=4 if mat=='board' else 3)
                name=mat+'_'+term+'_'+fold
                result[name]=stat(q)
                per_frame[name]=q
    return result,per_frame


def main():
    m,core,orig,z=setup()
    native=native_scene(m,core,z)
    report={}
    for path in sorted((HERE/'source_tracks').glob('*.npz')):
        k=path.stem
        d=vj.shot_from_inputs(k,m,orig)
        t=vj.load_checkpoint(path)
        np.testing.assert_array_equal(d.pts,t['pts'])
        use=z if d.program=='jin' else orig
        p=predictions(core,native,d,use[k+'_controls'],use[k+'_rig'],t)
        q,fr=summarize(d,t,p)
        report[k]=q
        folder=HERE/'baseline_motion'
        folder.mkdir(exist_ok=True)
        sj.atomic_arrays(folder/(k+'.npz'),**p,**fr,pts=d.pts)
        print(k,'board',[(label,q[label].get('p90'),q[label]['frames']) for label in
            ('board_adjacent_all','board_lag8_all','board_jitter_all')], 'ice',
            q['ice_adjacent_all'].get('p90'),q['ice_adjacent_all']['frames'],flush=True)
    sj.write_json(HERE/'source_motion_before.json',report)


if __name__=='__main__':
    main()
