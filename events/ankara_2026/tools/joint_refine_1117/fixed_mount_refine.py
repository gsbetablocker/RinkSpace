"""Transfer an exposure-varying rig to a fixed installation before real-image refitting."""
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr
from shared import PREPARED, OUT, arrays, prepared_packet_path, read, save, write
from lane_end_evidence import BASE, WORK
from repair_motion import project
from review import camera_state, native_for, ji
from rinkspace.domain.motion_chart import cubic_samples


def seed_points(native,name,old):
    a=arrays(prepared_packet_path(name))
    with np.load(OUT/BASE/'full_qa/curve_errors.npz') as err:
        st=err[name+'_station'];error=err[name+'_error']
    # Reference projections initialize the transfer only. Native source rows
    # and their full density are consumed separately by the ensuing solver.
    ix=np.flatnonzero((a['fold'][:len(st)]<5)&(error<6.)&np.isfinite(st))
    pixel=a['uv'][ix];frame=a['idx'][ix]
    bins=(pixel[:,0]//255).astype(int)+8*(pixel[:,1]//185).astype(int)
    order=np.lexsort((error[ix],bins,frame));combined=frame[order]*64+bins[order]
    rows=ix[order[np.r_[True,np.diff(combined)!=0]]]
    world=np.empty((len(rows),3));paths=a['path'][rows]
    table=ji.native_paths(native,np.unique(paths),count=513)
    for key in table:
        w,u=table[key];sel=paths==key
        world[sel]=np.column_stack([np.interp(st[rows[sel]],u,w[:,j]) for j in range(3)])
    return a['idx'][rows],world


def transfer(native,s,old,name,rig):
    t=old[name+'_times'];r=old[name+'_rigs'];ctrl=old[name+'_controls'].copy()
    idx,xyz=seed_points(native,name,old)
    target=project(native,s,old,name,idx,xyz)
    ok=np.isfinite(target).all(1);idx=idx[ok];xyz=xyz[ok];target=target[ok]
    n=len(ctrl);steps=np.array([3e-6,3e-6,3e-5])
    def pred(c):
        z=native.core.project_motion(rig,c,s['first'],s['dt'],1080,t[idx],xyz)
        uv=z['uv'].copy();uv[z['status']!=0]=np.nan
        return uv
    base_error=None
    for it in range(6):
        uv=pred(ctrl);res=target-uv
        if base_error is None:base_error=np.linalg.norm(res,axis=1)
        if not np.isfinite(res).all():raise ValueError((name,'invalid projection in seed transfer'))
        jac=np.stack([(pred(ctrl+v)-uv)/steps[j] for j,v in enumerate(np.diag(steps))],axis=-1)
        # Account for different row exposures at opposite sides of a fast pan.
        row_time=t[idx]+rig[37]*(np.clip(uv[:,1],0,1079)/1079.-.5)
        knots,weight=cubic_samples(n,s['first'],s['dt'],row_time)
        scale=np.array([.001,.001,.003]);coeff=jac[:,:,None,:]*weight[:,None,:,None]*scale
        cols=3*knots[:,:,None]+np.arange(3)
        row=np.broadcast_to(np.arange(2*len(idx)).reshape(-1,2,1,1),coeff.shape)
        mat=coo_matrix((coeff.ravel(),(row.ravel(),np.broadcast_to(cols[:,None],coeff.shape).ravel())),
                       shape=(2*len(idx),3*n)).tocsr()
        d=diags([np.ones(3*n-6),-2*np.ones(3*n-6),np.ones(3*n-6)],[0,3,6],shape=(3*n-6,3*n))
        mat=vstack((mat,4*d),format='csr')
        change=lsqr(mat,np.r_[res.ravel(),np.zeros(d.shape[0])],damp=.08,atol=2e-7,btol=2e-7)[0].reshape(n,3)*scale
        ctrl+=change
        if np.max(abs(change))<2e-7:break
    end=np.linalg.norm(pred(ctrl)-target,axis=1)
    return ctrl,dict(reference_points=len(idx),before_px=np.quantile(base_error,[.5,.9,.99]).tolist(),
                    after_px=np.quantile(end,[.5,.9,.99]).tolist(),iterations=it+1)


def create():
    parent='lane_motion_1118/end_geometry';state=camera_state(parent);old=dict(state)
    setup=read(PREPARED/'setup.json')['shots'];native=native_for(parent,'jin');changes={}
    nominal={cam:state[n+'_rigs'][0].copy() for cam,n in enumerate(['jin_s05','jin_s02','jin_s01'])}
    # Select by assigned installation even if the first suggested shot belongs
    # to another rig. Prefer an originally fixed, common shot in the pool.
    for cam in nominal:
        names=[n for n,s in setup.items() if s['program']=='jin' and s['camera_index']==cam and
               np.all(state[n+'_rigs']==state[n+'_rigs'][0])]
        piv=np.array([state[n+'_rigs'][0,15:18] for n in names])
        chosen=names[np.argmin(np.sum((piv-np.median(piv,axis=0))**2,axis=1))]
        nominal[cam]=state[chosen+'_rigs'][0].copy()
    for name,s in setup.items():
        r=old[name+'_rigs'];cam=s['camera_index'];program=s['program'];rig=r[len(r)//2].copy()
        if program=='jin' and np.any(np.ptp(r,axis=0)>1e-9):
            rig[15:24]=nominal[cam][15:24]
            reason='within-shot drift in mechanical installation'
        elif name=='chen_s20':
            peer=old['chen_s15_rigs'][0];rig[15:24]=peer[15:24]
            reason='same assigned main mount, baseline S20 pivot is more than .03L from other main shots'
        elif program=='nayeon' and cam==2:
            rig[18:24]=nominal[cam][18:24]
            reason='large lever and axis-spacing cancellation; same-installation working-hardware seed'
        else:continue
        ctrl,info=transfer(native,s,old,name,rig)
        changes[name]=dict(reason=reason,**info,original_rigs=len(np.unique(r,axis=0)),seed_rigs=1,
            old_pivot_range=np.ptp(r[:,15:18],axis=0).tolist(),fixed_pivot=rig[15:18].tolist())
        state[name+'_rigs']=np.broadcast_to(rig,r.shape).copy();state[name+'_controls']=ctrl
        print(name,info,flush=True)
    dest=WORK/'fixed_seed';save(dest/'state.npz',**state);write(dest/'transfer.json',changes)


if __name__=='__main__':create()
