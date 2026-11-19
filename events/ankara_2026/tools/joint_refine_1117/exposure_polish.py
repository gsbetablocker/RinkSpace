"""Resolve source-supported changes of magnification and fixed mechanical mounts."""
from argparse import ArgumentParser
import numpy as np
from lane_end_evidence import WORK
from fixed_mount_refine import transfer
from repair_motion import project
from review import camera_state, native_for
from shared import OUT, PREPARED, arrays, read, save, write
from source_bundle import flow, native_run, relative


def detail(parent,name,folder):
    b=camera_state(parent);n=native_for(parent,'jin')
    s=read(PREPARED/'setup.json')['shots'][name]
    z=arrays(WORK/'texture'/(name+'.npz'));q=arrays(WORK/'texture/quality'/(name+'.npz'))
    rr=np.flatnonzero(z['valid']&q['keep']&~z['held']&(z['lag']==1))
    idx=z['index'][rr];w=z['world'][rr]
    predicted=project(n,s,b,name,idx,w)-project(n,s,b,name,idx-1,w)
    difference=z['uv'][rr]-z['left_uv'][rr]-predicted
    flags=[]
    for i in np.unique(idx):
        j=idx==i;xy=z['left_uv'][rr[j]];v=difference[j]
        if len(v)<5 or np.ptp(xy[:,0])<140 or not np.isfinite(v).all():continue
        # A pan, tilt and change in magnification produce a coherent local
        # similarity field, including zooms whose vectors have opposite signs.
        u=(xy-np.mean(xy,axis=0))/500
        A=np.zeros((len(v),2,4));A[:,0,0]=1;A[:,1,1]=1
        A[:,0,2]=u[:,0];A[:,1,2]=u[:,1];A[:,0,3]=-u[:,1];A[:,1,3]=u[:,0]
        beta=np.linalg.lstsq(A.reshape(-1,4),v.ravel(),rcond=None)[0]
        fit=A@beta
        signal=np.quantile(np.linalg.norm(fit,axis=1),.70)
        scatter=np.median(np.linalg.norm(v-fit,axis=1))
        if signal>max(1.45,2.5*scatter):flags.append(int(i))
    # A cubic exposure depends on coefficients on both sides of its centre.
    supported=np.unique(np.r_[q['transient'],flags]).astype(int)
    if len(supported):
        supported=np.unique((supported[:,None]+np.arange(-2,3)).clip(0,len(z['pts'])-1))
    save(folder/(name+'.npz'),pts=z['pts'],transient=supported,
         source_similarity_exposures=np.array(flags,int),inherited=q['transient'])
    return flags


def fixed_seeds():
    setup=read(PREPARED/'setup.json')['shots'];parent=relative(WORK/'camera_assembly')
    b=camera_state(parent);native=native_for(parent,'jin');original=dict(b);report={}
    prior=camera_state(relative(WORK/'fixed_seed'));end=[n for n,s in setup.items() if s['program']=='nayeon' and s['camera_index']==2]
    shifted=[]
    for name in end:
        s=setup[name];t=b[name+'_times'];c=b[name+'_controls']
        def centres(rig):return native.core.evaluate_motion_camera(rig,c,s['first'],s['dt'],t)['centre']
        shifted.append(centres(b[name+'_rigs'][0])-centres(prior[name+'_rigs'][0]))
    shift=np.median(np.vstack(shifted),axis=0)
    for name,s in setup.items():
        r=original[name+'_rigs']
        if name in end:
            rig=prior[name+'_rigs'][0].copy();rig[15:18]+=shift
        elif name.startswith('jin') and np.ptp(r,axis=0).max()>1e-9:
            rig=r[len(r)//2].copy()
        elif name=='chen_s20':
            # The last short shot has a weak axis along viewing direction.
            # Keep the pivot on its established line of sight while testing a
            # partial move towards the main-camera peer as a separate solution.
            peer=original['chen_s15_rigs'][0];rig=r[0].copy()
            rig[15:18]=.5*(rig[15:18]+peer[15:18])
        else:continue
        c,record=transfer(native,s,original,name,rig)
        b[name+'_rigs']=np.broadcast_to(rig,r.shape).copy();b[name+'_controls']=c
        report[name]=dict(**record,seed_pivot=rig[15:18].tolist(),reference=parent)
    save(WORK/'physical_seed/state.npz',**b);write(WORK/'physical_seed/transfer.json',report)


def refine(parent,name,group,physical=False):
    folder=WORK/'detail'/group;detail(parent,name,folder)
    opts=['--mode','camera','--shots',name,'--stride','2','--unit-weights','--detail-dir',folder,
          '--station-seed-dir',WORK/'evaluation/camera',*flow(1.35)]
    if physical:opts+=['--rig','--rig-roll','--rig-fields','R0_z','cx','cy','eta','k10']
    if name.startswith('nayeon'):opts+=['--photo-bindings',OUT/'repair/prepared_v2/nayeon_photo_bindings.npz']
    native_run(WORK/group/name,parent,opts)


def compact_end(parent):
    state=camera_state(parent);old=dict(state);s=read(PREPARED/'setup.json')['shots']
    names=[n for n,v in s.items() if v['program']=='nayeon' and v['camera_index']==2]
    native=native_for(parent,'nayeon');design=[];target=[]
    for name in names:
        rig=state[name+'_rigs'][0];t=state[name+'_times']
        assert np.array_equal(rig,state[names[0]+'_rigs'][0])
        z=native.core.evaluate_motion_camera(rig,state[name+'_controls'],s[name]['first'],s[name]['dt'],t)
        rt=z['rotation'].reshape(-1,3,3).transpose(0,2,1)
        connector=np.cross(rig[9:12],rt@rig[12:15]);connector/=np.linalg.norm(connector,axis=1)[:,None]
        pupil=rt[:,:,2]*z['zoom_log_ratio'][:,None]
        design.append(np.concatenate((rt,pupil[:,:,None],(pupil*z['zoom_log_ratio'][:,None])[:,:,None]),axis=2).reshape(-1,5)/len(t)**.5)
        target.append((connector*rig[21]).ravel()/len(t)**.5)
    # Preserve P and the observed camera-centre path as closely as possible
    # when transferring the spacing displacement to the lever/pupil model.
    # Original source data, separately, determine whether this trial survives.
    change,_,rank,sv=np.linalg.lstsq(np.vstack(design),np.concatenate(target),rcond=None)
    report={}
    for name in names:
        r=old[name+'_rigs'];rig=r[0].copy()
        rig[18:21]+=change[:3];rig[21]=0.;rig[22:24]+=change[3:]
        ctrl,record=transfer(native,s[name],old,name,rig)
        state[name+'_controls']=ctrl;state[name+'_rigs']=np.broadcast_to(rig,r.shape).copy()
        report[name]=dict(**record,rank=int(rank),condition=float(sv[0]/sv[-1]),
            seed_lever=rig[18:21].tolist(),seed_spacing=float(rig[21]))
    folder=WORK/'compact_seed';save(folder/'state.npz',**state);write(folder/'transfer.json',report)
    for name in names:
        refine(relative(folder),name,'compact_end')


def lower_radial(parent):
    """Profile a first radial term against the observed sensor/zoom coverage."""
    from shared import prepared_packet_path
    b=camera_state(parent);native=native_for(parent,'okada');setup=read(PREPARED/'setup.json')['shots']
    result={}
    for cid in (1,2):
        shots=[n for n,s in setup.items() if s['program']=='okada' and s['camera_index']==cid]
        normal=np.zeros((3,3));rhs=np.zeros(3)
        for name in shots:
            s=setup[name];r=b[name+'_rigs'][0];t=b[name+'_times'];a=arrays(prepared_packet_path(name))
            cam=native.core.evaluate_motion_camera(r,b[name+'_controls'],s['first'],s['dt'],t)
            j=np.flatnonzero(a['fold']<5);i=a['idx'][j];counts=np.bincount(i,minlength=len(t))
            u=a['uv'][j,0]-cam['cx'][i];v=a['uv'][j,1]-cam['cy'][i]
            rad2=(u/cam['fx'][i])**2+(v/cam['fy'][i])**2
            weight=rad2*np.hypot(u,v)/np.sqrt(counts[i]*len(t))
            X=np.c_[np.ones(len(j)),cam['zoom_log_ratio'][i],np.maximum(0,r[38]-cam['log_f'][i])**2]
            X*=weight[:,None];y=weight*cam['k2'][i]*rad2
            normal+=X.T@X;rhs+=X.T@y
        delta=np.linalg.lstsq(normal,rhs,rcond=1e-11)[0]
        result[str(cid)]=dict(shots=shots,k1_initial_transfer=delta.tolist(),role='initialization on measured sensor coverage')
        for name in shots:
            r=b[name+'_rigs'].copy();r[:,[31,32,39]]+=delta;r[:,[33,34,40]]=0
            b[name+'_rigs']=r
    folder=WORK/'lower_radial_seed';save(folder/'state.npz',**b);write(folder/'transfer.json',result)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['seeds','physical','detail'])
    ap.add_argument('--shots',nargs='+');args=ap.parse_args()
    if args.mode=='seeds':fixed_seeds()
    else:
        for shot in args.shots:
            path=WORK/'physical_seed' if args.mode=='physical' else WORK/'cameras'/shot
            refine(relative(path),shot,'physical' if args.mode=='physical' else 'fine',args.mode=='physical')
