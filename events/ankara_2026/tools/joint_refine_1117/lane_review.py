"""Matched source-pixel comparisons for a complete shot and its measured motion."""
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from lane_end_evidence import BASE, WORK, ENDS
from repair_motion import project, frame_median, stat
from review import ji, camera_state, native_for
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write, programme_state_sha
from solve import stats


@lru_cache(3)
def native(candidate):
    return native_for(candidate,'jin')


def nearest_batch(projected,table,keys,pieces,observed):
    """Same clamped finite polyline distance as review.localize, in float64."""
    result=[]
    lengths=np.array([len(uv) for uv,_ in projected])
    for length in np.unique(lengths):
        ids=np.flatnonzero(lengths==length)
        chunks=[pieces[j] for j in ids]
        indices=np.concatenate(chunks);owner=np.repeat(np.arange(len(ids)),list(map(len,chunks)))
        poly=np.stack([np.where(projected[j][1][:,None]==0,projected[j][0],np.nan) for j in ids])
        tt=np.stack([table[keys[j]][1] for j in ids])
        u=torch.as_tensor(poly,device='cuda');t=torch.as_tensor(tt,device='cuda')
        a=u[:,:-1];d=u[:,1:]-a;norm=(d*d).sum(2)
        valid=torch.isfinite(a).all(2)&torch.isfinite(d).all(2)&(norm>1e-15)
        size=max(256,350000//int(length))
        for lo in range(0,len(indices),size):
            ix=indices[lo:lo+size];g=torch.as_tensor(owner[lo:lo+size],device='cuda')
            offset=torch.as_tensor(observed[ix],device='cuda')[:,None,:]-a[g]
            frac=((offset*d[g]).sum(2)/norm[g].clamp_min(1e-18)).clamp(0.,1.)
            dist=((offset-frac[:,:,None]*d[g])**2).sum(2)
            dist.masked_fill_(~valid[g],float('inf'))
            m,j=dist.min(1);ii=torch.arange(len(ix),device='cuda')
            station=t[g,j]+frac[ii,j]*(t[g,j+1]-t[g,j])
            result.append((ix,m.sqrt().cpu().numpy(),station.cpu().numpy()))
    return result


def pixel_errors(n,b,shot,name,a,selected):
    error=np.full(len(a['pts']),np.nan);station=np.full(len(error),np.nan)
    fixed=a.get('station_fixed',np.zeros(len(error),bool))
    ix=np.flatnonzero(selected&~fixed)
    if len(ix):
        paths,pi=np.unique(a['path'][ix],return_inverse=True)
        npaths=len(paths);packed=a['idx'][ix]*npaths+pi;order=np.argsort(packed,kind='stable')
        ordered=packed[order];cuts=np.r_[0,np.flatnonzero(np.diff(ordered))+1]
        pieces=np.split(ix[order],cuts[1:]);group=ordered[cuts]
        table=ji.native_paths(n,paths,count=257);world={k:v[0] for k,v in table.items()}
        jobs=[(int(g//npaths),paths[g%npaths],b[name+'_times'][g//npaths]) for g in group]
        for lo in range(0,len(jobs),300):
            slab=jobs[lo:lo+300]
            projection=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],shot['first'],shot['dt'],slab,world)
            for rows,e,u in nearest_batch(projection,table,[k for _,k,_ in slab],pieces[lo:lo+300],a['uv']):
                error[rows]=e;station[rows]=u
    ix=np.flatnonzero(selected&fixed)
    if len(ix):
        rigs,cam=np.unique(b[name+'_rigs'][a['idx'][ix]],axis=0,return_inverse=True)
        with n.resident(camera=dict(first_control_time=shot['first'],control_dt=shot['dt'],image_height=1080,
                reference_pts=0,time_base_num=1,time_base_den=12800),controls=b[name+'_controls'],rigs=rigs,
                camera_ids=cam.astype(np.int32),path_ids=a['path'][ix],stations=a['station'][ix],pts=a['pts'][ix],
                observed=a['uv'][ix],motion_times=b[name+'_times'][a['idx'][ix]]) as problem:
            problem.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.65+1e-6))
            error[ix]=problem.evaluate_errors();station[ix]=a['station'][ix]
    return error,station


def evaluate(candidate,name,folder,*,only_ends=False):
    shot=read(PREPARED/'setup.json')['shots'][name]
    b=camera_state(candidate);n=native(candidate);a=arrays(prepared_packet_path(name))
    stamp=programme_state_sha(b,[name]);path=folder/(name+'.npz');report_file=folder/(name+'.json')
    packet_sha=sha(prepared_packet_path(name))
    if path.exists() and report_file.exists():
        done=read(report_file)
        if done['shot_state']==stamp and done.get('packet_sha')==packet_sha and done.get('only_ends')==only_ends:return
    all_rows=np.ones(len(a['pts']),bool)
    add=a['material']=='paired_lane_end_source'
    annulus=a['material']=='real_paired_annulus_edges'
    if only_ends:all_rows=np.isin(a['path'],ENDS)
    old_error=old_st=None
    if candidate==BASE:
        with np.load(OUT/BASE/'full_qa/curve_errors.npz') as old:
            old_error=old[name+'_error'];old_st=old[name+'_station']
        all_rows[:len(old_error)]=False
    e,st=pixel_errors(n,b,shot,name,a,all_rows)
    if old_error is not None:
        e[:len(old_error)]=old_error;st[:len(old_st)]=old_st
    eligible=a['fold']<5
    def summary(use):
        vals=e[use]
        if only_ends:vals=vals[~np.isnan(vals)]
        return stats(vals)
    report=dict(shot_state=stamp,packet_sha=packet_sha,only_ends=only_ends,source=safe_source(shot),
        candidate=candidate,old=summary(eligible&~(add|annulus)),new_annulus=summary(eligible&annulus),
        all=summary(eligible),new_lane=summary(eligible&add),held_lane=summary((a['fold']==6)&add),
        families={f:summary(eligible&(a['family']==f)) for f in np.unique(a['family'][eligible])})
    z=arrays(WORK/'texture'/(name+'.npz'));quality=arrays(WORK/'texture/quality'/(name+'.npz'))
    rows=np.flatnonzero(z['valid']&quality['keep']);idx=z['index'][rows];lag=z['lag'][rows];left=idx-lag
    w=z['world'][rows];delta=project(n,shot,b,name,idx,w)-project(n,shot,b,name,left,w)
    displacement=delta-(z['uv'][rows]-z['left_uv'][rows]);flow=np.linalg.norm(displacement,axis=1)
    pts=z['pts'];replay={};motion={}
    for length in (1,8):
        for held in (True,False):
            key=f'lag{length}_'+('held' if held else 'fit')
            value=frame_median(idx,flow,(z['held'][rows]==held)&(lag==length),len(pts),3)
            replay[key]=value;motion[key]=stat(value)
    report['motion']=motion
    from flow_depths import bind
    h=z['held'][rows]
    idx_h=idx[h];left_h=left[h];length_h=lag[h]
    world,_=bind(n,shot,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],
        left_h,z['left_uv'][rows[h]],np.zeros(len(idx_h),int))
    good=np.isfinite(world).all(1)
    if good.any():
        delta=project(n,shot,b,name,idx_h[good],world[good])-project(n,shot,b,name,left_h[good],world[good])
        measured=z['uv'][rows[h][good]]-z['left_uv'][rows[h][good]]
        flow=np.linalg.norm(delta-measured,axis=1)
    else:flow=np.empty(0)
    report['current_depth_rows']=dict(held=len(good),valid=int(good.sum()))
    for length in (1,8):
        key=f'lag{length}_rebound_held'
        replay[key]=frame_median(idx_h[good],flow,length_h[good]==length,len(pts),3)
        motion[key]=stat(replay[key])
    save(path,error=e,station=st,pts=pts,**replay)
    write(report_file,report)
    print(name,candidate,report['old'].get('p90'),motion['lag1_held'].get('p95'),flush=True)


def safe_source(s):
    return dict(sha256=s['source_sha256'],camera_index=s['camera_index'],time_base=[1,12800])


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--shots',nargs='+')
    ap.add_argument('--only-ends',action='store_true')
    opt=ap.parse_args()
    for shot in opt.shots or read(PREPARED/'setup.json')['shots']:
        evaluate(opt.candidate,shot,opt.output,only_ends=opt.only_ends)
