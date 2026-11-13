"""Screen a saved continuous body and whole-skate fit at native exposure times."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from ..body_skate_fit import Assembly
from ..inputs import atomic_json, digest
from ..programme_motion_followup import bind


def static_assembly(r):
    assembly=Assembly(r.binding)
    L=r.binding.packet()['parameters'].sole_length_m
    # local forward +X, toe and heel on outsole, plus steel front/rear and cuff
    out=r.binding.packet()
    hx,tx=.035*L,.965*L
    floor=np.interp([hx,tx],out['xs'],out['floor'])
    ix=out['template']['part_ranges']['steel_runner']
    vertices=out['template']['vertices'][slice(*ix)]
    rx,x2=np.min(vertices[:,0]),np.max(vertices[:,0])
    return assembly,np.array([[hx,0.,floor[0]],[tx,0.,floor[1]],[rx,0.,vertices[vertices[:,0]==rx,2].min()],
                         [x2,0.,vertices[vertices[:,0]==x2,2].min()],
                         [.28*L,0.,max(out['rise'])*.9+floor[0]]])

def projection_blocks(r,c,stamps,assembly,mats,*,full_uv=False):
    stamp=np.asarray(stamps,int)
    tau=c.readout_times(stamp)
    t=r.times[stamp,None]+tau[:,None]*np.linspace(-.5,.5,5)
    native=fk_state(r,r.at(t.ravel()),np.zeros((t.size,69)))
    rota=native['R']
    origin=native['O']
    mat=origin[:,:,None]+np.einsum('nfij,kj->nfki',rota,mats)
    u=moving_projection(c,mat.reshape(len(stamp),5,-1,3),stamp)['uv'].reshape(len(stamp),2,len(mats),2)
    verts=None
    if full_uv:
        world=np.stack([assembly.points(native,f,transforms=(rota,origin)) for f in range(2)],axis=1)
        verts=moving_projection(c,world.reshape(len(stamp),5,-1,3),stamp)['uv'].reshape(len(stamp),2,-1,2)
    return u,verts

def angles_between(v,w):
    return np.degrees(np.arctan2(v[...,0]*w[...,1]-v[...,1]*w[...,0],np.sum(v*w,axis=-1)))

def silhouette_score(assembly, verts, polygon):
    if not np.isfinite(verts).all() or not np.isfinite(polygon).all():
        return np.nan
    try:
        boundary=np.concatenate(assembly.raster(verts))
    except ValueError:
        return np.nan
    return float(.55*np.median(cKDTree(boundary).query(polygon)[0])+
                 .45*np.median(cKDTree(polygon).query(boundary)[0]))

def main(args):
    cv2.setNumThreads(1)
    r,c,info,doc,cp=bind(args.prepared, args.evidence, args.parameter)
    pose=load(args.prepared/'observations.npz')
    data=load(args.visual)
    clock=data['source_pts'];shots=data['shot_ids']
    np.testing.assert_array_equal(clock,r.p['pts'])
    with np.load(args.result,allow_pickle=False) as base:
        cvuv=base['COCO_uv']
        r0=base['skate_R_world_mid']
        o0=base['skate_origin_world_mid']
        p0=base['P_world_mid']
        q0=base['Q_world_mid']
        s0=base['native22_world_mid']
        g0=base['skate_edge_world_mid'][...,2].min(axis=(2,3))
        boots=base['equipment_uv']
        observed=base['source_observation_use']
        iv=base['source_lower_body_inferred']
        local=base['local_R_mid']
        Q=base['Q_world_mid']
        ratio=float(base['body_length_ratio'])
    t=clock/12800.
    assembly,mats=static_assembly(r)
    known=(data['status']==1)&(data['extent']>0)&~(data['partial_occlusion']|data['identity_unresolved']|data['skate_occluded'])
    point=[]
    contour=np.full((len(t),2),np.nan)
    for a in range(0,len(t),64):
        frames=np.arange(a,min(a+64,len(t)))
        uv,verts=projection_blocks(r,c,frames,assembly,mats,full_uv=True)
        point.append(uv)
        for j,f in np.argwhere(known[frames]):
            contour[frames[j],f]=silhouette_score(assembly,verts[j,f],data['region'][frames[j],f])
    u=np.concatenate(point)
    both=np.isfinite(data['region']).all(axis=(2,3))
    centre=np.nanmedian(data['region'],axis=2)
    dev=np.linalg.norm(u[:,:,:2].mean(axis=2)-centre,axis=-1)
    vec=u[:,:,1]-u[:,:,0]
    centred=data['region']-centre[:,:,None]
    centred=np.where(np.isfinite(centred),centred,0.)
    val,axis=np.linalg.eigh(np.einsum('nfki,nfkj->nfij',centred,centred))
    uaxis=axis[...,1]
    model_angle=angles_between(vec,uaxis)
    ud=np.minimum(abs(model_angle),180.-abs(model_angle))
    width=np.linalg.norm(vec,axis=-1)
    soc=np.degrees(Rotation.from_matrix((r0[:-1].swapaxes(-1,-2)@r0[1:]).reshape(-1,3,3)).magnitude()).reshape(-1,2)
    soc=np.r_[np.zeros((1,2)),soc]
    move=np.r_[np.zeros((1,2)),np.linalg.norm(np.diff(o0,axis=0),axis=2)]
    # Shared-scale result world is already reported in metres at this layer.
    native_endpoint_velocity=np.r_[np.zeros((1,2)),np.linalg.norm(np.diff(u[:,:,:2].mean(axis=2),axis=0),axis=2)]
    direction_step=np.r_[np.zeros((1,2)),abs(angles_between(vec[:-1],vec[1:]))]
    ix=np.arange(len(t))
    same=np.r_[False,shots[1:]==shots[:-1]]
    support=(data['mask_agreement']>=.42)&known
    well=known&(val[...,1]>2.35*np.maximum(val[...,0],1.e-4))&(width>18)
    changes=soc.copy();changes[~same]=np.nan
    rank=np.nan_to_num(contour)*.7+np.where(well,ud,0)*.16+np.where(known,np.minimum(65,dev),0)*.19+np.where(same[:,None],np.minimum(soc,80),0)*.32
    # Peaks separated by half a second in each shot, ensuring different episodes.
    peaks=[]
    for sid in np.unique(shots):
        for f in range(2):
            m=np.flatnonzero(shots==sid)
            for j0 in np.argsort(-rank[m,f]):
                j=int(m[j0])
                if len([v for v in peaks if v['shot']==sid and v['foot']==f])>=8:
                    break
                if any(v['shot']==sid and abs(v['index']-j)<25 and v['foot']==f for v in peaks):
                    continue
                if not known[j,f] and soc[j,f]<40.:
                    continue
                peaks.append(dict(shot=str(sid),index=j,pts=int(clock[j]),time=float(t[j]),foot=f,
                    score=float(rank[j,f]),contour=None if not np.isfinite(contour[j,f]) else float(contour[j,f]),
                    displacement=None if not np.isfinite(dev[j,f]) else float(dev[j,f]),
                    axis=float(ud[j,f]),step=float(soc[j,f]),visible=bool(known[j,f])))
    np.savez_compressed(args.output/'full_saved_trajectory_audit.npz',pts=clock,shots=shots,
        source_region=data['region'],known=known,support=support,foot_uv=u,contour=contour,
        unsig_axis=ud,body= cvuv,dir_step=direction_step,world_step=move,same=same,so3=soc,priority=rank,
        p=p0,q=q0,runner_z=g0,equipment_uv=boots,world_joints=s0)
    opening=[]
    for i in [0,5,12,20,28,38,48,53,58,64,72,78,84,90,96,104,114,130,143,160]:
        used=pose['use'][i]
        opening.append(dict(t=float(t[i]),P=p0[i].round(3).tolist(),
          root_u=q0[i,:,1].round(3).tolist(),torso_u=((s0[i,12]-s0[i,0])/np.linalg.norm(s0[i,12]-s0[i,0])).round(3).tolist(),
          skate_mm=(g0[i]*1000).round(1).tolist(),skate_heel_toe_px=u[i,:,:2].round(1).tolist(),
          inferred=bool(iv[i]),keys=pose['model_joint_ids'][i,used].tolist(),observed=pose['uv'][i,used].round(2).tolist(),
          projected=cvuv[i,pose['model_joint_ids'][i,used]].round(2).tolist()))
    atomic_json(args.output/'saved_audit_summary.json',dict(clock=[int(clock[0]),int(clock[-1])],camera=info['state_sha256'],
                path=str(args.parameter),sha=digest(args.parameter),top=sorted(peaks,key=lambda z:(z['shot'],z['foot'],-z['score'])),opening=opening))
    for q in opening:
        print(q['t'],'P',q['P'],'root',q['root_u'],'torso',q['torso_u'],'blade',q['skate_mm'],'observed',q['keys'],flush=True)
    print('top',[(x['time'],x['foot'],round(x['score'],1)) for x in sorted(peaks,key=lambda z:-z['score'])[:34]],flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prepared", "evidence", "parameter", "result", "visual", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    main(args)
