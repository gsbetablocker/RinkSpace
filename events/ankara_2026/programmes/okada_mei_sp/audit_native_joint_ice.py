"""Read-only source-identity candidate audit, never a solver loss or weight."""
import collections,json
import numpy as np
import prepare_native_joint_shots as p
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline

def main():
    r=p.RUN;m=p.read(r/'joint_manifest.json');z=p.npz(r/'point_observations.npz');v=p.npz(r/'s04_board_initialize/accepted_pixels_latest.npz')
    tl=ShotMotionTimeline(m['shots'],m['control_dt']);mp=p.read(r/'mapping.json');paths=np.array(mp['path_ids'])[z['path_indices']]
    core=p.NativeCore(m['native_library']);model=p.SceneParameterModel(p.read(p.prep.SCENE),p.prep.SCENE,optimize_scene=True)
    native=p.NativeSceneGPU(p.prep.SCENE,core,parameter_state=p.ParameterState(model.initial_state.layout,v['scene_values']))
    camera={**p.read(p.prep.CAMERA),'rig':v['rig'].tolist(),'first_control_time':tl.first,'control_dt':tl.dt}
    names=[k for k,a in native.paths.items() if a['layer']=='ice'];ix=np.flatnonzero((z['pts']>=745984)&np.array([native.paths[k]['layer']=='ice' for k in paths]))
    pts=np.unique(z['pts'][ix]);sample=np.linspace(0,1,257);nm=len(names);ns=len(sample);distance=np.full((len(ix),nm),np.inf);stations=np.zeros_like(distance)
    for begin in range(0,len(pts),24):
        pp=pts[begin:begin+24];codes=[k for pt in pp for k in names for j in sample];ss=np.tile(sample,len(pp)*nm);pt=np.repeat(pp,nm*ns)
        s=m['shots'][1];times=(pt-s['reference_pts'])/12800+tl.offset(s)
        with native.resident(camera=camera,controls=v['controls'],path_ids=codes,stations=ss,pts=pt,observed=np.zeros((len(pt),2)),motion_times=times) as rr:
            rr.configure_projection_time_support(half_seconds=m['max_readout']/2);ev=rr.evaluate('cuda')
        uv=ev['values'][:,3:5].reshape(len(pp),nm,ns,2);valid=ev['status'].reshape(len(pp),nm,ns)==0
        for k,pt0 in enumerate(pp):
            rows=np.flatnonzero(z['pts'][ix]==pt0);obs=z['uv'][ix[rows]]
            for j in range(nm):
                ok=valid[k,j,:-1]&valid[k,j,1:]
                if not ok.any():continue
                left=uv[k,j,:-1][ok];delta=np.diff(uv[k,j],axis=0)[ok];length=np.sum(delta*delta,axis=1);d=obs[:,None]-left[None]
                u=np.clip(np.einsum('ijk,jk->ij',d,delta)/np.maximum(length,1e-300),0,1);ds=np.sum((d-u[:,:,None]*delta)**2,axis=2);best=ds.argmin(axis=1);ii=np.arange(len(rows))
                distance[rows,j]=np.sqrt(ds[ii,best]);stations[rows,j]=sample[:-1][ok][best]+u[ii,best]/(ns-1)
    native_index={p:i for i,p in enumerate(names)};original=np.array([native_index[k] for k in paths[ix]]);nearest=distance.argmin(axis=1)
    np.savez(r/'review/ice_identity_candidates.npz',rows=ix,path_ids=np.array(names),distance=distance,stations=stations,original=original,nearest=nearest)
    groups=collections.defaultdict(list)
    for i in range(len(ix)):
        if distance[i,original[i]]>50 and distance[i,nearest[i]]<10 and nearest[i]!=original[i]:groups[(names[original[i]],names[nearest[i]])].append(i)
    report=[]
    for (old,new),indices in sorted(groups.items(),key=lambda kv:-len(kv[1])):
        rows=ix[indices];report.append(dict(old=old,new=new,n=len(indices),first=float(z['pts'][rows].min()/12800),last=float(z['pts'][rows].max()/12800),
            rows=rows.tolist(),candidate_only=True))
    p.write(r/'review/ice_identity_candidate_groups.json',report)
    print(json.dumps([{k:v for k,v in g.items() if k!='rows'} for g in report[:30]]),flush=True)

if __name__=='__main__':main()
