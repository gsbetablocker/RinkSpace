"""Repair the limb branch in the flying entry with source knee/ankle and rigid-skate motion."""
from pathlib import Path
import json
from argparse import ArgumentParser
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from experiments.skate_video_v1.inputs import atomic_json, read, digest
from calibration.program_tracking.body_tracking.absolute_motion_curve import greville_times
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from ..programme_motion_followup import bind
from .source_audit import static_assembly
from .scoped_fit import Fit, tn, ti, save_candidate, smooth
from calibration.program_tracking.body_tracking.programme_opening import standing_contact_rows
from .direction_fit import authored_limits


def fit_entry(ref,camera,curve,controls,source,episode):
    rows=[d for d in source['rows'] if d['episode']==episode]
    rows.sort(key=lambda row:row['pts'])
    pts=ref.p['pts']
    anchor=np.searchsorted(pts,[d['pts'] for d in rows])
    np.testing.assert_array_equal(pts[anchor],[d['pts'] for d in rows])
    seconds=pts/12800
    times=ref.times
    gm=greville_times(curve.knots)
    centre=(anchor.min(),anchor.max())
    feet=sorted({q['foot'] for q in rows})
    inside=np.zeros(len(times),bool)
    inside[centre[0]:centre[1]+1]=True
    interval=times[anchor[[0,-1]]]
    pad=.17 if episode==0 else .185
    control=np.flatnonzero((gm>=interval[0]-pad)&(gm<=interval[1]+pad))
    frames=np.flatnonzero((times>=curve.knots[control[0]]-.03)&(times<=curve.knots[control[-1]+4]+.03))
    f=Fit(ref,camera,curve,frames,[track+foot for foot in feet for track in (1,4,7,10)],control,fit_position=False)
    located=np.searchsorted(frames,anchor)
    feature=np.array([('knee','ankle','assembly_centre').index(row['kind'])+3*row['foot'] for row in rows])
    target=np.array([row['uv'] for row in rows])
    sigma=np.array([row['sigma_px'] for row in rows])
    train=np.array([row['role']=='fit' for row in rows])
    ij=(ti(located[train]),ti(feature[train]))
    expected=tn(target[train])
    ss=tn(sigma[train])
    margin=tn(1.-smooth(interval[0]-.18,interval[0],times[frames])*(1.-smooth(interval[1],interval[1]+.18,times[frames])))
    a,m=static_assembly(ref)
    cuff=np.array([.39*(m[0,0]+m[1,0]),0.,.35*m[-1,2]+.65*m[1,2]])

    def compute(state,boots):
        j=state[5][:,[4,7,5,8]]
        shoe=boots.mean(dim=2)*.52+f.material_world(state)[:,:,4]*.48
        w=torch.stack([j[:,0],j[:,1],shoe[:,0],j[:,2],j[:,3],shoe[:,1]],dim=1)
        return w
    original=f.state()
    w0=f.material_world(original).detach()[2::5]
    direct0=compute(original,f.material_world(original)).detach()
    initp=f.project(direct0).detach()
    height0=f.contact_heights(original).detach()
    local0=original[2][2::5].detach()
    limits=[authored_limits(f,frames,foot,controls) for foot in feet]
    legs=[np.array([0,3,6])+foot for foot in feet]
    nominal=[p(local0[:,k]).detach() for (p,_),k in zip(limits,legs)]
    points=ti([3*foot+2 for foot in feet])
    event_mask=tn(inside[frames])

    def cost(state):
        assembly=f.material_world(state)
        node=compute(state,assembly)
        uv=f.project(node)
        errs=(uv[ij]-expected)/ss[:,None]
        source_loss=(torch.sqrt(1.+errs.square())-1.).mean()*16.
        # Penalize changes in material velocity, while observing the true
        # fast sweep. This does not force stationarity or insert extra contact.
        material=assembly[2::5][:,feet]
        rotation=state[6][2::5][:,feet]
        accel=torch.diff(material,n=2,dim=0)
        steer=torch.diff(rotation,n=2,dim=0)
        smoothness=(accel/.035).square().mean()*5.5+(steer/.17).square().mean()*.65
        diff=uv[:,points]-initp[:,points]
        bridge=((diff/21.).square().mean(dim=(1,2))*margin).mean()*4.
        outside=(((material-w0[:,feet])/.15).square().mean(dim=(1,2,3))*margin).mean()*1.35
        local=state[2][2::5]
        act=f.active-1
        pose=((local[:,act]-local0[:,act]).square().mean(dim=(1,2,3))*(.40+margin)).mean()
        anatomy=source_loss*0.
        for k,(trial,_),prev in zip(legs,limits,nominal):
            val=trial(local[:,k])
            anatomy+=8.*val.square().mean()+20.*torch.relu(val-prev-.13).square().mean()
            anatomy+=40.*torch.relu(val[:,:,:2]-prev[:,:,:2]).square().sum(dim=1).amax()
        height=f.contact_heights(state)
        penetr=(torch.relu(-height[:,feet]-.016)/.045).square().mean()*2.
        return dict(source_legs_and_whole_skates=source_loss,material_acceleration=smoothness,
                    boundaries=bridge+outside,shaped_joint_envelope=anatomy,soft_pose=pose,clearance=penetr)
    key="80.90–81.10 右腿起摆" if episode==0 else "81.30–81.52 双腿落位"
    trial=f.run(cost,50,key)
    def evaluation(current):
        from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
        state_ref=absolute_reference(ref,current)
        clock=(times[frames,None]+camera.readout_times(frames)[:,None]*np.linspace(-.5,.5,5)).ravel()
        state=fk_state(state_ref,state_ref.at(clock),np.zeros((len(clock),69)))
        boot=state['O'][:,:,None]+np.einsum('nfij,kj->nfki',state['R'],m)
        shoe=boot.mean(axis=2)*.52+boot[:,:,4]*.48
        j=state['native_world']
        world=np.stack((j[:,4],j[:,7],shoe[:,0],j[:,5],j[:,8],shoe[:,1]),axis=1)
        reproj=moving_projection(camera,world.reshape(len(frames),5,6,3),frames)['uv']
        model=state['local_R'][2::5]
        anatomy=[exact(model[:,j]).max(0).tolist() for j,(_,exact) in zip(legs,limits)]
        points=boot[2::5][:,feet,2:4]
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=-1).max(axis=2)
        accel=np.linalg.norm(np.diff(points,axis=0,n=2),axis=-1).max(axis=2)
        rr=state['R'][2::5][:,feet]
        rot=np.degrees(Rotation.from_matrix((rr[:-1].swapaxes(-1,-2)@rr[1:]).reshape(-1,3,3)).magnitude()).reshape(-1,len(feet))
        return dict(uv=reproj[located,feature],max_step_mm=(1000*ref.binding.ratio*lengths.max(0)).tolist(),
                    max_second_difference_mm=(1000*ref.binding.ratio*accel.max(0)).tolist(),max_rotation_deg=rot.max(0).tolist(),
                    joint_violations=anatomy)
    before,after=map(evaluation,(curve,trial))
    distance=np.stack([np.linalg.norm(a['uv']-target,axis=-1) for a in (before,after)])
    rms=np.sqrt((distance[:,~train]**2).mean(axis=1))
    section=np.array([row['kind']=='assembly_centre' for row in rows])
    gate=dict(held_source=bool(rms[1]<rms[0]*.86),
              fitted_source=bool(distance[1,train].mean()<distance[0,train].mean()*.85),
              motion=bool(max(after['max_second_difference_mm'])<max(before['max_second_difference_mm'])*.95),
              anatomy=bool(np.max(np.asarray(after['joint_violations'])-np.asarray(before['joint_violations']))<.38))
    result=dict(window=key,state="retained" if all(gate.values()) else "unresolved", gates=gate,
                pts=pts[frames[[0,-1]]].tolist(),feet=feet,
                observations=rows,checked_rms_px=rms.tolist(),image_distance=distance.tolist(),
                assembly_centre_held_px=distance[:,~train & section].mean(1).tolist(),
                states=[{k:v for k,v in r.items() if k!='uv'} for r in (before,after)])
    return (trial if all(gate.values()) else curve),result


def main(args):
    torch.set_num_threads(4)
    start, out, source = args.parameter, args.output, args.source
    out.mkdir(exist_ok=False)
    r,c,info,_,controls=bind(args.prepared, args.evidence, start)
    validate_checkpoint_identity(controls,r.fixed_checkpoint_identity)
    initial=r.absolute_curve
    doc=read(source)
    curve=initial
    rows=[]
    for episode in range(2):
        curve,result=fit_entry(r,c,curve,controls,doc,episode)
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('observations','image_distance')},ensure_ascii=False),flush=True)
        atomic_json(out/'progress.json',dict(rows=rows))
    save_candidate(load(args.parent),curve,out/'candidate.npz',parent=args.parent)
    atomic_json(out/'assessment.json',dict(windows=rows,parent_parameter_sha256=digest(start),
                result_sha256=digest(out/'candidate.npz'),source=str(source)))
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prepared", "evidence", "parameter", "parent", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
