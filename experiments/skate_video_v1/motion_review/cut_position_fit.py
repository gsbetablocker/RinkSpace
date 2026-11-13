"""Match one end-on episode using projected centres from both calibrated shots."""
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from experiments.skate_video_v1.inputs import atomic_json, read, digest
from calibration.program_tracking.body_tracking.absolute_motion_curve import greville_times
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from ..programme_motion_followup import bind
from .source_audit import static_assembly
from .scoped_fit import Fit, tn, ti, cpu, save_candidate, smooth
from .direction_fit import authored_limits, native_eval

def main(args):
    torch.set_num_threads(4)
    start, output, source_path = args.parameter, args.output, args.source
    output.mkdir(exist_ok=False)
    r,c,info,_,z=bind(args.prepared, args.evidence, start)
    validate_checkpoint_identity(z,r.fixed_checkpoint_identity)
    doc=read(source_path)
    rows=doc['rows']
    idx=np.searchsorted(r.p['pts'],[row['pts'] for row in rows])
    np.testing.assert_array_equal(r.p['pts'][idx],[row['pts'] for row in rows])
    foot=1
    curve=r.absolute_curve
    targets=np.array([row['steel_centre_uv'] for row in rows])
    scales=np.array([row['sigma_px'] for row in rows])
    use=np.array([row['role']=='fit' for row in rows])
    g=greville_times(curve.knots)
    t=r.times
    ctl=np.flatnonzero((g>=t[idx[0]]-.22)&(g<=t[idx[-1]]+.22))
    frames=np.flatnonzero((t>=curve.knots[ctl[0]]-.025)&(t<=curve.knots[ctl[-1]+4]+.025))
    local_rows=np.searchsorted(frames,idx)
    fit=Fit(r,c,curve,frames,[2,5,8,11],ctl,fit_position=False)
    cols=np.array([2,3])
    _,mats=static_assembly(r)
    before,h0,mat0,fk0=native_eval(r,c,curve,frames,foot,cols,mats)
    pick=ti(local_rows[use]);trg=tn(targets[use]);sigma=tn(scales[use])
    exact_penalty, limits=authored_limits(fit,frames,foot,z)
    legs=[1,4,7]
    previous=limits(fk0['local_R'][:,legs])
    base_penalty=tn(previous)
    track=tn(before)
    p0=tn(fk0['native_world'][:,[5,8]])
    w0=tn(mat0[:,foot])
    z0=tn(h0[:,foot])
    s=r.p['shot_ids'][idx[use]]
    ii=np.flatnonzero((s[1:]==s[:-1]) & (np.diff(t[idx[use]])<.105))
    p0_idx=ti(ii);p1_idx=ti(ii+1)
    prior=1-smooth(t[idx[0]]-.19,t[idx[0]],t[frames])*(1-smooth(t[idx[-1]],t[idx[-1]]+.19,t[frames]))
    border=tn(prior)
    pose=load(args.prepared/'observations.npz')
    m=(pose['model_joint_ids'][frames]==14)&pose['use'][frames]
    m[local_rows[~use]]=False
    ki,kj=np.where(m)
    uvk=tn(pose['uv'][frames[ki],kj])
    def objective(st):
        w=fit.material_world(st)[:,foot,cols]
        project=fit.project(w)
        centre=project.mean(dim=1)
        residual=(centre[pick]-trg)/sigma[:,None]
        direct=(torch.sqrt(1+residual.square())-1).mean()*12.
        motion_residual=(residual[p1_idx]-residual[p0_idx])
        displacement=(torch.sqrt(1.+motion_residual.square())-1.).mean()*2.
        body=st[2][2::5][:,legs]
        ang=exact_penalty(body)
        anatomy=18.*ang.square().mean()+65.*torch.relu(ang-base_penalty-.07).square().mean()
        world=w[2::5]
        temporal=2.0*(torch.diff(world,n=2,dim=0)/.018).square().mean()
        temporal+=.55*(torch.diff(st[6][2::5,foot],n=2,dim=0)/.14).square().mean()
        pose_hold=.9*((st[5][2::5][:,[5,8]]-p0)/.052).square().mean()
        retain=((project-track).square().mean(dim=(1,2))*border).mean()/165.
        world_hold=((world-w0)/.09).square().mean()*.7
        ground=fit.contact_heights(st)[:,foot]
        support=(torch.relu((ground-z0).abs()-.015)/.022).square().mean()*2.
        if len(ki):
            knee=fit.project(st[5][:,5:6])[:,0][ti(ki)]
            knee_cost=(torch.sqrt(1.+((knee-uvk)/13.).square())-1.).mean()
        else:
            knee_cost=direct*0.
        return dict(source_centre=direct,same_shot_motion=displacement,exact_joint_envelope=anatomy,
                    material_acceleration=temporal,pose_prior=pose_hold,ends=retain,
                    world_prior=world_hold,steel_height=support,knee=knee_cost)
    trial=fit.run(objective,45,'101.54–101.78 右鞋切镜连续')
    after,h1,mat1,fk1=native_eval(r,c,trial,frames,foot,cols,mats)
    err=np.stack([np.linalg.norm(a[local_rows].mean(axis=1)-targets,axis=-1) for a in (before,after)])
    anatomy=np.stack([previous.max(0),limits(fk1['local_R'][:,legs]).max(0)])
    step=np.stack([np.degrees(Rotation.from_matrix(
        a['R'][:-1,foot].swapaxes(-1,-2)@a['R'][1:,foot]).magnitude()) for a in (fk0,fk1)])
    accel=[np.linalg.norm(np.diff(a[:,foot],n=2,axis=0),axis=-1).max() for a in (mat0,mat1)]
    gate=dict(source=bool(err[1,~use].mean()<err[0,~use].mean()-3 and err[1,use].mean()<err[0,use].mean()-3),
              same_right_shoe=bool(np.max(np.abs(mat1[:,0]-mat0[:,0]))<1.e-8),
              anatomy=bool(np.all(anatomy[1]<=anatomy[0]+.16)),
              continuity=bool(step[1].max()<=max(32,step[0].max()+5) and accel[1]<=max(.045,accel[0]*1.3)),
              clearance=bool(h1[:,foot].min()>=min(h0[:,foot].min()-.008,-.026)))
    accepted=all(gate.values())
    save_candidate(load(args.parent),trial if accepted else curve,output/'candidate.npz',parent=args.parent)
    report=dict(state='retained' if accepted else 'unresolved',window='101.54–101.78 右鞋切镜连续',
                source_path=str(source_path),pts=[r['pts'] for r in rows],source_window_pts=r.p['pts'][frames[[0,-1]]].tolist(),
                foot=foot,held_before_px=float(err[0,~use].mean()),held_trial_px=float(err[1,~use].mean()),
                centre_error_px=err.tolist(),anatomy=anatomy.tolist(),world_step_peak_deg=step.max(1).tolist(),
                material_second_difference_m=list(map(float,accel)),minimum_runner_mm=(1000*np.array([h0[:,foot].min(),h1[:,foot].min()])).tolist(),
                window_brackets_cut=True,gates=gate,parent_parameter_sha256=digest(start),result_sha256=digest(output/'candidate.npz'))
    atomic_json(output/'assessment.json',report)
    print(report,flush=True)
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prepared", "evidence", "parameter", "parent", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
