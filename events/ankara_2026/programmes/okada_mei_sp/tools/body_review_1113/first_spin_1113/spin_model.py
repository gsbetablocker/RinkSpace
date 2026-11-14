from pathlib import Path
import os
import sys
import json

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, NAMES, angles_joint
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from experiments.skate_video_v1.body_skate_fit import Assembly
from experiments.skate_video_v1.inputs import atomic_json, read
from experiments.skate_video_v1.motion_review.scoped_fit import Fit, cpu, ti, tn, save_candidate
from experiments.skate_video_v1.programme_motion_followup import bind

RUN=Path(os.environ['RINKSPACE_AUDIT_RUN'])
JOB=Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', RUN.parent/'audit_af48_1113')).resolve()/'first_spin_1113'
CUTS=JOB.parent/'cut_continuity_1113'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cut_continuity_1113'))
from cavity_torch import Cavity
from cut_trial_geometry import DenseIce, DeformedUpper, SurfaceBounds

PREP=RUN/'prepared_refined'
PARENT=CUTS/'delivered/solve/candidate.npz'
r, camera, info, _, packet=bind(PREP,CUTS/'cut_source.json',PARENT)
doc=read(JOB/'spin_source.json')
r.source_document=doc
curve=curve_from_packet(packet)
clock=r.times
pts=r.p['pts']
seconds=pts/12800.
ratio=r.binding.ratio
camera.core.motion_projection_threads=8
for view in camera.cameras.values():
    view.core.motion_projection_threads=8
obs=load(PREP/'observations.npz')
roles=load(doc['whole_skate_source']['path'])
mass=ContinuousMassGeometry(r)
assembly=Assembly(r.binding)
config=doc['scoped_candidate_import']
anchor=load(doc['programme_structure']['chart_anchor'])
chart=JointChart(anchor['rest_R'],anchor['anatomical_basis'],anchor['reference_twist'])
policy=read(doc['programme_structure']['policy'])
envelopes=[policy['envelopes'][name.rsplit('_',1)[-1]] for name in NAMES]
exact_envelopes=[JointEnvelope(*np.deg2rad(e['flex_degrees']),np.deg2rad(e['flex_grid_degrees']),
                  np.deg2rad(e['swing_axes_degrees']),*np.deg2rad(e['scale_degrees']),policy['provenance'])
                 for e in envelopes]
bodies=tn(anchor['anatomical_basis'])
rest=tn(anchor['rest_R'])
joint_ids=JOINTS-1
bounds_data=[tuple(tn(np.deg2rad(e[k])) for k in
        ('flex_degrees','flex_grid_degrees','swing_axes_degrees','scale_degrees')) for e in envelopes]


def state(curve_at,t):
    ref=absolute_reference(r,curve_at)
    ref.relative_skate_curve=r.relative_skate_curve
    return fk_state(ref,ref.at(t),np.zeros((len(t),69)))


def sq(x):
    return x.square().sum(dim=-1)

def robust(x,delta=2.):
    return 2.*delta*delta*(torch.sqrt(1.+x.square()/delta**2)-1.)

def quat_axis(rotation):
    # Away from the pi branch the original anatomical charts use this
    # representation. Safe sign at pi is supplied by the frozen twist lift.
    w=(1.+rotation.diagonal(dim1=-1,dim2=-2).sum(-1)).clamp_min(1.e-7).sqrt()/2.
    v=torch.stack((rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],
                   rotation[...,1,0]-rotation[...,0,1]),-1)/(4.*w[...,None])
    return v,w

def angular_steps(R):
    relative=R[1:]@R[:-1].transpose(-1,-2)
    q,w=quat_axis(relative)
    n=torch.linalg.vector_norm(q,dim=-1)
    return q*((2.*torch.atan2(n,w)/n.clamp_min(1.e-7)).where(n>1.e-7,2.))[...,None]

def angular_accel(R,dt):
    return torch.diff(angular_steps(R),dim=0)/dt**2

def anatomy_values(pose,frames):
    data=chart.evaluate(pose['local_R'][:,joint_ids],sample_indices=frames)
    return np.stack([env.residual(angles_joint(data,k)) for k,env in enumerate(exact_envelopes)],1)

def anatomical(local,branch):
    R=bodies.transpose(-1,-2)@rest.transpose(-1,-2)@local[:,joint_ids]@bodies
    q,w=quat_axis(R)
    x,y,z=q.unbind(-1)
    twist=2.*torch.atan2(x,w)
    twist=branch+torch.remainder(twist-branch+torch.pi,2.*torch.pi)-torch.pi
    n=torch.linalg.vector_norm(q[...,1:],dim=-1)
    p=torch.sqrt(w.square()+x.square())
    swing=2.*torch.atan2(n,p)
    u=torch.stack((w*y-x*z,w*z+x*y),-1)/(p*n.clamp_min(1.e-7))[...,None]
    out=[]
    for j,(ends,grid,radii,scales) in enumerate(bounds_data):
        a=twist[:,j]
        k=torch.searchsorted(grid,a.contiguous(),right=True).clamp(1,len(grid)-1)
        f=((a-grid[k-1])/(grid[k]-grid[k-1])).clamp(0.,1.)
        radius=radii[k-1]+f[:,None]*(radii[k]-radii[k-1])
        allowed=torch.linalg.vector_norm(u[:,j]/radius,dim=-1).clamp_min(1.e-6).reciprocal()
        out.append(torch.relu(torch.stack(((ends[0]-a)/scales[0],(a-ends[1])/scales[0],
                                          (swing[:,j]-allowed)/scales[1]),-1)))
    return torch.stack(out,dim=1)

def ground_samples():
    pack=r.binding.packet()
    tr=pack['template']
    samples=[]
    for name in ('runner','toe_pick'):
        vertices=[]
        for arc in tr['contact_arcs']:
            if arc['semantic']!=name:continue
            t=np.linspace(arc['theta_start_rad'],arc['theta_end_rad'],73)
            vertices.extend(np.asarray(arc['center'])+arc['radius_m']*(
                np.cos(t[:,None])*arc['axis_cos']+np.sin(t[:,None])*arc['axis_sin']))
        for group in tr['contact_point_groups']:
            if group['semantic']==name:vertices.extend(group['points'])
        samples.append(tn(vertices))
    return (*samples,tn(tr['vertices']),tn(pack['upper']))
edge=ground_samples()


def mean(a, mask=None):
    return a.mean() if mask is None else a[mask].mean()


def log_event(**fields):
    print(json.dumps(fields,ensure_ascii=False),flush=True)


def to_world(P,Q,v,body):
    return P[:,None]+(Q[:,None]@(v-body.J[0])[...,None]).squeeze(-1)*body.body.scale
