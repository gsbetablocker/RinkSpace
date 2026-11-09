"""Reachability and native knee shifts implied by rigid source-only diagnostics."""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.programme_skate_replay import write
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


def norm(x):
    return np.linalg.norm(x,axis=-1)


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','parameter','geometry','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    ref,camera,_,e,_=load_reference(a.project,a.prepared,a.source)
    state=load(a.parameter)
    ref.absolute_curve=curve_from_packet(state,seed=True)
    ref.relative_skate_curve=shoe_curve_from_packet(state,seed=True)
    ss=SequenceState(ref,tuple(ref.times[[0,-1]]))
    data=json.loads(a.geometry.read_text(encoding='utf8'))['groups']
    frames=np.searchsorted(ref.p['pts'],[r['pts'] for r in data])
    feet=np.array([r['foot'] for r in data])
    clock=ref.times[frames,None]+np.linspace(-.5,.5,5)*camera.readout_times(frames)[:,None]
    times=clock.ravel()
    d=ss.increments(state['x'][17:],times)
    posed=fk_state(ref,ref.at(times),d)
    n=len(feet);idx=np.arange(n*5)
    foot=np.repeat(feet,5)
    H,K,A=(posed['native_world'][idx,offset+foot].reshape(n,5,3) for offset in (1,4,7))
    O=posed['O'][idx,foot].reshape(n,5,3)
    size=ref.binding.ratio
    frac=np.array([r['free_rigid']['chosen_step_fraction'] for r in data])
    u=np.array([r['free_rigid_shift_m'] for r in data])*frac[:,None]/size
    rot=np.deg2rad([r['free_rigid_rotation_deg'] for r in data])*frac[:,None]
    demanded=O+u[:,None]+np.einsum('nij,naj->nai',_exp(rot),A-O)
    upper=norm(K-H);lower=norm(K-A)
    direction=demanded-H
    dist=norm(direction)
    axis=direction/np.maximum(dist[...,None],1e-12)
    a1=(upper**2-lower**2+dist**2)/(2*dist)
    rad=np.sqrt(np.maximum(0.,upper**2-a1*a1))
    center=H+axis*a1[...,None]
    radial=K-center
    radial-=axis*np.einsum('nki,nki->nk',axis,radial)[...,None]
    knee=center+rad[...,None]*radial/np.maximum(norm(radial)[...,None],1e-12)
    nodes=np.stack([K,A,knee,demanded],axis=2)
    image=moving_projection(camera,nodes,frames)['uv']
    # Angles are geometric flexion between two links, not the complete
    # knee chart, and the conditional depth is kept explicitly diagnostic.
    old=np.rad2deg(np.arccos(np.clip(np.einsum('nki,nki->nk',K-H,A-K)/(upper*lower),-1.,1.)))
    new=np.rad2deg(np.arccos(np.clip((dist**2-upper**2-lower**2)/(2*upper*lower),-1.,1.)))
    result=[]
    for i,row in enumerate(data):
        beyond=float(size*(dist[i,2]-upper[i,2]-lower[i,2]))
        result.append(dict(pts=row['pts'],foot=row['foot'],raw_outline_rms_px=row['raw']['rms_px'],
           independent_outline_rms_px=row['free_rigid']['rms_px'],fraction=float(frac[i]),
           candidate_ankle_uv=image[i,3].tolist(),current_ankle_uv=image[i,1].tolist(),
           knee_uv_current=image[i,0].tolist(),knee_uv_min_world_change=image[i,2].tolist() if beyond<=0 else None,
           ankle_shift_px=float(norm(image[i,3]-image[i,1])),
           chosen_knee_shift_px=float(norm(image[i,2]-image[i,0])) if beyond<=0 else None,
           straight_leg_beyond_reach_m_conditional=max(0.,beyond),
           knee_flexion_geometry_deg=[float(old[i,2]),float(new[i,2])] if beyond<=0 else [float(old[i,2]),None],
           sampled_original_blade_phase=ref.p['shot_ids'][frames[i]].item()))
    write(a.output,dict(rows=result,scope='single tested depth branch, fixed hip and current link lengths',
                        production_acceptance=False,
                        metric_geometry_is_conditional=True))


if __name__=='__main__':
    main()
