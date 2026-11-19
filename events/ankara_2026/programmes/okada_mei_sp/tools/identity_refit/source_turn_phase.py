"""Source-shape time alignment proposes continuous ankle initializations.

The proposal is inferred from silhouettes, not new observations. It changes
actual articulated controls and must be checked with the original projector.
No source image, camera clock, root path or shoe placement is warped.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import diags, eye, vstack
from scipy.sparse.linalg import lsmr

from compare import bind
from source_review import digest
from source_hand_direction_init import lift
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, _exp, _log, greville_times,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state


def shape_descriptor(poly):
    moments=cv2.moments(np.asarray(poly,np.float32));area=moments['m00']
    if area<1.e-6:return np.full(32,np.nan)
    centre=np.array([moments['m10'],moments['m01']])/area
    angle=np.arange(32)*2*np.pi/32;directions=np.stack([np.cos(angle),np.sin(angle)],axis=1)
    return ((poly-centre)/np.sqrt(area)@directions.T).max(axis=0)


def align_shapes(model,source,visible,shots):
    """Monotone candidate clocks with finite lag and rate priors.

    Search spans 300 ms on either side. Missing source silhouettes force the
    existing clock; candidates never cross a physical camera edit.
    """
    n=len(model);lag=np.arange(-15,16);states=len(lag);zero=15
    offsets=lag[None,:]-lag[:,None]
    mapping=np.arange(n);records=[]
    for shot in np.unique(shots):
        frames=np.flatnonzero(shots==shot);first,last=frames[[0,-1]]
        current=np.mean((model[frames]-source[frames])**2,axis=1)
        finite=current[np.isfinite(current)&visible[frames]]
        scale=max(float(np.median(finite)) if len(finite) else 0.,1.e-5)
        transition=.10*scale*offsets**2
        transition[offsets < -1]=np.inf
        unary=np.full((len(frames),states),np.inf)
        for row,frame in enumerate(frames):
            if not visible[frame] or not np.isfinite(source[frame]).all():
                unary[row,zero]=0.;continue
            candidate=frame+lag;valid=(candidate>=first)&(candidate<=last)
            error=np.mean((model[np.clip(candidate,first,last)]-source[frame])**2,axis=1)
            error+=.10*scale*(lag/5.)**2
            unary[row,valid&np.isfinite(error)]=error[valid&np.isfinite(error)]
            if not np.isfinite(unary[row]).any():unary[row,zero]=0.
        previous=np.full(states,np.inf);previous[zero]=0.;parents=[]
        for row in unary:
            values=previous[:,None]+transition
            parent=np.argmin(values,axis=0);parents.append(parent)
            previous=values[parent,np.arange(states)]+row
        state=int(np.argmin(previous));chosen=np.empty(len(frames),int)
        for i in range(len(frames)-1,-1,-1):
            chosen[i]=lag[state];state=int(parents[i][state])
        mapping[frames]=frames+chosen
        records.append(dict(shot=str(shot),changed=int(np.count_nonzero(chosen)),
                            maximum_lag_seconds=float(np.max(abs(chosen))*.02),
                            descriptor_cost_before=float(np.nansum(current)),
                            descriptor_cost_after=float(np.nansum(np.mean((model[mapping[frames]]-source[frames])**2,axis=1)))))
    assert np.all(shots[mapping]==shots)
    assert np.all(mapping[~visible]==np.arange(n)[~visible])
    return mapping,records


def propose(prepared,source,parameter,screen,work,output):
    cache=dict(np.load(screen));observations=dict(np.load(work/'input/shoe_turn_source.npz'))
    if str(cache['parameter_sha256'])!=digest(parameter):raise ValueError('Projection cache belongs to a different curve')
    ref,_,_,_,packet=bind(prepared,source,parameter);curve=curve_from_packet(packet)
    np.testing.assert_array_equal(ref.p['pts'],cache['pts'])
    visible=cache['visible']&(cache['role']==1)
    model=np.full((len(ref.times),2,32),np.nan);raw=model.copy()
    for frame,foot in zip(*np.nonzero(visible)):
        model[frame,foot]=shape_descriptor(cache['contour'][frame,foot])
        raw[frame,foot]=shape_descriptor(observations['region'][frame,foot])
    target=np.empty((len(ref.times),2),int);records=[]
    for foot in (0,1):
        target[:,foot],record=align_shapes(model[:,foot],raw[:,foot],visible[:,foot],ref.p['shot_ids'])
        records.append(dict(foot=foot,shots=record))
    world=np.empty((len(ref.times),2,3,3))
    for first in range(0,len(ref.times),512):
        times=ref.times[first:first+512];state=fk_state(ref,ref.at(times),np.zeros((len(times),69)))
        world[first:first+len(times)]=state['Q'][:,None]@state['frames'][:,[7,8]]
    desired=world[target,np.arange(2)[None,:]]
    delta=_log(world.swapaxes(-1,-2)@desired)
    basis=curve.basis_sparse(ref.times);count=basis.shape[1]
    clock=greville_times(curve.knots);h=np.diff(clock)
    # Small finite curvature and displacement priors apply to the proposal;
    # the source-image refinement evaluates the resulting actual trajectory.
    second=diags([2/(h[:-1]*(h[:-1]+h[1:])),
                  -2/(h[:-1]*h[1:]),2/(h[1:]*(h[:-1]+h[1:]))],
                 [0,1,2],shape=(count-2,count),format='csr')
    matrix=vstack([basis*np.sqrt(.02),second*.0005*np.sqrt(.02),eye(count)*.02],format='csr')
    increments=np.empty((count,2,3));checks=[]
    for foot in (0,1):
        for axis in range(3):
            rhs=np.r_[delta[:,foot,axis]*np.sqrt(.02),np.zeros(2*count-2)]
            solution=lsmr(matrix,rhs,atol=1.e-10,btol=1.e-10)
            if solution[1] not in (1,2,4,5):raise RuntimeError('Phase proposal linear solve did not converge')
            increments[:,foot,axis]=solution[0];checks.append(dict(foot=foot,axis=axis,status=int(solution[1]),normal_residual=float(solution[4])))
    rotations=curve.rotation_controls.copy();rotations[:,[7,8]]=rotations[:,[7,8]]@_exp(increments)
    changed=np.linalg.norm(increments,axis=-1)>1.e-12
    from scipy.spatial.transform import Rotation
    q=lift(curve);relative=rotations[:-1].swapaxes(-1,-2)@rotations[1:]
    short=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
    short=np.where((short[...,3]<0)[...,None],-short,short)
    for foot,joint in enumerate((7,8)):
        edges=changed[:-1,foot]|changed[1:,foot];q[edges,joint]=short[edges,joint]
    candidate=AbsoluteMotionCurve(curve.knots,curve.position_controls,rotations,q)
    record=dict(parent=str(parameter),parent_sha256=digest(parameter),source_outline_sha256=digest(work/'input/shoe_turn_source.npz'),
                proposal_only=True,independent_source_observation=False,source_frames=len(ref.times),feet=records,
                source_clock_and_camera_unchanged=True,root_and_other_joints_unchanged=True,
                ankle_edge_initialization='Short source-shape-aligned path; not an accepted observation',linear_checks=checks)
    result=dict(packet);result.update(absolute_rotation_controls=rotations,absolute_seed_rotation_controls=rotations,
                                    absolute_rotation_edge_reference_quaternion=lift(candidate),
                                    source_turn_phase_json=np.array(json.dumps(record,sort_keys=True)))
    with output.open('wb') as f:np.savez_compressed(f,**result)
    np.savez_compressed(work/'source_review/phase_screen.npz',pts=ref.p['pts'],target_indices=target,
                        before_sha256=np.array(digest(parameter)),after_sha256=np.array(digest(output)),
                        shape_cost_before=np.mean((model-raw)**2,axis=2),
                        shape_cost_after=np.mean((model[target,np.arange(2)[None,:]]-raw)**2,axis=2))
    record.update(event='source_turn_phase_proposal',output=str(output),sha256=digest(output))
    with (work/'whole_continuity/progress.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','source','parameter','screen','work','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    propose(**vars(parser.parse_args()))
