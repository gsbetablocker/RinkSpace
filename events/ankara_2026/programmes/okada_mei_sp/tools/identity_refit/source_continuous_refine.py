"""Refine reviewed source intervals on the existing complete body curve.

This follows the programme-wide fit. Local source visibility corrections use
the same native projector, articulated geometry and analytic spline chain.
The saved output still contains every programme control, including its halos.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, eye, kron, vstack

from compare import bind
from source_review import digest
from source_hand_direction_init import lift
from calibration.program_tracking.first_shot_atomic_output import replace_file
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, RotationChartDomainError, greville_times, _exp, _log,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_motion_factors import ActualMotionFactors
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from calibration.program_tracking.body_tracking.source_motion_refit import NativeBodyPoints, NativeBodyRegions, NativeShoePoints, NativeSurfacePoints


_stage_worker = None


def initialize_stage_worker(args, queue):
    global _stage_worker
    # The parent records checkpoint/failure receipts itself. Diagnostic log
    # feeders must not prevent a failed worker from returning its exception.
    queue.cancel_join_thread()
    r,c,_,document,packet=bind(args.prepared,args.source,args.parameter)
    r.source_document=document
    points=json.loads(args.points.read_text(encoding='utf8'))
    observations=dict(np.load(args.observations))
    _stage_worker=(r,c,packet,points,observations,queue,args.check_derivatives)


def run_stage_worker(curve, spec):
    r,c,packet,points,observations,queue,check=_stage_worker
    def emit(event,**kw):queue.put((event,kw))
    result=stage(r,c,packet,curve,points,observations,spec,emit,
                 check and spec.get('check_derivatives',True))
    return result


def independent_stage_groups(curve, specs, source_offset):
    """Colour blocks by their entire spline influence, including readout halo."""
    clock=greville_times(curve.knots)+source_offset
    groups=[];supports=[]
    for spec in specs:
        lo,hi=spec['seconds'];ids=np.flatnonzero((clock>=lo)&(clock<=hi))
        if not len(ids):raise ValueError('Source stage has no spline controls')
        support=(curve.knots[ids[0]]+source_offset-.04,
                 curve.knots[ids[-1]+4]+source_offset+.04)
        for group,bounds in zip(groups,supports):
            if all(support[1]<a or support[0]>b for a,b in bounds):
                group.append(spec);bounds.append(support);break
        else:
            groups.append([spec]);supports.append([support])
    return groups


def merge_independent_curves(base, curves):
    """Combine disjoint control/edge updates without resampling or stitching."""
    P=base.position_controls.copy();R=base.rotation_controls.copy();q=lift(base)
    occupied=np.zeros(len(P),bool);occupied_edges=np.zeros(len(P)-1,bool)
    for child in curves:
        np.testing.assert_array_equal(child.knots,base.knots)
        changed=np.any(child.position_controls!=base.position_controls,axis=1)
        changed|=np.any(child.rotation_controls!=base.rotation_controls,axis=(1,2,3))
        edges=changed[:-1]|changed[1:]
        if np.any(occupied&changed) or np.any(occupied_edges&edges):
            raise ValueError('Parallel source blocks overlap controls or rotation edges')
        P[changed]=child.position_controls[changed];R[changed]=child.rotation_controls[changed]
        q[edges]=lift(child)[edges];occupied|=changed;occupied_edges|=edges
    return AbsoluteMotionCurve(base.knots,P,R,q)


def restored_curve(parent, baseline, offset):
    """Explicitly replace rejected ending displacement and occluded arm spikes."""
    g=greville_times(parent.knots)+offset
    P=parent.position_controls.copy();R=parent.rotation_controls.copy()
    weight=np.clip((g-188.12)/.20,0.,1.)
    P+=(baseline.position_controls-P)*weight[:,None]
    # Restore visible upper-body/leg pose; keep the newly fitted ankle turns.
    joints=np.array([j for j in range(22) if j not in (7,8)])
    R[:,joints]=R[:,joints]@_exp(_log(R[:,joints].swapaxes(-1,-2)@
        baseline.rotation_controls[:,joints])*weight[:,None,None])
    arms=np.array([13,14,16,17,18,19,20,21])
    first=int(np.searchsorted(g,85.08)-1);last=int(np.searchsorted(g,85.46))
    for i in range(first+1,last):
        amount=(g[i]-g[first])/(g[last]-g[first])
        R[i,arms]=R[first,arms]@_exp(_log(R[first,arms].swapaxes(-1,-2)@R[last,arms])*amount)
    changed=np.any(R!=parent.rotation_controls,axis=(-1,-2))
    edges=changed[:-1]|changed[1:]
    q=lift(parent)
    from scipy.spatial.transform import Rotation
    relative=R[:-1].swapaxes(-1,-2)@R[1:]
    short=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
    short=np.where((short[...,3]<0)[...,None],-short,short)
    q[edges]=short[edges]
    return AbsoluteMotionCurve(parent.knots,P,R,q),dict(
        restored_ending_controls=np.flatnonzero(weight>0).tolist(),
        interpolated_occluded_arm_controls=[first+1,last-1],
        interpolation_is_initialization_not_observation=True,
        explicitly_rebased_edges=np.argwhere(edges).tolist())


def restore_occluded_arms(parent,baseline,offset):
    """Retain the earlier source-compatible arm path in reviewed occlusion.

    This is a pose prior repair, not an invented observation of hidden joints.
    The three-dimensional curve remains one continuous spline.
    """
    g=greville_times(parent.knots)+offset;R=parent.rotation_controls.copy()
    records=[]
    for lo,inside,end,hi,joints in [
        (85.20,85.24,85.38,85.44,[14,17,19,21]),
        (85.58,85.64,85.76,85.82,[13,14,16,17,18,19,20,21])]:
        weight=np.minimum(np.clip((g-lo)/(inside-lo),0.,1.),np.clip((hi-g)/(hi-end),0.,1.))
        R[:,joints]=R[:,joints]@_exp(_log(R[:,joints].swapaxes(-1,-2)@
            baseline.rotation_controls[:,joints])*weight[:,None,None])
        records.append(dict(seconds=[lo,hi],joints=joints,controls=np.flatnonzero(weight>0).tolist(),
            evidence='Source review rejects limbs entering empty background; old path is retained as inferred pose, not measured hidden joints'))
    q=lift(parent);old=lift(baseline)
    changed=np.any(R!=parent.rotation_controls,axis=(-1,-2));edges=changed[:-1]|changed[1:]
    q[edges]=old[edges]
    candidate=AbsoluteMotionCurve(parent.knots,parent.position_controls,R,q)
    return candidate,records


def source_resolved_initialization(parent,baseline,offset):
    """Remove rejected sub-frame windings and resolve the fast opening turn."""
    from scipy.interpolate import BSpline
    from scipy.spatial.transform import Rotation
    g=greville_times(parent.knots)+offset;R=parent.rotation_controls.copy()
    for lo,inside,end,hi,joints in [
        (23.66,23.90,24.80,25.04,[1,2,4,5,7,8,14,17,19,21]),
        (84.90,85.00,85.42,85.52,[1,4,7])]:
        weight=np.minimum(np.clip((g-lo)/(inside-lo),0.,1.),np.clip((hi-g)/(hi-end),0.,1.))
        R[:,joints]=R[:,joints]@_exp(_log(R[:,joints].swapaxes(-1,-2)@
            baseline.rotation_controls[:,joints])*weight[:,None,None])
    q=lift(parent);old=lift(baseline)
    changed=np.any(R!=parent.rotation_controls,axis=(-1,-2));edges=changed[:-1]|changed[1:]
    q[edges]=old[edges]
    seed=AbsoluteMotionCurve(parent.knots,parent.position_controls,R,q)
    spline=BSpline(seed.knots,seed.position_controls,3)
    inserted=[]
    for t in np.arange(23.84,24.94,.04)-offset:
        if np.min(abs(spline.t-t))>1e-8:
            spline=spline.insert_knot(t);inserted.append(float(t+offset))
    clock=greville_times(spline.t)
    old_clock=greville_times(seed.knots)
    nearest=np.abs(clock[:,None]-old_clock).argmin(1)
    same=abs(clock-old_clock[nearest])<1e-10
    from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
    new_R=evaluate_curve(seed,clock)['rotations'];new_R[same]=seed.rotation_controls[nearest[same]]
    relative=new_R[:-1].swapaxes(-1,-2)@new_R[1:]
    new_q=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
    new_q=np.where((new_q[...,3]<0)[...,None],-new_q,new_q)
    preserved=same[:-1]&same[1:]&(np.diff(nearest)==1)
    new_q[preserved]=lift(seed)[nearest[:-1][preserved]]
    curve=AbsoluteMotionCurve(spline.t,spline.c,new_R,new_q)
    return curve,dict(inserted_source_knot_seconds=inserted,
        rejected_subframe_winding_regions=[[23.66,25.04],[84.90,85.52]],
        unchanged_control_rows=int(same.sum()),translation_knot_insertion_exact=True,
        local_rotation_sampling_is_initialization_not_measurement=True)


def reinitialize_reviewed_joints(parent, baseline, offset, regions):
    """Retry rejected turn branches from a retained continuous pose path.

    This changes initialization only. It neither labels the older pose as a
    source observation nor changes the subsequent image/motion objective.
    """
    from scipy.spatial.transform import Rotation
    np.testing.assert_array_equal(parent.knots, baseline.knots)
    clock=greville_times(parent.knots)+offset
    rotations=parent.rotation_controls.copy();records=[]
    for row in regions:
        lo,inside,end,hi=map(float,row['seconds'])
        if not lo<inside<=end<hi:raise ValueError('Invalid reviewed initialization window')
        joints=np.asarray(row['joints'],int)
        if len(joints)==0 or np.any((joints<0)|(joints>=22)):
            raise ValueError('Invalid reviewed initialization joints')
        weight=np.minimum(np.clip((clock-lo)/(inside-lo),0.,1.),
                          np.clip((hi-clock)/(hi-end),0.,1.))
        rotations[:,joints]=rotations[:,joints]@_exp(_log(
            rotations[:,joints].swapaxes(-1,-2)@baseline.rotation_controls[:,joints])
            *weight[:,None,None])
        records.append(dict(seconds=[lo,inside,end,hi],joints=joints.tolist(),
                            changed_control_rows=np.flatnonzero(weight>0).tolist(),
                            evidence=row.get('evidence','Source image review')))
    changed=np.any(rotations!=parent.rotation_controls,axis=(-1,-2))
    edges=changed[:-1]|changed[1:];quaternion=lift(parent)
    relative=rotations[:-1].swapaxes(-1,-2)@rotations[1:]
    short=Rotation.from_matrix(relative.reshape(-1,3,3)).as_quat().reshape(*relative.shape[:-2],4)
    short=np.where((short[...,3]<0)[...,None],-short,short)
    quaternion[edges]=short[edges]
    return AbsoluteMotionCurve(parent.knots,parent.position_controls,rotations,quaternion),dict(
        regions=records,source_observations_unchanged=True,
        older_pose_is_initialization_only=True,changed_edges=int(edges.sum()),
        reason='Source review rejects short opposite shoe rotations; retry the retained continuous branch before fitting the same original pixels')


def refine_native_rotation_clock(parent, native_times):
    """Subdivide the whole continuous curve at every source exposure.

    Translation insertion is exact. Geodesic Boehm insertion supplies rotation
    controls, then sparse Greville collocation retains the old rotation path as
    an initialization. The approximation is measured over the native clock;
    these old model samples never become additional image observations.
    """
    from scipy.interpolate import BSpline
    from scipy.sparse.linalg import splu
    from scipy.spatial.transform import Rotation
    from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve

    def multiply(a,b):
        return np.concatenate((a[...,3,None]*b[...,:3]+b[...,3,None]*a[...,:3]
            +np.cross(a[...,:3],b[...,:3]),
            (a[...,3]*b[...,3]-np.sum(a[...,:3]*b[...,:3],axis=-1))[...,None]),axis=-1)

    native_times=np.asarray(native_times,float)
    if np.any(np.diff(native_times)<=0):raise ValueError('Ordered native exposure times required')
    knots=parent.knots.copy();spline=BSpline(knots,parent.position_controls,3)
    old_edges=lift(parent);q=np.empty((*parent.rotation_controls.shape[:2],4))
    q[0]=Rotation.from_matrix(parent.rotation_controls[0].copy()).as_quat()
    for i in range(1,len(q)):q[i]=multiply(q[i-1],old_edges[i-1])
    q/=np.linalg.norm(q,axis=-1,keepdims=True)
    inserted=[]
    for t in native_times:
        if not knots[3]<t<knots[-4] or np.min(abs(knots-t))<1.e-9:continue
        span=int(np.searchsorted(knots,t,side='right')-1)
        indices=np.arange(span-2,span+1)
        alpha=((t-knots[indices])/(knots[indices+3]-knots[indices]))[:,None]
        left,right=q[indices-1],q[indices]
        dot=np.clip(np.sum(left*right,axis=-1),-1.,1.)
        angle=np.arccos(dot);sine=np.sin(angle)
        a=np.divide(np.sin((1-alpha)*angle),sine,out=np.broadcast_to(1-alpha,angle.shape).copy(),where=abs(sine)>1.e-10)
        b=np.divide(np.sin(alpha*angle),sine,out=np.broadcast_to(alpha,angle.shape).copy(),where=abs(sine)>1.e-10)
        middle=a[...,None]*left+b[...,None]*right
        middle/=np.linalg.norm(middle,axis=-1,keepdims=True)
        q=np.concatenate((q[:span-2],middle,q[span:]),axis=0)
        spline=spline.insert_knot(float(t));knots=spline.t;inserted.append(float(t))
    inverse=q[:-1].copy();inverse[...,:3]*=-1
    edges=multiply(inverse,q[1:]);edges/=np.linalg.norm(edges,axis=-1,keepdims=True)
    rotations=Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:2],3,3)
    curve=AbsoluteMotionCurve(knots,spline.c,rotations,edges)
    clock=greville_times(knots);target=evaluate_curve(parent,clock)['rotations']
    basis=curve.basis_sparse(clock);factor=splu(basis.tocsc())
    errors=[]
    while True:
        current=evaluate_curve(curve,clock,basis=basis)['rotations']
        residual=_log(current.swapaxes(-1,-2)@target)
        maximum=float(np.max(np.linalg.norm(residual,axis=-1)))
        errors.append(maximum)
        if maximum<1.e-8:break
        if len(errors)>3 and maximum>=min(errors[-4:-1])*(1-1.e-8):
            raise ValueError('Native-clock rotation initialization failed to converge')
        delta=factor.solve(residual.reshape(len(clock),-1)).reshape(len(clock),22,3)
        curve=curve.retracted(np.zeros_like(curve.position_controls),delta)
    angular=[];position_error=0.
    for first in range(0,len(native_times),512):
        times=native_times[first:first+512]
        a=evaluate_curve(parent,times);b=evaluate_curve(curve,times)
        position_error=max(position_error,float(np.max(abs(a['P']-b['P']))))
        angular.append(np.rad2deg(np.linalg.norm(_log(a['rotations'].swapaxes(-1,-2)@b['rotations']),axis=-1)))
    angular=np.concatenate(angular)
    return curve,dict(old_controls=len(parent.position_controls),new_controls=len(curve.position_controls),
        inserted_source_relative_seconds=inserted,source_exposures=len(native_times),
        translation_maximum_difference=position_error,greville_error_radians=errors,
        native_rotation_difference_degrees_maximum_by_joint=angular.max(0).tolist(),
        native_rotation_difference_degrees_p95=float(np.percentile(angular,95)),
        source_pixels_unchanged=True,old_model_path_is_initialization_not_observations=True,
        reason='Resolve full-programme rapid turns on the original 20 ms exposure clock before refitting image evidence')


def registered_cuff_initialization(parent,reference,source_outline):
    """Try the boot's registered shank axis where image direction is ambiguous.

    This changes the initial branch only. Subsequent fitting still uses the
    original source pixels; the inferred orientation is never an observation.
    """
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import gaussian_filter1d
    from calibration.program_tracking.body_tracking.source_foot_foreground import RegisteredBootCuff
    from calibration.program_tracking.body_tracking.programme_whole_skate import ambiguous_shoe_frames
    r=absolute_reference(reference,parent);clock=greville_times(parent.knots)
    cuff=RegisteredBootCuff(SimpleNamespace(reference=r,binding=r.binding),clock)
    with np.load(source_outline) as packet:
        active=ambiguous_shoe_frames(packet)
        weight=np.interp(clock+r.p['pts'][0]/12800,packet['source_pts']/12800,active.astype(float))
    rotations=parent.rotation_controls.copy();records=[]
    for foot,joint in enumerate((7,8)):
        current=cuff.shank[foot]@cuff.rotation[foot]
        # Both centre rows are homogeneous planes through the registered
        # ankle. Their intersection is the boot's shank exit direction.
        target=np.cross(cuff.centre[foot,0],cuff.centre[foot,1]);target/=np.linalg.norm(target)
        if target[2]<0:target=-target
        axis=np.cross(target,current);length=np.linalg.norm(axis)
        angle=np.arctan2(length,current@target)
        # Start from the shaped leg's neutral ankle yaw, rather than retaining
        # an arbitrary old yaw about an occluded shank. The source fit remains
        # free to turn it again wherever visible evidence supports that turn.
        neutral=cuff.rotation[foot]@_exp(axis/max(length,1.e-12)*angle)@cuff.rotation[foot].T
        tangent=_log(rotations[:,joint].swapaxes(-1,-2)@neutral)*weight[:,None]
        tangent=gaussian_filter1d(tangent,2.,axis=0,mode='nearest')
        rotations[:,joint]=rotations[:,joint]@_exp(tangent)
        records.append(dict(foot=foot,maximum_initial_rotation_deg=float(np.degrees(np.linalg.norm(tangent,axis=-1)).max())))
    quaternion=Rotation.from_matrix((rotations[:-1].swapaxes(-1,-2)@rotations[1:]).reshape(-1,3,3)).as_quat().reshape(len(rotations)-1,22,4)
    old_lift=lift(parent)
    quaternion=np.where((np.sum(quaternion*old_lift,axis=-1)<0)[...,None],-quaternion,quaternion)
    unchanged=np.ones(22,bool);unchanged[[7,8]]=False
    quaternion[:,unchanged]=old_lift[:,unchanged]
    return AbsoluteMotionCurve(parent.knots,parent.position_controls,rotations,quaternion),dict(
        geometric_boot_attachment_initialization=records,source_observation=False,source_pixels_unchanged=True)


def stage(reference,camera,packet,curve,points,obs,spec,emit,check=False,prior_reference=None):
    ref=absolute_reference(reference,curve)
    state=SequenceState(ref,packet['seconds'])
    body=SimpleNamespace(state=state,size=state.size)
    p=SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,body=body,size=17+state.size,
                      analytic_rigid_shoe_derivatives=True,fixed_relative_skates=True)
    native=ref.p['pts']/12800.
    lo,hi=spec['seconds'];g=greville_times(curve.knots)+native[0]
    controls=np.flatnonzero((g>=lo)&(g<=hi))
    channels=np.r_[np.arange(3) if spec.get('root_position') else np.empty(0,int),
        np.concatenate([3+3*j+np.arange(3) for j in spec['joints']])].astype(int)
    # Each rotational spline has its own joint channels. Computing inactive
    # arm derivatives in a leg-only block cannot contribute selected columns.
    p.active_body_pose_channels=channels
    columns=(controls[:,None]*69+channels).ravel()
    support_rule=reference.source_document.get('programme_double_support')
    if 'programme_double_support_json' in packet:
        inherited=json.loads(str(packet['programme_double_support_json']))
        if support_rule is not None and support_rule!=inherited:
            raise ValueError('Source and parent checkpoint declare different opening support')
        support_rule=inherited
    if support_rule:
        from opening_support import samples,verify
        verify(ref,curve,support_rule)
        protected=np.unique(curve.basis_sparse(samples(ref,support_rule)).indices)
        anchored=np.array([0,1,2,4,5,7,8])
        moving_support=(channels<3)|np.isin((channels-3)//3,anchored)
        # A coupled mapping has one set of channels per control. Keep the
        # supported stance fixed for any block that could change its blades.
        if moving_support.any():
            controls=np.setdiff1d(controls,protected)
            if not len(controls):
                emit('retained_confirmed_support',stage=spec['name'])
                return curve
            columns=(controls[:,None]*69+channels).ravel()
    mapping=None;latent_update_scale=None
    if spec.get('common_turn'):
        u=np.minimum(np.clip((g[controls]-lo)/.10,0.,1.),np.clip((hi-g[controls])/.10,0.,1.))
        mapping=kron(csr_matrix((u*u*(3-2*u))[:,None]),eye(len(channels)),format='csr')
    if spec.get('world_yaw_phase'):
        # One coherent timing correction of the skate's world yaw. A fixed
        # local ankle rotation cannot express this when the shank tilts while
        # spinning. Source PTS, camera and the rest of the body stay fixed.
        from calibration.program_tracking.body_tracking.skate_physics import fk_state
        policy=spec['world_yaw_phase'];foot=int(policy['foot']);joint=(7,8)[foot]
        if spec.get('common_turn') or spec['joints']!=[joint] or spec.get('root_position'):
            raise ValueError('World-yaw phase requires exactly its ankle joint')
        clock=g[controls]-native[0];h=.005
        centre=fk_state(ref,ref.at(clock),np.zeros((len(clock),69)))
        side=[]
        for shift in (-h,h):
            pose=fk_state(ref,ref.at(clock+shift),np.zeros((len(clock),69)))
            forward=pose['R'][:,foot,:,0]
            if np.any(np.linalg.norm(forward[:,:2],axis=1)<.5):
                raise ValueError('World-yaw phase is unresolved for a near-vertical skate')
            side.append(np.arctan2(forward[:,1],forward[:,0]))
        rate=np.arctan2(np.sin(side[1]-side[0]),np.cos(side[1]-side[0]))/(2*h)
        world_joint=centre['Q']@centre['frames'][:,joint]
        u=np.minimum(np.clip((g[controls]-lo)/.10,0.,1.),np.clip((hi-g[controls])/.10,0.,1.))
        axis=world_joint[:,2,:]
        direction=-rate[:,None]*axis*(u*u*(3-2*u))[:,None]
        mapping=csr_matrix(direction.reshape(-1,1))
        latent_update_scale=np.array([.20*np.sqrt(np.mean(np.sum(direction**2,axis=1)))])
    variables=len(columns) if mapping is None else mapping.shape[1]
    def control_delta(value):return value if mapping is None else mapping@value
    # Every source frame whose spline support changes, plus adjacent exposure.
    affected=np.asarray(curve.basis_sparse(ref.times)[:,controls].getnnz(axis=1)>0)
    frames=np.flatnonzero(affected);first=max(0,frames[0]-1);last=min(len(native),frames[-1]+2)
    frames=np.arange(first,last);stamps=set(map(int,ref.p['pts'][frames]))
    observed=[]
    for i in frames:
        for j in range(17):
            if not obs['use'][i,j]:continue
            joint=int(obs['model_joint_ids'][i,j])
            if joint not in spec['source_joints']:continue
            weight=.25 if joint<5 else 1.
            if obs['inside_tracked_shoe_proxy'][i,j]:weight*=.1
            observed.append(dict(pts=int(obs['pts'][i]),joint=joint,uv=obs['uv'][i,j].tolist(),
                role='fit',normalization_px=max(6.,float(obs['sigma_px'][i,j])),weight=weight,
                loss=spec.get('detector_robust_loss','pseudo_huber'),
                id=f"detector:{int(obs['pts'][i])}:{j}"))
    observed.extend(r for r in points['body_samples'] if r['role']=='fit' and r['pts'] in stamps)
    shoe=[r for r in points['samples'] if r['role']=='fit' and r['pts'] in stamps]
    factors=[NativeBodyPoints(p,dict(body_samples=observed))]
    regions=[r for r in points.get('body_regions',[]) if r['role']=='fit' and r['pts'] in stamps]
    if regions:factors.append(NativeBodyRegions(p,dict(body_regions=regions)))
    if spec.get('surface_points'):
        surfaces=[r for r in points.get('surface_samples',[]) if r['role']=='fit' and r['pts'] in stamps]
        if surfaces:factors.append(NativeSurfacePoints(p,dict(surface_samples=surfaces)))
    if shoe and spec.get('fit_shoe_points',True):factors.append(NativeShoePoints(p,dict(samples=shoe)))
    shoe_regions=[r for r in points.get('shoe_regions',[]) if r['role']=='fit' and r['pts'] in stamps]
    if shoe_regions and spec.get('fit_shoe_points',True):
        exclusion=NativeShoePoints(p,dict(samples=shoe_regions))
        exclusion.name='reviewed_native_shoe_regions'
        factors.append(exclusion)
    if spec.get('outlines'):
        from calibration.program_tracking.body_tracking.programme_whole_skate import WholeSkateSourceFactor
        outline_frames=frames
        for exclusion in points.get('source_outline_review_exclusions',[]):
            a,b=exclusion['seconds']
            outline_frames=outline_frames[(native[outline_frames]<a)|(native[outline_frames]>b)]
        outline_spec=dict(reference.source_document['whole_skate_source'])
        if spec.get('outline_source'):
            outline_spec.update(spec['outline_source'])
        outline=WholeSkateSourceFactor(p,outline_spec,frames=outline_frames)
        outline.refresh_each_state=True
        if outline.batches:factors.append(outline)
        if outline_spec.get('ambiguous_source_union'):
            from calibration.program_tracking.body_tracking.visible_skate_union import VisibleSkateUnion
            visible=VisibleSkateUnion(p,outline_spec,outline_frames)
            if visible.batches:factors.append(visible)
    if spec.get('source_foreground'):
        from calibration.program_tracking.body_tracking.source_foot_foreground import NativeFootForeground
        foreground=NativeFootForeground(p,spec['source_foreground'],frames)
        if len(foreground.frames):factors.append(foreground)
    if spec.get('complete_shoe_foreground'):
        from calibration.program_tracking.body_tracking.source_foot_foreground import NativeFootForeground
        specification=dict(spec['complete_shoe_foreground'],geometry='whole_skate')
        foreground=NativeFootForeground(p,specification,frames)
        if len(foreground.frames):factors.append(foreground)
    if spec.get('material_profile_foreground'):
        from calibration.program_tracking.body_tracking.source_foot_foreground import NativeFootForeground
        foreground=NativeFootForeground(p,spec['material_profile_foreground'],frames)
        foreground.name='reviewed_boot_source_profile'
        if len(foreground.frames):factors.append(foreground)
    if spec.get('reviewed_source_arcs'):
        from calibration.program_tracking.body_tracking.visible_skate_union import VisibleSkateUnion
        reviewed=VisibleSkateUnion(p,dict(spec['reviewed_source_arcs'],reviewed_open_arcs=True),frames)
        if reviewed.batches:factors.append(reviewed)
    if spec.get('source_hand_foreground'):
        from calibration.program_tracking.body_tracking.source_foot_foreground import NativeHandForeground
        foreground=NativeHandForeground(p,spec['source_hand_foreground'],frames)
        if len(foreground.frames):factors.append(foreground)
    prior_times=np.linspace(ref.times[frames[0]],ref.times[frames[-1]],4*(len(frames)-1)+1)
    if spec.get('support_inclination'):
        from calibration.program_tracking.body_tracking.continuous_motion_factors import ActualSkateInclinationRate
        policy=spec['support_inclination'];clock=prior_times+native[0]
        interval=policy['seconds']
        selected=prior_times[(clock>=interval[0])&(clock<=interval[1])]
        factors.append(ActualSkateInclinationRate(p,selected,feet=policy['feet'],
            scale_rad_s=policy.get('scale_rad_s',1.),fade_seconds=policy.get('fade_seconds',.18)))
    if spec.get('skate_ankle_acceleration'):
        from calibration.program_tracking.body_tracking.continuous_motion_factors import ActualSkateAnkleAcceleration
        policy=spec['skate_ankle_acceleration'];clock=prior_times+native[0]
        interval=policy['seconds']
        selected=prior_times[(clock>=interval[0])&(clock<=interval[1])]
        factors.append(ActualSkateAnkleAcceleration(p,selected,feet=policy['feet'],
            scale_m_s2=policy.get('scale_m_s2',8.)))
    if spec.get('skate_world_acceleration_weight',0)>0:
        from calibration.program_tracking.body_tracking.continuous_motion_factors import ActualSkateAngularAcceleration
        factors.append(ActualSkateAngularAcceleration(p,prior_times,
            feet=spec.get('skate_world_acceleration_feet',[0,1])))
    if spec.get('boot_cuff_weight',0)>0:
        from calibration.program_tracking.body_tracking.source_foot_foreground import RegisteredBootCuff
        factors.append(RegisteredBootCuff(p,prior_times,spec.get('boot_cuff_centre_weight',.15)))
    prior_pts=np.rint((prior_times+native[0])*12800).astype(np.int64)
    small=SimpleNamespace(times=prior_times,p=dict(ref.p,pts=prior_pts),
        at=ref.at,body=ref.body,source_document={})
    prior_problem=SimpleNamespace(reference=small,body=body)
    if prior_reference is None:prior_reference=ref
    prior=ActualMotionFactors(prior_problem,prior_reference,
        rotation_acceleration_loss=spec.get('rotation_acceleration_loss','pseudo_huber'))
    anatomy=None;anatomy_visibility_scale=None
    if spec.get('anatomical_domain_weight',0)>0:
        from calibration.program_tracking.body_tracking.programme_joint_limits import ProgrammeJointLimits
        structure=reference.source_document['programme_structure']
        configuration=json.loads(Path(structure['policy']).read_text(encoding='utf8'))
        anchor=dict(np.load(structure['chart_anchor']))
        # The authored branch is retained across the complete programme. Only
        # its coordinates are interpolated onto the 200 Hz prior quadrature;
        # these samples are not additional source-image observations.
        anchor['reference_twist']=np.column_stack([
            np.interp(prior_pts,anchor['source_pts'],row)
            for row in anchor['reference_twist'].T])
        anchor['source_pts']=prior_pts
        anatomy=ProgrammeJointLimits(prior_problem,configuration,np.zeros(state.size),anchor=anchor)
        if spec.get('occlusion_anatomy_multipliers'):
            # A foreground union cannot determine a hidden boot's direction:
            # folding it into the visible shin also satisfies that image term.
            # Retain the authored knee/ankle envelope more strongly where the
            # source lacks a resolved, independently visible shoe outline.
            policy=spec['occlusion_anatomy_multipliers']
            with np.load(spec['outline_source']['path']) as source_outline:
                admitted=source_outline['visible']&(source_outline['role']==1)
                partial=source_outline['partial_boundary']
                multiplier=np.where(~admitted,float(policy['missing']),
                    np.where(partial,float(policy['partial']),1.))
                source_pts=source_outline['source_pts']
            if np.any(multiplier<1.) or not np.isfinite(multiplier).all():
                raise ValueError('Occlusion anatomy multipliers must be finite and at least one')
            scale=np.ones((len(prior_pts),len(anatomy.joints),3))
            for foot,joints in enumerate(((4,7),(5,8))):
                values=np.sqrt(np.interp(prior_pts,source_pts,multiplier[:,foot]))
                scale[:,np.isin(anatomy.joints,joints)]=values[:,None,None]
            anatomy_visibility_scale=scale.ravel()
    zero=np.zeros(state.size);cached=[None,None];previous=[curve];part_ranges=[]
    accepted=[np.zeros(variables)];evaluations=[0];linearizations=[0];chart_rejections=[0]
    def evaluate(value,jac=False):
        if cached[0] is not None and np.array_equal(value,cached[0]) and (not jac or cached[1][1] is not None):return cached[1]
        full=zero.copy();full[columns]=control_delta(value)
        current=state.current_curve(full)
        current.check_continuous_edge_update(previous[0])
        rows=[];matrices=[];parts={};ranges=[];row_count=[0]
        def add(name,r,J=None,scale=1.):
            rows.append(np.asarray(r)*scale);parts[name]=float(rows[-1]@rows[-1]/2)
            ranges.append((name,row_count[0],row_count[0]+rows[-1].size));row_count[0]+=rows[-1].size
            if jac:
                selected=J[:,17+columns]
                matrices.append((selected if mapping is None else selected@mapping)*scale)
        for f in factors:
            v=f.evaluate(full,jac);name=f.name if hasattr(f,'name') else 'native_shoe_points'
            weight = (spec.get('foreground_weight',432.) if name in ('footwear_source_foreground','hand_source_foreground')
                      else spec.get('complete_foreground_weight',432.) if name=='complete_footwear_source_foreground'
                      else spec.get('reviewed_arc_weight',1.) if name=='reviewed_visible_skate_arcs'
                      else spec.get('material_profile_weight',432.) if name=='reviewed_boot_source_profile'
                      else spec['skate_world_acceleration_weight'] if name=='actual_skate_world_angular_acceleration'
                      else spec['support_inclination']['weight'] if name=='actual_skate_inclination_rate'
                      else spec['skate_ankle_acceleration']['weight'] if name=='actual_skate_ankle_acceleration'
                      else spec['boot_cuff_weight'] if name=='registered_boot_cuff'
                      else spec.get('ambiguous_outline_weight',spec.get('outline_weight',108.)) if name=='visible_ambiguous_skate_source'
                      else spec.get('outline_weight',108.) if name in ('whole_skate_source','visible_ambiguous_skate_source')
                      else spec.get('shoe_point_weight',1.) if name=='native_shoe_points' else 1.)
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError('Source factor weight must be positive and finite')
            add(name,v['residual'],v['jac'],np.sqrt(weight))
        if anatomy is not None:
            av=anatomy.evaluate(full,jac)
            active=np.flatnonzero(np.isin(anatomy.joints,spec['joints']))
            take=(np.arange(len(prior_times))[:,None,None]*len(anatomy.joints)*3+
                  active[None,:,None]*3+np.arange(3)).ravel()
            selected_scale=1. if anatomy_visibility_scale is None else anatomy_visibility_scale[take]
            add('anatomical_joint_domain',av['residual'][take]*selected_scale,
                None if not jac else av['jac'][take] if anatomy_visibility_scale is None
                else av['jac'][take].multiply(selected_scale[:,None]).tocsr(),np.sqrt(spec['anatomical_domain_weight']))
        pv=prior.evaluate(full,jac)
        for b in pv['weighted_blocks']:
            if b.name in ('finite_network_pose_prior','actual_so3_angular_acceleration'):
                n=len(prior_times) if b.name=='finite_network_pose_prior' else len(prior_times)-2
                take=(np.arange(n)[:,None,None]*66+np.array(spec['joints'])[None,:,None]*3+np.arange(3)).ravel()
                add(b.name,b.residual[take],None if not jac else b.jacobian[take],
                    np.sqrt(spec.get('motion_acceleration_weight',16.)) if b.name=='actual_so3_angular_acceleration' else 1.)
            elif spec.get('root_position'):
                add(b.name,b.residual,b.jacobian,.3 if b.name=='finite_parent_root_prior' else 1.)
        # Finite update regularization resolves depth/twist ambiguity without
        # prescribing hidden image features or copying source pixels.
        update_weight=float(spec.get('control_update_weight',1.))
        if not np.isfinite(update_weight) or update_weight<=0:
            raise ValueError('Continuous control update prior must be positive and finite')
        scale=(np.where((columns%69 if mapping is None else channels)<3,1.,.20)
               if latent_update_scale is None else latent_update_scale)*np.sqrt(update_weight)
        rows.append(value*scale);parts['finite_control_update']=float(rows[-1]@rows[-1]/2)
        ranges.append(('finite_control_update',row_count[0],row_count[0]+len(value)))
        if jac:matrices.append(eye(len(value),format='csr').multiply(scale[:,None]))
        result=(np.concatenate(rows),vstack(matrices,format='csr') if jac else None,parts)
        cached[:]=[value.copy(),result];part_ranges[:]=ranges;evaluations[0]+=1
        if jac:
            previous[0]=current;accepted[0]=value.copy();linearizations[0]+=1
            emit('source_refine_linearization',stage=spec['name'],iteration=linearizations[0],
                cost=float(result[0]@result[0]/2),parts=parts)
        return result
    emit('source_refine_start',stage=spec,controls=controls.tolist(),active_variables=variables,
        source_seconds=native[frames[[0,-1]]].tolist(),body_rows=len(observed),shoe_rows=len(shoe),
        motion_prior_samples=len(prior_times),motion_prior_samples_are_not_observations=True)
    initial=evaluate(accepted[0],True)
    if check:
        refreshing=[f for f in factors if getattr(f,'refresh_each_state',False)]
        for f in refreshing:f.refresh_each_state=False
        rng=np.random.default_rng(412);v=rng.normal(size=variables);v/=np.linalg.norm(v);h=1e-6
        num=(evaluate(h*v)[0]-evaluate(-h*v)[0])/(2*h);analytic=initial[1]@v
        for f in refreshing:f.refresh_each_state=True
        difference=np.abs(num-analytic);error=float(np.max(difference))
        emit('source_refine_derivative_check',stage=spec['name'],max_error=error,
             parts={name:float(np.max(difference[lo:hi],initial=0.)) for name,lo,hi in part_ranges})
        if error>2e-4:raise AssertionError('Source refinement analytic derivative mismatch')
    def fun(value):
        try:return evaluate(value)[0]
        except RotationChartDomainError as error:
            chart_rejections[0]+=1
            emit('source_refine_rejected_chart_trial',stage=spec['name'],reason=str(error))
            return np.full_like(initial[0],1e6*(1.+np.linalg.norm(value)))
    def jac(value):return evaluate(value,True)[1]
    start=np.zeros(variables)
    if spec.get('world_yaw_phase'):
        trials=[]
        for seconds in spec['world_yaw_phase'].get('initial_phase_seconds',[0.]):
            value=np.array([float(seconds)]);residual=fun(value)
            trials.append(dict(seconds=float(seconds),cost=float(residual@residual/2)))
        chosen=min(trials,key=lambda row:row['cost'])
        start[:]=chosen['seconds']
        emit('source_world_yaw_phase_search',stage=spec['name'],trials=trials,
             selected_initial_seconds=float(start[0]),source_clock_unchanged=True)
    result=least_squares(fun,start,jac=jac,x_scale='jac',
        ftol=spec.get('solver_ftol',2e-6),xtol=2e-6,gtol=2e-5,tr_solver='lsmr')
    full=zero.copy();full[columns]=control_delta(result.x);final=state.current_curve(full)
    if support_rule:verify(ref,final,support_rule)
    values=[f.evaluate(full) for f in factors]
    emit('source_refine_complete',stage=spec['name'],cost_before=float(initial[0]@initial[0]/2),
        cost_after=float(result.cost),evaluations=result.nfev,optimality=float(result.optimality),
        status=int(result.status),message=result.message,metrics=[v['metrics'] for v in values])
    if spec.get('world_yaw_phase'):
        emit('source_world_yaw_phase_result',stage=spec['name'],phase_seconds=float(result.x[0]),
             source_clock_unchanged=True,semantics='one scalar in a world-yaw control basis; not a source time shift')
    if chart_rejections[0] and result.optimality>2e-5 and initial[0]@initial[0]/2-result.cost>2e-6*max(1.,result.cost):
        # Rebase at the accepted curve, preserving every source-time pose.
        # An archived coordinate seam is not a bound on an athlete's motion.
        rebased=AbsoluteMotionCurve(final.knots,final.position_controls,final.rotation_controls,lift(final))
        emit('source_refine_chart_reanchor',stage=spec['name'],source_pose_unchanged=True,
            finite_pose_prior_retained=True,reason='Accepted progress reached previous numerical chart boundary')
        return stage(reference,camera,packet,rebased,points,obs,spec,emit,False,prior_reference)
    return final


def run(args):
    candidate=args.output/'candidate.npz'
    if candidate.exists():
        if not (getattr(args,'continue_in_place',False)
                and args.parameter.parent.resolve()==args.output.resolve()
                and args.parameter.resolve()!=candidate.resolve()
                and digest(args.parameter)==digest(candidate)):
            raise FileExistsError('Continuation requires an exact archived candidate in this output directory')
    args.output.mkdir(exist_ok=True)
    r,c,_,document,packet=bind(args.prepared,args.source,args.parameter)
    r.source_document=document
    points=json.loads(args.points.read_text(encoding='utf8'));obs=dict(np.load(args.observations))
    np.testing.assert_array_equal(obs['pts'],r.p['pts'])
    parent=curve_from_packet(packet);baseline=curve_from_packet(dict(np.load(args.baseline)))
    if getattr(args,'restart_motion_from_baseline',False):
        curve=AbsoluteMotionCurve(baseline.knots,baseline.position_controls,baseline.rotation_controls,lift(baseline))
        initialization=dict(motion_restarted_from_baseline=True,
            reason='Reject source-only fitting loops that introduced unsupported joint twisting; refit the complete programme with coupled anatomy and temporal priors')
    elif args.keep_initialization:
        curve=AbsoluteMotionCurve(parent.knots,parent.position_controls,parent.rotation_controls,lift(parent))
        initialization=dict(saved_curve_reanchored_without_geometric_change=True)
    else:
        curve,initialization=restored_curve(parent,baseline,r.p['pts'][0]/12800.)
    if args.restore_occluded_arms:
        curve,record=restore_occluded_arms(curve,baseline,r.p['pts'][0]/12800.)
        initialization['source_reviewed_occlusion_restore']=record
    if args.resolve_subframes:
        curve,record=source_resolved_initialization(curve,baseline,r.p['pts'][0]/12800.)
        initialization['source_resolved_motion']=record
    if getattr(args,'reinitialize_joints_key',None):
        if not np.array_equal(curve.knots,baseline.knots):
            baseline,record=refine_native_rotation_clock(baseline,r.times)
            np.testing.assert_array_equal(curve.knots,baseline.knots)
            initialization['retained_branch_native_clock']=record
        curve,record=reinitialize_reviewed_joints(curve,baseline,r.p['pts'][0]/12800.,
                                                points[args.reinitialize_joints_key])
        initialization['reviewed_turn_branch_retry']=record
    if getattr(args,'native_rotation_clock',False):
        curve,record=refine_native_rotation_clock(curve,r.times)
        initialization['native_rotation_clock']=record
    if getattr(args,'initialize_occluded_cuffs',False):
        entries=points[args.interval_key]
        curve,record=registered_cuff_initialization(curve,r,entries[0]['outline_source']['path'])
        initialization['registered_cuff_branch']=record
    specs=[dict(name='opening_support_and_raised_arm',seconds=[23.84,24.94],
                joints=[1,2,4,5,7,8,14,17,19],source_joints=list(range(5,17))),
           dict(name='spin_occlusion_arm_continuity',seconds=[84.96,85.64],
                joints=[13,14,16,17,18,19,20,21],source_joints=list(range(5,11))),
           dict(name='spin_visible_right_arm',seconds=[84.98,85.26],
                joints=[14,17,19],source_joints=[6,8,10]),
           dict(name='spin_visible_right_arm_exit',seconds=[85.36,85.58],
                joints=[14,17,19],source_joints=[6,8,10]),
           dict(name='airborne_left_skate_continuity',seconds=[106.90,107.28],
                joints=[1,4,7],source_joints=[13,15]),
           dict(name='jump_exit_both_skates',seconds=[77.94,78.48],
                joints=[1,2,4,5,7,8],source_joints=list(range(11,17))),
           dict(name='live_cut_end_body_and_both_skates',seconds=[101.20,101.70],
                joints=list(range(22)),root_position=True,source_joints=list(range(17))),
           dict(name='ending_both_skates',seconds=[188.10,188.70],
                joints=[1,2,4,5,7,8],source_joints=list(range(11,17))),
           dict(name='opening_skate_directions',seconds=[24.10,24.72],
                joints=[2,5,7,8],source_joints=[14,15,16]),
           dict(name='cut_end_skate_directions',seconds=[101.12,101.70],
                joints=[1,2,4,5,7,8],source_joints=list(range(11,17))),
           dict(name='spin_free_skate_direction',seconds=[85.08,85.36],
                joints=[1,4,7],source_joints=[13,15],common_turn=True)]
    if args.with_outlines:
        for spec in specs:spec['outlines']=True
    if getattr(args,'interval_key',None):
        specs=points[args.interval_key]
        if not specs or any(not (22.12<=s['seconds'][0]<s['seconds'][1]<=188.70) for s in specs):
            raise ValueError('Reviewed intervals must lie within the current source programme')
        if args.with_outlines:
            for spec in specs:spec['outlines']=True
    if args.only:
        specs=[s for s in specs if s['name'] in args.only]
        if len(specs)!=len(args.only):raise ValueError('Unknown source refinement stage')
    started=time.perf_counter()
    with (args.output/'progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        def emit(event,**kw):
            row=dict(event=event,elapsed_seconds=time.perf_counter()-started,**kw)
            log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if 'metrics' not in row:print(json.dumps(row,ensure_ascii=False),flush=True)
        receipt=dict(parent=str(args.parameter),parent_sha256=digest(args.parameter),
            baseline=str(args.baseline),baseline_sha256=digest(args.baseline),
            points=str(args.points),points_sha256=digest(args.points),
            observations=str(args.observations),observations_sha256=digest(args.observations),
            initialization=initialization,shared_geometry_and_camera_unchanged=True,
            complete_native_curve=True,source_time_scope=[22.12,188.68],user_accepted=False,
            workers=getattr(args,'workers',1),source_stages=[s['name'] for s in specs])
        emit('continuous_source_revision',**receipt)
        def save_current(completed):
            if 'programme_double_support' in document:
                from opening_support import verify
                contact = verify(r, curve, document['programme_double_support'])
                emit('verified_double_support', completed=completed, **contact)
            body_size=len(curve.position_controls)*69
            shoe_size=int(packet['relative_skate_parameter_count'])
            result=dict(packet);result.update(x=np.r_[packet['x'][:17],np.zeros(body_size+shoe_size)],
                knots=curve.knots,absolute_knots=curve.knots,body_parameter_count=np.array(body_size),
                relative_skate_parameter_offset=np.array(17+body_size),
                absolute_position_controls=curve.position_controls,absolute_rotation_controls=curve.rotation_controls,
                absolute_seed_position_controls=curve.position_controls,absolute_seed_rotation_controls=curve.rotation_controls,
                absolute_rotation_edge_reference_quaternion=lift(curve),
                relative_skate_seed_rotation_controls=packet['relative_skate_rotation_controls'],
                continuous_source_refine_json=np.array(json.dumps(dict(receipt,
                    completed_stages=completed,total_stages=len(specs)),sort_keys=True)))
            pending=args.output/'candidate.pending.npz'
            with pending.open('xb') as f:np.savez_compressed(f,**result)
            replace_file(pending,args.output/'candidate.npz')
        if getattr(args,'workers',1)>1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
            from queue import Empty
            groups=independent_stage_groups(curve,specs,r.p['pts'][0]/12800.)
            emit('continuous_parallel_blocks',workers=args.workers,
                 groups=[[s['name'] for s in group] for group in groups],
                 independence='Disjoint entire spline support including 40ms image halo; merge controls and rotation edges exactly')
            completed=0
            context=mp.get_context('spawn')
            queue=context.Queue()
            try:
                with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=initialize_stage_worker,
                                         initargs=(args,queue)) as pool:
                    for group_number,group in enumerate(groups):
                        base=curve;children=[]
                        futures={pool.submit(run_stage_worker,base,s):s['name'] for s in group}
                        while futures:
                            done,_=wait(futures,timeout=.25,return_when=FIRST_COMPLETED)
                            while True:
                                try:event,kw=queue.get_nowait()
                                except Empty:break
                                emit(event,**kw)
                            for future in done:
                                name=futures.pop(future)
                                try:child=future.result()
                                except BaseException as error:
                                    import traceback
                                    emit('continuous_stage_failed',stage=name,error=repr(error),
                                         traceback=traceback.format_exc())
                                    for pending_future in futures:pending_future.cancel()
                                    raise
                                children.append(child)
                                curve=merge_independent_curves(base,children);completed+=1
                                changed=np.any(child.rotation_controls!=base.rotation_controls,axis=(1,2,3))
                                changed|=np.any(child.position_controls!=base.position_controls,axis=1)
                                ids=np.flatnonzero(changed)
                                merge_error=0.
                                if len(ids):
                                    frames=np.flatnonzero(np.asarray(base.basis_sparse(r.times)[:,ids].getnnz(axis=1)>0))
                                    tau=c.readout_times(frames)
                                    clock=(r.times[frames,None]+tau[:,None]*np.linspace(-.5,.5,5)).ravel()
                                    from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
                                    a,b=evaluate_curve(child,clock),evaluate_curve(curve,clock)
                                    merge_error=max(float(np.max(abs(a[k]-b[k]))) for k in ('P','rotations'))
                                if merge_error>2.e-9:raise ValueError('Parallel merge changed the solved source motion')
                                try:
                                    save_current(completed)
                                except BaseException as error:
                                    emit('continuous_checkpoint_failed',stage=name,error=repr(error),
                                         pending=str(args.output/'candidate.pending.npz'),
                                         completed_stages=completed,total_stages=len(specs))
                                    for pending_future in futures:pending_future.cancel()
                                    # Python 3.11 has no public terminate_workers API.
                                    # Stop only this executor's children; do not wait
                                    # through queued work after checkpoint I/O failed.
                                    for worker in pool._processes.values():worker.terminate()
                                    raise
                                emit('continuous_stage_saved',stage=name,group=group_number,
                                     completed_stages=completed,total_stages=len(specs),
                                     parameter=str(candidate),sha256=digest(candidate),
                                     merged_native_readout_pose_error=merge_error)
                        while True:
                            try:event,kw=queue.get_nowait()
                            except Empty:break
                            emit(event,**kw)
            finally:
                queue.close()
                queue.join_thread()
        else:
            for number,spec in enumerate(specs):
                curve=stage(r,c,packet,curve,points,obs,spec,emit,
                            args.check_derivatives and spec.get('check_derivatives',True))
                if getattr(args,'checkpoint_each_stage',False):
                    save_current(number+1)
                    emit('continuous_stage_saved',completed_stages=number+1,total_stages=len(specs),
                        parameter=str(candidate),sha256=digest(candidate))
        save_current(len(specs))
        emit('continuous_source_saved',parameter=str(args.output/'candidate.npz'),
            sha256=digest(args.output/'candidate.npz'),source_exposures=len(r.times))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','source','parameter','baseline','points','observations','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--check-derivatives',action='store_true')
    parser.add_argument('--keep-initialization',action='store_true')
    parser.add_argument('--reinitialize-joints-key',
                        help='Source-review JSON key for continuous joint initialization windows')
    parser.add_argument('--native-rotation-clock',action='store_true',
                        help='Subdivide the full continuous curve on every original source exposure')
    parser.add_argument('--restart-motion-from-baseline',action='store_true')
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--interval-key',help='Reviewed interval list in the current source-point document')
    parser.add_argument('--continue-in-place',action='store_true')
    parser.add_argument('--checkpoint-each-stage',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--initialize-occluded-cuffs',action='store_true')
    parser.add_argument('--with-outlines',action='store_true')
    parser.add_argument('--restore-occluded-arms',action='store_true')
    parser.add_argument('--resolve-subframes',action='store_true')
    run(parser.parse_args())
