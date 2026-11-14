"""Prepare and compare the ExMR programme on the common body/skate solver."""
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil

import cv2
import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.sparse.linalg import lsmr
from scipy.spatial.transform import Rotation, RotationSpline

from .body_skate_fit import Problem
from .deliver_body_skate_fit import saved_ref
from .foot_socket import root_shift, rotate_bone, unit
from .inputs import PROGRAMME, PROJECT, atomic_json, atomic_npz, digest, emit, read
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, _log, clamped_cubic_knots, greville_times,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, curve_metadata, evaluate_curve
from calibration.program_tracking.body_tracking.camera_bridge import native_to_rectified, rectified_intrinsics
from calibration.program_tracking.body_tracking.action_clip_input import camera25_from_snapshot
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SHOE_SCHEMA
from calibration.program_tracking.body_tracking.skate_physics import fk_state


HIPS = np.array([1, 2])
KNEES = np.array([4, 5])
ANKLES = np.array([7, 8])
CASES = (29.68, 48.58, 66.24, 66.86, 83.52, 84.54, 84.64, 84.76, 84.80, 85.00, 86.94, 87.14,
         89.04, 89.06, 90.12, 92.04, 92.40, 92.70, 97.06, 100.84, 101.66, 110.18,
         136.16, 151.02, 159.38, 159.58, 167.34, 167.52)


def statistics(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(count=0)
    return dict(count=len(x), median=float(np.median(x)), p90=float(np.quantile(x, .9)), max=float(np.max(x)))


def baseline(visual, previous, job):
    """Verify the displayed candidate through its saved wear and socket adapters."""
    problem = Problem(visual, job)
    controls = previous / "controls.npz"
    ref = saved_ref(problem, controls)
    ratio = ref.binding.ratio
    stored = load(previous / "fitted/result.npz")
    parts = {name: [] for name in ("P", "Q", "local_R", "R", "O", "native_world", "frames")}
    body_pixels = []
    for frames in np.array_split(np.arange(len(ref.times)), 80):
        t = problem.times[frames].ravel()
        sample = ref.at(t)
        state = fk_state(ref, sample, np.zeros((len(t), 69)))
        for name in parts:
            a = state[name]
            parts[name].append(a.reshape(len(frames), 5, *a.shape[1:])[:, 2])
        hand = {} if "hand_local_R" not in sample else {"hand_local_R":sample["hand_local_R"]}
        local = ref.body.forward_rotations(state["local_R"], **hand)[0]
        world = state["P"][:, None] + np.einsum("nij,nkj->nki", state["Q"], local)
        body_pixels.append(moving_projection(problem.camera, world.reshape(len(frames), 5, 17, 3), frames)["uv"])
    state = {name: np.concatenate(v) for name, v in parts.items()}
    checks = {}
    for old, new, scale in (("native22_world_mid", "native_world", ratio),
                             ("skate_R_world_mid", "R", 1.), ("skate_origin_world_mid", "O", ratio)):
        err = float(np.abs(stored[old] - state[new] * scale).max())
        if err > 1.e-7:
            raise ValueError(f"Displayed baseline replay differs in {old}: {err}")
        checks[old] = err
    checks["COCO_uv"] = float(np.nanmax(np.abs(stored["COCO_uv"] - np.concatenate(body_pixels))))
    mount, _, _ = ref.binding.parameters()
    ankle = np.einsum("nfij,nfi->nfj", state["R"], state["native_world"][:, ANKLES] - state["O"])
    pivot = np.median(ankle, axis=0)
    np.testing.assert_allclose(ankle, np.broadcast_to(pivot, ankle.shape), atol=1.e-8, rtol=0)
    relative = (state["Q"][:, None] @ state["frames"][:, ANKLES] @ mount).swapaxes(-1, -2) @ state["R"]
    # Exact baseline evaluation includes between-exposure points for motion diagnostics.
    mid = (ref.times[:-1] + ref.times[1:]) * .5
    bridge = [ref.at(t)["P"] for t in np.array_split(mid, 64)]
    p_step = np.linalg.norm(np.diff(np.concatenate(bridge), axis=0), axis=1) * ratio
    atomic_npz(job / "displayed_baseline.npz", pts=problem.i.pts, time_base=problem.i.tb, **state,
               mount=mount, pivot=pivot, shared=ref.binding.x,
               fixed_R0=ref.binding.R0, fixed_t0=ref.binding.t0,
               fixed_base_parameters_json=np.array(json.dumps(asdict(ref.binding.base_parameters))),
               fixed_upper_height_log=np.array(ref.binding.upper_height_log))
    atomic_json(job / "displayed_baseline_replay.json", dict(controls=str(controls), controls_sha256=digest(controls),
                camera_sha256=problem.i.camera_sha, source_exposures=len(ref.times),
                native_projection_nodes=problem.times.size, between_exposure_samples=len(mid),
                maximum_saved_result_difference=checks, midpoint_root_step_m=statistics(p_step),
                registered_ankle_in_shoe_solver=pivot.tolist(),
                prior_relative_shoe_deg=[statistics(np.rad2deg(np.linalg.norm(_log(relative[:,f]),axis=-1))) for f in (0,1)]))
    emit("displayed_baseline_replayed", exposures=len(ref.times), checks=checks)


def directed_rows(packet):
    # Two ends of the outsole chord read from the saved, numbered native crop sheet.
    # Nearly frontal views are excluded because their projected chord is ambiguous.
    readings = ((84.54,(175,108),(55,118)), (84.72,(43,113),(177,122)),
                (84.76,(40,110),(185,122)), (84.80,(45,109),(184,123)),
                (84.84,(46,110),(183,123)), (84.88,(43,112),(182,126)),
                (84.94,(49,115),(176,129)))
    result = []
    for seconds, heel, toe in readings:
        stamp = round(seconds * 12800)
        i = np.searchsorted(packet["source_pts"], stamp)
        region = packet["region"][i, 1]
        center = np.rint(np.median(region, axis=0)).astype(int)
        half = np.array([int(max(65,np.ptp(region[:,0])*.85)), int(max(51,np.ptp(region[:,1])*.8))])
        lo = np.maximum(0, center - half)
        hi = np.minimum([1920, 1080], center + half)
        scale = (hi - lo) / [240., 182.]
        uv = lambda point: ((np.asarray(point)+.5)*scale + lo - .5).tolist()
        result.append(dict(pts=stamp, foot=1, heel_uv=uv(heel), toe_uv=uv(toe),
                           sigma_px=float(12*np.linalg.norm(scale)),
                           role="fit", source="visible heel/toe of the outsole in the native source; unsigned mask is separate"))
    return result


def source_roles(visual, previous, job, camera_sha, *, development_windows=()):
    packet = load(visual / "whole_follow_evidence.npz")
    history = load(previous / "held_out_silhouette.npz")
    np.testing.assert_array_equal(packet["source_pts"], history["source_pts"])
    visible = (packet["status"] == 1) & (packet["extent"] > 0)
    visible &= ~(packet["partial_occlusion"] | packet["identity_unresolved"] | packet["skate_occluded"])
    historical = np.isfinite(history["after_px"][..., 2])
    # Same foot, source anchor and neighbouring exposure stay with development.
    role = np.zeros(visible.shape, np.uint8)
    measure = np.zeros(visible.shape)
    for foot in (0, 1):
        stamps = packet["source_pts"]
        dev = historical[:, foot].copy()
        for first,last in development_windows:
            dev |= (stamps>=round(first*12800)) & (stamps<=round(last*12800))
        for offset in (256, 512):
            dev |= np.isin(stamps, stamps[historical[:,foot]] + offset)
            dev |= np.isin(stamps, stamps[historical[:,foot]] - offset)
        anchor = packet["tracking_anchor_pts"][:, foot]
        group = np.where(anchor >= 0, anchor, stamps)
        while True:
            protected = np.union1d(group[dev], stamps[dev])
            expanded = dev | np.isin(group, protected) | np.isin(stamps, protected)
            if np.array_equal(expanded,dev):
                break
            dev = expanded
        fit = visible[:, foot] & ~dev
        # Keep the original three-exposure contour sampling density and admit
        # all eligible source observations in the audited reverse turn.
        dense = (stamps >= round(84.54*12800)) & (stamps <= round(85.00*12800))
        selected = fit & ((np.arange(len(stamps)) % 3 == 0) | dense)
        role[dev, foot] = 2
        role[selected, foot] = 1
        identity, count = np.unique(group[selected], return_counts=True)
        measure[selected, foot] = .02 / np.sqrt(count[np.searchsorted(identity, group[selected])])
        if np.intersect1d(group[selected], group[dev]).size:
            raise ValueError("Whole-skate source family crosses the development/fit split")
    agreement = np.nan_to_num(packet["mask_agreement"], nan=0.)
    sigma = 3. / np.sqrt(np.clip(agreement, .25, 1.))
    dest = job / "whole_skate_roles.npz"
    atomic_npz(dest, region=packet["region"], source_pts=packet["source_pts"], shot_ids=packet["shot_ids"],
               time_base=packet["time_base"], role=role, sigma_px=sigma, measure_s=measure,
               tracking_anchor_pts=packet["tracking_anchor_pts"], visible=visible, historical_development=historical,
               camera_sha256=np.array(camera_sha), source_video_sha256=packet["source_video_sha256"])
    receipt = dict(roles={"0":"unselected_or_unknown","1":"fit","2":"development"},
                   visible=int(visible.sum()), fit=int((role == 1).sum()), development=int((role == 2).sum()),
                   historical_development=int(historical.sum()), independent_final_check=0,
                   input_sha256=digest(visual / "whole_follow_evidence.npz"),
                   by_shot={str(s):int((role[packet["shot_ids"] == s] == 1).sum()) for s in np.unique(packet["shot_ids"])})
    atomic_json(job / "whole_skate_role_receipt.json", receipt)
    return dict(path=str(dest), sha256=digest(dest), directed_boot_rows=directed_rows(packet))


def restore_structure_inputs(doc, recovery_path):
    structure = doc["programme_structure"]
    missing = [key for key in ("chart_anchor", "surface_witnesses")
               if not Path(structure[key]).is_file()]
    if not missing:
        return
    restored = read(recovery_path.parent / "continuation_prepared/source_evidence.json")["programme_structure"]
    for key in missing:
        path = Path(restored[key])
        if digest(path) != restored[f"{key}_sha256"]:
            raise ValueError(f"Recovered structure input changed: {path}")
        structure[key] = str(path)
        structure[f"{key}_sha256"] = restored[f"{key}_sha256"]


def pose_source(exmr, pts, shots):
    final, prefoot = (load(exmr / f"exmr_{v}.npz") for v in ("final", "prefoot"))
    selected = load(exmr / "prepared.npz")
    stamps = selected["source_pts"]
    for packet in (final, prefoot):
        np.testing.assert_array_equal(packet["source_pts"], stamps)
    body = final["body_pose"].copy()
    body[:, 18:24] = prefoot["body_pose"][:, 18:24]
    raw = np.concatenate(((final["R_w2c"].transpose(0,2,1)
                          @ Rotation.from_rotvec(final["global_orient"]).as_matrix())[:,None],
                          Rotation.from_rotvec(body.reshape(-1,3)).as_matrix().reshape(-1,21,3,3)),axis=1)
    hand = np.stack([final[k] for k in ("left_hand_pose", "right_hand_pose")], axis=1)
    hand = Rotation.from_rotvec(hand.reshape(-1,3)).as_matrix().reshape(len(stamps),30,3,3)
    joined = np.concatenate((raw, hand), axis=1)
    result = np.empty((len(pts), 52, 3, 3))
    filled = np.zeros(len(pts), bool)
    for shot in np.unique(shots):
        indices = np.flatnonzero(selected["shot_ids"] == shot)
        frames = np.flatnonzero(shots == shot)
        np.testing.assert_array_equal(stamps[indices], pts[frames[::2]])
        t = (stamps[indices] - pts[0]) / 12800.
        query = np.clip((pts[frames] - pts[0]) / 12800., t[0], t[-1])
        result[frames] = np.stack([RotationSpline(t, Rotation.from_matrix(joined[indices, j]))(query).as_matrix()
                                   for j in range(joined.shape[1])], axis=1)
        filled[frames] = True
    assert filled.all()
    return result[:, 0], result[:, 1:22], result[:, 22:].reshape(len(pts), 2, 15, 3, 3)


def root_on_original_camera(problem, Q, local, hand, start, obs):
    """Translation seed from pixels and the existing skate reconstruction's depth."""
    camera = problem.camera
    pose = camera.midpoint
    rays = camera25_from_snapshot(pose)
    local_points = problem.ref.body.forward_rotations(local, hand_local_R=hand)[0]
    points = np.einsum("nij,nkj->nki", pose["rotation"] @ Q, local_points)
    centre = pose["centre"] * float(problem.ref.p["metres_per_scene_unit"])
    guess = np.einsum("nij,nj->ni", pose["rotation"], start - centre)
    answer = guess.copy()
    for i in range(len(answer)):
        use = obs["use"][i] & np.isfinite(obs["uv"][i]).all(axis=-1)
        if use.sum() < 4 or guess[i,2] <= 0:
            continue
        pixels, good = native_to_rectified(obs["uv"][i,use], rays[i])
        if good.sum() < 4:
            continue
        K = rectified_intrinsics(rays[i])
        direction = (pixels[good] - K[:2,2]) / np.diag(K)[:2]
        loc = points[i,obs["model_joint_ids"][i,use]][good] + guess[i]
        e = direction * loc[:,2,None] - loc[:,:2]
        sigma = np.maximum(2., obs["sigma_px"][i,use][good])
        weight = 1. / sigma / np.sqrt(1. + (np.linalg.norm(e,axis=1)/.08)**2)
        answer[i,:2] += np.average(e, axis=0, weights=weight)
    return centre + np.einsum("nji,nj->ni", pose["rotation"], answer)


def seat_legs(problem, P, Q, local, old):
    """Seed the two-link chains from saved skate poses before the common solve."""
    ref = problem.ref
    pose = fk_state(ref, dict(P=P, Q=Q, local_R=local), np.zeros((len(P),69)))
    desired = old["O"] + np.einsum("nfij,fj->nfi", old["R"], old["pivot"])
    h,k,a = (pose["native"][:, ids] for ids in (HIPS, KNEES, ANKLES))
    target = np.einsum("nji,nfj->nfi",Q,desired-P[:,None])
    thigh,shin = k-h,a-k
    l1,l2 = (np.linalg.norm(x,axis=-1) for x in (thigh,shin))
    reach = l1+l2
    padding = np.minimum(.003,np.maximum(0.,reach-np.linalg.norm(a-h,axis=-1))*.5)
    centres=target-h
    separation=np.linalg.norm(centres[:,0]-centres[:,1],axis=-1)
    slack=np.maximum(0.,reach.sum(axis=1)-separation)
    room = reach - padding * np.minimum(1.,.9*slack/np.maximum(padding.sum(axis=1),1.e-12))[:,None]
    possible = (separation <= room.sum(axis=1)-1.e-10) & (separation >= np.abs(room[:,0]-room[:,1])+1.e-10)
    displacement=np.zeros_like(P)
    displacement[possible]=root_shift(centres[possible],room[possible])
    rays=target-displacement[:,None]-h
    used=np.clip(np.linalg.norm(rays,axis=-1),np.abs(l1-l2)+1.e-6,reach-1.e-6)
    v=unit(rays)
    ankle=h+used[...,None]*v
    offset=(l1*l1-l2*l2+used*used)/(2*used)
    side=thigh - (thigh*v).sum(axis=-1,keepdims=True)*v
    ambiguous=np.linalg.norm(side,axis=-1)<1.e-9
    side[ambiguous]=np.cross(pose["frames"][:,KNEES][...,0][ambiguous],v[ambiguous])
    knee=h+offset[...,None]*v+np.sqrt(np.maximum(0.,l1*l1-offset*offset))[...,None]*unit(side)
    hip_R=rotate_bone(thigh,knee-h)@pose["frames"][:,HIPS]
    knee_R=rotate_bone(shin,ankle-knee)@pose["frames"][:,KNEES]
    ankle_R=Q[:,None].swapaxes(-1,-2)@old["R"]@old["mount"].swapaxes(-1,-2)
    out=local.copy()
    out[:,HIPS-1]=pose["frames"][:,ref.body.parents[HIPS]].swapaxes(-1,-2)@hip_R
    out[:,KNEES-1]=hip_R.swapaxes(-1,-2)@knee_R
    out[:,ANKLES-1]=knee_R.swapaxes(-1,-2)@ankle_R
    return P+np.einsum("nij,nj->ni",Q,displacement), out, dict(
        exact_reach_exposures=int(possible.sum()), maximum_hip_translation_m=float(np.linalg.norm(displacement,axis=1).max()),
        initial_ankle_position_discrepancy_m=statistics(np.linalg.norm(ankle-(target-displacement[:,None]),axis=-1)))


def direct_spin_seed(problem, current, old, rows, *, episode):
    """Choose a continuous source-supported yaw branch for the seated-leg initialization."""
    if len(rows)<2:
        raise ValueError("Directed spin initialization requires two native source anchors")
    ref=saved_ref(problem,current)
    pts=problem.i.pts
    frame=np.searchsorted(pts,[r["pts"] for r in rows])
    times=problem.ref.times
    foot=episode["foot"]
    if foot not in (0,1) or any(row["foot"]!=foot for row in rows):
        raise ValueError("Directed spin rows must use the source support foot")
    np.testing.assert_array_equal(pts[frame],[row["pts"] for row in rows])
    basis=old["R"][:,foot]
    yaw=np.unwrap(np.arctan2(basis[:,1,0],basis[:,0,0]))
    pivot=old["O"][:,foot]+basis@old["pivot"][foot]
    p=ref.binding.packet()
    points=np.array([[0.,0.,p["floor"][0]],[p["xs"][-1],0.,p["floor"][-1]]])-old["pivot"][foot]
    choices=np.linspace(yaw[frame[0]]-np.pi*.5,yaw[frame[0]]+2*np.pi,541)
    allowed=np.diff(choices)[0]
    progress=choices[None]-choices[:,None]
    permitted=(progress>=-allowed*.5)&(progress<np.pi)
    goal=np.array([r["toe_uv"] for r in rows])-np.array([r["heel_uv"] for r in rows])

    def uv(i, target):
        phi=np.atleast_1d(target)-yaw[i]
        rotations=Rotation.from_rotvec(np.column_stack((np.zeros((len(phi),2)),phi))).as_matrix()@basis[i]
        world=pivot[i]+np.einsum("nij,kj->nki",rotations,points)
        uv=problem.camera.project(world,np.full(len(phi),i))
        return uv[:,1]-uv[:,0]

    err=np.stack([np.linalg.norm(uv(i,choices)-target,axis=1) for i,target in zip(frame,goal,strict=True)])
    sigma=np.asarray([r["sigma_px"] for r in rows])
    pelvis=np.unwrap(np.arctan2(old["Q"][:,1,0],old["Q"][:,0,0]))
    step=np.diff(pelvis[frame])
    cost=(err[0]/sigma[0])**2
    parents=[]
    for e,radius,turn in zip(err[1:],sigma[1:],step,strict=True):
        # The source torso turn supplies a finite transition prior. It leaves
        # room for the skate to lead or lag without rewarding a catch-up pulse.
        phase_cost=((progress-turn)/(.18+.35*abs(turn)))**2
        transitions=np.where(permitted,cost[:,None]+phase_cost,np.inf)
        parent=np.argmin(transitions,axis=0)
        parents.append(parent)
        cost=transitions[parent,np.arange(len(cost))]+(e/radius)**2
    best=[int(np.argmin(cost))]
    for previous in parents[::-1]:
        best.append(int(previous[best[-1]]))
    best=best[::-1]
    before=np.array([np.linalg.norm(uv(i,yaw[i])[0]-target) for i,target in zip(frame,goal,strict=True)])
    selected=choices[best]
    after=err[np.arange(len(frame)),best]
    kept=bool(np.isfinite(cost[best[-1]]) and np.median(after)<np.median(before))
    if kept:
        a,b=frame[[0,-1]]
        signal=yaw.copy()
        signal[a:b+1]=PchipInterpolator(times[frame],selected)(times[a:b+1])
        start,stop=np.searchsorted(pts,[episode["start_pts"],episode["end_pts_inclusive"]])
        if not 0<=start<a<b<stop<len(pts):
            raise ValueError("Directed anchors must lie inside the observed support episode")
        np.testing.assert_array_equal(pts[[start,stop]],
                                      [episode["start_pts"],episode["end_pts_inclusive"]])
        # Carry the measured phase into the episode rather than pulling back
        # towards the old yaw immediately after the last signed image chord.
        # Torso turn is an initialization cue. The main factors retain the
        # independent shoe motion, anatomy, support and source residuals.
        for edge,ids,endpoint in ((a,np.arange(start,a),start),(b,np.arange(b+1,stop+1),stop)):
            guided=signal[edge]+pelvis[ids]-pelvis[edge]
            old_branch=yaw[ids]+2*np.pi*np.round((guided-yaw[ids])/(2*np.pi))
            u=np.clip(np.abs(times[ids]-times[edge])/abs(times[endpoint]-times[edge]),0.,1.)
            taper=u*u*(3.-2.*u)
            signal[ids]=guided*(1.-taper)+old_branch*taper
        delta=signal-yaw
        R=Rotation.from_rotvec(np.column_stack((np.zeros((len(yaw),2)),delta))).as_matrix()@basis
        old=dict(old,R=old["R"].copy(),O=old["O"].copy())
        old["R"][:,foot]=R
        old["O"][:,foot]=pivot-R@old["pivot"][foot]
    return old,dict(kept=kept,source_pts=pts[frame].tolist(),before_chord_px=before.tolist(),
                    seed_chord_px=after.tolist(),selected_heading_deg=np.rad2deg(selected).tolist())


def make_curve(clock, P, Q, local, initial_knots):
    tracks = np.concatenate((Q[:,None],local),axis=1)
    curves = [RotationSpline(clock,Rotation.from_matrix(tracks[:,j])) for j in range(22)]
    breaks = np.unique(initial_knots)
    native_angles = np.linalg.norm(_log(tracks[:-1].swapaxes(-1,-2)@tracks[1:]),axis=-1)
    while True:
        knots=clamped_cubic_knots(breaks)
        gt=greville_times(knots)
        additions=[]
        for a,b in zip(gt[:-1],gt[1:]):
            i,j=np.searchsorted(clock,[a,b])
            if i+1<j and np.any(native_angles[i:j-1].sum(axis=0)>np.pi-.1):
                additions.append((a+b)*.5)
        if not additions:
            break
        breaks=np.unique(np.r_[breaks,additions])
    design=BSpline.design_matrix(clock,knots,3,extrapolate=True).tocsr()
    # Near dense intervals, exposure-midpoint values alone leave cubic
    # controls underdetermined between frames. Fit the moving interpolant
    # there too, and damp increments about its own Greville values.
    position=PchipInterpolator(clock,P,axis=0)
    between=(clock[:-1]+clock[1:])*.5
    position_clock=np.unique(np.r_[clock,between,gt])
    position_design=BSpline.design_matrix(position_clock,knots,3,extrapolate=True).tocsr()
    prior=position(np.clip(gt,clock[0],clock[-1]))
    correction=position(np.clip(position_clock,clock[0],clock[-1]))-position_design@prior
    pc=prior+np.stack([lsmr(position_design,correction[:,k],damp=.02,atol=1.e-9,btol=1.e-9,maxiter=400)[0]
                       for k in range(3)],axis=1)
    r=np.stack([curve(np.clip(gt,clock[0],clock[-1])).as_matrix() for curve in curves],axis=1)
    # Fit the same continuous curve to the native sampled pose, using sparse
    # spline support for a few manifold corrections at the actual clock.
    curve=AbsoluteMotionCurve(knots,pc,r)
    for _ in range(3):
        error=_log(evaluate_curve(curve,clock)["rotations"].swapaxes(-1,-2)@tracks)
        # The B-spline basis is the small-angle tangent of the cumulative chart.
        rotation=np.stack([lsmr(design,error[:,j,k],damp=.06,atol=1.e-6,btol=1.e-6,maxiter=120)[0]
                          for j in range(22) for k in range(3)],axis=1).reshape(len(pc),22,3)
        for scale in (1.,.5,.25):
            try:
                trial=curve.retracted(np.zeros_like(pc), rotation*scale)
                if np.linalg.norm(_log(evaluate_curve(trial,clock)["rotations"].swapaxes(-1,-2)@tracks)) < np.linalg.norm(error):
                    curve=trial
                    break
            except ValueError:
                continue
        else:
            break
    return curve


def prepare(visual, previous, job):
    problem=Problem(visual,job)
    recovery=problem.i.parent
    exmr=job/"exmr"
    inference=read(exmr/"inference.json")
    if inference["continuous_shots"]!=len(np.unique(problem.i.shots)):
        raise ValueError("ExMR did not infer every programme shot")
    old=load(job/"displayed_baseline.npz")
    np.testing.assert_array_equal(old["pts"],problem.i.pts)
    Q,network_local,hand=pose_source(exmr,problem.i.pts,problem.i.shots)
    obs=load(recovery["body_observations"]["path"])
    np.testing.assert_array_equal(obs["pts"],old["pts"])
    visual_data=load(visual/"whole_follow_evidence.npz")
    sigma=obs["sigma_px"].astype(float)
    hidden=np.zeros(sigma.shape,bool)
    for foot,joint in ((0,15),(1,16)):
        at=np.argwhere(obs["use"] & (obs["model_joint_ids"]==joint))
        for i,k in at:
            if (visual_data["status"][i,foot]!=1 or visual_data["extent"][i,foot]==0
                    or visual_data["identity_unresolved"][i,foot] or not np.isfinite(obs["uv"][i,k]).all()):
                continue
            xy=obs["uv"][i,k]
            dist=cv2.pointPolygonTest(visual_data["region"][i,foot],tuple(xy.astype(float)),True)
            if dist>sigma[i,k]:
                sigma[i,k]=np.hypot(sigma[i,k],dist)
                hidden[i,k]=True
    obs["sigma_px"]=sigma
    obs["inside_tracked_shoe_proxy"]=hidden
    whole=source_roles(visual,previous,job,problem.i.camera_sha)
    spin=next(e for e in read(problem.i.document_path)["source_episode_review"]["episodes"]
              if e["episode_id"]=="spin_support")
    old,direction=direct_spin_seed(problem,previous/"controls.npz",old,whole["directed_boot_rows"],episode=spin)
    P=root_on_original_camera(problem,Q,network_local,hand,old["P"],obs)
    P,local,seat=seat_legs(problem,P,Q,network_local,old)
    clock=problem.ref.times
    curve=make_curve(clock,P,Q,local,curve_from_packet(load(previous/"controls.npz")).knots)
    check=evaluate_curve(curve,clock)
    seat["curve_translation_difference_m"]=statistics(np.linalg.norm(check["P"]-P,axis=-1))
    orientation=np.concatenate((Q[:,None],local),axis=1)
    seat["curve_rotation_difference_deg"]=statistics(np.rad2deg(np.linalg.norm(
        _log(check["rotations"].swapaxes(-1,-2)@orientation),axis=-1)))
    stage=job/"prepared"
    stage.mkdir()
    original=problem.i.prepared
    for name in ("compact_smplx.npz","export_source.npz","model_identity.json","camera_context.json",
                 "grasp_relations.json","palm_grip_geometry.json","ice_soft_policy.json"):
        src=original/name
        if src.exists():
            shutil.copyfile(src,stage/name)
    if not (stage/"ice_soft_policy.json").is_file():
        shutil.copyfile(PROGRAMME/"ice_soft_1103.json",stage/"ice_soft_policy.json")
    packet=load(original/"input.npz")
    packet.update(pose_axis_angle=Rotation.from_matrix(network_local.reshape(-1,3,3)).as_rotvec().reshape(-1,21,3),
                  raw_root_prior_Q_world=Q)
    atomic_npz(stage/"input.npz",**packet)
    atomic_npz(stage/"initial.npz",pts=old["pts"],P=P,Q=Q,local_R=local,hand_local_R=hand,
               hand_pose_origin=np.array("Full programme ExMR HandMR; camera-only, fitted static skate mount"))
    atomic_npz(stage/"observations.npz",**obs)
    fixed=load(original/"fixed_binding.npz")
    fixed.update(base_skate_parameters_json=old["fixed_base_parameters_json"],
                 R_FB_initial=old["fixed_R0"], t_FB_initial_solver=old["fixed_t0"],x=old["shared"],
                 collar_constraints_enabled=np.array(True),
                 fixed_boot_geometry_json=np.array(json.dumps(dict(
                     upper_height_log=float(old["fixed_upper_height_log"]),respect_open_collar=True,fixed_skate_mount=True),
                     sort_keys=True)))
    atomic_npz(stage/"fixed_binding.npz",**fixed)
    shoe=clamped_cubic_knots(np.asarray(curve.knots[[3,-4]]))
    identity=np.broadcast_to(np.eye(3),(len(shoe)-4,2,3,3)).copy()
    atomic_npz(stage/"absolute_seed_state.npz",**curve_metadata(curve),absolute_knots=curve.knots,
               absolute_seed_position_controls=curve.position_controls,
               absolute_seed_rotation_controls=curve.rotation_controls,
               relative_skate_schema=np.array(SHOE_SCHEMA),relative_skate_knots=shoe,
               relative_skate_seed_rotation_controls=identity,
               prepared_input_sha256=np.array(digest(stage/"input.npz")),
               prepared_initial_sha256=np.array(digest(stage/"initial.npz")))
    doc=read(problem.i.document_path)
    restore_structure_inputs(doc,problem.i.recovery_path)
    doc["programme_body_observation_revision"]=dict(doc["programme_body_observation_revision"],
        path=str(stage/"observations.npz"),sha256=digest(stage/"observations.npz"))
    doc["programme_structure"]["solver_step"].pop("preserve_reviewed_flight_controls",None)
    episodes={e["episode_id"]:e for e in doc["source_episode_review"]["episodes"]}
    whole["support_shoe_phase"]=dict(episodes["spin_support"],
        turn_direction_sign=episodes["flying_camel_path"]["turn_direction_sign"],angular_scale_rad=.12)
    whole["weights"]={"whole_skate_source":9., "directed_boot_source":5., "support_shoe_phase":28.}
    doc["whole_skate_source"]=whole
    doc["programme_exmr"]=dict(receipt=str(exmr/"inference.json"),network_exposures=inference["frames"],
        source_exposures=len(clock),foot_pose="BodyNet ankles with HandMR, using fixed skate mounting",
        prefoot_sha256=digest(exmr/"exmr_prefoot.npz"),final_sha256=digest(exmr/"exmr_final.npz"))
    atomic_json(stage/"source_evidence.json",doc)
    atomic_json(job/"initialization.json",dict(exmr=doc["programme_exmr"],fixed_shoe_geometry_sha256=digest(stage/"fixed_binding.npz"),
                hidden_network_ankle_rows=int(hidden.sum()),semantic_sigma_change="distance of proxy inside visible skate outline",
                initialization=seat, directed_spin_seed=direction, body_controls=len(curve.position_controls)))
    emit("exmr_common_initialization_ready", source_exposures=len(clock),body_controls=len(curve.position_controls),**seat)


def refresh_surface(prepared, output, *, parameter=None, evidence_output=None, evidence_source=None):
    """Associate current template faces on the actual native and midpoint clock."""
    from calibration.program_tracking.body_tracking.programme_surface import scan_surface
    from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
    from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

    source=evidence_source or prepared/"source_evidence.json"
    ref,_,_,doc,_=load_reference(PROJECT,prepared,source)
    if parameter is not None:
        from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
        from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity

        controls=load(parameter)
        validate_checkpoint_identity(controls,ref.fixed_checkpoint_identity)
        ref.absolute_curve=curve_from_packet(controls)
        ref.relative_skate_curve=shoe_curve_from_packet(controls)
        ref.binding.x=controls["x"][:17].copy()
        provenance="programme_audit_repair" if "programme_audit_repair" in doc else "programme_exmr"
        doc[provenance]["surface_parameter_sha256"]=digest(parameter)
    state=SequenceState(ref,ref.times[[0,-1]])
    scan_surface(ref,state,np.zeros(state.size),doc,output,emit=emit)
    witness=output/"surface_scan.npz"
    doc["programme_structure"].update(surface_witnesses=str(witness),surface_witnesses_sha256=digest(witness))
    atomic_json(evidence_output or source,doc)


def geometry_trial(source, destination, profile, name, parameter=None):
    """Prepare a versioned static shape trial on the same programme and mount."""
    item, = (row for row in read(profile)["profile"] if row["name"] == name)
    values = dict(item["parameters"])
    height = values.pop("upper_height_scale")
    fixed = load(source/"fixed_binding.npz")
    mounting = json.loads(str(fixed["fixed_boot_geometry_json"]))
    mounting["upper_height_log"] = float(np.log(height))
    fixed["fixed_boot_geometry_json"] = np.array(json.dumps(mounting,sort_keys=True))
    fixed["base_skate_parameters_json"] = np.array(json.dumps(values))
    fixed["x"][12:] = 0.
    seed = load(source/"absolute_seed_state.npz")
    if parameter is not None:
        saved = load(parameter)
        for key in ("absolute_knots","relative_skate_knots","prepared_input_sha256"):
            np.testing.assert_array_equal(saved[key],seed[key])
        seed["absolute_seed_position_controls"] = saved["absolute_position_controls"]
        seed["absolute_seed_rotation_controls"] = saved["absolute_rotation_controls"]
        seed["relative_skate_seed_rotation_controls"] = saved["relative_skate_rotation_controls"]
    doc = read(source/"source_evidence.json")
    doc["programme_exmr"]["geometry_trial"] = dict(name=name, profile_sha256=digest(profile),
        shape=item["parameters"], same_mount=True,
        preceding_parameter_sha256=None if parameter is None else digest(parameter))
    destination.mkdir(exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(destination)
    revised = {"fixed_binding.npz","absolute_seed_state.npz","source_evidence.json"}
    for path in source.iterdir():
        if path.is_file() and path.name not in revised:
            shutil.copyfile(path,destination/path.name)
    atomic_npz(destination/"fixed_binding.npz",**fixed)
    atomic_npz(destination/"absolute_seed_state.npz",**seed)
    atomic_json(destination/"source_evidence.json",doc)
    emit("geometry_trial_prepared",path=str(destination),**doc["programme_exmr"]["geometry_trial"])


def repair_representation(visual, previous, job, source, destination):
    from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

    ref,camera,_,doc,_=load_reference(PROJECT,source)
    problem=Problem(visual,job)
    packet=load(source/"initial.npz")
    roles=load(job/"whole_skate_roles.npz")
    old=load(job/"displayed_baseline.npz")
    old,turn=direct_spin_seed(problem,previous/"controls.npz",old,doc["whole_skate_source"]["directed_boot_rows"],
                              episode=doc["whole_skate_source"]["support_shoe_phase"])
    time=ref.times
    frames=np.arange(len(time))
    raw_pose={key:packet[key] for key in ("P","Q","local_R","hand_local_R")}
    empty=np.zeros((len(time),69))
    raw_state=fk_state(ref,raw_pose,empty)
    current=fk_state(ref,ref.at(time),empty)
    if turn["kept"]:
        knees=raw_state["Q"][:,None]@raw_state["frames"][:,KNEES]
        world_ankles=old["R"]@old["mount"].swapaxes(-1,-2)
        raw_pose["local_R"][:,ANKLES-1]=knees.swapaxes(-1,-2)@world_ankles
        raw_state=fk_state(ref,raw_pose,empty)
    desired=camera.project(raw_state["O"],frames)
    observed_span=np.linalg.norm(np.ptp(roles["region"],axis=2),axis=-1)
    limit=np.maximum(3*roles["sigma_px"],.5*observed_span)

    def discrepancy(state):
        return np.linalg.norm(camera.project(state["O"],frames)-desired,axis=-1)

    before=discrepancy(current)
    error=before
    curve=ref.absolute_curve
    added=np.empty(0)
    attempted=np.zeros(len(time),bool)
    original_breaks=np.unique(ref.absolute_curve.knots)
    index=np.searchsorted(original_breaks,time)
    room=np.minimum(time-original_breaks[index-1],original_breaks[index]-time)
    native_spacing=float(np.median(np.diff(time)))
    while True:
        bad=np.flatnonzero(np.any((error>limit)&roles["visible"],axis=1)&~attempted)
        if not len(bad):
            break
        breaks=np.unique(curve.knots)
        for i in bad:
            knot=np.searchsorted(breaks,time[i])
            attempted|=(time>=breaks[max(0,knot-2)])&(time<=breaks[min(len(breaks)-1,knot+1)])
        added=time[attempted & (room>=native_spacing/4)]
        curve=make_curve(time,raw_pose["P"],raw_pose["Q"],raw_pose["local_R"],np.r_[ref.absolute_curve.knots,added])
        state=fk_state(ref,evaluate_curve(curve,time),empty)
        error=discrepancy(state)
        emit("source_curve_representation",added_exposure_times=len(added),controls=len(curve.position_controls),
             pixel_error=statistics(error[roles["visible"]]))

    destination.mkdir()
    revised={"initial.npz","absolute_seed_state.npz","source_evidence.json"}
    for path in source.iterdir():
        if path.is_file() and path.name not in revised:
            shutil.copyfile(path,destination/path.name)
    packet.update(raw_pose)
    atomic_npz(destination/"initial.npz",**packet)
    seed=load(source/"absolute_seed_state.npz")
    seed.update(curve_metadata(curve),absolute_knots=curve.knots,
                absolute_seed_position_controls=curve.position_controls,
                absolute_seed_rotation_controls=curve.rotation_controls,
                prepared_initial_sha256=np.array(digest(destination/"initial.npz")))
    atomic_npz(destination/"absolute_seed_state.npz",**seed)
    diagnostic=dict(before_px=statistics(before[roles["visible"]]),after_px=statistics(error[roles["visible"]]),
        maximum_fit_support_before=len(ref.absolute_curve.position_controls),controls_after=len(curve.position_controls),
        added_source_pts=ref.p["pts"][attempted].tolist(),directed_spin_seed=turn,
        residual_large_source_pts=ref.p["pts"][np.any((error>limit)&roles["visible"],axis=1)].tolist())
    doc["programme_exmr"]["representation_revision"]=diagnostic
    atomic_json(destination/"source_evidence.json",doc)
    atomic_json(destination/"representation_revision.json",diagnostic)


def check_shutter_seed(prepared):
    from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

    ref,camera,_,_,_=load_reference(PROJECT,prepared)
    camera.core.motion_projection_threads=4
    maximum=0.
    for group in np.array_split(np.arange(len(ref.times)),64):
        ts=ref.times[group,None]+camera.readout_times(group)[:,None]*np.linspace(-.5,.5,5)
        state=ref.at(ts.ravel())
        points=ref.body.forward_rotations(state["local_R"],hand_local_R=state["hand_local_R"])[0]
        world=state["P"][:,None]+np.einsum("nij,nkj->nki",state["Q"],points)
        try:
            projection=moving_projection(camera,world.reshape(len(group),5,17,3),group,jacobian=True)
        except ValueError as error:
            raise ValueError(f"Initial rolling shutter projection at PTS {ref.p['pts'][group[[0,-1]]].tolist()}: {error}") from error
        maximum=max(maximum,projection["body_row_contraction"])
    report=dict(exposures=len(ref.times),native_nodes=5,maximum_row_map_slope=maximum)
    atomic_json(prepared/"shutter_seed_check.json",report)
    emit("native_shutter_seed_ready",**report)


def repair_source_roles(source, destination):
    """Admit reviewed regressions as fitting evidence with their source families."""
    document=read(source)
    whole=document["whole_skate_source"]
    packet=load(whole["path"])
    seconds=packet["source_pts"]/12800.
    selected=np.zeros(packet["visible"].shape,bool)
    for foot,first,last in ((0,66.74,66.92),(0,150.90,151.14),(1,150.90,151.14)):
        selected[:,foot]|=(seconds>=first)&(seconds<=last)&packet["visible"][:,foot]
    pts=packet["source_pts"]
    for foot in (0,1):
        group=np.where(packet["tracking_anchor_pts"][:,foot]>=0,packet["tracking_anchor_pts"][:,foot],pts)
        current=selected[:,foot]
        reserve=current.copy()
        for delta in (256,512):
            reserve|=np.isin(pts,pts[current]+delta)|np.isin(pts,pts[current]-delta)
        related=np.isin(group,np.unique(group[reserve]))
        packet["role"][related&(packet["role"][:,foot]==2),foot]=0
        packet["role"][current,foot]=1
        groups,counts=np.unique(group[packet["role"][:,foot]==1],return_counts=True)
        fit=packet["role"][:,foot]==1
        packet["measure_s"][fit,foot]=.02/np.sqrt(counts[np.searchsorted(groups,group[fit])])
        if np.intersect1d(group[fit],group[packet["role"][:,foot]==2]).size:
            raise ValueError("Regression source family overlaps the development partition")
    packet["source_repair"]=selected
    path=destination.with_name("whole_skate_regression_roles.npz")
    atomic_npz(path,**packet)
    receipt=dict(count=int(selected.sum()),source_pts=[pts[selected[:,f]].tolist() for f in (0,1)],
                 historical_development_used_for_repair=int((selected&packet["historical_development"]).sum()),
                 total_fit=int(np.sum(packet["role"]==1)),independent_final_check=0)
    whole.update(path=str(path),sha256=digest(path),repair_role_receipt=receipt)
    atomic_json(destination,document)
    emit("regression_source_evidence_admitted",**receipt)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("baseline","prepare","surface","geometry","representation","shutter","regression"))
    parser.add_argument("visual",type=Path)
    parser.add_argument("previous",type=Path)
    parser.add_argument("job",type=Path)
    parser.add_argument("recovery",type=Path)
    parser.add_argument("--prepared",type=Path)
    parser.add_argument("--geometry-profile",type=Path)
    parser.add_argument("--geometry-name",choices=("current","midpoint","earlier"))
    parser.add_argument("--start-parameter",type=Path)
    parser.add_argument("--source-prepared",type=Path)
    parser.add_argument("--source-evidence-out",type=Path)
    parser.add_argument("--source-evidence",type=Path)
    a=parser.parse_args()
    os.environ["RINKSPACE_SKATE_RECOVERY_BINDING"]=str(a.recovery)
    if a.stage=="surface":
        stage=a.prepared or a.job/"prepared"
        if a.start_parameter is not None and a.source_evidence_out is None:
            parser.error("surface with a fitted parameter requires --source-evidence-out")
        out=(a.source_evidence_out.parent/"surface" if a.source_evidence_out is not None else
             a.job/("surface_seed" if a.prepared is None else stage.name+"_surface"))
        refresh_surface(stage,out,parameter=a.start_parameter,evidence_output=a.source_evidence_out,
                        evidence_source=a.source_evidence)
    elif a.stage=="shutter":
        check_shutter_seed(a.prepared or a.job/"prepared")
    elif a.stage=="regression":
        if not (a.source_evidence and a.source_evidence_out):
            parser.error("regression requires --source-evidence and --source-evidence-out")
        repair_source_roles(a.source_evidence,a.source_evidence_out)
    elif a.stage=="geometry":
        if not (a.prepared and a.geometry_profile and a.geometry_name):
            parser.error("geometry requires --prepared, --geometry-profile and --geometry-name")
        geometry_trial(a.job/"prepared",a.prepared,a.geometry_profile,a.geometry_name,a.start_parameter)
    elif a.stage=="representation":
        if not (a.prepared and a.source_prepared):
            parser.error("representation requires --prepared and --source-prepared")
        repair_representation(a.visual,a.previous,a.job,a.source_prepared,a.prepared)
    else:
        {"baseline":baseline,"prepare":prepare}[a.stage](a.visual,a.previous,a.job)


if __name__=="__main__":
    main()
