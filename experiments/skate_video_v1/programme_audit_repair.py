"""Import the committed Okada whole-skate replay into the continuous solver."""
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bsr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .body_skate_fit import Problem
from .deliver_body_skate_fit import saved_ref
from .inputs import PROGRAMME, PROJECT, atomic_json, atomic_npz, digest, emit, read
from .programme_exmr_fit import direct_spin_seed, restore_structure_inputs, source_roles, statistics
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, RotationChartDomainError, _log, clamped_cubic_knots, greville_times,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import (
    curve_from_packet, curve_metadata, evaluate_curve, inverse_right_jacobian,
)
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SHOE_SCHEMA
from calibration.program_tracking.body_tracking.skate_physics import fk_state


ANKLES = np.array([7, 8])
KNEES = np.array([4, 5])


def evaluate_replay(ref, times):
    rows = {}
    for block in np.array_split(times, max(1, int(np.ceil(len(times)/256)))):
        base = ref.at(block)
        pose = fk_state(ref, base, np.zeros((len(block), 69)))
        native = dict(base, **{name:pose[name] for name in ("R", "O", "native_world", "frames")})
        for key, value in native.items():
            rows.setdefault(key, []).append(value)
    return {key:np.concatenate(value) for key,value in rows.items()}


def source_projection(problem, state, index):
    pixels = []
    for first in range(0, len(index), 96):
        last = min(first+96, len(index))
        take = index[first:last].ravel()
        v = {k:state[k][take] for k in ("P","Q","local_R","hand_local_R")}
        coco = problem.ref.body.forward_rotations(v["local_R"],hand_local_R=v["hand_local_R"])[0]
        world = v["P"][:,None]+np.einsum("nij,nkj->nki",v["Q"],coco)
        pixels.append(moving_projection(problem.camera,world.reshape(last-first,5,17,3),
                                        np.arange(first,last))["uv"])
    return np.concatenate(pixels)


def fit_rotation_track(knots, clock, basis, weight, position, initial, reference, target):
    """Project the sampled rotation into the native cumulative SO(3) curve."""
    curve = AbsoluteMotionCurve(knots,position,initial[:,None],
                                None if reference is None else reference[:,None])
    dofs = 3*len(position)
    scale = np.sqrt(weight)
    n = len(clock)
    pointers = np.arange(0,4*(n+1),4,dtype=np.int32)
    for _ in range(6):
        value = evaluate_curve(curve,clock,with_blocks=True,basis=basis)
        current = value["rotations"][:,0]
        error = _log(current.swapaxes(-1,-2)@target)
        if np.max(np.linalg.norm(error,axis=-1)) < 1.e-8:
            break
        blocks = inverse_right_jacobian(-error)[:,None] @ value["rotation_right_blocks"][:,:,0]
        J = bsr_matrix(((scale[:,None,None,None]*blocks).reshape(-1,3,3),
                        value["control_indices"].astype(np.int32).ravel(),pointers),
                       shape=(3*n,dofs)).tocsr()
        rhs = (scale[:,None]*error).ravel()
        step = spsolve(J.T@J+1.e-8*eye(dofs,format="csc"),J.T@rhs).reshape(-1,1,3)
        before = np.linalg.norm(rhs)
        for fraction in (1.,.5,.25,.125):
            try:
                trial = curve.retracted(np.zeros_like(position),fraction*step)
            except RotationChartDomainError:
                continue
            r = evaluate_curve(trial,clock,basis=basis)["rotations"][:,0]
            after = np.linalg.norm(scale[:,None]*_log(r.swapaxes(-1,-2)@target))
            if after < before:
                curve = trial
                break
        else:
            break
        if (before-after) <= 1.e-5*max(before,1.e-8):
            break
    return curve.rotation_controls[:,0]


def fit_curve(old, clock, target, desired_at, new_breaks):
    breaks = np.unique(np.r_[old.knots,new_breaks])
    knots = clamped_cubic_knots(breaks)
    gt = greville_times(knots)
    old_gt = greville_times(old.knots)
    evaluated = desired_at(gt)
    init = np.concatenate((evaluated["Q"][:,None],evaluated["local_R"]),axis=1)
    # Controls whose complete knot support is identical keep the archived
    # value. The changed intervals use the sampled physical rotation path.
    right = np.clip(np.searchsorted(old_gt,gt),0,len(old_gt)-1)
    for i,j in enumerate(right):
        if np.array_equal(knots[i:i+5],old.knots[j:j+5]):
            init[i] = old.rotation_controls[j]
            # The ankle now carries the relative rotation of the old boot.
            init[i,ANKLES] = evaluated["local_R"][i,ANKLES-1]
    q = Rotation.from_matrix((init[:-1].swapaxes(-1,-2)@init[1:]).reshape(-1,3,3)).as_quat()
    q = q.reshape(len(gt)-1,22,4)
    q *= np.where(q[...,3:] < 0.,-1.,1.)
    if old.edge_reference_quaternion is not None:
        for i,j in enumerate(right[:-1]):
            if (right[i+1]==j+1 and np.array_equal(knots[i:i+6],old.knots[j:j+6])):
                q[i,:] = old.edge_reference_quaternion[j]
                actual = Rotation.from_matrix((init[i,ANKLES].swapaxes(-1,-2)@init[i+1,ANKLES])).as_quat()
                q[i,ANKLES] = np.where(actual[:,3:]<0.,-actual,actual)
    B = AbsoluteMotionCurve(knots,evaluated["P"],init,q).basis_sparse(clock)
    # Exposure nodes can lie a small fraction of a frame apart. Physical time
    # weights keep the fit from overcounting those dense samples.
    dt = np.diff(clock)
    weight = np.r_[dt[0],dt[:-1]+dt[1:],dt[-1]]*.5
    A = B.multiply(np.sqrt(weight)[:,None]).tocsr()
    regularizer = 1.e-10*eye(len(gt),format="csc")
    P = spsolve(A.T@A+regularizer,
                A.T@(np.sqrt(weight)[:,None]*target["P"])+regularizer@evaluated["P"])
    target_R = np.concatenate((target["Q"][:,None],target["local_R"]),axis=1)
    for j in range(22):
        init[:,j] = fit_rotation_track(knots,clock,B,weight,P,init[:,j],q[:,j],target_R[:,j])
    return AbsoluteMotionCurve(knots,P,init,q)


def direct_ankle_binding(ref, native, visible):
    previous,_t,_ = ref.binding.parameters()
    relative = (native["Q"][:,None] @ native["frames"][:,ANKLES] @ previous).swapaxes(-1,-2) @ native["R"]
    typical = np.stack([Rotation.from_matrix(relative[visible[:,foot],foot]).mean().as_matrix()
                        for foot in (0,1)])
    mount = previous@typical
    pivot = np.einsum("nfij,nfi->nfj",native["R"],native["native_world"][:,ANKLES]-native["O"])
    fixed_pivot = np.median(pivot,axis=0)
    np.testing.assert_allclose(pivot,np.broadcast_to(fixed_pivot,pivot.shape),rtol=0,atol=1.e-8)
    return mount,fixed_pivot


def refine_representation(curve, clock, target, desired_at, native_times, mid,
                          archived, problem, mount, pivot, roles):
    """Insert knots at native frames where conversion moves an observed boot."""
    shape = archived.binding.packet()
    material = np.array([[0.,0.,shape["floor"][0]], [shape["xs"][-1],0.,shape["floor"][-1]],
                         [.28*shape["xs"][-1],0.,max(shape["rise"])]])
    expected = target["O"][mid,:,None]+np.einsum("nfij,kj->nfki",target["R"][mid],material)
    frames = np.arange(len(mid))
    expected_uv = problem.camera.project(expected.reshape(len(mid),6,3),frames).reshape(len(mid),2,3,2)
    seen = roles["visible"]
    uncertainty = roles["sigma_px"]
    attempted = np.zeros(len(mid),bool)
    while True:
        value = evaluate_curve(curve,native_times)
        state = fk_state(archived,value,np.zeros((len(mid),69)))
        R = state["Q"][:,None]@state["frames"][:,ANKLES]@mount
        O = state["native_world"][:,ANKLES]-np.einsum("nfij,fj->nfi",R,pivot)
        xyz = O[:,:,None]+np.einsum("nfij,kj->nfki",R,material)
        uv = problem.camera.project(xyz.reshape(len(mid),6,3),frames).reshape(len(mid),2,3,2)
        error = np.max(np.linalg.norm(uv-expected_uv,axis=-1),axis=2)
        emit("committed_state_conversion_pixel_check",controls=len(curve.position_controls),
             boot_material_pixel_error=statistics(error[seen]))
        bad = np.flatnonzero(np.any(seen&(error>.5*uncertainty),axis=1)&~attempted)
        knots = np.unique(curve.knots)
        if not len(bad):
            break
        source = native_times[bad]
        bins = np.clip(np.searchsorted(knots,source),1,len(knots)-1)
        additions = []
        for b in np.unique(bins):
            choices = bad[bins==b]
            a,z = knots[b-1:b+1]
            stable = choices[(native_times[choices]-a>=.009)&(z-native_times[choices]>=.009)]
            if len(stable):
                i = stable[np.argmax(np.max(error[stable],axis=1))]
                additions.append(native_times[i])
            else:
                attempted[choices] = True
        if not additions:
            break
        next_curve = fit_curve(curve,clock,target,desired_at,additions)
        if len(next_curve.position_controls)<=len(curve.position_controls):
            break
        curve = next_curve
    return curve,error


def body_observations(problem, visual):
    obs = load(problem.i.parent["body_observations"]["path"])
    np.testing.assert_array_equal(obs["pts"],problem.i.pts)
    sigma = obs["sigma_px"].astype(float)
    hidden = np.zeros(sigma.shape,bool)
    for foot,joint in ((0,15),(1,16)):
        for i,k in np.argwhere(obs["use"] & (obs["model_joint_ids"]==joint)):
            xy = obs["uv"][i,k]
            if (visual["status"][i,foot]!=1 or visual["extent"][i,foot]==0
                    or visual["identity_unresolved"][i,foot] or not np.isfinite(xy).all()):
                continue
            distance = cv2.pointPolygonTest(visual["region"][i,foot],tuple(xy.astype(float)),True)
            if distance>sigma[i,k]:
                sigma[i,k] = np.hypot(sigma[i,k],distance)
                hidden[i,k] = True
    obs.update(sigma_px=sigma,inside_tracked_shoe_proxy=hidden)
    return obs


def prepare(visual, previous, job, *, refine=False):
    job.mkdir(exist_ok=True)
    receipt_path = job/("state_import_refined.json" if refine else "state_import.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    problem = Problem(visual,job)
    committed = read(PROGRAMME/"skate_video_1112/record.json")
    controls = previous/"controls.npz"
    if digest(controls)!=committed["controls"]["socket_aligned"]["sha256"]:
        raise ValueError("Repair seed differs from the committed socket candidate")
    if problem.visual_sha256!=committed["visual"]["sha256"] or problem.i.camera_sha!=committed["camera_sha256"]:
        raise ValueError("Repair source or camera differs from the committed candidate")
    archived = saved_ref(problem,controls)
    absolute = archived.absolute_curve
    t = archived.times
    between = (t[:-1]+t[1:])*.5
    times = np.unique(np.r_[problem.times.ravel(),t,between,greville_times(absolute.knots)])
    nodes = np.searchsorted(times,problem.times)
    mid = np.searchsorted(times,t)
    replay = evaluate_replay(archived,times)
    base = {k:v[mid] for k,v in replay.items()}
    ratio = archived.binding.ratio
    old_uv = source_projection(problem,replay,nodes)
    with np.load(previous/"fitted/result.npz",allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_pts"],problem.i.pts)
        baseline = {}
        for field,key,scale in (("native22_world_mid","native_world",ratio),
                                ("skate_R_world_mid","R",1.),("skate_origin_world_mid","O",ratio)):
            baseline[field] = float(np.max(np.abs(saved[field]-base[key]*scale)))
        baseline["COCO_uv"] = float(np.max(np.abs(saved["COCO_uv"]-old_uv)))
    if max(baseline.values())>1.e-7:
        raise ValueError("Committed full wear/socket replay differs from its output")
    emit("committed_socket_replay_verified",exposures=len(t),readout_nodes=int(nodes.size),
         between_exposure_samples=len(between),maximum_difference=baseline)

    whole = source_roles(visual,previous,job,problem.i.camera_sha,
                         development_windows=((150.90,151.14),))
    roles = load(job/"whole_skate_roles.npz")
    mount,pivot = direct_ankle_binding(archived,base,roles["visible"])
    spin=next(e for e in read(problem.i.document_path)["source_episode_review"]["episodes"]
              if e["episode_id"]=="spin_support")
    phase, direction = direct_spin_seed(problem,controls,dict(base,pivot=pivot),whole["directed_boot_rows"],
                                        episode=spin)
    phase_change = phase["R"][:,1]@base["R"][:,1].swapaxes(-1,-2)
    angle = np.unwrap(np.arctan2(phase_change[:,1,0],phase_change[:,0,0]))
    phase_at = PchipInterpolator(t,angle,extrapolate=False)

    def target_pose(clock, original):
        shift = phase_at(np.clip(clock,t[0],t[-1]))
        spin = Rotation.from_rotvec(shift[:,None]*np.array([0.,0.,1.])).as_matrix()
        boot = original["R"].copy()
        boot[:,1] = spin@boot[:,1]
        local = original["local_R"].copy()
        parent = original["Q"][:,None]@original["frames"][:,KNEES]
        local[:,ANKLES-1] = parent.swapaxes(-1,-2)@boot@mount.swapaxes(-1,-2)
        return dict(P=original["P"],Q=original["Q"],local_R=local,hand_local_R=original["hand_local_R"],
                    R=boot,O=original["native_world"][:,ANKLES]-np.einsum("nfij,fj->nfi",boot,pivot))

    target = target_pose(times,replay)
    before_R = replay["local_R"][:,ANKLES-1]
    new_R = target["local_R"][:,ANKLES-1]
    from calibration.program_tracking.body_tracking.body_clock import source_interval_mask
    transition = source_interval_mask(problem.i.pts, problem.i.tb, "84.46", "85.04")
    # Knots in the observed directional transition allow the existing curve
    # to follow the source branch without changing resolution across the programme.
    transition_breaks = t[transition] if direction["kept"] else np.empty(0)
    knots = np.unique(absolute.knots)
    take = np.searchsorted(knots,transition_breaks)
    distance = np.minimum(transition_breaks-knots[take-1],knots[take]-transition_breaks)
    breaks = transition_breaks[distance>.004]

    def desired_at(clock):
        return target_pose(clock,evaluate_replay(archived,clock))

    stage = job/("prepared_refined" if refine else "prepared")
    previous_seed = stage/"absolute_seed_state.npz"
    seed_exists = previous_seed.is_file()
    stage.mkdir(exist_ok=True)
    original = problem.i.prepared
    conversion_pixel = None
    if seed_exists:
        preserved = load(stage/"initial.npz")
        for name in ("P","Q","local_R","hand_local_R"):
            np.testing.assert_array_equal(preserved[name],target[name][mid])
        curve = curve_from_packet(load(previous_seed),seed=True)
    else:
        if refine:
            base_curve=curve_from_packet(load(job/"prepared/absolute_seed_state.npz"),seed=True)
            curve,conversion_pixel=refine_representation(base_curve,times,target,desired_at,t,mid,
                                                         archived,problem,mount,pivot,roles)
        else:
            curve = fit_curve(absolute,times,target,desired_at,breaks)
        for name in ("compact_smplx.npz","export_source.npz","model_identity.json","camera_context.json",
                     "grasp_relations.json","palm_grip_geometry.json","ice_soft_policy.json"):
            src = original/name
            if src.exists():
                shutil.copyfile(src,stage/name)
        if not (stage/"ice_soft_policy.json").is_file():
            shutil.copyfile(PROGRAMME/"ice_soft_1103.json",stage/"ice_soft_policy.json")
        shutil.copyfile(original/"input.npz",stage/"input.npz")
        atomic_npz(stage/"initial.npz",pts=problem.i.pts,**{k:target[k][mid]
                   for k in ("P","Q","local_R","hand_local_R")},
                   hand_pose_origin=np.array("committed programme replay"))
        atomic_npz(stage/"observations.npz",**body_observations(problem,load(visual/"whole_follow_evidence.npz")))
        shoe = clamped_cubic_knots(np.asarray(curve.knots[[3,-4]]))
        identity = np.broadcast_to(np.eye(3),(len(shoe)-4,2,3,3)).copy()
        atomic_npz(previous_seed,**curve_metadata(curve),
                   absolute_knots=curve.knots,absolute_seed_position_controls=curve.position_controls,
                   absolute_seed_rotation_controls=curve.rotation_controls,
                   relative_skate_schema=np.array(SHOE_SCHEMA),relative_skate_knots=shoe,
                   relative_skate_seed_rotation_controls=identity,
                   prepared_input_sha256=np.array(digest(stage/"input.npz")),
                   prepared_initial_sha256=np.array(digest(stage/"initial.npz")))
    fixed = load(original/"fixed_binding.npz")
    fixed.update(base_skate_parameters_json=np.array(json.dumps(asdict(archived.binding.base_parameters))),
                 R_FB_initial=mount,t_FB_initial_solver=-np.einsum("fij,fj->fi",mount,pivot),
                 x=np.r_[np.zeros(12),archived.binding.x[12:17]],collar_constraints_enabled=np.array(True),
                 fixed_boot_geometry_json=np.array(json.dumps(dict(
                     upper_height_log=float(archived.binding.upper_height_log),respect_open_collar=True,
                     fixed_skate_mount=True),sort_keys=True)))
    atomic_npz(stage/"fixed_binding.npz",**fixed)
    document = read(problem.i.document_path)
    restore_structure_inputs(document,problem.i.recovery_path)
    document["programme_body_observation_revision"].update(path=str(stage/"observations.npz"),
        sha256=digest(stage/"observations.npz"))
    structure = document["programme_structure"]
    structure["collar_fit"]="oriented_surface"
    structure["solver_step"].pop("preserve_reviewed_flight_controls",None)
    episodes={e["episode_id"]:e for e in document["source_episode_review"]["episodes"]}
    whole["support_shoe_phase"]=dict(episodes["spin_support"],
        turn_direction_sign=episodes["flying_camel_path"]["turn_direction_sign"],angular_scale_rad=.12)
    whole["weights"]={"whole_skate_source":9.,"directed_boot_source":5.,"support_shoe_phase":28.}
    document["whole_skate_source"]=whole
    receipt=dict(baseline_sha256=digest(controls),baseline_replay=baseline,
                 source_pts=problem.i.pts[[0,-1]].tolist(),exposures=len(t),readout_nodes=int(nodes.size),
                 between_exposure_samples=len(between),rotations_imported="committed full socket replay",
                 selected_model="one anatomical ankle rotation with a fixed, calibrated skate mount",
                 shape_same_as_commit=True,directed_spin_seed=direction,added_knots=breaks.tolist(),
                 archive_to_fixed_mount_ankle_change_deg=statistics(np.rad2deg(np.linalg.norm(
                     _log(before_R.swapaxes(-1,-2)@new_R),axis=-1))),
                 body_controls=len(curve.position_controls))
    if conversion_pixel is not None:
        receipt["conversion_shoe_material_pixels"] = statistics(conversion_pixel[roles["visible"]])
    document["programme_audit_repair"]=receipt
    atomic_json(stage/"source_evidence.json",document)

    from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
    converted,_,_,_,_=load_reference(PROJECT,stage)
    for name in ("upper","inner","adapter","edges"):
        np.testing.assert_allclose(converted.binding.packet()[name],archived.binding.packet()[name],
                                   rtol=0,atol=5.e-12)
    state = evaluate_replay(converted,times)
    new_uv = source_projection(problem,state,nodes)
    target_uv = source_projection(problem,target,nodes)
    receipt.update(
        conversion_body_pixel_error=statistics(np.linalg.norm(new_uv-target_uv,axis=-1)),
        conversion_skate_origin_error_mm=statistics(1000*ratio*np.linalg.norm(state["O"]-target["O"],axis=-1)),
        conversion_skate_rotation_error_deg=statistics(np.rad2deg(np.linalg.norm(
            _log(state["R"].swapaxes(-1,-2)@target["R"]),axis=-1))),
        native_clock_body_delta_to_commit_px=statistics(np.linalg.norm(new_uv-old_uv,axis=-1)),
        native_clock_skate_step_delta_deg=statistics(np.rad2deg(np.abs(
            np.linalg.norm(_log(state["R"][mid[:-1]].swapaxes(-1,-2)@state["R"][mid[1:]]),axis=-1)
            -np.linalg.norm(_log(target["R"][mid[:-1]].swapaxes(-1,-2)@target["R"][mid[1:]]),axis=-1)))))
    atomic_json(stage/"source_evidence.json",document)
    atomic_json(receipt_path,receipt)
    emit("committed_state_import_ready",**{k:receipt[k] for k in
         ("conversion_body_pixel_error","conversion_skate_origin_error_mm",
          "conversion_skate_rotation_error_deg","body_controls")})


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("visual","previous","job","recovery"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--refine-import",action="store_true")
    args = parser.parse_args()
    os.environ["RINKSPACE_SKATE_RECOVERY_BINDING"]=str(args.recovery)
    prepare(args.visual,args.previous,args.job,refine=args.refine_import)


if __name__=="__main__":
    main()
