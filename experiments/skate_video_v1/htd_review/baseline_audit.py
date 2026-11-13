"""Native source-domain audit of the frozen current programme."""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, NAMES, angles_joint
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from experiments.skate_video_v1.inputs import atomic_json, digest, emit, read
from experiments.skate_video_v1.programme_motion_followup import bind


def metric(value):
    value=np.asarray(value,dtype=float)
    valid=value[np.isfinite(value)]
    if len(valid)==0:
        return dict(n=0)
    return dict(n=len(valid),median=float(np.median(valid)),p95=float(np.percentile(valid,95)),
                rms=float(np.sqrt(np.mean(valid**2))),maximum=float(valid.max()))


def observations(path, pts):
    with np.load(path,allow_pickle=False) as doc:
        np.testing.assert_array_equal(doc["pts"],pts)
        xy=np.full(doc["uv"].shape,np.nan)
        ids=doc["model_joint_ids"]
        admitted=doc["use"]
        uv=doc["uv"]
        finite=np.isfinite(uv).all(axis=-1)
        for j in range(17):
            have=(ids==j)&admitted&finite
            idx=np.flatnonzero(have.sum(axis=1)==1)
            xy[idx,j]=uv[idx,have[idx].argmax(1)]
    return xy


def second(x, dt):
    return (x[2:]-2*x[1:-1]+x[:-2])/dt**2


def envelope(controls, body_R):
    chart=JointChart(controls["anatomical_rest_R"],controls["anatomical_basis"],controls["anatomical_reference_twist"])
    current=chart.evaluate(body_R[:,JOINTS-1])
    policy=json.loads(str(controls["anatomical_policy_json"]))
    residual=[]
    for i,name in enumerate(NAMES):
        p=policy["envelopes"][name.rsplit("_",1)[-1]]
        env=JointEnvelope(*np.deg2rad(p["flex_degrees"]),np.deg2rad(p["flex_grid_degrees"]),
            np.deg2rad(p["swing_axes_degrees"]),*np.deg2rad(p["scale_degrees"]),policy["provenance"])
        residual.append(np.max(env.residual(angles_joint(current,i)),axis=-1))
    return np.column_stack(residual)


def compute(root, prepared, evidence, out):
    param=root/"solve/candidate.npz"
    result_path=root/"fitted/result.npz"
    ref, cam, info, doc, controls=bind(prepared,evidence,param)
    pts=ref.p["pts"]
    shots=ref.p["shot_ids"]
    seconds=pts*info["source_time_base"][0]/info["source_time_base"][1]
    dt=float(np.diff(seconds)[0])
    np.testing.assert_allclose(np.diff(seconds),dt,atol=1.e-10,rtol=0)
    with np.load(result_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["source_pts"],pts)
        saved={k:z[k] for k in ("COCO_uv","COCO_world_mid","com_world_mid","P_world_mid","Q_world_mid",
                                "skate_origin_world_mid","skate_R_world_mid","foot_cavity_max_violation_m_mid")}
    obs=observations(prepared/"observations.npz",pts)
    dev=np.linalg.norm(obs-saved["COCO_uv"],axis=-1)
    ratio=ref.binding.ratio
    pose=fk_state(ref,ref.at(ref.times),np.zeros((len(pts),69)))
    mass=ContinuousMassGeometry(ref)
    centres,rot=mass.from_pose(pose)
    com=mass.com(centres)
    np.testing.assert_allclose(com,saved["com_world_mid"],atol=1e-9,rtol=0)
    np.testing.assert_allclose(pose["O"]*ratio,saved["skate_origin_world_mid"],atol=1.e-9,rtol=0)
    np.testing.assert_allclose(pose["R"],saved["skate_R_world_mid"],atol=1.e-9,rtol=0)
    x=np.arange(0,len(pts),67)
    query=(ref.times[x,None]+cam.readout_times(x)[:,None]*np.linspace(-.5,.5,5)).ravel()
    state=ref.at(query)
    coco=ref.body.forward_rotations(state["local_R"],hand_local_R=state.get("hand_local_R"))[0]
    world=state["P"][:,None]+np.einsum("nij,nkj->nki",state["Q"],coco)
    check=moving_projection(cam,world.reshape(len(x),5,17,3),x)["uv"]
    np.testing.assert_allclose(check,saved["COCO_uv"][x],atol=1.e-7,rtol=0)
    e=envelope(controls,pose["local_R"])
    g=batch_features(ref.binding,pose["R"],pose["O"])
    support=np.asarray(g["eligible"])
    runner=np.array(g["semantics"])=="runner"
    min_steel=g["height"][...,support].min(-1)
    min_runner=g["height"][...,runner].min(-1)
    valid3=(shots[:-2]==shots[1:-1])&(shots[1:-1]==shots[2:])
    a=second(com,dt)
    p=saved["P_world_mid"]
    optical=cam.midpoint["rotation"][1:-1,2]
    aP=second(p,dt)
    dacc=(aP*optical).sum(-1)
    same=(shots[1:]==shots[:-1])
    speed=np.linalg.norm(np.diff(com,axis=0)/dt,axis=1)
    rr=pose["R"][:-1].swapaxes(-1,-2)@pose["R"][1:]
    angular=np.degrees(Rotation.from_matrix(rr.reshape(-1,3,3)).magnitude()).reshape(-1,2)
    episodes=[]
    for row in doc["source_episode_review"]["episodes"]:
        if row["kind"]!="flight_visible_candidate":
            continue
        i,j=(np.searchsorted(pts,row[key]) for key in ("start_pts","end_pts_inclusive"))
        ii=np.arange(i,j+1);mask=valid3[ii[1:-1]-1]
        acc=a[ii[1:-1]-1]
        residual=np.linalg.norm(acc-np.array([0,0,-9.81]),axis=-1)[mask]
        t=seconds[ii]-seconds[i]
        compensated=com[ii]+np.c_[0*t,0*t,4.905*t**2]
        X=np.column_stack((np.ones(len(t)),t))
        fitted=X@np.linalg.lstsq(X,compensated,rcond=None)[0]
        path_res=np.linalg.norm(compensated-fitted,axis=-1)
        episodes.append(dict(name=row["episode_id"],source_seconds=seconds[ii[[0,-1]]].tolist(),
             difference_samples=len(residual),com_gravity_residual_mps2=metric(residual),
             com_distance_to_gravity_parabola_mm=metric(path_res*1000),
             lower_of_two_min_steel_mm=metric(min_steel[ii].min(1)*1000)))
    clips={}
    for tag,t0,t1 in (("s01",25.,35.),("s02",44.98,52.),("s05",76.,95.),("s05_spin",87.50,92.50)):
        use=(seconds>=t0-1.e-8)&(seconds<=t1+1.e-8)
        inner=use[:-2]&use[1:-1]&use[2:]&valid3
        pair=use[:-1]&use[1:]&same
        local_error=dev[use]
        percentiles=metric(np.linalg.norm(a[inner],axis=-1))
        top=np.flatnonzero(inner)
        worst=int(top[np.argmax(np.linalg.norm(a[top],axis=-1))]+1)
        clips[tag]=dict(source_seconds=[t0,t1],exposures=int(use.sum()),
           native_detector_discrepancy_px=metric(local_error),
           com_acceleration_mps2=percentiles,com_speed_mps=metric(speed[pair]),
           pelvis_acceleration_line_of_sight_energy_fraction=float((dacc[inner]**2).sum()/(aP[inner]**2).sum()),
           com_max_acceleration_source_seconds=float(seconds[worst]),
           com_max_acceleration_xyz_mps2=a[worst-1].tolist(),
           runner_minimum_mm=metric(min_runner[use]*1000),
           skate_rotation_deg_per_20ms=metric(angular[pair]),skate_cavity_mm=metric(saved["foot_cavity_max_violation_m_mid"][use]*1000),
           outside_model_joint_envelope={name:int((e[use,j]>0).sum()) for j,name in enumerate(NAMES)})
    events=np.argsort(np.where(valid3,np.linalg.norm(a,axis=1),-1.))[-1::-1]
    worst=[]
    for i in events:
        if len(worst)>=20:
            break
        t=seconds[i+1]
        if any(abs(k["source_seconds"]-t)<.20 for k in worst):
            continue
        worst.append(dict(source_seconds=float(t),source_pts=int(pts[i+1]),shot=str(shots[i+1]),
                com_acceleration_mps2=float(np.linalg.norm(a[i])),pelvis_los_acceleration_mps2=float(dacc[i])))
    summary=dict(parameter=str(param),parameter_sha256=digest(param),result_sha256=digest(result_path),
        source_pts=pts[[0,-1]].tolist(),camera_sha256=info["state_sha256"],exposures=len(pts),
        fixed_conditional_world=True,unit="conditional metres, same shaped body and skate mass model",
        original_camera_and_pts_replay_max_px=float(np.max(np.abs(check-saved["COCO_uv"][x]))),
        input_2d_role="existing detector residual, used by earlier fitting and not independent GT",
        clips=clips,source_confirmed_flight_windows=episodes,high_com_acceleration_events=worst,
        pva_prediction_used=False)
    atomic_json(out,summary)
    emit("baseline_audited",spin=clips["s05_spin"],flights=episodes[:4])


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for k in ("root","prepared","evidence","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    compute(a.root,a.prepared,a.evidence,a.output)
