"""Convert this event's preserved legacy candidates into the named v1.1 state.

Initialization conversion only: no source-pixel fit or physical acceptance.
The legacy superellipse exponent is preserved even when not implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .camera import (FixedHead, GaugeFixedSpline, UniformCubicBasis,
                     UniformCubicSpline, ZoomOptics, axis_rotation)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(values):
    a=np.asarray(values,dtype=np.float64)
    return {"rms":float(np.sqrt(np.mean(a*a))),"max_abs":float(np.max(np.abs(a)))}


def _native_controls(basis, times, targets):
    """Exact native interpolation plus declared linear boundary continuation.

    Ghost controls are fixed for initialization by zero second differences on
    the extra boundary triplets.  These equations are not final smoothness
    priors and do not reduce the final model's free control coordinates.
    """
    n=len(times); extension=(basis.n_controls-n)//2
    if n+2*extension != basis.n_controls:
        raise ValueError("unexpected native cubic boundary allocation")
    rows=[];cols=[];vals=[]
    for i,t in enumerate(times):
        indices,weights=basis.local(float(t))
        for j,w in zip(indices,weights):
            rows.append(i);cols.append(int(j));vals.append(float(w))
    triplets=list(range(extension))+list(range(basis.n_controls-2-extension,basis.n_controls-2))
    for j,start in enumerate(triplets):
        for offset,value in enumerate((1.,-2.,1.)):
            rows.append(n+j);cols.append(start+offset);vals.append(value)
    A=coo_matrix((vals,(rows,cols)),shape=(basis.n_controls,basis.n_controls)).tocsc()
    rhs=np.zeros((basis.n_controls,targets.shape[1]));rhs[:n]=targets
    controls=np.asarray(spsolve(A,rhs))
    if not np.all(np.isfinite(controls)):
        raise ValueError("native cubic seed interpolation failed")
    return controls


def bootstrap_state(seed_catalog_path, camera_seed_path, *, expected_frames=1248,
                    time_base_denominator=12800, max_abs_readout=.02,
                    distortion_radius_max=2.0):
    """Return a JSON-serializable initial state and complete conversion ledger.

    No implicit aux columns are read.  Every source field is named.  The
    output intentionally remains blocked for full-scene replay when p!=2.
    """
    seed_catalog_path=Path(seed_catalog_path);camera_seed_path=Path(camera_seed_path)
    seed=json.loads(seed_catalog_path.read_text(encoding="utf-8"))
    old=seed["legacy_head"]
    with np.load(camera_seed_path,allow_pickle=False) as arrays:
        required=("frame_indices","native_pts","R_world_to_camera","center_world",
                  "focal_px","principal_xy","brown_k1_k2_p1_p2")
        missing=set(required)-set(arrays.files)
        if missing:
            raise ValueError("missing named camera seed arrays: "+", ".join(sorted(missing)))
        data={key:np.array(arrays[key],copy=True) for key in required}
    frames=data["frame_indices"];pts=data["native_pts"];R=data["R_world_to_camera"]
    nframes=len(frames)
    if nframes != expected_frames or not np.array_equal(frames,np.arange(1,nframes+1)):
        raise ValueError("camera seeds must contain every ordered native frame exactly once")
    if pts.shape != (nframes,) or not np.all(np.diff(pts)==np.diff(pts)[0]) or np.diff(pts)[0]<=0:
        raise ValueError("only a verified same-shot uniform native PTS clock is supported")
    expected_shapes={"R_world_to_camera":(nframes,3,3),"center_world":(nframes,3),
                     "focal_px":(nframes,2),"principal_xy":(nframes,2),"brown_k1_k2_p1_p2":(nframes,4)}
    for key,shape in expected_shapes.items():
        if data[key].shape != shape or not np.all(np.isfinite(data[key])):
            raise ValueError(f"invalid named seed array {key}")
    if np.any(data["focal_px"]<=0) or time_base_denominator<=0:
        raise ValueError("positive focal lengths and source time base are required")
    if np.max(np.abs(np.swapaxes(R,1,2)@R-np.eye(3)))>1e-9 or np.max(np.abs(np.linalg.det(R)-1))>1e-9:
        raise ValueError("camera seed R must contain proper world-to-camera rotations")
    gauge=seed["world_gauge"]
    if float(gauge["rink_length"]) != 1.0:
        raise ValueError("this event conversion requires the already normalized L=1 source gauge")
    legacy_head=FixedHead(old["R0_world_to_camera"],old["pan_axis_world"],old["tilt_axis_camera"],
                          old["pivot"],old["lever"],a=old["axis_spacing"])
    n,b,Rold=legacy_head.n,legacy_head.b,legacy_head.R0
    v0=Rold.T@b
    angles=np.zeros((nframes,2));rotation_errors=[];solves=[]
    # Tilt leaves b invariant.  Its world image first supplies an analytic pan
    # branch, avoiding an arbitrary zero guess near the legacy pi-tilt branch.
    for i,target in enumerate(R):
        vi=target.T@b
        alpha=-np.arctan2(n@np.cross(v0,vi),v0@vi)
        B=target@(Rold@axis_rotation(n,alpha)).T
        beta=float(Rotation.from_matrix(B).as_rotvec()@b)
        def residual(ab):
            predicted=axis_rotation(b,ab[1])@Rold@axis_rotation(n,ab[0])
            return Rotation.from_matrix(predicted@target.T).as_rotvec()
        fit=least_squares(residual,[alpha,beta],ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=50)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError(f"two-angle seed conversion failed at native frame {int(frames[i])}")
        angles[i]=fit.x;rotation_errors.append(float(np.linalg.norm(fit.fun)))
        solves.append({"frame_index":int(frames[i]),"native_pts":int(pts[i]),
                       "rotation_error_rad":rotation_errors[-1],"nfev":int(fit.nfev)})
    angles=np.unwrap(angles,axis=0)
    offsets=angles[0].copy()
    R0=axis_rotation(b,offsets[1])@Rold@axis_rotation(n,offsets[0])
    angles-=offsets
    times=(pts-pts[0])/float(time_base_denominator)
    dt=float(pts[1]-pts[0])/float(time_base_denominator)
    signed_readout=float(old["readout_seconds"])
    coverage=max(float(max_abs_readout),abs(signed_readout))
    basis=UniformCubicBasis.for_native_frames(nframes,fps=1./dt,max_abs_readout=coverage)
    logf=np.mean(np.log(data["focal_px"]),axis=1)
    fref=float(np.exp(np.mean(logf)));z=logf-np.log(fref)
    full=_native_controls(basis,times,np.c_[angles,logf])
    reference_row=basis.row(0.);dependent=int(np.argmax(np.abs(reference_row)))
    free_indices=np.delete(np.arange(basis.n_controls),dependent)
    alpha=GaugeFixedSpline(basis,full[free_indices,0],0.)
    beta=GaugeFixedSpline(basis,full[free_indices,1],0.)
    focal=UniformCubicSpline(basis,full[:,2])
    replay_angles=np.array([[alpha(t),beta(t)] for t in times])
    replay_logf=np.array([focal(t) for t in times])
    # Preserve effective center seeds by fitting a *shared* axial pupil law;
    # absorb its constant into d_z so ell(0)=0 is exact in the new gauge.
    no_pupil=FixedHead(R0,n,b,legacy_head.P,legacy_head.d,a=legacy_head.a)
    axial=[]
    for ri,ab,center in zip(R,angles,data["center_world"]):
        base=no_pupil.pose(ab[0],ab[1],0.)
        axial.append(float((ri@(center-base.C))[2]))
    pupil_design=np.c_[np.ones(nframes),z,z*z]
    pupil=np.linalg.lstsq(pupil_design,np.asarray(axial),rcond=None)[0]
    d=np.array(legacy_head.d);d[2]+=pupil[0]
    head=FixedHead(R0,n,b,legacy_head.P,d,a=legacy_head.a,
                   pupil_linear=float(pupil[1]),pupil_quadratic=float(pupil[2]))
    logratio=np.log(data["focal_px"][:,0]/data["focal_px"][:,1]);eta=float(np.mean(logratio))
    optical_design=np.c_[np.ones(nframes),z]
    principal=np.linalg.lstsq(optical_design,data["principal_xy"],rcond=None)[0]
    radial=np.linalg.lstsq(optical_design,data["brown_k1_k2_p1_p2"][:,:2],rcond=None)[0]
    tangential=np.mean(data["brown_k1_k2_p1_p2"][:,2:],axis=0)
    optics=ZoomOptics(fref,tuple(principal[0]),eta,tuple(principal[1]),
                      tuple(radial[:,0]),tuple(radial[:,1]),float(tangential[0]),float(tangential[1]),
                      float(distortion_radius_max))
    center_errors=[];replayed_rotation=[];disk_bounds=[]
    focal_errors=[]
    for i,t in enumerate(times):
        K,D,zi=optics.at(replay_logf[i]);pose=head.pose(*replay_angles[i],zi)
        center_errors.append(np.linalg.norm(pose.C-data["center_world"][i]))
        replayed_rotation.append(np.linalg.norm(Rotation.from_matrix(pose.R@R[i].T).as_rotvec()))
        focal_errors.append([K.fx-data["focal_px"][i,0],K.fy-data["focal_px"][i,1]])
        disk_bounds.append(D.no_fold_lower_bound())
        solves[i].update({"cubic_rotation_error_rad":float(replayed_rotation[-1]),
                          "center_error_rink_length_units":float(center_errors[-1]),
                          "log_focal_error":float(replay_logf[i]-logf[i])})
    legacy_scene=seed["scene_parameters_legacy"];corner=legacy_scene["corner_curve"]
    unit=float(gauge["legacy_units_per_new_unit"]);bounds=np.asarray(legacy_scene["bounds"],dtype=float)/unit
    exponent=float(corner["exponent"]);radii=np.asarray(corner["tangent_extent_xy_scene_units"],dtype=float)/unit
    blockers=[]
    if exponent != 2.:
        blockers.append("superellipse_p_not_implemented: preserve p; implement stable endpoints before scene replay")
    failed_disks=int(np.count_nonzero(np.asarray(disk_bounds)<=0))
    if failed_disks:
        blockers.append("converted_low_dimensional_optics_uncertified: evaluate declared optical candidates before physical trials")
    state={
        "schema":"rinkfit.bootstrap_physical_state.v1","initialization_only":True,
        "original_pixel_fit_executed":False,"physical_accuracy_accepted":False,
        "status":"converted_seed_with_explicit_blockers" if blockers else "converted_seed_not_fitted",
        "blockers":blockers,
        "world_gauge":{"rink_length":1.,"verified_metre_scale":False,
                       "units":"one proposed rink length; no metric scale was invented"},
        "shared_head":{"R0":head.R0.tolist(),"n":head.n.tolist(),"b":head.b.tolist(),
                       "P":head.P.tolist(),"d":head.d.tolist(),"a":head.a,
                       "pupil_linear":head.pupil_linear,"pupil_quadratic":head.pupil_quadratic,
                       "axis_model":"orthogonal"},
        "camera_controls":{"basis":{"first_control_time":basis.first_control_time,"dt":basis.dt,
                                    "n_controls":basis.n_controls},
                           "reference_time":0.,"angle_dependent_index":dependent,
                           "alpha_free_controls":alpha.free_controls.tolist(),
                           "beta_free_controls":beta.free_controls.tolist(),
                           "log_f_controls":focal.controls.tolist(),
                           "domain_seconds":list(basis.domain),
                           "boundary_initialization":"zero second differences on extra ghost-control triplets; not final constraints"},
        "optics":{"f_ref":optics.f_ref,"principal_point":list(optics.principal_point),"eta":optics.eta,
                  "principal_linear":list(optics.principal_linear),"k1":list(optics.k1),"k2":list(optics.k2),
                  "p1":optics.p1,"p2":optics.p2,"radius_max":optics.radius_max,
                  "output_model":optics.output_model},
        "rolling_shutter":{"signed_readout_seconds":signed_readout,"role":"legacy_seed_only_not_measured",
                           "allocated_max_abs_readout_seconds":coverage,"unique_root_certified":False},
        "scene_seed":{"length":1.,"width":float(bounds[3]-bounds[1]),
                      "radius_x":float(radii[0]),"radius_y":float(radii[1]),"exponent":exponent,
                      "board_height":float(legacy_scene["board_height"])/unit,
                      "implementation_status":"unsupported_superellipse_preserved" if exponent != 2. else "ellipse_candidate",
                      "role":"shared_legacy_candidate_not_world_truth"},
        "clock":{"frame_indices":frames.tolist(),"native_pts":pts.tolist(),"pts_origin":int(pts[0]),
                 "time_base_numerator":1,"time_base_denominator":int(time_base_denominator)},
        "conversion":{"legacy_angle_offsets_rad":offsets.tolist(),"pupil_constant_absorbed_in_dz":float(pupil[0]),
                      "angle_gauge_function_values":[alpha(0.),beta(0.)],
                      "native_rotation_error_rad":_summary(replayed_rotation),
                      "native_focal_error_px":_summary(focal_errors),
                      "center_error_rink_length_units":_summary(center_errors),
                      "principal_conversion_error_px":_summary(optical_design@principal-data["principal_xy"]),
                      "radial_coefficient_conversion_error":_summary(optical_design@radial-data["brown_k1_k2_p1_p2"][:,:2]),
                      "eta_conversion_error":_summary(logratio-eta),
                      "distortion_disk_certificate_failed_frames":failed_disks,
                      "distortion_disk_minimum_bound":float(min(disk_bounds)),
                      "per_frame_replay":solves},
        "provenance":{"seed_catalog":str(seed_catalog_path),"seed_catalog_sha256":_sha256(seed_catalog_path),
                      "camera_seeds":str(camera_seed_path),"camera_seeds_sha256":_sha256(camera_seed_path),
                      "canonical_q_sha256":seed.get("canonical_q_sha256"),
                      "source_scene_sha256":seed.get("source_scene_sha256"),
                      "role":"named-field conversion of preserved historical candidates; no implicit aux columns"},
    }
    json.dumps(state,allow_nan=False)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sources=(args.data/"seeds.json",args.data/"camera_seeds.npz")
    if args.output.resolve() in [p.resolve() for p in sources]:
        parser.error("output must not overwrite a preserved seed source")
    state=bootstrap_state(*sources)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":state["status"],
                      "frames":len(state["clock"]["native_pts"]),"blockers":state["blockers"],
                      "rotation_error_rad":state["conversion"]["native_rotation_error_rad"],
                      "focal_error_px":state["conversion"]["native_focal_error_px"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

# IMPLEMENTED: real named-seed loading, two-angle fixed-head conversion, exact
# first-frame function gauge, native cubic interpolation with boundary support,
# shared pupil/low-dimensional optics conversion, serializable state/provenance
# and full frame replay.  NOT IMPLEMENTED: p!=2 geometry, certified optical/RS
# domains, all-observation S/branches, physical fitting or final acceptance.
