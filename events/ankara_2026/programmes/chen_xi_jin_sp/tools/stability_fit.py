"""Jointly fit named source pixels and measured static displacements across each shot."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="2"

import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, kron, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from common import HEIGHT, OUTPUT, frozen_scene, observation_packet, read, save, scope, source_pts, write
from fit import checkpoint, measurement_rows, row_layer, row_loss, row_weights, statistics
from ice_evidence import ICE_EVIDENCE_FIELDS, ice_roles
from measure import DT, FIRST, clock, load_camera, motion_clock
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos
from rinkspace.domain.camera_experiment import CAMERA_NAMES, _SCALAR_STORAGE_INDEX
from stability_audit import load_tracks
from stability_tracks import BLOCK
from solve_joint import Journal, finite_json


def motion_space(control,t,stride):
    chart=UniformMotionChart.create(len(control),FIRST,DT,stride,t[0])
    halo=continue_exposure_halos(np.repeat(chart.refinement[:,None],3,axis=1),FIRST,DT,t[[0,-1]])[:,0]
    ij,w=cubic_samples(len(control),FIRST,DT,t)
    sampled=np.einsum("ij,ijk->ik",w,halo[ij])
    alive=np.max(abs(sampled),axis=0)>1.e-8
    return halo[:,alive],csr_matrix(sampled[:,alive]),ij,w


def coarse_seed(control,t,stride):
    space,sampled,ij,w=motion_space(control,t,stride)
    exposure=np.einsum("ij,ijk->ik",w,control[ij])
    n=space.shape[1]
    bend=np.diff(np.eye(n),n=2,axis=0)
    design=vstack((sampled,csr_matrix(.08*bend)),format="csr")
    wanted=np.vstack((exposure,np.zeros((len(bend),3))))
    fit=np.column_stack([lsqr(design,y,atol=2.e-12,btol=2.e-12,iter_lim=900)[0] for y in wanted.T])
    return space@fit


def coherent_sample(q,uv,d,quality,limit):
    """Keep patches spread across columns with consistent local flow residuals."""
    if len(q)<2:
        return np.empty(0,np.int64)
    x=uv[q]/[1920.,1080.]
    basis=np.c_[np.ones(len(x)),x]
    central=np.linalg.norm(d[q]-np.median(d[q],axis=0),axis=1)
    initial=central<=np.quantile(central,.76)
    if np.count_nonzero(initial)>=4:
        model=np.linalg.lstsq(basis[initial],d[q][initial],rcond=None)[0]
        residual=np.linalg.norm(basis@model-d[q],axis=1)
        good=residual<=max(1.9,min(9.,3.1*np.median(residual)+.5))
        q=q[good]
    if len(q)<2:
        return np.empty(0,np.int64)
    choices=[]
    for cell in range(8):
        hits=q[(uv[q,0]//240)==cell]
        if len(hits):
            order=np.argsort(quality[hits],kind="stable")
            choices.extend(hits[order[:limit]])
    if len(choices)<4:
        by_confidence=q[np.argsort(quality[q],kind="stable")]
        choices.extend(k for k in by_confidence[:4] if k not in choices)
    return np.asarray(choices,np.int64)


def pairs_from_source(data,trust):
    # Gate on real measured motion coherence and visibility. A poor fitting
    # prediction must not discard the very frames it fails to follow.
    raw=data["uv"]-data["left_uv"]
    idx,lag,family,iq=data["index"],data["lag"],data["family"],data["quality"]
    use=data["usable"]&(iq<16.)&np.isfinite(raw).all(1)
    if "role" in data:
        use&=data["role"]==0
    else:
        right=idx%BLOCK
        left=(idx-lag)%BLOCK
        # Leave exposure halos next to the reserved interval out of the fit.
        after_reserved=(data["fold"]==0)&(left<2)
        before_reserved=(data["fold"]==2)&(right>=BLOCK-2)
        use&=(data["fold"]!=3)&~(before_reserved|after_reserved)
    q=np.flatnonzero(use)
    q=q[np.lexsort((lag[q],family[q],idx[q]))]
    matrix=np.c_[idx[q],family[q],lag[q]]
    chosen=[]
    weights=[]
    for group in np.split(q,np.flatnonzero(np.any(np.diff(matrix,axis=0)!=0,axis=1))+1):
        if not len(group):
            continue
        i=group[0]
        selected=coherent_sample(group,data["uv"],raw,iq,2 if family[i]==0 else 1)
        if len(selected)<2:
            continue
        mass=(13. if lag[i]==1 else 9.5)*(1. if family[i]==0 else .68)*trust
        mass*=min(1.,np.sqrt(len(selected)/4.))
        chosen.extend(selected.tolist())
        weights.extend([mass/np.sqrt(len(selected))]*len(selected))
    chosen=np.asarray(chosen,np.int64)
    t=idx[chosen]
    k=lag[chosen]
    before_two=data["prev2_uv"][chosen] if "prev2_uv" in data else np.full((len(chosen),2),np.nan)
    return dict(world=data["world"][chosen],frames=np.c_[t-k,t],
                uv=np.stack((data["left_uv"][chosen],data["uv"][chosen]),axis=1),
                prev2_uv=before_two,weight=np.asarray(weights),
                shot_track_rows=chosen,family=family[chosen],lag=k)


def data_bundle(record,observations,tracks,previous,source_stride,seed_stride,trust,shots,windows,
                path_roles,reused=None,shot_strides=None,preserve_track_holdouts=False,shot_axes=None,
                local_profiles=None,keep_fixed_view_rows=False,photo_anchors=None,
                protect_measurement_folds=False):
    rows={k:[] for k in ("pts","idx","uv","path","station","family","material","times","camera","shot",
                        *ICE_EVIDENCE_FIELDS)}
    tracks_out={k:[] for k in ("world","times","uv","prev2_uv","prev2_times","weight","family","lag",
                              "camera","shot","shot_track_rows")}
    spans={}
    controls=[]
    basis=[]
    rigs=[]
    slots={}
    groups={}
    exposure={}
    cursor=0
    for shot in record["shots"]:
        sid=shot["id"]
        pts,c,rig=load_camera(shot,previous)
        dt=motion_clock(shot,previous,pts)
        if not np.array_equal(dt,clock(pts)):
            exposure[sid]=dt
        camera=shot["camera_index"]
        # A calibrated view at a hard cut can carry its own bounded rig
        # chart while retaining the camera identity in the source registry.
        group=(camera,rig.tobytes(),sid if sid in (local_profiles or {}) else None)
        if group not in groups:
            groups[group]=len(rigs)
            rigs.append(rig)
        slot=slots[sid]=groups[group]
        spans[sid]=(cursor,cursor+len(c))
        enabled=(shots is None or sid in shots) and (windows is None or sid in windows)
        if not enabled and not keep_fixed_view_rows:
            basis.append(csr_matrix((c.size,0)))
            controls.append(c)
            cursor+=len(c)
            continue
        r=measurement_rows(observations,shot,record,path_roles)
        optimization_intervals = [] if preserve_track_holdouts else (windows or {}).get(sid,[])
        reused_intervals=optimization_intervals + (reused or {}).get(sid,[])
        data=load_tracks(observation_packet(tracks,sid),record,shot,reused_intervals)
        if protect_measurement_folds:
            from measure import fold_map
            nontraining=np.r_[0,np.cumsum(fold_map(shot,len(pts))!=0)]
            end=data["index"]
            start=np.maximum(0,end-np.maximum(data["lag"],2))
            data["usable"] &= nontraining[end+1]==nontraining[start]
        tr=pairs_from_source(data,trust)
        print(sid,"named points",len(r["pts"]),"measured pairs",len(tr["weight"]),flush=True)
        if enabled:
            local_stride=(shot_strides or {}).get(sid,source_stride)
            c=coarse_seed(c,dt,seed_stride) if seed_stride else c
            refinement,sampled,_,_=motion_space(c,dt,local_stride)
            # Fine control directions have to be supported by measured motion.
            temporal_frames=tr["frames"].ravel()
            supported=np.unique(temporal_frames if local_stride==1 else np.r_[r["idx"],temporal_frames])
            hit=np.asarray((abs(sampled[supported])>.04).sum(axis=0)).ravel()>=2
            if windows is not None:
                active_frames=np.zeros(len(pts),bool)
                for lo,hi in windows[sid]:
                    active_frames|=(pts>=lo)&(pts<=hi)
                inside=np.asarray(abs(sampled[active_frames]).max(axis=0).toarray()).ravel() if active_frames.any() else np.zeros(len(hit))
                outside=np.asarray(abs(sampled[~active_frames]).max(axis=0).toarray()).ravel() if not active_frames.all() else np.zeros(len(hit))
                hit&=(inside>0.)&(inside>=outside)
            axis_names = ("pan", "tilt", "log_focal")
            axes = (shot_axes or {}).get(sid, axis_names)
            if not axes or len(set(axes)) != len(axes) or set(axes).difference(axis_names):
                raise ValueError(f"Invalid active camera trajectory axes for {sid}")
            axis_basis = np.diag([.0014, .0014, .0030])[:, [axis_names.index(axis) for axis in axes]]
            block=kron(csr_matrix(refinement[:,hit]),csr_matrix(axis_basis),format="csr")
        else:
            block=csr_matrix((c.size,0))
        basis.append(block)
        controls.append(c)
        shift=cursor*DT-FIRST
        r["times"]=dt[r["idx"]]+shift
        r["camera"]=np.full(len(r["pts"]),slot,np.int32)
        r["shot"]=np.full(len(r["pts"]),sid)
        frames=tr.pop("frames")
        tr["times"]=dt[frames]+shift
        tr["prev2_times"]=dt[np.maximum(frames[:,1]-2,0)]+shift
        tr["camera"]=np.full(len(tr["weight"]),slot,np.int32)
        tr["shot"]=np.full(len(tr["weight"]),sid)
        for key,value in r.items():
            rows[key].append(value)
        if photo_anchors is not None:
            append_photo_groups(tr,photo_anchors,sid,pts,dt,shift,slot)
        for key,value in tr.items():
            tracks_out[key].append(value)
        cursor+=len(c)
    if not rows["pts"]:
        raise ValueError("No selected shot has current source measurements")
    return ({k:np.concatenate(v) for k,v in rows.items()},
            {k:np.concatenate(v) for k,v in tracks_out.items()},np.vstack(controls),
            np.asarray(rigs),block_diag(basis,format="csr"),spans,exposure,slots)


def load_photo_anchors(path, record, selected):
    """Read horizontal registrations of distinct print regions per exposure."""
    with np.load(path,allow_pickle=False) as f:
        for key,digest in (("source_sha256",record["source"]["sha256"]),
                           ("geometry_sha256",record["geometry_binding"]["value_sha256"])):
            if f[key].item()!=digest:
                raise ValueError(f"Photo anchor {key} differs from the active binding")
        data={k:f[k] for k in ("shot","pts","world","uv","weight")}
        regions=f["region"] if "region" in f else np.full(len(data["pts"]),"")
    n=len(data["pts"])
    if (not n or data["pts"].shape!=(n,) or data["pts"].dtype.kind not in "iu" or data["shot"].shape!=(n,) or
            regions.shape!=(n,) or regions.dtype.kind not in "US" or
            data["world"].shape!=(n,3) or data["uv"].shape!=(n,2) or
            data["weight"].shape!=(n,) or not set(data["shot"]).issubset(selected) or
            len(set(zip(data["shot"],data["pts"],regions)))!=n or
            not all(np.isfinite(data[k]).all() for k in ("world","uv","weight")) or
            np.any(data["weight"]<=0.)):
        raise ValueError("Invalid active source-photo registration constraints")
    return data


def append_photo_groups(tr,anchors,shot,pts,clock,shift,slot):
    use=anchors["shot"]==shot
    if not use.any():
        return
    p=anchors["pts"][use]
    index=np.searchsorted(pts,p)
    if (index>=len(pts)).any() or not np.array_equal(pts[index],p):
        raise ValueError(f"Photo anchors do not refer to native exposures in {shot}")
    uv=anchors["uv"][use]
    n=len(p)
    times=clock[index]+shift
    addition=dict(world=anchors["world"][use],times=np.repeat(times[:,None],2,axis=1),
                  uv=np.repeat(uv[:,None],2,axis=1),prev2_uv=np.full_like(uv,np.nan),
                  prev2_times=times,weight=anchors["weight"][use],family=np.full(n,-1),
                  lag=np.zeros(n,dtype=tr["lag"].dtype),camera=np.full(n,slot),
                  shot=np.full(n,shot),shot_track_rows=np.full(n,-1))
    for k,v in addition.items():
        tr[k]=np.concatenate((tr[k],v.astype(tr[k].dtype,copy=False)))


def track_samples(tracks):
    mask=np.column_stack((tracks["triplet"],np.ones_like(tracks["uv"][:,:,0],dtype=bool)))
    # Photo registration observes one exposure and has no preceding sample.
    mask[:,1]=tracks["lag"]!=0
    uv=np.concatenate((tracks["prev2_uv"][:,None,:],tracks["uv"]),axis=1)
    times=np.column_stack((tracks["prev2_times"],tracks["times"]))
    return mask, uv, times


def resident(native,rows,tracks,controls,rig):
    paths=list(dict.fromkeys(rows["path"]))
    labels={k:i for i,k in enumerate(paths)}
    n=len(tracks["world"])
    mask,uv,times=track_samples(tracks)
    length=mask.sum(axis=1)
    track_count=int(length.sum())
    definitions=[native.paths[k] for k in paths]+[dict(kind=4,vertices=tracks["world"],parameters=[])]
    model=ResidentResidualProblem(native.core,board=native.scene["board"],
          layout=native.scene["shared_layout"],options=native.scene["shared_geometry_options"],
          rig=rig,controls=controls,first_control_time=0.,control_dt=DT,image_height=HEIGHT,
          codes=np.r_[[labels[k] for k in rows["path"]],np.full(track_count,len(paths),np.int32)].astype(np.int32),
          stations=np.r_[rows["station"],np.repeat(np.linspace(0.,1.,n),length)],
          times=np.r_[rows["times"],times[mask]],
          camera_ids=np.r_[rows["camera"],np.repeat(tracks["camera"],length)],
          observed=np.vstack((rows["uv"],uv[mask])))
    try:
        model.configure_geometry(definitions)
        model.configure_projection_time_support(half_seconds=.64*max(abs(rig[:,37]))+1.e-6)
    except BaseException:
        model.close()
        raise
    return model


def rig_directions(model, motion, rig, profile):
    """Append explicitly bounded physical directions shared by the installation."""
    if set(profile) != {str(i) for i in range(len(rig))}:
        raise ValueError("A camera profile must name each confirmed installation")
    lower=np.full(rig.shape,-np.inf)
    upper=np.full(rig.shape,np.inf)
    n=motion.shape[1]
    q=n+sum(map(len,profile.values()))
    directions=np.zeros((len(rig)*31,q))
    active={}
    for camera in range(len(rig)):
        active[camera]=set()
        for name,pair in profile[str(camera)].items():
            scale,span=map(float,pair)
            if not np.isfinite([scale,span]).all() or not 0.<scale<=span:
                raise ValueError("Invalid current installation scalar range")
            directions[camera*31+CAMERA_NAMES.index(name),n]=scale
            if name=="R0_z":
                if span>.035:
                    raise ValueError("The local roll chart needs a small bound in radians")
                r0=rig[camera,:9].reshape(3,3)
                lo,hi=Rotation.from_rotvec([[0,0,-span],[0,0,span]]).as_matrix()@r0
                # The largest slope element is monotonic on the small
                # permitted interval and bounds the SO(3) roll directly.
                anchor=int(np.argmax(abs((hi-lo).ravel())))
                lower[camera,anchor],upper[camera,anchor]=sorted((lo.flat[anchor],hi.flat[anchor]))
                active[camera].update((*range(6),12,13,14))
            elif name in ("pan_axis_tangent1", "pan_axis_tangent2", "tilt_axis"):
                # The native chart retracts these rows onto the orthonormal
                # mount axes. Bound the stored axis values around the seed.
                if span > .035:
                    raise ValueError("The local mount axis chart needs a small bound in radians")
                indices = range(9, 12) if name.startswith("pan_axis") else range(12, 15)
                bound = np.sin(span)
                lower[camera, indices] = np.maximum(lower[camera, indices], rig[camera, indices] - bound)
                upper[camera, indices] = np.minimum(upper[camera, indices], rig[camera, indices] + bound)
                active[camera].update(indices)
                active[camera].update(range(12, 15))
            else:
                idx=_SCALAR_STORAGE_INDEX[name]
                lower[camera,idx]=rig[camera,idx]-span
                upper[camera,idx]=rig[camera,idx]+span
                active[camera].add(idx)
            n+=1
    time_bound = np.column_stack((rig[:, 37], lower[:, 37], upper[:, 37]))
    max_readout = 1.28 * np.max(abs(time_bound[np.isfinite(time_bound)])) + 2e-6
    model.configure_camera_parameters(directions=directions,max_readout=max_readout,
                                      lower=lower.ravel(),upper=upper.ravel())
    expanded=hstack((motion,csr_matrix((motion.shape[0],q-motion.shape[1]))),format="csr")
    return expanded,active


def configure(model,rows,tracks,directions,material_scale=1.,reviewed_scale=1.,triplet_scale=0.,
              path_weights=None,family_weights=None):
    n=len(rows["pts"])
    layer=row_layer(rows["family"])
    raw_w=row_weights(rows["pts"],rows["family"],layer)
    raw_w*=np.where(np.char.startswith(rows["material"],"dense_"),.70,1.)
    raw_w*=np.where((rows["shot"]=="s01")&(rows["material"]=="direct_purple_panel_boundary"),1.65,1.)
    direct=np.char.startswith(rows["material"],"source_")
    raw_w*=np.where(direct,material_scale,1.)
    reviewed=np.char.startswith(rows["material"],"source_reviewed_")
    raw_w*=np.where(reviewed,reviewed_scale,1.)
    for field, configured in (("path",path_weights),("family",family_weights)):
        for shot, weights in (configured or {}).items():
            for key, factor in weights.items():
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Invalid raw measurement weight")
                selected = (rows["shot"] == shot) & (rows[field] == key)
                if not selected.any():
                    raise ValueError(f"No active measurements for {shot} {key}")
                raw_w[selected] *= factor
    absolute=np.zeros((n,2,2))
    absolute[:,0,0]=absolute[:,1,1]=raw_w
    has_three=tracks["triplet"]
    photo=tracks["lag"]==0
    pair_groups=~(has_three|photo)
    length=(2+has_three-photo).astype(np.int32)
    ranks=np.where(photo,1,np.where(has_three,4,2)).astype(np.int32)
    blocks=2*length*ranks
    at=np.r_[0,np.cumsum(blocks,dtype=np.int32)]
    relative=np.empty(at[-1])
    w=tracks["weight"]
    two_eye=np.eye(2)
    pair=np.kron([[-1.,1.]],two_eye).ravel()
    relative[(at[:-1][pair_groups,None]+np.arange(8))]=w[pair_groups,None]*pair
    relative[at[:-1][photo,None]+np.arange(2)]=w[photo,None]*np.array([1.,0.])
    # The two contrasts describe the span and curvature of one source track.
    # Use one robust group so the middle exposure is not added as a third
    # independently weighted pair for this track.
    curvature=np.array([1.,-2.,1.])*(triplet_scale/np.sqrt(3.))
    triple=np.kron(np.vstack(([-1.,0.,1.],curvature)),two_eye).ravel()
    relative[(at[:-1][has_three,None]+np.arange(24))]=w[has_three,None]*triple
    track_count=int(length.sum())
    groups=np.r_[np.arange(n,dtype=np.int32),n+np.r_[0,np.cumsum(length,dtype=np.int32)]]
    weights=np.r_[4*np.arange(n,dtype=np.int32),4*n+at]
    parts=np.r_[np.arange(n,dtype=np.int32),np.full(track_count,-1,np.int32)]
    plan=model.configure_linearization(directions=directions,local_ids=parts,
          local_scales=np.where(layer=="board",.003,.009),group_offsets=groups,
          members=np.arange(n+track_count,dtype=np.int32),ranks=np.r_[np.full(n,2,np.int32),ranks],
          weight_offsets=weights,weights=np.r_[absolute.ravel(),relative],huber_delta=2.7)
    s=rows["station"]
    support=model._path_supports[model._codes[:n]]
    window=np.where(np.char.startswith(rows["path"],"board/continuous/"),.025,.15)
    window=np.where(rows["material"]=="source_ad2_named_right_stroke",.036,window)
    lower=np.maximum(s-window,support[:,0])
    upper=np.minimum(s+window,support[:,1])
    solver=model.configure_solver(lower=lower,upper=upper,device_budget_bytes=8*1024**3,
            host_budget_bytes=12*1024**3,initial_lambda_relative=.001,damping_metric="jacobi_image")
    return layer,raw_w,plan,solver


def result(model,rows,tracks,layer,raw_w):
    r=model.evaluate("cuda")
    n=len(layer)
    projected=r["values"][:,3:5]
    e=np.linalg.norm(projected[:n]-rows["uv"],axis=1)
    mask,observed,_=track_samples(tracks)
    delta=np.full_like(observed,np.nan)
    delta[mask]=projected[n:]-observed[mask]
    de=np.linalg.norm(delta[:,2]-delta[:,1],axis=1)
    photo=tracks["lag"]==0
    de[photo]=np.abs(delta[photo,2,0])
    e[r["status"][:n]!=0]=np.nan
    invalid=np.zeros(mask.shape,bool)
    invalid[mask]=r["status"][n:]!=0
    de[invalid.any(axis=1)]=np.nan
    weights=tracks["weight"]
    loss=row_loss(e,raw_w,layer)
    for lag in (1,8):
        v=weights[tracks["lag"]==lag]*de[tracks["lag"]==lag]
        loss[f"source_motion_{lag}"]=float(np.nansum(np.where(v<=2.7,.5*v*v,2.7*(v-1.35))))
    if photo.any():
        v=weights[photo]*de[photo]
        loss["source_photo_horizontal"]=float(np.nansum(np.where(v<=2.7,.5*v*v,2.7*(v-1.35))))
    return e,de,loss


def camera_slots(record, slots):
    return {slots[shot["id"]]:shot["camera_index"] for shot in record["shots"]}


def limit_trajectory(problem, record, spans, source_times, envelope, slots):
    intervals={index:[] for index in set(slots.values())}
    for shot in record["shots"]:
        sid=shot["id"]
        default=clock(source_pts(shot))
        shift=spans[sid][0]*DT-FIRST
        extents=source_times.get(sid,default)[[0,-1]]+shift
        intervals[slots[sid]].append(np.nextafter(extents,[-np.inf,np.inf]))
    for index,camera in camera_slots(record,slots).items():
        problem.configure_camera_envelope(**envelope[str(camera)],frame_time_intervals=intervals[index],
                                          camera_index=index)


def solve(args):
    folder=OUTPUT/args.out
    folder.mkdir(exist_ok=False)
    log=Journal(folder/"solve.jsonl")
    stage=folder.name
    profile=read(args.rig_profile) if args.rig_profile else None
    local_profiles=read(args.shot_rig_profile) if args.shot_rig_profile else {}
    reused=read(args.reused_track_windows) if args.reused_track_windows else None
    shot_strides=read(args.shot_strides) if args.shot_strides else None
    shot_axes=read(args.shot_axes) if args.shot_axes else None
    envelope=read(args.trajectory_envelope) if args.trajectory_envelope else None
    path_weights=read(args.path_weights) if args.path_weights else None
    family_weights=read(args.family_weights) if args.family_weights else None
    protect_measurement_folds=getattr(args,"protect_measurement_folds",False)
    log("start",variant=stage,geometry_frozen=True,rigs_frozen=profile is None and not local_profiles,
        source_motion_differentials=True,
        coefficient_stride=args.stride,coarse_seed_stride=args.seed_stride,tracks=args.tracks,
        measured_points=args.observations,previous=args.previous,local_source_windows=args.windows,
        photo_registration=args.photo_anchors,
        signed_material_weight_scale=args.material_scale,reviewed_profile_weight_scale=args.reviewed_scale,
        source_track_triplet_scale=args.triplet_scale,trajectory_envelope=args.trajectory_envelope,
        installation_scalar_profile=profile,shot_rig_profiles=local_profiles,
        per_shot_raw_measurement_weights=path_weights,per_shot_family_weights=family_weights,
        preserve_track_holdouts=args.preserve_track_holdouts,protect_measurement_folds=protect_measurement_folds,
        reused_source_track_windows=args.reused_track_windows,shot_coefficient_strides=shot_strides,
        shot_trajectory_axes=shot_axes)
    start=monotonic()
    try:
        record=scope()
        native=frozen_scene(record,verify=True)
        windows=read(args.windows) if args.windows else None
        selected={shot["id"] for shot in record["shots"]
                  if (args.shots is None or shot["id"] in args.shots)
                  and (windows is None or shot["id"] in windows)}
        if set(local_profiles)-selected:
            raise ValueError("A bounded rig profile must name an active source shot")
        if profile is not None and local_profiles:
            raise ValueError("Select shared installation or per-shot calibration for this solve")
        anchors=load_photo_anchors(args.photo_anchors,record,selected) if args.photo_anchors else None
        rows,tracks,c,r,directions,spans,exposure,slots=data_bundle(
             record,OUTPUT/args.observations,OUTPUT/args.tracks,
             OUTPUT/args.previous,args.stride,args.seed_stride,args.trust,args.shots,windows,ice_roles(native),
             reused,shot_strides,args.preserve_track_holdouts,shot_axes,local_profiles,profile is not None,anchors,
             protect_measurement_folds)
        if not directions.shape[1] or not len(tracks["weight"]):
            raise ValueError("No qualifying motion constraints")
        tracks["triplet"]=(args.triplet_scale>0.)&(tracks["lag"]==1)&np.isfinite(tracks["prev2_uv"]).all(1)
        mask=track_samples(tracks)[0]
        with resident(native,rows,tracks,c,r) as initial:
            status=initial.evaluate("cuda")["status"]
        n=len(rows["pts"])
        keep=status[:n]==0
        track_status=np.zeros(mask.shape,np.int32)
        track_status[mask]=status[n:]
        stable=(track_status==0).all(axis=1)
        if not keep.all() or not stable.all():
            save(folder/"unprojectable_input_rows.npz",named_status=status[:n][~keep],
                 named_pts=rows["pts"][~keep],named_path=rows["path"][~keep],
                 pair_status=track_status[~stable],pair_shots=tracks["shot"][~stable],
                 pair_track_rows=tracks["shot_track_rows"][~stable])
            log("domain_failure",variant=stage,named_rows=int(np.count_nonzero(~keep)),
                track_pairs=int(np.count_nonzero(~stable)),status=dict(Counter(map(int,status[status!=0]))))
        rows={k:v[keep] for k,v in rows.items()}
        tracks={k:v[stable] for k,v in tracks.items()}
        target=folder/"camera.npz"
        with resident(native,rows,tracks,c,r) as problem:
            active=None
            cameras=camera_slots(record,slots)
            if profile is not None:
                if len(set(cameras.values()))!=len(cameras) or set(profile)!={str(i) for i in cameras.values()}:
                    raise ValueError("A shared installation profile needs one starting chart per installation")
                slot_profile={str(idx):profile[str(camera)] for idx,camera in cameras.items()}
                directions,active=rig_directions(problem,directions,r,slot_profile)
            elif local_profiles:
                slot_profile={str(idx):{} for idx in cameras}
                for sid,fields in local_profiles.items():
                    idx=slots[sid]
                    slot_profile[str(idx)]=fields
                    if set(fields)&{"pivot_x","pivot_y","pivot_z","lever_x","lever_y","lever_z"}:
                        side="negative_x_end" if cameras[idx]==2 else "negative_y"
                        problem.configure_mount_constraints(side=side,camera_index=idx)
                directions,active=rig_directions(problem,directions,r,slot_profile)
            if envelope is not None:
                limit_trajectory(problem,record,spans,exposure,envelope,slots)

            def save_state():
                checkpoint(target,problem.export_state(),record,spans,r,native,False,exposure,active,slots)
            layer,w,plan,gpu=configure(problem,rows,tracks,directions,args.material_scale,args.reviewed_scale,
                                       args.triplet_scale,path_weights,family_weights)
            e0,p0,loss=result(problem,rows,tracks,layer,w)
            photo_groups=int(np.count_nonzero(tracks["lag"]==0))
            log("prepared",variant=stage,rows=len(e0),pair_groups=len(p0)-photo_groups,
                photo_groups=photo_groups,columns=directions.shape[1],
                objective_by_layer=loss,blocks=plan,gpu=gpu,checkpoint=str(target))
            save_state()
            accepted=0
            small=0

            def update(detail):
                nonlocal accepted,small,loss
                summary=detail["summary"]
                if summary["accepted"]:
                    accepted+=1
                    save_state()
                    _,_,loss=result(problem,rows,tracks,layer,w)
                    ratio=abs(summary["actual_decrease"])/max(1.,abs(summary["old_cost"]))
                    small=small+1 if ratio<2.5e-5 else 0
                log("trial",variant=stage,rows=len(e0),retained_columns=directions.shape[1],
                    pair_groups=len(p0)-photo_groups,photo_groups=photo_groups,
                    objective_by_layer=loss,checkpoint=str(target),**detail)
                return not (small>=2 or (folder/"stop").exists())

            final=problem.solve(relative_cost_tolerance=7.e-6,step_tolerance=1.e-5,progress=update,
                   numerical_progress=lambda rec:log("factorization_retry",variant=stage,**rec))
            state=problem.export_state()
            checkpoint(target,state,record,spans,r,native,False,exposure,active,slots)
            e1,p1,loss=result(problem,rows,tracks,layer,w)
        save(folder/"active_rows.npz",**rows,station_fitted=state["stations"][:len(e1)],before_px=e0,after_px=e1,
             source_sha256=record["source"]["sha256"],geometry_sha256=record["geometry_binding"]["value_sha256"])
        movement=tracks["lag"]!=0
        save(folder/"used_track_pairs.npz",shot=tracks["shot"][movement],index=tracks["shot_track_rows"][movement],
                   lag=tracks["lag"][movement],family=tracks["family"][movement],triplet=tracks["triplet"][movement],
                   before=p0[movement],after=p1[movement])
        if not movement.all():
            write(folder/"photo_registration_fit.json",dict(source=args.photo_anchors,
                      before=statistics(p0[~movement]),after=statistics(p1[~movement])))
        per_shot={k:dict(absolute=statistics(e1[rows["shot"]==k]),
              motion=statistics(p1[(tracks["shot"]==k)&movement])) for k in record["required_shot_ids"]}
        report=dict(stage=stage,active_rows=len(e1),temporal_pairs=int(movement.sum()),after=statistics(e1),before=statistics(e0),
              temporal_before=statistics(p0[movement]),temporal_after=statistics(p1[movement]),per_shot=per_shot,
              source_triplet_groups=int(tracks["triplet"].sum()),trajectory_envelope=args.trajectory_envelope,
              accepted_trials=accepted,objective_by_layer=loss,solver=finite_json(final),
              seconds=monotonic()-start,scene_frozen_sha256=record["geometry_binding"]["value_sha256"],
              checkpoint=str(target),separate_motion_across_hard_cuts=True,installation_scalar_profile=profile,
              shot_rig_profiles=local_profiles,calibration_slots=slots,active_trajectory_shots=sorted(selected))
        write(folder/"fit.json",report)
        log("finish",variant=stage,objective_by_layer=loss,checkpoint=str(target),
            reason=final.get("stop_reason"),result=str(folder/"fit.json"))
    except BaseException as error:
        log("failure",variant=stage,reason=str(error))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--previous",default="fit07")
    p.add_argument("--tracks",default="steady_tracks")
    p.add_argument("--observations",default="read07")
    p.add_argument("--stride",type=int,default=3)
    p.add_argument("--seed-stride",type=int,default=7)
    p.add_argument("--trust",type=float,default=1.)
    p.add_argument("--material-scale",type=float,default=1.,
                   help="relative weight of fresh signed physical material profiles")
    p.add_argument("--reviewed-scale",type=float,default=1.,
                   help="relative weight of the source-reviewed edge profiles")
    p.add_argument("--triplet-scale",type=float,default=0.,
                   help="enable true three-exposure curvature measurements and set their relative weight")
    p.add_argument("--trajectory-envelope",help="Explicit native per-installation trial envelope")
    p.add_argument("--path-weights", help="Optional shot-to-path weights for measured source material")
    p.add_argument("--family-weights",help="Optional shot-to-family weights for measured source material")
    p.add_argument("--rig-profile",help="JSON mapping camera indices to bounded shared scalar fields")
    p.add_argument("--shot-rig-profile",help="JSON mapping cut-delimited shots to bounded rig scalars")
    p.add_argument("--reused-track-windows")
    p.add_argument("--preserve-track-holdouts", action="store_true",
                   help="Keep original spatial roles when enabling local camera directions")
    p.add_argument("--protect-measurement-folds", action="store_true",
                   help="Keep motion tracks out of reserved material observation frames and guard intervals")
    p.add_argument("--shot-strides",help="JSON overrides for the camera correction knot stride")
    p.add_argument("--shot-axes", help="Optional shot-to-list mapping of active pan, tilt, and log_focal axes")
    p.add_argument("--shots",nargs="+")
    p.add_argument("--windows")
    p.add_argument("--photo-anchors", help="Source-bound horizontal photo registration constraints on frozen world references")
    solve(p.parse_args())
