"""Build PVA-Net crops from actual source exposures and keep the source clock."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import hashlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.action_clip_input import camera25_from_snapshot
from calibration.program_tracking.body_tracking.camera_bridge import native_to_rectified, rectified_intrinsics
from calibration.program_tracking.body_tracking.fs_jump3d_validation import camera as fs_camera
from experiments.skate_video_v1.inputs import atomic_json, digest, emit, read
from experiments.skate_video_v1.programme_motion_followup import bind
from rinkspace.review.render import source_frames


def sequence_clock(pts, tb, start, stop):
    native = pts * float(tb)
    grid = start + np.arange(int(np.floor((stop-start)*30 + 1.e-7))+1) / 30.
    j = np.searchsorted(native, grid).clip(1, len(pts)-1)
    j -= (grid-native[j-1] <= native[j]-grid)
    if np.any(np.diff(j) <= 0) or np.max(np.abs(native[j]-grid)) > float(tb) * np.diff(pts).max()/2 + 1.e-8:
        raise ValueError("No unique nearby native exposures")
    return grid, j


def full_rect_crop(frame, params, native_box):
    width, height = 192, 256
    x0, y0, x1, y1 = native_box
    t = np.linspace(0.,1.,9)
    x, y = x0+(x1-x0)*t, y0+(y1-y0)*t
    border = np.concatenate([np.c_[x,np.full_like(x,y0)],np.c_[x,np.full_like(x,y1)],
                             np.c_[np.full_like(y,x0),y],np.c_[np.full_like(y,x1),y]])
    rect_k = rectified_intrinsics(params)
    rect, valid = native_to_rectified(border,params,K_rect=rect_k,native_size_wh=frame.shape[1::-1])
    if valid.sum() < 16:
        raise ValueError("Person box misses valid native Brown rays")
    lo, hi = rect[valid].min(axis=0), rect[valid].max(axis=0)
    centre = (lo+hi)/2
    sh = 1.16*max(hi[1]-lo[1],(hi[0]-lo[0])/.75)
    sw = .75*sh
    left, top = centre - np.array([sw,sh])/2.
    source_k = np.array([[np.exp(params[6]),0.,params[8]],[0.,np.exp(params[7]),params[9]],[0.,0.,1.]])
    crop_from_rect = np.array([[(width-1)/sw,0.,-(width-1)*left/sw],
                               [0.,(height-1)/sh,-(height-1)*top/sh],[0.,0.,1.]])
    mx,my = cv2.initUndistortRectifyMap(source_k,params[11:],np.eye(3),crop_from_rect@rect_k,
                                        (width,height),cv2.CV_32FC1)
    crop=cv2.remap(frame,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    inside=(mx>=0)&(mx<frame.shape[1]-1)&(my>=0)&(my<frame.shape[0]-1)
    return crop[...,::-1],rect_k,np.array([left,top,left+sw,top+sh]),float(inside.mean())


def body_boxes(primary, secondary, pts):
    boxes=np.asarray(primary,dtype=np.float64).copy()
    good=np.isfinite(boxes).all(axis=1)&((boxes[:,2:]-boxes[:,:2])>20).all(axis=1)
    where=np.searchsorted(pts,secondary["source_pts"])
    fill=where[~good[where]]
    boxes[fill]=secondary["roi_native_xyxy"][np.searchsorted(secondary["source_pts"],pts[fill])]
    valid=np.isfinite(boxes).all(axis=1)&((boxes[:,2:]-boxes[:,:2])>20).all(axis=1)
    for col in range(4):
        boxes[:,col]=np.interp(pts,pts[valid],boxes[valid,col])
    return boxes, ~good



def extend_broadcast_boxes(boxes, prepared, roles_path, pts):
    with np.load(prepared/"observations.npz",allow_pickle=False) as p, np.load(roles_path,allow_pickle=False) as w:
        np.testing.assert_array_equal(p["pts"],pts)
        np.testing.assert_array_equal(w["source_pts"],pts)
        xy=p["uv"]
        valid=p["use"] & (p["sigma_px"]<=36.) & np.isfinite(xy).all(-1)
        visible=w["visible"] & (w["role"]>0) & np.isfinite(w["region"]).all(axis=(2,3))
        v=w["region"].reshape(len(pts),-1,2)
        have=np.repeat(visible,w["region"].shape[2],axis=1)
    extent=np.maximum(boxes[:,2:]-boxes[:,:2],30.)
    # Keep broad image evidence near the existing skater crop. A single
    # distant background detection cannot force the crop to full-frame.
    lo=boxes[:,:2]-extent*.30
    hi=boxes[:,2:]+extent*.30
    def near(x):
        return ((x>=lo[:,None])&(x<=hi[:,None])).all(-1)
    candidates=np.concatenate([xy,v],axis=1)
    use=np.concatenate([valid&near(xy),have&near(v)],axis=1)
    left=np.minimum(boxes[:,:2],np.where(use[...,None],candidates,np.inf).min(1))
    right=np.maximum(boxes[:,2:],np.where(use[...,None],candidates,-np.inf).max(1))
    base=np.c_[left,right]
    return base


def write_clips(output, tag, frames, camera25, rot, pts, tb, metadata, boxes, inferred, stop):
    grid, rows = sequence_clock(pts,tb,float(metadata["start_seconds"]),stop)
    output.mkdir(exist_ok=True, parents=True)
    uv, K, rgb, coverage, checks = [], [], [], [], []
    wanted=pts[rows]
    for n,(stamp,bgr) in enumerate(frames(wanted)):
        i=rows[n]
        assert stamp==pts[i]
        crop,k,box,inside=full_rect_crop(bgr,camera25[i],boxes[i])
        rgb.append(crop)
        K.append(k)
        uv.append(box)
        coverage.append(inside)
        checks.append(hashlib.sha256(bgr.tobytes()).hexdigest())
        if n%150==0:
            emit("pva_input_native_frames",clip=tag,decoded=n+1,pts=int(stamp))
    assert len(rgb)==len(rows)
    path=output/(tag+".npz")
    pending=path.with_suffix(".pending.npz")
    np.savez(pending,crops_rgb=np.asarray(rgb,np.uint8),bbx_xyxy=np.asarray(uv,np.float32),
             native_resolution=np.asarray(metadata["native_resolution"]),
             K_rect=np.asarray(K,np.float32),camera25=camera25[rows],R_w2c=rot[rows],sample_pts=wanted,
             source_time_base=np.array([tb.numerator,tb.denominator]),nominal_seconds=grid,
             source_seconds=wanted*float(tb),box_inferred=inferred[rows],valid_fraction=coverage,
             source_frame_bgr_sha256=np.asarray(checks),source_frame_index=rows)
    pending.replace(path)
    metadata.update(path=str(path),sha256=digest(path),frequency_hz=30,source_exposures=len(rows),
                    end_seconds=stop,native_pts=wanted[[0,-1]].tolist(),
                    maximum_sample_time_error_ms=float(1000*np.max(np.abs(wanted*float(tb)-grid))),
                    box_inferred_count=int(inferred[rows].sum()), min_valid_crop_fraction=min(coverage),
                    crop="192x256 RGB, calibrated Brown ray remap to rectified K, 1.16 times source person-box extent")
    atomic_json(output/(tag+".json"),metadata)
    emit("pva_input_ready",clip=tag,**metadata)


def broadcast(args):
    ref, camera, info, _, _ = bind(args.prepared,args.evidence,args.parameter)
    ctx=read(args.scope)["source"]
    video=Path(ctx["registered_path"])
    if digest(video)!=ctx["sha256"]:
        raise ValueError("Programme source bytes changed")
    with np.load(args.prepared/"input.npz",allow_pickle=False) as p, np.load(args.boxes,allow_pickle=False) as ex:
        pts = p["pts"]
        np.testing.assert_array_equal(pts,ref.p["pts"])
        if str(ex["camera_sha256"]) != info["state_sha256"]:
            raise ValueError("Box source camera identity differs")
        boxes,inferred = body_boxes(p["source_bbox"],ex,pts)
    boxes=extend_broadcast_boxes(boxes,args.prepared,args.roles,pts)
    tb=Fraction(*info["source_time_base"])
    spec=[("s05",76.,95.),("s02",44.98,52.),("s01",25.,35.)]
    camera25 = camera25_from_snapshot(camera.midpoint)
    for tag,start,stop in spec:
        grid, ix = sequence_clock(pts,tb,start,stop)
        if not (ref.p["shot_ids"][ix]==tag).all():
            raise ValueError("PVA windows must belong to a single camera shot")
        if (args.output/(tag+".npz")).is_file():
            continue
        # Read the contiguous original source interval once, then keep only the
        # requested exposures. source_frames enforces exact source PTS.
        from scipy.ndimage import maximum_filter1d, median_filter
        shot_rows=np.flatnonzero(ref.p["shot_ids"]==tag)
        centre=boxes[shot_rows].reshape(-1,2,2).mean(1)
        span=boxes[shot_rows,2:]-boxes[shot_rows,:2]
        neighbour=maximum_filter1d(span,size=7,axis=0,mode="nearest")
        # Preserve rapid movement, but widen for limbs intermittently lost by
        # the detector. Short smoothing of extent avoids crop-scale flicker.
        guard=np.maximum(span,median_filter(neighbour,size=(3,1),mode="nearest"))
        shot_boxes=boxes.copy()
        shot_boxes[shot_rows]=np.c_[centre-guard*.55,centre+guard*.55]
        wanted=pts[ix]
        continuous=pts[ix[0]:ix[-1]+1]
        def frames(_, selected=wanted, span=continuous):
            use=set(map(int,selected))
            return ((t,bgr) for t,bgr in source_frames(video,dict(width=ctx["width"],height=ctx["height"],time_base_num=tb.numerator,time_base_den=tb.denominator),span) if t in use)
        write_clips(args.output,tag,frames,camera25,camera.midpoint["rotation"],pts,tb,
                    dict(start_seconds=start,source=str(video),source_video_sha256=ctx["sha256"],
                         camera_sha256=info["state_sha256"],parameter_sha256=digest(args.parameter),shot=tag,
                         bbox_source=str(args.prepared/"input.npz"),alternate_bbox_source=str(args.boxes),source_extent_roles=str(args.roles),native_resolution=[ctx["width"],ctx["height"]]),shot_boxes,inferred,stop)


def fs_trial(args):
    root=args.fs
    config=read(root/"video_clocks.json")["7"]
    video=Path(config["path"])
    if digest(video)!=config["sha256"]:
        raise ValueError("FS-Jump3D cam7 video hash changed")
    with np.load(root/"input_cam7/prepared.npz",allow_pickle=False) as p:
        pts=p["source_pts"]
        tb=Fraction(*p["source_time_base"])
        crop=p["bbox_xyxy"]
        A=p["rect_to_input_affine"]
        lo=(crop[:,:2]-A[:,:2,2])/A[:,0,0,None]
        hi=(crop[:,2:]-A[:,:2,2])/A[:,0,0,None]
        rect=np.c_[lo,hi]
    cal=fs_camera(read(root/"official_cam_param.json")["fs_camera_7"])
    R=cal["R_w2c"]
    K=cal["K"]
    c=np.zeros(25)
    c[:3]=cal["C_world_m"]
    c[3:6]=Rotation.from_matrix(R).as_rotvec()
    c[6:8]=np.log([K[0,0],K[1,1]])
    c[8:10]=K[:2,2]
    c[11:16]=cal["distortion_opencv"]
    from calibration.program_tracking.body_tracking.camera_bridge import rectified_to_native
    points=np.stack([rect[:,:2],rect[:,2:],rect[:,[0,3]],rect[:,[2,1]]],axis=1)
    native=[]
    for box in points:
        uv,_=rectified_to_native(box,c,native_size_wh=(1920,1088))
        native.append(np.r_[uv.min(0),uv.max(0)])
    # Inferred source boxes are recorded by the acquisition; reference joints
    # are intentionally unused in PVA input creation.
    def frames(selected):
        use=set(map(int,selected))
        return ((t,bgr) for t,bgr in source_frames(video,dict(width=1920,height=1088,time_base_num=tb.numerator,time_base_den=tb.denominator),pts) if t in use)
    write_clips(args.output,"fs_axel_cam7",frames,np.repeat(c[None],len(pts),axis=0),
                np.repeat(R[None],len(pts),axis=0),pts,tb,
                dict(start_seconds=0.,source=str(video),source_video_sha256=config["sha256"],shot="cam7",
                     camera_sha256=digest(root/"official_cam_param.json"),
                     bbox_source=str(root/"input_cam7/prepared.npz"),native_resolution=[1920,1088]),np.asarray(native),np.zeros(len(pts),bool),149/30.)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    p=ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    for name in ("prepared","evidence","parameter","scope","boxes","fs","roles"):
        p.add_argument("--"+name,type=Path)
    args=p.parse_args()
    if args.fs:
        fs_trial(args)
    else:
        missing = [name for name in ("prepared", "evidence", "parameter", "scope", "boxes", "roles")
                   if getattr(args, name) is None]
        if missing:
            p.error("Broadcast preparation requires " + ", ".join("--" + name for name in missing))
        broadcast(args)
