"""Compare independent PVA predictions with archived programme observations on native PTS."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .baseline_audit import metric
from .audit_fs_pva import derivatives
from experiments.skate_video_v1.deliver_body_skate_fit import bounds, panel_crop, put
from experiments.skate_video_v1.inputs import PROGRAMME, atomic_json, atomic_npz, read, emit
from rinkspace.viewer3d.source import SourceVideo


def report(path,saved,obs,episodes):
    with np.load(path) as p:
        stamps=p["sample_pts"].copy()
        t=p["source_seconds"].copy()
        uv=p["source_image_keypoints"].copy()
        conf=p["heatmap_confidence"].copy()
        vel=p["velocity_world_mps"].copy()
        accel=p["acceleration_world_mps2"].copy()
        valid=p["source_keypoints_valid"].copy()
        R=p["camera_R_w2c"].copy()
    pts=saved["source_pts"]
    i=np.searchsorted(pts,stamps)
    np.testing.assert_array_equal(pts[i],stamps)
    j=np.arange(5,17)
    src=obs[i]
    tgt=saved["COCO_uv"][i]
    world=saved["COCO_world_mid"][i]
    truth_v,truth_a=derivatives(world,t)
    pv=vel[:,j]
    pa=accel[:,j]
    nrm=lambda x:np.linalg.norm(x,axis=-1)
    torso=np.array([5,6,11,12])
    pelvis_v=np.diff(saved["P_world_mid"][i],axis=0)/np.diff(t)[:,None]
    relative=lambda x:(x-x[:,[11,12]].mean(1)[:,None])[:,j]
    visible=valid & np.isfinite(src).all(-1)
    data=dict(pts=stamps,t=t,observed_uv=src,model_uv=tgt,pva_uv=uv,
              pva_velocity=vel,pva_accel=accel,base_velocity=truth_v,base_accel=truth_a,visibility=visible,confidence=conf)

    def window(start,stop):
        pick=(t>=start-1e-8)&(t<=stop+1e-8)
        vv=pick[:-1]&pick[1:]
        aa=pick[:-2]&pick[1:-1]&pick[2:]
        if not aa.any():
            return
        common=visible[pick][:,j]
        vd=pv[vv]-truth_v[vv][:,j]
        dd=pa[aa]-truth_a[aa][:,j]
        return dict(seconds=[start,stop],exposures=int(pick.sum()),source_native_points=int(common.sum()),
          body_to_existing_detector_px=metric(nrm(tgt[pick][:,j]-src[pick][:,j])[common]),
          pva_to_existing_detector_px=metric(nrm(uv[pick][:,j]-src[pick][:,j])[common]),
          pva_existing_world_velocity_difference_mps=metric(nrm(vd)),
          pva_existing_world_relative_velocity_difference_mps=metric(nrm(relative(vel[vv])-relative(truth_v[vv]))),
          pva_torso_speed_mps=metric(nrm(vel[vv][:,torso].mean(1))),
          body_torso_speed_mps=metric(nrm(truth_v[vv][:,torso].mean(1))),
          pelvis_speed_mps=metric(nrm(pelvis_v[vv])),
          pva_existing_acceleration_difference_mps2=metric(nrm(dd)),
          pva_existing_optical_accel_difference_mps2=metric(abs(np.einsum("ni,nki->nk",R[:-2][aa,2],dd))))

    t0,t1=map(float,t[[0,-1]])
    out=dict(source_pts=stamps[[0,-1]].tolist(),clips={"whole":window(t0,t1)})
    for name,start,stop in episodes:
        if start>=t0 and stop<=t1:
            v=window(start,stop)
            if v is not None:
                out["clips"][name]=v
    return out,data


def fs_compare(path):
    """Quantify how much of the lower acceleration error is also reached by a simple smoother."""
    with np.load(path) as f:
        t=f["source_seconds"]
        x=f["baseline_world_m"]
        v,a=f["reference_velocity"],f["reference_acceleration"]
        pv,pa=f["pva_velocity"],f["pva_acceleration"]
        okv,oka=f["velocity_valid"],f["acceleration_valid"]
    dv,da=derivatives(savgol_filter(x,7,2,axis=0),t)
    errs=lambda a,b,m:metric(np.linalg.norm(a-b,axis=-1)[m])
    den=np.linalg.norm(v,axis=-1)
    common=okv&(den>=.7)&(np.linalg.norm(dv,axis=-1)>=.1)&(np.linalg.norm(pv,axis=-1)>=.1)
    angles={}
    for tag,x in (("pva",pv),("exmr_sg_7",dv)):
        cosine=(x*v).sum(-1)/np.maximum(den*np.linalg.norm(x,axis=-1),1e-10)
        angles[tag]=metric(np.rad2deg(np.arccos(np.clip(cosine[common],-1,1))))
    return dict(matched_fs_smoothing_control={"velocity_error_mps":errs(dv,v,okv),
               "acceleration_error_mps2":errs(da,a,oka)},zero_acceleration_error_mps2=errs(np.zeros_like(a),a,oka),
               pva_velocity_and_acceleration_same_reference_rows=[errs(pv,v,okv),errs(pa,a,oka)],direction_common_rows=angles)


def witness_image(job,source,tags):
    video=SourceVideo(source["registered_path"],(1,12800))
    rows=[]
    for tag,data,seconds in tags:
        for query in seconds:
            t=data["t"]
            i=int(np.argmin(abs(t-query)))
            p=data["pts"][i]
            img=cv2.cvtColor(video.read(p),cv2.COLOR_RGB2BGR)
            uv=[data[k][i] for k in ("model_uv","pva_uv")]
            options=[img,img.copy(),img.copy()]
            box=bounds(uv,38,img.shape)
            for which,xy in enumerate(uv,1):
                for a,b in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
                    line=xy[[a,b]]
                    if not np.isfinite(line).all():
                        continue
                    col=(75,229,247) if which==1 else (107,234,142)
                    q,r=np.rint(line).astype(np.int32)
                    cv2.line(options[which],q,r,col,2,cv2.LINE_AA)
                    cv2.circle(options[which],q,3,col,-1,cv2.LINE_AA)
                    cv2.circle(options[which],r,3,col,-1,cv2.LINE_AA)
            out=np.full((230,1080,3),23,np.uint8)
            labels=(f"{tag.upper()}  Source {t[i]:.3f}s  PTS {p}","Current native projection","PVA 2D from the source")
            for k,(x,name) in enumerate(zip(options,labels,strict=True)):
                put(out,name,k*360+9,22,.53)
                panel_crop(out[31:,k*360:(k+1)*360],x,box)
            rows.append(out)
    video.close()
    for i in range(0,len(rows),4):
        cv2.imwrite(str(job/f"pva_image_audit_{i//4+1}.jpg"),np.vstack(rows[i:i+4]),[cv2.IMWRITE_JPEG_QUALITY,94])


def main(args):
    obs=np.load(args.observation,allow_pickle=False)
    base=np.load(args.result,allow_pickle=False)
    np.testing.assert_array_equal(base["source_pts"],obs["pts"])
    native=np.full(obs["uv"].shape,np.nan)
    row,col=np.where(obs["use"])
    native[row,obs["model_joint_ids"][row,col]]=obs["uv"][row,col]
    doc=read(args.evidence)
    episodes=[(r["episode_id"],r["start_pts"]/12800.,r["end_pts_inclusive"]/12800.)
              for r in doc["source_episode_review"]["episodes"]
              if r["kind"] in ("flight_visible_candidate","support_candidate")]
    episodes.extend([("spin_high",87.50,92.50),("exit_glide",92.74,94.44)])
    summary={}
    proof=[]
    for name,seconds in [("s01",[25.40,30.26]),("s02",[48.42,49.78]),("s05",[81.40,81.44,88.84,93.82])]:
        x,data=report(args.job/"pva_results"/(name+".npz"),base,native,episodes)
        summary[name]=x
        proof.append((name,data,seconds))
        atomic_npz(args.job/"pva_results"/(name+"_review_values.npz"),**data)
    fs=fs_compare(args.job/"fs_axel_with_pva.npz")
    atomic_json(args.job/"pva_programme_audit.json",dict(plots_input=summary,fs_controls=fs,
       model_trajectory_comparison_not_ground_truth=True, detector_2d_previously_used=True))
    witness_image(args.job,read(PROGRAMME/"rebuild_scope.json")["source"],proof)
    emit("pva_programme_audit_complete",s01=summary["s01"]["clips"]["whole"],s05=summary["s05"]["clips"]["whole"],
         fs_controls=fs)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for k in ("result","observation","evidence","job"):
        cli.add_argument("--"+k,type=Path,required=True)
    main(cli.parse_args())
