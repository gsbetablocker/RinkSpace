"""Source-only bidirectional LK transport of finite boot-contour regions.

The cache holds RGB navigation crops, not masks or material correspondences.
Only explicit boot_contour_axis anchors seed this preprocessing. No body result,
camera projection, fit result or network is read or evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLICY = dict(min_features=6, min_inlier_fraction=.70, feature_quality=.01,
              feature_min_distance=2, feature_block_size=3, feature_max_corners=80,
              feature_radius_cache_px=22, fb_median_cache_px=.75,
              fb_p90_cache_px=1.5, affine_p90_cache_px=1.5,
              ransac_cache_px=1.5, appearance_zncc=.65,
              min_feature_hull_area_cache_px2=24.,
              similarity_scale_min=.80, similarity_scale_max=1.25,
              endpoint_roundtrip_native_px=3.0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def cache_to_native(bounds):
    # PIL pixel centres: native=(cache+.5)*(right-left)/160+left-.5.
    b=np.asarray(bounds,float)
    scale=(b[2:]-b[:2])/160.
    return np.array([[scale[0],0,b[0]+.5*scale[0]-.5],
                     [0,scale[1],b[1]+.5*scale[1]-.5],[0,0,1.]])


def apply(matrix,points):
    p=np.asarray(points,float)
    return p@matrix[:2,:2].T+matrix[:2,2]


def select_view(cache,index,centre,preferred=None):
    choices=[]
    for view in (0,1):
        method=str(cache["feet_navigation_method"][index,view])
        if method not in ("source_qualified_ankle_uv","exact_pts_source_shoe_region"):
            continue
        T=cache_to_native(cache["feet_bounds_native_xyxy"][index,view])
        q=apply(np.linalg.inv(T),np.asarray(centre)[None])[0]
        margin=float(np.min(np.r_[q,159-q]))
        if margin<10:continue
        choices.append((margin+(3 if view==preferred else 0),view,T,q))
    if not choices:return None
    _,view,T,q=max(choices,key=lambda r:r[0])
    return dict(view=view,T=T,centre=q,gray=cv2.cvtColor(cache["feet_rgb"][index,view],cv2.COLOR_RGB2GRAY))


def local_mask(centre,radius):
    y,x=np.mgrid[:160,:160]
    return (((x-centre[0])**2+(y-centre[1])**2)<=radius**2).astype(np.uint8)*255


def zncc(a,b,mask):
    x=np.asarray(a,float)[mask];y=np.asarray(b,float)[mask]
    if len(x)<30 or x.std()<3 or y.std()<3:return None
    x-=x.mean();y-=y.mean()
    return float(x@y/np.sqrt((x@x)*(y@y)))


def similarity(a,b):
    M,inliers=cv2.estimateAffinePartial2D(np.asarray(a,np.float32),np.asarray(b,np.float32),
        method=cv2.RANSAC,ransacReprojThreshold=POLICY["ransac_cache_px"],maxIters=1000,confidence=.99)
    if M is None or inliers is None:return None,None
    H=np.eye(3);H[:2]=M
    return H,inliers.reshape(-1).astype(bool)


def step_flow(previous,current,centre_native,cumulative,anchor_view,anchor_centre):
    """Track texture supports around a region; supports are not template points."""
    centre=apply(np.linalg.inv(previous["T"]),np.asarray(centre_native)[None])[0]
    mask=local_mask(centre,POLICY["feature_radius_cache_px"])
    points=cv2.goodFeaturesToTrack(previous["gray"],maxCorners=POLICY["feature_max_corners"],
        qualityLevel=POLICY["feature_quality"],minDistance=POLICY["feature_min_distance"],
        mask=mask,blockSize=POLICY["feature_block_size"])
    info={"feature_count":0 if points is None else len(points)}
    if points is None or len(points)<POLICY["min_features"]:return None,"insufficient_local_texture",info
    native=apply(previous["T"],points[:,0])
    initial=apply(np.linalg.inv(current["T"]),native).astype(np.float32)[:,None]
    args=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,status,_=cv2.calcOpticalFlowPyrLK(previous["gray"],current["gray"],points,initial.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**args)
    if forward is None:return None,"forward_LK_failed",info
    back,back_status,_=cv2.calcOpticalFlowPyrLK(current["gray"],previous["gray"],forward,points.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**args)
    if back is None:return None,"backward_LK_failed",info
    fb=np.linalg.norm(back[:,0]-points[:,0],axis=1)
    good=status[:,0].astype(bool)&back_status[:,0].astype(bool)&np.isfinite(fb)
    good&=(forward[:,0]>=2).all(1)&(forward[:,0]<=157).all(1)
    info["LK_bidirectional_count"]=int(good.sum())
    if good.sum()<POLICY["min_features"]:return None,"insufficient_bidirectional_LK",info
    info.update(fb_median_cache_px=float(np.median(fb[good])),fb_p90_cache_px=float(np.quantile(fb[good],.9)))
    if info["fb_median_cache_px"]>POLICY["fb_median_cache_px"] or info["fb_p90_cache_px"]>POLICY["fb_p90_cache_px"]:
        return None,"forward_backward_residual",info
    good&=fb<=POLICY["fb_p90_cache_px"]
    a,b=points[good,0],forward[good,0]
    if len(a)<POLICY["min_features"]:return None,"insufficient_FB_inliers",info
    M,inliers=similarity(a,b)
    if M is None:return None,"robust_similarity_failed",info
    info.update(inlier_count=int(inliers.sum()),inlier_fraction=float(inliers.mean()))
    if inliers.sum()<POLICY["min_features"] or inliers.mean()<POLICY["min_inlier_fraction"]:
        return None,"insufficient_similarity_support",info
    a,b=a[inliers],b[inliers]
    hull=float(cv2.contourArea(cv2.convexHull(a.astype(np.float32))))
    info["feature_hull_area_cache_px2"]=hull
    if hull<POLICY["min_feature_hull_area_cache_px2"]:return None,"degenerate_feature_geometry",info
    residual=np.linalg.norm(apply(M,a)-b,axis=1)
    info["similarity_p90_cache_px"]=float(np.quantile(residual,.9))
    if info["similarity_p90_cache_px"]>POLICY["affine_p90_cache_px"]:return None,"similarity_residual",info
    scale=float(np.sqrt(np.linalg.det(M[:2,:2])))
    info["similarity_scale"]=scale
    if not POLICY["similarity_scale_min"]<=scale<=POLICY["similarity_scale_max"]:
        return None,"local_scale_or_turning_ambiguity",info
    # Independently robust-fit the backward correspondences, preserving FB noise.
    B,bi=similarity(b,back[good,0][inliers])
    if B is None or bi.sum()<POLICY["min_features"]:return None,"backward_similarity_failed",info
    native_M=current["T"]@M@np.linalg.inv(previous["T"])
    native_B=previous["T"]@B@np.linalg.inv(current["T"])
    H=native_M@cumulative
    new_centre=apply(H,np.asarray(anchor_centre)[None])[0]
    reverse_centre=apply(np.linalg.inv(cumulative)@native_B,new_centre[None])[0]
    endpoint_error=float(np.linalg.norm(reverse_centre-anchor_centre))
    info["anchor_endpoint_roundtrip_native_px"]=endpoint_error
    if endpoint_error>POLICY["endpoint_roundtrip_native_px"]:return None,"anchor_endpoint_roundtrip_failed",info
    # Cumulative direct anchor-image appearance check prevents local background drift.
    crop_H=np.linalg.inv(current["T"])@H@anchor_view["T"]
    warped=cv2.warpAffine(anchor_view["gray"],crop_H[:2],(160,160),flags=cv2.INTER_LINEAR)
    native_anchor_patch=local_mask(anchor_view["centre"],18)
    warped_mask=cv2.warpAffine(native_anchor_patch,crop_H[:2],(160,160),flags=cv2.INTER_NEAREST)>0
    valid_mask=cv2.warpAffine(np.ones((160,160),np.uint8),crop_H[:2],(160,160),flags=cv2.INTER_NEAREST)>0
    correlation=zncc(warped,current["gray"],warped_mask&valid_mask)
    info["anchor_patch_zncc"]=correlation
    if correlation is None or correlation<POLICY["appearance_zncc"]:return None,"anchor_appearance_or_occlusion_change",info
    # Conservative source-native error envelope, not calibrated covariance.
    step_error=max(info["fb_p90_cache_px"],info["similarity_p90_cache_px"],.25)*float(max(np.diag(current["T"])[:2]))
    F=np.eye(2)*step_error**2
    info.update(native_error_envelope_radius_px=step_error,texture_supports_are_not_fixed_material_points=True)
    return dict(centre=new_centre,H=H,native_M=native_M,flow_error=F,
                native_supports_previous=apply(previous["T"],a),native_supports_current=apply(current["T"],b)),None,info


def endpoint_backward_check(cache,start,endpoint,endpoint_centre,anchor_centre,preferred):
    """Re-detect texture at the endpoint and replay the entire chain to its seed."""
    view=select_view(cache,endpoint,endpoint_centre,preferred)
    if view is None:return False,{"reason":"endpoint_source_crop_missing"}
    endpoint_view=view;H=np.eye(3);centre=np.asarray(endpoint_centre,float)
    direction=-1 if endpoint>start else 1
    counts=[];fb=[]
    for index in range(endpoint+direction,start+direction,direction):
        target=select_view(cache,index,centre,view["view"])
        if target is None:return False,{"reason":"reverse_source_crop_missing","failed_pts":int(cache["pts"][index])}
        result,reason,info=step_flow(view,target,centre,H,endpoint_view,np.asarray(endpoint_centre,float))
        if reason:return False,{"reason":reason,"failed_pts":int(cache["pts"][index]),"detail":info}
        centre=result["centre"];H=result["H"];view=target
        counts.append(info["inlier_count"]);fb.append(info["fb_p90_cache_px"])
    error=float(np.linalg.norm(centre-anchor_centre))
    info={"source_anchor_pts":int(cache["pts"][start]),"endpoint_pts":int(cache["pts"][endpoint]),
          "full_reverse_steps":len(counts),"endpoint_recovered_anchor_native_uv":centre.tolist(),
          "manual_anchor_native_uv":np.asarray(anchor_centre,float).tolist(),
          "endpoint_to_manual_anchor_error_native_px":error,
          "minimum_reverse_inliers":min(counts),"maximum_reverse_FB_p90_cache_px":max(fb),
          "endpoint_features_redetected_independently":True}
    if error>POLICY["endpoint_roundtrip_native_px"]:info["reason"]="full_reverse_anchor_disagreement"
    return error<=POLICY["endpoint_roundtrip_native_px"],info


def track_branch(cache,anchor,feature,direction,protected_pts,propagated_region_shape):
    pts=cache["pts"];start=int(np.searchsorted(pts,anchor["pts"]))
    c=np.asarray(anchor[feature+"_uv"],float)
    S=np.diag(2*np.asarray(anchor[feature+"_half_extent_uv"],float)**2)
    role=anchor["role"]
    branch=dict(id=f"{anchor['id']}_{feature}_{'forward' if direction>0 else 'backward'}",anchor_id=anchor["id"],
        anchor_pts=int(anchor["pts"]),foot=anchor["foot"],feature=feature,role=role,direction=direction,steps=[])
    view=select_view(cache,start,c,anchor["foot"])
    if view is None:
        branch["failure"]={"pts":int(pts[start]),"reason":"anchor_not_in_qualified_source_crop"}
        return branch,[]
    seed_view=view
    H=np.eye(3);F=np.zeros((2,2));previous=start;records=[]
    for index in range(start+direction,len(pts) if direction>0 else -1,direction):
        pt=int(pts[index])
        if role=="fit" and pt in protected_pts:
            branch["failure"]={"pts":pt,"reason":"protected_review_exposure_barrier"};break
        target=select_view(cache,index,c,view["view"])
        if target is None:
            branch["failure"]={"pts":pt,"reason":"missing_or_distant_qualified_source_crop"};break
        result,reason,info=step_flow(view,target,c,H,seed_view,np.asarray(anchor[feature+"_uv"],float))
        step={"from_pts":int(pts[previous]),"pts":pt,"source_crop_view":target["view"],**info}
        if reason:
            step["status"]="ambiguous";step["failure_reason"]=reason
            branch["steps"].append(step);branch["failure"]={"pts":pt,"reason":reason};break
        endpoint_ok,endpoint_info=endpoint_backward_check(cache,start,index,result["centre"],np.asarray(anchor[feature+"_uv"],float),target["view"])
        step["full_endpoint_backward_check"]=endpoint_info
        if not endpoint_ok:
            step.update(status="ambiguous",failure_reason="full_endpoint_backward_check_failed")
            branch["steps"].append(step);branch["failure"]={"pts":pt,"reason":step["failure_reason"]};break
        F=result["native_M"][:2,:2]@F@result["native_M"][:2,:2].T+result["flow_error"]
        H=result["H"];c=result["centre"]
        shape=propagated_region_shape(S,H[:2,:2],F)
        radius=float(np.sqrt(np.linalg.eigvalsh(shape)[-1]))
        support_radius=POLICY["feature_radius_cache_px"]*float(min(np.diag(target["T"])[:2]))
        if radius>support_radius:
            step.update(status="ambiguous",failure_reason="region_uncertainty_exceeds_local_texture_support",region_radius_native_px=radius)
            branch["steps"].append(step);branch["failure"]={"pts":pt,"reason":step["failure_reason"]};break
        step.update(status="inferred",centre_uv=c.tolist(),region_shape_px2=shape.tolist(),native_affine=H.tolist(),
                    native_step_affine=result["native_M"].tolist(),native_flow_error_shape_px2=F.tolist(),
                    native_supports_previous=result["native_supports_previous"].tolist(),native_supports_current=result["native_supports_current"].tolist())
        branch["steps"].append(step)
        records.append(dict(sequence_index=index,pts=pt,feature=feature,foot=anchor["foot"],role=role,
             centre=c.copy(),shape=shape.copy(),anchor_id=anchor["id"],anchor_pts=int(anchor["pts"]),branch_id=branch["id"],diagnostic=info))
        previous=index;view=target
    branch["accepted_inferred_exposures"]=len(records)
    return branch,records


def cards(output,cache,anchors,branches):
    font_path=Path("C:/Windows/Fonts/consola.ttf")
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    outputs=[]
    for anchor in anchors:
        relevant=[b for b in branches if b["anchor_id"]==anchor["id"]]
        steps=[s for b in relevant for s in b["steps"] if s["status"]=="inferred"]
        times=sorted({anchor["pts"]}|{s["pts"] for s in steps})
        selected=sorted({times[0],anchor["pts"],times[-1]})
        card=Image.new("RGB",(430*len(selected),350),(24,28,35));draw=ImageDraw.Draw(card)
        draw.text((8,5),f"SOURCE TRACK | {anchor['id']} | {anchor['role']} | finite boot contour only",font=font,fill="white")
        cells=[]
        for j,pt in enumerate(selected):
            index=int(np.searchsorted(cache["pts"],pt));centre=np.mean([anchor["heel_uv"],anchor["toe_uv"]],axis=0)
            at=[s for b in relevant for s in b["steps"] if s["status"]=="inferred" and s["pts"]==pt]
            if at:centre=np.mean([s["centre_uv"] for s in at],axis=0)
            view=select_view(cache,index,centre,anchor["foot"])
            if view is None:view=dict(view=anchor["foot"],T=cache_to_native(cache["feet_bounds_native_xyxy"][index,anchor["foot"]]))
            foot=Image.fromarray(cache["feet_rgb"][index,view["view"]]).resize((240,240),Image.Resampling.NEAREST)
            fd=ImageDraw.Draw(foot)
            for feature,color in [("heel",(255,220,30)),("toe",(40,220,255))]:
                if pt==anchor["pts"]:c=np.asarray(anchor[feature+"_uv"],float);S=np.diag(2*np.asarray(anchor[feature+"_half_extent_uv"],float)**2);status="observed"
                else:
                    found=next((s for b in relevant if b["feature"]==feature for s in b["steps"] if s["status"]=="inferred" and s["pts"]==pt),None)
                    if found is None:continue
                    c=np.asarray(found["centre_uv"]);S=np.asarray(found["region_shape_px2"]);status="inferred"
                t=np.linspace(0,2*np.pi,65);poly=c+np.c_[np.cos(t),np.sin(t)]@np.linalg.cholesky(S).T
                q=(apply(np.linalg.inv(view["T"]),poly)+.5)*1.5-.5
                fd.line([tuple(v) for v in q],fill=color,width=2)
                fd.text((5,5 if feature=="heel" else 25),feature+":"+status,font=font,fill=color)
            x=430*j
            draw.text((x+7,32),f"{pt/12800:.2f}s PTS{pt}",font=font,fill="white")
            card.paste(Image.fromarray(cache["body_rgb"][index]),(x+7,61));card.paste(foot,(x+177,61))
            cells.append({"pts":int(pt),"source_seconds":pt/12800,"cache_index":index,"cache_view_proposal":int(view["view"])})
        draw.text((8,322),"No steel endpoints/contact/foot anatomy inferred; flow texture is not fixed template material.",font=font,fill=(255,210,140))
        path=output/f"track_card_{anchor['pts']}_{anchor['role']}.png";card.save(path)
        outputs.append({"path":str(path),"sha256":digest(path),"frames":cells})
    return outputs


def prepare(args):
    project=Path(args.project).resolve();sys.path[:0]=[str(project),str(project/"src")]
    from rinkspace.storage import AssetStore
    from calibration.program_tracking.body_tracking.source_track_evidence import propagated_region_shape
    from calibration.program_tracking.body_tracking.support_anchor_factors import NativeRegionObservation
    store=AssetStore(project,data_root=args.data_root)
    output=store.resolve(args.output)
    if not output.is_relative_to(store.data) or output.is_relative_to(project):raise ValueError("Products must stay in external Data")
    output.mkdir(parents=True,exist_ok=True)
    cache_path=output/"source_roi_cache.npz";source_manifest=output/"source_sequence.json"
    source_document=store.data/"runs/body_s01_20261029/blade_orientation_1105/source_boot_axis_document.json"
    doc=read_json(source_document);manifest=read_json(source_manifest)
    if digest(cache_path)!=manifest["cache"]["sha256"]:raise ValueError("ROI cache hash mismatch")
    with np.load(cache_path,allow_pickle=False) as z:
        cache={k:z[k].copy() for k in ("pts","feet_rgb","body_rgb","feet_bounds_native_xyxy","body_bounds_native_xyxy","feet_navigation_method")}
    pts=cache["pts"]
    if not np.array_equal(pts,np.arange(972800,1216001,256,dtype=np.int64)):raise ValueError("Complete native clock required")
    if doc["source_sha256"]!=manifest["source_video"]["sha256"]:raise ValueError("Source identities differ")
    anchors=[r for r in doc["boot_axis_observations"] if r.get("source_semantic")=="boot_heel_to_toe_contour_axis" and r.get("visibility")=="visible"]
    protected={int(r["pts"]) for r in anchors if r["role"]=="review"}
    for anchor in anchors:
        for feature in ("heel","toe"):
            q=np.asarray(anchor[feature+"_uv"],float)
            for view in (0,1):
                T=cache_to_native(cache["feet_bounds_native_xyxy"][int(np.searchsorted(pts,anchor["pts"])),view])
                assert np.allclose(apply(T,apply(np.linalg.inv(T),q[None])),q[None],atol=1e-10)
    cv2.setNumThreads(4);cv2.setRNGSeed(1105)
    branches=[];candidates=[]
    for anchor in anchors:
        for feature in ("heel","toe"):
            for direction in (-1,1):
                branch,rows=track_branch(cache,anchor,feature,direction,protected,propagated_region_shape)
                branches.append(branch);candidates.extend(rows)
        print(json.dumps({"anchor":anchor["id"],"role":anchor["role"],"inferred_candidates_so_far":len(candidates)}),flush=True)
    by_key=defaultdict(list)
    for r in candidates:by_key[(r["pts"],r["foot"],r["feature"],r["role"])].append(r)
    observed={(int(r["pts"]),r["foot"],feature,r["role"]):r for r in anchors for feature in ("heel","toe")}
    selected=[];conflicts=[]
    for key,rows in by_key.items():
        if key in observed:continue
        if key[3]=="fit" and key[0] in protected:raise ValueError("Review-derived information leaked into fit")
        consistent=True
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                distance=np.linalg.norm(a["centre"]-b["centre"])
                envelope=np.sqrt(np.linalg.eigvalsh(a["shape"])[-1])+np.sqrt(np.linalg.eigvalsh(b["shape"])[-1])
                if distance>envelope:consistent=False
        if not consistent:
            conflicts.append({"pts":key[0],"foot":key[1],"feature":key[2],"role":key[3],"reason":"anchor_propagations_disagree","branches":[r["branch_id"] for r in rows]});continue
        # One region per foot/feature/role/exposure prevents correlated duplicates.
        row=min(rows,key=lambda r:np.linalg.det(r["shape"]))
        row["consistent_branch_ids"]=[r["branch_id"] for r in rows]
        selected.append(row)
    periods={role:.02*sum(r["role"]==role for r in selected) for role in ("fit","review")}
    increments=[]
    for r in sorted(selected,key=lambda r:(r["pts"],r["foot"],r["feature"],r["role"])):
        row=dict(id=f"temporal_boot_{r['role']}_{r['pts']}_foot{r['foot']}_{r['feature']}",source_id=doc["source_id"],source_sha256=doc["source_sha256"],shot_id=doc["shot_id"],
            pts=int(r["pts"]),time_base=doc["time_base"],foot=int(r["foot"]),kind="boot_marker_region",feature=r["feature"],
            centre_uv=r["centre"].tolist(),region_shape_px2=r["shape"].tolist(),visibility="visible",contact_state="unknown",role=r["role"],
            reliability=.65,objective_weight=.02/periods[r["role"]],source_exposure_weight_seconds=.02,
            provenance="Adjacent cached source RGB bidirectional LK + robust similarity + independent backward fit + full endpoint-to-seed replay + anchor appearance/roundtrip checks; no model projection",
            source_semantic="boot_adapter_region_"+r["feature"],adapter_point_ids=[0,7] if r["feature"]=="heel" else [1,8],
            evidence_status="inferred",source_evidence_origin="temporal_source_region_inference",material_identity_confirmed=False,
            optimization_enabled=r["role"]=="fit",dependency_seed_pts=[r["anchor_pts"]],dependency_seed_id=r["anchor_id"],
            track_branch_ids=r["consistent_branch_ids"],source_uncertainty="Transported finite boot silhouette region with conservative affine/FB envelope; centre is not a measured foot point, steel endpoint or contact",
            anatomical_association_status="inherited conditional episode foot proposal; not independently established physical left/right",human_confirmed=False)
        NativeRegionObservation(**{k:row[k] for k in NativeRegionObservation.__dataclass_fields__})
        increments.append(row)
    inc_index={(r["pts"],r["foot"],r["feature"],r["role"]):r for r in increments}
    failure_by_key=defaultdict(list)
    for b in branches:
        if b.get("failure"):failure_by_key[(b["failure"]["pts"],b["foot"],b["feature"],b["role"])].append(b["failure"]["reason"])
    conflict_keys={(r["pts"],r["foot"],r["feature"],r["role"]) for r in conflicts}
    ledger=[]
    for index,pt in enumerate(pts):
        regions=[]
        for foot in (0,1):
            for feature in ("heel","toe"):
                for role in ("fit","review"):
                    key=(int(pt),foot,feature,role)
                    row=dict(foot=foot,feature=feature,role=role)
                    if key in observed:row.update(status="observed",anchor_id=observed[key]["id"],optimization_enabled=role=="fit")
                    elif key in inc_index:row.update(status="inferred",observation_id=inc_index[key]["id"],optimization_enabled=role=="fit")
                    elif key in conflict_keys:row.update(status="ambiguous",reason="anchor_propagations_disagree",optimization_enabled=False)
                    elif key in failure_by_key:row.update(status="ambiguous",reason=failure_by_key[key],optimization_enabled=False)
                    else:row.update(status="unknown",reason="no_confirmed_boot_contour_anchor_for_this_foot" if foot==0 else "not_reached_by_accepted_source_track",optimization_enabled=False)
                    regions.append(row)
        ledger.append(dict(sequence_index=index,pts=int(pt),source_seconds=int(pt)/12800,regions=regions))
    image_cards=cards(output,cache,anchors,branches)
    diagnostics=output/"region_track_diagnostics.json"
    save_json(diagnostics,dict(schema="rinkspace.source-region-track-diagnostics.v1",policy=POLICY,branches=branches,conflicts=conflicts,
        policy_scope="Frozen image tracking engineering gates, not precision certification; failures are retained, no coverage target or time cap",network=False,fit=False,model_projection=False))
    # Sparse numeric diagnostics share the complete native951 clock.
    numeric=output/"region_track_diagnostics.npz"
    centres=np.full((len(branches),951,2),np.nan);shapes=np.full((len(branches),951,2,2),np.nan);matrices=np.full((len(branches),951,3,3),np.nan)
    for bidx,b in enumerate(branches):
        for s in b["steps"]:
            if s["status"]=="inferred":
                i=int(np.searchsorted(pts,s["pts"]));centres[bidx,i]=s["centre_uv"];shapes[bidx,i]=s["region_shape_px2"];matrices[bidx,i]=s["native_affine"]
    np.savez_compressed(numeric,pts=pts,time_base=np.array([1,12800]),branch_ids=np.asarray([b["id"] for b in branches]),
                        region_centres_native_uv=centres,region_shapes_native_px2=shapes,anchor_to_current_native_affine=matrices)
    status_counts=Counter(r["status"] for f in ledger for r in f["regions"])
    result=dict(schema="rinkspace.source-support-regions.v1",source_id=doc["source_id"],source_sha256=doc["source_sha256"],shot_id=doc["shot_id"],time_base=doc["time_base"],pixel_domain="native Brown1920x1080",robust_delta=doc["robust_delta"],
        created_at=datetime.now(timezone.utc).isoformat(),scope="One complete951 exposure source ledger with multiple source-visibility track segments; no independent fitting windows",observations=increments,
        anchors_preserved=anchors,clock=ledger,status_counts=dict(status_counts),
        dependency=dict(source_sequence=str(source_manifest),source_sequence_sha256=digest(source_manifest),source_cache=str(cache_path),source_cache_sha256=manifest["cache"]["sha256"],source_document=str(source_document),source_document_sha256=digest(source_document)),
        source_exposure_quadrature=dict(native_period_seconds=.02,normalization="Each row owns its native0.02s cell; divide all fit rows by total fit region-seconds and all review rows separately, so each role sums to1 without multiplying weight by50Hz",role_region_seconds=periods,
            role_total_objective_weight={role:sum(r["objective_weight"] for r in increments if r["role"]==role) for role in ("fit","review")},independent_observation_count=False,unknown_cells_not_filled=True),
        review_isolation=dict(review_anchor_pts=sorted(protected),fit_branches_stop_before_review_pixels=True,review_seed_tracks_remain_review=True,review_measurements_used_to_correct_fit=False),
        diagnostics=dict(path=str(diagnostics),sha256=digest(diagnostics),numeric_path=str(numeric),numeric_sha256=digest(numeric)),cards=image_cards,
        uncertainty_semantics="region_shape_px2 is a conservative finite set, not Gaussian covariance or accuracy claim. Texture flow follows local appearance, not fixed template material. Heel/toe denote source boot contour regions; no steel/contact inference.",
        code=dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__).resolve())),execution=dict(fit=False,network=False,model_projection_read=False,video_generated=False,original_source_uv_modified=False))
    target=output/"temporal_regions.json";save_json(target,result)
    assert len(ledger)==951 and not any(r["role"]=="fit" and r["dependency_seed_pts"][0] in protected for r in increments)
    assert all(r["kind"]=="boot_marker_region" and r["evidence_status"]=="inferred" and r["contact_state"]=="unknown" for r in increments)
    for card in image_cards:
        with Image.open(card["path"]) as image:image.verify()
    print(json.dumps(dict(output=str(target),increment_rows=len(increments),fit_rows=sum(r["role"]=="fit" for r in increments),review_rows=sum(r["role"]=="review" for r in increments),status_counts=dict(status_counts),failures=dict(Counter(b.get("failure",{}).get("reason","sequence_boundary") for b in branches))),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",required=True);p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    prepare(p.parse_args())


if __name__=="__main__":main()
