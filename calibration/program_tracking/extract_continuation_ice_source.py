"""Read authenticated current-source ice candidates for programme frames 101–199.

The frozen 8a28 scene/cameras navigate each independent native RGB exposure.
Circle coordinates are signed material readouts. Long-line representatives
share an explicitly recorded estimate from the same frame's actual readouts.
No previous-frame point coordinates or projected UVs become observations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.optimize import least_squares
from calibration.program_tracking.projection import project_fast
from calibration.program_tracking.extract_single_frame_ice_source import (
    candidate_support,source_ellipse_guide,reread_source_ring,simplify,ellipse_distance,sha,read_json,
)
from calibration.program_tracking.apply_ice_source_review_amendment import NativeReadouts,fit_source_line,runs
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader,_batched_actual_profiles
from rinkspace.vision_programme_ice_review import _source_queries
from rinkspace.vision_source_shapes import _MaterialProfileFrame

DEFAULT_BATCH=Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/"
    "frame_by_frame_review_20261013/continuation_101_200_50fps")
DATASET=ROOT/"events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013"
MODEL=ROOT/"events/ankara_2026/models/framewise_8a28_20261013"
TEMPLATE=DATASET/"frame_000200_50fps/ice_features.json"
LINE_TARGETS={"ice.lane3:edge2","ice.lane2:edge0"}


def broadcast_exclusion(bgr):
    """Read this exposure's actual timer rectangle; font holes are excluded."""
    out=np.zeros(bgr.shape[:2],bool)
    out[25:105,25:557]=True
    white=(bgr[920:1020,1650:1900].min(2)>216).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(white,8)
    valid=[]
    for st in stats[1:]:
        x,y,w,h,area=map(int,st)
        if area<600 or not 55<=w<=230 or not 10<=h<=70:
            continue
        rect=bgr[920+y:920+y+h,1650+x:1650+x+w]
        if int((rect.max(2)<110).sum())<40:
            continue
        valid.append(st)
    rectangle=None
    if valid:
        st=max(valid,key=lambda s:int(s[4]))
        x,y,w,h=map(int,st[:4]);x+=1650;y+=920
        rectangle=[x,y,x+w-1,y+h-1]
        timer=np.zeros(out.shape,np.uint8)
        timer[y:y+h,x:x+w]=1
        out|=cv2.dilate(timer,np.ones((9,9),np.uint8))>0
    return out,dict(actual_native_rectangle_xyxy_inclusive=rectangle,edge_halo_px=4,
        method="current-native white timer component rectangle with dark text, no historical timer mask",
        current_source_timer_present=rectangle is not None,
        applied_to_source_readouts_and_line_visibility=True)


def blank_feature(path):
    alias=path["feature_id"]
    role={"outer":"外层材料界面","middle":"中层材料界面","inner":"内层材料界面",
        "center":"最内空心小圈"}.get(path["world_curve_role"],path["world_curve_role"])
    return dict(id="ice."+path["curve_id"],target_id="ice."+path["curve_id"],
        label_zh=alias+" "+role,world_curve=path["curve_id"],measurement_operator=path["operator"],
        material_signal=path["material_signal"],material_transition=path["material_transition"],
        segments=[],point_count=0,source_evidence_measured_in_this_frame=True,
        projected_uv_used_as_observations=False,user_confirmed=False,
        physical_identity_evidence="same-shot physical template and each current native material readout",
        status="current_source_candidates_waiting_visual_and_user_review")


def transformed_pattern_ellipse(path,cam,cam200,template):
    """Warp the reviewed complete-circle hypothesis, never its observations."""
    old=next((f for f in template["features"] if f["world_curve"]==path["curve_id"]),None)
    geometry=old.get("geometric_interpretation") if old else None
    if not geometry or "axes_px" not in geometry:return None,None
    ground=np.array([[x,y,0.] for x in (-20.,0.,20.) for y in (-10.,0.,10.)])
    oldnav,olddepth=project_fast(cam200,ground);newnav,newdepth=project_fast(cam,ground)
    valid=np.isfinite(oldnav).all(1)&np.isfinite(newnav).all(1)&(olddepth>.05)&(newdepth>.05)
    h,_=cv2.findHomography(oldnav[valid],newnav[valid],0)
    if h is None:return None,None
    theta=np.linspace(0.,2*np.pi,513)
    angle=np.deg2rad(geometry["angle_degrees"])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    points=(np.c_[np.cos(theta),np.sin(theta)]*(np.asarray(geometry["axes_px"])/2.))@rotation.T+geometry["center_uv"]
    current=cv2.perspectiveTransform(points.astype(np.float64)[None],h)[0]
    ellipse=cv2.fitEllipse(current.astype(np.float32))
    residual=np.linalg.norm(cv2.perspectiveTransform(oldnav[valid][None],h)[0]-newnav[valid],axis=1)
    center,axes,angle=ellipse
    t=np.linspace(0.,2*np.pi,513);a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    perimeter=(np.c_[np.cos(t),np.sin(t)]*(np.asarray(axes)/2.))@rotation.T+center
    # An entirely out-of-frame physical circle provides no native profile.
    if not ((perimeter>=[4,4]).all(1)&(perimeter<=[1915,1075]).all(1)).any():return None,None
    return ellipse,dict(kind="reviewed_complete_circle_pattern_warped_by_ice_plane_navigation",
        frame200_template_path=str(TEMPLATE),frame200_template_sha256=sha(TEMPLATE),
        homography_from_frame200_ice_navigation=h.tolist(),maximum_navigation_homography_residual_px=float(residual.max()),
        source_readouts_copied=False,transformed_uv_used_as_observations=False,
        use="whole physical-circle topology and source branch search only; every output UV reread in current native RGB")


def safe_reread_source_ring(frame,path,ellipse,exclusion):
    center,axes,angle=ellipse;t=np.linspace(0.,2*np.pi,513);a=np.deg2rad(angle)
    rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    projected=(np.c_[np.cos(t),np.sin(t)]*(np.asarray(axes)/2.))@rotation.T+center
    queries,normals,intervals=_source_queries(dict(path,_projected_center=np.asarray(center)),projected,
        np.ones(len(projected)),frame.rgb.shape[:2],24.,20.)
    if not len(queries):
        return ([],[dict(reason="complete_circle_search_has_no_in_frame_native_profile_support")],np.empty(0,int),projected,
            dict(query_uv=queries,query_normal=normals,query_interval=intervals))
    return reread_source_ring(frame,path,ellipse,exclusion)


def source_ring_feature(path,cam,raw,profiles,query_ids,candidate_by_query,frame,exclusion,archive,pattern=None,pattern_origin=None):
    feature=blank_feature(path)
    is_ring="center_xyz" in path
    readings=[];rejections=[]
    for qi in query_ids:
        eligible=[]
        for ci in candidate_by_query[int(qi)]:
            if raw["candidate_flags"][ci]&3:
                continue
            support,why=candidate_support(path,int(qi),ci,raw,profiles)
            if support is None:
                continue
            shift=abs(support["native_profile_shift_px"])
            if shift<=(48. if is_ring else 12.):
                eligible.append((shift,ci,support))
        eligible.sort(key=lambda a:a[0])
        if not eligible or (len(eligible)>1 and eligible[1][0]-eligible[0][0]<3.):
            readings.append(None)
            rejections.append(dict(query_index=int(qi),reason="no_unique_supported_current_native_material_branch"))
        else:
            _,ci,support=eligible[0]
            support["measurement_origin"]="current_native_signed_RGB_material_profiles"
            readings.append((raw["candidate_uv"][ci].astype(float),support))
    projected,depth=project_fast(cam,path["xyz"])
    navigation_projected=projected.copy()
    nav_ids=np.array([int(np.argmin(np.linalg.norm(projected-raw["query_uv"][qi],axis=1))) for qi in query_ids])
    ellipse=None
    if is_ring:
        if pattern is not None:
            ellipse=pattern
            seed=[r for r in readings if r is not None]
            if len(seed)>=4:
                center,axes,angle=ellipse;t=np.linspace(0.,2*np.pi,1025)
                a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                local=np.c_[np.cos(t),np.sin(t)]*(np.asarray(axes)/2.)
                perimeter=local@rotation.T+center
                normal=(local/(np.asarray(axes)/2.)**2)@rotation.T
                normal/=np.linalg.norm(normal,axis=1)[:,None]
                points=np.asarray([r[0] for r in seed]);closest=np.argmin(np.linalg.norm(points[:,None]-perimeter[None],axis=2),axis=1)
                near=perimeter[closest];normals=normal[closest]
                offsets=np.einsum("ij,ij->i",normals,points-near)
                design=np.c_[normals,np.einsum("ij,ij->i",normals,near-center)]
                solution=least_squares(lambda v:np.r_[design@v-offsets,v[0]/20.,v[2]/.02],np.zeros(3),
                    bounds=([-50.,-40.,-.06],[50.,40.,.06]),loss="soft_l1",f_scale=2.)
                correction=solution.x
                ellipse=(tuple(np.asarray(center)+correction[:2]),tuple(np.asarray(axes)*(1.+correction[2])),angle)
                feature["current_source_pattern_initial_alignment"]=dict(translation_uv=correction[:2].tolist(),isotropic_scale_change=float(correction[2]),
                    basis="current source broad native search readings align a complete-circle hypothesis; no alignment UV emitted",native_readout_count=len(seed))
            readings,extra,nav_ids,projected,detail=safe_reread_source_ring(frame,path,ellipse,exclusion)
            rejections.extend(extra)
            archive.update({path["curve_id"].replace(":","_")+"__"+k:v for k,v in detail.items()})
            measured=[r for r in readings if r is not None]
            if len(measured)>=4:
                center=np.asarray(ellipse[0]);normals=np.asarray([r[1]["profile_normal"] for r in measured])
                queries=np.asarray([r[1]["query_uv"] for r in measured]);offsets=np.asarray([r[1]["native_profile_shift_px"] for r in measured])
                design=np.c_[normals,np.einsum("ij,ij->i",normals,queries-center)]
                solution=least_squares(lambda v:design@v-offsets,np.zeros(3),bounds=([-20.,-20.,-.04],[20.,20.,.04]),loss="soft_l1",f_scale=2.)
                correction=solution.x
                if np.linalg.norm(correction[:2])>.4 or abs(correction[2])>.001:
                    ellipse=(tuple(center+correction[:2]),tuple(np.asarray(ellipse[1])*(1.+correction[2])),ellipse[2])
                    readings,extra,nav_ids,projected,detail=safe_reread_source_ring(frame,path,ellipse,exclusion)
                    rejections.extend(extra)
                    archive.update({path["curve_id"].replace(":","_")+"__corrected__"+k:v for k,v in detail.items()})
                feature["current_source_complete_circle_guide_correction"]=dict(translation_uv=correction[:2].tolist(),isotropic_scale_change=float(correction[2]),
                    basis="this frame's native signed-material offsets only; complete-circle topology retained",native_readout_count=len(measured),
                    emitted_as_observations=False)
            for i,reading in enumerate(readings):
                if reading is None:continue
                # The R4 upper-right false branches were explicitly rejected
                # in the full source-circle review. Broad blur uncertainty is
                # not permission to cross into a different material branch.
                maximum=3.5 if path["curve_id"]=="R4:outer" else 7.
                if abs(reading[1]["native_profile_shift_px"])>maximum:
                    rejections.append(dict(query_index=i,reason="current_material_readout_crosses_complete_circle_branch_window"))
                    readings[i]=None
                elif path["curve_id"]=="R4:outer":
                    # The same static blurred patch remains visibly mixed in
                    # current exposures. Carry only its identity as a search
                    # region; do not carry the old rejected coordinates as
                    # observations or replace them with circle-generated UVs.
                    correction=read_json(DATASET/"frame_000200_50fps/user_correction_01.json")
                    old=np.asarray(correction["remove_native_points"]["uv"],float)
                    h=np.asarray(pattern_origin["homography_from_frame200_ice_navigation"])
                    ambiguous=cv2.perspectiveTransform(old[None],h)[0]
                    scale=float(np.sqrt(abs(np.linalg.det(h[:2,:2]))))
                    if np.linalg.norm(ambiguous-reading[0],axis=1).min()<18.*scale:
                        rejections.append(dict(query_index=i,source_uv=reading[0].tolist(),
                            reason="current_visible_R4_upper_right_blurred_mixed_material_patch",
                            current_native_visual_review=True,static_patch_identity_anchor="frame200 user red-circle source review; current source localization reread"))
                        readings[i]=None
            if path["curve_id"]=="R3:outer":
                # Re-read the user-confirmed broad left arc with its original
                # blurred material width, while retaining native coordinates.
                center,axes,angle=ellipse;t=np.linspace(0.,2*np.pi,513)
                a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                guide=(np.c_[np.cos(t),np.sin(t)]*(np.asarray(axes)/2.))@rotation.T+center
                queries,normals,_=_source_queries(dict(path,_projected_center=np.asarray(center)),guide,
                    np.ones(len(guide)),frame.rgb.shape[:2],24.,32.)
                native=NativeReadouts(dict(source_video_sha256=frame.binding["source_sha256"],
                    native_pts=frame.binding["source_pts"],time_base=frame.binding["time_base"]),frame.rgb,exclusion)
                rows=native.read("R3_outer_complete_current_arc",queries,normals,"material_interface",path["material_transition"],radius=32.)
                for row in rows:
                    if not len(detail["query_uv"]):continue
                    q=np.asarray(row["source_evidence"]["query_uv"])
                    qi=int(np.argmin(np.linalg.norm(detail["query_uv"]-q,axis=1)))
                    if np.linalg.norm(detail["query_uv"][qi]-q)>1e-5:continue
                    if abs(row["source_evidence"]["native_profile_shift_px"])<=7.:
                        row["source_evidence"]["measurement_origin"]="current_native_signed_RGB_material_profiles"
                        readings[qi]=(np.asarray(row["uv"]),row["source_evidence"])
                archive.update({"R3_outer_blurred_arc__"+k:v for k,v in native.archive.items()})
            feature["complete_circle_search_pattern"]=pattern_origin
        else:
            ellipse=None
        seed=[r[0] for r in readings if r is not None]
        proposed=source_ellipse_guide(seed) if pattern is None else None
        # A narrow isolated arc cannot independently determine a new ellipse.
        # Check the current-source guide against the coarse circle family;
        # navigation is a branch prior, never a UV measurement or exact truth.
        nav_good=np.isfinite(navigation_projected).all(1)&(depth>.05)
        nav_ellipse=cv2.fitEllipse(navigation_projected[nav_good].astype(np.float32)) if nav_good.sum()>=8 else None
        use_refinement=False
        if proposed is not None and nav_ellipse is not None:
            axis_ratio=np.asarray(proposed[1])/np.asarray(nav_ellipse[1])
            centre_shift=np.linalg.norm(np.asarray(proposed[0])-nav_ellipse[0])
            angle_shift=abs((float(proposed[2])-float(nav_ellipse[2])+90)%180-90)
            use_refinement=bool((axis_ratio>.55).all() and (axis_ratio<1.7).all() and centre_shift<180. and angle_shift<5.)
        if use_refinement:
            ellipse=proposed
            readings,extra,nav_ids,projected_refined,detail=safe_reread_source_ring(frame,path,ellipse,exclusion)
            projected=projected_refined
            rejections.extend(extra)
            archive.update({path["curve_id"].replace(":","_")+"__"+k:v for k,v in detail.items()})
        elif proposed is not None:
            feature["current_source_ellipse_refinement_withheld_reason"]="insufficient full-circle branch agreement for an independently determined ellipse"
            ellipse=None
    groups=[];current=[];previous=None
    for ordinal,reading in enumerate(readings):
        if reading is None:
            if current:groups.append(current);current=[]
            previous=None
            continue
        gap=(previous is not None and abs(int(nav_ids[ordinal])-int(nav_ids[previous]))>(48 if ellipse is not None else 24))
        if current and (np.linalg.norm(reading[0]-current[-1][0])>72. or gap):
            groups.append(current);current=[]
        current.append(reading);previous=ordinal
    if current:groups.append(current)
    full_in_frame=bool(len(projected) and (depth>.05).all() and np.isfinite(projected).all()
        and (projected>=[8,8]).all() and (projected<=[1911,1071]).all())
    entirely_read=bool(readings) and all(r is not None for r in readings)
    if is_ring and full_in_frame and len(groups)>1 and readings[0] is not None and readings[-1] is not None:
        if np.linalg.norm(groups[-1][-1][0]-groups[0][0][0])<48.:
            groups=[groups[-1]+groups[0]]+groups[1:-1]
    for group in groups:
        if len(group)<2:continue
        points=np.asarray([r[0] for r in group])
        closed=bool(is_ring and full_in_frame and entirely_read and len(groups)==1)
        keep=simplify(points,tolerance=2.5,maximum_gap=260.,closed=closed)
        evidence=[group[k][1] for k in keep]
        feature["segments"].append(dict(uv=np.round(points[keep],4).tolist(),closed=closed,vertex_count=len(keep),
            sigma_px=[round(e["transition_sigma_px"],4) for e in evidence],source_evidence=evidence,
            measurement_origin="current_native_signed_RGB_material_profiles",pixels_measured_in_this_frame=True,
            source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=True,
            hidden_span_filled=False,user_confirmed=False))
    feature["point_count"]=sum(s["vertex_count"] for s in feature["segments"])
    observed=np.asarray([uv for s in feature["segments"] for uv in s["uv"]],np.float32).reshape(-1,2)
    if is_ring and len(observed)>=6:
        physical=source_ellipse_guide(observed)
        if physical is not None:
            distances=ellipse_distance(observed,physical)
            sigma=np.asarray([v for s in feature["segments"] for v in s["sigma_px"]])
            feature["geometric_interpretation"]=dict(kind="projected_physical_circle_ellipse",inference_only=True,
                basis="one current physical circle; all current native observed segments share one ellipse",
                center_uv=list(physical[0]),axes_px=list(physical[1]),angle_degrees=physical[2],
                maximum_source_readout_to_ellipse_distance_px=float(distances.max()),
                all_readouts_within_maximum_of_2point5_px_or_2point5_sigma=bool((distances<=np.maximum(2.5,2.5*sigma)).all()),
                inferred_uv_not_emitted_as_source_points=True,hidden_spans_are_inference_not_source_observations=True)
    if ellipse is not None:
        feature["current_source_curve_branch_guide"]=dict(center_uv=list(ellipse[0]),axes_px=list(ellipse[1]),
            angle_degrees=ellipse[2],emitted_as_observations=False,
            camera_projection_coordinates_emitted=False,
            source=("reviewed_complete_circle_navigation_pattern; current native RGB reread" if pattern is not None else "current_native_material_readouts_only"))
    feature["note_zh"]=("同一实体材料层按完整物理圆的投影椭圆共同识别；坐标来自本帧原生剖面，出画、人物和字幕遮挡不补源点。"
        if feature["point_count"] else "该层在本帧未得到唯一清晰原图材料段，或尚未入画；未复制模型投影、邻帧点或遮挡段。")
    if path["curve_id"]=="R3:outer":
        feature["physical_identity_confirmation"]=dict(path=str(DATASET/"frame_000200_50fps/user_correction_01.json"),
            sha256=sha(DATASET/"frame_000200_50fps/user_correction_01.json"),scope="same-shot R3 outer physical-circle identity; current UV independently reread")
        feature["note_zh"]+=" 用户已确认R3最外圈实体，左端与右段共同保持同一圆形，不再作倒影归属重复待核。"
    return feature,dict(target_id=feature["target_id"],query_count=len(readings),
        actual_source_supported_queries=sum(r is not None for r in readings),final_points=feature["point_count"],
        note_zh=feature["note_zh"],rejection_count=len(rejections),rejections=rejections)


def current_source_long_line(path,cam,reader,person,broadcast,pts,cam200,template):
    feature=blank_feature(path)
    projected,depth=project_fast(cam,path["xyz"])
    good=np.isfinite(projected).all(1)&(depth>.05)
    if good.sum()<4 or np.ptp(projected[good,0])<100:
        feature["note_zh"]="该长标线未进入可读取的原图范围，未取点。"
        return feature,[],None
    navigation=np.polyfit(projected[good,0],projected[good,1],1)
    old=next((f for f in template["features"] if f["world_curve"]==path["curve_id"]),None)
    if old and old.get("source_line_fit"):
        ground=np.array([[x,y,0.] for x in (-20.,0.,20.) for y in (-10.,0.,10.)])
        oldnav,_=project_fast(cam200,ground);newnav,_=project_fast(cam,ground)
        h,_=cv2.findHomography(oldnav,newnav,0)
        coefficient200=np.asarray(old["source_line_fit"]["coefficient_y_equals_ax_plus_b"])
        oldpoints=np.c_[[0.,1919.],np.polyval(coefficient200,[0.,1919.])]
        guide=cv2.perspectiveTransform(oldpoints[None],h)[0]
        navigation=np.polyfit(guide[:,0],guide[:,1],1)
        feature["complete_line_search_pattern"]=dict(frame200_template_path=str(TEMPLATE),frame200_template_sha256=sha(TEMPLATE),
            homography_from_frame200_ice_navigation=h.tolist(),source_points_copied=False,transformed_uv_emitted_as_observations=False,
            use="same-shot full physical line search only; current source RGB reread determines every line estimate")
    xs=np.arange(15.,1910.,64.)
    queries=np.c_[xs,np.polyval(navigation,xs)]
    queries=queries[(queries[:,1]>4)&(queries[:,1]<1076)]
    normal=np.tile(np.array([-navigation[0],1.])/np.sqrt(1+navigation[0]**2),(len(queries),1))
    rows=reader.read(path["curve_id"].replace(":","_")+"_full_current_line",queries,normal,"paint_center",radius=48.)
    rows=[r for r in rows if abs(r["uv"][1]-np.polyval(navigation,r["uv"][0]))<24.]
    if len(rows)<3 or np.ptp(np.asarray([r["uv"] for r in rows])[:,0])<150:
        feature["note_zh"]="长标线尚未入画或本帧原图缺少跨区域的清晰双边读数，未用导航直线坐标补点。"
        feature["native_candidate_readouts"]=rows
        return feature,[],None
    coefficient,covariance,active,residual=fit_source_line(rows)
    if active.sum()<3 or np.ptp(np.asarray([r["uv"] for r,ok in zip(rows,active) if ok])[:,0])<150. or abs(coefficient[0]-navigation[0])>.015:
        feature["note_zh"]="当前原图双边读数未形成与该实体方向一致的长直线，未取点。"
        feature["native_candidate_readouts"]=rows
        return feature,[],None
    group_id=f"official50_{pts}_"+feature["id"].replace(".","_").replace(":","_")+"_source_line"
    feature["source_evidence_group_id"]=group_id
    feature["source_line_fit"]=dict(coefficient_y_equals_ax_plus_b=coefficient.tolist(),covariance_ab=covariance.tolist(),
        native_source_pts=int(pts),current_frame_only=True,old_model_projection_used_as_output_points=False,
        direct_readouts=[dict(r,used_in_line_fit=bool(ok),source_fit_residual_px=float(res)) for r,ok,res in zip(rows,active,residual)],
        actual_native_readout_count=int(active.sum()))
    xs=np.arange(1920);ys=np.polyval(coefficient,xs)
    inside=(ys>=0)&(ys<=1079);iy=np.clip(np.rint(ys).astype(int),0,1079)
    hidden=person[iy,xs]|broadcast[iy,xs]
    for first,last in runs(np.flatnonzero(inside&~hidden)):
        positions=np.linspace(first,last,max(1,int(np.ceil((last-first)/220.)))+1)
        uv=np.c_[positions,np.polyval(coefficient,positions)]
        sigmas=[max(1.,float(np.sqrt(np.array([x,1.])@covariance@np.array([x,1.])))) for x in positions]
        feature["segments"].append(dict(uv=np.round(uv,4).tolist(),closed=False,vertex_count=len(uv),sigma_px=sigmas,
            measurement_origin="same_frame_joint_straight_edge_fit",representative_points_share_line_estimate=True,
            source_evidence_group_id=group_id,pixels_measured_in_this_frame=False,source_evidence_measured_in_this_frame=True,
            individual_native_pixel_readouts=False,inferred_from_current_source_readouts=True,hidden_span_filled=False,
            user_confirmed=False,source_evidence=[dict(kind="shared_current_source_line_representative",
                source_evidence_group_id=group_id,individual_native_pixel_readout=False,
                coefficient_y_equals_ax_plus_b=coefficient.tolist()) for _ in uv]))
    feature["point_count"]=sum(s["vertex_count"] for s in feature["segments"])
    inferred=[]
    for first,last in runs(np.flatnonzero(inside&hidden)):
        reason="人物遮挡的同一物理直线延续，仅几何推断，不计原图取点。"
        if broadcast[iy[first:last+1],xs[first:last+1]].any():reason="当前计时字幕框及4像素边缘安全区遮挡，仅直线延续推断，不计取点。"
        inferred.append(dict(target_id=feature["target_id"],uv=[[float(first),float(np.polyval(coefficient,first))],
            [float(last),float(np.polyval(coefficient,last))]],inference_only=True,reason_zh=reason,
            source_evidence_group_id=group_id))
    feature["note_zh"]="本帧原图材料读数共同估计同一长直线，按约220像素覆盖完整可见范围；人物/字幕段只存推断。代表点共享同一估计，不冒充独立像素测量。"
    feature["geometric_interpretation"]=dict(kind="straight_ice_paint_centerline",inference_only=True,
        current_source_line_y_equals_ax_plus_b=coefficient.tolist(),current_source_line_covariance_ab=covariance.tolist(),
        full_visible_native_frame_extent=True,source_line_representatives_are_not_independent_pixel_readouts=True)
    group=dict(group_id=group_id,target_id=feature["target_id"],native_source_pts=int(pts),
        correlation_reason="same current-source straight-line estimate",fit_parameter_order=["slope_a","intercept_b"],
        covariance_ab=covariance.tolist(),representative_point_count=feature["point_count"],native_individual_readout_count=int(active.sum()))
    return feature,inferred,group


def verify_current_source_minor_line(feature,path,cam):
    """Admit newly clear modeled lines from this frame's own source evidence.

    Navigation and continuity nominate identity; independently read current
    material sides must form that physical straight edge across visible areas.
    An old review's omission does not veto a newly visible source feature.
    """
    rows=[(np.asarray(uv,float),sigma,evidence) for segment in feature["segments"]
        for uv,sigma,evidence in zip(segment["uv"],segment["sigma_px"],segment["source_evidence"])]
    strong=[r for r in rows if min(r[2]["signed_interface_iqr_contrast"])>=6.
        and r[1]<=10.
        and np.linalg.norm(np.diff(r[2]["actual_source_sides_uv"],axis=0))<=14.]
    projected,depth=project_fast(cam,path["xyz"])
    good=np.isfinite(projected).all(1)&(depth>.05)
    reason=None
    if len(strong)<4 or np.ptp(np.asarray([r[0][0] for r in strong]))<300.:
        reason="本帧该细弱标线没有形成跨可见区域、双边均可分辨的原图材料读数；不将模糊纹理强认成标线。"
    elif good.sum()<4:
        reason="本帧细弱材料的模型实体范围无法可靠导航，保留原生候选但不强认实体。"
    else:
        navigation=np.polyfit(projected[good,0],projected[good,1],1)
        native=np.asarray([r[0] for r in strong])
        coefficient=np.polyfit(native[:,0],native[:,1],1)
        residual=np.abs(native[:,1]-np.polyval(coefficient,native[:,0]))
        active=residual<=np.maximum(2.5,2.*np.asarray([r[1] for r in strong]))
        if active.sum()<4 or np.ptp(native[active,0])<300. or abs(coefficient[0]-navigation[0])>.015:
            reason="本帧原生材料分支未组成该静止实体应有的共同直线；局部反光、邻圈响应不进入取点。"
        else:
            kept=[r for r,ok in zip(strong,active) if ok]
            keep_uv={tuple(r[0]) for r in kept}
            retained=[]
            for old in feature["segments"]:
                group=[]
                for uv,sigma,evidence in zip(old["uv"],old["sigma_px"],old["source_evidence"]):
                    if tuple(uv) in keep_uv:
                        group.append((np.asarray(uv),sigma,evidence))
                    elif group:
                        retained.append(group);group=[]
                if group:retained.append(group)
            feature["segments"]=[]
            for kept_segment in retained:
                if len(kept_segment)<2:continue
                keep=simplify(np.asarray([r[0] for r in kept_segment]),tolerance=2.5,maximum_gap=220.)
                selected=[kept_segment[k] for k in keep]
                feature["segments"].append(dict(uv=[np.round(r[0],4).tolist() for r in selected],closed=False,
                    vertex_count=len(selected),sigma_px=[r[1] for r in selected],source_evidence=[r[2] for r in selected],
                    measurement_origin="current_native_signed_RGB_material_profiles",pixels_measured_in_this_frame=True,
                    source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=True,
                    hidden_span_filled=False,user_confirmed=False))
            feature["point_count"]=sum(s["vertex_count"] for s in feature["segments"])
            feature["note_zh"]="本帧新入画静止细标线按当前原生双边材料读数及同一实体的共同直线形态识别；输出只保留实际读得的稀疏坐标。"
            feature["geometric_interpretation"]=dict(kind="current_source_shared_straight_edge_identity",inference_only=True,
                coefficient_y_equals_ax_plus_b=coefficient.tolist(),used_as_output_coordinates=False,
                current_native_supported_count=len(kept),static_identity_confirmed_by_current_source_material=True)
    if reason:
        feature["withheld_candidate_point_count"]=feature["point_count"]
        feature["segments"]=[];feature["point_count"]=0;feature["note_zh"]=reason
    return feature


def draw_overlay(bgr,packet):
    output=bgr.copy()
    for feature in packet["features"]:
        color=(222,85,230) if "R4:" in feature["id"] else (25,210,250) if "R3:" in feature["id"] else (70,235,70)
        for segment in feature["segments"]:
            uv=np.rint(np.asarray(segment["uv"])*256).astype(np.int32)
            cv2.polylines(output,[uv],segment["closed"],color,1,cv2.LINE_AA,shift=8)
            for p in uv:
                cv2.circle(output,tuple(map(int,p)),3*256,(15,15,15),-1,cv2.LINE_AA,shift=8)
                cv2.circle(output,tuple(map(int,p)),2*256,color,-1,cv2.LINE_AA,shift=8)
    for segment in packet["inferred_segments"]:
        uv=np.asarray(segment["uv"],float)
        for a,b in zip(uv[:-1],uv[1:]):
            vector=b-a;length=np.linalg.norm(vector)
            if not length:continue
            for distance in np.arange(0,length,14.):
                p=np.rint(a+vector*(distance/length)).astype(int)
                q=np.rint(a+vector*(min(distance+6,length)/length)).astype(int)
                cv2.line(output,tuple(p),tuple(q),(140,140,140),1,cv2.LINE_AA)
    return output


def extract(index,batch=DEFAULT_BATCH):
    if not 101<=index<=199:raise ValueError("Only newly authorized 101–199 source extraction")
    batch=Path(batch)
    manifest=read_json(batch/"frame_manifest.json")
    row=next(r for r in manifest["frames"] if r["index"]==index)
    source=deepcopy(row["source"])
    image=Path(row["image_path"]);mask_path=Path(row["mask_path"])
    if sha(image)!=source["png_sha256"]:raise ValueError("Current native source PNG changed")
    bgr=cv2.imread(str(image));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=source["decoded_RGB_sha256"]:raise ValueError("Current native source RGB changed")
    mask_bytes=mask_path.read_bytes();mask_digest=hashlib.sha256(mask_bytes).hexdigest()
    if source.get("mask_sha256") and mask_digest!=source["mask_sha256"]:
        raise ValueError("Current native person mask changed after source-manifest binding")
    person=cv2.imdecode(np.frombuffer(mask_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)>0
    if person.shape!=(1080,1920):raise ValueError("Current authenticated native person mask required")
    broadcast,broadcast_receipt=broadcast_exclusion(bgr)
    exclusion=person|broadcast
    archive=np.load(batch/"camera_seeds.npz")
    ci=int(np.flatnonzero(archive["programme_indices"]==index)[0]);cam=archive["camera25"][ci]
    if int(archive["pts"][ci])!=source["native_pts"]:raise ValueError("Camera navigation PTS mismatch")
    scene=read_json(MODEL/"scene.json");template=read_json(TEMPLATE)
    identities={f["world_curve"] for f in template["features"]}
    reader=IceProgrammeSourceReader(scene,source_sha256=source["source_video_sha256"],time_base=source["time_base"],
        sample_spacing_px=24.,search_radius_px=64.)
    circle_centers={f["id"]:f["center_xyz"] for f in scene["ice_features"] if f["kind"]=="circle"}
    for path in reader.paths:
        if path["feature_id"] in circle_centers and "center_xyz" not in path:
            path["center_xyz"]=np.asarray(circle_centers[path["feature_id"]],float)
    # The current source may expose a new modeled entity. The frame-200
    # feature list is a pattern anchor, never an authorization restriction.
    visible_ids=set()
    for path in reader.paths:
        uv,depth=project_fast(cam,path["xyz"])
        visible=np.isfinite(uv).all(1)&(depth>.05)&(uv>=[-64,-64]).all(1)&(uv<=[1983,1143]).all(1)
        if visible.any():visible_ids.add(path["curve_id"])
    identities|=visible_ids
    identities.discard("curling_centerline_row3:segment0")
    reader.paths=[p for p in reader.paths if p["curve_id"] in identities]
    active_reader_paths=[p for p in reader.paths if p["world_curve_role"]!="center"
        and "ice."+p["curve_id"] not in LINE_TARGETS]
    inventory=reader.paths;reader.paths=active_reader_paths
    raw=reader.read(rgb,source["native_pts"],cam,exclusion,shot_id=1)
    frame=_MaterialProfileFrame(rgb,raw["source_binding"])
    profiles=_batched_actual_profiles(frame,raw["query_uv"],raw["query_normal"],raw["query_interval"],
        np.asarray([reader.paths[i]["material_signal"] for i in raw["query_curve"]]))
    by_query=defaultdict(list)
    for k,qi in enumerate(raw["candidate_query"]):by_query[int(qi)].append(k)
    evidence={k:v for k,v in raw.items() if isinstance(v,np.ndarray)}
    evidence.update(profile_offsets=profiles[0],profile_values=profiles[1],profile_smoothed=profiles[2],profile_valid=profiles[3])
    features=[];reports=[];inferred=[];groups=[]
    native=NativeReadouts(source,rgb,exclusion)
    for path in inventory:
        target="ice."+path["curve_id"]
        if path["world_curve_role"]=="center":
            feature=blank_feature(path)
            feature["note_zh"]="用户允许最内空心圈暂不硬取；本帧不以同心、半径比或旧点制造小圈观测。"
            report=dict(target_id=target,query_count=0,actual_source_supported_queries=0,final_points=0,note_zh=feature["note_zh"])
        elif target in LINE_TARGETS:
            feature,hidden,group=current_source_long_line(path,cam,native,person,broadcast,source["native_pts"],archive["camera25"][99],template)
            inferred.extend(hidden)
            if group:groups.append(group)
            report=dict(target_id=target,query_count=len(feature.get("source_line_fit",{}).get("direct_readouts",[])),
                actual_source_supported_queries=feature.get("source_line_fit",{}).get("actual_native_readout_count",0),
                final_points=feature["point_count"],note_zh=feature["note_zh"])
        else:
            pi=next(i for i,p in enumerate(reader.paths) if p["curve_id"]==path["curve_id"])
            query_ids=np.flatnonzero(raw["query_curve"]==pi)
            pattern,origin=transformed_pattern_ellipse(path,cam,archive["camera25"][99],template) if "center_xyz" in path else (None,None)
            feature,report=source_ring_feature(path,cam,raw,profiles,query_ids,by_query,frame,exclusion,evidence,pattern,origin)
            if "center_xyz" not in path:
                feature=verify_current_source_minor_line(feature,path,cam)
                report.update(final_points=0,note_zh=feature["note_zh"])
                report["final_points"]=feature["point_count"]
            elif path["curve_id"]=="A2:outer":
                observed=np.asarray([uv for s in feature["segments"] for uv in s["uv"]]).reshape(-1,2)
                if len(observed)<6 or np.ptp(observed[:,0])<300. or not feature.get("geometric_interpretation"):
                    feature["withheld_candidate_point_count"]=feature["point_count"]
                    feature["segments"]=[];feature["point_count"]=0
                    feature["note_zh"]="本帧该区域只见少量微弱红色材料响应，尚未形成可辨认的同一圆弧；原生候选保留，不将局部色纹强认成A2。"
                    report.update(final_points=0,note_zh=feature["note_zh"])
        features.append(feature);reports.append(report)
    evidence.update({"shared_line__"+k:v for k,v in native.archive.items()})
    dest=batch/"ice_candidates";dest.mkdir(parents=True,exist_ok=True)
    archives=batch/"ice_native_evidence";archives.mkdir(parents=True,exist_ok=True)
    evidence_path=archives/f"{index:06d}.npz"
    np.savez_compressed(evidence_path,**evidence)
    source.update(mask_path=str(mask_path.resolve()),mask_sha256=mask_digest,mask_provisional=False,
        exclusion_halo_px=0,current_mask_already_conservatively_dilated_native_px=10,
        broadcast_timer_exclusion=broadcast_receipt)
    packet=dict(schema="rinkspace.single-frame-source-ice-features.v1",source=source,source_pts=source["native_pts"],
        programme_frame_index_1based=index,status="awaiting_user_review",user_accepted=False,frame_count=1,
        next_frame_processed=False,solver_inputs_modified=False,optimization_performed=False,
        navigation_camera_path=str((batch/"camera_seeds.npz").resolve()),navigation_camera_sha256=sha(batch/"camera_seeds.npz"),
        navigation_camera_row=ci,navigation_camera25_sha256=hashlib.sha256(cam.tobytes()).hexdigest(),
        model_scene_sha256=sha(MODEL/"scene.json"),physical_aliases={},
        user_naming_correction=dict(canonical_ice_circle_ids=["R3","R4"],
            old_B3_B4_were_oral_misnomers=True,target_or_coordinate_identity_changed=False,
            historical_aliases={"B3":"R3","B4":"R4"},
            note_zh="用户已更正名称：当前冰圈统一称模型R3/R4；此前B3/B4只保留为历史口头误称，对应实体和坐标未改变。"),
        duplicate_physical_material_suppressed={"ice.curling_centerline_row3:segment0":"ice.house_connection_F3_R3:segment0"},
        features=features,point_count=sum(f["point_count"] for f in features),
        unresolved_source_regions=[r for r in reports if r["final_points"]==0],curve_readout_summary=reports,
        inferred_segments=inferred,source_correlated_groups=groups,projected_uv_used_as_observations=False,
        source_evidence_archive=str(evidence_path.resolve()),source_evidence_archive_sha256=sha(evidence_path),
        frame200_pattern_reference=dict(path=str(TEMPLATE),sha256=sha(TEMPLATE),source_points_copied=False),
        source_sampling_policy=dict(straight_representative_maximum_spacing_px=220.,
            curved_source_simplification_maximum_deviation_px=2.5,curved_maximum_chord_px=260.,
            each_frame_native_material_readouts=True,prior_frame_points_copied=False,center_button_forced=False),
        visual_review=dict(status="awaiting_extraction_agent_native_overlay_review",original_and_overlay_inspected=False,
            camera_or_model_accuracy_accepted=False,human_acceptance=False),
        generator_script_sha256=sha(Path(__file__)))
    for f in features:
        for s in f["segments"]:
            if len(s["uv"])!=len(s["sigma_px"]) or len(s["uv"])!=len(s["source_evidence"]):raise ValueError("Source evidence length mismatch")
            for uv in s["uv"]:
                x,y=np.rint(uv).astype(int)
                if not (0<=x<1920 and 0<=y<1080) or exclusion[y,x]:raise ValueError("Current source point lies outside actual image or on person/broadcast")
    if packet["point_count"]>340:raise ValueError("Unexpected excessive source density; inspect extraction before board merge")
    (dest/f"{index:06d}.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    overlays=batch/"ice_overlays";overlays.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(overlays/f"{index:06d}.png"),draw_overlay(bgr,packet),[cv2.IMWRITE_PNG_COMPRESSION,1])
    summary=dict(index=index,source_pts=source["native_pts"],point_count=packet["point_count"],
        by_feature={f["id"]:f["point_count"] for f in features},missing=[f["id"] for f in features if not f["point_count"]],
        packet_path=str((dest/f"{index:06d}.json").resolve()),packet_sha256=sha(dest/f"{index:06d}.json"))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def contact_sheets(batch=DEFAULT_BATCH):
    batch=Path(batch);paths=sorted((batch/"ice_candidates").glob("*.json"))
    tiles=[]
    for path in paths:
        packet=read_json(path);index=packet["programme_frame_index_1based"]
        overlay=cv2.imread(str(batch/"ice_overlays"/f"{index:06d}.png"))
        tile=cv2.resize(overlay,(960,540),interpolation=cv2.INTER_AREA)
        band=np.zeros((40,960,3),np.uint8)
        cv2.putText(band,f"{index:03d}  {packet['source']['source_time']}  ice {packet['point_count']} points",(10,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1)
        tiles.append((index,np.vstack([band,tile])))
    dest=batch/"ice_contact_sheets";dest.mkdir(parents=True,exist_ok=True)
    out=[]
    for first in range(0,len(tiles),4):
        selection=tiles[first:first+4]
        canvas=np.zeros((1160,1920,3),np.uint8)
        for ordinal,(_,tile) in enumerate(selection):
            y,x=divmod(ordinal,2);canvas[y*580:(y+1)*580,x*960:(x+1)*960]=tile
        path=dest/f"{selection[0][0]:03d}-{selection[-1][0]:03d}.png"
        cv2.imwrite(str(path),canvas,[cv2.IMWRITE_PNG_COMPRESSION,1]);out.append(str(path))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,default=DEFAULT_BATCH)
    parser.add_argument("--indices",type=int,nargs="+")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--contacts-only",action="store_true")
    args=parser.parse_args()
    if args.contacts_only:
        print(json.dumps(dict(contact_sheets=contact_sheets(args.batch))));return
    selected=args.indices or list(range(101,200))
    if any(not 101<=i<=199 for i in selected):raise ValueError("Only 101–199 may be extracted by this module")
    started=time.monotonic()
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(extract,i,args.batch) for i in selected]
            summaries=[future.result() for future in as_completed(futures)]
    else:summaries=[extract(i,args.batch) for i in selected]
    summaries.sort(key=lambda r:r["index"])
    result=dict(schema="rinkspace.continuation-native-ice-extraction.v1",processed_indices=selected,
        source_pts_start=281856,source_pts_last=306944,native_step_pts=256,source_time_base="1/12800",
        frame_count=len(summaries),rows=summaries,elapsed_seconds=time.monotonic()-started,
        first100_modified=False,frame200_modified=False,processing201_performed=False,optimization_performed=False,
        contact_sheets=contact_sheets(args.batch))
    (args.batch/"ice_extraction_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(completed=len(summaries),points=[r["point_count"] for r in summaries],elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":main()
