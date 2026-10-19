"""Experimental ice-only photometric fit under an explicit rotating-camera prior.

The optical centre is assumed fixed, principal point centred and pixels square.
Initial focal length is a declared assumption, not calibration. Images, masks,
sampling and all trial footprints obey masked_photometric's isolation contract.
Four geometric degrees of freedom replace unconstrained affine shear. No
independent contours enter fitting. This is a falsifiable candidate model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import cv2
import numpy as np

from .masked_photometric import (_isolated, _bilinear, _project, _ranges,
                                _support, _loss, _correlation)
from .photometric_registration import PhotometricConfig, PhotometricRegistrar

ALGORITHM_VERSION = "ice-only-rotation-zoom-photometric-1"


def intrinsic(shape, focal):
    h, w = shape
    return np.array([[focal, 0., (w-1)/2], [0., focal, (h-1)/2], [0., 0., 1.]])


def camera_h(shape, reference_focal, current_focal, rotation):
    h = intrinsic(shape, reference_focal) @ rotation @ np.linalg.inv(intrinsic(shape, current_focal))
    return h/h[2, 2]


def estimate_rotation(reference, current, ref_mask, cur_mask, *, reference_focal,
                      initial_current_focal, initial_rotation):
    ref, rs = _isolated(reference, ref_mask)
    cur, cs = _isolated(current, cur_mask)
    if ref.shape != cur.shape:
        raise ValueError("camera images must have the same dimensions")
    height, width = cur.shape
    p = np.r_[cv2.Rodrigues(np.asarray(initial_rotation, float))[0].ravel(),
              math.log(initial_current_focal), 0., 0.]
    result = dict(schema="rinkspace.rotation-photometric.v1", status="unavailable",
        reason=None, H_reference_from_current=None, independent_validation=False,
        motion_model="rotation_zoom", model_assumption="fixed_optical_centre_centred_principal_point_square_pixels_unknown_initial_focal",
        reference_focal_px=float(reference_focal), sampling_pool_current_xy=[], iterations=[],
        fit_current_xy=[], fit_reference_xy=[], optimized_parameter_count_including_gain_bias=6,
        protection=dict(outside_mask_constant=0, gaussian_radius_px=2, safe_mask_erosion_radius_px=3,
            reference_interpolation="explicit_bilinear_all_four_corners_safe",
            derivative="same_four_safe_values_and_geometry_only_numerical_derivatives"),
        limits=dict(max_iterations=35, minimum_correlation=.85, minimum_std=.002,
            minimum_samples=160, minimum_initial_support_retained=.9,
            local_focal_ratio=[.65,1.55], maximum_corner_correction_px=40.))

    def fail(reason):
        result["reason"] = reason
        return result

    def matrix(params):
        return camera_h(cur.shape, reference_focal, math.exp(float(np.clip(params[3],-20,20))),
                        cv2.Rodrigues(params[:3])[0])

    h0 = matrix(p)
    corners = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],float)
    initial_corners = _project(h0,corners)
    initial_area = cv2.contourArea(initial_corners.astype(np.float32),oriented=True)
    step = max(1,math.ceil(math.sqrt(np.count_nonzero(cs)/7000)))
    yy,xx = np.mgrid[step//2:height:step,step//2:width:step]
    pool = np.c_[xx.ravel(),yy.ravel()]
    pool = pool[cs[pool[:,1],pool[:,0]]]
    if len(pool)>7000:
        pool=pool[np.linspace(0,len(pool)-1,7000,dtype=int)]
    ids,_,_,_=_bilinear(ref,rs,_project(h0,pool))
    pool=pool[ids]
    result["sampling_pool_current_xy"]=pool.tolist()
    if len(pool)<160:
        return fail("insufficient_safe_overlap")
    support=_support(pool,cur.shape)
    result["initial_support"]=support
    if support["hull_fraction"]<.02 or support["span_x_fraction"]<.18 or support["span_y_fraction"]<.1:
        return fail("insufficient_spatial_support")
    current_values=cur[pool[:,1],pool[:,0]]
    _,reference_values,_,_=_bilinear(ref,rs,_project(h0,pool))
    if min(np.std(current_values),np.std(reference_values))<.002:
        return fail("insufficient_texture")
    exposure=np.linalg.lstsq(np.c_[reference_values,np.ones(len(pool))],current_values,rcond=None)[0]
    if .5<=exposure[0]<=2 and abs(exposure[1])<=.35:
        p[4:]=[math.log(exposure[0]),exposure[1]]

    def evaluate(params,iteration,scale):
        h=matrix(params)
        mapped=_project(h,pool)
        ids,values,gx,gy=_bilinear(ref,rs,mapped)
        gain=math.exp(float(np.clip(params[4],-50,50)))
        residual=gain*values+params[5]-current_values[ids]
        row=dict(evaluation=len(result["iterations"]),iteration=iteration,step_scale=scale,
                 accepted=False,H_reference_from_current=h.tolist(),gain=gain,bias=float(params[5]),
                 valid_count=len(ids),valid_pool_index_ranges=_ranges(ids),
                 robust_loss=_loss(residual) if len(ids) else None)
        result["iterations"].append(row)
        guard=None
        projected=_project(h,corners)
        den=(np.c_[corners,np.ones(4)]@h.T)[:,2]
        area=cv2.contourArea(projected.astype(np.float32),oriented=True)
        if len(ids)<max(160,.9*len(pool)):
            guard="safe_support_lost"
        elif not .5<=gain<=2 or abs(params[5])>.35:
            guard="exposure_out_of_bounds"
        elif (not np.all(np.isfinite(projected)) or min(den)*max(den)<=0 or area*initial_area<=0
              or np.max(np.linalg.norm(projected-initial_corners,axis=1))>40
              or not .65<=math.exp(float(np.clip(params[3],-20,20)))/initial_current_focal<=1.55):
            guard="motion_or_camera_geometry_out_of_bounds"
        row["guard"]=guard
        return dict(h=h,mapped=mapped,ids=ids,values=values,gx=gx,gy=gy,
                    residual=residual,row=row,guard=guard,gain=gain)

    def jacobian(state,params):
        columns=[]
        selected=pool[state["ids"]]
        # Differentiate coordinates only; no extra pixels are read here.
        for j in range(4):
            offset=np.zeros(6);offset[j]=1e-6
            d=(_project(matrix(params+offset),selected)-_project(matrix(params-offset),selected))/2e-6
            columns.append(state["gain"]*(state["gx"]*d[:,0]+state["gy"]*d[:,1]))
        return np.column_stack(columns+[state["gain"]*state["values"],np.ones(len(selected))])

    state=evaluate(p,0,0)
    if state["guard"]:
        return fail(state["guard"])
    state["row"]["accepted"]=True
    result["initial_correlation"]=_correlation(state["values"],current_values[state["ids"]])
    converged=False
    for iteration in range(1,36):
        jac=jacobian(state,p)
        weights=np.minimum(1.,.015/np.maximum(np.abs(state["residual"]),1e-15))
        weighted=jac*np.sqrt(weights[:,None])
        lengths=np.maximum(np.linalg.norm(weighted,axis=0),1e-12)
        scaled=weighted/lengths
        singular=np.linalg.svd(scaled,compute_uv=False)
        rank=float(singular[-1]/singular[0])
        result["design_rank_ratio"]=rank
        if rank<1e-5:
            return fail("photometric_system_rank_deficient")
        delta=np.linalg.solve(scaled.T@scaled+1e-5*np.eye(6),
                              scaled.T@(-state["residual"]*np.sqrt(weights)))/lengths
        improved=False
        for line in range(9):
            scale=.5**line
            proposal=p+scale*delta
            trial=evaluate(proposal,iteration,scale)
            if trial["guard"]:continue
            common,a,b=np.intersect1d(state["ids"],trial["ids"],return_indices=True)
            before,after=_loss(state["residual"][a]),_loss(trial["residual"][b])
            trial["row"].update(common_count=len(common),common_loss_before=before,common_loss_after=after)
            if after<=before+1e-15:
                move=np.max(np.linalg.norm(trial["mapped"][common]-state["mapped"][common],axis=1))
                trial["row"]["accepted"]=True
                p,state=proposal,trial
                improved=True
                converged=bool(move<1e-3 and np.max(np.abs(scale*delta[4:]))<1e-5)
                break
        if converged or not improved:break
    ids=state["ids"]
    corr=_correlation(state["values"],current_values[ids])
    result.update(final_correlation=corr,gain=state["gain"],bias=float(p[5]),
        final_rmse_0_to_1=float(np.sqrt(np.mean(state["residual"]**2))),converged=converged,
        safe_support_retained=len(ids)/len(pool),support=_support(pool[ids],cur.shape),
        final_evaluation=state["row"]["evaluation"],fit_current_xy=pool[ids].tolist(),
        fit_reference_xy=state["mapped"][ids].tolist(),current_focal_px=float(math.exp(p[3])),
        rotation_reference_from_current=cv2.Rodrigues(p[:3])[0].tolist())
    if corr<.85:return fail("insufficient_photometric_correlation")
    if min(np.std(state["values"]),np.std(current_values[ids]))<.002:
        return fail("insufficient_final_texture")
    result.update(status="candidate",reason="rotation_zoom_candidate_requires_independent_ice_check",
                  H_reference_from_current=state["h"].tolist())
    return result


@dataclass(frozen=True)
class RotationPhotometricConfig(PhotometricConfig):
    motion_model: str = "rotation_zoom"
    initial_focal_width_ratio: float = 1.


class RotationPhotometricRegistrar(PhotometricRegistrar):
    def __init__(self,config=None,fit_config=None):
        # Parent initialization/reference texture test uses its frozen affine
        # helper; the identity contains no estimated camera compensation.
        super().__init__(config or RotationPhotometricConfig(),fit_config)
        self.rotation=np.eye(3)
        self.focal=None

    def push(self,image,sample,masks,force_break=False):
        if self.focal is None or force_break:
            self.rotation=np.eye(3)
            self.focal=image.shape[1]*self.config.initial_focal_width_ratio
        # Parent's self-reference call has only its supported model names.
        from dataclasses import replace
        original=self.config
        self.config=replace(self.config,motion_model="affine")
        try:
            row=super().push(image,sample,masks,force_break)
        finally:
            self.config=original
        row.update(motion_model="rotation_zoom",initial_focal_width_ratio=original.initial_focal_width_ratio,
                   current_focal_px=self.focal if row["status"]=="candidate" else None,
                   camera_model_independently_verified=False)
        return row

    def _key(self,gray,usable,sample,h):
        key=super()._key(gray,usable,sample,h)
        key.update(rotation=self.rotation.copy(),focal=self.focal)
        return key

    def _photo_edge(self,key,gray,usable,isolated,safe):
        relative=key["rotation"].T@self.rotation
        audit=estimate_rotation(key["raw"],gray,key["usable"],usable,
            reference_focal=key["focal"],initial_current_focal=self.focal,initial_rotation=relative)
        audit.update(edge_keyframe_sample_index=key["sample_index"],edge_keyframe_time_s=key["time"],
            edge_keyframe_source_pts=key["source_pts"],edge_keyframe_time_base=key["source_time_base"],
            initializer="last_observed_camera_pose_no_heldout_data",
            H_reference_from_keyframe=key["H_reference_from_keyframe"].tolist())
        self.last_photometric_audit["attempts"].append(audit)
        edge=dict(status=audit["status"],reason=audit["reason"],edge_keyframe_sample_index=key["sample_index"],
            edge_keyframe_time_s=key["time"],initializer=audit["initializer"],
            photometric_final_correlation=audit.get("final_correlation"),
            photometric_final_rmse_0_to_1=audit.get("final_rmse_0_to_1"))
        if audit["status"]!="candidate":return edge
        rotation=key["rotation"]@np.asarray(audit["rotation_reference_from_current"])
        focal=audit["current_focal_px"]
        combined=camera_h(gray.shape,gray.shape[1]*self.config.initial_focal_width_ratio,focal,rotation)
        current=np.asarray(audit["fit_current_xy"]);key_xy=np.asarray(audit["fit_reference_xy"])
        support=np.asarray(audit["support"]["hull_xy"])
        den=(np.c_[support,np.ones(len(support))]@combined.T)[:,2]
        # A finite rotation may leave the initial camera's projective chart.
        # Retain its observed camera pose, but explicitly require the renderer
        # to find ONE finite chart for the full shot or refuse that display.
        chart_horizon=bool(min(den)*max(den)<=0)
        audit["reference_chart_horizon_crosses_support"]=chart_horizon
        self.rotation,self.focal=rotation,focal
        selected=np.linspace(0,len(current)-1,min(200,len(current)),dtype=int)
        edge.update(H_reference_from_current=combined.tolist(),H_keyframe_from_current=audit["H_reference_from_current"],
            support_polygon=support.tolist(),support_fraction=audit["support"]["hull_fraction"],
            fit_current_xy=current[selected].tolist(),fit_keyframe_xy=key_xy[selected].tolist(),
            fit_reference_xy=_project(combined,current[selected]).tolist(),
            fit_points_full_count=len(current),fit_points_are_subset=len(selected)<len(current),
            fit_points_uniform_indices=selected.tolist(),fit_points_role="photometric_sampling_not_independent_correspondences",
            photometric_design_rank_ratio=audit["design_rank_ratio"],dense_optimization_used=True,
            train_inliers=0,train_inliers_semantics="dense_fit_has_no_feature_inliers",
            reference_chart_horizon_crosses_support=chart_horizon,
            display_requires_whole_shot_finite_chart_check=True,
            model_assumption=audit["model_assumption"])
        return edge
