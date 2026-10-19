"""The frozen external Ad13 normal scans, on ONE shared print geometry.

No pixel normal band becomes a two-axis box. The original source tangent
coordinate is fixed; the outer projected-circle intersection along the
source outward normal supplies the one scalar. Angle is an implicit curve
coordinate, not an observed world correspondence or a nearest-point guess.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import cv2
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from z190_shared_board_pattern_source import _project

PACKET_SHA = 'ed121036301b33c1b8cb47b90178709121cf984eca6523c8b5e94c799e60ee85'
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
FACTOR_KIND = 'external_original_normal_interval'
OUTER_STROKE_AUTHORITY = dict(
    kind='same-event real source comparison, independently reviewed',
    physical_entity='Ad13 outermost circular seal stroke outside its annular lettering',
    excludes='inner circle enclosing the central emblem; printed letters',
    source_native_pts=[230422016,15657728],
    source_only_comparison='D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/ice_alignment_review/external_pattern_source_adapter_audit_v1/same_Ad13_outer_seal_SOURCE_pixels_comparison.png',
    projection_used_to_determine_identity=False,
    colour_difference_does_not_define_another_ring=True)


def admit_external_pattern_frames(external_bundle):
    """Validate hashes/actual aliases; preserve all 64 original normal scans.

    Returns four Ad13 frames (3 development and the JPEG held) and a manifest.
    The 12 real Ad12 pre-window scans are report-only: no fictitious actual
    state or world material-point correspondence is introduced for them.
    """
    reference=external_bundle.manifest['source_authorities']['patterns']
    path=Path(reference['path']); raw=path.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    if sha!=PACKET_SHA or reference['sha256']!=PACKET_SHA:
        raise ValueError('Frozen external print source authority changed')
    data=json.loads(raw.decode('utf8'))
    if (data.get('source_sha256')!=SOURCE_SHA or data.get('source_measurements_frozen_before_fit')is not True
            or data.get('not_model_selected')is not True or data.get('fitted')is not False
            or external_bundle.pattern_source_records!=data['rows']):
        raise ValueError('Original independent source pattern observations required')
    checked={}; frames={}; excluded=[]
    for source in data['rows']:
        source=deepcopy(source); ref=source['source_ref']; p=int(source['source_native_pts'])
        png=Path(ref['raw_png'])
        if str(png)not in checked:
            if hashlib.sha256(png.read_bytes()).hexdigest()!=ref['raw_png_sha256']:
                raise ValueError('External native PNG changed')
            pixels=cv2.imread(str(png))
            if (pixels is None or pixels.shape!=(1080,1920,3)
                    or hashlib.sha256(pixels.tobytes()).hexdigest()!=ref['native_bgr_sha256']):
                raise ValueError('External native BGR source changed')
            checked[str(png)]=dict(raw_png_sha256=ref['raw_png_sha256'],native_bgr_sha256=ref['native_bgr_sha256'])
        if (source.get('source_visible')is not True or source.get('person_occlusion')is not False
                or source.get('manually_qualified_source_identity')is not True
                or source.get('model_projection_used')is not False):
            raise ValueError('True visible source-only printed boundary required')
        if source['physical_id']=='Ad12':
            if p in external_bundle.aliases or source.get('outside_actual38')is not True or source.get('fit_admitted')is not False:
                raise ValueError('Pre-window Ad12 must remain explicitly outside actual38')
            excluded.append(source); continue
        if source['physical_id']!='Ad13' or source['physical_pattern_id']!='TCF_outer_red_annulus':
            raise ValueError('Only the independently confirmed outermost Ad13 stroke is admitted')
        actual=int(external_bundle.aliases[p]); split='held'if actual in external_bundle.held_pts else'fit'
        if source['source_actual_pts']!=actual or source['split']!=('held'if split=='held'else'development_source'):
            raise ValueError('Frozen external source actual role changed')
        uv=np.asarray(source['uv_native'],float); normal=np.asarray(source['normal_native'],float)
        interval=np.asarray(source['normal_support_interval_px'],float)
        if (uv.shape!=(2,)or normal.shape!=(2,)or interval.shape!=(2,)
                or not np.isfinite(np.r_[uv,normal,interval]).all()or interval[0]>interval[1]
                or abs(float(normal@normal)-1.)>1e-10 or source['source_sigma_px']!=4.):
            raise ValueError('Original finite unit-normal support and engineering sigma4 required')
        if actual not in frames:
            frames[actual]=dict(source_native_pts=p,source_actual_pts=actual,
                physical_ad_id=13,physical_pattern_id='Ad13_outermost_circular_seal',
                split=split,engineering_sigma_native_px=[4.,4.],source_factor_kind=FACTOR_KIND,
                raw_png=str(png),raw_png_sha256=ref['raw_png_sha256'],native_bgr_sha256=ref['native_bgr_sha256'],
                native_size_wh=[1920,1080],source_pts_timebase=[1,15360],
                observer_installation='long_side_main',source_curve_support=[],
                source_visible=True,source_support_manually_qualified=True,
                model_projection_used_to_generate_or_select_uv=False,frozen_before_fit=True,
                source_packet_authority=dict(path=str(path),sha256=sha),
                physical_outer_stroke_authority=OUTER_STROKE_AUTHORITY,
                world_position_radius_unknown_and_shared=True)
        row=deepcopy(source)
        row.update(uv=deepcopy(source['uv_native']),source_support_manually_qualified=True,
            source_factor_kind=FACTOR_KIND,native_support_bounds_xy_px=None,
            source_material='outermost printed seal stroke against light board',
            source_original_row=deepcopy(source))
        frames[actual]['source_curve_support'].append(row)
    answer=[frames[p]for p in sorted(frames)]
    if (set(frames)!={15653888,15655168,15657728,15661568}
            or any(len(f['source_curve_support'])!=16 for f in answer)
            or len(excluded)!=12 or len(checked)!=5):
        raise ValueError('External physical source inventory changed')
    return answer,dict(packet=dict(path=str(path),sha256=sha),actual_frames=sorted(frames),
        fit_normal_intervals=48,held_normal_intervals=16,source_held_actual_pts=[15657728],
        source_held_native_aliases=[15657728,15657984],raw_native_pngs_verified=checked,
        source_uv_normal_interval_sigma_and_roles_unchanged=True,
        no_normal_interval_converted_to_bounding_box=True,
        same_physical_outermost_stroke_authority=OUTER_STROKE_AUTHORITY,
        excluded_Ad12_report_only_original_rows=excluded,Ad12_private_state_not_created=True,
        new_world_centre_or_per_camera_geometry_created=False,no_solve_invoked=True)


class ExternalBoardPatternSourceFactor:
    """Original 1D normal support and exact implicit branch derivatives.

    geometry can be the common perimeter station-Z chart or its historical
    planar chart. It supplies the same nine shared metre parameters; this
    class never owns a second pattern geometry or camera pose.
    """
    def __init__(self,geometry,*,engineering_circle_samples=256):
        if engineering_circle_samples<64:
            raise ValueError('Explicit whole closed-curve engineering sampling >=64 required')
        self.geometry=geometry;self.samples=int(engineering_circle_samples)
        self.theta=np.linspace(0.,2*np.pi,self.samples+1)

    def evaluate(self,q9,camera25,frame,with_jacobian=True):
        self.geometry.guard(q9);q=np.asarray(q9,float)
        if (frame.get('source_factor_kind')!=FACTOR_KIND or frame.get('physical_ad_id')not in(13,26)
                or frame.get('split')not in('fit','held') or frame.get('frozen_before_fit')is not True
                or frame.get('model_projection_used_to_generate_or_select_uv')is not False
                or frame.get('engineering_sigma_native_px')!=[4.,4.]):
            raise ValueError('Only frozen original external normal intervals are supported')
        ad=int(frame['physical_ad_id']);world,_=self.geometry.circle(q,self.theta,ad_id=ad)
        coarse,_,_=_project(camera25,world)
        residual=[];jq=[];jc=[];reports=[]
        for source in frame['source_curve_support']:
            if (source.get('source_visible')is not True or source.get('source_support_manually_qualified')is not True
                    or source.get('model_projection_used')is not False or source.get('source_sigma_px')!=4.
                    or source.get('native_support_bounds_xy_px')is not None):
                raise ValueError('Original visible normal interval cannot be replaced by a box')
            uv0=np.asarray(source['uv'],float);normal=np.asarray(source['normal_native'],float)
            tangent=np.array([-normal[1],normal[0]])
            interval=np.asarray(source['normal_support_interval_px'],float)
            if (uv0.shape!=(2,)or normal.shape!=(2,)or interval.shape!=(2,)
                    or not np.isfinite(np.r_[uv0,normal,interval]).all()
                    or abs(float(normal@normal)-1.)>1e-10 or interval[0]>interval[1]):
                raise ValueError('Finite source unit-normal scanline required')
            values=(coarse-uv0)@tangent
            roots=[]
            def scan(theta):
                w,_=self.geometry.circle(q,[theta],ad_id=ad)
                uv,_,_=_project(camera25,w)
                return float((uv[0]-uv0)@tangent)
            for i in range(self.samples):
                a,c=values[i],values[i+1]
                if a==0.:roots.append(float(self.theta[i]))
                if a*c<0.:
                    roots.append(float(brentq(scan,self.theta[i],self.theta[i+1],xtol=1e-14)))
            roots=sorted(r%(2*np.pi)for r in roots)
            unique=[]
            for angle in roots:
                if not unique or abs(angle-unique[-1])>1e-10:unique.append(angle)
            if len(unique)>1 and 2*np.pi-unique[-1]+unique[0]<1e-10:unique.pop()
            if not unique:
                raise ValueError('Source tangent scanline has no real projected outer-circle intersection')
            hits=[]
            for angle in unique:
                w,jw=self.geometry.circle(q,[angle],ad_id=ad)
                uv,camera_j,world_j=_project(camera25,w,True)
                d_world=-q[2]*np.sin(angle)*jw[0,:,0]+q[2]*np.cos(angle)*jw[0,:,1]
                d_uv=world_j[0]@d_world
                if abs(float(tangent@d_uv))<=1e-8:
                    raise ValueError('Source tangent scan is tangent to the curve: implicit derivative undefined')
                hits.append((float((uv[0]-uv0)@normal),angle,uv[0],camera_j[0],world_j[0],jw[0],d_uv))
            hits.sort(key=lambda item:item[0],reverse=True)
            if len(hits)>1 and hits[0][0]-hits[1][0]<=1e-8:
                raise ValueError('Source outward branch is not uniquely separable')
            distance,angle,uv,camera_j,world_j,world_q,d_uv=hits[0]
            outside=distance-float(np.clip(distance,*interval));sigma=float(source['source_sigma_px'])
            residual.append(outside/sigma)
            if with_jacobian:
                chain=(normal-float(normal@d_uv)/float(tangent@d_uv)*tangent)*float(outside!=0.)/sigma
                jq.append(chain@world_j@world_q);jc.append(chain@camera_j)
            reports.append(dict(source_id=source['id'],source_native_pts=int(frame['source_native_pts']),
                source_normal_interval_px=interval.tolist(),projected_outer_branch_native_uv=uv.tolist(),
                projected_curve_theta_implicit=angle,real_scanline_intersections=len(hits),
                chosen_branch='maximum along frozen source outward normal',
                normal_offset_native_px=distance,outside_normal_support_native_px=outside,
                frozen_source_tangent_coordinate_preserved=True,
                ellipse_centre_used_as_world_measurement=False,nearest_curve_angle_used=False))
        return SimpleNamespace(residual=np.asarray(residual),
            shared_q9_jacobian=np.asarray(jq).reshape(-1,9)if with_jacobian else None,
            camera_state11_jacobian=np.asarray(jc).reshape(-1,11)if with_jacobian else None,
            source_reports=reports,source_held=frame['split']=='held',
            source_extrema_or_pixel_ellipse_center_used=False)


class ExternalBoardPatternDistanceSourceFactor(ExternalBoardPatternSourceFactor):
    """Finite distance to the ORIGINAL normal-support segment.

    Each source UV/normal/interval defines a segment, not a two-axis box.
    Profile theta over only the projected outward half-contour: its outward
    normal must face the frozen source outward normal. This remains a finite
    positive error when a large initial displacement leaves the source
    scanline outside the circle. The old strict scanline operator is retained
    above for replay. No opposite branch or ellipse centre is substituted.
    """
    def evaluate(self,q9,camera25,frame,with_jacobian=True):
        self.geometry.guard(q9);q=np.asarray(q9,float)
        if (frame.get('source_factor_kind')!=FACTOR_KIND or frame.get('physical_ad_id')not in(13,26)
                or frame.get('split')not in('fit','held')or frame.get('frozen_before_fit')is not True
                or frame.get('model_projection_used_to_generate_or_select_uv')is not False
                or frame.get('engineering_sigma_native_px')!=[4.,4.]):
            raise ValueError('Original frozen finite normal source segment required')
        ad=int(frame['physical_ad_id'])
        # A circle crossing a retained board-polyline knot is not smoothed.
        # Keep its genuine piece boundaries in the theta profiling domain.
        knot_map={}
        if hasattr(self.geometry,'course_station_m'):
            for station in self.geometry.course_station_m:
                fraction=(float(station)-q[0])/q[2]
                if -1.<fraction<1.:
                    angle=float(np.arccos(fraction))
                    knot_map[angle]=float(station);knot_map[2*np.pi-angle]=float(station)
        grid=np.unique(np.r_[self.theta,list(knot_map)])
        w,jw=self.geometry.circle(q,grid,ad_id=ad)
        uv,cam_j,world_j=_project(camera25,w,True)
        tangent_world=-q[2]*np.sin(grid)[:,None]*jw[:,:,0]+q[2]*np.cos(grid)[:,None]*jw[:,:,1]
        tangent_uv=np.einsum('nij,nj->ni',world_j,tangent_world)
        twice_area=float(np.sum(uv[:-1,0]*uv[1:,1]-uv[1:,0]*uv[:-1,1]))
        if abs(twice_area)<=1e-8:
            raise ValueError('Projected closed circle orientation is degenerate')
        orientation=float(np.sign(twice_area));residual=[];q_j=[];camera_j=[];reports=[]
        def projected(theta,jac=False):
            point,jpoint=self.geometry.circle(q,[theta],ad_id=ad)
            pixel,jcamera,jworld=_project(camera25,point,jac)
            if not jac:return pixel[0]
            d_world=-q[2]*np.sin(theta)*jpoint[0,:,0]+q[2]*np.cos(theta)*jpoint[0,:,1]
            d_uv=jworld[0]@d_world
            combined=np.c_[jworld[0]@jpoint[0],jcamera[0]]
            return pixel[0],d_uv,combined
        for source in frame['source_curve_support']:
            if (source.get('source_visible')is not True or source.get('source_support_manually_qualified')is not True
                    or source.get('model_projection_used')is not False or source.get('source_sigma_px')!=4.
                    or source.get('native_support_bounds_xy_px')is not None):
                raise ValueError('No box conversion or model-selected source is permitted')
            uv0=np.asarray(source['uv'],float);n=np.asarray(source['normal_native'],float)
            interval=np.asarray(source['normal_support_interval_px'],float)
            if (uv0.shape!=(2,)or n.shape!=(2,)or interval.shape!=(2,)
                    or not np.isfinite(np.r_[uv0,n,interval]).all()or interval[0]>interval[1]
                    or abs(float(n@n)-1.)>1e-10):
                raise ValueError('Original finite support and unit source normal required')
            t=np.array([-n[1],n[0]]);sigma=float(source['source_sigma_px'])
            def branch(theta):return orientation*float(t@projected(theta,True)[1])
            def excess(theta,jac=False):
                value=projected(theta,jac)
                pixel=value[0]if jac else value
                nearest=uv0+n*float(np.clip((pixel-uv0)@n,*interval))
                return (pixel-nearest)/sigma,value
            def cost(theta):
                e,_=excess(theta);return float(e@e)
            def slope(theta):
                e,value=excess(theta,True);return float(2*e@value[1]/sigma)
            g=orientation*(tangent_uv@t);bounds={0.:'periodic',float(2*np.pi):'periodic'}
            for angle in knot_map:bounds[angle]='retained_polyline_knot'
            for i in range(len(grid)-1):
                if grid[i]in knot_map or grid[i+1]in knot_map:
                    # At each retained knot the one-sided tangent is used.
                    delta=min(1e-8,(grid[i+1]-grid[i])/1000.)
                    left,right=grid[i]+delta,grid[i+1]-delta
                    ga,gc=branch(left),branch(right)
                else:left,right=grid[i],grid[i+1];ga,gc=g[i],g[i+1]
                if ga*gc<0:
                    angle=float(brentq(branch,left,right,xtol=1e-14));bounds[angle]='outward_half_boundary'
                elif ga==0.:bounds[float(left)]='outward_half_boundary'
            order=sorted(bounds);candidates=[];intervals=[]
            for left,right in zip(order,order[1:]):
                if right-left<=1e-12 or branch((left+right)/2.)<0:continue
                intervals.append([left,right])
                candidates.extend((cost(theta),theta,bounds[theta])for theta in(left,right))
                fit=minimize_scalar(cost,bounds=(left,right),method='bounded',options={'xatol':1e-12})
                angle=float(fit.x)
                delta=min(1e-8,(right-left)/1000.)
                if slope(left+delta)<0<slope(right-delta):
                    angle=float(brentq(slope,left+delta,right-delta,xtol=1e-14))
                candidates.append((cost(angle),angle,'interior_profile'))
            if not candidates:raise ValueError('No genuine source-facing half-contour exists')
            _,angle,kind=min(candidates,key=lambda item:item[0])
            e,value=excess(angle,True);pixel,d_uv,all_j=value;r=float(np.linalg.norm(e))
            # A real segment/curve intersection is zero; do not divide by
            # floating point roundoff after solving its profiled coordinate.
            numerical_zero=r*sigma<=1e-8
            if numerical_zero:r=0.;e=np.zeros(2)
            residual.append(r)
            if with_jacobian:
                chain=np.zeros(2)if r==0. else e/r/sigma
                theta_j=np.zeros(20)
                if kind=='retained_polyline_knot':
                    denominator=-q[2]*np.sin(angle)
                    if abs(denominator)<=1e-10:raise ValueError('Circle touches a retained course knot tangentially')
                    theta_j[0]=-1./denominator;theta_j[2]=-np.cos(angle)/denominator
                elif kind=='outward_half_boundary':
                    # Differentiate the moving boundary g=orientation*t.u_theta=0.
                    # Mixed derivatives use the analytic point J at theta±h;
                    # no source/world values or parameters are changed.
                    h=1e-5;plus=projected(angle+h,True);minus=projected(angle-h,True)
                    denominator=orientation*float(t@(plus[1]-minus[1]))/(2*h)
                    if abs(denominator)<=1e-8:raise ValueError('Outward half-contour endpoint has undefined derivative')
                    gp=orientation*(t@(plus[2]-minus[2]))/(2*h)
                    theta_j=-gp/denominator
                total=chain@all_j+float(chain@d_uv)*theta_j
                q_j.append(total[:9]);camera_j.append(total[9:])
            reports.append(dict(source_id=source['id'],source_native_pts=int(frame['source_native_pts']),
                operator='distance to original finite native normal support segment',
                segment_endpoints_native=(uv0+interval[:,None]*n).tolist(),
                source_sigma_px=sigma,profiled_curve_theta_nuisance=angle,
                profiled_coordinate_kind=kind,projected_curve_uv_native=pixel.tolist(),
                outside_segment_vector_native_px=(e*sigma).tolist(),distance_native_px=r*sigma,
                source_outward_half_intervals_rad=intervals,
                source_outward_branch_rule='projected physical outward normal faces frozen source normal',
                reverse_half_contour_excluded=True,near_support_zero_means_same_closed_circle_touches_original_segment=True,
                numerical_intersection_roundoff_zero_native_px=1e-8,
                ellipse_centre_used_as_world_measurement=False,source_packet_modified=False))
        return SimpleNamespace(residual=np.asarray(residual),
            shared_q9_jacobian=np.asarray(q_j).reshape(-1,9)if with_jacobian else None,
            camera_state11_jacobian=np.asarray(camera_j).reshape(-1,11)if with_jacobian else None,
            source_reports=reports,source_held=frame['split']=='held',
            source_extrema_or_pixel_ellipse_center_used=False,
            strict_source_scanline_operator_retained_for_replay=True)
