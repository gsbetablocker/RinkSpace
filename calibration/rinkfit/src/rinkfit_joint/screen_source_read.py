"""Read real ice material profiles using the current screen state for search.

The historical chart supplies navigation only. Returned source UV never comes
from that chart, a fitted camera or interpolation across missing material.
"""
import hashlib
import numpy as np
from rinkspace.vision_programme_ice_review import _source_queries,_candidate_context_excluded
from rinkspace.vision_source_shapes import (read_source_dark_stroke_profiles,read_source_chromatic_stroke_centers)


def read_screen_ice_curves(state,rgb,mask,binding,curves,*,spacing_px=12.,search_radius_px=64.,keep_profiles=False):
    rgb=np.asarray(rgb);mask=np.asarray(mask,bool)
    if rgb.shape!=(1080,1920,3)or mask.shape!=rgb.shape[:2]:raise ValueError('Original native RGB and same-source exclusions required')
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=binding['decoded_native_rgb_sha256']:raise ValueError('Actual source RGB binding differs')
    pts=int(binding['source_pts']);sid=binding['shot_id'];outputs=[]
    for curve in curves:
        endpoints=np.asarray(curve['old_endpoints_L'],float)
        xy=np.linspace(endpoints[0],endpoints[1],1025)
        nav,valid=state.project_ice(pts,sid,xy,world_chart_sha256=state.world_chart_sha256)
        points,normals,intervals=_source_queries({},nav,np.where(valid,1.,-1.),rgb.shape[:2],spacing_px,search_radius_px)
        signal=curve['material_signal']
        common=dict(binding=binding,search_points_uv=points,normals=normals,search_intervals_px=intervals,keep_profiles=keep_profiles)
        if signal in ('red','blue'):
            result=read_source_chromatic_stroke_centers(rgb,material_signal='red_minus_green'if signal=='red'else'blue_minus_red',**common)
            candidate_key='all_actual_source_chromatic_stroke_candidates';uvkey='actual_source_center_candidate_uv'
        else:
            result=read_source_dark_stroke_profiles(rgb,**common)
            candidate_key='all_actual_source_dark_stroke_candidates';uvkey='actual_source_dark_stroke_center_candidate_uv'
        selected=[];rejected=[];candidates=[]
        for qi,reading in enumerate(result['readings']):
            clean=[]
            for candidate in reading[candidate_key]:
                uv=np.asarray(candidate[uvkey],float)
                if signal in ('red','blue'):
                    sides=np.asarray([x['actual_source_uv']for x in candidate['actual_RGB_band_interfaces']])
                else:
                    sides=np.asarray([candidate['actual_RGB_entering_interface']['actual_source_transition_candidate_uv'],candidate['actual_RGB_leaving_interface']['actual_source_transition_candidate_uv']])
                reasons=[k for k,v in candidate['source_flags'].items()if v]
                if any(_candidate_context_excluded(mask,p,normals[qi])for p in [uv,*sides]):reasons.append('actual_candidate_context_excluded')
                item=dict(query_index=qi,current_uv=uv.tolist(),source_flanks_uv=sides.tolist(),source_normal=normals[qi].tolist(),
                    actual_material_reading=candidate,rejection_reasons=reasons)
                candidates.append(item)
                if not reasons:clean.append(item)
            if len(clean)==1:
                item=clean[0];item.update(material_instance=curve['current_material_label'],physical_surface='ice',
                    operator='paired_material_boundary_midline',instance_basis='same finite static ice carrier in current source; historical chart is search only',
                    model_UV_used_as_observation=False,source_clock_preserved=True)
                selected.append(item)
            else:rejected.append(dict(query_index=qi,navigation_uv=points[qi].tolist(),reason='no_current_clean_material_candidate'if not clean else'competing_actual_material_candidates',candidate_count=len(clean)))
        outputs.append(dict(curve_id=curve['curve_id'],material_label=curve['current_material_label'],material_signal=signal,
            source_candidates=candidates,selected_source_rows=selected,unresolved=rejected,
            queries=len(points),native_source_readings=result,
            historical_chart_station_is_not_physical_correspondence=True))
    return dict(schema='rinkfit.screen_navigated_current_ice_profiles.v1',source_binding=binding,
        screen_state_sha256=state.sha256,curve_outputs=outputs,source_truth_from_model=False,
        geometry_precision_claimed=False,static_surface_only=True)
