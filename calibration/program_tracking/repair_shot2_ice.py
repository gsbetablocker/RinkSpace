"""Independent current-source ice reread for the selected shot-two review.

The old review exclusions are historical failures, not an authority for this
reread. Source guides locate identities only; observations require current PNG
material interfaces. The source clock, mask and exact PNG digest are checked.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import extract_first_shot_ice as shared
from .extract_first_shot_ice import (
    Extractor, NativeProfiles, blank_feature, read, write, sha, simplify,
    circle_geometry_digest, ice_curve_inventory, project_fast, runs,
    _source_queries, current_broadcast_exclusion,
)
from .extract_shot2_ice import SelectedExtractor, SelectedHouseReader, current_ice_domain
from .native_ice_material_repair import sample_chroma, narrow_dark_paint
from .physical_ice_width import ground_midpoint_from_source_sides
from scipy.optimize import least_squares
from .static_ice_source_exclusions import source_records, exclusion_mask, filter_source_readouts
from .static_ice_source_exclusions_v2 import filter_source_family_readouts


class SourceNormalProfiles(NativeProfiles):
    """Read the source's two exterior flanks across a visible image section.

    For two parallel material boundaries on z=0, the ground midpoint of their
    intersections with any transverse section lies on the physical midline.
    This avoids a nearly tangential section dictated by an inaccurate camera
    initializer, while preserving the existing Brown ground-midpoint operator.
    """
    def __init__(self, rgb, source, mask, camera):
        super().__init__(rgb, source, mask)
        self.frame.binding['shot_id']=int(source.get('shot_index',source.get('shot_id',2)))
        self.camera = camera

    def sample(self, path, guide, radius=28, spacing=48):
        local = dict(path)
        if path.get('_current_source_guide'):
            radius = min(radius, 28 if path['material_signal']=='dark' else 45)
        if path['feature_id'] in ('B1','B2','C'):
            rows, diagnostic = self.sample_complete_colour_band(local, guide, spacing)
        else:
            rows, diagnostic = super().sample(local, guide, radius=radius, spacing=32)
        rows, rejected = filter_source_readouts(rows, getattr(self, 'static_material_exclusions', []))
        if rejected:
            diagnostic['static_material_excluded_readouts'] = rejected
            diagnostic['measured_count'] = len(rows)
        if getattr(self, 'source_family_full_footprint_admission', False):
            rows, failed = filter_source_family_readouts(rows, self.mask,
                getattr(self, 'static_material_exclusions', []), operator=path.get('operator','material_interface'))
            diagnostic['full_reader_support_rejections_before_line_fit'] = failed
            diagnostic['full_reader_support_admitted_readouts'] = len(rows)
            self.source_family_admission_history.append(dict(kind='line',curve_id=path['curve_id'],
                admitted=len(rows),rejected=failed))
        return rows, diagnostic

    def sample_complete_colour_band(self, path, guide, spacing):
        local = dict(path, _whole_band_profile_radius=path.get('_whole_band_profile_radius',65))
        # Independently identified blue paint can be almost achromatic in the
        # broadcast. Its two own current luminance shoulders still determine
        # its width; the guide cannot determine either shoulder.
        if path['material_signal']=='blue'and not path.get('_force_complete_blue_chromatic_band'):
            q,n,interval = _source_queries(local,guide,np.ones(len(guide)),(1080,1920),32,
                path.get('_blue_native_profile_radius',40))
            if not len(q):return [],dict(query_count=0,measured_count=0)
            offsets,profiles,valid = sample_chroma(self.frame.rgb,q,n,interval,'dark')
            rows=[]
            for k in range(len(q)):
                choices=narrow_dark_paint(q[k],n[k],offsets[k],profiles[k],valid[k],
                    maximum_width=path.get('_current_native_maximum_band_width',75))
                choices=[r for r in choices if abs((np.asarray(r['uv'])-q[k])@n[k])<path.get('_current_native_band_identity_radius',18)
                         and 4<=r['source_evidence']['narrow_paint_width_px']<=path.get('_current_native_maximum_band_width',65)]
                if not choices:continue
                # Search is identity-constrained; select the target band's own
                # valley, not a different painted row crossing its strip.
                choices.sort(key=lambda r:abs((np.asarray(r['uv'])-q[k])@n[k]))
                row=choices[0];ev=row['source_evidence'];ev['query_index']=k
                ev['material_signal']='blue'
                ev['measurement_origin']='current_source_verified_blue_band_two_exterior_luminance_flanks'
                ev['blue_identity_from_native_visual_and_temporal_continuity']=True
                rows.append(row)
            diagnostic=dict(query_count=len(q),measured_count=len(rows))
        else:
            rows,diagnostic=NativeProfiles.sample_complete_colour_band(self,local,guide,32)
        # Opt in only for a source-confirmed complete, almost achromatic
        # painted strip. A broad neighbouring ring/abrasion can share its
        # centre station while providing neither of the strip's exterior
        # interfaces. Compare the two directly measured sides to nearby
        # current-source sections before any line or ground-midpoint fit.
        # This is an input qualification, never a replacement width.
        if path.get('_qualify_current_source_band_width_cohort') and len(rows)>=6:
            widths=np.asarray([np.linalg.norm(np.diff(r['source_evidence']['actual_source_sides_uv'],axis=0)) for r in rows])
            stations=np.asarray([r['source_evidence']['query_uv'] for r in rows])
            retained=[];rejected=[]
            for k,row in enumerate(rows):
                nearby=np.linalg.norm(stations-stations[k],axis=1)<=200
                nearby[k]=False
                cohort=widths[nearby]
                median=float(np.median(cohort)) if len(cohort)>=5 else float(np.median(np.delete(widths,k)))
                if widths[k]>max(2.5*median,median+8.):
                    rejected.append(dict(native_readout=row,nearby_current_source_band_width_median_px=median,
                        reason='current_source_flanks_span_other_materials_instead_of_the_confirmed_complete_strip'))
                    continue
                row['source_evidence']['nearby_current_source_complete_band_width_median_px']=median
                retained.append(row)
            rows=retained
            diagnostic['cross_material_width_support_rejected_before_line_fit']=rejected
        output=[]
        rejected_identity=[]
        for row in rows:
            ev=row['source_evidence'];sides=np.asarray(ev['actual_source_sides_uv'],float)
            matching=[]
            for record in path.get('_material_identity_exclusion_rois',[]):
                x0,y0,x1,y1=record['source_roi_xyxy']
                candidates=np.vstack([sides,np.asarray(row['uv'])[None],np.asarray(ev['query_uv'])[None]])
                if ((candidates>=[x0,y0])&(candidates<[x1,y1])).all(1).any():matching.append(record)
            if matching:
                rejected_identity.append(dict(native_readout=row,material_identity_exclusion_rois=matching));continue
            result=ground_midpoint_from_source_sides(self.camera,sides)
            if result is None:continue
            uv,center,ground_sides=result
            context=np.vstack([sides,np.asarray(ev['actual_source_side_quantiles_uv']).reshape(-1,2)])
            lo,hi=ev['complete_band_native_interval_px'];q=np.asarray(ev['query_uv']);n=np.asarray(ev['profile_normal'])
            context=np.vstack([context,q+np.arange(np.ceil(lo),np.floor(hi)+1)[:,None]*n])
            if (context<[0,0]).any() or (context>[1919,1079]).any():continue
            ip=np.rint(context).astype(int)
            if self.mask[ip[:,1],ip[:,0]].any():continue
            row['uv']=np.round(uv,4).tolist()
            ev.update(actual_measured_sides_ground_xy=ground_sides.tolist(),
                actual_measured_midpoint_ground_xy=center.tolist(),
                width_midpoint_operator='inverse_Brown_z0_measured_two_interfaces_ground_average_then_native_Brown_projection',
                width_profile_operator='current_source_image_transverse_section_of_same_parallel_material_boundaries',
                physical_midpoint_minus_image_arithmetic_midpoint_px=(uv-sides.mean(0)).tolist(),
                measured_boundary_pixels_from_current_source=True,
                scene_camera_used_only_for_ground_midpoint_coordinate_conversion=True,
                interior_ridges_not_used_as_band_exterior_edges=True)
            output.append(row)
        diagnostic.update(measured_count=len(output),operator='native_exterior_material_flanks_then_physical_ground_midpoint')
        if path.get('_material_identity_exclusion_rois'):
            diagnostic.update(material_identity_exclusions_applied_before_source_line_fit=True,
                rejected_material_identity_readouts=rejected_identity)
        return output,diagnostic


class CurrentHouseReader(SelectedHouseReader):
    """A nearby red straight band cannot impersonate the grey C0 paint."""
    def queries(self,group,index,role,parameters,spacing=35):
        theta,q,n,interval=super().queries(group,index,role,parameters,spacing=24)
        if group is self.groups.get('C0'):
            interval=np.maximum(interval,[-26.,-np.inf])
            interval=np.minimum(interval,[np.inf,26.])
        return theta,q,n,interval

    def measurements(self,group,index,native,parameters):
        rows,diagnostic=super().measurements(group,index,native,parameters)
        outline_roles=getattr(self,'material_outline_roles',{})
        if group['outline'] and group['paths'][0]['curve_id']in outline_roles:
            for row in rows:self.exterior_outline_interface(row)
        exclusions=[r for r in getattr(self,'material_identity_exclusions',[])
                    if r['native_pts']==native.frame.binding['source_pts']
                    and any(p['curve_id']==r['curve_id']for p in group['paths'])]
        if exclusions:
            kept=[];rejected=[]
            for row in rows:
                actual=np.asarray(row['uv']);query=np.asarray(row['source_evidence']['query_uv'])
                matching=[]
                for record in exclusions:
                    if row['role']!=record['curve_id'].split(':')[-1]:continue
                    x0,y0,x1,y1=record['source_roi_xyxy']
                    if any(x0<=uv[0]<x1 and y0<=uv[1]<y1 for uv in (actual,query)):
                        matching.append(record)
                if matching:rejected.append(dict(native_readout=row,material_identity_exclusion_rois=matching))
                else:kept.append(row)
            rows=kept
            for role,stats in diagnostic.items():
                role_rejected=[r for r in rejected if r['native_readout']['role']==role]
                stats.update(source_readout_candidates_before_identity_exclusion=stats['eligible_native_material_readouts'],
                    eligible_native_material_readouts=stats['eligible_native_material_readouts']-len(role_rejected),
                    material_identity_exclusions_applied_before_shared_family_fit=True,
                    rejected_non_static_material_readouts=role_rejected)
        rows, static_rejected = filter_source_readouts(rows, getattr(native, 'static_material_exclusions', []))
        if static_rejected:
            for role, stats in diagnostic.items():
                rejected_for_role = [r for r in static_rejected if r['native_readout']['role'] == role]
                stats['static_material_excluded_readouts'] = rejected_for_role
                stats['eligible_native_material_readouts'] -= len(rejected_for_role)
        if getattr(native, 'source_family_full_footprint_admission', False):
            rows, failed = filter_source_family_readouts(rows, native.mask,
                getattr(native, 'static_material_exclusions', []),
                operator='paint_center' if group['outline'] else 'material_interface')
            native.source_family_admission_history.append(dict(kind='circle',
                curve_ids=[p['curve_id'] for p in group['paths']],admitted=len(rows),rejected=failed))
            for role,stats in diagnostic.items():
                role_failed=[x for x in failed if x['native_readout']['role']==role]
                stats['full_reader_support_rejections_before_shared_family_fit']=role_failed
                stats['eligible_native_material_readouts']-=len(role_failed)
                stats['full_reader_support_admitted_readouts']=sum(r['role']==role for r in rows)
        if group is not self.groups.get('C0'):return rows,diagnostic
        kept=[];rejected=[]
        rgb=native.frame.rgb.astype(np.float32)
        red=rgb[...,0]-(rgb[...,1]+rgb[...,2])*.5
        for row in rows:
            ev=row['source_evidence'];uv=np.asarray(row['uv']);normal=np.asarray(ev['profile_normal'])
            sides=np.asarray(ev['actual_source_sides_uv'])
            outside=np.vstack([sides[0]-normal*5,sides[1]+normal*5])
            sample=np.vstack([uv,outside]).astype(np.float32)
            values=cv2.remap(red,sample[:,0][None],sample[:,1][None],cv2.INTER_LINEAR)[0]
            red_contrast=float(values[0]-values[1:].mean())
            if red_contrast>5:
                rejected.append(dict(uv=row['uv'],reason='red_straight_band_crossing_grey_circle',red_chroma_contrast=red_contrast))
                continue
            ev['current_native_grey_C0_vs_red_straight_band_disambiguation']=dict(red_chroma_contrast=red_contrast)
            kept.append(row)
        diagnostic.setdefault('outer',{}).update(eligible_native_material_readouts=len(kept),
            crossing_red_band_candidates=rejected)
        return kept,diagnostic

    @staticmethod
    def exterior_outline_interface(row):
        ev=row['source_evidence'];sides=np.asarray(ev['actual_source_sides_uv'],float)
        quantiles=np.asarray(ev['actual_source_side_quantiles_uv'],float)
        if len(sides)!=2:raise ValueError('Outline material needs both current source flanks')
        ev.update(actual_outline_inner_outer_uv=sides.tolist(),actual_outline_inner_outer_quantiles_uv=quantiles.tolist(),
            former_paint_width_midpoint_uv=row['uv'],outside_material_interface_used_for_outer_circle=True,
            actual_source_sides_uv=[sides[1].tolist()]*2,actual_source_side_quantiles_uv=[quantiles[1].tolist()]*2,
            measurement_origin='current_native_unfilled_circle_outside_paint_to_ice_interface')
        row['uv']=np.round(sides[1],4).tolist()
        row['sigma_px']=max(1.,float(np.linalg.norm(quantiles[1,2]-quantiles[1,0])/1.349))
        ev['transition_sigma_px']=row['sigma_px']


class IceReread(SelectedExtractor):
    def __init__(self,data_root,model_root,navigation=None,source_guides=None,shot_index=None):
        super().__init__(data_root,model_root,review_path=data_root/'no_prior_exclusion_authority.json',shot_index=shot_index)
        self.material_reviews={}
        source_review=read(source_guides)if source_guides and source_guides.exists()else{}
        self.source_review=source_review
        central_roles={r if isinstance(r,str)else r['curve_id']
                       for r in source_review.get('confirmed_material_circle_center_roles',[])}
        if central_roles:
            # The center role is a physical white disk boundary, never an
            # abstract circle-center point. It joins the same source family
            # only where its own inner/outer materials can actually be read.
            central=[p for p in ice_curve_inventory(self.scene)
                     if p['world_curve_role']=='center'and p['curve_id']in central_roles]
            if {p['curve_id']for p in central}!=central_roles:
                raise ValueError('Confirmed material center boundary is not a supplied scene center role')
            for p in central:p['_circle']=True
            self.paths.extend(central)
        self.material_outline_roles={r['curve_id']:r for r in source_review.get('confirmed_material_circle_outline_roles',[])}
        for path in self.paths:
            if path['curve_id']in self.material_outline_roles:path['operator']='material_interface'
        self.house_reader=CurrentHouseReader(self.scene,self.paths,self.cameras)
        self.house_reader.material_outline_roles=self.material_outline_roles
        self.wide_profiles={}
        for record in source_review.get('wide_band_search_profiles',[]):
            index=int(record['programme_frame_index_1based'])
            if record['native_pts']!=self.rows[index]['source']['native_pts']:raise ValueError('Wide profile source clock mismatch')
            self.wide_profiles[(index,record['curve_id'])]=record
        material_roles={r['curve_id']:r for r in source_review.get('source_material_curve_roles',[])}
        unresolved=[p for p in ice_curve_inventory(self.scene)
                    if p['feature_id'].startswith('goal_crease') and p['curve_id']in material_roles]
        for p in unresolved:
            role=material_roles[p['curve_id']]
            p.update(_circle=False,operator=role['measurement_operator'],material_signal=role['material_signal'],
                     material_transition=role.get('material_transition','light_to_dark'),
                     material_role_authority=role['authority'])
        self.paths.extend(unresolved)
        self.source_guides={}
        self.continuous_arc_reviews={}
        self.source_spots={}
        self.source_arc_records={}
        self.source_regions={}
        self.material_identity_exclusions=[]
        if source_guides and source_guides.exists():
            for record in source_review.get('material_identity_exclusion_rois',[]):
                index=int(record['programme_frame_index_1based'])
                if record['native_pts']!=self.rows[index]['source']['native_pts'] or record['time_base']!='1/12800':
                    raise ValueError('Material identity exclusion source clock mismatch')
                self.material_identity_exclusions.append(record)
            self.house_reader.material_identity_exclusions=self.material_identity_exclusions
            for record in source_review.get('unresolved_source_regions',[]):
                index=int(record['programme_frame_index_1based'])
                if record['native_pts']!=self.rows[index]['source']['native_pts']:raise ValueError('Region source clock mismatch')
                self.source_regions.setdefault(index,[]).append(record)
            for record in source_review.get('static_arc_guides',[]):
                self.source_arc_records.setdefault(int(record['programme_frame_index_1based']),[]).append(record)
            for record in source_review.get('static_spot_rois',[]):
                self.source_spots.setdefault(int(record['programme_frame_index_1based']),[]).append(record)
            for record in source_review.get('current_visually_continuous_material_arcs',[]):
                index=int(record['programme_frame_index_1based'])
                if record['native_pts']!=self.rows[index]['source']['native_pts'] or record['time_base']!='1/12800':
                    raise ValueError('Visible arc source clock mismatch')
                self.continuous_arc_reviews[(index,record['curve_id'])]=record
            for record in source_review.get('guides',[]):
                index=int(record['programme_frame_index_1based'])
                if record['native_pts']!=self.rows[index]['source']['native_pts'] or record['time_base']!='1/12800':
                    raise ValueError('Source guide clock mismatch')
                self.source_guides[(index,record['curve_id'])]=record
        self.navigation={}
        if navigation and navigation.exists():
            for row in read(navigation).get('guides',[]):
                if row.get('usable',True):
                    index=int(row.get('target_frame',row.get('target_programme_index')))
                    target=row.get('curve_id',row.get('target_id')).removeprefix('ice.')
                    key=(index,target)
                    previous=self.navigation.get(key)
                    if previous is None or row.get('native_adjacent_links',9999)<previous.get('native_adjacent_links',9999):
                        self.navigation[key]=row

    def guide(self,p,index):
        if p['feature_id'].startswith('goal_crease'):
            feature=next(f for f in self.scene['ice_features'] if f['id']==p['feature_id'])
            xyz=np.asarray(feature['polygon_xyz'],float)
            # Only the user's physical semicircular perimeter is curved. The
            # polygon's artificial closing chord must not become a frame/ROI
            # clipping edge or an extra observed paint boundary.
            world=np.vstack([np.linspace(a,b,8,endpoint=False)for a,b in zip(xyz[:-1],xyz[1:])]+[xyz[-1:]])
            uv,depth=project_fast(self.cameras[index],world)
            valid=np.isfinite(uv).all(1)&(depth>.05)
            return uv[valid],'8a28_same_physical_half_circle_boundary_navigation_only'
        record=self.source_guides.get((index,p['curve_id']))
        manual=record['uv'] if record else None
        transported=self.navigation.get((index,p['curve_id']))
        if manual is not None:
            return np.linspace(*np.asarray(manual,float),257),'independent_current_native_visual_identity_search'
        if transported:
            uv=np.asarray(transported['uv'],float)
            return uv,'adjacent_official_PTS_material_identity_navigation_only'
        return Extractor.guide(self,p,index)

    def line(self,p,guide,native,family_slope=None):
        index=next(i for i,r in self.rows.items()if r['source']['native_pts']==native.frame.binding['source_pts'])
        profile=self.wide_profiles.get((index,p['curve_id']),{})
        local=dict(p,_current_source_guide=True,
                   _line_identity_policy=dict(current_identity_constrained=True,
                   reference_frame=next(i for i,r in self.rows.items() if r['source']['native_pts']==native.frame.binding['source_pts']),
                   identity_source='8a28_identity_plus_current_native_source_visual_continuity',
                   normal_identity_tolerance_px=45 if p['material_signal']=='red' else 20,
                   search_radius_px=45 if p['material_signal']=='red' else 28,
                   navigation_coordinates_are_not_observations=True))
        for field,target in [('profile_radius_px','_whole_band_profile_radius'),('blue_profile_radius_px','_blue_native_profile_radius'),
                             ('maximum_current_band_width_px','_current_native_maximum_band_width'),('identity_radius_px','_current_native_band_identity_radius'),
                             ('require_complete_chromatic_band','_force_complete_blue_chromatic_band'),
                             ('qualify_current_source_band_width_cohort','_qualify_current_source_band_width_cohort')]:
            if field in profile:local[target]=profile[field]
        local['_material_identity_exclusion_rois']=[r for r in self.material_identity_exclusions
            if r['programme_frame_index_1based']==index and r['curve_id']==p['curve_id']]
        return Extractor.line(self,local,guide,native,family_slope)

    def nested_house_features(self,index,native):
        # Reuse the full joint physical-circle family reader. Disable only the
        # predecessor's global extra-circle rejection by schematic identity;
        # current material and actual ownership are reviewed on the output.
        original=shared.qualify_extra_circle_material
        try:
            shared.qualify_extra_circle_material=lambda values:values
            return Extractor.nested_house_features(self,index,native)
        finally:
            shared.qualify_extra_circle_material=original

    def outline(self,p,guide,native):
        """One source circle with fixed shape and observable position/scale."""
        feature=blank_feature(p)
        if len(guide)<5:return feature
        e=cv2.fitEllipse(np.asarray(guide,np.float32));center=np.asarray(e[0]);axes=np.asarray(e[1])
        if min(axes)<5 or max(axes)>50000:return feature
        local=dict(p,_maximum_width=35,_minimum_contrast=1.3,_maximum_sigma=14)
        exterior=p['curve_id']in self.material_outline_roles
        if exterior:local['operator']='paint_center'
        rows,diag=NativeProfiles.sample(native,local,guide,radius=48,spacing=24)
        rows,rejected=self.current_identity_readout_exclusions(rows,native,p['curve_id'])
        if rejected:diag['rejected_material_identity_readouts_before_source_curve_fit']=rejected
        if exterior:
            for row in rows:CurrentHouseReader.exterior_outline_interface(row)
        if len(rows)<3:return feature
        uv=np.asarray([r['uv'] for r in rows]);q=np.asarray([r['source_evidence']['query_uv'] for r in rows])
        n=np.asarray([r['source_evidence']['profile_normal'] for r in rows]);off=np.einsum('ij,ij->i',uv-q,n)
        design=np.c_[n,np.einsum('ij,ij->i',q-center,n)]
        fit=least_squares(lambda z:np.r_[design@z-off,z[0]/20,z[1]/20,z[2]/.02],np.zeros(3),
            bounds=([-35,-35,-.04],[35,35,.04]),loss='soft_l1',f_scale=2.5)
        ellipse=(tuple(center+fit.x[:2]),tuple(axes*(1+fit.x[2])),e[2])
        curve=(np.asarray(guide)-center)*(1+fit.x[2])+center+fit.x[:2] if p['feature_id'].startswith('goal_crease') else shared.ellipse_points(ellipse)
        rows,diag=NativeProfiles.sample(native,local,curve,radius=32,spacing=24)
        rows,rejected=self.current_identity_readout_exclusions(rows,native,p['curve_id'])
        if rejected:diag['rejected_material_identity_readouts_before_source_curve_fit']=rejected
        if exterior:
            for row in rows:CurrentHouseReader.exterior_outline_interface(row)
        if not rows:return feature
        actual=np.asarray([r['uv'] for r in rows],np.float32)
        nearest=cv2.getClosestEllipsePoints(ellipse,actual).reshape(-1,2)
        errors=np.linalg.norm(actual-nearest,axis=1)
        selected=[]
        group_id=f"native_circle_{native.frame.binding['source_pts']}_{p['curve_id']}"
        for row,representative,error in zip(rows,nearest,errors):
            if error>3:continue
            row.update(representative_uv=np.round(representative.astype(float),4).tolist(),representative_to_actual_material_px=float(error))
            selected.append(row)
        if len(selected)<3:return feature
        parts=[];part=[];last=None
        for row in selected:
            qi=row['source_evidence']['query_index']
            if part and (qi-last>1 or np.linalg.norm(np.asarray(row['uv'])-part[-1]['uv'])>75):parts.append(part);part=[]
            part.append(row);last=qi
        if part:parts.append(part)
        for part in parts:
            points=np.asarray([r['representative_uv'] for r in part]);take=simplify(points,tolerance=.7,maximum_gap=90,closed=False)
            selected_part=[part[k] for k in take]
            feature['segments'].append(dict(uv=[r['representative_uv'] for r in selected_part],vertex_count=len(take),closed=False,
                sigma_px=[r['sigma_px'] for r in selected_part],measurement_origin='same_frame_joint_single_physical_circle_source_fit',
                pixels_measured_in_this_frame=False,source_evidence_measured_in_this_frame=True,hidden_span_filled=False,
                source_evidence=[dict(r['source_evidence'],actual_native_readout_uv=r['uv'],source_evidence_group_id=group_id,
                    kind='shared_current_source_circle_representative',individual_native_pixel_readout=False)for r in selected_part]))
        feature.update(point_count=sum(s['vertex_count'] for s in feature['segments']),source_evidence_group_id=group_id,
            source_circle_fit=dict(current_frame_only=True,direct_readouts=selected,actual_native_readout_count=len(selected),
                material_readouts_are_not_model_or_temporal_pixels=True),
            geometric_interpretation=dict(kind='projected_physical_circle_ellipse',inference_only=True,center_uv=list(ellipse[0]),
                axes_px=list(ellipse[1]),angle_degrees=ellipse[2],maximum_readout_to_ellipse_px=float(max(r['representative_to_actual_material_px']for r in selected))),
            extraction_diagnostic=dict(**diag,eligible_native_material_readouts=len(selected),
                acceptance='current declared native circular material interfaces; one circle shape supplies correlated representatives'))
        return feature

    def current_identity_readout_exclusions(self,rows,native,curve_id):
        records=[r for r in self.material_identity_exclusions
                 if r['native_pts']==native.frame.binding['source_pts'] and r['curve_id']==curve_id]
        kept=[];rejected=[]
        for row in rows:
            ev=row['source_evidence'];samples=np.vstack([np.asarray(row['uv'])[None],np.asarray(ev['query_uv'])[None],
                                                         np.asarray(ev['actual_source_sides_uv'])])
            matching=[]
            for record in records:
                x0,y0,x1,y1=record['source_roi_xyxy']
                if ((samples>=[x0,y0])&(samples<[x1,y1])).all(1).any():matching.append(record)
            if matching:rejected.append(dict(native_readout=row,material_identity_exclusion_rois=matching))
            else:kept.append(row)
        return kept,rejected

    def crease(self,p,guide,native):
        local=dict(p,_circle=True)
        feature=self.outline(local,guide,native)
        feature['note_zh']='按本programme核实材料定义读取当前球门区半圆；整条半圆使用同一投影实体圆，保留本帧真实材料读数，不将画幅裁切边或闭合弦当实体边。'
        feature['material_role_authority']=p.get('material_role_authority')
        return feature

    def continuous_circle_representatives(self,feature,index,native):
        """Current visibly continuous C0 arcs share the entire frame's fit.

        A missed individual gradient profile is not an absent physical arc.
        Raw interfaces authenticate and locate one common current-source circle;
        representatives sample its independently inspected visible source span.
        They are explicitly correlated estimates, not extra pixel readouts.
        """
        review=self.continuous_arc_reviews.get((index,feature['world_curve']))
        if not review:return
        fit=feature.get('source_circle_fit');geometry=feature.get('geometric_interpretation')
        if not fit or not geometry or fit['actual_native_readout_count']<3:return
        ellipse=(tuple(geometry['center_uv']),tuple(geometry['axes_px']),float(geometry['angle_degrees']))
        curve=shared.ellipse_points(ellipse,n=4097)
        visible=np.isfinite(curve).all(1)&(curve>=[3,3]).all(1)&(curve<=[1916,1076]).all(1)
        ip=np.rint(curve[visible]).astype(int);ids=np.flatnonzero(visible)
        visible[ids]&=~native.mask[ip[:,1],ip[:,0]]
        parts=[];group=feature['source_evidence_group_id']
        actual=fit['direct_readouts'];sigma=float(np.median([r['sigma_px']for r in actual]))
        for first,last in runs(np.flatnonzero(visible)):
            if last-first<2:continue
            part=curve[first:last+1];length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(part,axis=0),axis=1))]
            if length[-1]<12:continue
            stations=np.unique(np.r_[np.arange(0,length[-1],100),length[-1]])
            uv=np.column_stack([np.interp(stations,length,part[:,axis])for axis in range(2)])
            parts.append(dict(uv=np.round(uv,4).tolist(),vertex_count=len(uv),closed=False,
                sigma_px=[max(1.,sigma)]*len(uv),measurement_origin='same_frame_common_circle_fit_over_visually_continuous_current_material_arc',
                representative_points_share_circle_family_estimate=True,pixels_measured_in_this_frame=False,
                source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=False,
                hidden_span_filled=False,source_evidence_group_id=group,
                source_evidence=[dict(kind='shared_current_source_circle_representative',source_evidence_group_id=group,
                    individual_native_pixel_readout=False,current_native_circle_interface_group_count=len(actual),
                    current_visible_material_arc_confirmed_by_independent_native_image_review=True)for _ in uv]))
        feature['segments']=parts;feature['point_count']=sum(s['vertex_count']for s in parts)
        feature['current_native_visual_continuity']=dict(programme_frame_index_1based=index,
            source_pts=native.frame.binding['source_pts'],identity=feature['world_curve'],source_visual_review_record=review,
            visible_material_arc_reviewed_on_original_and_contrast_display=True,
            individual_gradient_rejections_do_not_delete_visually_continuous_arc=True,
            current_circle_group_direct_readouts_preserved=True,person_timer_and_out_of_image_spans_excluded=True,
            user_accepted=False)

    def spots(self,index,native):
        output=[];rgb=native.frame.rgb.astype(np.float32)
        red=rgb[...,0]-(rgb[...,1]+rgb[...,2])*.5
        for record in self.source_spots.get(index,[]):
            if record['native_pts']!=native.frame.binding['source_pts']:raise ValueError('Spot clock mismatch')
            x0,y0,x1,y1=record['source_roi_xyxy'];roi=red[y0:y1,x0:x1]
            background=float(np.quantile(roi,.2));peak=float(np.quantile(roi,.98));level=background+(peak-background)*.5
            if peak-background<4:continue
            mask=(roi>=level).astype(np.uint8)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours=[c for c in contours if cv2.contourArea(c)>50]
            if not contours:continue
            contour=max(contours,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
            if len(contour)<5:continue
            visible=~native.mask[contour[:,1],contour[:,0]]
            if not visible.all():continue
            crop_contact=(contour[:,0]<=x0+1)|(contour[:,0]>=x1-2)|(contour[:,1]<=y0+1)|(contour[:,1]>=y1-2)
            parts=[(contour,True)]
            if crop_contact.any():
                if not record.get('allow_partial_source_crop'):continue
                # An image-border clipping chord is not printed material. Keep
                # only actual half-level arcs away from that image border.
                source_border=((contour[:,0]<=1)&(x0==0))|((contour[:,0]>=1918)&(x1==1920))|((contour[:,1]<=1)&(y0==0))|((contour[:,1]>=1078)&(y1==1080))
                if (crop_contact&~source_border).any():continue
                keep=~source_border
                first=int(np.flatnonzero(source_border)[0]);contour=np.roll(contour,-first,axis=0);keep=np.roll(keep,-first)
                parts=[(contour[a:b+1],False)for a,b in runs(np.flatnonzero(keep))if b-a>=2]
                if not parts:continue
            target=record['target_id'];segments=[]
            for part,closed in parts:
                take=simplify(part,tolerance=1.5,maximum_gap=50,closed=closed);points=part[take]
                evidence=[dict(kind='current_native_red_ice_material_half_level_boundary',
                    current_source_material_boundary_uv=q.tolist(),roi=record['source_roi_xyxy'],
                    current_native_red_signal_background=background,current_native_red_signal_paint=peak,
                    observed_half_level=level,model_projection_used_to_localize_boundary=False,
                    exact_model_landmark_identity_confirmed=False)for q in points]
                segments.append(dict(uv=points.astype(float).tolist(),vertex_count=len(points),closed=closed,
                    sigma_px=[3.]*len(points),source_evidence=evidence,measurement_origin='current_native_red_ice_material_half_level_boundary',
                    pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=True,
                    hidden_span_filled=False,source_image_border_not_emitted_as_material_boundary=True))
            output.append(dict(id=target,target_id=target,label_zh='冰面红色静止标识：模型身份待人工确认',
                world_curve=None,physical_surface='ice',measurement_operator='material_interface',material_signal='red',
                review_layer='unbound_ice_material',allowed_for_fit=False,model_binding_confirmed=False,
                type='filled_static_ice_material_spot',material='red_ice_paint',static=True,binding_unresolved=True,
                source_evidence=dict(native_pts=record['native_pts'],time_base=record['time_base'],
                    source_roi_xyxy=record['source_roi_xyxy'],boundary_read_from_current_native_source=True,
                    full_model_identity_not_confirmed=True),
                current_source_static_painted_material_visible=True,physical_identity_reference=record['physical_identity_reference'],
                segments=segments,point_count=sum(s['vertex_count']for s in segments),source_evidence_measured_in_this_frame=True,
                user_confirmed=False,projected_uv_used_as_observations=False,status='unbound_current_native_material_waiting_user_identity_review'))
        return output

    def source_arcs(self,index,native):
        output=[]
        for record in self.source_arc_records.get(index,[]):
            if record['native_pts']!=native.frame.binding['source_pts']:raise ValueError('Arc clock mismatch')
            vertices=np.asarray(record['rough_identity_search_uv'],float)
            guide=np.vstack([np.linspace(a,b,16,endpoint=False)for a,b in zip(vertices[:-1],vertices[1:])]+[vertices[-1:]])
            path=dict(feature_id='unbound_red_arc',operator='paint_center',material_signal='red',
                      material_transition='dark_to_light',_circle=False,_minimum_contrast=1.3,_maximum_width=35,_maximum_sigma=14)
            rows,diagnostic=NativeProfiles.sample(native,path,guide,radius=18,spacing=35)
            if not rows:
                query,normal,interval=_source_queries(path,guide,np.ones(len(guide)),(1080,1920),35,25)
                offsets,profiles,valid=sample_chroma(native.frame.rgb,query,normal,interval,'red')
                for qi in range(len(query)):
                    good=valid[qi];off=offsets[qi,good];values=profiles[qi,good]
                    if len(values)<15:continue
                    sm=cv2.GaussianBlur(values.astype(np.float32)[:,None],(1,5),1).ravel()
                    centre=np.abs(off)<15
                    if not centre.any():continue
                    peak=int(np.flatnonzero(centre)[np.argmax(sm[centre])]);background=float(np.quantile(sm,.2))
                    amplitude=float(sm[peak]-background)
                    if amplitude<3:continue
                    level=background+.5*amplitude;left=peak;right=peak
                    while left>0 and sm[left]>=level:left-=1
                    while right<len(sm)-1 and sm[right]>=level:right+=1
                    if left==0 or right==len(sm)-1 or not 2<=off[right]-off[left]<=35:continue
                    crossings=[]
                    for a,b in ((left,left+1),(right-1,right)):
                        crossings.append(float(off[a]+(level-sm[a])/(sm[b]-sm[a])*(off[b]-off[a])))
                    sides=query[qi]+np.asarray(crossings)[:,None]*normal[qi];uv=sides.mean(0)
                    context=query[qi]+off[left:right+1,None]*normal[qi];xy=np.rint(context).astype(int)
                    if native.mask[xy[:,1],xy[:,0]].any():continue
                    rows.append(dict(uv=np.round(uv,4).tolist(),sigma_px=3.,source_evidence=dict(query_index=qi,
                        current_source_query_uv=query[qi].tolist(),actual_source_sides_uv=sides.tolist(),
                        native_red_paint_signal_background=background,native_red_paint_signal_peak=float(sm[peak]),
                        observed_half_level=level,measurement_origin='current_native_red_paint_two_ice_shoulder_half_level_crossings',
                        both_complete_flanks_visible_in_current_source=True)))
                diagnostic.update(current_two_ice_shoulder_half_level_readout_count=len(rows))
            if not rows:continue
            parts=[];part=[];last=None
            for row in rows:
                qi=row['source_evidence']['query_index']
                if part and qi-last>1:parts.append(part);part=[]
                part.append(row);last=qi
            if part:parts.append(part)
            target=record['target_id'];segments=[]
            for part in parts:
                segments.append(dict(uv=[r['uv']for r in part],vertex_count=len(part),closed=False,
                    sigma_px=[r['sigma_px']for r in part],source_evidence=[r['source_evidence']for r in part],
                    pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,hidden_span_filled=False,
                    measurement_origin='current_native_red_paint_two_interface_midpoint_identity_unbound'))
            output.append(dict(id=target,target_id=target,label_zh=record['label_zh'],world_curve=None,
                physical_surface='ice',type='visible_static_ice_material_arc',material='red_ice_paint',static=True,
                allowed_for_fit=False,binding_unresolved=True,model_binding_confirmed=False,
                physical_identity_reference=record['physical_identity_reference'],review_layer='unbound_ice_material',
                source_evidence=dict(native_pts=record['native_pts'],time_base=record['time_base'],
                    source_roi_xyxy=record['source_roi_xyxy'],current_native_material_readouts=rows,
                    rough_search_coordinates_are_not_observations=True,model_projection_supplies_no_observed_pixels=True),
                segments=segments,point_count=sum(s['vertex_count']for s in segments),extraction_diagnostic=diagnostic,
                source_evidence_measured_in_this_frame=True,user_confirmed=False,status='source_material_read_identity_binding_pending'))
        return output

    def source_print_materials(self,index,native):
        """Read current static ink boundaries, including visible internal holes.

        A programme-confirmed ROI supplies material ownership only. Native
        source intensity supplies every boundary point; cropped or occluded
        contours remain open and never acquire an ROI/mask closing chord.
        """
        output=[];rgb=native.frame.rgb.astype(np.float32)
        luma=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
        base_safety=cv2.dilate(native.mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
        for record in self.source_review.get('static_print_material_rois',[]):
            if record['programme_frame_index_1based']!=index:continue
            if record['native_pts']!=native.frame.binding['source_pts'] or record['time_base']!='1/12800':
                raise ValueError('Static print source clock mismatch')
            if not record.get('current_source_static_material_confirmed'):raise ValueError('Static print ownership unconfirmed')
            x0,y0,x1,y1=record['source_roi_xyxy']
            if not 0<=x0<x1<=1920 or not 0<=y0<y1<=1080:raise ValueError('Static print source ROI invalid')
            # A wrong-material ROI cannot contribute to the ink/ice level
            # estimate either. Filtering only final contour points would keep
            # the rejected ring/glint pixels' indirect effect on all edges.
            safety=base_safety.copy()
            exclusions=record.get('material_identity_exclusion_rois',[])
            for exclusion in exclusions:
                a,b,c,d=exclusion['source_roi_xyxy'];safety[max(0,b-2):min(1080,d+2),max(0,a-2):min(1920,c+2)]=True
            roi=luma[y0:y1,x0:x1];available=~safety[y0:y1,x0:x1]
            if available.sum()<50:continue
            background=float(np.quantile(roi[available],.85));paint=float(np.quantile(roi[available],.06))
            contrast=background-paint
            if contrast<record.get('minimum_current_contrast',4.):continue
            field=background-roi;level=contrast*.5
            contours,hierarchy=cv2.findContours(((field>=level)&available).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            gx=cv2.Sobel(field,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(field,cv2.CV_32F,0,1,ksize=3)/8
            segments=[]
            for ci,contour in enumerate(contours):
                if cv2.contourArea(contour)<record.get('minimum_contour_area_px',12.) or cv2.arcLength(contour,True)<12:continue
                local=contour.reshape(-1,2);points=local.astype(float)+[x0,y0]
                gradients=np.c_[gx[local[:,1],local[:,0]],gy[local[:,1],local[:,0]]]
                squared=np.einsum('ij,ij->i',gradients,gradients)
                shift=(level-field[local[:,1],local[:,0]])[:,None]*gradients/np.maximum(squared[:,None],1e-8)
                length=np.linalg.norm(shift,axis=1);shift*=np.minimum(1.,1.5/np.maximum(length,1e-8))[:,None]
                points+=shift;xy=np.rint(points).astype(int)
                sigma=contrast*.5/np.maximum(np.sqrt(squared),1e-8)/1.349
                keep=(local[:,0]>1)&(local[:,0]<x1-x0-2)&(local[:,1]>1)&(local[:,1]<y1-y0-2)
                safe_xy=np.clip(xy,[0,0],[1919,1079])
                keep&=(sigma<=14)&~safety[safe_xy[:,1],safe_xy[:,0]]
                if record.get('require_blue_or_neutral_ink'):
                    color=rgb[safe_xy[:,1],safe_xy[:,0]]
                    keep&=(color[:,0]-color[:,2]<=3.)
                depth=0;parent=int(hierarchy[0,ci,3])
                while parent>=0:depth+=1;parent=int(hierarchy[0,parent,3])
                role='outer_ink_to_ice' if depth%2==0 else 'inner_ice_hole_to_ink'
                parts=[(np.arange(len(points)),True)] if keep.all() else []
                if not keep.all() and keep.any():
                    start=int(np.flatnonzero(~keep)[0]);ids=np.roll(np.arange(len(points)),-start);valid=keep[ids]
                    parts=[(ids[a:b+1],False)for a,b in runs(np.flatnonzero(valid))if b-a>=2]
                for ids,closed in parts:
                    part=points[ids];take=simplify(part,tolerance=1.5,maximum_gap=50,closed=closed);selected=ids[take]
                    if len(selected)<(3 if closed else 2):continue
                    uv=np.round(points[selected],4).tolist();sigmas=np.maximum(1.,sigma[selected]).tolist()
                    evidence=[dict(kind='current_native_static_ink_material_half_level_interface',
                        current_source_material_boundary_uv=q,boundary_role=role,
                        current_native_luminance_ice_level=background,current_native_luminance_ink_level=paint,
                        observed_luminance_half_level=background-level,transition_sigma_px=unc,
                        current_source_gradient_xy=gradients[k].tolist(),source_roi_is_identity_search_only=True,
                        reference_pixels_not_copied=True)for q,unc,k in zip(uv,sigmas,selected)]
                    segments.append(dict(uv=uv,vertex_count=len(uv),closed=closed,boundary_role=role,
                        sigma_px=sigmas,source_evidence=evidence,
                        measurement_origin='current_native_static_ink_material_half_level_interface',
                        pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,
                        individual_native_pixel_readouts=True,hidden_span_filled=False,
                        crop_and_mask_edges_not_emitted_as_material_boundaries=True))
            if not segments:continue
            target=record['target_id']
            output.append(dict(id=target,target_id=target,label_zh=record['label_zh'],physical_surface='ice',world_curve=None,
                measurement_operator='material_interface',material_signal='dark',material='grey_blue_static_ice_ink',
                static=True,type='visible_static_ice_print_material_boundaries',allowed_for_fit=False,binding_unresolved=True,
                model_binding_confirmed=False,review_layer='unbound_ice_material',physical_identity_reference=record['physical_identity_reference'],
                source_evidence=dict(native_pts=record['native_pts'],time_base=record['time_base'],source_roi_xyxy=record['source_roi_xyxy'],
                    current_source_static_material_confirmed=True,temporal_evidence_supplies_identity_only=True,
                    semantic_letter_identity_not_confirmed=True,boundary_pixels_from_current_native_source=True,
                    material_identity_exclusions_applied_before_source_levels_and_contours=True,
                    excluded_source_material_regions=deepcopy(exclusions)),
                segments=segments,point_count=sum(s['vertex_count']for s in segments),user_confirmed=False,
                projected_uv_used_as_observations=False,status='unbound_current_native_material_waiting_user_identity_review'))
        return output

    def source_single_interfaces(self,index,native):
        output=[]
        for record in self.source_review.get('static_single_interface_guides',[]):
            if record['programme_frame_index_1based']!=index:continue
            if record['native_pts']!=native.frame.binding['source_pts']:raise ValueError('Single-interface source clock mismatch')
            guide=np.linspace(*np.asarray(record['rough_identity_search_uv'],float),129)
            path=dict(feature_id='unbound_single_ice_interface',operator='material_interface',_circle=False,
                material_signal=record['material_signal'],material_transition=record['material_transition'],
                _minimum_contrast=record.get('minimum_current_material_contrast',3),_maximum_sigma=14)
            rows,diagnostic=NativeProfiles.sample(native,path,guide,radius=record.get('search_radius_px',28),spacing=24)
            parts=[];part=[];last=None
            for row in rows:
                qi=row['source_evidence']['query_index']
                if part and(qi-last>1 or np.linalg.norm(np.asarray(row['uv'])-part[-1]['uv'])>75):parts.append(part);part=[]
                part.append(row);last=qi
            if part:parts.append(part)
            if not parts:continue
            segments=[dict(uv=[r['uv']for r in part],vertex_count=len(part),closed=False,
                sigma_px=[r['sigma_px']for r in part],source_evidence=[r['source_evidence']for r in part],
                pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,
                individual_native_pixel_readouts=True,hidden_span_filled=False)for part in parts]
            target=record['target_id']
            output.append(dict(id=target,target_id=target,label_zh=record['label_zh'],physical_surface='ice',world_curve=None,
                material=record['material_signal'],static=True,type='visible_static_ice_single_material_interface',
                allowed_for_fit=False,binding_unresolved=True,model_binding_confirmed=False,review_layer='unbound_ice_material',
                physical_identity_reference=record['physical_identity_reference'],source_evidence=dict(native_pts=record['native_pts'],
                    time_base=record['time_base'],second_material_flank_not_supplied_by_model=True,search_guide_is_not_observation=True),
                segments=segments,point_count=sum(s['vertex_count']for s in segments),extraction_diagnostic=diagnostic,user_confirmed=False))
        return output

    def outline_inner_materials(self,features,source):
        output=[]
        for feature in features:
            role=self.material_outline_roles.get(feature['world_curve'])
            if not role or not role.get('retain_visible_inner_boundary_as_unbound'):continue
            rows=feature.get('source_circle_fit',{}).get('direct_readouts',[])
            rows=[r for r in rows if 'actual_outline_inner_outer_uv'in r['source_evidence']]
            parts=[];part=[];last=None
            for row in rows:
                uv=np.asarray(row['source_evidence']['actual_outline_inner_outer_uv'][0]);qi=row.get('query_index',row['source_evidence']['query_index'])
                if part and(qi-last>1 or np.linalg.norm(uv-part[-1][0])>120):parts.append(part);part=[]
                part.append((uv,row));last=qi
            if part:parts.append(part)
            segments=[]
            for part in parts:
                uv=np.asarray([v[0]for v in part]);take=simplify(uv,tolerance=1.5,maximum_gap=120,closed=False)
                chosen=[part[k]for k in take]
                evidence=[];sigmas=[]
                for q,r in chosen:
                    ev=deepcopy(r['source_evidence']);quantile=ev['actual_outline_inner_outer_quantiles_uv'][0]
                    sigma=max(1.,float(np.linalg.norm(np.asarray(quantile)[2]-quantile[0])/1.349))
                    ev.update(current_source_inner_material_boundary_uv=q.tolist(),outer_model_circle_curve=feature['world_curve'],
                        inner_circle_model_geometry_not_defined=True,actual_source_sides_uv=[q.tolist()]*2,
                        actual_source_side_quantiles_uv=[quantile]*2,transition_sigma_px=sigma,
                        measurement_origin='current_native_unfilled_circle_inside_ice_to_paint_interface')
                    evidence.append(ev);sigmas.append(sigma)
                segments.append(dict(uv=np.round(uv[take],4).tolist(),vertex_count=len(take),closed=False,
                    sigma_px=sigmas,source_evidence=evidence,
                    measurement_origin='current_native_unfilled_circle_inside_ice_to_paint_interface',
                    pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,
                    individual_native_pixel_readouts=True,hidden_span_filled=False))
            if not segments:continue
            target='unbound.ice.'+feature['world_curve'].split(':')[0]+'.inner_material_boundary'
            output.append(dict(id=target,target_id=target,label_zh=feature['world_curve'].split(':')[0]+'圈漆内侧实体边界：模型未定义',
                physical_surface='ice',world_curve=None,material=feature['material_signal'],static=True,
                type='visible_static_ice_circle_inner_material_boundary',allowed_for_fit=False,
                binding_unresolved=True,model_binding_confirmed=False,review_layer='unbound_ice_material',
                physical_identity_reference=feature['world_curve']+'对应同一圈漆材料的当前源内侧边界',
                source_evidence=dict(native_pts=source['native_pts'],time_base=source['time_base'],
                    boundary_read_from_current_native_source=True,registered_outer_boundary_is_not_a_width_midline=True),
                segments=segments,point_count=sum(s['vertex_count']for s in segments),user_confirmed=False))
        return output

    def extract(self,index):
        started=time.perf_counter();row=self.rows[index];source=deepcopy(row['source'])
        raw_path=Path(row.get('image_path',source['image_path']));raw=cv2.imread(str(raw_path))
        if raw is None or raw.shape!=(1080,1920,3):raise ValueError('Native PNG unavailable')
        if source.get('png_sha256') and sha(raw_path)!=source['png_sha256']:raise ValueError('Native PNG file changed')
        if hashlib.sha256(raw.tobytes()).hexdigest()!=source['decoded_BGR_sha256']:raise ValueError('Native PNG changed')
        mask_path=Path(row.get('mask_path') or source.get('mask_path') or self.output_root/f'masks_current/{index:06d}.png');person=cv2.imread(str(mask_path),0)
        if person is None:raise ValueError('Current person mask unavailable')
        mask_sha256=sha(mask_path)
        if source.get('mask_sha256') and mask_sha256!=source['mask_sha256']:raise ValueError('Mask changed')
        if row.get('mask_sha256') and mask_sha256!=row['mask_sha256']:raise ValueError('Manifest mask changed')
        binding_reference=deepcopy(row.get('source_binding'))
        if binding_reference:
            binding_path=Path(binding_reference['path'])
            if sha(binding_path)!=binding_reference['sha256']:raise ValueError('Exact source binding changed')
            binding=read(binding_path)
            for key in ('native_pts','time_base','programme_frame_index_1based',
                        'source_video_sha256','decoded_BGR_sha256','png_sha256'):
                if binding.get(key)!=source.get(key):raise ValueError(f'Exact source binding {key} mismatch')
            if binding.get('mask_sha256')!=mask_sha256:raise ValueError('Exact source binding current mask mismatch')
            if Path(binding['mask_path']).resolve()!=mask_path.resolve():raise ValueError('Exact source binding mask path mismatch')
            source['source_binding']=deepcopy(binding_reference)
        broadcast,bcast=current_broadcast_exclusion(raw);outside,domain=current_ice_domain(self.output_root,index)
        static_exclusions = source_records(source, self.source_review.get('static_material_exclusion_rois', []))
        mask=(person>0)|broadcast|outside|exclusion_mask(static_exclusions)
        native=SourceNormalProfiles(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),source,mask,self.cameras[index])
        native.static_material_exclusions = static_exclusions
        native.source_family_full_footprint_admission = bool(self.source_review.get('source_family_full_footprint_admission'))
        native.source_family_admission_history = []
        if native.source_family_full_footprint_admission:
            # Historical cache keys did not bind the complete source footprint.
            self.use_circle_cache = False
        if static_exclusions:
            source['static_material_exclusion_rois'] = deepcopy(static_exclusions)
            source['original_person_mask_unchanged'] = True
        nested,counts=self.nested_house_features(index,native);features=[];coverage=[]
        for p in self.paths:
            guide,origin=self.guide(p,index)
            if p['_circle']:
                feature=nested.get(p['curve_id'],blank_feature(p))
                if p['feature_id'].startswith('A'):
                    feature=self.outline(p,guide,native)
            elif p['feature_id'].startswith('goal_crease'):
                feature=self.crease(p,guide,native)
            elif len(guide)>=2:
                feature,_,_=self.line(p,guide,native)
            else:feature=blank_feature(p)
            if p['_circle'] and not p['feature_id'].startswith('A'):
                feature['identity_navigation_origin']=origin
                feature['navigation_origin']='8a28_same_circle_family_initialization_then_current_native_joint_material_reread'
            else:
                feature['navigation_origin']=origin
            self.continuous_circle_representatives(feature,index,native)
            feature['review_layer']='ice_features'
            feature['allowed_for_fit']=False
            feature['independent_current_source_reread']=True
            exclusions=[r for r in self.material_identity_exclusions if r['programme_frame_index_1based']==index
                        and r['curve_id']==p['curve_id']]
            if exclusions:
                feature['material_identity_exclusions_before_source_family_fit']=deepcopy(exclusions)
                feature.get('source_circle_fit',{}).update(material_identity_exclusions_applied_before_fit=True,
                    excluded_source_regions=deepcopy(exclusions))
            if feature['point_count']:features.append(feature)
            coverage.append(dict(target_id=feature['target_id'],points=feature['point_count'],
                current_material_diagnostic=feature.get('extraction_diagnostic'),
                missing_reason=None if feature['point_count'] else 'current_selected_source_material_not_yet_read'))
        source.update(mask_path=str(mask_path),mask_sha256=mask_sha256,
                      broadcast_timer_exclusion=bcast,ice_visible_domain=domain)
        unbound=(self.spots(index,native)+self.source_arcs(index,native)+self.source_print_materials(index,native)
                 +self.outline_inner_materials(features,source)+self.source_single_interfaces(index,native))
        for feature in unbound:
            for si,segment in enumerate(feature['segments']):
                segment['point_ids']=[hashlib.sha256(
                    f"{source['source_video_sha256']}:{source['native_pts']}:{feature['target_id']}:{si}:{pi}:{uv}".encode('utf-8')
                    ).hexdigest()[:20]for pi,uv in enumerate(segment['uv'])]
        packet=dict(schema='rinkspace.single-frame-source-ice-features.v1',source=source,
            programme_frame_index_1based=index,status='independent_reread_pending_visual_review',
            user_accepted=False,features=features,unbound_static_materials=unbound,inferred_segments=[],
            point_count=sum(f['point_count'] for f in features),per_feature_coverage=coverage,
            unbound_point_count=sum(f['point_count']for f in unbound),
            all_current_source_point_count=sum(f['point_count']for f in features+unbound),
            unresolved_source_regions=deepcopy(self.source_regions.get(index,[])),
            model_scene_sha256=sha(self.model_root/'scene.json'),
            circle_geometry_sha256=self.circle_geometry_sha256,optimization_performed=False,
            old_review_exclusions_reused=False,selected_frames_only=True,
            intermediate_exposures_source_points_extracted=False,
            observation_coordinates_from_projection=False,elapsed_seconds=time.perf_counter()-started)
        if native.source_family_full_footprint_admission:
            packet['source_family_support_admission'] = dict(
                algorithm='original_full_profile_tangent7_cv2_bilinear32_and_current_gaussian_r3_before_family_fit',
                helper_sha256=sha(Path(__file__).with_name('static_ice_source_exclusions_v2.py')),
                every_estimation_iteration_guarded=True,failed_readouts_preserved=True,
                history=native.source_family_admission_history)
        if binding_reference:
            packet['source_binding']=binding_reference
            packet['current_exact_source_binding_and_mask_verified']=True
        write(self.output_root/f'ice_candidates/{index:06d}.json',packet)
        self.render(index,raw,features+unbound)
        print(json.dumps(dict(index=index,points=packet['point_count'],targets=[f['target_id'] for f in features]),ensure_ascii=False),flush=True)
        return packet

    def render(self,index,raw,features):
        marked=raw.copy();flanks=raw.copy()
        for f in features:
            color=(230,60,180) if 'source_circle_fit' in f else (50,220,50)
            for s in f['segments']:
                points=np.rint(s['uv']).astype(int).reshape(-1,2)
                geometry=f.get('geometric_interpretation',{})
                if geometry.get('kind')=='projected_physical_circle_ellipse' and len(points)>=2:
                    center=np.asarray(geometry['center_uv']);axes=np.asarray(geometry['axes_px'])/2
                    angle=np.deg2rad(geometry['angle_degrees']);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    local=(np.asarray(s['uv'])-center)@rotation/axes
                    theta=np.unwrap(np.arctan2(local[:,1],local[:,0]))
                    dense=[]
                    for a,b in zip(theta[:-1],theta[1:]):
                        stations=np.linspace(a,b,max(3,int(abs(b-a)*max(axes)/4)+1))
                        dense.extend((np.c_[np.cos(stations),np.sin(stations)]*axes@rotation.T+center).tolist())
                    cv2.polylines(marked,[np.rint(dense).astype(int)],False,color,1)
                else:cv2.polylines(marked,[points],False,color,1)
                for uv in points:cv2.circle(marked,tuple(uv),2,color,-1)
                if len(points):cv2.putText(marked,f.get('world_curve') or f['id'],tuple(points[len(points)//2]),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
            for r in f.get('source_line_fit',f.get('source_circle_fit',{})).get('direct_readouts',[]):
                sides=np.rint(r['source_evidence']['actual_source_sides_uv']).astype(int)
                cv2.line(flanks,tuple(sides[0]),tuple(sides[1]),(0,220,220),1)
                for uv in sides:cv2.circle(flanks,tuple(uv),1,(0,0,255),-1)
        folder=self.output_root/'ice_overlays';folder.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(folder/f'{index:06d}.png'),marked)
        folder=self.output_root/'ice_diagnostics/current_flanks';folder.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(folder/f'{index:06d}.png'),flanks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--navigation',type=Path)
    parser.add_argument('--source-guides',type=Path)
    parser.add_argument('--indices')
    parser.add_argument('--shot-index',type=int)
    args=parser.parse_args();cv2.setNumThreads(1)
    extractor=IceReread(args.data_root,args.model_root,args.navigation,args.source_guides,args.shot_index)
    indices=[int(x) for x in args.indices.split(',')] if args.indices else sorted(extractor.rows)
    for index in indices:extractor.extract(index)


if __name__=='__main__':main()
