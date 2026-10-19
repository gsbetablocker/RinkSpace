"""Read ice materials only at the selected native official50 shot-2 exposures.

Reuse the repaired native line, physical width and joint circle mathematics.
The reference scene/cameras and image registration navigate physical identity;
all admitted material evidence is measured in the current selected PNG.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from . import extract_first_shot_ice as shared
from .extract_first_shot_ice import (
    Extractor, ROOT, read, write, sha, circle_geometry_digest, image_line,
    ice_curve_inventory, project_fast, runs, blank_feature, ground_xy,
)
from .repair_physical_wide_bands import PhysicalBandProfiles
from .repair_physical_B1_visibility import LocalPhysicalBlue
from .native_ice_material_repair import NestedHouseReader
from .native_ice_line_continuity import straight_guide, NativeLineIdentity
from .audit_first_shot_circle_material_identity import conflicts, apply_rejections
from rinkspace.vision_source_shapes import _MaterialProfileFrame
import cv2
import numpy as np

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014')
MODEL = ROOT/'events/ankara_2026/models/shot2_8a28_20261014'
REVIEW = ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot2_review_20261014/ice_material_review.json'


CONTACT_DOMAIN_INPUT_DIGEST_SCHEMA='rinkspace.current-source-contact-domain-input.v1'


def contact_domain_input_payload(packet):
    """The exact current contact coordinates used to define the ice domain.

    Printed detail, unused representatives and unrelated packet metadata do
    not affect this dependency. A segment with no horizontal extent cannot
    contribute to the interpolated domain and is not part of the payload.
    """
    parts=[]
    for feature in packet['features']:
        target=feature.get('target_id','')
        if not target.startswith('board.contact'):continue
        for index,segment in enumerate(feature['segments']):
            field='source_native_readout_uv' if 'source_native_readout_uv'in segment else 'uv'
            uv=np.asarray(segment[field],dtype=float)
            if len(uv)<2:continue
            if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():
                raise ValueError('Invalid current source contact coordinates')
            if len(np.unique(uv[:,0]))<2:continue
            parts.append(dict(target_id=target,segment_index=index,
                              readout_coordinate_field=field,uv=uv.tolist()))
    parts.sort(key=lambda part:(part['target_id'],part['segment_index']))
    source=packet['source']
    return dict(schema=CONTACT_DOMAIN_INPUT_DIGEST_SCHEMA,
                native_pts=int(source['native_pts']),time_base=source['time_base'],contact_parts=parts)


def contact_domain_input_sha256(packet):
    canonical=json.dumps(contact_domain_input_payload(packet),ensure_ascii=False,
                         sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def current_ice_domain(data_root, index):
    """Conservatively exclude everything above currently measured contact."""
    path=data_root/f'board_candidates/{index:06d}.json';packet=read(path)
    contact=[f for f in packet['features'] if f.get('target_id','').startswith('board.contact')]
    border=np.full(1920,np.nan);axis=np.arange(1920);parts=[];all_points=[]
    for f in contact:
        for si,s in enumerate(f['segments']):
            uv=np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            if len(uv)<2:continue
            all_points.extend(uv.tolist());uv=uv[np.argsort(uv[:,0])]
            xx,inv=np.unique(uv[:,0],return_inverse=True)
            yy=np.array([uv[inv==j,1].max() for j in range(len(xx))])
            if len(xx)<2:continue
            visible=(axis>=xx[0])&(axis<=xx[-1])
            border[visible]=np.fmax(border[visible],np.interp(axis[visible],xx,yy))
            parts.append(dict(target_id=f['target_id'],segment_index=si,
                              source_x_interval=[float(xx[0]),float(xx[-1])]))
    supported=np.isfinite(border)
    if supported.sum()<50:raise ValueError(f'Current contact outline insufficient: {index}')
    filled=np.interp(axis,np.flatnonzero(supported),border[supported]);gaps=[]
    for first,last in runs(np.flatnonzero(~supported)):
        flanks=[border[q] for q in (first-1,last+1) if 0<=q<1920 and supported[q]]
        if flanks:filled[first:last+1]=max(flanks)
        gaps.append([int(first),int(last)])
    mask=np.arange(1080)[:,None]<=filled[None]
    return mask,dict(method='current_native_contact_parts_conservative_gap_exclusion',
        board_contact_candidate_path=str(path),board_contact_candidate_sha256=sha(path),
        board_contact_input_sha256=contact_domain_input_sha256(packet),
        board_contact_input_digest_schema=CONTACT_DOMAIN_INPUT_DIGEST_SCHEMA,
        native_source_pts=packet['source']['native_pts'],source_contact_parts=parts,
        current_source_contact_readout_count=len(all_points),
        unsupported_x_intervals_used_only_as_conservative_exclusion=gaps,
        model_projection_used_for_domain=False,
        exclusion_gap_coordinates_emitted_as_observations=False)


class ShotProfiles(PhysicalBandProfiles):
    def __init__(self,rgb,source,mask,camera):
        super().__init__(rgb,source,mask,camera)
        self.frame=_MaterialProfileFrame(rgb,dict(source_sha256=source['source_video_sha256'],
            source_pts=source['native_pts'],time_base=source['time_base'],native_hw=[1080,1920],
            shot_id=int(source.get('shot_index',source.get('shot_id',2)))))

    def sample(self,path,guide,radius=28,spacing=48):
        local=dict(path)
        if not path.get('_circle') and path['material_signal']=='dark' and path['feature_id'] not in ('B1','B2','C'):
            # Native review of this view found shallow grey haze and crossing
            # ring/red-band responses being mistaken for a continuous stripe.
            # Read the stripe's own two flanks, then verify its support densely.
            local['_minimum_contrast']=max(4.,local.get('_minimum_contrast',0.))
            local['_maximum_width']=min(30.,local.get('_maximum_width',30.))
            if path.get('_dense_current_source_verification'):
                radius=12;spacing=20
        rows,diagnostic=super().sample(local,guide,radius=radius,spacing=spacing)
        return rows,diagnostic


class SelectedHouseReader(NestedHouseReader):
    def __init__(self,scene,paths,cameras):
        # The repaired parent constructor has first-shot index/guide assumptions.
        # Rebuild only its identity inventory; inherit every material/fit method.
        self.project=project_fast;self.cameras=cameras;self.groups={};self.state={}
        for feature in scene['ice_features']:
            if feature.get('kind')!='circle':continue
            fid=feature['id'];group_paths=[p for p in paths if p['feature_id']==fid]
            if not group_paths:continue
            definitions=feature.get('rings') or [dict(role='outer',center_xyz=feature['center_xyz'],radius=feature['radius'])]
            outer=next(r for r in definitions if r['role']=='outer');best=None
            for index,camera in cameras.items():
                uv,depth=project_fast(camera,np.asarray(group_paths[0]['xyz']))
                visible=np.isfinite(uv).all(1)&(uv>=[-75,-75]).all(1)&(uv<=[1994,1154]).all(1)&(depth>.05)
                score=int(visible.sum())
                if best is None or score>best[0]:best=(score,index)
            if best is None or best[0]<2:continue
            self.groups[fid]=dict(center=np.asarray(outer['center_xyz'],float)[:2],
                radii={r['role']:float(r['radius']) for r in definitions if r['role']!='center'
                       or any(p['world_curve_role']=='center'for p in group_paths)},
                anchor=best[1],authority='8a28_named_physical_circle_current_selected_camera_navigation',
                paths=group_paths,outline=not bool(feature.get('rings')))

    def read(self,fid,index,native):
        group=self.groups[fid]
        uv,depth=self.projection(group,index,'outer',np.linspace(0,2*np.pi,513),None)
        visible=np.isfinite(uv).all(1)&(uv>=[-75,-75]).all(1)&(uv<=[1994,1154]).all(1)&(depth>.05)
        if visible.sum()<2:return {},dict(outer=dict(query_count=0,eligible_native_material_readouts=0))
        return super().read(fid,index,native)


class SelectedLineIdentity(NativeLineIdentity):
    """Register only consecutive selected frames, within measured ice support."""
    def __init__(self,output_root):
        self.states={};self.guides=[];self.last_policy={};self.pairs={}
        self.motion_quality_path=output_root/'temporal_diagnostics/selected_native_motion.json'
        self.previous=None;self.previous_index=None;self.registration=[]

    def observe(self,index,gray,mask,source):
        sift=cv2.SIFT_create(nfeatures=5000)
        points,descriptors=sift.detectAndCompute(gray,(~mask).astype(np.uint8)*255)
        if self.previous is not None:
            old_points,old_desc,old_source=self.previous;record=dict(index=index,
                reference_frame=self.previous_index,native_pts=source['native_pts'],
                reference_native_pts=old_source['native_pts'],usable=False,
                source_registration_points_are_navigation_only=True,
                intermediate_exposures_read_for_points=False)
            if descriptors is not None and old_desc is not None:
                matches=cv2.BFMatcher().knnMatch(old_desc,descriptors,k=2)
                good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
                record['ratio_match_count']=len(good)
                if len(good)>=8:
                    a=np.asarray([old_points[m.queryIdx].pt for m in good],np.float32)
                    b=np.asarray([points[m.trainIdx].pt for m in good],np.float32)
                    h,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.5)
                    if h is not None and inliers is not None and int(inliers.sum())>=8:
                        keep=inliers.ravel()>0;res=np.linalg.norm(cv2.perspectiveTransform(a[None],h)[0]-b,axis=1)[keep]
                        hull=cv2.convexHull(a[keep]).reshape(-1,2)
                        spread=np.ptp(a[keep],axis=0);p95=float(np.quantile(res,.95))
                        record.update(homography=h.tolist(),inlier_count=int(keep.sum()),p95_reprojection_px=p95,
                            inlier_convex_hull_uv=hull.tolist(),ice_support_span_px=spread.tolist(),
                            usable=bool(p95<=6 and spread[0]>=120 and spread[1]>=60))
            self.pairs[(self.previous_index,index)]=record;self.registration.append(record)
        self.previous=(points,descriptors,deepcopy(source));self.previous_index=index

    def _transport(self,a,b,uv):
        record=self.pairs.get((a,b))
        if not record or not record['usable']:return None
        uv=np.asarray(uv,float);hull=np.asarray(record['inlier_convex_hull_uv'],np.float32)
        within=np.asarray([cv2.pointPolygonTest(hull,tuple(map(float,p)),True)>=-2 for p in uv])
        if within.sum()<2 or np.linalg.norm(np.ptp(uv[within],axis=0))<35:return None
        h=np.asarray(record['homography'],float)
        moved=cv2.perspectiveTransform(uv[within][None],h)[0]
        current_hull=cv2.perspectiveTransform(hull[None],h)[0]
        if not np.isfinite(moved).all():return None
        error=record['p95_reprojection_px']
        return moved,error,error,current_hull


class SelectedExtractor(Extractor):
    def __init__(self,data_root,model_root,review_path=None,shot_index=None):
        self.output_root=data_root;self.model_root=model_root
        self.scene=read(model_root/'scene.json');self.paths=ice_curve_inventory(self.scene)
        self.circle_geometry_sha256=circle_geometry_digest(self.scene)
        circles={f['id'] for f in self.scene['ice_features'] if f.get('kind')=='circle'}
        for p in self.paths:p['_circle']=p['feature_id'] in circles
        self.paths=[p for p in self.paths if p['world_curve_role']!='center' and p['operator']!='unresolved']
        manifest=read(data_root/'frame_manifest.json');self.rows={r['index']:r for r in manifest['frames']}
        with np.load(data_root/'camera_seeds.npz',allow_pickle=False) as saved:
            self.cameras={int(i):c.copy() for i,c in zip(saved['programme_indices'],saved['camera25'])}
        if set(self.cameras)!=set(self.rows):raise ValueError('Selected source/camera indices differ')
        scope=manifest['shot_scope']
        self.shot_index=int(scope['shot_index'])
        if shot_index is not None and self.shot_index!=int(shot_index):
            raise ValueError('Selected source manifest shot differs from requested shot')
        if self.shot_index<1:raise ValueError('Invalid selected source shot index')
        for row in self.rows.values():
            source=row['source']
            if source['time_base']!='1/12800' or not scope['first_pts']<=source['native_pts']<scope['stop_pts_exclusive']:
                raise ValueError('Selected source clock is outside its official50 shot')
            if int(source.get('shot_index',self.shot_index))!=self.shot_index:
                raise ValueError('Selected source row belongs to another shot')
            source['shot_id']=self.shot_index
        self.guides=[];self.previous={};self.previous_index=None;self.hcache={}
        self.query_qualification=None;self.use_circle_cache=False
        self.house_reader=SelectedHouseReader(self.scene,self.paths,self.cameras)
        self.line_identity=SelectedLineIdentity(data_root)
        self.review_path=review_path or REVIEW
        self.material_reviews={r['source_pts']:r for r in read(self.review_path).get('records',[])} if self.review_path.exists() else {}

    def guide(self,p,index):
        # Selected exposures are non-consecutive. Transport the last actually
        # measured same-shot shape through the undistorted ice plane, then
        # reread its material in this exposure. No intervening frame is sampled.
        if self.previous_index is not None and p['curve_id'] in self.previous:
            uv=self.previous[p['curve_id']]
            xy,valid=ground_xy(self.cameras[self.previous_index],np.asarray(uv,float))
            if valid.sum()>=2:
                xy=xy[valid]
                if not p['_circle']:
                    center=xy.mean(0);direction=np.linalg.svd(xy-center,full_matrices=False)[2][0]
                    extent=(np.asarray(p['xyz'])[:,:2]-center)@direction
                    xy=center+np.linspace(extent.min(),extent.max(),257)[:,None]*direction
                guide=dict(frame=self.previous_index,curve_id=p['curve_id'],
                    authority='previous_selected_current_material_distortion_aware_navigation',
                    _world_xyz=np.c_[xy,np.zeros(len(xy))])
                self.guides.append(guide)
                try:return super().guide(p,index)
                finally:self.guides.pop()
        return super().guide(p,index)

    def line(self,p,guide,native,family_slope=None):
        # Only the visibly identified B1 band in the opening exposure may use
        # its luminance interfaces when reflected yellow suppresses blue tint.
        # The LocalPhysicalBlue reader still checks both exact physical sides.
        source_pts=native.frame.binding['source_pts'];review=self.material_reviews.get(source_pts,{})
        if p['feature_id'] in review.get('luminance_wide_targets',[]):
            index=review['programme_frame_index_1based']
            verified=LocalPhysicalBlue(native.frame.rgb,self.rows[index]['source'],native.mask,self.cameras[index])
            verified.frame=native.frame
            p=dict(p,_wide_local_identity_reason=review['wide_material_note_zh'])
            feature,hidden,refined=self.wide_source_choices(p,guide,verified,family_slope)
            if feature['point_count']:
                feature['native_visual_identity_review']=dict(source_pts=source_pts,
                    material_note=review['wide_material_note_zh'],source_review_path=str(self.review_path),
                    assistant_reviewed=True,user_confirmed=False)
            return feature,hidden,refined
        if p['feature_id'] in ('B1','B2','C'):
            return self.wide_source_choices(p,guide,native,family_slope)
        feature,hidden,refined=super().line(p,guide,native,family_slope)
        if not feature['point_count'] or p['material_signal']!='dark' or p['feature_id'] in ('B1','B2','C'):return feature,hidden,refined
        # A source line must persist between the originally measured stations.
        # A fit through isolated circles/colour ridges cannot fill white ice.
        dense,_,_=super().line(dict(p,_dense_current_source_verification=True),refined,native,family_slope)
        if not dense['point_count']:return dense,[],None
        fit=dense['source_line_fit'];origin=np.asarray(fit['point_origin']);v=np.asarray(fit['unit_tangent']);n=np.asarray(fit['unit_normal'])
        rows=fit['direct_readouts'];stations=np.asarray([(np.asarray(r['uv'])-origin)@v for r in rows]);order=np.argsort(stations)
        groups=np.split(order,np.flatnonzero(np.diff(stations[order])>55)+1);segments=[]
        for group in groups:
            lo,hi=stations[group].min(),stations[group].max()
            if len(group)<2 or hi-lo<35:continue
            ts=np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/120))+1));uv=origin+ts[:,None]*v+np.polyval(fit['normal_offset_coefficients'],ts)[:,None]*n
            segments.append(dict(uv=np.round(uv,4).tolist(),vertex_count=len(uv),closed=False,
                sigma_px=[max(1.,float(np.median([rows[j]['sigma_px'] for j in group])))]*len(uv),
                measurement_origin='same_frame_joint_straight_edge_fit_dense_current_material_support',
                representative_points_share_line_estimate=True,source_evidence_group_id=dense['source_evidence_group_id'],
                pixels_measured_in_this_frame=False,source_evidence_measured_in_this_frame=True,
                individual_native_pixel_readouts=False,hidden_span_filled=False,
                source_evidence=[dict(kind='shared_current_source_line_representative',source_evidence_group_id=dense['source_evidence_group_id'],
                    individual_native_pixel_readout=False,dense_current_material_interval=[float(lo),float(hi)]) for _ in uv]))
        dense['segments']=segments;dense['point_count']=sum(s['vertex_count'] for s in segments)
        dense['current_source_dense_material_verification']=dict(spacing_px=20,maximum_supported_station_gap_px=55,
            shallow_grey_haze_not_admitted=True,isolated_crossing_ring_or_red_band_cannot_fill_long_white_span=True)
        return dense,[],refined if dense['point_count'] else None

    def wide_source_choices(self,p,guide,native,family_slope=None):
        # A reference camera and a transported previous source shape are two
        # identity/search hypotheses. The latter cannot hide a newly visible
        # current material when reference camera errors differ between frames.
        current=next(i for i,r in self.rows.items() if r['source']['native_pts']==native.frame.binding['source_pts'])
        model_guide,_=Extractor.guide(self,p,current)
        choices=[('current_reference_camera',model_guide),('previous_selected_material',guide)]
        results=[]
        for origin,candidate in choices:
            if len(candidate)<2:continue
            result=super().line(p,candidate,native,family_slope)
            f=result[0];rows=f.get('source_line_fit',{}).get('direct_readouts',[])
            score=len(rows)
            results.append((score,origin,result))
        if not results:return blank_feature(p),[],None
        _,origin,result=max(results,key=lambda item:item[0])
        result[0]['current_material_search_choice']=dict(selected=origin,
            current_source_readout_counts={kind:score for score,kind,_ in results},
            scene_or_transport_coordinates_emitted_as_points=False)
        return result

    def family_audit(self,index,features,searches,native):
        """Use current measured parallel family and current camera row order."""
        candidates={f['world_curve']:f for f in features if 'source_line_fit' in f
                    and f['world_curve'] in shared.LONG_LINE_ORDER}
        if not candidates:return features,[],dict(status='no_parallel_line_points')
        camera=self.cameras[index];model_lines={}
        for p in self.paths:
            if p['curve_id'] not in shared.LONG_LINE_ORDER:continue
            uv,depth=project_fast(camera,np.asarray(p['xyz']))
            valid=np.isfinite(uv).all(1)&(depth>.05)&(uv>=[-300,-300]).all(1)&(uv<=[2219,1379]).all(1)
            uv=uv[valid]
            if len(uv)<2:continue
            direction=uv[-1]-uv[0]
            if abs(direction[0])<1:continue
            slope=float(direction[1]/direction[0]);y=float(uv[0,1]+slope*(960-uv[0,0]))
            model_lines[p['curve_id']]=(slope,y)
        source_lines=[(cid,image_line(f)) for cid,f in candidates.items()]
        reference=[(cid,z) for cid,z in source_lines if z and z['readout_span_px']>=180 and z['readout_count']>=3]
        aa=None;bb=None;origin='current_camera_physical_parallel_family_navigation'
        if len(reference)>=3:
            yy=np.asarray([z['y960'] for _,z in reference]);mm=np.asarray([z['slope'] for _,z in reference])
            ratios=[(mm[j]-mm[k])/(yy[j]-yy[k]) for j in range(len(yy)) for k in range(j) if abs(yy[j]-yy[k])>100]
            if ratios:
                aa=float(np.median(ratios));bb=float(np.median(mm-aa*yy));origin='current_selected_native_parallel_material_family'
        if aa is None and len(model_lines)>=2:
            values=np.asarray(list(model_lines.values()));aa,bb=np.polyfit(values[:,1],values[:,0],1)
        rejected=[];out=[];refined={}
        for f in features:
            cid=f['world_curve'];z=image_line(f) if cid in candidates else None
            if z is None or aa is None:out.append(f);continue
            expected=float(aa*z['y960']+bb)
            error=abs(z['slope']-expected)*z['readout_span_px']/np.sqrt(1+expected**2)
            if error>8:
                path,guide,nav_origin=searches[cid]
                replacement,_,new_guide=self.line(path,guide,native,family_slope=expected)
                rz=image_line(replacement) if replacement['point_count'] else None
                error2=abs(rz['slope']-(aa*rz['y960']+bb))*rz['readout_span_px']/np.sqrt(1+expected**2) if rz else None
                rejected.append(dict(target_id=f['target_id'],reason='current_material_branch_conflicts_with_parallel_physical_family',
                    orientation_error_px=float(error),replacement_orientation_error_px=float(error2) if error2 is not None else None,
                    original_candidate=deepcopy(f)))
                if rz is None or error2>8:continue
                f=replacement;f['navigation_origin']=nav_origin;z=rz;error=error2;refined[cid]=new_guide
            f['parallel_family_geometry_check']=dict(reference_origin=origin,
                orientation_departure_over_readout_span_px=float(error),current_frame_material_support_required=True)
            out.append(f)
        # Sort the physical rows in this view instead of inheriting first-shot orientation.
        rank={cid:j for j,(cid,_) in enumerate(sorted(model_lines.items(),key=lambda item:item[1][1]))}
        accepted=[];removed=set()
        for f in sorted([f for f in out if f['world_curve'] in rank and 'source_line_fit' in f],
                        key=lambda f:-f['source_line_fit']['actual_native_readout_count']):
            cid=f['world_curve'];z=image_line(f)
            if z is None:continue
            bad=None
            for other in accepted:
                oid=other['world_curve'];oz=image_line(other);separation=z['y960']-oz['y960']
                if (rank[cid]-rank[oid])*separation<=0 or abs(separation)<5:bad=(oid,separation);break
            if bad:
                removed.add(cid);rejected.append(dict(target_id=f['target_id'],reason='distinct_physical_rows_share_or_reverse_native_stroke',
                    conflicting_target_id='ice.'+bad[0],separation_px=float(bad[1]),original_candidate=deepcopy(f)))
            else:accepted.append(f)
        return [f for f in out if f['world_curve'] not in removed],rejected,dict(status='checked',
            reference_origin=origin,current_camera_physical_row_order=list(rank),rejected_candidate_count=len(rejected),
            remeasured_navigation_shapes=refined)

    def extract(self,index):
        started=time.perf_counter();row=self.rows[index];source=deepcopy(row['source'])
        raw_path=Path(row.get('image_path',source['image_path']));raw=cv2.imread(str(raw_path))
        if raw is None or raw.shape!=(1080,1920,3):raise ValueError('Native source image unavailable')
        if hashlib.sha256(raw.tobytes()).hexdigest()!=source['decoded_BGR_sha256']:raise ValueError('Current source pixels changed')
        review=self.material_reviews.get(source['native_pts'])
        if review and review['decoded_BGR_sha256']!=source['decoded_BGR_sha256']:raise ValueError('Material review belongs to different source pixels')
        mask_path=Path(row.get('mask_path',source.get('mask_path',str(self.output_root/f'masks_current/{index:06d}.png'))))
        person=cv2.imread(str(mask_path),0)
        if person is None:raise ValueError(f'Current source person mask unavailable: {mask_path}')
        if source.get('mask_sha256') and sha(mask_path)!=source['mask_sha256']:raise ValueError('Current source mask changed from manifest')
        broadcast,bcast=shared.current_broadcast_exclusion(raw);outside,domain=current_ice_domain(self.output_root,index)
        mask=(person>0)|broadcast|outside;native=ShotProfiles(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),source,mask,self.cameras[index])
        self.line_identity.observe(index,cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),mask,source)
        nested,_=self.nested_house_features(index,native);features=[];reports=[];searches={};inferred=[];current={}
        for original in self.paths:
            p=dict(original);guide,origin=self.guide(p,index)
            if len(guide)<(5 if p['_circle'] else 2):
                reports.append(dict(target_id='ice.'+p['curve_id'],points=0,missing_reason='physical_curve_not_visible_in_current_selected_camera_domain'));continue
            if not p['_circle']:
                guide,origin,policy=self.line_identity.select_guide(p,index,guide,origin);p['_line_identity_policy']=policy
            visible=np.isfinite(guide).all(1)&(guide>=[-75,-75]).all(1)&(guide<=[1994,1154]).all(1)
            if not visible.any() and p['curve_id'] not in nested:
                reports.append(dict(target_id='ice.'+p['curve_id'],points=0,missing_reason='physical_curve_outside_current_image'));continue
            searches[p['curve_id']]=(p,guide,origin)
            if p['curve_id'] in nested:f=nested[p['curve_id']];refined=None;origin=f.get('navigation_origin','joint_current_physical_circle_family')
            elif p['_circle']:
                # A nested family failure cannot be replaced by independent layer ellipses.
                f=blank_feature(p);refined=None
                f['extraction_diagnostic']=dict(missing_reason='insufficient_current_material_support_for_joint_physical_circle_family')
            else:f,hidden,refined=self.line(p,guide,native);inferred.extend(hidden)
            f['navigation_origin']=origin
            if f['point_count']:
                features.append(f)
                if refined is not None:current[p['curve_id']]=refined
            reports.append(dict(target_id=f['target_id'],points=f['point_count'],navigation_origin=origin,
                native_material_diagnostic=f.get('extraction_diagnostic'),
                missing_reason=None if f['point_count'] else f.get('extraction_diagnostic',{}).get('missing_reason','no_clear_current_native_material_support')))
        features,rejected,family=self.family_audit(index,features,searches,native)
        disqualified={r['target_id']:r for r in (review or {}).get('unqualified_targets',[])}
        for f in features:
            if f['target_id'] in disqualified:
                rejected.append(dict(target_id=f['target_id'],reason='current_native_visual_material_ownership_unqualified',
                    reason_zh=disqualified[f['target_id']]['reason_zh'],source_review_path=str(self.review_path),original_candidate=deepcopy(f)))
        features=[f for f in features if f['target_id'] not in disqualified]
        self.line_identity.record(index,features)
        source.update(mask_path=str(mask_path),mask_sha256=sha(mask_path),broadcast_timer_exclusion=bcast,ice_visible_domain=domain)
        packet=dict(schema='rinkspace.single-frame-source-ice-features.v1',programme_frame_index_1based=index,
            source=source,status='awaiting_user_review',user_accepted=False,features=features,inferred_segments=inferred,
            point_count=sum(f['point_count'] for f in features),optimization_performed=False,
            model_scene_sha256=sha(self.model_root/'scene.json'),circle_geometry_sha256=self.circle_geometry_sha256,
            per_feature_coverage=reports,parallel_family_geometry_audit={k:v for k,v in family.items() if k!='remeasured_navigation_shapes'},
            source_correlated_groups=[dict(group_id=f['source_evidence_group_id'],target_id=f['target_id'],native_source_pts=source['native_pts'],
                representative_point_count=f['point_count'],correlation_reason='same current native material curve; representatives are correlated',
                native_individual_readout_count=f.get('source_line_fit',f.get('source_circle_fit',{}))['actual_native_readout_count'])
                for f in features if 'source_line_fit' in f or 'source_circle_fit' in f],
            selected_frames_only=True,intermediate_exposures_source_points_extracted=False,
            no_first_shot_manual_guides_imported=True,observation_coordinates_from_projection=False,
            elapsed_seconds=time.perf_counter()-started)
        if review:packet['current_native_material_review']=dict(path=str(self.review_path),sha256=sha(self.review_path),record=review)
        apply_rejections(packet,conflicts(packet['features']))
        final_counts={f['target_id']:f['point_count'] for f in packet['features']}
        for report in packet['per_feature_coverage']:
            if report['points'] and not final_counts.get(report['target_id']):report['missing_reason']='current_material_identity_or_parallel_family_rejection'
            if report['target_id'] in disqualified:report.update(missing_reason=disqualified[report['target_id']]['reason_zh'],source_material_identity_unqualified=True)
            report['points']=final_counts.get(report['target_id'],0)
        packet['source_correlated_groups']=[g for g in packet['source_correlated_groups'] if g['target_id'] in final_counts]
        accepted_curves={f['world_curve'] for f in packet['features']}
        self.previous={cid:shape for cid,shape in current.items() if cid in accepted_curves};self.previous_index=index
        destination=self.output_root/f'ice_candidates/{index:06d}.json'
        if destination.exists():
            # Re-reading preserves the prior current-source candidate. Failed
            # source branches and manual-review exclusions remain inspectable.
            previous_sha=sha(destination)
            archive=self.output_root/f'ice_diagnostics/revisions/{index:06d}_{previous_sha[:16]}.json'
            if not archive.exists():
                archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(destination.read_bytes())
        write(destination,packet)
        if rejected:write(self.output_root/f'ice_diagnostics/rejected_candidates/{index:06d}.json',dict(index=index,source=source,rejected_candidates=rejected))
        write(self.line_identity.motion_quality_path,dict(schema='rinkspace.selected-native-ice-motion-navigation.v1',
            registration=self.line_identity.registration,selected_frames_only=True,registration_keypoints_are_not_fitting_observations=True))
        return packet


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=DATA);ap.add_argument('--model-root',type=Path,default=MODEL)
    ap.add_argument('--material-review',type=Path,default=REVIEW)
    ap.add_argument('--indices');ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(1);extractor=SelectedExtractor(args.data_root.resolve(),args.model_root.resolve(),args.material_review.resolve())
    indices=sorted(map(int,args.indices.split(','))) if args.indices else sorted(extractor.rows)
    for index in indices:
        destination=extractor.output_root/f'ice_candidates/{index:06d}.json'
        if destination.exists() and not args.overwrite:continue
        packet=extractor.extract(index)
        print(__import__('json').dumps(dict(index=index,points=packet['point_count'],seconds=packet['elapsed_seconds'],
            features=[(f['target_id'],f['point_count']) for f in packet['features']]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
