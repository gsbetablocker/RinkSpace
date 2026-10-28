"""Current-native finite lines, historical whole-band reading and source motion.

Only actual direct_readouts are exported. Fitted/transported guides are search
locations and do not change the native scene or become measured coordinates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.extract_first_shot_ice import Extractor,NativeProfiles,LONG_LINE_ORDER
from calibration.program_tracking.native_ice_line_continuity import NativeLineIdentity,straight_guide
from calibration.program_tracking.repair_first_shot_wide_line_identity import EntryProfiles
from rinkspace.compat.d78_entities import native_definition
from reread_s04_house import NativeHouseReader
from review_points_s04 import CurrentNativeGrid,OUT,ADS,read,write
from ice_points_s04 import source_domain,REC,FiniteSourceIdentity


class SourceMotionIdentity(NativeLineIdentity):
    def __init__(self,photographic, *, motion_path=None):
        self.photographic=photographic
        self.motion_quality_path=Path(motion_path) if motion_path is not None else OUT/'evidence/source_navigation.json'
        self.motion={r['source_pts']:r for r in read(self.motion_quality_path)['pairs']}
        self.guides=[];self.states={};self.last_policy={}

    def _transport(self,a,b,uv):
        variance=0.;maximum=0.;hull=None
        for pts in range(a+256,b+1,256):
            r=self.motion.get(pts)
            if not r or r['status']!='candidate' or r['diagnostic_median_reference_px']>2.:return None
            error=2.*r['diagnostic_p95_reference_px'];variance+=error**2;maximum=max(maximum,error)
            if np.sqrt(variance)>6.:return None
            old,oc=self.photographic.transforms[pts-256];new,nc=self.photographic.transforms[pts]
            if oc!=nc:return None
            uv=cv2.perspectiveTransform(np.asarray(uv,float)[None],new@np.linalg.inv(old))[0]
            hull=np.asarray(r['fit_support_current_xy'],np.float32)*2.
            if not np.isfinite(uv).all():return None
        return uv,float(np.sqrt(variance)),maximum,hull


class NativeLineReader:
    def __init__(self,grid, *, photographic=None, motion_path=None, source_prefix='s04', source_cross_line_guides=False, source_parallel_references=False, source_finite_extents=False, recover_partial_blue=False, photographic_cross_direction=False, native_crossline_partitions=False, temporal_achromatic_blue=False):
        self.grid=grid
        self.source_prefix=source_prefix
        self.source_cross_line_guides=source_cross_line_guides
        self.source_parallel_references=source_parallel_references
        self.source_finite_extents=source_finite_extents
        self.recover_partial_blue=recover_partial_blue
        self.photographic_cross_direction=photographic_cross_direction
        self.native_crossline_partitions=native_crossline_partitions
        self.temporal_achromatic_blue=temporal_achromatic_blue
        self.house=NativeHouseReader(grid,motion_path=motion_path)
        self.photographic=photographic if photographic is not None else FiniteSourceIdentity()
        self.identity=SourceMotionIdentity(self.photographic,motion_path=motion_path)
        self.paths=[]
        for p in self.house.scene['paths']:
            if p['layer']!='ice':continue
            kind,params=native_definition(self.house.scene,p,grid.core)
            if kind!=0:continue
            cid=p['label'].removeprefix('ice.');fid,role=cid.split(':')
            signal='blue' if fid in ('B1','B2') else 'red' if fid=='C' or fid.startswith('goal_') else 'dark'
            self.paths.append(dict(p,curve_id=cid,feature_id=fid,world_curve_role=role,
                operator='paint_center',material_signal=signal,material_transition=None,_circle=False,
                xyz=np.asarray(grid.core.sample_path(self.house.scene['board'],kind,params,[],[0.,1.])).tolist()))
        # Read lane paint before the fine centre connections, reusing the
        # historical current-frame width reference against house dark fills.
        self.paths.sort(key=lambda p:(not p['feature_id'].startswith('lane'),p['label']))

    def line(self,*args,**kwargs):return Extractor.line(self,*args,**kwargs)

    def current_native_parallel_direction(self,pts):
        rows=[]
        for path in self.paths:
            if path['curve_id']not in LONG_LINE_ORDER:continue
            p=self.current_projections[path['id']];q=p['uv'];good=p['valid']&np.isfinite(q).all(1)
            q=q[good]
            if len(q)<2 or np.ptp(q[:,0])<35:continue
            slope,level=np.polyfit(q[:,0]-960,q[:,1],1);rows.append([level,slope])
        if len(rows)<2:return None
        yy,mm=np.asarray(rows).T
        a,b=np.linalg.lstsq(np.c_[yy,np.ones(len(yy))],mm,rcond=None)[0]
        return float(a),float(b)

    def crossline_partition(self,path,projections):
        """Native ordering limits a search, without emitting model pixels."""
        by_label={p['label']:p for p in self.paths};frames={}
        for label in ['ice.B1:segment0','ice.B2:segment0','ice.C:segment0']:
            if label not in by_label:return None
            p=projections[by_label[label]['id']];q=p['uv'];good=p['valid']&np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1920);q=q[good]
            if len(q)<2:return None
            origin=q.mean(0);_,_,axes=np.linalg.svd(q-origin,full_matrices=False);tangent=axes[0];normal=np.array([-tangent[1],tangent[0]])
            frames[label]=(origin,normal)
        target=frames[path['label']][0];boundaries=['ice.C:segment0']if path['feature_id']in('B1','B2')else['ice.B1:segment0','ice.B2:segment0']
        yy,xx=np.ogrid[:1080,:1920];mask=np.zeros((1080,1920),bool)
        for label in boundaries:
            origin,normal=frames[label];side=np.sign((target-origin)@normal)
            if side:mask|=side*((xx-origin[0])*normal[0]+(yy-origin[1])*normal[1])<=0
        return mask

    def read(self,pts,bgr,excluded,source_sha):
        native=NativeProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),dict(source_video_sha256=source_sha,
            native_pts=pts,time_base='1/12800'),excluded)
        projections=self.grid.project(pts);self.current_projections=projections
        features={};diagnostics={};accepted=[];searches={};path_lookup={};native_guides={}
        if self.source_parallel_references:self.source_parallel_reference_ids=set()
        for path in self.paths:
            projected=projections[path['id']]
            uv=projected['uv'];good=projected['valid']&(uv>=[-120,-120]).all(1)&(uv<=[2039,1199]).all(1)
            photo=self.photographic.at(path['label'],pts)
            support=None
            if photo is not None:
                g,h=photo;support=cv2.perspectiveTransform(np.vstack(g['spans'])[None],h)[0]
                if self.source_parallel_references and len(support)>=2 and path['curve_id']in LONG_LINE_ORDER:
                    self.source_parallel_reference_ids.add(path['curve_id'])
            if good.sum()<2:
                source_good=(np.isfinite(support).all(1)&(support>=[-120,-120]).all(1)&(support<=[2039,1199]).all(1))if support is not None else np.zeros(0,bool)
                if source_good.sum()<2:
                    diagnostics[path['label']]=dict(status='native_finite_line_outside_source_image',source_identity_support_also_outside=True)
                    continue
                # An auxiliary camera is not a visibility oracle. Keep the
                # current native identity, but search its actual finite photo
                # support when a not-yet-optimized path projects offscreen.
                fallback=straight_guide(support[source_good],support[source_good]);origin='same_shot_finite_source_support_with_native_navigation_outside'
            else:
                fallback=straight_guide(uv[good],uv[good]);origin='current_native_finite_line'
            native_guides[path['curve_id']]=fallback.copy()
            if photo is not None:
                g,h=photo
                # Photographic source controls image alignment. Current native
                # controls the finite path; historical world geometry is absent.
                proposal=straight_guide(support,support if self.source_finite_extents else fallback)
                if len(proposal)>=2:
                    fallback=proposal;origin='same_shot_actual_source_readouts:'+str(g['pts'])
            if len(fallback)<2:continue
            guide,origin,policy=self.identity.select_guide(path,pts,fallback,origin)
            if getattr(self,'temporal_achromatic_blue',False)and path['feature_id']in('B1','B2')and not policy.get('current_identity_constrained')and photo is not None:
                # A full actual photographic trace may initialize the same
                # existing identity constraint before a previous frame exists.
                # This is not a user-acceptance label or a model measurement.
                source_points=np.vstack(g['spans'])
                if len(source_points)>=4 and np.linalg.norm(np.ptp(source_points,axis=0))>=400:
                    error=0. if g['pts']==pts else None
                    if g['pts']<pts:
                        moved=self.identity._transport(g['pts'],pts,source_points)
                        if moved is not None:error=moved[1]
                    if error is not None:
                        policy=dict(policy,current_identity_constrained=True,
                            identity_source='qualified_whole_actual_source_photograph',reference_frame=g['pts'],
                            normal_identity_tolerance_px=min(14.,6.+2.*error),search_radius_px=min(24.,14.+2.*error),
                            registration_error_px=error,source_photograph=g['path'])
            if self.source_finite_extents and support is not None and len(support)>=2 and len(guide)>=2:
                # Temporal identity may set the line direction, but may not
                # restore the wider projected extent past real source support.
                guide=straight_guide(guide,support)
                policy=dict(policy,finite_source_extent=True,source_extent_pts=g['pts'])
                if len(guide)<2:continue
            local=dict(path,_line_identity_policy=policy)
            direction=None
            if path['feature_id']in ('B1','B2','C'):
                # Cross-lines have their own direction; a short reflected or
                # neighbouring longitudinal stripe must not become B/C.
                # Reuse the existing family-direction candidate discriminator.
                actual=guide if getattr(self,'photographic_cross_direction',False)and photo is not None else native_guides[path['curve_id']]
                delta=actual[-1]-actual[0]
                if abs(delta[0])>1e-8:direction=float(delta[1]/delta[0])
                if not self.source_cross_line_guides or photo is None:
                    guide=actual;origin='current_native_cross_line_with_actual_RGB_branch_disambiguation'
                # Cross-line direction and identity are separate. Keeping a
                # special direction must not discard the already registered
                # same-material temporal constraint.
                local=dict(path,_line_identity_policy=policy)
            searches[path['curve_id']]=(local,guide,origin);path_lookup[path['curve_id']]=path
            material_native=native;material_excluded=excluded
            if getattr(self,'native_crossline_partitions',False)and path['feature_id']in('B1','B2','C'):
                partition=self.crossline_partition(path,projections)
                if partition is not None:
                    material_excluded=excluded|partition
                    material_native=NativeProfiles(native.frame.rgb,dict(source_video_sha256=source_sha,native_pts=pts,time_base='1/12800'),material_excluded)
            feature,_,refined=Extractor.line(None,local,guide,material_native,family_slope=direction)
            allow_achromatic=not getattr(self,'temporal_achromatic_blue',False)or bool(feature.get('source_line_fit'))or policy.get('current_identity_constrained',False)
            if path['feature_id'] in ('B1','B2') and allow_achromatic and (not feature.get('source_line_fit')or self.recover_partial_blue):
                # Restore the historical entry/exit repair: yellow reflection
                # can suppress blue chroma although both paint flanks remain
                # measurable in luminance. This still uses the same named,
                # finite source guide and the same whole-line consistency.
                bounded=EntryProfiles(native.frame.rgb,dict(source_video_sha256=source_sha,
                    native_pts=pts,time_base='1/12800'),material_excluded)
                alternative,_,alternate_guide=Extractor.line(None,dict(local,
                    _wide_local_identity_reason='current_native_finite_blue_strip_and_same_shot_source_identity'),guide,bounded,family_slope=direction)
                replace=bool(alternative.get('source_line_fit'))
                original_fit=feature.get('source_line_fit')
                if replace and original_fit:
                    # A partial chroma reading is positive branch evidence.
                    # The existing luminance reader can extend coverage only
                    # on the same straight strip, under its existing 4 px
                    # whole-line support criterion. No raw UV is straightened.
                    af=alternative['source_line_fit'];old=np.array([r['uv']for r in original_fit['direct_readouts']])
                    a=np.array(af['point_origin']);tangent=np.array(af['unit_tangent']);normal=np.array(af['unit_normal'])
                    distance=np.abs((old-a)@normal-np.polyval(af['normal_offset_coefficients'],(old-a)@tangent))
                    replace=len(af['direct_readouts'])>len(old) and bool(np.all(distance<=4.))
                    alternative['extraction_diagnostic']['partial_chroma_branch_check']=dict(
                        original_actual_rows=len(old),maximum_normal_disagreement_px=float(distance.max()),
                        same_existing_whole_line_inlier_limit_px=4.,accepted=replace)
                if replace:
                    alternative['extraction_diagnostic']['whole_chroma_attempt']=feature.get('extraction_diagnostic',{})
                    feature,refined=alternative,alternate_guide
            diag=feature.get('extraction_diagnostic',{})
            diag.update(search_origin=origin,finite_extent_authority='current_native_model',
                temporal_identity=policy,source_guide_is_not_observation=True)
            diagnostics[path['label']]=diag
            fit=feature.get('source_line_fit')
            if fit is None:continue
            accepted.append(feature)
        checked,rejected,family=Extractor.family_audit(self,pts,accepted,searches,native)
        present={f['world_curve']for f in checked};rescued=[]
        for failure in rejected:
            cid=failure['target_id'].removeprefix('ice.')
            if cid in present or cid not in searches:continue
            path=path_lookup[cid];guide=searches[cid][1] if self.source_finite_extents else native_guides[cid]
            delta=guide[-1]-guide[0]
            if abs(delta[0])<1e-8:continue
            level=float(guide[0,1]+delta[1]/delta[0]*(960-guide[0,0]))
            a,b=family['slope_vs_y960'];direction=a*level+b
            # A family retry is still the same physical material. Retain the
            # source identity policy and its exclusion surface at this entry.
            local=dict(searches[cid][0]);retry_native=native
            if getattr(self,'native_crossline_partitions',False)and path['feature_id']in('B1','B2','C'):
                partition=self.crossline_partition(path,projections)
                if partition is not None:
                    retry_native=NativeProfiles(native.frame.rgb,dict(source_video_sha256=source_sha,native_pts=pts,time_base='1/12800'),excluded|partition)
            replacement,_,_=Extractor.line(None,local,guide,retry_native,family_slope=direction)
            if not replacement.get('source_line_fit')and path['feature_id']in('B1','B2')and local.get('_line_identity_policy',{}).get('current_identity_constrained'):
                retry_excluded=excluded if not getattr(self,'native_crossline_partitions',False)or partition is None else excluded|partition
                bounded=EntryProfiles(native.frame.rgb,dict(source_video_sha256=source_sha,native_pts=pts,time_base='1/12800'),retry_excluded)
                replacement,_,_=Extractor.line(None,dict(local,_wide_local_identity_reason='same_named_material_during_family_retry'),guide,bounded,family_slope=direction)
            if replacement.get('source_line_fit'):
                rescued.append(replacement);present.add(cid)
                searches[cid]=(local,guide,'current_native_finite_path_after_photographic_branch_conflict')
        if rescued:
            checked,more,second=Extractor.family_audit(self,pts,checked+rescued,searches,native)
            rejected+=more;family['source_reread_after_identity_conflict']=len(rescued)
        diagnostics['parallel_family']=dict(**{k:v for k,v in family.items()if k!='remeasured_navigation_shapes'},
            rejected_source_candidates=rejected)
        for feature in checked:
            path=path_lookup[feature['world_curve']];fit=feature['source_line_fit'];rows=[]
            for i,r in enumerate(fit['direct_readouts']):
                row=dict(r,native_path_id=path['id'],feature_id=path['label'],native_pts=pts,
                    source_row_id=f'{self.source_prefix}_whole_line_RGB/{pts}/{path["curve_id"]}/{i}')
                rows.append(row)
            features[path['label']]=dict(native_path_id=path['id'],rows=rows,navigation=dict(
                search_origin=searches[path['curve_id']][2],temporal_identity=feature.get('native_static_line_identity_policy',{}),
                source_line_fit={k:v for k,v in fit.items()if k!='direct_readouts'},parallel_family_geometry_check=feature.get('parallel_family_geometry_check'),
                current_native_finite_support=True,fitted_representatives_exported=False))
        self.identity.record(pts,checked)
        return features,diagnostics


def reread_final_support(path, fit, native):
    """Re-read RGB when releasing a direction prior invalidated line support.

    The shared extractor selects profiles with a direction prior, then computes
    its final unconstrained material line. Those two fits can disagree at a
    crossing. Use the final actual material only to navigate a new RGB read;
    retain the established 4 px support criterion and every measured UV.
    """
    observed=np.asarray([r['uv']for r in fit['direct_readouts']])
    origin=observed.mean(0);_,_,axes=np.linalg.svd(observed-origin,full_matrices=False);tangent=axes[0];normal=axes[1]
    if len(observed)<3 or np.max(np.abs((observed-origin)@normal))<4.:
        return None
    if np.dot(tangent,fit['unit_tangent'])<0:tangent=-tangent
    stations=(observed-origin)@tangent
    query=np.linspace(float(stations.min()),float(stations.max()),257)
    guide=origin+query[:,None]*tangent
    feature,_,_=Extractor.line(None,dict(path,_source_query_spacing_px=35,_separate_profile_support=True),guide,native)
    checked=feature.get('source_line_fit')
    if checked is None or checked['residual_max_px']>=4.:
        return None
    previous_ends=guide[[0,-1]]-checked['point_origin']
    branch_error=np.abs(previous_ends@checked['unit_normal']-np.polyval(checked['normal_offset_coefficients'],previous_ends@checked['unit_tangent']))
    if np.any(branch_error>=4.):
        return None
    feature['extraction_diagnostic']['final_direction_prior_release_reread']=dict(
        previous_actual_readout_count=len(observed),previous_maximum_normal_error_px=fit['residual_max_px'],
        maximum_normal_error_px=checked['residual_max_px'],existing_support_limit_px=4.,
        previous_material_endpoint_normal_disagreement_px=branch_error.tolist(),
        navigation_from_actual_material=True,all_coordinates_reread_from_current_RGB=True)
    return feature


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pts',type=int,nargs='+');parser.add_argument('--all',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();cv2.setNumThreads(2)
    grid=CurrentNativeGrid();reader=NativeLineReader(grid)
    vis=read(ADS/'evidence/visibility.json');lookup={r['pts']:r for r in vis['frames']}
    dest=args.output_dir or OUT/'evidence/whole_line_RGB';dest.mkdir(parents=True,exist_ok=True)
    algorithm=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'calibration/program_tracking/extract_first_shot_ice.py').read_bytes()
        +(ROOT/'calibration/program_tracking/native_ice_material_repair.py').read_bytes()).hexdigest()
    requested=sorted(lookup)if args.all else args.pts
    if not requested:parser.error('Use --all or --pts')
    for index,pts in enumerate(requested):
        start=time.monotonic();v=lookup[pts];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Wrong original exposure')
        graphics,border,_,_=source_domain(im,mask,read(REC/f'dense_board_v1/qualified_frames/{pts}.json'))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        features,diag=reader.read(pts,im,excluded,vis['source_video_sha256'])
        record=dict(source_pts=pts,source_BGR_sha256=v['source_BGR_sha256'],algorithm_sha256=algorithm,
            features=features,diagnostics=diag,native_authority=grid.receipt,original_sources_modified=False,
            model_or_fitted_curve_pixels_emitted=False,seconds=time.monotonic()-start)
        pending=dest/f'{pts}.pending.json';write(pending,record);pending.replace(dest/f'{pts}.json')
        if not args.all or index%50==0 or index==len(requested)-1:
            print(json.dumps(dict(frame=index+1,total=len(requested),pts=pts,counts={k:len(v['rows'])for k,v in features.items()},seconds=record['seconds'])),flush=True)
    grid.unchanged()


if __name__=='__main__':main()
