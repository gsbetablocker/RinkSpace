"""Pending key contours with current photographed-pattern navigation.

Whole-print semantic failures remain unchanged. This reader can retain useful
current material readouts for independent source association; it never grants
physical identity or fit eligibility. Original complete semantic checks remain
in the returned diagnostics. Numeric, ink and foreground gates are unchanged.
"""
from __future__ import annotations
import cv2
import numpy as np
from rinkspace.session.print_patterns import project,sample,registration_geometry_check,read_source_material_transition_profiles
from rinkspace.session.print_shape_queries import KeyShapePrintPattern,key_shape_stations,KEY_SHAPE_POLICY


class PhotographicContourCandidates(KeyShapePrintPattern):
    def read_paths(self,image,exclusion,registration,binding,spacing=None):
        status=registration.get('status')
        if status not in ('source_pattern_registered','whole_pattern_unresolved'):
            return [],dict(reason=status,fit_eligible=False)
        if registration.get('correlation',-1.)<=.43:
            return [],dict(reason='current_photographic_pattern_unresolved',fit_eligible=False)
        H=np.asarray(registration['H'])
        geometry=registration_geometry_check(H,np.eye(3),self.points,warp_roi=(self.points.min(0),self.points.max(0)))
        if not geometry['valid']or registration.get('registration_geometry',{}).get('valid')is False:
            return [],dict(reason='current_registration_geometry_invalid',geometry=geometry,fit_eligible=False)
        inverse=np.linalg.inv(H);query=[];normals=[];labels=[];key_plans=[]
        semantic_resolved,structure=self.resolved_glyphs(image,exclusion,H,horizontal_blur_px=registration['shared_horizontal_blur_px'])
        checks=registration.get('glyph_checks',[])
        resolved={g['path_id']for g in checks if g['correlation']>.45}
        if not checks or sum(g['correlation']>.45 for g in checks)<.5*len(checks):
            return [],dict(reason='current_whole_photographic_variation_unresolved',original_whole_pattern_status=status,structure=structure,fit_eligible=False)
        exclusion=cv2.dilate(exclusion,np.ones((7,7),np.uint8))
        for p in self.paths:
            if p['path_id'] not in resolved:continue
            points=project(H,p['pixels']);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            if length[-1]<4:continue
            t,key_plan=key_shape_stations(points,closed=p.get('source_closed',p.get('closed')),horizontal_blur_px=registration['shared_horizontal_blur_px'])
            key_plans.append(dict(path_id=p['path_id'],**key_plan))
            q=np.column_stack([np.interp(t,length,points[:,k]) for k in (0,1)])
            l=np.column_stack([np.interp(t-.8,length,points[:,k]) for k in (0,1)])
            r=np.column_stack([np.interp(t+.8,length,points[:,k]) for k in (0,1)])
            n=np.c_[-(r-l)[:,1],(r-l)[:,0]];n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
            delta=sample(self.ink,project(inverse,q+n*1.25))-sample(self.ink,project(inverse,q-n*1.25))
            n[delta<0]*=-1
            if self.ink_polarity=='dark':n*=-1
            valid=(q[:,0]>5)&(q[:,0]<image.shape[1]-6)&(q[:,1]>5)&(q[:,1]<image.shape[0]-6)&(abs(delta)>2)
            q=q[valid];n=n[valid];xy=np.rint(q).astype(int);visible=exclusion[xy[:,1],xy[:,0]]==0
            query.extend(q[visible]);normals.extend(n[visible]);labels.extend([p['path_id']]*int(visible.sum()))
        if not query:return [],dict(reason='no_resolved_visible_glyphs',structure=structure)
        query=np.array(query);normals=np.array(normals)
        radius=np.maximum(4.,registration['shared_horizontal_blur_px']*abs(normals[:,0])/2+3)
        readings=read_source_material_transition_profiles(np.ascontiguousarray(image[:,:,::-1]),binding=binding,
            search_points_uv=query,normals=normals,search_intervals_px=np.c_[-radius,radius],transition='dark_to_light',
            tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
        # One exposure-wide ink/background split, not a self-justifying split of
        # a mislabeled local patch. Foreground and out-of-frame pixels are absent.
        mask=cv2.warpPerspective(self.mask,H,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        values=self.ink_signal(image)[(mask>0)&(exclusion==0)]
        if len(values)<20:return [],dict(reason='current_ink_material_unresolved')
        boundary,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        accepted=[];reasons={};signal=self.ink_signal(image).astype(np.float32)
        for i,(r,q,n,name) in enumerate(zip(readings,query,normals,labels)):
            flags=r['source_flags'];offsets=np.asarray(r['source_q10_q50_q90_normal_offsets_px'])
            reason='source_profile_unresolved' if any(flags.values()) or not np.isfinite(offsets).all() else None
            if reason is None:
                xy=q+np.linspace(offsets[0]-1,offsets[2]+1,33)[:,None]*n
                if np.any(xy<0) or np.any(xy>[image.shape[1]-1,image.shape[0]-1]):reason='profile_outside_source'
                elif np.any(sample(exclusion,xy)>0):reason='profile_foreground'
                else:
                    sides=sample(signal,q+np.array([offsets[0]-1,offsets[2]+1])[:,None]*n)
                    lower,upper=sides[::-1] if self.ink_polarity=='dark' else sides
                    if not lower<boundary<upper:reason='profile_is_not_current_ink_boundary'
            if reason:
                reasons[reason]=reasons.get(reason,0)+1;continue
            accepted.append(dict(path_id='print_instance/'+name.split('.')[0]+'::'+name,candidate=i,
                uv=np.asarray(r['actual_source_uv']).tolist(),search_uv=q.tolist(),normal=n.tolist(),actual_normal_quantiles=offsets.tolist(),
                actual_current_material_sides=sides.tolist(),current_ink_class_boundary=float(boundary),
                source_profile_flags=flags,current_source_RGB_measured=True,model_UV_used_as_measurements=False,physical_identity_qualified=False,fit_eligible=False,original_whole_pattern_status=status))
        return accepted,dict(queries=len(query),accepted=len(accepted),reasons=reasons,current_ink_class_boundary=float(boundary),structure=structure,query_policy=KEY_SHAPE_POLICY,key_query_plans=key_plans,original_whole_pattern_status=status,original_semantic_resolved_paths=sorted(semantic_resolved),photographic_supported_paths=sorted(resolved),physical_identity_qualification_required=True,fit_eligible=False)
