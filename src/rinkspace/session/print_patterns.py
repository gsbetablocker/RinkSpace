"""Register one semantically checked complete print, then read current RGB edges.

The initial chart only supplies a search. Local fragments cannot confer labels.
Motion blur is a nuisance of pattern registration, never a source of edge UV.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles


def project(H, points):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[None], H)[0]


def sample(image, points):
    q=np.asarray(points,np.float32)
    shape=q.shape[:-1]
    return cv2.remap(image,q[...,0].reshape(1,-1),q[...,1].reshape(1,-1),cv2.INTER_LINEAR).reshape(*shape,*image.shape[2:]).astype(float)


def registration_geometry_check(H,initial_H,reference_points,*,warp_roi):
    """Check the finite print and the complete ECC warp ROI analytically.

    A quadrilateral's signed area can look ordinary while its projective map
    crosses a horizon. The denominator is affine, so its extrema on each
    rectangular domain occur at its corners. With no pole, the analytic local
    Jacobian determinant also proves orientation over the complete domain.
    Directional scales are reported; they are not an invented identity bound.
    """
    reasons=[]
    def normalized(value):
        value=np.asarray(value,float)
        if value.shape!=(3,3) or not np.isfinite(value).all() or not np.any(value):
            raise ValueError('nonfinite_or_empty_homography')
        return value/np.max(np.abs(value))
    def domain(matrix,lo,hi):
        probes=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]],(lo+hi)/2],float)
        v=np.c_[probes,np.ones(len(probes))]@matrix.T;den=v[:,2]
        horizon=not (np.all(den>0) or np.all(den<0))
        result=dict(denominator_range=[float(den.min()),float(den.max())],horizon_crosses_domain=horizon)
        if horizon:return result,None
        J=(matrix[:2,:2][None]*den[:,None,None]-v[:,:2,None]*matrix[2,:2][None,None,:])/den[:,None,None]**2
        if not np.isfinite(J).all():return result,None
        determinant=np.linalg.det(J);scale=np.linalg.svd(J,compute_uv=False)
        result.update(local_J=J.tolist(),local_determinant_range=[float(determinant.min()),float(determinant.max())],
            local_scale_singular_values=scale.tolist())
        return result,J
    try:
        H=normalized(H);initial=normalized(initial_H);points=np.asarray(reference_points,float)
        if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
            raise ValueError('invalid_finite_reference_domain')
        lo=points.min(0);hi=points.max(0)
        current,J=domain(H,lo,hi);previous,J0=domain(initial,lo,hi)
        if current['horizon_crosses_domain']:reasons.append('registered_shape_horizon')
        if previous['horizon_crosses_domain']:reasons.append('initial_shape_horizon')
        relative=None
        if J is None or J0 is None:
            if not reasons:reasons.append('nonfinite_shape_J')
        else:
            relative=J@np.linalg.inv(J0);det=np.linalg.det(relative);scale=np.linalg.svd(relative,compute_uv=False)
            if not np.isfinite(relative).all() or not np.all(det>0):reasons.append('shape_orientation_not_preserved')
            relative=dict(local_determinant_range=[float(det.min()),float(det.max())],
                local_scale_singular_values=scale.tolist(),
                local_anisotropy_max=float(np.max(scale[:,0]/scale[:,1])) if np.all(scale[:,1]>0) else None)
        roi_lo,roi_hi=np.asarray(warp_roi,float)
        if np.any(roi_hi<=roi_lo):raise ValueError('empty_warp_ROI')
        shift=np.eye(3);shift[:2,2]=-roi_lo
        E=normalized(shift@H@np.linalg.inv(initial)@np.linalg.inv(shift))
        warp,_=domain(E,np.zeros(2),roi_hi-roi_lo)
        if warp['horizon_crosses_domain']:reasons.append('warp_ROI_horizon')
        if np.linalg.det(H)==0:reasons.append('singular_registered_homography')
        return dict(valid=not reasons,reasons=reasons,registered_shape=current,initial_shape=previous,
            relative_shape=relative,warp_ROI=warp,finite_shape_bbox=np.r_[lo,hi].tolist(),
            warp_roi=np.asarray(warp_roi).tolist(),anisotropy_is_diagnostic_only=True)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
        return dict(valid=False,reasons=reasons+['invalid_registration_geometry'],detail=str(error))


class PrintPattern:
    def __init__(self, atlas, image):
        self.atlas=atlas
        self.paths=atlas['paths']
        names=[p['path_id'] for p in self.paths]
        if len(set(names))!=len(names):
            raise ValueError('A complete semantic reference requires unique path identities')
        self.image=np.array(image,copy=True)
        self.ink_polarity=atlas.get('ink_polarity','light')
        if self.ink_polarity not in ('light','dark'):raise ValueError('Unknown ink polarity')
        # Minimum channel separates white ink from saturated blue/purple decoration.
        self.ink=self.ink_signal(self.image).astype(np.float32)
        self.points=np.concatenate([p['pixels'] for p in self.paths])
        lo=np.floor(self.points.min(0)-5).astype(int)
        hi=np.ceil(self.points.max(0)+6).astype(int)
        self.mask=np.zeros(self.ink.shape,np.uint8)
        self.mask[max(0,lo[1]):hi[1],max(0,lo[0]):hi[0]]=255
        self._reference_material_topology()

    def ink_signal(self,image):
        signal=np.asarray(image).min(2)
        return 255-signal if self.ink_polarity=='dark' else signal

    def _reference_material_topology(self):
        """Paths may be holes or separate arcs of the SAME ink component."""
        values=self.ink[self.mask>0].astype(np.uint8)
        threshold,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary=(self.ink>threshold).astype(np.uint8)
        _,self.reference_components,_,_=cv2.connectedComponentsWithStats(binary,8)
        _,background,bg_stats,_=cv2.connectedComponentsWithStats(1-binary,8)
        border_ids=set(np.concatenate([background[0],background[-1],background[:,0],background[:,-1]]))
        self.reference_path_components={};self.reference_holes={}
        offsets=np.array([(x,y)for x in range(-2,3)for y in range(-2,3)])
        for path in self.paths:
            name=path['path_id'];points=np.array(path['pixels'],float)
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            along=np.arange(0,distance[-1]+.01,1.)
            q=np.column_stack([np.interp(along,distance,points[:,k])for k in(0,1)])
            near=np.rint(q[:,None]+offsets).astype(int).reshape(-1,2)
            near=near[(near[:,0]>=0)&(near[:,1]>=0)&(near[:,0]<binary.shape[1])&(near[:,1]<binary.shape[0])]
            ids,counts=np.unique(self.reference_components[near[:,1],near[:,0]],return_counts=True)
            good=ids>0
            if not good.any():continue
            self.reference_path_components[name]=int(ids[good][np.argmax(counts[good])])
            ids,counts=np.unique(background[near[:,1],near[:,0]],return_counts=True);good=ids>0
            if not good.any():continue
            bg=int(ids[good][np.argmax(counts[good])])
            # Exterior arcs do not require a closed hole. A true enclosed
            # background region must still exist in the current exposure.
            if bg not in border_ids:
                region=np.zeros(binary.shape,np.uint8)
                cv2.fillPoly(region,[np.rint(points).astype(np.int32)],1)
                hole=background==bg
                if np.count_nonzero(hole&(region>0))>.5*np.count_nonzero(hole):
                    self.reference_holes[name]=hole.astype(np.uint8)
        self._source_contour_topology()

    def _source_contour_topology(self):
        """Keep source-declared closed contours/holes separate from RGB bridges.

        Some source atlases carry exact observed contours through an asset proof.
        Read topology only after matching its ORIGINAL source pixels and image
        SHA. Material/template UV is never read as reference pixels. Without a
        bound declaration, the actual reference RGB component remains the basis.
        """
        hints={};documents={};evidence={}
        for path in self.paths:
            name=path['path_id'];hint={k:path[k] for k in
                ('source_closed','role','parent_contour_key','semantic_glyph_id') if k in path}
            proof=path.get('proof',{});asset=proof.get('asset')
            if proof.get('kind')=='current_native_shape_source' and asset:
                location=asset.get('path') if isinstance(asset,dict) else asset
                if location not in documents:
                    p=Path(location)
                    documents[location]=(p.read_bytes() if p.is_file() else None)
                raw=documents[location]
                if raw is not None:
                    digest=hashlib.sha256(raw).hexdigest()
                    if not isinstance(asset,dict) or asset.get('sha256',digest)==digest:
                        doc=json.loads(raw)
                        source_path=next((q for t in doc.get('templates',[]) for q in t['paths'] if q['id']==name),None)
                        if source_path is not None:
                            origin=source_path.get('shape_provenance',{});boundary=origin.get('source_native_boundary')
                            frame=origin.get('source_frame',{})
                            exact=boundary is not None and np.array_equal(np.asarray(boundary),np.asarray(path['pixels']))
                            repeated_close=(source_path.get('closed') is True and boundary is not None
                                and len(path['pixels'])==len(boundary)+1
                                and np.array_equal(np.asarray(boundary),np.asarray(path['pixels'][:-1]))
                                and np.array_equal(np.asarray(boundary[0]),np.asarray(path['pixels'][-1])))
                            if ((exact or repeated_close)
                                    and frame.get('sha256')==self.atlas.get('image_sha256')):
                                hint.update(source_closed=source_path.get('closed'),role=source_path.get('role'),
                                    parent_contour_key=source_path.get('parent_contour_key'))
                                evidence[name]=dict(asset=location,asset_sha256=digest,
                                    exact_source_pixels_and_image_SHA_verified=True,
                                    atlas_repeats_explicitly_closed_source_endpoint=bool(repeated_close))
            if hint:hints[name]=hint
        names={p['path_id']:p for p in self.paths};roots={}
        def root(name,seen=()):
            if name in seen:raise ValueError('Cyclic source contour parent topology')
            hint=hints.get(name,{})
            if hint.get('semantic_glyph_id') is not None:return ('explicit',hint['semantic_glyph_id'])
            parent=hint.get('parent_contour_key')
            if parent in names:return root(parent,seen+(name,))
            if (hint.get('source_closed') is True and hint.get('role') in
                    ('printed_ink_external_boundary','letter_outer_boundary','letter_material_boundary')):
                return ('source_outer',name)
            return None
        for name in names:
            value=root(name)
            if value is not None:roots.setdefault(value,[]).append(name)
        self.source_declared_groups=set();next_id=int(self.reference_components.max())+1
        for key,members in roots.items():
            region=np.zeros(self.ink.shape,np.uint8);holes=[]
            for name in members:
                hint=hints.get(name,{});path=names[name]
                if hint.get('source_closed') is True:
                    contour=np.zeros(self.ink.shape,np.uint8)
                    cv2.fillPoly(contour,[np.rint(path['pixels']).astype(np.int32)],1)
                    if hint.get('parent_contour_key') in names:
                        holes.append((name,contour))
                    else:region|=contour
                elif key[0]=='explicit' and name in self.reference_path_components:
                    region|=(self.reference_components==self.reference_path_components[name]).astype(np.uint8)
            if not np.any(region):continue
            for name,hole in holes:
                region[hole>0]=0;self.reference_holes[name]=hole
            self.reference_components[region>0]=next_id
            for name in members:self.reference_path_components[name]=next_id
            self.source_declared_groups.add(next_id);next_id+=1
        used=set(self.reference_path_components.values())
        self.reference_components[~np.isin(self.reference_components,list(used))]=0
        self.reference_topology_evidence=evidence

    def register(self, image, exclusion, initial_H):
        corners=project(initial_H,self.points)
        lo=np.maximum(np.floor(corners.min(0)-20).astype(int),[0,0])
        hi=np.minimum(np.ceil(corners.max(0)+21).astype(int),[image.shape[1],image.shape[0]])
        if np.any(hi-lo<12):return dict(status='outside_image')
        shift=np.eye(3);shift[:2,2]=-lo
        size=tuple(hi-lo)
        target=self.ink_signal(image[lo[1]:hi[1],lo[0]:hi[0]]).astype(np.float32)
        warped=cv2.warpPerspective(self.ink,shift@initial_H,size)
        valid=cv2.warpPerspective(self.mask,shift@initial_H,size,flags=cv2.INTER_NEAREST)
        valid=cv2.erode(valid,np.ones((5,5),np.uint8))
        valid[cv2.dilate(exclusion,np.ones((7,7),np.uint8))[lo[1]:hi[1],lo[0]:hi[0]]>0]=0
        if np.count_nonzero(valid)<100:return dict(status='occluded_or_clipped')
        best=None;attempts=[];geometry_attempts=[]
        for width in (1,5,9,17,25,33,49,65):
            blurred=cv2.blur(warped,(width,1))
            try:
                cc,E=cv2.findTransformECC(blurred,target,np.eye(3,dtype=np.float32),cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-6),valid,5)
            except cv2.error:
                attempts.append([width,None]);continue
            H=np.linalg.inv(shift)@E@shift@initial_H
            # Reject folded/collapsed search solutions; this does not admit a measurement.
            probes=np.array([[0,0],[size[0],0],list(size),[0,size[1]]],float)
            warped_probes=project(E.astype(float),probes)
            area=cv2.contourArea(warped_probes.astype(np.float32),oriented=True)
            shape_ok=np.isfinite(H).all() and .25*np.prod(size)<area<4*np.prod(size)
            geometry=registration_geometry_check(H,initial_H,self.points,warp_roi=(lo,hi))
            geometry_attempts.append(dict(horizontal_blur_px=width,valid=geometry['valid'],
                reasons=geometry['reasons'],
                shape_denominator_range=geometry.get('registered_shape',{}).get('denominator_range'),
                relative_determinant_range=(geometry.get('relative_shape')or{}).get('local_determinant_range'),
                warp_ROI_denominator_range=geometry.get('warp_ROI',{}).get('denominator_range')))
            shape_ok=shape_ok and geometry['valid']
            attempts.append([width,float(cc),bool(shape_ok)])
            if shape_ok and (best is None or cc>best[0]):best=(float(cc),H,width,geometry)
        if best is None:return dict(status='registration_failed',attempts=attempts,geometry_attempts=geometry_attempts)
        cc,H,width,geometry=best
        # A whole-word score alone can be high under extreme blur. Require several
        # separate glyph regions to have actual image variation and agreement.
        refwarp=cv2.blur(cv2.warpPerspective(self.ink,shift@H,size),(width,1))
        glyphs=[];names=set()
        for p in self.paths:
            q=project(shift@H,p['pixels']);a=np.maximum(np.floor(q.min(0)-3).astype(int),[0,0]);b=np.minimum(np.ceil(q.max(0)+4).astype(int),size)
            if np.any(b-a<3):continue
            ok=valid[a[1]:b[1],a[0]:b[0]]>0
            if ok.sum()<20 or ok.mean()<.5:continue
            x=refwarp[a[1]:b[1],a[0]:b[0]][ok];y=target[a[1]:b[1],a[0]:b[0]][ok]
            sx=float(x.std());sy=float(y.std());corr=float(np.corrcoef(x,y)[0,1]) if min(sx,sy)>3 else -1.
            glyphs.append(dict(path_id=p['path_id'],correlation=corr,current_ink_signal_std=sy))
            if corr>.45:names.add(p['path_id'])
        resolved,structure=self.resolved_glyphs(image,exclusion,H,horizontal_blur_px=width)
        reference_groups=set(self.reference_path_components.values())
        supported_groups={self.reference_path_components[name] for name in names
            if name in self.reference_path_components}
        supported_groups&={self.reference_path_components[name] for name in resolved}
        required_groups=min(3,len(reference_groups))
        qualified=(cc>.43 and required_groups>0 and len(supported_groups)>=required_groups
            and sum(g['correlation']>.45 for g in glyphs)>=.5*len(glyphs))
        representatives=[]
        for identity in sorted(supported_groups):
            members=[p['path_id'] for p in self.paths if self.reference_path_components.get(p['path_id'])==identity]
            representatives.append(next((name for name in members if name not in self.reference_holes),members[0]))
        return dict(status='source_pattern_registered' if qualified else 'whole_pattern_unresolved',H=H.tolist(),
            correlation=cc,shared_horizontal_blur_px=width,supported_letters=representatives,
            supported_paths=sorted(names),glyph_checks=glyphs,
            semantic_identity_check=dict(reference_material_groups=len(reference_groups),
                required_visible_material_groups=required_groups,supported_material_groups=len(supported_groups),
                holes_are_not_additional_glyphs=True,current_source_structure=structure),
            attempts=attempts,geometry_attempts=geometry_attempts,registration_geometry=geometry,
            search_initialization_is_not_identity_evidence=True)

    def read_paths(self,image,exclusion,registration,binding,spacing=4.):
        if registration['status']!='source_pattern_registered':return [],dict(reason=registration['status'])
        H=np.asarray(registration['H'])
        # Archived registration status is not a geometry certificate. Recheck
        # the finite reference at the readout boundary before sampling RGB.
        geometry=registration_geometry_check(H,np.eye(3),self.points,
            warp_roi=(self.points.min(0),self.points.max(0)))
        if not geometry['valid'] or registration.get('registration_geometry',{}).get('valid') is False:
            return [],dict(reason='current_registration_geometry_invalid',geometry=geometry)
        inverse=np.linalg.inv(H);query=[];normals=[];labels=[]
        resolved,structure=self.resolved_glyphs(image,exclusion,H,
            horizontal_blur_px=registration['shared_horizontal_blur_px'])
        reference_groups=set(self.reference_path_components.values())
        resolved_groups={self.reference_path_components[name] for name in resolved}
        required_groups=min(3,len(reference_groups))
        if not required_groups or len(resolved_groups)<required_groups:
            return [],dict(reason='current_source_semantic_identity_unresolved',
                required_visible_material_groups=required_groups,
                resolved_material_groups=len(resolved_groups),structure=structure)
        exclusion=cv2.dilate(exclusion,np.ones((7,7),np.uint8))
        for p in self.paths:
            if p['path_id'] not in resolved:continue
            points=project(H,p['pixels']);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            if length[-1]<4:continue
            t=np.arange(1,length[-1]-1,spacing)
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
                source_profile_flags=flags,current_source_RGB_measured=True,model_UV_used_as_measurements=False))
        return accepted,dict(queries=len(query),accepted=len(accepted),reasons=reasons,current_ink_class_boundary=float(boundary),structure=structure)

    def resolved_glyphs(self,image,exclusion,H,*,horizontal_blur_px=1):
        """Current ink components must separate letters before contour admission.

        Registration can recognize a whole blurred word whose individual contours
        have disappeared. One merged bright band must never stand for many letters.
        Binary components establish only visible structure; UV still comes from RGB.
        """
        xy=project(H,self.points);lo=np.maximum(np.floor(xy.min(0)-5).astype(int),[0,0]);hi=np.minimum(np.ceil(xy.max(0)+6).astype(int),[image.shape[1],image.shape[0]])
        if np.any(hi-lo<3):return set(),dict(reason='outside_source')
        signal=self.ink_signal(image[lo[1]:hi[1],lo[0]:hi[0]])
        hidden=cv2.dilate(exclusion,np.ones((7,7),np.uint8))[lo[1]:hi[1],lo[0]:hi[0]]>0
        values=signal[~hidden]
        if len(values)<20:return set(),dict(reason='no_current_material')
        boundary,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary=((signal>boundary)&~hidden).astype(np.uint8)
        count,components,stats,_=cv2.connectedComponentsWithStats(binary,8)
        shift=np.eye(3);shift[:2,2]=-lo
        clean_reference=cv2.warpPerspective(self.reference_components.astype(np.float32),shift@H,tuple(hi-lo),flags=cv2.INTER_NEAREST).astype(np.int32)
        width=max(1,int(horizontal_blur_px))
        photometry=None
        if width>1:
            rendered=cv2.blur(cv2.warpPerspective(self.ink,shift@H,tuple(hi-lo)),(width,1))
            # Compare structure at the SAME measured material split. Separately
            # thresholding a clean reference can reconnect faint blurred strokes
            # which are below the current exposure's ink/background boundary.
            # Only a whole-pattern positive affine exposure change is fitted;
            # there are no per-glyph thresholds or invented source edges.
            x=rendered[~hidden].astype(np.float64);y=signal[~hidden].astype(np.float64)
            xc=x-x.mean();variance=float(np.mean(xc*xc))
            scale=float(np.mean(xc*(y-y.mean()))/variance) if variance>0 else 0.
            offset=float(y.mean()-scale*x.mean())
            if not np.isfinite([scale,offset]).all() or scale<=0:
                return set(),dict(reason='whole_pattern_photometry_unresolved')
            split=(boundary-offset)/scale
            photometry=dict(scale=scale,offset=offset,reference_threshold=float(split),
                current_threshold=float(boundary),scope='one_registered_whole_pattern')
            # Blur changes observed support, never semantic ownership. Mapping
            # every clean letter to a merged blurred component would erase the
            # very distinct-letter conflict that the current source must pass.
            reference=np.zeros(clean_reference.shape,np.int32)
            nearest=np.full(clean_reference.shape,np.inf,np.float32)
            for reference_id in sorted(set(self.reference_path_components.values())):
                distance=cv2.distanceTransform((clean_reference!=reference_id).astype(np.uint8),cv2.DIST_L2,3)
                closer=distance<nearest;ties=distance==nearest
                reference[closer]=reference_id;reference[ties]=0
                nearest=np.minimum(nearest,distance)
            support=cv2.blur((clean_reference>0).astype(np.float32),(width,1))>0
            reference[~((rendered>split)&support)]=0
        else:reference=clean_reference
        groups={}
        for name,reference_id in self.reference_path_components.items():
            groups.setdefault(reference_id,[]).append(name)
        path_groups={name:reference_id for reference_id,names in groups.items()for name in names}
        assignment={};owned={};checks={};resolved=set();group_checks={};visibility={};hole_checks={}
        for reference_id,names in groups.items():
            expected=(reference==reference_id)&~hidden
            # The same source exclusion may split one real letter into several
            # visible pieces. Match each visible reference piece to current ink;
            # never use the hidden reference pixels as source material evidence.
            piece_count,pieces=cv2.connectedComponents(expected.astype(np.uint8),8)
            matched=[]
            for piece in range(1,piece_count):
                pixels=components[(pieces==piece)&(components>0)]
                if len(pixels)<3:continue
                ids,counts=np.unique(pixels,return_counts=True)
                if reference_id in self.source_declared_groups:matched.extend(ids.astype(int).tolist())
                else:matched.append(int(ids[np.argmax(counts)]))
            if not matched:group_checks[reference_id]='no_current_ink_component';continue
            assignment[reference_id]=tuple(sorted(set(matched)))
            for component in assignment[reference_id]:owned.setdefault(component,[]).append(reference_id)
        foreground_border=cv2.dilate(hidden.astype(np.uint8),np.ones((3,3),np.uint8))>0
        for reference_id,matched in assignment.items():
            actual=np.isin(components,matched);whole_expected=reference==reference_id
            expected=whole_expected&~hidden
            intersection=np.count_nonzero(actual&expected);union=np.count_nonzero(actual|expected)
            cut=bool(np.any(whole_expected&hidden))
            visibility[str(reference_id)]=dict(current_components=list(matched),
                reference_pixels=int(np.count_nonzero(whole_expected)),
                visible_reference_pixels=int(np.count_nonzero(expected)),
                current_pixels=int(np.count_nonzero(actual)),visible_shape_iou=intersection/max(1,union),
                visible_reference_bbox_xywh=list(cv2.boundingRect(expected.astype(np.uint8))),
                current_component_bboxes_xywh=[stats[c,:4].tolist() for c in matched],
                reference_cut_by_foreground=cut)
            if any(len(owned[component])>1 for component in matched):
                group_checks[reference_id]='current_ink_merges_distinct_letters';continue
            if np.any(actual&foreground_border) and not cut:
                # A source component growing into foreground where this
                # reference letter is not cut may be a hidden merged band.
                group_checks[reference_id]='current_component_cut_by_foreground';continue
            if intersection/max(1,union)<.5:
                group_checks[reference_id]='current_ink_shape_not_separated_at_reference_letter';continue
            group_checks[reference_id]=('separate_visible_current_ink_components' if cut
                else 'separate_current_ink_component')
        for path in self.paths:
            name=path['path_id'];reference_id=path_groups.get(name)
            reason=group_checks.get(reference_id,'reference_ink_component_unresolved');checks[name]=reason
            if reason not in ('separate_current_ink_component','separate_visible_current_ink_components'):continue
            if name in self.reference_holes:
                expected=cv2.warpPerspective(self.reference_holes[name],shift@H,tuple(hi-lo),flags=cv2.INTER_NEAREST)>0
                actual=np.zeros(binary.shape,np.uint8)
                partial=bool(np.any(expected&foreground_border))
                if partial:
                    # An actual exclusion can open an enclosed hole. Compare
                    # only its surviving background inside the source hole's
                    # search box; a filled/missing hole still supplies no match.
                    yy,xx=np.nonzero(expected)
                    if len(xx):
                        region=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
                        actual[region]=((signal[region]<=boundary)&~hidden[region]).astype(np.uint8)
                    expected=expected&~hidden
                else:
                    current=np.isin(components,assignment[reference_id]).astype(np.uint8)
                    contours,hierarchy=cv2.findContours(current,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    if hierarchy is not None:
                        cv2.drawContours(actual,[c for c,h in zip(contours,hierarchy[0])if h[3]>=0],-1,1,-1)
                intersection=np.count_nonzero((actual>0)&expected);union=np.count_nonzero((actual>0)|expected)
                hole_checks[name]=dict(partial_source_visibility=partial,
                    visible_reference_pixels=int(np.count_nonzero(expected)),
                    current_pixels=int(np.count_nonzero(actual)),visible_hole_iou=intersection/max(1,union))
                if intersection/max(1,union)<.35:
                    checks[name]='current_hole_unresolved';continue
                checks[name]=('current_visible_hole_separated' if partial else 'current_closed_hole_separated')
            resolved.add(name)
        return resolved,dict(current_ink_threshold=float(boundary),path_checks=checks,
            merged_components={str(k):[groups[r][0]for r in v]for k,v in owned.items()if len(v)>1},
            reference_component_paths={str(k):v for k,v in groups.items()},
            component_visibility_checks=visibility,hole_visibility_checks=hole_checks,
            photographic_horizontal_blur_px=width,
            photographic_material_split=photometry,
            source_declared_topology_evidence=self.reference_topology_evidence,
            blur_preserves_clean_semantic_groups=True,
            blur_can_join_reference_parts_but_does_not_create_source_edges=True,
            binary_used_only_for_structure_not_measured_UV=True)
