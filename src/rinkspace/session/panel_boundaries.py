"""Read a colored panel's exterior from RGB, including blurred vertical edges.

The filled color component separates the exterior from lettering and QR holes.
It supplies topology only; original RGB interface quantiles supply every UV.
"""
import cv2
import numpy as np
from rinkspace.session.projected_curve_queries import projected_curve_queries
from rinkspace.vision_programme_ice_review import _batched_actual_profiles, _actual_profile_candidates
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.session.material_tracks import qualify_line_candidates
from rinkspace.session.native_shape_guidance import register_native_shapes


def _qualify_closed_outline(measurement, curve, path, exclusion):
    """Organize measured silhouette pixels with the actual closed native shape.

    A local similarity is source-search geometry only. Every emitted pixel remains
    an original RGB interface quantile. A line in an unrolled normal strip is
    not a valid shape model for a translated closed contour.
    """
    queries=np.flatnonzero(measurement['query_curve']==curve)
    ids=np.flatnonzero(measurement['query_curve'][measurement['candidate_query']]==curve)
    report=dict(curve=int(curve),curve_geometry='native_closed_outline',
        selected_candidate_ids=[],source_uv_unchanged=True,native_fit_residual_used=False,
        organizing_shape_is_not_output_measurement=True)
    def missing(reason):return [],dict(report,disposition=reason)
    if not len(queries) or len(ids)<7:return missing('insufficient_closed_outline_evidence')
    query=measurement['candidate_query'][ids]
    if len(np.unique(query))!=len(query):return missing('competing_closed_outline_source_readings')
    path=np.asarray(path,float)
    if not np.isfinite(path).all():return missing('incomplete_native_outline_navigation')
    if np.linalg.norm(path[-1]-path[0])>1e-8:path=np.vstack([path,path[0]])
    uv=measurement['candidate_uv'][ids]
    try:
        registered=register_native_shapes(uv,np.full(len(uv),'outline'),{'outline':path},transform_model='similarity')
    except ValueError as error:
        return missing('unresolved_closed_outline_shape: '+str(error))
    if np.linalg.det(registered.affine[:2,:2])<=0:return missing('reversed_closed_outline_shape')
    # Match the existing source-line organization's 3px geometric resolution.
    # Wider actual RGB transition intervals may organize a blurred interface;
    # they do not become weights or tolerances in the native fitting problem.
    quantiles=measurement['candidate_side_quantiles_uv'][ids,0]
    normal=measurement['query_normal'][query]
    intervals=np.sort(np.einsum('nij,nj->ni',quantiles[:,[0,2]]-uv[:,None],normal),axis=1)
    width=np.maximum(3.,np.max(np.abs(intervals),axis=1))
    keep=registered.matches.distance_px<=width
    kept=ids[keep];good_queries=set(map(int,measurement['candidate_query'][kept]))
    positions=measurement['query_uv'][queries]
    pairs=list(zip(range(len(queries)-1),range(1,len(queries))))
    h,w=exclusion.shape
    complete=((path>=0).all() and (path<[w,h]).all())
    if complete:pairs.append((len(queries)-1,0))
    visible=0.;supported=0.
    for a,b in pairs:
        length=float(np.linalg.norm(positions[b]-positions[a]))
        visible+=length
        if int(queries[a]) in good_queries and int(queries[b]) in good_queries:supported+=length
    report.update(source_affine=registered.affine.tolist(),source_transform_model='similarity',source_shape_rms_px=registered.rms_distance_px,
        source_shape_distance_px=registered.matches.distance_px.tolist(),source_candidate_ids=ids.tolist(),
        visible_query_length_px=visible,supported_query_length_px=supported,
        source_shape_fit_stop=registered.stop_reason,actual_source_quantiles_used=True)
    # Require distributed evidence in addition to the low-dimensional source
    # registration, rather than accepting a small point cluster.
    if len(kept)<7 or visible<=0 or supported<.5*visible:
        return missing('insufficient_distributed_closed_outline_support')
    report.update(disposition='source_closed_outline_supported',selected_candidate_ids=kept.tolist())
    return kept.tolist(),report


def _color_filled_components(binary,available,lo,hi,image_hw):
    """Fill bounded interiors using reliable ROI background, never frame edges."""
    height,width=image_hw
    background=(binary==0)&available
    seeds=np.zeros(binary.shape,bool)
    if lo[0]>0:seeds[:,0]=background[:,0]
    if hi[0]<width:seeds[:,-1]=background[:,-1]
    if lo[1]>0:seeds[0]=background[0]
    if hi[1]<height:seeds[-1]=background[-1]
    if lo[0]==0:seeds[:,0]=False
    if hi[0]==width:seeds[:,-1]=False
    if lo[1]==0:seeds[0]=False
    if hi[1]==height:seeds[-1]=False
    if not seeds.any():return None
    _,labels=cv2.connectedComponents(background.astype('uint8'),connectivity=4)
    outside_labels=np.unique(labels[seeds]);outside_labels=outside_labels[outside_labels!=0]
    if not len(outside_labels):return None
    outside=np.isin(labels,outside_labels)
    # Masked pixels are unknown: they supply neither outside connectivity nor
    # material support. The original mask/probe checks still guard every UV.
    filled=(~outside)&available
    _,components=cv2.connectedComponents(filled.astype('uint8'),connectivity=8)
    return components



def _source_supported_visibility(measurement, reports, exclusion):
    """Use an actual source-line hypothesis only to locate missing masked spans.

    Wide normal searches can remain partly unmasked even when the physical edge
    is hidden. An RGB-supported line locates that edge for the mask query. It
    never supplies observations or hides a query with a measured candidate.
    """
    points=np.asarray(measurement['query_uv']);normals=np.asarray(measurement['query_normal'])
    intervals=np.asarray(measurement['query_interval']);support=np.ones(len(points),bool)
    candidate_query=np.asarray(measurement['candidate_query'],int)
    candidate_flags=np.asarray(measurement.get('candidate_flags',np.zeros(len(candidate_query),int)),int)
    # Some readers retain rejected raw RGB alternatives for evidence. Those
    # are not admitted measurements; every clean candidate still protects its
    # query from being removed by the inferred visibility check.
    measured=set(map(int,candidate_query[(candidate_flags&3)==0]));inferred={}
    for report in reports:
        organizer=report.get('source_organizer')
        if report.get('disposition')!='insufficient_distributed_source_line_span' or organizer is None:continue
        if organizer['competing_family_support_length_px']>0:continue
        queries=np.asarray(report['visible_source_queries'],int)
        if len(queries)<2:continue
        positions=points[queries];arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
        slope,intercept=organizer['coefficients']
        offset=slope*(arc-organizer['station_center'])+intercept
        source_line=positions+normals[queries]*offset[:,None]
        ij=np.rint(source_line).astype(int)
        inside=(ij[:,0]>=0)&(ij[:,0]<exclusion.shape[1])&(ij[:,1]>=0)&(ij[:,1]<exclusion.shape[0])
        inside&=(offset>=intervals[queries,0])&(offset<=intervals[queries,1])
        hidden=[];tolerance=float(report.get('source_geometry_tolerance_px',3.))
        for local in np.flatnonzero(inside):
            qi=int(queries[local]);x,y=ij[local]
            if qi in measured or not exclusion[y,x]:continue
            # The complete permitted source-line center halo, including pixel
            # interpolation contributors, must be masked. The broad navigation
            # search strip is not the physical edge's visibility footprint.
            halo=source_line[local]+normals[qi]*np.linspace(-tolerance,tolerance,max(3,int(np.ceil(4*tolerance))+1))[:,None]
            lower=np.floor(halo).astype(int);upper=np.ceil(halo).astype(int)
            if np.any(lower<0)or np.any(upper>np.array(exclusion.shape[::-1])-1):continue
            if all(exclusion[yy[:,1],xx[:,0]].all()for yy in(lower,upper)for xx in(lower,upper)):
                support[qi]=False;hidden.append(qi)
        if hidden:inferred[int(report['curve'])]=hidden
    return support,inferred


def read_panel_boundary(rgb, projected_paths, exclusion, *, binding, search_radius=96., spacing=24.,
                        material_signal='dark', closed_paths=None):
    if material_signal not in ('dark','red','blue'):raise ValueError('Panel material_signal must be dark, red or blue')
    height,width=rgb.shape[:2]
    paths=[]
    for path in projected_paths:
        path=np.asarray(path)
        if not path.size:path=np.empty((0,2))
        if path.ndim!=2 or path.shape[1]!=2:raise ValueError('Projected panel paths must have shape (N, 2)')
        paths.append(path)
    projected_paths=paths
    closed_paths=[False]*len(paths) if closed_paths is None else list(closed_paths)
    if len(closed_paths)!=len(paths):raise ValueError('One closed-path flag is required per projected path')
    finite=[p[np.isfinite(p).all(1)] for p in projected_paths]
    if not any(len(p) for p in finite):return [],dict(status='outside_source')
    all_uv=np.vstack(finite)
    lo=np.maximum(np.floor(all_uv.min(0)-search_radius).astype(int),[0,0])
    hi=np.minimum(np.ceil(all_uv.max(0)+search_radius).astype(int),[width,height])
    if np.any(hi-lo<8):return [],dict(status='outside_source')
    # The dark panel and its monochrome QR area share the same light surround.
    # A chroma-only component would incorrectly cut the QR block off the panel.
    # Red excess selects a dark component; it must not locate the boundary,
    # because broadcast chroma can lose the fine luminance outline.
    signal=rgb.min(2).astype(float)
    patch=signal[lo[1]:hi[1],lo[0]:hi[0]]
    available=~exclusion[lo[1]:hi[1],lo[0]:hi[0]]
    values=patch[available]
    if len(values)<20 or np.ptp(values)<4:return [],dict(status='panel_material_unresolved')
    threshold,_=cv2.threshold(np.clip(values,0,255).astype('uint8').reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary=((patch<=threshold)&available).astype('uint8')
    color_seed=None
    topology_method=(material_signal+'_source_flood_exterior' if material_signal in ('red','blue')
                     else 'registered_source_components_envelope')
    source_component_count=None
    if material_signal in ('red','blue'):
        channel=0 if material_signal=='red' else 2
        other=[i for i in range(3) if i!=channel]
        excess=rgb[...,channel].astype(float)-rgb[...,other].max(2)
        excess=excess[lo[1]:hi[1],lo[0]:hi[0]]
        color_threshold,_=cv2.threshold(np.clip(excess[available],0,255).astype('uint8').reshape(-1,1),
            0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        color_seed=(excess>color_threshold)&available
    expected=np.zeros(binary.shape,'uint8')
    hull=cv2.convexHull((all_uv-lo).astype('float32'))
    cv2.fillConvexPoly(expected,np.rint(hull).astype('int32'),1)
    if color_seed is not None:
        components=_color_filled_components(binary,available,lo,hi,(height,width))
        if components is None:return [],dict(status='panel_topology_unresolved')
        count=int(components.max())+1
        if count<=1:return [],dict(status='no_source_panel_component')
        area=np.bincount(components[expected!=0],minlength=count)
        seed_area=np.bincount(components[(expected!=0)&color_seed],minlength=count)
        selected=max(range(1,count),key=lambda i:(seed_area[i],area[i]))
        if not seed_area[selected]:return [],dict(status='no_source_'+material_signal+'_panel_component')
        component=(components==selected).astype('uint8')
        material_component=component
    else:
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:return [],dict(status='no_source_panel_component')
        candidates=[]
        for contour in contours:
            region=np.zeros(binary.shape,'uint8');cv2.fillPoly(region,[contour],1)
            overlap=int(np.count_nonzero(region&expected))
            candidates.append((overlap,int(np.count_nonzero(region)),contour,region))
        # A person, white lettering or a white gap can divide one physical
        # rectangular graphic into several RGB components. Retain the source
        # pieces predominantly inside its registered footprint, not just the
        # largest piece. The convex envelope supplies exterior topology only.
        # Inward material support must come from the actual hole-filled pieces:
        # an open white gap between them is not material inside the envelope.
        pieces=[(contour,region) for overlap,area,contour,region in candidates if overlap>area-overlap]
        source_component_count=len(pieces)
        if pieces:
            envelope=cv2.convexHull(np.concatenate([contour for contour,_ in pieces],axis=0))
            component=np.zeros(binary.shape,'uint8');cv2.fillConvexPoly(component,envelope,1)
            material_component=np.maximum.reduce([region for _,region in pieces])
        else:
            _,_,_,component=max(candidates,key=lambda x:x[0])
            material_component=component
    # Clipped and occluded boundaries cannot create source measurements.
    exterior=np.zeros((height,width),'uint8');exterior[lo[1]:hi[1],lo[0]:hi[0]]=component
    material=np.zeros((height,width),'uint8');material[lo[1]:hi[1],lo[0]:hi[0]]=material_component&available
    q=[];normal=[];interval=[];curve=[]
    source_query_translation=np.zeros(2)
    if material_signal=='blue':
        # A blurred blue shield can be visibly displaced from the current
        # camera prediction. Its measured filled footprint relocates the
        # search chart; neither its centroid nor hull supplies observations.
        actual_moment=cv2.moments(component,binaryImage=True)
        expected_moment=cv2.moments(expected,binaryImage=True)
        if actual_moment['m00'] and expected_moment['m00']:
            source_query_translation=np.array([
                actual_moment[key]/actual_moment['m00']-expected_moment[key]/expected_moment['m00']
                for key in ('m10','m01')])
    center=np.mean(all_uv,axis=0)+source_query_translation
    for ci,original_path in enumerate(projected_paths):
        path=original_path+source_query_translation
        queries=projected_curve_queries(path,np.isfinite(path).all(1),(height,width),spacing,search_radius,
            center_uv=center,closed=closed_paths[ci])
        q.extend(queries['points_uv']);normal.extend(queries['normals']);interval.extend(queries['intervals']);curve.extend([ci]*len(queries['points_uv']))
    if not q:return [],dict(status='no_visible_panel_search')
    q=np.asarray(q);normal=np.asarray(normal);interval=np.asarray(interval);curve=np.asarray(curve)
    frame=_MaterialProfileFrame(np.ascontiguousarray(rgb),binding)
    offsets,profiles,smooth,valid=_batched_actual_profiles(frame,q,normal,interval,np.full(len(q),'dark'))
    uv=[];sides=[];quantiles=[];candidate_query=[];flags=[]
    profile_debug=[]
    for qi in range(len(q)):
        points=_actual_profile_candidates(q[qi],normal[qi],offsets[qi,valid[qi]],profiles[qi,valid[qi]],smooth[qi,valid[qi]],
            operator='material_interface',signal='dark',transition='dark_to_light',with_quantiles=True,
            separate_bands=True)
        if curve[qi] in (1,3):profile_debug.append(dict(curve=int(curve[qi]),query=q[qi].tolist(),normal=normal[qi].tolist(),
            candidates=[dict(uv=p.tolist(),cropped=bool(b),width=float(np.linalg.norm(qq[0,2]-qq[0,0]))) for p,_,b,qq in points]))
        admitted=[]
        for point,pair,bad,qq in points:
            if bad:continue
            ext=max(4.,float(np.linalg.norm(qq[0,2]-qq[0,0]))+3.)
            probes=point+np.array([-ext,ext])[:,None]*normal[qi]
            ij=np.rint(probes).astype(int)
            if np.any(ij<0) or np.any(ij>=[width,height]):continue
            segment=point+np.linspace(-ext,ext,max(3,int(2*ext)+1))[:,None]*normal[qi]
            xy=np.rint(segment).astype(int)
            if exclusion[xy[:,1],xy[:,0]].any():continue
            if not material[ij[0,1],ij[0,0]] or exterior[ij[1,1],ij[1,0]]:continue
            admitted.append((point,pair,qq))
        if color_seed is not None and admitted:
            admitted=[max(admitted,key=lambda item:float(item[0]@normal[qi]))]
        for point,pair,qq in admitted:
            uv.append(point);sides.append(pair);quantiles.append(qq);candidate_query.append(qi);flags.append(0)
    if not uv:return [],dict(status='source_exterior_not_resolved')
    measurement=dict(query_uv=q,query_normal=normal,query_interval=interval,query_curve=curve,
        candidate_uv=np.asarray(uv),candidate_sides=np.asarray(sides),candidate_side_quantiles_uv=np.asarray(quantiles),
        candidate_query=np.asarray(candidate_query),candidate_flags=np.asarray(flags))
    line_indices=[i for i,closed in enumerate(closed_paths) if not closed]
    selected,reports=qualify_line_candidates(measurement,line_indices,exclusion,
        native_shape_guided=True,use_source_quantiles=False)
    support,inferred_occlusion=_source_supported_visibility(measurement,reports,exclusion)
    if inferred_occlusion:
        selected,reports=qualify_line_candidates(measurement,line_indices,exclusion,
            native_shape_guided=True,use_source_quantiles=False,source_support_queries=support)
        for report in reports:
            report['source_line_masked_queries']=inferred_occlusion.get(int(report['curve']),[])
            report['visibility_basis']='actual source-line organizer intersected with exclusion mask; no new UV'
    selected=selected.tolist()
    for ci,closed in enumerate(closed_paths):
        if closed:
            kept,report=_qualify_closed_outline(measurement,ci,projected_paths[ci]+source_query_translation,exclusion)
            selected.extend(kept);reports.append(report)
    rows=[dict(curve=int(curve[candidate_query[i]]),uv=np.asarray(uv[i]).tolist(),
        normal=normal[candidate_query[i]].tolist(),actual_source_quantiles=np.asarray(quantiles[i]).tolist(),
        model_UV_synthesized=False,measurement_operator='source_outer_panel_material_interface') for i in selected]
    return rows,dict(status='source_panel_exterior_read',threshold=float(threshold),candidate_count=len(uv),
        reports=reports,profile_debug=profile_debug,source_UV_unchanged=True,source_component_used_for_topology_only=True,
        source_component_method=topology_method,source_component_count=source_component_count,
        source_material_support_method='hole_filled_source_components_union',
        source_query_translation=source_query_translation.tolist(),source_query_translation_is_navigation_only=True)
