"""Read-only gray-contour competition review using current source stems.

No model coordinates, fitted residuals, or majority selection enter this
review. Current named stems must be separately witnessed in their own image.
Missing continuation remains pending; every original numeric array is kept.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.spatial import Delaunay,QhullError


def _tangent_interval(points,quantiles,axis,step):
    """Feasible local 2D side-path directions from native readout intervals."""
    normal=np.array([-axis[1],axis[0]])
    centers=np.asarray(quantiles,float).mean(1)
    # The actual two signed-edge gradient quartiles describe center readout
    # support. Physical stripe width never becomes center-error tolerance.
    x=points@axis;y=points@normal
    xq=centers@axis;yq=centers@normal
    xlo=np.minimum(x,xq.min(1))-.5*step*np.abs(axis).sum()
    xhi=np.maximum(x,xq.max(1))+.5*step*np.abs(axis).sum()
    ylo=np.minimum(y,yq.min(1))-.5*step*np.abs(normal).sum()
    yhi=np.maximum(y,yq.max(1))+.5*step*np.abs(normal).sum()
    lo=-np.inf;hi=np.inf;pairs=0
    order=np.argsort(x,kind='stable')
    for pos,i in enumerate(order):
        for j in order[pos+1:]:
            dxlo=xlo[j]-xhi[i];dxhi=xhi[j]-xlo[i]
            if dxlo<=0:continue
            bounds=[dy/dx for dy in (ylo[j]-yhi[i],yhi[j]-ylo[i]) for dx in (dxlo,dxhi)]
            lo=max(lo,min(bounds));hi=min(hi,max(bounds));pairs+=1
    return None if not pairs or not np.isfinite([lo,hi]).all() else [float(lo),float(hi)]


def _material_signature_intervals(colours):
    """RGB deficit ratios with the native 8-bit channel quantization bound."""
    deficit=colours[:,[0,2]]-colours[:,1,None]
    total=deficit.sum(2)
    lo=np.full_like(deficit,-np.inf);hi=np.full_like(deficit,np.inf)
    for i in range(len(colours)):
        for side in range(2):
            # Each exterior-minus-paint channel is a difference of two
            # quantized channel readings, so its support is +/- one unit.
            denlo=total[i,side]-3.;denhi=total[i,side]+3.
            if denlo<=0:continue
            for channel in range(3):
                candidates=[d/den for d in (deficit[i,side,channel]-1.,deficit[i,side,channel]+1.)
                    for den in (denlo,denhi)]
                lo[i,side,channel]=min(candidates);hi[i,side,channel]=max(candidates)
    return lo,hi


def _all_sample_pixels_supported(mask,positions,normals):
    """Check the exact 3x5 remap samples and nonzero bilinear contributors."""
    tangent=np.c_[-normals[:,1],normals[:,0]]
    samples=positions[:,:,None,:]+tangent[:,None,None,:]*np.arange(-2.,3.)[None,None,:,None]
    xy=np.rint(samples.astype(np.float32)*32.)/32.
    lower=np.floor(xy).astype(np.int64);upper=np.ceil(xy).astype(np.int64)
    valid=np.ones(len(positions),bool)
    for xs in (lower[...,0],upper[...,0]):
        for ys in (lower[...,1],upper[...,1]):
            inside=(xs>=0)&(ys>=0)&(xs<mask.shape[1])&(ys<mask.shape[0])
            supported=np.zeros(inside.shape,bool)
            supported[inside]=mask[ys[inside],xs[inside]]
            valid&=supported.all((1,2))
    return valid


def analyze_current_gray_native_contour_pixels(rgb,uv,sides,quantiles,*,source_valid_mask,source_axis,
        native_quantization_step_px,source_navigation_paths=(),native_rgb_field=None):
    """Read source geometry only; this function supplies no named identity.

    The axis and paths organize searches from source observations. The caller
    must establish named source lineage separately. All returned pixels,
    gradients, materials and continuation samples belong to this own image.
    """
    from rinkspace.vision_ice_model_context_identity import _actual_paint_colour_context
    rgb=np.asarray(rgb);points=np.asarray(uv,float);sides=np.asarray(sides,float);q=np.asarray(quantiles,float)
    n=len(points);mask=np.asarray(source_valid_mask,bool);axis=np.asarray(source_axis,float)
    if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or points.shape!=(n,2)
            or sides.shape!=(n,2,2) or q.shape!=(n,2,3,2) or mask.shape!=rgb.shape[:2]
            or not np.isfinite(points).all() or not np.isfinite(sides).all()
            or axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<=0
            or not np.isfinite(native_quantization_step_px) or native_quantization_step_px<=0):
        raise ValueError('Exact native RGB geometry and a source navigation axis required')
    axis=axis/np.linalg.norm(axis)
    q50_available=np.isfinite(q[:,:,1]).all((1,2))
    if not np.array_equal(points,sides.mean(1)) or not np.array_equal(sides[q50_available],q[q50_available,:,1]):
        raise ValueError('Use the literal original q50 center and its two measured interfaces')
    complete=np.isfinite(q).all((1,2,3));width=np.linalg.norm(sides[:,1]-sides[:,0],axis=1)
    normals=np.divide(sides[:,1]-sides[:,0],width[:,None],out=np.zeros((n,2)),where=width[:,None]>0)
    native_rgb_field=rgb.astype(np.float32) if native_rgb_field is None else native_rgb_field
    readable,colours,contrast,positions=_actual_paint_colour_context(rgb,points,sides,normals,'dark',native_rgb_field=native_rgb_field)
    valid=_all_sample_pixels_supported(mask,positions,normals)&complete&(width>0)&readable
    station=points@axis
    neighbors=[set() for _ in range(n)];continuation=[]
    valid_indices=np.flatnonzero(valid)
    candidate_pairs=set()
    if len(valid_indices)>1:
        # A source-coordinate triangulation supplies adjacency for RGB
        # inspection without comparing every pair in a densely sampled
        # stripe. Its coordinates supply navigation, never new observations
        # or an acceptance radius. Actual RGB still decides each edge.
        if len(valid_indices)>=3:
            try:
                triangles=Delaunay(points[valid_indices]).simplices
                for tri in triangles:
                    for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                        candidate_pairs.add(tuple(sorted((int(valid_indices[a]),int(valid_indices[b])))))
            except QhullError:
                pass  # Collinear support still has exact station neighbors.
        # Sparse observed support has no artificial maximum valid span:
        # check its own native RGB between station neighbors and explicitly
        # witnessed stem neighbors even when the samples are far apart.
        order=valid_indices[np.argsort(station[valid_indices],kind='stable')]
        candidate_pairs.update(tuple(sorted((int(a),int(b)))) for a,b in zip(order[:-1],order[1:]))
        for path in source_navigation_paths:
            candidate_pairs.update(tuple(sorted((a,b))) for a,b in zip(path[:-1],path[1:]) if valid[a] and valid[b])
    # Check every locally available measured-side continuation in its own RGB.
    # Interpolation supplies search samples only, never output observations.
    for i,j in sorted(candidate_pairs):
            count=max(2,int(np.ceil(np.linalg.norm(points[j]-points[i])/native_quantization_step_px))+1)
            alpha=np.linspace(0.,1.,count)[:,None,None]
            test_sides=sides[i]+alpha*(sides[j]-sides[i]);test_points=test_sides.mean(1)
            test_width=np.linalg.norm(test_sides[:,1]-test_sides[:,0],axis=1)
            if np.any(test_width<=0):continue
            test_normals=(test_sides[:,1]-test_sides[:,0])/test_width[:,None]
            locations=np.stack([test_sides[:,0]-3.*test_normals,test_points,test_sides[:,1]+3.*test_normals],axis=1)
            connected=bool(_all_sample_pixels_supported(mask,locations,test_normals).all())
            if connected:
                # OpenCV remap dimensions must fit signed 16-bit indices.
                # Chunk only the computation; every native sample is read.
                remap_rows=np.iinfo(np.int16).max-1
                for start in range(0,count,remap_rows):
                    stop=min(count,start+remap_rows)
                    supported,_,_,_=_actual_paint_colour_context(rgb,test_points[start:stop],test_sides[start:stop],test_normals[start:stop],'dark',native_rgb_field=native_rgb_field)
                    if not supported.all():connected=False;break
            continuation.append(dict(original_local_point_indices=[i,j],actual_native_RGB_two_interface_continuation=connected,
                navigation_samples_are_not_new_observations=True))
            if connected:neighbors[i].add(j);neighbors[j].add(i)
    intervals=[]
    for i in range(n):
        local=sorted({i,*neighbors[i]})
        interval=_tangent_interval(points[local],q[local],axis,native_quantization_step_px) if len(local)>=2 and complete[local].all() else None
        intervals.append(interval)
    return dict(valid=valid,complete=complete,normals=normals,colours=colours,contrast=contrast,
        neighbors=neighbors,continuation=continuation,tangent_intervals=intervals,axis=axis)


def review_current_gray_source_contour(rgb,uv,sides,quantiles,*,source_binding,
        trusted_current_stem_paths,stem_witness,source_valid_mask,
        native_quantization_step_px,maximum_source_neighbor_gap_px=80.):
    """Return point-level source-shape evidence and pending, never new UVs.

``trusted_current_stem_paths`` indexes these exact current observations.
Witnesses explicitly identify the visible named material in this actual/native
image; a producer's general seed flag or a model display is insufficient.
The 80-native-pixel neighbor search is the existing public paint-coherence
navigation footprint, not an acceptance distance to any world/model curve.
    """
    from rinkspace.vision_ice_model_context_identity import _actual_paint_colour_context
    rgb=np.asarray(rgb);points=np.asarray(uv,float);sides=np.asarray(sides,float);q=np.asarray(quantiles,float)
    n=len(points);mask=np.asarray(source_valid_mask,bool)
    if (rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or points.shape!=(n,2)
            or sides.shape!=(n,2,2) or q.shape!=(n,2,3,2) or mask.shape!=rgb.shape[:2]
            or not np.isfinite(points).all() or not np.isfinite(sides).all()
            or not np.isfinite(native_quantization_step_px) or native_quantization_step_px<=0
            or not np.isfinite(maximum_source_neighbor_gap_px) or maximum_source_neighbor_gap_px<=0):
        raise ValueError('Exact current native pixels, signed sides and readout support required')
    if hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()!=source_binding['decoded_native_BGR_sha256']:
        raise ValueError('Source-contour review requires its own unchanged native image')
    if (stem_witness.get('source_binding')!=source_binding
            or stem_witness.get('actual_visible_named_material_reviewed') is not True
            or stem_witness.get('model_projection_used_as_source_truth') is not False
            or not stem_witness.get('source_review_references')):
        raise ValueError('Separately witnessed current actual named stems are required')
    stems=[list(path) for path in trusted_current_stem_paths]
    flat=[i for path in stems for i in path]
    if (not flat or len(set(flat))!=len(flat) or any(type(i)is not int or not 0<=i<n for i in flat)
            or any(len(path)<1 for path in stems)
            or stem_witness.get('trusted_current_stem_paths')!=stems):
        raise ValueError('Witnessed disjoint current source stem indices required')
    centered=points[flat]-points[flat].mean(0)
    axis=np.linalg.svd(centered,full_matrices=False)[2][0] if len(flat)>=2 else np.array([1.,0.])
    if axis[0]<0:axis=-axis
    geometry=analyze_current_gray_native_contour_pixels(rgb,points,sides,q,
        source_valid_mask=mask,source_axis=axis,native_quantization_step_px=native_quantization_step_px,
        source_navigation_paths=stems)
    valid=geometry['valid'];complete=geometry['complete'];colours=geometry['colours'];contrast=geometry['contrast']
    neighbors=geometry['neighbors'];continuation=geometry['continuation']
    stem_intervals=[]
    for path in stems:
        if not valid[path].all():continue
        interval=_tangent_interval(points[path],q[path],axis,native_quantization_step_px)
        if interval is not None and interval[0]<=interval[1]:stem_intervals.append(interval)
    trusted=set(flat);decisions=[];pending=[];shape_verified=[]
    colour_lo,colour_hi=_material_signature_intervals(colours)
    supported_stems=[i for i in flat if valid[i] and np.isfinite(colour_lo[i]).all() and np.isfinite(colour_hi[i]).all()]
    material_bounds=(None if not supported_stems else
        [colour_lo[supported_stems].min(0),colour_hi[supported_stems].max(0)])
    for i in range(n):
        local=sorted({i,*neighbors[i]})
        interval=_tangent_interval(points[local],q[local],axis,native_quantization_step_px) if len(local)>=2 and complete[local].all() else None
        compatible=(interval is not None and interval[0]<=interval[1] and any(
            max(interval[0],base[0])<=min(interval[1],base[1]) for base in stem_intervals))
        reached={i};stack=[i]
        while stack:
            for j in neighbors[stack.pop()]:
                if j not in reached:reached.add(j);stack.append(j)
        connected_stem=bool(reached&trusted)
        material_comparable=bool(material_bounds is not None and np.isfinite(colour_lo[i]).all()
            and np.isfinite(colour_hi[i]).all())
        material_competes=(material_comparable and bool(np.any(np.maximum(colour_lo[i],material_bounds[0])
            >np.minimum(colour_hi[i],material_bounds[1]))))
        if not valid[i]:status='pending_own_RGB_two_interfaces_quantiles_or_source_scope'
        elif i in trusted:status='witnessed_current_named_source_stem_preserved'
        elif material_competes:status='pending_current_source_material_signature_competes_with_named_gray_stem'
        elif interval is not None and interval[0]>interval[1]:status='pending_multiple_actual_contour_directions_or_readout_incoherence'
        elif interval is not None and stem_intervals and not compatible:status='pending_actual_contour_tangent_competes_with_witnessed_named_stem'
        elif not connected_stem:status='pending_current_named_contour_continuation_not_source_verified'
        elif not compatible:status='pending_current_actual_tangent_support_unresolved'
        elif not material_comparable:status='pending_current_named_gray_material_signature_unresolved'
        else:status='current_source_shape_supported_by_witnessed_named_contour'
        record=dict(original_local_point_index=i,status=status,actual_contour_neighbor_indices=local,
            actual_tangent_slope_interval=interval,witnessed_stem_tangent_slope_intervals=stem_intervals,
            own_two_exterior_material_contrast=contrast[i].tolist(),actual_RGB_continuation_reaches_stem=connected_stem)
        record['own_native_RGB_material_signature_comparable']=material_comparable
        record['own_native_RGB_material_signature_competes']=material_competes
        decisions.append(record)
        (pending if status.startswith('pending_') else shape_verified).append(i)
    return dict(schema='rinkspace.current-own-RGB-gray-contour-review.v1',source_binding=dict(source_binding),
        source_stem_witness=stem_witness,point_decisions=decisions,pending_original_local_point_indices=pending,
        source_shape_verified_original_local_point_indices=shape_verified,actual_two_interface_continuation=continuation,
        independently_witnessed_singleton_stem_indices=[path[0] for path in stems if len(path)==1],
        singleton_source_witness_supplies_no_two_dimensional_tangent=True,
        RGB_continuation_navigation='actual_source_Delaunay_station_and_witnessed_stem_adjacency',
        original_source_points_removed=0,original_numeric_pixel_values_changed=False,
        model_coordinates_or_fit_errors_used=False,source_shape_check_is_not_independent_named_identity_certificate=True,
        source_FIT_admitted=False)
