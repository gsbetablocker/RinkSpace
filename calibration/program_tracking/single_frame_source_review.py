"""Measure a reviewed single source image; never advance its source clock.

Navigation polygons select an image region, not observation coordinates.
Saved vertices distinguish local colour transitions, derived rectangular
corners, and representatives of a shared source-line estimate. Dense image
queries are temporary; only a sparse representation is retained for review.
No observations are approved or installed into a solver by this module.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def transition(profile, centre):
    """Read the middle and quartiles of this image's local colour change."""
    p = gaussian_filter1d(np.asarray(profile, float), 1.0)
    low, high = np.median(p[:7]), np.median(p[-7:])
    amplitude = high-low
    if abs(amplitude) < 9:
        return None
    q = (p-low)/amplitude
    crossings = []
    for fraction in (.25, .5, .75):
        edges = np.flatnonzero((q[:-1] <= fraction) & (q[1:] > fraction))
        if not len(edges):
            return None
        k = edges[np.argmin(np.abs(edges-centre))]
        crossings.append(float(k+(fraction-q[k])/(q[k+1]-q[k])))
    if not crossings[0] <= crossings[1] <= crossings[2]:
        return None
    return crossings, abs(float(amplitude))


def simplify_indices(points, tolerance):
    """Keep only measured vertices; tolerance is tied to observed edge blur."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return np.arange(len(p))
    direction = p[-1]-p[0]
    t = np.clip(((p-p[0]) @ direction)/max(float(direction@direction), 1e-12), 0, 1)
    distances = np.linalg.norm(p-(p[0]+t[:, None]*direction), axis=1)
    k = int(np.argmax(distances))
    if distances[k] <= tolerance:
        return np.array([0, len(p)-1])
    return np.r_[simplify_indices(p[:k+1], tolerance)[:-1],
                 k+simplify_indices(p[k:], tolerance)]


def compact_run(points, widths, *, closed=False):
    p = np.asarray(points, float)
    widths = np.asarray(widths, float)
    # One quarter of the observed 25--75% transition width. This represents
    # image information, not a fixed per-frame point budget or solver error.
    tolerance = max(.75, float(np.median(widths))/4)
    if closed:
        far = int(np.argmax(np.linalg.norm(p-p[0], axis=1)))
        first = simplify_indices(p[:far+1], tolerance)
        tail = np.vstack([p[far:], p[0]])
        last = (far+simplify_indices(tail, tolerance)) % len(p)
        keep = np.r_[first[:-1], last[:-1]]
    else:
        keep = simplify_indices(p, tolerance)
    selected = p[keep]
    poly = np.vstack([selected, selected[0]]) if closed else selected
    errors = np.full(len(p), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        d=b-a; fraction=np.clip(((p-a)@d)/max(float(d@d),1e-12),0,1)
        errors=np.minimum(errors,np.linalg.norm(p-(a+fraction[:,None]*d),axis=1))
    assert np.max(errors) <= tolerance+1e-7
    return dict(uv=np.round(selected, 3).tolist(), closed=bool(closed),
                transition_iqr_px=np.round(widths[keep], 3).tolist(),
                localization_sigma_px=np.round(widths[keep]/1.349, 3).tolist(),
                simplification_tolerance_px=round(tolerance, 3),
                reconstruction_max_px=round(float(np.max(errors)), 3),
                source_query_count=len(p), vertex_count=len(keep),
                pixels_measured_in_this_frame=True)


def scan_curve(raw, mask, spec):
    """Scan current pixels around a reviewed horizontal/vertical edge ROI."""
    b, g, r = cv2.split(raw.astype(float))
    signals = {'blue': b-(g+r)/2, 'yellow': (r+g)/2-b,
               'magenta': (r+b)/2-g, 'luma': (r+g+b)/3}
    scalar=signals[spec['signal']]
    anchors=np.asarray(spec['navigation_xy'],float)
    axis=spec.get('axis',0); other=1-axis
    start,end=anchors[[0,-1],axis]
    span=spec.get('normal_radius_px',24)
    rows=[]; runs=[]
    # Current-image readout every two pixels detects unsupported gaps. These
    # samples are not persisted as thousands of optimizer observations.
    for location in np.r_[np.arange(start,end,2.),end]:
        nav=np.interp(location,anchors[:,axis],anchors[:,other])
        coords=np.arange(int(round(nav))-span,int(round(nav))+span+1)
        cross=int(round(location))
        if axis==0:
            if coords[0]<0 or coords[-1]>=raw.shape[0]:continue
            columns=np.arange(max(0,cross-2),min(raw.shape[1],cross+3))
            profile=np.median(scalar[np.ix_(coords,columns)],axis=1)
        else:
            if coords[0]<0 or coords[-1]>=raw.shape[1]:continue
            columns=np.arange(max(0,cross-2),min(raw.shape[0],cross+3))
            profile=np.median(scalar[np.ix_(columns,coords)],axis=0)
        found=transition(profile,span)
        if found is None:
            if rows:runs.append(rows);rows=[]
            continue
        quartiles,strength=found
        coordinate=coords[0]+quartiles[1]
        point=np.array([location,coordinate]) if axis==0 else np.array([coordinate,location])
        x,y=np.rint(point).astype(int)
        qlo,qhi=np.floor(coords[0]+np.array([quartiles[0],quartiles[2]])).astype(int)
        occupied=(mask[qlo:qhi+2,max(0,x-2):x+3].any() if axis==0 else
                  mask[max(0,y-2):y+3,qlo:qhi+2].any())
        if occupied or abs(coordinate-nav)>span*.7:
            if rows:runs.append(rows);rows=[]
            continue
        rows.append((point,quartiles[2]-quartiles[0]))
    if rows:runs.append(rows)
    segments=[]
    for run in runs:
        if len(run)<2:continue
        segment=compact_run([r[0] for r in run],[r[1] for r in run])
        # Temporary native readouts support later sparse coverage samples.
        # They are removed before the review packet is written.
        segment['_native_uv']=[r[0].tolist() for r in run]
        segment['_native_iqr']=[r[1] for r in run]
        segments.append(segment)
    return segments


def print_contours(raw, mask, roi, *, minimum_transition_iqr_px=0.0,
                   require_magenta_white_pair=False, profile_radius_px=20,
                   foreground_clearance_px=10):
    """Extract white Ad2 print; keep cuts at the person and frame boundary."""
    x0,y0,x1,y1=roi
    radius=int(profile_radius_px)
    clearance=int(foreground_clearance_px)
    if radius < 8 or clearance < 0:
        raise ValueError('Print profiles need real material context')
    green=raw[y0:y1,x0:x1,1]
    threshold,binary=cv2.threshold(green,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary[mask[y0:y1,x0:x1]]=0
    from rinkspace.vision_instances import source_print_boundary_contours
    # Reuse the existing native material-contour interface. White is the
    # selected material here; both polarities are remeasured below.
    full_mask=np.zeros(raw.shape[:2],np.uint8)
    full_mask[y0:y1,x0:x1]=binary
    contours=source_print_boundary_contours({'ink_mask':full_mask},stride=1)
    result=[]
    for contour in contours:
        path=np.asarray(contour['uv'],float)
        if len(path)<20:continue
        closed=bool(contour['closed'])
        centre=np.mean(path,axis=0)
        label=('U' if 1340<centre[0]<1565 else 'hyphen' if 1565<=centre[0]<1680
               else 'S' if 1680<=centre[0]<1845 else
               'right_cropped_letter' if centre[0]>=1845 else 'left_S_visible_fragment')
        rows=[];pair_votes=[]
        for i,point in enumerate(path):
            # Frame-clipped paths are open. Their ends are unrelated boundary
            # locations and must never share a wrapped tangent or run.
            before=(i-5)%len(path) if closed else max(0,i-5)
            after=(i+5)%len(path) if closed else min(len(path)-1,i+5)
            tangent=path[after]-path[before]
            length=np.linalg.norm(tangent)
            if length<.1:rows.append(None);continue
            normal=np.array([-tangent[1],tangent[0]])/length
            coords=point+np.arange(-radius,radius+1)[:,None]*normal
            if (coords[:,0].min()<0 or coords[:,0].max()>raw.shape[1]-1 or
                    coords[:,1].min()<0 or coords[:,1].max()>raw.shape[0]-1):
                rows.append(None);continue
            x,y=point.astype(int)
            if mask[max(0,y-clearance):y+clearance+1,max(0,x-clearance):x+clearance+1].any():rows.append(None);continue
            values=cv2.remap(raw[:,:,1],coords[:,0].astype('float32')[:,None],
                             coords[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0]
            if require_magenta_white_pair:
                colors=cv2.remap(raw,coords[:,0].astype('float32')[:,None],
                                 coords[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float)
                sides=np.array([np.median(colors[:5],axis=0),np.median(colors[-5:],axis=0)])
                white=sides[np.argmax(sides[:,1])];purple=sides[np.argmin(sides[:,1])]
                pair_votes.append(not (white[1]<140 or white[1]-purple[1]<40 or np.ptp(white)>55
                        or (purple[0]+purple[2])/2-purple[1]<25))
            measured=transition(values,radius)
            if measured is None:rows.append(None);continue
            qs,strength=measured
            if abs(qs[1]-radius)>radius/2:rows.append(None);continue
            if qs[2]-qs[0]<minimum_transition_iqr_px:rows.append(None);continue
            uv=point+(qs[1]-radius)*normal
            rows.append((uv,qs[2]-qs[0]))
        # Use adjacent material colours to authenticate the whole connected
        # print component. At a real acute glyph corner, a 20px normal can
        # cross another stroke; that alone must not erase the corner.
        if require_magenta_white_pair and (not pair_votes or np.mean(pair_votes)<.55):continue
        if all(r is not None for r in rows):
            segments=[compact_run([r[0] for r in rows],[r[1] for r in rows],closed=contour['closed'])]
        else:
            # Start after a rejected sample, so cyclic contour order does not
            # create a synthetic connection across an occlusion.
            if closed:
                begin=next(i for i,r in enumerate(rows) if r is None)
                ordered=rows[begin+1:]+rows[:begin+1]
            else:
                ordered=rows+[None]
            runs=[];run=[]
            for row in ordered:
                if row is None:
                    if len(run)>4:runs.append(run)
                    run=[]
                else:run.append(row)
            segments=[compact_run([r[0] for r in run],[r[1] for r in run])for run in runs]
        if segments:result.append(dict(id='Ad2.print.'+label,label=label,segments=segments,
              physical_surface='board',world_correspondence_status='template_binding_pending',
              source_contour_internal_hole=bool(contour.get('internal_hole',False)),
              source_magenta_white_pair_fraction=(float(np.mean(pair_votes)) if pair_votes else None),
              source_print_threshold=float(threshold),allowed_for_fit=False))
    return result


def ad_corners(raw, boxes):
    """Measure advertisement corner arcs from source chroma, not grid corners."""
    b,g,r=cv2.split(raw.astype(float));scalar=(r+b)/2-g
    segments=[]
    for x0,y0,x1,y1 in boxes:
        crop=scalar[y0:y1,x0:x1]
        low,high=np.percentile(crop,[10,90])
        level=(low+high)/2
        binary=(crop>level).astype(np.uint8)
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        path=max(contours,key=cv2.contourArea)[:,0,:]+[x0,y0]
        inside=(path[:,0]>x0+1)&(path[:,0]<x1-2)&(path[:,1]>y0+1)&(path[:,1]<y1-2)
        indices=np.flatnonzero(inside)
        # The contour boundary has one genuine corner arc; crop sides are
        # navigation only and are never retained as a material boundary.
        runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
        for run in runs:
            if len(run)<4:continue
            values=[];widths=[]
            for i in run:
                tangent=path[(i+3)%len(path)]-path[(i-3)%len(path)]
                normal=np.array([-tangent[1],tangent[0]],float)
                normal/=max(np.linalg.norm(normal),1e-9)
                xy=path[i]+np.arange(-20,21)[:,None]*normal
                profile=cv2.remap(scalar.astype('float32'),xy[:,0].astype('float32')[:,None],
                                  xy[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0]
                measured=transition(profile,20)
                if measured is None:continue
                qs,_=measured
                values.append(path[i]+(qs[1]-20)*normal);widths.append(qs[2]-qs[0])
            if len(values)>=2:segments.append(compact_run(values,widths))
    return segments


def rectangular_left_outline(features, advertisement_id):
    """Share two source-line intersection corners between three visible edges."""
    by_id={f['id']:f for f in features}
    upper=by_id[advertisement_id+'.top_edge']['segments'].pop(0)
    lower=by_id[advertisement_id+'.bottom_edge']['segments'].pop(0)
    side=by_id[advertisement_id+'.left_vertical']['segments'][0]

    def intersection(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        da=a[-1]-a[0];db=b[-1]-b[0]
        t=np.linalg.solve(np.column_stack([da,-db]),b[0]-a[0])[0]
        return a[0]+t*da

    top=intersection(upper['uv'],side['uv'])
    bottom=intersection(lower['uv'],side['uv'])
    uv=np.array([upper['uv'][-1],top,bottom,lower['uv'][-1]])
    segment=dict(uv=np.round(uv,3).tolist(),closed=False,vertex_count=4,
        protected_vertex_indices=[0,1,2,3],
        measurement_origin='two_source_measured_straight_edge_intersections',
        edge_roles=['top_edge','left_vertical','bottom_edge'],
        corner_vertex_indices={'top_left':1,'bottom_left':2},
        source_line_uv={'top':upper['uv'],'left':side['uv'],'bottom':lower['uv']},
        _edge_readouts=[upper,side,lower],
        model_projection_used_for_corner_coordinates=False,
        corners_are_derived_feature_positions=True,user_confirmed=False)
    features[:]=[f for f in features if f['id'] not in
                 {advertisement_id+'.left_vertical',advertisement_id+'.visible_corner_arcs'}]
    features.append(dict(id=advertisement_id+'.visible_left_outline',
        label_zh='Ad2上边左段—左上角—左下角—下边左段',
        physical_surface='board',world_curve=advertisement_id+'.external_border',
        segments=[segment],allowed_for_fit=False,
        corner_semantics='user_requested_rectangular_border_no_rounded_arc_samples'))


def compact_review_points(features, maximum):
    """Remove redundant measured vertices while retaining contour topology.

    Open ends, closed-contour extrema, explicit corners and manual source
    readouts remain. The existing source coordinates are never shifted.
    The user-specified count is local to this review packet.
    """
    records=[]
    for feature in features:
        for segment in feature['segments']:
            p=np.asarray(segment['uv'],float);n=len(p)
            protected=set(segment.get('protected_vertex_indices',[]))
            if segment.get('measurement_origin')=='assistant_native_visual_readout':
                protected.update(range(n))
            if segment.get('closed'):
                protected.update(int(i) for i in
                    [np.argmin(p[:,0]),np.argmax(p[:,0]),np.argmin(p[:,1]),np.argmax(p[:,1])])
            else:protected.update([0,n-1])
            width=np.asarray(segment.get('transition_iqr_px',np.full(n,4.)),float)
            records.append(dict(segment=segment,p=p,keep=list(range(n)),
                protected=protected,scale=max(1.,float(np.median(width))/2)))
    before=sum(len(r['keep'])for r in records)

    def error(p,keep,closed):
        poly=p[keep]
        if closed:poly=np.vstack([poly,poly[0]])
        distances=np.full(len(p),np.inf)
        for a,b in zip(poly[:-1],poly[1:]):
            d=b-a;t=np.clip(((p-a)@d)/max(float(d@d),1e-12),0,1)
            distances=np.minimum(distances,np.linalg.norm(p-(a+t[:,None]*d),axis=1))
        return float(np.max(distances))

    count=before
    while count>maximum:
        choices=[]
        for ri,row in enumerate(records):
            keep=row['keep'];closed=row['segment'].get('closed',False)
            if len(keep)<=(6 if closed else 2):continue
            for index in keep:
                if index in row['protected']:continue
                selected=[i for i in keep if i!=index]
                deviation=error(row['p'],selected,closed)
                choices.append((deviation/row['scale'],deviation,ri,index,selected))
        if not choices:
            raise ValueError('User count cannot be met without losing protected features')
        _,_,ri,_,selected=min(choices)
        records[ri]['keep']=selected;count-=1
    for row in records:
        segment=row['segment'];keep=row['keep'];n=len(row['p'])
        for key in ('uv','transition_iqr_px','localization_sigma_px','source_y_interval_px'):
            if key in segment and len(segment[key])==n:
                segment[key]=[segment[key][i]for i in keep]
        segment['vertex_count']=len(keep)
        segment['retained_pre_compaction_vertex_indices']=keep
        segment['pre_compaction_vertex_count']=n
        segment['sparse_representation_max_deviation_px']=round(error(row['p'],keep,segment.get('closed',False)),3)
        if 'reconstruction_max_px' in segment:
            segment['pre_compaction_reconstruction_max_px']=segment.pop('reconstruction_max_px')
            segment['pre_compaction_simplification_tolerance_px']=segment.pop('simplification_tolerance_px')
        if 'protected_vertex_indices' in segment:
            segment['protected_vertex_indices']=[keep.index(i)for i in row['protected']]
    return dict(maximum_points_requested_by_user=maximum,points_before=before,
        points_after=count,all_segments_and_closure_retained=True,
        original_retained_coordinates_unchanged=True,
        image_accuracy_accepted=False)


def straight_source_boundary(feature, evidence_segments, visible_x_intervals):
    """Estimate one straight edge from this frame, retaining visible breaks.

    The straight-edge representatives share one estimated line, and are
    explicitly distinguished from the original individual colour readouts.
    No camera/world projection or hidden interval becomes a source point.
    """
    from scipy.optimize import least_squares
    points=[];uncertainty=[]
    for segment in evidence_segments:
        points.extend(segment['uv'])
        if 'localization_sigma_px' in segment:
            uncertainty.extend(segment['localization_sigma_px'])
        else:
            bounds=np.asarray(segment['source_y_interval_px'],float)
            uncertainty.extend((bounds[:,1]-bounds[:,0])/2)
    uv=np.asarray(points,float);sigma=np.asarray(uncertainty,float)
    x=(uv[:,0]-960.)/960.
    design=np.column_stack([np.ones(len(x)),x])
    initial=np.linalg.lstsq(design/sigma[:,None],uv[:,1]/sigma,rcond=None)[0]
    fitted=least_squares(lambda c:(design@c-uv[:,1])/sigma,initial,loss='soft_l1')
    if not fitted.success:raise ValueError('Source straight-edge estimation failed')
    intercept=float(fitted.x[0]-fitted.x[1]);slope=float(fitted.x[1]/960.)
    feature['source_straight_line_estimate']=dict(
        y_intercept_px=intercept,y_per_x=slope,
        evidence_uv=uv.tolist(),evidence_localization_sigma_px=sigma.tolist(),
        evidence_source='this_frame_native_colour_readouts_and_source_visual_intervals',
        model_projection_used=False,user_requested_straight_edge=True,
        estimator='source_uncertainty_weighted_soft_l1_line',
        original_readout_max_residual_px=float(np.max(np.abs(design@fitted.x-uv[:,1]))),
        user_accuracy_acceptance=False)
    feature['segments']=[]
    for lo,hi in visible_x_intervals:
        endpoints=np.array([[lo,intercept+slope*lo],[hi,intercept+slope*hi]])
        feature['segments'].append(dict(uv=endpoints.tolist(),closed=False,vertex_count=2,
            measurement_origin='same_frame_joint_straight_edge_fit',
            representative_points_share_line_estimate=True,
            pixels_measured_in_this_frame=False,
            source_evidence_measured_in_this_frame=True,
            original_readouts_retained_in_feature=True,
            hidden_span_filled=False,user_confirmed=False))


def space_boundary_samples(features, maximum_spacing):
    """Cover long visible edges; bind each sample to its own physical edge.

    Additional material-edge samples are selected from actual native colour
    readouts, not interpolated. Only the explicitly fitted straight board top
    shares a line estimate. Corner identity is separate from edge identity.
    """
    def edge_samples(a,b,readout,shared_line=False):
        length=float(np.linalg.norm(b-a))
        divisions=max(1,int(np.ceil(length/maximum_spacing)))
        points=[a];origins=['existing_endpoint'];widths=[None]
        for fraction in np.arange(1,divisions)/divisions:
            target=a+(b-a)*fraction
            if shared_line:
                point=target;width=None;origin='shared_source_line_representative'
            else:
                native=np.asarray(readout['_native_uv'],float)
                index=int(np.argmin(np.linalg.norm(native-target,axis=1)))
                point=native[index];width=float(readout['_native_iqr'][index])
                origin='native_colour_half_transition'
            points.append(point);origins.append(origin);widths.append(width)
        points.append(b);origins.append('existing_endpoint');widths.append(None)
        return points,origins,widths

    for feature in features:
        if '.print.' in feature['id']:continue
        shared_line='source_straight_line_estimate' in feature
        for segment in feature['segments']:
            original=np.asarray(segment['uv'],float)
            edge_roles=segment.get('edge_roles')
            corners={v:k for k,v in segment.get('corner_vertex_indices',{}).items()}
            points=[];bindings=[];expanded_roles=[];new_corners={}
            for edge_index,(a,b) in enumerate(zip(original[:-1],original[1:])):
                role=edge_roles[edge_index] if edge_roles else feature['world_curve']
                target='Ad2.'+role if edge_roles else role
                readout=segment['_edge_readouts'][edge_index] if edge_roles else segment
                p,origins,widths=edge_samples(a,b,readout,shared_line)
                for local_index,point in enumerate(p):
                    if edge_index and local_index==0:continue
                    original_index=(edge_index if local_index==0 else
                                    edge_index+1 if local_index==len(p)-1 else None)
                    corner=corners.get(original_index)
                    if corner:
                        new_corners[corner]=len(points)
                        binding=dict(kind='corner',physical_target='Ad2.'+corner,
                            required_residual='point_to_corresponding_model_corner_2d',
                            measurement_origin='source_straight_edge_intersection',
                            adjacent_model_edges=(['Ad2.top_edge','Ad2.left_vertical']
                                if corner=='top_left' else ['Ad2.left_vertical','Ad2.bottom_edge']))
                    else:
                        origin=origins[local_index]
                        if shared_line:origin='shared_source_line_representative'
                        elif origin=='existing_endpoint':origin='native_colour_half_transition'
                        binding=dict(kind='edge_sample',physical_target=target,
                            required_residual='point_to_named_finite_model_edge',
                            measurement_origin=origin)
                        if widths[local_index] is not None:
                            binding['localization_sigma_px']=widths[local_index]/1.349
                    binding['solver_binding_status']='not_installed_pending_user_acceptance'
                    binding['allowed_for_fit']=False
                    points.append(point);bindings.append(binding)
                expanded_roles.extend([role]*(len(p)-1))
            segment['uv']=np.round(points,3).tolist()
            segment['vertex_count']=len(points)
            segment['point_bindings']=bindings
            segment['protected_vertex_indices']=list(range(len(points)))
            segment['maximum_spacing_requested_px']=maximum_spacing
            segment['measured_maximum_spacing_px']=float(np.max(np.linalg.norm(np.diff(points,axis=0),axis=1)))
            # Existing per-vertex arrays no longer index the expanded set.
            for key in ('transition_iqr_px','localization_sigma_px'):
                if key in segment:
                    segment['endpoint_evidence_'+key]=segment.pop(key)
            if edge_roles:segment['edge_roles']=expanded_roles
            if corners:segment['corner_vertex_indices']=new_corners
            if shared_line:
                segment['samples_are_independent_measurements']=False
                segment['future_fit_weighting']='shared_line_evidence_not_independent_sample_count'
            else:segment['additional_points_measured_in_native_pixels']=True


def remove_temporary_readouts(value):
    if isinstance(value,dict):
        for key in list(value):
            if key.startswith('_'):del value[key]
            else:remove_temporary_readouts(value[key])
    elif isinstance(value,list):
        for child in value:remove_temporary_readouts(child)


def extract(raw_path, binding_path, specification_path, output_path):
    raw=cv2.imread(str(raw_path));mask=cv2.imread(str(Path(raw_path).with_name('person_mask.png')),0)>0
    binding=json.loads(Path(binding_path).read_text(encoding='utf8'))
    assert hashlib.sha256(raw.tobytes()).hexdigest()==binding['decoded_BGR_sha256']
    spec=json.loads(Path(specification_path).read_text(encoding='utf8'))
    features=[]
    for feature in spec['features']:
        row={k:v for k,v in feature.items() if k!='scans'}
        row['segments']=[segment for scan in feature['scans'] for segment in scan_curve(raw,mask,scan)]
        for visual in feature.get('visual_review_segments',[]):
            uv=np.asarray(visual['uv'],float)
            bounds=np.asarray(visual['source_y_interval_px'],float)
            row['segments'].append(dict(uv=uv.tolist(),closed=False,vertex_count=len(uv),
                source_y_interval_px=bounds.tolist(),measurement_origin='assistant_native_visual_readout',
                measurement_note=visual['note'],pixels_measured_in_this_frame=True,
                user_confirmed=False,model_projection_is_pixel_truth=False))
        row.update(allowed_for_fit=False,physical_surface='board')
        features.append(row)
    if spec.get('ad_corner_boxes') and not spec.get('rectangular_advertisement_corners'):
        features.append(dict(id='Ad2.visible_corner_arcs',label_zh='Ad2左上及左下转角',
            world_curve='Ad2.external_border',physical_surface='board',allowed_for_fit=False,
            segments=ad_corners(raw,spec['ad_corner_boxes'])))
    features.extend(print_contours(raw,mask,spec['print_roi']))
    counts={}
    for feature in features:counts[feature['id']]=counts.get(feature['id'],0)+1
    occurrences={}
    for feature in features:
        base=feature['id'];index=occurrences.get(base,0);occurrences[base]=index+1
        feature['id']=base+('.fragment'+str(index+1) if counts[base]>1 else '')
    for advertisement_id in spec.get('rectangular_advertisement_corners',[]):
        rectangular_left_outline(features,advertisement_id)
    line_specs=spec.get('straight_source_boundaries',{})
    line_evidence={f['id']:deepcopy(f['segments']) for f in features if f['id']in line_specs}
    compaction=None
    if spec.get('maximum_review_points') is not None:
        compaction=compact_review_points(features,int(spec['maximum_review_points']))
    for feature in features:
        if feature['id']in line_specs:
            straight_source_boundary(feature,line_evidence[feature['id']],
                                     line_specs[feature['id']]['visible_x_intervals'])
    if compaction and line_specs:
        compaction['points_after_vertex_compaction']=compaction['points_after']
        compaction['points_after']=sum(len(s['uv'])for f in features for s in f['segments'])
        compaction['straight_source_boundaries_applied_after_vertex_compaction']=list(line_specs)
        compaction['original_retained_coordinates_unchanged_except_shared_source_line_representatives']=True
        compaction.pop('original_retained_coordinates_unchanged',None)
    if spec.get('maximum_visible_boundary_spacing_px') is not None:
        space_boundary_samples(features,float(spec['maximum_visible_boundary_spacing_px']))
        prior_compaction=compaction
        compaction=compact_review_points(features,int(spec['maximum_review_points']))
        compaction['before_boundary_spacing_compaction']=prior_compaction
        compaction['boundary_coverage_samples_protected']=True
        compaction['maximum_visible_boundary_spacing_px']=spec['maximum_visible_boundary_spacing_px']
    remove_temporary_readouts(features)
    report=dict(schema='rinkspace.single-frame-source-review.v1',source=binding,
        status='awaiting_user_acceptance',user_accepted=False,solver_inputs_modified=False,
        frame_count=1,next_frame_processed=False,features=features,
        source_specification=str(Path(specification_path).resolve()),
        point_count=sum(s['vertex_count']for f in features for s in f['segments']),
        source_measurement='current_native_image_local_colour_half_transitions',
        density='vertices simplified within one quarter of measured edge transition IQR',
        projection_used_for_navigation_only=True,observation_coordinates_from_projection=False,
        independent_accuracy_claim=False,excluded_features_by_user=spec.get('excluded_features_by_user',[]),
        unresolved_source_regions=spec.get('unresolved_source_regions',[]))
    if compaction:
        report['user_requested_compaction']=compaction
        report['density']='user_requested_sparse_vertices_preserving_segment_ends_corners_and_glyph_extrema'
    if spec.get('rectangular_advertisement_corners'):
        report['rectangular_corner_advertisements']=spec['rectangular_advertisement_corners']
        report['source_measurement']='current_native_colour_transitions_and_source_line_intersection_corners'
    if line_specs:
        report['source_measurement']+=';_user_requested_same_frame_straight_boundary_estimates'
    Path(output_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report
