"""Read the two sides of a coloured ring as one actual source band.

The image conic groups measured bands only. Its coordinates never become
observations, and the caller must establish the physical ring's identity.
"""
from collections import defaultdict
import cv2
import numpy as np
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from .material_tracks import assemble_fragments
from .material_rings import source_ring_hypotheses, ellipse_matrix


def continuous_paired_boundary_candidates(measurement, candidate_ids, *, max_gap_px=72., minimum_span_px=70.):
    """Reject isolated interface readings without completing missing pixels.

    Neighbours must be close on *both* measured sides. A side's endpoint is
    not interchangeable with the centre of a fitted or projected annulus.
    """
    from scipy.spatial import cKDTree
    ids=np.unique(np.asarray(candidate_ids,int))
    if not len(ids):return [],[]
    uv=np.asarray(measurement['candidate_side_quantiles_uv'])[ids,:,1]
    parent=np.arange(len(ids))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in cKDTree(uv[:,0]).query_pairs(max_gap_px):
        if np.linalg.norm(uv[a,1]-uv[b,1])<=max_gap_px:parent[find(a)]=find(b)
    groups=defaultdict(list)
    for i,cid in enumerate(ids):groups[find(i)].append(int(cid))
    kept=[];receipt=[]
    all_uv=np.asarray(measurement['candidate_side_quantiles_uv'])[:,:,1]
    for values in groups.values():
        span=np.linalg.norm(np.ptp(all_uv[values],axis=0),axis=1)
        good=len(values)>=4 and float(span.min())>=minimum_span_px
        receipt.append(dict(candidate_ids=values,side_spans_px=span.tolist(),continuous_source_arc=good))
        if good:kept.extend(values)
    return sorted(kept),receipt


def compatible_query_consensus(alternatives, candidate_queries):
    """Missing support is not disagreement; different bands at a query are."""
    claims=defaultdict(set)
    for alternative in alternatives:
        for candidate in alternative['candidate_ids']:
            claims[int(candidate_queries[candidate])].add(int(candidate))
    return dict(selected=sorted(next(iter(ids)) for ids in claims.values() if len(ids)==1),
        conflicting_queries={str(q):sorted(ids) for q,ids in claims.items() if len(ids)>1})


def observed_band_color(rgb, measurement, color):
    """Check actual pigment against both observed flanks, not a projected fill.

    The caller supplies a source-established material class. This class is
    identity evidence for an arc; it never supplies coordinates or fit weights.
    """
    if color not in ('red','neutral_dark'):
        raise ValueError('Unsupported observed band color: '+str(color))
    ids=np.arange(len(measurement['candidate_uv']))
    if not len(ids):return np.zeros(0,bool),dict(color=color,eligible=0)
    q=np.asarray(measurement['candidate_query'],int)
    normal=np.asarray(measurement['query_normal'],float)[q]
    sides=np.asarray(measurement['candidate_side_quantiles_uv'])
    middle=np.asarray(measurement['candidate_uv'],float)
    xy=np.stack([sides[:,0,0]-normal*3.,middle,sides[:,1,2]+normal*3.],1).astype('f')
    pixels=cv2.remap(np.asarray(rgb,np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)
    difference=pixels[:,1]-.5*(pixels[:,0]+pixels[:,2])
    red=difference[:,0]-.5*(difference[:,1]+difference[:,2])
    dark=-difference.mean(1)
    pigment_red=pixels[:,1,0]-.5*(pixels[:,1,1]+pixels[:,1,2])
    # A dark ripple inside red paint can be less red than its two flanks.
    # Relative contrast alone must not relabel that ripple as the neutral
    # outer annulus. Inspect the measured pigment itself as well.
    good=(red>1.5) if color=='red' else ((red<=1.5)&(pigment_red<=1.5)&(dark>1.5))
    return good,dict(color=color,eligible=int(good.sum()),total=len(ids),
        source_red_contrast=red.tolist(),source_dark_contrast=dark.tolist(),
        measured_pigment_red_chroma=pigment_red.tolist())


def read_coloured_ring_band(rgb, invalid, projected, *, source_sha256, pts,
                           center_uv, search_radius_px=160., spacing_px=28., prepared_source_context=None,
                           material_signal='red', refine_source_origin=False, band_color=None,
                           profile_direction='outward_radial',query_sampling=None):
    if profile_direction not in ('outward_radial','outward_normal'):
        raise ValueError('Unknown source profile direction')
    path=dict(curve_id='coloured_annulus',feature_id='unassigned_ring',world_curve_role='band',
        native_path_id='source_search/coloured_annulus',xyz=np.zeros((2,3)),center_xyz=np.zeros(3),
        operator='paint_center',material_signal=material_signal,material_transition=None,profile_direction=profile_direction)
    if query_sampling is not None:
        if query_sampling!='continuous_arc_length':raise ValueError('Unsupported ring query sampling')
        path.update(query_sampling=query_sampling,closed=True)
    reader=IceProgrammeSourceReader(None,source_sha256=source_sha256,time_base='1/12800',
        sample_spacing_px=spacing_px,search_radius_px=search_radius_px,native_paths=[path],
        prepared_source_context=prepared_source_context)
    projection=dict(uv=projected['uv'],valid=projected['valid'],centre_uv=center_uv)
    measured=reader.read(rgb,pts,None,invalid,shot_id='source_band',
        native_projections={path['native_path_id']:projection})
    eligibility,color_receipt=(observed_band_color(rgb,measured,band_color) if band_color else (None,None))
    fragments=assemble_fragments(measured,[path],eligible_candidates=eligibility)
    hypotheses=source_ring_hypotheses(np.asarray(rgb,np.float32),measured,0,fragments,
        band_centers=True,material_signal=material_signal,eligible_candidates=eligibility)
    if not hypotheses:return measured,dict(status='partial_source_bands_without_complete_ring' if fragments else 'no_source_ring_band',
        selected=[],hypotheses=[],fragments=fragments,color_receipt=color_receipt,
        partial_arcs_need_source_identity_and_correspondence=True)
    plausible=[h for h in hypotheses if h['score']>=.85*hypotheses[0]['score']]
    consensus=compatible_query_consensus(plausible,measured['candidate_query'])
    result=dict(status='paired_source_ring_band_candidates',hypotheses=hypotheses,
        plausible_hypotheses=len(plausible),**consensus,
        band_middle_never_exported_as_observation=True,
        source_measurement_signal=material_signal,
        physical_identity_assigned=False,model_UV_is_search_only=True,
        fragments=fragments,color_receipt=color_receipt)
    if refine_source_origin and len(consensus['selected'])>=12:
        # An inaccurate camera can place its search origin in the annulus.
        # Use the observed outer ellipse's interior for the next RGB read.
        # This image centre is navigation, never a material point or a 3D
        # circle-centre observation. The second pass remeasures original RGB.
        outer=measured['candidate_side_quantiles_uv'][consensus['selected'],1,1]
        try:
            source_center,source_conic=ellipse_matrix(outer)
        except (ValueError,cv2.error):
            return measured,result
        new_measurement,new_result=read_coloured_ring_band(rgb,invalid,projected,
            source_sha256=source_sha256,pts=pts,center_uv=source_center,
            search_radius_px=search_radius_px,spacing_px=spacing_px,
            prepared_source_context=prepared_source_context,material_signal=material_signal,
            refine_source_origin=False,band_color=band_color,profile_direction=profile_direction,query_sampling=query_sampling)
        new_result['source_origin_revision']=dict(initial_search_origin=np.asarray(center_uv).tolist(),
            actual_outer_samples=outer.tolist(),source_ellipse_center=source_center.tolist(),
            source_conic=source_conic.tolist(),coordinates_used_as_observations=False)
        return new_measurement,new_result
    return measured,result
