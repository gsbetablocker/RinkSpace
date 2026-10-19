"""Replace unverified top-edge identities with reviewed original-image strips.

The unchanged raw cache remains evidence. All old main-camera board-top
callbacks are replaced as a class: absent reviewed source support is unknown,
never permission to retain an advertisement edge under the old name. Other
observation families and every camera parameter remain unchanged.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from continuous_central_edge_optin import guard_for_spec
from observation_weighting import ObservationQuality
from track_sequence import SourceEdges, _closed_edge_ring, _add_source_edge_factors


class _FrozenBlurQuality:
    """Keep the ordinary/native source-edge objectives numerically identical."""
    def __init__(self, stamp, observed, scales):
        self.stamp=int(stamp)
        self.observed=np.asarray(observed,float).copy()
        self.multipliers=np.asarray(scales,float).copy()
        self.manifest=self

    def sigma_multiplier(self, pts, uv):
        if int(pts)!=self.stamp or not np.array_equal(uv,self.observed):
            raise ValueError('Reviewed source quality requested for different UV')
        return self.multipliers.copy()

    def scales(self, pts, uv, kind):
        return self.sigma_multiplier(pts,uv)

    wrap=staticmethod(ObservationQuality.wrap)


class _SourceSpecCollector:
    def __init__(self):
        self.specs=[]

    def add_observation(self, keys, residual, **kwargs):
        self.specs.append(dict(kind='observation',keys=tuple(map(int,keys)),
            args=(list(keys),residual),kwargs=kwargs,
            file='track_sequence.py',function='_add_source_edge_factors',
            loop='reviewed_main_source_cap',
            source_role='reviewed blue-cap source transition, not projected pseudo-observation'))


def replace_main_top_source_specs(data, model, states, rows, provenance):
    """Create opt-in source specs; do not mutate the original captured graph.

    Each row has native_pts, observed_uv, vertical_transition_interval_px,
    physical_identity, identity_status and source_sha256. A source review or
    checked bidirectional source match must certify the identity before entry.
    Only the earliest measured native row of a repeated exposure contributes.
    """
    if data['camera_id']!='main_long_side' or not provenance.get('source_identity_review_complete'):
        raise ValueError('Complete main camera and reviewed source identity are required')
    by_exposure={}
    reviewed_states={'source_visual_review','bidirectional_source_match_to_reviewed_seeds'}
    for row in rows:
        stamp=int(row['native_pts'])
        if (stamp not in model.frames or row.get('physical_identity')!='board_upper_blue_cap'
                or row.get('identity_status') not in reviewed_states
                or row.get('source_sha256')!=data['diag']['source_sha256']
                or not row.get('source_pixel_sha256')
                or not row.get('identity_evidence')):
            raise ValueError('Unqualified source row cannot become a board-top factor')
        uv=np.asarray(row['observed_uv'],float)
        bounds=np.asarray(row['vertical_transition_interval_px'],float)
        if (uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or bounds.shape!=uv.shape
                or not np.isfinite(uv).all() or not np.isfinite(bounds).all()
                or np.any(bounds[:,0]>0) or np.any(bounds[:,1]<0)):
            raise ValueError('Malformed actual source pixels or blur support')
        rep=int(model.frames[stamp].frame_key)
        previous=by_exposure.get(rep)
        if previous is None or stamp<int(previous['native_pts']):
            by_exposure[rep]=row
    if not by_exposure:
        raise ValueError('No independently identified blue-cap source rows')
    retained=[]
    removed=[]
    for spec in data['retained']:
        guard=guard_for_spec(spec)
        if guard is not None and guard.edge_name=='board_top':
            if guard.source_pts not in data['pts']:
                raise ValueError('Cannot replace a different camera source edge')
            removed.append(dict(pts=int(guard.source_pts),
                                raw_uv_count=len(guard.observed),
                                raw_observed_sha256=guard.observed_sha256))
        else:
            retained.append(spec)
    if Counter(r['pts'] for r in removed)!=Counter(data['pts']):
        raise ValueError('Original main top-edge inventory changed or was not captured')
    world={'board_top':_closed_edge_ring(np.asarray(data['scene']['board_surface']['top_edge_xyz'],float)),
           'ice_board_contact':_closed_edge_ring(np.asarray(data['scene']['ice_boundary']['points_xyz'],float))}
    collector=_SourceSpecCollector()
    quality=data['captured']['source_quality']
    measurements=[]
    for rep,row in sorted(by_exposure.items()):
        stamp=int(row['native_pts'])
        uv=np.asarray(row['observed_uv'],np.float64)
        bounds=np.asarray(row['vertical_transition_interval_px'],np.float64)
        # Sigma is an explicit fitting weight, not measured Gaussian precision.
        # A broad visible transition must not receive narrower source weight
        # than its farthest q10/q90 extent around the measured midpoint.
        old_scale=quality.scales(stamp,uv,'reviewed_native_edge:board_top')
        blur_extent=np.max(np.abs(bounds),axis=1)
        scales=np.maximum(np.broadcast_to(old_scale,(len(uv),)),blur_extent/2.)
        source_quality=_FrozenBlurQuality(stamp,uv,scales)
        edges=SourceEdges(np.asarray([stamp],np.int64),np.asarray([0,len(uv)],np.int64),
            uv,np.zeros(len(uv),np.int8),data['diag']['source_sha256'],data['diag']['model_sha256'])
        _add_source_edge_factors(collector,stamp,0,model.frames[stamp].template25,
            model.camera25(stamp,states),edges,world,fixed_mount=False,
            finite_vector=True,dynamic_candidates=True,source_quality=source_quality)
        spec=collector.specs[-1]
        guard=guard_for_spec(spec)
        if (guard is None or guard.edge_name!='board_top'
                or not np.array_equal(guard.observed,uv)
                or not np.array_equal(guard.scales,scales)):
            raise ValueError('Native/Python reviewed source objective registration differs')
        measurements.append(dict(native_pts=stamp,representative_pts=rep,
            uv_count=len(uv),uv_sha256=guard.observed_sha256,
            identity_status=row['identity_status'],
            source_pixel_sha256=row['source_pixel_sha256'],
            effective_sigma_min_max=[float(2*scales.min()),float(2*scales.max())]))
    changed={**data,'retained':retained+collector.specs}
    report=dict(policy='replace_entire_original_main_board_top_class_with_source_identified_cap',
        original_top_factors=len(removed),original_top_uv=sum(r['raw_uv_count'] for r in removed),
        new_top_exposure_factors=len(collector.specs),
        new_top_source_uv=sum(m['uv_count'] for m in measurements),
        unknown_exposures=len(model.frame_keys)-len(by_exposure),
        unchanged_other_spec_count=len(retained),
        original_cache_modified=False,source_uv_moved=False,priors_modified=False,
        camera_parameters_fixed=False,
        blur_policy='sigma=max(original image-quality sigma, maximum measured 10/90 transition extent); soft fitting weight, not confidence interval',
        native_objective='same finite Brown15 point-to-curve residual and source multipliers as Python',
        removed_raw_top_factors=removed,measurements=measurements,provenance=provenance)
    return changed,report
