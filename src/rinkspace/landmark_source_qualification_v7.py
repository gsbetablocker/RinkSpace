"""Partial source arcs and low-saturation ink evidence; no camera qualification."""
from copy import deepcopy
import numpy as np

from .landmark_band_profile import point_array_sha256
from .landmark_source_qualification import qualify_source_frame


def qualify_source_frame_v7(frame, model, profiles=None, *, chromatic_profiles=None, appearance=None):
    result=qualify_source_frame(frame,model,profiles,appearance=appearance)
    chromatic={p['observation_id']:p for p in chromatic_profiles or []}
    circles={p['id'] for p in model['structure']['primitives'] if p['kind']=='circle'}
    small=sorted(i for i in circles if i.startswith(('R','F')))
    for observation,report in zip(frame['landmark_observations'],result['observations']):
        identity=observation['observation_id'];colour=observation.get('colour_class','gray')
        profile=chromatic.get(identity)
        if profile and profile['point_array_sha256']!=point_array_sha256(observation['points_source_px']):
            raise ValueError('Chromatic profile belongs to different measured source points')
        blue_supported=bool(profile and profile['colour_evidence']=='blue_ink_supported')
        partial=bool(report['source_shape'].get('strong_source_curvature'))
        pair=bool(report['source_circle_evidence_eligible'])
        ids=list(report['allowed_model_ids'])
        if observation.get('association_status')!='masked_out' and (pair or partial):
            # Native "gray" includes unmeasurably low-saturation blue. Unknown
            # colour preserves competing large/small-circle explanations.
            ids=(['centre_circle'] if blue_supported or colour=='blue' else
                 small+(['centre_circle'] if colour!='red' else sorted(i for i in circles if i.startswith('big_red'))))
            report['reason']='visible_curved_partial_arc_with_colour_candidates' if not pair else 'paired_visible_curved_outer_boundaries_with_colour_candidates'
        if observation.get('association_status')=='masked_out':ids=[]
        result['qualified_model_ids'][identity]=ids
        report.update(allowed_model_ids=ids,
            source_circle_evidence_eligible=bool(ids and (pair or partial)),
            partial_visible_arc_only=bool(partial and not pair),
            recommended_measurement_uncertainty_source_px=8. if partial and not pair else 4.,
            chromatic_evidence=({'colour_evidence':profile['colour_evidence'],
                'point_array_sha256':profile['point_array_sha256'],
                'median_chromatic_excess_br_bg_levels':profile['median_chromatic_excess_br_bg_levels'],
                'blue_excess_point_fraction':profile['blue_excess_point_fraction'],
                'reliable_point_count':profile['reliable_point_count']} if profile else None),
            unknown_colour_is_not_neutral_gray_identity=True,
            one_arc_or_colour_is_not_sufficient_full_rink_initialization=True)
    result['qualification_version']='v7_partial_source_arcs_and_local_chromatic_support'
    return result
