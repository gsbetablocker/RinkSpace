"""Source-pixel evidence eligibility, before any model-camera fitting.

Eligibility is not an identity or accuracy decision. Weak contours remain real
observations; they cannot acquire a circle identity from a fitted camera alone.
"""
from dataclasses import asdict, dataclass
import numpy as np

from .landmark_band_profile import point_array_sha256
from .landmark_association import source_complementary_pairs


@dataclass(frozen=True)
class SourceQualificationConfig:
    minimum_curve_points: int = 8
    minimum_span_source_px: float = 100.
    minimum_quadratic_improvement: float = 1.5
    minimum_bend_to_residual: float = 2.


def source_shape(observation, config=None, *, major_axis=None):
    """Fit a source-coordinate polynomial only to actually visible samples.

    The quadratic is a shape diagnostic, never replacement observed pixels.
    Signed bend uses an explicitly declared normal, enabling opposite-arc tests.
    """
    config=config or SourceQualificationConfig()
    points=np.asarray(observation.get('points_source_px',[]),float).reshape(-1,2)
    if len(points)<config.minimum_curve_points or not np.isfinite(points).all():
        return {'eligible_extent':False,'reason':'too_few_finite_source_points'}
    centered=points-points.mean(0)
    if major_axis is None:
        axis=np.linalg.svd(centered,full_matrices=False)[2][0]
        if axis[np.argmax(abs(axis))]<0:axis=-axis
    else:
        axis=np.zeros(2);axis[major_axis]=1.
    normal=np.array([-axis[1],axis[0]])
    x=centered@axis;y=centered@normal;span=float(np.ptp(x))
    if span<config.minimum_span_source_px:
        return {'eligible_extent':False,'span_source_px':span,'reason':'too_little_source_extent'}
    x=2*(x-x.min())/span-1
    linear=np.polynomial.polynomial.polyfit(x,y,1)
    quadratic=np.polynomial.polynomial.polyfit(x,y,2)
    line_error=y-np.polynomial.polynomial.polyval(x,linear)
    curve_error=y-np.polynomial.polynomial.polyval(x,quadratic)
    noise=float(np.percentile(abs(curve_error),90))
    ratio=float(np.sqrt(np.mean(line_error**2))/max(.25,np.sqrt(np.mean(curve_error**2))))
    bend=float(quadratic[2])
    strong=ratio>=config.minimum_quadratic_improvement and abs(bend)>=config.minimum_bend_to_residual*max(.5,noise)
    return {'eligible_extent':True,'span_source_px':span,'source_axis':axis.tolist(),
            'source_normal':normal.tolist(),'quadratic_bend_source_px':bend,
            'quadratic_residual_p90_source_px':noise,'line_to_curve_rms_ratio':ratio,
            'strong_source_curvature':bool(strong),'uses_camera_or_model_projection':False}


def qualify_source_frame(frame, model, profiles=None, *, config=None, appearance=None):
    """Return allowed IDs and an evidence report without modifying observations."""
    config=config or SourceQualificationConfig()
    appearance=appearance or {'centre_circle_colour':'blue','heldout_ids':['blue_left','blue_right']}
    profile_map={p['observation_id']:p for p in (profiles or [])}
    observations=frame.get('landmark_observations',[])
    primitive_ids={p['id'] for p in model['structure']['primitives']}
    circle_ids={p['id'] for p in model['structure']['primitives'] if p['kind']=='circle'}
    small_ids=sorted(i for i in circle_ids if i.startswith(('R','F')))
    red_ids=small_ids+sorted(i for i in circle_ids if i.startswith('big_red'))
    shapes=[source_shape(o,config) for o in observations]
    reports=[];allowed={};strong=set()
    # Candidate pairs come solely from source overlap, colour and brightness.
    pairs=source_complementary_pairs(observations,maximum_pairs=max(1,len(observations)**2),
                                     profiles=profile_map,allow_blue=True)
    pair_reports=[]
    for pair in pairs:
        i,j=pair['observation_indices']
        a=np.asarray(observations[i]['points_source_px'],float)
        b=np.asarray(observations[j]['points_source_px'],float)
        major=int(np.argmax(np.ptp(a,axis=0)+np.ptp(b,axis=0)))
        first=source_shape(observations[i],config,major_axis=major)
        second=source_shape(observations[j],config,major_axis=major)
        opposite=first.get('quadratic_bend_source_px',0)*second.get('quadratic_bend_source_px',0)<0
        curved=bool(first.get('strong_source_curvature') and second.get('strong_source_curvature') and opposite)
        both_outer=pair['source_pair_brightness']==['source_pair_outward_brightness_supported']*2
        eligible=curved and both_outer
        if eligible:strong.update((i,j))
        pair_reports.append({**pair,'source_curvature':[first,second],
            'opposite_source_bend':bool(opposite),'both_source_outer_brightness_supported':both_outer,
            'circle_evidence_eligible':eligible,
            'reason':'paired_visible_curved_outer_boundaries' if eligible else 'weak_shape_or_missing_complementary_outer_brightness'})
    for index,observation in enumerate(observations):
        identity=observation['observation_id'];colour=observation.get('colour_class','gray')
        profile=profile_map.get(identity)
        if profile and profile.get('point_array_sha256')!=point_array_sha256(observation['points_source_px']):
            raise ValueError('Source qualification profile does not bind original measured points')
        ids=[]
        if observation.get('association_status')=='masked_out':
            reason='existing_observation_masked_out'
        elif index in strong:
            ids=(['centre_circle'] if colour=='blue' else red_ids if colour=='red' else small_ids)
            reason='paired_visible_curved_outer_boundaries'
        elif colour=='red' and shapes[index].get('eligible_extent') and not shapes[index].get('strong_source_curvature'):
            ids=['red_center'] if 'red_center' in primitive_ids else []
            reason='red_linear_evidence_only_not_circle_evidence'
        else:
            reason='weak_visible_contour_retained_without_circle_identity'
        ids=[i for i in ids if i not in appearance['heldout_ids']]
        allowed[identity]=ids
        reports.append({'observation_id':identity,'allowed_model_ids':ids,
            'source_circle_evidence_eligible':index in strong,
            'source_shape':shapes[index],'reason':reason,
            'source_points_unchanged':True,'weak_shape_does_not_mean_not_a_real_ring':True,
            'camera_projection_used_for_qualification':False})
    return {'qualified_model_ids':allowed,'observations':reports,'source_pair_evidence':pair_reports,
            'config':asdict(config),'appearance':appearance,
            'qualification_is_not_landmark_identity_or_accuracy_validation':True}
