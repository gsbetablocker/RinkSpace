"""Read source-owned near-board cap/ice silhouette optical transitions.

The wide cap surface and its printed texture are not the vertical advertisement
surface. Only a reviewed iceward top silhouette may bind to board.top. Search
navigation is source-only; every emitted UV is reread in the current native RGB.
The q10/q90 support is photometric, never covariance, E, a measured world height,
or a certificate that the optical half-rise is an exact geometric surface point.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from .material_profile_candidates import _transition

SCHEMA = 'rinkfit.native_near_board_top_source_generation.v1'
QUALIFICATION_SCHEMA = 'rinkfit.native_near_board_top_source_qualification.v1'
OPERATOR = 'native_grey_cap_iceward_top_optical_half_rise_v1'
SOURCE_SHA = 'e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(file=str(p), sha256=sha(p))


def verified(reference):
    p = Path(reference.get('file', reference.get('path'))).resolve()
    if sha(p) != reference['sha256']:
        raise ValueError('Changed exact native near-board source: '+str(p))
    return p


def measure_native_top_profiles(raw_bgr, excluded, navigation_rows, binding):
    """Return local raw-pixel optical readouts and every unsupported query.

    navigation_rows are integer [y,x,...] source-gradient candidates, not image
    observations. They cannot supply a UV when the current RGB/mask is missing.
    One horizontal profile is not a two-direction material texture point.
    """
    image = np.asarray(raw_bgr)
    excluded = np.asarray(excluded)
    if image.shape != (1080,1920,3) or image.dtype != np.uint8:
        raise ValueError('Exact native uint8 BGR exposure required')
    if excluded.shape != (1080,1920) or excluded.dtype != np.bool_:
        raise ValueError('Exact current bool exclusion, nonzero means excluded, required')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if (binding.get('decoded_native_BGR_sha256') != digest
        or binding.get('source_video_sha256') != SOURCE_SHA
        or binding.get('time_base') != [1,12800]
        or binding.get('actual_pts') != binding.get('native_pts')
        or not isinstance(binding.get('native_pts'),int)):
        raise ValueError('Official current native PTS/RGB binding differs')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    accepted, pending = [], []
    for query_index, nav in enumerate(navigation_rows):
        y,x = map(int,nav[:2])
        base = dict(native_pts=binding['native_pts'],query_index=query_index,
                    navigation_seed_uv=[x,y], source_navigation_is_observation=False)
        if not(2<=y<1078 and 26<=x<1894):
            pending.append(dict(base,status='unknown_missing_two_native_plateaux'));continue
        if excluded[y-2:y+3,x-26:x+27].any():
            pending.append(dict(base,status='unknown_current_exclusion_intersects_profile'));continue
        profile = np.median(gray[y-2:y+3,x-26:x+27],axis=0).astype(float)
        left = float(np.median(profile[2:17]));right = float(np.median(profile[36:51]));amplitude=right-left
        before = np.median(image[y-2:y+3,x-24:x-9,:],axis=(0,1)).astype(float)
        if not(45<left<185 and 90<right<235 and amplitude>=16 and np.max(abs(before-left))<=32):
            pending.append(dict(base,status='unknown_cap_ice_colour_order_not_currently_resolved'));continue
        crossing = _transition(profile,10,42,1,left,amplitude)
        if crossing is None:
            pending.append(dict(base,status='unknown_competing_raw_profile_crossings'));continue
        q10,q50,q90 = [float(x-26+v) for v in crossing]
        if not(q10<q50<q90 and .5<=q90-q10<=32 and abs(q50-x)<=12):
            pending.append(dict(base,status='unknown_nonunique_or_unbounded_optical_transition'));continue
        uv=[q50,float(y)]
        source_profile=dict(x_coordinates=list(range(x-26,x+27)),y_rows=list(range(y-2,y+3)),
            median_native_gray=profile.tolist(),grey_plateau=left,iceward_plateau=right,
            actual_native_BGR_patch_sha256=hashlib.sha256(image[y-2:y+3,x-26:x+27].tobytes()).hexdigest(),
            q10_q50_q90_x=[q10,q50,q90],operator='_transition_unique_raw_median_gray_10_50_90',
            current_exclusion_clear=True,profile_support_halfheight_rows=2)
        token=hashlib.sha256(json.dumps(source_profile,separators=(',',':'),sort_keys=True).encode()).hexdigest()
        pts=binding['native_pts'];rid=f'native_near_top_v1/s06/{pts}/y{y}/{token[:16]}'
        accepted.append(dict(base,source_row_id=rid,source_family_id=f'native_near_top_v1/s06/{pts}',shot_id='s06',
            source_uv=uv,current_uv=uv,normal=[1.,0.],source_profile=source_profile,
            source_profile_sha256=token,current_rgb_sha256=digest,
            source_video_sha256=SOURCE_SHA,time_base=[1,12800],actual_pts=pts,
            surface_class='board',surface_instance_id='near_continuous_board_iceward_top_silhouette',
            material_surface='iceward_upper_rim_of_grey_board_cap',joint_material_id='board.top',
            measurement_operator_resolved=OPERATOR,source_geometry_eligible=True,
            point_correspondence=False,entity_station_is_unknown=True,
            station_unknown_and_not_shared_across_source_query_indices=True,
            source_pixel_location='actual_optical_half_rise_with_photometric_q10_q90_support',
            source_physical_boundary_mean_exact=False,source_optical_support_interval_x=[q10,q90],
            pixels_from_model=False,ice_H_used=False,physical_accuracy_claimed=False,
            source_position_covariance_issued=False,statistical_objective_eligible=False,
            numeric_statistics_ready=False,solver_eligible=False,
            fit_role=binding.get('fit_role','unassigned'),frame_role=binding.get('frame_role','unassigned'),
            source_navigation_seed_uv=[float(x),float(y)],source_measured_alpha=q50-x,
            source_acquisition_query=dict(alpha=[-16.,16.],native_uv=[[float(x-16),float(y)],[float(x+16),float(y)]],
                normal=[1.,0.],source_acquisition_window_only=True,source_query_index_is_not_world_station=True,
                source_profile_native_pts=pts,source_profile_query_index=query_index),
            optical_support_is_not_sigma_E_or_covariance=True))
    return accepted,pending


def replay_source_generation(generation_path):
    """Re-decode original exposures; exact UV/profile/mask mismatch fails closed."""
    import av
    gp=Path(generation_path).resolve();g=json.loads(gp.read_text(encoding='utf8'))
    if g.get('schema')!=SCHEMA or g.get('source_UV_from_model_or_screen')is not False:
        raise ValueError('Near-board generation is not source-only')
    for r in g['reader_modules']:verified(r)
    scope=json.loads(verified(g['scope']).read_text(encoding='utf8'));source=scope['source']
    if source['sha256']!=SOURCE_SHA or sha(source['path'])!=SOURCE_SHA:
        raise ValueError('Official native video changed')
    ledger=[json.loads(x)for x in verified(g['frame_ledger']).read_text(encoding='utf8').splitlines()]
    rows=[json.loads(x)for x in verified(g['source_rows']).read_text(encoding='utf8').splitlines()]
    expected={}
    for row in rows:expected.setdefault(row['native_pts'],[]).append(row)
    wanted={x['native_pts']:x for x in ledger if x['navigation_rows']}
    all_rows=[];checked=[];pending=[]
    with av.open(source['path'])as container:
        stream=container.streams.video[0]
        if str(stream.time_base)!='1/12800' or str(stream.average_rate)!='50':raise ValueError('Native clock changed')
        container.seek(min(wanted)-12800,stream=stream,backward=True)
        for frame in container.decode(stream):
            pts=int(frame.pts)
            if pts>max(wanted):break
            if pts not in wanted:continue
            record=wanted[pts];im=frame.to_ndarray(format='bgr24');mask=cv2.imread(str(verified(record['mask'])),cv2.IMREAD_UNCHANGED)
            if mask is None:raise ValueError('Current native exclusion unavailable')
            mask=np.any(mask!=0,axis=2)if mask.ndim==3 else mask!=0
            binding=dict(source_video_sha256=SOURCE_SHA,native_pts=pts,actual_pts=pts,time_base=[1,12800],decoded_native_BGR_sha256=hashlib.sha256(im.tobytes()).hexdigest(),fit_role=record['fit_role'],frame_role=record['frame_role'])
            actual,unknown=measure_native_top_profiles(im,mask,record['navigation_rows'],binding)
            if actual!=expected.get(pts,[])or binding['decoded_native_BGR_sha256']!=record['decoded_native_BGR_sha256']or unknown!=record['pending_profiles']:
                raise ValueError('Near-board actual native profile/UV/unknown denominator changed: '+str(pts))
            all_rows.extend(actual);pending.extend(unknown);checked.append(pts)
    if sorted(checked)!=sorted(wanted)or len(all_rows)!=g['qualified_optical_rows']:
        raise ValueError('Near-board native replay is incomplete')
    return dict(passed=True,actual_native_exposures_redecoded=len(checked),actual_source_rows_replayed=len(all_rows),
        pending_profiles_retained=len(pending),source_UV_max_abs_difference_px=0.,raw_profile_max_abs_difference=0.,
        all_current_RGB_hashes_and_exclusion_masks_valid=True,source_UV_changed=False,
        source_covariance_issued=False,source_support_is_not_exact_physical_mean=True)
