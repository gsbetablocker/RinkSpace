"""Original high-view user print corners and identified camera-entity support.

Pure small source loading/binding; no prepared programme graph or solver.
World positions use the caller's current actual shared geometry. Historical
ray intersections/camera fits are provenance only, never observations.
"""
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
from copy import deepcopy
import hashlib
import json

import numpy as np
from rinkspace.storage import AssetStore

REPO = Path('D:/OneDrive/codex_projs/RinkSpace')
HERE = _runtime_data_module_dir(__file__)
STORAGE = AssetStore(REPO)
AD_FILE = STORAGE.resolve('outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/high_view_ad_corners_56218880.json')
OBSERVATION_FILE = STORAGE.resolve('three_frame_refit/inputs/step02_observations.json')
PTS = 56218880
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
ATTACHMENT = Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00013.png')
ANNOTATED_ENTITY_BOX = (35., 788., 73., 841.)
NATIVE_ENTITY_BOX = tuple(.96*x-.02 for x in ANNOTATED_ENTITY_BOX)
SECONDARY_INSTALLATION_ID = 'okada_sp_installation_1_conditional_z190'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path):
    return dict(path=str(Path(path)), sha256=_sha(path))


def load_wide_user_ad_source():
    """Return JSON-safe packet with all 26 literal native source UVs.

    No uncertainty or weights are invented. Original FIT/group integration is
    included; caller must declare any numerical sigma or strengthened weight.
    """
    data = json.loads(AD_FILE.read_bytes())
    if data['schema_version'] != 'rinkspace.high-view-advert-corners.v1':
        raise ValueError('Original high-view source schema changed')
    source, annotation = data['source_frame'], data['user_annotation']
    if (source['source_pts'] != PTS or source['time_base'] != '1/15360'
            or source['source_sha256'] != SOURCE_SHA
            or source['image_size_wh'] != [1920,1080]
            or annotation['image_size_wh'] != [2000,1125]):
        raise ValueError('Original source exposure/hash/coordinate space changed')
    image = STORAGE.resolve(source['image_path']); marked = STORAGE.resolve(annotation['snapshot_path'])
    if _sha(image) != source['image_sha256'] or _sha(marked) != annotation['sha256']:
        raise ValueError('Original source or user-annotation image bytes changed')
    observations = json.loads(OBSERVATION_FILE.read_bytes())['observations']
    selected = {r['id']:r for r in observations if r.get('frame_id') == 'high_56218880' and r.get('category') == 'ads'}
    if len(selected) != 26:
        raise ValueError('Original 26-ad-corner observation inventory changed')
    rows = []
    for ad in data['advertisements']:
        number = int(ad['user_number']); scope = ad['observed_feature_scope']
        corners = ad['corner_observations']
        if len(corners) != (2 if number == 1 else 4):
            raise ValueError('Do not complete the partial advertisement or delete corners')
        if number == 1 and scope != 'external_ad_border_partial':
            raise ValueError('Ad1 source is its two visible left corners only')
        if number == 5 and scope != 'internal_graphic_rectangle_not_ad_extent':
            raise ValueError('Ad5 marks the internal graphic, not the whole print extent')
        for corner in corners:
            letter, role = corner['letter'], corner['role']
            identity = f'high_56218880:ad{number}:{letter}'
            uv = np.asarray(corner['source_frame_uv'], float)
            annotated_uv = np.asarray(corner['annotated_image_uv'], float)
            expected = .96*annotated_uv-.02
            original = selected.pop(identity)
            if (uv.shape != (2,) or not np.isfinite(uv).all()
                    or np.max(np.abs(uv-expected)) > 3e-13
                    or not np.array_equal(np.asarray(original['pixels_uv'])[0], uv)
                    or original['target']['corner_role'] != role
                    or original['target']['scope'] != scope
                    or original['use'] != 'fit'):
                raise ValueError('Original user corner UV/role/scope/split changed: '+identity)
            if number == 5 and ((letter == 'r' and role != 'right_top')
                                or (letter == 'q' and role != 'right_bottom')):
                raise ValueError('Preserve the corrected original Ad5 r/q identities')
            rows.append(dict(id=identity, source_pts=PTS, source_time_base='1/15360',
                source_video_sha256=SOURCE_SHA, source_role='fit', user_ad_number=number,
                letter=letter, role=role, observed_feature_scope=scope,
                annotated_image_uv=annotated_uv.tolist(), source_frame_uv=uv.tolist(),
                source_pointer=f'/advertisements/{number-1}/corner_observations/{corners.index(corner)}',
                original_observation_target=deepcopy(original['target']),
                original_uncertainty=deepcopy(original['uncertainty']),
                original_group_weight=deepcopy(original['weight']),
                original_sqrt_weights=deepcopy(original.get('sqrt_weights')),
                independent_validation=False,
                source_geometry_status='user_identity_and_pixels; current_world_is_shared_conditional_candidate'))
    if selected or len(rows) != 26 or len({r['id']for r in rows}) != 26:
        raise ValueError('Original source inventory is not exactly 26 unique corners')
    return dict(schema='rinkspace.actual-wide-user-26-ad-corners-source.v1',
        source_frame=deepcopy(source), original_annotation=deepcopy(annotation),
        source_files=[_ref(AD_FILE),_ref(OBSERVATION_FILE),_ref(image),_ref(marked)],
        annotation_to_native_formula='(uv_annotated+0.5)*0.96-0.5',
        rows=rows, source_uvs_roles_and_original_group_integration_unchanged=True,
        Ad1_partial_right_edge_or_width_observation_added=False,
        Ad5_internal_rectangle_is_not_external_ad_extent=True,
        Ad5_corrected_r_right_top_q_right_bottom=True,
        historical_camera_or_ray_intersections_used_as_measurement=False,
        engineering_sigma_or_weight_must_be_explicit_caller_policy=True,
        existing_pattern9_Ad12_13_has_no_matching_corner_in_this_Ad1_7_inventory=True,
        no_accuracy_claim=True)


def bind_wide_ad_world(packet, geometry, qworld, *, pattern_geometry=None, pattern_q=None):
    """Return current world26/Jworld and explicit original print ledger.

    geometry must be the caller's real source geometry (q11/23/29), not a copy
    of an old rendered scene. Current Ad1/14 and any installed paired/course
    parameters dispatch through its original rebind/world_jacobian APIs.
    """
    from ad_landmarks import AdGeometry
    qworld = np.asarray(qworld, float)
    if qworld.shape != (geometry.parameter_count,) or not np.isfinite(qworld).all():
        raise ValueError('Complete actual shared-world chart required')
    geometry.validate_parameters(qworld)
    worlds, jacobians, ledger = [], [], []
    for r in packet['rows']:
        n, role = r['user_ad_number'], r['role']
        old = AdGeometry(geometry.base,n)
        if role not in old.roles:
            raise ValueError(f'Original source print has no physical corner {n}:{role}')
        source_row = dict(id=r['id'], physical_ad_id=n,
            physical_world_corner_roles=[role],
            world=np.asarray([old.corner(old.initial,role)],float))
        current = geometry.rebind(source_row,qworld)
        world = np.asarray(current['world'],float)
        jac = np.asarray(geometry.world_jacobian(source_row,qworld),float)
        if world.shape != (1,3) or jac.shape != (1,3,len(qworld)):
            raise ValueError('Actual shared source world callback changed shape')
        worlds.append(world[0]); jacobians.append(jac[0])
        ledger.append(dict(id=r['id'],user_ad_number=n,role=role,
            observed_feature_scope=r['observed_feature_scope'],
            source_frame_uv=r['source_frame_uv'],source_role=r['source_role'],
            actual_world_dispatch='geometry.rebind + geometry.world_jacobian on untouched original print corner',
            current_world_has_active_parameter=bool(np.any(jac)),
            unmeasured_conditional_world_not_camera_ground_truth=True))
    if (pattern_geometry is None) != (pattern_q is None):
        raise ValueError('Optional actual pattern owner and full pattern q must be supplied together')
    pattern_count = 0 if pattern_q is None else len(np.asarray(pattern_q))
    if pattern_q is not None and pattern_count != 9:
        raise ValueError('Existing Ad12/13 shared pattern has nine explicit coordinates')
    return dict(world=np.asarray(worlds),world_jacobian=np.asarray(jacobians),
                pattern_jacobian=np.zeros((26,3,pattern_count)),ledger=ledger,
                pattern_binding='No Ad12/13 pattern source in Ad1..7; zero columns explicit, no relabelled Ad5 internal graphic')


def load_secondary_entity_support(packet_path=None):
    """Strict frozen visible-entity interval, not an exact pivot/origin pixel."""
    p = HERE/'wide_secondary_entity_source_packet.json' if packet_path is None else Path(packet_path)
    data = json.loads(p.read_bytes())
    if (data['schema'] != 'rinkspace.user-identified-secondary-camera-visible-entity-support.v1'
            or data['source_pts'] != PTS or data['source_video_sha256'] != SOURCE_SHA
            or data['installation_id'] != SECONDARY_INSTALLATION_ID
            or data['native_support_box_ltrb'] != list(NATIVE_ENTITY_BOX)
            or data['exact_optical_center_or_mount_origin_observed'] is not False
            or _sha(data['user_attachment_snapshot']['path']) != data['user_attachment_snapshot']['sha256']):
        raise ValueError('Frozen user camera-entity identity/support contract changed')
    return data


def secondary_support_residual(native_projected_uv, support):
    """Two raw px interval violations; no circle centre or invented sigma.

    Zero means inside the visible-entity support only. It is not evidence that
    the physical fixed-XYZ point equals the unknown optical centre/pivot.
    Caller must declare that modelling approximation and its working weight.
    """
    uv = np.asarray(native_projected_uv,float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError('One finite actual wide-camera projected point required')
    left,top,right,bottom = support['native_support_box_ltrb']
    nearest = np.clip(uv,[left,top],[right,bottom])
    jac = np.diag(((uv < [left,top]) | (uv > [right,bottom])).astype(float))
    return uv-nearest,jac


def write_ad_source_packet(output=None):
    p = HERE/'wide_user_26_ad_source_packet.json' if output is None else Path(output)
    packet = load_wide_user_ad_source()
    p.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return p
