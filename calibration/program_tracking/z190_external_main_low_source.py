"""Source-only factory for the real 16:59 main-to-low short window.

No programme view is reused and preparation never solves.  The 38 actual
exposures use the same main installation/lever/optics.  Each of two unknown
static structural points has ONE positive anchor optical depth, shared by
all later views.  A wall/ice plane or surveyed point is not introduced.

The user's explicit red-line definition binds the qualified BLUE UPPER edge
to the existing zero-thickness board_top, at its current estimated height.
The distinct lower blue-to-white edge is history/report-only.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from z190_fixed_target_cross_view_factor import fixed_target_projection, fixed_target_source_factor, native_bbox_support
from z190_secondary_connected_cap_source import connected_lower_left_cap
from shared_centrosymmetric_advertisement_geometry import SharedCentrosymmetricAdvertisementGeometry

SCENE_UNIT_M = 0.31353729994266055
SOURCE_ROOT = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/ice_alignment_review/user_cross_camera_jpeg_source_trace_v1')
AUTHORITY = {
    'aliases': ('external_native_actual_exposure_aliases_SOURCE_QUALIFIED_FROZEN.json', '7bca239417faf70bbc2bf4ad47c061c7575c77af2d29da53d17dd4623dd8524a'),
    'upper': ('external_blue_cap_upper_lower_semantics_SOURCE_QUALIFIED_FROZEN.json', '08f5ae37bd7847458a72277c3fb09194b2859ef9e164d366431a2d5e15afd8b0'),
    'tracks': ('external_main_observer_short_window_source_QUALIFIED_FROZEN.json', '6a069aca5ce4ba2a26827527c1460c07a85669cc2c33483e9e4777da736448d0'),
    'heads': ('main_warmup_to_low_whole_pan_tilt_head_support_QUALIFIED_FROZEN_v2.json', 'be1ea9f60ab82da2f33ba47297c2a9e8753c54b76e8e291129faab531fa5df69'),
    'ad14': ('ad14_source_only_v1/external_Ad14_visible_corners_source_QUALIFIED_FROZEN_v1.json', 'ea215e4812f7af6a6d2d8823abf58fe00f3a40acbef462db4424716efe1cb61c'),
    'patterns': ('ad13_ad12_pattern_source_only_v2/external_Ad13_Ad12_patterns_SOURCE_QUALIFIED_FROZEN_v2.json', 'ed121036301b33c1b8cb47b90178709121cf984eca6523c8b5e94c799e60ee85'),
}
USER_TOP_ENTITY_AUTHORITY = {
    'kind': 'human_explicit_project_entity_definition',
    'instruction': '用户红线确认：近侧灰色朝冰斜边与远侧蓝色上边统一为零厚度围挡上沿，同一高度。',
    'definition_overrides_old_source_packet_world_pending': True,
    'numeric_height': 'existing working scene estimate, not a user measured number',
    'extra_width_thickness_or_lip_parameters': False,
}


def _load_authority(root, kind):
    name, expected = AUTHORITY[kind]
    path = root/name
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError('Frozen external source authority changed: '+kind)
    return json.loads(raw.decode('utf-8')), dict(path=str(path), sha256=actual)


def _native_ray(uv, logf, pp, k1):
    """Central inverse Brown ray, and exact derivative with respect to logf."""
    f = float(np.exp(float(logf)))
    if not np.isfinite(f) or f <= 0:
        raise ValueError('External focal must be finite and positive')
    distorted = (np.asarray(uv, float)-pp)/f
    rd = float(np.linalg.norm(distorted))
    if rd == 0:
        return np.array([0., 0., 1.]), np.zeros(3)
    radius = rd
    for _ in range(30):
        derivative = 1.+3.*k1*radius*radius
        if derivative <= 0:
            raise ValueError('Anchor ray outside central monotonic Brown domain')
        step = (radius+k1*radius**3-rd)/derivative
        radius -= step
        if radius < 0 or not np.isfinite(radius):
            raise ValueError('No positive central inverse Brown anchor ray')
        if abs(step) < 1e-13:
            break
    else:
        raise ValueError('Inverse Brown anchor ray did not converge')
    n = distorted*(radius/rd)
    radial = 1.+k1*float(n@n)
    if radial <= 0 or 1.+3.*k1*float(n@n) <= 0:
        raise ValueError('Anchor ray outside central monotonic Brown domain')
    dn = np.linalg.solve(radial*np.eye(2)+2.*k1*np.outer(n, n), -distorted)
    return np.r_[n, 1.], np.r_[dn, 0.]


def shared_anchor_point(observer_xyz_m, anchor_view, anchor_uv_native, log_depth_m,
                        lever_m, principal_px, k1):
    """One unknown material point, not a plane or an absolute world anchor.

    exp(log_depth_m) is positive forward optical depth at the anchor.  Source
    UV is a conditional pixel-pair reference with engineering sigma6.
    All anchor rotation/focal/lever dependencies remain in the Jacobian.
    """
    observer = np.asarray(observer_xyz_m, float)
    view = np.asarray(anchor_view, float)
    depth = float(np.exp(float(log_depth_m)))
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError('One shared positive finite anchor depth is required')
    ray, dlogf = _native_ray(anchor_uv_native, view[3], principal_px, k1)
    rotation, raw = cv2.Rodrigues(view[:3])
    dr = raw.reshape(3, 3, 3)
    local = lever_m+depth*ray
    point = observer+rotation.T@local
    jview = np.c_[np.column_stack([d.T@local for d in dr]), rotation.T@(depth*dlogf)]
    return SimpleNamespace(world_xyz_m=point, anchor_view_jacobian=jview,
        log_depth_jacobian=rotation.T@(depth*ray), observer_xyz_jacobian=np.eye(3),
        anchor_optical_depth_m=depth, point_position_unknown=True, assigned_surface=None)


def _project_curve(world_m, observer, view, lever, pp, k1):
    """Exact vertex projection/J. Invalid ends are not invented or clamped."""
    rotation, raw = cv2.Rodrigues(view[:3]); dr = raw.reshape(3, 3, 3)
    delta = world_m-observer
    camera = delta@rotation.T-lever
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 0)
    n = np.full((len(world_m), 2), np.nan)
    n[valid] = camera[valid, :2]/z[valid, None]
    r2 = np.einsum('ni,ni->n', n, n)
    radial = 1.+k1*r2
    valid &= (radial > 0) & (1.+3.*k1*r2 > 0)
    f = float(np.exp(view[3]))
    if not np.isfinite(f) or f <= 0:
        raise ValueError('External focal outside finite positive domain')
    uv = np.full((len(world_m), 2), np.nan)
    uv[valid] = pp+f*n[valid]*radial[valid, None]
    jp = np.zeros((len(world_m), 2, 3)); jv = np.zeros((len(world_m), 2, 4))
    ix = np.flatnonzero(valid); nn = n[ix]
    brown = radial[ix, None, None]*np.eye(2)+2.*k1*nn[:, :, None]*nn[:, None, :]
    normalized = np.zeros((len(ix), 2, 3)); normalized[:, :, :2] = np.eye(2)
    normalized[:, :, 2] = -nn; normalized /= z[ix, None, None]
    pc = f*np.einsum('nij,njk->nik', brown, normalized)
    jp[ix] = -np.einsum('nij,jk->nik', pc, rotation)
    tangent = np.einsum('aij,nj->nia', dr, delta[ix])
    jv[ix, :, :3] = np.einsum('nij,njk->nik', pc, tangent)
    jv[ix, :, 3] = uv[ix]-pp
    return SimpleNamespace(uv=uv, valid=valid, observer_xyz_jacobian=jp, view_jacobian=jv)


def _unique_y_at_source_x(projection, source_x):
    uv = projection.uv
    dx = uv[1:, 0]-uv[:-1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = (source_x-uv[:-1, 0])/dx
    mask = projection.valid[:-1] & projection.valid[1:] & (np.abs(dx)>1e-8) & (fraction>=0.) & (fraction<=1.)
    ids = np.flatnonzero(mask)
    # A shared projected polyline vertex is one intersection, not two branches.
    if len(ids)==2 and ids[1]==ids[0]+1 and abs(fraction[ids[0]]-1.)<1e-10 and abs(fraction[ids[1]])<1e-10:
        ids=ids[:1]
    if len(ids)!=1:
        raise ValueError('Source fixed-X needs exactly one connected board_top Y intersection; got '+str(len(ids)))
    i=int(ids[0]); t=float(fraction[i]); slope=(uv[i+1, 1]-uv[i, 1])/dx[i]
    y=float(uv[i, 1]+t*(uv[i+1, 1]-uv[i, 1]))
    def combine(j):
        return (1.-t)*(j[i, 1]-slope*j[i, 0])+t*(j[i+1, 1]-slope*j[i+1, 0])
    return SimpleNamespace(y=y, segment_index=i, fraction=t,
        observer_xyz_jacobian=combine(projection.observer_xyz_jacobian),
        view_jacobian=combine(projection.view_jacobian))


class ExternalMainLowSourceBundle:
    """Private local state: [38*(R3,log native f), 2*log anchor depth m]."""
    def __init__(self, scene, calibration, source_root=SOURCE_ROOT):
        root=Path(source_root); loaded={}; refs={}
        for kind in AUTHORITY:
            loaded[kind], refs[kind] = _load_authority(root, kind)
        a=loaded['aliases']; s=loaded['tracks']
        self.canonical_pts=np.array([r['canonical_source_native_pts'] for r in a['groups']], np.int64)
        self.aliases={int(k):int(v) for k,v in a['aliases_source_native_to_canonical_native'].items()}
        self.index={int(p):i for i,p in enumerate(self.canonical_pts)}
        self.held_pts=frozenset(self.aliases[int(p)] for p in a['held_native_pts'])
        if len(self.canonical_pts)!=38 or self.held_pts!={15657728}:
            raise ValueError('Real external actual exposure inventory changed')
        self.calibration=calibration
        self.ad_geometry=SharedCentrosymmetricAdvertisementGeometry(scene)
        self.scene=self.ad_geometry.scene()
        # Support only the fixed optical contract used by the current cameras.
        if any(calibration.axial) or any(calibration.k2) or any(calibration.tangential_p1_p2) or len(calibration.k1)!=1 or len(calibration.principal_x_fraction)!=1 or len(calibration.principal_y_fraction)!=1 or calibration.aspect_fx_over_fy!=1.:
            raise ValueError('External block requires the unchanged simple fixed Z190 optical contract')
        self.lever_m=np.asarray(calibration.lever_xyz, float)*SCENE_UNIT_M
        self.pp=np.array([1920*calibration.principal_x_fraction[0],1080*calibration.principal_y_fraction[0]])
        self.k1=float(calibration.k1[0])
        if not np.any(self.lever_m):
            raise ValueError('The current nonzero main lens lever must be retained')
        context_world, topology=connected_lower_left_cap(scene)
        # These readings are on the grey left-end course adjoining Ad14/13.
        # Ad14 is the centre-symmetric upper-right Ad1 (a corner advertisement),
        # so the short-side/corner join is retained, with no invented tangent
        # station. The near long side is a different source physical branch.
        x0=float(scene['parameters']['bounds'][0])
        rx=float(scene['parameters']['corner_curve']['tangent_extent_xy_scene_units'][0])
        short=context_world[context_world[:,0]<=x0+rx+1e-7]
        if len(short)<2:
            raise ValueError('Existing physical left-short top branch missing')
        self.top_world_scene=short
        self.top_world_m=short*SCENE_UNIT_M
        topology.update(source_factor_physical_facets=['left_short_side','lower_left_corner'],
            branch_source_authority='All12 source x1400/1650/1850 readings lie on the grey-end course above the same user-confirmed TC Ad13 adjoining corner APP Ad14; continuous raw frames, not model projection. No exact corner tangent station assigned.',
            full_connected_course_retained_for_display_only=True,
            wrong_near_lower_long_branch_excluded_by_source_identity=True)
        self.upper_rows=[]
        for original in loaded['upper']['rows']:
            row=deepcopy(original); native=int(row['source_native_pts']); actual=self.aliases[native]
            if row['physical_seen_boundary']!='blue_rail_visible_upper_colour_rise' or row['source_visible']is not True or row['person_occlusion']is not False or row['model_projection_used']is not False:
                raise ValueError('Only qualified true blue UPPER source boundary is admitted')
            row.update(actual_pts=actual, split='held' if actual in self.held_pts else 'fit',
                id=f'external_main_board_top_upper_{native}_x{row["source_u_native"]}',
                physical_geometry_id='board_top_left_short_lower_left_corner',
                physical_entity_authority=USER_TOP_ENTITY_AUTHORITY,
                curve_observation_kind='axis_scanline_branch',
                scanline_axis='x', branch_requirement='unique_y', world_units='metres')
            self.upper_rows.append(row)
        self.track_rows=[]; self.track_sources=[]
        for ti, track in enumerate(s['persistent_static_source_tracks']):
            if not track['qualification'].startswith('QUALIFIED same visible static') or track['model_projection_used']is not False:
                raise ValueError('Unknown or model-selected material cannot become persistent structure')
            natives=list(map(int,track['native_pts'])); uvs=track['uv_native']
            if len(natives)!=46 or len(uvs)!=46:
                raise ValueError('Structural original native ledger changed')
            lookup=dict(zip(natives, uvs)); anchor=int(self.canonical_pts[0])
            info=dict(track_id='external_door_c'+str(track['proposal_id']), anchor_actual_pts=anchor,
                anchor_source_native_pts=anchor, anchor_uv_native=deepcopy(lookup[anchor]),
                unknown_world_position=True, surface_id=None, original_motion_index=None,
                source_sigma_px=6., sigma_semantics='conditional pixel-pair engineering scale; source anchor ray is reference, not a surveyed point',
                all_native_ledger=deepcopy(track), static_material_qualified=True)
            self.track_sources.append(info)
            for p in self.canonical_pts:
                p=int(p)
                if p==anchor: continue
                self.track_rows.append(dict(id=info['track_id']+'_to_'+str(p), track_index=ti,
                    track_id=info['track_id'], source_native_pts=p, actual_pts=p,
                    target_uv_native=deepcopy(lookup[p]), anchor_actual_pts=anchor,
                    source_sigma_px=6., split='held'if p in self.held_pts else'fit',
                    shared_static_point=True, surface_id=None, original_motion_index=None,
                    source_visible=True, target_visible=True, model_projection_used=False,
                    source_authority=refs['tracks']))
        self.head_rows=[]
        for original in loaded['heads']['rows']:
            row=deepcopy(original); p=int(row['source_native_pts']); actual=self.aliases[p]
            lower=row['physical_axis_broad_support_xy_min']; upper=row['physical_axis_broad_support_xy_max']
            row.update(actual_pts=actual, source_actual_pts=actual, split='held'if actual in self.held_pts else'fit',
                source_visible=True, source_support_manually_qualified=True,
                physical_target_kind='fixed_pan_head_pivot', fixed_pivot_source_qualified=True,
                physical_target_authority=dict(frozen_source=refs['heads'], physical_limit=row['physical_origin_uncertainty']),
                native_support_bounds_xy_px=[[lower[0],upper[0]],[lower[1],upper[1]]],
                sigma_native_px=[row['source_sigma_engineering_native_px']]*2)
            self.head_rows.append(row)
        self.ad14_rows=[]
        for original in loaded['ad14']['rows']:
            row=deepcopy(original);actual=self.aliases[int(row['source_native_pts'])]
            if row['physical_id']!='Ad14' or row['source_corner_role']not in ('right_top','right_bottom') or row['source_visible']is not True or row['person_occlusion']is not False or row['model_projection_used_to_generate_or_select_uv']is not False:
                raise ValueError('Ad14 source must name a true visible physical print corner')
            world=self.ad_geometry.corner(14,row['source_corner_role'])*SCENE_UNIT_M
            row.update(actual_pts=actual,split='held'if actual in self.held_pts else'fit',world_xyz_m=world.tolist(),
                world_binding_authority=dict(kind='shared actual AdGeometry with user centre-symmetry',physical_print_corner_role=row['source_corner_role'],source_and_render_same=True))
            self.ad14_rows.append(row)
        self.pattern_source_records=deepcopy(loaded['patterns']['rows'])
        self.local_size=4*len(self.canonical_pts)+len(self.track_sources)
        self.manifest=dict(schema='rinkspace.external-main-low-source-bundle.v1', source_authorities=refs,
            user_entity_definition=USER_TOP_ENTITY_AUTHORITY, canonical_pts=self.canonical_pts.tolist(),
            native_aliases=a['aliases_source_native_to_canonical_native'], actual_count=38, native_count=46,
            top_development_rows=9, top_held_rows=3, head_development_rows=3, head_held_rows=1,
            Ad14_development_corners=6,Ad14_held_corners=2,
            Ad14_world_authority=self.ad_geometry.manifest(),
            additional_actual_print_pattern_sources=dict(Ad13_development_outer_ring_intervals=48,Ad13_held_outer_ring_intervals=16,Ad12_pre_window_outer_silhouette_intervals=12,
                physical_identity_user_confirmed=True,source_visibility_qualified=True,
                shared_board_pattern_world_position_shape_parameters_not_yet_materialized=True,
                not_used_as_invented_absolute_world_anchors=True),
            persistent_target_development_rows=72, persistent_target_held_rows=2,
            persistent_reference_anchors=2, unknown_static_world_points=2,
            physical_course_topology=topology, original_source_packets_unchanged=True,
            old_blue_lower_front_rows_not_used=True, numbered_advertisements_or_ice_paint_world_anchors_in_this_window=['Ad14:right_top','Ad14:right_bottom'],
            programme_view_reuse=False, preparation_has_solve_side_effect=False,
            held_actual_without_direct_training_motion=[15657728],
            held_view_policy='All JPEG source targets/absolute supports held. Its Rf needs explicitly labelled neighbour-time inference or a separately qualified different-material motion family; never claim it directly source recovered.',
            local_q_schema=['38 actual (R3,log native focal px)','2 log shared anchor optical depth m'],
            static_world_points_are_not_absolute_anchors=True, fixed_optics_and_nonzero_lever=True)

    def unpack(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.local_size,) or not np.isfinite(q).all():
            raise ValueError('External local q has wrong size or nonfinite values')
        views=q[:4*len(self.canonical_pts)].reshape(-1,4)
        logs=q[4*len(self.canonical_pts):]
        return views, logs

    def shared_points(self,q,observer_xyz_m):
        views,logs=self.unpack(q); result=[]
        for ti,source in enumerate(self.track_sources):
            vi=self.index[source['anchor_actual_pts']]
            result.append(shared_anchor_point(observer_xyz_m, views[vi], source['anchor_uv_native'], logs[ti], self.lever_m,self.pp,self.k1))
        return result

    def evaluate(self,q,observer_xyz_m,target_xyz_m,*,jacobian=True,held=False,include_roll=True,include_heads=True):
        """Exact local/mainXYZ/lowXYZ columns, no implicit global solve.

        Held uses only frozen held measurements and never adds them to fit.
        JPEG local view remains an explicit state; see manifest held_view_policy.
        """
        views,logs=self.unpack(q); observer=np.asarray(observer_xyz_m,float); target=np.asarray(target_xyz_m,float)
        if observer.shape!=(3,)or target.shape!=(3,)or not np.isfinite(observer).all()or not np.isfinite(target).all():
            raise ValueError('Fixed observer/target installation XYZ must be finite metres')
        split='held'if held else'fit'; residuals=[]; jl=[]; jm=[]; jt=[]; ledger=[]; curves={}
        points=self.shared_points(q,observer)
        def add(value,local,main,low,source):
            value=np.asarray(value,float).reshape(-1); residuals.append(value)
            if jacobian:
                jl.append(np.asarray(local,float).reshape(len(value),self.local_size))
                jm.append(np.asarray(main,float).reshape(len(value),3)); jt.append(np.asarray(low,float).reshape(len(value),3))
            ledger.append(dict(id=source['id'],actual_pts=int(source['actual_pts']),split=split,
                residual_slice=[sum(len(x)for x in residuals[:-1]),sum(len(x)for x in residuals)],
                source_native_pts=int(source.get('source_native_pts',source['actual_pts']))))
        for row in self.upper_rows:
            if row['split']!=split:continue
            p=row['actual_pts']; vi=self.index[p]
            if p not in curves:curves[p]=_project_curve(self.top_world_m,observer,views[vi],self.lever_m,self.pp,self.k1)
            intersection=_unique_y_at_source_x(curves[p],float(row['source_u_native']))
            lo,hi=row['source_broad_y_support_px']; outside=intersection.y-np.clip(intersection.y,lo,hi)
            factor=float(outside!=0.)/float(row['source_sigma_px'])
            local=np.zeros((1,self.local_size));local[0,4*vi:4*vi+4]=factor*intersection.view_jacobian
            add([outside/row['source_sigma_px']],local,factor*intersection.observer_xyz_jacobian,np.zeros(3),row)
        for row in self.track_rows:
            if row['split']!=split:continue
            ti=row['track_index']; vi=self.index[row['actual_pts']]; ai=self.index[row['anchor_actual_pts']]; point=points[ti]
            projection=fixed_target_projection(observer,point.world_xyz_m,views[vi,:3],views[vi,3],self.lever_m,self.pp,self.k1)
            sigma=row['source_sigma_px']; value=(projection.uv_native_px-np.asarray(row['target_uv_native']))/sigma
            local=np.zeros((2,self.local_size))
            local[:,4*vi:4*vi+4]+=projection.observer_view_R3_logf_jacobian/sigma
            local[:,4*ai:4*ai+4]+=projection.target_xyz_m_jacobian@point.anchor_view_jacobian/sigma
            local[:,4*len(self.canonical_pts)+ti]=projection.target_xyz_m_jacobian@point.log_depth_jacobian/sigma
            # W shifts with the same main P, so the source-only unknown material
            # cannot be misreported as an absolute installation position anchor.
            main=(projection.observer_xyz_m_jacobian+projection.target_xyz_m_jacobian)/sigma
            add(value,local,main,np.zeros((2,3)),row)
        for row in self.head_rows:
            if row['split']!=split or not include_heads:continue
            vi=self.index[row['actual_pts']]
            factor=fixed_target_source_factor(row,observer,target,views[vi,:3],views[vi,3],self.lever_m,self.pp,self.k1)
            local=np.zeros((2,self.local_size));local[:,4*vi:4*vi+4]=factor.observer_view_R3_logf_jacobian
            add(factor.residual,local,factor.observer_xyz_m_jacobian,factor.target_xyz_m_jacobian,row)
        for row in self.ad14_rows:
            if row['split']!=split:continue
            vi=self.index[row['actual_pts']]
            projection=fixed_target_projection(observer,row['world_xyz_m'],views[vi,:3],views[vi,3],self.lever_m,self.pp,self.k1)
            factor=native_bbox_support(projection,row['native_support_bounds_xy_px'],row['source_sigma_native_px'])
            local=np.zeros((2,self.local_size));local[:,4*vi:4*vi+4]=factor.observer_view_R3_logf_jacobian
            add(factor.residual,local,factor.observer_xyz_m_jacobian,np.zeros((2,3)),row)
        if include_roll and not held:
            sigma=np.deg2rad(.5)
            for vi,p in enumerate(self.canonical_pts):
                rotation,raw=cv2.Rodrigues(views[vi,:3]);dr=raw.reshape(3,3,3)
                x,y=rotation[0,2],-rotation[1,2];norm=x*x+y*y
                if norm<1e-8:raise ValueError('External physical horizon roll undefined')
                local=np.zeros((1,self.local_size));local[0,4*vi:4*vi+3]=(y*dr[:,0,2]+x*dr[:,1,2])/norm/sigma
                add([np.arctan2(x,y)/sigma],local,np.zeros(3),np.zeros(3),dict(id='roll_'+str(p),actual_pts=p))
        value=np.concatenate(residuals)if residuals else np.empty(0)
        if not np.isfinite(value).all():raise ValueError('External source residual left finite domain')
        return SimpleNamespace(residual=value,local_jacobian=np.vstack(jl)if jacobian else None,
            observer_xyz_m_jacobian=np.vstack(jm)if jacobian else None,
            target_xyz_m_jacobian=np.vstack(jt)if jacobian else None,
            source_row_ledger=ledger,shared_unknown_points=points,held=held,
            main_static_material_translation_jacobian_is_zero=True)

    def guard(self,q,observer_xyz_m,target_xyz_m,*,check_held=False):
        try:
            fit=self.evaluate(q,observer_xyz_m,target_xyz_m,jacobian=False,held=False)
            if check_held:self.evaluate(q,observer_xyz_m,target_xyz_m,jacobian=False,held=True)
            return dict(valid=True,fit_residual_count=len(fit.residual),unknown_material_world_xyz_m=[p.world_xyz_m.tolist()for p in fit.shared_unknown_points],
                unknown_material_world_xyz_is_a_conditional_result_not_a_measured_anchor=True)
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError)as exc:
            return dict(valid=False,reason=str(exc))

    def infer_held_view_from_neighbours(self,q):
        """Explicit temporal inference, not source recovery or a new UV read."""
        q=np.asarray(q,float).copy();views,_=self.unpack(q);reports=[]
        for p in sorted(self.held_pts):
            i=self.index[p];left=i-1;right=i+1
            if left<0 or right>=len(views):raise ValueError('Held view has no real neighbouring development states')
            pl,pr=int(self.canonical_pts[left]),int(self.canonical_pts[right]);t=(p-pl)/(pr-pl)
            rotations=Rotation.from_rotvec(views[[left,right],:3]);views[i,:3]=Slerp([pl,pr],rotations)([p]).as_rotvec()[0]
            views[i,3]=(1.-t)*views[left,3]+t*views[right,3]
            reports.append(dict(actual_pts=p,left_actual_pts=pl,right_actual_pts=pr,
                method='SLERP plus linear logf between recovered neighbours, same fixed installation',
                source_camera_state_directly_recovered=False,source_measurements_used=[],inference_fraction=t))
        return q,reports

    def initialize(self,observer_xyz_m,target_xyz_m,*,native_focal_seed_px,anchor_depth_seed_m):
        """Coarse same-window seed only; no minimization and no reused view.

        Explicit focal/depth seeds are numerical guesses, not measured values.
        A true Ad14 right-border midpoint locates an initial aim only; fit
        uses its two full frozen corner supports. Caller conditionally recovers the
        37 observed Rf states with evaluate/J (main/low XYZ held fixed first).
        """
        f=float(native_focal_seed_px);depths=np.broadcast_to(np.asarray(anchor_depth_seed_m,float),(2,))
        if not np.isfinite(f)or f<=0 or not np.isfinite(depths).all()or np.any(depths<=0):
            raise ValueError('Caller must explicitly give positive numerical focal/depth seeds')
        observer=np.asarray(observer_xyz_m,float);target=np.asarray(target_xyz_m,float);rv=[];stamps=[]
        points_by_pts={}
        for row in self.ad14_rows:
            if row['split']=='fit':points_by_pts.setdefault(row['actual_pts'],[]).append(row)
        for p,rows in sorted(points_by_pts.items()):
            if len(rows)!=2:raise ValueError('Initial source aim requires both genuinely visible right-border endpoints')
            world_mid=np.mean([row['world_xyz_m']for row in rows],axis=0)
            uv=np.mean([row['nominal_uv_search_only']for row in rows],axis=0)
            delta=world_mid-observer;distance=np.linalg.norm(delta)
            forward=delta/distance;right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            base=np.vstack([right,down,forward])
            ray,_=_native_ray(uv,np.log(f),self.pp,self.k1)
            aa=ray@ray;bb=2.*float(self.lever_m@ray);cc=float(self.lever_m@self.lever_m)-distance*distance
            discriminant=bb*bb-4.*aa*cc
            if discriminant<=0:raise ValueError('Numerical head aim has no positive observer optical depth')
            z=(-bb+np.sqrt(discriminant))/(2.*aa);desired=(self.lever_m+z*ray)/distance
            axis=np.cross([0.,0.,1.],desired);norm=np.linalg.norm(axis)
            if norm<1e-14:align=np.eye(3)
            else:align=cv2.Rodrigues(axis/norm*np.arctan2(norm,desired[2]))[0]
            rv.append(cv2.Rodrigues(align@base)[0].reshape(3));stamps.append(p)
        rv=np.asarray(rv);stamps=np.asarray(stamps);views=np.zeros((len(self.canonical_pts),4));views[:,3]=np.log(f)
        # Only initial guesses. Final Rf stays separate for every actual image.
        for j in range(3):views[:,j]=np.interp(self.canonical_pts,stamps,rv[:,j])
        q=np.r_[views.ravel(),np.log(depths)]
        return SimpleNamespace(q=q,guard=self.guard(q,observer,target),
            report=dict(solved=False,numerical_focal_seed_px=f,numerical_anchor_depth_seed_m=depths.tolist(),
                source_housing_or_support_centre_fitted_as_point=False,programme_view_reused=False,
                initial_aim_source='same-window three genuinely visible Ad14 right-border endpoint midpoints, initialization only; fit keeps both source corner intervals',
                remaining_step='conditional 38-state window source recovery; only37 states have development material observations. Infer held state explicitly from recovered real neighbours and then evaluate all frozen held source.',
                unknown_static_points_are_numerical_nuisances_not_world_anchors=True))

    def recover_conditional(self,observer_xyz_m,target_xyz_m,*,q_seed,max_iterations,
                            include_heads=False,relative_cost_tolerance=1e-9):
        """Explicit opt-in LOCAL initializer, never a programme/global solve.

        Main/low XYZ, lever/PP/k1 and all world geometry stay fixed. By default
        only the real absolute Ad14/top plus unknown persistent structure
        recover observer Rf. The unknown low P is not used as a pose anchor;
        dev/held head supports remain available for the subsequent common XYZ
        system. With include_heads=True this is explicitly conditional on the
        supplied low P estimate. The caller must label either result unaccepted.

        The fully held JPEG Rf has no fit measurements and is excluded from
        the solve. Afterwards it receives explicit neighbour-time inference.
        """
        q=np.asarray(q_seed,float).copy();self.unpack(q)
        if not isinstance(max_iterations,int) or max_iterations<=0:
            raise ValueError('Caller must specify a positive local initialization iteration limit')
        excluded={4*self.index[p]+j for p in self.held_pts for j in range(4)}
        active=np.array([i for i in range(self.local_size)if i not in excluded],int)
        damping=1e-2;history=[];stop='caller_iteration_limit'
        for iteration in range(max_iterations):
            e=self.evaluate(q,observer_xyz_m,target_xyz_m,include_heads=include_heads)
            residual=e.residual;cost=.5*float(residual@residual);j=e.local_jacobian[:,active]
            h=j.T@j;g=j.T@residual;diag=np.maximum(np.diag(h),1.)
            try:step=np.linalg.solve(h+damping*np.diag(diag),-g)
            except np.linalg.LinAlgError:damping*=10.;continue
            accepted=False
            for power in range(17):
                scale=.5**power;candidate=q.copy();candidate[active]+=scale*step
                try:
                    v=self.evaluate(candidate,observer_xyz_m,target_xyz_m,jacobian=False,include_heads=include_heads).residual
                    trial=.5*float(v@v)
                except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):continue
                if trial<cost:
                    q=candidate;history.append(dict(iteration=iteration,cost_before=cost,cost_after=trial,step_fraction=scale,damping=damping))
                    accepted=True;damping=max(damping/3.,1e-12)
                    if (cost-trial)/max(cost,1.)<relative_cost_tolerance:stop='relative_cost_convergence'
                    break
            if stop=='relative_cost_convergence':break
            if not accepted:
                damping*=10.
                if damping>1e12:stop='source_domain_or_no_descent';break
        q,inference=self.infer_held_view_from_neighbours(q)
        final=self.evaluate(q,observer_xyz_m,target_xyz_m,include_heads=include_heads)
        sv=np.linalg.svd(final.local_jacobian[:,active],compute_uv=False)
        rank=int(np.count_nonzero(sv>sv[0]*max(final.local_jacobian[:,active].shape)*np.finfo(float).eps))
        return SimpleNamespace(q=q,history=history,guard=self.guard(q,observer_xyz_m,target_xyz_m,check_held=True),
            report=dict(kind='conditional local38 exposure initialization only',external_camera_accuracy_accepted=False,
                fixed_observer_xyz_m=np.asarray(observer_xyz_m).tolist(),fixed_target_xyz_m=np.asarray(target_xyz_m).tolist(),
                target_head_support_used_during_observer_initialization=bool(include_heads),
                retained_source_uv_sigmas_aliases_roles=True,held_measurements_fitted=False,
                held_camera_inference=inference,unknown_static_points_not_absolute_anchors=True,
                active_local_parameters=len(active),final_numerical_rank=rank,singular_values=sv.tolist(),
                final_source_cost=.5*float(final.residual@final.residual),stop_reason=stop,
                no_programme_optimizer_invoked=True,no_default_or_scene_modified=True))


def prepare_external_main_low_source(scene,calibration,*,source_root=SOURCE_ROOT):
    """Return an immutable-source evaluator; no solver, cache or scene write."""
    if isinstance(scene,(str,Path)):scene=json.loads(Path(scene).read_text(encoding='utf-8'))
    return ExternalMainLowSourceBundle(scene,calibration,source_root)
