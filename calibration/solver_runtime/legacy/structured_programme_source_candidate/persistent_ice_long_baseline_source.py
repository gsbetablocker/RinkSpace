"""Append real same-material ice correspondences to the existing source.

Every evaluation intersects an unchanged native anchor UV with z=0 using the
current anchor camera25, then projects that current point into the actual
target camera25. No historical world point or new camera column is fitted.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from .native_camera_local import native_camera_local
from scipy.sparse import csr_matrix

from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from .native_ice_curve_source import SOURCE_SHA, _sha
from .fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_coordinate_derivative,
)
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES

PACKET_SCHEMA = 'rinkspace.persistent-ice-z0-source.v1'


def current_native_rays(camera, native_uv):
    """The existing circle initializer's full Brown14/skew inverse chart."""
    c = np.asarray(camera, float)
    uv = np.asarray(native_uv, float).reshape(-1, 2)
    if c.shape != (25,) or not np.isfinite(c).all() or not np.isfinite(uv).all():
        raise ValueError('Finite actual camera25 and unchanged native UV required')
    fx, fy = np.exp(c[6:8])
    deskew = uv.copy()
    deskew[:, 0] -= c[10]*(uv[:, 1]-c[9])/fy
    K = np.array([[fx,0.,c[8]], [0.,fy,c[9]], [0.,0.,1.]])
    normalized = cv2.undistortPointsIter(deskew.reshape(-1,1,2), K, c[11:25], None, None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
    rays = np.column_stack((normalized, np.ones(len(normalized)))) @ cv2.Rodrigues(c[3:6])[0]
    projected, _, valid = project_central_valid_full_brown(c, c[:3]+100*rays)
    closure = np.linalg.norm(projected-uv, axis=1)
    if not np.isfinite(rays).all() or not np.all(valid) or np.any(closure>1e-6):
        raise ValueError('Current inverse ray does not close the actual full-Brown native chart')
    return rays


def current_anchor_ice_points(camera, native_uv):
    c = np.asarray(camera, float)
    rays = current_native_rays(c, native_uv)
    if np.any(abs(rays[:,2])<1e-12):
        raise ValueError('Actual anchor ray is parallel to the declared ice plane')
    distance = -c[2]/rays[:,2]
    if np.any(distance<=0) or not np.isfinite(distance).all():
        raise ValueError('Actual anchor source ray has no positive z=0 ice intersection')
    world = c[:3]+distance[:,None]*rays
    world[:,2] = 0.
    if not np.isfinite(world).all():
        raise ValueError('Current source-ray ice intersection is not finite')
    return world


def _inside_finite_ice(points, polygon):
    """Float64 odd/even membership on the actual finite course polygon."""
    xy = np.asarray(points, float)[:,:2]
    border = np.asarray(polygon, float)[:,:2]
    if len(border)>1 and np.array_equal(border[0],border[-1]):
        border = border[:-1]
    if len(border)<3 or not np.isfinite(border).all():
        raise ValueError('Current finite ice course required')
    a, b = border, np.roll(border,-1,axis=0)
    dx, dy = (b-a).T
    y, x = xy[:,1,None], xy[:,0,None]
    crossings = (a[:,1]>y)!=(b[:,1]>y)
    denominator = np.where(dy!=0,dy,1.)
    intersection_x = a[:,0]+dx*(y-a[:,1])/denominator
    inside = np.logical_xor.reduce(crossings & (x<intersection_x),axis=1)
    edges=b-a;length2=np.einsum('ni,ni->n',edges,edges)
    if np.any(length2<=0): raise ValueError('Current ice polygon has degenerate finite edges')
    delta=xy[:,None,:]-a[None,:,:]
    fraction=np.clip(np.einsum('pni,ni->pn',delta,edges)/length2,0.,1.)
    distance2=np.sum((delta-fraction[:,:,None]*edges)**2,axis=2)
    return inside | (np.min(distance2,axis=1)<=1e-20)


class PersistentIceLongBaselineFactor:
    def __init__(self, state, packet_path, parameter_count):
        self.state, self.parameter_count = state, int(parameter_count)
        self.geometry = state.geometry.current
        self.packet_path = Path(packet_path).resolve()
        packet = json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema') != PACKET_SCHEMA or packet.get('source_sha256') != SOURCE_SHA
                or packet.get('status') != 'SOURCE_RAW_MATERIAL_REVIEWED_FROZEN_NEW_FIT_AND_HELD'
                or packet.get('time_base') != '1/15360' or packet.get('frozen_before_fit') is not True
                or packet.get('model_projection_used_to_generate_or_select_uv') is not False
                or packet.get('repeated_anchor_uv_correlated') is not True
                or packet.get('source_no_world_xyz_supplied') is not True):
            raise ValueError('Frozen same-video persistent real ice-z0 source packet required')
        parent=packet.get('parent_source_packet',packet.get('parent_v1_source_packet'))
        if parent is not None:
            parent_path=Path(parent['path']).resolve()
            if _sha(parent_path)!=parent['sha256']:
                raise ValueError('Persistent source append parent changed')
            preceding=json.loads(parent_path.read_text(encoding='utf8'))
            prefix=preceding['new_rows']
            if packet['tracks']!=preceding['tracks'] or packet['new_rows'][:len(prefix)]!=prefix:
                raise ValueError('Persistent append changed earlier source UV, track, sigma or FIT/HELD role')
        anchors={int(t['anchor']['source_native_pts'])for t in packet.get('tracks',[])}
        if len(anchors)!=1:raise ValueError('One explicit common raw anchor required')
        self.anchor_pts=anchors.pop()
        self.fit_target_pts={int(r['target_native_pts'])for r in packet.get('new_rows',[])if r.get('split')=='fit'}
        self.held_target_pts={int(r['target_native_pts'])for r in packet.get('new_rows',[])if r.get('split')=='held'}
        if not self.fit_target_pts or not self.held_target_pts or self.fit_target_pts & self.held_target_pts:
            raise ValueError('Separate explicit FIT and HELD source target exposures required')
        self.endpoint_pts={self.anchor_pts}|self.fit_target_pts|self.held_target_pts
        fit_actual={state.original.aliases.get(p)for p in self.fit_target_pts}
        held_actual={state.original.aliases.get(p)for p in self.held_target_pts}
        if None in fit_actual|held_actual or fit_actual & held_actual:
            raise ValueError('FIT and HELD targets must be distinct authenticated actual exposures')
        self._frame_references = {}
        clock_path=Path(packet['source_clock_authority_path']).resolve()
        clock_sha=packet['source_clock_authority_sha256']
        if _sha(clock_path)!=clock_sha: raise ValueError('Authenticated real exposure-clock source changed')
        self.clock_reference=dict(path=str(clock_path),sha256=clock_sha)
        required_pts=self.endpoint_pts
        self.clock_rows={}
        with clock_path.open('r',encoding='utf8') as stream:
            for line in stream:
                source=json.loads(line)
                if int(source['full_pts']) in required_pts:
                    self.clock_rows[int(source['full_pts'])]=source
        tracks = {}
        for record in packet.get('tracks',[]):
            track = deepcopy(record)
            name = track.get('track_id')
            if (not isinstance(name,str) or not name or name in tracks
                    or track.get('surface_id') != 'ice_z0' or not track.get('material_identity')):
                raise ValueError('Distinct explicit same-static-ice-material track identities required')
            endpoint, uv = self._endpoint(track['anchor'])
            if endpoint['native_pts'] != self.anchor_pts:
                raise ValueError('Only the reviewed common spin anchor source is admitted')
            tracks[name] = dict(record=track,anchor=endpoint,uv=uv)
        if not tracks: raise ValueError('No persistent real ice source tracks')
        self.rows, self.held = [], []
        names, associations = set(), set()
        for record in packet.get('new_rows',[]):
            row = deepcopy(record)
            if (not isinstance(row.get('id'),str) or row['id'] in names or row.get('track_id') not in tracks
                    or row.get('surface_id') != 'ice_z0'
                    or row.get('same_static_material_confirmed') is not True
                    or row.get('both_end_manually_qualified') is not True
                    or row.get('source_visible') is not True or row.get('target_visible') is not True
                    or row.get('frozen_before_fit') is not True
                    or row.get('model_projection_used_to_generate_or_select_uv') is not False
                    or any(row.get(k) is not None for k in ('world','world_xyz','world_xyz_scene_units'))):
                raise ValueError('Explicit visible qualified source-only same-material correspondence required')
            names.add(row['id'])
            anchor, xy0 = self._endpoint(row['anchor'])
            target, xy1 = self._endpoint(row['target'])
            track = tracks[row['track_id']]
            if (anchor != track['anchor'] or not np.array_equal(xy0,track['uv'])
                    or not np.array_equal(np.asarray(row['xy0'],float),xy0)
                    or not np.array_equal(np.asarray(row['xy1'],float),xy1)
                    or int(row['source_native_pts']) != anchor['native_pts']
                    or int(row['target_native_pts']) != target['native_pts']):
                raise ValueError('Persistent track endpoint UV, native PTS or raw-source provenance changed')
            identity = (row['track_id'],target['native_pts'])
            if identity in associations: raise ValueError('Duplicate same-source ice correspondence row')
            associations.add(identity)
            sigma = np.asarray(row['sigma_native_px'],float)
            if sigma.shape != (2,) or not np.array_equal(sigma,np.array([6.,6.])) or not row.get('sigma_semantics'):
                raise ValueError('Unchanged declared two-axis engineering sigma6 required')
            split = row.get('split')
            if (split == 'fit' and target['native_pts'] in self.fit_target_pts):
                destination = self.rows
            elif (split == 'held' and target['native_pts'] in self.held_target_pts):
                destination = self.held
            else:
                raise ValueError('Persistent endpoint differs from its explicit FIT/HELD source role')
            row.update(anchor_endpoint=anchor,target_endpoint=target,anchor_uv=xy0.copy(),target_uv=xy1.copy(),sigma=sigma.copy())
            destination.append(row)
        if not self.rows or not self.held:
            raise ValueError('Real FIT correspondences and separately frozen HELD targets are both required')
        self.tracks = tracks
        self._border_cache = OrderedDict()
        self._cache = OrderedDict()
        actual = {r[k]['actual_pts'] for r in self.rows+self.held for k in ('anchor_endpoint','target_endpoint')}
        self.blocks = {p:state.bypts[p] for p in actual}
        fit_actual = {r[k]['actual_pts'] for r in self.rows for k in ('anchor_endpoint','target_endpoint')}
        self.camera_steps, active = {}, []
        for stamp, block in self.blocks.items():
            view=block.start+4*block.bundle.index[stamp]
            cols=np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,view+np.arange(4)].astype(int)
            if stamp in fit_actual: active.extend(cols)
            for col in cols: self.camera_steps[int(col)] = 1e-6
            for col in block.optics_chart.indices: self.camera_steps[int(col)] = 1e-4
            for col in range(view,view+3): self.camera_steps[col] = 1e-7
        self.active = np.unique(active).astype(int)
        self.core_count=int(state.structured_programme_coordinate_chart.core_count)
        self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        if (np.any(self.active<0) or np.any(self.active>=self.core_count)
                or np.intersect1d(self.active,self.world_indices).size):
            raise ValueError('Only actual endpoint XYZ/optics/mount/R3logf camera slots may be active')
        self.scales = np.concatenate([r['sigma'] for r in self.rows])
        self.quadratic = np.ones(len(self.scales),bool)
        self.row_count = len(self.scales)
        self.contract = dict(schema='rinkspace.actual-native-persistent-ice-long-baseline-factor.v1',
            packet_path=str(self.packet_path),packet_sha256=_sha(self.packet_path),
            source_video_sha256=SOURCE_SHA,time_base='1/15360',parameter_count=self.parameter_count,
            common_anchor_native_pts=self.anchor_pts,fit_ids=[r['id'] for r in self.rows],held_ids=[r['id'] for r in self.held],
            fit_target_native_pts=sorted({r['target_endpoint']['native_pts'] for r in self.rows}),
            held_target_native_pts=sorted(self.held_target_pts),scalar_rows=self.row_count,
            active_existing_camera_parameter_indices=self.active.tolist(),new_parameters=0,
            source_operator='current full-Brown25 anchor native ray -> current z0 intersection -> actual target full-Brown25 native reprojection vector',
            old_WORLD_not_used_as_observation=True,current_anchor_intersection_recomputed_every_numeric_camera_state=True,
            anchor_and_target_camera_chain='bundle.camera(native_camera_local(block,core), actual authenticated exposure)',
            raw_sigma_native_px=[6.,6.],sigma_is_engineering_weight_not_measured_uncertainty=True,
            source_clock_authority=deepcopy(self.clock_reference),
            common_anchor_correlated=True,tracks_or_targets_statistically_independent_not_claimed=True,
            covariance_or_effective_sample_count_not_measured=True,group_sqrt_N_multiplier_applied=False,
            held_used_in_fit=False,finite_ice_membership_is_domain_guard_not_new_observation=True,
            original_source_rows_masks_scales_profiles_and_clock_preserved=True,accuracy_accepted=False)

    def _endpoint(self, record):
        reference = deepcopy(record)
        native = int(reference['source_native_pts'])
        if native not in self.state.original.aliases:
            raise ValueError('Persistent source exposure has no authenticated original camera alias')
        actual = int(self.state.original.aliases[native])
        if actual not in self.state.bypts or native not in self.endpoint_pts:
            raise ValueError('Persistent ice endpoint lies outside this reviewed continuous spin interval')
        if 'source_actual_pts' in reference and int(reference['source_actual_pts']) != actual:
            raise ValueError('Persistent source actual exposure disagrees with authenticated alias')
        clock_path=Path(reference['source_clock_authority_path']).resolve()
        if (str(clock_path)!=self.clock_reference['path']
                or reference['source_clock_authority_sha256']!=self.clock_reference['sha256']):
            raise ValueError('Persistent endpoint exposure authority differs from frozen packet')
        clock=self.clock_rows.get(native)
        if clock is None or clock.get('accepted_official_pts') is None or int(clock['accepted_official_pts'])!=int(reference['official_pts']):
            raise ValueError('Persistent endpoint has no exact accepted official exposure identity')
        path, digest = Path(reference['raw_png']).resolve(), reference['raw_png_sha256']
        if path not in self._frame_references:
            if _sha(path) != digest: raise ValueError('Persistent ice raw native source frame changed')
            raw=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
            if (raw is None or raw.shape!=(1080,1920,3)
                    or hashlib.sha256(raw.tobytes()).hexdigest()!=reference.get('native_bgr_sha256')):
                raise ValueError('Persistent source pixels differ from frozen native BGR authority')
            self._frame_references[path] = digest
        elif self._frame_references[path] != digest:
            raise ValueError('Conflicting raw native source-frame identity')
        bgr = reference.get('native_bgr_sha256')
        if not isinstance(bgr,str) or len(bgr)!=64:
            raise ValueError('Frozen native BGR provenance must remain explicit')
        uv=np.asarray(reference['uv'],float)
        if uv.shape!=(2,) or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>=[1920.,1080.]):
            raise ValueError('Unchanged full-resolution finite native source UV required')
        return dict(native_pts=native,actual_pts=actual,raw_png=str(path),raw_png_sha256=digest,native_bgr_sha256=bgr),uv

    def _q(self, q):
        value=np.asarray(q,float)
        if value.shape!=(self.parameter_count,) or not np.isfinite(value).all():
            raise ValueError('Complete preserved current source state required')
        return value

    def _cameras(self, q, rows):
        core=np.asarray(q,float)[:self.core_count]
        stamps={r[k]['actual_pts'] for r in rows for k in ('anchor_endpoint','target_endpoint')}
        return {p:self.blocks[p].bundle.camera(native_camera_local(self.blocks[p],core),p) for p in stamps}

    def _border(self, q):
        world=np.asarray(q,float)[self.world_indices]
        key=world.tobytes()
        if key not in self._border_cache:
            course,cq,_=self.geometry.print_course_binding(world)
            self._border_cache[key]=np.asarray(course.sample_perimeter(cq)[0],float).copy()
            while len(self._border_cache)>2: self._border_cache.popitem(last=False)
        return self._border_cache[key]

    def _evaluate(self, q, rows, cameras=None):
        cameras=self._cameras(q,rows) if cameras is None else cameras
        ids=list(dict.fromkeys(r['track_id'] for r in rows))
        anchor=self.tracks[ids[0]]['anchor']['actual_pts']
        points=current_anchor_ice_points(cameras[anchor],np.array([self.tracks[k]['uv'] for k in ids]))
        if not np.all(_inside_finite_ice(points,self._border(q))):
            raise ValueError('Current source-ray z0 intersection leaves the actual finite ice')
        worlds=dict(zip(ids,points));predictions={}
        for target in {r['target_endpoint']['actual_pts'] for r in rows}:
            selected=[r for r in rows if r['target_endpoint']['actual_pts']==target]
            projected,_,valid=project_central_valid_full_brown(cameras[target],np.array([worlds[r['track_id']] for r in selected]))
            if not np.all(valid) or not np.isfinite(projected).all():
                raise ValueError('Persistent source point leaves current target central optical chart')
            predictions.update((r['id'],uv.copy()) for r,uv in zip(selected,projected))
        residual=np.concatenate([(predictions[r['id']]-r['target_uv'])/r['sigma'] for r in rows])
        if not np.isfinite(residual).all(): raise ValueError('Persistent source reprojection is not finite')
        return residual,predictions,worlds

    def guard(self, q):
        value=self._q(q)
        self._evaluate(value,self.rows+self.held)

    def block(self, q, jacobian=True, *, original_start=0):
        value=self._q(q)
        key=value[self.active].tobytes()
        cached=self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):
            # Course shape may change independently of camera slots. Enforce
            # finite physical support on each call, even for an r/J cache hit.
            self._evaluate(value,self.rows)
            residual,matrix=cached
        else:
            residual,_,_=self._evaluate(value,self.rows);matrix=None
            if jacobian:
                columns=np.empty((len(residual),len(self.active)))
                height_columns=positive_pivot_height_columns(self.blocks.values())
                try:
                    for i,col in enumerate(self.active):
                        h=self.camera_steps[int(col)]
                        columns[:,i]=positive_pivot_coordinate_derivative(
                            lambda current:self._evaluate(current,self.rows)[0],
                            value,col,h,height_columns,base=residual)
                finally:
                    self._cameras(value,self.rows)
                rr,cc=np.nonzero(columns)
                matrix=csr_matrix((columns[rr,cc],(rr,self.active[cc])),shape=(len(residual),self.parameter_count))
            self._cache[key]=(residual.copy(),matrix)
            while len(self._cache)>4: self._cache.popitem(last=False)
        ledger=dict(family='additional_real_native_persistent_ice_long_baseline',
            original_start=int(original_start),original_stop=int(original_start)+self.row_count,
            packet_sha256=self.contract['packet_sha256'],fit_ids=self.contract['fit_ids'],held_ids=self.contract['held_ids'],
            common_anchor_native_pts=self.anchor_pts,fit_target_native_pts=self.contract['fit_target_native_pts'],
            existing_active_camera_columns=self.active.tolist(),common_anchor_correlated=True,
            old_WORLD_not_used_as_observation=True,source_UV_sigma_roles_and_clock_preserved=True)
        return SourceRowBlock(residual,matrix,self.quadratic,self.scales,ledger)

    def review(self, q):
        value=self._q(q)
        _,predictions,worlds=self._evaluate(value,self.rows+self.held)
        rows=[]
        for row in self.rows+self.held:
            raw=predictions[row['id']]-row['target_uv']
            rows.append(dict(id=row['id'],track_id=row['track_id'],split=row['split'],
                source_native_pts=row['anchor_endpoint']['native_pts'],target_native_pts=row['target_endpoint']['native_pts'],
                anchor_actual_pts=row['anchor_endpoint']['actual_pts'],target_actual_pts=row['target_endpoint']['actual_pts'],
                source_uv=row['anchor_uv'].tolist(),target_uv=row['target_uv'].tolist(),
                projected_target_uv=predictions[row['id']].tolist(),raw_native_error_vector_px=raw.tolist(),
                native_error_norm_px=float(np.linalg.norm(raw)),sigma_native_px=row['sigma'].tolist(),
                current_anchor_ray_ice_intersection_xyz_scene_units=worlds[row['track_id']].tolist(),
                current_intersection_is_derived_state_not_observed_WORLD=True))
        return dict(contract=deepcopy(self.contract),rows=rows,accuracy_accepted=False)


class SourceWithPersistentIceLongBaseline:
    def __init__(self, base, factor):
        self.base,self.persistent_ice_factor=base,factor
        self.count,self.updater=int(base.count),base.updater
        if factor.parameter_count!=self.count: raise ValueError('Persistent ice source changed the complete parameter prefix')
        self.quadratic=np.r_[base.quadratic,factor.quadratic]
        self.scales=np.r_[base.scales,factor.scales]
        self.contract=dict(schema='rinkspace.complete-source-plus-real-persistent-ice-long-baseline.v1',
            original_source_contract=deepcopy(base.contract),persistent_ice_contract=deepcopy(factor.contract),
            parameter_count=self.count,original_source_rows=len(base.scales),additional_scalar_rows=factor.row_count,
            total_rows=len(self.scales),old_source_prefix_masks_sigmas_roles_profiles_and_parameters_preserved=True,
            new_camera_or_world_parameters=0,accuracy_accepted=False)

    def __getattr__(self, name):
        base=self.__dict__.get('base')
        if base is None: raise AttributeError(name)
        return getattr(base,name)

    def guard(self, q):
        self.base.guard(q)
        self.persistent_ice_factor.guard(q)

    def blocks(self, q, jacobian=True):
        yield from self.base.blocks(q,jacobian)
        yield self.persistent_ice_factor.block(q,jacobian,original_start=len(self.base.scales))


def wrap_persistent_ice_long_baseline(base, state, packet_path):
    return SourceWithPersistentIceLongBaseline(base,PersistentIceLongBaselineFactor(state,packet_path,base.count))
