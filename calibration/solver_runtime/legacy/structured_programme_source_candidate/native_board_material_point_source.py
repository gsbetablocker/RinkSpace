"""Additional real Ad12/25 printed material points on the existing q9 graph.

The three named points are the same independent station/Z coordinates used
by CurrentPatterns and the complete scene descriptor. No shield outline,
normalized rectangle, circle, new parameter, or fixed old XYZ is introduced.
"""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .native_camera_local import native_camera_local
from scipy.sparse import csr_matrix

from z190_streaming_source_normal import SourceRowBlock
from z190_shared_board_pattern_source import SHIELD_ROLES, _project
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from .native_ice_curve_source import (SOURCE_SHA, _sha, _native_frame_reference,
    _source_optical_support, _native_error_with_optical_support,
    _current_source_parameter_layout)
from .fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_coordinate_derivative,
)
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES


class NativeBoardMaterialPointFactor:
    def __init__(self, state, packet_path, parameter_count):
        self.state, self.parameter_count = state, int(parameter_count)
        self.core_count, self.world_indices = _current_source_parameter_layout(state, self.parameter_count)
        self.world_parameter_count = len(self.world_indices)
        self.packet_path = Path(packet_path).resolve()
        packet = json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema') != 'rinkspace.additional-native-board-material-points.v1'
                or packet.get('source_video_sha256') != SOURCE_SHA
                or packet.get('time_base') != '1/15360'
                or packet.get('status') != 'explicit_new_fit_target_authorized'):
            raise ValueError('Explicit admitted same-source board material-point packet required')
        self.patterns = getattr(state, 'patterns', None)
        if (self.patterns is None or not callable(getattr(self.patterns, 'bind_current_geometry', None))
                or not isinstance(self.patterns.start, (int, np.integer))):
            raise ValueError('Existing current-course shared q9 pattern producer required')
        self.pattern_start = int(self.patterns.start)
        self.pattern_columns = np.arange(self.pattern_start+3, self.pattern_start+9)
        _, _, course_chain = state.geometry.current.print_course_binding(state.geometry.current.initial)
        course_chain = np.asarray(course_chain, float)
        if course_chain.shape != (6, self.world_parameter_count):
            raise ValueError('Current logical world/course derivative chain required')
        self.active_world = self.world_indices[np.flatnonzero(np.any(course_chain != 0, axis=0))]
        if (self.pattern_start < 0 or self.pattern_start+9 > self.core_count
                or np.intersect1d(self.pattern_columns, self.world_indices).size):
            raise ValueError('Existing independent q9 station/Z slots required')
        self.rows, self.held = [], []
        ids = set()
        for record in packet['rows']:
            row = deepcopy(record)
            if row['id'] in ids:
                raise ValueError('Additional material point source IDs must be unique')
            ids.add(row['id'])
            curve = row.get('curve', {})
            number = curve.get('advertisement_id')
            roles = row.get('material_roles', [])
            points = (curve.get('kind') == 'shared_board_pattern_material_points'
                and type(number) is int and number in (12,25)
                and row.get('point_correspondence') is True
                and isinstance(roles,list) and 1 <= len(roles) <= 3
                and len(set(roles)) == len(roles)
                and all(role in SHIELD_ROLES for role in roles))
            circle = (curve.get('kind') == 'shared_board_pattern_circle_curve'
                and type(number) is int and number in (13,26)
                and curve.get('edge') == 'outermost_visible_circular_print_stroke'
                and not roles and row.get('point_correspondence') is not True)
            if (not (points or circle) or row.get('feature_id') != 'Ad'+str(number)
                    or row.get('world') is not None):
                raise ValueError('Existing named Ad12/25 points or Ad13/26 actual outer-rim curves required')
            native = int(row['pts'])
            if native not in state.original.aliases:
                raise ValueError('Source PTS has no authenticated original camera alias')
            actual = int(state.original.aliases[native])
            if ('actual_pts' in row and int(row['actual_pts']) != actual) or actual not in state.bypts:
                raise ValueError('Material source exposure disagrees with authenticated camera alias')
            if not 229114112 <= native < 231705600:
                raise ValueError('Material source lies outside the authorized programme')
            path, expected = _native_frame_reference(row)
            if _sha(path) != expected:
                raise ValueError('Material source raw frame bytes changed')
            uv = np.asarray(row['uv'],float)
            sigma = np.broadcast_to(np.asarray(row['sigma_px'],float),(len(uv),)).copy()
            valid_shape = uv.shape == (len(roles),2) if points else (uv.ndim == 2 and uv.shape[1:] == (2,) and len(uv) >= 3)
            if (not valid_shape or not np.isfinite(uv).all()
                    or not np.isfinite(sigma).all() or np.any(sigma<=0)
                    or np.any(uv<0) or np.any(uv >= [1920.,1080.])):
                raise ValueError('Finite unchanged native source point UV and positive raw sigma required')
            row.update(actual_pts=actual,uv=uv,sigma=sigma)
            if points and row.get('source_point_bounds_authorized') is True:
                bounds = np.asarray(row.get('point_support_bounds_xy_px'), float)
                if (bounds.shape != (len(uv),2,2) or not np.isfinite(bounds).all()
                        or np.any(bounds[:,:,0]>bounds[:,:,1])
                        or np.any(uv<bounds[:,:,0]) or np.any(uv>bounds[:,:,1])):
                    raise ValueError('Explicit actual named-point XY blur support must enclose original finite source UV')
                row['_source_point_bounds'] = bounds

            if circle:
                row['_source_optical_support'] = _source_optical_support(row)
            if row['split']=='fit': self.rows.append(row)
            elif row['split']=='holdout': self.held.append(row)
            else: raise ValueError('Explicit new FIT or HELD role required')
        if not self.rows:
            raise ValueError('At least one explicitly admitted new material point source required')
        if any(r['curve']['kind'] == 'shared_board_pattern_circle_curve' for r in self.rows+self.held):
            self.pattern_columns = np.arange(self.pattern_start, self.pattern_start+9)
        self.blocks = {r['actual_pts']:state.bypts[r['actual_pts']] for r in self.rows+self.held}
        self.camera_steps, columns = {}, []
        for actual, block in self.blocks.items():
            view = block.start+4*block.bundle.index[actual]
            slots = np.r_[block.xyz_slots,block.optics_chart.indices,
                          block.optics_chart.mount_indices,view+np.arange(4)].astype(int)
            if any(r['actual_pts']==actual for r in self.rows): columns.extend(slots)
            for col in slots: self.camera_steps[int(col)]=1e-6
            for col in block.optics_chart.indices: self.camera_steps[int(col)]=1e-4
            for col in range(view,view+3): self.camera_steps[col]=1e-7
        self.camera_columns = np.unique(columns).astype(int)
        if np.intersect1d(self.camera_columns,np.r_[self.world_indices,self.pattern_columns]).size:
            raise ValueError('Original camera, current course, and q9 point slots overlap')
        self.active_material = np.r_[self.active_world,self.pattern_columns]
        self.active = np.unique(np.r_[self.active_material,self.camera_columns]).astype(int)
        self.scales = np.concatenate([np.repeat(r['sigma'],2) for r in self.rows])
        self.quadratic = np.ones(len(self.scales),bool)
        self.row_count, self._cache = len(self.scales), OrderedDict()
        self.contract = dict(schema='rinkspace.additional-real-native-board-material-point-factor.v1',
            packet_path=str(self.packet_path),packet_sha256=_sha(self.packet_path),
            source_video_sha256=SOURCE_SHA,time_base='1/15360',
            current_material='CurrentPatterns.bind_current_geometry -> existing shield(q9,ad_id) or circle(q9,theta,ad_id)',
            point_roles=list(SHIELD_ROLES),existing_q9_start=self.pattern_start,
            evaluation_core_parameter_count=self.core_count,
            logical_world_parameter_count=self.world_parameter_count,
            logical_world_parameter_indices=self.world_indices.tolist(),
            active_existing_q9_point_columns=self.pattern_columns.tolist(),
            active_named_world_course_columns=self.active_world.tolist(),
            fit_ids=[r['id']for r in self.rows],held_ids=[r['id']for r in self.held],
            scalar_rows=self.row_count,source_UV_sigma_roles_and_clock_preserved=True,
            raw_sigma_used_without_extra_sqrt_N=True,observation_operator='native full-Brown named-point reprojection vector',
            geometry_or_camera_DOF_added=False,fixed_old_WORLD_as_observation=False,
            normalized_rectangle_or_shield_circle_added=False,unobserved_shield_outline_created=False,
            native_circle_curve_ids=[r['id'] for r in self.rows+self.held if r['curve']['kind']=='shared_board_pattern_circle_curve'],
            shield_upper_heights_or_symmetry_imposed=False,held_rows_used_in_fit=False,
            working_sigma_not_measured_covariance=True,accuracy_accepted=False)

    def _cameras(self,q):
        core=np.asarray(q,float)[:self.core_count]
        return {actual:block.bundle.camera(native_camera_local(block,core),actual)for actual,block in self.blocks.items()}

    def _world(self,q,row):
        q=np.asarray(q,float)
        with self.patterns.bind_current_geometry(q[:self.core_count]) as geometry:
            q9=q[self.pattern_start:self.pattern_start+9]
            geometry.guard(q9)
            if row['curve']['kind'] == 'shared_board_pattern_circle_curve':
                world = geometry.circle(q9,np.linspace(0.,2*np.pi,361),ad_id=row['curve']['advertisement_id'])[0]
                return np.asarray(world,float)
            world=geometry.shield(q9,ad_id=row['curve']['advertisement_id'])[0]
        return np.asarray(world,float)[[SHIELD_ROLES.index(role)for role in row['material_roles']]]

    def _residual(self,q,rows,*,cameras=None):
        cameras=self._cameras(q)if cameras is None else cameras
        values=[]
        for row in rows:
            if row['curve']['kind'] == 'shared_board_pattern_circle_curve':
                raw = curve_vector_error_full_brown(cameras[row['actual_pts']],self._world(q,row),row['uv'],with_jacobian=False)
                if isinstance(raw,tuple):raw=raw[0]
                values.append(_native_error_with_optical_support(row,raw)/np.repeat(row['sigma'],2))
                continue
            projected=_project(cameras[row['actual_pts']],self._world(q,row),False)[0]
            if row.get('source_point_bounds_authorized') is True:
                bounds = row['_source_point_bounds']
                raw = np.minimum(projected-bounds[:,:,0],0.) + np.maximum(projected-bounds[:,:,1],0.)
                values.append((raw/row['sigma'][:,None]).reshape(-1))
            else:
                values.append(((projected-row['uv'])/row['sigma'][:,None]).reshape(-1))
        result=np.concatenate(values)if values else np.empty(0)
        if not np.isfinite(result).all(): raise ValueError('Material source has invalid physical projection')
        return result

    def block(self,q,jacobian=True,*,original_start=0):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():
            raise ValueError('Complete same current physical programme state required')
        key=q[self.active].tobytes()
        cached=self._cache.get(key)
        if cached is not None and (not jacobian or cached[1]is not None):
            residual,matrix=cached
        else:
            cameras=self._cameras(q)
            residual,matrix=self._residual(q,self.rows,cameras=cameras),None
            if jacobian:
                compact=np.empty((len(residual),len(self.active)))
                height_columns=positive_pivot_height_columns(self.blocks.values())
                try:
                    from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
                    compact=native_factor_dependency_jacobian(self,q,self.rows,self.active,
                        np.r_[self.world_indices,self.active_material],residual,
                        lambda value,groups:self._residual(value,[self.rows[i]for i in groups],
                            cameras=cameras if np.array_equal(value[self.camera_columns],q[self.camera_columns])else None),height_columns)
                finally:self._cameras(q)
                rr,cc=np.nonzero(compact)
                matrix=csr_matrix((compact[rr,cc],(rr,self.active[cc])),shape=(len(residual),self.parameter_count))
            self._cache[key]=(residual.copy(),matrix)
            while len(self._cache)>4:self._cache.popitem(last=False)
        ledger=dict(family='additional_real_native_board_material_points',
            original_start=int(original_start),original_stop=int(original_start)+self.row_count,
            packet_sha256=self.contract['packet_sha256'],
            fit_observation_ids=self.contract['fit_ids'],held_observation_ids=self.contract['held_ids'],
            native_pts=[r['pts']for r in self.rows],actual_pts=[r['actual_pts']for r in self.rows],
            material_roles=[r.get('material_roles',[])for r in self.rows],
            primitive_kinds=[r['curve']['kind']for r in self.rows],
            old_WORLD_not_used_as_observation=True,geometry_or_camera_DOF_added=False)
        return SourceRowBlock(residual,matrix,self.quadratic,self.scales,ledger)

    def review(self,q):
        cameras=self._cameras(q)
        rows=[]
        for row in self.rows+self.held:
            raw=self._residual(q,[row],cameras=cameras).reshape(-1,2)*row['sigma'][:,None]
            distance=np.linalg.norm(raw,axis=1)
            rows.append(dict(id=row['id'],pts=row['pts'],actual_pts=row['actual_pts'],split=row['split'],
                feature_id=row['feature_id'],material_roles=row.get('material_roles',[]),source_uv_count=len(distance),
                primitive_kind=row['curve']['kind'],
                source_uv=row['uv'].tolist(),projected_uv=(row['uv']+raw).tolist(),
                mean_native_px=float(distance.mean()),max_native_px=float(distance.max()),
                source_sigma_is_engineering_not_calibrated_accuracy=True))
        return dict(contract=self.contract,rows=rows,accuracy_accepted=False)


class SourceWithAdditionalNativeBoardMaterialPoints:
    def __init__(self,base,factor):
        self.base,self.native_board_material_point_factor=base,factor
        self.count,self.updater=int(base.count),base.updater
        self.quadratic=np.r_[base.quadratic,factor.quadratic]
        self.scales=np.r_[base.scales,factor.scales]
        self.contract=dict(schema='rinkspace.complete-source-plus-real-native-board-material-points.v1',
            original_source_contract=deepcopy(base.contract),additional_material_point_contract=deepcopy(factor.contract),
            original_source_rows=len(base.scales),additional_scalar_rows=factor.row_count,total_rows=len(self.scales),
            original_blocks_profiles_source_UV_sigma_clock_and_geometry_DOF_preserved=True,accuracy_accepted=False)

    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)

    def guard(self,q):
        self.base.guard(q);self.native_board_material_point_factor.block(q,False)

    def blocks(self,q,jacobian=True):
        yield from self.base.blocks(q,jacobian)
        yield self.native_board_material_point_factor.block(q,jacobian,original_start=len(self.base.scales))


def wrap_additional_native_board_material_points(base,state,packet_path):
    return SourceWithAdditionalNativeBoardMaterialPoints(base,NativeBoardMaterialPointFactor(state,packet_path,base.count))
