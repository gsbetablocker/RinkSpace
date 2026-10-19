"""Measured board-body pixel transfer in the complete existing owner chart.

This objective admits only packets with direct original-RGB material witnesses.
Source-only LK/FB/texture qualification is insufficient. Native pixel residuals
use the public component Huber2 as an engineering objective, with no inferred
probability, sigma calibration, added camera state, or relaxed physical guard.
"""
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack

from .first_shot_source_transition import reference, read


SCHEMA = 'rinkspace.measured-board-motion-packet.v1'


def _authenticated(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or reference(ref['path']) != ref:
        raise ValueError('Exact immutable motion evidence reference required')
    return ref


def _readonly(value, dtype=None):
    value = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _load_material_support(ref):
    from .native_board_material_support import load_support
    return load_support(ref)


@dataclass(frozen=True)
class MotionPair:
    a: int
    b: int
    native_pts: tuple
    source_uv: object
    target_uv: object
    source_point_ids: tuple
    split: str


class BoardMotionPacket:
    def __init__(self, packet_reference):
        self._verified_paths={}
        self.reference = deepcopy(self._authenticate(packet_reference))
        packet = read(self.reference)
        if packet.get('schema') != SCHEMA:
            raise ValueError('A material-admitted measured board motion packet is required; legacy flow is not admissible')
        if packet.get('material_support') is None:
            raise ValueError('Missing authenticated original-RGB board-body material support')
        support_ref = self._authenticate(packet['material_support'])
        self.support = _load_material_support(support_ref)
        self.source = deepcopy(packet['source'])
        raw_pts=np.asarray(self.source['native_pts'])
        if raw_pts.dtype.kind not in 'iu':raise ValueError('Original native PTS must be exact integers')
        self.pts = _readonly(raw_pts, np.int64)
        if (self.pts.ndim != 1 or len(self.pts) < 2 or np.any(np.diff(self.pts) <= 0)
                or not self.source.get('source_video_sha256') or not self.source.get('time_base')):
            raise ValueError('Motion packet requires exact source identity and increasing original native PTS')
        if (self.support.source_video_sha256 != self.source['source_video_sha256']
                or self.support.time_base != self.source['time_base']):
            raise ValueError('Material support and motion source identities differ')
        radius=float(packet.get('tracking_support_radius_px',-1.))
        if not np.isfinite(radius)or radius<0.:
            raise ValueError('Exact original tracker material support radius required')
        self.support_radius_px=radius
        loss = packet.get('loss')
        expected = dict(residual_unit='native_px', working_unit_native_px=1.,
            integration_weight=1., robust_loss='public_component_Huber2', calibrated_probability=False)
        if loss != expected:
            raise ValueError('Explicit uncalibrated native-pixel public Huber2 objective required')
        arrays_ref = self._authenticate(packet['arrays'])
        quality_ref=self._authenticate(packet['source_qualification'])
        quality=read(quality_ref);quality_arrays_ref=self._authenticate(packet['source_qualification_arrays'])
        if (quality.get('camera_model_or_residual_used')is not False
                or quality.get('source_track_pixels_modified')is not False):
            raise ValueError('Motion correspondence qualification must use unchanged source pixels only')
        chain_ref=packet.get('material_chain');chain_arrays_ref=None
        if chain_ref is not None:
            self._authenticate(chain_ref);chain_doc=read(chain_ref)
            if (chain_doc.get('schema')!='rinkspace.original-rgb-entire-footprint-chain.v1'
                    or chain_doc.get('source_support')!=support_ref
                    or chain_doc.get('source_flow_arrays')!=arrays_ref
                    or chain_doc.get('source_qualification')!=quality_ref
                    or chain_doc.get('forward_and_reverse_actual_original_RGB_LK')is not True
                    or chain_doc.get('each_step_full_35px_body_support_required')is not True
                    or chain_doc.get('model_projection_or_model_error_selection')is not False
                    or chain_doc.get('original_direct_endpoints_modified')is not False
                    or float(chain_doc.get('chain_endpoint_and_FB_tolerance_px',-1.))!=.5):
                raise ValueError('Original RGB full-footprint forward/reverse material chain required')
            chain_arrays_ref=self._authenticate(chain_doc['arrays'])
            for name in ('source_flow','producer'):self._authenticate(chain_doc[name])
        self.pairs = []; used = set()
        with np.load(arrays_ref['path'],allow_pickle=False)as arrays, np.load(quality_arrays_ref['path'],allow_pickle=False)as qualified, (np.load(chain_arrays_ref['path'],allow_pickle=False)if chain_arrays_ref is not None else nullcontext(None))as chain:

            for item in packet.get('pairs', []):
                raw_frames=np.asarray([item['a'],item['b']]);pair_pts=np.asarray(item['native_pts'])
                if raw_frames.dtype.kind not in 'iu'or pair_pts.dtype.kind not in 'iu':
                    raise ValueError('Exact integer native frame indices and PTS required')
                a, b = map(int,raw_frames)
                if not 1 <= a < b <= len(self.pts) or list(item['native_pts']) != [int(self.pts[a-1]), int(self.pts[b-1])]:
                    raise ValueError('Pair endpoints must use exact native source PTS')
                key = item['array_prefix']; raw_indices = np.asarray(item['source_indices'])
                if raw_indices.dtype.kind not in 'iu':
                    raise ValueError('Source indices must be exact integers')
                indices = raw_indices.astype(np.int64)
                if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0):
                    raise ValueError('Explicit unique source-only selected point indices required')
                all_u=np.asarray(arrays[key+'_previous_uv']);all_v=np.asarray(arrays[key+'_current_uv'])
                if (all_u.ndim!=2 or all_u.shape[1:]!=(2,) or all_v.shape!=all_u.shape
                        or np.any(indices>=len(all_u))):
                    raise ValueError('Selected point indices leave the original measured pair inventory')
                u=all_u[indices];v=all_v[indices]
                mask=np.asarray(qualified[key+'_source_2d_identifiable'])
                if mask.shape!=(len(all_u),)or mask.dtype.kind!='b'or not mask[indices].all():
                    raise ValueError('Selected point lacks frozen source-only correspondence qualification')
                if b-a>1:self._verify_chain(chain,key,indices,a,b,u,v,radius)
                ids = tuple(map(str, item['source_point_ids']))
                if (u.shape != (len(ids), 2) or v.shape != u.shape or not np.isfinite(u).all()
                        or not np.isfinite(v).all() or len(set(ids)) != len(ids)):
                    raise ValueError('Exact finite measured source/target pixel inventory required')
                for point_id, left, right in zip(ids, u, v, strict=True):
                    if point_id in used:
                        raise ValueError('Motion point identities must be unique across the packet')
                    used.add(point_id)
                    self.support.require(a,int(self.pts[a-1]),left,radius)
                    self.support.require(b,int(self.pts[b-1]),right,radius)
                split = item.get('split')
                if split not in ('fit', 'development'):
                    raise ValueError('Explicit fit/development split required; old inspected holdout is development')
                self.pairs.append(MotionPair(a, b, tuple(item['native_pts']), _readonly(u, float), _readonly(v, float), ids, split))
        self.pairs = tuple(self.pairs)
        self.fit_pairs = tuple(pair for pair in self.pairs if pair.split == 'fit')
        if not self.fit_pairs:
            raise ValueError('No material-admitted motion training observations; refuse an empty or legacy-only factor')
        self.rows = 2*sum(len(pair.source_uv) for pair in self.fit_pairs)
        self.references = [self.reference,deepcopy(support_ref),deepcopy(arrays_ref),deepcopy(quality_ref),deepcopy(quality_arrays_ref)]
        if chain_ref is not None:self.references.extend([deepcopy(chain_ref),deepcopy(chain_arrays_ref)])
        observed = sorted({i for pair in self.fit_pairs for i in (pair.a, pair.b)})
        self.receipt = dict(schema=SCHEMA, packet=deepcopy(self.reference),
            material_support=deepcopy(support_ref),arrays=deepcopy(arrays_ref),
            source_qualification=deepcopy(quality_ref),source_qualification_arrays=deepcopy(quality_arrays_ref),
            material_chain=deepcopy(chain_ref),material_chain_arrays=deepcopy(chain_arrays_ref),
            fit_pairs=len(self.fit_pairs), fit_points=self.rows//2,
            development_pairs=sum(pair.split == 'development' for pair in self.pairs),
            source_frames=len(self.pts), motion_supported_frames=observed,
            unsupported_frames=[i for i in range(1,len(self.pts)+1) if i not in observed],
            loss=expected, support_radius_px=radius, old_texture_only_flow_not_admitted=True,
            native_source_target_observed_rows=True, material_world_points_added=False)

    def _verify_chain(self,chain,key,indices,a,b,u,v,radius):
        if chain is None:raise ValueError('Non-adjacent motion requires the entire original-RGB material support chain')
        source_indices=np.asarray(chain[key+'_source_indices'])
        if source_indices.ndim!=1 or source_indices.dtype.kind not in 'iu' or len(np.unique(source_indices))!=len(source_indices):
            raise ValueError('Exact unique original source indices required in material chain')
        mapping={int(value):i for i,value in enumerate(source_indices)}
        if any(int(i)not in mapping for i in indices):raise ValueError('Motion point absent from authenticated full-footprint chain')
        selected=np.asarray([mapping[int(i)]for i in indices]);steps=b-a+1
        forward=np.asarray(chain[key+'_forward_uv'])[selected];reverse=np.asarray(chain[key+'_reverse_uv'])[selected]
        forward_pts=np.asarray(chain[key+'_forward_pts']);reverse_pts=np.asarray(chain[key+'_reverse_pts'])
        passed=np.asarray(chain[key+'_full_support_chain_passed'])
        if (forward.shape!=(len(indices),steps,2)or reverse.shape!=forward.shape
                or not np.isfinite(forward).all()or not np.isfinite(reverse).all()
                or not np.array_equal(forward_pts,self.pts[a-1:b])
                or not np.array_equal(reverse_pts,self.pts[a-1:b][::-1])
                or passed.shape!=(len(source_indices),)or passed.dtype.kind!='b'or not passed[selected].all()
                or not np.array_equal(forward[:,0],u)or not np.array_equal(reverse[:,0],forward[:,-1])
                or np.any(np.linalg.norm(forward[:,-1]-v,axis=1)>.5)
                or np.any(np.linalg.norm(reverse[:,-1]-u,axis=1)>.5)):
            raise ValueError('Original chain PTS, endpoints, FB agreement or complete support qualification differs')
        for samples,frames,stamps in ((forward,range(a,b+1),forward_pts),(reverse,range(b,a-1,-1),reverse_pts)):
            for j,(frame,stamp)in enumerate(zip(frames,stamps,strict=True)):
                for uv in samples[:,j]:self.support.require(frame,int(stamp),uv,radius)

    def _authenticate(self, ref):
        if not isinstance(ref,dict)or set(ref)!={'path','sha256'}:
            raise ValueError('Exact immutable motion evidence reference required')
        key=str(Path(ref['path']).resolve()).casefold();prior=self._verified_paths.get(key)
        if prior is not None:
            if prior!=ref['sha256']:raise ValueError('The same motion evidence path has conflicting hashes')
            return ref
        _authenticated(ref);self._verified_paths[key]=ref['sha256'];return ref

    def authenticate_state(self, state):
        if (not np.array_equal(self.pts, np.asarray(state.pts))
                or self.source['time_base'] != state.source_clock['time_base']):
            raise ValueError('Owner and motion packet native source clock differ')
        binding = read(state.binding_reference)
        if binding['source']['sha256'] != self.source['source_video_sha256']:
            raise ValueError('Owner and motion packet source video identity differ')

    def checkpoint_arrays(self):
        # Keep the complete admitted evidence identity, actual UV and original
        # source timestamps in every numerical checkpoint, beside inherited aux.
        pairs = self.pairs; offsets = np.r_[0,np.cumsum([len(p.source_uv) for p in pairs])]
        return dict(measured_board_motion_identity_json=np.asarray(json.dumps(self.receipt,sort_keys=True)),
            measured_board_motion_source_uv=np.vstack([p.source_uv for p in pairs]),
            measured_board_motion_target_uv=np.vstack([p.target_uv for p in pairs]),
            measured_board_motion_pair_offsets=offsets,
            measured_board_motion_pair_frames=np.asarray([(p.a,p.b)for p in pairs],np.int64),
            measured_board_motion_pair_native_pts=np.asarray([p.native_pts for p in pairs],np.int64),
            measured_board_motion_pair_is_fit=np.asarray([p.split=='fit' for p in pairs]),
            measured_board_motion_source_point_ids=np.asarray([i for p in pairs for i in p.source_point_ids]))


def _course_boundary_differentials(state, physical, columns, scene):
    """Pull back the explicit course sampler used by the current actual scene."""
    provider=state.provider;current=getattr(provider,'current',None)
    binding=getattr(current,'print_course_binding',None)
    if not callable(binding):return None
    count=int(provider.parameter_count)
    world=np.asarray(state.world_parameter_indices,dtype=int)
    if count<=0 or len(world)<count or len(np.unique(world))!=len(world):
        raise ValueError('Explicit unique provider world-coordinate mapping required')
    geometry_columns=world[:count]
    if np.any(geometry_columns<0)or np.any(geometry_columns>=len(physical)):
        raise ValueError('Provider geometry leaves the complete physical chart')
    course,cq,course_chain=binding(np.asarray(physical)[geometry_columns])
    bottom,_,bottom_chain,_,_=course.sample_perimeter(cq,with_jacobian=True)
    bottom=np.asarray(bottom,float);boundary=np.asarray(scene['ice_boundary']['points_xyz'],float)
    course_chain=np.asarray(course_chain,float);bottom_chain=np.asarray(bottom_chain,float)
    if (bottom.shape!=boundary.shape or not np.array_equal(bottom,boundary)
            or course_chain.shape!=(6,count) or bottom_chain.shape!=bottom.shape+(6,)
            or not np.isfinite(course_chain).all()or not np.isfinite(bottom_chain).all()):
        raise ValueError('Analytic course sampler must reproduce the actual current scene boundary exactly')
    state.guard_geometry(physical)
    complete=np.zeros((6,len(physical)))
    complete[:,geometry_columns]=course_chain
    derivative=np.einsum('nik,kj->nij',bottom_chain[:,:2],complete[:,columns])
    reports=[dict(column=int(column),method='analytic_current_course_sample',
        structural_course_column_present=bool(np.any(complete[:5,column]!=0.)),
        boundary_XY_derivative_identically_zero=bool(not np.any(complete[:5,column]!=0.)),
        original_full_current_geometry_guard_passed=True,actual_scene_bottom_byte_exact=True,
        domain_failures=[])for column in columns]
    return scene,derivative,reports


def boundary_differentials(state, physical, columns, *, fd_step):
    """Actual course analytic chain; unknown producers retain guarded geometry FD.

    Every original scene construction/domain check runs. No per-point surface
    finite differences, frozen surface, or source-domain-based point dropping.
    """
    physical = np.asarray(physical); columns = np.asarray(columns,dtype=int)
    scene = state.scene(physical)
    analytic=_course_boundary_differentials(state,physical,columns,scene)
    if analytic is not None:return analytic
    # Unknown producers retain guarded FD. Failed trials never imply zero.
    boundary = np.asarray(scene['ice_boundary']['points_xyz'],float)[:,:2]
    derivative = np.zeros(boundary.shape+(len(columns),)); reports=[]
    if not np.isfinite(fd_step) or fd_step <= 0.:
        raise ValueError('Original positive geometry producer FD step required')
    try:
        for j,column in enumerate(columns):
            samples=[];failures=[]
            for sign in (1.,-1.):
                trial=physical.copy();trial[column]+=sign*fd_step
                try:
                    state.guard_geometry(trial)
                    other=state.scene(trial)
                    value=np.asarray(other['ice_boundary']['points_xyz'],float)[:,:2]
                    if value.shape!=boundary.shape:
                        raise ValueError('Current physical boundary topology changed')
                except ValueError as exc:
                    samples.append(None);failures.append(str(exc))
                else:samples.append(value);failures.append(None)
            plus,minus=samples
            if plus is None and minus is None:
                raise ValueError('Both original-step boundary geometry FD sides leave the domain: '+str(column))
            if plus is not None and minus is not None:
                derivative[:,:,j]=(plus-minus)/(2.*fd_step);method='central_original_step'
            elif plus is not None:
                derivative[:,:,j]=(plus-boundary)/fd_step;method='forward_original_step'
            else:
                derivative[:,:,j]=(boundary-minus)/fd_step;method='backward_original_step'
            reports.append(dict(column=int(column),method=method,original_step=float(fd_step),domain_failures=failures))
    finally:
        # Physical model's scene context is mutable; restore exact current q.
        state.scene(physical)
    return scene, derivative, reports


class MeasuredBoardMotionObjective:
    """Append admitted material motion while retaining the full original owner."""
    def __init__(self, base, packet_reference, *, entry_reference):
        self.base=base;self.packet=(packet_reference if isinstance(packet_reference,BoardMotionPacket)else BoardMotionPacket(packet_reference))
        self.programme=base.programme;self.parameter_count=int(base.parameter_count)
        self.programme_count=int(base.programme_count);self.start=base.start.copy()
        self.packet.authenticate_state(self.programme.state)
        self.scales=np.r_[base.scales,np.ones(self.packet.rows)]
        self.quadratic=np.r_[base.quadratic,np.zeros(self.packet.rows,bool)]
        self.source_producers=deepcopy(base.source_producers)
        self.recovery_references=deepcopy(base.recovery_references)
        self._references=deepcopy(getattr(base,'_references',[]))
        from . import board_motion_geometry,board_motion_factor,brown_inverse_rows
        refs=[reference(__file__),reference(board_motion_geometry.__file__),
              reference(board_motion_factor.__file__),reference(brown_inverse_rows.__file__),
              entry_reference,*self.packet.references]
        for current in (self.source_producers,self.recovery_references,self._references):
            for ref in refs:
                if ref not in current:current.append(deepcopy(ref))
        self._initial_complete_J_reuse=False
        self._motion_cache=OrderedDict()
        self.motion_receipt=dict(schema='rinkspace.complete-owner-measured-board-motion.v1',
            producer=reference(__file__),packet=deepcopy(self.packet.receipt),
            original_rows=len(base.scales),motion_rows=self.packet.rows,
            complete_rows=len(self.scales),parameter_count=self.parameter_count,
            original_parameter_domain_and_fixed_coordinates_retained=True,
            original_full_source_rows_W_and_loss_preserved=True,
            original_406_or_other_auxiliaries_retained=True,
            fresh_combined_J_required=True,checkpoint_adds_actual_motion_evidence=True,
            absolute_pose_embedding_anchors_added=False,accuracy_accepted=False)

    def __getattr__(self,name):return getattr(self.base,name)

    def _motion_block(self,q,jacobian):
        from z190_streaming_source_normal import SourceRowBlock
        from .first_shot_row_camera import prepare_frame_row_camera
        from .reference_board_pair import ReferenceBoardPair
        from .board_motion_geometry import evaluate_board_motion_geometry
        q=np.asarray(q);token=(q.dtype.str,q.shape,q.tobytes(),bool(jacobian))
        if token in self._motion_cache:return self._motion_cache[token]
        physical,chain=self.base.effective_q_and_chain(q)
        physical=np.asarray(physical)[:self.programme_count];chain=csr_matrix(chain)[:self.programme_count]
        state=self.programme.state
        columns=np.asarray(state.world_parameter_indices,dtype=int)
        # Preserve the entire declared world chart, including weak/zero current
        # columns. A zero derivative at one state is not a fixed-parameter proof.
        if jacobian:
            scene,boundary_j,stencils=boundary_differentials(state,physical,columns,fd_step=float(state.provider.fd_step))
        else:
            scene=state.scene(physical);boundary_j=None;stencils=[]
        first=self.packet.fit_pairs[0]
        @lru_cache(maxsize=16)
        def trajectory(index):return prepare_frame_row_camera(state,physical,index)
        cam=trajectory(first.a).camera25_at_row(float(first.source_uv[0,1]))
        surface=ReferenceBoardPair(first.source_uv,first.target_uv,scene,
            np.arange(len(scene['ice_boundary']['points_xyz'])),cam,cam)
        values=[];matrices=[]
        for pair in self.packet.fit_pairs:
            left,right=trajectory(pair.a),trajectory(pair.b)
            ca=left.camera25_at_rows(pair.source_uv[:,1]);cb=right.camera25_at_rows(pair.target_uv[:,1])
            r,ja,jb,jg,_=evaluate_board_motion_geometry(surface,ca,cb,pair.source_uv,pair.target_uv,
                boundary_xy_jacobian=boundary_j,jacobian=bool(jacobian))
            values.append(r)
            if jacobian:
                n=len(pair.source_uv);rows=[]
                for i in range(n):
                    _,ac,aj=left.camera15_mapping(float(pair.source_uv[i,1]))
                    _,bc,bj=right.camera15_mapping(float(pair.target_uv[i,1]))
                    part=(csr_matrix(ja[2*i:2*i+2]@aj)@chain[ac]
                          +csr_matrix(jb[2*i:2*i+2]@bj)@chain[bc])
                    if len(columns):part=part+csr_matrix(jg[2*i:2*i+2])@chain[columns]
                    rows.append(part)
                matrices.append(vstack(rows,format='csr'))
        residual=np.concatenate(values)
        matrix=vstack(matrices,format='csr')if jacobian else None
        ledger=dict(family='direct-original-RGB-board-body-motion',packet=deepcopy(self.packet.reference),
            pairs=len(self.packet.fit_pairs),points=self.packet.rows//2,
            native_source_target_row_cameras=True,current_surface_reintersected=True,
            surface_geometry_columns=columns.tolist(),
            residual_unit='native_px',calibrated_probability=False)
        self.last_boundary_derivative_stencils=stencils
        block=SourceRowBlock(residual,matrix,np.zeros(len(residual),bool),np.ones(len(residual)),ledger)
        self._motion_cache[token]=block
        while len(self._motion_cache)>2:self._motion_cache.popitem(last=False)
        return block

    def directional_check(self,q,*,direction,epsilon=1e-6,pair_indices=None,fallback_direction=None,allow_direction_fallback=True):
        """Probe actual motion through feasible factor-relevant canonical trials.

        The explicit boundary chain and actual row-camera mappings remove
        unrelated print coordinates. Every real trial keeps the entire base
        guard and all motion visibility domains. Active inequalities can use a
        valid one-sided stencil without relaxing the physical domain.
        """
        from .first_shot_row_camera import prepare_frame_row_camera
        from .reference_board_pair import ReferenceBoardPair
        from .board_motion_geometry import evaluate_board_motion_geometry
        q=np.asarray(q,float);requested=np.asarray(direction,float);eps=float(epsilon)
        if requested.shape!=q.shape or not np.isfinite(requested).all()or not np.isfinite(eps)or eps<=0.:
            raise ValueError('Explicit finite canonical direction and positive derivative step required')
        fixed=np.asarray(self.base.fixed,dtype=int)
        if np.any(requested[fixed]!=0.):
            raise ValueError('Derivative probe must preserve all inherited fixed coordinates')
        count=len(self.packet.fit_pairs)
        selected=sorted(set([0,count//2,count-1]if pair_indices is None else pair_indices))
        if not selected or min(selected)<0 or max(selected)>=count:
            raise ValueError('Existing admitted pair indices required for mathematical probe')
        pairs=[self.packet.fit_pairs[i]for i in selected];state=self.programme.state
        physical,chain=self.base.effective_q_and_chain(q);p=np.asarray(physical)[:self.programme_count]
        chain=csr_matrix(chain)[:self.programme_count]
        columns=np.asarray(state.world_parameter_indices,dtype=int)
        scene,boundary_j,stencils=boundary_differentials(state,p,columns,fd_step=float(state.provider.fd_step))
        affected_geometry=columns[np.any(boundary_j!=0.,axis=(0,1))]
        physical_rows=set(map(int,affected_geometry));camera_rows=set()
        for pair in pairs:
            for index,uv in ((pair.a,pair.source_uv[:2]),(pair.b,pair.target_uv[:2])):
                camera=prepare_frame_row_camera(state,p,index)
                for row in uv[:,1]:
                    _,cc,_=camera.camera15_mapping(float(row));camera_rows.update(map(int,cc))
        physical_rows.update(camera_rows)
        supported=np.unique(chain[sorted(physical_rows)].indices)
        d=np.zeros_like(requested);d[supported]=requested[supported];d[fixed]=0.
        if not np.any(d):raise ValueError('Supplied direction has no actual motion-factor derivative support')
        d*=np.linalg.norm(requested)/np.linalg.norm(d)
        self.last_directional_direction=d.copy()
        self.last_directional_probe=dict(requested_direction_sha256=hashlib.sha256(requested.tobytes()).hexdigest(),
            used_direction_sha256=hashlib.sha256(d.tobytes()).hexdigest(),
            factor_relevant_canonical_columns=supported.tolist(),
            removed_unrelated_nonzero_columns=np.flatnonzero((requested!=0.)&(d==0.)).tolist(),
            original_epsilon=eps,attempts=[],boundary_stencils=stencils)
        try:
            self.guard(q)
            def values(canonical,analytic=False):
                actual,_=self.base.effective_q_and_chain(canonical);actual=np.asarray(actual)[:self.programme_count]
                current=state.scene(actual);rr=[];jj=[]
                dp=np.asarray(chain@d).ravel()
                boundary_d=np.einsum('nik,k->ni',boundary_j,dp[columns])[:,:,None]
                for pair in pairs:
                    u,v=pair.source_uv[:2],pair.target_uv[:2]
                    left=prepare_frame_row_camera(state,actual,pair.a);right=prepare_frame_row_camera(state,actual,pair.b)
                    ca=left.camera25_at_rows(u[:,1]);cb=right.camera25_at_rows(v[:,1])
                    wall=ReferenceBoardPair(u,v,current,np.arange(len(current['ice_boundary']['points_xyz'])),ca[0],cb[0])
                    r,ja,jb,jg,_=evaluate_board_motion_geometry(wall,ca,cb,u,v,
                        boundary_xy_jacobian=boundary_d if analytic else None,jacobian=analytic)
                    rr.append(r)
                    if analytic:
                        value=np.asarray(jg.toarray()).ravel()
                        for i in range(len(u)):
                            _,ac,aj=left.camera15_mapping(float(u[i,1]));_,bc,bj=right.camera15_mapping(float(v[i,1]))
                            value[2*i:2*i+2]+=ja[2*i:2*i+2]@aj@dp[ac]+jb[2*i:2*i+2]@bj@dp[bc]
                        jj.append(value)
                return np.concatenate(rr),np.concatenate(jj)if analytic else None
            baseline_values,_=values(q)
            def trial(sign,step):
                candidate=q+sign*step*d;candidate[fixed]=q[fixed]
                if np.array_equal(candidate,q):return None,'no_representable_motion_displacement'
                try:
                    actual,_=self.base.effective_q_and_chain(candidate)
                    if np.array_equal(np.asarray(actual)[:self.programme_count],p):
                        return None,'no_representable_motion_displacement'
                    measured,_=values(candidate)
                    # q can have zero coordinates and keep changing down to
                    # subnormals after every pixel is already byte-identical.
                    # Such a trial cannot measure a derivative. Check actual
                    # factor observability before another expensive full guard.
                    if np.array_equal(measured,baseline_values):
                        return None,'no_representable_motion_displacement'
                    self.guard(candidate)
                except ValueError as error:return None,type(error).__name__+': '+str(error)
                return measured,None
            camera_only_fallback=False;reference_fallback=False;pan_fallback=False
            if fallback_direction is not None:
                fallback_direction=np.asarray(fallback_direction,float)
                if (fallback_direction.shape!=q.shape or not np.isfinite(fallback_direction).all()
                        or np.any(fallback_direction[fixed]!=0.)):
                    raise ValueError('A complete same-chart fixed-coordinate-preserving probe reference is required')
            while True:
                plus,plus_error=trial(1.,eps);minus,minus_error=trial(-1.,eps)
                self.last_directional_probe['attempts'].append(dict(epsilon=eps,plus_failure=plus_error,minus_failure=minus_error))
                if plus is not None or minus is not None:break
                if not allow_direction_fallback:
                    raise ValueError('Requested derivative block has no feasible nominal-step side: '+str(self.last_directional_probe['attempts'][-1]))
                if not reference_fallback and fallback_direction is not None:
                    reference_fallback=True
                    if np.linalg.norm(fallback_direction)>0.:
                        d=fallback_direction*np.linalg.norm(requested)/np.linalg.norm(fallback_direction)
                        self.last_directional_direction=d.copy()
                        self.last_directional_probe.update(previous_accepted_secant_fallback=True,
                            used_direction_sha256=hashlib.sha256(d.tobytes()).hexdigest())
                        eps=float(epsilon);continue
                if not pan_fallback:
                    # Pan controls do not enter the existing tilt/logf/shared-axis
                    # mount envelope. This chooses a mathematical QA direction;
                    # it never fixes optimization coordinates or relaxes a guard.
                    from .first_shot_native_time_stage import KINDS
                    controls=self.base.checkpoint_arrays().get('native_time_control_slots')
                    pan_fallback=True
                    if controls is not None:
                        pan_columns=np.asarray(controls)[list(KINDS).index('pan')]
                        pan_columns=np.intersect1d(pan_columns,supported)
                        pan=np.zeros_like(d);pan[pan_columns]=requested[pan_columns];pan[fixed]=0.
                        if np.any(pan):
                            d=pan*np.linalg.norm(requested)/np.linalg.norm(pan)
                            self.last_directional_direction=d.copy()
                            self.last_directional_probe.update(pan_control_feasible_direction_fallback=True,
                                pan_control_canonical_columns=pan_columns.tolist(),
                                used_direction_sha256=hashlib.sha256(d.tobytes()).hexdigest())
                            eps=float(epsilon);continue
                if not camera_only_fallback:
                    # At an intersection of world inequalities, use only camera
                    # columns with exactly zero pullback into all world rows.
                    world_support=np.unique(chain[columns].indices)
                    camera_support=np.setdiff1d(np.unique(chain[sorted(camera_rows)].indices),world_support)
                    fallback=np.zeros_like(d);fallback[camera_support]=requested[camera_support];fallback[fixed]=0.
                    camera_only_fallback=True
                    if np.any(fallback):
                        d=fallback*np.linalg.norm(requested)/np.linalg.norm(fallback)
                        self.last_directional_direction=d.copy()
                        self.last_directional_probe.update(camera_only_feasible_direction_fallback=True,
                            used_direction_sha256=hashlib.sha256(d.tobytes()).hexdigest())
                        eps=float(epsilon);continue
                if plus_error==minus_error=='no_representable_motion_displacement':
                    raise ValueError('No representable feasible motion derivative probe; original physical domain retained')
                eps*=.5
                if eps==0.:raise ValueError('No finite feasible motion derivative probe')
            r,analytic=values(q,True)
            if plus is not None and minus is not None:
                finite=(plus-minus)/(2.*eps);method='central_feasible_canonical'
            else:
                sign=1. if plus is not None else -1.;first=plus if sign>0 else minus
                second,second_error=trial(sign,2.*eps)
                self.last_directional_probe['second_one_sided_failure']=second_error
                if second is not None:
                    finite=sign*(-3.*r+4.*first-second)/(2.*eps);method='second_order_one_sided_feasible_canonical'
                else:
                    finite=sign*(first-r)/eps;method='first_order_one_sided_feasible_canonical'
            error=analytic-finite
            self.last_directional_probe.update(epsilon=eps,method=method,
                original_full_guard_and_all_motion_domains_passed=True)
            return dict(schema='rinkspace.admitted-board-motion-directional-QA.v1',
                pair_indices=selected,source_point_ids=[list(pair.source_point_ids[:2])for pair in pairs],
                epsilon=eps,rows=len(r),max_absolute=float(np.max(abs(error))),
                analytic_norm=float(np.linalg.norm(analytic)),finite_norm=float(np.linalg.norm(finite)),
                largest_sample_displacement_px=float(max(np.max(abs(x-r))for x in (plus,minus)if x is not None)),
                canonical_direction_norm=float(np.linalg.norm(d)),
                physical_world_direction_norm=float(np.linalg.norm((chain@d)[columns])),
                relative_norm=float(np.linalg.norm(error)/max(np.linalg.norm(finite),1e-15)),
                probe=deepcopy(self.last_directional_probe),
                full_motion_or_original_source_J_constructed=False,source_fit_inventory_unchanged=True)
        finally:state.scene(p)

    def source_blocks(self,q,jacobian):
        cursor=0
        for block in self.base.source_blocks(q,bool(jacobian)):
            cursor+=len(block.r);yield block
        if cursor!=len(self.base.scales):raise ValueError('Original complete source rows changed')
        yield self._motion_block(q,bool(jacobian))

    def evaluate(self,q,jacobian=False):
        blocks=list(self.source_blocks(q,jacobian));r=np.concatenate([b.r for b in blocks])
        return(r,vstack([b.J for b in blocks],format='csr'))if jacobian else r

    def cost(self,q):
        import math
        from calibration.solver_runtime.source_loss import weighted_source_cost
        return math.fsum(weighted_source_cost(b.r,b.quadratic,b.loss_weights)for b in self.source_blocks(q,False))

    def guard(self,q):
        # Original model/source domain remains decisive. Motion uses its own
        # first-visible/occlusion domain; no invalid row is silently removed.
        result=self.base.guard(q);self._motion_block(q,False);return result

    def checkpoint_arrays(self):
        result=self.base.checkpoint_arrays()
        for key,value in self.packet.checkpoint_arrays().items():
            if key in result:
                actual=np.asarray(result[key]);expected=np.asarray(value)
                if (actual.dtype!=expected.dtype or actual.shape!=expected.shape
                        or actual.tobytes()!=expected.tobytes()):
                    raise ValueError('Restored motion evidence auxiliary differs from the authenticated packet: '+key)
            result[key]=value.copy()
        return result

    def checkpoint_identity(self):
        result=deepcopy(self.base.checkpoint_identity())
        result.update(measured_board_motion=deepcopy(self.motion_receipt),
            source_producers=deepcopy(self.source_producers),recovery_references=deepcopy(self.recovery_references))
        return result

    def complete_state_payload(self,q):
        result=self.base.complete_state_payload(q)
        result.update({'aux__'+key:value for key,value in self.checkpoint_arrays().items()})
        return result

    def solver_kwargs(self):
        result=dict(self.base.solver_kwargs())
        result.update(source_block_factory=self.source_blocks,checkpoint_arrays=self.checkpoint_arrays(),
            checkpoint_identity=self.checkpoint_identity(),reuse_source_jacobian=False)
        return result
