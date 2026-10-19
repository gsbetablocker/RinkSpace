"""Full original programme source on the common named66 world.

Every original source row keeps its UV, sigma, role and operator.  Only its
physical material and finite boundary follow the current common geometry.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import copy,time
import numpy as np
import cv2
from scipy.sparse import csr_matrix,vstack,hstack
from .paired_programme_chart import CORE_COUNT,WORLD_INDICES
from shared_board_course_candidate.course_pattern_adapter import CourseMountOpticsPatterns,_factor_geometry
from shared_board_course_candidate.course_pattern_geometry import rebuild_patterns_geometry
from shared_board_course_candidate.board_course_material import certified_finite_course_binding
from installation_mount_candidate.mount_graph import evaluate_external_with_mount,bind_external_main_mount,validate_mounts
from z190_fixed_target_cross_view_factor import fixed_target_source_factor
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA
from z190_joint_source_solver import make_profile_update
from z190_streaming_source_normal import SourceRowBlock
from z190_parallel_source_evaluation import ordered_block_evaluations,make_parallel_profile_update
from z190_fixed_pivot_board_side_support import fixed_pivot_board_side_support
from .native_camera_local import native_camera_local

COURSE_INDICES=WORLD_INDICES[44:50]
INSTALLATIONS={'long_side_main':0,'long_side_secondary':1,'low_end':2}

class CurrentPatterns(CourseMountOpticsPatterns):
    @classmethod
    def install(cls,owner,state):
        out=cls.__new__(cls);out.__dict__.update(vars(owner));out.state=state
        out.installation_optics=state.installation_optics
        out._original_course_pattern_geometry=owner.geometry
        out._original_course_pattern_factor=owner.factor
        out._original_course_external_pattern_factor=owner.external_factor
        out.course_world_geometry=state.geometry;out.course_indices=COURSE_INDICES.copy()
        out.course_fd_steps=np.full(6,1e-6)
        out.manifest=dict(owner.manifest,current_named66_course=True,original_q9_and_source_factors_preserved=True)
        return out
    def _course_fd_stencil(self,q,slot,h):
        from .batched_absolute_geometry_fd import _feasible_world_stencil
        indices=np.asarray(self.state.geometry_parameter_indices,int)
        columns=np.flatnonzero(indices==slot)
        if len(columns)!=1:raise ValueError('Current pattern course slot needs one complete world coordinate')
        points,_,report=_feasible_world_stencil(self.course_world_geometry,q[indices],int(columns[0]),h)
        samples=[]
        for world in points:
            value=q.copy();value[indices]=world;samples.append(value)
        return samples,report
    @contextmanager
    def bind_current_geometry(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.state.structured_programme_coordinate_chart.core_count,):raise ValueError('Complete physical pattern state required')
        old=(self.geometry,self.factor,self.external_factor)
        course=self.course_world_geometry.source_course.course
        cq=self.course_world_geometry.current.current_course6(q[self.state.geometry_parameter_indices])
        self.geometry=rebuild_patterns_geometry(self._original_course_pattern_geometry,course,cq)
        self.factor=_factor_geometry(self._original_course_pattern_factor,self.geometry)
        self.external_factor=_factor_geometry(self._original_course_external_pattern_factor,self.geometry)
        try:yield self.geometry
        finally:self.geometry,self.factor,self.external_factor=old

class CurrentExternal:
    def __init__(self,state):
        self.state=state;self.external=state.external;self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        if self.external is None:return
        self.top=self.external.top_world_scene.copy();self.top_m=self.external.top_world_m.copy()
        self.ad14=self.external.ad14_rows
        self.material=certified_finite_course_binding(state.geometry.source_course.course,self.top,
            identity='original_external_board_top_left_short_lower_left_corner',
            authority='original ExternalMainLowSourceBundle finite top branch',z_policy='top')
        self.active=np.array(sorted(set(range(44,50))|set(state.geometry.current.material_parameter_indices(14))),int)
    @contextmanager
    def bind(self,q):
        if self.external is None:yield;return
        world=np.asarray(q)[self.world_indices];g=self.state.geometry.current
        old=(self.external.top_world_scene,self.external.top_world_m,self.external.ad14_rows)
        self.external.top_world_scene=self.material.evaluate(g.current_course6(world))
        self.external.top_world_m=self.external.top_world_scene*SCENE_UNIT_M
        self.external.ad14_rows=[dict(r,world_xyz_m=g.corner(14,r['source_corner_role'],world)*SCENE_UNIT_M,
            world_xyz_scene_units=g.corner(14,r['source_corner_role'],world).tolist())for r in self.ad14]
        try:yield
        finally:self.external.top_world_scene,self.external.top_world_m,self.external.ad14_rows=old
    def raw(self,state,q,*,jacobian=True,held=False,local_override=None):
        local=q[state.external_start:state.external_stop]if local_override is None else local_override
        with self.bind(q):return self.external.evaluate(local,q[state.slots[state.ids[0]]],q[state.slots[state.ids[2]]],jacobian=jacobian,held=held)
    def evaluate(self,q,*,jacobian=True,held=False,local_override=None):
        state=self.state
        result=evaluate_external_with_mount(state,q,jacobian=jacobian,held=held,local_override=local_override,evaluator=self.raw)
        if not jacobian:return result
        prefix=np.zeros((len(result.residual),14));prefix[:,state.slots[state.ids[0]]]=result.observer_xyz_m_jacobian
        prefix[:,state.slots[state.ids[2]]]=result.target_xyz_m_jacobian
        J=hstack((csr_matrix(prefix),csr_matrix((len(result.residual),state.external_start-14)),
            csr_matrix(result.local_jacobian),csr_matrix((len(result.residual),len(q)-state.external_stop))),format='csr')
        # Camera chart columns are explicit; current world columns come from
        # replaying the exact inverse-anchor / finite-top / Ad14 source chain.
        rr,cc=np.nonzero(result.optics_jacobian);J+=csr_matrix((result.optics_jacobian[rr,cc],(rr,np.asarray(result.optics_parameter_indices)[cc])),shape=J.shape)
        rr,cc=np.nonzero(result.mount_jacobian);J+=csr_matrix((result.mount_jacobian[rr,cc],(rr,np.asarray(result.mount_parameter_indices)[cc])),shape=J.shape)
        columns=np.zeros((len(result.residual),len(self.active)));expected=[r['id']for r in result.source_row_ledger]
        from .batched_absolute_geometry_fd import _feasible_world_stencil
        domain_checks=[]
        for j,k in enumerate(self.active):
            points,_,stencil=_feasible_world_stencil(state.geometry,q[self.world_indices],int(k),1e-6)
            values=[]
            for world in points:
                value=q.copy();value[self.world_indices]=world
                item=result if np.array_equal(value,q)else self.evaluate(value,jacobian=False,held=held,local_override=local_override)
                if [r['id']for r in item.source_row_ledger]!=expected:raise ValueError('Original external ledger changed')
                values.append(item.residual)
            if stencil['method']=='central':columns[:,j]=(values[0]-values[1])/(2*stencil['step'])
            else:
                columns[:,j]=stencil['side']*(4*(values[1]-values[0])-(values[2]-values[0]))/(2*stencil['step'])
                domain_checks.append(stencil)
        self._last_external_geometry_fd_domain_checks=domain_checks
        rr,cc=np.nonzero(columns);J+=csr_matrix((columns[rr,cc],(rr,self.world_indices[self.active[cc]])),shape=J.shape)
        return SimpleNamespace(**vars(result),global_jacobian=J)

class CurrentProgrammeSource:
    def __init__(self,state,args,old_cap):
        self.state=state;self.args=args;self.old_cap=old_cap;self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        self.external=CurrentExternal(state)
        if state.patterns is not None:state.patterns=CurrentPatterns.install(state.patterns,state)
        self.quadratic=np.asarray(old_cap['quadratic']).copy();self.scales=np.asarray(old_cap['scales']).copy()
        self.actors=None
        if getattr(args,'residual_source_workers',0):
            from .residual_only_source_actors import ResidualOnlySourceActors
            from pathlib import Path
            self.actors=ResidualOnlySourceActors(state.blocks,task_root=Path(__file__).resolve().parents[1],
                native_threads=int(getattr(args,'residual_native_threads',3)),q_columns=len(state.q))
            state.source_actors=self.actors
        elif getattr(args,'source_processes',False):
            from .idle_parent_fallback_source_actors import IdleParentFallbackSourceActors
            from pathlib import Path
            self.actors=IdleParentFallbackSourceActors(state.blocks,task_root=Path(__file__).resolve().parents[1],
                native_threads=int(getattr(args,'source_actor_threads',3)),q_columns=len(state.q))
            state.source_actors=self.actors
        self.profile_update=self.actors.profile_update if self.actors is not None else make_parallel_profile_update(state.blocks,1)
        self.actual_family_times=[]
        self.refs={n:next(b for b in state.blocks if b.bundle.calibration.installation_id==n)for n in state.ids}
    def cross(self,q,row,jacobian):
        state=self.state;p=row['actual_pts'];b=state.bypts[p];local=native_camera_local(b,q);cal=b.bundle.current_calibration(local)
        observer=state.ids[INSTALLATIONS[row['observer_installation']]];target=state.ids[INSTALLATIONS[row['target_installation']]]
        index=b.start+4*b.bundle.index[p];view=q[index:index+4]
        result=fixed_target_source_factor(row['factor_source'],q[state.slots[observer]],q[state.slots[target]],view[:3],view[3],
            np.asarray(cal.lever_xyz)*SCENE_UNIT_M,np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]]),cal.k1[0])
        if not jacobian:return result.residual,result
        J=np.zeros((2,len(q)));J[:,state.slots[observer]]+=result.observer_xyz_m_jacobian;J[:,state.slots[target]]+=result.target_xyz_m_jacobian
        J[:,index:index+4]+=result.observer_view_R3_logf_jacobian
        try:
            for col,width,scale in zip(b.optics_chart.indices,b.optics_chart.physical_steps,(1.,1.,.001)):
                h=float(width/scale);p,m=q.copy(),q.copy();p[col]+=h;m[col]-=h;J[:,col]=(self.cross(p,row,False)[0]-self.cross(m,row,False)[0])/(2*h)
            for col in b.optics_chart.mount_indices:
                h=1e-7;p,m=q.copy(),q.copy();p[col]+=h;m[col]-=h;J[:,col]=(self.cross(p,row,False)[0]-self.cross(m,row,False)[0])/(2*h)
        finally:native_camera_local(b,q)
        return result.residual,csr_matrix(J),result
    def evaluations(self,q,jacobian=True):
        s=self.state
        begin=time.perf_counter()
        evaluations=(ordered_block_evaluations(s.blocks,q,jacobian,1)if self.actors is None else self.actors.ordered_evaluations(q,jacobian))
        for b,result in evaluations:
            ready=time.perf_counter();self.actual_family_times.append(dict(family='programme:'+b.name,jacobian=bool(jacobian),ready_after_previous_consume_seconds=ready-begin))
            print(__import__('json').dumps(dict(stage='actual_programme_source_block_ready',block=b.name,jacobian=bool(jacobian),rows=len(result[0]if jacobian else result),seconds=ready-begin,actors=self.actors is not None)),flush=True)
            yield 'programme:'+b.name,result[0]if jacobian else result,result[1]if jacobian else None
            begin=time.perf_counter()
        callback=getattr(self,'after_programme_jacobian',None)
        if jacobian and callback is not None:callback()
        yield from self.tail_evaluations(q,jacobian)
    def tail_evaluations(self,q,jacobian=True):
        s=self.state
        for row in s.cross_fit:
            result=self.cross(q,row,jacobian);yield 'physical_target:'+row['id'],result[0],result[1]if jacobian else None
        if s.external is not None:
            result=self.external.evaluate(q,jacobian=jacobian)
            if [r['id']for r in result.source_row_ledger]!=s.external_contract[2]:raise ValueError('Original external rows changed')
            yield 'external_main_to_low',result.residual,result.global_jacobian if jacobian else None
        if s.patterns is not None:
            begin=time.perf_counter();result=s.patterns.evaluate(q,jacobian)
            self.actual_family_times.append(dict(family='shared_board_patterns',jacobian=bool(jacobian),evaluation_seconds=time.perf_counter()-begin))
            yield 'shared_board_patterns',result[0]if jacobian else result,result[1]if jacobian else None
            callback=getattr(self,'after_shared_pattern_evaluation',None)
            if callback is not None:callback()
        for name in s.ids:
            b=self.refs[name];local=native_camera_local(b,q);cal=b.bundle.current_calibration(local);camera,_,vj=cal.camera25_and_jacobians(b.bundle.view(local,b.bundle.reference_pts))
            value=np.array([(camera[2]*SCENE_UNIT_M-1.6)/.18]);J=None
            if jacobian:
                J=np.zeros((1,len(q)));J[0,s.slots[name][2]]=1/.18;pos=b.start+4*b.bundle.index[b.bundle.reference_pts];J[0,pos:pos+4]=vj[2]*SCENE_UNIT_M/.18
                J[0,b.optics_chart.mount_indices]=cv2.Rodrigues(np.asarray(camera[3:6]))[0].T[2]/.18;J=csr_matrix(J)
            yield 'optical_height:'+name,value,J
        # Current symmetric row centres enforce the former midpoint relation.
        # Keep its original scalar position as a redundant structural zero.
        yield 'common_row_midpoint',np.zeros(1),csr_matrix((1,len(q)))if jacobian else None
        if getattr(self.args,'pivot_border_policy','outside-soft')=='outside-soft':
            world=q[self.world_indices];g=s.geometry.current;course,cq,chain=g.print_course_binding(world)
            polygon=course.sample_perimeter(cq)[0][:,:2]*SCENE_UNIT_M
            for name in s.ids:
                r,pivotj,_=fixed_pivot_board_side_support(polygon,q[s.slots[name]],self.args.pivot_board_soft_sigma_m);J=None
                if jacobian:
                    J=np.zeros((1,len(q)));J[:,s.slots[name]]=pivotj
                    for k in range(44,50):
                        h=1e-6;p,m=world.copy(),world.copy();p[k]+=h;m[k]-=h
                        pp=course.sample_perimeter(g.current_course6(p))[0][:,:2]*SCENE_UNIT_M;pm=course.sample_perimeter(g.current_course6(m))[0][:,:2]*SCENE_UNIT_M
                        J[:,self.world_indices[k]]=(fixed_pivot_board_side_support(pp,q[s.slots[name]],self.args.pivot_board_soft_sigma_m)[0]-fixed_pivot_board_side_support(pm,q[s.slots[name]],self.args.pivot_board_soft_sigma_m)[0])/(2*h)
                    J=csr_matrix(J)
                yield 'fixed_pivot_board_side:'+name,r,J
    def blocks(self,q,jacobian=True):
        cursor=0
        for family,r,J in self.evaluations(q,jacobian):
            stop=cursor+len(r)
            if stop>len(self.scales):raise ValueError('Original source row count changed')
            yield SourceRowBlock(r,J,self.quadratic[cursor:stop],self.scales[cursor:stop],dict(family=family,original_start=cursor,original_stop=stop,source_sha256=SOURCE_SHA))
            cursor=stop
        if cursor!=len(self.scales):raise ValueError('Original source family omitted')
    def guard(self,q):
        s=self.state;q=np.asarray(q,float);s.geometry.validate_parameters(q[self.world_indices]);validate_mounts(s,q)
        for name in s.ids:
            if q[s.slots[name]][2]<=0:raise ValueError('Physical panhead height must be positive')
        for b in s.blocks:
            local=b.local(q);f=local[7::4]
            if np.any((f<np.log(100.))|(f>np.log(50000.))):raise ValueError('Physical focal domain')
            b.bundle.validate_finite_extents(b.bundle.cache(local,False))
        for row in s.cross_fit:self.cross(q,row,False)
        if s.external is not None:
            with bind_external_main_mount(s,q),self.external.bind(q):
                result=s.external.guard(q[s.external_start:s.external_stop],q[s.slots[s.ids[0]]],q[s.slots[s.ids[2]]])
                if result['valid']is not True:raise ValueError('Original external domain: '+result['reason'])
        if s.patterns is not None:
            with s.patterns.bind_current_geometry(q):s.patterns.geometry.guard(q[s.patterns.start:s.patterns.start+9])
    def capture(self):
        shared=np.r_[np.arange(5,14),np.arange(self.state.patterns.start,self.state.patterns.start+9),self.state.mount_parameter_indices,self.state.optics_parameter_indices,self.world_indices]
        if len(shared)!=36+len(self.world_indices) or len(set(shared))!=len(shared):raise ValueError('Explicit complete current shared geometry chart required')
        return dict(iter_source_blocks=self.blocks,profile_update=self.profile_update,guard=self.guard,
            quadratic=self.quadratic,scales=self.scales,shared_schur_indices=shared,args=self.args,
            current_external_evaluator=self.external.evaluate)
