"""Resident routing of the existing ad geometry, board and camera CUDA kernels.

This module changes execution placement only. It supplies no optimizer or new
objective. Ceres, the current source layouts, exact full W, E and AL remain the
existing public implementation. Attach only at a solver construction boundary.
"""
from __future__ import annotations
import ctypes as C
import hashlib,time,types
from pathlib import Path
import numpy as np
from .gpu_board import CudaBoardBatch,pack_board
from .optimizer import ForwardDomainRejection


class OriginalAdsGpuPipeline:
    def __init__(self,engine):
        p=engine.params
        if len(engine.profile_rows):
            raise ValueError('resident original-ad routing currently requires the actual no-Query graph; no row may be omitted')
        self.engine=engine;self.device=engine.gpu.device;self.generation=engine.gpu.generation;self.buffers=[]
        self.original_compute=engine._compute
        self.board=CudaBoardBatch(p.count,device=self.device,generation=self.generation)
        model=p.model;board=model.scene.board(model.state);self.board_signature=model.state.get(p.board_key).tobytes()
        self._update(self.board.boards,np.array([pack_board(board)],dtype=np.float64))
        self._update(self.board.nodes,np.ascontiguousarray(board.nodes));self._update(self.board.weights,np.ascontiguousarray(board.weights))
        curves=list(p.world_groups);offsets=[];sizes=[];anchors=[];points=[];row_curve=np.empty(p.count,np.int32);ad_offsets=np.empty(p.count,np.int64)
        for ci,key in enumerate(curves):
            definition=model.scene.resolved_definition(p.curve_lookup[key],model.state)
            if definition['kind']!='board_material_polyline' or not definition.get('shared_layout_block_id'):
                raise ValueError('current ad routing accepts only declared continuous-board material polylines')
            v=np.asarray(definition['unwrapped_station_height'],dtype=np.float64)
            if v.ndim!=2 or v.shape[1]!=2 or not len(v) or not np.isfinite(v).all():raise ValueError('invalid immutable ad path')
            offsets.append(sum(sizes));sizes.append(len(v));points.append(v);anchors.append(definition['shared_content_anchor_unwrapped'])
            rr=p.world_groups[key];row_curve[rr]=ci;ad_offsets[rr]=p.parameter_offsets[p.semantic_to_native[p.curve_group_keys[key]]]
        base=p.parameter_offsets[p.local_block];selected=p.station_column>=0
        station_offsets=np.full(p.count,-1,dtype=np.int64);station_offsets[selected]=base[selected]+p.station_column[selected]
        row_offsets=base+p.row_column
        ad_jbase=np.empty(p.count,np.int64)
        for b in range(len(p.views)):
            loc=p.block_row_locations[b];ad_jbase[loc[:,0]]=loc[:,1]
        self.meta={k:self._upload(np.ascontiguousarray(v)) for k,v in {
            'points':np.concatenate(points),'curve_offsets':np.array(offsets,np.int64),'curve_sizes':np.array(sizes,np.int32),
            'curve_ids':row_curve,'ad_offsets':ad_offsets,'anchors':np.asarray(anchors,dtype=np.float64),
            'station_offsets':station_offsets,'row_offsets':np.asarray(row_offsets,np.int64),'seeds':p.station_seed,
            'frame_times':p.frame_times,'ad_jbase':ad_jbase,'local_jbase':engine._local_offsets,
            'local_width':np.asarray(p.block_sizes,dtype=np.int32)[p.local_block],
            'station_column':p.station_column,'row_column':p.row_column}.items()}
        self.x=self._allocate(p.initial.shape);self.dq=self._allocate((p.count,2));self.q_status=self._allocate((p.count,),np.uint8)
        self.compact=self._allocate((p.count,12));self.jac=self._allocate(engine.Jprojection.shape)
        self.kernels,self.compilation=self.device.compile(Path(__file__).with_name('original_ads_chain.cu'),
            ('ad_unwrapped','ad_board_camera_rows','ad_projection_chain','ad_zero','ad_compact_uv','ad_constraints'))
        budget_index=np.full(p.count,-1,dtype=np.int32);budget_index[engine.budget_rows]=np.arange(len(engine.budget_rows),dtype=np.int32)
        self.constraint_meta={k:self._upload(np.ascontiguousarray(v)) for k,v in {
            'target':p.target_uv,'budget_index':budget_index,'budget_values':engine.budgets,
            'budget_offsets':engine.budget_layout.jacobian_offsets[engine.budget_layout.factor_edge_offsets[:-1]],
            'equality_offsets':engine.roweq_layout.jacobian_offsets[engine.roweq_layout.factor_edge_offsets[:-1]]}.items()}
        self.constraints=self._allocate((engine.constraint_set.layout.residual_count,))
        self.constraint_jac=self._allocate((engine.constraint_set.layout.jacobian_value_count,))
        self.closed=False
        self.description={'schema':'rinksolve.original_ads_resident_gpu.v1','current_sources':p.count,'native_parameters':len(p.initial),
            'existing_board_kernel':self.board.compilation,'existing_camera_kernel':engine.gpu.compilation,
            'routing_kernel':self.compilation,'float64':True,'Ceres_unchanged':True,'original_full_W_unchanged':True,
            'geometry_vertices_free':False,'camera_ice_board_frozen':True,'Query_sources':len(engine.profile_rows),
            'static_geometry_uploaded_once':True,'board_camera_intermediate_host_roundtrip':False,
            'camera_full_69_column_host_download':False,'constraint_assembly':'GPU_same_original_scalar_E_and_row_J','PHR':'existing_numpy_CPU',
            'linear_solver':'existing_Ceres_policy'}
    def _allocate(self,shape,dtype=np.float64):
        v=self.device.allocate(shape,dtype,generation=self.generation);self.buffers.append(v);return v
    def _upload(self,array):
        v=self.device.upload(array,generation=self.generation);self.buffers.append(v);return v
    def _update(self,buffer,array):self.device.update(buffer,np.ascontiguousarray(array),generation=self.generation)
    def compute(self,x,need_jacobian):
        e=self.engine;p=e.params
        if e._last_x is not None and np.array_equal(x,e._last_x) and (not need_jacobian or e._jacobian_ready):return e._cache
        started=time.perf_counter();d=self.device;gpu=e.gpu;m=self.meta;n=p.count
        try:
            model=p.model_for(x)
            if model.state.get(p.board_key).tobytes()!=self.board_signature:raise ValueError('frozen board changed')
            op=model.operator('S01');stations,rows,alpha=p.auxiliaries(x,model)
            # The camera/controls are frozen by OriginalReferenceGuard. Their existing resident buffers remain valid.
            e.original_reference_guard.assert_model(model)
            self._update(self.x,np.ascontiguousarray(x))
            d.launch(self.kernels['ad_unwrapped'],n,(self.x,m['points'],m['curve_offsets'],m['curve_sizes'],m['curve_ids'],m['ad_offsets'],m['anchors'],m['station_offsets'],m['row_offsets'],m['seeds'],C.c_int(n),self.board.q,self.dq,self.q_status))
            d.launch(self.board.kernel,n,(self.board.q,self.board.boards,self.board.nodes,self.board.weights,C.c_int(n),C.c_int(1),self.board.output,self.board.status))
            tau=op.camera_definition['signed_readout_seconds'];h=p.rig.image_height-1
            d.launch(self.kernels['ad_board_camera_rows'],n,(self.board.output,self.x,m['row_offsets'],m['frame_times'],C.c_double(tau/h),C.c_double(tau),C.c_int(n),gpu.xyz,gpu.time))
            gpu.launch(full=need_jacobian)
            if need_jacobian:
                d.launch(self.kernels['ad_zero'],self.jac.shape[0],(self.jac,C.c_longlong(self.jac.shape[0])))
                d.launch(self.kernels['ad_projection_chain'],n,(gpu.output,self.board.output,self.board.q,self.dq,self.x,m['curve_ids'],m['anchors'],m['ad_offsets'],m['ad_jbase'],m['local_jbase'],m['local_width'],m['station_column'],m['row_column'],C.c_double(tau/h),C.c_int(n),self.compact,self.jac))
                value=d.download(self.compact);np.copyto(e.Jprojection,d.download(self.jac));np.copyto(e.Jerror,e.Jprojection)
                forward={'uv':value[:,:2],'depth':value[:,2],'ray_radius':value[:,3],'jacobian_world':value[:,4:10].reshape(-1,2,3),'jacobian_time':value[:,10:12]}
                e.counters['full_linearizations']+=1
            else:
                value=d.download(gpu.uv_output);forward={'uv':value}
            status=d.download(gpu.status);qstatus=d.download(self.q_status);bstatus=d.download(self.board.status)[0]
            forward.update(status=status,valid=(status==0)&(qstatus==0)&(bstatus==0))
            if not forward['valid'].all():
                i=int(np.flatnonzero(~forward['valid'])[0]);raise ForwardDomainRejection('current resident ad trial rejected: '+p.observation_ids[i]+' camera/shape/board statuses '+str((int(status[i]),int(qstatus[i]),int(bstatus[i]))))
            xyz=d.download(gpu.xyz);times=p.frame_times+tau*(rows/h-.5)
            error=np.ascontiguousarray(forward['uv']-p.target_uv)
            cache={'model':model,'operator':op,'stations':stations,'rows':rows,'alpha':alpha,'xyz':xyz,'times':times,'forward':forward,
                'error':error,'query_error':error.copy(),'roweq':rows-forward['uv'][:,1],
                'query_derivative':np.zeros((n,2)),'target_uv_current_query':p.target_uv.copy()}
            e.counters['forward_calls']+=1;e.timings['last_full_linearization_seconds' if need_jacobian else 'last_residual_evaluation_seconds']=time.perf_counter()-started
            self.forward_is_full=need_jacobian
            e._last_x=np.array(x,copy=True);e._jacobian_ready=need_jacobian;e._cache=cache;return cache
        except (ValueError,FloatingPointError) as exc:
            e.counters['domain_rejections']+=1;e.domain_rejection_index+=1
            event={'schema':'rinksolve.original_ads_domain_rejection.v1','error':str(exc),'trial_index':e.domain_rejection_index,
                'camera_ice_board_parameters_cannot_change':True,'native_parameters_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'execution':'resident_existing_CUDA_kernels'}
            e.last_domain_failure=event
            if e.domain_event_callback is not None:e.domain_event_callback(event,x)
            e._last_x=None;e._jacobian_ready=False
            raise ForwardDomainRejection(str(exc)) from exc
    def evaluate_constraints(self,x,constraint_out,jacobian_out,new_point):
        e=self.engine;p=e.params
        try:self.compute(x,jacobian_out is not None)
        except ForwardDomainRejection:return False
        m=self.meta;c=self.constraint_meta;nb=len(e.budget_rows)
        self.device.launch(self.kernels['ad_constraints'],p.count,(e.gpu.output if self.forward_is_full else e.gpu.uv_output,
            C.c_int(69 if self.forward_is_full else 2),c['target'],self.x,m['row_offsets'],c['budget_index'],c['budget_values'],
            c['budget_offsets'],c['equality_offsets'],m['ad_jbase'],m['local_jbase'],m['local_width'],m['row_column'],self.jac,
            C.c_int(p.count),C.c_int(nb),C.c_longlong(p.budget_offset),C.c_longlong(e.budget_layout.jacobian_value_count),
            C.c_int(jacobian_out is not None),self.constraints,self.constraint_jac))
        np.copyto(constraint_out,self.device.download(self.constraints))
        if jacobian_out is not None:np.copyto(jacobian_out,self.device.download(self.constraint_jac))
    def close(self):
        if self.closed:return
        self.board.close()
        for v in self.buffers:v.close()
        self.closed=True


def attach_original_ads_gpu(engine):
    """Attach at a fresh engine boundary; never modifies on-disk live code/state."""
    if hasattr(engine,'resident_ad_gpu'):raise ValueError('resident ad routing already attached')
    pipeline=OriginalAdsGpuPipeline(engine);engine.resident_ad_gpu=pipeline
    engine._compute=types.MethodType(lambda self,x,need_jacobian:pipeline.compute(x,need_jacobian),engine)
    engine.evaluate_constraints=types.MethodType(lambda self,x,r,j,new_point:pipeline.evaluate_constraints(x,r,j,new_point),engine)
    engine.al.constraint_callback=engine.evaluate_constraints
    original_close=engine.close
    def close(self):
        pipeline.close();original_close()
    engine.close=types.MethodType(close,engine)
    engine._last_x=None;engine._jacobian_ready=False
    return pipeline
