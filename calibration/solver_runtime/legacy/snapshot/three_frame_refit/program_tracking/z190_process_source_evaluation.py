"""Persistent isolated processes for independent original source blocks.

Each owned process receives only one minimal source block. Native residuals
and CSR storage travel through reusable shared-memory buffers; q and small
metadata travel through a pipe. Parent iteration and checkpoint order remain
the original source ledger order. This module never stops external processes.
"""
from __future__ import annotations
import copy
import importlib.util
import multiprocessing as mp
from multiprocessing import shared_memory
import os,pickle,sys,time,traceback
from pathlib import Path


class RemoteSourceError(ValueError):
    """Source-domain failures still reject the caller's ordinary LM trial."""


class SourceActorTransportError(ConnectionError):
    """An owned actor transport failure cannot become an ordinary LM trial."""


def _minimal_pickle(block):
    """Shallow copies retain ndarray identity during the single serialization."""
    from prepared_cache_helpers import install_mappingproxy_pickle
    install_mappingproxy_pickle()
    small=copy.copy(block)
    if hasattr(small,'_original_motion_far_board_role_binding'):
        from functools import partial
        from structured_programme_source_candidate.original_motion_far_board_roles import _role_synchronize_course
        small.synchronize_course=partial(_role_synchronize_course,small)
    if hasattr(small,'prepared'):small.prepared=None
    bundle=copy.copy(block.bundle);small.bundle=bundle
    kernel=getattr(bundle,'motion_kernel',None)
    if kernel is not None and hasattr(kernel,'__dict__'):
        kernel=copy.copy(kernel);bundle.motion_kernel=kernel
        if hasattr(kernel,'latest_checks'):kernel.latest_checks=[]
    prior=vars(bundle).get('priors')
    if callable(prior) and getattr(prior,'__name__',None)=='<lambda>':
        bundle._isolated_cache_heightless_prior=True
        del vars(bundle)['priors']
    modules={}
    for function in(getattr(small,'absolute',None),getattr(bundle,'absolute_evaluator',None)):
        name=getattr(function,'__module__',None);module=sys.modules.get(name)
        path=getattr(module,'__file__',None)
        if name and path and name.endswith('_factory'):modules[name]=str(Path(path).resolve())
    return pickle.dumps(small,protocol=5),modules


def _worker_bootstrap(task_root,native_threads,dynamic_modules):
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']=str(native_threads)
    os.environ['RINKSPACE_MOTION_THREADS']=str(native_threads)
    os.environ['RINKSPACE_SCATTER_THREADS']=str(native_threads)
    sys.dont_write_bytecode=True
    from isolated_solver_bootstrap import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    import cv2
    cv2.setNumThreads(1)
    from continuous_central_brown4_optin import enable_projection_backend
    enable_projection_backend()
    for name,path in dynamic_modules.items():
        if name in sys.modules:continue
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)


def _attach(spec):
    import numpy as np
    memory=shared_memory.SharedMemory(name=spec['name'])
    arrays={name:np.ndarray(tuple(meta['shape']),dtype=np.dtype(meta['dtype']),
        buffer=memory.buf,offset=meta['offset'])for name,meta in spec['arrays'].items()}
    return memory,arrays


def _actor_worker(connection,payload,task_root,native_threads,dynamic_modules,spec):
    memory=None;arrays=None;block=None
    try:
        _worker_bootstrap(task_root,native_threads,dynamic_modules)
        import numpy as np
        block=pickle.loads(payload);del payload
        if getattr(block.bundle,'_isolated_cache_heightless_prior',False):
            from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
            block.bundle.priors=lambda q,cache:tuple(x[1:]for x in MovingPivotSourceBundle.priors(block.bundle,q,cache))
        memory,arrays=_attach(spec)
        connection.send(dict(status='ready',pid=os.getpid(),name=block.name))
        while True:
            command=connection.recv();mode=command['mode']
            if mode=='close':break
            begin=time.perf_counter();cpu=time.process_time()
            try:
                if mode=='profiles':
                    block.bundle.robust_phase=bool(command['robust'])
                    block.bundle.profile_weights=arrays['weights'].copy()if command['present']else None
                    reply=dict(status='ok')
                elif mode=='resize_output':
                    replacement=command['spec']
                    # Only the output mapping changes. The independently
                    # unpickled source block and its frozen profile survive.
                    # Validate before attaching or dropping the old mapping.
                    if any(replacement[key]!=spec[key]for key in('row_capacity','nnz_capacity')):
                        raise ValueError('Original source output inventory changed during resize')
                    if replacement['arrays']['weights']['shape']!=spec['arrays']['weights']['shape']:
                        raise ValueError('Original source profile inventory changed during resize')
                    new_memory,new_arrays=_attach(replacement)
                    old_memory=memory;arrays.clear()
                    memory,arrays,spec=new_memory,new_arrays,replacement
                    old_memory.close()
                    reply=dict(status='ok',pid=os.getpid(),output_jacobian=bool(spec['output_jacobian']),
                        shared_memory_name=spec['name'],output_bytes=spec['bytes'])
                elif mode=='profile_update':
                    q=command['q'];block.bundle.robust_phase=True
                    if command['motion_only']:
                        preliminary=block.bundle.profile_motion_residual(block.local(q))
                    else:preliminary=block.evaluate(q,False)[:block.bundle.motion_rows]
                    weights=np.minimum(1.,2./np.maximum(np.abs(preliminary),1e-12)).reshape(-1,2)
                    block.bundle.profile_weights=weights
                    np.copyto(arrays['weights'],weights)
                    reply=dict(status='ok',present=True,robust=True)
                elif mode=='evaluate':
                    jac=command['jacobian']
                    if jac and not spec.get('output_jacobian',True):
                        raise ValueError('Source actor requires Jacobian output capacity before evaluation')
                    result=block.evaluate(command['q'],jac)
                    residual=np.asarray(result[0]if jac else result,float)
                    if residual.shape!=(spec['row_capacity'],):raise ValueError('Original source residual row inventory changed')
                    np.copyto(arrays['residual'],residual)
                    reply=dict(status='ok',rows=len(residual),jacobian=jac)
                    if jac:
                        matrix=result[1].tocsr(copy=False);nnz=int(matrix.nnz)
                        if nnz>spec['nnz_capacity']:raise ValueError('Source CSR exceeded original layout capacity')
                        np.copyto(arrays['data'][:nnz],matrix.data)
                        np.copyto(arrays['indices'][:nnz],matrix.indices,casting='safe')
                        np.copyto(arrays['indptr'],matrix.indptr,casting='safe')
                        reply.update(nnz=nnz,columns=matrix.shape[1],sorted_indices=bool(matrix.has_sorted_indices),
                            canonical=bool(matrix.has_canonical_format))
                    del result,residual
                    if jac:del matrix
                    checks=getattr(getattr(block.bundle,'motion_kernel',None),'latest_checks',None)
                    if checks is not None:checks.clear()
                elif mode=='review':
                    # absolute_records reads retained source groups directly;
                    # prepared input factories/manifests are not involved.
                    # Keep review outside both output buffers and the mutable
                    # frozen-profile state used by the numerical source stream.
                    bundle=block.bundle;groups=bundle.abs_groups
                    weights=bundle.profile_weights;robust=bundle.robust_phase
                    try:
                        records=list(block.absolute_records(command['q']))
                        records.extend(block.absolute_records(command['q'],command['held_rows']))
                    finally:
                        bundle.abs_groups=groups
                        bundle.profile_weights=weights;bundle.robust_phase=robust
                    reply=dict(status='ok',records=records)
                else:raise ValueError('Unknown source actor command: '+str(mode))
                reply.update(wall_seconds=time.perf_counter()-begin,process_cpu_seconds=time.process_time()-cpu)
            except Exception as exc:
                reply=dict(status='error',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                    wall_seconds=time.perf_counter()-begin,process_cpu_seconds=time.process_time()-cpu)
            connection.send(reply)
    except Exception as exc:
        try:connection.send(dict(status='error',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        except(OSError,EOFError,BrokenPipeError):pass
    finally:
        arrays=None;block=None
        if memory is not None:memory.close()
        connection.close()


def _output_memory_spec(block,q_columns=None,output_jacobian=True):
    """Exact owned output layout, without allocating a shared section.

    Capacities and index width come from the complete original source layout
    in either mode. Residual-only transport retains zero-length CSR entries
    so consumers can inspect the same array keys without reserving full J.
    """
    import numpy as np
    bundle=block.bundle
    explicit=getattr(block,'actor_output_capacity',None)
    if explicit is not None:
        rows=int(explicit['rows']);nnz=int(explicit['nnz']);motion_rows=int(explicit['motion_rows'])
    else:
        rows=len(block.quadratic);motion_rows=int(bundle.motion_rows)
        absolute_rows=int(bundle.source_residual_rows)-motion_rows
        # Scatter omits one shared padding slot per source row and can add
        # every shared material column only on the absolute source span.
        nnz=int(bundle.pointer[-1])+3*len(bundle.stamps)-int(bundle.source_residual_rows)+absolute_rows*int(block.geometry_parameter_count)
    if rows<0 or nnz<0 or motion_rows<0 or motion_rows%2:raise ValueError('Original source layout capacity required')
    dtype='<i8'if max(nnz,rows,int(q_columns or 0))>=2**31 else'<i4'
    output_jacobian=bool(output_jacobian);stored_nnz=nnz if output_jacobian else 0
    stored_indptr=rows+1 if output_jacobian else 0
    layout={};offset=0
    for name,shape,dt in(('residual',(rows,),'<f8'),('data',(stored_nnz,),'<f8'),('indices',(stored_nnz,),dtype),
                        ('indptr',(stored_indptr,),dtype),('weights',(motion_rows//2,2),'<f8')):
        offset=(offset+7)//8*8
        layout[name]=dict(shape=shape,dtype=dt,offset=offset)
        offset+=int(np.prod(shape))*np.dtype(dt).itemsize
    return dict(arrays=layout,row_capacity=rows,nnz_capacity=nnz,bytes=max(offset,1),
        output_jacobian=output_jacobian)


def _make_memory(block,q_columns=None,output_jacobian=True):
    import numpy as np
    spec=_output_memory_spec(block,q_columns,output_jacobian)
    memory=shared_memory.SharedMemory(create=True,size=spec['bytes'])
    spec['name']=memory.name;layout=spec['arrays']
    arrays={name:np.ndarray(tuple(meta['shape']),dtype=np.dtype(meta['dtype']),buffer=memory.buf,offset=meta['offset'])
        for name,meta in layout.items()}
    return memory,spec,arrays


class ProcessBlockProfileUpdate:
    """Same parent checkpoint/token protocol, synchronized resident weights."""
    def __init__(self,manager):
        self.manager=manager;self.blocks=manager.blocks
        from z190_joint_source_solver import _BlockProfileUpdate
        self.reference=_BlockProfileUpdate(self.blocks)
    def __call__(self,q):self.manager.update_profiles(q)
    def iter_checkpoint_arrays(self):return self.reference.iter_checkpoint_arrays()
    def checkpoint_arrays(self):return self.reference.checkpoint_arrays()
    def restore_checkpoint(self,arrays):
        self.reference.restore_checkpoint(arrays)
        if self.manager.started:self.manager.synchronize_profiles()


class PersistentSourceActors:
    """One explicit, owned process per distinct original source bundle.

    Results alias actor buffers until that actor's next evaluation. Consume
    them in the ordinary source stream and do not retain aliases for a later
    evaluation or output resize. Caller-owned cache replay must copy arrays
    before retaining. Resizing is explicit so an admission owner can reserve
    the exact new section once while the old section is still alive.
    """
    def __init__(self,blocks,*,task_root=None,native_threads=3,q_columns=None,output_jacobian=True):
        self.blocks=tuple(blocks);self.native_threads=int(native_threads)
        if self.native_threads<1:raise ValueError('Positive native threads per source actor required')
        if len({id(b.bundle)for b in self.blocks})!=len(self.blocks):raise ValueError('Source actors require distinct source bundles')
        self.task_root=Path(task_root or Path(__file__).resolve().parents[3]).resolve()
        self.q_columns=q_columns;self.output_jacobian=bool(output_jacobian)
        self.started=False;self.closed=False;self._busy=False
        self._actors=[];self.last_stats=[];self.last_resize_stats=[];self.profile_update=ProcessBlockProfileUpdate(self)
    def start(self):
        if self.closed:raise RuntimeError('Closed source actors cannot restart')
        if self.started:return self
        context=mp.get_context('spawn')
        # Spawn imports the caller's __main__ before the worker initializer.
        # Set only math-library process limits during these owned starts.
        saved={key:os.environ.get(key)for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        try:
            for block in self.blocks:
                memory,spec,arrays=_make_memory(block,self.q_columns,self.output_jacobian)
                parent,child=context.Pipe()
                actor=dict(block=block,memory=memory,spec=spec,arrays=arrays,connection=parent,process=None)
                self._actors.append(actor)
                weights=block.bundle.profile_weights
                if weights is not None:arrays['weights'][:]=weights
                payload,modules=_minimal_pickle(block)
                from calibration.solver_runtime.worker_entry import run_source_actor
                process=context.Process(target=run_source_actor,args=(child,payload,str(self.task_root),self.native_threads,modules,spec),
                    name='rink-source-'+block.name)
                actor['process']=process;process.start();child.close();del payload
            for actor in self._actors:self._receive(actor)
            self.started=True;self.synchronize_profiles()
            return self
        except Exception:
            self.close();raise
        finally:
            for key,value in saved.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    def _receive(self,actor):
        try:answer=actor['connection'].recv()
        except(EOFError,OSError)as exc:
            detail='Owned source actor receive failed: '+actor['block'].name+' (pid '+str(actor['process'].pid)+', exit '+str(actor['process'].exitcode)+')'
            self.close()
            raise SourceActorTransportError(detail)from exc
        if answer['status']=='error':
            raise RemoteSourceError(actor['block'].name+': '+answer['type']+': '+answer['message']+'\n'+answer['traceback'])
        return answer
    def _send(self,actor,command):
        try:actor['connection'].send(command)
        except(EOFError,OSError)as exc:
            detail='Owned source actor send failed during '+command['mode']+': '+actor['block'].name+' (pid '+str(actor['process'].pid)+', exit '+str(actor['process'].exitcode)+')'
            self.close()
            raise SourceActorTransportError(detail)from exc
    def _check_ready(self):
        if not self.started:self.start()
        if self.closed:raise RuntimeError('Source actors have closed')
        if self._busy:raise RuntimeError('One source actor evaluation may be active at a time')
    def resize_output(self,jacobian):
        """Switch idle owned output mappings, retaining each PID and source.

        One actor's new section is allocated and acknowledged at a time,
        then its old parent/worker mappings are closed. Parent profile weights
        remain linked to the current shared weights; worker profiles remain
        independent. Before startup this only selects the initial layout.
        """
        if self.closed:raise RuntimeError('Source actors have closed')
        if self._busy:raise RuntimeError('Cannot resize an active owned source actor')
        jacobian=bool(jacobian);self.last_resize_stats=[]
        if not self.started:
            self.output_jacobian=jacobian;return self
        self._busy=True
        try:
            for actor in self._actors:
                if actor['spec'].get('output_jacobian',True)==jacobian:continue
                memory,spec,arrays=_make_memory(actor['block'],self.q_columns,jacobian)
                adopted=False
                try:
                    bundle=actor['block'].bundle;value=bundle.profile_weights
                    if value is not None:arrays['weights'][:]=value
                    try:
                        self._send(actor,dict(mode='resize_output',spec=spec))
                        answer=self._receive(actor)
                    except Exception:
                        # An unsuccessful control reply cannot certify which
                        # mapping a worker owns. Close our actors rather than
                        # continue against potentially mismatched buffers.
                        self.close();raise
                    old_memory,old_arrays=actor['memory'],actor['arrays']
                    actor.update(memory=memory,spec=spec,arrays=arrays);adopted=True
                    if value is not None:bundle.profile_weights=arrays['weights']
                    old_arrays.clear();old_memory.close();old_memory.unlink()
                    self.last_resize_stats.append(dict(block=actor['block'].name,**answer))
                finally:
                    if not adopted:
                        arrays.clear();memory.close();memory.unlink()
            self.output_jacobian=jacobian
            return self
        finally:self._busy=False
    def synchronize_profiles(self):
        self._check_ready();self._busy=True;received=0;sent=0
        try:
            for actor in self._actors:
                bundle=actor['block'].bundle;value=bundle.profile_weights
                if value is not None:
                    actor['arrays']['weights'][:]=value;bundle.profile_weights=actor['arrays']['weights']
                self._send(actor,dict(mode='profiles',present=value is not None,robust=bool(bundle.robust_phase)));sent+=1
            for actor in self._actors:
                received+=1;self._receive(actor)
        finally:
            for actor in self._actors[received:sent]:
                try:self._receive(actor)
                except SourceActorTransportError:raise
                except Exception:pass
            self._busy=False
    def update_profiles(self,q):
        import numpy as np
        self._check_ready();self._busy=True;self.last_stats=[];received=0;sent=0
        try:
            q=np.asarray(q,float);motion_only=os.environ.get('RINKSPACE_PROFILE_MOTION_ONLY','0')=='1'
            for actor in self._actors:
                self._send(actor,dict(mode='profile_update',q=q,motion_only=motion_only));sent+=1
            for actor in self._actors:
                received+=1;reply=self._receive(actor);self.last_stats.append(dict(block=actor['block'].name,**reply))
                actor['block'].bundle.profile_weights=actor['arrays']['weights'];actor['block'].bundle.robust_phase=True
        finally:
            for actor in self._actors[received:sent]:
                try:self._receive(actor)
                except SourceActorTransportError:raise
                except Exception:pass
            self._busy=False
    def ordered_evaluations(self,q,jacobian):
        import numpy as np
        from scipy.sparse import csr_matrix
        self._check_ready()
        if jacobian and any(not actor['spec'].get('output_jacobian',True)for actor in self._actors):
            raise RuntimeError('Call resize_output(True) before a Jacobian source evaluation')
        self._busy=True;self.last_stats=[];received=0;sent=0
        try:
            q=np.asarray(q,float)
            for actor in self._actors:
                self._send(actor,dict(mode='evaluate',q=q,jacobian=bool(jacobian)));sent+=1
            for actor in self._actors:
                received+=1;answer=self._receive(actor);self.last_stats.append(dict(block=actor['block'].name,**answer))
                arrays=actor['arrays'];residual=arrays['residual']
                if jacobian:
                    nnz=answer['nnz'];matrix=csr_matrix((arrays['data'][:nnz],arrays['indices'][:nnz],arrays['indptr']),
                        shape=(answer['rows'],answer['columns']),copy=False)
                    matrix.has_sorted_indices=answer['sorted_indices'];matrix.has_canonical_format=answer['canonical']
                    yield actor['block'],(residual,matrix)
                else:yield actor['block'],residual
        finally:
            # A source stream may close early after an exception. Drain only
            # our outstanding replies before the next evaluation or shutdown.
            for actor in self._actors[received:sent]:
                try:self._receive(actor)
                except SourceActorTransportError:raise
                except Exception:pass
            self._busy=False
    def ordered_absolute_records(self,q,held_by_name):
        """Yield (original block, complete fit-then-held records) in order.

        Review never writes residual/CSR/profile buffers, so a same-state
        numerical replay remains valid across an intervening source review.
        Held groups are transmitted as supplied on every call, preserving
        caller corrections and every original source row and timestamp.
        """
        import numpy as np
        self._check_ready();self._busy=True;self.last_stats=[];received=0;sent=0
        try:
            q=np.asarray(q,float)
            for actor in self._actors:
                self._send(actor,dict(mode='review',q=q,held_rows=held_by_name[actor['block'].name]));sent+=1
            for actor in self._actors:
                received+=1;answer=self._receive(actor)
                records=answer.pop('records');self.last_stats.append(dict(block=actor['block'].name,record_count=len(records),**answer))
                yield actor['block'],records
        finally:
            for actor in self._actors[received:sent]:
                try:self._receive(actor)
                except SourceActorTransportError:raise
                except Exception:pass
            self._busy=False
    def close(self):
        if self.closed:return
        self.closed=True
        for actor in self._actors:
            connection=actor['connection']
            try:connection.send(dict(mode='close'))
            except(OSError,EOFError,BrokenPipeError):pass
        for actor in self._actors:
            process=actor['process']
            if process is not None and process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    # Only processes this manager created are eligible.
                    process.terminate();process.join(timeout=10)
            bundle=actor['block'].bundle;value=bundle.profile_weights
            if value is not None and getattr(value,'base',None)is not None:bundle.profile_weights=value.copy()
            actor['arrays'].clear();actor['connection'].close()
            try:actor['memory'].close()
            except BufferError:pass
            actor['memory'].unlink()
        self._actors.clear();self.started=False
    def __enter__(self):return self.start()
    def __exit__(self,*args):self.close()


def make_process_profile_update(manager):return manager.profile_update
