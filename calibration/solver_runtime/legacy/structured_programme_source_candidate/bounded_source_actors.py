"""Measured-memory admission around the original persistent source actors.

Only scheduling changes: original blocks, ordered rows, CSR transport and
profile formulas are retained. The verified six-block pool is a request,
while current physical/commit headroom decides actual residency/concurrency.
"""
import os,json,math,pickle,time
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace
from z190_process_source_evaluation import PersistentSourceActors,RemoteSourceError,_minimal_pickle
from z190_parallel_source_evaluation import make_parallel_profile_update


class GuardedSourceActors(PersistentSourceActors):
    """Original pipe/shared-CSR transport with owned wait monitoring."""
    def __init__(self,*args,memory_owner,**kwargs):
        super().__init__(*args,**kwargs);self.memory_owner=memory_owner;self.identities={}
    def _receive(self,actor):
        from rinkspace.runtime_memory import process_identity
        process=actor['process']
        if process.pid not in self.identities:self.identities[process.pid]=process_identity(process.pid)
        while not actor['connection'].poll(.5):
            try:self.memory_owner.sample('waiting_'+actor['block'].name)
            except Exception:
                self.close();raise
            if not process.is_alive():break
        self.memory_owner.sample('receive_'+actor['block'].name)
        return super()._receive(actor)
    def close(self):
        from rinkspace.runtime_memory import process_identity
        for actor in self._actors:
            process=actor['process']
            if process is None or process.pid is None or not process.is_alive():continue
            expected=self.identities.get(process.pid)
            if expected is None:expected=process_identity(process.pid);self.identities[process.pid]=expected
            if process_identity(process.pid)!=expected:raise RuntimeError('Owned actor PID identity changed; refusing termination')
        return super().close()


class BoundedProfileUpdate:
    def __init__(self,manager):
        self.manager=manager;self.reference=make_parallel_profile_update(manager.blocks,1)
    def __call__(self,q):self.manager.update_profiles(q)
    def iter_checkpoint_arrays(self):return self.reference.iter_checkpoint_arrays()
    def checkpoint_arrays(self):return self.reference.checkpoint_arrays()
    def restore_checkpoint(self,values):
        self.reference.restore_checkpoint(values);self.manager.synchronize_profiles()


class BoundedSourceActors:
    def __init__(self,blocks,*,task_root,native_threads=3,q_columns=None,requested_workers=6):
        self.blocks=tuple(blocks);self.task_root=Path(task_root);self.native_threads=int(native_threads);self.q_columns=q_columns
        if len({b.name for b in self.blocks})!=len(self.blocks):raise ValueError('Distinct original block names required')
        if len({id(b.bundle)for b in self.blocks})!=len(self.blocks):raise ValueError('Distinct original source bundles required')
        if not 1<=int(requested_workers)<=6:raise ValueError('Requested actors must be within the verified 1..6 configuration')
        self.requested_workers=min(int(requested_workers),len(self.blocks));self.pool={};self.last_used={};self.use_counter=0
        self.last_stats=[];self.memory_samples=[];self.observed_worker_peak=None;self.worker_samples={}
        self.payload_measurements={};self.parent_start_measurements={};self._starting_parent=None
        self.queued_blocks=[];self.queue_reason=None
        # Startup-only samples cannot certify either evaluation mode. Keep the
        # original evaluation flag/blocks as the Jacobian compatibility view.
        self.evaluation_peak_measured=False;self.calibrated_blocks=set()
        self.residual_peak_measured=False;self.residual_calibrated_blocks=set()
        self.observed_residual_worker_peak=None;self.observed_jacobian_worker_peak=None
        self.evaluation_worker_samples={'residual':{},'jacobian':{}};self.active_evaluation_mode=None
        self.current_worker_samples={}
        self.active_names=set();self.active_inflight_bytes=0
        self.runtime_report_path=None;self.reserve_bytes=4*1024**3;self.sample_failure=None;self.closed=False;self._busy=False
        self.profile_update=BoundedProfileUpdate(self)
        from rinkspace.runtime_memory import OwnedMemoryMonitor,process_identity
        self.memory_monitor=OwnedMemoryMonitor([process_identity(os.getpid())])
    def _output_bytes(self,block,output_jacobian=True):
        # Exact dtype threshold and 8-byte starts from original _make_memory.
        c=block.actor_output_capacity;rows=int(c['rows']);nnz=int(c['nnz']);motion=int(c['motion_rows'])
        index_bytes=8 if max(nnz,rows,int(self.q_columns or 0))>=2**31 else 4
        offset=0
        stored_nnz=nnz if output_jacobian else 0
        stored_indptr=rows+1 if output_jacobian else 0
        for count,width in ((rows,8),(stored_nnz,8),(stored_nnz,index_bytes),(stored_indptr,index_bytes),(motion,8)):
            offset=(offset+7)//8*8;offset+=count*width
        return max(offset,1)
    @staticmethod
    def _residual_shared_bytes(block):
        c=block.actor_output_capacity
        return (int(c['rows'])+int(c['motion_rows']))*8
    def _event(self,item):
        self.memory_samples.append(item)
        if len(self.memory_samples)>128:self.memory_samples=self.memory_samples[-128:]
    def _measure_payload(self,block):
        """Count the original minimal pickle without a second payload/graph.

        The original _minimal_pickle code still performs its exact shallow
        bundle/kernel preparation. Only its private dumps name is substituted
        in independent function globals; the transport/module is untouched.
        Protocol5 metadata is streamed to a count sink, contiguous buffers
        are counted by nbytes rather than materialized/copied.
        """
        metadata=0;buffers=[]
        class Counter:
            def write(self,value):
                nonlocal metadata
                metadata+=len(value)
        def buffer_count(value):
            view=value.raw()
            try:buffers.append(view.nbytes)
            finally:view.release()
        def count_dumps(value,protocol):
            if protocol!=5:raise ValueError('The authenticated source actor protocol must remain 5')
            pickle.Pickler(Counter(),protocol=protocol,buffer_callback=buffer_count).dump(value)
            return None
        namespace=dict(_minimal_pickle.__globals__);namespace['pickle']=SimpleNamespace(dumps=count_dumps)
        counter=FunctionType(_minimal_pickle.__code__,namespace,name=_minimal_pickle.__name__,argdefs=_minimal_pickle.__defaults__,closure=_minimal_pickle.__closure__)
        unused,modules=counter(block)
        # NEXT_BUFFER/READONLY_BUFFER are replaced by in-band bytes opcodes;
        # allow 32 bytes per buffer for length/opcode and frame boundaries.
        estimate=metadata+sum(buffers)+32*len(buffers)
        record=dict(protocol=5,metadata_bytes=metadata,buffer_bytes=sum(buffers),buffer_count=len(buffers),
            inband_framing_allowance_bytes=32*len(buffers),payload_bytes_upper_estimate=estimate,
            dynamic_module_count=len(modules),original_minimal_pickle_code_reused=True,
            full_payload_or_source_graph_copied=False)
        self.payload_measurements[block.name]=record
        return record
    def _parent_sample(self,result=None):
        value=self.memory_monitor.latest if result is None else result
        if value is None or not value.complete:raise MemoryError('Complete owned parent memory sample required')
        identity=self.memory_monitor.roots[0]
        return next(p for p in value.processes if p.identity==identity)
    def sample(self,phase,*,evaluation_completed=None):
        from rinkspace.runtime_memory import read_system_memory,memory_pressure,subtree_from_snapshot,measured_worker_peak,WorkerPeak
        result=self.memory_monitor.sample();system=read_system_memory();recovery=None
        if not phase.startswith('queued_')and memory_pressure(system,
                reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes):
            # Submission reserves can be consumed by later parent allocations
            # or other work while an original actor command is running. Idle
            # owned peers have no outstanding replies and their profile owner
            # copies weights on close. Release them before treating pressure
            # as an unrecoverable active-command failure; never touch an active
            # or starting actor, or any process outside this pool.
            before=system;released=[]
            starting=None if self._starting_parent is None else self._starting_parent[0]
            candidates=sorted(self.pool,key=lambda n:self.last_used.get(n,-1))
            for name in candidates:
                if name in self.active_names or name==starting or self.pool[name]._busy:continue
                self._remove(name);released.append(name)
                result=self.memory_monitor.sample();system=read_system_memory()
                if not memory_pressure(system,reserve_physical_bytes=self.reserve_bytes,
                        reserve_commit_bytes=self.reserve_bytes):break
            if released:
                recovery=dict(released_idle_owned_blocks=released,before=asdict(before),after=asdict(system),
                    active_blocks_preserved=sorted(self.active_names),reserve_bytes_unchanged=self.reserve_bytes)
        item=dict(phase=phase,owned=asdict(result),system=asdict(system))
        if recovery is not None:item['idle_pressure_recovery']=recovery
        if self._starting_parent is not None and result.complete:
            name,before=self._starting_parent;parent=self._parent_sample(result)
            self.parent_start_measurements[name]=dict(parent_rss_growth_bytes=max(0,parent.rss_bytes-before.rss_bytes),
                parent_private_growth_bytes=max(0,parent.private_bytes-before.private_bytes),
                parent_lifetime_peak_rss_increment_bytes=max(0,parent.peak_rss_bytes-before.peak_rss_bytes),
                parent_lifetime_peak_private_increment_bytes=max(0,parent.peak_private_bytes-before.peak_private_bytes),
                earlier_driver_peak_can_mask_this_startup_peak=True)
        if phase.startswith('waiting_')and self.memory_samples and self.memory_samples[-1].get('phase')==phase:self.memory_samples[-1]=item
        else:self._event(item)
        for manager in tuple(self.pool.values()):
            for actor in manager._actors:
                process=actor['process']
                if process is None or process.pid is None or not process.is_alive():continue
                identity=manager.identities.get(process.pid)
                if identity is None:continue
                try:worker=subtree_from_snapshot(result,identity)
                except MemoryError as exc:self.sample_failure=str(exc);continue
                name=actor['block'].name
                # Current counters are attributed to this command. Lifetime
                # counters remain diagnostics and never certify another mode.
                current=WorkerPeak(worker.rss_bytes,worker.private_bytes,1)
                self.current_worker_samples[name]=current
                peak=measured_worker_peak((worker,));old=self.worker_samples.get(name)
                if old is not None:peak=WorkerPeak(max(old.rss_bytes,peak.rss_bytes),max(old.private_bytes,peak.private_bytes),old.samples+1)
                self.worker_samples[name]=peak
                if name in self.active_names and self.active_evaluation_mode is not None:
                    # Sample only this block while its evaluate command runs.
                    # A previous J lifetime peak must not inflate residuals.
                    samples=self.evaluation_worker_samples[self.active_evaluation_mode]
                    mode_peak=current;previous=samples.get(name)
                    if previous is not None:
                        mode_peak=WorkerPeak(max(previous.rss_bytes,mode_peak.rss_bytes),
                            max(previous.private_bytes,mode_peak.private_bytes),previous.samples+1)
                    samples[name]=mode_peak
        for mode,samples in self.evaluation_worker_samples.items():
            if samples:
                peak=WorkerPeak(max(v.rss_bytes for v in samples.values()),max(v.private_bytes for v in samples.values()),
                    sum(v.samples for v in samples.values()))
                if mode=='jacobian':self.observed_jacobian_worker_peak=peak
                else:self.observed_residual_worker_peak=peak
        if self.worker_samples:
            self.observed_worker_peak=WorkerPeak(max(v.rss_bytes for v in self.worker_samples.values()),
                max(v.private_bytes for v in self.worker_samples.values()),sum(v.samples for v in self.worker_samples.values()))
        if not result.complete:self.sample_failure='Incomplete owned driver memory sample: '+repr(result.errors)
        if not phase.startswith('queued_')and memory_pressure(system,reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes):
            self.sample_failure='Insufficient physical/commit headroom during owned source evaluation'
        if evaluation_completed is not None and self.sample_failure is None:
            name,jacobian=evaluation_completed
            if jacobian:
                if self.observed_jacobian_worker_peak is None:raise MemoryError('Complete real Jacobian worker peak required')
                self.evaluation_peak_measured=True;self.calibrated_blocks.add(name)
            else:
                if self.observed_residual_worker_peak is None:raise MemoryError('Complete real residual worker peak required')
                self.residual_peak_measured=True;self.residual_calibrated_blocks.add(name)
        if self.runtime_report_path is not None:
            report=dict(latest=item,sample_count=self.memory_monitor.samples,
                owned_peak_rss_bytes=self.memory_monitor.peak_rss_bytes,owned_peak_private_bytes=self.memory_monitor.peak_private_bytes,
                measured_one_worker_peak=None if self.observed_worker_peak is None else asdict(self.observed_worker_peak),
                evaluation_peak_measured=self.evaluation_peak_measured,calibrated_blocks=sorted(self.calibrated_blocks),
                residual_peak_measured=self.residual_peak_measured,residual_calibrated_blocks=sorted(self.residual_calibrated_blocks),
                measured_residual_worker_peak=None if self.observed_residual_worker_peak is None else asdict(self.observed_residual_worker_peak),
                measured_jacobian_worker_peak=None if self.observed_jacobian_worker_peak is None else asdict(self.observed_jacobian_worker_peak),
                active_evaluation_mode=self.active_evaluation_mode,mode_peaks_use_conservative_process_lifetime_counters=False,
                mode_peaks_use_current_command_samples=True,
                measured_mode_block_peaks={mode:{name:asdict(peak)for name,peak in samples.items()}for mode,samples in self.evaluation_worker_samples.items()},
                requested_concurrency=self.requested_workers,concurrency=len(self.active_names),resident_workers=len(self.pool),
                resident_blocks=[b.name for b in self.blocks if b.name in self.pool],active_inflight_estimate_bytes=self.active_inflight_bytes,
                minimal_pickle_measurements=self.payload_measurements,parent_actor_start_measurements=self.parent_start_measurements,
                queued_source_blocks=self.queued_blocks,queue_reason=self.queue_reason,
                failure=self.sample_failure,reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes,accuracy_accepted=False)
            self.runtime_report_path.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')),encoding='utf8')
        if self.sample_failure:raise MemoryError(self.sample_failure)
        return result
    def _mode_calibration(self,mode='evaluate',jacobian=False):
        if mode=='evaluate'and not jacobian:
            return 'residual',self.residual_peak_measured,self.observed_residual_worker_peak
        # Non-evaluation operations retain the original Jacobian budget.
        return 'jacobian',self.evaluation_peak_measured,self.observed_jacobian_worker_peak
    def _block_peak(self,block,mode='evaluate',jacobian=False):
        evaluation_mode,measured,representative=self._mode_calibration(mode,jacobian)
        # Once measured, each original block owns its actual mode peak.
        # The first mode sample is only the fallback for still unseen blocks.
        peak=self.evaluation_worker_samples[evaluation_mode].get(block.name)
        if peak is not None:return peak
        if measured:return representative
        return self.current_worker_samples.get(block.name)
    def _future_growth(self,mode='evaluate',jacobian=False,*,blocks=(),owned_snapshot=None):
        """Reserve future growth only for blocks about to execute."""
        from rinkspace.runtime_memory import sample_owned_tree,subtree_from_snapshot
        identities={}
        for block in blocks:
            manager=self.pool.get(block.name)
            if manager is not None:
                actor=manager._actors[0]
                identity=manager.identities.get(actor['process'].pid)
                if identity is not None:identities[block.name]=identity
        # Every budget check still uses one fresh OS snapshot. The dispatch
        # sample can be supplied directly; never use an old cached reading.
        if identities and owned_snapshot is None:
            owned_snapshot=sample_owned_tree(identities.values())
        phys=commit=0
        for block in blocks:
            manager=self.pool.get(block.name)
            if manager is None:continue
            actor=manager._actors[0];identity=manager.identities.get(actor['process'].pid)
            if identity is None:continue
            sample=subtree_from_snapshot(owned_snapshot,identity)
            if not sample.complete:raise MemoryError('Incomplete actor sample cannot admit parallel work')
            peak=self._block_peak(block,mode,jacobian)
            if peak is None:continue
            phys+=max(0,math.ceil(peak.rss_bytes*1.25)-sample.rss_bytes)
            commit+=max(0,math.ceil(peak.private_bytes*1.25)-sample.private_bytes)
            if mode=='evaluate'and jacobian and not self.evaluation_peak_measured:
                # Single first J grows from existing residual residency.
                # Preserve idle peers; reserve only this original block.
                extra=2*self._output_bytes(block);phys+=extra;commit+=extra
        return phys,commit
    def _admission(self,block,mode='evaluate',jacobian=False):
        from rinkspace.runtime_memory import choose_worker_count,read_system_memory,memory_pressure,WorkerPeak
        self.sample('queued_before_payload_measure_'+block.name)
        if memory_pressure(read_system_memory(),reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes):return False
        payload=self._measure_payload(block)
        self.sample('queued_before_admit_'+block.name);system=read_system_memory()
        output=self._output_bytes(block,output_jacobian=mode=='evaluate'and jacobian)
        # Spawn is serial and no evaluate command is in flight here. Idle
        # peers already reduce system availability; do not reserve their peaks.
        growth_phys=growth_commit=0
        parent=2*payload['payload_bytes_upper_estimate']
        if self.parent_start_measurements:
            parent=max(parent,max(v['parent_rss_growth_bytes']for v in self.parent_start_measurements.values()),
                max(v['parent_private_growth_bytes']for v in self.parent_start_measurements.values()))
        evaluation_mode,measured,unused=self._mode_calibration(mode,jacobian)
        peak=self._block_peak(block,mode,jacobian)
        if not measured:
            # Only the first original block calibrates a previously unseen
            # mode. Existing idle workers are retained when switching modes.
            pending=any(b.name in self.pool for b in self.blocks[:self.blocks.index(block)])
            physical=parent+self._residual_shared_bytes(block)+payload['payload_bytes_upper_estimate']
            committed=parent+output+payload['payload_bytes_upper_estimate']
            count=int(not pending and not memory_pressure(system,
                reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes,
                inflight_physical_bytes=physical,inflight_commit_bytes=committed))
            admission='one_mode_bootstrap_exact_shared_commit_and_residual_residency'
        else:
            # A worker RSS includes its touched shared output mapping. Count
            # that physical residency once via its measured peak. Windows
            # commits the entire SharedMemory section even for residual-only
            # commands, separately from child private bytes: reserve it once.
            count=choose_worker_count(1,peak,system=system,
                reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes,
                inflight_physical_bytes=parent,inflight_commit_bytes=parent+output)
            admission='measured_'+evaluation_mode+'_block_peak_exact_shared_commit_once'
        self._event(dict(phase='admit_'+block.name,admission=admission,allowed_additional_workers=count,
            evaluation_mode=evaluation_mode,evaluation_peak_measured=measured,
            output_bytes=output,parent_inflight_bytes=parent,minimal_pickle=payload,
            shared_commit_count=1,residual_full_J_physical_reserve=False,
            existing_worker_growth_physical_bytes=growth_phys,existing_worker_growth_commit_bytes=growth_commit,
            system=asdict(system),requested_workers=self.requested_workers))
        return bool(count)
    def _remove(self,name):
        if name in self.active_names:raise RuntimeError('Cannot release an active owned source actor')
        manager=self.pool.pop(name);self.last_used.pop(name,None);manager.close()
    def _ensure(self,block,protected,mode='evaluate',jacobian=False):
        if self.closed:raise RuntimeError('Source actors have closed')
        output_jacobian=mode=='evaluate'and bool(jacobian)
        if block.name in self.pool:
            manager=self.pool[block.name]
            if manager._actors[0]['spec'].get('output_jacobian',True)!=output_jacobian:
                from rinkspace.runtime_memory import read_system_memory,memory_pressure
                # The old section is already counted by the system. Reserve
                # the replacement once until the owned worker ACK releases it.
                new_bytes=self._output_bytes(block,output_jacobian)
                if memory_pressure(read_system_memory(),reserve_physical_bytes=self.reserve_bytes,
                    reserve_commit_bytes=self.reserve_bytes,inflight_physical_bytes=self._residual_shared_bytes(block),
                    inflight_commit_bytes=new_bytes):return False
                manager.resize_output(output_jacobian)
                self.sample('after_output_resize_'+block.name)
            return True
        if len(self.pool)<self.requested_workers and self._admission(block,mode,jacobian):
            manager=GuardedSourceActors((block,),task_root=self.task_root,native_threads=self.native_threads,
                q_columns=self.q_columns,memory_owner=self,output_jacobian=output_jacobian)
            # Register before start so waiting samples include this owned child.
            self.pool[block.name]=manager
            self._starting_parent=(block.name,self._parent_sample())
            try:manager.start();self.sample('after_actor_start_'+block.name)
            except Exception:
                self.pool.pop(block.name,None);manager.close();raise
            finally:self._starting_parent=None
            return True
        if protected:return False
        # No rows are omitted: only idle owned blocks may queue/release to make
        # room for the next original block. Affordable actors otherwise stay.
        candidates=sorted(self.pool,key=lambda n:self.last_used.get(n,-1))
        for name in candidates:
            self._remove(name)
            if self._ensure(block,{"retry_without_eviction"},mode,jacobian):return True
        return False
    def _caller_inflight_bytes(self,blocks,mode,jacobian=False):
        if mode=='evaluate':
            if jacobian:return 128*1024**2+2*sum(self._output_bytes(b)for b in blocks)
            # Trial-cost consumers copy normalized r, not J/CSR buffers.
            return 128*1024**2+sum(int(b.actor_output_capacity['rows'])*8 for b in blocks)
        if mode=='profile_update':
            return 128*1024**2+2*sum(int(b.actor_output_capacity['motion_rows'])*8 for b in blocks)
        # Review metadata and profile synchronization do not copy full CSR.
        return 128*1024**2+sum(int(b.actor_output_capacity['rows'])*8 for b in blocks)
    def _can_execute(self,blocks,mode='evaluate',jacobian=False,*,owned_snapshot=None):
        from rinkspace.runtime_memory import read_system_memory,memory_pressure
        _,measured,_=self._mode_calibration(mode,jacobian)
        if not measured and len(blocks)>1:return False
        if mode=='evaluate'and jacobian and len(blocks)>1 and any(b.name not in self.calibrated_blocks for b in blocks):return False
        gp,gc=self._future_growth(mode,jacobian,blocks=blocks,owned_snapshot=owned_snapshot);inflight=self._caller_inflight_bytes(blocks,mode,jacobian)
        return not memory_pressure(read_system_memory(),reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes,
            inflight_physical_bytes=gp+inflight,inflight_commit_bytes=gc+inflight)
    def _queue_one(self,block,mode,jacobian=False):
        """Release only our idle peers, then wait instead of losing a row."""
        if self.active_names:raise RuntimeError('Source queue requires no active commands')
        self.queued_blocks=[block.name];self.queue_reason='Waiting for measured physical/commit headroom for '+mode
        for name in tuple(self.pool):
            if name!=block.name:self._remove(name)
        while True:
            if self._ensure(block,{'queued_single'},mode,jacobian)and self._can_execute([block],mode,jacobian):
                self.queued_blocks=[];self.queue_reason=None;return
            # A single denied command cannot keep its own idle allocation
            # forever while waiting for the growth reserve it prevents.
            if block.name in self.pool:self._remove(block.name)
            self.sample('queued_single_'+block.name);time.sleep(.5)
    def _groups(self,mode='evaluate',jacobian=False):
        index=0
        while index<len(self.blocks):
            group=[]
            while index<len(self.blocks)and len(group)<self.requested_workers:
                block=self.blocks[index]
                if not self._ensure(block,{b.name for b in group},mode,jacobian):
                    if group:break
                    self._queue_one(block,mode,jacobian)
                if not self._can_execute(group+[block],mode,jacobian):
                    if group:break
                    self._queue_one(block,mode,jacobian)
                group.append(block);index+=1
                if not self._mode_calibration(mode,jacobian)[1]:break
            yield group
    def _run_group(self,blocks,mode,q=None,jacobian=False,held_by_name=None):
        """Recheck submission, splitting/queuing in original source order."""
        pending=list(blocks)
        while pending:
            batch=[]
            for block in pending:
                if not self._ensure(block,{b.name for b in batch},mode,jacobian):break
                if not self._can_execute(batch+[block],mode,jacobian):break
                batch.append(block)
            if not batch:
                self._queue_one(pending[0],mode,jacobian);batch=[pending[0]]
            submitted=yield from self._submit_group(batch,mode,q,jacobian,held_by_name)
            if submitted:pending=pending[len(batch):]
            else:
                self.queued_blocks=[b.name for b in pending];self.queue_reason='Submission headroom changed; original blocks requeued'
                self._event(dict(phase='queued_dispatch_recheck',mode=mode,jacobian=bool(jacobian),blocks=self.queued_blocks.copy()))
        self.queued_blocks=[];self.queue_reason=None
    def _submit_group(self,blocks,mode,q=None,jacobian=False,held_by_name=None):
        import numpy as np
        from scipy.sparse import csr_matrix
        dispatch_snapshot=self.sample('queued_dispatch_check_'+mode)
        if not self._can_execute(blocks,mode,jacobian,owned_snapshot=dispatch_snapshot):return False
        actors=[(self.pool[b.name],self.pool[b.name]._actors[0])for b in blocks]
        sent=received=0;self.active_names={b.name for b in blocks};self.active_inflight_bytes=self._caller_inflight_bytes(blocks,mode,jacobian)
        self.active_evaluation_mode=('jacobian'if jacobian else'residual')if mode=='evaluate'else None
        try:
            for manager,actor in actors:
                manager._busy=True;command=dict(mode=mode)
                if mode!='profiles':command['q']=np.asarray(q,float)
                if mode=='evaluate':command['jacobian']=bool(jacobian)
                elif mode=='profile_update':command['motion_only']=os.environ.get('RINKSPACE_PROFILE_MOTION_ONLY','0')=='1'
                elif mode=='profiles':
                    bundle=actor['block'].bundle;value=bundle.profile_weights
                    if value is not None:actor['arrays']['weights'][:]=value;bundle.profile_weights=actor['arrays']['weights']
                    command.update(present=value is not None,robust=bool(bundle.robust_phase))
                elif mode=='review':command['held_rows']=held_by_name[actor['block'].name]
                manager._send(actor,command);sent+=1
            for manager,actor in actors:
                received+=1;answer=manager._receive(actor);block=actor['block'];arrays=actor['arrays']
                self.use_counter+=1;self.last_used[block.name]=self.use_counter
                if mode=='evaluate':
                    self.sample('measured_'+self.active_evaluation_mode+'_'+block.name,evaluation_completed=(block.name,jacobian))
                    self.last_stats.append(dict(block=block.name,**answer))
                    if jacobian:
                        nnz=answer['nnz'];matrix=csr_matrix((arrays['data'][:nnz],arrays['indices'][:nnz],arrays['indptr']),
                            shape=(answer['rows'],answer['columns']),copy=False)
                        matrix.has_sorted_indices=answer['sorted_indices'];matrix.has_canonical_format=answer['canonical']
                        # Streaming normal retention keeps the returned r/J by
                        # reference. Detach before yield so a later shared-buffer
                        # write or queued idle eviction cannot alter GN replay.
                        # These caller copies are in the in-flight admission
                        # reserve, as is the residual-only copy below.
                        detached=(arrays['residual'].copy(),matrix.copy());del matrix
                        yield block,detached
                    else:
                        # A completed trial row can still be held by the
                        # caller while we wait for the next original block.
                        # Detach its small r buffer before any idle recovery
                        # may close/resize that actor's shared section.
                        yield block,arrays['residual'].copy()
                elif mode=='profile_update':
                    block.bundle.profile_weights=arrays['weights'];block.bundle.robust_phase=True
                    self.last_stats.append(dict(block=block.name,**answer))
                elif mode=='review':
                    records=answer.pop('records');self.last_stats.append(dict(block=block.name,record_count=len(records),**answer));yield block,records
        finally:
            for manager,actor in actors[received:sent]:
                try:manager._receive(actor)
                except RemoteSourceError:pass
                except Exception:
                    self.active_names.clear();self.close();raise
            for manager,actor in actors:manager._busy=False
            self.active_names.clear();self.active_inflight_bytes=0;self.active_evaluation_mode=None
        return True
    def ordered_evaluations(self,q,jacobian):
        if self._busy:raise RuntimeError('One source evaluation may be active at a time')
        self._busy=True;self.last_stats=[]
        try:
            if jacobian and not self.evaluation_peak_measured and self.pool:
                self._event(dict(phase='jacobian_single_mode_bootstrap',retained_residual_blocks=list(self.pool)))
            for group in self._groups('evaluate',jacobian):yield from self._run_group(group,'evaluate',q,jacobian)
        finally:self._busy=False
    def ordered_absolute_records(self,q,held_by_name):
        if self._busy:raise RuntimeError('One source evaluation may be active at a time')
        self._busy=True;self.last_stats=[]
        try:
            for group in self._groups('review',False):yield from self._run_group(group,'review',q,held_by_name=held_by_name)
        finally:self._busy=False
    def update_profiles(self,q):
        if self._busy:raise RuntimeError('Cannot update profiles during source evaluation')
        resident=[b for b in self.blocks if b.name in self.pool]
        remaining=[b for b in self.blocks if b.name not in self.pool]
        if remaining:
            from rinkspace.runtime_memory import choose_worker_count,read_system_memory
            gp,gc=self._future_growth('profile_update',blocks=resident);inflight=self._caller_inflight_bytes(remaining,'profile_update')
            count=choose_worker_count(min(self.requested_workers,len(remaining)),self.observed_jacobian_worker_peak if self.evaluation_peak_measured else None,
                system=read_system_memory(),reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes,
                inflight_physical_bytes=gp+inflight,inflight_commit_bytes=gc+inflight)
            self.sample('before_parent_profiles');make_parallel_profile_update(remaining,max(1,count))(q);self.sample('after_parent_profiles')
        # Original actor formula shares frozen weights back; sends precede
        # ordered receives. No actor/bundle is updated twice concurrently.
        self.last_stats=[]
        for i in range(0,len(resident),max(1,self.requested_workers)):
            group=resident[i:i+self.requested_workers]
            list(self._run_group(group,'profile_update',q))
    def synchronize_profiles(self):
        for manager in self.pool.values():manager.synchronize_profiles()
    def close_resident(self):
        for name in tuple(self.pool):self._remove(name)
    def close(self):
        self.active_names.clear();self.close_resident();self.closed=True
