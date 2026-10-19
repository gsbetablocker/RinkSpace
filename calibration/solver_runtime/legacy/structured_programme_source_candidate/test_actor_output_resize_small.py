"""Six real Windows source processes; tiny transport fixture, no programme.

This exercises owned shared-section replacement and the original r/CSR and
profile formulas. It is not evidence of rink accuracy or programme speed.
"""
from calibration.solver_runtime.config import SOURCE_ROOT,data_root
from pathlib import Path
import hashlib,json,os,sys,time
from multiprocessing import shared_memory
sys.path.insert(0,str(SOURCE_ROOT));sys.dont_write_bytecode=True
from isolated_solver_bootstrap import install
install()
import numpy as np
from structured_programme_source_candidate.bounded_actor_transport_fixture_factory import TinySourceBlock
from z190_process_source_evaluation import PersistentSourceActors,_output_memory_spec
from z190_joint_source_solver import _frozen_profile_token


def _evaluate(manager,q,jacobian):
    names=[]
    for block,value in manager.ordered_evaluations(q,jacobian):
        expected=block.evaluate(q,jacobian)
        if jacobian:
            assert np.array_equal(value[0],expected[0])
            assert value[1].shape==expected[1].shape
            for key in('data','indices','indptr'):
                assert np.array_equal(getattr(value[1],key),getattr(expected[1],key))
        else:assert np.array_equal(value,expected)
        names.append(block.name)
    assert names==[block.name for block in manager.blocks]


def _state(manager):
    return {actor['block'].name:dict(pid=actor['process'].pid,name=actor['spec']['name'],
        bytes=actor['spec']['bytes'],memory=actor['memory'],arrays=actor['arrays'])for actor in manager._actors}


def _released(state):
    for value in state.values():
        assert value['memory']._mmap is None and not value['arrays']
        try:probe=shared_memory.SharedMemory(name=value['name'])
        except FileNotFoundError:continue
        else:
            probe.close();raise AssertionError('Old owned output section remains open: '+value['name'])


def _profile_links(manager):
    for actor in manager._actors:
        value=actor['block'].bundle.profile_weights
        assert value is actor['arrays']['weights']
        assert np.shares_memory(value,actor['arrays']['weights'])


def main():
    if sys.platform!='win32':raise RuntimeError('This shared-section release fixture requires Windows')
    output=data_root()/('actor_output_resize_small_'+str(time.time_ns()));output.mkdir()
    q=np.array([.7,-1.2,2.5,.1]);blocks=[TinySourceBlock('resize_fixture_'+str(i),i)for i in range(6)]
    manager=PersistentSourceActors(blocks,task_root=SOURCE_ROOT,q_columns=4,output_jacobian=False)
    old_motion_only=os.environ.get('RINKSPACE_PROFILE_MOTION_ONLY')
    report=dict(real_programme_evaluated=False,real_windows_owned_processes=6)
    try:
        manager.start();start=_state(manager);pids={name:value['pid']for name,value in start.items()}
        for actor in manager._actors:
            spec=actor['spec'];capacity=actor['block'].actor_output_capacity
            assert not spec['output_jacobian'] and spec['bytes']==8*(capacity['rows']+capacity['motion_rows'])
            assert (spec['row_capacity'],spec['nnz_capacity'])==(capacity['rows'],capacity['nnz'])
            assert all(actor['arrays'][key].size==0 for key in('data','indices','indptr'))
        _evaluate(manager,q,False);_evaluate(manager,q+.031,False)
        manager.resize_output(False);assert _state(manager)==start
        try:list(manager.ordered_evaluations(q,True))
        except RuntimeError as exc:assert 'resize_output(True)'in str(exc)
        else:raise AssertionError('Missing Jacobian storage was not rejected before evaluate')
        active=manager.ordered_evaluations(q,False);next(active)
        try:
            try:manager.resize_output(True)
            except RuntimeError as exc:assert 'active owned source actor'in str(exc)
            else:raise AssertionError('Active output resize was not rejected')
        finally:active.close()
        assert _state(manager)==start
        os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='1';manager.update_profiles(q)
        _profile_links(manager);_evaluate(manager,q,False)
        token=_frozen_profile_token(manager.profile_update)
        profiles=manager.profile_update.checkpoint_arrays()
        manager.resize_output(True);full=_state(manager);_released(start)
        assert {name:value['pid']for name,value in full.items()}==pids
        assert len(manager.last_resize_stats)==6
        assert _frozen_profile_token(manager.profile_update)==token
        _profile_links(manager);_evaluate(manager,q,True);_evaluate(manager,q-.017,False)
        assert all(actor['spec']['bytes']==_output_memory_spec(actor['block'],4,True)['bytes']for actor in manager._actors)
        manager.resize_output(False);residual=_state(manager);_released(full)
        assert {name:value['pid']for name,value in residual.items()}==pids
        assert _frozen_profile_token(manager.profile_update)==token
        _profile_links(manager);_evaluate(manager,q,False)
        manager.update_profiles(q+.017);_profile_links(manager);_evaluate(manager,q,False)
        # A worker's frozen profile is independent of its transport mapping.
        # An unsynchronized parent mutation must remain unsynchronized even
        # when the output is resized; explicit synchronization then applies it.
        baseline={block.name:value.copy()for block,value in manager.ordered_evaluations(q,False)}
        for block in blocks:block.bundle.profile_weights[:]*=.5
        previous=_state(manager);manager.resize_output(True);_released(previous)
        for block,value in manager.ordered_evaluations(q,False):assert np.array_equal(value,baseline[block.name])
        manager.synchronize_profiles();_profile_links(manager);_evaluate(manager,q,True)
        manager.profile_update.restore_checkpoint(profiles);_profile_links(manager)
        assert _frozen_profile_token(manager.profile_update)==token
        _evaluate(manager,q,True)
        previous=_state(manager);manager.resize_output(False);_released(previous)
        final=_state(manager);assert {name:value['pid']for name,value in final.items()}==pids
        _evaluate(manager,q,False)
        report.update(residual_first_shared_bytes=sum(value['bytes']for value in start.values()),
            full_J_shared_bytes=sum(value['bytes']for value in full.values()),
            residual_final_shared_bytes=sum(value['bytes']for value in final.values()),resident_pids=pids,
            original_r_CSR_and_row_order_exact=True,residual_to_J_to_residual_same_pids=True,
            profile_update_and_checkpoint_exact=True,parent_profiles_link_current_shared_storage=True,
            worker_frozen_profiles_remain_independent=True,old_shared_sections_released=True,
            active_resize_rejected_before_allocation=True,same_mode_resize_noop=True,
            J_requires_explicit_capacity=True)
    finally:
        manager.close()
        if old_motion_only is None:os.environ.pop('RINKSPACE_PROFILE_MOTION_ONLY',None)
        else:os.environ['RINKSPACE_PROFILE_MOTION_ONLY']=old_motion_only
    _released(final)
    # The existing default still allocates the complete original CSR layout.
    default=PersistentSourceActors([TinySourceBlock('default_output',0)],task_root=SOURCE_ROOT,q_columns=4)
    try:
        default.start();assert default.output_jacobian and default._actors[0]['spec']['output_jacobian']
        _evaluate(default,q,True)
    finally:default.close()
    wide=_output_memory_spec(TinySourceBlock('wide',0),2**31,False)
    assert wide['arrays']['indices']['dtype']=='<i8' and wide['bytes']==80
    implementation=Path(sys.modules[PersistentSourceActors.__module__].__file__).resolve()
    report.update(default_J_interface_compatible=True,int64_capacity_threshold_preserved=True,
        implementation_source_path=str(implementation),implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest())
    (output/'small_output_resize_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(output),**report)))


if __name__=='__main__':main()
