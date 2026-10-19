"""Tiny real transport regressions for mode-specific resident admission."""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import sys,os,json,time,hashlib
from dataclasses import replace
BASE=_runtime_data_root()
sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from isolated_solver_bootstrap import install
install()
import numpy as np
from structured_programme_source_candidate.bounded_source_actors import BoundedSourceActors
from structured_programme_source_candidate.bounded_actor_transport_fixture_factory import TinySourceBlock
import rinkspace.runtime_memory as memory

def evaluate(manager,blocks,q,jac):
    results=[]
    for b,result in manager.ordered_evaluations(q,jac):
        expected=b.evaluate(q,jac)
        if jac:
            assert np.array_equal(result[0],expected[0])
            for k in ('data','indices','indptr'):assert np.array_equal(getattr(result[1],k),getattr(expected[1],k))
        else:assert np.array_equal(result,expected)
        results.append(b.name)
    assert results==[b.name for b in blocks]

def pids(manager):
    return {name:m._actors[0]['process'].pid for name,m in manager.pool.items()}

def main():
    output=BASE/('bounded_actor_mode_budget_small_'+str(time.time_ns()));output.mkdir()
    q=np.array([.7,-1.2,2.5,.1]);blocks=[TinySourceBlock('fixture_'+str(i),i)for i in range(6)]
    manager=BoundedSourceActors(blocks,task_root=BASE,q_columns=4)
    manager.runtime_report_path=output/'memory_live.json'
    original_sample=memory.sample_owned_tree
    # Deliberately unrelated OS lifetime high-water must not budget this
    # residual command. Current process counters remain authentic.
    def inflated_lifetime(roots):
        value=original_sample(roots)
        return replace(value,processes=tuple(replace(p,peak_rss_bytes=max(p.peak_rss_bytes,2**40),
            peak_private_bytes=max(p.peak_private_bytes,2**40))for p in value.processes))
    report=dict(real_programme_evaluated=False)
    try:
        memory.sample_owned_tree=inflated_lifetime
        evaluate(manager,blocks,q,False)
        assert manager.residual_peak_measured and len(manager.pool)==6
        start=pids(manager)
        assert all(manager._output_bytes(b,output_jacobian=False)==manager.pool[b.name]._actors[0]['spec']['bytes']for b in blocks)
        assert manager.observed_worker_peak.rss_bytes>=2**40
        assert manager.observed_residual_worker_peak.rss_bytes<2**40
        evaluate(manager,blocks,q+.031,False);assert pids(manager)==start
        evaluate(manager,blocks,q,True);assert manager.evaluation_peak_measured and pids(manager)==start
        evaluate(manager,blocks,q-.017,False);assert pids(manager)==start
        os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='1';manager.profile_update(q)
        evaluate(manager,blocks,q,False);assert pids(manager)==start
        report.update(residual_first_six_resident=True,warm_residual_pid_reuse=True,
            residual_to_J_keeps_all_pids=True,J_to_residual_keeps_all_pids=True,
            unrelated_lifetime_highwater_ignored_by_mode_budgets=True,
            original_r_CSR_and_row_order_exact=True,actual_shared_layout_exact=True,
            profile_update_residual_pid_reuse=True,resident_pids=start)
    finally:
        memory.sample_owned_tree=original_sample;manager.close()
    # Original index-width threshold, including an otherwise small q chart.
    probe=BoundedSourceActors([TinySourceBlock('probe',0)],task_root=BASE,q_columns=2**31)
    try:
        c=probe.blocks[0].actor_output_capacity
        expected=8*(int(c['rows'])+2*int(c['nnz'])+int(c['rows'])+1+int(c['motion_rows']))
        assert probe._output_bytes(probe.blocks[0])==expected
    finally:probe.close()
    # Low memory must reject before creating any process; no indefinite queue
    # is entered by this direct admission check.
    actual=memory.read_system_memory();original_read=memory.read_system_memory
    for field in ('physical_available_bytes','commit_available_bytes'):
        memory.read_system_memory=lambda field=field:replace(actual,**{field:0})
        denied=BoundedSourceActors([TinySourceBlock('denied',0)],task_root=BASE,q_columns=4)
        try:assert not denied._admission(denied.blocks[0],'evaluate',False) and not denied.pool
        finally:denied.close();memory.read_system_memory=original_read
    report.update(int64_chart_threshold_exact=True,low_memory_denied_before_spawn=True,
        implementation_source_path=str(Path(sys.modules[BoundedSourceActors.__module__].__file__).resolve()),
        implementation_sha256=hashlib.sha256(Path(sys.modules[BoundedSourceActors.__module__].__file__).read_bytes()).hexdigest())
    (output/'small_mode_budget_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(output),**report)))
if __name__=='__main__':main()

