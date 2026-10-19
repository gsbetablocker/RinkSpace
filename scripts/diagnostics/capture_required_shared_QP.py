"""Capture the unchanged first required reduced QP, then unwind without solving."""
from pathlib import Path
import functools
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

class RequiredQPCaptured(BaseException):
    pass

def main():
    config=Path(sys.argv[1]).resolve()
    output=Path(sys.argv[2]).resolve()
    if not output.is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Diagnostic output belongs in the Data root')
    if output.exists():raise FileExistsError(output)
    cfg=json.loads(config.read_text(encoding='utf8'))
    os.environ['RINKSPACE_SOLVER_DATA_ROOT']=cfg['resource_root']
    os.environ['RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR']=str(output)+'.bootstrap'
    os.environ['RINKSPACE_BENCHMARK_TIMINGS_PATH']=str(output)+'.bootstrap/benchmark_timings_latest.json'
    import calibration.solver_runtime.__main__ as entry
    import z190_global_schur_linear_step as qp
    from structured_programme_source_candidate import feasibility_priority_restoration as priority
    import numpy as np
    original=qp._bounded_shared_step
    old_priority=priority.contract_original_support_bounds
    prior={}
    @functools.wraps(old_priority)
    def remember_priority(*args,**kwargs):
        result=old_priority(*args,**kwargs)
        prior.clear()
        for name,value in zip(('lower','upper','constraint_matrix','original_constraint_lower',
                               'row_group_index','current_group_violations'),args):
            prior[name]=np.asarray(value)
        prior['report']=result['report']
        return result
    @functools.wraps(original)
    def capture(matrix,gradient,lower,upper,constraint_matrix=None,constraint_lower=None):
        if constraint_matrix is None or not len(constraint_lower):
            return original(matrix,gradient,lower,upper,constraint_matrix,constraint_lower)
        destination=output/'required_reduced_QP'
        destination.mkdir(parents=True,exist_ok=False)
        arrays=dict(matrix=matrix,gradient=gradient,lower=lower,upper=upper,
                    constraint_matrix=constraint_matrix,constraint_lower=constraint_lower)
        arrays.update({'priority_'+key:value for key,value in prior.items() if key!='report'})
        path=destination/'original_problem.npz'
        np.savez(path,**arrays)
        receipt=dict(schema='rinkspace.original-required-reduced-QP-diagnostic-capture.v1',
            original_required_QP_calls=0,trial_calls=0,diagnostic_only=True,
            config_reference=dict(path=str(config),sha256=hashlib.sha256(config.read_bytes()).hexdigest()),
            reduced_chart='actual damped scaled shared coordinates passed to _bounded_shared_step',
            pid=os.getpid(),wall_time_ns=time.time_ns(),
            arrays={k:dict(shape=list(np.shape(v)),dtype=str(np.asarray(v).dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())for k,v in arrays.items()},
            array_path=str(path),array_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            priority_report=prior.get('report'),
            complete_source_J_or_normal_saved=False)
        (destination/'capture.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
        raise RequiredQPCaptured(str(destination/'capture.json'))
    priority.contract_original_support_bounds=remember_priority
    qp._bounded_shared_step=capture
    arguments=['capture_required_shared_QP',cfg['driver'],'--output',str(output)]
    for key,value in cfg['source_options'].items():
        if isinstance(value,bool):
            if value:arguments.append(key)
        else:arguments.extend((key,str(value)))
    sys.argv=arguments
    try:
        entry.main()
    except RequiredQPCaptured as signal:
        print(json.dumps(dict(stage='required_reduced_QP_capture_complete',receipt=str(signal),
                              original_required_QP_calls=0,trial_calls=0)),flush=True)
        return 0
    finally:
        priority.contract_original_support_bounds=old_priority
        qp._bounded_shared_step=original
    raise RuntimeError('No required reduced-QP boundary was reached')

if __name__=='__main__':raise SystemExit(main())
