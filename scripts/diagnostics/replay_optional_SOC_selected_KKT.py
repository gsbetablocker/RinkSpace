"""Replay one authenticated reduced SOC QP; no source graph or J evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding='utf8'))
    if capture['schema'] != 'rinkspace.original-optional-SOC-reduced-QP-capture.v1':
        raise ValueError('Original reduced SOC capture required')
    array_path = Path(capture['array_path'])
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != capture['array_file_sha256']:
        raise ValueError('Original reduced QP archive changed')
    os.environ.pop('RINKSPACE_OPTIONAL_SOC_CAPTURE_DIR', None)
    warm = capture.get('warm_faces')
    if warm and warm.get('scope'):
        os.environ['RINKSPACE_SOLVER_DATA_ROOT'] = warm['scope']
    import numpy as np
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.optional_soc_runtime import (
        install_optional_SOC_numerical_failure_route, OptionalSOCNumericalFailure,
        _optional_soc, _digest)
    installation = install_optional_SOC_numerical_failure_route()
    import z190_global_schur_linear_step as qp
    with np.load(array_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in capture['arrays']}
    for key, value in arrays.items():
        identity = capture['arrays'][key]
        if list(value.shape) != identity['shape'] or str(value.dtype) != identity['dtype'] or _digest(value) != identity['sha256']:
            raise ValueError('Original array identity changed: ' + key)
    if warm:
        warm = dict(warm, shape=tuple(warm['shape']), rows=tuple(warm['rows']))
        qp._linearly_constrained_shared_step._last_verified_separation_faces = warm
    context = {key: capture[key] for key in ('original_q_sha256',
        'previous_direction_sha256', 'strictly_tighter_measured_SOC_faces')}
    token = _optional_soc.set(context)
    begun = time.perf_counter()
    try:
        value, held, result = qp._linearly_constrained_shared_step(**arrays)
        outcome = dict(optional_correction_KKT_verified=True, complete_QP_report=result)
    except OptionalSOCNumericalFailure as error:
        outcome = dict(optional_correction_KKT_verified=False,
                       retained_previous_complete_KKT_direction=True,
                       failure=error.optional_soc_report)
    finally:
        _optional_soc.reset(token)
    report = dict(schema='rinkspace.authenticated-optional-SOC-selected-KKT-replay.v1',
        capture=str(args.capture.resolve()), capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        original_q_sha256=capture['original_q_sha256'], seconds=time.perf_counter()-begun,
        source_graph_constructed=False, source_J_recomputed=False,
        original_arrays_byte_identity_verified=True, installation=installation, **outcome)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({key: report[key] for key in ('original_q_sha256','seconds',
        'optional_correction_KKT_verified','source_graph_constructed')}), flush=True)


if __name__ == '__main__':
    main()
