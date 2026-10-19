"""Run the new execution scheduling against an existing complete baseline.

The previous run is evidence, not rerun here. No source rows, physical limits,
solver trial schedule or original candidate acceptance are replaced.
"""
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
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    config = json.loads(options.config.read_text(encoding='utf8'))
    output = options.output.resolve()
    if output.exists():
        raise ValueError('Keep existing run evidence; choose a new output directory')
    boot = Path(str(output) + '.bootstrap')
    boot.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['RINKSPACE_SOLVER_DATA_ROOT'] = config['resource_root']
    os.environ['RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR'] = str(boot)
    os.environ['RINKSPACE_BENCHMARK_TIMINGS_PATH'] = str(boot / 'benchmark_timings_latest.json')

    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.source_cache_output_relocation import install_output_relocation_compatibility
    install_output_relocation_compatibility()
    import run_current_named66_full_programme as driver
    from calibration.solver_runtime.execution_acceleration import install_named66_execution
    receipts = dict(install_named66_execution(driver, boot))
    receipts.update(config=str(options.config.resolve()),
        config_sha256=hashlib.sha256(options.config.read_bytes()).hexdigest(),
        output=str(output), registered_source_files_unchanged=True,
        previous_complete_baseline_rerun=False,
        execution_overrides={'--residual-source-workers': '6', '--residual-native-threads': '3'})
    (boot / 'execution_candidate_receipt.json').write_text(
        json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

    source_options = dict(config['source_options'])
    source_options.update(receipts['execution_overrides'])
    arguments = ['named66', '--output', str(output)]
    for key, value in source_options.items():
        if isinstance(value, bool):
            if value:
                arguments.append(key)
        else:
            arguments.extend((key, str(value)))
    sys.argv = arguments
    began = time.perf_counter()
    print(json.dumps(dict(stage='new_execution_candidate_started', **receipts)), flush=True)
    try:
        driver.main()
    finally:
        print(json.dumps(dict(stage='new_execution_candidate_finished',
            wall_seconds=time.perf_counter() - began)), flush=True)


if __name__ == '__main__':
    main()
