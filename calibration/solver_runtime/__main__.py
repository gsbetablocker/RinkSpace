"""Light Windows-spawn entry for the reusable source-owned solver."""
from pathlib import Path
import json
import os
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(RINKSPACE_MOTION_BACKEND="native", RINKSPACE_MOTION_THREADS="3",
                  RINKSPACE_SCATTER_BACKEND="native", RINKSPACE_SCATTER_THREADS="3",
                  RINKSPACE_PROJECTION_BACKEND="native", RINKSPACE_PROJECTION_VALUE_BACKEND="opencv",
                  RINKSPACE_SOLVER_ACCELERATION="auto")
from . import install
BASE = install()
from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract
install_source_worker_backend_contract()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Use: python -m calibration.solver_runtime status|verify-backends|test-transport|named66|v12 [arguments]")
    command = sys.argv.pop(1)
    if command == "status":
        from . import status
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif command == "verify-backends":
        from . import configure_native_runtime
        print(json.dumps(configure_native_runtime(stage="owned_cli_backend_check"), ensure_ascii=False, indent=2))
    elif command == "test-transport":
        from structured_programme_source_candidate.test_bounded_mode_budget_small import main as test
        test()
    elif command == "named66":
        config_path = Path(__file__).resolve().parents[2] / 'configs/solver_runtime/okada_named66.json'
        if '--input-config' in sys.argv:
            index = sys.argv.index('--input-config')
            if index + 1 == len(sys.argv):
                raise SystemExit('--input-config requires an input configuration path')
            config_path = Path(sys.argv[index + 1])
            del sys.argv[index:index + 2]
        from .strict_model_scope_inputs import require_solver_scope_policy
        scope = require_solver_scope_policy(config_path, require_policy=True)
        if scope and scope.get('schema') == 'rinkspace.strict-finite-dataset-startup-policy.v2':
            for option, key in [('--strict-source-fit-dataset','fit_dataset'),('--strict-motion-inputs','motion_inputs')]:
                matches=[argument for argument in sys.argv if argument==option or argument.startswith(option+'=')]
                if matches != [option] or sys.argv.index(option)+1 == len(sys.argv):
                    raise SystemExit('Strict configured source route missing from actual command: '+option)
                if Path(sys.argv[sys.argv.index(option)+1]).resolve() != Path(scope[key]['path']).resolve():
                    raise SystemExit('Actual source route differs from authenticated configuration: '+option)
        from .source_cache_output_relocation import install_output_relocation_compatibility
        install_output_relocation_compatibility()
        # Keeping the complete driver out of module scope prevents each child
        # process from importing its geometry/audit tree during Windows spawn.
        import run_current_named66_full_programme as driver
        diagnostic_directory = os.environ.get('RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR')
        if not diagnostic_directory:
            for index, argument in enumerate(sys.argv):
                if argument == '--output' and index + 1 < len(sys.argv):
                    diagnostic_directory = sys.argv[index + 1] + '.bootstrap'
                elif argument.startswith('--output='):
                    diagnostic_directory = argument.partition('=')[2] + '.bootstrap'
        if diagnostic_directory:
            from .execution_acceleration import install_named66_execution
            receipt = install_named66_execution(driver, diagnostic_directory)
            print(json.dumps(dict(stage='complete_source_execution_scheduling_installed',
                                  receipt=receipt), ensure_ascii=False), flush=True)
        driver.main()
    elif command == "v12":
        from run_solver_fast import main as run
        run()
    else:
        raise SystemExit(f"Unknown solver runtime command: {command}")


if __name__ == "__main__":
    main()
