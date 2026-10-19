"""Tiny original SHA/file-invalidations/AST checks; no programme graph."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
from calibration.solver_runtime import startup_source_auth_reuse_runtime as runtime
from calibration.solver_runtime.solver_control_runtime import SourceControlStop

DESTINATION = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/diagnostics/current_solver_takeover_20261009/startup_source_auth_small')


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    original_calls = []
    def original(path):
        original_calls.append(str(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    a, b = DESTINATION / 'source_A.bin', DESTINATION / 'source_B.bin'
    a.write_bytes(b'ab'); b.write_bytes(b'xy')
    with runtime.SourceAuthenticationScope(DESTINATION / 'outputs') as scope:
        expected = runtime._source_sha(a, original)
        assert runtime._source_sha(a, original) == expected
        runtime._source_sha(b, original)
    assert original_calls == [str(a), str(b), str(a)]
    assert scope.closed and not scope.files and runtime._ACTIVE.get() is None
    original_calls.clear()
    rejected = False
    before = a.stat()
    try:
        with runtime.SourceAuthenticationScope(DESTINATION / 'outputs') as scope:
            runtime._source_sha(a, original)
            runtime._source_sha(a, original)
            a.write_bytes(b'cd')
            os.utime(a, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert a.stat().st_size == before.st_size and a.stat().st_mtime_ns == before.st_mtime_ns
    except ValueError:
        rejected = True
    assert rejected and len(original_calls) == 2 and runtime._ACTIVE.get() is None and not scope.files
    control = SourceControlStop({'reason': 'explicit-test-stop'})
    try:
        with runtime.SourceAuthenticationScope(DESTINATION / 'outputs') as scope:
            runtime._source_sha(a, original)
            raise control
    except SourceControlStop as actual:
        assert actual is control and runtime._ACTIVE.get() is None and not scope.files
    output = DESTINATION / 'outputs' / 'mutable.bin'
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b'old')
    with runtime.SourceAuthenticationScope(output.parent) as scope:
        old = runtime._source_sha(output, original)
        output.write_bytes(b'new')
        assert runtime._source_sha(output, original) != old
        assert scope.calls == scope.hits == 0
    receipt = runtime.install_startup_source_auth_reuse_runtime()
    from structured_programme_source_candidate.human_stroke_truth import _sha
    with runtime.SourceAuthenticationScope(DESTINATION / 'outputs') as scope:
        assert _sha(a) == hashlib.sha256(a.read_bytes()).hexdigest()
        assert _sha(a) == hashlib.sha256(a.read_bytes()).hexdigest()
        assert scope.hits == 1
    # Compile the unchanged real main with the precise attach boundary.
    from calibration.solver_runtime.runtime import source_path
    tree = ast.parse(source_path('run_current_named66_full_programme').read_text(encoding='utf8'))
    namespace = {}
    changed = runtime.attach_startup_source_auth_reuse(tree, namespace, DESTINATION)
    compile(changed, '<actual-original-driver-AST-source-auth-check>', 'exec')
    report = dict(passed=True, original_SHA_duplicate_file_read_once_and_hit_only_exit_recheck=True,
        same_size_and_restored_mtime_source_content_change_rejected=True,
        original_control_BaseException_retained_and_cache_cleared=True,
        mutable_run_outputs_not_cached=True, real_registered_SHA_wrapper_equivalent=True,
        original_driver_AST_source_build_and_diagnostic_boundaries_compile=True,
        source_graph_or_solver_or_process_pool_not_constructed=True,
        installed_functions=len(receipt['original_SHA_functions']))
    (DESTINATION / 'startup_source_auth_small_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report))
    print(hashlib.sha256((ROOT / 'calibration/solver_runtime/startup_source_auth_reuse_runtime.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
