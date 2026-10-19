"""Small source-input/serialization/AST checks; never construct a programme."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace

import numpy as np

from calibration.solver_runtime import named66_source_skeleton_adapter_runtime as adapter
from calibration.solver_runtime import source_state_skeleton_cache_runtime as skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-directory', type=Path, required=True)
    opt = parser.parse_args()
    opt.report_directory.mkdir(parents=True, exist_ok=True)
    checks = {}
    with tempfile.TemporaryDirectory(prefix='named66_skeleton_small_', dir=opt.report_directory) as temp:
        directory = Path(temp)
        source = directory/'source.json'
        source.write_text('{"source_UV":[1,2],"sigma":6}', encoding='utf8')
        seed = directory/'current_seed.npz'
        np.savez(seed, q=np.arange(12., dtype=float), damping=np.asarray(1e-6),
                 next_line_search_fraction=np.asarray(.5))
        controller = adapter.Named66SourceSkeletonAdapter(cache_directory=directory/'cache',
            report_directory=directory/'receipts', declared_source_roots=(source,),
            complete_static_input_closure_confirmed=True, current_seed_parts=lambda *_args, **_kw: None)
        request = SimpleNamespace(camera_seed=seed)
        with controller.capture_input_reads(request):
            assert json.loads(source.read_bytes())['sigma'] == 6
            with np.load(seed, allow_pickle=False) as archive:
                assert archive['q'].shape == (12,)
            (directory/'output.json').write_text('{}', encoding='utf8')
        reads = controller.last_input_reads
        assert reads.files == {source.resolve():hashlib.sha256(source.read_bytes()).hexdigest()}
        assert not reads.errors
        checks['actual_input_SHA_observed_current_seed_and_writes_excluded'] = True
        original = reads.files[source.resolve()]
        source.write_text('{"source_UV":[1,2],"sigma":7}', encoding='utf8')
        assert skeleton._sha(source) != original
        checks['actual_source_sigma_change_changes_SHA'] = True

        shared = np.arange(65537, dtype=np.float64).reshape(-1, 1)
        state = SimpleNamespace(q=np.arange(4.), source_actors=None,
            profile_weights=np.ones((2,2)), robust_phase=True,
            _facet_course_cache=('current', shared),
            _absolute_geometry_fd_stencil_cache={'current':shared},
            _derived_course_center_cache={'current':shared},
            _current_binding_cache=('current', shared), source_cameras={'current':shared},
            original_source_array=shared, aliased_original_source_array=shared)
        state.self = state
        payload = dict(state=state, source=shared)
        expected = io.BytesIO()
        skeleton._OriginalSkeletonPickler(expected, state).dump(payload)
        destination = directory/'streamed.pickle'
        with destination.open('xb') as stream:
            writer = skeleton._HashingOutput(stream)
            skeleton._OriginalSkeletonPickler(writer, state).dump(payload)
        assert destination.read_bytes() == expected.getvalue()
        assert writer.nbytes == len(expected.getvalue())
        assert writer.digest.hexdigest() == skeleton._sha(destination)
        restored = pickle.loads(destination.read_bytes())
        clone = restored['state']
        assert clone.q is None and clone.self is clone
        assert clone._facet_course_cache is None and clone._current_binding_cache is None
        assert clone._absolute_geometry_fd_stencil_cache == clone._derived_course_center_cache == clone.source_cameras == {}
        assert clone.profile_weights is None and clone.robust_phase is False
        assert clone.original_source_array is clone.aliased_original_source_array is restored['source']
        assert np.array_equal(restored['source'], shared)
        assert state._facet_course_cache[1] is shared and state.profile_weights is not None
        checks['large_protocol5_buffer_stream_byteexact_aliases_and_live_state_preserved'] = True
        checks['current_q_profiles_and_known_binding_derivatives_removed'] = True
        invalid = dict(q=np.asarray([np.nan]), damping=np.asarray(1e-6))
        try:
            adapter._validate_current_checkpoint_numbers(invalid)
        except ValueError:
            checks['current_numerical_error_remains_original_error'] = True
        else:
            raise AssertionError('Invalid current q was accepted')

        path = Path(adapter.__file__).parent/'legacy/run_current_named66_full_programme.py'
        tree = ast.parse(path.read_text(encoding='utf8'))
        namespace = {'current_seed_parts':lambda *_args, **_kw: None}
        adapter.attach_named66_source_skeleton_reuse(tree, namespace,
            cache_directory=directory/'astcache', report_directory=directory/'astreceipts',
            declared_source_roots=(source,), complete_static_input_closure_confirmed=True)
        compile(tree, str(path), 'exec')
        assert "capold = _rinkspace_final_skeleton_values['capold']" in ast.unparse(tree)
        checks['original_named66_AST_compile_and_capold_hit_binding'] = True
    result = dict(checks=checks, all_pass=all(checks.values()),
        programme_graph_or_solver_or_process_pool_started=False,
        production_pickle_capture_and_checkpoint_hit_not_yet_observed=True,
        helper_sha256={Path(module.__file__).name:skeleton._sha(module.__file__)
                       for module in (adapter, skeleton)})
    (opt.report_directory/'named66_source_skeleton_small_check.json').write_text(
        json.dumps(result, sort_keys=True, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
