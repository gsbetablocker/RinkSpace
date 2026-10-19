"""Original three-wrapper bound context check; no event graph or process jobs."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
import numpy as np
from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements
from structured_programme_source_candidate.persistent_ice_long_baseline_source import SourceWithPersistentIceLongBaseline
from structured_programme_source_candidate.static_motion_source_role_retirement import SourceWithStaticMotionRoleRetirement
from structured_programme_source_candidate.native_board_track_source import SourceWithNativeBoardTracks


def main():
    placement = SourceWithIndependentPatternPlacements.__new__(SourceWithIndependentPatternPlacements)
    placement.count, placement.placement_count, placement.placement_indices = 5, 2, np.array([3, 4])
    placement.patterns = SimpleNamespace(_independent_pair_offsets=np.array([.1, .2, 0., 0.]))
    placement.material_factor = SimpleNamespace(_cache=OrderedDict())
    source = placement
    for cls in (SourceWithPersistentIceLongBaseline, SourceWithStaticMotionRoleRetirement, SourceWithNativeBoardTracks):
        wrapped = cls.__new__(cls)
        wrapped.base, wrapped.count = source, source.count
        source = wrapped
    method = source.pattern_placement_context
    actual_owner = getattr(method, '__self__', None)
    assert type(actual_owner) is SourceWithIndependentPatternPlacements
    assert getattr(method, '__func__', None) is SourceWithIndependentPatternPlacements.pattern_placement_context
    assert actual_owner is placement
    assert actual_owner.patterns is source.patterns
    assert actual_owner.material_factor is source.material_factor
    q = np.array([1., 2., 3., .4, .5])
    before = placement.patterns._independent_pair_offsets.copy()
    with method(q):
        delegated = placement.patterns._independent_pair_offsets.copy()
    assert np.array_equal(placement.patterns._independent_pair_offsets, before)
    with actual_owner.pattern_placement_context(q):
        assert delegated.tobytes() == placement.patterns._independent_pair_offsets.tobytes()
    assert np.array_equal(placement.patterns._independent_pair_offsets, before)
    failed = False
    try:
        with actual_owner.pattern_placement_context(q):
            raise ValueError('test-original-context-exit')
    except ValueError:
        failed = True
    assert failed and np.array_equal(placement.patterns._independent_pair_offsets, before)
    report = dict(passed=True, original_wrapper_chain=[cls.__name__ for cls in (
        SourceWithNativeBoardTracks, SourceWithStaticMotionRoleRetirement, SourceWithPersistentIceLongBaseline)],
        original_bound_context_function_and_typed_self_exact=True,
        delegated_and_direct_original_offsets_byte_equal=True,
        original_offsets_restore_on_success_and_error=True,
        live_wrappers_and_worker_source_helpers_not_modified=True,
        production_full_context_still_to_be_observed=True)
    destination = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/diagnostics/current_solver_takeover_20261009')
    (destination / 'confirmed_columns_delegated_placement_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
