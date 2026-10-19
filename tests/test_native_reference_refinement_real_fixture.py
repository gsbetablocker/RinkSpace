"""Independent production-wrapper replay of the sealed real source failure."""
import json
import os
from pathlib import Path

import numpy as np
import pytest


CAPTURE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/'
    'okada_strict_v6_828b_reference_v9_20261013/solver/failed_native_reference/'
    '828b09ae9c5b32a5_1791866513534280600')


@pytest.mark.skipif(not (CAPTURE/'capture_complete.json').is_file(), reason='Sealed real failure fixture is unavailable')
def test_production_wrapper_independently_refines_real_equation_and_passes_original_all_source_gate():
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.failed_native_reference_capture import (
        load_failed_native_reference, iter_failed_native_reference_blocks)
    from calibration.solver_runtime.native_reference_refinement import refine_undamped_native_reference
    from z190_streaming_source_normal import SourceBlockContract, BlockSignature, replay_native_step_maxima

    metadata, numeric = load_failed_native_reference(CAPTURE)
    saved = metadata['source_contract']
    contract = SourceBlockContract(saved['parameter_count'], saved['row_count'],
        tuple(BlockSignature(**record) for record in saved['signatures']), saved['frozen_profile_token'])
    def replay(steps):
        return replay_native_step_maxima(lambda: iter_failed_native_reference_blocks(CAPTURE),
            steps, contract, frozen_profile_token=contract.frozen_profile_token)
    original = replay([numeric['gn'], numeric['reference_step']-numeric['gn']])
    assert original[1] == metadata['reference_error_native_px'] == .006172500681189456
    assert metadata['derivative_model'] == 'fresh_source_J'
    original_bytes = {name: numeric[name].tobytes() for name in ('gradient','diagonal','gn','reference_step')}
    normal_bytes = tuple(array.tobytes() for array in (numeric['normal'].data,
                                                      numeric['normal'].indices,numeric['normal'].indptr))
    gn, reference, maximum_gn, error, report = refine_undamped_native_reference(
        numeric['normal'], numeric['gradient'], numeric['diagonal'], numeric['gn'], numeric['reference_step'],
        numeric['shared_parameter_indices'], numeric['fixed_parameter_indices'],
        native_replay=replay, initial_native_error=float(original[1]))
    assert error <= .001 and error < 5e-7
    assert maximum_gn > 0 and np.isfinite(gn).all() and np.isfinite(reference).all()
    fixed = numeric['fixed_parameter_indices']
    assert len(fixed)==5 and np.array_equal(gn[fixed],np.zeros(5)) and np.array_equal(reference[fixed],np.zeros(5))
    assert report['original_coordinate_system'] and not report['reversible_equilibration_used']
    full, schur = report['independent_full_sparse_LU'], report['independent_global_Schur']
    assert full['equation_identity'] == schur['equation_identity']
    assert full['rhs_identity'] == schur['rhs_identity']
    assert full['accepted_corrections'] == schur['accepted_corrections'] == 1
    assert all(numeric[name].tobytes()==data for name,data in original_bytes.items())
    assert normal_bytes == tuple(array.tobytes() for array in (numeric['normal'].data,
                                                             numeric['normal'].indices,numeric['normal'].indptr))
    assert metadata['original_source_blocks']==23 and metadata['original_source_rows']==2572778
    result = dict(production_wrapper_real_fixture_passed=True,
        original_native_error_px=float(original[1]),refined_native_error_px=error,
        complete_original_source_blocks=23,complete_original_source_rows=2572778,
        exact_fixed_zero_columns=5,report=report)
    output = os.environ.get('RINKSPACE_REAL_REFINEMENT_REPORT')
    if output:
        Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key!='report'}),flush=True)
