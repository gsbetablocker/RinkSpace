"""Replay a saved real failure and test independent same-equation refinement."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

# This diagnostic owns its separate process; configure its numerical imports
# without installing another package or changing the solver's environment.
for _thread_setting in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_thread_setting] = '1'

import numpy as np
from scipy.sparse import diags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.failed_native_reference_capture import (
        load_failed_native_reference, iter_failed_native_reference_blocks)
    from calibration.solver_runtime.normal_equation_refinement_candidate import (
        SameEquationLU, SameEquationSchur, refine_same_equation,
        original_equation_error, system_identity)
    from z190_streaming_source_normal import (
        SourceBlockContract, BlockSignature, replay_native_step_maxima)

    metadata, numeric = load_failed_native_reference(args.capture)
    if float(numeric['numerical_ridge_factor']) != 1e-8 or metadata['original_threshold_native_px'] != 1e-3:
        raise ValueError('Original explicit ridge and native-pixel gate required')
    normal, gradient = numeric['normal'], numeric['gradient']
    matrix = normal + diags(1e-8 * numeric['diagonal'], format='csc')
    rhs = -gradient
    steps = [numeric['gn'], numeric['reference_step'] - numeric['gn']]
    labels = ['saved_original_GN_maximum_native_step', 'saved_original_reference_error']
    records = []
    started = time.perf_counter()
    with nullcontext():
        for equilibrate in (False, True):
            beginning = time.perf_counter()
            lu = SameEquationLU(matrix, equilibrate=equilibrate)
            schur = SameEquationSchur(matrix, numeric['shared_parameter_indices'], equilibrate=equilibrate)
            gn, gn_report = refine_same_equation(matrix, rhs, numeric['gn'], lu.solve)
            reference, reference_report = refine_same_equation(matrix, rhs, numeric['reference_step'], schur.solve)
            label = 'equilibrated' if equilibrate else 'original_coordinates'
            gate_position = len(steps)
            steps.extend((reference - gn, gn - numeric['gn'], reference - numeric['reference_step']))
            labels.extend((label + '_reference_error', label + '_GN_change', label + '_reference_change'))
            records.append(dict(variant=label, equilibrate=equilibrate,
                gate_position=gate_position, independent_GN_and_Schur_factors=True,
                GN=gn_report, Schur=reference_report,
                GN_original_equation_error=original_equation_error(matrix, rhs, gn),
                Schur_original_equation_error=original_equation_error(matrix, rhs, reference),
                reduced_maximum_asymmetry=schur.reduced_maximum_asymmetry,
                seconds=time.perf_counter()-beginning))
            del lu, schur
        saved = metadata['source_contract']
        contract = SourceBlockContract(saved['parameter_count'], saved['row_count'],
            tuple(BlockSignature(**record) for record in saved['signatures']), saved['frozen_profile_token'])
        maxima = replay_native_step_maxima(lambda: iter_failed_native_reference_blocks(args.capture),
            np.stack(steps), contract, frozen_profile_token=contract.frozen_profile_token)
    if not np.isclose(maxima[1], metadata['reference_error_native_px'], rtol=0., atol=1e-12):
        raise ValueError('Saved original complete native-pixel failure did not replay exactly')
    for record in records:
        position = record.pop('gate_position')
        record['original_complete_source_reference_error_native_px'] = float(maxima[position])
        record['passes_unchanged_original_0_001px_gate'] = bool(maxima[position] <= 1e-3)
        record['GN_change_native_px'] = float(maxima[position+1])
        record['reference_change_native_px'] = float(maxima[position+2])
    report = dict(schema='rinkspace.real-failed-native-reference-independent-refinement-probe.v1',
        capture_path=str(args.capture.resolve()), q_sha256=metadata['q_sha256'],
        derivative_model=metadata.get('derivative_model'), equation_identity=system_identity(matrix),
        original_ridge_factor=1e-8, original_native_pixel_threshold=1e-3,
        saved_failure_exactly_reproduced=True, native_maxima=dict(zip(labels, map(float, maxima))),
        candidates=records, seconds=time.perf_counter()-started,
        original_full_H_gradient_J_scales_and_source_operator_unchanged=True,
        prior_or_source_rows_added=0, production_solver_modified_or_installed=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(report_path=str(args.report.resolve()),
        saved_failure_native_px=float(maxima[1]),
        candidates=[dict(variant=r['variant'], error_native_px=r['original_complete_source_reference_error_native_px'],
            passed=r['passes_unchanged_original_0_001px_gate'],
            GN_corrections=r['GN']['accepted_corrections'],
            Schur_corrections=r['Schur']['accepted_corrections']) for r in records]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
