"""Replay saved complete original phase-I arrays without source J or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    from calibration.solver_runtime.priority_feasibility_witness_runtime import install_priority_feasibility_witness_runtime
    receipt = install_priority_feasibility_witness_runtime(report_directory=args.output)
    with np.load(args.inputs, allow_pickle=False) as saved:
        names = ('lower', 'upper', 'constraint_matrix', 'original_constraint_lower',
                 'row_group_index', 'current_group_violations')
        arrays = {name: saved[name if name in saved.files else 'step_'+name].copy() for name in names}
        if 'matrix' in saved.files:
            raise ValueError('This replay requires the captured original matrix=None chart')
    before = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    lo, hi, A, b, ids, v = (arrays[name] for name in names)
    n = len(lo)
    finite_lo, finite_hi = np.flatnonzero(np.isfinite(lo)), np.flatnonzero(np.isfinite(hi))
    eye = np.eye(n)
    rows = np.vstack((A, eye[finite_lo], -eye[finite_hi]))
    rhs = np.r_[b, lo[finite_lo], -hi[finite_hi]]
    coeff = np.zeros(len(b)); owned = ids >= 0
    coeff[owned] = np.where(v[ids[owned]] > 1e-8, v[ids[owned]], 0.)
    expansion = np.r_[coeff, np.zeros(len(finite_lo)+len(finite_hi))]
    cn = np.max(np.abs(rows), axis=0)
    D = np.divide(1., cn, out=np.ones(n), where=cn > 0.)
    B = rows * D[None, :]
    norm = np.sqrt(np.einsum('ij,ij->i', B, B)+expansion*expansion)
    begun = time.perf_counter()
    fallback = source._verified_same_original_conic_phaseI
    observed = []
    def observe_fallback(*args, **kwargs):
        candidate = fallback(*args, **kwargs)
        observed.append(candidate)
        return candidate
    # Re-enter the unchanged OUTER domain contract and its original map-back
    # and every original-unit check. Reuse the recorded primary failure; no
    # repeat of the known failed HiGHS solve is needed to test its fallback.
    recorded_failure = SimpleNamespace(success=False, message='(HiGHS Status 0: Not Set); saved original failure, not rerun')
    with patch.object(source, 'linprog', return_value=recorded_failure), patch.object(
            source, '_verified_same_original_conic_phaseI', side_effect=observe_fallback):
        contracted = source.contract_original_support_bounds(lo, hi, A, b, ids, v)
    assert len(observed) == 1
    result = observed[0]
    elapsed = time.perf_counter()-begun
    direction = D * result.x[:-1]
    fraction = float(contracted['common_fraction'])
    assert fraction == .5*(1.+float(result.x[-1]))
    np.testing.assert_array_equal(contracted['matrix'], A)
    np.testing.assert_array_equal(contracted['lower'], b-fraction*coeff)
    slack = rows @ direction + expansion*fraction-rhs
    assert np.min(slack/norm) >= -1e-9
    assert all(hashlib.sha256(value.tobytes()).hexdigest() == before[name] for name, value in arrays.items())
    support = slack[:len(b)]
    groups = []
    for group in np.unique(ids[ids >= 0]):
        mask = ids == group
        groups.append(dict(group_index=int(group), original_violation=float(v[group]),
            original_face_count=int(mask.sum()), contracted_minimum_slack=float(support[mask].min()),
            contracted_normalized_minimum_slack=float((support[mask]/norm[:len(b)][mask]).min())))
    protected = expansion == 0.
    report = dict(schema='rinkspace.saved-original-phaseI-complete-primal-witness-replay.v1',
        inputs=str(args.inputs.resolve()), input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        original_support_rows=len(b), original_shared_columns=n, original_groups=len(v),
        total_support_and_coordinate_box_rows=len(rows), original_arrays_unchanged=True,
        original_HiGHS_rerun=False, full_J_or_production_fit_run=False,
        outer_original_contract_returned_complete_contracted_domain=True,
        returned_constraint_matrix_byte_exact=True,
        returned_targets_byte_exact_with_original_formula=True,
        elapsed_seconds=elapsed, witness_fraction=float(result.x[-1]), original_midpoint_fraction=fraction,
        original_unit_contracted_all_face_minimum_slack=float(slack.min()),
        original_unit_protected_face_minimum_slack=float(slack[protected].min()),
        original_normalized_contracted_all_face_minimum_slack=float((slack/norm).min()),
        original_normalized_protected_face_minimum_slack=float((slack[protected]/norm[protected]).min()),
        groups=groups, outer_original_contract_report=contracted['report'],
        original_backend_witness=result.original_phaseI_report, runtime_installation=receipt,
        source_QP_and_full_nonlinear_acceptance_still_required=True, accuracy_accepted=False)
    np.savez(args.output/'original_phaseI_complete_primal_witness.npz',
        direction=direction, coordinate=result.x, common_fraction=np.asarray(fraction),
        contracted_lower=contracted['lower'])
    (args.output/'original_phaseI_complete_primal_witness_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({key:report[key] for key in ('original_support_rows','original_shared_columns',
        'elapsed_seconds','witness_fraction','original_midpoint_fraction',
        'original_normalized_contracted_all_face_minimum_slack')}))


if __name__ == '__main__':
    main()
