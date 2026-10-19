"""Read saved complete states and quantify existing non-image priors.

This audit never evaluates a new target, changes weights, or starts a solver.
It reports roll on the saved actual views and only attributes saved family
costs to a state when their exact complete-q hash agrees.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np


def audit_state(directory):
    directory = Path(directory).resolve()
    identity = json.loads((directory / 'state_identity.json').read_text(encoding='utf-8-sig'))
    state = directory / 'complete_state_UNACCEPTED.npz'
    with np.load(state, allow_pickle=False) as saved:
        q = saved['q'].copy()
        evaluation_layout = None
        if 'aux__current_complete_parameter_layout_json' in saved:
            evaluation_layout = json.loads(str(saved['aux__current_complete_parameter_layout_json'].item()))
            key = 'aux__current_complete_evaluation_to_canonical'
            if key not in saved:
                candidates = [name for name in saved.files if 'evaluation_to_canonical' in name]
                if len(candidates) != 1:
                    raise ValueError('Saved evaluation layout requires one explicit permutation')
                key = candidates[0]
            permutation = np.asarray(saved[key], dtype=int)
            if (permutation.shape != q.shape or
                    not np.array_equal(np.sort(permutation), np.arange(len(q)))):
                raise ValueError('Saved complete evaluation permutation is not bijective')
            # Auxiliary prior indices describe the evaluation chart; q itself
            # deliberately preserves the original complete canonical prefix.
            q = q[permutation]
        starts = saved['aux__block_start'].copy()
        lengths = saved['aux__block_lengths'].copy()
        names = saved['aux__block_names'].copy()
        roll = json.loads(str(saved['aux__correlated_roll_prior_contract_json'].item()))
    sigma = float(roll['component_sigma_radians'])
    bias = q[np.asarray(roll['statistical_bias_parameter_indices'], int)]
    rows = []
    for name, start, length in zip(names, starts, lengths):
        name = str(name)
        installation = 1 if name == 'secondary' else 2 if name == 'low' else 0
        if int(length) % 4:
            raise ValueError('Saved actual view block is not a four-coordinate camera chart')
        rotations = q[int(start):int(start + length)].reshape(-1, 4)[:, :3]
        theta = []
        for vector in rotations:
            rotation = cv2.Rodrigues(vector)[0]
            x, y = rotation[0, 2], -rotation[1, 2]
            if x*x + y*y < 1e-8:
                raise ValueError('Existing physical roll domain is undefined')
            theta.append(np.arctan2(x, y))
        delta = np.asarray(theta) - bias[installation]
        degrees = np.rad2deg(delta)
        rows.append(dict(
            block=name, actual_views=len(rotations), installation=installation,
            installation_bias_degrees=float(np.rad2deg(bias[installation])),
            roll_delta_rms_degrees=float(np.sqrt(np.mean(degrees**2))),
            roll_delta_range_degrees=[float(degrees.min()), float(degrees.max())],
            roll_engineering_prior_cost=float(.5*np.sum((delta/sigma)**2))))
    run = directory.parent.parent
    exact = []
    for path in sorted(run.glob('actual_trial_family_cost_pass_*.json')):
        receipt = json.loads(path.read_text(encoding='utf-8-sig'))
        if receipt['complete_q_sha256'] != identity['q_sha256']:
            continue
        exact.append(dict(
            path=str(path), frozen_profile_token=receipt['frozen_profile_token'],
            source_manifest_sha256=receipt['source_manifest_sha256'],
            families=[dict(family=block['original_row_ledger']['family'],
                           rows=block['scalar_rows'], normalized_cost=block['normalized_cost'])
                      for block in receipt['families']]))
    return dict(
        directory=str(directory), q_sha256=identity['q_sha256'],
        complete_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        saved_auxiliary_indices_evaluated_in_explicit_chart=True,
        evaluation_layout=evaluation_layout,
        programme_roll_prior_cost=sum(row['roll_engineering_prior_cost'] for row in rows),
        hardware_bias_prior_cost=float(.5*np.sum((bias/sigma)**2)),
        roll_component_sigma_degrees=float(np.rad2deg(sigma)),
        actual_views=sum(row['actual_views'] for row in rows), programme_blocks=rows,
        external_rolls_excluded_from_this_direct_view_total=roll['additional_active_external_rolls'],
        exact_q_saved_family_cost_receipts=exact,
        saved_profile_receipt_is_reported_separately_from_q=True,
        engineering_prior_cost_is_not_source_alignment=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-directory', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        schema='rinkspace.saved-complete-state-engineering-prior-audit.v1',
        operation='read-only saved q and saved cost receipts; no source or solver evaluation',
        target_or_weights_changed=False,
        cost_comparison_requires_identical_target_manifest=True,
        states=[audit_state(path) for path in args.state_directory],
        interpretation='Costs quantify existing hypotheses; they do not prove that a prior is appropriate or caused a source conflict.',
        accuracy_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), states=[dict(
        q=state['q_sha256'], actual_views=state['actual_views'],
        roll_cost=state['programme_roll_prior_cost'],
        exact_cost_receipts=len(state['exact_q_saved_family_cost_receipts']))
        for state in result['states']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
