"""Same-q production wide loss audit, without fitting or source image copies.

Exercise the actual 5,808-row wide producer and its original integer expansion.
All parameters stay at q8.  Only the optional integration payload changes.
Independent unique-row arithmetic checks the expanded robust cost and normal.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import argparse
import json
import time

import joint_absolute_source_context_20261011 as ctx
from rinkspace.storage import AssetStore
from rinkspace.path_policy import guard_project_write

a = ctx.a
np = ctx.np
POLICY = Path(__file__).resolve().parents[2] / (
    'events/ankara_2026/observations/source_packets/diagnostics/'
    'current56_shared_active_set_filter_20261009/systematic_continuation_20261010/'
    'original_WIDE45_current_physical_curve_isolated_anchor_POLICY_OPTIONAL_20261011.json')
POLICY = AssetStore(Path(__file__).resolve().parents[2]).resolve(POLICY)
OUTPUT = a.H / 'optional_wide45_same_q8_actual_stream_audit_20261011.json'


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=a.sha(path))


def numeric_sha(value):
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((arr.shape, arr.dtype.str)).encode('ascii'))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    output = guard_project_write(parser.parse_args().output)
    if output.exists():
        raise FileExistsError(output)
    from scipy.sparse import csr_matrix, hstack
    from rinkspace.source_sampling import compile_curve_sampling
    from calibration.solver_runtime.source_loss import (
        SourceLossWeights, source_irls_weights, weighted_source_cost,
        source_loss_checkpoint_arrays, source_loss_from_checkpoint)
    from calibration.solver_runtime.original_wide_curve_integration import (
        wrap_original_wide_curve_integration)
    from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource
    from z190_streaming_source_normal import (
        SourceRowBlock, accumulate_source_normal, replay_source_cost)

    context = ctx.load_context()
    factor = ctx.wide_factory(context)
    chart = context.state.structured_programme_coordinate_chart
    core = int(chart.core_count)
    private = np.asarray(chart.high_indices, int)
    frozen_q = context.q.copy()
    base = SimpleNamespace(count=core + 3, core_count=core, contract={},
        scales=np.empty(0), quadratic=np.empty(0, bool), updater=lambda q: None,
        blocks=lambda q, jacobian: iter(()), guard=lambda q: None)
    wide = CompleteWideSource(base, factor, 4)
    value = np.r_[context.q[:core], context.q[np.asarray(chart.bias_indices, int)],
                  context.q[private]]
    # This reduced container omits other families but exercises the same
    # production wide factory, columns, source rows and integer expansion.
    before = list(wide.blocks(value, True))[0]
    owner = wrap_original_wide_curve_integration(wide, factor, POLICY, wide_source=wide)
    after = owner.apply_block(before)
    assert all(getattr(before, key) is getattr(after, key)
               for key in ('r', 'J', 'quadratic', 'scales'))
    assert wrap_original_wide_curve_integration(wide, factor, None, wide_source=wide) is wide
    assert before.loss_weights is None
    assert len(after.r) == 23776
    assert after.loss_weights.working_normalization_multiplier is None

    policy = a.read(POLICY)
    packet_path = policy['source_packet_reference']['path']
    assert a.sha(packet_path) == policy['source_packet_reference']['sha256']
    original = a.read(packet_path)['observations']
    records = []
    for index, (row, observed, stated) in enumerate(zip(original, factor.observations, policy['rows'])):
        assert row == observed
        assert stated['original_v5_index'] == index
        uv = np.asarray(row['pixels_uv'], float)
        sigma = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(uv),))
        assert np.array_equal(factor.uvs[index], uv)
        assert np.array_equal(factor.sigmas[index], sigma)
        records.append(dict(id=row['id'], split='fit',
            source_video_sha256=policy['source_video_sha256'], actual_pts=56218880,
            physical_curve_id=stated['physical_curve_id'], operator_id=stated['operator_id'],
            uv=uv, stored_sigma_px=sigma, working_sigma_px=sigma,
            prior_sampling_multiplier=np.ones(len(uv)), point_integration_mass=np.ones(len(uv)),
            sampling_kind=['measured'] * len(uv)))
    # Independent compilation from authenticated old UV, sigma and identities;
    # no use of the production wrapper's precomputed payload or stored weights.
    plan = compile_curve_sampling(records)
    alpha = np.r_[np.concatenate([np.repeat(plan['rows'][r['id']]['point_integration_weights'], 2)
                                  for r in records]), np.ones(1520)]
    expected_payload = alpha[wide.row_indices]
    assert np.array_equal(after.loss_weights.integration_weights, expected_payload)
    assert np.all(alpha[:4288] > 0.) and np.all(alpha[4288:] == 1.)
    assert len(plan['groups']) == 40
    unique_r, core_j, private_j = factor.evaluate(context.q[:core], context.q[private], True)
    gold = np.asarray(a.read(context.directory/'source_review_UNACCEPTED.json')['wide']['source_normalized_residual'])
    assert np.array_equal(unique_r, gold)
    assert np.array_equal(before.r, unique_r[wide.row_indices])
    unique_j = hstack((core_j, csr_matrix((5808, 3)), csr_matrix(private_j)), format='csr')
    assert (before.J != unique_j[wide.row_indices]).nnz == 0
    active = np.unique(unique_j.indices)
    j = unique_j[:, active].toarray()
    expected_loss = SourceLossWeights(4. * np.asarray(factor.cost_weights) * alpha)
    expected_cost = weighted_source_cost(unique_r, factor.quadratic, expected_loss)
    expected_irls = source_irls_weights(unique_r, factor.quadratic, expected_loss)
    expected_g = j.T @ (expected_irls * unique_r)
    expected_h = j.T @ (expected_irls[:, None] * j)
    compact = SourceRowBlock(after.r, after.J[:, active], after.quadratic, after.scales,
                             after.row_ledger, after.loss_weights)
    normal = accumulate_source_normal(lambda: iter((compact,)), len(active),
        expected_row_count=23776, frozen_profile_token='same_q8_source_only_audit')
    np.testing.assert_allclose(normal.cost, expected_cost, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(normal.gradient, expected_g, rtol=2e-11, atol=1e-8)
    np.testing.assert_allclose(normal.normal.toarray(), expected_h, rtol=2e-11, atol=2e-6)
    old_compact = SourceRowBlock(before.r, before.J[:, active], before.quadratic,
                                 before.scales, before.row_ledger, before.loss_weights)
    rejected = False
    try:
        replay_source_cost(lambda: iter((old_compact,)), normal.contract,
                           frozen_profile_token='same_q8_source_only_audit')
    except ValueError:
        rejected = True
    assert rejected
    stored = source_loss_checkpoint_arrays(after.loss_weights, len(after.r))
    restored = source_loss_from_checkpoint(stored)
    assert restored.identity_digest == after.loss_weights.identity_digest
    assert np.array_equal(frozen_q, context.q)
    # Literal near-right 4 and outer 4 are still exact independent factors.
    near = next(i for i, row in enumerate(original) if row.get('weight', 1.) == 4.)
    offsets = np.r_[0, np.cumsum([2 * len(r['uv']) for r in records])]
    assert np.all(wide.row_counts[offsets[near]:offsets[near+1]] == 16)
    comparison = []
    parent_directory = a.R/'okada_review86_samea454_actual947_minimal_pivot_trial_resume_memory_sha'/'accepted_states'/a.PARENT
    for directory in (parent_directory, context.directory):
        review = a.read(directory/'source_review_UNACCEPTED.json')
        r = np.asarray(review['wide']['source_normalized_residual'])
        comparison.append(dict(q=directory.name,
            source_review=ref(directory/'source_review_UNACCEPTED.json'),
            old_objective_cost=weighted_source_cost(r, factor.quadratic,
                SourceLossWeights(4. * factor.cost_weights)),
            optional_objective_cost=weighted_source_cost(r, factor.quadratic, expected_loss),
            changed_loss_units_are_not_image_improvement=True))
    result = dict(schema='rinkspace.optional-wide45-same-q-actual-stream-audit.v1',
        passed=True, model_parameters_changed=False, active_config_changed=False,
        accuracy_accepted=False, complete_q_sha256=a.CHILD,
        complete_state=ref(context.path), policy=ref(POLICY), script=ref(__file__),
        integration_implementation=ref(Path(__file__).resolve().parents[2]/
            'calibration/solver_runtime/original_wide_curve_integration.py'),
        production_wide_implementation=ref(Path(__file__).resolve().parents[2]/
            'calibration/solver_runtime/legacy/wide_anchor_candidate/complete_wide_free_optics_source.py'),
        unique_rows=5808, expanded_rows=23776, original_curve_scalar_rows=4288,
        unchanged_other_scalar_rows=1520, original_points=2144, integration_groups=40,
        source_r_J_sigma_quadratic_object_identity_preserved=True,
        unique_residual_bitexact_saved_q8=True, current_complete_source_other_families_not_recomputed=True,
        no_complete_programme_J_or_candidate_or_video=True,
        raw_residual_sha256=numeric_sha(unique_r),
        active_wide_J_columns=active.tolist(), compact_actual_J_sha256=numeric_sha(j),
        production_loss_identity=after.loss_weights.identity_digest.hex(),
        unique_and_expanded_costs=[expected_cost, normal.cost],
        unique_and_expanded_gradient_max_abs_difference=float(abs(normal.gradient-expected_g).max()),
        unique_and_expanded_normal_max_abs_difference=float(abs(normal.normal.toarray()-expected_h).max()),
        independent_gradient_max_abs=float(abs(expected_g).max()),
        independent_normal_max_abs=float(abs(expected_h).max()),
        unique_and_expanded_gradient_relative_frobenius_difference=float(
            np.linalg.norm(normal.gradient-expected_g)/np.linalg.norm(expected_g)),
        unique_and_expanded_normal_relative_frobenius_difference=float(
            np.linalg.norm(normal.normal.toarray()-expected_h)/np.linalg.norm(expected_h)),
        old_objective_replay_rejected=rejected, checkpoint_loss_identity_exact=True,
        original_literal_near_right_4_and_outer_4_preserved=True,
        all_other_1520_scalar_integration_weights_one=True,
        guards_unchanged_by_loss_adapter_not_recomputed_in_this_audit=True,
        board_regions_not_certified_as_single_material_curve=True,
        separate_user_TOP_mean_targets_not_regrouped=True,
        saved_state_objective_comparison=comparison,
        new_source_images=0, new_persistent_J_arrays=0,
        seconds=time.perf_counter()-started)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(output), passed=True, seconds=result['seconds'],
                         cost=comparison), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
