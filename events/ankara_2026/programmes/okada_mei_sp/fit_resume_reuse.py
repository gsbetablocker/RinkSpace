"""Continue released-domain fitting with the existing public J-reuse policy."""
from pathlib import Path
import sys

import fit_resume_mount as mount
from calibration.program_tracking.first_shot_jacobian_reuse import (
    install_source_jacobian_reuse, annotate_current_report, reference)
import hashlib
import json
from types import MethodType
import numpy as np

BaseBundle = mount.ReleasedBundle
BaseReplay = mount.ReleasedReplay
active_owner = None
output_directory = None


class ExecutionBundle(BaseBundle):
    def __init__(self, *args, **kwargs):
        global active_owner
        super().__init__(*args, **kwargs)
        if mount.recover_finite_extent:
            from calibration.program_tracking.first_shot_finite_recovery_fast import install_finite_endpoint_recovery_fast
            self.finite_endpoint_recovery_fast_receipt = install_finite_endpoint_recovery_fast(
                self, allowed_columns=(95, 96, 97))
        active_owner = self


def install_execution(replay):
    from calibration.program_tracking.first_shot_raw_execution import install
    install(replay.adapter)
    install_source_jacobian_reuse(replay.adapter,
        cache_directory=output_directory / 'source_J_cache', entry_reference=reference(__file__))
    replay.receipt.update(all_changed_states_fresh=False,
        all_changed_states_fresh_source_values=True,
        changed_state_J_is_explicit_public_modified_Newton_model=True,
        source_J_execution_policy=replay.adapter.source_jacobian_reuse_policy)
    return replay


class ExecutionReplay(BaseReplay):
    def install(self):
        super().install()
        self.adapter._initial_complete_J_reuse = True
        return install_execution(self)


class FreshCurrentReplay:
    """Authenticate the accepted state, then evaluate its original source anew."""
    def __init__(self, adapter, *, prior_report, checkpoint, pool_transition, **unused):
        from calibration.program_tracking.first_shot_resume_cache import CompleteSourceReplay, source_inventory
        self.adapter = adapter
        verifier = object.__new__(CompleteSourceReplay)
        verifier.pool_transition = Path(pool_transition)
        verifier.transitions, verifier.verified = {}, {}
        report_ref, checkpoint_ref = reference(prior_report), reference(checkpoint)
        report = json.loads(Path(prior_report).read_text(encoding='utf8'))
        verifier.verify_references(report)
        def same(actual, expected, name):
            if not np.array_equal(actual, expected):
                raise ValueError('Fresh current-state continuation differs: ' + name)
        with np.load(checkpoint, allow_pickle=False) as saved:
            identity = json.loads(str(saved['identity_json']))
            verifier.verify_references(identity)
            q = saved['q'].copy()
            same(q, adapter.start, 'complete q')
            same(saved['scales'], adapter.scales, 'scales')
            same(saved['quadratic'], adapter.quadratic, 'source mask')
            for name, value in adapter.checkpoint_arrays().items():
                same(saved['aux__' + name], value, 'aux ' + name)
            self.damping, self.fraction = float(saved['damping']), float(saved['next_line_search_fraction'])
        token = hashlib.sha256(np.asarray(q, float).tobytes()).hexdigest()
        if (report['qaug_sha256'] != token or report['binding'] != adapter.bundle.state.binding_reference
                or identity['binding'] != adapter.bundle.state.binding_reference
                or identity['association'] != adapter.association_reference):
            raise ValueError('Fresh accepted checkpoint/report/binding identity differs')
        if (adapter.seed_receipt['initial_damping'] != self.damping
                or adapter.seed_receipt['initial_line_search_fraction'] != self.fraction):
            raise ValueError('Fresh continuation altered checkpoint damping/fraction')
        inventory, _ = source_inventory(adapter.bundle)
        with np.load(report['snapshot']['path'], allow_pickle=False) as saved:
            for name, value in adapter.complete_state_payload(q).items():
                same(saved[name], value, 'snapshot ' + name)
            same(saved['source_point_ids'], inventory['source_point_ids'], 'native source IDs')
            same(saved['native_pts_per_point'], inventory['native_pts'], 'native source PTS')
        self.q = np.empty(0)  # Never enter the driver's borrowed-initial-J export branch.
        self.receipt = dict(schema='rinkspace.authenticated-current-state-fresh-source-start.v1',
            checkpoint=checkpoint_ref, prior_report=report_ref, qaug_sha256=token,
            parameter_count=len(q), source_rows=len(adapter.scales), source_points=len(inventory['source_point_ids']),
            initial_source_J_reuse=False, initial_source_r_reuse=False, initial_raw_reuse=False,
            cached_world_or_J_loaded=False, original_current_state_source_producer_required=True,
            source_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact=True,
            initial_damping=self.damping, initial_line_search_fraction=self.fraction,
            damping_and_line_search_fraction_unchanged=True,
            prior_accepted_q_preserved_without_returning_to_an_older_J_anchor=True,
            producer_transitions=list(verifier.transitions.values()), exact_other_references=len(verifier.verified),
            producer=reference(__file__))

    def install(self):
        owner = self.adapter
        original_identity = owner.checkpoint_identity
        receipt = self.receipt
        def identity(adapter):
            result = original_identity()
            result['authenticated_current_state_fresh_source_start'] = dict(receipt)
            result['initial_exact_complete_current_state_full_J_reuse'] = False
            return result
        owner.checkpoint_identity = MethodType(identity, owner)
        owner._initial_complete_J_reuse = False
        return install_execution(self)


def main():
    global output_directory
    fresh = '--fresh-current-state' in sys.argv
    if fresh:
        sys.argv.remove('--fresh-current-state')
        if '--prepare-only' in sys.argv:
            raise ValueError('--fresh-current-state evaluates a new source J; it cannot use --prepare-only')
    import argparse
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', required=True)
        args, _ = parser.parse_known_args()
        output_directory = Path(args.output)
    from calibration.program_tracking import row_scene_projection_fast
    row_scene_projection_fast.install()
    original_write = mount.driver.write

    def write(path, value):
        if active_owner is not None:
            annotate_current_report(value, active_owner)
        return original_write(path, value)

    mount.driver.write = write
    mount.ReleasedBundle = ExecutionBundle
    mount.ReleasedReplay = FreshCurrentReplay if fresh else ExecutionReplay
    mount.main()


if __name__ == '__main__':
    main()
