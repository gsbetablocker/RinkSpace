"""Continue the paired latest accepted cache with explicit fused value execution.

This entry retains the public modified-Newton policy, connects the existing
pivot outside face, and requires one fresh full-source same-seed r/J proof.
It never chooses an older J anchor or starts from the legacy default seed.
"""
from pathlib import Path
import argparse
import sys
from types import MethodType

import fit_resume_reuse as reuse
from calibration.program_tracking import first_shot_value_execution as value
from calibration.program_tracking.first_shot_fused_execution import (
    install_worker_execution, certify_seed_source, reference)
from calibration.program_tracking.first_shot_pivot_constraints import install_pivot_outside_step_constraints
from calibration.program_tracking.first_shot_source_value_replay import install as install_source_values

BaseBundle = reuse.ExecutionBundle
parent_installation = None
validation_references = None


class FusedBundle(BaseBundle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_pivot_outside_step_constraints(self)


def install_execution(replay):
    from calibration.program_tracking.first_shot_raw_execution import install as install_raw
    owner = replay.adapter
    try:
        install_raw(owner)
        install_worker_execution(owner, parent_installation=parent_installation,
            entry_reference=reference(__file__), validation_references=validation_references)
        reuse.install_source_jacobian_reuse(owner,
            cache_directory=reuse.output_directory / 'source_J_cache', entry_reference=reference(__file__))
        previous_kwargs = owner.solver_kwargs
        restart = dict(authenticated_checkpoint_fraction=replay.fraction,
            continuation_initial_fraction=1., damping_unchanged=replay.damping,
            complete_q_r_J_and_parameter_values_unchanged=True,
            reason='The existing fixed-pivot outside-or-on fact is now connected to the shared QP; retry the original real full step with the same nonlinear guard instead of inheriting outside-rejection backtracking')
        replay.receipt['changed_feasible_domain_line_search_restart'] = restart

        def solver_kwargs(adapter):
            result = previous_kwargs()
            result['initial_line_search_fraction'] = 1.
            result['checkpoint_identity']['changed_feasible_domain_line_search_restart'] = dict(restart)
            return result

        owner.solver_kwargs = MethodType(solver_kwargs, owner)
        install_source_values(owner, update_profiles=None)
        certify_seed_source(replay, reuse.output_directory / 'fused_full_source_QA.json')
        replay.receipt.update(all_changed_states_fresh=False,
            all_changed_states_original_source_values_required=True,
            same_state_values_may_be_original_value_replay=True,
            changed_state_J_is_explicit_public_modified_Newton_model=True,
            source_J_execution_policy=owner.source_jacobian_reuse_policy,
            fused_same_value_execution=owner.fused_execution_receipt,
            source_value_replay=owner.source_value_replay_receipt)
        return replay
    except BaseException:
        owner.close()
        raise


def main():
    global parent_installation, validation_references
    if '--fresh-current-state' in sys.argv:
        raise ValueError('Fused continuation requires paired exact-current checkpoint/r/J/world cache; no alternate fresh anchor')
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        for name in ('seed', 'cache-manifest', 'world-cache-manifest', 'prior-report', 'output'):
            parser.add_argument('--' + name, required=True)
        parser.add_argument('--geometry-context', action='store_true', required=True)
        parser.parse_known_args()
        root = reuse.mount.driver.Z
        validation_references = [reference(root / 'value_real_state_QA/report.json'),
                                 reference(root / 'pivot_step_QA/report.json')]
        parent_installation = value.install()
    original_annotate = reuse.annotate_current_report

    def annotate(report, owner):
        original_annotate(report, owner)
        if isinstance(report, dict) and 'fresh_current_source_J_every_new_state' in report:
            report['fused_same_value_execution'] = owner.checkpoint_identity()['fused_same_value_execution']
            report['same_state_original_source_value_replay'] = dict(owner.source_value_replay_receipt)
            report['original_fixed_pivot_outside_shared_QP_bridge'] = owner.pivot_outside_step_receipt
        return report

    reuse.annotate_current_report = annotate
    reuse.ExecutionBundle = FusedBundle
    reuse.install_execution = install_execution
    reuse.main()


if __name__ == '__main__':
    main()
