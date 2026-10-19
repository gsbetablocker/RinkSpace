"""Thin opt-in continuation with the unmeasured photo numeric window released."""
from pathlib import Path
import argparse
import sys
import types

import fit_resume as driver
from calibration.program_tracking.first_shot_mount_domain import install_unmeasured_mount_domain, reference

BaseBundle = driver.CompleteSourceMetricTransverseBundle
BaseReplay = driver.CompleteSourceReplay
failure_directory = None
recover_finite_extent = False


class ReleasedBundle(BaseBundle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_unmeasured_mount_domain(self, entry_reference=reference(__file__),
                                       failure_directory=failure_directory)
        if recover_finite_extent:
            from calibration.program_tracking.first_shot_finite_recovery import install_finite_endpoint_recovery
            self.finite_endpoint_recovery_receipt = install_finite_endpoint_recovery(self, allowed_columns=(95, 96, 97))


class ReleasedReplay(BaseReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receipt.update(same_physical_guard_source_weights_and_solver_policy=False,
            source_r_J_and_objective_mathematics_unchanged=True,
            all_non_mount_physical_guards_and_solver_acceptance_policy_unchanged=True,
            photo_numeric_mount_window_released=self.adapter._unmeasured_mount_domain)

    def install(self):
        super().install()
        owner = self.adapter
        original_kwargs = owner.solver_kwargs
        policy = dict(authenticated_checkpoint_fraction=self.fraction,
            continuation_initial_fraction=1., damping_unchanged=self.damping,
            complete_q_r_J_and_parameter_values_unchanged=True,
            reason='Unmeasured photo window released and finite endpoint feasibility recovery connected; old rejected-domain backtracking seed is obsolete')
        self.receipt['changed_feasible_domain_line_search_restart'] = policy

        def solver_kwargs(adapter):
            result = original_kwargs()
            result['initial_line_search_fraction'] = 1.
            result['checkpoint_identity']['changed_feasible_domain_line_search_restart'] = policy
            return result

        owner.solver_kwargs = types.MethodType(solver_kwargs, owner)
        return self


def main():
    global failure_directory, recover_finite_extent
    recover_finite_extent = '--recover-finite-extent' in sys.argv
    if recover_finite_extent:
        sys.argv.remove('--recover-finite-extent')
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', required=True)
        args, _ = parser.parse_known_args()
        failure_directory = Path(args.output) / 'guard_failures'
    driver.CompleteSourceMetricTransverseBundle = ReleasedBundle
    driver.CompleteSourceReplay = ReleasedReplay
    driver.main()


if __name__ == '__main__':
    main()
