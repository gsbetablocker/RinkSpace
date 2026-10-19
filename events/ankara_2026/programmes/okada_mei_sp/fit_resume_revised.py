"""Fresh new-source r/J at the complete accepted q after a verified RGB edit."""
from pathlib import Path
import argparse
import sys
from types import MethodType

import fit_resume_fused as fused
from calibration.program_tracking.first_shot_source_transition import (
    RevisedSourceStart, validate_manifest, reference)
from calibration.program_tracking.first_shot_revised_bundle import (
    install_revised_source_bundle, attach_revised_pipeline)

reuse = fused.reuse
transition_manifest = None
parent_installation = None
validation_references = None


class RevisedBundle(fused.FusedBundle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        attach_revised_pipeline(self)


class RevisedReplay(RevisedSourceStart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, transition_manifest=transition_manifest, **kwargs)

    def install(self):
        super().install()
        owner = self.adapter
        from calibration.program_tracking.first_shot_raw_execution import install as install_raw
        from calibration.program_tracking.first_shot_fused_execution import install_worker_execution
        try:
            install_raw(owner)
            receipt = install_worker_execution(owner, parent_installation=parent_installation,
                entry_reference=reference(__file__), validation_references=validation_references)
            # Existing execution is already certified at the preceding source.
            # New semantic targets intentionally define a different residual.
            receipt.update(full_same_seed_source_r_J_proof_required=False,
                full_same_seed_source_r_J_proof=None,
                cross_source_basis_old_new_r_J_equality_not_claimed=True,
                new_source_initial_full_r_J_required=True,
                new_source_basis_transition=self.receipt['transition_manifest'])
            reuse.install_source_jacobian_reuse(owner,
                cache_directory=reuse.output_directory / 'source_J_cache', entry_reference=reference(__file__))
            disabled=dict(enabled=False, original_changed_state_source_factory_required=True,
                reason='Whole-source frozen-input hashing on every cache lookup exceeded its saved projection time; this optional layer is disabled for the new source basis',
                cache_hits_claimed=False, raw_retention_and_public_J_reuse_remain_enabled=True)
            previous_identity=owner.checkpoint_identity
            def identity(adapter):
                result=previous_identity();result['same_state_source_value_replay_execution']=dict(disabled)
                return result
            owner.checkpoint_identity=MethodType(identity,owner)
            owner.source_value_replay_disabled_receipt=disabled
            self.receipt.update(fused_same_value_execution=receipt,
                source_J_execution_policy=owner.source_jacobian_reuse_policy,
                same_state_value_replay=disabled,
                first_actual_raw_hit_full_original_bitwise_shadow_required=True,
                source_value_replay_not_installed=True)
            return self
        except BaseException:
            owner.close()
            raise


def main():
    global transition_manifest, parent_installation, validation_references
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--revision-manifest', required=True)
        for key in ('binding','association','seed','prior-report','output'):
            parser.add_argument('--'+key, required=True)
        parser.add_argument('--geometry-context', action='store_true', required=True)
        parser.add_argument('--recover-finite-extent', action='store_true', required=True)
        args, remaining = parser.parse_known_args()
        transition_manifest = Path(args.revision_manifest)
        _, manifest, *_ = validate_manifest(transition_manifest)
        if (reference(args.binding)!=manifest['revised_binding']
                or reference(args.association)!=manifest['revised_association']):
            raise ValueError('CLI source does not equal the verified new source basis')
        # Remove only our new flag; the original driver receives its whole CLI.
        i=sys.argv.index('--revision-manifest');del sys.argv[i:i+2]
        root=reuse.mount.driver.Z
        validation_references=[reference(root/'value_real_state_QA/report.json'),
            reference(root/'pivot_step_QA/report.json')]
        parent_installation=fused.value.install()
        install_revised_source_bundle(transition_manifest)
    previous_annotate=reuse.annotate_current_report

    def annotate(report, owner):
        previous_annotate(report,owner)
        if isinstance(report,dict) and 'fresh_current_source_J_every_new_state' in report:
            report.update(source_basis='RGB-verified source target revision; new objective at the preserved complete physical state',
                source_basis_transition=owner.source_basis_transition_receipt,
                old_new_source_costs_are_different_objectives=True,
                cross_basis_cost_reduction_not_claimed=True,
                fused_same_value_execution=owner.checkpoint_identity()['fused_same_value_execution'],
                same_state_original_source_value_replay=dict(owner.source_value_replay_disabled_receipt),
                original_fixed_pivot_outside_shared_QP_bridge=owner.pivot_outside_step_receipt)
        return report

    reuse.annotate_current_report=annotate
    reuse.ExecutionBundle=RevisedBundle
    reuse.FreshCurrentReplay=RevisedReplay
    if '--fresh-current-state' not in sys.argv:sys.argv.append('--fresh-current-state')
    reuse.main()


if __name__=='__main__':
    main()
