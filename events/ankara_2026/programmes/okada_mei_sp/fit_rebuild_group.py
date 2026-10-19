"""Continue the latest complete v21 q after one reviewed source-group rebuild."""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib,json,sys
from types import SimpleNamespace
import numpy as np
import fit_continue_v21 as same
from calibration.program_tracking import first_shot_group_rebuild as group
from calibration.program_tracking import first_shot_source_transition as transition

revised=same.revised
reuse=revised.reuse
manifest_path=None
pivot_roundoff=False
live_review=False


class GroupBundle(revised.RevisedBundle):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        group.attach_group_pipeline(self)


class _OldView:
    def __init__(self,owner,manifest):
        self._owner=owner
        self.bundle=SimpleNamespace(frames=owner.bundle._authenticated_group_previous_frames,
            state=SimpleNamespace(binding_reference=manifest['base_binding']))
        self.scales=owner.bundle._authenticated_group_previous_scales
        self.quadratic=owner.bundle._authenticated_group_previous_mask
        self.association_reference=manifest['base_association']
    def __getattr__(self,key):return getattr(self._owner,key)


class GroupReplay(same.SameBasisReplay):
    def __init__(self,adapter,**kwargs):
        ref,manifest,*_=group.read_transition(manifest_path)
        # Existing FreshCurrentReplay verifies complete q/aux, old masks/scales,
        # snapshot IDs/PTS and actual checkpoint damping/fraction. This view
        # borrows the already-authenticated pre-delta inventory, not old r/J.
        view=_OldView(adapter,manifest)
        same.SameBasisReplay.__init__(self,view,**kwargs)
        self.adapter=adapter;self._new_source_ref=ref
        self.receipt['old_checkpoint_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact']=self.receipt.pop('source_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact')
        self.receipt.pop('same_basis_continuation_entry',None)
        prior=transition.read(transition.reference(kwargs['prior_report']))
        self.receipt.update(transition_manifest=ref,
            same_source_basis_as_checkpoint=False,same_source_objective_as_checkpoint=False,
            first_v20_to_v21_conversion_performed=False,first_verified_whole_group_conversion_performed=True,
            old_new_costs_are_different_objectives=True,old_source_basis_prior_cost=prior['source_objective_cost'],
            cross_basis_cost_reduction_not_claimed=True,
            only_verified_whole_group_mean_D_covariance_W_reconstructed=True,
            complete_source_transition_comparison=deepcopy(adapter.bundle.group_rebuild_receipt['source_scope']),
            old_source_inventory_authentication_used_existing_frames_without_second_source_load=True,
            latest_accepted_q_aux_damping_and_fraction_preserved=True,
            producer=transition.reference(__file__),new_group_continuation_entry=transition.reference(__file__))
        for key in ('_authenticated_group_previous_frames','_authenticated_group_previous_scales','_authenticated_group_previous_mask'):
            delattr(adapter.bundle,key)

    def install(self):
        # Reuse all frozen fusion/raw/valueworker/finite/Jreuse execution.
        revised.RevisedReplay.install(self)
        owner=self.adapter
        if pivot_roundoff:
            from calibration.program_tracking.first_shot_pivot_recovery import install_pivot_boundary_recovery
            install_pivot_boundary_recovery(owner,entry_reference=transition.reference(__file__),
                qa_references=(transition.reference(reuse.mount.driver.Z/'pivot_nonlinear_QA/recovery_report.json'),))
        if live_review:
            from calibration.program_tracking.first_shot_live_review_export import install_live_review_export
            install_live_review_export(owner,output_directory=reuse.output_directory,
                prototype_dir=reuse.mount.driver.Z,annotation_module=reuse,entry_reference=transition.reference(__file__))
        ref=transition.reference(__file__)
        if ref not in owner.source_producers:owner.source_producers.append(ref)
        return self


def main():
    global manifest_path,pivot_roundoff,live_review
    own=argparse.ArgumentParser(add_help=False)
    own.add_argument('--group-revision-manifest',required='--help' not in sys.argv and '-h' not in sys.argv)
    own.add_argument('--recover-pivot-roundoff',action='store_true')
    own.add_argument('--live-review-export',action='store_true')
    args,remaining=own.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    pivot_roundoff=args.recover_pivot_roundoff;live_review=args.live_review_export
    if '--help' not in sys.argv and '-h' not in sys.argv:
        manifest_path=Path(args.group_revision_manifest)
        ref,manifest,*_=group.read_transition(manifest_path)
        parser=argparse.ArgumentParser(add_help=False)
        for key in ('binding','association','seed','prior-report','output'):
            parser.add_argument('--'+key,required=True)
        parser.add_argument('--geometry-context',action='store_true',required=True)
        parser.add_argument('--recover-finite-extent',action='store_true',required=True)
        parser.add_argument('--pool-transition',default=str(reuse.mount.driver.Z/'geometry_pool_reuse'))
        state,_=parser.parse_known_args()
        if (transition.reference(state.binding)!=manifest['revised_binding']
                or transition.reference(state.association)!=manifest['revised_association']):
            raise ValueError('CLI binding/association differ from the exact group source transition')
        report=transition.read(transition.reference(state.prior_report))
        verifier=transition.checkpoint_reference_verifier(state.pool_transition);verifier.verify_references(report)
        with np.load(state.seed,allow_pickle=False)as saved:
            identity=json.loads(str(saved['identity_json']));verifier.verify_references(identity)
            if (saved['q'].shape!=(7674,) or hashlib.sha256(saved['q'].tobytes()).hexdigest()!=report['qaug_sha256']
                    or report['binding']!=manifest['base_binding'] or identity['binding']!=manifest['base_binding']
                    or identity['association']!=manifest['base_association']):
                raise ValueError('Explicit latest complete checkpoint/report must be the exact old v21 source')
        revised.transition_manifest=Path(manifest['authenticated_v21_transition']['path'])
        revised.parent_installation=revised.fused.value.install()
        revised.validation_references=[transition.reference(reuse.mount.driver.Z/'value_real_state_QA/report.json'),
            transition.reference(reuse.mount.driver.Z/'pivot_step_QA/report.json')]
        group.install_group_rebuilt_bundle(manifest_path)
    previous=reuse.annotate_current_report
    def annotate(report,owner):
        previous(report,owner)
        if isinstance(report,dict) and 'fresh_current_source_J_every_new_state'in report:
            report.update(source_basis='Reviewed post-native complete source-group mean/D/covariance/W rebuild; fresh objective at preserved latest full q',
                old_new_source_costs_are_different_objectives=True,cross_basis_cost_reduction_not_claimed=True,
                same_source_basis_as_checkpoint=False,same_source_objective_as_checkpoint=False,
                verified_whole_group_source_revision=deepcopy(owner.bundle.group_rebuild_receipt),
                fused_same_value_execution=owner.checkpoint_identity()['fused_same_value_execution'],
                same_state_original_source_value_replay=dict(owner.source_value_replay_disabled_receipt),
                original_fixed_pivot_outside_shared_QP_bridge=owner.pivot_outside_step_receipt)
        return report
    reuse.annotate_current_report=annotate
    reuse.ExecutionBundle=GroupBundle;reuse.FreshCurrentReplay=GroupReplay
    if '--fresh-current-state'not in sys.argv:sys.argv.append('--fresh-current-state')
    reuse.main()


if __name__=='__main__':main()
