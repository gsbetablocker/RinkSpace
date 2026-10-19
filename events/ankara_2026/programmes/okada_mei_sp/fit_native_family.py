"""Fresh new-source fitting at the explicit latest complete accepted v22 q."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,sys
from types import SimpleNamespace
import numpy as np
import fit_resume_revised as revised
from calibration.program_tracking import first_shot_native_family_bundle as family
from calibration.program_tracking import first_shot_source_transition as transition
from calibration.program_tracking import first_shot_resume_cache as cache

reuse=revised.reuse
BaseFreshCurrentReplay=reuse.FreshCurrentReplay
manifest_path=None
pivot_roundoff=False
live_review=False


class FamilyBundle(revised.RevisedBundle):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        family.attach_native_family_pipeline(self)


class OldInventoryView:
    """Bind snapshot producers to this old inventory, never to the new owner."""
    def __init__(self,owner,manifest):
        self._owner=owner
        self.bundle=SimpleNamespace(frames=owner.bundle._native_family_previous_frames,
            state=owner.bundle._native_family_previous_state)
        self.scales=owner.bundle._native_family_previous_scales
        self.quadratic=owner.bundle._native_family_previous_mask
        self.association_reference=manifest['base_association']
    def __getattr__(self,key):return getattr(self._owner,key)
    def checkpoint_arrays(self):
        return self._owner.checkpoint_arrays.__func__(self)
    def _ids_and_pts(self):
        inventory,_=cache.source_inventory(self.bundle)
        return inventory['source_point_ids'],inventory['native_pts']
    def complete_state_payload(self,q):
        return self._owner.complete_state_payload.__func__(self,q)


class FamilyReplay(revised.RevisedReplay):
    def __init__(self,adapter,**kwargs):
        ref,manifest,*_=family.read_transition(manifest_path)
        view=OldInventoryView(adapter,manifest)
        strict=transition.checkpoint_reference_verifier(kwargs['pool_transition'])
        Original=cache.CompleteSourceReplay
        class RichReferenceVerifier(Original):
            def verify_reference(self,value):
                strict.verified=self.verified;strict.transitions=self.transitions
                return strict.verify_reference(value)
        cache.CompleteSourceReplay=RichReferenceVerifier
        try:BaseFreshCurrentReplay.__init__(self,view,**kwargs)
        finally:cache.CompleteSourceReplay=Original
        # Also authenticate the immutable full old UV/target/PTS inventory;
        # the accepted raw snapshot itself owns IDs/PTS and the complete q.
        prior=transition.read(transition.reference(kwargs['prior_report']))
        root=Path(prior['snapshot']['path']).parent.parent
        identity=transition.read(transition.reference(root/'run_identity.json'))
        if identity['binding']!=manifest['base_binding']or identity['association']!=manifest['base_association']:
            raise ValueError('Previous immutable inventory belongs to another source basis')
        inventory,_=cache.source_inventory(view.bundle)
        saved_ref=identity['source_inventory']
        if transition.reference(saved_ref['path'])!=saved_ref:raise ValueError('Previous source inventory bytes changed')
        with np.load(saved_ref['path'],allow_pickle=False)as saved:
            if set(saved.files)!=set(inventory):raise ValueError('Old source inventory fields differ')
            for key,value in inventory.items():
                if not np.array_equal(saved[key],value):raise ValueError('Old complete source inventory differs: '+key)
        if (adapter.bundle.state.binding_reference!=manifest['revised_binding']
                or adapter.association_reference!=manifest['revised_association']):
            raise ValueError('New source identity differs from the admitted transition')
        self.adapter=adapter;self._new_source_ref=ref
        self.receipt['old_source_rows']=self.receipt.pop('source_rows')
        self.receipt['old_source_points']=self.receipt.pop('source_points')
        self.receipt.update(transition_manifest=ref,old_binding=manifest['base_binding'],new_binding=manifest['revised_binding'],
            source_rows=len(adapter.scales),source_points=sum(f['source_point_count']for f in adapter.bundle.frames),
            same_source_basis_as_checkpoint=False,same_source_objective_as_checkpoint=False,
            old_source_basis_prior_cost=prior['source_objective_cost'],old_new_costs_are_different_objectives=True,
            cross_basis_cost_reduction_not_claimed=True,old_complete_UV_ID_target_PTS_inventory_exact=True,
            source_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact=False,
            old_checkpoint_source_inventory_exact_new_inventory_separately_verified=True,
            complete_source_transition_comparison=deepcopy(adapter.bundle.native_family_receipt['source_scope']),
            latest_accepted_q_aux_damping_and_fraction_preserved=True,new_source_full_r_J_world_required=True,
            native_family_continuation_entry=transition.reference(__file__))
        for key in ('_native_family_previous_frames','_native_family_previous_state','_native_family_previous_scales','_native_family_previous_mask'):
            delattr(adapter.bundle,key)

    def install(self):
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
    own.add_argument('--family-revision-manifest',required='--help'not in sys.argv and'-h'not in sys.argv)
    own.add_argument('--recover-pivot-roundoff',action='store_true')
    own.add_argument('--live-review-export',action='store_true')
    own.add_argument('--from-current',help='Resolve the exact latest accepted canonical/report at launch time')
    flags,remaining=own.parse_known_args()
    if flags.from_current:
        if '--seed'in remaining or '--prior-report'in remaining:
            raise ValueError('--from-current supplies both the latest seed and its matching prior report')
        current=transition.read(transition.reference(flags.from_current));canonical=current['canonical_checkpoint']
        if transition.reference(canonical['path'])!={k:canonical[k]for k in ('path','sha256')}:
            raise ValueError('Latest complete canonical bytes changed')
        remaining.extend(('--seed',canonical['path'],'--prior-report',flags.from_current))
    sys.argv=[sys.argv[0],*remaining]
    pivot_roundoff=flags.recover_pivot_roundoff;live_review=flags.live_review_export
    if '--help'not in sys.argv and'-h'not in sys.argv:
        manifest_path=Path(flags.family_revision_manifest)
        _,manifest,*_=family.read_transition(manifest_path)
        parser=argparse.ArgumentParser(add_help=False)
        for key in ('binding','association','seed','prior-report','output'):parser.add_argument('--'+key,required=True)
        parser.add_argument('--geometry-context',action='store_true',required=True)
        parser.add_argument('--recover-finite-extent',action='store_true',required=True)
        parser.add_argument('--pool-transition',default=str(reuse.mount.driver.Z/'geometry_pool_reuse'))
        args,_=parser.parse_known_args()
        if transition.reference(args.binding)!=manifest['revised_binding']or transition.reference(args.association)!=manifest['revised_association']:
            raise ValueError('CLI source does not equal the exact admitted family revision')
        report=transition.read(transition.reference(args.prior_report))
        verifier=transition.checkpoint_reference_verifier(args.pool_transition);verifier.verify_references(report)
        with np.load(args.seed,allow_pickle=False)as saved:
            identity=json.loads(str(saved['identity_json']));verifier.verify_references(identity)
            if (saved['q'].shape!=(7674,)or hashlib.sha256(saved['q'].tobytes()).hexdigest()!=report['qaug_sha256']
                    or report['binding']!=manifest['base_binding']or identity['binding']!=manifest['base_binding']
                    or identity['association']!=manifest['base_association']):
                raise ValueError('Explicit latest accepted canonical/report must be the exact old v22 basis')
        revised.parent_installation=revised.fused.value.install()
        revised.validation_references=[transition.reference(reuse.mount.driver.Z/'value_real_state_QA/report.json'),
            transition.reference(reuse.mount.driver.Z/'pivot_step_QA/report.json')]
        family.install_native_family_bundle(manifest_path)
    previous=reuse.annotate_current_report
    def annotate(report,owner):
        previous(report,owner)
        if isinstance(report,dict)and'fresh_current_source_J_every_new_state'in report:
            report.update(source_basis='RGB-verified native material-interface family points on the existing finite-curve/observed-y camera source',
                same_source_basis_as_checkpoint=False,same_source_objective_as_checkpoint=False,
                old_new_source_costs_are_different_objectives=True,cross_basis_cost_reduction_not_claimed=True,
                verified_native_family_source_revision=deepcopy(owner.bundle.native_family_receipt),
                same_state_original_source_value_replay=dict(owner.source_value_replay_disabled_receipt))
        return report
    reuse.annotate_current_report=annotate
    reuse.ExecutionBundle=FamilyBundle;reuse.FreshCurrentReplay=FamilyReplay
    if '--fresh-current-state'not in sys.argv:sys.argv.append('--fresh-current-state')
    reuse.main()


if __name__=='__main__':main()
