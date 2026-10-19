"""Fresh full-J continuation of an explicit latest accepted v21 state.

Optional --recover-pivot-roundoff reuses the existing zero-buffer pivot-domain
initializer only after the actual complete recover reports an invalid pivot.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import fit_resume_revised as revised
from calibration.program_tracking import first_shot_source_transition as transition
from calibration.program_tracking import first_shot_resume_cache as cache

BaseBundle=revised.RevisedBundle
BaseReplay=revised.RevisedReplay
FreshCurrentReplay=revised.reuse.FreshCurrentReplay
pivot_roundoff_recovery=False


class SameBasisBundle(BaseBundle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            data=transition.validate_manifest(revised.transition_manifest)
            old=getattr(self.bundle,'_authenticated_previous_frames',None)
            if old is None:raise ValueError('Authenticated original operator inventory is required')
            proof=transition.compare_source_frames(old,self.bundle.frames,data[2],data[3],data[-1])
            proof.pop('_retained_formal_indices');proof.pop('frames')
            self.seed_receipt['same_basis_revised_source_reconstruction']=proof
        except BaseException:
            self.close();raise
        finally:
            if hasattr(self.bundle,'_authenticated_previous_frames'):
                del self.bundle._authenticated_previous_frames


class SameBasisReplay(BaseReplay):
    def __init__(self,adapter,**kwargs):
        # Reuse the complete existing fresh-state checks. Only the temporary
        # verifier object's enriched file references need their identity fields.
        strict=transition.checkpoint_reference_verifier(kwargs['pool_transition'])
        Original=cache.CompleteSourceReplay
        class RichReferenceVerifier(Original):
            def verify_reference(self,value):
                strict.verified=self.verified;strict.transitions=self.transitions
                return strict.verify_reference(value)
        cache.CompleteSourceReplay=RichReferenceVerifier
        try:FreshCurrentReplay.__init__(self,adapter,**kwargs)
        finally:cache.CompleteSourceReplay=Original
        manifest=transition.read(transition.reference(revised.transition_manifest))
        if (adapter.bundle.state.binding_reference!=manifest['revised_binding']
                or adapter.association_reference!=manifest['revised_association']):
            raise ValueError('Same-basis continuation requires the identical verified v21 source')
        self._new_source_ref=transition.reference(revised.transition_manifest)
        self.receipt.update(transition_manifest=self._new_source_ref,
            same_source_basis_as_checkpoint=True,same_source_objective_as_checkpoint=True,
            first_v20_to_v21_conversion_performed=False,
            old_new_costs_are_different_objectives=False,cross_basis_cost_reduction_not_claimed=True,
            same_basis_continuation_entry=transition.reference(__file__),
            latest_accepted_q_aux_damping_and_fraction_preserved=True)

    def install(self):
        super().install()
        owner=self.adapter
        owner.fused_execution_receipt.update(same_source_basis_as_checkpoint=True,
            new_source_initial_full_r_J_required=False,fresh_current_state_full_r_J_required=True)
        owner.fused_execution_receipt.pop('new_source_basis_transition',None)
        owner.fused_execution_receipt['verified_source_recipe_manifest']=self._new_source_ref
        ref=transition.reference(__file__)
        if ref not in owner.source_producers:owner.source_producers.append(ref)
        owner.fused_execution_receipt['pivot_boundary_roundoff_recovery_requested']=pivot_roundoff_recovery
        if pivot_roundoff_recovery:
            from calibration.program_tracking.first_shot_pivot_recovery import install_pivot_boundary_recovery
            qa=transition.reference(revised.reuse.mount.driver.Z/'pivot_nonlinear_QA/recovery_report.json')
            install_pivot_boundary_recovery(owner,entry_reference=ref,qa_references=(qa,))
        return self


def main():
    global pivot_roundoff_recovery
    own=argparse.ArgumentParser(add_help=False)
    own.add_argument('--recover-pivot-roundoff',action='store_true')
    flags,remaining=own.parse_known_args()
    pivot_roundoff_recovery=flags.recover_pivot_roundoff
    sys.argv=[sys.argv[0],*remaining]
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser=argparse.ArgumentParser(add_help=False)
        for key in ('seed','prior-report','revision-manifest'):
            parser.add_argument('--'+key,required=True)
        parser.add_argument('--pool-transition',default=str(revised.reuse.mount.driver.Z/'geometry_pool_reuse'))
        args,_=parser.parse_known_args()
        manifest=transition.read(transition.reference(args.revision_manifest))
        report=transition.read(transition.reference(args.prior_report))
        verifier=transition.checkpoint_reference_verifier(args.pool_transition)
        verifier.verify_references(report)
        with np.load(args.seed,allow_pickle=False)as saved:
            identity=json.loads(str(saved['identity_json']));verifier.verify_references(identity)
            q=saved['q']
            if (q.shape!=(7674,) or hashlib.sha256(q.tobytes()).hexdigest()!=report['qaug_sha256']
                    or report['binding']!=manifest['revised_binding']
                    or identity['binding']!=manifest['revised_binding']
                    or identity['association']!=manifest['revised_association']):
                raise ValueError('Explicit latest checkpoint/report must have the same verified v21 basis')
    original_write=revised.reuse.mount.driver.write
    def write(path,value):
        if isinstance(value,dict) and 'fresh_current_source_J_every_new_state' in value:
            value.update(source_basis='Identical verified v21 source and objective; explicit latest accepted full-state continuation',
                same_source_basis_as_checkpoint=True,same_source_objective_as_checkpoint=True,
                old_new_source_costs_are_different_objectives=False,
                first_v20_to_v21_conversion_performed=False,
                same_basis_continuation_entry=transition.reference(__file__))
        return original_write(path,value)
    revised.reuse.mount.driver.write=write
    revised.RevisedBundle=SameBasisBundle
    revised.RevisedReplay=SameBasisReplay
    revised.main()


if __name__=='__main__':
    main()
