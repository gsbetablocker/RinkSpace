"""Thin same-model review of the explicitly verified revised source basis."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def _review_proof_view(bundle):
    # The legacy reviewer consumes this original native-readout count. Keep the
    # entire new proof and expose its already authenticated original count only
    # on this review instance; no fit producer or revision fact is rewritten.
    proof=bundle.source_revision_proof
    count=proof['original_authenticated_basis']['added_native_readouts']
    bundle.source_revision_proof=dict(proof,added_native_readouts=count)
    return count


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False)
    for name in ('revision-manifest','prototype-dir','binding','association'):
        parser.add_argument('--'+name,required=True)
    args,_=parser.parse_known_args(arguments)
    forwarded=[]
    i=0
    while i<len(arguments):
        if arguments[i]=='--revision-manifest':i+=2;continue
        if arguments[i].startswith('--revision-manifest='):i+=1;continue
        forwarded.append(arguments[i]);i+=1
    sys.path.insert(0,str(Path(args.prototype_dir).resolve()))
    from calibration.solver_runtime import install,configure_native_runtime
    install();configure_native_runtime(stage='verified-source-revised-same-model-review',native_threads=1)
    import cv2
    cv2.setNumThreads(1)
    import native_transverse_bundle as native
    from calibration.program_tracking.first_shot_revised_bundle import (
        install_revised_source_bundle,attach_revised_pipeline)
    from calibration.program_tracking import first_shot_source_transition as transition
    from calibration.program_tracking import first_shot_mount_domain as mount
    from calibration.program_tracking import review_transverse_fit as review
    installation=install_revised_source_bundle(args.revision_manifest)
    manifest=transition.read(installation['transition_manifest'])
    if (transition.reference(args.binding)!=manifest['revised_binding']
            or transition.reference(args.association)!=manifest['revised_association']):
        raise ValueError('Review CLI differs from the explicitly verified revised source')
    changes={(int(item['native_pts']),str(item['source_point_id'])):item
             for item in manifest['verified_changes']}
    quarantines={(int(item['native_pts']),str(item['source_point_id'])):item
                 for item in manifest.get('verified_quarantines',[])}
    Original=native.CompleteTransverseBundle

    class RevisedReviewBundle(Original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            try:
                pipeline=attach_revised_pipeline(self)
                original_added_native_readouts=_review_proof_view(self.bundle)
                release=mount.install_unmeasured_mount_domain(self,entry_reference=transition.reference(__file__))
                old_frames=getattr(self.bundle,'_authenticated_previous_frames',None)
                if old_frames is None:raise ValueError('Review source scope requires the already authenticated old inventory')
                comparison=transition.compare_source_frames(old_frames,self.bundle.frames,changes,{},quarantines)
                comparison.pop('_retained_formal_indices')
                per_frame=comparison.pop('frames')
                comparison['per_frame_scope_sha256']=hashlib.sha256(json.dumps(per_frame,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                receipt=dict(producer=transition.reference(__file__),pipeline=pipeline,
                    transition_manifest=installation['transition_manifest'],source_scope=comparison,
                    old_frames_released_after_exact_inventory_proof=True,
                    unchanged_core_geometry_camera_native_source_checks=True,
                    legacy_review_added_native_readouts=original_added_native_readouts,
                    legacy_review_count_from='source_revision_proof.original_authenticated_basis.added_native_readouts')
                self.seed_receipt['verified_revised_source_review']=receipt
                self.seed_receipt['review_mount_photo_domain_release']=release
                for ref in (transition.reference(__file__),transition.reference(transition.__file__),
                            transition.reference(mount.__file__),installation['transition_manifest']):
                    if ref not in self.source_producers:self.source_producers.append(ref)
            except BaseException:
                self.close();raise
            finally:
                if hasattr(self.bundle,'_authenticated_previous_frames'):
                    del self.bundle._authenticated_previous_frames

    native.CompleteTransverseBundle=RevisedReviewBundle
    return review.main(forwarded)


if __name__=='__main__':
    main()
