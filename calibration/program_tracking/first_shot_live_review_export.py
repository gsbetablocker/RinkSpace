"""Optional request-driven review transport from a live accepted fit owner.

A request is {"id": "unique request", "check_frames": [1, 842, 1248]} in
output/review_request.json. No source bundle is constructed and no video is
rendered. Every request id is consumed once, including diagnostic failures.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import MethodType

DATA=Path('D:/CodexData/RinkSpace_Data')


def _write(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def _export(owner,report,request,directory,prototype):
    # Heavy review imports are intentionally behind an explicit request.
    import numpy as np
    from .official_static_surface_dataset import reference
    from . import review_transverse_fit as core
    from .first_shot_review import FirstShotReview
    from . import parallel_first_shot_review as parallel
    import sys
    if str(prototype) not in sys.path:sys.path.insert(0,str(prototype))
    import native_camera_review as native
    if Path(native.__file__).resolve()!=prototype/'native_camera_review.py':
        raise ValueError('Live review prototype module differs from its explicit directory')
    ReviewStateMixin,ReviewBundle=native.ReviewStateMixin,native.ReviewBundle

    canonical=report['canonical_checkpoint']
    if reference(canonical['path'])!={k:canonical[k] for k in ('path','sha256')}:
        raise ValueError('Saved accepted canonical bytes changed before live review')
    with np.load(canonical['path'],allow_pickle=False) as saved:
        q=saved['q'].copy()
        identity=json.loads(str(saved['identity_json']))
        for key,value in owner.checkpoint_arrays().items():
            if not np.array_equal(saved['aux__'+key],value):
                raise ValueError('Live review canonical auxiliary state differs: '+key)
    qhash=hashlib.sha256(q.tobytes()).hexdigest()
    if (q.shape!=(7674,) or qhash!=report['qaug_sha256']
            or identity['binding']!=owner.bundle.state.binding_reference
            or identity['association']!=owner.association_reference
            or report['binding']!=owner.bundle.state.binding_reference):
        raise ValueError('Live review must use the exact current saved same-basis state')
    frames=tuple(request.get('check_frames',(1,842,1248)))
    if (len(frames)!=3 or any(type(i) is not int or not 1<=i<=1248 for i in frames)
            or len(set(frames))!=3):
        raise ValueError('Transport requires exactly three distinct native first-shot frames')
    proxy=core._review_bundle(owner,ReviewStateMixin,ReviewBundle)
    proof=owner.bundle.source_revision_proof
    if proof is not None and 'added_native_readouts' not in proof:
        proxy.source_revision_proof=dict(proof,added_native_readouts=
            proof['original_authenticated_basis']['added_native_readouts'])
    try:
        owner.model.install_dispatch()
        geometry=core._geometry_identity(owner,proxy,q)
        cameras=core._camera_identity(owner,proxy,q,frames)
        review_identity=dict(schema='rinkspace.live-owner-saved-canonical-review.v1',
            archive=deepcopy(canonical),q_sha256=qhash,parameter_count=len(q),
            binding=deepcopy(owner.bundle.state.binding_reference),association=deepcopy(owner.association_reference),
            source_producers=deepcopy(owner.source_producers),geometry=geometry,
            live_source_inventory_reused_without_reconstruction=True)
        view=FirstShotReview(proxy,q,review_identity)
        pack=parallel.pack_review(view,owner,prototype,check_frames=frames)
        pack_ref=parallel.save_pack(pack,directory/'review_pack.pkl')
        transport=parallel.validate_transport(view,pack_ref,directory/'transport_QA',workers=1)
        return dict(canonical=deepcopy(canonical),qaug_sha256=qhash,pack=pack_ref,
            transport_check=transport,camera_checks=cameras,geometry=geometry,
            all_1248_native_cameras_and_current_UV_retained=True,
            full_source_bundle_reconstructed=False,live_owner_closed=False,
            actual_model_and_full_geometry_checks_retained=True,video_generated=False,accuracy_accepted=False)
    finally:
        # Core review uses the same model. Restore its explicit dispatch and the
        # accepted shared-geometry context even when optional drawing fails.
        try:owner.model.install_dispatch()
        finally:owner.state._activate(q)


def install_live_review_export(owner,*,output_directory,prototype_dir,annotation_module,entry_reference=None):
    """Install before solver/pool startup; default checkpoints do no review work."""
    from .official_static_surface_dataset import reference
    output=Path(output_directory).resolve()
    if not output.is_relative_to(DATA.resolve()):
        raise ValueError('Live review diagnostics must stay in Data')
    prototype=Path(prototype_dir).resolve()
    request_path=output/'review_request.json'
    producer=reference(__file__)
    receipt=dict(schema='rinkspace.optional-live-owner-review-export.v1',producer=producer,
        entry_reference=deepcopy(entry_reference),request_path=str(request_path),enabled=True,
        source_bundle_reconstruction=False,default_without_request_heavy_work=False,
        three_frame_transport_workers=1,video_generated=False,failures_are_diagnostic_only=True)
    previous_identity=owner.checkpoint_identity
    def identity(adapter):
        result=previous_identity();result['optional_live_owner_review_export']=deepcopy(receipt)
        return result
    owner.checkpoint_identity=MethodType(identity,owner)
    if producer not in owner.source_producers:owner.source_producers.append(producer)
    previous=annotation_module.annotate_current_report
    consumed=set()
    def annotate(report,adapter):
        result=previous(report,adapter)
        if (adapter is not owner or not isinstance(report,dict)
                or 'canonical_checkpoint' not in report or not request_path.is_file()):
            return result
        started=time.perf_counter();request=None;record=None
        try:
            payload=request_path.read_bytes()
            request=json.loads(payload)
            if not isinstance(request,dict):raise ValueError('A review request object is required')
            request_id=request.get('id')
            if not isinstance(request_id,str) or not request_id:
                raise ValueError('An explicit nonempty review request id is required')
            token=hashlib.sha256(request_id.encode('utf-8')).hexdigest()
            bank=output/'live_review_exports';record=bank/'requests'/(token+'.json')
            if token in consumed or record.exists():return result
            consumed.add(token);record.parent.mkdir(parents=True,exist_ok=True)
            event=dict(request_id=request_id,request_sha256=hashlib.sha256(payload).hexdigest(),
                producer=producer,canonical=deepcopy(report['canonical_checkpoint']),status='running',
                diagnostic_only=True,original_fit_parameters_loss_and_control_unchanged=True)
            _write(record,event)
            directory=bank/(token[:16]+'_'+report['qaug_sha256'][:12])
            directory.mkdir(exist_ok=False)
            event.update(_export(owner,report,request,directory,prototype),status='complete')
        except Exception as error:
            event=dict(status='failed',request_id=request.get('id') if isinstance(request,dict) else None,
                producer=producer,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                diagnostic_only=True,optimizer_continues=True,video_generated=False)
            if record is None:record=output/'live_review_export_request_error.json'
        event['elapsed_seconds']=time.perf_counter()-started
        try:_write(record,event)
        except Exception:pass
        report['optional_live_review_export']=deepcopy(event)
        return result
    annotation_module.annotate_current_report=annotate
    owner.live_review_export_receipt=receipt
    return deepcopy(receipt)
