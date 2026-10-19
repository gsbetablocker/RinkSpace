"""Authenticate a complete accepted programme/static-camera parameter parent.

The historical programme constructor receives an explicitly labelled prefix
initializer. Its result is subsequently checked against the complete accepted
state and old source. No prefix extraction is called an accepted continuation.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from .first_shot_source_transition import reference, read, checkpoint_reference_verifier
from .first_shot_resume_cache import source_inventory
from .first_shot_safe_frame_export import _decode


def _same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if (a.dtype != b.dtype or a.shape != b.shape
            or not np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                                  np.ascontiguousarray(b).view(np.uint8))):
        raise ValueError('Complete accepted stage differs: '+label)


def read_complete_parent(manifest, *, pool_transition):
    report=read(manifest['accepted_parent_report'])
    ref=manifest['accepted_parent_checkpoint']
    if report['canonical_checkpoint']!=ref:
        raise ValueError('Explicit accepted report/canonical pair required')
    if reference(ref['path'])!=ref:raise ValueError('Complete parent canonical bytes changed')
    verifier=checkpoint_reference_verifier(pool_transition)
    verifier.verify_references(report)
    with np.load(ref['path'],allow_pickle=False)as saved:
        values={key:saved[key].copy()for key in saved.files}
    q=values['q'];identity=json.loads(str(values['identity_json']))
    verifier.verify_references(identity)
    if (q.dtype!=np.float64 or q.shape!=(7687,)or not np.isfinite(q).all()
            or hashlib.sha256(q.tobytes()).hexdigest()!=report['qaug_sha256']):
        raise ValueError('Latest complete accepted full7687 required')
    aux={key[5:]:value for key,value in values.items()if key.startswith('aux__')}
    if int(aux['programme_parameter_count'][0])!=7674:
        raise ValueError('Explicit programme prefix chart changed')
    _same(aux['static_venue_camera_columns'],np.arange(7674,7687),'wide13 chart')
    return dict(report=report,reference=deepcopy(ref),values=values,q=q,aux=aux,identity=identity)


def write_programme_prefix_initializer(parent, path):
    """Extract only the old constructor's chart, retaining every parent aux."""
    path=Path(path)
    if path.exists():raise ValueError('Prefix initializer must not overwrite an artifact')
    count=int(parent['aux']['programme_parameter_count'][0])
    identity=deepcopy(parent['identity'])
    identity.update(schema='rinkspace.explicit-programme-prefix-parameter-initializer.v1',
        accepted_state_claimed=False,parameter_initializer_only=True,
        accepted_complete_stage_parent=parent['reference'],
        parameter_count=count,full_parent_parameter_count=len(parent['q']),
        original_wide13_not_discarded_from_complete_new_stage=True)
    np.savez_compressed(path,q=parent['q'][:count].copy(),
        damping=parent['values']['damping'],
        next_line_search_fraction=parent['values']['next_line_search_fraction'],
        identity_json=np.array(json.dumps(identity,ensure_ascii=False,allow_nan=False)),
        **{'aux__'+key:value for key,value in parent['aux'].items()})
    return reference(path)


def authenticate_prefix_parameters(owner, parent):
    """This preliminary check does not claim full source authentication."""
    _same(owner.start,parent['q'][:len(owner.start)],'programme prefix')
    for key,value in owner.checkpoint_arrays().items():
        _same(value,parent['aux'][key],'original aux '+key)
    if (owner.seed_receipt['initial_damping']!=float(parent['values']['damping'])
            or owner.seed_receipt['initial_line_search_fraction']!=float(parent['values']['next_line_search_fraction'])):
        raise ValueError('Complete parent damping/fraction changed')
    return dict(schema='rinkspace.complete-stage-prefix-preliminary-authentication.v1',
        accepted_complete_parent=parent['reference'],prefix_parameter_bits_exact=True,
        all_original_programme_aux_exact=True,wide13_retained_separately=True,
        full_old_source_authentication_still_required_after_explicit_group_quarantine=True,
        old_source_r_J_or_world_cache_loaded=False)


def authenticate_old_stage_source(owner, old_factor, parent, *, inventory_reference, output):
    """Replay old37 on the one newly constructed programme owner, without J."""
    from z190_joint_source_solver import source_cost
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    q=parent['q'];prefix=q[:len(owner.start)]
    inventory,_=source_inventory(owner.bundle)
    if reference(inventory_reference['path'])!=inventory_reference:
        raise ValueError('Complete parent programme inventory bytes changed')
    with np.load(inventory_reference['path'],allow_pickle=False)as saved:
        if set(saved.files)!=set(inventory):raise ValueError('Complete parent inventory fields changed')
        for key,value in inventory.items():_same(value,saved[key],'inventory '+key)
    old_rows=len(owner.scales)
    _same(parent['values']['scales'][:old_rows],owner.scales,'programme scales')
    _same(parent['values']['quadratic'][:old_rows],owner.quadratic,'programme mask')
    _same(parent['values']['scales'][old_rows:],old_factor.scales,'old static scales')
    _same(parent['values']['quadratic'][old_rows:],old_factor.quadratic,'old static mask')
    _same(parent['aux']['static_venue_initial_camera_state'],old_factor.camera_policy.encode(),
          'historical initial static-camera auxiliary values')
    _same(parent['aux']['static_venue_world_columns'],old_factor.geometry_columns,'static world columns')
    actual_identity=old_factor.identity
    expected=parent['identity']['persistent_static_venue_stage']['static_venue_factor']
    if actual_identity!=expected:raise ValueError('Previous static factor identity changed')
    owner.guard(prefix)
    blocks=list(owner.source_blocks(prefix,False))
    r=np.concatenate([b.r for b in blocks]);raw=owner.raw(prefix)
    snapshot_ref=parent['report']['snapshot']
    if reference(snapshot_ref['path'])!=snapshot_ref:raise ValueError('Complete parent snapshot bytes changed')
    with np.load(snapshot_ref['path'],allow_pickle=False)as snapshot:
        _same(snapshot['q_augmented'],q,'complete snapshot canonical q')
        _same(snapshot['programme_q_augmented'],prefix,'snapshot programme prefix')
        _same(snapshot['programme_point_errors_px'],raw,'all programme raw point errors')
        _same(snapshot['source_point_ids'],inventory['source_point_ids'],'all snapshot IDs')
        _same(snapshot['native_pts_per_point'],inventory['native_pts'],'all snapshot PTS')
        for key,value in parent['aux'].items():_same(snapshot['aux__'+key],value,'snapshot aux '+key)
    wide=old_factor.block(q,False)
    review=read(parent['report']['review']['static'])
    if reference(review['arrays']['path'])!=review['arrays']:raise ValueError('Old static review arrays changed')
    with np.load(review['arrays']['path'],allow_pickle=False)as saved:
        payload=_decode(review['payload'],{key:saved[key].copy()for key in saved.files})
    _same(payload['q'],q,'old static complete q')
    _same(payload['wide_normalized_residual'],wide.r,'every old37 residual')
    _same(payload['wide_raw_pixel_vectors'],(wide.r*wide.scales).reshape(-1,2),'every old37 pixel vector')
    programme_cost=math.fsum(source_cost(b.r,b.quadratic,b.loss_weights)for b in blocks)
    wide_cost=source_cost(wide.r,wide.quadratic,wide.loss_weights)
    if (abs(programme_cost-parent['report']['programme_source_cost'])>1e-8
            or abs(wide_cost-parent['report']['wide_source_cost'])>1e-8):
        raise ValueError('Complete accepted programme/static costs changed')
    np.savez(output/'actual_parent_source.npz',q=q,programme_r=r,programme_raw=raw,wide_r=wide.r,
        programme_scales=owner.scales,programme_quadratic=owner.quadratic)
    proof=dict(schema='rinkspace.actual-full-accepted-programme-static-parent-replay.v1',passed=True,
        accepted_parent=parent['reference'],q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        programme_source_points=len(raw),programme_rows=len(r),wide_rows=len(wide.r),
        all_complete_q_and_aux_preserved=True,all_programme_UV_ID_PTS_and_raw_exact=True,
        all_old_static_residuals_and_pixel_vectors_bitexact=True,
        original_W_masks_scales_and_loss_weights_preserved=True,
        programme_source_cost=programme_cost,wide_source_cost=wide_cost,
        actual_source_arrays=reference(output/'actual_parent_source.npz'),
        old_J_or_world_derivatives_loaded=False,optimizer_called=False,producer=reference(__file__))
    (output/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return proof
