"""Explicit reviewed group/point corrections at one preformal source boundary.

This module never initializes an owner or chooses an identity from residuals.
The caller authenticates the old complete source, closes its pools and supplies
the separately reviewed group and ordinary-point manifests.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
import hashlib
import json

import numpy as np

from .first_shot_source_transition import reference,read,_digest


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def preflight_material_source(inventory,*,binding,group_manifest_reference,
        ordinary_revision_reference,selected_frames,old_formal_rows,allow_preview=False):
    """Check immutable addresses and actual complete W before owner creation.

    Selected safe frames provide actual W checks where available. Remaining
    groups receive the complete ordered-member/W check on the one real owner,
    before any source installation. Ordinary changes outside safe frames get
    their full descriptor check there too; this pass checks labels and UV.
    """
    from .first_shot_alignment_monitor import _target_label
    from .first_shot_revision import SOURCE_SCHEMA,_target
    groups=read(group_manifest_reference);ordinary=read(ordinary_revision_reference)
    if not allow_preview and (groups.get('preview_only_not_applied')or ordinary.get('preview_only_not_applied')):
        raise ValueError('A preview source review cannot authorize a production transition')
    if (ordinary.get('schema')!=SOURCE_SCHEMA
            or ordinary.get('source_video_sha256')!=binding['source']['sha256']
            or ordinary.get('source_time_base')!=binding['source_clock']['time_base']
            or ordinary.get('original_source_UV_IDs_and_approvals_modified')is not False):
        raise ValueError('Ordinary revision must identify unchanged official native source')
    base=read(ordinary['base_binding'])
    if any(base[key]!=binding[key]for key in('complete_fit_input','model','source')):
        raise ValueError('Ordinary revision belongs to another binding')
    if any(ordinary.get(key)for key in('added_native_readouts','correlated_operator_replacements',
            'correlated_estimator_replacements')):
        raise ValueError('This explicit material transition admits no new UV or operators')
    ids=inventory['source_point_ids'];pts=inventory['native_pts'];uv=inventory['source_uv']
    addresses={(int(t),str(i)):j for j,(t,i)in enumerate(zip(pts,ids,strict=True))}
    if len(addresses)!=len(ids):raise ValueError('Immutable source contains duplicate native addresses')
    actual={};actual_groups={};actual_grouped=set();safe_pts=set()
    for frame in selected_frames:
        safe_pts.add(int(frame['native_pts']))
        flattened=[(row,j)for row in frame['rows']for j in range(len(row['uv']))]
        for row,j in flattened:
            key=int(frame['native_pts']),str(row['source_point_ids'][j])
            if key not in addresses:raise ValueError('Safe source metadata has a foreign native address')
            index=addresses[key]
            if (not np.array_equal(row['uv'][j],uv[index])
                    or _target_label(row['target'])!=str(inventory['source_target_labels'][index])):
                raise ValueError('Safe metadata and immutable current source disagree')
            actual[key]=(row,j)
        for group in frame['whitening_groups']:
            members=tuple(str(flattened[i][0]['source_point_ids'][flattened[i][1]])for i in group['point_indices'])
            key=int(frame['native_pts']),members
            actual_groups[key]=group['operator'].matrix
            actual_grouped.update((key[0],pid)for pid in members)
    evidence={};removed=set();removed_formal=0;checked_groups=[];pending_groups=[];saved_origins=[];datasets={}
    def collect(refs):
        for ref in refs:
            if ref['path']in evidence and evidence[ref['path']]!=ref:
                raise ValueError('Conflicting hashes for the same material evidence')
            evidence[ref['path']]=ref
    def saved_origin(item,key):
        origin=item.get('operator_source')
        if origin is None:return False
        ref=origin['frame_dataset'];token=ref['path'],ref['sha256']
        if token not in datasets:
            data=read(ref)
            datasets[token]={int(frame['index']):frame for frame in data.get('frames',[])+data.get('appended_frames',[])}
        frame=datasets[token][int(origin['frame'])]
        if int(frame['native_pts'])!=key[0]:raise ValueError('Saved W frame/PTS origin differs')
        group=frame['whitening_groups'][int(origin['whitening_group_index'])];operator=group['operator']
        matrix=np.asarray(operator['whitening_matrix'],np.float64)
        if (operator['group_id']!=origin['group_id']or tuple(map(str,operator['source_point_ids']))!=key[1]
                or _sha(matrix)!=item['operator_matrix_sha256']or list(matrix.shape)!=item['operator_matrix_shape']
                or frame['source_packet']!=origin['source_packet']
                or group['source_evidence']['source_readout_packet']!=origin['native_readout_packet']):
            raise ValueError('Expected W is not the unchanged saved complete source operator')
        sigma=np.asarray([frame['source_scales_by_point_id'][pid]for pid in key[1]],np.float64)
        if _sha(sigma)!=item['original_sigma_px_sha256']:
            raise ValueError('Expected group sigma differs from its saved complete source origin')
        collect((ref,origin['source_packet'],origin['native_readout_packet']))
        saved_origins.append(key[0]);return True
    specifications=groups['specifications']
    if not specifications:raise ValueError('An explicit nonempty complete material quarantine is required')
    for item in specifications:
        members=tuple(map(str,item['source_point_ids']));key=int(item['native_pts']),members
        if (not members or len(set(members))!=len(members)or item.get('material_identity_confirmed')is not False
                or not item.get('reason')or not item.get('evidence_references')):
            raise ValueError('Distinct reviewed uncertain complete material groups required')
        shape=item['operator_matrix_shape'];digest=item['operator_matrix_sha256']
        if (not isinstance(shape,list)or len(shape)!=2 or any(type(n)is not int or n<=0 for n in shape)
                or shape[1]!=2*len(members)or not isinstance(digest,str)or len(digest)!=64
                or any(c not in '0123456789abcdef'for c in digest)):
            raise ValueError('Complete group requires exact source W shape and SHA256')
        origin_checked=saved_origin(item,key)
        if key in actual_groups:
            matrix=actual_groups[key]
            if _sha(matrix)!=digest or list(matrix.shape)!=shape:
                raise ValueError('Complete group W differs from actual safe source')
            checked_groups.append(key[0])
        elif key[0]in safe_pts:
            raise ValueError('Complete group ordered members differ from actual safe source')
        else:
            if not origin_checked:raise ValueError('Pending actual W requires authenticated saved packet/operator origin')
            pending_groups.append(key[0])
        current={(key[0],pid)for pid in members}
        if current&removed or not current<=addresses.keys():raise ValueError('Missing or repeated complete material group')
        removed.update(current);removed_formal+=shape[0]
        if 'source_UV_float64_sha256'in item:
            indices=[addresses[(key[0],pid)]for pid in members]
            if _sha(uv[indices])!=item['source_UV_float64_sha256']:
                raise ValueError('Complete group immutable source UV digest differs from review')
        collect(item['evidence_references'])
    changed=set();quarantined=set();full_descriptors=0
    for field,destination in(('identity_corrections',changed),('quarantined_correspondences',quarantined)):
        for item in ordinary.get(field,[]):
            key=int(item['native_pts']),str(item['source_point_id'])
            if (key in changed or key in quarantined or key in removed or key not in addresses
                    or key in actual_grouped):raise ValueError('Ordinary material edit has missing, duplicate or correlated address')
            index=addresses[key]
            if (not np.array_equal(np.asarray(item['original_UV'],float),uv[index])
                    or _target_label(item['original_target'])!=str(inventory['source_target_labels'][index])):
                raise ValueError('Ordinary material original UV/current immutable identity differs')
            if ('frame'in item and int(item['frame'])!=int(inventory['frame_indices'][index])):
                raise ValueError('Ordinary material source frame and native PTS disagree')
            if (item.get('authority')!='agent_rgb_review'or not item.get('reason')
                    or item.get('identity_confirmed_by_user')is True or not item.get('evidence_references')):
                raise ValueError('Actual RGB review cannot invent user confirmation')
            if field=='identity_corrections':_target(item['target'])
            if key in actual:
                row,j=actual[key]
                if item['original_target']!=row['target']or len(row.get('native_line_queries',())):
                    raise ValueError('Ordinary edit differs from actual full target or native query scope')
                full_descriptors+=1
            destination.add(key)
            collect(item['evidence_references'])
    for ref in evidence.values():
        if reference(ref['path'])!=ref:raise ValueError('Material RGB evidence bytes changed')
    return dict(schema='rinkspace.offline-reviewed-material-source-preflight.v1',passed=True,
        group_manifest=group_manifest_reference,ordinary_revision=ordinary_revision_reference,
        old_source_points=len(ids),source_points=len(ids)-len(removed)-len(quarantined),
        old_formal_rows=int(old_formal_rows),formal_rows=int(old_formal_rows)-removed_formal-2*len(quarantined),
        complete_groups=len(specifications),complete_group_points=len(removed),complete_group_formal_rows=removed_formal,
        identity_corrections=len(changed),ordinary_quarantines=len(quarantined),
        every_ordinary_original_UV_current_label_and_native_address_checked=True,
        ordinary_full_descriptors_crosschecked_against_safe_frames=full_descriptors,
        remaining_ordinary_full_descriptor_check_required_on_actual_owner=True,
        actual_complete_W_and_ordered_members_checked_groups=len(checked_groups),
        actual_complete_W_checked_native_pts=checked_groups,
        pending_actual_owner_W_and_ordered_members_groups=len(pending_groups),
        pending_actual_owner_W_native_pts=pending_groups,
        expected_W_authenticated_saved_operator_origin_groups=len(saved_origins),
        every_quarantined_group_actual_W_and_ordered_members_checked=not pending_groups,
        all_groups_actual_W_check_required_before_source_installation=True,
        source_graph_or_owner_constructed=False,optimizer_called=False,preview_only=bool(allow_preview))


@contextmanager
def parent_source_without_raw_retention(programme,original_factory):
    """Authenticate the old source without filling its same-q raw cache."""
    receipt=programme.raw_retention_receipt
    if receipt['verified'] or receipt['cache_hits']:
        raise ValueError('Old-source raw retention must still be empty at the explicit revision boundary')
    previous=programme.source_blocks;start=len(programme.worker_receipts)
    programme.source_blocks=MethodType(original_factory,programme)
    try:
        yield
    finally:
        programme.source_blocks=previous
        for worker in programme.worker_receipts[start:]:
            value=worker.pop('_same_state_cost_raw',None)
            if value is not None:
                worker['old_parent_only_raw_sha256']=_sha(value)
                worker['old_parent_only_raw_points']=len(value)
        if receipt['verified'] or receipt['cache_hits']:
            raise ValueError('Old-source authentication populated the new-source same-q raw cache')


def close_source_revision_pools(programme):
    """Quiescent preformal boundary, preserving receipts and the loaded graph."""
    closed=[];seen=set()
    for owner,name in ((programme,'pool'),(programme.bundle,'_source_pool'),
                       (programme.bundle,'_geometry_pool')):
        pool=getattr(owner,name)
        if pool is not None and id(pool)not in seen:
            pool.shutdown(wait=True);seen.add(id(pool));closed.append(name)
        setattr(owner,name,None)
    if hasattr(programme,'_geometry_pool_delta'):programme._geometry_pool_delta=None
    return dict(closed_execution_pools=closed,all_shutdown_waited=True,
        source_graph_or_parameter_state_reconstructed=False,old_worker_receipts_preserved=True)


def transform_material_source(frames,*,group_manifest_reference,ordinary_revision_reference,binding,policy):
    """Use the original whole-W and source-addressed ordinary public utilities."""
    from .first_shot_group_quarantine import quarantine_complete_groups
    from .first_shot_ordinary_delta import apply_ordinary_delta
    group_manifest=read(group_manifest_reference)
    specifications={(int(item['native_pts']),tuple(map(str,item['source_point_ids']))):item
        for item in group_manifest['specifications']}
    addressed_pts={key[0]for key in specifications};origin_verified=[];origin_datasets={};normalized_origin_metadata=[]
    for frame in frames:
        pts=int(frame['native_pts'])
        if pts not in addressed_pts:continue
        flattened=[(row,j)for row in frame['rows']for j in range(len(row['uv']))]
        for group in frame['whitening_groups']:
            members=[flattened[int(i)]for i in group['point_indices']]
            ids=tuple(str(row['source_point_ids'][j])for row,j in members);key=pts,ids
            if key not in specifications:continue
            item=specifications[key]
            if ('source_UV_float64_sha256'in item and _sha(np.asarray([row['uv'][j]for row,j in members]))
                    !=item['source_UV_float64_sha256']):raise ValueError('Actual complete group source UV differs from review')
            if ('original_sigma_px_sha256'in item and _sha(np.asarray([row['sigma'][j]for row,j in members],np.float64))
                    !=item['original_sigma_px_sha256']):raise ValueError('Actual complete group source sigma differs from review')
            origin=item.get('operator_source')
            if origin is not None:
                # The public runtime group normalizer retains the operator and
                # point map, not necessarily the literal source_evidence field.
                # Authenticate that origin separately, then require the entire
                # actual runtime operator and source values to match it exactly.
                dataset_ref=origin['frame_dataset'];token=dataset_ref['path'],dataset_ref['sha256']
                if token not in origin_datasets:
                    dataset=read(dataset_ref)
                    origin_datasets[token]={int(f['index']):f for f in dataset.get('frames',[])+dataset.get('appended_frames',[])}
                literal_frame=origin_datasets[token][int(origin['frame'])]
                literal=literal_frame['whitening_groups'][int(origin['whitening_group_index'])]
                operator=literal['operator'];matrix=np.asarray(operator['whitening_matrix'],np.float64)
                source=group.get('source_evidence')
                if (int(literal_frame['native_pts'])!=pts or int(literal_frame['index'])!=int(frame['index'])
                        or operator['group_id']!=origin['group_id']or tuple(map(str,operator['source_point_ids']))!=ids
                        or literal_frame['source_packet']!=origin['source_packet']
                        or literal['source_evidence']['source_readout_packet']!=origin['native_readout_packet']
                        or _sha(matrix)!=item['operator_matrix_sha256']or list(matrix.shape)!=item['operator_matrix_shape']
                        or group['operator'].group_id!=origin['group_id']or frame['source_packet']!=origin['source_packet']
                        or group['operator'].matrix.shape!=matrix.shape
                        or group['operator'].matrix.tobytes()!=matrix.tobytes()
                        or(source is not None and source['source_readout_packet']!=origin['native_readout_packet'])):
                    raise ValueError('Actual complete operator/source packet differs from authenticated saved origin')
                for origin_ref in (origin['source_packet'],origin['native_readout_packet']):
                    if reference(origin_ref['path'])!=origin_ref:raise ValueError('Original material operator source bytes changed')
                if source is None:normalized_origin_metadata.append(pts)
                origin_verified.append(pts)
    for item in group_manifest['specifications']:
        for evidence in item['evidence_references']:
            if reference(evidence['path'])!=evidence:raise ValueError('Group material RGB evidence bytes changed')
    after_groups,group_proof=quarantine_complete_groups(frames,group_manifest['specifications'],
        evidence_reference=group_manifest_reference)
    revised,ordinary_proof=apply_ordinary_delta(after_groups,ordinary_revision_reference,binding,policy=policy)
    ordinary_indices=ordinary_proof['complete_source_comparison']['_retained_formal_indices']
    formal_indices=group_proof['retained_formal_indices'][ordinary_indices]
    proof=dict(schema='rinkspace.combined-reviewed-material-source-revision.v1',
        group_manifest=group_manifest_reference,ordinary_revision=ordinary_revision_reference,
        group_quarantine={k:deepcopy(v)for k,v in group_proof.items()if k!='retained_formal_indices'},
        ordinary_identity={k:deepcopy(v)for k,v in ordinary_proof.items()if k!='complete_source_comparison'},
        complete_ordinary_comparison={k:deepcopy(v)for k,v in ordinary_proof['complete_source_comparison'].items()
            if k!='_retained_formal_indices'},
        old_source_points=sum(f['source_point_count']for f in frames),
        source_points=sum(f['source_point_count']for f in revised),formal_rows=len(formal_indices),
        retained_old_formal_indices_sha256=_sha(formal_indices),
        source_UV_ID_clock_unchanged_except_explicit_archived_members=True,
        correlated_W_not_sliced_or_reestimated=True,automatic_nearest_assignment=False,
        actual_source_UV_sigma_packet_and_operator_origin_verified_groups=len(origin_verified),
        runtime_groups_without_literal_source_evidence=len(normalized_origin_metadata),
        absent_runtime_origin_authenticated_via_literal_and_full_operator_bytes=True,
        source_equations_unchanged_except_explicit_material_identities=True)
    return revised,formal_indices,proof


def revision_comparison_maps(old_frames,new_frames,formal_indices,ordinary_revision_reference):
    """Locate exactly the retained components whose named material changed."""
    from .first_shot_fit import FrozenWhitening
    revision=read(ordinary_revision_reference)
    changed={(int(item['native_pts']),str(item['source_point_id']))for item in revision.get('identity_corrections',[])}
    old_addresses={};affected=[];point_offset=formal_offset=0
    for frame in old_frames:
        w=FrozenWhitening(frame);cursor=0;local_changed=[]
        for row in frame['rows']:
            for pid in row['source_point_ids']:
                key=int(frame['native_pts']),str(pid)
                if key in old_addresses:raise ValueError('Repeated old source native address')
                old_addresses[key]=point_offset+cursor
                if key in changed:local_changed.extend((2*cursor,2*cursor+1))
                cursor+=1
        local=np.isin(w.ordinary,np.asarray(local_changed,int))
        if int(local.sum())!=len(local_changed):raise ValueError('Changed ordinary identity intersects correlated W')
        affected.extend((formal_offset+w.ordinary_start+np.flatnonzero(local)).tolist())
        point_offset+=cursor;formal_offset+=w.count
    point_map=[];found=set()
    for frame in new_frames:
        for row in frame['rows']:
            for pid in row['source_point_ids']:
                key=int(frame['native_pts']),str(pid)
                if key not in old_addresses or key in found:raise ValueError('Revised source invented or duplicated a native address')
                point_map.append(old_addresses[key]);found.add(key)
    if not changed<=found:raise ValueError('Changed material identity disappeared from revised source')
    old_point_changed=np.zeros(point_offset,bool)
    for key in changed:old_point_changed[old_addresses[key]]=True
    point_map=np.asarray(point_map,np.int64);formal_indices=np.asarray(formal_indices,np.int64)
    formal_changed=np.isin(formal_indices,np.asarray(affected,np.int64))
    if int(formal_changed.sum())!=2*len(changed):raise ValueError('Changed ordinary formal component map is incomplete')
    return dict(retained_old_point_indices=point_map,retained_old_formal_indices=formal_indices,
        retained_point_identity_changed=old_point_changed[point_map],
        retained_formal_identity_changed=formal_changed)


def install_revised_source(programme,frames,formal_indices,*,revision_reference,proof):
    """Rebuild source maps on the existing physical/camera parameter owner."""
    from .first_shot_fit import FrozenWhitening,_target_key
    if any(getattr(owner,name)is not None for owner,name in ((programme,'pool'),
            (programme.bundle,'_source_pool'),(programme.bundle,'_geometry_pool'))):
        raise ValueError('Every execution pool must close before the source revision')
    if getattr(programme,'initial_J_manifest',None)is not None:
        raise ValueError('A corrected source cannot inherit an initial Jacobian')
    if programme.raw_retention_receipt['verified'] or programme.raw_retention_receipt['cache_hits']:
        raise ValueError('A corrected source cannot inherit an old same-q raw cache')
    old_frames=programme.bundle.frames
    coverage=lambda current:[r for f in current for r in f['rows']if r.get('_continuous_world_line')]
    if _digest(coverage(old_frames))!=_digest(coverage(frames)):
        raise ValueError('This correction needs an explicit native-line coverage reconstruction')
    descriptors=OrderedDict();whitening=[];ledgers=[]
    for frame,previous in zip(frames,programme.bundle.ledgers,strict=True):
        for row in frame['rows']:
            if row['_target_key']!=_target_key(row['target']):
                raise ValueError('Corrected source target key does not match its full descriptor')
            descriptors.setdefault(row['_target_key'],row['target'])
        w=FrozenWhitening(frame);whitening.append(w);h=hashlib.sha256()
        for row in frame['rows']:
            h.update(json.dumps(row['source_point_ids'],separators=(',',':')).encode())
            h.update(row['uv'].tobytes());h.update(row['sigma'].tobytes())
        for group in frame['whitening_groups']:h.update(group['operator'].matrix.tobytes())
        ledgers.append(dict(previous,source_points=frame['source_point_count'],formal_rows=w.count,
            source_inventory_sha256=h.hexdigest(),explicit_material_source_revision=revision_reference))
    scales=np.concatenate([w.scales for w in whitening]);formal_indices=np.asarray(formal_indices,np.int64)
    if not np.array_equal(scales,programme.scales[formal_indices]):
        raise ValueError('Source correction changed retained formal scales/order')
    mask=programme.quadratic[formal_indices].copy();bundle=programme.bundle
    target_proof=bundle.state.validate_initial_targets(tuple(descriptors.values()))
    bundle.frames,bundle.whitening,bundle.ledgers=frames,whitening,ledgers
    bundle.descriptors,bundle.keys=descriptors,tuple(descriptors)
    bundle.target_descriptors=tuple(descriptors.values());bundle.initial_target_proof=target_proof
    bundle.scales,bundle.quadratic=scales,mask
    programme.scales,programme.quadratic=scales.copy(),mask.copy()
    programme.formal_offsets=np.r_[0,np.cumsum([w.count for w in whitening])]
    for owner,name in ((bundle,'_world_cache'),(bundle,'_finite_coverage_cache'),
                       (programme,'transverse_world_cache'),(programme,'guard_cache')):
        getattr(owner,name).clear()
    programme.metric=None;programme._initial_complete_J_reuse=False
    receipt=dict(proof,manifest=revision_reference,producer=reference(__file__),
        new_source_value_world_derivative_and_whitening_cache_identity=True,
        old_source_r_J_world_D_or_raw_cache_reused=False,
        source_map_whitening_descriptors_ledgers_offsets_rebuilt=True)
    for ref in (reference(__file__),revision_reference,proof['group_manifest'],proof['ordinary_revision']):
        if ref not in programme.source_producers:programme.source_producers.append(deepcopy(ref))
    previous_identity=programme.checkpoint_identity
    def identity(owner):
        result=previous_identity();result['explicit_postaccepted_material_source_revision']=deepcopy(receipt)
        result['initial_exact_complete_current_state_full_J_reuse']=False
        return result
    programme.checkpoint_identity=MethodType(identity,programme)
    programme.explicit_material_source_receipt=receipt
    return receipt
