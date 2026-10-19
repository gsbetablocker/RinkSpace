from collections import OrderedDict
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from test_first_shot_revision import source,save
from calibration.program_tracking.first_shot_fit import FrozenWhitening
from calibration.program_tracking.first_shot_material_source_revision import (
    parent_source_without_raw_retention,close_source_revision_pools,
    transform_material_source,install_revised_source,preflight_material_source,revision_comparison_maps)


def test_old_parent_authentication_cannot_poison_same_q_new_source_raw():
    cache={};q=np.array([1.,2.])
    owner=SimpleNamespace(input=np.array([5.,6.,7.]),worker_receipts=[],
        raw_retention_receipt={'verified':False,'cache_hits':0})
    def original(adapter,value,jacobian):
        adapter.worker_receipts.append({'_same_state_cost_raw':adapter.input.copy()})
        yield adapter.input.copy()
    def accelerated(value,jacobian):
        cache[value.tobytes()]=owner.input.copy()
        yield owner.input.copy()
    owner.source_blocks=accelerated
    with parent_source_without_raw_retention(owner,original):
        np.testing.assert_array_equal(next(owner.source_blocks(q,False)),[5.,6.,7.])
    assert not cache
    assert '_same_state_cost_raw'not in owner.worker_receipts[0]
    assert owner.worker_receipts[0]['old_parent_only_raw_points']==3
    owner.input=np.array([9.,10.])
    next(owner.source_blocks(q,False))
    np.testing.assert_array_equal(cache[q.tobytes()],[9.,10.])
    owner.raw_retention_receipt['verified']=True
    with pytest.raises(ValueError,match='still be empty'):
        with parent_source_without_raw_retention(owner,original):pass


def test_execution_pool_boundary_preserves_loaded_state_and_receipts():
    calls=[];pool=SimpleNamespace(shutdown=lambda **kw:calls.append(kw))
    q=np.array([1.,-0.]);receipts=[{'parent_only':True}]
    owner=SimpleNamespace(pool=pool,bundle=SimpleNamespace(_source_pool=pool,_geometry_pool=None),
        start=q,worker_receipts=receipts,_geometry_pool_delta=np.ones(5))
    proof=close_source_revision_pools(owner)
    assert calls==[{'wait':True}]
    assert owner.start is q and owner.worker_receipts is receipts
    assert owner.pool is None and owner.bundle._source_pool is None
    assert owner._geometry_pool_delta is None
    assert proof['source_graph_or_parameter_state_reconstructed']is False


def test_combined_whole_W_and_ordinary_identity_rebuilds_exact_retained_source(source,tmp_path):
    row=source.row
    source.frame['rows']=[dict(row,uv=row['uv'][i:i+1],sigma=row['sigma'][i:i+1],
        source_point_ids=[row['source_point_ids'][i]],original_vertex_indices=[i])for i in range(4)]
    old_w=FrozenWhitening(source.frame);matrix=source.frame['whitening_groups'][0]['operator'].matrix
    group=save(tmp_path/'group.json',{'specifications':[dict(native_pts=256256,
        source_point_ids=['p0','p1'],operator_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        operator_matrix_shape=list(matrix.shape),material_identity_confirmed=False,
        reason='Unresolved original material family',evidence_references=[source.evidence])]})
    item=source.item(2);item['target']=dict(row['target'],contour_key='component018')
    source.revision['identity_corrections']=[item]
    revision=save(tmp_path/'ordinary.json',source.revision)
    import json
    binding=json.loads(open(source.revision['base_binding']['path'],encoding='utf8').read())
    frames,indices,proof=transform_material_source([source.frame],group_manifest_reference=group,
        ordinary_revision_reference=revision,binding=binding,policy=None)
    np.testing.assert_array_equal(indices,[2,3,4,5])
    assert frames[0]['rows'][0]['target']['contour_key']=='component018'
    assert frames[0]['rows'][1]['target']==row['target']
    np.testing.assert_array_equal(np.concatenate([r['uv']for r in frames[0]['rows']]),row['uv'][2:])
    assert proof['source_points']==2 and proof['formal_rows']==4
    maps=revision_comparison_maps([source.frame],frames,indices,revision)
    np.testing.assert_array_equal(maps['retained_old_point_indices'],[2,3])
    np.testing.assert_array_equal(maps['retained_point_identity_changed'],[True,False])
    np.testing.assert_array_equal(maps['retained_formal_identity_changed'],[True,True,False,False])
    owner=SimpleNamespace(start=np.array([1.,-0.]),pool=None,initial_J_manifest=None,
        raw_retention_receipt={'verified':False,'cache_hits':0},
        scales=old_w.scales,quadratic=np.zeros(old_w.count,bool),
        source_producers=[],checkpoint_identity=lambda:{'original':'preserved'},
        guard_cache=OrderedDict(old=True),transverse_world_cache=OrderedDict(old=True),metric=object())
    owner.bundle=SimpleNamespace(frames=[source.frame],_source_pool=None,_geometry_pool=None,
        ledgers=[{'original_ledger':True}],_world_cache=OrderedDict(old=True),
        _finite_coverage_cache=OrderedDict(old=True),
        state=SimpleNamespace(validate_initial_targets=lambda values:{'targets':len(values)}))
    complete=save(tmp_path/'combined.json',{'group':group,'ordinary':revision})
    before=owner.start.view(np.uint64).copy()
    installed=install_revised_source(owner,frames,indices,revision_reference=complete,proof=proof)
    np.testing.assert_array_equal(owner.start.view(np.uint64),before)
    np.testing.assert_array_equal(owner.scales,old_w.scales[indices])
    assert not owner.guard_cache and not owner.transverse_world_cache
    assert not owner.bundle._world_cache and not owner.bundle._finite_coverage_cache
    assert owner.formal_offsets.tolist()==[0,4]
    assert owner.checkpoint_identity()['explicit_postaccepted_material_source_revision']==installed
    assert installed['old_source_r_J_world_D_or_raw_cache_reused']is False


def test_offline_preflight_checks_actual_whole_W_and_rejects_preview_for_production(source,tmp_path):
    from calibration.program_tracking.first_shot_resume_cache import source_inventory
    import json
    inventory,_=source_inventory(SimpleNamespace(frames=[source.frame]))
    matrix=source.frame['whitening_groups'][0]['operator'].matrix
    specification=dict(native_pts=256256,source_point_ids=['p0','p1'],
        operator_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),operator_matrix_shape=list(matrix.shape),
        material_identity_confirmed=False,reason='Unresolved material family',evidence_references=[source.evidence])
    group=dict(specifications=[specification],preview_only_not_applied=True)
    group_ref=save(tmp_path/'group.json',group)
    item=source.item(2);item['target']=dict(source.row['target'],contour_key='component018')
    source.revision['identity_corrections']=[item]
    ordinary=save(tmp_path/'ordinary.json',source.revision)
    binding=json.loads(open(source.revision['base_binding']['path'],encoding='utf8').read())
    options=dict(binding=binding,group_manifest_reference=group_ref,ordinary_revision_reference=ordinary,
        selected_frames=[source.frame],old_formal_rows=6)
    with pytest.raises(ValueError,match='preview'):preflight_material_source(inventory,**options)
    proof=preflight_material_source(inventory,**options,allow_preview=True)
    assert (proof['source_points'],proof['formal_rows'],proof['identity_corrections'])==(2,4,1)
    assert proof['ordinary_full_descriptors_crosschecked_against_safe_frames']==1
    assert proof['every_quarantined_group_actual_W_and_ordered_members_checked']is True
    dataset=save(tmp_path/'saved_dataset.json',{'frames':[dict(index=1,native_pts=256256,
        source_packet=source.evidence,source_scales_by_point_id={'p0':1.,'p1':2.},
        whitening_groups=[dict(operator=dict(group_id='fixture/complete',source_point_ids=['p0','p1'],
            whitening_matrix=matrix.tolist()),source_evidence=dict(source_readout_packet=source.evidence))])]})
    pending_group=json.loads(json.dumps(group));pending_item=pending_group['specifications'][0]
    pending_item['operator_source']=dict(frame_dataset=dataset,frame=1,whitening_group_index=0,
        group_id='fixture/complete',source_packet=source.evidence,native_readout_packet=source.evidence)
    pending_item['original_sigma_px_sha256']=hashlib.sha256(np.array([1.,2.]).tobytes()).hexdigest()
    pending_ref=save(tmp_path/'pending_group.json',pending_group)
    pending=preflight_material_source(inventory,**dict(options,selected_frames=[],group_manifest_reference=pending_ref),allow_preview=True)
    assert pending['pending_actual_owner_W_and_ordered_members_groups']==1
    assert pending['every_quarantined_group_actual_W_and_ordered_members_checked']is False
    assert pending['expected_W_authenticated_saved_operator_origin_groups']==1
    specification['operator_matrix_sha256']='0'*64
    options['group_manifest_reference']=save(tmp_path/'group.json',group)
    with pytest.raises(ValueError,match='W differs'):
        preflight_material_source(inventory,**options,allow_preview=True)


def test_actual_complete_group_sigma_is_checked_even_when_its_W_is_unchanged(source,tmp_path):
    import json
    matrix=source.frame['whitening_groups'][0]['operator'].matrix
    group=save(tmp_path/'group.json',{'specifications':[dict(native_pts=256256,
        source_point_ids=['p0','p1'],operator_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        operator_matrix_shape=list(matrix.shape),original_sigma_px_sha256=hashlib.sha256(source.row['sigma'][:2].tobytes()).hexdigest(),
        material_identity_confirmed=False,reason='Unresolved family',evidence_references=[source.evidence])]})
    revision=save(tmp_path/'ordinary.json',source.revision)
    binding=json.loads(open(source.revision['base_binding']['path'],encoding='utf8').read())
    # The correlated likelihood uses W, so this cannot be caught merely by
    # comparing its formal scales. Preserve the source evidence itself too.
    source.row['sigma'][0]+=1.
    with pytest.raises(ValueError,match='source sigma differs'):
        transform_material_source([source.frame],group_manifest_reference=group,
            ordinary_revision_reference=revision,binding=binding,policy=None)


def test_normalized_runtime_group_authenticates_literal_origin_without_inventing_metadata(source,tmp_path):
    import json
    row=source.row
    source.frame['rows']=[dict(row,uv=row['uv'][i:i+1],sigma=row['sigma'][i:i+1],
        source_point_ids=[row['source_point_ids'][i]],original_vertex_indices=[i])for i in range(4)]
    group=source.frame['whitening_groups'][0]
    group['operator'].group_id='native/complete-family'
    source.frame['source_packet']=source.evidence
    matrix=group['operator'].matrix
    dataset=save(tmp_path/'literal.json',{'frames':[dict(index=1,native_pts=256256,
        source_packet=source.evidence,whitening_groups=[dict(operator=dict(
            group_id=group['operator'].group_id,source_point_ids=['p0','p1'],
            whitening_matrix=matrix.tolist()),source_evidence=dict(source_readout_packet=source.evidence))])]})
    specification=dict(native_pts=256256,source_point_ids=['p0','p1'],
        operator_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        operator_matrix_shape=list(matrix.shape),material_identity_confirmed=False,
        reason='Unresolved source family',evidence_references=[source.evidence],
        operator_source=dict(frame_dataset=dataset,frame=1,whitening_group_index=0,
            group_id=group['operator'].group_id,source_packet=source.evidence,native_readout_packet=source.evidence))
    manifest=save(tmp_path/'groups.json',{'specifications':[specification]})
    item=source.item(2);item['target']=dict(source.row['target'],contour_key='component018')
    source.revision['identity_corrections']=[item]
    revision=save(tmp_path/'ordinary.json',source.revision)
    binding=json.loads(open(source.revision['base_binding']['path'],encoding='utf8').read())
    assert 'source_evidence'not in group
    _,_,proof=transform_material_source([source.frame],group_manifest_reference=manifest,
        ordinary_revision_reference=revision,binding=binding,policy=None)
    assert proof['actual_source_UV_sigma_packet_and_operator_origin_verified_groups']==1
    assert proof['runtime_groups_without_literal_source_evidence']==1
    assert 'source_evidence'not in group
    matrix[0,0]+=1.
    with pytest.raises(ValueError,match='differs from authenticated saved origin'):
        transform_material_source([source.frame],group_manifest_reference=manifest,
            ordinary_revision_reference=revision,binding=binding,policy=None)
