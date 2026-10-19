from copy import deepcopy

import pytest

from calibration.solver_runtime.strict_dataset_target_bindings import _qualify_group


def case():
    target=dict(kind='advertisement_boundary',advertisement_number=5,edge='top_edge',extent_scope='external_border')
    pack=dict(path='exact-current-source.npz',sha256='1'*64)
    annotation=dict(path='exact-original-corners.json',sha256='2'*64)
    header=dict(model_target=target,curve=deepcopy(target),actual_pts=100,native_pts=101,
        decoded_BGR_sha256='3'*64,physical_material_id='Ad5.top_edge',
        source_anchor_id='corners:100:Ad5:top_edge',source_annotation_reference=annotation)
    group=dict(actual_pts=100,model_target=deepcopy(target),source_locators=[dict(source_pack=pack,
        source_row_index=0,original_native_pts=101,decoded_BGR_sha256='3'*64)])
    return group,[header],dict(source_pack=deepcopy(pack),corner_annotation_reference=deepcopy(annotation),
        allowed_source_anchor_ids=['corners:100:Ad5:top_edge'])


def test_only_exact_authenticated_annotation_row_can_receive_scope_label_binding():
    group,headers,binding=case()
    assert _qualify_group(group,headers,binding)
    assert group['model_target']['extent_scope']=='external_border'


@pytest.mark.parametrize('field,value',[
    ('actual_pts',999),('native_pts',999),('decoded_BGR_sha256','4'*64),
    ('physical_material_id','Ad5.bottom_edge'),('source_anchor_id','corners:101:Ad5:top_edge'),
    ('source_annotation_reference',dict(path='other-corners.json',sha256='2'*64)),
    ('curve',dict(kind='advertisement_boundary',advertisement_number=5,edge='top_edge',extent_scope='internal_graphic_only'))])
def test_changed_original_source_binding_is_refused(field,value):
    group,headers,binding=case();headers[0][field]=value
    with pytest.raises(ValueError):_qualify_group(group,headers,binding)


def test_unreviewed_external_scope_cannot_be_globally_aliased():
    group,headers,binding=case();group['source_locators'][0]['source_pack']['sha256']='4'*64
    with pytest.raises(ValueError):_qualify_group(group,headers,binding)
    group,headers,binding=case();group['model_target']['advertisement_number']=18
    assert not _qualify_group(group,headers,binding)
    group,headers,binding=case();group['model_target']['extent_scope']='internal_graphic_only'
    assert not _qualify_group(group,headers,binding)
