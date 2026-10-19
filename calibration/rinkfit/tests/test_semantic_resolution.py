import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rinkfit_start.data import sha256_file
from rinkfit_start.runtime import atomic_json
from rinkfit_start.semantic_resolution import export_resolution,load_semantics,resolve_materials


def fixture(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();n=3
    def array(name,a,dtype):
        a=np.asarray(a,dtype=dtype);p=parent/name;a.tofile(p)
        return {'file':str(p),'sha256':sha256_file(p),'dtype':a.dtype.str,'shape':list(a.shape)}
    raw=array('material.u32',[0,0,0],'<u4')
    runs=parent/'source_runs.jsonl';runs.write_text('',encoding='utf-8')
    m={'schema':'rinkfit.semantic_packet.v1','observations':n,'variant_ids':['w0'],
       'immutable_source_arrays':{'immutable':'same'},
       'arrays':{'kind':array('kind.u8',[0,0,2],'|u1'),
                 'variant_index':array('variant.u32',[0,0,0],'<u4'),
                 'weights':array('weights.f64',[1.,.2,2.],'<f8')},
       'assets':{'source_runs.jsonl':{'file':str(runs),'sha256':sha256_file(runs)}}}
    atomic_json(parent/'manifest.json',m)
    obs=SimpleNamespace(native_pts=np.array([10,20,30]),source_point_ids=np.array(['p0','p1','p2']),
        source_target_labels=np.array(['T','T','T']),source_uv=np.array([[0.,0.],[1.,2.],[3.,4.]]))
    confirmation=tmp_path/'human.json'
    atomic_json(confirmation,{'schema':'rinkfit.direct-human-material-confirmation.v1','source_row':1,
        'native_pts':20,'raw_source_point_id':'p1','raw_source_label':'T','source_uv':[1.,2.],
        'original_npz_source_identity':'source','confirmation_applies_to_other_source_rows':False,
        'raw_arrays_W_E_obligations_remain_immutable':True})
    geometry=tmp_path/'geometry.json';atomic_json(geometry,{'A':{'closed':True}})
    binding={'source_row':1,'native_pts':20,'source_point_id':'p1','source_uv':[1.,2.],
        'original_source_label':'T','original_material_index':0,'original_variant_id':'w0',
        'resolved_model_material_index':1,'resolved_model_target_id':'A','resolved_variant_id':'A_variant',
        'allowed_path_indices':[0],'direct_human_confirmation':{'path':str(confirmation),'sha256':sha256_file(confirmation)},
        'exact_source_geometry_provenance':{'path':str(geometry),'sha256':sha256_file(geometry),'json_pointer':'/A'}}
    model={'materials':[{'label':'T','paths':[{'variant_id':'w0'}]},
                         {'label':'A','paths':[{'variant_id':'A_variant'}]}]}
    return parent/'manifest.json',raw,binding,obs,model


def test_overlay_preserves_parent_and_changes_only_confirmed_world_interpretation(tmp_path):
    parent,raw,binding,obs,model=fixture(tmp_path)
    frozen=parent.read_bytes();path=export_resolution(parent,raw,[binding],tmp_path/'new','source')
    m,a,_,info=load_semantics(path,3,expected_sha256=sha256_file(path))
    materials,variants,records=resolve_materials(info,np.zeros(3,dtype='<u4'),np.array(['w0']*3),obs,model,np.zeros(3,int),'source')
    assert materials.tolist()==[0,1,0] and variants.tolist()==['w0','A_variant','w0']
    assert a['variant_index'].tolist()==[0,0,0] and m['variant_ids']==['w0']
    assert records[0]['saved_branch_in_confirmed_support'] and parent.read_bytes()==frozen
    assert info['query_semantic_sha256']==sha256_file(parent)


def test_overlay_rejects_changed_parent_statistics_even_if_rehashed(tmp_path):
    parent,raw,binding,_,_=fixture(tmp_path);path=export_resolution(parent,raw,[binding],tmp_path/'new','source')
    m=json.loads(path.read_text(encoding='utf-8'));p=path.parent/'changed.f64';np.array([9.,.2,2.],'<f8').tofile(p)
    m['arrays']['weights'].update(file=str(p),sha256=sha256_file(p));atomic_json(path,m)
    with pytest.raises(ValueError,match='changed original semantic array'):load_semantics(path,3)


def test_overlay_rejects_raw_key_or_confirmation_or_selected_identity_mismatch(tmp_path):
    parent,raw,binding,obs,model=fixture(tmp_path);path=export_resolution(parent,raw,[binding],tmp_path/'new','source')
    with pytest.raises(ValueError,match='Selected semantic identity'):load_semantics(path,3,expected_sha256='wrong')
    *_,info=load_semantics(path,3);obs.source_uv[1,0]+=1
    with pytest.raises(ValueError,match='immutable composite source key'):resolve_materials(info,np.zeros(3,dtype='<u4'),np.array(['w0']*3),obs,model,np.zeros(3,int),'source')
    proof=tmp_path/'human.json';proof.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='confirmation checksum'):load_semantics(path,3)


def test_overlay_rejects_ordered_run_rebinding_instead_of_dropping_obligation(tmp_path):
    parent,raw,binding,_,_=fixture(tmp_path);m=json.loads(parent.read_text(encoding='utf-8'))
    runs=parent.parent/'source_runs.jsonl';runs.write_text(json.dumps({'source_rows':[0,1]})+'\n',encoding='utf-8')
    m['assets']['source_runs.jsonl']['sha256']=sha256_file(runs);atomic_json(parent,m)
    with pytest.raises(ValueError,match='ordered source run'):export_resolution(parent,raw,[binding],tmp_path/'new','source')


def image_review_fixture(tmp_path):
    parent,raw,binding,obs,model=fixture(tmp_path);obs.native_pts[0]=20
    anchor=tmp_path/'anchor.json';a=json.loads((tmp_path/'human.json').read_text(encoding='utf-8'))
    a.update(source_row=0,raw_source_point_id='p0',source_uv=[0.,0.]);atomic_json(anchor,a)
    assets={}
    for name in ['exact_native_frame','native_source_video','visual_diagnostic','original_fragment_lineage']:
        p=tmp_path/(name+'.bin');p.write_bytes(name.encode());assets[name]={'path':str(p),'sha256':sha256_file(p)}
    assets['exact_native_frame'].update(native_pts=20,width=1920,height=1080)
    proof={'schema':'rinkfit.native-image-material-review.v1','review_origin':'codex_primary_native_frame_visual_review',
        'source_row':1,'native_pts':20,'raw_source_point_id':'p1','raw_source_label':'T','source_uv':[1.,2.],
        'resolved_model_target_id':'A','original_npz_source_identity':'source','review_applies_to_other_source_rows':False,
        'raw_arrays_W_E_obligations_remain_immutable':True,'not_a_direct_human_confirmation':True,
        'no_extension_of_anchor_human_confirmation':True,**assets,
        'separate_human_confirmed_anchor':{'path':str(anchor),'sha256':sha256_file(anchor),'source_row':0}}
    p=tmp_path/'review.json';atomic_json(p,proof);binding.pop('direct_human_confirmation')
    binding['source_image_review']={'path':str(p),'sha256':sha256_file(p)}
    return parent,raw,binding,obs,model,p,proof


def test_native_image_review_remains_distinct_from_human_and_checks_anchor_raw_identity(tmp_path):
    parent,raw,binding,obs,model,_,_=image_review_fixture(tmp_path)
    path=export_resolution(parent,raw,[binding],tmp_path/'new','source');*_,info=load_semantics(path,3)
    mats,_,records=resolve_materials(info,np.zeros(3,dtype='<u4'),np.array(['w0']*3),obs,model,np.zeros(3,int),'source')
    assert mats.tolist()==[0,1,0] and 'source_image_review'in records[0] and 'direct_human_confirmation'not in records[0]
    obs.source_point_ids[0]='wrong-anchor'
    with pytest.raises(ValueError,match='anchor does not match'):resolve_materials(info,np.zeros(3,dtype='<u4'),np.array(['w0']*3),obs,model,np.zeros(3,int),'source')


@pytest.mark.parametrize('change',[{'review_origin':'untrusted'}, {'review_applies_to_other_source_rows':True},
    {'not_a_direct_human_confirmation':False}, {'no_extension_of_anchor_human_confirmation':False},
    {'resolved_model_target_id':'wrong'}])
def test_native_image_review_rejects_authority_scope_and_target_changes(tmp_path,change):
    parent,raw,binding,_,_,p,proof=image_review_fixture(tmp_path);proof.update(change);atomic_json(p,proof)
    binding['source_image_review']['sha256']=sha256_file(p)
    with pytest.raises(ValueError,match='distinct native-image'):export_resolution(parent,raw,[binding],tmp_path/'new','source')


def test_review_cannot_be_relabelled_as_human_or_combined_with_human_evidence(tmp_path):
    parent,raw,binding,_,_,p,proof=image_review_fixture(tmp_path)
    binding['direct_human_confirmation']=copy.deepcopy(binding['source_image_review'])
    with pytest.raises(ValueError,match='exactly one'):export_resolution(parent,raw,[binding],tmp_path/'ambiguous','source')
    binding.pop('source_image_review');proof.update(schema='rinkfit.direct-human-material-confirmation.v1',confirmation_applies_to_other_source_rows=False)
    atomic_json(p,proof);binding['direct_human_confirmation']['sha256']=sha256_file(p)
    with pytest.raises(ValueError,match='direct human-confirmed'):export_resolution(parent,raw,[binding],tmp_path/'fake_human','source')
