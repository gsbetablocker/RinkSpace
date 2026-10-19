"""Actual pinned event guards; skipped when the reviewed Data fixture is absent."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start.data import sha256_file
from rinkfit_start.runtime import atomic_json
from rinkfit_start.semantic_resolution import load_semantics, resolve_materials
from rinkfit_start.semantic_order_overlay import _rebase
from rinkfit_start.source_generator_recovery import load_source_generator_recovery, PARENT, REVIEW

DATA=Path(os.environ.get('RINKSPACE_DATA_ROOT',r'D:\CodexData\RinkSpace_Data'))
RUN=DATA/'runs/okada_s1_r11'
SELECTED=RUN/'semantic_generator_recovery/manifest.json'


@pytest.fixture(scope='module')
def reviewed():
    if not SELECTED.exists():pytest.skip('pinned native event fixture is not installed')
    m,a,p,info=load_semantics(SELECTED,182017)
    pm,pa,pp,pi=load_semantics(RUN/'semantic_true_order/manifest.json',182017,expected_sha256=PARENT)
    return m,a,p,info,(pm,pa,pp,pi)


@pytest.fixture
def candidate(reviewed,tmp_path):
    m,a,p,info,parent=reviewed
    now=_rebase(copy.deepcopy(m),p.parent)
    def parent_loader(path,n,expected_sha256):
        assert Path(path).resolve()==parent[2].resolve()
        assert n==182017 and expected_sha256==PARENT
        return parent
    return tmp_path,now,parent_loader


def load(candidate):
    root,m,parent_loader=candidate
    return load_source_generator_recovery(root/'manifest.json',m,182017,parent_loader)


def rewrite_proof(candidate,edit):
    root,m,_=candidate
    source=Path(m['source_generator_recovery']['file'])
    proof=json.loads(source.read_text(encoding='utf-8'));edit(proof)
    output=root/'proof.json';atomic_json(output,proof)
    m['source_generator_recovery']={'file':str(output),'sha256':sha256_file(output)}


def rewrite_array(candidate,key,edit):
    root,m,_=candidate;spec=m[key];array=np.fromfile(spec['file'],dtype=spec['dtype'])
    edit(array);output=root/(key+'.u32');array.tofile(output)
    spec['file']=str(output);spec['sha256']=sha256_file(output)


def test_public_complete_native_load_and_exact_53_delta(reviewed):
    m,a,p,i,parent=reviewed
    assert m['observations']==182017 and m['frames']==1248 and m['active_observations']==182016
    assert np.flatnonzero(~i['source_exclusion']['active_mask']).tolist()==[25537]
    proof=i['source_generator_recovery'];bindings=proof['bindings']
    assert proof['review_sha256']==REVIEW and len(bindings)==53
    changed=np.flatnonzero((i['resolved_material_index']!=parent[3]['resolved_material_index'])|
                          (i['resolved_variant_index']!=parent[3]['resolved_variant_index']))
    assert set(changed)=={b['source_row']for b in bindings}
    assert i['bindings']==parent[3]['bindings']
    assert [b['source_row']for b in i['bindings']if 'direct_human_confirmation'in b]==[11534]
    assert i['query_semantic_sha256']==parent[3]['query_semantic_sha256']
    assert i['source_order_overlay']['audit_run_count']==15419
    assert i['source_order_overlay']['active_hard_run_count']==0
    for key in a:assert np.array_equal(a[key],parent[1][key],equal_nan=True)


def test_independent_relocated_complete_load(candidate):
    assert len(load(candidate)[3]['source_generator_recovery']['bindings'])==53


def test_any_54th_material_change_is_rejected(candidate):
    rewrite_array(candidate,'resolved_material_index',lambda a:a.__setitem__(0,(int(a[0])+1)%171))
    with pytest.raises(ValueError,match='unaudited additional row'):load(candidate)


def test_any_54th_variant_change_is_rejected(candidate):
    rewrite_array(candidate,'resolved_variant_index',lambda a:a.__setitem__(0,(int(a[0])+1)%166))
    with pytest.raises(ValueError,match='unaudited additional row'):load(candidate)


@pytest.mark.parametrize('key',['source_uv','source_point_ids','source_target_labels','native_pts','frame_indices'])
def test_raw_five_array_identity_change_is_rejected(candidate,key):
    candidate[1]['immutable_source_arrays'][key]['sha256_c_contiguous_array_bytes']='0'*64
    with pytest.raises(ValueError,match='parent authority/raw/W/E/Q changed'):load(candidate)


@pytest.mark.parametrize('key',['weights','source_sigma','query_uv','alpha_bounds'])
def test_W_noise_or_query_identity_change_is_rejected(candidate,key):
    candidate[1]['arrays'][key]['sha256']='0'*64
    with pytest.raises(ValueError,match='parent authority/raw/W/E/Q changed'):load(candidate)


def test_false_relocation_with_retained_W_hash_is_rejected(candidate):
    root,m,_=candidate;bad=root/'false_W.f64';bad.write_bytes(b'changed W')
    m['arrays']['weights']['file']=str(bad)
    with pytest.raises(ValueError,match='checksum changed'):load(candidate)


def test_wrong_parent_is_rejected(candidate):
    candidate[1]['source_generator_parent_manifest']['sha256']='0'*64
    with pytest.raises(ValueError,match='wrong reviewed parent'):load(candidate)


@pytest.mark.parametrize('mutation',[lambda p:p['bindings'].pop(),
    lambda p:p['bindings'].append(copy.deepcopy(p['bindings'][0])),
    lambda p:p['bindings'][0].__setitem__('source_row',0)])
def test_exact_53_review_scope_is_required(candidate,mutation):
    rewrite_proof(candidate,mutation)
    with pytest.raises(ValueError,match='exact review scope'):load(candidate)


def test_human_authority_cannot_be_added(candidate):
    rewrite_proof(candidate,lambda p:p.__setitem__('direct_human_confirmation',{'source_row':11534}))
    with pytest.raises(ValueError,match='unrecognized proof authority'):load(candidate)


def test_human_anchor_cannot_be_borrowed_per_binding(candidate):
    rewrite_proof(candidate,lambda p:p['bindings'][0].__setitem__('separate_human_confirmed_anchor',11534))
    with pytest.raises(ValueError,match='exact review scope'):load(candidate)


@pytest.mark.parametrize('key',['confirmed_bindings','active_source_mask','source_order_overlay'])
def test_existing_A103_exclusion_and_order_authorities_are_immutable(candidate,key):
    candidate[1][key]['sha256']='0'*64
    with pytest.raises(ValueError,match='parent authority/raw/W/E/Q changed'):load(candidate)


def test_public_material_resolution_keeps_A103_and_adds_native_bindings(reviewed):
    from rinkfit_start.data import Observations
    m,a,p,info,_=reviewed
    binding=json.loads(Path(m['binding']['path']).read_text(encoding='utf-8'))
    obs=Observations.load(Path(binding['input_manifest']['path']).parent)
    model=json.loads((RUN/'model_checks/print_W_seed/model.json').read_text(encoding='utf-8'))
    branches=np.fromfile(RUN/'model_checks/generator_53_seed/path.u32','<u4')
    raw=info['original_material_index']
    variants=np.asarray(m['variant_ids'])[a['variant_index']]
    resolved,expected,records=resolve_materials(info,raw,variants,obs,model,branches,info['source_identity'])
    assert np.array_equal(resolved,info['resolved_material_index'])
    assert len(records)==56 and sum('direct_human_confirmation'in b for b in records)==1
    native=[b for b in records if b.get('no_human_anchor_used') is True]
    assert len(native)==53 and all(b['not_a_direct_human_confirmation'] is True for b in native)
    assert all(b['saved_branch_in_confirmed_support'] for b in native)
