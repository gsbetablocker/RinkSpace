
"""Actual event-source authority boundaries, including grouped observations."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start.semantic_resolution import load_semantics
from rinkfit_start.source_component_recovery import resolve_component_materials,_signature
from rinkfit_start.data import sha256_file,Observations
from rinkfit_start.pipeline import read_run

R=Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11')
pytestmark=pytest.mark.skipif(not (R/'semantic_component_v2/manifest.json').is_file(),reason='actual event snapshot unavailable')

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    return path

@pytest.fixture(scope='module')
def full():
    p=R/'semantic_component_v2/manifest.json';m,a,_,info=load_semantics(p,182017,expected_sha256=sha256_file(p))
    pm,_,_,parent=load_semantics(R/'semantic_generator_recovery_v2/manifest.json',182017)
    _,run=read_run(R);obs=Observations.load(run['data_dir'])
    model=json.loads((R/'model_checks/print_W_seed/model.json').read_text())
    branches=np.fromfile(R/'model_checks/component84_seed/path.u32','<u4')
    return m,a,info,pm,parent,obs,model,branches

def test_full_source_exact_delta_and_previous_authority(full):
    m,a,info,pm,parent,obs,model,branches=full
    rows={b['source_row']for b in info['source_component_recovery']['bindings']}
    delta=set(np.flatnonzero((info['resolved_material_index']!=parent['resolved_material_index'])|(info['resolved_variant_index']!=parent['resolved_variant_index'])))
    assert rows==delta and len(rows)==84
    protected={b['source_row']for b in parent['source_generator_recovery']['bindings']}
    assert len(protected)==108 and not rows&protected
    assert info['source_generator_recovery']==parent['source_generator_recovery']
    assert np.flatnonzero(~info['source_exclusion']['active_mask']).tolist()==[25537]
    assert info['source_order_overlay']['active_hard_run_count']==0
    for k,v in pm.items():
        if k not in ['resolved_material_index','resolved_variant_index']:assert _signature(m[k])==_signature(v)
    # Confirmed interpretation records preserve actual original UV/key/labels.
    records=resolve_component_materials(info,obs,model,branches,[])
    assert len(records)==84 and all(x['not_a_direct_human_confirmation']and x['no_human_anchor_used']for x in records)
    assert all(x['review_origin']=='native_source_image_and_component_topology_crosscheck'for x in records)

def test_old68_snapshot_remains_loadable():
    _,_,_,info=load_semantics(R/'semantic_component_v1/manifest.json',182017)
    assert len(info['source_component_recovery']['bindings'])==68
    assert len(info['source_generator_recovery']['bindings'])==108

@pytest.mark.parametrize('case',['parent','W','dtype','extra_row','extra_authority','extra_binding','unknown_evidence','relocated_bad_bytes'])
def test_reject_component_authority_pollution(full,tmp_path,case):
    m,a,info,pm,parent,obs,model,branches=full;m=copy.deepcopy(m)
    # Selected manifest paths are rebased, and existing runtime assets stay immutable.
    proof=json.loads(Path(info['source_component_recovery']['overlay_path']).read_text())
    if case=='parent':m['source_component_parent_manifest']['sha256']='0'*64
    elif case=='W':m['arrays']['weights']['sha256']='0'*64
    elif case=='dtype':m['resolved_material_index']['dtype']='<u8'
    elif case=='extra_row':
        z=info['resolved_material_index'].copy();row=next(i for i in range(len(z))if i not in {b['source_row']for b in proof['bindings']});z[row]=(z[row]+1)%171
        f=tmp_path/'changed.u32';z.astype('<u4').tofile(f);m['resolved_material_index']['file']=str(f);m['resolved_material_index']['sha256']=sha256_file(f)
    elif case=='extra_authority':proof['direct_human_confirmation']=True
    elif case=='extra_binding':proof['bindings'].append(copy.deepcopy(proof['bindings'][0]))
    elif case=='unknown_evidence':proof['evidence_manifest']['sha256']='0'*64
    elif case=='relocated_bad_bytes':
        f=tmp_path/'material.u32';f.write_bytes(b'corrupted');m['resolved_material_index']['file']=str(f)
    if case in ['extra_authority','extra_binding','unknown_evidence']:
        f=write(tmp_path/'proof.json',proof);m['source_component_recovery']={'file':str(f),'sha256':sha256_file(f)}
    # The two new full-size arrays are relative to the selected frozen directory.
    for k in ['resolved_material_index','resolved_variant_index']:
        q=Path(m[k]['file'])
        if not q.is_absolute():m[k]['file']=str(R/'semantic_component_v2'/q)
    f=write(tmp_path/'manifest.json',m)
    with pytest.raises((ValueError,FileNotFoundError)):
        load_semantics(f,182017,expected_sha256=sha256_file(f))

@pytest.mark.parametrize('case',['native_pts','source_point_ids','source_uv','source_target_labels','model_label','model_variant'])
def test_actual_runtime_material_cannot_override_raw_or_variant(full,case):
    m,a,info,pm,parent,original,originalmodel,branches=full
    obs=copy.deepcopy(original);model=copy.deepcopy(originalmodel);b=info['source_component_recovery']['bindings'][0];i=b['source_row'];mi=b['resolved_model_material_index']
    if case=='native_pts':obs.native_pts[i]+=256
    elif case=='source_point_ids':obs.source_point_ids[i]='another-source-key'
    elif case=='source_uv':obs.source_uv[i,0]+=.25
    elif case=='source_target_labels':obs.source_target_labels[i]='Ad5.print.component022'
    elif case=='model_label':model['materials'][mi]['label']='another-material'
    elif case=='model_variant':model['materials'][mi]['paths'][0]['variant_id']='w000'
    with pytest.raises(ValueError):resolve_component_materials(info,obs,model,branches,[])
