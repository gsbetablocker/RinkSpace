"""Pinned source-image/component identity deltas over the frozen 108-row parent.

This authority is algorithmic native-image evidence. It never extends a human
confirmation or relabels raw observations, and never uses a physical fit to
choose a material identity. Unknown snapshots are deliberately rejected.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .data import sha256_file
from .runtime import atomic_json
from .semantic_order_overlay import _path, _verified, _rebase, _verify_descriptors
from .source_generator_recovery import _signature, _require

PARENT = 'cc50c4117d48e0e0875c08c1ad775682fe6198aae63edab7a71aa510135b0bb5'
SCHEMA = 'rinkfit.source_component_recovery.v1'
RULE = 'native_source_component_correspondence_v1'
# Filled only after the actual native-source evidence batch has been frozen.
# Counts alone, arbitrary caller ledgers, and empty policies grant no authority.
EVIDENCE_POLICIES = {'56d67d9b96897dacc8ee0f799ff1e8d73d811dc09174550c65e6bb8a2c292186': {'changed': 68, 'targets': {'Ad5.print.component016': 3, 'Ad5.print.component009': 2, 'Ad5.print.component017_inner': 1, 'Ad5.print.component014_inner': 3, 'Ad5.print.component013_inner': 4, 'Ad5.print.component011_inner': 3, 'Ad5.print.component020': 4, 'Ad5.print.component018': 14, 'Ad5.print.component015': 33, 'Ad5.print.component010_inner': 1}, 'facts_sha256': '606529dc5f1850a27a0463e7c2584ac86f21b4532b4729f2799088e820ffa1e3', 'inventory_sha256': '241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99'}, '230162e0b0a95c7221d5427e3174df4a238c34dfd4daa76b87a4bbccb924ad2c': {'changed': 84, 'targets': {'Ad5.print.component016': 3, 'Ad5.print.component009': 2, 'Ad5.print.component017_inner': 1, 'Ad5.print.component014_inner': 3, 'Ad5.print.component013_inner': 4, 'Ad5.print.component011_inner': 3, 'Ad5.print.component020': 12, 'Ad5.print.component018': 14, 'Ad5.print.component015': 33, 'Ad5.print.component010_inner': 1, 'Ad5.print.component021_inner': 8}, 'facts_sha256': '83e8372e9b5df5e3bc80fab8c351faa028214e0850bead6702b8389808990210', 'inventory_sha256': '241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99'}, '9279ea527a9f04c668d62daafb65060bfe3136a9e05ae3481c408775c5320eac': {'changed': 262, 'targets': {'Ad5.print.component016': 5, 'Ad5.print.component009': 2, 'Ad5.print.component017_inner': 19, 'Ad5.print.component014_inner': 22, 'Ad5.print.component013_inner': 20, 'Ad5.print.component011_inner': 10, 'Ad5.print.component020': 53, 'Ad5.print.component018': 31, 'Ad5.print.component015': 46, 'Ad5.print.component010_inner': 15, 'Ad5.print.component021_inner': 26, 'Ad5.print.component012': 2, 'Ad5.print.component019': 11}, 'facts_sha256': '9ead29a590224abe1d6f6b8309e0620098331af5da472ce73773ef371ce576f0', 'inventory_sha256': '241c68e8302d857fbf7716d3dfc9ca4dbd2a1014b05cb359233e511199dd4d99'}}
BINDING_KEYS = {'source_row','native_pts','source_point_id','source_uv',
    'original_source_label','original_material_index','original_variant_id',
    'resolved_model_material_index','resolved_model_target_id','resolved_variant_id',
    'allowed_path_indices','evidence_fragment_id'}


def load_source_component_recovery(path, manifest, n, parent_loader):
    from .semantic_resolution import read_binary
    root=Path(path).parent
    extra={'source_component_parent_manifest','source_component_recovery'}
    _require(extra<=set(manifest),'incomplete native-component overlay')
    ref=manifest['source_component_parent_manifest']
    _require(ref.get('sha256')==PARENT,'another native-component parent')
    pm,arrays,_,inherited=parent_loader(_verified(root,ref),n,expected_sha256=PARENT)
    _require(n==182017 and pm['frames']==1248 and pm['active_observations']==182016
             and inherited.get('source_generator_recovery')
             and not inherited.get('source_component_recovery'), 'incomplete component parent chain')
    active=inherited['source_exclusion']['active_mask']
    _require(np.flatnonzero(~active).tolist()==[25537]
             and inherited['source_order_overlay']['active_hard_run_count']==0
             and inherited['source_order_overlay']['audit_run_count']==15419,
             'native-component overlay changed original obligations')
    _require(set(manifest)==set(pm)|extra,'native-component field inventory changed')
    for key,before in pm.items():
        now=manifest[key]
        if key in ('resolved_material_index','resolved_variant_index'):
            x=_signature(before);y=_signature(now);x.pop('sha256',None);y.pop('sha256',None)
            _require(x==y,'native-component array ABI changed')
        else:
            _require(_signature(before)==_signature(now),'native-component parent authority/raw/W/E/Q changed: '+key)
    # The parent loader owns its historical source_snapshot ledger. Its old
    # mutable project binding pointer is an audit fact, not a runtime asset.
    # Verify every selected runtime/authority descriptor, as the parent does.
    for key in ['arrays','assets','parent_semantic_manifest','original_material_index',
                'resolved_material_index','resolved_variant_index','confirmed_bindings',
                'source_exclusion_parent_manifest','source_exclusion','active_source_mask',
                'active_statistics','source_order_parent_manifest','source_order_overlay',
                'source_generator_parent_manifest','source_generator_recovery',
                'source_component_parent_manifest','source_component_recovery']:
        _verify_descriptors(root,manifest[key])
    proof_path=_verified(root,manifest['source_component_recovery'])
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    _require(set(proof)=={'schema','rule_id','parent_semantic_sha256','observations','evidence_manifest','bindings'}
             and proof.get('schema')==SCHEMA and proof.get('rule_id')==RULE
             and proof.get('parent_semantic_sha256')==PARENT and proof.get('observations')==n,
             'native-component proof authority/scope changed')
    digest=proof['evidence_manifest'].get('sha256')
    _require(digest in EVIDENCE_POLICIES,'unreviewed native-component evidence snapshot')
    policy=EVIDENCE_POLICIES[digest]
    ep=_verified(proof_path.parent,proof['evidence_manifest']);ed=ep.parent
    evidence=json.loads(ep.read_text(encoding='utf-8'))
    _verify_descriptors(ed,evidence)
    _require(evidence.get('schema')=='rinkfit.native_source_component_correspondence_evidence.v1'
             and evidence.get('rule_id')==RULE
             and evidence.get('origin')=='native_source_image_and_component_topology_crosscheck'
             and evidence.get('parent_semantic_sha256')==PARENT
             and evidence.get('source_identity')==inherited['source_identity']
             and evidence.get('raw_observations')==n and evidence.get('active_observations')==182016
             and all(evidence.get(k) is True for k in ['not_a_direct_human_confirmation',
                'no_extension_of_any_human_confirmation','no_human_anchor_used',
                'current_model_nearest_label_is_not_identity_evidence',
                'raw_UV_ID_labels_W_E_and_existing_authority_unchanged']),
             'native-component human/source authority contamination')
    bindings=proof['bindings']
    _require(bindings==evidence.get('bindings') and len(bindings)==policy['changed']
             and evidence.get('changed_rows')==policy['changed']
             and evidence.get('target_counts')==policy['targets']
             and dict(Counter(b['resolved_model_target_id'] for b in bindings))==policy['targets'],
             'native-component exact delta scope changed')
    _require(_signature(evidence['source_row_registry'])==_signature(pm['assets']['source_rows.jsonl']),
             'native-component source-row registry changed')
    _require(evidence['source_correspondence']['sha256']==policy['facts_sha256']
             and evidence['source_inventory']['sha256']==policy['inventory_sha256'],
             'native-component provenance snapshot changed')
    model=json.loads(_verified(ed,evidence['model']).read_text(encoding='utf-8'))
    labels=[m['label'] for m in model['materials']]
    raw=read_binary(root,manifest['original_material_index'])
    material=read_binary(root,manifest['resolved_material_index'])
    variant=read_binary(root,manifest['resolved_variant_index'])
    old_material=inherited['resolved_material_index'];old_variant=inherited['resolved_variant_index']
    ids=manifest['resolved_variant_ids']
    _require(all(a.shape==(n,) and a.dtype==np.dtype('<u4') for a in [raw,material,variant]),
             'native-component full uint32 ABI required')
    binding=json.loads(_path(root,manifest['binding']).read_text(encoding='utf-8'))
    ip=_verified(_path(root,manifest['binding']).parent,binding['input_manifest'])
    op=ip.parent/'observations.npz'
    _require(sha256_file(op)==inherited['source_identity'],'native-component raw source archive changed')
    with np.load(op,allow_pickle=False) as archive:
        source={k:archive[k] for k in manifest['immutable_source_arrays']}
    facts=json.loads(_verified(ed,evidence['source_correspondence']).read_text(encoding='utf-8'))
    _require(facts.get('schema')=='rinkfit.native_component_identity_facts.v1'
             and facts.get('physical_projection_used_for_identity') is False,
             'native-component source-only facts missing')
    factrows={x['source_row']:x for x in facts['records']}
    _require(len(factrows)==len(facts['records'])==len(bindings),'native-component fact rows incomplete')
    registry={}
    wanted={b['source_row'] for b in bindings}
    with _verified(ed,evidence['source_row_registry']).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['source_row'] in wanted:registry[row['source_row']]=row
    _require(set(registry)==wanted,'native-component source provenance missing')
    protected={b['source_row'] for b in inherited['source_generator_recovery']['bindings']}
    packets={};byrow={}
    for b in bindings:
        row=b['source_row'];mi=b['resolved_model_material_index']
        _require(set(b)==BINDING_KEYS and type(row) is int and 0<=row<n
                 and row not in byrow and row not in protected and active[row] and arrays['kind'][row]==0
                 and type(mi) is int and 0<=mi<len(labels), 'invalid/duplicate/protected component binding')
        _require(b['native_pts']==int(source['native_pts'][row])
                 and b['source_point_id']==str(source['source_point_ids'][row])
                 and b['source_uv']==source['source_uv'][row].tolist()
                 and b['original_source_label']==str(source['source_target_labels'][row])==labels[int(raw[row])]
                 and b['original_source_label'].startswith('Ad5.print.')
                 and b['original_material_index']==int(raw[row])
                 and b['original_variant_id']==pm['variant_ids'][int(arrays['variant_index'][row])]
                 and b['resolved_model_target_id']==labels[mi] and labels[mi].startswith('Ad5.print.')
                 and b['allowed_path_indices']==[0] and b['resolved_variant_id'] in ids
                 and material[row]==mi and variant[row]==ids.index(b['resolved_variant_id'])
                 and (material[row]!=old_material[row] or variant[row]!=old_variant[row]),
                 'native-component binding differs from immutable raw source/exact target')
        _require(model['materials'][mi]['paths'][0]['variant_id']==b['resolved_variant_id'],
                 'native-component target finite variant mismatch')
        fact=factrows[row]
        _require(fact.get('identity_confirmed') is True
                 and fact.get('resolved_model_target_id')==b['resolved_model_target_id']
                 and fact.get('evidence_fragment_id')==b['evidence_fragment_id'],
                 'native-component binding lacks its admitted source-only fact')
        reg=registry[row];sp=_verified(ep.parent,reg['source_packet']);key=str(sp)
        if key not in packets:packets[key]=json.loads(sp.read_text(encoding='utf-8'))
        packet=packets[key];features=[x for x in packet['features'] if x['id']==reg['source_feature_id']]
        _require(len(features)==1 and packet['source']['native_pts']==b['native_pts']
                 and features[0]['segments'][reg['source_segment_index']]['uv'][reg['original_vertex_index']]==b['source_uv'],
                 'native-component original packet/member/PTS changed')
        byrow[row]=b
    changed=np.flatnonzero((material!=old_material)|(variant!=old_variant)).tolist()
    _require(set(changed)==set(byrow),'native-component unaudited extra/missing delta')
    info=dict(inherited)
    info.update(resolved_material_index=material,resolved_variant_index=variant,
        source_component_recovery=dict(_rebase(proof,proof_path.parent),overlay_path=str(proof_path),
            overlay_sha256=manifest['source_component_recovery']['sha256'],evidence_sha256=digest))
    return manifest,arrays,Path(path),info


def resolve_component_materials(info,obs,model,branches,records):
    proof=info.get('source_component_recovery')
    if not proof:return records
    result=list(records)
    for original in proof['bindings']:
        b=copy.deepcopy(original);row=b['source_row'];mi=b['resolved_model_material_index']
        _require(int(obs.native_pts[row])==b['native_pts']
                 and str(obs.source_point_ids[row])==b['source_point_id']
                 and str(obs.source_target_labels[row])==b['original_source_label']
                 and np.array_equal(obs.source_uv[row],b['source_uv'])
                 and model['materials'][mi]['label']==b['resolved_model_target_id']
                 and model['materials'][mi]['paths'][0]['variant_id']==b['resolved_variant_id'],
                 'actual component observation/finite material mismatch')
        b.update(saved_branch_in_confirmed_support=int(branches[row]) in b['allowed_path_indices'],
            review_origin='native_source_image_and_component_topology_crosscheck',
            not_a_direct_human_confirmation=True,no_human_anchor_used=True,
            recovery_evidence_sha256=proof['evidence_sha256'])
        result.append(b)
    return result


def export_source_component_recovery(parent_manifest,evidence_manifest,output_dir):
    from .semantic_resolution import load_semantics
    parent=Path(parent_manifest).resolve();ep=Path(evidence_manifest).resolve()
    _require(sha256_file(parent)==PARENT and sha256_file(ep) in EVIDENCE_POLICIES,
             'unreviewed component export input')
    pm,_,_,info=load_semantics(parent,182017,expected_sha256=PARENT)
    evidence=json.loads(ep.read_text(encoding='utf-8'));bindings=evidence['bindings']
    material=info['resolved_material_index'].copy();variant=info['resolved_variant_index'].copy()
    for b in bindings:
        material[b['source_row']]=b['resolved_model_material_index']
        variant[b['source_row']]=pm['resolved_variant_ids'].index(b['resolved_variant_id'])
    out=Path(output_dir).resolve()
    _require(not out.exists() or not any(out.iterdir()),'never overwrite frozen component interpretation')
    out.mkdir(parents=True,exist_ok=True);manifest=_rebase(copy.deepcopy(pm),parent.parent)
    for key,name,z in [('resolved_material_index','resolved_material.u32',material),
                       ('resolved_variant_index','resolved_variant.u32',variant)]:
        z.astype('<u4').tofile(out/name);d=copy.deepcopy(pm[key]);d.pop('path',None)
        d.update(file=name,sha256=sha256_file(out/name));manifest[key]=d
    proof=dict(schema=SCHEMA,rule_id=RULE,parent_semantic_sha256=PARENT,observations=182017,
        evidence_manifest=dict(path=str(ep),sha256=sha256_file(ep)),bindings=copy.deepcopy(bindings))
    atomic_json(out/'source_component_recovery.json',proof)
    manifest.update(source_component_parent_manifest=dict(path=str(parent),sha256=PARENT),
        source_component_recovery=dict(file='source_component_recovery.json',sha256=sha256_file(out/'source_component_recovery.json')))
    atomic_json(out/'manifest.json',manifest)
    load_semantics(out/'manifest.json',182017,expected_sha256=sha256_file(out/'manifest.json'))
    return out/'manifest.json'
