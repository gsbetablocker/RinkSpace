"""Pinned native-fragment identity recovery over the complete reviewed parent.

Only model material/variant interpretation changes. Source arrays, statistics,
exclusions, source-order audit and the separate A103 authority remain inherited.
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

SCHEMA = 'rinkfit.source_generator_recovery.v1'
RULE = 'native_letter_fragment_identity_recovery_v1'
PARENT = '416f6f4254be483089b38fdf1d74ac64b32235e78ca87c276e5127f11bd09df4'
REVIEW = '250cf0e111410acdeea3e87465d03f3201baf8eabd089e5a8aa4235102a9ae45'
TARGET_COUNTS = {'Ad2.print.K':39, 'Ad2.print.SKATING_A_inner':14}
REVIEW_V2 = '5620ec6a2519821782f4d181f840eff251c03e44baf19fb1d8cca6837b160e69'
TARGET_COUNTS_V2 = {'Ad2.print.COM_O':1, 'Ad2.print.COM_O_inner':8,
                    'Ad2.print.K':39, 'Ad2.print.SKATING_A':28,
                    'Ad2.print.SKATING_A_inner':14, 'Ad2.print.SKATING_S':11,
                    'Ad2.print.hyphen':7}
REVIEW_POLICIES = {
    REVIEW:dict(changed=53,retained=53,fragments=22,members=106,frames=18,targets=TARGET_COUNTS),
    REVIEW_V2:dict(changed=108,retained=132,fragments=41,members=240,frames=32,targets=TARGET_COUNTS_V2),
}


def _signature(value):
    if isinstance(value, dict):
        return {k:_signature(v) for k,v in value.items()
                if not ('sha256' in value and k in ('file','path'))}
    if isinstance(value,list):return [_signature(v) for v in value]
    return value


def _require(ok, message):
    if not ok:raise ValueError('Source generator recovery: '+message)


def load_source_generator_recovery(path, manifest, n, parent_loader):
    """Load parent first, then certify the exact pinned native interpretation delta."""
    from .semantic_resolution import read_binary
    root=Path(path).parent;keys={'source_generator_parent_manifest','source_generator_recovery'}
    _require(keys<=set(manifest),'incomplete overlay')
    ref=manifest['source_generator_parent_manifest']
    _require(ref.get('sha256')==PARENT,'wrong reviewed parent')
    pm,arrays,_,inherited=parent_loader(_verified(root,ref),n,expected_sha256=PARENT)
    _require(not inherited.get('source_generator_recovery') and n==182017
             and pm.get('frames')==1248 and pm.get('active_observations')==182016
             and bool(inherited.get('source_order_overlay'))
             and inherited.get('source_exclusion',{}).get('enabled'), 'incomplete parent chain')
    exclusion=inherited['source_exclusion'];active=exclusion['active_mask']
    order=inherited['source_order_overlay']
    _require(np.flatnonzero(~active).tolist()==[25537]
             and order.get('audit_run_count')==15419
             and order.get('active_hard_run_count')==0
             and order.get('active_hard_row_count')==0,'changed active/order obligations')
    _require(set(manifest)==set(pm)|keys,'unrelated field inventory changed')
    for key,value in pm.items():
        current=manifest[key]
        if key in ('resolved_material_index','resolved_variant_index'):
            before=_signature(value);after=_signature(current)
            before.pop('sha256',None);after.pop('sha256',None)
            _require(before==after,'resolved-array dtype/shape/metadata changed')
        else:_require(_signature(current)==_signature(value),'parent authority/raw/W/E/Q changed: '+key)
    # Check bytes at relocated descriptors, not just claimed hash equality.
    for key in ['arrays','assets','parent_semantic_manifest','original_material_index',
                'resolved_material_index','resolved_variant_index','confirmed_bindings',
                'source_exclusion_parent_manifest','source_exclusion','active_source_mask',
                'active_statistics','source_order_parent_manifest','source_order_overlay']:
        _verify_descriptors(root,manifest[key])
    proof_path=_verified(root,manifest['source_generator_recovery'])
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    _require(set(proof)=={'schema','rule_id','parent_semantic_sha256','observations','review_manifest','bindings'},
             'unrecognized proof authority/fields')
    _require(proof.get('schema')==SCHEMA and proof.get('rule_id')==RULE
             and proof.get('parent_semantic_sha256')==PARENT
             and proof.get('observations')==n,'wrong proof scope')
    review_ref=proof['review_manifest']
    review_digest=review_ref.get('sha256')
    _require(review_digest in REVIEW_POLICIES,'unreviewed native ledger')
    policy=REVIEW_POLICIES[review_digest]
    review_path=_verified(proof_path.parent,review_ref)
    review=json.loads(review_path.read_text(encoding='utf-8'));rd=review_path.parent
    _require(review.get('schema')=='rinkfit.native-source-generator-recovery-review.v1'
             and review.get('rule_id')==RULE
             and review.get('review_origin')=='codex_primary_native_frame_visual_review'
             and all(review.get(k) is True for k in ['not_a_direct_human_confirmation',
                 'no_extension_of_any_human_confirmation','no_human_anchor_used',
                 'current_model_nearest_label_is_not_identity_evidence',
                 'raw_UV_ID_labels_W_E_and_existing_authority_unchanged'])
             and review.get('parent_semantic_sha256')==PARENT
             and review.get('source_identity')==inherited['source_identity']
             and review.get('raw_observations')==n
             and review.get('active_observations')==182016,'human/source authority contamination')
    bindings=proof['bindings']
    _require(bindings==review.get('bindings') and len(bindings)==policy['changed']
             and review.get('changed_rows')==policy['changed']
             and review.get('retained_outer_rows')==policy['retained']
             and review.get('reviewed_full_fragments')==policy['fragments']
             and review.get('reviewed_fragment_rows')==policy['members']
             and review.get('reviewed_native_frames')==policy['frames'],'incomplete exact review scope')
    if review_digest==REVIEW_V2:
        previous_ref=review['previous_native_review']
        _require(previous_ref.get('sha256')==REVIEW,'v2 lost original native recovery authority')
        previous=json.loads(_verified(rd,previous_ref).read_text(encoding='utf-8'))
        for key,id_key in [('bindings','source_row'),('fragments','fragment_id'),('frames','frame')]:
            current={v[id_key]:v for v in review[key]}
            _require(all(current.get(v[id_key])==v for v in previous[key]),'v2 rewrote the original native recovery facts')
    _require(_signature(review['source_row_registry'])==_signature(pm['assets']['source_rows.jsonl']),
             'another source-row registry')
    _verify_descriptors(rd,review)
    binding=json.loads(_path(root,manifest['binding']).read_text(encoding='utf-8'))
    input_path=_verified(_path(root,manifest['binding']).parent,binding['input_manifest'])
    observation_path=input_path.parent/'observations.npz'
    _require(sha256_file(observation_path)==inherited['source_identity'],'original observation archive changed')
    with np.load(observation_path,allow_pickle=False) as archive:
        source={k:archive[k] for k in manifest['immutable_source_arrays']}
    for key,spec in manifest['immutable_source_arrays'].items():
        a=source[key]
        _require(a.dtype.str==spec['dtype_str'] and list(a.shape)==spec['shape']
                 and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==spec['sha256_c_contiguous_array_bytes'],
                 'immutable raw array changed: '+key)
    raw=read_binary(root,manifest['original_material_index'])
    old_material=inherited['resolved_material_index'];old_variant=inherited['resolved_variant_index']
    material=read_binary(root,manifest['resolved_material_index'])
    variant=read_binary(root,manifest['resolved_variant_index']);ids=manifest['resolved_variant_ids']
    _require(all(a.shape==(n,) and a.dtype==np.dtype('<u4') for a in [raw,material,variant]),'full uint32 ABI required')
    model=json.loads(_verified(rd,review['model']).read_text(encoding='utf-8'))
    labels=[m['label'] for m in model['materials']]
    byrow={}
    ordinary=set(map(int,arrays['ordinary_rows']))
    allowed_keys={'source_row','native_pts','source_point_id','source_uv','original_source_label',
                  'original_material_index','original_variant_id','resolved_model_material_index',
                  'resolved_model_target_id','resolved_variant_id','allowed_path_indices','evidence_fragment_id'}
    for b in bindings:
        row=b['source_row'];mi=b['resolved_model_material_index']
        _require(set(b)==allowed_keys and type(row) is int and 0<=row<n and row not in byrow
                 and row in ordinary and active[row] and arrays['kind'][row]==0
                 and type(mi) is int and 0<=mi<len(labels),'invalid/nonordinary/contaminated binding')
        _require(b['native_pts']==int(source['native_pts'][row])
                 and b['source_point_id']==str(source['source_point_ids'][row])
                 and b['source_uv']==source['source_uv'][row].tolist()
                 and b['original_source_label']==str(source['source_target_labels'][row])
                 and b['original_source_label']==labels[int(raw[row])]
                 and b['original_material_index']==int(raw[row])
                 and b['original_variant_id']==pm['variant_ids'][int(arrays['variant_index'][row])]
                 and b['resolved_model_target_id']==labels[mi]
                 and b['allowed_path_indices']==[0] and b['resolved_variant_id'] in ids
                 and material[row]==mi and variant[row]==ids.index(b['resolved_variant_id'])
                 and (material[row]!=old_material[row] or variant[row]!=old_variant[row]),
                 'binding differs from original source or reviewed target')
        byrow[row]=b
    _require(Counter(b['resolved_model_target_id'] for b in bindings)==policy['targets'],'target counts changed')
    changed=np.flatnonzero((material!=old_material)|(variant!=old_variant)).tolist()
    _require(set(changed)==set(byrow),'unaudited additional row or missing reviewed change')
    packets={};frames={}
    for frame in review['frames']:
        f=frame['frame'];pts=frame['native_pts'];image=frame['native_image']
        packet=json.loads(_verified(rd,frame['source_packet']).read_text(encoding='utf-8'))
        _require(type(f) is int and 1<=f<=1248 and f not in packets
                 and pts==256256+256*(f-1) and frame['time_base']=='1/12800'
                 and frame['native_PTS_matches_source_packet'] is True
                 and image['matches_source_packet'] is True
                 and image['decoded_BGR_sha256']==frame['original_decoded_BGR_sha256']
                 and packet['source']['native_pts']==pts and packet['source']['actual_pts']==pts
                 and packet['source']['programme_frame_index_1based']==f
                 and packet['source']['time_base']=='1/12800'
                 and packet['source']['source_video_sha256']==review['source_video']['sha256']
                 and packet['source']['decoded_BGR_sha256']==image['decoded_BGR_sha256']
                 and packet['source']['mask_sha256']==frame['foreground_mask']['sha256'],
                 'native PTS/frame/video/BGR/mask identity changed')
        packets[f]=packet;frames[f]=frame
    _require(len(frames)==policy['frames'],'native frame coverage changed')
    members=set();changed_members=set();fragment_ids=set();retained=0
    for fragment in review['fragments']:
        fid=fragment['fragment_id'];f=fragment['frame']
        _require(fid not in fragment_ids and f in frames
                 and fragment['native_pts']==frames[f]['native_pts']
                 and fragment['source_packet']==frames[f]['source_packet']
                 and fragment['native_image']==frames[f]['native_image']
                 and fragment['native_review_completed'] is True
                 and fragment['native_review_origin']=='codex_primary_native_frame_visual_review'
                 and fragment['complete_fragment_membership_reviewed'] is True,'incomplete primary fragment review')
        fragment_ids.add(fid);feature_id=fragment['source_feature_id']
        features=[v for v in packets[f]['features'] if v['id']==feature_id]
        _require(len(features)==1,'ambiguous/missing source feature');feature=features[0]
        canonical=json.dumps(feature,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        _require(hashlib.sha256(canonical).hexdigest()==fragment['source_feature_sha256_canonical_json']
                 and sum(len(s['uv']) for s in feature['segments'])==len(fragment['members']),
                 'changed source feature or omitted fragment member')
        for member in fragment['members']:
            row=member['source_row'];si=member['segment'];vertex=member['vertex'];b=byrow.get(row)
            _require(type(row) is int and 0<=row<n and row not in members and active[row]
                     and arrays['kind'][row]==0 and source['frame_indices'][row]==f
                     and source['native_pts'][row]==fragment['native_pts']
                     and source['source_uv'][row].tolist()==member['source_uv']
                     and feature['segments'][si]['uv'][vertex]==member['source_uv']
                     and str(source['source_point_ids'][row])==f'{feature_id}/segment/{si}/vertex/{vertex}'
                     and member['interpretation_changed'] is (b is not None),'fragment membership/raw identity changed')
            members.add(row)
            if b is not None:
                _require(b['evidence_fragment_id']==fid and b['resolved_model_target_id']==member['reviewed_model_target_id'],
                         'binding lacks the reviewed native member')
                changed_members.add(row)
            else:
                _require(member['reviewed_model_target_id']==labels[int(old_material[row])], 'retained exterior member changed')
                retained+=1
    _require(len(fragment_ids)==policy['fragments'] and len(members)==policy['members']
             and changed_members==set(byrow) and retained==policy['retained'],
             'incomplete full native-fragment audit')
    info=dict(inherited)
    info.update(resolved_material_index=material,resolved_variant_index=variant,
                source_generator_recovery=dict(_rebase(proof,proof_path.parent),
                    overlay_path=str(proof_path),overlay_sha256=manifest['source_generator_recovery']['sha256'],
                    review_sha256=review_digest))
    return manifest,arrays,Path(path),info


def resolve_generator_materials(info, obs, model, branches, records):
    """Check native bindings against actual observations and selected geometry."""
    proof=info.get('source_generator_recovery')
    if not proof:return records
    result=list(records)
    for original in proof['bindings']:
        b=copy.deepcopy(original);row=b['source_row'];mi=b['resolved_model_material_index']
        _require(int(obs.native_pts[row])==b['native_pts']
                 and str(obs.source_point_ids[row])==b['source_point_id']
                 and str(obs.source_target_labels[row])==b['original_source_label']
                 and np.array_equal(obs.source_uv[row],b['source_uv']), 'actual immutable observation changed')
        material=model['materials'][mi]
        _require(material['label']==b['resolved_model_target_id']
                 and all(0<=j<len(material['paths']) and material['paths'][j]['variant_id']==b['resolved_variant_id']
                         for j in b['allowed_path_indices']), 'selected finite material/variant changed')
        b['saved_branch_in_confirmed_support']=int(branches[row]) in b['allowed_path_indices']
        b['review_origin']='codex_primary_native_frame_visual_review'
        b['not_a_direct_human_confirmation']=True
        b['no_human_anchor_used']=True
        b['recovery_review_sha256']=proof['review_sha256']
        result.append(b)
    return result


def export_source_generator_recovery(parent_manifest, review_manifest, output_dir):
    """Export two complete derived vectors; reference every immutable raw asset."""
    from .semantic_resolution import load_semantics
    parent=Path(parent_manifest).resolve();review_path=Path(review_manifest).resolve()
    review_digest=sha256_file(review_path)
    _require(sha256_file(parent)==PARENT and review_digest in REVIEW_POLICIES,'wrong reviewed input identity')
    pm=json.loads(parent.read_text(encoding='utf-8'));n=pm['observations']
    _,_,_,info=load_semantics(parent,n,expected_sha256=PARENT)
    review=json.loads(review_path.read_text(encoding='utf-8'));bindings=review['bindings']
    material=info['resolved_material_index'].copy();variant=info['resolved_variant_index'].copy()
    for b in bindings:
        row=b['source_row'];material[row]=b['resolved_model_material_index']
        variant[row]=pm['resolved_variant_ids'].index(b['resolved_variant_id'])
    out=Path(output_dir).resolve()
    _require(not out.exists() or not any(out.iterdir()),'never overwrite a frozen candidate')
    out.mkdir(parents=True,exist_ok=True);manifest=_rebase(copy.deepcopy(pm),parent.parent)
    for key,name,a in [('resolved_material_index','resolved_material.u32',material),
                       ('resolved_variant_index','resolved_variant.u32',variant)]:
        a.astype('<u4').tofile(out/name);spec=copy.deepcopy(pm[key]);spec.pop('path',None)
        spec.update(file=name,sha256=sha256_file(out/name));manifest[key]=spec
    proof={'schema':SCHEMA,'rule_id':RULE,'parent_semantic_sha256':PARENT,'observations':n,
           'review_manifest':{'path':str(review_path),'sha256':review_digest},'bindings':copy.deepcopy(bindings)}
    atomic_json(out/'source_generator_recovery.json',proof)
    manifest.update(source_generator_parent_manifest={'path':str(parent),'sha256':PARENT},
                    source_generator_recovery={'file':'source_generator_recovery.json',
                                               'sha256':sha256_file(out/'source_generator_recovery.json')})
    atomic_json(out/'manifest.json',manifest)
    load_semantics(out/'manifest.json',n,expected_sha256=sha256_file(out/'manifest.json'))
    return out/'manifest.json'


def main():
    import argparse
    from .semantic_resolution import load_semantics
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);args=parser.parse_args()
    manifest,_,_,info=load_semantics(args.manifest,182017)
    _require(bool(info.get('source_generator_recovery')),'selected packet has no recovery')
    print(json.dumps({'semantic_sha256':sha256_file(args.manifest),'raw_observations':manifest['observations'],
                      'active_observations':manifest['active_observations'],
                      'changed_rows':len(info['source_generator_recovery']['bindings']),
                      'target_counts':REVIEW_POLICIES[info['source_generator_recovery']['review_sha256']]['targets'],
                      'review_sha256':info['source_generator_recovery']['review_sha256'],'parent_sha256':PARENT},sort_keys=True))

if __name__=='__main__':main()
