"""Evidence-scoped correction of a falsely derived source-run order registry.

This policy is deliberately tied to the reviewed immutable parent and producer.
Original navigation records remain exact audit assets; they are not boundary
traces merely because extraction query indices were consecutive.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from .data import sha256_file
from .runtime import atomic_json

SCHEMA = 'rinkfit.source_order_evidence_overlay.v1'
RULE = 'current_native_independent_normal_queries_without_boundary_trace_v1'
PARENT = 'bb74173425444d4dbe925c50c5bf42f3739663b602d0bb839a0392f040dfb64d'
PRODUCERS = {
    'normal_query_producer': '3d30d5d595fa7299ad53191ea5e3c2844ae96e955ce42e18b5df67d938bf4f85',
    'derived_order_stamper': '00b4d293bcaa0049d17f71430f40cccf00151e22d27f07775f0ff44c9f1f6d6e',
    'semantic_run_exporter': 'e9072669f64e56f490f03eea23db38bbe3f90bcce6ddd580de7ebb530d6e7b17',
}
FACTS = {
    'spec': 'd085d351959380c6dbd5bd581d84cad119640cdd80818343ec6dbb11b4abac93',
    'order_counterexample': '38318f642da42767cd171818bcff0a3f02b4c56966c7d06184855a75979b4e5c',
    'hole_support_diagnostic': 'd79abf8a483484d7b47419e152d9355d98b3c3fa3bc27d275a460725083c94e6',
}
# Exact known producer output vocabulary: an added trace/order proof cannot be
# silently reclassified by this policy. Region identity is not a boundary trace.
SEGMENT_KEYS = set('closed localization_sigma_px measurement_origin native_connected_component_rgb_profile_support_fraction native_simplification_tolerance_px pixels_measured_in_this_frame point_ids pre_compaction_vertex_count retained_pre_compaction_vertex_indices source_clear_component_identity_verified_by_current_shape source_query_count sparse_representation_max_deviation_px transition_iqr_px uv vertex_count'.split())
FEATURE_KEYS = set('allowed_for_fit current_source_only_readout id label label_zh model_or_template_generated_observation physical_surface review_layer segments source_identity_recovery target_id world_correspondence_status'.split())


def _path(root, spec):
    return Path(root)/spec.get('file', spec.get('path', ''))


def _verified(root, spec):
    path = _path(root, spec)
    if sha256_file(path) != spec['sha256']:
        raise ValueError('Source-order evidence checksum changed: '+str(path))
    return path


def _signature(value):
    if isinstance(value, dict):
        return {k:_signature(v) for k,v in value.items() if k not in ('file','path')}
    if isinstance(value, list):return [_signature(v) for v in value]
    return value


def _rebase(value, root):
    if isinstance(value, dict):
        value = {k:_rebase(v,root) for k,v in value.items()}
        if 'sha256' in value:
            for key in ('file','path'):
                if key in value:value[key] = str((root/value[key]).resolve())
        return value
    if isinstance(value,list):return [_rebase(v,root) for v in value]
    return value


def _verify_descriptors(root, value):
    if isinstance(value, dict):
        if 'sha256' in value and ('file' in value or 'path' in value):
            _verified(root,value)
        for child in value.values():_verify_descriptors(root,child)
    elif isinstance(value, list):
        for child in value:_verify_descriptors(root,child)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def classify_runs(source_path, n):
    records = []; cache = {}; total_rows = 0
    for ordinal, source in enumerate(_jsonl(source_path)):
        rows = source['source_rows']; vertices = source['source_vertex_indices']
        if (len(rows)<2 or len(rows)!=len(vertices) or len(set(rows))!=len(rows)
                or any(type(i) is not int or i<0 or i>=n for i in rows)
                or any(type(i) is not int or i<0 for i in vertices)
                or any(b!=a+1 for a,b in zip(vertices,vertices[1:]))):
            raise ValueError('Invalid original navigation run registry')
        ref = source['source_packet']; key = (ref['path'],ref['sha256'])
        if key not in cache:
            packet_path = _verified(source_path.parent,ref)
            packet = json.loads(packet_path.read_text(encoding='utf-8'))
            cache[key] = {f['id']:f for f in packet['features']}
        feature = cache[key][source['source_feature_id']]
        segment = feature['segments'][source['source_segment_index']]
        if (set(feature)-FEATURE_KEYS or set(segment)-SEGMENT_KEYS
                or feature.get('current_source_only_readout') is not True
                or feature.get('model_or_template_generated_observation') is not False
                or feature.get('world_correspondence_status') != 'named_same_physical_print_component'
                or segment.get('measurement_origin') != 'current_native_material_normal_half_transition'
                or segment.get('pixels_measured_in_this_frame') is not True
                or max(vertices)>=len(segment['uv'])):
            raise ValueError('Run is outside the independent normal-query policy; preserve its trace obligations')
        record_bytes = json.dumps(source,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        records.append({'run_id':f'run/{ordinal:06d}', 'source_record_sha256':hashlib.sha256(record_bytes).hexdigest(),
            'source_rows':rows, 'source_vertex_indices':vertices,
            'classification':'navigation_query_sequence_only',
            'measurement_origin':segment['measurement_origin'],
            'independent_boundary_trace_verified':False,
            'region_identity_is_not_boundary_trace':True,
            'active_hard_order_factor':False})
        total_rows += len(rows)
    return records, total_rows, len(cache)


def _check_evidence(root, values, expected):
    if len(values)!=len(expected) or {s.get('role') for s in values}!=set(expected):
        raise ValueError('Incomplete source-order producer/fact evidence')
    for spec in values:
        if spec.get('sha256')!=expected[spec['role']]:
            raise ValueError('Unreviewed source-order producer/fact identity')
        _verified(root,spec)


def load_order_overlay(path, manifest, n, parent_loader):
    root = Path(path).parent
    keys = {'source_order_parent_manifest','source_order_overlay'}
    if not keys <= set(manifest):raise ValueError('Incomplete source-order overlay')
    ref = manifest['source_order_parent_manifest']
    if ref['sha256'] != PARENT:raise ValueError('Source-order policy has the wrong reviewed parent')
    pm, arrays, _, inherited = parent_loader(_path(root,ref),n,expected_sha256=ref['sha256'])
    if inherited.get('source_order_overlay'):raise ValueError('Stacked source-order corrections are not authorized')
    if set(manifest)!=set(pm)|keys:raise ValueError('Source-order overlay changed unrelated fields')
    for key,value in pm.items():
        if key=='assets':continue
        if _signature(manifest[key])!=_signature(value):
            raise ValueError('Source-order overlay changed parent semantic identity: '+key)
    if set(manifest['assets'])!=set(pm['assets']):raise ValueError('Source-order overlay deleted original audit assets')
    for key,value in pm['assets'].items():
        if key!='source_runs.jsonl' and _signature(manifest['assets'][key])!=_signature(value):
            raise ValueError('Source-order overlay changed original semantic asset: '+key)
        _verified(root,manifest['assets'][key])
    # A relocation is permitted only when the file at its new address really
    # has the original bytes. Never merely compare descriptor hash claims.
    for section in ('arrays','assets','original_material_index','resolved_material_index',
                    'resolved_variant_index','confirmed_bindings','source_exclusion',
                    'active_source_mask','active_statistics'):
        if section in manifest:_verify_descriptors(root,manifest[section])
    overlay_path = _verified(root,manifest['source_order_overlay'])
    overlay = json.loads(overlay_path.read_text(encoding='utf-8'))
    if (overlay.get('schema')!=SCHEMA or overlay.get('rule_id')!=RULE
            or overlay.get('parent_semantic_sha256')!=ref['sha256']
            or overlay.get('observations')!=n
            or overlay.get('raw_arrays_W_E_and_authorities_unchanged') is not True
            or _signature(overlay.get('audit_source_runs'))!=_signature(pm['assets']['source_runs.jsonl'])
            or _signature(overlay.get('active_hard_source_runs'))!=_signature(manifest['assets']['source_runs.jsonl'])):
        raise ValueError('Source-order correction authority/parent/audit mismatch')
    _check_evidence(overlay_path.parent,overlay['producer_evidence'],PRODUCERS)
    _check_evidence(overlay_path.parent,overlay['evidence'],FACTS)
    source_path = _verified(overlay_path.parent,overlay['audit_source_runs'])
    classifications, total_rows, packets = classify_runs(source_path,n)
    saved = _jsonl(_verified(overlay_path.parent,overlay['classifications']))
    hard_path = _verified(overlay_path.parent,overlay['active_hard_source_runs'])
    if (saved!=classifications or hard_path.read_bytes()!=b''
            or overlay.get('audit_run_count')!=len(saved) or overlay.get('audit_row_count')!=total_rows
            or overlay.get('verified_source_packet_count')!=packets
            or overlay.get('active_hard_run_count')!=0 or overlay.get('active_hard_row_count')!=0):
        raise ValueError('Incomplete or fabricated source-order classification/hard registry')
    info = dict(inherited)
    info['source_order_overlay'] = dict(_rebase(overlay,overlay_path.parent), overlay_path=str(overlay_path),overlay_sha256=manifest['source_order_overlay']['sha256'])
    return manifest, arrays, Path(path), info


def export_order_overlay(parent_manifest, output_dir, *, producer_evidence, evidence):
    from .semantic_resolution import load_semantics
    parent = Path(parent_manifest).resolve(); digest = sha256_file(parent)
    if digest!=PARENT:raise ValueError('Wrong immutable parent for reviewed source-order correction')
    pm = json.loads(parent.read_text(encoding='utf-8'));n=pm['observations']
    load_semantics(parent,n,expected_sha256=digest)
    _check_evidence(parent.parent,producer_evidence,PRODUCERS)
    _check_evidence(parent.parent,evidence,FACTS)
    source_spec = _rebase(pm['assets']['source_runs.jsonl'],parent.parent)
    classifications,rows,packets = classify_runs(_verified(parent.parent,source_spec),n)
    out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Never overwrite an immutable source-order overlay')
    out.mkdir(parents=True,exist_ok=True)
    (out/'active_source_runs.jsonl').write_bytes(b'')
    with (out/'classifications.jsonl').open('w',encoding='utf-8',newline='\n') as handle:
        for record in classifications:handle.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    def spec(name):return {'file':name,'sha256':sha256_file(out/name)}
    overlay={'schema':SCHEMA,'rule_id':RULE,'parent_semantic_sha256':digest,'observations':n,
        'audit_source_runs':source_spec,'classifications':spec('classifications.jsonl'),
        'active_hard_source_runs':spec('active_source_runs.jsonl'),
        'audit_run_count':len(classifications),'audit_row_count':rows,'verified_source_packet_count':packets,
        'active_hard_run_count':0,'active_hard_row_count':0,
        'producer_evidence':_rebase(producer_evidence,parent.parent),'evidence':_rebase(evidence,parent.parent),
        'raw_arrays_W_E_and_authorities_unchanged':True}
    atomic_json(out/'source_order_overlay.json',overlay)
    manifest=_rebase(copy.deepcopy(pm),parent.parent)
    manifest['assets']['source_runs.jsonl']=spec('active_source_runs.jsonl')
    manifest.update(source_order_parent_manifest={'path':str(parent),'sha256':digest},
        source_order_overlay=spec('source_order_overlay.json'))
    atomic_json(out/'manifest.json',manifest)
    load_semantics(out/'manifest.json',n)
    return out/'manifest.json'
