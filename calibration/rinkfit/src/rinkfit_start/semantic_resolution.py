"""Versioned evidence-scoped model bindings over immutable source observations.

The original five arrays, source variants, source runs and whitening operators
remain the parent packet's exact bytes.  A resolution changes only the world
interpretation of explicitly confirmed, independent ordinary rows.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .data import sha256_file
from .runtime import atomic_json
from .semantic_exclusion import load_exclusion

SCHEMA = 'rinkfit.confirmed_material_resolution.v1'


def _path(root, spec):
    return Path(root)/spec.get('file', spec.get('path', ''))


def read_binary(root, spec):
    path = _path(root, spec)
    if sha256_file(path) != spec['sha256']:
        raise ValueError('Resolved semantic binary checksum changed: '+str(path))
    value = np.fromfile(path, dtype=spec['dtype'])
    if value.size != int(np.prod(spec['shape'])):
        raise ValueError('Resolved semantic binary shape changed')
    return value.reshape(spec['shape'])


def _array_signature(spec):
    return {k: spec[k] for k in ['sha256', 'dtype', 'shape']}


def load_semantics(manifest_path, n, *, expected_sha256=None, _ancestors=()):
    """Load the selected packet and validate any immutable parent/overlay chain."""
    path = Path(manifest_path).resolve()
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Selected semantic identity does not match the state')
    if str(path) in _ancestors or len(_ancestors) >= 8:
        raise ValueError('Invalid semantic parent chain')
    m = json.loads(path.read_text(encoding='utf-8'))
    if 'source_audit_parent_manifest' in m or 'source_audit_revision' in m:
        from .source_audit_revision import load_source_audit_revision
        return load_source_audit_revision(path,m,n,
            lambda p,count,**kw:load_semantics(p,count,_ancestors=(*_ancestors,str(path)),**kw))
    if 'source_component_parent_manifest' in m or 'source_component_recovery' in m:
        from .source_component_recovery import load_source_component_recovery
        return load_source_component_recovery(path,m,n,
            lambda p,count,**kw:load_semantics(p,count,_ancestors=(*_ancestors,str(path)),**kw))
    if 'source_generator_parent_manifest' in m or 'source_generator_recovery' in m:
        from .source_generator_recovery import load_source_generator_recovery
        return load_source_generator_recovery(path,m,n,
            lambda p,count,**kw:load_semantics(p,count,_ancestors=(*_ancestors,str(path)),**kw))
    if 'source_order_parent_manifest' in m or 'source_order_overlay' in m:
        from .semantic_order_overlay import load_order_overlay
        return load_order_overlay(path,m,n,
            lambda p,count,**kw:load_semantics(p,count,_ancestors=(*_ancestors,str(path)),**kw))
    arrays = {k: read_binary(path.parent, s) for k, s in m['arrays'].items()}
    if arrays['kind'].shape != (n,) or m['observations'] != n:
        raise ValueError('Selected semantic source count mismatch')
    for spec in m.get('assets', {}).values():
        if sha256_file(_path(path.parent, spec)) != spec['sha256']:
            raise ValueError('Selected semantic provenance changed')
    exclusion = load_exclusion(m,path,arrays,n,
        lambda p,count,**kw:load_semantics(p,count,_ancestors=(*_ancestors,str(path)),**kw))
    parent = m.get('parent_semantic_manifest')
    if not parent:
        if any(k in m for k in ['resolved_material_index', 'resolved_variant_index',
                                'confirmed_bindings', 'resolved_variant_ids']):
            raise ValueError('A material resolution requires its immutable parent')
        return m, arrays, path, {'enabled': False, 'query_semantic_sha256': digest,
                                'bindings': [],'source_exclusion':exclusion}
    parent_path = _path(path.parent, parent)
    pm, pa, _, inherited = load_semantics(parent_path, n,
        expected_sha256=parent['sha256'], _ancestors=(*_ancestors, str(path)))
    if inherited['enabled']:
        raise ValueError('Stacked binding overlays need an explicit merged transition')
    if set(m['arrays']) != set(pm['arrays']) or set(m['assets']) != set(pm['assets']):
        raise ValueError('A material overlay cannot add or remove original arrays/assets')
    for k, spec in pm['arrays'].items():
        if _array_signature(m['arrays'][k]) != _array_signature(spec):
            raise ValueError('A material overlay changed original semantic array: '+k)
    for k, spec in pm['assets'].items():
        if m['assets'][k]['sha256'] != spec['sha256']:
            raise ValueError('A material overlay changed original semantic asset: '+k)
    for k, value in pm.items():
        if k not in ['arrays', 'assets'] and m.get(k) != value:
            raise ValueError('A material overlay changed original semantic metadata: '+k)
    overlay_spec = m['confirmed_bindings']; overlay_path = _path(path.parent, overlay_spec)
    if sha256_file(overlay_path) != overlay_spec['sha256']:
        raise ValueError('Human-confirmed binding overlay checksum changed')
    overlay = json.loads(overlay_path.read_text(encoding='utf-8'))
    if (overlay.get('schema') != SCHEMA or overlay.get('observations') != n
            or overlay.get('parent_semantic_sha256') != parent['sha256']):
        raise ValueError('Binding overlay parent/source registry mismatch')
    raw_material = read_binary(path.parent, m['original_material_index'])
    resolved = read_binary(path.parent, m['resolved_material_index'])
    variants = read_binary(path.parent, m['resolved_variant_index'])
    if any(x.shape != (n,) or x.dtype != np.dtype('<u4')
           for x in [raw_material, resolved, variants]):
        raise ValueError('Resolved semantic indices require complete uint32 vectors')
    ids = m['resolved_variant_ids']
    if ids[:len(pm['variant_ids'])] != pm['variant_ids'] or len(set(ids)) != len(ids):
        raise ValueError('Original source variant registry changed')
    if np.any(variants >= len(ids)):
        raise ValueError('Unknown resolved material variant')
    bindings = overlay.get('bindings', []); rows = [b['source_row'] for b in bindings]
    if (not rows or len(set(rows)) != len(rows) or any(type(i) is not int or i < 0 or i >= n for i in rows)):
        raise ValueError('Invalid unique confirmed source-row registry')
    changed = np.flatnonzero((resolved != raw_material) | (variants != pa['variant_index']))
    if set(changed.tolist()) != set(rows):
        raise ValueError('Resolved array differences do not match confirmed bindings')
    if np.any(pa['kind'][rows] != 0):
        raise ValueError('This binding transition only supports ordinary finite rows')
    source_run_path = _path(path.parent, m['assets']['source_runs.jsonl'])
    with source_run_path.open(encoding='utf-8') as handle:
        for line in handle:
            if set(json.loads(line)['source_rows']).intersection(rows):
                raise ValueError('A binding inside an ordered source run requires a separate resolved topology proof')
    for b in bindings:
        i = b['source_row']
        if (int(raw_material[i]) != b['original_material_index']
                or int(resolved[i]) != b['resolved_model_material_index']
                or pm['variant_ids'][int(pa['variant_index'][i])] != b['original_variant_id']
                or ids[int(variants[i])] != b['resolved_variant_id']):
            raise ValueError('Confirmed row binding does not match its complete index arrays')
        human = 'direct_human_confirmation' in b
        review = 'source_image_review' in b
        if human == review:
            raise ValueError('Binding requires exactly one direct-human or source-image evidence kind')
        proof = b['direct_human_confirmation'] if human else b['source_image_review']
        proof_path = _path(overlay_path.parent, proof)
        if sha256_file(proof_path) != proof['sha256']:
            raise ValueError('Direct human confirmation checksum changed')
        confirmation = json.loads(proof_path.read_text(encoding='utf-8'))
        if human:
            if (confirmation.get('schema') != 'rinkfit.direct-human-material-confirmation.v1'
                    or confirmation.get('source_row') != i
                    or confirmation.get('original_npz_source_identity') != overlay['source_identity']
                    or confirmation.get('confirmation_applies_to_other_source_rows') is not False
                    or confirmation.get('raw_arrays_W_E_obligations_remain_immutable') is not True
                    or 'review_origin' in confirmation
                    or confirmation.get('not_a_direct_human_confirmation') is True):
                raise ValueError('Binding is not scoped to the direct human-confirmed source row')
        else:
            if (confirmation.get('schema') != 'rinkfit.native-image-material-review.v1'
                    or confirmation.get('review_origin') != 'codex_primary_native_frame_visual_review'
                    or confirmation.get('source_row') != i
                    or confirmation.get('original_npz_source_identity') != overlay['source_identity']
                    or confirmation.get('review_applies_to_other_source_rows') is not False
                    or confirmation.get('raw_arrays_W_E_obligations_remain_immutable') is not True
                    or confirmation.get('not_a_direct_human_confirmation') is not True
                    or confirmation.get('no_extension_of_anchor_human_confirmation') is not True
                    or confirmation.get('resolved_model_target_id') != b['resolved_model_target_id']):
                raise ValueError('Binding is not scoped to the distinct native-image material review')
            for key in ['exact_native_frame','native_source_video','visual_diagnostic','original_fragment_lineage']:
                asset = confirmation[key]
                if sha256_file(_path(proof_path.parent,asset)) != asset['sha256']:
                    raise ValueError('Native-image review asset checksum changed: '+key)
            frame = confirmation['exact_native_frame']
            if (frame.get('native_pts') != confirmation['native_pts']
                    or frame.get('width') != 1920 or frame.get('height') != 1080):
                raise ValueError('Native-image review frame clock/dimensions differ')
            anchor_spec = confirmation['separate_human_confirmed_anchor']
            anchor_path = _path(proof_path.parent,anchor_spec)
            if sha256_file(anchor_path) != anchor_spec['sha256']:
                raise ValueError('Separate human-confirmed anchor checksum changed')
            anchor = json.loads(anchor_path.read_text(encoding='utf-8'))
            if (anchor.get('schema') != 'rinkfit.direct-human-material-confirmation.v1'
                    or anchor.get('source_row') != anchor_spec.get('source_row')
                    or type(anchor.get('source_row')) is not int
                    or not 0 <= anchor['source_row'] < n
                    or anchor.get('source_row') == i
                    or anchor.get('native_pts') != confirmation['native_pts']
                    or anchor.get('original_npz_source_identity') != overlay['source_identity']
                    or anchor.get('confirmation_applies_to_other_source_rows') is not False
                    or anchor.get('raw_arrays_W_E_obligations_remain_immutable') is not True):
                raise ValueError('Source-image review cannot extend or impersonate its separate human anchor')
            b['_verified_anchor'] = anchor
        evidence = b.get('exact_source_geometry_provenance', [])
        if isinstance(evidence, dict): evidence = [evidence]
        if not evidence:
            raise ValueError('A resolved target requires its traced original material geometry')
        for reference in evidence:
            reference_path = _path(overlay_path.parent, reference)
            if sha256_file(reference_path) != reference['sha256']:
                raise ValueError('Resolved target geometry provenance checksum changed')
            if 'json_pointer' in reference:
                value = json.loads(reference_path.read_text(encoding='utf-8'))
                for token in reference['json_pointer'].split('/')[1:]:
                    token = token.replace('~1', '/').replace('~0', '~')
                    value = value[int(token)] if isinstance(value, list) else value[token]
        b['_verified_confirmation'] = confirmation
    return m, arrays, path, {'enabled': True, 'parent_semantic_sha256': parent['sha256'],
        'query_semantic_sha256': inherited['query_semantic_sha256'],
        'overlay_path': str(overlay_path), 'overlay_sha256': overlay_spec['sha256'],
        'source_identity': overlay['source_identity'], 'bindings': bindings,
        'original_material_index': raw_material, 'resolved_material_index': resolved,
        'resolved_variant_index': variants, 'resolved_variant_ids': ids,
        'source_exclusion':exclusion}


def resolve_materials(info, original_material, original_variants, obs, model, branches,
                      source_identity):
    """Verify actual raw keys and map only the confirmed model interpretation."""
    original_material = np.asarray(original_material, dtype='<u4')
    expected = np.asarray(original_variants).copy()
    if not info['enabled']:
        return original_material.copy(), expected, []
    if (info['source_identity'] != source_identity
            or not np.array_equal(info['original_material_index'], original_material)):
        raise ValueError('Resolved material source identity/raw material registry mismatch')
    resolved = info['resolved_material_index'].copy()
    ids = info['resolved_variant_ids']
    # A wider string dtype is essential when a newly named variant is longer.
    expected = np.asarray(ids)[info['resolved_variant_index']]
    records = []
    if np.any(resolved >= len(model['materials'])):
        raise ValueError('Resolved material is missing from the selected model')
    for binding in info['bindings']:
        b = {k:v for k,v in binding.items() if not k.startswith('_')}; i = b['source_row']
        c = binding['_verified_confirmation']; m = model['materials'][int(resolved[i])]
        if (int(obs.native_pts[i]) != b['native_pts'] or int(obs.native_pts[i]) != c['native_pts']
                or str(obs.source_point_ids[i]) != b['source_point_id']
                or str(obs.source_point_ids[i]) != c['raw_source_point_id']
                or str(obs.source_target_labels[i]) != b['original_source_label']
                or str(obs.source_target_labels[i]) != c['raw_source_label']
                or not np.array_equal(obs.source_uv[i], b['source_uv'])
                or not np.array_equal(obs.source_uv[i], c['source_uv'])):
            raise ValueError('Human binding does not match the immutable composite source key/UV')
        anchor = binding.get('_verified_anchor')
        if anchor is not None:
            j = anchor['source_row']
            if (int(obs.native_pts[j]) != anchor['native_pts']
                    or str(obs.source_point_ids[j]) != anchor['raw_source_point_id']
                    or str(obs.source_target_labels[j]) != anchor['raw_source_label']
                    or not np.array_equal(obs.source_uv[j],anchor['source_uv'])):
                raise ValueError('Separate human anchor does not match its own immutable source key/UV')
        if m['label'] != b['resolved_model_target_id']:
            raise ValueError('Resolved target ID is inconsistent with the selected model index')
        allowed = b['allowed_path_indices']
        if (not allowed or any(type(j) is not int or j < 0 or j >= len(m['paths'])
                              or m['paths'][j].get('variant_id') != b['resolved_variant_id'] for j in allowed)):
            raise ValueError('Resolved finite path provenance does not match the selected model')
        # The state can still have an illegal branch; replay keeps that row and
        # reports its support failure instead of rewriting the saved state.
        b['saved_branch_in_confirmed_support'] = int(branches[i]) in allowed
        records.append(b)
    from .source_generator_recovery import resolve_generator_materials
    records = resolve_generator_materials(info, obs, model, branches, records)
    from .source_component_recovery import resolve_component_materials
    records = resolve_component_materials(info, obs, model, branches, records)
    return resolved, expected, records


def export_resolution(parent_manifest, original_material_spec, bindings, output_dir,
                      source_identity):
    """Freeze a separate semantic candidate; caller supplies traced target IDs."""
    parent_manifest = Path(parent_manifest).resolve(); parent_hash = sha256_file(parent_manifest)
    pm = json.loads(parent_manifest.read_text(encoding='utf-8')); n = pm['observations']
    _, pa, _, inherited = load_semantics(parent_manifest, n)
    if inherited['enabled']:
        raise ValueError('Export requires the original immutable source semantic parent')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Never overwrite a frozen semantic candidate')
    out.mkdir(parents=True, exist_ok=True)
    original_material_spec = copy.deepcopy(original_material_spec)
    original_material_spec['file'] = str(Path(original_material_spec['file']).resolve())
    raw = read_binary(out, original_material_spec)
    material = raw.copy(); variant = pa['variant_index'].copy(); ids = list(pm['variant_ids'])
    for b in bindings:
        i = b['source_row']; material[i] = b['resolved_model_material_index']
        if b['resolved_variant_id'] not in ids: ids.append(b['resolved_variant_id'])
        variant[i] = ids.index(b['resolved_variant_id'])
    def save(name, value):
        np.asarray(value, dtype='<u4').tofile(out/name)
        return {'file':name,'dtype':'<u4','shape':[n],'layout':'C','sha256':sha256_file(out/name)}
    m = copy.deepcopy(pm)
    for category in ['arrays', 'assets']:
        for spec in m[category].values():spec['file'] = str(_path(parent_manifest.parent, spec).resolve())
    overlay = {'schema':SCHEMA,'observations':n,'source_identity':source_identity,
        'parent_semantic_sha256':parent_hash,'raw_five_arrays_and_original_W_E_unchanged':True,
        'bindings':copy.deepcopy(bindings)}
    atomic_json(out/'confirmed_bindings.json',overlay)
    m.update(parent_semantic_manifest={'path':str(parent_manifest),'sha256':parent_hash},
        original_material_index=original_material_spec,
        resolved_material_index=save('resolved_material.u32',material),
        resolved_variant_index=save('resolved_variant.u32',variant),resolved_variant_ids=ids,
        confirmed_bindings={'file':'confirmed_bindings.json','sha256':sha256_file(out/'confirmed_bindings.json')})
    atomic_json(out/'manifest.json',m)
    load_semantics(out/'manifest.json',n,expected_sha256=sha256_file(out/'manifest.json'))
    return out/'manifest.json'
