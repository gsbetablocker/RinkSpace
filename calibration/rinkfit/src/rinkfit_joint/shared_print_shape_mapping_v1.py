"""Read-only audit of actual shared print shape consumers and identity mappings.

This tool never creates source observations or promotes numerical seed vertices to
physical templates. Alias links need an existing source-template declaration or an
explicit producer alias record and exact current normalized contour equivalence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read_ref(ref):
    path = Path(ref.get('file', ref.get('path', '')).replace('\\\\', '/'))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('shape mapping dependency SHA mismatch: ' + str(path))
    return json.loads(raw), dict(file=str(path), sha256=ref['sha256'])


def file_ref(path):
    path = Path(path)
    return dict(file=str(path), sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())


def nested_objects(value):
    if isinstance(value, str):
        try:
            yield from nested_objects(json.loads(value))
        except (ValueError, TypeError):
            pass
    elif isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nested_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_objects(child)


def normalized_points(model, path):
    q = np.asarray(path['points'], dtype='<f8')
    bi = path.get('geometry_block', -1)
    if bi >= 0:
        a = np.asarray(model['blocks'][bi]['values'], dtype=float)
        if a.shape != (6,):
            raise ValueError('kind2 shape is not the existing six coefficient affine')
        q = q @ np.array([[a[0], a[3]], [a[1], a[4]]]) + a[[2, 5]]
    return np.ascontiguousarray(q, dtype='<f8')


def shape_inventory(model):
    records = {}
    for mi, mat in enumerate(model['materials']):
        for pi, path in enumerate(mat['paths']):
            if path['kind'] != 2 or path.get('physical_point') or len(path['points']) < 2:
                continue
            if '.print.' not in mat['label']:
                continue
            q = normalized_points(model, path)
            source_points = np.asarray(path['points'], dtype='<f8')
            source_repeated = np.array_equal(source_points[0], source_points[-1])
            repeated = bool(path.get('closed', False)) and source_repeated
            if np.array_equal(q[0], q[-1]) and not repeated:
                raise ValueError('coincident normalized endpoints have no declared source endpoint alias')
            # kind2 consumes N-1 segments even when its unused closed flag says
            # true. A kind6 migration must not invent an absent closing edge.
            records[(mi, pi)] = dict(
                material_index=mi, path_index=pi, label=mat['label'],
                current_path_id=path['id'], current_variant_id=path['variant_id'],
                geometry_block=path.get('geometry_block', -1),
                advertisement_block=path['advertisement_block'],
                advertisement_offset=path.get('advertisement_offset', 0),
                points=q.tolist(), point_count=len(q), runtime_segments=len(q)-1,
                source_closed_flag=bool(path.get('closed', False)),
                closure_evidence=dict(kind='explicit_original_path_closed_and_repeated_ordered_source_vertex' if repeated else 'legacy_runtime_open_chain', declared_closed=bool(path.get('closed', False)), original_first_last_exactly_identical=bool(source_repeated), normalized_float_coincidence_used_as_authority=False),
                normalized_kind6_closed=repeated,
                normalized_kind6_vertices=len(q)-int(repeated),
                misleading_closed_flag=bool(path.get('closed', False)) != repeated,
                initializer_claim=any(
                    any('initial' in str(k).lower() and bool(v) for k, v in obj.items())
                    for obj in nested_objects(path.get('provenance', ''))),
                raw_rows=0, fit_rows=0, fit_shots={}, fit_frames=set(),
                normalized_seed_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                numerical_seed_is_physical_template=False,
            )
    return records


def existing_links(model, records, alias_proof=None, source_links=()):
    labels = {m['label']: i for i, m in enumerate(model['materials'])}
    links, unresolved = [], []
    seen = set()

    def link(a, b, evidence):
        if a == b or a not in records or b not in records:
            return
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        x, y = records[a], records[b]
        # Same shape normal form is required for an exact chart migration.
        # Declared same entity with different seeds is reported for inspection;
        # this tool never chooses the lower-error model as source identity.
        same = x['runtime_segments'] == y['runtime_segments'] and np.array_equal(x['points'], y['points'])
        item = dict(a=list(a), b=list(b), evidence=evidence,
                    exact_current_normalized_shape=same)
        if same:
            links.append(item)
        else:
            unresolved.append(item)

    for a, b, evidence in source_links:
        link(tuple(a), tuple(b), evidence)

    for (mi, pi), rec in records.items():
        path = model['materials'][mi]['paths'][pi]
        for obj in nested_objects(path.get('provenance', '')):
            source = obj.get('source_template')
            if not isinstance(source, str) or source not in labels:
                continue
            target = labels[source]
            if pi >= len(model['materials'][target]['paths']):
                continue
            rep = obj.get('representation', '')
            if rep not in ('same_confirmed_print_template_shared_unfolded_dimensions',
                           'previously_missing_instance_of_user_confirmed_same_print'):
                continue
            link((mi, pi), (target, pi), dict(kind='existing_explicit_source_template',
                 representation=rep, source_template=source, user_pair=obj.get('user_pair')))
    for mi_text, refinement in model.get('printed_material_path_refinements', {}).items():
        targets = refinement.get('canonical_materials', [])
        if not targets:
            continue
        mi = int(mi_text)
        for pi in range(len(model['materials'][mi]['paths'])):
            source_key = (mi, pi)
            if source_key not in records:
                continue
            rec = records[source_key]
            matches = []
            for target in targets:
                for tpi in range(len(model['materials'][target]['paths'])):
                    target_key = (target, tpi)
                    other = records.get(target_key)
                    if other and rec['advertisement_block'] == other['advertisement_block'] and rec['advertisement_offset'] == other['advertisement_offset'] and rec['runtime_segments'] == other['runtime_segments'] and np.array_equal(rec['points'], other['points']):
                        matches.append(target_key)
            for target_key in matches:
                link(source_key, target_key, dict(kind='existing_explicit_source_refinement_canonical_material',
                     refinement_material_index=mi, canonical_materials=targets,
                     source_evidence=refinement.get('source_evidence', [])))
    if alias_proof:
        for pair in alias_proof['pairs']:
            originals = {model['materials'][mi]['label'].split('.', 1)[1]: mi
                         for mi in pair['updated_materials']}
            for mi in pair['display_aliases']:
                label = model['materials'][mi]['label']
                source = originals.get(label.split('.', 1)[1])
                if source is None:
                    raise ValueError('declared producer alias lost its exact original label')
                for pi in range(len(model['materials'][mi]['paths'])):
                    link((mi, pi), (source, pi), dict(kind='existing_producer_display_alias',
                         paired_ids=pair['ids'], alias_material_index=mi))
    return links, unresolved


def summarize(records, links):
    parent = {key: key for key in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for link in links:
        a, b = (tuple(link[k]) for k in ('a', 'b'))
        parent[find(a)] = find(b)
    groups = {}
    for key, rec in records.items():
        canonical = find(key)
        group = groups.setdefault(canonical, dict(members=[], fit_rows=0, shots={},
                                  frames=set(), vertices=rec['normalized_kind6_vertices']))
        group['members'].append(list(key))
        group['fit_rows'] += rec['fit_rows']
        group['frames'].update(rec['fit_frames'])
        for shot, n in rec['fit_shots'].items():
            group['shots'][shot] = group['shots'].get(shot, 0) + n
    output = []
    for key, group in groups.items():
        group['canonical_path'] = list(key)
        group['fit_frames'] = len(group.pop('frames'))
        group['fit_shots'] = group.pop('shots')
        group['shared_shape_scalar_coordinates'] = group['vertices'] * 2
        group['observability_proven'] = False
        output.append(group)
    return output


def run(model_file, request_file, accepted_aux_file, output):
    model_ref, request_ref, aux_ref = map(file_ref, (model_file, request_file, accepted_aux_file))
    model = json.loads(Path(model_file).read_text(encoding='utf-8'))
    request = json.loads(Path(request_file).read_text(encoding='utf-8'))
    aux = json.loads(Path(accepted_aux_file).read_text(encoding='utf-8'))
    selection, selection_ref = read_ref(request['manifest'])
    records = shape_inventory(model)
    arrays = {}
    n = request['raw_rows']
    for key in ('material_index', 'shot_index', 'native_pts'):
        descriptor = request['arrays'][key]
        ref = file_ref(descriptor['file'])
        if ref['sha256'] != descriptor['sha256']:
            raise ValueError('raw array SHA mismatch: ' + key)
        arrays[key] = np.memmap(descriptor['file'], dtype=descriptor['dtype'], mode='r', shape=(n,))
    if 'files' in aux:
        path_ref = dict(aux['files']['path.u32'], file=str(Path(accepted_aux_file).parent / 'path.u32'))
    else:
        path_ref = aux['arrays']['path'] if 'arrays' in aux else aux['path']
    path_file = path_ref.get('file', path_ref.get('path'))
    if not Path(path_file).is_absolute():
        path_file = str(Path(accepted_aux_file).parent / path_file)
    if file_ref(path_file)['sha256'] != path_ref['sha256']:
        raise ValueError('accepted actual path SHA mismatch')
    actual_path = np.memmap(path_file, dtype='<u4', mode='r', shape=(n,))
    fit_ref = selection['representative_selection']['fit_rows'] if 'representative_selection' in selection else selection['fit_rows']
    fit_file = fit_ref.get('file', fit_ref.get('path'))
    if file_ref(fit_file)['sha256'] != fit_ref['sha256']:
        raise ValueError('actual fit selection SHA mismatch')
    fit = np.memmap(fit_file, dtype='<u8', mode='r')
    for mi, pi in zip(arrays['material_index'], actual_path):
        record = records.get((int(mi), int(pi)))
        if record:
            record['raw_rows'] += 1
    for row in fit:
        i = int(row)
        record = records.get((int(arrays['material_index'][i]), int(actual_path[i])))
        if record:
            shot = request['shot_ids'][int(arrays['shot_index'][i])]
            record['fit_rows'] += 1
            record['fit_shots'][shot] = record['fit_shots'].get(shot, 0) + 1
            record['fit_frames'].add((shot, int(arrays['native_pts'][i])))
    alias_proof, alias_ref = read_ref(model['same_print_pairs_complete'])
    source_links=[]
    source_link_refs=[]
    if model.get('SPORTOTO_literal_repair'):
        proof,proof_ref=read_ref(model['SPORTOTO_literal_repair'])
        if proof['schema']!='rinkfit.sportoto_actual_native_shape.v1':raise ValueError('SPORTOTO source identity proof schema')
        annotation,annotation_ref=read_ref(dict(file=proof['source'],sha256=proof['source_sha256']))
        # The archived producer binds both physical installations to these exact
        # source feature IDs. Verify its changed-label inventory and ordered raw
        # contour, rather than matching current model proximity or a glyph name.
        changed={x['label']:x for x in proof['changes']}
        labels={m['label']:i for i,m in enumerate(model['materials'])}
        for feature in annotation['features']:
            label=feature['id']
            if not label.startswith('Ad10.print.'):continue
            other='Ad23.print.'+label[len('Ad10.print.'):]
            if label not in changed or other not in changed or label not in labels or other not in labels:continue
            if len(feature['segments'])!=1:raise ValueError('SPORTOTO source has unrecorded disconnected boundary')
            segment=feature['segments'][0]
            count=len(segment.get('source_native_readout_uv',segment['uv']))
            if changed[label]['source_vertices']!=count or changed[other]['source_vertices']!=count:raise ValueError('SPORTOTO source ordered vertex inventory differs')
            source_links.append(([labels[label],0],[labels[other],0],dict(
                kind='same_exact_ordered_native_source_contour_in_two_confirmed_print_installations',
                source_feature_id=label,source_annotation=annotation_ref,
                source_binding_proof=proof_ref,source_native_pts=proof['source_pts'],
                numerical_world_seed_is_not_physical_truth=True)))
        source_link_refs.extend((proof_ref,annotation_ref))
    links, unresolved = existing_links(model, records, alias_proof, source_links)
    groups = summarize(records, links)
    for rec in records.values():
        rec['fit_frames'] = len(rec['fit_frames'])
        rec.pop('points')
    report = dict(schema='rinkfit.shared_print_shape_mapping.v1', model=model_ref,
        request=request_ref, accepted_aux=aux_ref, selection=selection_ref,
        alias_proof=alias_ref, source_link_dependencies=source_link_refs, raw_rows=n, fit_rows=len(fit),
        shape_records=list(records.values()), exact_existing_links=links,
        declared_links_with_different_current_seed=unresolved, shared_shapes=groups,
        summary=dict(raw_consumed_paths=sum(r['raw_rows']>0 for r in records.values()),
            fit_consumed_paths=sum(r['fit_rows']>0 for r in records.values()),
            fit_print_rows=sum(r['fit_rows'] for r in records.values()),
            fit_shared_shape_groups=sum(g['fit_rows']>0 for g in groups),
            multi_shot_fit_shape_groups=sum(g['fit_rows']>0 and len(g['fit_shots'])>1 for g in groups),
            shared_shape_scalar_coordinates=sum(g['shared_shape_scalar_coordinates'] for g in groups if g['fit_rows']),
            misleading_closed_flags=sum(r['misleading_closed_flag'] for r in records.values() if r['raw_rows'])),
        source_observation_arrays_changed=False, model_modified=False,
        numerical_shape_authority_claimed=False, rank_or_image_acceptance_claimed=False)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report['summary']


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'request', 'accepted-aux', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.request, args.accepted_aux, args.output)))


if __name__ == '__main__':
    main()
