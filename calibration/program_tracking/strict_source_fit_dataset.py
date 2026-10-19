"""Immutable finite-support FIT view over disjoint reviewed source packs.

Only index arrays are stored. Original pixels, measured sigma, proof tables,
typed operators and every HELD observation stay in the existing source packs.
This observation interface does not install the full solver graph adapter.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .compact_source_header_inventory import read_minimal_compact_source_headers
from .strict_absolute_source import StrictAbsoluteSourcePack
from .strict_model_scope import authenticate_source_reference, file_sha
from .strict_source_named_identity_filter_with_exclusions import _slice_current_row
from .strict_source_representatives import _declared_paths


SCHEMA = 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1'
POINT_FIELDS = ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
    'source_side_quantiles_uv', 'original_point_indices')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def partition_support(headers, offsets, sigma):
    """All finite-positive FIT, all HELD, and missing-support FIT partition UV."""
    offsets, sigma = np.asarray(offsets), np.asarray(sigma)
    if (offsets.shape != (len(headers)+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(sigma)
            or np.any(np.diff(offsets) < 0) or sigma.ndim != 1
            or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)):
        raise ValueError('Exact row offsets and positive-or-missing original sigma required')
    fit = np.zeros(len(sigma), bool)
    held = np.zeros(len(sigma), bool)
    for i, row in enumerate(headers):
        lo, hi = map(int, offsets[i:i+2])
        if row['point_count'] != hi-lo or row['split'] not in ('fit', 'held'):
            raise ValueError('Original ordered row and FIT/HELD role required')
        (fit if row['split'] == 'fit' else held)[lo:hi] = True
    finite = np.isfinite(sigma) & (sigma > 0)
    result = dict(fit_point_indices=np.flatnonzero(fit & finite),
        held_point_indices=np.flatnonzero(held),
        pending_support_point_indices=np.flatnonzero(fit & ~finite))
    if not np.array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(len(sigma))):
        raise ValueError('Finite FIT / complete HELD / missing support must partition original points')
    return result


def finite_source_row(row, local_indices, source_reference, source_row_index, *, output_split=None):
    """Select measured stations and split declared paths across missing ones."""
    local = np.asarray(local_indices, np.int64)
    n = len(row['uv'])
    if (local.ndim != 1 or not len(local) or np.any(local < 0)
            or np.any(local >= n) or np.any(np.diff(local) <= 0)):
        raise ValueError('Exact ordered nonempty source stations required')
    if row['split'] == 'held' and not np.array_equal(local, np.arange(n)):
        raise ValueError('HELD observations cannot be discarded or changed to FIT')
    output_split = row['split'] if output_split is None else output_split
    if output_split not in ('fit', 'held') or row['split'] == 'held' and output_split != 'held':
        raise ValueError('HELD observations cannot change to FIT')
    if output_split == 'fit' and not np.isfinite(np.asarray(row['sigma_px'])[local]).all():
        raise ValueError('Missing source support cannot be fitted')
    paths, basis = _declared_paths(row)
    flat = [i for path in paths for i in path]
    if sorted(flat) != list(range(n)):
        raise ValueError('Declared source paths must partition original stations')
    back = {int(old):new for new,old in enumerate(local)}
    retained_paths = []
    for path in paths:
        run = []
        for station in path:
            if station in back:
                run.append(back[station])
            elif run:
                retained_paths.append(run)
                run = []
        if run:
            retained_paths.append(run)
    result = _slice_current_row(row, local)
    for name in ('normal_interval', 'point_support_bounds_xy_px', 'source_tangent_interval_px'):
        if name in row:
            value = np.asarray(row[name])
            if value.ndim and len(value) == n:
                result[name] = value[local].copy()
    result.update(point_count=len(local), split=output_split, source_visible_paths=retained_paths,
        source_dataset_original_path_basis=basis,
        source_dataset_locator=dict(source_pack=dict(source_reference),
            source_row_index=int(source_row_index), local_point_indices=local.tolist()))
    result['working_sigma_status'] = ('explicit_source_working_sigma'
        if np.isfinite(result['sigma_px']).all()
        else 'missing_or_partial_source_working_sigma_not_fit_ready')
    return result


def build_dataset(cut_references, output_dir, *, model_reference, coverage_reference,
        authorities=()):
    """Index exact merged cuts without storing another copy of their pixels."""
    output = Path(output_dir).resolve()
    manifest_path = output/'fit_dataset.json'
    if manifest_path.exists():
        raise FileExistsError('Keep earlier frozen source dataset unchanged')
    output.mkdir(parents=True, exist_ok=True)
    for ref in (model_reference, coverage_reference, *authorities):
        authenticate_source_reference(ref)
    records = []
    source = expected = None
    processed = set()
    counts = Counter()
    exposure_fit = set()
    by_material = Counter()
    for cut_index, ref in enumerate(cut_references):
        authenticate_source_reference(ref)
        headers, offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'], allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata_json'].item()))
            sigma = saved['sigma_px']
            actual = saved['source_frame_actual_pts']
            native = saved['source_frame_native_pts']
            hashes = saved['source_frame_BGR_sha256']
        if source is None:
            source, expected = metadata['source'], metadata['expected_actual_pts']
        if source != metadata['source'] or expected != metadata['expected_actual_pts']:
            raise ValueError('Dataset cuts must share exact original video and programme clock')
        current = set(metadata['processed_actual_pts'])
        if processed.intersection(current):
            raise ValueError('Dataset cuts must be merged before indexing overlapping exposures')
        processed.update(current)
        bound = {int(p) for p,n,h in zip(actual,native,hashes) if n >= 0 and h}
        if bound != current:
            raise ValueError('Every processed cut exposure must retain its own native RGB binding')
        partition = partition_support(headers, offsets, sigma)
        for name, value in partition.items():
            counts[name] += len(value)
        index_path = output/f'cut_{cut_index:02d}_support_indices.npz'
        if index_path.exists():
            raise FileExistsError('Preserve earlier dataset index')
        np.savez_compressed(index_path, row_offsets=offsets, **partition)
        finite = np.isfinite(sigma) & (sigma > 0)
        for ri, header in enumerate(headers):
            lo, hi = map(int, offsets[ri:ri+2])
            if header['scope'] not in ('ice', 'board') or header['actual_pts'] not in current:
                raise ValueError('Model-external or different-clock source row in dataset')
            if header['split'] == 'fit':
                amount = int(finite[lo:hi].sum())
                by_material[header['physical_material_id']] += amount
                if amount:
                    exposure_fit.add(header['actual_pts'])
        records.append(dict(source_pack=deepcopy(ref), support_index=reference(index_path),
            source_rows=len(headers), source_points=len(sigma),
            fit_points=len(partition['fit_point_indices']),
            held_points=len(partition['held_point_indices']),
            pending_support_points=len(partition['pending_support_point_indices']),
            processed_actual_exposures=len(current)))
        print(json.dumps(dict(dataset_cut=cut_index, **{k:v for k,v in records[-1].items()
            if k not in ('source_pack','support_index')})), flush=True)
    if processed != set(expected):
        raise ValueError('Dataset must retain complete original exposure clock')
    result = dict(schema=SCHEMA, source=source, expected_actual_pts=expected,
        cuts=records, frozen_model_reference=model_reference,
        original_native_coverage_reference=coverage_reference,
        source_review_authorities=list(authorities),
        fit_points=counts['fit_point_indices'], held_points=counts['held_point_indices'],
        pending_support_points=counts['pending_support_point_indices'],
        actual_exposures=len(processed), actual_exposures_with_finite_fit=len(exposure_fit),
        actual_exposures_without_finite_fit=len(processed-exposure_fit),
        finite_fit_points_by_material=dict(sorted(by_material.items())),
        observation_inputs_have_finite_measured_support=True,
        all_HELD_pixels_preserved=True, missing_support_retained_in_original_packs=True,
        source_UV_sigma_weights_and_operators_changed=False, pixels_or_time_subsampled=False,
        repeated_large_numeric_arrays_stored=False, independent_accuracy_accepted=False,
        full_solver_graph_adapter_installed=False, solver_started=False)
    with manifest_path.open('x', encoding='utf8') as saved:
        json.dump(result, saved, ensure_ascii=False, separators=(',', ':'))
    return reference(manifest_path)


def _named_pending_indices(headers, offsets, source_ref, source, frame_bindings, filters):
    """Bind current exclusion facts to their exact original row/point indices."""
    pending = set()
    for ref in filters:
        authenticate_source_reference(ref)
        fact = json.loads(Path(ref['path']).read_text(encoding='utf-8-sig'))
        if (fact.get('schema') != 'rinkspace.exact-source-named-identity-pending-filter.v1'
                or fact.get('retain_pending_source_readouts_full_literal_evidence') is not True):
            raise ValueError('Literal exact-point pending filter required')
        groups = [g for g in fact['source_pack_groups'] if
            Path(g['source_pack_reference']['path']).resolve() == Path(source_ref['path']).resolve()
            and g['source_pack_reference']['sha256'] == source_ref['sha256']]
        if len(groups) != 1:
            raise ValueError('Pending filter must bind the selected original source bytes')
        group = groups[0]
        for witness in group['source_review_witnesses']:
            authenticate_source_reference(witness)
        count = 0
        for entry in group['entries']:
            ri = entry['pack_row_index']
            if type(ri) is not int or not 0 <= ri < len(headers):
                raise ValueError('Pending row is outside exact original source')
            row = headers[ri]
            binding = entry['source_binding']
            lo,hi = map(int,offsets[ri:ri+2])
            indices = entry['pack_point_indices']
            digest = binding.get('decoded_native_BGR_sha256', binding.get('decoded_BGR_sha256'))
            video_sha = binding.get('source_sha256', binding.get('video_sha256'))
            if (row['split'] != 'fit' or row['physical_material_id'] != entry['physical_material_id']
                    or row['actual_pts'] != binding['actual_pts']
                    or row['native_pts'] != binding['native_pts']
                    or (row['native_pts'],row['decoded_BGR_sha256']) != frame_bindings[row['actual_pts']]
                    or digest != row['decoded_BGR_sha256'] or video_sha != source['video_sha256']
                    or binding['time_base'] != source['time_base']
                    or entry.get('exclude_from_final_named_FIT') is not True
                    or entry.get('status') != 'source_material_measured_named_identity_pending'
                    or not indices or any(type(i) is not int or not lo <= i < hi for i in indices)
                    or indices != sorted(set(indices)) or pending.intersection(indices)):
                raise ValueError('Exact named exclusion differs from original material, role, clock or indices')
            pending.update(indices)
            count += len(indices)
        if count != group['pending_named_point_count']:
            raise ValueError('Exact named exclusion count changed')
    return np.asarray(sorted(pending), np.int64)


def build_logical_dataset(source_records, output_dir, *, model_reference, coverage_reference,
        duplicate_audit_reference, authorities=()):
    """Qualified original bases plus zero-duplicate deltas need no pixel rewrite."""
    for ref in (model_reference,coverage_reference,duplicate_audit_reference,*authorities):
        authenticate_source_reference(ref)
    audit = json.loads(Path(duplicate_audit_reference['path']).read_text(encoding='utf8'))
    wanted = {(str(Path(r['source_pack']['path']).resolve()),r['source_pack']['sha256']) for r in source_records}
    audited = {(str(Path(i['source_reference']['path']).resolve()),i['source_reference']['sha256'])
        for r in audit['records'] for i in r['inputs']}
    if (audit['schema'] != 'rinkspace.source-input-cross-pack-exact-pixel-operator-audit.v1'
            or wanted != audited or len(wanted) != len(source_records)
            or any(v for k,v in audit['counts'].items() if 'duplicate' in k or 'conflict' in k)
            or any(r.get('operator_conflicts') for r in audit['records'])
            or audit['points'] != audit['unique_exact_pixel_target_count']):
        raise ValueError('Logical source view requires this complete zero-duplicate/operator-conflict audit')
    output = Path(output_dir).resolve()
    if (output/'fit_dataset.json').exists():
        raise FileExistsError('Preserve earlier frozen dataset')
    output.mkdir(parents=True,exist_ok=True)
    source = expected = None
    frames = {}
    prepared = []
    held_features,held_materials = set(),set()
    for record in source_records:
        ref = record['source_pack']
        authenticate_source_reference(ref)
        headers,offsets = read_minimal_compact_source_headers(ref['path'])
        with np.load(ref['path'],allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata_json'].item()))
            sigma = z['sigma_px']
            bindings = {int(p):(int(n),bytes(h).decode('ascii')) for p,n,h in zip(
                z['source_frame_actual_pts'],z['source_frame_native_pts'],z['source_frame_BGR_sha256']) if n>=0 and h}
        if source is None:
            source,expected = metadata['source'],metadata['expected_actual_pts']
        if source != metadata['source'] or expected != metadata['expected_actual_pts']:
            raise ValueError('Logical source view must retain one video and original exposure clock')
        if set(bindings) != set(metadata['processed_actual_pts']):
            raise ValueError('Source pack lost actual/native/RGB bindings')
        for pts,binding in bindings.items():
            if pts in frames and frames[pts] != binding:
                raise ValueError('Overlapping source packs changed native RGB bindings')
            frames[pts] = binding
        for h in headers:
            if (h['scope'] not in ('ice','board') or h['actual_pts'] not in bindings
                    or (h['native_pts'],h['decoded_BGR_sha256']) != bindings[h['actual_pts']]):
                raise ValueError('Model-external or unbound row cannot enter source dataset')
            if h['split'] == 'held':
                held_features.add((h['actual_pts'],h['feature_id']))
                held_materials.add((h['actual_pts'],h['physical_material_id']))
        named = _named_pending_indices(headers,offsets,ref,source,bindings,
            record.get('pending_filter_references',[]))
        prepared.append((record,headers,offsets,sigma,named))
    with np.load(coverage_reference['path'],allow_pickle=False) as z:
        ids = np.flatnonzero(z['representative'])
        original_frames = {int(z['actual_pts'][i]):(int(z['native_pts'][i]),
            bytes(z['BGR_sha256'][i]).decode('ascii')) for i in ids}
        native_frames = len(z['native_pts'])
    if set(frames) != set(expected) or frames != original_frames:
        raise ValueError('Logical dataset must preserve every original exposure and RGB binding')
    records=[];counts=Counter();material=Counter();fit_exposures=set()
    for index,(record,headers,offsets,sigma,named) in enumerate(prepared):
        effective = []
        promoted=[]
        for ri,h in enumerate(headers):
            new=dict(h)
            if ((h['actual_pts'],h['feature_id']) in held_features or
                    (h['actual_pts'],h['physical_material_id']) in held_materials):
                if h['split']=='fit':promoted.append(ri)
                new['split']='held'
            effective.append(new)
        partition=partition_support(effective,offsets,sigma)
        for name in tuple(partition):
            partition[name]=np.setdiff1d(partition[name],named,assume_unique=True)
        partition['pending_named_point_indices']=named
        if not np.array_equal(np.sort(np.concatenate(list(partition.values()))),np.arange(len(sigma))):
            raise ValueError('Named pending, missing support, HELD and finite FIT must partition original UV')
        path=output/f'source_{index:02d}_support_indices.npz'
        if path.exists():raise FileExistsError('Preserve earlier dataset index')
        np.savez_compressed(path,row_offsets=offsets,promoted_held_rows=np.asarray(promoted,np.int64),**partition)
        for name,value in partition.items():counts[name]+=len(value)
        selected=partition['fit_point_indices']
        for ri,h in enumerate(effective):
            lo,hi=map(int,offsets[ri:ri+2]);a,b=np.searchsorted(selected,[lo,hi])
            if b>a:
                fit_exposures.add(h['actual_pts']);material[h['physical_material_id']]+=int(b-a)
        records.append(dict(source_pack=record['source_pack'],support_index=reference(path),
            pending_filter_references=record.get('pending_filter_references',[]),
            source_rows=len(headers),source_points=len(sigma),
            fit_points=len(selected),held_points=len(partition['held_point_indices']),
            pending_support_points=len(partition['pending_support_point_indices']),
            pending_named_points=len(named),held_protected_rows_promoted=len(promoted)))
        print(json.dumps(dict(dataset_source=index,fit_points=len(selected),
            pending_named_points=len(named),held_points=records[-1]['held_points'])),flush=True)
    result=dict(schema=SCHEMA,source=source,expected_actual_pts=expected,cuts=records,
        frozen_model_reference=model_reference,original_native_coverage_reference=coverage_reference,
        exact_duplicate_operator_audit_reference=duplicate_audit_reference,
        source_review_authorities=list(authorities),
        fit_points=counts['fit_point_indices'],held_points=counts['held_point_indices'],
        pending_support_points=counts['pending_support_point_indices'],pending_named_points=counts['pending_named_point_indices'],
        actual_exposures=len(frames),native_frames=native_frames,
        actual_exposures_with_finite_fit=len(fit_exposures),actual_exposures_without_finite_fit=len(frames)-len(fit_exposures),
        finite_fit_points_by_material=dict(sorted(material.items())),
        HELD_material_and_feature_exposure_protection_applied=True,
        observation_inputs_have_finite_measured_support=True,all_HELD_pixels_preserved=True,
        missing_and_named_pending_retained_in_original_packs=True,
        source_UV_sigma_weights_and_operators_changed=False,pixels_or_time_subsampled=False,
        repeated_large_numeric_arrays_stored=False,independent_accuracy_accepted=False,
        full_solver_graph_adapter_installed=False,solver_started=False)
    path=output/'fit_dataset.json'
    with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'))
    return reference(path)


def iter_dataset_rows(manifest_reference, *, split='fit'):
    """Public exact-source validation and finite row consumption, one cut at a time."""
    if split not in ('fit', 'held'):
        raise ValueError('Consume FIT and HELD separately')
    authenticate_source_reference(manifest_reference)
    manifest = json.loads(Path(manifest_reference['path']).read_text(encoding='utf8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Frozen finite-support source dataset required')
    for cut in manifest['cuts']:
        authenticate_source_reference(cut['support_index'])
        ref = cut['source_pack']
        pack = StrictAbsoluteSourcePack(ref['path'], expected_sha256=ref['sha256'])
        with np.load(cut['support_index']['path'], allow_pickle=False) as saved:
            selection = saved[split+'_point_indices']
            headers=deepcopy(pack.rows)
            if 'promoted_held_rows' in saved:
                for ri in saved['promoted_held_rows']:
                    if headers[int(ri)]['split']!='fit':raise ValueError('Only FIT rows can acquire HELD protection')
                    headers[int(ri)]['split']='held'
            expected = partition_support(headers, pack.values['row_offsets'], pack.values['sigma_px'])
            if 'pending_named_point_indices' in saved:
                named=saved['pending_named_point_indices']
                bound={p:(b['native_pts'],b['decoded_BGR_sha256']) for p,b in pack.frame_bindings.items()}
                actual_named=_named_pending_indices(pack.rows,pack.values['row_offsets'],ref,
                    pack.metadata['source'],bound,cut.get('pending_filter_references',[]))
                if not np.array_equal(named,actual_named):raise ValueError('Exact named pending selection changed')
                expected={k:np.setdiff1d(v,named,assume_unique=True) for k,v in expected.items()}
            for name, indices in expected.items():
                if not np.array_equal(saved[name], indices):
                    raise ValueError('Dataset index changed original source roles or finite support')
            if not np.array_equal(saved['row_offsets'], pack.values['row_offsets']):
                raise ValueError('Dataset index changed original row boundaries')
        for ri, header in enumerate(pack.rows):
            lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
            a, b = np.searchsorted(selection, [lo,hi])
            local = selection[a:b]-lo
            if not len(local):
                continue
            row = dict(header, **{key:pack.values[key][lo:hi] for key in POINT_FIELDS})
            yield finite_source_row(row, local, ref, ri, output_split=split)
