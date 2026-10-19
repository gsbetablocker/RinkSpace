"""Low-only source identity admission, independent of camera residuals.

Keep proposal pixels and their history for review. A development-fit split is
not physical ice-feature confirmation. This module never fits or moves a
camera, edits the shared scene, or changes another installation's admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

from z190_continuous_source_bundle import HARD_SCENE_SHA, SOURCE_SHA
from z190_source_camera_diagnostic import jsonable

SCHEMA = 'rinkspace.low-source-identity-confirmations.v1'
PENDING = 'unfitted_identity_pending'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_hash(row):
    return hashlib.sha256(np.asarray(row['uv'], np.float64).tobytes()).hexdigest()


def source_key(row):
    prior = row.get('source_authority_admission', {})
    name = prior.get('input_source_row_id', row['id'])
    return (int(row.get('source_native_pts', row['pts'])), name, pixel_hash(row))


def is_ice_feature(row, scene):
    name = source_key(row)[1]
    features = {f['id'] for f in scene['ice_features']}
    return name in features or any(name.startswith(f+'_') for f in features)


def load_confirmations(path):
    if path is None:
        return {}, None
    packet = json.loads(Path(path).read_text(encoding='utf8'))
    if (packet.get('schema') != SCHEMA or packet.get('installation_id') != 2 or
            packet.get('scene_sha256') != HARD_SCENE_SHA or packet.get('source_sha256') != SOURCE_SHA):
        raise ValueError('Identity confirmations are not scoped to this low installation/source/scene')
    result = {}
    for entry in packet['confirmations']:
        key = (int(entry['source_native_pts']), entry['input_source_row_id'], entry['source_uv_float64_sha256'])
        if key in result:
            raise ValueError('Duplicate source identity confirmation')
        evidence = entry['human_evidence']
        if evidence.get('kind') != 'direct_user_confirmation' or not str(evidence.get('exact_text', '')).strip():
            raise ValueError('Ice identity requires retained direct human confirmation text')
        record = Path(evidence['record_path'])
        if not record.is_absolute() or digest(record) != evidence['record_sha256']:
            raise ValueError('Direct human confirmation record absent or changed')
        identity = entry['confirmed_identity']
        if identity.get('kind') != 'circle' or identity.get('ring_role') not in ('outer','middle','inner','center'):
            raise ValueError('This low confirmation adapter currently supports explicit circle/ring identity only')
        result[key] = copy.deepcopy(entry)
    return result, dict(path=str(Path(path).resolve()), sha256=digest(path))


def confirmed_circle(row, scene, identity):
    feature = next((f for f in scene['ice_features'] if f['id'] == identity['feature']), None)
    if feature is None or feature['kind'] != 'circle':
        raise ValueError('User-confirmed circle is absent from the unchanged hard scene')
    role = identity['ring_role']
    ring = next((r for r in feature.get('rings', []) if r['role'] == role), None)
    if ring is None and role == 'outer':
        ring = feature
    if ring is None:
        raise ValueError('User-confirmed ring role is absent from the unchanged hard scene')
    count = len(row['world'])
    if count < 3:
        raise ValueError('A source curve requires a retained finite sampled circle inventory')
    angle = np.linspace(0, 2*np.pi, count)
    center = np.asarray(ring['center_xyz'], float)
    return identity['feature']+'_'+role, center+float(ring['radius'])*np.c_[np.cos(angle), np.sin(angle), np.zeros(count)]


def admit_low_source_rows(rows, scene, *, confirmations_path=None, input_binding=None, raw_source_base=None, timeline=None):
    """Return copied rows plus an admission audit; no error/camera input exists.

    All low ice identities require an exact-row direct human record. Existing
    held splits remain held after confirmation. Pending rows are a distinct
    review inventory, never claimed to be independent validation holdouts.
    """
    confirmations, confirmation_binding = load_confirmations(confirmations_path)
    admitted, decisions, used = [], [], set()
    for source in rows:
        row = copy.deepcopy(source)
        actual = pixel_hash(row)
        if 'source_uv_float64_sha256' in row and row['source_uv_float64_sha256'] != actual:
            raise ValueError('Original measured pixels changed before admission')
        if not is_ice_feature(row, scene):
            admitted.append(row)
            continue
        key = source_key(row)
        previous = row.get('source_authority_admission', {})
        original_split = previous.get('input_source_split', row['split'])
        trace = dict(input_source_row_id=key[1], input_source_split=original_split,
            original_source_split=row.get('original_source_split'), input_origin=row.get('origin'),
            input_authority=row.get('authority'), source_native_pts=key[0], canonical_pts=int(row['pts']),
            source_uv_float64_sha256=actual, input_world_float64_sha256=hashlib.sha256(np.asarray(row['world'], np.float64).tobytes()).hexdigest(),
            input_packet=input_binding, admission_uses_camera_or_residual=False, source_pixels_modified=False)
        if raw_source_base is not None and timeline is not None:
            cut = next((i for i in (3,5,8) if timeline['shots'][i]['start_pts'] <= key[0] < timeline['shots'][i]['end_pts']), None)
            if cut is None:
                raise ValueError('Scoped low source admission received a non-low native PTS')
            rawfile = Path(raw_source_base)/f'shot{cut}_actual_source/source_rows.json'
            rawrows = json.loads(rawfile.read_text(encoding='utf8'))
            raw = next((r for r in rawrows if int(r['pts']) == key[0] and r['id'] == key[1] and pixel_hash(r) == actual), None)
            if raw is None:
                raise ValueError('Retained low ice row differs from its original raw source library')
            trace.update(raw_source_rows_path=str(rawfile.resolve()), raw_source_rows_sha256=digest(rawfile),
                         raw_source_split=raw['split'], raw_source_origin=raw.get('origin'))
        confirmation = confirmations.get(key)
        if confirmation is None:
            row['split'] = PENDING
            trace.update(status=PENDING, reason='physical ice identity has no direct human confirmation',
                         original_fit_split_is_not_identity_authority=True)
        else:
            used.add(key)
            row['id'], row['world'] = confirmed_circle(row, scene, confirmation['confirmed_identity'])
            # Admission may restore a historical development-fit row only
            # after actual identity confirmation. True held rows stay held.
            row['split'] = original_split if original_split != PENDING else 'fit'
            row['world_scene_binding'] = dict(kind='direct_user_confirmed_circle_and_ring',
                feature=confirmation['confirmed_identity']['feature'], ring_role=confirmation['confirmed_identity']['ring_role'],
                scene_sha256=HARD_SCENE_SHA,
                source_uv_changed=False, actual_renderer_same_geometry=True)
            trace.update(status='human_identity_confirmed', confirmation=confirmation,
                         confirmation_manifest=confirmation_binding, identity_rebound_to_actual_hard_scene=True)
        row['source_authority_admission'] = trace
        if pixel_hash(row) != actual:
            raise ValueError('Admission changed original measured pixels')
        admitted.append(row)
        decisions.append(trace)
    unused = set(confirmations)-used
    if unused:
        raise ValueError('Identity confirmations do not exactly match retained low source rows: '+str(sorted(unused)))
    report = dict(schema='rinkspace.low-source-authority-admission.v1', installation_id=2,
        scope='low installation only; no automatic changes to main or secondary source layers',
        source_sha256=SOURCE_SHA, scene_sha256=HARD_SCENE_SHA, confirmation_manifest=confirmation_binding,
        source_rows_preserved=len(admitted), pending_ice_rows=sum(r['split'] == PENDING for r in admitted),
        fit_rows=sum(r['split'] == 'fit' for r in admitted), held_rows=sum(r['split'] not in ('fit', PENDING) for r in admitted),
        decisions=decisions, source_uv_modified=False, input_library_modified=False,
        rows_selected_by_residual=False, camera_fitted=False, geometry_fitted=False)
    return admitted, report


def split_rows(rows):
    return dict(fit_rows=[r for r in rows if r['split'] == 'fit'],
                held_rows=[r for r in rows if r['split'] not in ('fit', PENDING)],
                identity_pending_rows=[r for r in rows if r['split'] == PENDING])


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    if digest(a.scene) != HARD_SCENE_SHA:
        raise ValueError('Hard scene changed')
    scene = json.loads(a.scene.read_text(encoding='utf8'))
    packet = json.loads(a.rows.read_text(encoding='utf8'))
    original = packet['fit_rows']+packet['held_rows']+packet.get('identity_pending_rows', [])
    timeline = json.loads(a.timeline.read_text(encoding='utf8')) if a.timeline else None
    binding = dict(path=str(a.rows.resolve()), sha256=digest(a.rows))
    rows, report = admit_low_source_rows(original, scene, confirmations_path=a.source_confirmations,
        input_binding=binding, raw_source_base=a.raw_source_base, timeline=timeline)
    # Byte-identical copy preserves old selected-layer ordering and every
    # historical array/field; no historical file is changed or relocated.
    (a.output/'original_source_rows_exact_input.json').write_bytes(a.rows.read_bytes())
    (a.output/'source_authority_rows.json').write_text(json.dumps(jsonable(split_rows(rows)), ensure_ascii=False, indent=2), encoding='utf8')
    (a.output/'source_authority_admission.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding='utf8')
    template = dict(schema=SCHEMA, installation_id=2, scene_sha256=HARD_SCENE_SHA, source_sha256=SOURCE_SHA,
        confirmations=[], unresolved_rows=[dict(source_native_pts=x['source_native_pts'], input_source_row_id=x['input_source_row_id'],
            source_uv_float64_sha256=x['source_uv_float64_sha256']) for x in report['decisions'] if x['status'] == PENDING],
        note='Empty confirmations; unresolved proposals are not actual human records. Populate only after retaining the real user reply and its SHA.')
    (a.output/'pending_confirmation_template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='low_source_admission_prepared_only', pending_ice_rows=report['pending_ice_rows'],
                         fit_rows=report['fit_rows'], held_rows=report['held_rows'], output=str(a.output))), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene','rows','output'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('source-confirmations','raw-source-base','timeline'):
        p.add_argument('--'+key, type=Path)
    run(p.parse_args())
