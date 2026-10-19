"""Enumerate existing source identity annotations without importing old UV as facts."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np


def _reference(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _shot(shots, pts):
    rows = [r for r in shots if int(r['start_pts']) <= pts < int(r['end_pts'])]
    if len(rows) != 1:raise ValueError('Original source identity has no unique shot')
    return rows[0]['id']


def _curve_id(row, names):
    feature = row.get('feature_id', row.get('world_scene_binding', {}).get('feature'))
    curve = row.get('curve', {})
    if curve.get('kind') == 'house_ring':
        feature = curve['side']+str(curve['row'])
        key = feature+':'+curve.get('ring', 'outer')
    elif curve.get('kind') == 'hockey_circle':
        key = curve.get('feature_id', feature)+':outer'
    elif curve.get('kind') == 'current_goal_crease':
        key = feature+':boundary'
    elif feature and feature.startswith('lane'):
        key = feature+':edge'+str(curve.get('segment_index', 0))
    elif feature:
        key = feature+':segment'+str(curve.get('segment_index', 0))
    else:return None
    return key if key in names else None


def make_source_identity_seed_registry(inventory, *, source_sha256, time_base, source_shots,
        original_absolute_inventory=None, native_ice_packets=(), user_strokes=(),
        additional_confirmed_seeds=()):
    """Return all known per-source identity hints and earliest curve/shot anchors.

    Named source annotation identity is preserved. Its old pixel coordinates
    are search/identity hints only and must be remeasured in actual current RGB.
    Explicitly unresolved physical IDs do not become confirmed seeds.
    """
    names = {row['curve_id'] for row in inventory}
    records, navigation_only_hints = [], []
    def confirmation_problem(row, authority):
        if authority.get('kind','').startswith('direct_user'):return None
        status=str(row.get('physical_identity_status',row.get('identity_status',''))).lower()
        if any(word in status for word in ('unresolved','unknown','unverified','conditional')):
            return 'explicitly_conditional_or_unconfirmed_source_identity'
        if row.get('source_identity_review_required') is True:
            return 'source_identity_review_explicitly_required'
        if row.get('physical_identity_admission'):
            reference=row['physical_identity_admission']
            if _reference(reference['path'])['sha256']!=reference['sha256']:
                raise ValueError('Existing physical source identity admission changed')
            return None
        name=str(row.get('id','')).lower()
        if name.startswith('user_') or row.get('source_anchor_admission'):
            return None
        if any(text in status for text in ('source-confirmed','held-frame source identity','existing cut4 visible house identity')):
            return None
        return 'old_named_candidate_is_navigation_only_until_actual_identity_confirmation'
    def add(row, uv, authority, native_pts=None):
        key = _curve_id(row, names)
        if key is None:return
        pts = int(row.get('actual_pts', row.get('pts')))
        points = np.asarray(uv, float).reshape(-1, 2)
        if len(points) < 2 or not np.isfinite(points).all():return
        seed=dict(curve_id=key, physical_id=key.split(':')[0], actual_pts=pts,
            native_pts=pts if native_pts is None else int(native_pts), source_sha256=source_sha256,
            time_base=str(Fraction(time_base)), shot_id=_shot(source_shots, pts),
            navigation_uv=points.tolist(), source_original_split=row.get('split', 'fit'),
            source_material_class='static_painted_ice', surface_and_physical_identity_confirmed=True,
            authority={**authority, 'kind':authority['kind'] if authority.get('kind','').startswith('direct_user')
                else 'existing_named_original_source_identity_annotation'},
            source_annotation_id=row.get('id'),
            decoded_native_RGB_sha256=None, bind_current_original_RGB_at_exact_seed_pts=True,
            is_exact_centerline_pixel_ground_truth=False,
            previous_pixel_accuracy_not_assumed=True)
        problem=confirmation_problem(row,authority)
        if problem:
            seed.update(surface_and_physical_identity_confirmed=False,navigation_only=True,
                source_identity_confirmation_missing=problem)
            navigation_only_hints.append(seed)
        else:records.append(seed)
    if original_absolute_inventory is not None:
        ref = _reference(original_absolute_inventory)
        with np.load(original_absolute_inventory, allow_pickle=False) as data:
            rows = json.loads(str(data['rows_json']))
            for ri, row in enumerate(rows):
                if _curve_id(row, names) is None:continue
                indices = np.flatnonzero(data['row_index'] == ri)
                add(row, data['uv'][indices], {**ref, 'original_row_index':ri,
                    'original_point_indices':data['original_point_index'][indices].tolist()}, row.get('source_native_pts'))
    for packet_path in native_ice_packets:
        ref = _reference(packet_path)
        packet = json.loads(Path(packet_path).read_text(encoding='utf-8-sig'))
        if packet['source_video_sha256'] != source_sha256 or str(Fraction(packet['time_base'])) != str(Fraction(time_base)):
            raise ValueError('Native source identity packet belongs to another original video clock')
        for ri, row in enumerate(packet['rows']):
            add(row, row['uv'], {**ref, 'packet_row_index':ri,
                'physical_identity_admission':row.get('physical_identity_admission')}, row.get('pts'))
    for stroke_path in user_strokes:
        ref = _reference(stroke_path)
        packet = json.loads(Path(stroke_path).read_text(encoding='utf-8-sig'))
        row = dict(feature_id=packet['feature_id'], actual_pts=int(packet['actual_exposure_pts']),
                   id='registered_user_stroke:'+str(packet['native_pts']), split='development_source')
        add(row, packet['native_source_uv'], {**ref, 'kind':'direct_user_registered_original_source_stroke',
            'statement':packet.get('user_statement')}, packet['native_pts'])
    records.extend(additional_confirmed_seeds)
    # Prefer a user identity trace over another same-exposure navigation hint.
    unique = {}
    for row in records:
        key = (row['shot_id'], row['curve_id'], row['actual_pts'])
        if key not in unique or row['authority']['kind'].startswith('direct_user'):
            unique[key] = row
    rows = sorted(unique.values(), key=lambda r: (r['actual_pts'], r['curve_id']))
    earliest = {}
    for row in rows:earliest.setdefault((row['shot_id'], row['curve_id']), row)
    return dict(schema='rinkspace.original-source-ice-identity-seed-registry.v1',
        source_sha256=source_sha256, time_base=str(Fraction(time_base)), records=rows,
        navigation_only_seed_hints=navigation_only_hints,
        earliest_curve_shot_anchors=[dict(shot_id=s, curve_id=c, actual_pts=r['actual_pts'],
            authority=r['authority']) for (s,c),r in earliest.items()],
        old_UV_used_as_exact_measurements=False, source_seed_positions_require_current_RGB_reread=True)


def bind_source_seed_to_current_RGB(seed, measurement):
    """Bind existing annotation identity to the exact stream image, never new PTS."""
    binding = measurement['source_binding']
    for field in ('source_sha256', 'time_base', 'shot_id'):
        if seed[field] != binding[field]:raise ValueError('Source identity seed frame differs: '+field)
    if int(seed['actual_pts']) != int(binding['source_pts']):
        raise ValueError('Source annotation cannot seed a different exposure')
    if seed.get('decoded_native_RGB_sha256') not in (None, measurement['decoded_native_RGB_sha256']):
        raise ValueError('Source identity seed original RGB changed')
    return {**seed, 'decoded_native_RGB_sha256':measurement['decoded_native_RGB_sha256']}
