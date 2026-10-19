"""Bind five authenticated legacy labels to the same current finite print.

This changes neither observations nor their admission.  The former q23
provider recognized named corners but silently treated these source labels
as static world curves.  Each additional identity below is certified against
its immutable, source-only authority before the original finite-material
transport is used.  There is no image-to-world association or geometry snap.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from shared_advertisement_pairs_candidate.shared_advertisement_pairs_numeric_geometry import (
    DEFAULT_PAIRS, SharedAdvertisementPairsNumericGeometry,
)
from shared_advertisement_pairs_candidate.shared_rink_numeric_geometry_lanes_ad_pairs import (
    SharedRinkNumericGeometryLanesAdPairs,
)


VERSION = 'qualified_advertisement_source_binding_v1'
SOURCE_SHA256 = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
ORIGINAL_SOURCE_ROOT = Path(
    'D:/CodexData/.codex/visualizations/2026/10/02/'
    'private-thread-00008/RinkSpace_tmp'
)
AUTHORITIES = {
    'ad7_boundary': (
        ORIGINAL_SOURCE_ROOT/'ice_alignment_review/ad7_source_boundaries_qualified.json',
        '9cf126825314962e65ac9885cc3ef704eeb15c43a8197810d3b07a858a02a2fb',
    ),
    'main1_ad7': (
        ORIGINAL_SOURCE_ROOT/'ice_alignment_review/main1_ice_source/'
        'main1_F3_extent_Ad7_source_packet_qualified.json',
        '758c96a024a8c8d09d5bf92946d5d6bd8cf498ab242bd4023888e55ffe2e4175',
    ),
    'main9_ad2': (
        ORIGINAL_SOURCE_ROOT/'main9_board_ice_key_source_v1/'
        'core_key_visible_board_edges_source.json',
        '8f10051947755b546a7313e11fa5ae1b82f4e90d88f422b67375a3b9634016a4',
    ),
}
ALIASES = {
    'ad7_left_vertical': ('main1_ad7', 7, 'left_vertical_xyz'),
    'ad2_left_source_boundary': ('main9_ad2', 2, 'left_vertical_xyz'),
    'ad7_left_outer_source_blur_interval': ('ad7_boundary', 7, 'left_vertical_xyz'),
    'advertisement7.left_outer_boundary': ('ad7_boundary', 7, 'left_vertical_xyz'),
    'advertisement7.right_outer_boundary': ('ad7_boundary', 7, 'right_vertical_xyz'),
}


@lru_cache(maxsize=3)
def _authority(name):
    path, expected = AUTHORITIES[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Original qualified advertisement authority changed: '+str(path))
    value = json.loads(raw)
    source = value.get('source_sha256', value.get('source_video_sha256'))
    if source != SOURCE_SHA256:
        raise ValueError('Qualified advertisement source video identity changed')
    return value


def authority_contract():
    for name in AUTHORITIES:
        _authority(name)
    return {name: {'path': str(path), 'sha256': digest}
            for name, (path, digest) in AUTHORITIES.items()}


def _same_array(row, key, expected):
    if key not in row or not np.array_equal(np.asarray(row[key], float), np.asarray(expected, float)):
        raise ValueError('Qualified source '+key+' changed for '+str(row.get('id')))


def _same_origin(row, name, *, review=False):
    path, digest = AUTHORITIES[name]
    if review:
        value = row.get('review_source_packet', {})
        if value.get('sha256') != digest:
            raise ValueError('Qualified report source authority hash changed')
        original = value.get('path')
    else:
        original = row.get('origin')
    if original is None or Path(original).resolve() != path.resolve():
        raise ValueError('Qualified advertisement source authority path changed')


def _certificate_key(row):
    """Content key, never object identity: mutation cannot reuse a certificate."""
    h = hashlib.sha256()
    scalar_keys = ('id', 'pts', 'source_native_pts', 'split', 'source_role',
                   'original_source_role', 'point_correspondence', 'physical_ids',
                   'origin', 'review_source_packet', 'review_only_never_admitted',
                   'diagnostic_unit_sigma_not_fit_weight', 'source_bgr_sha256')
    for key in scalar_keys:
        h.update(key.encode('utf8'))
        h.update(json.dumps(row.get(key), sort_keys=True, ensure_ascii=False,
                            default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)).encode('utf8'))
    for key in ('world', 'uv', 'normal', 'normal_interval', 'sigma'):
        a = np.asarray(row[key], float)
        h.update(key.encode('ascii')); h.update(str(a.shape).encode('ascii'))
        h.update(a.tobytes(order='C'))
    return h.digest()


def certify_legacy_alias(row):
    """Return the literal physical identity after original source checks.

    No new row is admitted.  Unrecognized labels return None; recognized but
    altered legacy sources fail explicitly rather than becoming fixed-world.
    """
    name = str(row.get('id', ''))
    declared = ALIASES.get(name)
    if declared is None:
        return None
    authority_name, number, role = declared
    packet = _authority(authority_name)
    pts = int(row.get('source_native_pts', row.get('pts', -1)))
    if int(row.get('pts', -1)) != pts or row.get('point_correspondence') is not False:
        raise ValueError('Qualified finite source clock or correspondence changed')

    if authority_name != 'ad7_boundary':
        if row.get('physical_ids') is not None:
            raise ValueError('Declared legacy source identity unexpectedly changed')
        matches = [v for v in packet['groups'] if v['id'] == name and int(v['pts']) == pts]
        if len(matches) != 1:
            raise ValueError('Unqualified exposure for known legacy advertisement label')
        original = matches[0]
        _same_origin(row, authority_name)
        if row.get('split') != original['split']:
            raise ValueError('Qualified advertisement FIT/HELD role changed')
        _same_array(row, 'world', original['world'])
        _same_array(row, 'uv', original['uv'])
        _same_array(row, 'normal', original.get('normal', original.get('source_normal_uv')))
        _same_array(row, 'normal_interval', original.get('normal_interval', original.get('normal_interval_px')))
        sigma = original.get('sigma')
        if sigma is None:
            sigma = np.full(len(original['uv']), float(original['sigma_px']))
        _same_array(row, 'sigma', sigma)
        return [(number, role)]

    physical = ('advertisement7.left_outer_boundary' if role == 'left_vertical_xyz'
                else 'advertisement7.right_outer_boundary')
    matches = [v for v in packet['records']
               if v['physical_id'] == physical and int(v['source_pts']) == pts]
    if len(matches) != 1:
        raise ValueError('Unqualified Ad7 boundary source exposure')
    original = matches[0]
    points = original['rows']
    _same_array(row, 'uv', [v['source_uv'] for v in points])
    _same_array(row, 'normal', [v['source_normal_uv'] for v in points])
    _same_array(row, 'normal_interval', [v['normal_interval_px'] for v in points])

    if name == 'ad7_left_outer_source_blur_interval':
        if (list(row.get('physical_ids', [])) != [physical] or
            row.get('split') != 'fit' or row.get('original_source_role') != 'development' or
            original['source_role'] != 'development' or
            row.get('source_bgr_sha256') != original['source_bgr_sha256']):
            raise ValueError('Original admitted Ad7 blur source identity/role changed')
        _same_origin(row, authority_name)
        # Original engineering 6.19px and sqrt(9) correlated-scan budget.
        _same_array(row, 'sigma', np.full(9, 18.57))
        knot_count = 33
    else:
        if (row.get('physical_ids') is not None or row.get('split') != 'held' or
            row.get('source_role') != 'held' or original['source_role'] != 'held' or
            row.get('review_only_never_admitted') is not True or
            row.get('diagnostic_unit_sigma_not_fit_weight') is not True):
            raise ValueError('Original report-only Ad7 role changed')
        _same_origin(row, authority_name, review=True)
        _same_array(row, 'sigma', np.ones(len(points)))
        knot_count = 121
    # The exact original generator, including order and rounding, is retained.
    ends = np.asarray(original['world_endpoints'], float)
    t = np.linspace(0., 1., knot_count)[:, None]
    _same_array(row, 'world', (1.-t)*ends[0]+t*ends[1])
    return [(number, role)]


class CorrectedSharedAdvertisementPairsNumericGeometry(SharedAdvertisementPairsNumericGeometry):
    """The original finite transport with five certified spelling aliases."""
    def __init__(self, base, scene_unit_m=None, *, pairs=DEFAULT_PAIRS):
        super().__init__(base, scene_unit_m, pairs=pairs)
        authority_contract()
        self._legacy_source_certificate_cache = {}

    def identities(self, row):
        if str(row.get('id', '')) not in ALIASES:
            return super().identities(row)
        key = _certificate_key(row)
        value = self._legacy_source_certificate_cache.get(key)
        if value is None:
            value = certify_legacy_alias(row)
            if len(self._legacy_source_certificate_cache) >= 128:
                self._legacy_source_certificate_cache.clear()
            self._legacy_source_certificate_cache[key] = value
        return value


class CorrectedSharedRinkNumericGeometryLanesAdPairs(SharedRinkNumericGeometryLanesAdPairs):
    """Same 23 world parameters, with current finite targets for legacy rows."""
    def __init__(self, base, scene_unit_m=None, *, pairs=DEFAULT_PAIRS):
        super().__init__(base, scene_unit_m, pairs=pairs)
        self.pair_panels = CorrectedSharedAdvertisementPairsNumericGeometry(
            self.base, self.scene_unit_m, pairs=pairs)
        # Constructor values are unchanged, including initial original q11.
        self.initial = np.r_[self.original.initial, self.pair_panels.initial]


def _count_aliases(geometry, groups):
    result = {}
    for block, pts, row in groups:
        if str(row.get('id', '')) not in ALIASES:
            continue
        ids = geometry.pair_panels.identities(row)
        key = str(row['id'])
        value = result.setdefault(key, {'records': 0, 'UV': 0, 'instances': []})
        value['records'] += 1; value['UV'] += len(row['uv'])
        value['instances'].append({'block': block, 'actual_pts': int(pts),
                                   'source_native_pts': int(row.get('source_native_pts', row['pts'])),
                                   'source_split': row['split'], 'physical_identities': ids})
    return result


def install_corrected_advertisement_source_geometry(state):
    """Install on a fresh paired-q23 state, before factors or source actors.

    All six blocks and their existing owner references use one provider.
    No row object, parameter index, q, profile array, sigma, prior or source
    admission is replaced.  The current physical camera chart is unaffected.
    """
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Install corrected geometry before owned source actors')
    if hasattr(state, 'corrected_advertisement_source_binding_contract'):
        if not isinstance(state.geometry, CorrectedSharedRinkNumericGeometryLanesAdPairs):
            raise ValueError('Corrected source contract/provider ownership mismatch')
        if any(b.geometry is not state.geometry for b in state.blocks):
            raise ValueError('Source blocks no longer share the corrected provider')
        return state.corrected_advertisement_source_binding_contract
    old = state.geometry
    if (type(old) is not SharedRinkNumericGeometryLanesAdPairs or old.parameter_count != 23 or
        old.pair_panels.pairs != DEFAULT_PAIRS or len(state.blocks) != 6 or
        not hasattr(state, 'paired_ad_contract')):
        raise ValueError('Original complete paired-q23 graph required; no course chart truncation')
    expected = np.r_[np.array([0, 1, 2, 3, 4, 34189, 34190, 34191, 34192, 34193, 34194]),
                     np.arange(34213, 34225)]
    if not np.array_equal(state.geometry_parameter_indices, expected):
        raise ValueError('Original unchanged geometry parameter indices required')
    geometry = CorrectedSharedRinkNumericGeometryLanesAdPairs(old.base, old.scene_unit_m)
    geometry.original = old.original; geometry.rink = old.rink
    geometry.panel = old.panel; geometry.base = old.base
    fit_groups = [(b.name, p, r) for b in state.blocks for p, rows in b.groups for r in rows]
    held_groups = [(name, p, r) for name, groups in state.held.items()
                   for p, rows in groups for r in rows]
    fit = _count_aliases(geometry, fit_groups); held = _count_aliases(geometry, held_groups)
    totals = lambda v: (sum(x['records'] for x in v.values()), sum(x['UV'] for x in v.values()))
    if totals(fit) != (7, 51) or totals(held) != (18, 156):
        raise ValueError('Qualified source alias inventory changed: '+str((totals(fit), totals(held))))
    # Validate every corrected finite source through the original primitive.
    # At zero pair displacement even the sampled world is exactly unchanged.
    for _, _, row in fit_groups+held_groups:
        if str(row.get('id', '')) in ALIASES:
            if not np.array_equal(geometry.rebind(row, geometry.initial)['world'],
                                  old.rebind(row, old.initial)['world']):
                raise ValueError('Corrected alias changed initial finite source domain')

    replacement = {}
    for block in state.blocks:
        if block.geometry is not old:
            raise ValueError('All original source blocks must share one world provider')
        new = type(block).__new__(type(block)); new.__dict__.update(vars(block))
        new.geometry = geometry
        new.bundle = copy.copy(block.bundle)
        if getattr(block, 'prepared', None) is not None:
            new.prepared = copy.copy(block.prepared); new.prepared.bundle = new.bundle
        # Replace existing world-owner aliases; unrelated q9 pattern geometry
        # and source-only original scene are deliberately different owners.
        for owner in (new, new.bundle, getattr(new, 'prepared', None)):
            if owner is None:
                continue
            for name, value in list(vars(owner).items()):
                if value is old:
                    setattr(owner, name, geometry)
        replacement[id(block)] = new
    blocks = [replacement[id(b)] for b in state.blocks]
    bypts = {p: replacement[id(b)] for p, b in state.bypts.items()}
    state.blocks = blocks
    state.bypts = bypts
    state.geometry = geometry
    contract = {
        'schema': 'rinkspace.qualified-advertisement-source-alias-correction.v1',
        'version': VERSION, 'source_authorities': authority_contract(),
        'alias_physical_material_identities': {k: {'authority': v[0], 'number': v[1], 'role': v[2]}
                                             for k, v in ALIASES.items()},
        'affected_admitted': fit, 'affected_HELD': held,
        'affected_admitted_records': 7, 'affected_admitted_UV': 51,
        'affected_HELD_records': 18, 'affected_HELD_UV': 156,
        'physical_parameter_count': 34225, 'world_parameter_count': 23,
        'geometry_parameter_indices': expected.tolist(),
        'parameter_indices_camera_functions_and_profiles_unchanged': True,
        'source_UV_support_sigma_roles_counts_order_and_budget_unchanged': True,
        'original_finite_material_samples_domain_and_order_preserved': True,
        'all_six_blocks_share_corrected_provider_before_actors': True,
        'initial_pair_world_and_residual_equal_to_legacy_target': True,
        'previous_missing_panel_J_columns_are_corrected_not_equal': True,
        'old_full_H_g_equivalence_claim': False,
        'must_pass_fresh_complete_source_gate_before_solver': True,
        'accuracy_accepted': False, 'production_modified': False,
    }
    state.corrected_advertisement_source_binding_contract = contract
    state.paired_ad_contract = dict(state.paired_ad_contract, corrected_source_bindings=contract)
    if hasattr(state, 'optics_contract'):
        state.optics_contract = dict(state.optics_contract, paired_ad_contract=state.paired_ad_contract)
    return contract
