"""Exact-source correction of the independently reviewed Ad5 scope label.

The historical corner reader labelled every corner envelope external_border.
This binding applies only to its authenticated Ad5 MEIWA colour-bar rows in
one frozen v6 pack. UV, sigma, source paths, roles and original targets survive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_model_scope import authenticate_source_reference

_ROOT = Path(__file__).resolve().parents[2]
_BINDING_PATH = _ROOT / 'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/strict_solver_adapter_20261013/Ad5_meiwa_color_bar_scope_binding_v2.json'
_BINDING_SHA256 = '4acc2ab0061be5375e791c82d22a48807ed21726cd3e1bc11089187a20725ad1'


def _same_reference(first, second):
    return (first.get('sha256') == second.get('sha256')
        and Path(first.get('path', '')).resolve() == Path(second.get('path', '')).resolve())


def _qualify_group(group, headers, binding):
    target = group['model_target']
    if not (target.get('kind') == 'advertisement_boundary'
            and target.get('advertisement_number') == 5 and target.get('extent_scope') == 'external_border'):
        return False
    edge = target['edge']
    if edge not in ('top_edge','bottom_edge','left_vertical','right_vertical'):
        raise ValueError('Reviewed Ad5 label binding requires an original finite edge')
    for locator in group['source_locators']:
        if not _same_reference(locator['source_pack'], binding['source_pack']):
            raise ValueError('Unreviewed Ad5 external scope cannot alias the internal graphic material')
        header = headers[locator['source_row_index']]
        if (header['model_target'] != target or header.get('curve') != target
                or header['actual_pts'] != group['actual_pts']
                or header['native_pts'] != locator['original_native_pts']
                or header['decoded_BGR_sha256'] != locator['decoded_BGR_sha256']
                or header['physical_material_id'] != 'Ad5.' + edge
                or header.get('source_anchor_id') not in binding['allowed_source_anchor_ids']
                or not header['source_anchor_id'].endswith(':Ad5:' + edge)
                or not _same_reference(header.get('source_annotation_reference', {}), binding['corner_annotation_reference'])):
            raise ValueError('Ad5 scope correction lost its exact original annotation/source row identity')
    return True


def apply_confirmed_v6_target_bindings(fit, held, dataset_reference):
    """Bind only the exact reviewed v6 identity; unknown datasets stay strict."""
    reference = dict(path=str(_BINDING_PATH), sha256=_BINDING_SHA256)
    authenticate_source_reference(reference)
    binding = json.loads(_BINDING_PATH.read_text(encoding='utf8'))
    if not _same_reference(dataset_reference, binding['dataset_reference']):
        return dict(applied=False, binding_reference=reference, dataset_identity_matches=False)
    for item in (binding['source_pack'], binding['corner_annotation_reference'], binding['source_frame_reference'],
            binding['exact_original_row_binding_closure_reference']):
        authenticate_source_reference(item)
    closure=json.loads(Path(binding['exact_original_row_binding_closure_reference']['path']).read_text(encoding='utf8'))
    if (closure['invalid_binding_rows'] or closure['fit_points'] != binding['expected_finite_FIT_points']
            or set(closure['source_anchor_points']) != set(binding['allowed_source_anchor_ids'])):
        raise ValueError('Complete original Ad5 corner seed and point-proof closure changed')
    annotation = json.loads(Path(binding['corner_annotation_reference']['path']).read_text(encoding='utf8'))
    observations = [row for row in annotation['observations'] if row.get('user_number') == 5]
    if (len(observations) != binding['original_Ad5_annotation_observations']
            or any(row.get('measurement_component') != 'meiwa_color_bar_envelope' for row in observations)):
        raise ValueError('Original Ad5 annotation is not the independently verified MEIWA internal colour-bar envelope')
    with np.load(binding['source_pack']['path'], allow_pickle=False) as pack:
        headers = unpack_source_json(pack['rows_json'])
    selected = [(role, group) for role, rows in (('fit',fit),('held',held)) for group in rows
        if _qualify_group(group, headers, binding)]
    points = sum(len(group['uv']) for _,group in selected)
    if points != binding['expected_finite_FIT_points'] or any(role != 'fit' for role,_ in selected):
        raise ValueError('Reviewed Ad5 finite scope correction inventory changed')
    for _,group in selected:
        group['original_model_target_before_confirmed_label_binding'] = dict(group['model_target'])
        group['model_target'] = dict(group['model_target'], extent_scope='internal_graphic_only')
        group['confirmed_target_label_binding_reference'] = dict(reference)
    return dict(applied=True, binding_reference=reference, groups=len(selected), points=points,
        original_scope='external_border', effective_scope='internal_graphic_only',
        annotation_component='meiwa_color_bar_envelope',
        source_pixels_sigma_paths_roles_clock_and_original_locators_unchanged=True,
        other_Ad5_external_sources_unreviewed_and_refused=True)
