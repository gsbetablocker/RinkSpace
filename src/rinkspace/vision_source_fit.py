"""Reread original FIT pixels through their registered physical definitions.

Neural probabilities, legacy assistant UV and fitted world coordinates locate
searches only. Returned row edits/retirements are candidates for the existing
authenticated solver input transaction, never a silently installed new FIT.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .vision_instances import source_binding, same_frame
from .vision_source_shapes import (_MaterialProfileFrame,
    read_source_material_transition_profiles, read_source_stroke_midlines)

SCHEMA = 'rinkspace.original-FIT-source-RGB-revision-candidate.v1'


def _plain(value):
    if isinstance(value, np.ndarray):return _plain(value.tolist())
    if isinstance(value, np.generic):return _plain(value.item())
    if isinstance(value, (float,)) and not np.isfinite(value):return None
    if isinstance(value, dict):return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):return [_plain(item) for item in value]
    return value


def _reference(reference):
    path = Path(reference['path']).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != reference['sha256']:
        raise ValueError('Original FIT visual request reference bytes changed')
    return path


def _physical_definition(geometry, row, definition):
    _, binding = geometry._binding(row)
    identity = binding['contract'].get('physical_identity', binding['contract'].get('physical_identities'))
    if identity is None or _plain(identity) != _plain(definition.get('physical_identity')):
        raise ValueError('Visual measurement definition differs from registered original material')
    if not definition.get('physical_role') or not definition.get('identity_authority'):
        raise ValueError('Source measurement needs its confirmed role and recorded identity authority')
    operator = definition.get('measurement_operator')
    if operator == 'paired_material_boundary_midline':
        ordinary = binding['kind'] in ('line', 'carrier', 'lane', 'crease')
        feature = geometry.features.get(binding.get('name'), {})
        style = feature.get('style', {})
        circle = (binding['kind'] == 'circle' and not feature.get('rings')
            and ('fill_srgb' in feature or 'fill_srgb' in style)
            and feature.get('fill_srgb') is None and style.get('fill_srgb') is None)
        if not (ordinary or circle) or definition.get('two_sides_belong_to_same_painted_stroke') is not True:
            raise ValueError('Paint midline requires the same ordinary stroke, never a layered or filled boundary')
    elif operator == 'signed_material_transition':
        if definition.get('material_boundary_role_confirmed') is not True:
            raise ValueError('A material edge needs its confirmed outer/inner/print/top/contact role')
    else:
        raise ValueError('Explicit physical measurement operator required; labels alone do not select one')
    return identity, operator


def reread_original_fit_source_frame(rgb, *, binding, geometry, requests, original_packet_reference=None):
    """Batch current native profiles, preserving every original point decision.

    ``requests`` each contain the immutable source row/locator, original row
    index, and explicit source material definition with normal search brackets.
    Operators apply to all registered physical identities, without a feature
    name whitelist. A missing side, mixed interface or clipped search retires
    no old data silently: it emits a source-qualification candidate reason.
    """
    from .feedback_runtime import SourceFeedbackRead
    requests = list(requests)
    def declared_ids(value):
        if isinstance(value, str):return [value]
        if isinstance(value, (list, tuple)):
            return [name for item in value for name in declared_ids(item)]
        if isinstance(value, dict):
            return [name for key in ('id', 'feature_id', 'curve_id', 'name', 'physical_identity')
                    if key in value for name in declared_ids(value[key])]
        return []
    entities = [name for request in requests for name in declared_ids(request['definition'].get('physical_identity'))]
    feedback = SourceFeedbackRead(binding, entity_ids=entities, tags=['measurement_operator', 'ice_paint'])
    frame = _MaterialProfileFrame(rgb, binding)
    decisions, edits = [], []
    seen = set()
    for request in requests:
        ordinal = request['original_packet_row_index']
        if type(ordinal) is not int or ordinal in seen:
            raise ValueError('Distinct original FIT source row indices required')
        seen.add(ordinal)
        locator, row = request['locator'], request['source_row']
        if locator['collection_role'] != 'FIT' or row.get('split') not in ('fit', 'development'):
            raise ValueError('Original FIT source rereading cannot promote HELD rows')
        if int(locator['group_actual_pts']) != int(frame.binding['source_pts']):
            raise ValueError('Original FIT row belongs to a different native exposure')
        definition = request['definition']
        defined_binding = source_binding(definition['source_binding'], frame.rgb.shape[:2])
        same_frame(frame.binding, defined_binding)
        if defined_binding.get('decoded_native_rgb_sha256', frame.digest) != frame.digest:
            raise ValueError('Original FIT definition belongs to different bound original RGB')
        identity, operator = _physical_definition(geometry, row, definition)
        points = np.asarray(row['uv'], float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Exact finite original source UV required')
        if not feedback.require_operator(declared_ids(identity), operator):
            pending = list(range(len(points)))
            decisions.extend(dict(original_packet_row_index=ordinal, original_point_index=index,
                source_pts=int(frame.binding['source_pts']), locator=deepcopy(locator),
                physical_identity=_plain(identity), measurement_operator=operator,
                action='withhold_conflicting_operator_revision_remeasure_both_material_flanks',
                original_uv=points[index].tolist(), actual_source_uv=None,
                source_flags=dict(confirmed_measurement_operator_conflict=True)) for index in pending)
            edits.append(dict(original_packet_row_index=ordinal, locator=deepcopy(locator),
                corrected_original_point_indices=[], corrected_uv=[], retired_original_point_indices=[],
                pending_original_point_indices=pending, all_original_points_resolved_before_FIT_admission=False,
                original_UV_preserved=True, confirmed_measurement_operator_conflict=True))
            continue
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Exact finite original source UV required')
        selected = np.asarray(definition.get('original_point_indices', np.arange(len(points))))
        if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(np.unique(selected)) != len(selected)
                or np.any(selected < 0) or np.any(selected >= len(points))):
            raise ValueError('Distinct exact original point indices required')
        common = dict(binding=frame.binding, search_points_uv=points[selected],
            normals=definition['source_search_normals'], expected_RGB_sha256=frame.digest,
            material_signal_rgb_weights=definition.get('material_signal_rgb_weights'),
            tangent_half_width_px=definition.get('tangent_half_width_px', 2),
            keep_profiles=False, _validated_frame=frame)
        if operator == 'paired_material_boundary_midline':
            measured = read_source_stroke_midlines(frame.rgb,
                entering_search_intervals_px=definition['entering_search_intervals_px'],
                leaving_search_intervals_px=definition['leaving_search_intervals_px'],
                entering_context_search_intervals_px=definition.get('entering_context_search_intervals_px'),
                leaving_context_search_intervals_px=definition.get('leaving_context_search_intervals_px'),
                paint_signal_increases=definition.get('paint_signal_increases', False), **common)
        else:
            measured = read_source_material_transition_profiles(frame.rgb,
                search_intervals_px=definition['search_intervals_px'],
                context_search_intervals_px=definition.get('context_search_intervals_px'),
                transition=definition['transition'], **common)
        visibility = definition.get('source_visible_mask', np.ones(len(points), bool))
        visibility = np.asarray(visibility)
        if visibility.dtype != np.bool_ or visibility.shape != (len(points),):
            raise ValueError('Source visibility must retain one explicit boolean per original point')
        corrected, retired, pending, actual = [], [], [], []
        human = row.get('manual_annotation_ground_truth') is True or row.get('direct_user_annotation') is True
        for local_index, reading in enumerate(measured['readings']):
            index = int(selected[local_index])
            flags = deepcopy(reading['source_flags'])
            flags['known_source_occlusion'] = not bool(visibility[index])
            available = np.asarray(reading['actual_source_uv'])
            readable = available.shape == (2,) and np.isfinite(available).all() and not any(flags.values())
            action = ('retain_human_source_requires_explicit_user_revision' if human else
                'correct_UV_candidate' if readable else
                'retire_from_native_input_candidate' if not visibility[index] else
                'withhold_FIT_measurement_source_review_pending')
            if human or (not readable and visibility[index]):pending.append(index)
            elif readable:corrected.append(index); actual.append(available.tolist())
            else:retired.append(index)
            decisions.append(dict(original_packet_row_index=ordinal, original_point_index=index,
                source_pts=int(frame.binding['source_pts']), original_source_pts=int(row['pts']), locator=deepcopy(locator),
                physical_identity=_plain(identity), physical_role=definition['physical_role'], measurement_operator=operator,
                action=action, original_uv=points[index].tolist(), actual_source_uv=available.tolist() if readable else None,
                source_flags=flags, measurement_evidence=_plain(reading)))
        for index in np.setdiff1d(np.arange(len(points)), selected):
            index = int(index)
            pending.append(index)
            decisions.append(dict(original_packet_row_index=ordinal, original_point_index=index,
                source_pts=int(frame.binding['source_pts']), original_source_pts=int(row['pts']), locator=deepcopy(locator),
                physical_identity=_plain(identity), physical_role=definition['physical_role'], measurement_operator=operator,
                action='unmeasured_original_point_pending_source_qualification', original_uv=points[index].tolist(),
                actual_source_uv=None, source_flags=dict(explicit_material_search_not_supplied=True)))
        edits.append(dict(original_packet_row_index=ordinal, locator=deepcopy(locator), original_id=row['id'],
            original_source_pts=int(row['pts']), actual_source_pts=int(frame.binding['source_pts']),
            original_full_uv_float64_sha256=hashlib.sha256(points.astype(np.float64).tobytes()).hexdigest(),
            physical_identity=_plain(identity), physical_role=definition['physical_role'],
            identity_authority=deepcopy(definition['identity_authority']), measurement_operator=operator,
            corrected_original_point_indices=corrected, corrected_uv=actual,
            retired_original_point_indices=retired, pending_original_point_indices=pending,
            all_original_points_resolved_before_FIT_admission=not pending,
            current_RGB_pixel_boundary_quality_is_not_inferred_from_readable=True,
            physical_localization_review_required=True,
            original_sigma=_plain(row['sigma']), original_WORLD_sha256=hashlib.sha256(np.asarray(row['world'], np.float64).tobytes()).hexdigest()))
    result = dict(schema=SCHEMA, source_binding=frame.binding, RGB_sha256=frame.digest,
        original_packet_reference=deepcopy(original_packet_reference), row_edits=edits, point_decisions=decisions,
        corrected_point_count=sum(len(row['corrected_original_point_indices']) for row in edits),
        retirement_candidate_count=sum(len(row['retired_original_point_indices']) for row in edits),
        source_review_pending_count=sum(len(row['pending_original_point_indices']) for row in edits),
        all_requested_original_points_have_explicit_decisions=True, all_registered_physical_identities_supported=True,
        neural_probability_or_old_assistant_UV_used_as_measurement=False, world_or_camera_fitted=False,
        original_WORLD_sigma_clock_and_source_bytes_unchanged=True, pixel_labels_or_solver_sigma_created=False,
        public_correction_qualification_and_physical_compaction_transaction_required=True,
        source_revision_candidates_installed=False, accuracy_accepted=False)
    result['source_feedback'] = feedback.finish(frame.rgb, dict(
        corrected_point_count=result['corrected_point_count'],
        retirement_candidate_count=result['retirement_candidate_count'],
        source_review_pending_count=result['source_review_pending_count']),
        tool='OriginalFITSourceReader.read')
    return result


class OriginalFITSourceReader:
    """One immutable packet/material producer for a complete native sequence.

    Prepare once before decoding a shot. A frame is never represented by rows
    from another PTS. The network/decoder are owned by the existing outer
    SourceRinkReader/review loop; this object neither loads weights nor opens a
    video. All exact original FIT locators are retained at preparation time.
    """
    def __init__(self, request_path):
        from .vision_camera_initialization import current_complete_source_geometry
        from .vision_data import _DataOnlyUnpickler
        request_path = Path(request_path).resolve()
        request_bytes = request_path.read_bytes()
        packet = json.loads(request_bytes.decode('utf-8-sig'))
        if packet.get('schema') != 'rinkspace.original-FIT-source-RGB-reread-request.v1':
            raise ValueError('Explicit original FIT reread request required')
        original_path = _reference(packet['original_absolute_packet'])
        with original_path.open('rb') as stream:
            original = _DataOnlyUnpickler(stream).load()['programme_absolute_rows']
        state_path = _reference(packet['complete_state'])
        auxiliary = _reference(packet['world_initializer'])
        self.geometry, _, self.geometry_receipt = current_complete_source_geometry(state_path, auxiliary)
        frames = packet.get('frames', [dict(source_binding=packet.get('source_binding'), records=packet.get('records'))])
        self.rows_by_pts, self.bindings_by_pts = {}, {}
        self.original_packet_reference = deepcopy(packet['original_absolute_packet'])
        self.request_reference = dict(path=str(request_path), sha256=hashlib.sha256(request_bytes).hexdigest())
        self.source_identity = None
        seen = set()
        for frame in frames:
            binding = source_binding(frame['source_binding'])
            stamp = int(binding['source_pts'])
            identity = (binding['source_sha256'], binding['time_base'])
            if self.source_identity is None:self.source_identity = identity
            if identity != self.source_identity or stamp in self.rows_by_pts:
                raise ValueError('Distinct original native frames on one verified source clock required')
            requests = []
            for entry in frame['records']:
                ordinal = entry['original_packet_row_index']
                if type(ordinal) is not int or not 0 <= ordinal < len(original) or ordinal in seen:
                    raise ValueError('Distinct exact original source row indices required')
                seen.add(ordinal)
                item = original[ordinal]
                if int(item['group_actual_pts']) != stamp:
                    raise ValueError('Prepared original FIT locator belongs to another native frame')
                defined_binding = source_binding(entry['definition']['source_binding'])
                same_frame(binding, defined_binding)
                if defined_binding.get('decoded_native_rgb_sha256') != binding.get('decoded_native_rgb_sha256'):
                    raise ValueError('Prepared original FIT frame and definition RGB identities differ')
                self.geometry.register_row(item['source_row'], actual_pts=stamp)
                requests.append(dict(entry, locator={key: item[key] for key in
                    ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')}, source_row=item['source_row']))
            self.rows_by_pts[stamp] = requests
            self.bindings_by_pts[stamp] = binding

    def read(self, rgb, binding):
        actual = source_binding(binding, rgb.shape[:2])
        if (actual['source_sha256'], actual['time_base']) != self.source_identity:
            raise ValueError('Prepared original FIT reader belongs to another source/clock')
        stamp = int(actual['source_pts'])
        runtime_binding = deepcopy(actual)
        if stamp in self.bindings_by_pts:
            registered = source_binding(self.bindings_by_pts[stamp], rgb.shape[:2])
            # read-frame and review-shot name their runtime clips separately
            # from the immutable FIT programme block. Authenticate the native
            # exposure and RGB, then retain this explicit label correspondence;
            # never weaken the public same_frame contract or reassign PTS.
            same_frame(dict(actual, shot_id=registered['shot_id']), registered)
            expected = registered.get('decoded_native_rgb_sha256')
            if expected is not None and actual.get('decoded_native_rgb_sha256', expected) != expected:
                raise ValueError('Prepared FIT frame and runtime decoded RGB identity differ')
            actual = dict(actual, shot_id=registered['shot_id'])
            if expected is not None:actual['decoded_native_rgb_sha256'] = expected
        result = reread_original_fit_source_frame(rgb, binding=actual, geometry=self.geometry,
            requests=self.rows_by_pts.get(stamp, []), original_packet_reference=self.original_packet_reference)
        result['runtime_source_binding'] = runtime_binding
        result['runtime_shot_to_original_FIT_block_label_mapping'] = dict(
            runtime_shot_id=runtime_binding['shot_id'], original_FIT_block_shot_id=actual['shot_id'],
            same_native_source_clock_PTS_dimensions_and_RGB_authenticated=True)
        result['actual_complete_material_producer'] = self.geometry_receipt
        result['request_reference'] = self.request_reference
        result['frame_original_FIT_locator_status'] = ('original_FIT_rows_reread' if stamp in self.rows_by_pts else 'no_original_FIT_locator_at_this_frame')
        result['prepared_original_packet_and_geometry_reused'] = True
        return result


def reread_original_fit_request(rgb, binding, request_path, *, prepared_reader=None):
    """Single-frame convenience; streaming callers pass/reuse the prepared reader."""
    reader = OriginalFITSourceReader(request_path) if prepared_reader is None else prepared_reader
    if Path(request_path).resolve() != Path(reader.request_reference['path']):
        raise ValueError('Prepared original FIT reader belongs to a different request')
    return reader.read(rgb, binding)
