"""Optional RGB qualification of original absolute FIT point slots.

The existing source stream and nonlinear guards are retained. Original source
records remain immutable; only explicitly reviewed residual/J rows retire.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np

from .source_point_qualification import (
    SCHEMA, _direct_user_annotation, _reference, file_sha256,
    mask_scalar_rows, original_uv_sha256,
)

FAMILY = 'original_programme_absolute'
LOCATOR = ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')


def _plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist()
        if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)))


def _verify_runtime_row(runtime, original, item, indices, state):
    """Allow only the separately certified pixel edit on other point slots."""
    expected = deepcopy(original)
    patch = runtime.get('active_source_pixel_correction')
    if patch is not None:
        authority = getattr(state, 'original_ice_pixel_correction', None)
        if (not isinstance(authority, dict)
                or patch.get('packet_sha256') != authority.get('packet_sha256')):
            raise ValueError('Runtime pixel edit lacks its installed source authority')
        records = [r for r in authority['corrected_rows']
            if r.get('original_packet_row_index') == item['original_packet_row_index']]
        if len(records) != 1:
            raise ValueError('Runtime pixel correction row is not uniquely authenticated')
        revision = records[0]
        corrected = revision['point_indices']
        if set(indices) & set(corrected):
            raise ValueError('A currently source-corrected/user pixel cannot be retired')
        if (_plain(revision['original_source_row']) != _plain(original)
                or _plain(revision['locator']) != _plain({k: item[k] for k in LOCATOR})):
            raise ValueError('Installed pixel correction no longer binds the original source row')
        expected['uv'] = np.asarray(original['uv']).copy()
        expected['uv'][corrected] = revision['corrected_uv']
        if revision.get('optical_support_updated'):
            expected['normal_interval'] = np.asarray(original['normal_interval']).copy()
            expected['normal_interval'][corrected] = revision['corrected_normal_interval_px']
        if revision.get('original_normal_interval_removed_from_operator'):
            footprint = {key: _plain(original[key]) for key in ('normal_interval', 'normal_interval_px') if key in original}
            expected.pop('normal_interval', None)
            expected.pop('normal_interval_px', None)
            expected['measurement_role'] = 'painted_line_center'
            expected['source_paint_footprint_evidence'] = dict(original_source_fields=footprint,
                measured_paint_footprint_evidence=deepcopy(revision.get('paint_footprint_evidence')),
                paint_footprint_is_not_location_uncertainty=True)
            retired = revision.get('retired_original_point_indices', [])
            reference = revision.get('source_point_qualification_reference')
            if (set(retired) != set(indices) or not isinstance(reference, dict)
                    or patch.get('source_point_qualification_reference') != reference):
                raise ValueError('Painted row retirement must match its exact installed source qualification')
        if 'source_uv' in original and np.array_equal(original['source_uv'], original['uv']):
            expected['source_uv'] = expected['uv'].copy()
        expected['active_source_pixel_correction'] = patch
    if any(_plain(runtime.get(key)) != _plain(value) for key, value in expected.items()):
        raise ValueError('Runtime original source UV/sigma/world/role/operator/history changed')
    if _direct_user_annotation(runtime):
        raise ValueError('Direct user source pixels cannot be retired')


class OriginalSourcePointQualification:
    def __init__(self, packet_path):
        path = Path(packet_path).resolve()
        manifest = json.loads(path.read_text(encoding='utf8'))
        if (manifest.get('schema') != SCHEMA or manifest.get('source_family') != FAMILY
                or not isinstance(manifest.get('source_video_sha256'), str)
                or len(manifest['source_video_sha256']) != 64
                or not isinstance(manifest.get('time_base'), str)):
            raise ValueError('Explicit original absolute source-only qualification required')
        parent = manifest.get('original_packet', {})
        if parent.get('format') != 'original_absolute_compact_pickle':
            raise ValueError('Exact immutable original absolute compact packet required')
        original_path = _reference(parent, path, 'Original absolute packet')
        original = pickle.loads(original_path.read_bytes())['programme_absolute_rows']
        records = manifest.get('records')
        if not isinstance(records, list) or not records:
            raise ValueError('At least one source-only reviewed point record required')
        self.records, seen = [], set()
        for record in records:
            index = record.get('original_packet_row_index')
            if type(index) is not int or not 0 <= index < len(original) or index in seen:
                raise ValueError('Unique exact original packet row index required')
            seen.add(index)
            item = original[index]; row = item['source_row']
            if (item['collection_role'] != 'FIT' or row.get('split') not in ('fit', 'development')
                    or _direct_user_annotation(row)):
                raise ValueError('Only existing FIT points; HELD and direct user pixels stay unchanged')
            if (_plain(record.get('locator')) != _plain({k: item[k] for k in LOCATOR})
                    or record.get('row_id') != row['id']
                    or type(record.get('source_pts')) is not int or record['source_pts'] != int(row['pts'])
                    or record.get('original_uv_float64_sha256') != original_uv_sha256(row)):
                raise ValueError('Original row locator, identity, PTS or UV bytes changed')
            indices = record.get('excluded_point_indices')
            if (not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(row['uv']) for i in indices)
                    or len(indices) != len(set(indices))):
                raise ValueError('Unique in-range original point indices required')
            if (record.get('review_basis') != 'original_rgb_only'
                    or record.get('decision_is_not_from_model_residual_or_cost') is not True
                    or not isinstance(record.get('reason'), str) or not record['reason'].strip()):
                raise ValueError('RGB-only qualification; model residual/cost cannot retire source points')
            _reference(record.get('review_authority'), path, 'Original RGB source review')
            evidence = record.get('native_rgb_evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('Exact native RGB evidence required')
            for ref in evidence:
                _reference(ref, path, 'Native RGB evidence')
                if (type(ref.get('source_pts')) is not int or ref['source_pts'] != int(item['group_actual_pts'])
                        or ref.get('source_video_sha256') != manifest['source_video_sha256']
                        or ref.get('time_base') != manifest['time_base']):
                    raise ValueError('RGB must bind this actual exposure, source video and clock')
            self.records.append(dict(record=deepcopy(record), original=deepcopy(row),
                item=dict({k: item[k] for k in LOCATOR}, original_packet_row_index=index)))
        self.contract = dict(schema=SCHEMA, source_family=FAMILY,
            packet_path=str(path), packet_sha256=file_sha256(path),
            original_packet=dict(path=str(original_path), sha256=parent['sha256'], format=parent['format']),
            source_video_sha256=manifest['source_video_sha256'], time_base=manifest['time_base'],
            reviewed_records=deepcopy(records), original_packet_bytes_untouched=True,
            source_UV_sigma_world_split_operator_and_history_unchanged=True,
            excluded_points_not_in_active_FIT_accuracy_statistics=True,
            inactive_zero_residual_is_accuracy_measurement=False,
            scalar_slots_removed=0, new_FIT_samples_added=0,
            direct_user_annotation_and_HELD_unchanged=True,
            original_nonlinear_truth_and_physical_guards_unchanged=True,
            fresh_complete_source_profiles_and_J_required=True, accuracy_accepted=False)
        self._blocks = {}

    def qualify_state(self, state):
        """Derive current slots from actual full blocks, including compaction."""
        prequalified = getattr(state, 'original_absolute_point_qualification_prequalification', None)
        if prequalified is not None:
            if (prequalified.contract['packet_sha256'] != self.contract['packet_sha256']
                    or _plain(prequalified.contract['reviewed_records']) != _plain(self.contract['reviewed_records'])):
                raise ValueError('Physically compacted points need their exact prior RGB qualification')
            self.records = deepcopy(prequalified.records)
            blocks = {block.name: block for block in state.blocks}
            verified_retired = []
            for entry in self.records:
                item, record = entry['item'], entry['record']
                block = blocks[item['block']]
                group, position = int(item['group_order']), int(item['row_order'])
                pts, rows = block.groups[group]; runtime = rows[position]
                if int(pts) != int(item['group_actual_pts']):
                    raise ValueError('Physically retired source exposure changed')
                mapping = runtime.get('active_original_absolute_input_compaction')
                indices = record['excluded_point_indices']
                from .original_absolute_input_compaction import original_points_to_active, slice_observed_points
                if not isinstance(mapping, dict):
                    raise ValueError('Qualified source points require physical input removal, not zero slots')
                keep = np.asarray(mapping.get('retained_original_point_indices'), np.int64)
                if (mapping.get('original_point_count') != len(entry['original']['uv'])
                        or sorted(set(range(len(entry['original']['uv']))) - set(keep.tolist())) != sorted(indices)):
                    raise ValueError('Actual source input removal differs from exact qualified original points')
                original_points_to_active(runtime, keep)
                prior_pts, prior_rows = block.original_absolute_groups[group]
                prior = prior_rows[position]
                _verify_runtime_row(prior, entry['original'], item, indices, state)
                expected = slice_observed_points(prior, keep)
                if int(prior_pts) != int(pts) or any(_plain(runtime.get(k)) != _plain(v) for k, v in expected.items()):
                    raise ValueError('Physically retired source pixels/sigma/operator differ from verified retained input')
                revision = prior.get('active_source_pixel_correction')
                if revision and revision.get('retired_points_require_verified_physical_input_compaction'):
                    ref = revision.get('source_point_qualification_reference')
                    if (not isinstance(ref, dict) or Path(ref['path']).resolve() != Path(self.contract['packet_path']).resolve()
                            or ref['sha256'] != self.contract['packet_sha256']
                            or sorted(revision['retired_original_point_indices']) != sorted(indices)):
                        raise ValueError('Painted row must use the final exact physical qualification manifest')
                    verified_retired.append(dict(original_packet_row_index=item['original_packet_row_index'],
                        retired_original_point_indices=sorted(indices), retained_original_point_indices=keep.tolist()))
            self._blocks = {}
            self.contract.update(active_FIT_points_retired=sum(len(r['record']['excluded_point_indices']) for r in self.records),
                active_FIT_scalars_retired=0,
                scalar_slots_removed=2*sum(len(r['record']['excluded_point_indices']) for r in self.records),
                active_runtime_global_scalar_slots=[],
                original_precompaction_global_scalar_slots=prequalified.contract['active_runtime_global_scalar_slots'],
                source_identity_points_removed_before_native_r_J=True,
                painted_row_retirement_physical_removal_verified=verified_retired,
                input_compaction=deepcopy(state.original_absolute_input_compaction_contract))
            if verified_retired:
                state.original_ice_pixel_correction['painted_row_retirement_physical_removal_verified'] = deepcopy(verified_retired)
            return
        blocks, cursor = {}, 0
        for block in state.blocks:
            if block.name in blocks:
                raise ValueError('Original programme block name duplicated')
            blocks[block.name] = (block, cursor, cursor + len(block.quadratic))
            cursor += len(block.quadratic)
        selected = defaultdict(list)
        for entry in self.records:
            record, item, original = entry['record'], entry['item'], entry['original']
            block, start, stop = blocks[item['block']]
            group, position = int(item['group_order']), int(item['row_order'])
            pts, rows = block.groups[group]
            if (int(pts) != int(item['group_actual_pts'])
                    or int(state.original.aliases.get(int(original['pts']), original['pts'])) != int(pts)):
                raise ValueError('Runtime original row exposure/alias changed')
            runtime = rows[position]
            indices = record['excluded_point_indices']
            _verify_runtime_row(runtime, original, item, indices, state)
            revision = runtime.get('active_source_pixel_correction')
            if revision and revision.get('retired_points_require_verified_physical_input_compaction'):
                ref = revision['source_point_qualification_reference']
                if (Path(ref['path']).resolve() != Path(self.contract['packet_path']).resolve()
                        or ref['sha256'] != self.contract['packet_sha256']):
                    raise ValueError('Painted row must use the final exact physical qualification manifest')
            before = sum(2 * len(row['uv']) for _, group_rows in block.groups[:group] for row in group_rows)
            before += sum(2 * len(row['uv']) for row in rows[:position])
            row_start = start + int(block.bundle.motion_rows) + before
            slots = np.asarray([[row_start + 2*i, row_start + 2*i + 1] for i in indices], np.int64).ravel()
            if np.any(slots < start) or np.any(slots >= stop):
                raise ValueError('Qualified absolute scalar slots outside original programme block')
            entry['global_scalar_slots'] = slots.tolist()
            selected[(f'programme:{block.name}', start, stop)].extend(slots.tolist())
        self._blocks = {key: np.asarray(sorted(slots), np.int64) for key, slots in selected.items()}
        self.contract.update(active_FIT_points_retired=sum(len(r['record']['excluded_point_indices']) for r in self.records),
            active_FIT_scalars_retired=sum(len(r) for r in self._blocks.values()),
            active_runtime_global_scalar_slots=sorted(i for rows in self._blocks.values() for i in rows.tolist()),
            source_slot_location_derived_from_complete_original_group_order=True,
            original_motion_compaction_resolved_from_actual_current_block_motion_rows=True)

    def block(self, block, jacobian):
        from z190_streaming_source_normal import SourceRowBlock
        ledger = block.row_ledger
        key = (ledger.get('family'), ledger.get('original_start'), ledger.get('original_stop'))
        slots = self._blocks.get(key)
        if slots is None:
            if any(key[0] == expected[0] for expected in self._blocks):
                raise ValueError('Qualified original source block scalar range changed')
            return block, None
        if (len(block.r) != key[2] - key[1]
                or ledger.get('source_sha256') != self.contract['source_video_sha256']
                or (jacobian and block.J is None)):
            raise ValueError('Qualified original source stream layout/video/J changed')
        residual, matrix = mask_scalar_rows(block.r, block.J if jacobian else None, slots-key[1])
        inactive = sorted(set(ledger.get('inactive_scalar_slots_excluded_from_source_accuracy_statistics', [])) | set(slots.tolist()))
        revised = dict(ledger, original_source_point_qualification=deepcopy(self.contract),
            inactive_scalar_slots_excluded_from_source_accuracy_statistics=inactive,
            inactive_zero_residual_is_accuracy_measurement=False,
            source_exclusion_is_RGB_identity_not_model_error=True)
        return SourceRowBlock(residual, matrix, block.quadratic, block.scales, revised,
            getattr(block, 'loss_weights', None)), key

    def history_review(self):
        return dict(contract=deepcopy(self.contract), rows=[dict(
            locator=deepcopy(entry['record']['locator']), original_source_row=_plain(entry['original']),
            excluded_point_indices=entry['record']['excluded_point_indices'],
            global_scalar_slots=entry.get('global_scalar_slots'), active_source_role='source_identity_pending',
            included_in_active_FIT_accuracy_statistics=False, accuracy_metric=None,
            inactive_zero_residual_is_pixel_alignment=False) for entry in self.records], accuracy_accepted=False)


class SourceWithOriginalPointQualification:
    def __init__(self, base, state, packet_path):
        self.base = base
        self.original_point_qualification = OriginalSourcePointQualification(packet_path)
        self.original_point_qualification.qualify_state(state)
        if (getattr(state, 'original_ice_pixel_correction', None)
                and any(r.get('retired_points_require_verified_physical_input_compaction')
                        for r in state.original_ice_pixel_correction['corrected_rows'])
                and not self.original_point_qualification.contract.get('source_identity_points_removed_before_native_r_J')):
            raise ValueError('Painted row source stream requires verified physical input compaction before residual/J')
        self.count, self.updater = base.count, base.updater
        self.scales, self.quadratic = base.scales, base.quadratic
        self.contract = dict(schema=SCHEMA+'.original-complete-source', original_source_contract=deepcopy(base.contract),
            original_source_point_qualification=deepcopy(self.original_point_qualification.contract),
            total_rows=len(self.scales), original_parameter_source_mask_scale_profile_and_geometry_layout_preserved=True,
            accuracy_accepted=False)
        refs = [Path(__file__).resolve(), Path(__file__).with_name('source_point_qualification.py').resolve()]
        self.contract['original_source_point_qualification']['implementation_sources'] = [
            dict(path=str(p), sha256=file_sha256(p)) for p in refs]

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None: raise AttributeError(name)
        return getattr(base, name)

    def blocks(self, q, jacobian=True):
        found = set()
        for block in self.base.blocks(q, jacobian):
            block, key = self.original_point_qualification.block(block, jacobian)
            if key is not None:
                if key in found: raise ValueError('Qualified original source block duplicated')
                found.add(key)
            yield block
        if found != set(self.original_point_qualification._blocks):
            raise ValueError('Complete source omitted qualified original programme block')


def wrap_original_source_point_qualification(base, state, packet_path=None):
    """Same stream interface as original source retirement; None is identity."""
    return base if packet_path is None else SourceWithOriginalPointQualification(base, state, packet_path)
