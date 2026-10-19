"""Optional source-budget adapter around the existing complete source stream.

No geometry, reader, profile, guard, optimizer or admission rule lives here.
The caller supplies a source-authenticated grouping policy after constructing
all original source wrappers. No policy returns the exact original owner.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from rinkspace.source_sampling import compile_curve_sampling
from .source_loss import SourceLossWeights
from .source_point_qualification import file_sha256, original_uv_sha256


SCHEMA = 'rinkspace.equal-physical-curve-source-integration.v1'


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError('Explicit source contract JSON required')


def source_contract_sha256(contract):
    text = json.dumps(contract, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False, default=_json_default)
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def _array_sha(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _reference(ref, manifest_path, label, authenticated_files=None):
    if not isinstance(ref, dict) or not {'path', 'sha256'} <= ref.keys():
        raise ValueError(label + ' exact path and SHA256 required')
    path = Path(ref['path'])
    if not path.is_absolute(): path = manifest_path.parent / path
    path = path.resolve()
    cache_key = (str(path), 'authenticated_sha256')
    if authenticated_files is None:
        digest = file_sha256(path)
    else:
        # Receipts can contain all native RGB profiles of a continuous span.
        # Authenticate once per constructor, never once per measurement row.
        if cache_key not in authenticated_files:
            authenticated_files[cache_key] = file_sha256(path)
        digest = authenticated_files[cache_key]
    if digest != ref['sha256']:
        raise ValueError(label + ' hash changed')
    return path


def _source_row(ref, manifest_path, files):
    path = _reference(ref, manifest_path, 'Source grouping original row packet', files)
    key = (str(path), ref.get('format'))
    if key not in files:
        if ref.get('format') == 'json':
            files[key] = json.loads(path.read_text(encoding='utf8'))
        elif ref.get('format') == 'original_absolute_compact_pickle':
            files[key] = pickle.loads(path.read_bytes())
        else:
            raise ValueError('Explicit owned source row packet format required')
    value = files[key]
    pointer = ref.get('pointer')
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Exact original source row pointer required')
    for part in pointer[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    if not isinstance(value, dict) or 'uv' not in value or 'id' not in value:
        raise ValueError('Pointer must resolve to an original source UV row')
    return value


class SourceWithCurveGrouping:
    """Fixed loss-integration payload, preserving original R/J/native scales."""
    def __init__(self, base, policy_path):
        self.base = base
        path = Path(policy_path).resolve()
        policy = json.loads(path.read_text(encoding='utf8'))
        if (policy.get('schema') != SCHEMA or policy.get('budget_policy') != 'equal_physical_curve'
                or policy.get('curve_budget') != 1. or type(policy.get('curve_budget')) is bool):
            raise ValueError('Explicit equal engineering budget1 per physical curve required')
        self.count, self.updater = base.count, base.updater
        self.scales, self.quadratic = base.scales, base.quadratic
        if (policy.get('source_contract_sha256') != source_contract_sha256(base.contract)
                or policy.get('source_scalar_count') != len(self.scales)
                or policy.get('source_scales_float64_sha256') != _array_sha(self.scales, '<f8')
                or policy.get('source_quadratic_bool_sha256') != _array_sha(self.quadratic, bool)):
            raise ValueError('Complete source contract/scalar/mask/scale binding changed')
        records = policy.get('records')
        if not isinstance(records, list) or not records:
            raise ValueError('Actual source curve sampling records required')
        files, selected, used = {}, defaultdict(list), set()
        for record in records:
            original = _source_row(record['source_row_reference'], path, files)
            if (original['id'] != record.get('source_row_id')
                    or original.get('split') != 'fit'
                    or original_uv_sha256(original) != record.get('original_uv_float64_sha256')
                    or original_uv_sha256(record) != original_uv_sha256(original)):
                raise ValueError('Grouping changed original source row ID, UV or FIT/HELD role')
            if record.get('physical_identity_basis') != 'source_authenticated':
                raise ValueError('Physical curve must be authenticated from source, not residual ranking')
            _reference(record.get('physical_identity_authority'), path, 'Physical curve source identity')
            family = record.get('block_family')
            if not isinstance(family, str) or not family:
                raise ValueError('Original complete source block family required')
            start, stop = record.get('block_start'), record.get('block_stop')
            indices = record.get('global_scalar_indices')
            if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(self.scales)
                    or not isinstance(indices, list) or len(indices) != len(original['uv'])
                    or any(not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair)
                           or pair[1] != pair[0]+1 or not start <= pair[0] < pair[1] < stop for pair in indices)):
                raise ValueError('Exact original paired scalar slots inside source block required')
            indices = np.asarray(indices, int).ravel()
            if any(int(i) in used for i in indices):
                raise ValueError('Source grouping scalar slot selected more than once')
            used.update(indices.tolist())
            stored = np.repeat(np.broadcast_to(np.asarray(record['stored_sigma_px'], float), (len(original['uv']),)), 2)
            if not np.array_equal(stored, np.asarray(self.scales)[indices]):
                raise ValueError('Grouping stored sigma differs from original complete source scales')
            selected[(family, start, stop)].append((record, indices-start))
        plan = compile_curve_sampling(records)
        self._payloads = {}
        self._inactive = {}
        for block, selected_rows in selected.items():
            integration = np.ones(block[2]-block[1]); normalization = np.ones_like(integration)
            for record, local in selected_rows:
                row = plan['rows'][record['id']]
                integration[local] = np.repeat(row['point_integration_weights'], 2)
                normalization[local] = np.repeat(row['working_normalization_multiplier'], 2)
            self._payloads[block] = SourceLossWeights(integration,
                normalization if np.any(normalization != 1.) else None)
            self._inactive[block] = (np.flatnonzero(integration == 0.) + block[1]).tolist()
        self.contract = dict(schema=SCHEMA+'.complete-source',
            original_source_contract=deepcopy(base.contract),
            source_curve_integration=dict(policy_path=str(path), policy_sha256=file_sha256(path),
                original_source_contract_sha256=policy['source_contract_sha256'], groups=plan['groups'],
                engineering_equal_curve_budget=1., statistical_independence_claimed=False,
                original_UV_sigma_HELD_truth_profiles_and_scalar_slots_unchanged=True,
                working_sigma_and_prior_sampling_decomposition_explicit=True,
                integration_applied_after_original_component_Huber2=True,
                source_R_J_and_native_scales_not_copied_or_modified=True,
                group_policy_precomputed_once=True, fresh_source_J_and_profiles_required=True,
                inactive_interpolation_is_accuracy_measurement=False, accuracy_accepted=False))
        import rinkspace.source_sampling as shared_sampling
        paths = [Path(__file__).resolve(), Path(__file__).with_name('source_loss.py').resolve(),
                 Path(shared_sampling.__file__).resolve()]
        self.contract['source_curve_integration']['implementation_sources'] = [
            dict(path=str(p), sha256=file_sha256(p)) for p in paths]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def blocks(self, q, jacobian=True):
        from z190_streaming_source_normal import SourceRowBlock
        found = set()
        for block in self.base.blocks(q, jacobian):
            ledger = block.row_ledger
            key = (ledger.get('family'), ledger.get('original_start'), ledger.get('original_stop'))
            payload = self._payloads.get(key)
            if payload is None:
                yield block
                continue
            if key in found or block.loss_weights is not None:
                raise ValueError('Grouped original block duplicated or already loss-weighted')
            found.add(key)
            if len(block.r) != key[2]-key[1]:
                raise ValueError('Grouped original source block scalar layout changed')
            revised = dict(ledger, source_curve_integration_policy_sha256=self.contract['source_curve_integration']['policy_sha256'],
                curve_budget_policy='equal_physical_curve',
                inactive_scalar_slots_excluded_from_source_accuracy_statistics=self._inactive[key],
                interpolation_zero_mass_is_not_pixel_accuracy=True)
            yield SourceRowBlock(block.r, block.J, block.quadratic, block.scales, revised, payload)
        if found != set(self._payloads):
            raise ValueError('Complete source stream omitted an authenticated curve block')


def wrap_curve_source_grouping(base, policy_path=None):
    """Attach only after the caller's final source wrappers; None is identity."""
    return base if policy_path is None else SourceWithCurveGrouping(base, policy_path)


NATIVE_BASE_SCHEMA = 'rinkspace.native-base-curve-source-integration-candidate.v1'


def _verify_rgb_boundary_integration(record, original, row, authority, files):
    """Bind soft boundary quadrature to its independent native RGB receipt."""
    key = (str(authority), 'rgb_boundary_measurement')
    if key not in files:
        files[key] = json.loads(authority.read_text(encoding='utf8'))
    receipt = files[key]
    index = record.get('authenticated_rgb_measurement_index')
    reference = record['physical_identity_authority']
    board_location=(record['family']=='native_board'
        and original.get('boundary_location_observation')=='measured_transition_midpoint'
        and original.get('boundary_location_measurement')==reference)
    curve=original.get('curve',{})
    ice_location=(record['family']=='native_ice'
        and original.get('measurement_role')=='filled_outer_material_boundary'
        and original.get('measurement_operator')=='integrated_actual_signed_luminance_transition_q50'
        and original.get('actual_RGB_measurement_receipt')==reference
        and curve.get('kind')=='house_ring' and curve.get('ring')=='outer'
        and curve.get('side') in ('F','R') and curve.get('row') in (1,2,3,4)
        and original.get('normal_interval_px')is None)
    if (receipt.get('schema') != 'rinkspace.actual-RGB-measured-boundary-location-receipt.v1'
            or receipt.get('source_video_sha256') != record['source_video_sha256']
            or type(index) is not int or not 0 <= index < len(receipt.get('records', []))
            or not (board_location or ice_location)):
        raise ValueError('Exact original RGB soft boundary measurement required')
    measured = receipt['records'][index]['measured_boundary_row']
    if (measured['id'] != original['id'] or measured['split'] != 'fit'
            or measured['actual_pts'] != int(row['actual_pts'])
            or measured['actual_pts'] != record['actual_pts']
            or measured['curve'] != original['curve'] or measured['curve'] != record['curve']
            or measured['source_raw_frame'] != original['source_raw_frame']
            or measured['source_file_sha256'] != original['source_file_sha256']
            or measured['source_integration_paths'] != record['source_integration_paths']):
        raise ValueError('RGB boundary physical identity, exposure or visible support changed')
    for name in ('uv', 'normal', 'normal_interval_px'):
        if not np.array_equal(np.asarray(measured.get(name)), np.asarray(original.get(name))):
            raise ValueError('Original RGB boundary observation changed: ' + name)
    if (not np.array_equal(np.broadcast_to(measured['sigma_px'], len(row['uv'])),
                           np.broadcast_to(original['sigma_px'], len(row['uv'])))
            or not np.array_equal(np.asarray(measured['point_integration_mass']),
                                  np.asarray(record['point_integration_mass']))):
        raise ValueError('Original RGB boundary sigma or integration mass changed')
    rgb_key = (measured['source_raw_frame'], 'rgb_sha256')
    if rgb_key not in files:
        files[rgb_key] = file_sha256(measured['source_raw_frame'])
    if files[rgb_key] != measured['source_file_sha256']:
        raise ValueError('Original boundary RGB pixels changed')


class SourceWithNativeBaseCurveIntegration:
    """Apply authenticated curve masses only to the original native blocks.

    Human centre factors already have their own mean normalization. Their
    residuals, and the full human guards, never pass through this adapter.
    """
    _families = {'native_ice': ('native_ice_factor', 'additional_real_native_ice_curves'),
                 'native_board': ('native_board_factor', 'additional_real_native_board_edges')}

    def __init__(self, base, policy_path):
        self.base = base
        path = Path(policy_path).resolve()
        policy = json.loads(path.read_text(encoding='utf8'))
        if (policy.get('schema') != NATIVE_BASE_SCHEMA
                or policy.get('apply_to_base_native_blocks_only') is not True
                or policy.get('do_not_apply_to_mean_normalized_direct_manual_centre_blocks') is not True):
            raise ValueError('Explicit native base-only curve integration policy required')
        records = policy.get('records')
        if not isinstance(records, list) or not records:
            raise ValueError('Authenticated native source rows required')
        files, layouts, scales, source_refs = {}, {}, {}, {}
        self._payloads = {}
        for name, (attribute, family) in self._families.items():
            factor = getattr(base, attribute, None)
            if factor is None: raise ValueError('Native integration factor is absent')
            reference = policy['packets']['ice' if name == 'native_ice' else 'board']
            if (_reference(reference, path, 'Native integration source packet') != Path(factor.packet_path).resolve()
                    or reference['sha256'] != factor.contract['packet_sha256']):
                raise ValueError('Native integration packet ownership changed')
            source_refs[name] = dict(reference)
            rows, cursor = {}, 0
            qualification = getattr(factor, 'source_point_qualification', None)
            for row in factor.rows:
                indices = np.arange(len(row['uv'])) if qualification is None else np.flatnonzero(qualification.point_mask(row))
                rows[row['id']] = (row, indices, cursor)
                cursor += 2 * len(indices)
            if cursor != factor.row_count: raise ValueError('Active native factor scalar layout changed')
            layouts[name] = rows
            scales[name] = np.asarray(factor.scales)
            self._payloads[family] = np.ones(cursor)
        selected, used = [], set()
        for record in records:
            name = record['family']
            if name not in self._families: raise ValueError('Only native base curve families may be integrated')
            original = _source_row(record['source_row_reference'], path, files)
            digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()
            packet = files[(str(_reference(record['source_row_reference'], path, 'Native row source')), 'json')]
            index = record['packet_row_index']
            if (type(index) is not int or not 0 <= index < len(packet['rows'])
                    or packet['rows'][index] != original or original['split'] != 'fit'
                    or original['id'] != record['source_row_id'] or digest != record['original_row_canonical_json_sha256']
                    or original_uv_sha256(original) != record['original_uv_float64_sha256']
                    or not np.array_equal(np.asarray(original['uv']), np.asarray(record['uv']))):
                raise ValueError('Native integration original row/UV/role changed')
            row, keep, offset = layouts[name][original['id']]
            pairs = np.arange(offset, offset + 2 * len(keep)).reshape(-1, 2)
            if (len(keep) != len(original['uv'])
                    or not np.array_equal(pairs, record['native_active_base_scalar_pairs'])
                    or not np.array_equal(np.asarray(row['uv']), np.asarray(original['uv']))
                    or not np.array_equal(scales[name][pairs.ravel()],
                        np.repeat(np.broadcast_to(record['stored_sigma_px'], len(keep)), 2))):
                raise ValueError('Native integration active paired slots or original sigma changed')
            slots = {(name, int(i)) for i in pairs.ravel()}
            if used & slots: raise ValueError('Native integration slot selected twice')
            used.update(slots)
            basis = record.get('physical_identity_basis')
            authority = _reference(record['physical_identity_authority'], path, 'Native curve source authority', files)
            if basis == 'registered_direct_user_complete_stroke':
                trace = json.loads(authority.read_text(encoding='utf8'))
                trace_indices = np.asarray(record['authenticated_registered_trace_indices'])
                if (trace_indices.dtype.kind not in 'iu' or trace_indices.shape != (len(keep),)
                        or np.any(trace_indices < 0) or np.any(trace_indices >= len(trace['user_stroke_raw_native_px']))
                        or not np.array_equal(np.asarray(trace['user_stroke_raw_native_px'])[trace_indices], row['uv'])
                        or trace['physical_curve'] != record['curve']
                        or row['curve'] != record['curve'] or int(row['actual_pts']) != record['actual_pts']):
                    raise ValueError('Native integration physical identity/actual source trace changed')
            elif basis == 'authenticated_original_RGB_boundary_locations':
                _verify_rgb_boundary_integration(record, original, row, authority, files)
            else:
                raise ValueError('Native curve identity needs an authenticated source measurement')
            from rinkspace.source_sampling import source_support_integration_weights
            expected = source_support_integration_weights(row['uv'], record['source_integration_paths'])
            if (not np.allclose(expected, record['point_integration_mass'], rtol=1e-12, atol=1e-12)
                    or not np.array_equal(np.broadcast_to(record['prior_sampling_multiplier'], len(keep)), np.ones(len(keep)))
                    or not np.array_equal(np.broadcast_to(record['stored_sigma_px'], len(keep)),
                                          np.broadcast_to(record['working_sigma_px'], len(keep)))):
                raise ValueError('Native integration support or explicit working sigma decomposition changed')
            selected.append((record, pairs.ravel()))
        plan = compile_curve_sampling(records)
        for record, slots in selected:
            family = self._families[record['family']][1]
            weights = plan['rows'][record['id']]['point_integration_weights']
            if not np.allclose(weights, record['point_integration_weights'], rtol=1e-12, atol=1e-12):
                raise ValueError('Native integration supplied weights differ from public source grouping')
            self._payloads[family][slots] = np.repeat(weights, 2)
        # Previously admitted packet rows may explicitly store a curve budget
        # in sigma. Separate that declared factor outside robust loss; do not
        # multiply another budget onto an already weighted quadratic source.
        # The option is explicit so historical policies retain their target.
        packet_sampling=[]
        normalizations={family:np.ones(len(w)) for family,w in self._payloads.items()}
        if policy.get('separate_declared_packet_sampling_normalization') is True:
            from rinkspace.source_sampling import source_support_integration_weights
            for name,(attribute,family) in self._families.items():
                packet_path=Path(source_refs[name]['path']).resolve()
                packet=json.loads(packet_path.read_text(encoding='utf8'))
                for index,original in enumerate(packet['rows']):
                    if original.get('split')!='fit' or original.get('original_working_sigma_px') is None:
                        continue
                    row,keep,offset=layouts[name][original['id']]
                    if len(keep)!=len(original['uv']):
                        raise ValueError('Declared packet sampling needs its exact full active anchor layout')
                    pairs=np.arange(offset,offset+2*len(keep)).reshape(-1,2)
                    if used&{(name,int(i)) for i in pairs.ravel()}:
                        raise ValueError('Declared packet sampling overlaps a separately integrated native row')
                    working=np.broadcast_to(np.asarray(original['original_working_sigma_px'],float),(len(keep),))
                    stored=np.broadcast_to(np.asarray(original['sigma_px'],float),(len(keep),))
                    weights=np.asarray(original.get('source_integration_weights'),float)
                    paths=original.get('source_visible_paths')
                    expected=source_support_integration_weights(original['uv'],paths)
                    if (not np.isfinite(working).all() or np.any(working<=0.)
                            or weights.shape!=(len(keep),) or np.any(weights<=0.)
                            or not np.allclose(weights,expected,rtol=1e-12,atol=1e-12)
                            or not np.allclose(stored,working/np.sqrt(weights),rtol=1e-12,atol=1e-12)
                            or not np.array_equal(np.asarray(row['uv']),np.asarray(original['uv']))
                            or not np.array_equal(scales[name][pairs.ravel()],np.repeat(stored,2))
                            or not np.isclose(weights.sum(),1.,rtol=1e-12,atol=1e-12)):
                        raise ValueError('Declared native packet working sigma or exact integration support changed')
                    self._payloads[family][pairs.ravel()]=np.repeat(weights,2)
                    normalizations[family][pairs.ravel()]=np.repeat(stored/working,2)
                    packet_sampling.append(dict(family=name,packet_row_index=index,id=original['id'],
                        actual_pts=int(row['actual_pts']),curve=original['curve'],points=len(keep),
                        source_row_reference=dict(source_refs[name],json_pointer=f'/rows/{index}'),
                        working_sigma_px=working.tolist(),stored_sigma_px=stored.tolist(),
                        integration_weight_sum=float(weights.sum()),point_integration_weights=weights.tolist(),
                        quadratic_information_budget_unchanged=True))
        self._payloads = {family: SourceLossWeights(w,normalizations[family]
            if np.any(normalizations[family]!=1.) else None) for family,w in self._payloads.items()}
        self.native_base_curve_integration = dict(schema=NATIVE_BASE_SCHEMA + '.runtime',
            policy_reference=dict(path=str(path), sha256=file_sha256(path)), source_packets=source_refs,
            groups=plan['groups'], apply_to_base_native_blocks_only=True,
            direct_manual_centre_and_all_human_truth_supports_unchanged=True,
            original_R_J_sigma_quadratic_roles_and_scalar_slots_unchanged=True,
            integration_precomputed_once=True, fresh_source_J_and_profiles_required=True,
            statistical_independence_claimed=False, accuracy_accepted=False)
        if policy.get('separate_declared_packet_sampling_normalization') is True:
            self.native_base_curve_integration.update(declared_packet_sampling_normalizations=packet_sampling,
                packet_stored_sigma_budget_removed_only_with_explicit_policy=True,
                quadratic_packet_budget_unchanged=True)
        import rinkspace.source_sampling as sampling
        self.native_base_curve_integration['implementation_sources'] = [
            dict(path=str(p.resolve()), sha256=file_sha256(p)) for p in
            (Path(__file__), Path(__file__).with_name('source_loss.py'), Path(sampling.__file__))]
        self.contract = dict(base.contract, native_base_curve_integration=self.native_base_curve_integration)

    def __getattr__(self, name): return getattr(self.base, name)

    def apply_block(self, block):
        payload = self._payloads.get(block.row_ledger.get('family'))
        if payload is None: return block
        if (len(block.r) != len(payload.integration_weights) or block.loss_weights is not None
                or 'native_base_curve_integration' in block.row_ledger):
            raise ValueError('Native base integration scalar layout changed or budget applied twice')
        from z190_streaming_source_normal import SourceRowBlock
        return SourceRowBlock(block.r, block.J, block.quadratic, block.scales,
            dict(block.row_ledger, native_base_curve_integration=self.native_base_curve_integration['policy_reference']), payload)

    def blocks(self, q, jacobian=True):
        found = set()
        for block in self.base.blocks(q, jacobian):
            family = block.row_ledger.get('family')
            if family in self._payloads:
                if family in found: raise ValueError('Native integration base block duplicated')
                found.add(family)
            yield self.apply_block(block)
        if found != set(self._payloads): raise ValueError('Native integration base block missing')


def wrap_native_base_curve_integration(base, policy_path=None):
    return base if policy_path is None else SourceWithNativeBaseCurveIntegration(base, policy_path)
