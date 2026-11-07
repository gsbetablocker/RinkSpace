"""Read-only exact skate geometry on every saved physical sample.

This is a receipt for the saved reference-plane geometry, not a measurement of
real ice penetration, separation or contact. It does not evaluate a Jacobian,
run an optimizer, classify contact or introduce an accuracy/acceptance gate.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json

import numpy as np

from .absolute_motion_state import curve_from_packet
from .body_clock import source_seconds, time_base
from .sequence_skate_state import shoe_curve_from_packet
from .skate_checkpoint import IDENTITY_FIELDS, validate_checkpoint_identity
from .skate_physics import ALLOWANCE, batch_features, fk_state


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def array_sha256(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode('utf8'))
    h.update(a.tobytes())
    return h.hexdigest()


def distribution(value):
    a = np.asarray(value, float)
    finite = a[np.isfinite(a)]
    out = dict(total=int(a.size), finite=int(finite.size), nan=int(np.isnan(a).sum()),
               positive_infinity=int(np.isposinf(a).sum()), negative_infinity=int(np.isneginf(a).sum()),
               unknown=int((~np.isfinite(a)).sum()))
    out.update({key: None for key in ('minimum', 'median', 'p95', 'maximum')})
    if finite.size:
        out.update(minimum=float(finite.min()), median=float(np.median(finite)),
                   p95=float(np.percentile(finite, 95)), maximum=float(finite.max()))
    return out


def minimum_features(height, indices):
    """Keep an unknown minimum unknown, including empty feature membership."""
    h = np.asarray(height)
    ids = np.asarray(indices, int)
    result = np.full(h.shape[:2], np.nan)
    witness = np.full(h.shape[:2], -1, dtype=np.int32)
    if ids.size:
        selected = h[..., ids]
        known = np.isfinite(selected).all(axis=-1)
        best = np.argmin(np.where(np.isfinite(selected), selected, np.inf), axis=-1)
        values = np.take_along_axis(selected, best[..., None], axis=-1)[..., 0]
        result[known] = values[known]
        witness[known] = ids[best[known]]
    return result, witness


def nearest_source(native_seconds, sample_seconds):
    right = np.clip(np.searchsorted(native_seconds, sample_seconds), 0, len(native_seconds) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(native_seconds[left] - sample_seconds) <=
                    abs(native_seconds[right] - sample_seconds), left, right)


def feature_catalog(binding, features):
    """Use actual typed metadata and explicit support IDs, never ID spelling."""
    packet = binding.packet()
    template = packet['template']
    ids = list(features['ids'])
    if len(ids) != len(set(ids)):
        raise ValueError('Exact geometric feature IDs must be unique')
    catalog = []
    primitives = {p['id']: dict(p, kind='exact_arc') for p in template['contact_arcs']}
    for group in template['contact_point_groups']:
        points = np.asarray(group['points'])
        primitives[group['id']] = dict(kind='solid_or_points_vertex_minimum',
            point_count=len(points), points_solver_sha256=array_sha256(points))
    for i, name in enumerate(ids):
        primitive = primitives.get(name)
        if primitive is None:
            if name != 'upper_shell_guard':
                raise ValueError('Unregistered exact feature: ' + name)
            primitive = dict(kind='existing_deformed_upper_vertex_minimum',
                neutral_points_solver_sha256=array_sha256(packet['upper']),
                point_count=len(packet['upper']))
        semantic = features['semantics'][i]
        catalog.append(dict(index=i, id=name, semantic=semantic,
            support_eligible=bool(features['eligible'][i]),
            existing_allowance_m=ALLOWANCE.get(semantic), primitive=primitive))
    groups = {}
    for mode, members in template['support_feature_ids'].items():
        if len(members) != len(set(members)) or any(name not in ids for name in members):
            raise ValueError('Invalid actual template support IDs: ' + mode)
        group = [ids.index(name) for name in members]
        if any(not features['eligible'][i] for i in group):
            raise ValueError('Template support group contains an ineligible guard: ' + mode)
        groups[mode] = group
    if 'glide' not in groups:
        raise ValueError('Actual template has no explicit glide support group')
    runner = [i for i, row in enumerate(catalog)
              if row['semantic'] == 'runner' and row['support_eligible']]
    if any(catalog[i]['semantic'] != 'runner' for i in groups['glide']):
        raise ValueError('Actual glide support group contains a non-runner feature')
    return catalog, groups, runner


def saved_structure_receipts(parameter):
    """Quote existing cavity/collar diagnostics without skinning again.

    Sidecar location is reported as provenance, not checksum authentication of
    an absent original parameter hash. Both original JSON files remain intact.
    """
    fields = ('feasible', 'max_violation_m', 'rows_over_tolerance', 'tested_rows',
              'worst_row', 'worst_label', 'scope', 'collar_unknown_sections', 'ice_is_delivery_gate')
    stage = parameter.parent.parent
    records = []
    for path, key in ((stage / 'final_metrics.json', 'structure'),
                      (parameter.with_suffix('.json'), None)):
        if not path.is_file():
            records.append(dict(path=str(path), status='missing', unknown=True))
            continue
        document = json.loads(path.read_text(encoding='utf8'))
        value = document.get(key, {}) if key else document
        row = dict(path=str(path), sha256=file_sha256(path), status='existing_saved_diagnostic',
                   fields={name: value[name] for name in fields if name in value},
                   unknown_fields=[name for name in fields if name not in value], recomputed=False)
        if not key:
            checkpoint = document.get('candidate_checkpoint')
            row['recorded_candidate_checkpoint'] = checkpoint
            row['recorded_path_matches_requested_parameter'] = (
                checkpoint is not None and Path(checkpoint).resolve() == parameter)
        records.append(row)
    return dict(records=records,
        interpretation='Original foot cavity/collar row diagnostics only; not blade contact or ice truth.',
        provenance_limit='Existing run sidecars do not record a parameter content hash; adjacency is not independent authentication.',
        no_new_skinning_or_constraints=True)


def generate(project, prepared, parameter, output, *, batch_samples=256):
    project, prepared, parameter, output = map(lambda p: Path(p).resolve(),
                                              (project, prepared, parameter, output))
    from rinkspace.storage import AssetStore
    from .trajectory_backend import load_reference
    data_root = Path(AssetStore(project).data).resolve()
    if output == data_root or not output.is_relative_to(data_root):
        raise ValueError('Diagnostic output must be a new directory under the configured Data root')
    if output.exists():
        raise FileExistsError('Preserve existing diagnostic output: ' + str(output))
    if type(batch_samples) is not int or batch_samples < 1:
        raise ValueError('Positive integer batch_samples required')
    paths = [prepared / name for name in ('input.npz', 'observations.npz', 'initial.npz',
             'fixed_binding.npz', 'camera_context.json', 'model_identity.json', 'source_evidence.json')]
    paths.append(parameter)
    hashes = {str(path): file_sha256(path) for path in paths}
    # The requested loader consumes a prepared seed. Do not fall back to its
    # broader initial seed-construction path when that current file is absent.
    if not (prepared / 'absolute_seed_state.npz').is_file():
        raise FileNotFoundError('Read-only receipt requires the prepared absolute_seed_state.npz')
    hashes[str(prepared / 'absolute_seed_state.npz')] = file_sha256(prepared / 'absolute_seed_state.npz')
    with np.load(parameter, allow_pickle=False) as saved:
        validate_checkpoint_identity(saved)
        for field, name in (('prepared_input_sha256', 'input.npz'),
                            ('prepared_observations_sha256', 'observations.npz')):
            if str(saved[field]) != hashes[str(prepared / name)]:
                raise ValueError('Saved final parameter differs from actual prepared identity: ' + field)
        times = np.asarray(saved['physics_times'], float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Saved complete physics_times must be finite and strictly increasing')
        ref, camera, info, document, seed_report = load_reference(project, prepared)
        validate_checkpoint_identity(saved, ref.fixed_checkpoint_identity)
        ref.absolute_curve = curve_from_packet(saved, seed=False)
        ref.relative_skate_curve = shoe_curve_from_packet(saved, seed=False)
        if ref.relative_skate_curve is None:
            raise ValueError('Final candidate has no saved relative-shoe curve')
        for curve in (ref.absolute_curve, ref.relative_skate_curve):
            if times[0] < curve.knots[3] or times[-1] > curve.knots[-4]:
                raise ValueError('Saved physics samples extend beyond an actual saved curve domain')
        ref.binding.x = np.asarray(saved['x'][:17], float).copy()
        if ref.binding.x.shape != (17,) or not np.isfinite(ref.binding.x).all():
            raise ValueError('Expected 17 finite saved shared skate parameters')
        ref.binding._parameter_key = ref.binding._packet = None
        ref.binding.reference = ref
        pts = np.asarray(ref.p['pts'])
        native_seconds = source_seconds(ref.p)
        reference_times = np.asarray(ref.times, float)
        if pts.ndim != 1 or pts.dtype.kind not in 'iu' or len(pts) < 2 or np.any(np.diff(pts) <= 0):
            raise ValueError('Ordered actual source PTS required')
        if reference_times.shape != native_seconds.shape or not np.isfinite(reference_times).all():
            raise ValueError('Reference and source clock shapes differ')
        origin = float(native_seconds[0] - reference_times[0])
        np.testing.assert_allclose(reference_times + origin, native_seconds, rtol=0, atol=1e-10,
                                   err_msg='Reference clock does not align with actual source PTS')
        with np.load(prepared / 'observations.npz', allow_pickle=False) as obs:
            if 'pts' in obs:
                np.testing.assert_array_equal(obs['pts'], pts)
        source_times = times + origin
        nearest = nearest_source(native_seconds, source_times)
        # This tolerance checks floating-point clock equality only. It is not
        # a physical, source-accuracy or contact threshold.
        is_native_mid = np.isclose(source_times, native_seconds[nearest], rtol=0, atol=1e-10)
        packet = ref.binding.packet()
        ratio = float(ref.binding.ratio)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('A positive actual binding unit ratio is required')
        output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(dict(event='programme_ice_receipt_started', output=str(output),
                              source_exposures=len(pts), physical_samples=len(times))), flush=True)
        arrays = dict(physics_time_seconds=times, source_absolute_seconds=source_times,
            nearest_source_pts=pts[nearest], nearest_source_seconds=native_seconds[nearest],
            sample_minus_nearest_source_seconds=source_times-native_seconds[nearest],
            coincides_with_native_source_midpoint=is_native_mid)
        selections = {}
        catalog = groups = runner = None
        for first in range(0, len(times), batch_samples):
            last = min(first + batch_samples, len(times))
            t = times[first:last]
            state = fk_state(ref, ref.at(t), np.zeros((len(t), 69)))
            features = batch_features(ref.binding, state['R'], state['O'], state)
            if catalog is None:
                catalog, groups, runner = feature_catalog(ref.binding, features)
                indices = np.arange(len(catalog))
                selections = dict(raw=indices, clearance=indices,
                                  eligible_runner=np.asarray(runner, int),
                                  glide=np.asarray(groups['glide'], int))
                arrays['feature_height_m'] = np.full((len(times), 2, len(catalog)), np.nan)
                arrays['upper_guard_vertex_index'] = np.full((len(times), 2), -1, np.int32)
                for name in selections:
                    arrays[name+'_minimum_m'] = np.full((len(times), 2), np.nan)
                    arrays[name+'_feature_index'] = np.full((len(times), 2), -1, np.int32)
                    arrays[name+'_witness_local_solver'] = np.full((len(times), 2, 3), np.nan)
            if list(features['ids']) != [row['id'] for row in catalog] or list(features['semantics']) != [row['semantic'] for row in catalog]:
                raise ValueError('Actual geometry feature identity changed between batches')
            height = np.asarray(features['height'])
            arrays['feature_height_m'][first:last] = height
            if features['upper_index'] is not None:
                arrays['upper_guard_vertex_index'][first:last] = features['upper_index']
            allowances = np.asarray([ALLOWANCE.get(row['semantic'], np.nan) for row in catalog])
            for name, indices in selections.items():
                value = height + allowances if name == 'clearance' else height
                minimum, witness = minimum_features(value, indices)
                arrays[name+'_minimum_m'][first:last] = minimum
                arrays[name+'_feature_index'][first:last] = witness
                known = witness >= 0
                i, f = np.nonzero(known)
                arrays[name+'_witness_local_solver'][first+i, f] = features['local_point'][i, f, witness[i, f]]
            print(json.dumps(dict(event='programme_ice_receipt_batch', completed=last,
                                  total=len(times))), flush=True)

        def extreme(name, maximum=False):
            values = arrays[name+'_minimum_m']
            finite = np.isfinite(values)
            if not finite.any():
                return dict(status='unknown', unknown=int(values.size))
            choose = np.argmax if maximum else np.argmin
            index = int(choose(np.where(finite, values, -np.inf if maximum else np.inf)))
            i, f = np.unravel_index(index, values.shape)
            j = int(arrays[name+'_feature_index'][i, f])
            return dict(physical_sample_index=int(i), physics_time_seconds=float(times[i]),
                source_absolute_seconds=float(source_times[i]), foot=int(f), foot_name=('left', 'right')[f],
                feature_index=j, feature_id=catalog[j]['id'], value_m=float(values[i, f]),
                witness_local_solver=arrays[name+'_witness_local_solver'][i, f].tolist(),
                nearest_source_pts=int(pts[nearest[i]]), nearest_source_seconds=float(native_seconds[nearest[i]]),
                sample_minus_nearest_source_seconds=float(source_times[i]-native_seconds[nearest[i]]),
                coincides_with_native_source_midpoint=bool(is_native_mid[i]),
                sample_type='native_source_midpoint' if is_native_mid[i] else 'interpolated_physics_sample',
                native_exposure_asserted_at_interpolated_time=False)

        metrics = {name: dict(all_feet=distribution(arrays[name+'_minimum_m']),
            by_foot={foot: distribution(arrays[name+'_minimum_m'][:, f]) for f, foot in enumerate(('left', 'right'))},
            minimum_sample=extreme(name), maximum_sample=extreme(name, maximum=True)) for name in selections}
        fixed_identity = {key: json.loads(str(saved[key])) if key.endswith('_json') else
                          np.asarray(saved[key]).tolist() for key in IDENTITY_FIELDS}
        policy = json.loads(str(saved['ice_soft_policy_json'])) if 'ice_soft_policy_json' in saved else None
        arrays.update(feature_ids=np.asarray([row['id'] for row in catalog]),
            feature_semantics=np.asarray([row['semantic'] for row in catalog]),
            feature_support_eligible=np.asarray([row['support_eligible'] for row in catalog]),
            existing_feature_allowance_m=allowances, source_origin_seconds=np.array(origin),
            source_time_base=np.asarray(time_base(ref.p)), actual_source_pts=pts,
            body_length_ratio=np.array(ratio))
        np.savez_compressed(output / 'physics_samples.npz', **arrays)
        changed = [str(path) for path, checksum in hashes.items() if file_sha256(path) != checksum]
        if changed:
            raise ValueError('Read inputs changed while evaluating: ' + ', '.join(changed))
        report = dict(schema='rinkspace.programme-exact-reference-plane-receipt.v1',
            parameter=str(parameter), prepared=str(prepared), input_sha256=hashes,
            samples=dict(source_exposures=len(pts), complete_saved_physics_times=len(times),
                native_source_midpoint_samples=int(is_native_mid.sum()), interpolated_samples=int((~is_native_mid).sum()),
                outside_native_midpoint_range=int(((source_times<native_seconds[0])|(source_times>native_seconds[-1])).sum()),
                source_seconds_range=native_seconds[[0,-1]].tolist(),
                physics_absolute_seconds_range=source_times[[0,-1]].tolist(), source_origin_seconds=origin,
                reference_plus_origin_matches_source_seconds=True,
                nearest_source_pts_meaning='Navigation only; interpolated samples are not invented native exposures.'),
            geometry=dict(template_version=str(saved['template_version']), binding_version=str(saved['binding_version']),
                fixed_checkpoint_identity=fixed_identity, binding_x_sha256=array_sha256(ref.binding.x),
                body_length_ratio=ratio, units=dict(reported_heights='m', FK_and_local_witness='historical solver units',
                    conversion='batch_features: height_m = (O_z + (R @ local_point)_z) * binding.ratio',
                    template='SkateBinding.packet converts the metric kernel to solver units; retain this exact existing conversion.'),
                actual_parameters_solver=asdict(packet['parameters']),
                support_feature_ids=packet['template']['support_feature_ids'],
                eligible_runner_feature_ids=[catalog[i]['id'] for i in runner], feature_catalog=catalog,
                feature_height_unknown_counts=[dict(id=row['id'], counts=distribution(arrays['feature_height_m'][..., i]))
                                               for i, row in enumerate(catalog)]),
            ice_plane_known=bool(ref.ice_plane_known), existing_ice_soft_policy=policy,
            clearance_definition='minimum over actual features of raw height_m + existing semantic ALLOWANCE; no additional deadzone applied',
            metrics=metrics, existing_foot_cavity_collar=saved_structure_receipts(parameter),
            output=dict(samples=str(output/'physics_samples.npz'), samples_sha256=file_sha256(output/'physics_samples.npz')),
            interpretation_zh='全采样高度是当前已保存模板几何相对参考 Z=0 的数值，不是实测穿冰、离冰或真实接触。这里只报告 saved.physics_times 样本极值，不是连续曲线所有时刻的解析界。',
            evidence_limits_zh='nearest_source_pts 仅供定位原片；插值物理样本不伪造为原片曝光。鞋或踝的源点不改称刀刃/接触，不从网络静止概率推造接冰。',
            unknown_policy='NaN/nonfinite or empty membership remains unknown, never replaced with zero; minima require every selected feature finite.',
            no_acceptance_gate=True, no_new_unified_cm_standard=True, no_contact_classification=True,
            no_jacobian_or_optimizer=True, camera_scale_template_and_saved_parameters_modified=False,
            evaluation_batch_samples=batch_samples)
        (output / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'prepared', 'parameter', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-samples', type=int, default=256)
    args = parser.parse_args()
    report = generate(args.project, args.prepared, args.parameter, args.output,
                      batch_samples=args.batch_samples)
    print(json.dumps(dict(receipt=str(args.output.resolve()/'receipt.json'),
                          physical_samples=report['samples']['complete_saved_physics_times']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
