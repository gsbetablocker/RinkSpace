"""Compare saved whole-programme candidates and scan ledgers without evaluating a model."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.body_clock import time_base
from calibration.program_tracking.body_tracking.programme_com import acceleration_operator
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from report_programme_com_motion import LSP_WINDOW_S

SCOPE = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json'
OPENING_PTS = (283136, 293120)
HAND_WAIST_PTS = (283136, 295680, 299520)
HAND_WAIST_FIELDS = ('hand_waist_source_pts', 'hand_waist_patch_distances_m')
JOINTS = ('left_knee', 'right_knee', 'left_elbow', 'right_elbow')
POSITIONS = {'P': 'P_world_mid', 'G': 'G_world_mid', 'COM': 'com_world_mid'}
SAMPLE_KINDS = ('native', 'interpolated_midpoint', 'interpolated_adaptive')
TOPOLOGY_FIELDS = ('whole_skin_faces', 'retained_skin_faces', 'skin_label_seams_removed',
                   'faces', 'vertices', 'topology_filter', 'signed_branch')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def missing(reason):
    return dict(available=False, reason=reason)


def checked(reader, context):
    try:
        return reader()
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        return missing(f'{context}: {exc}')


def optional(path, reader):
    if path is None or not path.is_file():
        return missing(f'Saved input unavailable: {path or "no path supplied"}')
    return checked(lambda: reader(path), path)


def stats(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Expected finite values on every selected sample')
    q = np.percentile(values, [0, 50, 95, 100])
    return dict(samples=values.size, min=float(q[0]), p50=float(q[1]), p95=float(q[2]), max=float(q[3]))


def check_clock(packet, pts, tb):
    available = [key for key in ('source_pts', 'pts') if key in packet]
    if not available or any(not np.array_equal(packet[key], pts) for key in available) or time_base(packet) != tb:
        raise ValueError('Result does not match the registered complete source PTS/time base')


def read_hand_waist_distances(saved, expected_pts):
    pts, distances = (np.asarray(saved[name]) for name in HAND_WAIST_FIELDS)
    if pts.dtype.kind not in 'iu' or not np.array_equal(pts, expected_pts):
        raise ValueError('Hand-waist source PTS differ from the registered window')
    if (distances.shape != (len(pts), 3) or distances.dtype.kind not in 'fiu'
            or not np.isfinite(distances).all() or np.any(distances < 0)):
        raise ValueError('Expected three finite nonnegative hand-waist patch distances per source PTS')
    return distances


def opening_report(result, pts, positions, diagnostics, metrics):
    ids = np.flatnonzero((pts >= OPENING_PTS[0]) & (pts <= OPENING_PTS[1]))
    out = dict(source_pts_range=list(OPENING_PTS), source_frames=len(ids))
    root = positions.get('P')
    out['quiet_P_range_m'] = (float(np.linalg.norm(root[ids]-root[ids[0]], axis=1).max())
                              if root is not None else missing('P_world_mid is absent'))
    out['sampled_runner_edge_min_z_m'] = missing('skate_edge_world_mid is absent')
    if 'skate_edge_world_mid' in result:
        edge = result['skate_edge_world_mid'][ids, ..., 2]
        out['sampled_runner_edge_min_z_m'] = stats(edge.reshape(len(ids), -1).min(axis=1))
    saved = diagnostics.get('opening_source_standing') or {}
    if np.array_equal(saved.get('source_pts'), pts[ids]):
        out['exact_minimum_eligible_steel_height_m'] = stats(saved['minimum_steel_height_m'])
        out['standing_contact_mode'] = saved.get('standing_contact_mode', 'legacy_unspecified')
        if 'steel_height_by_foot_m' in saved:
            each = np.asarray(saved['steel_height_by_foot_m'], float)
            if each.shape != (len(ids), 2) or not np.isfinite(each).all():
                raise ValueError('Per-foot opening steel heights differ from the source clock')
            out['exact_eligible_steel_height_by_foot_m'] = [stats(each[:, foot]) for foot in (0, 1)]
            out['both_contact_feet_max_absolute_height_m'] = (float(np.abs(each).max())
                if saved.get('standing_contact_mode') == 'both_feet' else None)
        else:
            out['exact_eligible_steel_height_by_foot_m'] = missing('Old saved factor did not export both feet separately')
        if 'hand_waist_max_distance_m' in saved:
            out['hand_waist_max_distance_m'] = saved['hand_waist_max_distance_m']
    elif 'opening_scope' in metrics:
        scope = metrics['opening_scope']
        if (scope.get('standing_rows') == len(ids)
                and (scope.get('source_pts_first'), scope.get('source_pts_last')) == OPENING_PTS
                and scope.get('exact_quiet_source_scope')):
            height = scope['minimum_steel_height_m']
            out['exact_minimum_eligible_steel_height_m'] = dict(
                samples=len(ids), min=height['min'], p50=height['median'], max=height['max'],
            )
            if 'hand_waist_max_distance_m' in scope:
                out['hand_waist_max_distance_m'] = scope['hand_waist_max_distance_m']
    out.setdefault('exact_minimum_eligible_steel_height_m',
                   missing('No saved exact standing envelope on these source frames'))
    out.setdefault('hand_waist_max_distance_m', missing('No saved hand-waist diagnostic'))
    start, release_start, end = HAND_WAIST_PTS
    hand_pts = pts[(pts >= start) & (pts <= end)]
    held = hand_pts <= release_start
    distances = missing('No saved per-PTS hand-waist distances for phase statistics')
    if any(name in saved for name in HAND_WAIST_FIELDS):
        distances = checked(lambda: read_hand_waist_distances(saved, hand_pts), 'Hand-waist distances')

    def phase_statistics(mask):
        if isinstance(distances, dict):
            return distances
        selected = distances[mask]
        summary = stats(selected)
        peak = int(np.argmax(selected.max(axis=1)))
        return dict(available=True, patch_distances_m=summary, max_distance_m=summary['max'],
                    max_distance_pts=int(hand_pts[mask][peak]))

    out['hand_waist_distance_by_phase'] = dict(
        full_window=dict(
            source_pts_range=[start, end],
            max_distance_field='hand_waist_max_distance_m',
            includes_release=True,
        ),
        held=dict(
            start_pts_inclusive=start, end_pts_inclusive=release_start,
            source_frames=int(np.count_nonzero(held)),
            **phase_statistics(held),
        ),
        release=dict(
            start_pts_exclusive=release_start, end_pts_inclusive=end,
            source_frames=int(np.count_nonzero(~held)),
            **phase_statistics(~held),
        ),
    )
    return out


def offset_accelerations(result, motion_path, pts, tb, positions, operator):
    root, com = positions['P'], positions['COM']
    fields = ('com_segment_centres_world_m', 'com_segment_body_joint_ids', 'com_segment_mass_fractions')
    if all(name in result for name in fields):
        centres, joint_ids, weights = (result[name] for name in fields)
        count = len(joint_ids)
        if (centres.shape != (len(pts), count+2, 3) or weights.shape != (count+2,)
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError('Unexpected body/shoe mass proxy layout')
        np.testing.assert_allclose(np.einsum('s,nsc->nc', weights, centres), com, atol=1e-8, rtol=0)
        offsets = centres-root[:, None]
        parts = {
            name: operator @ np.einsum('s,nsc->nc', weights[sl], offsets[:, sl])
            for name, sl in (('body', slice(None, count)), ('shoe', slice(count, None)))
        }
        return parts, dict(source='saved segment centres and mass fractions')
    if motion_path is None or not motion_path.is_file():
        return {}, missing('Saved body/shoe segment centres or aligned COM motion NPZ are unavailable')
    with np.load(motion_path, allow_pickle=False) as motion:
        check_clock(motion, pts, tb)
        for name in ('P', 'COM'):
            np.testing.assert_allclose(motion[POSITIONS[name]], positions[name], atol=1e-8, rtol=0)
        parts = {
            kind: motion[f'{kind}_mass_offset_from_P_acceleration_world_mps2'][1:-1]
            for kind in ('body', 'shoe')
        }
    return parts, dict(source=str(motion_path.resolve()))


def acceleration_report(result, motion_path, pts, tb, positions):
    if not {'P', 'COM'} <= positions.keys():
        return missing('P and COM positions are required for the same-source decomposition')
    seconds = pts*tb[0]/tb[1]
    relative = (pts-pts[0])*tb[0]/tb[1]
    operator = acceleration_operator(relative)
    acc = {
        'COM': operator@positions['COM'],
        'P': operator@positions['P'],
        'COM_minus_P': operator@(positions['COM']-positions['P']),
    }
    exported = 'com_acceleration_world_mps2' in result
    if exported:
        np.testing.assert_allclose(result['com_acceleration_world_mps2'][1:-1], acc['COM'], atol=1e-8, rtol=0)
    try:
        parts, provenance = offset_accelerations(result, motion_path, pts, tb, positions, operator)
        if parts:
            np.testing.assert_allclose(parts['body']+parts['shoe'], acc['COM_minus_P'], atol=1e-8, rtol=0)
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parts, provenance = {}, missing(f'Body/shoe decomposition rejected: {exc}')
    acc.update(parts)
    magnitude = np.linalg.norm(acc['COM'], axis=1)

    def scope(indices):
        i = indices[np.argmax(magnitude[indices])]
        unit = acc['COM'][i]/max(magnitude[i], np.finfo(float).tiny)
        return dict(
            acceleration_magnitude_mps2=stats(magnitude[indices]),
            peak=dict(
                source_pts=int(pts[i+1]), source_seconds=float(seconds[i+1]),
                acceleration_xyz_mps2={name: v[i].tolist() for name, v in acc.items()},
                contributions_along_COM_mps2={name: float(v[i]@unit) for name, v in acc.items() if name != 'COM'},
            ),
        )

    lsp = np.flatnonzero((seconds[1:-1] >= LSP_WINDOW_S[0]) & (seconds[1:-1] <= LSP_WINDOW_S[1]))
    return dict(
        derivative='Existing three-sample acceleration operator on original source time, including cut-adjacent samples',
        matches_saved_backend_acceleration=exported,
        body_and_shoe_decomposition=provenance,
        whole_programme_interior=scope(np.arange(len(magnitude))),
        LSp_source_seconds=list(LSP_WINDOW_S), LSp=scope(lsp),
    )


def anatomy_report(path, pts, result):
    with np.load(path, allow_pickle=False) as p:
        if not np.array_equal(p['source_pts'], pts):
            raise ValueError('Anatomical frames use a different source clock')
        for a, b in (('local_R', 'local_R_mid'), ('global_root_R', 'Q_world_mid')):
            np.testing.assert_allclose(p[a], result[b], atol=1e-8, rtol=0)
        names = list(p['joint_names'])
        values = p['normalized_violation']
    if values.shape != (len(pts), len(names), 3) or not np.isfinite(values).all():
        raise ValueError('Incomplete normalized anatomical chart values')
    joints = {}
    for name in JOINTS:
        if name not in names:
            joints[name] = missing('No saved anatomical chart for this joint')
            continue
        v = values[:, names.index(name)]
        row, component = np.unravel_index(np.argmax(v), v.shape)
        norm = np.linalg.norm(v, axis=1)
        peak = np.argmax(norm)
        joints[name] = dict(
            peak_dimensionless_component=float(v[row, component]), component=int(component),
            component_peak_pts=int(pts[row]) if v[row, component] > 0 else None,
            peak_dimensionless_vector_norm=float(norm[peak]),
            vector_norm_peak_pts=int(pts[peak]) if norm[peak] > 0 else None,
            positive_samples=int(np.any(v > 0, axis=1).sum()),
        )
    return dict(source=str(path.resolve()), clinical_angle_claim=False, joints=joints)


def grasp_report(diagnostics, pts):
    saved = diagnostics.get('source_supported_grasp')
    if not saved:
        return missing('No saved source_supported_grasp hand-to-shoe distances')
    episodes = []
    for e in saved['episodes']:
        stamps, distances = np.asarray(e['source_pts']), np.asarray(e['distance_m'])
        if stamps.shape != distances.shape or not np.isin(stamps, pts).all():
            raise ValueError('Grasp distance rows do not match their source samples')
        episodes.append(dict(
            hand=e['hand'], foot=e['foot'], source_seconds=e['source_seconds'],
            distance_m=stats(distances), peak_distance_pts=int(stamps[np.argmax(distances)]),
        ))
    return dict(geometry='Source-selected palm patch to nearest held-shoe triangle, conditional model distance',
                episodes=episodes)


def scan_receipt(path, parameter, n, samples, pairs):
    receipt = read_json(path.with_suffix('.json'))
    if (receipt['native_exposures'] != n or receipt.get('total_samples', receipt.get('samples')) != samples
            or receipt['total_crossing_pairs'] != pairs):
        raise ValueError('Saved scan and count receipt disagree')
    expected = receipt.get('parameter_sha256')
    binding = 'Explicit input; saved parameter identity unavailable'
    if expected and parameter is not None:
        if digest(parameter) != expected:
            raise ValueError('Surface scan belongs to a different saved parameter')
        binding = 'Saved stage parameter hash verified'
    topology = receipt
    if 'base_scan' in receipt:
        base = Path(receipt['base_scan'])
        if not base.is_file() or digest(base) != receipt['base_scan_sha256']:
            raise ValueError('Adaptive scan base identity is missing or changed')
        topology = read_json(base.with_suffix('.json'))
    return receipt, binding, {key: topology.get(key) for key in TOPOLOGY_FIELDS}


def read_scan(path, pts, tb, parameter):
    with np.load(path, allow_pickle=False) as s:
        if not np.array_equal(s['source_pts'], pts):
            raise ValueError('Surface scan uses a different source PTS clock')
        seconds = s['source_seconds']
        adaptive = s['adaptive_sample'] if 'adaptive_sample' in s else np.zeros(len(seconds), dtype=bool)
        parts = s['parts'].tolist()
        ledger = s['pair_counts']
        counts, grip, uncertain = (s[key] for key in (
            'crossing_counts', 'grip_crossing_counts', 'topology_uncertain_candidates',
        ))
    if seconds.ndim != 1 or not np.isfinite(seconds).all():
        raise ValueError('Invalid scan sample times')
    keys = np.rint(seconds*1e9).astype(np.int64)
    native = pts*tb[0]/tb[1]
    base = np.sort(np.r_[native, .5*(native[:-1]+native[1:])])
    if adaptive.shape != keys.shape or adaptive.dtype.kind != 'b' or np.any(np.diff(keys) <= 0):
        raise ValueError('Scan samples must have distinct time keys and declared kinds')
    np.testing.assert_array_equal(keys[~adaptive], np.rint(base*1e9).astype(np.int64))
    kinds = np.where(adaptive, 2, 1)
    kinds[np.flatnonzero(~adaptive)[::2]] = 0
    if any(v.shape != keys.shape or v.dtype.kind not in 'iu' or np.any(v < 0) for v in (counts, grip, uncertain)):
        raise ValueError('Missing or invalid per-sample scan counts')
    if (len(set(parts)) != len(parts) or ledger.ndim != 2 or ledger.shape[1] != 4
            or ledger.dtype.kind not in 'iu' or np.any(ledger < 0)
            or np.any(ledger[:, 0] >= len(keys)) or np.any(ledger[:, 1:3] >= len(parts))
            or np.any(ledger[:, 1] > ledger[:, 2]) or np.any(ledger[:, 3] == 0)
            or len(np.unique(ledger[:, :3], axis=0)) != len(ledger)):
        raise ValueError('Invalid semantic-pair sample ledger')
    total = np.zeros_like(counts, dtype=np.int64)
    np.add.at(total, ledger[:, 0], ledger[:, 3])
    np.testing.assert_array_equal(total, counts)
    receipt, binding, topology = scan_receipt(path, parameter, len(pts), len(keys), int(counts.sum()))
    events = {}
    for first, second in np.unique(ledger[:, 1:3], axis=0):
        select = (ledger[:, 1] == first) & (ledger[:, 2] == second)
        pair = tuple(sorted((parts[first], parts[second])))
        events[pair] = ledger[select, 0]
    summary = dict(
        source=str(path.resolve()), parameter_binding=binding,
        parameter_sha256=receipt.get('parameter_sha256'), source_document_sha256=receipt.get('source_document_sha256'),
        topology=topology, samples_by_kind=kind_counts(kinds),
        retained_positive_samples_by_kind=kind_counts(kinds[counts > 0]),
        retained_triangle_pair_samples=int(counts.sum()),
        conditional_grip_triangle_pair_samples=int(grip.sum()),
        head_topology_uncertain_candidates=int(uncertain.sum()),
    )
    return dict(keys=keys, kinds=kinds, events=events, counts=counts, parts=sorted(parts), summary=summary)


def kind_counts(kinds):
    return {name: int(np.count_nonzero(kinds == k)) for k, name in enumerate(SAMPLE_KINDS)}


def event_ranges(mask, keys):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    return [[int(keys[run[0]]), int(keys[run[-1]]), len(run)] for run in runs]


def compare_scans(before, after):
    if 'summary' not in before or 'summary' not in after:
        return missing('Two valid saved scans are required')
    topology = before['summary']['topology']
    if (before['parts'] != after['parts'] or topology != after['summary']['topology']
            or any(v is None for v in topology.values())):
        return missing('Saved scan mesh inventories, topology filters or semantic parts differ or are unrecorded')
    keys, i, j = np.intersect1d(before['keys'], after['keys'], assume_unique=True, return_indices=True)
    same = before['kinds'][i] == after['kinds'][j]
    mismatched = int(np.count_nonzero(~same))
    keys, i, j = keys[same], i[same], j[same]
    if not len(keys):
        return missing('The scans share no sample time and kind')
    totals = Counter()
    events = {}
    for pair in sorted(before['events'].keys() | after['events'].keys()):
        a = np.isin(i, before['events'].get(pair, []))
        b = np.isin(j, after['events'].get(pair, []))
        new, gone = ~a & b, a & ~b
        totals.update(new=int(new.sum()), disappeared=int(gone.sum()), retained=int((a & b).sum()))
        if new.any() or gone.any():
            events[' / '.join(pair)] = dict(new=event_ranges(new, keys), disappeared=event_ranges(gone, keys))
    return dict(
        common_samples=len(keys), common_samples_by_kind=kind_counts(before['kinds'][i]),
        before_only_samples=len(before['keys'])-len(keys), after_only_samples=len(after['keys'])-len(keys),
        same_time_different_kind_samples=mismatched,
        retained_triangle_pair_samples=dict(before=int(before['counts'][i].sum()), after=int(after['counts'][j].sum())),
        retained_positive_samples_by_kind=dict(
            before=kind_counts(before['kinds'][i[before['counts'][i] > 0]]),
            after=kind_counts(after['kinds'][j[after['counts'][j] > 0]]),
        ),
        semantic_pair_sample_transitions=dict(totals), changed_event_sets=events,
    )


def stage_diagnostics(spec, pts, tb):
    path = spec.get('result')
    summary_path = spec.get('summary')
    if summary_path is not None and not summary_path.is_file():
        return missing(f'Stage has no completed run_summary.json: {summary_path.parent}'), {}, {}
    summary = read_json(summary_path) if summary_path else {}
    if path is None or not path.is_file():
        return missing(f'Saved result unavailable: {path or "no path supplied"}'), {}, {}
    if summary.get('source_frames', len(pts)) != len(pts):
        raise ValueError('Run summary does not describe the complete source clock')
    metrics_path = spec.get('metrics')
    metrics = optional(metrics_path, read_json)
    metrics_source = str(metrics_path.resolve()) if metrics.get('available') is not False else metrics
    metrics = metrics.get(spec['label'], metrics)
    parameter = spec.get('parameter')
    if ('opening_scope' in metrics and
            (parameter is None or 'candidate' not in metrics or Path(metrics['candidate']).resolve() != parameter.resolve())):
        metrics_source = missing('Factor checks do not identify this saved stage parameter')
        metrics = {}
    diagnostics = metrics.get('physical_diagnostics', metrics.get('metrics', metrics))
    with np.load(path, allow_pickle=False) as result:
        check_clock(result, pts, tb)
        positions = {name: result[key] for name, key in POSITIONS.items() if key in result}
        if any(v.shape != (len(pts), 3) or not np.isfinite(v).all() for v in positions.values()):
            raise ValueError('Saved positions must have finite XYZ on the complete source clock')
        output = dict(
            result=str(path.resolve()), source_frames=len(pts),
            run_summary={key: summary[key] for key in (
                'retained_step', 'additional_accepted_steps', 'original_primary_guards_pass',
                'trajectory_candidate', 'accuracy_accepted',
            ) if key in summary},
            missing_position_fields=[POSITIONS[name] for name in POSITIONS if name not in positions],
            metrics_source=metrics_source,
            opening=checked(lambda: opening_report(result, pts, positions, diagnostics, metrics), 'Opening'),
            COM_acceleration=checked(
                lambda: acceleration_report(result, spec.get('motion'), pts, tb, positions), 'COM acceleration',
            ),
            knees_and_elbows=optional(spec.get('anatomy'), lambda p: anatomy_report(p, pts, result)),
            source_supported_grasp=checked(lambda: grasp_report(diagnostics, pts), 'Source-supported grasp'),
        )
    scan = optional(spec.get('surface'), lambda p: read_scan(p, pts, tb, spec.get('parameter')))
    output['surface_scan'] = scan.get('summary', scan)
    return output, positions, scan


def named_path(value):
    name, sep, path = value.partition('=')
    if not name or not sep or not path:
        raise argparse.ArgumentTypeError('Use LABEL=PATH')
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('stage', 'result', 'anatomy', 'surface', 'motion', 'metrics'):
        parser.add_argument('--'+option, type=named_path, action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--baseline', required=True, metavar='LABEL')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specs = {}
    for label, path in args.stage:
        if label in specs:
            parser.error('Repeated stage '+label)
        specs[label] = dict(
            label=label, summary=path/'run_summary.json', result=path/'fitted/result.npz',
            anatomy=path/'anatomical_frames.npz', metrics=path/'final_metrics.json',
            surface=path/'surface/surface_scan.npz', parameter=path/'solve/candidate.npz',
        )
    for label, path in args.result:
        if label in specs:
            parser.error('Choose either --stage or --result for '+label)
        specs[label] = dict(label=label, result=path)
    for field in ('anatomy', 'surface', 'motion', 'metrics'):
        seen = set()
        for label, path in getattr(args, field):
            if label not in specs or label in seen:
                parser.error('Unknown or repeated candidate '+label+' for --'+field)
            seen.add(label)
            specs[label][field] = path
    if args.baseline not in specs:
        parser.error('Baseline must name a --stage or --result candidate')
    pts, _, tb = required_clock(read_json(SCOPE))
    candidates, position_sets, scans = {}, {}, {}
    for label, spec in specs.items():
        try:
            candidates[label], position_sets[label], scans[label] = stage_diagnostics(spec, pts, tb)
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            candidates[label], position_sets[label], scans[label] = missing(str(exc)), {}, {}
    comparisons = {}
    base = args.baseline
    for label in (name for name in specs if name != base):
        a, b = position_sets[base], position_sets[label]
        comparisons[label] = dict(
            displacement_from_baseline_m={
                name: stats(np.linalg.norm(a[name]-b[name], axis=1)) if name in a and name in b
                else missing(f'Both saved results must contain {POSITIONS[name]} on the complete clock')
                for name in POSITIONS
            },
            surface_events=compare_scans(scans[base], scans[label]),
        )
    report = dict(
        schema='rinkspace.programme-repair-progress.v1', baseline=base,
        source_frames=len(pts), source_pts_range=[int(pts[0]), int(pts[-1])], source_time_base=list(tb),
        candidates=candidates, comparison_to_baseline=comparisons,
        surface_event_semantics=(
            'Saved retained semantic-part pair incidence at equal nanosecond and sample kind. '
            'Each event interval is [first_source_ns, last_source_ns, matching_sample_count] on the common sample sequence. '
            'Counts depend on each scan policy; conditional grip and uncertain head candidates are separate. '
            'Unscanned times and continuous contact remain unknown.'
        ),
        accuracy_accepted=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(args.output.resolve()), candidates=list(candidates))))


if __name__ == '__main__':
    main()
