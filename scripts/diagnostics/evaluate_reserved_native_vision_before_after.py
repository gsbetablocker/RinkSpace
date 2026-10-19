"""Frozen direct-source evaluation of every native frame via public inference."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import av
import numpy as np
import torch
from torch.nn import functional as F
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from rinkspace.vision import RinkVisionModule, configure, load_training_manifest


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def direct_metrics(result, records):
    names = tuple(result['classes'])
    probabilities = result['class_probabilities']
    h, w = result['native_hw']
    uv = torch.tensor([r['literal_observed_uv'] for r in records], device=probabilities.device)
    grid = uv * uv.new_tensor([2 / (w - 1), 2 / (h - 1)]) - 1
    sampled = F.grid_sample(probabilities, grid[None, :, None], align_corners=True)[0, :, :, 0].T
    ys = torch.arange(h, device=uv.device, dtype=uv.dtype)
    xy = torch.stack((uv[:, 0, None].expand(-1, h), ys[None].expand(len(uv), -1)), dim=-1)
    column_grid = xy * uv.new_tensor([2 / (w - 1), 2 / (h - 1)]) - 1
    profiles = F.grid_sample(probabilities, column_grid[None], align_corners=True)[0].permute(1, 0, 2)
    values = sampled.cpu().numpy()
    profile_values = profiles.cpu().numpy()
    board = names.index('board_top')
    static_types = {'ice_contact', 'board_top', 'ice_paint', 'ad_boundary'}
    rows = []
    for i, record in enumerate(records):
        label = names[int(values[i].argmax())]
        row = dict(source_global_point_index=record['source_global_point_index'],
            native_pts=record['native_pts'], literal_observed_uv=record['literal_observed_uv'],
            direct_source_decision=record['decision'], predicted_type_at_literal_uv=label,
            type_probabilities_at_literal_uv={name: float(values[i, j]) for j, name in enumerate(names)},
            physical_upper_rail_instance_identity_provided=False)
        if record['decision'] == 'actual_upper_blue_board_rail':
            typed = profile_values[i].argmax(0) == board
            row['board_top_type_at_literal_uv'] = label == 'board_top'
            row['board_top_type_missing_in_full_native_column'] = not bool(typed.any())
            row['source_vertical_10_to_90_colour_transition_interval_px'] = record['source_vertical_10_to_90_transition_interval_px']
            if typed.any():
                y = int(np.where(typed, profile_values[i, board], -np.inf).argmax())
                source_y = record['literal_observed_uv'][1]
                lo, hi = (source_y + d for d in record['source_vertical_10_to_90_transition_interval_px'])
                row.update(full_column_strongest_board_top_typed_y=y,
                    q50_vertical_position_error_px=y - source_y,
                    outside_actual_colour_transition_distance_px=max(lo - y, 0., y - hi),
                    strongest_board_top_position_inside_actual_colour_transition=lo <= y <= hi)
        elif record['decision'] == 'foreground_person_not_upper_board_rail':
            row['foreground_point_mistyped_as_static_fixture'] = label in static_types
        else:
            row['printed_blue_exclusion_is_not_a_global_model_negative'] = True
        rows.append(row)
    return rows


def aggregate(rows):
    positives = [r for r in rows if r['direct_source_decision'] == 'actual_upper_blue_board_rail']
    available = [r for r in positives if not r['board_top_type_missing_in_full_native_column']]
    errors = np.asarray([r['q50_vertical_position_error_px'] for r in available])
    outside = np.asarray([r['outside_actual_colour_transition_distance_px'] for r in available])
    printed = [r for r in rows if r['direct_source_decision'] == 'printed_blue_not_upper_board_rail']
    people = [r for r in rows if r['direct_source_decision'] == 'foreground_person_not_upper_board_rail']
    return dict(direct_reviewed_native_frames=len({r['native_pts'] for r in rows}), direct_upper_rail_points=len(positives),
        direct_non_upper_rail_points=len(printed) + len(people), board_top_type_at_literal_uv=sum(r['board_top_type_at_literal_uv'] for r in positives),
        board_top_type_missing_in_full_native_column=len(positives) - len(available),
        full_column_position_available_points=len(available),
        full_column_strongest_type_q50_abs_error_median_px=float(np.median(abs(errors))) if len(errors) else None,
        full_column_strongest_type_q50_abs_error_max_px=float(np.max(abs(errors))) if len(errors) else None,
        full_column_strongest_type_outside_colour_transition_mean_px=float(outside.mean()) if len(outside) else None,
        full_column_strongest_type_inside_actual_colour_transition_points=sum(r['strongest_board_top_position_inside_actual_colour_transition'] for r in available),
        printed_blue_point_type_histogram=dict(Counter(r['predicted_type_at_literal_uv'] for r in printed)),
        foreground_point_type_histogram=dict(Counter(r['predicted_type_at_literal_uv'] for r in people)),
        foreground_points_mistyped_as_static_fixture=sum(r['foreground_point_mistyped_as_static_fixture'] for r in people),
        physical_upper_rail_instance_identity_provided=False)


def run(args):
    config, data = configure(args.config)
    torch.set_num_threads(int(config['strict_source_dataset_training'].get('cpu_threads', 1)))
    seeds = json.loads(args.direct_seeds.read_text(encoding='utf8'))
    evidence = json.loads(args.native_evidence.read_text(encoding='utf8'))
    source = evidence['source_video_reference']
    for reference in (ref(args.direct_seeds), ref(args.native_evidence), evidence['source_numeric_evidence_reference'], source):
        authenticate_source_reference(reference)
    if (seeds['direct_original_reviewed_rail_points'] != 98 or seeds['direct_original_reviewed_nonrail_points'] != 10
            or seeds['permanent_whole_shot_training_and_tuning_exclusion'] is not True
            or seeds['candidate_model_output_used_for_ground_truth'] is not False):
        raise ValueError('Original independent direct-source scope changed')
    with np.load(evidence['source_numeric_evidence_reference']['path'], allow_pickle=False) as arrays:
        native_pts = arrays['native_pts']
        expected_hashes = {int(n): bytes(h).decode('ascii') for n, h in zip(native_pts, arrays['decoded_BGR_sha256'])}
        candidate_uv = arrays['candidate_observed_uv']
    if len(native_pts) != 2556 or not np.all(np.diff(native_pts) == 256):
        raise ValueError('The complete source-native clock is required')
    manifest, shot_plan = load_training_manifest(config, data)
    shot = next(r for r in shot_plan['shots'] if r['id'] == evidence['shot_id'])
    start, end = evidence['exact_reserved_native_interval']
    if shot['split'] != 'independent_evaluation' or not shot['whole_shot'] or [shot['start_pts'], shot['end_pts']] != [start, end]:
        raise ValueError('Independent complete shot role changed')
    if any(start <= int(s['source_pts']) < end for s in manifest['samples'] if s['source_hash'] == source['sha256']):
        raise ValueError('The reserved source overlaps training samples')
    training_bundle = json.loads(args.training_bundle.read_text(encoding='utf8'))
    if any(start <= int(n) < end for n in training_bundle['crop_origins_by_native']):
        raise ValueError('The reserved source entered current training')
    by_native = {}
    for record in seeds['records']:
        if not np.array_equal(candidate_uv[record['source_global_point_index']], np.asarray(record['literal_observed_uv'])):
            raise ValueError('Direct original reviewed point UV changed')
        if expected_hashes[record['native_pts']] != record['decoded_BGR_sha256']:
            raise ValueError('Direct source frame identity changed')
        authenticate_source_reference(record['original_source_reference'])
        by_native.setdefault(record['native_pts'], []).append(record)
    if args.output.exists():
        raise FileExistsError('Preserve earlier original independent evaluation')
    args.output.mkdir(parents=True)
    plan = dict(schema='rinkspace.frozen-independent-native-image-only-evaluation-plan.v1',
        direct_source_seed_reference=ref(args.direct_seeds), complete_native_evidence_reference=ref(args.native_evidence),
        source_video_reference=source, training_bundle_reference=ref(args.training_bundle),
        formal_model_reference=ref(args.formal_model), candidate_model_reference=ref(args.candidate_model),
        implementation_reference=ref(__file__), original_native_interval=[start, end], native_frames=2556,
        direct_source_frames=16, direct_positive_points=98, direct_exclusion_points=10,
        direct_source_upper_rail_type='board_top',
        prediction_mode='Public RinkVisionModule.predict on each continuous original RGB; no camera or world shapes; shot reset only at start',
        type_sampling='Bilinear class probabilities at each direct literal source UV; argmax of unchanged six types',
        position_policy='For each direct upper-rail x, evaluate all 1080 native rows; choose highest board_top probability among rows whose interpolated six-class argmax is board_top; missing if none',
        position_comparison='Signed y difference from actual source q50 and distance outside actual source 10-to-90 colour transition; no added acceptance radius or sigma interpretation',
        printed_blue_exclusion_policy='Report type histogram only; these 9 board-body points are not global negatives and module supplies no upper-rail instance names',
        foreground_exclusion_policy='For the 1 direct person point report all types and static fixture false typing in ice_contact/board_top/ice_paint/ad_boundary',
        unlabeled_frames_accuracy_policy='2540 frames are unreviewed direct material truth; report successful inference, missing prediction and native type inventories only',
        source_link_derived_points_used_as_GT=False, direct_source_seed_given_to_predict=False,
        independent_source_ever_used_for_training_or_tuning=False, whole_shot_accuracy_accepted=False)
    plan_path = args.output / 'frozen_evaluation_plan.json'
    write(plan_path, plan)
    print(json.dumps(dict(status='independent_native_evaluation_plan_frozen', reference=ref(plan_path))), flush=True)
    summaries = {}
    for label, model_path in (('before', args.formal_model), ('after', args.candidate_model)):
        begun = time.perf_counter()
        module = RinkVisionModule(model_path, ROOT / config['flow_weights'], device=config['device'],
            exposure_registry=config.get('source_clock_registry'))
        if list(module.classes) != config['classes']:
            raise ValueError('Prediction semantic types changed')
        usage = module.model_usage()
        points, processed = [], []
        counts = Counter()
        journal_path = args.output / f'{label}_native_prediction_inventory.jsonl'
        maps_path = args.output / f'{label}_native_class_maps.zip'
        with av.open(source['path']) as container, journal_path.open('x', encoding='utf8') as journal, ZipFile(maps_path, 'x', compression=ZIP_DEFLATED, compresslevel=1) as maps:
            stream = container.streams.video[0]
            stream.thread_count = int(config['strict_source_dataset_training'].get('cpu_threads', 1))
            if str(stream.time_base) != evidence['source_time_base']:
                raise ValueError('Actual evaluation source clock changed')
            container.seek(start, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < start:
                    continue
                if frame.pts >= end:
                    break
                native = int(frame.pts)
                bgr = frame.to_ndarray(format='bgr24')
                digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
                if native not in expected_hashes or digest != expected_hashes[native] or bgr.shape != (1080, 1920, 3):
                    raise ValueError('Actual independent RGB/frame binding changed')
                binding = dict(source_sha256=source['sha256'], source_pts=native,
                    time_base=evidence['source_time_base'], shot_id=evidence['shot_id'], native_hw=[1080, 1920])
                result = module.predict(np.ascontiguousarray(bgr[..., ::-1]), binding)
                if result['probabilities_conditioned_on_geometry'] or result['projected_shapes'] or result['camera_shape_priors']:
                    raise ValueError('Independent image-only inference received geometry')
                if not torch.isfinite(result['class_probabilities']).all():
                    raise ValueError('Nonfinite actual native prediction')
                labels = result['class_probabilities'].argmax(1)[0].cpu().numpy().astype(np.uint8)
                histogram = np.bincount(labels.ravel(), minlength=len(module.classes))
                with maps.open(f'{native}.npy', 'w') as entry:
                    np.save(entry, labels, allow_pickle=False)
                metrics = direct_metrics(result, by_native[native]) if native in by_native else []
                points.extend(metrics)
                row = dict(native_pts=native, decoded_BGR_sha256=digest, prediction_native_hw=[1080, 1920],
                    source_success=True, prediction_missing=False, class_map_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                    class_pixel_counts={name: int(histogram[i]) for i, name in enumerate(module.classes)},
                    direct_source_reviewed=native in by_native, direct_source_point_metrics=metrics,
                    prediction_source_binding=result['prediction_source_binding'], memory_update=result['memory_update'],
                    geometry_prior_used=False, physical_upper_rail_instance_identity_provided=False,
                    accuracy_accepted=False)
                journal.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                journal.flush()
                processed.append(native)
                counts.update(row['class_pixel_counts'])
                if len(processed) % 50 == 0:
                    print(json.dumps(dict(model=label, native_frames=len(processed), native_pts=native,
                        elapsed_seconds=round(time.perf_counter() - begun, 3))), flush=True)
                del result, labels
        if processed != native_pts.tolist():
            raise ValueError('A complete original native frame was lost or duplicated')
        summary = dict(model_usage=usage, actual_native_frames=len(processed), native_source_and_prediction_missing_frames=0,
            first_native_pts=processed[0], last_native_pts=processed[-1], all_native_frame_and_RGB_bindings_verified=True,
            class_pixel_counts=dict(counts), direct_metrics=aggregate(points), unreviewed_direct_native_frames=2540,
            direct_source_only_metrics=True, source_link_derived_points_used_as_GT=False,
            complete_native_inventory_reference=ref(journal_path), complete_native_class_maps_reference=ref(maps_path),
            whole_shot_accuracy_accepted=False, elapsed_seconds=time.perf_counter() - begun)
        write(args.output / f'{label}_summary.json', summary)
        summaries[label] = summary
        del module
        gc.collect()
        torch.cuda.empty_cache()
    output = dict(schema='rinkspace.complete-independent-native-image-only-before-after-evaluation.v1',
        frozen_plan_reference=ref(plan_path), summaries=summaries,
        physical_upper_rail_instance_identity_provided=False, whole_shot_accuracy_accepted=False,
        unreviewed_direct_native_frames_not_counted_as_accuracy_pass=2540,
        formal_model_replaced=False, independent_source_used_for_training_or_tuning=False)
    write(args.output / 'before_after_evaluation.json', output)
    print(json.dumps(dict(status='independent_native_before_after_evaluation_complete',
        report=ref(args.output / 'before_after_evaluation.json'), summaries={k: v['direct_metrics'] for k, v in summaries.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('config', 'direct-seeds', 'native-evidence', 'training-bundle', 'formal-model', 'candidate-model', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    run(parser.parse_args())
