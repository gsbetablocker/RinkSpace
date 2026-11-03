"""Static source-crop and saved-motion diagnostics; no fitting or video decoding."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT)]
from calibration.program_tracking.body_tracking.fullbody_review import COCO_EDGES, load, sha256

TIME_BASE = 12800
SELECTED_SECONDS = (22.12, 22.76, 23.08, 23.40, 23.72, 24.04)
MODES = (('old_opening', 'Old opening', '--'), ('new_motion', 'New motion', '-'))


def checked_path(path):
    path = Path(path).resolve()
    if len(str(path).encode('utf-16-le'))//2 >= 260 or len(path.name.encode('utf-16-le'))//2 > 255:
        raise ValueError('Output exceeds the project Windows path policy: '+str(path))
    return path


def exact_index(pts, point):
    index = int(np.searchsorted(pts, point))
    if index == len(pts) or int(pts[index]) != point:
        raise ValueError('Missing exact source exposure: '+str(point))
    return index


def ready_inputs(root, output, source_dir=None):
    source_dir = output if source_dir is None else Path(source_dir).resolve()
    paths = dict(old=root/'opening_repair_1102/fitted/result.npz',
                 new=output/'fitted/result.npz', observations=output/'observations.npz',
                 review_frames=source_dir/'source_review_frames.json', report=output/'motion_report.json')
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Saved results are not ready; no diagnostic output written: '+', '.join(missing))
    hashes = {name: sha256(path) for name, path in paths.items()}
    old, new, obs = [load(paths[name]) for name in ('old', 'new', 'observations')]
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    frames = json.loads(paths['review_frames'].read_text(encoding='utf8'))
    pts = np.asarray(obs['pts'])
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or not np.all(np.diff(pts) > 0):
        raise ValueError('Source identities must be strictly increasing integer PTS')
    for name, packet in [('old', old), ('new', new), ('observations', obs)]:
        np.testing.assert_array_equal(packet['pts'], pts, err_msg=name+' source PTS')
        np.testing.assert_array_equal(packet['time_base'], [1, TIME_BASE], err_msg=name+' clock')
    for name, packet in [('old', old), ('new', new)]:
        if packet['COCO_uv'].shape != (len(pts), 17, 2):
            raise ValueError(name+' COCO source pixel shape is invalid')
    if obs['use'].shape != (len(pts), 17) or obs['uv'].shape != (len(pts), 17, 2):
        raise ValueError('Current source observations have an invalid shape')
    np.testing.assert_array_equal(report['source_time_base'], [1, TIME_BASE])
    curve_pts = np.asarray(report['source_pts'])
    if curve_pts.ndim != 1 or curve_pts.dtype.kind not in 'iu' or not np.all(np.diff(curve_pts) > 0):
        raise ValueError('Motion curves must retain strictly increasing integer source PTS')
    for point in curve_pts:
        exact_index(pts, int(point))
    inferred = np.isin(pts, np.asarray(report['lower_body_inferred']['source_pts'], dtype=np.int64))
    if 'opening_inferred_mask' in new:
        np.testing.assert_array_equal(np.asarray(new['opening_inferred_mask'], bool), inferred)
    by_pts = {}
    for frame in frames:
        point = int(frame['pts'])
        if point in by_pts:
            raise ValueError('Duplicate source image identity: '+str(point))
        if abs(float(frame['seconds'])-point/TIME_BASE) > 1e-10:
            raise ValueError('Source image seconds do not match PTS')
        by_pts[point] = frame
    selected = []
    for seconds in SELECTED_SECONDS:
        point = round(seconds*TIME_BASE)
        if point not in by_pts:
            raise ValueError('No existing source image for PTS '+str(point))
        row = dict(by_pts[point], frame_index=exact_index(pts, point))
        path = Path(row['raw_image'])
        if not path.is_absolute():
            path = source_dir/path
        path = path.resolve()
        if path.parent != source_dir or path.name != 'source_'+str(point)+'.png':
            raise ValueError('Expected the existing exact source_<PTS>.png in the output directory')
        if not path.is_file():
            raise FileNotFoundError(path)
        row['raw_image'] = str(path)
        row['image_sha256'] = sha256(path)
        selected.append(row)
    return paths, hashes, old, new, obs, report, inferred, selected


def source_comparison(output, old, new, obs, inferred, selected, plt):
    from PIL import Image
    target = checked_path(output/'source_motion_comparison.png')
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.7*len(selected)),
                             squeeze=False, constrained_layout=True)
    records = []
    for row, frame in enumerate(selected):
        fi = frame['frame_index']
        with Image.open(frame['raw_image']) as source:
            width, height = source.size
            crop = np.asarray(frame['display_crop'])
            if crop.shape != (4,) or not np.isfinite(crop).all() or not np.equal(crop, np.floor(crop)).all():
                raise ValueError('Display crop must be four integer source pixel bounds')
            x0, y0, x1, y1 = crop.astype(int)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('Display crop lies outside its original source image')
            image = np.array(source.convert('RGB').crop((x0, y0, x1, y1)))
        use = np.asarray(obs['use'][fi], bool)
        selected_uv = obs['uv'][fi, use]
        if not np.isfinite(selected_uv).all():
            raise ValueError('An admitted current source point is nonfinite')
        for col, (label, packet) in enumerate((('Source crop', None), ('Old projection', old), ('New projection', new))):
            ax = axes[row, col]
            ax.imshow(image)
            if packet is not None:
                uv = packet['COCO_uv'][fi]-[x0, y0]
                for a, b in COCO_EDGES:
                    if not np.isfinite(uv[[a, b]]).all():
                        continue
                    dashed = bool(inferred[fi]) and (a >= 11 or b >= 11)
                    ax.plot(uv[[a, b], 0], uv[[a, b], 1], color='#11c8ff', lw=1.3,
                            ls='--' if dashed else '-', alpha=.95)
                measured = selected_uv-[x0, y0]
                ax.scatter(measured[:, 0], measured[:, 1], s=18, facecolors='none',
                           edgecolors='#ffbd22', linewidths=1.)
            ax.set_xlim(-.5, x1-x0-.5)
            ax.set_ylim(y1-y0-.5, -.5)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{frame["pts"]/TIME_BASE:.3f}s | PTS {frame["pts"]} | {label}\n'
                         f'CROP source pixels [{x0}:{x1}, {y0}:{y1}]', fontsize=9)
        records.append(dict(pts=frame['pts'], seconds=frame['pts']/TIME_BASE,
            raw_image=frame['raw_image'], image_sha256=frame['image_sha256'],
            source_size=[width, height], display_crop_xyxy=[int(x0), int(y0), int(x1), int(y1)],
            displayed_current_use_joint_ids=np.flatnonzero(use).tolist(),
            lower_body_inferred=bool(inferred[fi])))
    fig.suptitle('Saved source crops | cyan: model projection | gold: current exposure use points\n'
                 'Dashed lower-body segments are inferred. Crops improve visibility; source UV is unchanged. '
                 'Static development diagnostics; no accuracy verdict.', fontsize=11)
    try:
        fig.savefig(target, dpi=150)
    finally:
        plt.close(fig)
    return target, records


def motion_curves(output, report, plt):
    target = checked_path(output/'motion_curves.png')
    pts = np.asarray(report['source_pts'], dtype=np.int64)
    times = pts/TIME_BASE
    fig, axes = plt.subplots(3, 3, figsize=(19, 12), constrained_layout=True)
    contract_path = output/'inputs.json'
    contract = json.loads(contract_path.read_text(encoding='utf8')) if contract_path.is_file() else {}
    quiet = contract.get('quiet_phase_seconds')

    def values(trace, key, width=None):
        array = np.asarray(trace[key], dtype=float)
        expected = (len(pts),) if width is None else (len(pts), width)
        if array.shape != expected:
            raise ValueError('Motion curve shape does not match original PTS: '+key)
        return array

    for mode, label, style in MODES:
        trace = report['curves'][mode]
        P = values(trace, 'pelvis_world_m', 3)
        axes[0, 0].plot(times, P[:, 2], style, label=label)
        for index, axis in enumerate('XY'):
            axes[0, 1].plot(times, P[:, index]-P[0, index], style,
                            color=('tab:blue', 'tab:orange')[index], label=label+' '+axis)
        wrist = values(trace, 'right_wrist_body_coordinates_m', 3)
        for index, axis in enumerate('XYZ'):
            axes[0, 2].plot(times, wrist[:, index], style,
                color=('tab:blue', 'tab:orange', 'tab:green')[index], label=label+' '+axis)
        angles = trace['local_angle_from_dynamic_raw_deg']
        for joint, color in [('right_shoulder', 'tab:blue'), ('right_elbow', 'tab:orange')]:
            axes[1, 0].plot(times, values(angles, joint), style, color=color,
                            label=label+' '+joint.replace('right_', ''))
        axes[1, 1].plot(times, values(trace, 'right_elbow_internal_angle_deg'), style, label=label)
        axes[1, 2].plot(times, values(trace, 'right_wrist_local_change_from_window_first_deg'), style, label=label)
        axes[2, 0].plot(times, values(trace, 'root_heading_unwrapped_deg'), style, label=label)
        axes[2, 1].plot(times, values(trace, 'torso_from_world_up_deg'), style, label=label)
        axes[2, 2].plot(times, values(trace, 'minimum_saved_blade_z_m'), style, label=label)
    axes[2, 0].plot(times, values(report['curves']['new_motion'], 'raw_root_heading_unwrapped_deg'),
                    ':', color='gray', label='Raw prior')
    axes[2, 2].axhline(0, color='gray', lw=.7)
    titles = [('Pelvis Z', 'Conditional model m'), ('Pelvis XY change from own first sample', 'Conditional model m'),
              ('Right wrist in body coordinates', 'Conditional model m'),
              ('Local SO(3) angle from dynamic raw', 'Degrees'), ('Right elbow internal angle (180 = straight)', 'Degrees'),
              ('Right wrist local change from first sample', 'Degrees'), ('Root heading (unwrapped model axis)', 'Degrees'),
              ('Torso angle from world up', 'Degrees'), ('Minimum saved blade Z (contact unknown)', 'Conditional model m')]
    for ax, (title, ylabel) in zip(axes.ravel(), titles):
        if quiet:
            ax.axvline(quiet[1], color='purple', lw=.9, ls=':', label=f'Quiet body until {quiet[1]:.2f}s')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Original source seconds = integer PTS / 12800')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(times[0], times[-1])
    fig.suptitle('Saved opening motion | dashed: old opening | solid: new motion\n'
                 'Every curve uses original exposure PTS; no temporal smoothing. '
                 'Model quantities and raw priors are not independent 3D truth; no automatic winner.', fontsize=12)
    try:
        fig.savefig(target, dpi=150)
    finally:
        plt.close(fig)
    return target


def run(root, output, source_dir=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths, hashes, old, new, obs, report, inferred, frames = ready_inputs(root, output, source_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison, records = source_comparison(output, old, new, obs, inferred, frames, plt)
    curves = motion_curves(output, report, plt)
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError('Input changed during diagnostics: '+str(path))
    for frame in frames:
        if sha256(frame['raw_image']) != frame['image_sha256']:
            raise ValueError('Source image changed during diagnostics: '+frame['raw_image'])
    record = dict(schema='rinkspace.opening-motion-static-visuals.v1', source_time_base=[1, TIME_BASE],
        sources={name: dict(path=str(path), sha256=hashes[name]) for name, path in paths.items()},
        implementation=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
        frames=records, curve_source_pts=report['source_pts'],
        outputs={name: dict(path=str(path), sha256=sha256(path)) for name, path in
                 [('source_comparison', comparison), ('motion_curves', curves)]},
        source_points='Only original current-exposure observations.use points; no propagated or inferred points added.',
        crop='Recorded source-pixel xyxy bounds, applied equally to all three columns; source UV only translated by crop origin.',
        scope='Six static development examples, not continuous review or independent validation.',
        accuracy_accepted=False, automatic_quality_winner=None)
    metadata = checked_path(output/'visual_diagnostics.json')
    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return dict(source_comparison=str(comparison), motion_curves=str(curves), metadata=str(metadata),
                accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-dir', help='Reuse the existing reviewed source PNG directory')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output, args.source_dir), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
