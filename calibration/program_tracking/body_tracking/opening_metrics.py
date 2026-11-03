"""Plot saved opening candidates only; no solver, source decoding or smoothing.

Preserved from make_opening_metrics_plot.py. Read candidates from root/output
and place new figures/metadata in the explicit report directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from rinkspace.path_policy import guard_project_write


FIRST_PTS = 283136
LAST_PTS = 312832
TIME_BASE = 12800


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in
                ('pts', 'time_base', 'P_world_mid', 'native22_world_mid',
                 'gear_world_mid', 'metres_per_scene_unit')}


def values(packet, selected):
    P = packet['P_world_mid'][selected]
    torso = packet['native22_world_mid'][selected, 12]-packet['native22_world_mid'][selected, 0]
    length = np.linalg.norm(torso, axis=1)
    if np.any(length <= 0):
        raise ValueError('Degenerate torso axis; no plot written')
    per_foot = np.min(packet['gear_world_mid'][selected, :, 14:, 2], axis=2)*100.
    result = dict(pelvis_XY_distance_from_own_first_frame_conditional_m=np.linalg.norm(P[:, :2]-P[0, :2], axis=1),
                  torso_angle_from_world_up_deg=np.rad2deg(np.arccos(np.clip(torso[:, 2]/length, -1., 1.))),
                  minimum_sample_Z_across_both_blades_cm=np.min(per_foot, axis=1),
                  blade_0_minimum_sample_Z_cm=per_foot[:, 0],
                  blade_1_minimum_sample_Z_cm=per_foot[:, 1])
    if any(value.shape != (117,) or not np.isfinite(value).all() for value in result.values()):
        raise ValueError('A trace lacks 117 finite source samples; no plot written')
    return result


def trace_metadata(value, pts):
    return dict(samples=len(value), minimum=float(value.min()), maximum=float(value.max()),
                minimum_source_pts=pts[value == value.min()].tolist(),
                maximum_source_pts=pts[value == value.max()].tolist(),
                values=value.tolist())


def run(root, output, *, report_dir):
    root, output = Path(root).resolve(), Path(output).resolve()
    report_dir = guard_project_write(report_dir)
    png = guard_project_write(report_dir/'metrics_plot.png')
    metadata_path = guard_project_write(report_dir/'metrics_plot.json')
    if png.exists() or metadata_path.exists():
        raise FileExistsError('Choose a new report directory; existing plots are preserved')
    paths = dict(old=root/'boot_integration_1102/fitted/result.npz',
                 new=output/'fitted/result.npz')
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Final export not ready; no plot written: '+', '.join(missing))
    before_hashes = {mode: digest(path) for mode, path in paths.items()}
    packets = {mode: load(path) for mode, path in paths.items()}
    expected = np.arange(FIRST_PTS, LAST_PTS+256, 256, dtype=np.int64)
    traces = {}
    for mode, packet in packets.items():
        np.testing.assert_array_equal(packet['time_base'], [1, TIME_BASE])
        selected = (packet['pts'] >= FIRST_PTS) & (packet['pts'] <= LAST_PTS)
        np.testing.assert_array_equal(packet['pts'][selected], expected)
        traces[mode] = values(packet, selected)
    np.testing.assert_equal(packets['old']['metres_per_scene_unit'], packets['new']['metres_per_scene_unit'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seconds = expected/TIME_BASE
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    panels = [
        ('pelvis_XY_distance_from_own_first_frame_conditional_m', 'Pelvis XY distance from its own first frame', 'Conditional m'),
        ('torso_angle_from_world_up_deg', 'Torso native12 - native0 angle from world +Z', 'Degrees'),
        ('minimum_sample_Z_across_both_blades_cm', 'Lowest saved sample across both blades', 'Z (cm, ice plane = 0)')]
    colors = dict(old='#555555', new='#147dcc')
    for ax, (key, title, label) in zip(axes, panels):
        for mode in ('old', 'new'):
            ax.plot(seconds, traces[mode][key], color=colors[mode],
                    ls='--' if mode == 'old' else '-', marker='.', markersize=2.8,
                    lw=1.25, label=mode)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylabel(label); ax.grid(True, alpha=.22); ax.legend(loc='best', ncol=2)
        ax.set_xlim(seconds[0], seconds[-1])
    axes[2].axhline(0., color='#161616', lw=.85, alpha=.7)
    axes[-1].set_xlabel('Original source time (s), exact PTS / 12800')
    figure.suptitle('Opening: saved old/new candidates | all 117 source exposures, 22.120-24.440 s\nConditional model geometry; no smoothing; not measured motion or confirmed skate contact', fontsize=12)
    metadata = dict(schema='rinkspace.opening-metrics-plot.v1',
        source_pts=expected.tolist(), source_seconds=seconds.tolist(), source_time_base=[1, TIME_BASE],
        samples_per_trace=117, source_interval_inclusive_s=[22.12, 24.44],
        sources={mode: dict(path=str(path), sha256=before_hashes[mode]) for mode, path in paths.items()},
        script=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        panels=[dict(key=key, title=title, ylabel=label) for key, title, label in panels],
        traces={mode: {key: trace_metadata(value, expected) for key, value in rows.items()} for mode, rows in traces.items()},
        definitions=dict(root='Each candidate uses its own first exposure as XY distance origin; anatomical pelvis, conditional metres.',
                         torso='Signed world-up angle: 0 upright, 90 horizontal, 180 inverted.',
                         blades='Third panel is the per-frame minimum over both feet and all saved blade samples (gear indices 14:48). Metadata also preserves each foot separately; anatomical/contact identity is not established.'),
        plotting=dict(all_117_samples_retained=True, smoothing=False, temporal_resampling=False, clipping_or_filtering=False),
        accuracy_accepted=False)
    if any(digest(path) != before_hashes[mode] for mode, path in paths.items()):
        plt.close(figure)
        raise ValueError('An input changed during plotting; no output written')
    historical = output/'make_opening_metrics_plot.py'
    if historical.is_file():
        metadata['historical_script'] = dict(path=str(historical), sha256=digest(historical))
    report_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=160); plt.close(figure)
    metadata['image'] = dict(path=str(png), sha256=digest(png))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return dict(image=str(png), metadata=str(metadata_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing body_s01 run directory')
    parser.add_argument('--output', required=True, help='Existing opening repair directory')
    parser.add_argument('--report-dir', required=True, help='New figure/metadata directory')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output, report_dir=args.report_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
