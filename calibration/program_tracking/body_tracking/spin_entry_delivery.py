"""Thin saved entry-motion delivery; no work runs on import.

Uses the shared native mesh renderer and an explicit completed-run binding.
The old eight-step support-foot state is restored from its original case;
the new entry state is restored from its own output case.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import spin_support_foot_delivery as shared

SECONDS = (80.8, 81.3, 81.4, 81.5, 81.7, 82.2, 89.56)


def _bindings(project, base, output, foot_output=None, foot_reference=None):
    project, base, output = (Path(p).resolve() for p in (project, base, output))
    foot = Path(foot_output).resolve() if foot_output else base / 'spin_support_foot_1104'
    reference = (Path(foot_reference).resolve() if foot_reference else
                 base / 'spin_grasp_1104/support_joint_v2')
    spec = shared.DeliverySpec(
        variant='entry_motion', parameter_from_summary=True,
        source_config='viewer3d_spin_support_foot_1104.json',
        config_name='viewer3d_spin_entry_motion_1104.json',
        launcher_name='START_3D_SPIN_ENTRY.cmd', baseline_variant='support_foot',
        old_result=foot / shared.CASE / 'fitted/result.npz',
        old_parameter=project / shared.PROGRAMME / 'spin_support_foot_1104.npz',
        old_reference_case=reference,
        label='整段入旋可动候选（{accepted}步，待原片审核）',
        panel_labels=('上一轮支撑脚8步（保留对照）', '整段入旋可动候选（{accepted}步，待审核）'),
        seconds=SECONDS, refinement_intervals=((76., 95.),),
        use_current_case_inputs=True, initial_source_seconds=81.3,
        window_scope='Single continuous 76-95s state; all 951 source exposures movable',
    )
    return project, base, foot, output, spec


def register(project, base, output, *, foot_output=None, foot_reference=None):
    """Append entry_motion to a separate config preserving all foot variants."""
    project, base, foot, output, spec = _bindings(project, base, output, foot_output, foot_reference)
    return shared.register(project, base, foot, output, previous=output, spec=spec)


def render_stills(project, base, output, *, foot_output=None, foot_reference=None):
    """Six exact source-PTS comparisons using the shared five-node renderer."""
    project, base, foot, output, spec = _bindings(project, base, output, foot_output, foot_reference)
    return shared.render_stills(project, base, foot, output, previous=output, spec=spec)


def trajectory(project, base, output, *, foot_output=None, foot_reference=None):
    """Plot all 951 saved pelvis XY samples, highlighting source 81-82s.

    Coordinates are used exactly as saved. They are conditional model metres,
    not measured centre-of-mass or athlete displacement; no points are removed.
    """
    project, base, foot, output, spec = _bindings(project, base, output, foot_output, foot_reference)
    saved = shared._saved(project, base, foot, output, previous=output, spec=spec)
    packets, pts, evidence = saved[6], saved[8], saved[9]
    image_path = shared._guard(output / 'entry_motion_saved_pelvis_xy.png', project)
    report_path = shared._guard(output / 'entry_motion_saved_pelvis_xy.json', project)
    for path in (image_path, report_path):
        if path.exists():
            raise FileExistsError(f'Preserve existing trajectory delivery: {path}')
    positions = {}
    for name, packet in packets.items():
        value = np.asarray(packet['P_world_mid'])
        if value.shape != (951, 3) or not np.isfinite(value).all():
            raise ValueError(f'{name}: expected all 951 finite saved pelvis positions')
        positions[name] = value
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    highlight = (pts >= 81 * 12800) & (pts <= 82 * 12800)
    fig, ax = plt.subplots(figsize=(9, 8))
    try:
        for name, label, color in (('old_result', 'Previous support-foot 8 steps', '#bd6b32'),
                                   ('result', 'Entry-motion candidate', '#2466a8')):
            xy = positions[name][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.2, alpha=.6,
                    label=label + ' | all 951 exposures')
            segment = xy[highlight]
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=3.5,
                    label=label + ' | source 81.0-82.0s')
            ax.scatter(segment[[0, -1], 0], segment[[0, -1], 1], color=color, s=30)
        ax.set(xlabel='Saved world X (conditional model m)',
               ylabel='Saved world Y (conditional model m)',
               title='Saved pelvis XY: source 76-95s; no coordinate alignment')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.25)
        ax.legend(fontsize=8); fig.tight_layout()
        with image_path.open('xb') as stream:
            fig.savefig(stream, format='png', dpi=150)
    finally:
        plt.close(fig)
    report = dict(evidence, plot=str(image_path), plot_sha256=shared._digest(image_path),
                  plotted_points_per_case=951, all_source_points_preserved=True,
                  highlight_source_seconds=[81., 82.], highlight_source_pts=pts[highlight].tolist(),
                  coordinates='unaltered saved P_world_mid[:, :2]; conditional model metres',
                  interpretation='model pelvis projection, not measured CoM or athlete displacement')
    shared._write_new_json(report_path, report, project)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'stills', 'trajectory'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--foot-output', type=Path)
    parser.add_argument('--foot-reference', type=Path)
    args = parser.parse_args(argv)
    action = dict(register=register, stills=render_stills, trajectory=trajectory)[args.action]
    return action(args.project, args.base_root, args.output,
                  foot_output=args.foot_output, foot_reference=args.foot_reference)


if __name__ == '__main__':
    main()
