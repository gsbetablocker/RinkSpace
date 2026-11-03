"""Headless physical overview from saved packets; never fits or invents support."""
from pathlib import Path
import argparse
import json

import numpy as np


REVIEWED_SUPPORT_SECONDS = (26.22, 35.20, 35.60, 36.02, 36.60, 39.78, 43.86)
REFERENCE_BOUNDARIES = (28.12, 34.8, 35.2, 36.8, 37.2, 43.8)
FOCUS_SECONDS = (26.22, 39.78, 43.86)
EXPECTED_PTS = np.arange(283136, 575489, 256)


def strict_collar_values(report, pts):
    """Worst of both actual foot rows; an unknown/missing foot keeps NaN."""
    pts = np.asarray(pts)
    if report.get('source_time_base') != [1, 12800]:raise ValueError('Unexpected collar time base')
    if report.get('source_frames') != len(pts):raise ValueError('Collar/source exposure count mismatch')
    if report.get('source_first_last_pts') != [int(pts[0]), int(pts[-1])]:raise ValueError('Collar/source PTS bounds mismatch')
    values = np.full((len(pts), 2), np.nan)
    seen = set();lookup = {int(p): i for i, p in enumerate(pts)}
    for row in report['records']:
        key = (int(row['pts']), int(row['foot']))
        if key in seen:raise ValueError('Duplicate collar exposure/foot row')
        seen.add(key)
        if key[0] not in lookup or key[1] not in (0, 1):raise ValueError('Unexpected collar PTS or foot')
        if row['status'] == 'measured_geometry':
            value = row.get('maximum_outside_m')
            if value is None or not np.isfinite(value) or value < 0:raise ValueError('Invalid measured collar outside distance')
            values[lookup[key[0]], key[1]] = value
    strict = np.max(values, axis=1)
    return strict, dict(foot_rows_expected=2*len(pts), foot_rows_present=len(seen),
        unknown_or_missing_foot_rows=int((~np.isfinite(values)).sum()),
        strict_known_exposures=int(np.isfinite(strict).sum()),
        convention='both feet known required; no nanmax or zero-filling')


def reviewed_support_points(packet):
    """Finite saved group-support heights only at the seven reviewed PTS."""
    pts = np.asarray(packet['pts']);gap = np.asarray(packet['skate_support_gap_m_mid'], float)
    if gap.shape == (len(pts), 1):gap = gap[:, 0]
    if gap.shape != (len(pts),):raise ValueError('Expected saved per-exposure reviewed support-group height')
    reviewed = np.isin(pts, [round(t*12800) for t in REVIEWED_SUPPORT_SECONDS])
    finite = np.isfinite(gap);selected = reviewed & finite
    return pts[selected], gap[selected], dict(reviewed_exposures_in_batch=int(reviewed.sum()),
        finite_reviewed_exposures=int(selected.sum()),
        reviewed_unknown_exposures=int((reviewed & ~finite).sum()),
        finite_unreviewed_values_excluded=int((~reviewed & finite).sum()),
        convention='scatter only; no interpolation, propagation, inferred support, or side identification')


def _read_result(path):
    keys = ('pts', 'time_base', 'skate_clearance_m_mid', 'skate_support_gap_m_mid', 'foot_cavity_max_violation_m_mid')
    with np.load(path, allow_pickle=False) as archive:packet = {key: archive[key] for key in keys}
    if not np.array_equal(packet['pts'], EXPECTED_PTS) or not np.array_equal(packet['time_base'], [1, 12800]):
        raise ValueError('Expected all 1143 available S01 source exposures, time base 1/12800')
    for key in ('skate_clearance_m_mid', 'foot_cavity_max_violation_m_mid'):
        if packet[key].shape != (len(EXPECTED_PTS), 2):raise ValueError('Expected two feet: '+key)
    return packet


def plot_physical_overview(old_result, new_result, old_collar, new_collar, out):
    """Write only out/physical_overview.png using matplotlib's Agg backend."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    packets = [_read_result(path) for path in (old_result, new_result)]
    collars = [json.loads(Path(path).read_text(encoding='utf8')) for path in (old_collar, new_collar)]
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    details = {}
    try:
        for label, color, packet, collar in zip(('Old candidate', 'New continuous candidate'),
                                               ('#777777', '#0072B2'), packets, collars):
            pts = packet['pts'];seconds = pts/12800.
            violation = np.maximum(0., -np.min(packet['skate_clearance_m_mid'], axis=1))
            cavity = np.max(packet['foot_cavity_max_violation_m_mid'], axis=1)
            outside, collar_info = strict_collar_values(collar, pts)
            support_pts, support, support_info = reviewed_support_points(packet)
            axes[0].plot(seconds, violation*1000, label=label, color=color, linewidth=1)
            axes[1].scatter(support_pts/12800., support*1000, label=label, color=color,
                            marker='o' if label.startswith('Old') else 'x', s=35, zorder=4)
            axes[2].plot(seconds, cavity*1000, label=label, color=color, linewidth=1)
            axes[3].plot(seconds, outside*1000, label=label, color=color, linewidth=1)
            details[label] = dict(collar=collar_info, support=support_info)
        axes[0].set_ylabel('Maximum nonpenetration\nconstraint violation (mm)')
        axes[1].set_ylabel('Reviewed support-group\nraw height (mm)')
        axes[1].axhspan(-2., 3., color='#E69F00', alpha=.18, label='Debug band [-2,+3] mm: uncalibrated')
        axes[2].set_ylabel('Worst generic-foot\ncavity violation (mm)')
        axes[3].set_ylabel('Strict collar-section\nmaximum outside (mm)')
        for ax in axes:
            ax.grid(alpha=.18)
            for t in REFERENCE_BOUNDARIES:ax.axvline(t, color='#999999', linestyle='--', linewidth=.7, alpha=.6)
            for t in FOCUS_SECONDS:ax.axvline(t, color='#D55E00', linestyle=':', linewidth=1, alpha=.8)
        for t in FOCUS_SECONDS:
            axes[0].text(t, 1.02, f'{t:.2f}s', transform=axes[0].get_xaxis_transform(),
                         ha='center', va='bottom', color='#A34000', fontsize=9)
        axes[0].legend(loc='upper right', fontsize=9)
        axes[1].legend(loc='upper right', fontsize=8)
        axes[-1].set_xlim(EXPECTED_PTS[0]/12800., EXPECTED_PTS[-1]/12800.)
        axes[-1].set_xlabel('Source seconds (PTS / 12800)')
        fig.suptitle('Source50Hz diagnostics, not accuracy certification', fontsize=15)
        fig.text(.5, .016,
            'Dashed lines: old boundary references, not clips. Support: reviewed exposures only; all other labels unknown; no interpolation.\n'
            'Collar: one horizontal section; either foot unknown keeps NaN. Generic foot is not an athlete measurement. Debug band is uncalibrated.',
            ha='center', va='bottom', fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .955))
        destination = Path(out)/'physical_overview.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
        return dict(path=str(destination), source_exposures=len(EXPECTED_PTS), series=details)
    finally:plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-result', required=True);parser.add_argument('--new-result', required=True)
    parser.add_argument('--old-collar', required=True);parser.add_argument('--new-collar', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(plot_physical_overview(args.old_result, args.new_result, args.old_collar, args.new_collar, args.output)))
