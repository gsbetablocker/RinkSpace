"""Report source-clock root and mass-proxy kinematics of a saved programme state."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from check_programme_repair_factors import saved_problem

import numpy as np

from calibration.program_tracking.body_tracking.body_clock import source_seconds, time_base
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.phase_ledger import flight_mask
from calibration.program_tracking.body_tracking.programme_com import (
    acceleration_operator,
    export_com_diagnostics,
)
from calibration.program_tracking.body_tracking.rebuild_prepare import digest

CUT_NEIGHBORHOOD_S = .10
LSP_WINDOW_S = (156.2, 165.2)


def source_annotations(reference, evidence):
    pts = reference.p['pts']
    shots = np.asarray(reference.p['shot_ids'], dtype=str)
    cut_indices = np.flatnonzero(shots[:-1] != shots[1:])+1
    cut_points = pts[cut_indices]
    numerator, denominator = time_base(reference.p)
    distances = (pts[:, None]-cut_points)*numerator/denominator
    neighborhood = np.abs(distances) <= CUT_NEIGHBORHOOD_S
    cut_crossing = np.zeros(len(pts), dtype=bool)
    cut_crossing[cut_indices] = True
    cut_crossing[cut_indices-1] = True
    episodes = evidence['source_episode_review']['episodes']
    flights = [episode for episode in episodes if episode['kind'] == 'flight_visible_candidate']
    flight_membership = np.column_stack([
        (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
        for episode in flights
    ]) if flights else np.empty((len(pts), 0), dtype=bool)
    flight = flight_mask(SimpleNamespace(source_document=evidence), pts)
    np.testing.assert_array_equal(flight, flight_membership.any(axis=1))
    return dict(
        shot_id=shots, cut_source_pts=cut_points, cut_source_indices=cut_indices,
        cut_neighborhood_s=np.array(CUT_NEIGHBORHOOD_S),
        cut_neighborhood_membership=neighborhood, near_cut=neighborhood.any(axis=1),
        acceleration_stencil_crosses_cut=cut_crossing,
        flight_episode_ids=np.array([episode['episode_id'] for episode in flights], dtype=str),
        flight_episode_membership=flight_membership, in_flight_core=flight,
    )


def motion_packet(problem, evidence):
    reference, state = problem.reference, problem.body.state
    pts = reference.p['pts']
    numerator, denominator = time_base(reference.p)
    seconds = source_seconds(reference.p)
    times = (pts-pts[0])*numerator/denominator
    np.testing.assert_allclose(reference.times, times, atol=1e-12, rtol=0)
    base = reference.at(times)
    delta = state.increments(np.zeros(state.size), times)
    mass = ContinuousMassGeometry(reference)
    centres, rotations = mass.evaluate(base, delta)
    packet = export_com_diagnostics(mass, times, centres, rotations)
    root = (base['P']+delta[:, :3])*reference.binding.ratio
    ground = root.copy()
    ground[:, 2] = 0.
    relative = centres-root[:, None]
    body_offset, shoe_offset = (
        np.einsum('s,nsc->nc', mass.mass[sl], relative[:, sl])
        for sl in (slice(None, mass.count), slice(mass.count, None))
    )
    operator = acceleration_operator(times)

    def kinematics(name, positions):
        velocity = np.gradient(positions, times, axis=0, edge_order=2)
        acceleration = operator@positions
        # Use the same endpoint convention as the delivered COM diagnostics.
        acceleration = np.vstack((acceleration[:1], acceleration, acceleration[-1:]))
        packet.update({
            f'{name}_world_mid': positions,
            f'{name}_velocity_world_mps': velocity,
            f'{name}_acceleration_world_mps2': acceleration,
        })

    kinematics('P', root)
    kinematics('G', ground)
    kinematics('com_minus_P', packet['com_world_mid']-root)
    kinematics('body_mass_offset_from_P', body_offset)
    kinematics('shoe_mass_offset_from_P', shoe_offset)
    acc = lambda name: packet[f'{name}_acceleration_world_mps2']
    np.testing.assert_allclose(acc('com'), acc('P')+acc('com_minus_P'), atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        acc('com_minus_P'), acc('body_mass_offset_from_P')+acc('shoe_mass_offset_from_P'),
        atol=1e-9, rtol=0,
    )
    packet.update(
        pts=pts, source_time_base=np.array([numerator, denominator]), source_seconds=seconds,
        com_specific_force_interpretation=np.array(
            'Model kinematic estimate of required non-gravitational external force per unit mass. '
            'Direct blade or plantar force measurements are unavailable.'
        ),
        **source_annotations(reference, evidence),
    )
    return packet


def peak_record(packet, candidates):
    acc = packet['com_acceleration_world_mps2']
    magnitude = np.linalg.norm(acc, axis=1)
    frame = int(candidates[np.argmax(magnitude[candidates])])
    unit = acc[frame]/max(magnitude[frame], np.finfo(float).tiny)
    names = ('P', 'G', 'com', 'com_minus_P', 'body_mass_offset_from_P', 'shoe_mass_offset_from_P')
    vectors = {name: packet[f'{name}_acceleration_world_mps2'][frame] for name in names}
    root_norm = float(np.linalg.norm(vectors['P']))
    offset_norm = float(np.linalg.norm(vectors['com_minus_P']))
    nearby = range(max(0, frame-1), min(len(acc), frame+2))
    membership = packet['flight_episode_membership'][frame]
    return dict(
        source_index=frame,
        source_pts=int(packet['pts'][frame]),
        source_seconds=float(packet['source_seconds'][frame]),
        shot_id=str(packet['shot_id'][frame]),
        near_cut=bool(packet['near_cut'][frame]),
        acceleration_stencil_crosses_cut=bool(packet['acceleration_stencil_crosses_cut'][frame]),
        flight_episodes=packet['flight_episode_ids'][membership].tolist(),
        acceleration_xyz_mps2={name: vector.tolist() for name, vector in vectors.items()},
        com_acceleration_magnitude_mps2=float(magnitude[frame]),
        root_translation_magnitude_mps2=root_norm,
        relative_pose_and_shoe_magnitude_mps2=offset_norm,
        larger_vector_norm='root_translation' if root_norm > offset_norm else 'relative_pose_and_shoe',
        contribution_along_com_acceleration_mps2={
            name: float(vectors[name]@unit)
            for name in ('P', 'com_minus_P', 'body_mass_offset_from_P', 'shoe_mass_offset_from_P')
        },
        source_stencil=[
            dict(
                source_pts=int(packet['pts'][i]),
                shot_id=str(packet['shot_id'][i]),
                P_world_m=packet['P_world_mid'][i].tolist(),
                com_minus_P_world_m=packet['com_minus_P_world_mid'][i].tolist(),
            )
            for i in nearby
        ],
    )


def motion_summary(packet):
    clock = packet['source_seconds']
    interior = packet['com_derivative_interior']
    lsp = np.flatnonzero(interior & (clock >= LSP_WINDOW_S[0]) & (clock <= LSP_WINDOW_S[1]))
    return dict(
        source_frames=len(clock),
        source_pts_first=int(packet['pts'][0]),
        source_pts_last=int(packet['pts'][-1]),
        source_time_base=packet['source_time_base'].tolist(),
        cut_neighborhood_s=CUT_NEIGHBORHOOD_S,
        cut_boundaries=[
            dict(
                first_source_pts=int(packet['pts'][i]), source_seconds=float(clock[i]),
                from_shot=str(packet['shot_id'][i-1]), to_shot=str(packet['shot_id'][i]),
                neighborhood_frames=int(packet['cut_neighborhood_membership'][:, k].sum()),
            )
            for k, i in enumerate(packet['cut_source_indices'])
        ],
        flight_frames_by_episode=dict(zip(
            packet['flight_episode_ids'].tolist(), packet['flight_episode_membership'].sum(axis=0).tolist(),
        )),
        kinematics_definition=dict(
            P='Pelvis root in world metres.',
            G='Vertical projection of P onto world Z=0.',
            COM='Body segment volume proxy plus two rigid shoes in the same fitted state.',
            body_and_shoe_offsets='Separate mass-weighted contributions to COM minus P, in world coordinates.',
            derivative=str(packet['com_derivative_method']),
            peak_scope='Interior source frames. Endpoint acceleration values repeat the adjacent interior samples.',
            specific_H='Model mass-normalized angular momentum about COM in m²/s, with undefined endpoints.',
            required_specific_force=str(packet['com_specific_force_interpretation']),
            interpretation='Model trajectory diagnostics. Root and relative offset are a kinematic decomposition.',
        ),
        worst_whole_programme=peak_record(packet, np.flatnonzero(interior)),
        LSp_source_window_seconds=list(LSP_WINDOW_S),
        worst_LSp=peak_record(packet, lsp) if len(lsp) else None,
    )


def plot_motion(packet, summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    clock = packet['source_seconds']
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 8), layout='constrained')
    for name, label, color, style in (
        ('P', 'P (pelvis)', '#76559c', '-'),
        ('G', 'G (pelvis projection)', '#72767e', '--'),
        ('com', 'COM (model)', '#087e83', '-'),
    ):
        speed = np.linalg.norm(packet[f'{name}_velocity_world_mps'][:, :2], axis=1)
        height = packet[f'{name}_world_mid'][:, 2]
        acceleration = np.linalg.norm(packet[f'{name}_acceleration_world_mps2'], axis=1)
        for ax, values in zip(axes, (speed, height, acceleration)):
            ax.plot(clock, values, color=color, linestyle=style, linewidth=1, label=label)
    for ax, label in zip(axes, ('Horizontal speed (m/s)', 'World Z (m)', 'Acceleration magnitude (m/s²)')):
        ax.set_ylabel(label)
        ax.grid(alpha=.17)
        for index, boundary in enumerate(summary['cut_boundaries']):
            label = 'Source cut' if index == 0 and ax is axes[0] else None
            ax.axvline(boundary['source_seconds'], color='#9d7556', alpha=.5, linewidth=.8, label=label)
    for key, label in (('worst_whole_programme', 'Programme peak'), ('worst_LSp', 'LSp peak')):
        peak = summary[key]
        if peak is not None:
            x, y = peak['source_seconds'], peak['com_acceleration_magnitude_mps2']
            axes[-1].scatter([x], [y], color='#087e83', s=18, zorder=4)
            axes[-1].annotate(
                f'{label}  {x:.2f} s', xy=(x, y), xytext=(-6, 8), textcoords='offset points',
                ha='right', fontsize=8,
            )
    axes[0].legend(loc='upper right', frameon=False, ncols=4)
    axes[-1].margins(y=.10)
    axes[-1].set_xlabel('Original source time (s)')
    axes[-1].set_xlim(clock[0], clock[-1])
    fig.suptitle('Saved programme candidate kinematics')
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'prepared', 'parameter', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-evidence', type=Path, help='Override the prepared source annotation ledger')
    args = parser.parse_args()
    problem, _ = saved_problem(args.project, args.prepared, args.parameter)
    prepared_evidence = problem.reference.source_document
    evidence_path = args.source_evidence or args.prepared/'source_evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if args.source_evidence else prepared_evidence
    for key in ('source_id', 'source_sha256', 'time_base'):
        if evidence[key] != prepared_evidence[key]:
            raise ValueError('Source evidence does not identify the prepared source: '+key)
    packet = motion_packet(problem, evidence)
    summary = motion_summary(packet)
    summary.update(
        candidate=dict(path=str(args.parameter.resolve()), sha256=digest(args.parameter)),
        prepared=str(args.prepared.resolve()),
        source_evidence=dict(path=str(evidence_path.resolve()), sha256=digest(evidence_path)),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output/'programme_com_motion'
    plot_motion(packet, summary, stem.with_suffix('.png'))
    np.savez_compressed(stem.with_suffix('.npz'), **packet)
    stem.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(stem.with_suffix('.json').resolve()), source_frames=len(packet['pts']))))


if __name__ == '__main__':
    main()
