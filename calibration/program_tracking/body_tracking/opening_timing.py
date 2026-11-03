"""Opening phase repair from the user's source-timed motion correction."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.spatial.transform import Rotation

from .opening_motion import (MotionReference, MotionProblem, START, END,
    ATTACH_START, RELEASE_END, interpolate_motion, build as build_motion)
from .opening_repair import smoothstep
from .fullbody_state import matrices
from .fullbody_fit import verify_problem
from .boot_integration import export
from .fullbody_round import write_json
from .rebuild_prepare import load, digest

QUIET_END = 22.90
QUIET_BODY_JOINTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
QUIET_POSITION_SCALE = .01
QUIET_ROTATION_SCALE_DEG = 1.


def quiet_state_prior(state, times, origin=START):
    """Constant *actual* body state before22.90; its common value stays free.

    The reference has constant P/Q/trunk/legs in this phase. Therefore equal
    right corrections imply equal actual poses. Hands, arms, neck and head are
    excluded, and no row beyond the source-confirmed quiet phase is added.
    """
    times = np.asarray(times)
    selected = times + origin <= QUIET_END + 1e-10
    B = state.basis(times[selected]) - state.basis([START-origin])
    body = np.r_[np.arange(6),
        [6+(j-1)*3+a for j in QUIET_BODY_JOINTS for a in range(3)]]
    scale = np.where(state.active < 3, QUIET_POSITION_SCALE,
                     np.deg2rad(QUIET_ROTATION_SCALE_DEG))
    inverse = np.isin(state.active, body)/scale
    return kron(csr_matrix(B), csr_matrix(np.diag(inverse)), format='csr')/np.sqrt(2)


class TimingReference(MotionReference):
    def __init__(self, old, camera, packet):
        super().__init__(old, camera, packet)
        self.quiet_raw_Q, pivot = interpolate_motion(packet, [QUIET_END])
        self.quiet_raw_local = pivot[0]
        self.initial_body_local = self.initial_raw_local.copy()
        for j in (1, 2, 4, 5, 7, 8, 10, 11):
            self.initial_body_local[j-1] = self.preparation['local_R'][j-1]
        self.seed_first = self.old.at([0.])['P'][0]
        self.seed_pivot = self.old.at([QUIET_END-START])['P'][0]

    def at(self, t):
        t = np.atleast_1d(t).astype(float)
        base = self.parent.at(t)
        seconds = t+self.origin
        selected = seconds < END
        out = {key: base[key].copy() for key in ('P', 'Q', 'local_R')}
        out['times'] = t
        if selected.any():
            source = seconds[selected]
            raw_Q, local = interpolate_motion(self.packet, source)
            phase = np.maximum(source, QUIET_END)
            phase_Q, phase_local = interpolate_motion(self.packet, phase)
            release = 1-smoothstep((source-QUIET_END)/(RELEASE_END-QUIET_END))
            # Preserve the opening body until22.90, then transfer observed raw
            # increments relative to22.90. The alignment releases by24.60.
            target_Q = self.initial_raw_Q @ self.quiet_raw_Q.transpose(0, 2, 1) @ phase_Q
            change = Rotation.from_matrix(raw_Q.transpose(0, 2, 1)@target_Q).as_rotvec()
            Q = raw_Q@matrices(release[:, None]*change)
            for j in QUIET_BODY_JOINTS:
                target = self.initial_body_local[j-1] @ self.quiet_raw_local[j-1].T @ phase_local[:, j-1]
                change = Rotation.from_matrix(local[:, j-1].transpose(0, 2, 1)@target).as_rotvec()
                local[:, j-1] = local[:, j-1]@matrices(release[:, None]*change)
            seed = self.old.at(t[selected])['P'].copy()
            quiet = source <= QUIET_END
            seed[:, :2] = self.seed_first[:2]+seed[:, :2]-self.seed_pivot[:2]
            seed[quiet, :2] = self.seed_first[:2]
            trial = self.world(dict(P=np.zeros_like(seed), Q=Q, local_R=local, times=t[selected]))
            seed[:, 2] = -np.min(trial['gear'][:, :, 14:, 2], axis=(1, 2))
            attach = 1-smoothstep((source-ATTACH_START)/(END-ATTACH_START))
            out['P'][selected] = base['P'][selected]+attach[:, None]*(seed-base['P'][selected])
            for key, target in [('Q', Q), ('local_R', local)]:
                original = base[key][selected]
                rv = Rotation.from_matrix((np.swapaxes(original, -1, -2)@target).reshape(-1, 3, 3)).as_rotvec()
                rv = rv.reshape(*target.shape[:-2], 3)
                factor = attach.reshape((len(attach),)+(1,)*(rv.ndim-1))
                out[key][selected] = original@matrices(factor*rv)
        for state, x in self.refinements:
            delta = state.increments(x, t)
            out['P'] += delta[:, :3]
            out['Q'] = out['Q']@matrices(delta[:, 3:6])
            out['local_R'] = out['local_R']@matrices(delta[:, 6:].reshape(-1, 21, 3))
        return out


class TimingProblem(MotionProblem):
    def __init__(self, reference, camera, observations, log=None):
        super().__init__(reference, camera, observations, log)
        self.quiet = reference.p['pts'][self.frames]/12800. <= QUIET_END+1e-10
        self.prior_matrices.append(('source_confirmed_quiet_body',
            quiet_state_prior(self.state, self.mid_times, reference.origin)))

    def physical_rows(self, delta):
        rows = super().physical_rows(delta)
        # Standing during this source-confirmed phase supports a lowest-blade
        # envelope. It does not identify a physical blade station or left foot.
        rows[self.quiet, 2] *= 8.  # 8cm -> 1cm engineering support scale
        return rows


def build(project, root, out):
    previous, camera, info, catalog = build_motion(project, root, out)
    return TimingReference(previous.old, camera, previous.packet), camera, info, catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'export', 'run'], default='run')
    parser.add_argument('--warm-start')
    args = parser.parse_args(argv)
    project, root, out = map(Path, (args.project, args.root, args.output))
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))), flush=True)
    reference, camera, info, _ = build(project, root, out)
    obs_path = root/'boot_integration_1102/observations.npz'
    previous_contract = json.loads((root/'opening_motion_1103/inputs.json').read_text(encoding='utf8'))
    contract = dict(previous_contract, schema='rinkspace.opening-timing-input.v1',
        implementation_sha256=digest(__file__), base_implementation_sha256=digest(Path(__file__).with_name('opening_motion.py')),
        quiet_phase_seconds=[START, QUIET_END],
        quiet_body_native_joints=list(QUIET_BODY_JOINTS),
        quiet_position_scale_m=QUIET_POSITION_SCALE,
        quiet_rotation_scale_deg=QUIET_ROTATION_SCALE_DEG,
        quiet_support_scale_m=.01,
        actual_velocity_factor=False, fixed_location_factor=False,
        quiet_body_time_equality_factor=True,
        user_correction='Until about22.90 only hands move; lowering begins afterwards. Reject previous1.11m opening displacement.')
    if (out/'inputs.json').is_file() and json.loads((out/'inputs.json').read_text(encoding='utf8')) != contract:
        raise ValueError('Timing inputs changed; preserve previous candidate')
    write_json(out/'inputs.json', contract)
    np.savez_compressed(out/'motion_reference.npz', **reference.packet)
    np.savez_compressed(out/'observations.npz', **load(obs_path))
    ledger = json.loads((root/'opening_motion_1103/ledger.json').read_text(encoding='utf8'))
    ledger.update(opening_motion_contract=contract,
        opening_timing_user_correction=contract['user_correction'],
        rejected_candidate=str(root/'opening_motion_1103/fitted/result.npz'))
    write_json(out/'ledger.json', ledger)
    if args.stage == 'prepare':
        return
    with (out/'fit_progress.jsonl').open('a', encoding='utf8', buffering=1) as log:
        problem = TimingProblem(reference, camera, out/'observations.npz', log)
        if (out/'fit/result.npz').is_file():
            fit = load(out/'fit/result.npz')
            np.testing.assert_array_equal(fit['knots'], problem.state.knots)
            np.testing.assert_array_equal(fit['active'], problem.state.active)
            x = fit['x']
            report = json.loads((out/'fit/result.json').read_text(encoding='utf8'))
        elif args.stage in ('run', 'fit'):
            print(json.dumps(dict(derivatives=verify_problem(problem, out/'derivative_checks.json'))), flush=True)
            initial = None
            if args.warm_start:
                saved = load(args.warm_start)
                np.testing.assert_array_equal(saved['knots'], problem.state.knots)
                np.testing.assert_array_equal(saved['active'], problem.state.active)
                initial = saved['x'].reshape(problem.state.m, problem.state.d).copy()
                body_columns = np.r_[np.arange(6),
                    [6+(j-1)*3+a for j in QUIET_BODY_JOINTS for a in range(3)]]
                initial[:, np.isin(problem.state.active, body_columns)] = 0.
                initial = initial.ravel()
                write_json(out/'initialization.json', dict(path=str(Path(args.warm_start).resolve()),
                    sha256=digest(args.warm_start), scope='Only arms, wrists, neck/head controls; body controls reset for corrected source timing'))
            x, report = problem.solve(out/'fit', initial_x=initial)
            print(json.dumps(report), flush=True)
        else:
            raise FileNotFoundError('No fitted phase-timing candidate')
    write_json(out/'joint_result.json', report)
    reference.refinements.append((problem.state, x))
    if args.stage in ('run', 'export'):
        config = json.loads((out/'fit/config.json').read_text(encoding='utf8'))
        config.update(mode='source_timed_opening', motion_contract=contract,
            frozen=['camera', 'scene', 'beta', 'body scale', 'bone lengths', 'shared foot-boot registration'],
            wrist_orientation='Dynamic cached wrist plus optimized right wrist and sliding hand-waist relation; fingers default',
            quiet_phase='P/Q/trunk/legs share a freely fitted opening state until22.90; arms and hands remain dynamic',
            boundary='C2 return to boot parent at25s; all arrays exact from25s inclusive')
        write_json(out/'fit/config.json', config)
        packet = export(reference, camera, info, out,
            source_path=root/'boot_integration_1102/fitted/result.npz')
        packet.update(opening_inferred_mask=reference.p['lower_body_inferred'],
            opening_motion_reference_json=np.array(json.dumps(contract)),
            stopping_reason_json=np.array(json.dumps(report['stopping_reason'])),
            solver_config_json=np.array(json.dumps(dict(opening='fit/config.json', result='fit/result.json', ledger='ledger.json'))))
        np.savez_compressed(out/'fitted/result.npz', **packet)
        from .opening_motion_review import run as review
        review(root, out)
        print(json.dumps(dict(export=str(out/'fitted/result.npz'), frames=len(packet['pts']))), flush=True)


if __name__ == '__main__':
    main()
