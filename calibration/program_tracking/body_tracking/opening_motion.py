"""Time-varying opening articulation on the frozen body/camera/boot model.

The old opening is only a translation seed. Its static pose, actual-velocity,
upright and fixed-position factors are not part of this problem.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.transform import Rotation, Slerp
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from .rebuild_prepare import load, digest
from .fullbody_state import matrices
from .fullbody_fit import FullBodyProblem, verify_problem
from .opening_repair import setup, OpeningWindow, START, END, smoothstep
from .boot_integration import edges_for, export
from .fullbody_round import write_json

RELEASE_END = 24.60
ATTACH_START = 24.44
LOWER_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
HAND_RELEASE_START, HAND_RELEASE_END = 23.10, 23.40


def surface_contact_vectors(palms, waist, softness=.03):
    """Sliding skin contact; no fixed world point or preferred waist vertex."""
    difference = palms[:, :, None]-waist[:, None]
    exponent = -np.sum(difference**2, axis=-1)/(2*softness**2)
    weights = np.exp(exponent-logsumexp(exponent, axis=2, keepdims=True))
    return np.sum(weights[..., None]*difference, axis=2)


class MotionWindow(OpeningWindow):
    def __init__(self, reference, seconds, mode):
        super().__init__(reference, seconds, mode)
        # Source wrist pixels do not measure hand orientation. The observed
        # hand/waist relationship and dynamic reference inform this wrist.
        self.active = np.r_[self.active, [66, 67, 68]]
        self.d = len(self.active)
        self.size = self.m*self.d


def interpolate_motion(packet, times):
    """Evaluate the exact, source-mapped raw bracket; refuse extrapolation."""
    np.testing.assert_array_equal(packet['source_time_base'], [1, 12800])
    stamps = packet['raw_official_pts']/12800.
    times = np.atleast_1d(times).astype(float)
    if np.any(times < stamps[0]) or np.any(times > stamps[-1]):
        raise ValueError('Raw motion does not bracket requested source times')
    local = np.stack([Slerp(stamps, Rotation.from_matrix(packet['local_R'][:, j]))(
        times).as_matrix() for j in range(21)], axis=1)
    Q = Slerp(stamps, Rotation.from_matrix(packet['Q_world']))(times).as_matrix()
    return Q, local


def motion_packet(parent, camera):
    """Recover only affected pose rows, using the existing confirmed clock map."""
    raw_path = parent.root/'source_cache/raw_shot_01.npz'
    raw = load(raw_path)
    # This older cache's official_exposure_representative_pts is in the old
    # clock. Never treat that misnamed field as current official source PTS.
    source = json.loads((parent.root/'input.json').read_text(encoding='utf8'))
    ledger = Path(next(p for p in source['frozen_inputs'] if p.endswith('native60_to_official50.jsonl')))
    mapping = {}
    for line in ledger.read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        if row.get('accepted_official_pts') is not None and row['shot_index'] == 1:
            mapping[int(row['full_pts'])] = int(row['accepted_official_pts'])
    pairs = sorted((mapping[int(p)], i) for i, p in enumerate(raw['source_pts'])
                   if int(p) in mapping and raw['chosen_window'][i] >= 0)
    pairs = [(p, i) for k, (p, i) in enumerate(pairs) if k == 0 or p != pairs[k-1][0]]
    stamps = np.array([p for p, i in pairs], np.int64)
    rows = np.array([i for p, i in pairs])
    # Keep one genuine raw sample on either side of the affected window,
    # including the camera readout halo. This is not a programme re-prepare.
    first = max(0, int(np.searchsorted(stamps, round((START-.05)*12800)))-1)
    stop = min(len(stamps), int(np.searchsorted(stamps, round((END+.05)*12800)))+1)
    stamps, rows = stamps[first:stop], rows[first:stop]
    if len(stamps) < 2 or not np.all(np.diff(stamps) > 0):
        raise ValueError('Raw official source PTS are not unique and increasing')
    # The selected projector already contains the current shot's time shift.
    shift = float(camera.p['motion_times'][0])-float(parent.p['pts'][0])/12800.
    pose = camera.core.evaluate_motion_camera(camera.p['rig'], camera.p['controls'],
        float(camera.p['first_control_time']), float(camera.p['control_dt']), stamps/12800.+shift)
    if pose['status'].any():
        raise ValueError('Current camera failed at a dynamic raw exposure')
    return dict(raw_official_pts=stamps, raw_legacy_pts=raw['source_pts'][rows],
        raw_cache_rows=rows, local_R=matrices(raw['body_pose'][rows].reshape(-1, 21, 3)),
        Q_world=pose['rotation'].transpose(0, 2, 1)@matrices(raw['global_orient'][rows]),
        source_time_base=np.array([1, 12800]),
        raw_cache_sha256=np.array(digest(raw_path)), clock_map_sha256=np.array(digest(ledger)))


class MotionReference:
    def __init__(self, old, camera, packet):
        self.old = old
        self.parent = old.parent
        self.refinements = []
        self.packet = packet
        self.origin = START
        self.initial_raw_Q, initial = interpolate_motion(packet, [START])
        self.initial_raw_local = initial[0]
        self.preparation = old.preparation
        if self.preparation is None:
            raise ValueError('Existing preparation pose is missing')

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def at(self, t):
        t = np.atleast_1d(t).astype(float)
        base = self.parent.at(t)
        source = t+self.origin
        selected = source < END
        out = {key: base[key].copy() for key in ('P', 'Q', 'local_R')}
        out['times'] = t
        if selected.any():
            seconds = source[selected]
            Q, local = interpolate_motion(self.packet, seconds)
            # Preparation defines only an inferred initial lower pose. Retain
            # raw *changes* throughout the release, rather than freezing legs.
            strength = 1-smoothstep((seconds-START)/(RELEASE_END-START))
            for j in LOWER_JOINTS:
                relative = self.initial_raw_local[j-1].T@local[:, j-1]
                transferred = self.preparation['local_R'][j-1]@relative
                change = Rotation.from_matrix(local[:, j-1].transpose(0, 2, 1)@transferred).as_rotvec()
                local[:, j-1] = local[:, j-1]@matrices(strength[:, None]*change)
            seed = self.old.at(t[selected])['P'].copy()
            trial = self.world(dict(P=np.zeros_like(seed), Q=Q, local_R=local, times=t[selected]))
            # Height follows changing legs. This is a coarse support seed,
            # subsequently optimized; no fixed pelvis height or velocity.
            seed[:, 2] = -np.min(trial['gear'][:, :, 14:, 2], axis=(1, 2))
            weight = 1-smoothstep((seconds-ATTACH_START)/(END-ATTACH_START))
            out['P'][selected] = base['P'][selected]+weight[:, None]*(seed-base['P'][selected])
            for key, target in [('Q', Q), ('local_R', local)]:
                original = base[key][selected]
                rv = Rotation.from_matrix((np.swapaxes(original, -1, -2)@target).reshape(-1, 3, 3)).as_rotvec()
                rv = rv.reshape(*target.shape[:-2], 3)
                factor = weight.reshape((len(weight),)+(1,)*(rv.ndim-1))
                out[key][selected] = original@matrices(factor*rv)
        for state, x in self.refinements:
            delta = state.increments(x, t)
            out['P'] += delta[:, :3]
            out['Q'] = out['Q']@matrices(delta[:, 3:6])
            out['local_R'] = out['local_R']@matrices(delta[:, 6:].reshape(-1, 21, 3))
        return out

    def world(self, reference, delta=None):
        return self.parent.world(reference, delta)


class MotionProblem(FullBodyProblem):
    def __init__(self, reference, camera, observations, log=None):
        super().__init__(reference, camera, observations, (START, END), 'full', log,
                         state_factory=MotionWindow)
        self.mid_times = reference.times[self.frames]
        self.mid_ref = reference.at(self.mid_times)
        self.support_mask = reference.p['pts'][self.frames]/12800. <= ATTACH_START
        seconds = reference.p['pts'][self.frames]/12800.
        self.hand_weight = 1-smoothstep((seconds-HAND_RELEASE_START)/(HAND_RELEASE_END-HAND_RELEASE_START))
        surface = reference.binding.surface
        shaped = surface.shaped
        rest = (shaped-surface.body.J[0])*surface.body.scale
        torso = surface.weights[:, [0, 1, 2, 3, 6, 9]].sum(axis=1) > .8
        # The right lateral waist between pelvis and lower chest, in the
        # existing model's rest frame. Contact may slide over this whole patch.
        self.waist_ids = np.flatnonzero(torso & (rest[:, 0] < -.05) &
            (rest[:, 1] > 0) & (rest[:, 1] < .28))
        hand = surface.weights[:, [21, *range(40, 55)]].sum(axis=1) > .8
        hand_ids = np.flatnonzero(hand)
        # Three palm centres between the wrist and finger-base joints. Each is
        # an average of neighbouring skin vertices, not an invented wrist UV.
        centres = .45*surface.body.J[21]+.55*surface.body.J[[40, 43, 49]]
        _, nearest = cKDTree(shaped[hand_ids]).query(centres, k=8)
        self.palm_ids = hand_ids[nearest]
        if not len(self.waist_ids):
            raise ValueError('Body model has no right waist surface patch')

    def visible_boot_edges(self, world):
        return edges_for(world, self.frames, self.projector, self.reference.binding)

    def physical_rows(self, delta):
        world = self.reference.world(self.mid_ref, delta)
        # A 1mm soft minimum keeps the derivative defined when blade samples
        # exchange the lowest position. It is an explicit factor approximation,
        # not a contact measurement or an accuracy tolerance.
        feet = -.001*logsumexp(-world['gear'][:, :, 14:, 2]/.001, axis=2)
        support = -.001*logsumexp(-feet/.001, axis=1)
        depth = np.maximum(-feet, 0)
        penetration = np.where(depth < .001, depth**2/.002, depth-.0005)
        local_joints = np.einsum('nji,nkj->nki', world['Q'], world['native']-world['P'][:, None])
        ids = np.r_[self.waist_ids, self.palm_ids.ravel()]
        surface = self.reference.binding.surface.sample_vertices(local_joints,
            world['frames'], vertex_ids=ids, double_precision=True)
        waist = surface[:, :len(self.waist_ids)]
        palms = surface[:, len(self.waist_ids):].reshape(self.n, 3, 8, 3).mean(axis=2)
        hand = surface_contact_vectors(palms, waist).reshape(self.n, 9)
        hand = hand*self.hand_weight[:, None]/(.02*np.sqrt(3))
        # Geometric nonpenetration plus a weak lowest-blade support envelope.
        # No torso-upright, fixed location, or actual-velocity factor.
        return np.c_[penetration/.03,
            support*self.support_mask/.08, hand]/np.sqrt(2)

    def evaluate(self, x, jacobian=False):
        stream = self.log
        self.log = None
        try:
            base = super().evaluate(x, jacobian)
        finally:
            self.log = stream
        if base.get('motion_added'):
            return base
        delta = self.state.increments(x, self.mid_times)
        physical = self.physical_rows(delta)
        if jacobian:
            columns = []
            h = 1e-5
            for active in self.state.active:
                plus, minus = delta.copy(), delta.copy()
                plus[:, active] += h
                minus[:, active] -= h
                columns.append((self.physical_rows(plus)-self.physical_rows(minus))/(2*h))
            local = np.stack(columns, -1)
            B = self.state.basis(self.mid_times)
            frame, knot = np.nonzero(B)
            values = local[frame]*B[frame, knot, None, None]
            rr = np.broadcast_to(frame[:, None, None]*physical.shape[1]+np.arange(physical.shape[1])[None, :, None], values.shape)
            cc = np.broadcast_to(knot[:, None, None]*self.state.d+np.arange(self.state.d), values.shape)
            J = coo_matrix((values.ravel(), (rr.ravel(), cc.ravel())), shape=(physical.size, self.size)).tocsr()
            base['jac'] = vstack((base['jac'], J), format='csr')
        base['residual'] = np.r_[base['residual'], physical.ravel()]
        base['parts']['blade_plane_envelope'] = float(np.sum(physical[:, :3]**2)/2)
        base['parts']['observed_right_hand_waist_relation'] = float(np.sum(physical[:, 3:]**2)/2)
        base['motion_added'] = True
        if stream:
            stream.write(json.dumps(dict(event='motion_evaluation', evaluation=self.evaluations,
                with_jacobian=jacobian, cost=float(base['residual']@base['residual']/2),
                parts=base['parts'], evaluation_seconds=base['eval_seconds'],
                variables=self.size, source_frames=self.n, domain_failures=0,
                seconds=time.perf_counter()-self.start))+'\n')
        return base


def build(project, root, out):
    old, camera, info, catalog = setup(project, root, out)
    fitted = load(root/'opening_repair_1102/fit/result.npz')
    state = OpeningWindow(old, (START, END), 'full')
    np.testing.assert_array_equal(state.knots, fitted['knots'])
    np.testing.assert_array_equal(state.active, fitted['active'])
    old.refinements.append((state, fitted['x']))
    packet = motion_packet(old.parent, camera)
    return MotionReference(old, camera, packet), camera, info, catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'export', 'run'], default='run')
    parser.add_argument('--warm-start', help='Existing accepted controls; remap common columns explicitly')
    args = parser.parse_args(argv)
    project, root, out = Path(args.project), Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))), flush=True)
    reference, camera, info, catalog = build(project, root, out)
    obs_path = root/'boot_integration_1102/observations.npz'
    contract = dict(schema='rinkspace.opening-motion-input.v1', camera=info,
        parent_sha256=digest(root/'boot_integration_1102/fitted/result.npz'),
        translation_seed_sha256=digest(root/'opening_repair_1102/fit/result.npz'),
        raw_cache_sha256=str(reference.packet['raw_cache_sha256']),
        clock_map_sha256=str(reference.packet['clock_map_sha256']),
        preparation_sha256=digest(reference.old.preparation_path),
        observations_sha256=digest(obs_path), source_seconds=[START, END],
        lower_pose_release_end=RELEASE_END, parent_attachment_start=ATTACH_START,
        actual_velocity_factor=False, fixed_location_factor=False, upright_factor=False,
        support_scale_m=.08, penetration_scale_m=.03,
        blade_minimum_softness_m=.001, penetration_C1_transition_m=.001,
        hand_waist_relation=dict(hold_until=HAND_RELEASE_START, released_at=HAND_RELEASE_END,
            surface_distance_scale_m=.02, sliding_softness_m=.03, palm_skin_patches=3,
            evidence='Existing19.960s preparation image and source22.12-23.08s; user confirms right hand on waist'),
        implementation_sha256=digest(__file__))
    if (out/'inputs.json').is_file() and json.loads((out/'inputs.json').read_text(encoding='utf8')) != contract:
        raise ValueError('Motion inputs changed; preserve the previous fitted candidate')
    write_json(out/'inputs.json', contract)
    np.savez_compressed(out/'motion_reference.npz', **reference.packet)
    np.savez_compressed(out/'observations.npz', **load(obs_path))
    ledger = json.loads((root/'boot_integration_1102/ledger.json').read_text(encoding='utf8'))
    ledger.update(parent_ledger=str(root/'boot_integration_1102/ledger.json'),
        opening_user_evidence='Right hand rests on waist, opens with turning; small lowering then rising; natural translation allowed',
        opening_motion_contract=contract, lower_body_inference_preserved=True)
    write_json(out/'ledger.json', ledger)
    if args.stage == 'prepare':
        return
    with (out/'fit_progress.jsonl').open('a', encoding='utf8', buffering=1) as log:
        problem = MotionProblem(reference, camera, out/'observations.npz', log)
        if (out/'fit/result.npz').is_file():
            fit = load(out/'fit/result.npz')
            np.testing.assert_array_equal(fit['knots'], problem.state.knots)
            np.testing.assert_array_equal(fit['active'], problem.state.active)
            x = fit['x']
            report = json.loads((out/'fit/result.json').read_text(encoding='utf8'))
        elif args.stage in ('fit', 'run'):
            print(json.dumps(dict(derivatives=verify_problem(problem, out/'derivative_checks.json'))), flush=True)
            initial = None
            if args.warm_start:
                previous = load(args.warm_start)
                np.testing.assert_array_equal(previous['knots'], problem.state.knots)
                initial = np.zeros((problem.state.m, problem.state.d))
                controls = previous['x'].reshape(problem.state.m, -1)
                for index, active in enumerate(previous['active']):
                    initial[:, np.flatnonzero(problem.state.active == active)[0]] = controls[:, index]
                initial = initial.ravel()
                write_json(out/'initialization.json', dict(path=str(Path(args.warm_start).resolve()),
                    sha256=digest(args.warm_start), new_right_wrist_controls_zero=True))
            x, report = problem.solve(out/'fit', initial_x=initial)
            print(json.dumps(report), flush=True)
        else:
            raise FileNotFoundError('No fitted motion candidate')
    write_json(out/'joint_result.json', report)
    reference.refinements.append((problem.state, x))
    if args.stage in ('export', 'run'):
        config = json.loads((out/'fit/config.json').read_text(encoding='utf8'))
        config.update(mode='dynamic_opening', frozen=['camera', 'scene', 'beta', 'body scale',
            'bone lengths', 'shared foot-boot registration'],
            wrist_orientation='dynamic cached local rotation; not directly observed',
            active_head_neck=True, physical_contact_factor='weak blade plane envelope; physical contact identity unknown',
            actual_velocity_factor=False, fixed_location_factor=False, upright_factor=False,
            motion_contract=contract)
        write_json(out/'fit/config.json', config)
        packet = export(reference, camera, info, out,
                        source_path=root/'boot_integration_1102/fitted/result.npz')
        packet.update(opening_inferred_mask=reference.p['lower_body_inferred'],
            opening_motion_reference_json=np.array(json.dumps(contract)),
            stopping_reason_json=np.array(json.dumps(report['stopping_reason'])),
            costs_json=np.array(json.dumps(report['final_parts'])))
        np.savez_compressed(out/'fitted/result.npz', **packet)
        write_json(out/'binding.json', reference.binding.record())
        from .opening_motion_review import run as review
        print(json.dumps(review(root, out)), flush=True)
        print(json.dumps(dict(export=str(out/'fitted/result.npz'), frames=len(packet['pts']))), flush=True)


if __name__ == '__main__':
    main()
