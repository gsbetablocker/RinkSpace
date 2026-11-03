"""Right-arm-only opening polish on the saved, source-timed body state."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from .opening_timing import build as build_timing, START, END
from .opening_motion import (MotionProblem, MotionWindow, HAND_RELEASE_START,
    HAND_RELEASE_END)
from .opening_repair import smoothstep
from .fullbody_state import WindowState
from .fullbody_fit import FullBodyProblem, verify_problem
from .fullbody_round import write_json
from .rebuild_prepare import load, digest
from .boot_integration import export

HAND_END = HAND_RELEASE_END
RIGHT_ARM_JOINTS = (17, 19, 21)


class HandWindow(WindowState):
    def __init__(self, reference, seconds=None, mode='full'):
        super().__init__(reference, (START-.52, HAND_END), mode,
                         spacing=.20, taper=.25)
        self.seconds = (START, HAND_END)
        self.active = np.array([6+(j-1)*3+a for j in RIGHT_ARM_JOINTS for a in range(3)])
        self.d = len(self.active)
        self.size = self.m*self.d


class HandProblem(MotionProblem):
    def __init__(self, reference, camera, observations, log=None):
        FullBodyProblem.__init__(self, reference, camera, observations,
            (START, HAND_END), 'full', log, state_factory=HandWindow)
        self.mid_times = reference.times[self.frames]
        self.mid_ref = reference.at(self.mid_times)
        self.support_mask = np.ones(self.n, bool)
        seconds = reference.p['pts'][self.frames]/12800.
        self.hand_weight = 1-smoothstep((seconds-HAND_RELEASE_START)/(HAND_RELEASE_END-HAND_RELEASE_START))
        surface = reference.binding.surface
        rest = (surface.shaped-surface.body.J[0])*surface.body.scale
        torso = surface.weights[:, [0, 1, 2, 3, 6, 9]].sum(axis=1) > .8
        # The palm/finger bases can extend below pelvis height while resting
        # on the upper hip. The former y>0 patch forced this hand upward.
        self.waist_ids = np.flatnonzero(torso & (rest[:, 0] < -.05) &
            (rest[:, 1] > -.12) & (rest[:, 1] < .28))
        hand_ids = np.flatnonzero(surface.weights[:, [21, *range(40, 55)]].sum(axis=1) > .8)
        centres = .45*surface.body.J[21]+.55*surface.body.J[[40, 43, 49]]
        _, nearest = cKDTree(surface.shaped[hand_ids]).query(centres, k=8)
        self.palm_ids = hand_ids[nearest]
        if not len(self.waist_ids):
            raise ValueError('Missing right waist/upper-hip skin')


def build(project, root, out):
    reference, camera, info, catalog = build_timing(project, root, out)
    parameters = Path(project)/'events/ankara_2026/programmes/okada_mei_sp/opening_timing_1103.npz'
    saved = load(parameters)
    state = MotionWindow(reference, (START, END), 'full')
    np.testing.assert_array_equal(state.knots, saved['knots'])
    np.testing.assert_array_equal(state.active, saved['active'])
    reference.refinements.append((state, saved['x']))
    reference.hashes = dict(reference.hashes)
    reference.hashes[str(parameters)] = digest(parameters)
    return reference, camera, info, catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['run', 'export'], default='run')
    args = parser.parse_args(argv)
    project, root, out = map(Path, (args.project, args.root, args.output))
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))), flush=True)
    reference, camera, info, _ = build(project, root, out)
    base_parameters = project/'events/ankara_2026/programmes/okada_mei_sp/opening_timing_1103.npz'
    contract = json.loads((root/'opening_timing_1103/inputs.json').read_text(encoding='utf8'))
    contract.update(schema='rinkspace.opening-hand-polish-input.v1', camera=info,
        base_timing_parameters=dict(path=str(base_parameters), sha256=digest(base_parameters)),
        body_timing_fit_sha256=digest(root/'opening_timing_1103/fit/result.npz'),
        implementation_sha256=digest(__file__),
        hand_polish_seconds=[START, HAND_END], hand_polish_native_joints=list(RIGHT_ARM_JOINTS),
        hand_contact_patch_rest_y_m=[-.12, .28],
        body_timing_frozen=True,
        hand_polish_reason='Keep source-timed trunk/legs/translation exact; permit fingers-down palm contact on right waist and upper hip')
    if (out/'inputs.json').is_file() and json.loads((out/'inputs.json').read_text(encoding='utf8')) != contract:
        raise ValueError('Hand polish contract changed; preserve previous result')
    write_json(out/'inputs.json', contract)
    np.savez_compressed(out/'motion_reference.npz', **reference.packet)
    np.savez_compressed(out/'observations.npz', **load(root/'boot_integration_1102/observations.npz'))
    ledger = json.loads((root/'opening_timing_1103/ledger.json').read_text(encoding='utf8'))
    ledger.update(opening_hand_polish_contract=contract)
    write_json(out/'ledger.json', ledger)
    with (out/'fit_progress.jsonl').open('a', encoding='utf8', buffering=1) as log:
        problem = HandProblem(reference, camera, out/'observations.npz', log)
        if (out/'fit/result.npz').is_file():
            saved = load(out/'fit/result.npz')
            np.testing.assert_array_equal(saved['knots'], problem.state.knots)
            np.testing.assert_array_equal(saved['active'], problem.state.active)
            x = saved['x']
            report = json.loads((out/'fit/result.json').read_text(encoding='utf8'))
        elif args.stage == 'run':
            print(json.dumps(dict(derivatives=verify_problem(problem, out/'derivative_checks.json'))), flush=True)
            x, report = problem.solve(out/'fit')
            print(json.dumps(report), flush=True)
        else:
            raise FileNotFoundError('No saved hand refinement')
    write_json(out/'joint_result.json', report)
    reference.refinements.append((problem.state, x))
    config = json.loads((out/'fit/config.json').read_text(encoding='utf8'))
    config.update(mode='right_arm_only_on_source_timed_body', motion_contract=contract,
        frozen=['camera', 'scene', 'body scale', 'beta', 'bone lengths', 'root P/Q',
                'trunk/legs', 'left arm', 'head/neck', 'shared foot-boot registration'],
        wrist_orientation='Right wrist optimized; fingers retain existing defaults',
        boundary='Right-arm refinement ends C2 at23.40s; body timing state preserved')
    write_json(out/'fit/config.json', config)
    packet = export(reference, camera, info, out, source_path=root/'boot_integration_1102/fitted/result.npz')
    packet.update(opening_inferred_mask=reference.p['lower_body_inferred'],
        opening_motion_reference_json=np.array(json.dumps(contract)),
        stopping_reason_json=np.array(json.dumps(report['stopping_reason'])))
    np.savez_compressed(out/'fitted/result.npz', **packet)
    from .opening_motion_review import run as review
    review(root, out)
    print(json.dumps(dict(export=str(out/'fitted/result.npz'), frames=len(packet['pts']))), flush=True)


if __name__ == '__main__':
    main()
