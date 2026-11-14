from argparse import ArgumentParser
from pathlib import Path
import json
import os

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from experiments.skate_video_v1.inputs import atomic_json, read
from experiments.skate_video_v1.motion_review.scoped_fit import geometry, save_candidate


parser = ArgumentParser()
parser.add_argument('name')
parser.add_argument('inputs', type=Path, nargs='+')
args = parser.parse_args()
run = Path(os.environ['RINKSPACE_AUDIT_RUN'])
job = Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', run.parent/'audit_af48_1113')).resolve()/'cut_continuity_1113'
parent_path = job.parent / 'opening_lunge/delivered/solve/candidate.npz'
parent = load(parent_path)
initial = curve_from_packet(parent)
positions = initial.position_controls.copy()
rotations = initial.rotation_controls.copy()
used = np.zeros(len(positions), dtype=bool)
camera = None
details = []
for path in args.inputs:
    candidate = load(path)
    curve = curve_from_packet(candidate)
    np.testing.assert_array_equal(initial.knots, curve.knots)
    np.testing.assert_array_equal(parent['x'][:17], candidate['x'][:17])
    np.testing.assert_array_equal(parent['relative_skate_rotation_controls'], candidate['relative_skate_rotation_controls'])
    rows = np.any(curve.position_controls != initial.position_controls, axis=1)
    rows |= np.any(curve.rotation_controls != initial.rotation_controls, axis=(1, 2, 3))
    if np.any(used & rows):
        raise ValueError(f'Spline support overlaps another proposal: {path}')
    used |= rows
    positions[rows] = curve.position_controls[rows]
    rotations[rows] = curve.rotation_controls[rows]
    if camera is None:
        camera = candidate['camera_binding_json']
    else:
        np.testing.assert_array_equal(camera, candidate['camera_binding_json'])
    details.append(dict(path=str(path), active=np.flatnonzero(rows).tolist()))
assert used.any()
out = job / 'proposals' / args.name
out.mkdir(parents=True, exist_ok=True)
save_candidate(dict(parent, camera_binding_json=camera), geometry(initial, positions, rotations),
               out / (args.name+'.npz'), parent=parent_path)
atomic_json(out / 'assessment.json', dict(parts=details, changed_controls=int(used.sum()),
    camera_sha256=json.loads(str(camera))['state_sha256']))
print(out / (args.name+'.npz'), int(used.sum()), flush=True)
