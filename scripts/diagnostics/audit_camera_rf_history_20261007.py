"""Read immutable physical-camera archives; do not smooth or mutate a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
REPO = Path(__file__).resolve().parents[2]
BASE = ROOT / 'okada_lane23_Ad7_fast_complete_20261007/initial_states/d2295be51b075c1b0767ebf046568b77676fe9ddc93a4f760cb9c60e9df2cd60'


def load_camera(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ('pts', 'exposure_representative_pts', 'view_state4', 'camera25', 'installation_id', 'metadata_json')}


def camera_identity(state_dir):
    return json.loads((state_dir / 'state_identity.json').read_text(encoding='utf8'))


def compare(base, current, shots):
    assert np.array_equal(base['pts'], current['pts'])
    assert np.array_equal(base['exposure_representative_pts'], current['exposure_representative_pts'])
    _, use = np.unique(base['exposure_representative_pts'], return_index=True)
    use.sort()
    pts = base['pts'][use]
    a, b = base['view_state4'][use], current['view_state4'][use]
    logf = b[:, 3] - a[:, 3]
    focal = 100 * np.expm1(logf)
    relative = Rotation.from_rotvec(a[:, :3]).inv() * Rotation.from_rotvec(b[:, :3])
    angle = np.rad2deg(relative.magnitude())
    adjacent_angle = np.rad2deg((relative[:-1].inv() * relative[1:]).magnitude())
    adjacent_logf = np.diff(logf)
    out = []
    for n, shot in enumerate(shots, 1):
        mask = (pts >= shot['start_pts']) & (pts < shot['end_pts'])
        rows = np.flatnonzero(mask)
        adj = np.flatnonzero(mask[:-1] & mask[1:])
        if not len(rows):
            continue
        f = rows[np.argmax(abs(focal[rows]))]
        r = rows[np.argmax(angle[rows])]
        item = dict(shot=n, start_pts=shot['start_pts'], end_pts_exclusive=shot['end_pts'],
                    representative_exposures=len(rows), max_focal_delta_percent=float(focal[f]),
                    max_focal_delta_pts=int(pts[f]), max_rotation_delta_degrees=float(angle[r]),
                    max_rotation_delta_pts=int(pts[r]))
        if len(adj):
            j = adj[np.argmax(abs(adjacent_logf[adj]))]
            k = adj[np.argmax(adjacent_angle[adj])]
            item.update(max_added_adjacent_logf=float(adjacent_logf[j]),
                        max_added_adjacent_logf_pts=[int(pts[j]), int(pts[j + 1])],
                        max_added_adjacent_rotation_degrees=float(adjacent_angle[k]),
                        max_added_adjacent_rotation_pts=[int(pts[k]), int(pts[k + 1])])
        out.append(item)
    samples = []
    for p in (229543424, 230157312, 230501120, 230506752, 230955776, 231705344):
        at = np.flatnonzero(base['pts'] == p)
        assert len(at) == 1
        i = at[0]
        rr = Rotation.from_rotvec(base['view_state4'][i, :3]).inv() * Rotation.from_rotvec(current['view_state4'][i, :3])
        samples.append(dict(pts=p, focal_delta_percent=float(100*np.expm1(current['view_state4'][i, 3]-base['view_state4'][i, 3])),
                            rotation_delta_degrees=float(np.rad2deg(rr.magnitude())),
                            optical_center_delta_conditional_scene_units=(current['camera25'][i, :3]-base['camera25'][i, :3]).tolist()))
    return dict(shots=out, selected_source_times=samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    opt = ap.parse_args()
    shots = json.loads((REPO/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    bi = camera_identity(BASE)
    base = load_camera(bi['camera_path'])
    accepted = []
    for run in ('okada_lane23_Ad7_fast_complete_20261007', 'okada_lane23_Ad7_material_accelerated_20261007'):
        states = sorted((ROOT/run/'accepted_states').iterdir(), key=lambda p:p.stat().st_mtime)
        for seq, state in enumerate(states, 1):
            ident = camera_identity(state)
            cam = load_camera(ident['camera_path'])
            assert hashlib.sha256(Path(ident['camera_path']).read_bytes()).hexdigest() == ident['camera_sha256']
            accepted.append(dict(run=run, accepted_sequence=seq, q_sha256=ident['q_sha256'],
                                 camera_path=ident['camera_path'], camera_sha256=ident['camera_sha256'],
                                 **compare(base, cam, shots)))
    old_path = ROOT/'okada_user_priorities_shared_actual_fit_fast_20261005/complete_physical_camera_model_UNACCEPTED_917c204dafde4027.npz'
    old = load_camera(old_path)
    result = dict(schema='rinkspace.raw-camera-rf-accepted-history-audit.v1',
                  baseline_q_sha256=bi['q_sha256'], baseline_camera_path=bi['camera_path'],
                  source_shots_path=str(REPO/'calibration/program_tracking/source_shots.json'),
                  comparison='All immutable accepted raw R3/logf camera values on distinct verified exposures. Adjacent increments computed within each full source shot only.',
                  smoothing_applied=False, numerical_cost_is_not_visual_accuracy=True,
                  old_917c_to_d229=compare(old, base, shots), accepted_states=accepted)
    if opt.output:
        opt.output.parent.mkdir(parents=True, exist_ok=True)
        opt.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print('accepted_states', len(accepted))
    for label, report in [('old917c_to_d229', result['old_917c_to_d229'])] + [(s['q_sha256'][:12],s) for s in accepted if s['q_sha256'].startswith(('478afaf', '4e212'))] + [(accepted[-1]['q_sha256'][:12],accepted[-1])]:
        print(label)
        for item in report['shots']:
            print(json.dumps(item))
        print('source_times', json.dumps(report['selected_source_times']))


if __name__ == '__main__':
    main()
