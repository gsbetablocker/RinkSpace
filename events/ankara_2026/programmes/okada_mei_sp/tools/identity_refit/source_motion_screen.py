"""Locate full-programme source mismatches without exporting dense meshes."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import bind, GROUPS
from source_review import digest, NAMES
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection


def screen(prepared, source, parameter, output, body_observations=None):
    ref, camera, info, document, packet = bind(prepared, source, parameter)
    ledger=ref.observations_path if body_observations is None else body_observations
    observations = dict(np.load(ledger))
    pts = ref.p['pts']
    uv = np.empty((len(pts), 17, 2))
    for start in range(0, len(pts), 96):
        frames = np.arange(start, min(start+96, len(pts)))
        tau = camera.readout_times(frames)
        times = (ref.times[frames, None]+tau[:, None]*np.linspace(-.5, .5, 5)).ravel()
        world = ref.world(ref.at(times), np.zeros((len(times), 69)))['world']
        nodes = world.reshape(len(frames), 5, -1, 3)[:, :, ref.layout.body_landmarks]
        uv[frames] = moving_projection(camera, nodes, frames)['uv']
    assignments = observations['model_joint_ids']
    predicted = uv[np.arange(len(uv))[:, None], assignments]
    error = np.linalg.norm(predicted-observations['uv'], axis=-1)
    used = observations['use']
    # Keep one small reusable projection cache, not a dense fitted surface.
    np.savez_compressed(output, pts=pts, shot_ids=ref.p['shot_ids'], uv=uv,
        error=error, source_uv=observations['uv'], use=used,
        sigma_px=observations['sigma_px'], model_joint_ids=assignments,
        parameter_sha256=np.array(digest(parameter)),body_observations_sha256=np.array(digest(ledger)))
    for shot in np.unique(ref.p['shot_ids']):
        print(json.dumps(dict(shot=str(shot), body={
            name: dict(count=int(mask.sum()), median_px=float(np.median(error[mask])),
                       p95_px=float(np.percentile(error[mask],95)))
            for name, ids in GROUPS.items()
            if (mask := used & np.isin(assignments,ids) & (ref.p['shot_ids']==shot)[:,None]).any()
        })), flush=True)
    # Highest source disagreements separated in time, for visual diagnosis;
    # detector disagreement is not an accuracy certificate.
    for label, ids in GROUPS.items():
        mask = used & np.isin(assignments,ids)
        scores = np.where(mask,error,-np.inf).max(1)
        selected=[]
        for f in np.argsort(scores)[::-1]:
            if not np.isfinite(scores[f]):
                break
            if any(abs(int(pts[f])-int(pts[g])) < 12800 for g in selected):
                continue
            selected.append(int(f))
            if len(selected)==12:
                break
        print(json.dumps(dict(group=label, review=[dict(seconds=float(pts[f]/12800),
            source_joint=NAMES[int(np.argmax(np.where(mask[f],error[f],-np.inf)))],
            error_px=float(scores[f])) for f in selected])),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','source','parameter','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--body-observations',type=Path)
    screen(**vars(parser.parse_args()))
