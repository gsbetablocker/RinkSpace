"""Replay the existing engineering roll prior, without fitting or source images."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(directory):
    identity = read(directory/'state_identity.json')
    camera_path = Path(identity['camera_path'])
    if ref(camera_path)['sha256'] != identity['camera_sha256']:
        raise ValueError('Immutable saved camera changed')
    state_path = directory/'complete_state_UNACCEPTED.npz'
    if ref(state_path)['sha256'] != identity['state_npz_sha256']:
        raise ValueError('Immutable saved state changed')
    with np.load(state_path, allow_pickle=False) as z:
        q = z['q'].copy()
        contract = json.loads(str(z['aux__correlated_roll_prior_contract_json'].item()))
        permutation = z['aux__current_evaluation_to_canonical_parameter_indices'].copy()
        names = z['aux__block_names'].tolist()
        starts = z['aux__block_start'].copy()
        lengths = z['aux__block_lengths'].copy()
        stamps = z['aux__canonical_pts'].copy()
    with np.load(camera_path, allow_pickle=False) as z:
        cameras = {k: z[k].copy() for k in ('pts', 'installation_id', 'view_state4')}
    lookup = {int(p): i for i, p in enumerate(cameras['pts'])}
    # The prior contract uses evaluation coordinates; the saved q is canonical.
    bias_columns = permutation[np.asarray(contract['statistical_bias_parameter_indices'], int)]
    bias = q[bias_columns]
    inst_order = contract['installation_order']
    sigma = contract['component_sigma_radians']
    rows = []; cursor = 0
    for name, start, length in zip(names, starts, lengths):
        n = int(length)//4
        pts = stamps[cursor:cursor+n]; cursor += n
        original = q[int(start):int(start+length)].reshape(n, 4)
        indices = np.array([lookup[int(p)] for p in pts])
        # Exported view_state4 stores log(f/domain_width); canonical q stores
        # log(f_native_px). This prior uses only the identical rotation R3.
        if not np.array_equal(original[:, :3], cameras['view_state4'][indices, :3]):
            raise ValueError('Original view rotation differs from camera export')
        installations = cameras['installation_id'][indices].tolist()
        if len(set(installations)) != 1:
            raise ValueError('Original programme block installation changed')
        installation = installations[0]; inst = inst_order.index(installation)
        theta = []
        for vector in original[:, :3]:
            rotation = cv2.Rodrigues(vector)[0]
            x, y = rotation[0, 2], -rotation[1, 2]
            if x*x+y*y < 1e-8:
                raise ValueError('Original roll domain undefined')
            theta.append(np.arctan2(x, y))
        theta = np.asarray(theta)
        residual = (theta-bias[inst])/sigma
        rows.append(dict(block=name, pts=pts, theta=theta, residual=residual,
                         cost=.5*residual**2, installation=installation))
    if cursor != len(stamps):
        raise ValueError('Complete programme RF4 partition required')
    return dict(identity=identity, state=ref(state_path), camera=ref(camera_path),
                contract=contract, rows=rows, bias=bias, bias_columns=bias_columns,
                bias_cost=.5*float(np.sum((bias/sigma)**2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old, new = load(args.before), load(args.after)
    if old['contract'] != new['contract']:
        raise ValueError('Original roll prior contract changed')
    out = []
    for a, b in zip(old['rows'], new['rows']):
        if (a['block'] != b['block'] or a['installation'] != b['installation']
                or not np.array_equal(a['pts'], b['pts'])):
            raise ValueError('Original exposure/installation partition changed')
        out.append(dict(block=a['block'], actual_exposures=len(a['pts']),
                        installation=a['installation'],
                        prior_cost_before=float(a['cost'].sum()),
                        prior_cost_after=float(b['cost'].sum()),
                        prior_cost_change=float((b['cost']-a['cost']).sum()),
                        residual_degrees_quantiles_before=np.quantile(np.rad2deg(a['residual']*old['contract']['component_sigma_radians']), [0, .5, .95, 1]).tolist(),
                        residual_degrees_quantiles_after=np.quantile(np.rad2deg(b['residual']*new['contract']['component_sigma_radians']), [0, .5, .95, 1]).tolist()))
    result = dict(schema='rinkspace.saved-complete-programme-roll-prior-contribution.v1',
                  before={k: old[k] for k in ('identity', 'state', 'camera')},
                  after={k: new[k] for k in ('identity', 'state', 'camera')},
                  original_contract=old['contract'],
                  canonical_bias_columns=old['bias_columns'].tolist(),
                  bias_degrees_before=np.rad2deg(old['bias']).tolist(),
                  bias_degrees_after=np.rad2deg(new['bias']).tolist(),
                  explicit_three_bias_row_cost_before=old['bias_cost'],
                  explicit_three_bias_row_cost_after=new['bias_cost'],
                  per_block=out,
                  programme_prior_cost_before=sum(r['prior_cost_before'] for r in out),
                  programme_prior_cost_after=sum(r['prior_cost_after'] for r in out),
                  source=ref(ROOT/'calibration/solver_runtime/legacy/correlated_roll_prior_candidate/explicit_bias_source.py'),
                  script=ref(__file__),
                  source_images_written=0, new_fit_or_full_source_J=False,
                  source_observations_roles_weights_and_solver_state_changed=False,
                  limitations=['Engineering prior values are not actual roll measurements.',
                               'Costs alone do not prove gradient conflict or identify a true physical model error.',
                               'Active external-camera roll rows are outside the programme total here.'],
                  accuracy_accepted=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(args.output), per_block=out,
                         programme_cost_change=result['programme_prior_cost_after']-result['programme_prior_cost_before'])))


if __name__ == '__main__':
    main()
