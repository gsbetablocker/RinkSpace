"""Resolve duplicated board registrations in S08 with adjacent source transfer."""
from __future__ import annotations

import argparse

import numpy as np

from common import OUTPUT, frozen_scene, save, scope, write
from measure import DT, FIRST, clock, initialize_controls, load_camera
from navigate import Pose, choose_grid, transfer
from rinkspace.domain.motion_chart import cubic_samples


def transfer_window(pose, rig, a, b, knots, homographies, support):
    grid=choose_grid()
    forward=[knots[a]]
    backward=[knots[b]]
    for matrix,good in zip(homographies[a:b],support[a:b],strict=True):
        q=forward[-1]
        forward.append(transfer(pose,rig,q,matrix,grid) if good else q)
    for matrix,good in zip(homographies[a:b][::-1],support[a:b][::-1],strict=True):
        q=backward[-1]
        backward.append(transfer(pose,rig,q,np.linalg.inv(matrix),grid) if good else q)
    weights=np.linspace(0.,1.,b-a+1)[:,None]
    return (1.-weights)*np.asarray(forward)+weights*np.asarray(backward[::-1])


def repair(baseline, seed, output):
    record=scope()
    shot=next(s for s in record["shots"] if s["id"]=="s08")
    pose=Pose(frozen_scene(record).core)
    pts,controls,rig=load_camera(shot,baseline)
    indices,weights=cubic_samples(len(controls),FIRST,DT,clock(pts))
    positions=np.einsum("ij,ijk->ik",weights,controls[indices])
    with np.load(OUTPUT/"navigation/s08_flow.npz",allow_pickle=False) as flow:
        np.testing.assert_array_equal(pts,flow["pts"])
        matrices=flow["adjacent_h"]
        support=(flow["counts"]>=35)&(flow["error_p90"]<=3.2)
    repaired=positions.copy()
    recovery=[]
    # The endpoints are checked against the actual board sequence.
    for span in [(68.60,70.16),(71.56,73.04)]:
        expect=np.rint(np.asarray(span)*12800.).astype(np.int64)
        a,b=np.searchsorted(pts,expect)
        np.testing.assert_array_equal(pts[[a,b]],expect)
        if np.count_nonzero(~support[a:b])>2:
            raise ValueError("Too little adjacent source support for S08")
        repaired[a:b+1]=transfer_window(pose,rig,a,b,positions,matrices,support)
        recovery.append(dict(first_pts=int(pts[a]),last_pts=int(pts[b]),source_pairs=int(b-a),
              supported_source_pairs=int(np.count_nonzero(support[a:b])),
              maximum_absolute_pan_correction_rad=float(np.max(abs(repaired[a:b+1,0]-positions[a:b+1,0])))))
    with np.load(seed/"camera.npz",allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    np.testing.assert_array_equal(pts,arrays["s08_pts"])
    np.testing.assert_array_equal(rig,arrays["s08_rig"])
    arrays["s08_controls"]=controls+initialize_controls(pts,repaired-positions)
    output.mkdir(exist_ok=False)
    save(output/"camera.npz",**arrays)
    write(output/"repair.json",dict(source=str(baseline),receipt=str(OUTPUT/"navigation/s08_flow.npz"),
                                    windows=recovery,guide_only=True))
    print(recovery,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",required=True)
    p.add_argument("--seed",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    repair(OUTPUT/a.baseline,OUTPUT/a.seed,OUTPUT/a.out)
