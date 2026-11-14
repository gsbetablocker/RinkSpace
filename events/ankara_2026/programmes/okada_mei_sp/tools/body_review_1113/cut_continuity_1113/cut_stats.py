import sys
from pathlib import Path

import numpy as np
from fit_cuts import JOB, curve, clock, pts, config, refit_state, mass, ratio
from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.rebuild_prepare import load


def stats(c, cut):
    time = pts/12800.
    near = abs(time-pts[cut]/12800.+.01)<=.16
    use = np.flatnonzero(near)
    world=refit_state(c,clock[use])
    xyz,_=mass.from_pose(world)
    com=mass.com(xyz)
    _n,h=component_ice(mass.ref.binding,world) if hasattr(mass,'ref') else component_ice(__import__('fit_cuts').r.binding,world)
    X = np.linalg.norm(np.diff(com,n=2,axis=0)/.02**2,axis=-1)
    V = np.linalg.norm(np.diff(com,axis=0)/.02,axis=-1)
    R = world['R']
    angles=np.rad2deg(np.linalg.norm(_log(R[1:]@R[:-1].swapaxes(-1,-2)),axis=-1))
    shock=use[1:-1][np.argmax(X)]
    return dict(com=round(float(X.max()),2), com_shock_s=float(time[shock]),
                max_speed=round(float(V.max()),2), step=np.round(angles.max(0),2).tolist(),
                angle_shock_s=time[use[1:][np.argmax(angles,axis=0)]].tolist(),
                min_blade_mm=np.round(h[:,:,0].min(0)*1000,1).tolist())


for k in (1,2,3,4,5,6,7,8):
    cut=np.searchsorted(pts,config['camera_cut_continuity']['cut_pts'][k-1])
    print('Cut',k,pts[cut]/12800.,'prior',stats(curve,cut),flush=True)
    for name in sys.argv[1:]:
        dr = JOB/name
        path = dr/f'cut_{k}.npz' if dr.is_dir() else dr
        if path.is_file():
            print(name,stats(curve_from_packet(load(path)),cut),flush=True)
