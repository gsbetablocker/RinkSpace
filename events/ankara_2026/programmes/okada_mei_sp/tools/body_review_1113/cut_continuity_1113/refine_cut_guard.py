import numpy as np
import torch

from fit_cuts import (JOB, angular_steps, assembly, clock, config, curve, doc, edge,
                     r, refit_state, tn)
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from cut_trial_geometry import DenseIce, SurfaceBounds


GUARD = load(JOB/'import_verified/scoped_guard_baseline.npz')
TIMES = load(JOB/'import_verified/solve/candidate.npz')['physics_times']
SCAN = load(doc['programme_structure']['surface_witnesses'])


class CutTrialGuard:
    def __init__(self, f, initial):
        self.f = f
        bounds = curve.knots[np.r_[f.ctrl[0],f.ctrl[-1]+4]]
        self.native = (clock[f.frames]>bounds[0])&(clock[f.frames]<bounds[1])
        steps = np.flatnonzero(self.native[:-1]|self.native[1:])
        self.step = steps
        self.angle_cap = tn(np.deg2rad(GUARD['angle'][f.frames[steps]].max(0)+2.05))
        self.anatomy = tn(GUARD['anatomy'][f.frames][self.native])
        clock_dense = TIMES[(TIMES>bounds[0])&(TIMES<bounds[1])]
        self.ice = DenseIce(f,clock_dense,refit_state(curve,clock_dense),edge,config['ice']['tolerance_m'])
        backup = JOB/'refined'/'scan_carry'/f'cut_{int(f.cut)}.npz'
        earlier = load(backup) if backup.is_file() else None
        self.surface = SurfaceBounds(f,initial,r,SCAN,doc,bounds,refit_state,prior_scan=earlier)
        self.surface_cache = backup
        self.upper = self.ice.upper
        self.contour_rows = {}
        for foot,_,ab,blend,*_ in f.whole:
            vertices = assembly.vertices[ab]
            mask = ab < assembly.upper_count
            self.contour_rows[foot] = ([self.upper.weights(vertices[:,:,j][:,None],mask[:,:,j][:,None]) for j in (0,1)],tn(blend))

    def loss(self, st, local, anatomy):
        parameter = self.f.fit.parameter
        step = torch.linalg.vector_norm(angular_steps(local)[self.step],dim=-1)
        step_limit = (torch.relu(step-self.angle_cap)/np.deg2rad(1.5)).square().amax(dim=0)
        data = anatomy[self.native]
        peak = torch.relu(data.amax(0)-self.anatomy.amax(0)-.012)
        mean = torch.relu(torch.relu(data-self.anatomy).mean(0)-.025)
        return dict(
            local_rotation_step=8.*step_limit.sum(),
            local_anatomy=12.*(peak/.025).square().sum()+10.*(mean/.02).square().sum(),
            exact_components=self.ice.loss(parameter),
            actual_surfaces=self.surface.loss(parameter))
