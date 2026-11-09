"""Protect visible skate paths between direct source exposures in a proposal."""

import numpy as np

from .ordered_native_skate import SemanticSkateMesh
from .skate_physics import fk_state
from .source_visible_model import unoccluded


class InterExposureRunnerGuard:
    """Detect a new large loss of modeled visibility between observed traces."""

    def __init__(self, replay, frames):
        self.replay = replay
        self.frames = np.asarray(frames, dtype=int)
        self.times = replay.ref.times[self.frames]
        self.base = replay.ref.at(self.times)
        self.mesh = SemanticSkateMesh(replay.problem.binding)
        self.camera = replay.camera.camera_centres(self.frames)
        evidence = {(r['pts'],r['foot']) for r in replay.ordered.rows if r['semantic']=='runner_lower_edge'}
        self.no_direct_runner = np.array([
            [(replay.pts[i],foot) not in evidence for foot in (0,1)] for i in frames])
        self.original = self.visible_spans(replay.x)

    def visible_spans(self,x):
        r = self.replay
        pose = fk_state(r.ref,self.base,r.problem.body.state.increments(x[17:],self.times))
        origin = pose['O']
        eye = np.einsum('nfji,nfj->nfi',pose['R'],self.camera[:,None]-origin)
        out = np.zeros((len(self.frames),2))
        f = np.tile([0,1],len(self.frames))
        repeated = {k:np.repeat(pose[k],2,axis=0) for k in ('native','frames','skate_relative_R')}
        # The model's outer shell and rocker geometry are from the same
        # binding used by the ordered native residual.
        local = self.mesh.local(repeated,f)
        for i,foot in np.ndindex(len(self.frames),2):
            verts = local[2*i+foot]
            ids = self.mesh.runner['runner_lower_edge'][int(eye[i,foot,1]>=0)]
            path = verts[ids]
            mid = .5*(path[1:]+path[:-1])
            step = np.linalg.norm(np.diff(path,axis=0),axis=1)
            visible = unoccluded(eye[i,foot],mid,verts[self.mesh.faces])
            out[i,foot] = step[visible].sum()/step.sum()
        return out

    def newly_hidden(self,x):
        visible = self.visible_spans(x)
        # This is a model-change check where a direct trace has no say.
        # A view can legitimately become occluded if a measured trace drives
        # the trial, so only a new large collapse between measurements is
        # retired by this check.
        changed = (self.original>=.65) & (visible<.53*self.original) & self.no_direct_runner
        return [dict(pts=int(self.replay.pts[self.frames[i]]),foot=int(f),
                     previous_visible_fraction=float(self.original[i,f]),
                     proposed_visible_fraction=float(visible[i,f]))
                for i,f in np.argwhere(changed)]
