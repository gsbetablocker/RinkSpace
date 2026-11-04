"""Immutable original-A comparison, distinct from stage-local retention."""
from copy import deepcopy
import numpy as np
from .fullbody_round import write_json


def error_summary(error,pts):
    error=np.asarray(error,float);pts=np.broadcast_to(np.asarray(pts),error.shape)
    valid=np.isfinite(error)
    if not valid.any():return dict(count=0,median_px=None,p95_px=None,worst_px=None,worst_pts=None)
    e=error[valid];p=pts[valid];k=int(np.argmax(e))
    return dict(count=len(e),exposures=int(len(np.unique(p))),median_px=float(np.median(e)),
        p95_px=float(np.quantile(e,.95)),worst_px=float(e[k]),worst_pts=int(p[k]))


def body_groups(predicted,observed,use,pts):
    error=np.linalg.norm(np.asarray(predicted)-observed,axis=-1)
    error=np.where(use,error,np.nan);clock=np.broadcast_to(np.asarray(pts)[:,None],error.shape)
    groups={'body':error_summary(error,clock)}
    for foot,ids in [(0,[11,13,15]),(1,[12,14,16])]:
        groups['body_foot_'+str(foot)]=error_summary(error[:,ids],clock[:,ids])
    return groups


def attach_saved_A_comparison(problem,root,identity):
    """For the earlier runner, original A remains fixed across root/pose seeds."""
    from .rebuild_prepare import load,digest
    path=root/'A/fitted/result.npz';original=load(path)
    frames=problem.body.frames;obs=problem.body.obs
    original_groups=body_groups(original['COCO_uv'][frames],obs['uv'][frames],obs['use'][frames],obs['pts'][frames])
    original_groups['shoe_groups']={'status':'not admitted in the original new-shot runner; no zero-error claim'}
    def measure(x,value):
        cache=problem.body.evaluate(x[17:],False)
        result=body_groups(cache['uv'][:,problem.body.layout.body_landmarks],
            obs['uv'][frames],obs['use'][frames],obs['pts'][frames])
        result.update(stage_image_cost=value['image_cost'],shoe_groups=deepcopy(original_groups['shoe_groups']))
        return result
    problem.stage_quality=StageQualityTrace(problem.output/'cross_stage_quality.json',
        dict(identity,original_A_result_sha256=digest(path)),original_groups,measure)


class StageQualityTrace:
    def __init__(self,path,identity,original_A,measure):
        self.path=path;self.identity=deepcopy(identity);self.original_A=deepcopy(original_A)
        self.measure=measure;self.seed=None;self.best=None;self.delivery=None;self.accepted=0

    def begin(self,x,value):
        self.seed=deepcopy(self.measure(x,value));self.best=deepcopy(self.seed)
        self._write()

    def finish(self,x,value,*,accepted_steps):
        if self.seed is None:raise ValueError('Stage seed must be recorded before solving')
        self.accepted=int(accepted_steps);self.best=deepcopy(self.measure(x,value))
        self.delivery=dict(metrics=deepcopy(self.best),status='requires_visual_review',not_truth=True)
        self._write()

    def record(self):
        return dict(schema='rinkspace.cross-stage-quality.v1',identity=deepcopy(self.identity),
            original_A=deepcopy(self.original_A),original_A_is_truth=False,stage_seed=deepcopy(self.seed),
            stage_best=deepcopy(self.best),delivery_candidate=deepcopy(self.delivery),
            accepted_steps=self.accepted,stage_update='updated' if self.accepted else 'no_update',
            stage_last_good_scope='local monitor only; never replaces original_A',
            accuracy_accepted=False,automatic_cross_stage_percentage_gate=False)

    def _write(self):write_json(self.path,self.record())
