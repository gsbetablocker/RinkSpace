"""One actual weighted-J probe: source move and a physical camera-ray move."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
from .source_guided_refit import prepare,native_metrics
from .audit_probe_contract import choose_source_names,restore_saved_objective
from .directional_receipt import Block,receipt,source_guided_direction
from .weighted_blocks import verify
from .fullbody_round import write_json

GROUPS=dict(face=(0,1,2,3,4),torso=(5,6,11,12),lower_limbs=(13,14,15,16),upper_limbs=(7,8,9,10))

def body_groups(problem,value,direction=None,observation_mask=None):
    block=next(b for b in value['weighted_blocks']if b.name=='body_pixels');j=problem.body.oj
    model_j=getattr(problem.body,'model_oj',j)
    pixels=np.linalg.norm(problem.body.cache['uv'][problem.body.of,problem.body.layout.body_landmarks[model_j]]-problem.body.obs['uv'][problem.body.frames[problem.body.of],j],axis=-1)
    effects=None if direction is None else (block.residual*(block.jacobian@direction)).reshape(-1,2).sum(1)
    rows={}
    for name,indices in GROUPS.items():
        mask=np.isin(j,indices)
        if observation_mask is not None:mask &= np.asarray(observation_mask)[problem.body.frames[problem.body.of],j]
        r=block.residual.reshape(-1,2)[mask]
        row=dict(observations=int(mask.sum()),weighted_cost=float(np.sum(r*r)/2),pixel_median=float(np.median(pixels[mask])),pixel_p95=float(np.percentile(pixels[mask],95)))
        if effects is not None:row['directional_slope']=float(effects[mask].sum())
        rows[name]=row
    return rows

def run(project,stage,parameter,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);log_path=output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path.resolve()),actual_weighted_probe=True)),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1)as log:
        p,x,execution,info,mass,saved=prepare(project,stage,parameter,output,log)
        restored=restore_saved_objective(p,stage,saved);p.body.compatibility_difference_step=1e-8
        with execution.limits():
            started=time.perf_counter();first=p.evaluate(x,True);jsec=time.perf_counter()-started
            write_json(output/'registry.json',verify(first['weighted_blocks'],first['residual'],first['jac']))
            blocks=[Block(b.name,b.residual,b.jacobian,b.association)for b in first['weighted_blocks']]
            source_names=choose_source_names(blocks)
            d,record=source_guided_direction(blocks,source_names,parameter_scales=np.full(len(x),.1),damping=.1,active_columns=np.arange(17,len(x)),maxiter=160)
            peak=np.max(abs(d[17:]))
            if peak>.1:d*=.1/peak
            curve=p.body.state.curve;ray=p.reference.at(p.reference.times)['P']-p.body.camera_centres;ray/=np.linalg.norm(ray,axis=-1)[:,None]
            B=csr_matrix(p.body.state.basis(p.reference.times));controls=np.stack([lsmr(B,ray[:,j]*.05,atol=1e-10,btol=1e-10)[0]for j in range(3)],axis=1)
            depth=np.zeros_like(x);depth[17:17+p.body.state.body_size].reshape(p.body.state.m,69)[:,:3]=controls
            result=dict(objective_replay=restored,full_jacobian_seconds=jsec,baseline=native_metrics(p,first),groups=body_groups(p,first),source_direction=record,directions={})
            for name,direction in [('source',d),('ray_depth_5cm',depth)]:
                p.body.cache=first.get('body_cache',p.body.cache)
                row=dict(receipt=receipt(blocks,direction),body_groups=body_groups(p,first,direction),trials=[])
                for alpha in (-.25,.25):
                    trial=p.evaluate(x+alpha*direction,False)
                    row['trials'].append(dict(alpha=alpha,cost=float(trial['residual']@trial['residual']/2),groups=body_groups(p,trial),parts={b.name:float(b.residual@b.residual/2)for b in trial['weighted_blocks']}))
                result['directions'][name]=row
                p.evaluate(x,False)
            write_json(output/'trajectory_direction.json',result);np.savez_compressed(output/'proposal.npz',parent_parameter_sha256=saved['parent_parameter_sha256']if 'parent_parameter_sha256'in saved else np.array(''),source_pts=p.reference.p['pts'],baseline_x=x,source_direction=d,ray_depth_direction=depth)
    execution.close();print(json.dumps(dict(event='trajectory_probe_complete',first_J_seconds=jsec)),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    for name in ('project','stage','parameter','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.project,a.stage,a.parameter,a.output)
if __name__=='__main__':main()
