"""Infer dynamic legs from new boot pixels with calibrated geometry frozen."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import least_squares
from .footwear_fit import FootwearProblem
from .footwear_geometry import INITIAL,decode
from .rebuild_prepare import load,digest


def run(root,review_path,dataset,geometry_source,exposure_nodes=3):
    p=FootwearProblem(root,review_path,'articulated',dataset=dataset,exposure_nodes=exposure_nodes);source=load(geometry_source);geometry=source['x'][:len(INITIAL)].copy()
    x0=source['x'][len(INITIAL):].copy();out=Path(dataset)/'observed_pose'
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'source_review_at_run.json').write_bytes(Path(review_path).read_bytes());start=time.perf_counter()
    def full(x):return np.r_[geometry,x]
    config=dict(schema='rinkspace.fixed-boot-geometry-pose-transfer.v1',camera_scene_pelvis_beta_frozen=True,equipment_geometry_frozen=True,
        geometry_source=str(Path(geometry_source).resolve()),geometry_source_sha256=digest(geometry_source),
        admission_sha256=digest(Path(dataset)/'admission.json'),review_sha256=digest(review_path),input_sha256=digest(p.root/'input.npz'),
        base_B_sha256=digest(p.root/'pose/result.npz'),new_window_white_boot_pixels_consumed=True,new_window_blade_pixels_consumed=False,
        exposure_nodes=p.exposure_nodes,pose_variables=len(x0),contact_height_factor=False,source_time_base='1/12800',model_parameters=decode(geometry),device='cpu')
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(out/'initial_state.npz',x=full(x0),correction_knots=p.knots)
    with (out/'fit_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        p.log=log;log.write(json.dumps(dict(event='start',**config))+'\n')
        def fun(x):return p.fun(full(x))
        def accepted(x):
            v=p.evaluate(full(x));log.write(json.dumps(dict(event='accepted',cost=float(v['residual']@v['residual']/2),parts=v['parts'],elapsed_seconds=time.perf_counter()-start))+'\n')
            np.savez_compressed(out/'accepted.pending.npz',x=full(x),correction_knots=p.knots);(out/'accepted.pending.npz').replace(out/'last_accepted.npz')
        fit=least_squares(fun,x0,jac='2-point',jac_sparsity=p.sparsity[:,len(INITIAL):],bounds=(p.bounds[0][len(INITIAL):],p.bounds[1][len(INITIAL):]),
            method='trf',tr_solver='lsmr',x_scale='jac',diff_step=1e-4,ftol=1e-8,xtol=1e-8,gtol=1e-4,callback=accepted,tr_options={'atol':1e-7,'btol':1e-7})
        result=full(fit.x);last=p.evaluate(result);p.log=None;state=p.state(result,np.arange(len(p.p['pts'])))
        np.savez_compressed(out/'result.npz',x=result,correction_knots=p.knots,pts=p.p['pts'],equipment_uv=state['uv'],equipment_world_m=state['world'],COCO_uv=state['body_uv'],
            nominal_blade_minimum_world_m=state['nominal_blade_minimum'],P_world_m=p.base['P_world_m'],Q_world=p.base['Q_world'],fit_frames=p.frames)
        report=dict(optimizer_success=bool(fit.success),numerically_converged=bool(fit.optimality<=1e-4),termination=fit.message,cost=float(fit.cost),optimality=float(fit.optimality),
            nfev=fit.nfev,njev=fit.njev,elapsed_seconds=time.perf_counter()-start,parts=last['parts'],equipment_geometry_changed=False,pelvis_changed=False,
            independent_3d_truth=False,knives_in_new_windows_fitted=False)
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**report))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--geometry-source',type=Path,required=True);ap.add_argument('--exposure-nodes',type=int,choices=[3,5],default=3);a=ap.parse_args();run(a.root,a.review,a.dataset,a.geometry_source,a.exposure_nodes)
