"""Replay saved main Z190 source objective rows; never fit or decode images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

from main_installation_source_bundle_candidate import prepare
from main7_source_camera_candidate import sha
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA
from z190_physical_camera import PhysicalCameraModel
from continuous_central_brown4_optin import enable_projection_backend
from main7_source_bundle_candidate import huber_cost

CURRENT = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp')
DEFAULT_CURRENT = CURRENT/'z190_main_fixed_xy_shared_height_v1'
IDENTIFIER = 'okada_sp_installation_0_conditional_z190'


def ref(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def load_current_graph(current):
    manifest=json.loads((current/'input_manifest.json').read_text(encoding='utf8'))
    argv=manifest['run_command_argv']
    def argument(name):return Path(argv[argv.index('--'+name)+1])
    base,previous,scene_path=map(argument,['base','previous-source-data','scene'])
    if sha(scene_path)!=HARD_SCENE_SHA:raise ValueError('Hard scene changed')
    for b in manifest['source_bindings']:
        if sha(Path(b['path']))!=b['sha256']:raise ValueError('Current source input changed: '+b['path'])
    promoted=[Path(b['path']) for b in manifest['source_bindings'] if b.get('role')=='promoted_actual_source_review_no_longer_holdout']
    outputs=prepare(base,previous,base/'render_markings/main7_geometry/source_bridge_closure_v1',scene_path,promoted)
    scene,streams,aliases,cuts,rows,dense,bindings,suppressed,bundle,_=outputs
    model,native,physical_aliases,views=PhysicalCameraModel.load_npz(current/'physical_main_soft_height_candidate_unaccepted.npz')
    q=np.load(current/'optimized_state6rf.npz',allow_pickle=False)['q']
    if aliases!=physical_aliases or len(q)!=6+4*len(bundle.stamps):raise ValueError('Saved source/exposure state does not match current graph')
    if len(bundle.a)!=manifest['actual_source_pairs'] or len(bundle.native_residual_sigma)!=manifest['source_residual_rows']:
        raise ValueError('Saved source residual inventory differs from replay')
    return SimpleNamespace(current=current,manifest=manifest,base=base,previous=previous,scene_path=scene_path,scene=scene,
        streams=streams,aliases=aliases,cuts=cuts,rows=rows,dense=dense,bindings=bindings,suppressed=suppressed,
        bundle=bundle,model=model,native=native,views=views,q=q,calibration=model.calibrations[IDENTIFIER])


def audit_current(g,source_native_pts):
    p=g.aliases[int(source_native_pts)];index=g.bundle.index[p]
    groups=dict(g.bundle.abs_groups);rows=groups[p]
    keep=(g.bundle.ia==index)|(g.bundle.ib==index)
    select=np.repeat(keep,2)
    sigma=np.r_[g.bundle.motion_sigma[select],np.concatenate([np.repeat(r['sigma'],2) for r in rows])]
    local=SimpleNamespace(stamps=g.bundle.stamps,index=g.bundle.index,a=g.bundle.a[keep],b=g.bundle.b[keep],
        ia=g.bundle.ia[keep],ib=g.bundle.ib[keep],motion_sigma=g.bundle.motion_sigma[select],
        abs_groups=[(p,rows)],native_residual_sigma=sigma)
    objective=StandingSourceBundle(local,g.calibration,prior_wide_focal_px=g.manifest['frozen_wide_prior_focal_px'],fixed_height=False)
    enable_projection_backend()
    value,jac=objective.evaluate(g.q,include_priors=False)
    weight=np.minimum(1.,2./np.maximum(np.abs(value),1e-12))
    cols=6+4*index+np.arange(4);viewjac=jac[:,cols].toarray()
    original_global_start=g.bundle.motion_rows
    for stamp,source_rows in g.bundle.abs_groups:
        if stamp==p:break
        original_global_start+=2*sum(len(r['uv']) for r in source_rows)
    records=[];start=objective.motion_rows;global_start=original_global_start
    gradients={'motion':viewjac[:start].T@(weight[:start]*value[:start])}
    for row in rows:
        n=2*len(row['uv']);r=value[start:start+n];w=weight[start:start+n]
        gradient=viewjac[start:start+n].T@(w*r)
        gradients[row['id']]=gradients.get(row['id'],np.zeros(4))+gradient
        records.append(dict(id=row['id'],source_pts=row['pts'],source_native_pts=row.get('source_native_pts',row['pts']),
            split=row['split'],origin=row['origin'],point_correspondence=bool(row.get('point_correspondence')),
            physical_ids=row.get('physical_ids'),source_uv=row['uv'],sigma_native_px=row['sigma'],
            global_scalar_residual_slice=[global_start,global_start+n],
            residual_native_xy=(r*np.repeat(row['sigma'],2)).reshape(-1,2),
            normalized_residual_xy=r.reshape(-1,2),huber_irls_component_weights=w.reshape(-1,2),
            huber_cost=huber_cost(r,True),view_gradient_r3_logf=gradient,
            finite_curve_search_used=not bool(row.get('point_correspondence'))))
        start+=n;global_start+=n
    prior,prior_jac=objective.priors(g.q)
    roll_index=6+index # soft height: PP2,height1,k1-reference3,then per-view roll
    roll=prior_jac[roll_index,cols].toarray().ravel()*prior[roll_index]
    gradients['roll_prior']=roll
    checks=[]
    for column in cols:
        h=1e-7;plus,minus=g.q.copy(),g.q.copy();plus[column]+=h;minus[column]-=h
        numeric=(objective.evaluate(plus,False,include_priors=False)-objective.evaluate(minus,False,include_priors=False))/(2*h)
        analytic=jac[:,column].toarray().ravel()
        checks.append(dict(global_column=int(column),relative_l2=float(np.linalg.norm(numeric-analytic)/max(1.,np.linalg.norm(numeric))),
                          max_absolute=float(np.max(np.abs(numeric-analytic)))))
    if max(c['relative_l2'] for c in checks)>2e-4:raise ValueError('Current source RF derivative replay failed')
    source_motion_pairs=[]
    for ia,ib in sorted(set(zip(local.ia.tolist(),local.ib.tolist()))):
        ids=(local.ia==ia)&(local.ib==ib);s=local.motion_sigma[np.repeat(ids,2)]
        source_motion_pairs.append(dict(canonical_pts=[g.bundle.stamps[ia],g.bundle.stamps[ib]],points=int(ids.sum()),
            sigma_native_px_min=float(s.min()),sigma_native_px_max=float(s.max())))
    return dict(schema='rinkspace.z190-main-current-source-objective-audit.v1',source_native_pts=int(source_native_pts),canonical_pts=p,
        current_state=ref(g.current/'optimized_state6rf.npz'),current_manifest=ref(g.current/'input_manifest.json'),
        hard_scene_sha256=HARD_SCENE_SHA,no_solver_or_camera_or_source_change=True,
        source_objective_inventory_exact_replay=True,absolute_rows=records,
        suppression_at_exposure=[r for r in g.suppressed if r['canonical_pts']==p],
        original_ad_point_row_count=sum(bool(r.get('point_correspondence')) for r in rows),
        incident_motion_points=len(local.a),incident_motion_pairs=source_motion_pairs,
        normalization='source XY native residual divided once by original sigma; motion pair sigma=max(2,sqrt(N)/3)',
        robust_rule='per scalar XY Huber threshold2 normalized units; priors quadratic; no physical-group renormalization',
        actual_huber_view_gradients=gradients,gradient_sum=np.sum(list(gradients.values()),axis=0),
        derivatives_at_saved_source_state=checks,
        saved_height_m=g.calibration.pivot_xyz[2]*0.31353729994266055,
        original_absolute_scalar_residual_rows=len(g.bundle.native_residual_sigma)-g.bundle.motion_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=DEFAULT_CURRENT)
    p.add_argument('--source-native-pts',type=int,default=229186560)
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_source_objective_audit_v1')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);g=load_current_graph(a.current);report=audit_current(g,a.source_native_pts)
    (a.output/'objective_audit.json').write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(jsonable({k:report[k] for k in ['canonical_pts','suppression_at_exposure','original_ad_point_row_count','incident_motion_points','gradient_sum','derivatives_at_saved_source_state']})))
