"""Complete finite-material capture initializer; source/W/E stay immutable."""
from pathlib import Path
import json,shutil,time,numpy as np
from .native_scene import NativeScene,camera_from_event
from .data import sha256_file
from .runtime import atomic_json


def capture_finite_arrays(scene,camera,uv,frames,materials,kind,variants,variant_ids,old_station,old_branch,*,same_branch_only=False,progress=print):
    """Reusable finite-support initializer on explicit, source-resolved arrays.

    This uses the existing grid/observed-row-time bounded GN algorithm. It is
    a capture seed, not a global nearest point or a true-RS certificate.
    Inputs are never mutated. Corners and genuine profile stations are fixed.
    """
    started=time.monotonic();uv=np.asarray(uv,float);frames=np.asarray(frames);materials=np.asarray(materials);kind=np.asarray(kind);variants=np.asarray(variants);old_station=np.asarray(old_station);old_branch=np.asarray(old_branch);n=len(uv)
    if uv.shape!=(n,2) or any(np.shape(x)!=(n,)for x in [frames,materials,kind,variants,old_station,old_branch]):raise ValueError('Complete aligned capture arrays required')
    if not np.isfinite(uv).all()or not np.isfinite(old_station).all()or np.any((old_station<0)|(old_station>1)):raise ValueError('Finite original source/station values required')
    station=old_station.copy();branch=old_branch.copy();times=(frames-1)*.02+camera.tau*(uv[:,1]/1079.-.5)
    oldX,finite,_=scene.world_rows(materials,branch,station);oldp=camera.at(oldX,times);best=np.sum((oldp['uv']-uv)**2,axis=1);best[~(oldp['valid']&finite)]=np.inf
    old_error=np.sqrt(best);ordinary=kind==0;accounted=np.zeros(n,bool);accounted[~ordinary]=True;grid_evaluations=0
    for mi,material in enumerate(scene.model['materials']):
        rm=np.flatnonzero((materials==mi)&ordinary);accounted[rm]=True
        for bi,path in enumerate(material['paths']):
            rr=rm[variants[rm]==variant_ids.index(path['variant_id'])]
            if same_branch_only:rr=rr[old_branch[rr]==bi]
            if not len(rr):continue
            count=257 if path['kind']==3 else (129 if path['kind']==1 else 65)
            breaks=np.arange(len(path.get('points',[])))/max(1,len(path.get('points',[]))-1)
            grid=np.unique(np.r_[np.linspace(0.,1.,count),breaks])
            for frame in np.unique(frames[rr]):
                rows=rr[frames[rr]==frame];original=old_station[rows[old_branch[rows]==bi]];ss=np.unique(np.r_[grid,original]);world=scene.world(mi,bi,ss)
                projected=camera.at(world,np.full(len(ss),(frame-1)*.02));grid_evaluations+=len(ss);valid=projected['valid'];pixel=projected['uv']
                if not valid.any():continue
                # Full finite path chart, including every polygon vertex.
                # Linear segment candidates are only a seed; exact per-row
                # time projection decides whether to replace the old seed.
                delta=uv[rows,None]-pixel[None];cost=np.sum(delta*delta,axis=2);cost[:,~valid]=np.inf;node=cost.argmin(axis=1);selected=ss[node].copy();lowest=cost[np.arange(len(rows)),node]
                D=np.diff(pixel,axis=0);den=np.sum(D*D,axis=1);fraction=np.clip(np.sum((uv[rows,None]-pixel[None,:-1])*D[None],axis=2)/np.maximum(den,1e-30),0,1)
                e=uv[rows,None]-pixel[None,:-1]-fraction[:,:,None]*D[None];distance=np.sum(e*e,axis=2);distance[:,~(valid[:-1]&valid[1:])]=np.inf;j=distance.argmin(axis=1);value=distance[np.arange(len(rows)),j];replace=value<lowest
                selected[replace]=ss[j[replace]]+fraction[np.arange(len(rows))[replace],j[replace]]*(ss[j[replace]+1]-ss[j[replace]])
                actual=camera.at(scene.world(mi,bi,selected),times[rows]);value=np.sum((actual['uv']-uv[rows])**2,axis=1);take=actual['valid']&(value<best[rows]);chosen=rows[take];station[chosen]=selected[take];branch[chosen]=bi;best[chosen]=value[take]
        if progress and(mi+1)%20==0:progress(json.dumps(dict(stage='all_finite_path_capture',materials=mi+1,total_materials=len(scene.model['materials']),accounted=int(accounted.sum()),grid_camera_evaluations=grid_evaluations,elapsed=time.monotonic()-started)),flush=True)
    if not accounted.all():raise ValueError('A source obligation was not visited')
    # Exact native observed-row-time bounded Gauss-Newton refinement, all
    # ordinary rows together; disconnected branches remain explicitly fixed.
    selected=np.flatnonzero(ordinary)
    for iteration in range(6):
        s=station[selected];b=branch[selected];m=materials[selected];t=times[selected];world,_,_=scene.world_rows(m,b,s);base=camera.at(world,t)
        plus=np.minimum(1.,s+1e-5);minus=np.maximum(0.,s-1e-5);xp,_,_=scene.world_rows(m,b,plus);xm,_,_=scene.world_rows(m,b,minus);pp=camera.at(xp,t);pm=camera.at(xm,t);d=(pp['uv']-pm['uv'])/(plus-minus)[:,None]
        error=base['uv']-uv[selected];den=np.sum(d*d,axis=1);step=np.divide(np.sum(error*d,axis=1),den,out=np.zeros(len(s)),where=den>1e-20);step=np.clip(step,-.05,.05)
        for reduction in range(8):
            candidate=np.clip(s-step,0.,1.);X,_,_=scene.world_rows(m,b,candidate);trial=camera.at(X,t);cost=np.sum((trial['uv']-uv[selected])**2,axis=1);take=trial['valid']&np.isfinite(cost)&(cost<best[selected]);station[selected[take]]=candidate[take];best[selected[take]]=cost[take];step*=.5
        if progress:progress(json.dumps(dict(stage='all_finite_exact_row_time_refine',iteration=iteration+1,rows=len(selected),elapsed=time.monotonic()-started)),flush=True)
    if np.any(station[~ordinary]!=old_station[~ordinary])or np.any(branch[~ordinary]!=old_branch[~ordinary]):raise ValueError('Profile/entity-point seed was unexpectedly altered')
    for i in range(n):
        if scene.model['materials'][int(materials[i])]['paths'][int(branch[i])]['variant_id']!=variant_ids[int(variants[i])]:raise ValueError('Source variant changed')
    return dict(stations=station,branches=branch,old_error_px=old_error,candidate_fixed_row_cost_sq=best,candidate_fixed_row_error_px=np.sqrt(best),grid_camera_evaluations=grid_evaluations,elapsed_seconds=time.monotonic()-started,all_rows_accounted=bool(accounted.all()),same_branch_only=bool(same_branch_only))


def capture_all_finite(run_dir,model_path,seed_dir,output_dir,progress=print):
    started=time.monotonic();run=Path(run_dir);packet=run/'backend_input';seed=Path(seed_dir);out=Path(output_dir)
    if out.exists()and any(out.iterdir()):raise ValueError('Never overwrite a previous capture candidate')
    out.mkdir(parents=True,exist_ok=True);model=json.loads(Path(model_path).read_text(encoding='utf-8'));scene=NativeScene(model);camera=camera_from_event(model)
    uv=np.fromfile(packet/'uv.f64','<f8').reshape(-1,2);n=len(uv);frames=np.fromfile(packet/'frame.i32','<i4');materials=np.fromfile(packet/'material.u32','<u4')
    kind=np.fromfile(run/'semantic_packet/kind.u8','u1');variants=np.fromfile(run/'semantic_packet/variant_index.u32','<u4');variant_ids=json.loads((run/'semantic_packet/manifest.json').read_text())['variant_ids']
    old_station=np.fromfile(seed/'station.f64','<f8');old_branch=np.fromfile(seed/'path.u32','<u4');station=old_station.copy();branch=old_branch.copy();times=(frames-1)*.02+camera.tau*(uv[:,1]/1079.-.5)
    if n!=182017 or len(np.unique(frames))!=1248:raise ValueError('Complete original native shot required')
    captured=capture_finite_arrays(scene,camera,uv,frames,materials,kind,variants,variant_ids,old_station,old_branch,progress=progress)
    station=captured['stations'];branch=captured['branches'];old_error=captured['old_error_px'];best=captured['candidate_fixed_row_cost_sq'];grid_evaluations=captured['grid_camera_evaluations'];ordinary=kind==0
    station.astype('<f8').tofile(out/'station.f64');branch.astype('<u4').tofile(out/'path.u32')
    initial_row=np.fromfile(seed/'initial_row.f64','<f8')if(seed/'initial_row.f64').exists()else uv[:,1].copy()
    initial_row[ordinary]=uv[ordinary,1];initial_row.astype('<f8').tofile(out/'initial_row.f64')
    if(seed/'initial_alpha.f64').exists():shutil.copyfile(seed/'initial_alpha.f64',out/'initial_alpha.f64')
    report=dict(schema='rinkfit.full_source_finite_material_recapture.v1',source_rows=n,frames=1248,materials=171,ordinary_rows=int(ordinary.sum()),profile_rows=int(np.count_nonzero(kind==2)),source_identity=model['source_identity'],model=dict(path=str(model_path),sha256=sha256_file(model_path)),seed=str(seed),changed_stations=int(np.count_nonzero(station!=old_station)),changed_branches=int(np.count_nonzero(branch!=old_branch)),unresolved_rows=np.flatnonzero(~np.isfinite(best)).tolist(),
      exact_row_time_error_before=dict(median=float(np.median(old_error)),p95=float(np.quantile(old_error,.95)),max=float(np.max(old_error))),exact_row_time_error_after=dict(median=float(np.median(np.sqrt(best))),p95=float(np.quantile(np.sqrt(best),.95)),max=float(np.max(np.sqrt(best)))),
      station_sha256=sha256_file(out/'station.f64'),path_sha256=sha256_file(out/'path.u32'),initial_row_sha256=sha256_file(out/'initial_row.f64'),initial_row_policy='ordinary changed-station rows use original observed v; unchanged profile/entity rows keep their previous explicit witness; no implied true-RS success',grid_camera_evaluations=grid_evaluations,elapsed_seconds=time.monotonic()-started,source_order_audit_and_repair_required=True,true_RS_and_query_recovery_required=True,
      scope='Initial finite-chart capture with exact observed-row-time refinement; no global nearest-point claim; original UV/W/E preserved and no derived factor added.')
    atomic_json(out/'station_initialization.json',report);return report
