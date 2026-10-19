"""Whole-run finite-grid initialization using certified true-RS projections.

Grid DP is a capture objective, never the correlated W likelihood or a proof of
nearest distance. Original legal whole-run seeds are admissible candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .semantic_capture import _domains,monotone_grid_dp
from .semantic_order import audit_source_runs,_board_lengths
from .material_foot import _certified_candidates


def true_rs_domain_grid(domain,camera,frame_time,original,*,spacing_px=4.,curvature_px=.1):
    """Finite true-RS grid plus certified visibility-boundary witnesses."""
    grid=np.unique(np.r_[domain.initial_breaks(original),np.linspace(0.,1.,65)])
    def project(s):return camera.project_rs(domain.world(s),np.full(len(s),frame_time))
    while True:
        projected=project(grid);middle=(grid[:-1]+grid[1:])*.5;mid=project(middle)
        uv=projected['uv'];length=np.linalg.norm(np.diff(uv,axis=0),axis=1)
        bend=np.linalg.norm(mid['uv']-(uv[:-1]+uv[1:])*.5,axis=1)
        supported=projected['valid'][:-1]|projected['valid'][1:]|mid['valid']
        split=supported&((length>spacing_px)|(bend>curvature_px))
        split&=(middle>grid[:-1])&(middle<grid[1:])
        if not np.any(split):break
        grid=np.sort(np.r_[grid,middle[split]])
    # A true-RS/image-domain transition is a legal capture boundary, even if
    # the closest unconstrained row-time point lies outside the image.
    edges=np.flatnonzero(projected['valid'][:-1]!=projected['valid'][1:])
    if len(edges):
        lo=grid[edges].copy();hi=grid[edges+1].copy();left=projected['valid'][edges]
        while True:
            middle=(lo+hi)*.5
            if np.all((middle==lo)|(middle==hi)):break
            legal=project(middle)['valid'];same=legal==left
            lo=np.where(same,middle,lo);hi=np.where(same,hi,middle)
        grid=np.unique(np.r_[grid,np.where(left,lo,hi)])
    if domain.closed:grid=grid[grid<1.]
    result=project(grid)
    return grid,result['uv'],result['valid']


def capture_source_runs_arrays(semantic_manifest,model,scene,camera,source_uv,
        native_pts,materials,expected_variants,stations,branches,*,
        baseline_stations=None,baseline_branches=None,active=None,progress=None):
    """Compare all original runs and return a complete legal source-index seed.

    ``model`` contains the exact current theta used by ``scene``/``camera``.
    No original row, source variant, path topology or statistical asset changes.
    """
    started=time.monotonic();semantic_manifest=Path(semantic_manifest)
    manifest=json.loads(semantic_manifest.read_text(encoding='utf-8'))
    spec=manifest['assets']['source_runs.jsonl'];run_path=semantic_manifest.parent/spec['file']
    runs=[json.loads(l)for l in run_path.read_text(encoding='utf-8').splitlines()if l.strip()]
    n=len(stations);active=np.ones(n,bool)if active is None else np.asarray(active,bool)
    if baseline_stations is None:baseline_stations=stations
    if baseline_branches is None:baseline_branches=branches
    source_uv=np.asarray(source_uv);materials=np.asarray(materials);expected_variants=np.asarray(expected_variants)
    result_s=np.array(stations,float,copy=True);result_b=np.array(branches,np.uint32,copy=True)
    times=(np.asarray(native_pts)-256256)/12800.;lengths=_board_lengths(model)
    before=audit_source_runs(semantic_manifest,model,stations,branches)
    original=audit_source_runs(semantic_manifest,model,baseline_stations,baseline_branches)
    captured=_certified_candidates(scene,camera,materials,branches,stations,times)
    base=_certified_candidates(scene,camera,materials,baseline_branches,baseline_stations,times)
    used=np.zeros(n,bool);pools={}
    for source in runs:
        ii=np.asarray(source['source_rows'],int)
        if np.any(used[ii])or not active[ii].all():raise ValueError('Overlapping/inactive ordered source obligations')
        used[ii]=True
        m=int(materials[ii[0]]);variant=manifest['variant_ids'][source['variant_index']]
        if np.any(materials[ii]!=m)or np.any(expected_variants[ii]!=variant):
            raise ValueError('Ordered source run changed its material or finite variant')
        if not np.all(native_pts[ii]==native_pts[ii[0]]):raise ValueError('A source run spans distinct native frames')
        key=(int(native_pts[ii[0]]),m,variant);pools.setdefault(key,[]).extend(ii.tolist())
    cache={};records=[];failures=[]
    for ordinal,(source,audit,base_audit)in enumerate(zip(runs,before['records'],original['records'],strict=True)):
        ii=np.asarray(source['source_rows'],int);m=int(materials[ii[0]]);variant=manifest['variant_ids'][source['variant_index']]
        key=(int(native_pts[ii[0]]),m,variant);domains=_domains(m,model['materials'][m],variant,scene,lengths)
        candidates=[];best=None
        for name,ss,bb,rr,aa in [('existing_legal_capture',stations,branches,captured,audit),
                                  ('original_legal_baseline',baseline_stations,baseline_branches,base,base_audit)]:
            if aa['valid']and rr['valid'][ii].all():
                cost=float(np.sum((rr['uv'][ii]-source_uv[ii])**2));choice={'domain':name,'direction_seed':aa['legal_direction_candidates'][0],
                    'cost_raw_capture_px2':cost,'feasible':True,'grid_points':None}
                candidates.append(choice)
                if best is None or cost<best[0]:best=(cost,np.asarray(bb)[ii].copy(),np.asarray(ss)[ii].copy(),choice)
        for domain in domains:
            cache_key=(*key,domain.label)
            if cache_key not in cache:
                pool=np.asarray(pools[key],int)
                coordinates=domain.original_coordinates(np.asarray(branches)[pool],np.asarray(stations)[pool])
                coordinates+=domain.original_coordinates(np.asarray(baseline_branches)[pool],np.asarray(baseline_stations)[pool])
                cache[cache_key]=true_rs_domain_grid(domain,camera,times[ii[0]],coordinates)
            grid,uv,valid=cache[cache_key]
            costs=np.sum((uv[None,:,:]-source_uv[ii,None,:])**2,axis=2);costs[:,~valid]=np.inf
            for direction in audit['allowed_direction_candidates']:
                selected,total=monotone_grid_dp(costs,closed=domain.closed,direction=direction)
                choice={'domain':domain.label,'direction_seed':int(direction),'cost_raw_capture_px2':float(total)if np.isfinite(total)else None,
                    'feasible':selected is not None,'grid_points':len(grid),'true_RS_valid_grid_points':int(valid.sum())}
                candidates.append(choice)
                if selected is not None and(best is None or total<best[0]):
                    bb,ss=domain.map(grid[selected]);best=(total,bb,ss,choice)
        record={'run_id':audit['run_id'],'frame':source['frame'],'source_rows':ii.tolist(),'variant_id':variant,
            'candidates':candidates,'source_feature_id':source['source_feature_id'],'unresolved':best is None,
            'world_direction_is_initialization_choice_only':source.get('world_direction')is None}
        if best is None:failures.append(audit['run_id'])
        else:
            _,bb,ss,choice=best;result_s[ii]=ss;result_b[ii]=bb;record['selected']=choice
        records.append(record)
        if progress and((ordinal+1)%250==0 or ordinal+1==len(runs)):
            progress({'stage':'all_source_runs_true_RS_DP','processed_runs':ordinal+1,'total_runs':len(runs),
                'cached_frame_material_domains':len(cache),'unresolved_runs':len(failures),'elapsed_seconds':time.monotonic()-started})
    after=audit_source_runs(semantic_manifest,model,result_s,result_b)
    if not np.array_equal(result_s[~used],np.asarray(stations)[~used])or not np.array_equal(result_b[~used],np.asarray(branches)[~used]):
        raise ValueError('Run capture changed an unrelated source row')
    return {'stations':result_s,'branches':result_b,'records':records,'source_order_before':before,
            'source_order_after':after,'unresolved_run_ids':failures,'source_run_rows':int(used.sum()),
            'cached_frame_material_domains':len(cache),'elapsed_seconds':time.monotonic()-started,
            'scope':'all finite-grid costs are certified true-RS raw native distances; not a W optimum or a global-nearest proof'}
