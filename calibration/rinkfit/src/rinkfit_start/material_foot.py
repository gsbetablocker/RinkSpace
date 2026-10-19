"""Read-only finite-material station diagnostics on a complete fixed snapshot.

The existing capture grid is an initializer. All reported upper bounds require
the native RS certificate, include the original legal saved station, and make
no global-nearest or jointly ordered/statistically optimal station claim.
"""
from pathlib import Path
import json,time
import numpy as np
from .data import Observations,sha256_file
from .pipeline import read_run
from .runtime import atomic_json,atomic_npz
from .native_scene import NativeScene,camera_from_event
from .native_replay import semantic_arrays,row_state,read_array
from .semantic_resolution import resolve_materials
from .semantic_order import audit_source_runs
from .material_capture import capture_finite_arrays
from .capture_roots import _statistics


def _implicit_capture(scene,camera,material,branch,station,times,observed,*,iterations=8,initial_row=None):
    """Bounded local S/row GN capture; finite differences are seeds, not proof."""
    s=np.asarray(station,float).copy();v=(np.asarray(observed,float)[:,1] if initial_row is None else np.asarray(initial_row,float)).copy()
    dt=1e-5;eps=1e-5;clock=camera.tau/1079.
    for _ in range(iterations):
        X,_,_=scene.world_rows(material,branch,s);t=times+camera.tau*(v/1079.-.5)
        p=camera.at(X,t);pv=p['uv'];h=v-pv[:,1]
        sp=np.minimum(1.,s+eps);sm=np.maximum(0.,s-eps)
        xp,_,_=scene.world_rows(material,branch,sp);xm,_,_=scene.world_rows(material,branch,sm)
        ds=(camera.at(xp,t)['uv']-camera.at(xm,t)['uv'])/(sp-sm)[:,None]
        dv=(camera.at(X,t+dt)['uv']-camera.at(X,t-dt)['uv'])*(clock/(2*dt))
        hv=1.-dv[:,1];safe=np.where(np.abs(hv)>1e-10,hv,np.nan)
        tangent=ds+dv*(ds[:,1]/safe)[:,None]
        error=pv-observed-dv*(h/safe)[:,None];den=np.sum(tangent*tangent,axis=1)
        step=-np.divide(np.sum(tangent*error,axis=1),den,out=np.zeros(len(s)),where=den>1e-20)
        step=np.clip(np.nan_to_num(step),-.05,.05);step=np.clip(s+step,0.,1.)-s
        rowstep=(ds[:,1]*step-h)/safe
        bounded=np.clip(v+rowstep,1e-7,1079.-1e-7);rowstep=bounded-v
        hit=(v+((ds[:,1]*step-h)/safe)<0)|(v+((ds[:,1]*step-h)/safe)>1079.)
        restore=np.divide(h+hv*rowstep,ds[:,1],out=np.zeros(len(s)),where=np.abs(ds[:,1])>1e-20)
        step=np.where(hit,np.clip(np.nan_to_num(restore),-.05,.05),step)
        step=np.clip(s+step,0.,1.)-s;rowstep=np.clip(np.nan_to_num(rowstep),-300.,300.)
        best=np.sum((pv-observed)**2,axis=1)+1e4*h*h
        nexts=s.copy();nextv=v.copy()
        for reduction in range(7):
            factor=2.**(-reduction);trial_s=np.clip(s+factor*step,0.,1.);trial_v=np.clip(v+factor*rowstep,0.,1079.)
            world,finite,_=scene.world_rows(material,branch,trial_s)
            q=camera.at(world,times+camera.tau*(trial_v/1079.-.5));qh=trial_v-q['uv'][:,1]
            merit=np.sum((q['uv']-observed)**2,axis=1)+1e4*qh*qh
            take=finite&q['valid']&np.isfinite(merit)&(merit<best)
            nexts[take]=trial_s[take];nextv[take]=trial_v[take];best[take]=merit[take]
        s=nexts;v=nextv
    return s


def _certified_candidates(scene,camera,material,branch,station,times):
    world,finite,reason=scene.world_rows(material,branch,station)
    rs=camera.project_rs(np.where(finite[:,None],world,[0.,0.,1.]),times)
    rs['valid'] &= finite;rs['finite_support']=finite;rs['support_reason']=reason
    for key,value in rs['certificate'].items():
        if isinstance(value,np.ndarray):rs['RS_certificate_'+key]=value
    return rs


def select_certified_upper_bound(saved_s,saved_b,saved_result,candidate_s,candidate_b,candidate_result,observed):
    """Never discard an existing legal saved-S upper bound for a worse seed."""
    old=np.sum((saved_result['uv']-observed)**2,axis=1)
    new=np.sum((candidate_result['uv']-observed)**2,axis=1)
    take=candidate_result['valid']&(~saved_result['valid']|(new<old))
    s=np.where(take,candidate_s,saved_s);b=np.where(take,candidate_b,saved_b)
    result={k:np.array(v,copy=True)for k,v in saved_result.items()if isinstance(v,np.ndarray)and v.shape[:1]==(len(s),)}
    for k,v in result.items():
        if k in candidate_result:v[take]=candidate_result[k][take]
    return s,b,result,take


def _true_rs_grid_recovery(scene,camera,material,frames,times,observed,ordinary,
                           expected_variants,station,branch,result,*,
                           same_branch_only,threshold_px=20.,grid_count=4097,
                           progress=None):
    """Extra computation on difficult rows, never an observation selection.

    Finite stations and every available polyline knot are tested with true RS.
    Visibility transition bisection retains an actually certified endpoint;
    neither the grid nor local refinement proves global nearest distance.
    """
    started=time.monotonic();error=np.linalg.norm(result['uv']-observed,axis=1)
    requested=np.flatnonzero(ordinary&((~result['valid'])|(error>threshold_px)))
    tasks={}
    for i in requested:
        paths=scene.model['materials'][int(material[i])]['paths']
        candidates=[int(branch[i])] if same_branch_only else [b for b,p in enumerate(paths) if p['variant_id']==expected_variants[i]]
        for b in candidates:tasks.setdefault((int(frames[i]),int(material[i]),b),[]).append(int(i))
    seed_s=station.copy();seed_b=branch.copy();seed_cost=np.full(len(station),np.inf)
    boundaries=[];projected=0
    def offer(ii,s,b,uv,valid):
        cost=np.sum((uv-observed[ii])**2,axis=1);take=valid&(cost<seed_cost[ii]);jj=ii[take]
        seed_cost[jj]=cost[take];seed_s[jj]=np.asarray(s)[take];seed_b[jj]=b
    for task_id,((frame,m,b),rows) in enumerate(tasks.items()):
        ii=np.asarray(rows,int);path=scene.path(m,b)
        knots=np.arange(len(path.get('points',[])))/max(1,len(path.get('points',[]))-1)
        grid=np.unique(np.r_[np.linspace(0.,1.,grid_count),knots,station[ii]])
        gr=_certified_candidates(scene,camera,np.full(len(grid),m),np.full(len(grid),b),grid,np.full(len(grid),times[ii[0]]));projected+=len(grid)
        valid_ids=np.flatnonzero(gr['valid'])
        if len(valid_ids):
            d=np.sum((gr['uv'][valid_ids][None,:,:]-observed[ii,None,:])**2,axis=2);jj=valid_ids[np.argmin(d,axis=1)]
            offer(ii,grid[jj],b,gr['uv'][jj],gr['valid'][jj])
        # This also catches camera-domain visibility boundaries, while every
        # retained candidate still needs the normal native RS certificate.
        edges=np.flatnonzero(gr['valid'][1:]!=gr['valid'][:-1])
        for edge in edges:
            boundaries.append((task_id,grid[edge],grid[edge+1],bool(gr['valid'][edge]),m,b,times[ii[0]]))
    if boundaries:
        lo=np.asarray([x[1]for x in boundaries]);hi=np.asarray([x[2]for x in boundaries]);left_valid=np.asarray([x[3]for x in boundaries])
        mm=np.asarray([x[4]for x in boundaries]);bb=np.asarray([x[5]for x in boundaries]);tt=np.asarray([x[6]for x in boundaries])
        for _ in range(36):
            middle=(lo+hi)*.5;rr=_certified_candidates(scene,camera,mm,bb,middle,tt);projected+=len(middle)
            left=rr['valid']==left_valid;lo=np.where(left,middle,lo);hi=np.where(left,hi,middle)
        edge_s=np.where(left_valid,lo,hi);edge_rs=_certified_candidates(scene,camera,mm,bb,edge_s,tt);projected+=len(edge_s)
        task_items=list(tasks.items())
        for j,item in enumerate(boundaries):
            ii=np.asarray(task_items[item[0]][1],int)
            offer(ii,np.full(len(ii),edge_s[j]),int(bb[j]),np.broadcast_to(edge_rs['uv'][j],(len(ii),2)),np.full(len(ii),edge_rs['valid'][j]))
    found=requested[np.isfinite(seed_cost[requested])]
    old_invalid=int(np.count_nonzero(~result['valid'][requested]));improved=0
    if len(found):
        candidate=_certified_candidates(scene,camera,material[found],seed_b[found],seed_s[found],times[found])
        old={k:v[found]for k,v in result.items()if isinstance(v,np.ndarray)and v.shape[:1]==(len(station),)}
        ss,bb,rr,take=select_certified_upper_bound(station[found],branch[found],old,seed_s[found],seed_b[found],candidate,observed[found])
        station[found]=ss;branch[found]=bb
        for k in result:
            if isinstance(result[k],np.ndarray)and k in rr:result[k][found]=rr[k]
        improved+=int(take.sum())
        local=_implicit_capture(scene,camera,material[found],seed_b[found],seed_s[found],times[found],observed[found],iterations=16,initial_row=candidate['row'])
        refined=_certified_candidates(scene,camera,material[found],seed_b[found],local,times[found])
        old={k:v[found]for k,v in result.items()if isinstance(v,np.ndarray)and v.shape[:1]==(len(station),)}
        ss,bb,rr,take=select_certified_upper_bound(station[found],branch[found],old,local,seed_b[found],refined,observed[found])
        station[found]=ss;branch[found]=bb
        for k in result:
            if isinstance(result[k],np.ndarray)and k in rr:result[k][found]=rr[k]
        improved+=int(take.sum())
    report=dict(requested_rows=len(requested),scheduling_threshold_px=threshold_px,
        rows_with_legal_RS_grid_candidate=len(found),rows_without_candidate=requested[~np.isfinite(seed_cost[requested])].tolist(),
        initially_invalid_requested_rows=old_invalid,finally_invalid_requested_rows=int(np.count_nonzero(~result['valid'][requested])),
        frame_material_branch_tasks=len(tasks),grid_and_boundary_RS_projections=projected,
        visibility_transition_brackets=len(boundaries),candidate_improvements=improved,
        same_branch_only=same_branch_only,global_nearest_certified=False,
        source_obligations_removed=0,elapsed_seconds=time.monotonic()-started)
    if progress:progress(json.dumps(dict(stage='difficult_rows_true_RS_grid',**report)),flush=True)
    return report


def _summary(value,valid):
    good=np.asarray(value)[np.asarray(valid)&np.isfinite(value)]
    return dict(rows=len(value),valid_rows=int(np.count_nonzero(valid)),invalid_rows=int(np.count_nonzero(~np.asarray(valid))),
        median_px=float(np.median(good))if len(good)else None,p95_px=float(np.quantile(good,.95))if len(good)else None,
        max_px=float(good.max())if len(good)else None,over_2px=int(np.count_nonzero(good>2)),over_20px=int(np.count_nonzero(good>20)),over_50px=int(np.count_nonzero(good>50)))


def diagnose_finite_foot(run_dir,model_path,state_path,native_replay_dir,output_dir,*,progress=print):
    """Account for all source rows; write separate same-theta candidate ledgers."""
    started=time.monotonic();root,run=read_run(run_dir);out=Path(output_dir)
    if out.exists()and any(out.iterdir()):raise ValueError('Do not overwrite a frozen diagnostic')
    out.mkdir(parents=True,exist_ok=True);model_path=Path(model_path);state_path=Path(state_path);native=Path(native_replay_dir)
    hashes={str(p):sha256_file(p)for p in [model_path,state_path,native/'verification.json',native/'final_predictions.npz',native/'final_source_residuals.npz',native/'profile_global_frozen_inputs.npz',native/'profile_global_certificates.npz']}
    model=json.loads(model_path.read_text(encoding='utf-8'));state=json.loads(state_path.read_text(encoding='utf-8'));verification=json.loads((native/'verification.json').read_text(encoding='utf-8'))
    if verification['state_sha256']!=hashes[str(state_path)]or verification['model_sha256']!=hashes[str(model_path)]:raise ValueError('Native ledger and fixed complete snapshot differ')
    obs=Observations.load(run['data_dir']);n=len(obs.source_uv);identity=state['identity'];saved=row_state(state,state_path,n)
    semantic,sem,sem_path,resolution=semantic_arrays(root,n,state['semantic_manifest_path'],identity['statistics_id'],include_resolution=True)
    packet_path=Path(state['packet_manifest_path']);packet=json.loads(packet_path.read_text(encoding='utf-8'))
    if sha256_file(packet_path)!=state['packet_manifest_sha256']:raise ValueError('Packet identity differs')
    for name,key in [('resolved_material_index','resolved_material_sha256'),('resolved_variant_index','resolved_variant_sha256'),('confirmed_bindings','confirmed_bindings_sha256')]:
        if resolution['enabled']and identity.get(key)!=semantic[name]['sha256']:raise ValueError('Resolved semantic identity differs')
    tables_spec=packet['string_tables'];tables_path=packet_path.parent/tables_spec['file']
    if sha256_file(tables_path)!=tables_spec['sha256']:raise ValueError('Raw material table changed')
    tables=json.loads(tables_path.read_text(encoding='utf-8'));lookup={k:i for i,k in enumerate(tables['material_labels'])}
    raw_material=np.asarray([lookup[x]for x in obs.source_target_labels],dtype='<u4');source_variants=np.asarray(semantic['variant_ids'])[sem['variant_index']]
    material,expected_variants,bindings=resolve_materials(resolution,raw_material,source_variants,obs,model,saved['branches'],run['input_hashes']['observations.npz'])
    if identity['source_id']!=run['input_hashes']['observations.npz']or model['source_identity']!=identity['source_id']:raise ValueError('Immutable source identity differs')
    with np.load(native/'final_predictions.npz',allow_pickle=False)as z:oldp={k:z[k]for k in z.files}
    with np.load(native/'final_source_residuals.npz',allow_pickle=False)as z:olde={k:z[k]for k in z.files}
    with np.load(native/'profile_global_frozen_inputs.npz',allow_pickle=False)as z:q={k:z[k]for k in z.files}
    with np.load(native/'profile_global_certificates.npz',allow_pickle=False)as z:qcert={k:z[k]for k in z.files}
    if not np.array_equal(oldp['source_rows'],np.arange(n))or not np.array_equal(oldp['native_pts'],obs.native_pts)or not np.array_equal(oldp['source_point_ids'],obs.source_point_ids)or not np.array_equal(oldp['source_target_labels'],obs.source_target_labels):raise ValueError('Original source rows changed')
    if not verification['source_identity_exact']or not np.array_equal(oldp['resolved_material_indices'],material):raise ValueError('Source/overlay ledger differs')
    scene=NativeScene(model,state);camera=camera_from_event(model,state);current_model=json.loads(model_path.read_text(encoding='utf-8'))
    for block,value in zip(current_model['blocks'],scene.values):block['values']=value.tolist()
    # The helper reads scene.model only for immutable topology. Its numeric
    # world evaluator owns the complete current theta from the saved state.
    times=(obs.native_pts-256256)/12800.;ordinary=sem['kind']==0;profile=sem['kind']==2;corners=sem['kind']==1
    if n!=182017 or ordinary.sum()!=179364 or profile.sum()!=961 or corners.sum()!=1692:raise ValueError('Full original source obligations required')
    if not np.array_equal(q['source_rows'],np.flatnonzero(profile))or not np.all(qcert['global_unique_certified']):raise ValueError('Same-state original finite query proof incomplete')
    original_s=saved['stations'].astype(float);original_b=saved['branches'].astype(np.uint32)
    baseline=_certified_candidates(scene,camera,material,original_b,original_s,times)
    # Resolved variants use their own exact registry, never raw row labels.
    variant_ids=list(dict.fromkeys(expected_variants.tolist()));variant_index=np.asarray([variant_ids.index(v)for v in expected_variants],dtype='<u4')
    capture=capture_finite_arrays(scene,camera,obs.source_uv,obs.frame_indices,material,sem['kind'],variant_index,variant_ids,original_s,original_b,same_branch_only=True,progress=progress)
    ids=np.flatnonzero(ordinary);captured_s=capture['stations'].copy()
    refined_s=captured_s.copy();refined_s[ids]=_implicit_capture(scene,camera,material[ids],original_b[ids],captured_s[ids],times[ids],obs.source_uv[ids])
    same_s=original_s.copy();same_b=original_b.copy();same=baseline
    for name,s in [('observed_row_capture',captured_s),('local_implicit_capture',refined_s)]:
        candidate=_certified_candidates(scene,camera,material,original_b,s,times)
        same_s,same_b,same,take=select_certified_upper_bound(same_s,same_b,same,s,original_b,candidate,obs.source_uv)
        if progress:progress(json.dumps(dict(stage=name,accepted_candidate_upper_bounds=int(take.sum()),valid_rows=int(same['valid'].sum()),elapsed=time.monotonic()-started)),flush=True)
    difficult_recovery={'same_branch':_true_rs_grid_recovery(scene,camera,material,obs.frame_indices,times,obs.source_uv,
        ordinary,expected_variants,same_s,same_b,same,same_branch_only=True,progress=progress)}
    union_s=same_s.copy();union_b=same_b.copy();union={k:v.copy()for k,v in same.items()}
    multi=np.asarray([ordinary[i]and sum(path['variant_id']==expected_variants[i]for path in model['materials'][int(material[i])]['paths'])>1 for i in range(n)])
    jj=np.flatnonzero(multi)
    if len(jj):
        alt=capture_finite_arrays(scene,camera,obs.source_uv[jj],obs.frame_indices[jj],material[jj],sem['kind'][jj],variant_index[jj],variant_ids,same_s[jj],same_b[jj],progress=None)
        alt_s=_implicit_capture(scene,camera,material[jj],alt['branches'],alt['stations'],times[jj],obs.source_uv[jj]);alt_result=_certified_candidates(scene,camera,material[jj],alt['branches'],alt_s,times[jj])
        old={k:v[jj]for k,v in union.items()};ss,bb,rr,take=select_certified_upper_bound(union_s[jj],union_b[jj],old,alt_s,alt['branches'],alt_result,obs.source_uv[jj]);union_s[jj]=ss;union_b[jj]=bb
        for k in union:union[k][jj]=rr[k]
    difficult_recovery['legal_union']=_true_rs_grid_recovery(scene,camera,material,obs.frame_indices,times,obs.source_uv,
        ordinary,expected_variants,union_s,union_b,union,same_branch_only=False,progress=progress)
    # Genuine Q roots are an explicit alternative candidate state. Reuse the
    # same fixed theta's certified root, not the invalid saved-Q witness.
    for s,b,r in [(same_s,same_b,same),(union_s,union_b,union)]:
        s[profile]=q['station'];rquery=_certified_candidates(scene,camera,material[profile],b[profile],s[profile],times[profile])
        for key in r:r[key][profile]=rquery[key]
        if not r['valid'][profile].all():raise ValueError('Certified same-state Q root failed independent RS replay')
    if not np.array_equal(same_s[corners],original_s[corners])or not np.array_equal(union_s[corners],original_s[corners])or not np.array_equal(union_b[corners],original_b[corners]):raise ValueError('Physical entity point changed')
    olderror=np.linalg.norm(baseline['uv']-obs.source_uv,axis=1);same_error=np.linalg.norm(same['uv']-obs.source_uv,axis=1);union_error=np.linalg.norm(union['uv']-obs.source_uv,axis=1)
    if np.any(ordinary&baseline['valid']&((~same['valid'])|(same_error>olderror+1e-10)))or np.any(ordinary&same['valid']&((~union['valid'])|(union_error>same_error+1e-10))):raise ValueError('Legal saved-S upper bound was lost')
    for b in [same_b,union_b]:
        for i in range(n):
            if model['materials'][int(material[i])]['paths'][int(b[i])]['variant_id']!=expected_variants[i]:raise ValueError('Candidate changed source variant')
    alpha=np.full(n,np.nan);alpha[profile]=q['alpha']
    arrays=dict(source_rows=np.arange(n,dtype='<u8'),native_pts=obs.native_pts,frame_indices=obs.frame_indices,raw_material_index=raw_material,resolved_material_index=material,
        saved_station=original_s,saved_branch=original_b,saved_true_RS_uv=baseline['uv'],saved_true_RS_valid=baseline['valid'],saved_true_RS_error_2d_px=olderror,
        saved_explicit_row_uv=olde['saved_explicit_predicted_native_uv'],same_branch_station=same_s,same_branch_branch=same_b,same_branch_uv=same['uv'],same_branch_root_row=same['row'],same_branch_valid=same['valid'],same_branch_error_2d_px=same_error,
        union_station=union_s,union_branch=union_b,union_uv=union['uv'],union_root_row=union['row'],union_valid=union['valid'],union_error_2d_px=union_error,profile_alpha=alpha,
        source_kind=sem['kind'],same_branch_at_finite_endpoint=(same_s<=1e-8)|(same_s>=1.-1e-8),union_at_finite_endpoint=(union_s<=1e-8)|(union_s>=1.-1e-8),saved_valid=baseline['valid'])
    for name,result in [('saved',baseline),('same_branch',same),('union',union)]:
        for key,value in result.items():
            if key.startswith('RS_certificate_')or key in ['bracketed','root_equality_px']:
                arrays[name+'_'+key]=value
    atomic_npz(out/'finite_foot_candidates.npz',arrays)
    orders={};statistics={};summaries={}
    for name,s,b,r,err in [('saved',original_s,original_b,baseline,olderror),('same_branch',same_s,same_b,same,same_error),('legal_union',union_s,union_b,union,union_error)]:
        order=audit_source_runs(sem_path,current_model,s,b);atomic_json(out/(name+'_source_order.json'),order);orders[name]={k:v for k,v in order.items()if k!='records'}
        cost,_=_statistics(r['uv']-obs.source_uv,sem,sem_path);statistics[name]=dict(original_full_W_half_squared_norm=cost,arithmetic_all_rows_retained=True,rank=214112,original_groups=9173,valid_all_rows=bool(r['valid'].all()),source_order_passed=order['passed'],cost_is_not_a_joint_W_station_optimum=True,not_the_accepted_checkpoint_state=name!='saved')
        coverage=err.copy();coverage[profile]=np.abs(np.einsum('ij,ij->i',(r['uv']-obs.source_uv)[profile],sem['normal'][profile]));summaries[name]=dict(all_rows=_summary(coverage,r['valid']),ordinary_rows=_summary(err[ordinary],r['valid'][ordinary]),unresolved_source_rows=np.flatnonzero(~r['valid']).tolist())
    material_summary=[]
    for label in np.unique(obs.source_target_labels):
        ii=np.flatnonzero(obs.source_target_labels==label);material_summary.append(dict(raw_label=str(label),rows=len(ii),frames=len(np.unique(obs.frame_indices[ii])),saved=_summary(olderror[ii],baseline['valid'][ii]),same_branch=_summary(same_error[ii],same['valid'][ii]),legal_union=_summary(union_error[ii],union['valid'][ii])))
    atomic_json(out/'by_raw_material.json',material_summary)
    for path,digest in hashes.items():
        if sha256_file(path)!=digest:raise ValueError('Fixed source/checkpoint/ledger changed during diagnostic')
    direct=verification['evaluator_provenance']['direct_module_sha256'];package=Path(__file__).parent
    if any(sha256_file(package/name)!=digest for name,digest in direct.items()):raise ValueError('Frozen eleven direct evaluators changed')
    report=dict(schema='rinkfit.full_source_finite_foot_diagnostic.v2',state_token=state['state_token'],source_observations=n,frames=1248,raw_material_labels=171,ordinary_rows=int(ordinary.sum()),fixed_entity_points=int(corners.sum()),genuine_query_rows=int(profile.sum()),source_identity=identity,confirmed_binding_rows=[x['source_row']for x in bindings],inputs={p:d for p,d in hashes.items()},semantic_manifest=dict(path=str(sem_path),sha256=sha256_file(sem_path)),packet_manifest=dict(path=str(packet_path),sha256=sha256_file(packet_path)),summaries=summaries,original_full_W_arithmetic=statistics,source_orders=orders,
        original_legal_saved_station_in_candidate_set=True,each_reported_valid_foot_is_a_certified_true_RS_upper_bound=True,global_nearest_certified=False,difficult_rows_true_RS_recovery=difficult_recovery,finite_capture_scope='Existing full finite grid/all polyline knots plus observed-row-time bounded GN; bounded local implicit S/row refinement. All invalid or >20px ordinary rows additionally use shared true-RS finite grids, all available polyline knots, legal visibility-boundary candidates and further bounded local refinement. The threshold schedules computation and removes no obligations. No sampling grid or FD derivative is used as an optimality/root certificate. Native full-readout RS interval certificate validates each reported candidate.',physical_scene=scene.check_physics(),whole_optical_domain=camera.optical_domain_certificate(),candidate_source_order_joint_repair_performed=False,source_order_failure_requires_joint_capture_or_DP=True,
        original_checkpoint_or_W_or_source_or_semantic_changed=False,original_W_is_not_replaced_by_independent_foot_cost=True,final_physical_fit_accepted=False,elapsed_seconds=time.monotonic()-started,code_sha256={name:sha256_file(package/name)for name in ['material_foot.py','material_capture.py']},frozen_direct_module_sha256=direct,
        files={p.name:sha256_file(p)for p in out.iterdir()if p.is_file()})
    atomic_json(out/'report.json',report);return report
