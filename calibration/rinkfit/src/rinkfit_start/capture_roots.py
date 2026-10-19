"""Full-obligation primal root initialization on an unchanged physical model.

This writes a separate candidate. Numerical query roots are not interval
certificates and never change the source measurement, W, or finite domains.
"""
from pathlib import Path
import argparse, json, shutil, time
import numpy as np
from .data import sha256_file
from .native_scene import NativeScene,camera_from_event
from .native_query import load_query_packet,recover_query_roots,rational_query
from .native_replay import read_array,semantic_arrays
from .semantic_order import audit_source_runs
from .runtime import atomic_json


def _summary(x):
    x=np.asarray(x,float);finite=np.isfinite(x);v=x[finite]
    return dict(rows=int(x.size),finite=int(finite.sum()),nonfinite=int((~finite).sum()),
      median=float(np.median(v)) if len(v) else None,p95=float(np.quantile(v,.95)) if len(v) else None,
      max=float(np.max(v)) if len(v) else None)


def _physical(scene,model):
    values=scene.values;g=values[0];result=[];ids=[]
    def add(name,v):ids.append(name);result.append(float(v))
    add('rink/nonzero_short_straight',(2*g[2]-g[0]+2e-5)/.5)
    for i,(spec,q) in enumerate(zip(model['blocks'],values)):
        name=spec['name']
        if name.startswith('ad/'):
            add(name+'/bottom_on_board',-q[2]/.02);add(name+'/nonzero_height',(q[2]-q[3]+1e-6)/.02)
            add(name+'/top_on_board',(q[3]-g[4])/.02);add(name+'/span_not_larger_than_perimeter',q[1]-scene.perimeter)
        elif name.startswith('ice/circle/'):
            for j in range(2,len(q)):add(name+'/positive_radius_'+str(j),(1e-6-q[j])/.05)
            for j in range(2,len(q)-1):add(name+'/nested_radius_'+str(j),(q[j]-q[j+1]+1e-6)/.05)
            for sx in [-1,1]:
                for sy in [-1,1]:add(name+f'/conservative_outer_square_{sx}_{sy}',scene.containment(q[:2]+q[-1]*np.array([sx,sy])))
        elif name.startswith('ice/line/'):
            add(name+'/endpoint0_in_rink',scene.containment(q[:2]));add(name+'/endpoint1_in_rink',scene.containment(q[2:4]));add(name+'/nonzero_length',(1e-10-np.sum((q[:2]-q[2:4])**2))/.25)
        elif name.startswith('print/'):
            vertices=[];ad=None
            for m in model['materials']:
                for p in m['paths']:
                    if p.get('geometry_block',-1)==i:vertices.extend(p['points']);ad=values[p['advertisement_block']]
            uv=np.asarray(vertices);vv=q[3]*uv[:,0]+q[4]*uv[:,1]+q[5];h=ad[2]+(ad[3]-ad[2])*vv
            add(name+'/board_height_-1',-h.min()/.02);add(name+'/board_height_1',(h.max()-g[4])/.02);add(name+'/positive_affine_orientation',1e-6-(q[0]*q[4]-q[1]*q[3]))
    return ids,np.asarray(result)


def _orders(scene,model,stations,branches,materials,seed):
    lengths=np.array([scene.long,scene.arc,scene.short,scene.arc]*2);prefix=np.r_[0,np.cumsum(lengths)]
    ids=[];values=[]
    for record in seed['records']:
        rows=np.asarray(record['source_rows'],int);coordinate=stations[rows].copy()
        for j,row in enumerate(rows):
            path=model['materials'][int(materials[row])]['paths'][int(branches[row])]
            if path['kind']==3 and path.get('boundary_segment',-1)>=0:
                k=path['boundary_segment'];coordinate[j]=(prefix[k]+lengths[k]*coordinate[j])/scene.perimeter
        x=coordinate+np.asarray(record['chosen_integer_lifts']);d=record['chosen_seed_direction']
        for j in range(1,len(rows)):ids.append('order/'+record['run_id']+'/'+str(j));values.append(d*(x[j-1]-x[j]))
        if record['world_path_closed']:ids.append('order_span/'+record['run_id']);values.append(d*(x[-1]-x[0])-1)
    return ids,np.asarray(values)


def _statistics(error,sem,sem_path):
    weights=sem['weights'];members=sem['group_rows'];residual=[];count=np.zeros(len(error),np.int8)
    manifest=json.loads(Path(sem_path).read_text(encoding='utf-8'))
    group_path=Path(sem_path).parent/manifest['assets']['groups.jsonl']['file']
    with group_path.open(encoding='utf-8') as handle:
        for line in handle:
            group=json.loads(line);start=group['member_start'];n=group['member_count'];rank=group['rank'];offset=group['weight_start'];rows=members[start:start+n].astype(int)
            W=weights[offset:offset+rank*2*n].reshape(rank,2*n);residual.append(W@error[rows].reshape(-1));count[rows]+=1
    independent=sem['ordinary_rows'].astype(int);residual.append((error[independent]/sem['source_sigma'][independent,None]).reshape(-1));count[independent]+=1
    out=np.concatenate(residual)
    if np.any(count!=1) or len(out)!=214112:raise ValueError('Complete original statistical accounting failed')
    return .5*float(out@out),out


def _domain(camera,instant):
    z=instant['z'];R=float(camera.optics['radius_max']);q=R*R
    k1=np.polynomial.polynomial.polyval(z,camera.optics['k1']);k2=np.polynomial.polynomial.polyval(z,camera.optics['k2'])
    def minimum(c1,c2):
        endpoint=np.minimum(1.,1+c1*q+c2*q*q);vertex=np.divide(-c1,2*c2,out=np.zeros_like(c1),where=c2!=0)
        return np.minimum(endpoint,np.where((c2>0)&(vertex>0)&(vertex<q),1+c1*vertex+c2*vertex*vertex,np.inf))
    lower=np.minimum(minimum(k1,k2),minimum(3*k1,5*k2))-6*np.hypot(camera.optics['p1'],camera.optics['p2'])*R
    return np.c_[instant['ray_radius']**2/q-1.,-instant['depth'],-lower]


def capture_roots(run_dir,model_path,seed_dir,output_dir,progress=print):
    started=time.monotonic();run=Path(run_dir);packet=run/'backend_input';seed=Path(seed_dir);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Never overwrite a previous root capture candidate')
    out.mkdir(parents=True,exist_ok=True);model_path=Path(model_path);model=json.loads(model_path.read_text(encoding='utf-8'))
    semantic,sem,sem_path=semantic_arrays(run,182017);n=semantic['observations'];profile=np.flatnonzero(sem['kind']==2)
    _,query=load_query_packet(run/'query_packet/manifest.json',read_array,sha256_file(sem_path),profile)
    uv=np.fromfile(packet/'uv.f64','<f8').reshape(n,2);pts=np.fromfile(packet/'pts.i64','<i8');material=np.fromfile(packet/'material.u32','<u4');budget=np.fromfile(packet/'budget.f64','<f8')
    station=np.fromfile(seed/'station.f64','<f8');branch=np.fromfile(seed/'path.u32','<u4')
    if station.shape!=(n,) or branch.shape!=(n,):raise ValueError('Root capture requires every original row')
    initial=json.loads((run/'query_initialization.json').read_text(encoding='utf-8'));initial_alpha=read_array(run,initial['alpha']);alpha0=np.zeros(n);alpha0[profile]=initial_alpha
    row0=uv[:,1].copy()
    if (seed/'initial_row.f64').exists():row0=np.fromfile(seed/'initial_row.f64','<f8')
    if (seed/'initial_alpha.f64').exists():alpha0=np.fromfile(seed/'initial_alpha.f64','<f8')
    if row0.shape!=(n,)or alpha0.shape!=(n,)or not np.isfinite(row0).all()or not np.isfinite(alpha0).all()or np.any((row0<0)|(row0>1079))or np.any(alpha0[sem['kind']!=2]!=0)or np.any((alpha0[profile]<query['bounds'][:,0])|(alpha0[profile]>query['bounds'][:,1])):raise ValueError('Original complete auxiliary seed is not finite/in its immutable domains')
    times=(pts-256256)/12800.;scene=NativeScene(model);camera=camera_from_event(model)
    order_seed=json.loads((seed/'run_constraint_seed.json').read_text(encoding='utf-8'))
    before_order=audit_source_runs(sem_path,model,station,branch)
    if not before_order['passed']:raise ValueError('Original run-legal seed must remain available')
    world,finite,_=scene.world_rows(material,branch,station);rs=camera.project_rs(world,times);rs['valid'] &= finite
    progress(json.dumps(dict(stage='true_RS_initial_roots',accounted=n,valid=int(rs['valid'].sum()),unresolved=int((~rs['valid']).sum()))))
    recovered=recover_query_roots(scene,camera,material[profile],branch[profile],station[profile],times[profile],query,alpha_seed=alpha0[profile])
    solved=recovered['valid'];new_station=station.copy();new_station[profile[solved]]=recovered['stations'][solved]
    alpha=alpha0.copy();alpha[profile[solved]]=recovered['alpha'][solved]
    row=row0.copy();row[rs['valid']]=rs['row'][rs['valid']];row[profile[solved]]=recovered['row'][solved]
    if np.any(~np.isfinite(row)) or np.any(row<0) or np.any(row>1079):raise ValueError('No out-of-image RS witness may be inserted')
    after_order=audit_source_runs(sem_path,model,new_station,branch)
    if not after_order['passed']:raise ValueError('Query root recovery changed a verified order obligation')
    rows_in_runs=np.zeros(n,bool)
    for rec in before_order['records']:rows_in_runs[np.asarray(rec['source_rows'],int)]=True
    if np.any(new_station[rows_in_runs]!=station[rows_in_runs]):raise ValueError('Root recovery unexpectedly touched ordered runs')
    pids,physical=_physical(scene,model);oids,orders=_orders(scene,model,station,branch,material,order_seed)
    if len(physical)!=415:raise ValueError('Whole frozen physical inequality inventory changed')
    if np.any(physical>1e-8) or np.any(orders>1e-9):raise ValueError('Unchanged physical/order seed is not feasible')
    def evaluate(label,S,V,A):
        X,finite,_=scene.world_rows(material,branch,S);instant=camera.at(X,times+camera.tau*(V/1079.-.5));valid=finite&instant['valid'];e=instant['uv']-uv
        if not valid.all():raise ValueError(label+' has a true lifted arithmetic failure')
        q=np.sum(e*e,axis=1)/budget**2;q[profile]=(np.sum(sem['normal'][profile]*e[profile],axis=1)/budget[profile])**2
        hrs=V-instant['uv'][:,1];quv,_,qvalid=rational_query(query,A[profile]);hquery=instant['uv'][profile]-quv
        if not qvalid.all():raise ValueError('Immutable finite query arithmetic failed')
        cost,Wres=_statistics(e,sem,sem_path)
        true=camera.project_rs(X,times);true['valid'] &= finite;te=true['uv']-uv;tq=np.sum(te*te,axis=1)/budget**2;tq[profile]=(np.sum(sem['normal'][profile]*te[profile],axis=1)/budget[profile])**2
        summary=dict(all_source_rows=n,all_lifted_arithmetic_valid=bool(valid.all()),lifted_error_px=_summary(np.linalg.norm(e,axis=1)),lifted_q=_summary(q),lifted_max_q=float(q.max()),
          rs_equality_px=_summary(np.abs(hrs)),query_equality_px=_summary(np.linalg.norm(hquery,axis=1)),query_scalar_bounds_valid=bool(np.all((A[profile]>=query['bounds'][:,0])&(A[profile]<=query['bounds'][:,1]))),
          original_full_statistical_cost=cost,original_statistical_residual_count=len(Wres),true_RS_valid_rows=int(true['valid'].sum()),true_RS_invalid_rows=int((~true['valid']).sum()),
          true_RS_error_px_valid_subset=_summary(np.linalg.norm(te[true['valid']],axis=1)),true_RS_max_q_valid_subset=float(tq[true['valid']].max()),
          true_native_t_hat=float(np.sqrt(tq.max())) if true['valid'].all() else None,true_native_whole_W_cost=_statistics(te,sem,sem_path)[0] if true['valid'].all() else None)
        return summary,dict(source_rows=np.arange(n,dtype='<u8'),raw_error=e,q=q,h_RS=hrs,h_query=hquery,W_residual=Wres,true_RS_valid=true['valid'],true_RS_uv=true['uv'],true_RS_q=tq,domain_g=_domain(camera,instant))
    baseline,barrays=evaluate('original_order_capture',station,row0,alpha0);candidate,carrays=evaluate('root_capture',new_station,row,alpha)
    common_b=baseline['lifted_max_q']
    for report,arrays in [(baseline,barrays),(candidate,carrays)]:
        pixel=arrays['q']-common_b;g=np.r_[pixel,arrays['domain_g'].reshape(-1),physical,orders];h=np.r_[arrays['h_RS'],arrays['h_query'].reshape(-1)]
        report.update(common_budget_sq=common_b,phase_I_objective_at_same_b=common_b,all_g_count=len(g),all_h_count=len(h),max_g=float(g.max()),max_abs_h=float(np.max(np.abs(h))),
          active_inequalities_at_zero_multiplier=int(np.count_nonzero(g>=-1e-10)),violated_inequalities=int(np.count_nonzero(g>0)),
          same_initial_PHR_rho1_lambda0_nu0_cost=float(common_b+.5+.5*np.sum(np.maximum(g,0)**2)+.5*np.sum(h*h)),proposed_new_phase_I_budget_sq=float(max(1.,arrays['q'].max())),
          physical_inequality_count=len(physical),source_domain_inequality_count=3*n,domain_scales=[1.,1.,1.],domain_max_g_by_kind=arrays['domain_g'].max(axis=0).tolist(),order_inequality_count=len(orders),physical_max_g=float(physical.max()),order_max_g=float(orders.max()))
        arrays.update(g=g,h=h,physical_g=physical,order_g=orders)
    new_station.astype('<f8').tofile(out/'station.f64');branch.astype('<u4').tofile(out/'path.u32');row.astype('<f8').tofile(out/'initial_row.f64');alpha.astype('<f8').tofile(out/'initial_alpha.f64')
    order_seed['station_sha256']=sha256_file(out/'station.f64');order_seed['branch_sha256']=sha256_file(out/'path.u32')
    order_seed['model']=dict(path=str(model_path),sha256=sha256_file(model_path))
    order_seed['root_capture_provenance']='Only genuine profiles not present in source runs were recovered; every recorded run row station/path/lift is unchanged.'
    atomic_json(out/'run_constraint_seed.json',order_seed);atomic_json(out/'source_order_before.json',before_order);atomic_json(out/'source_order_after.json',after_order)
    np.savez_compressed(out/'baseline_all_constraints.npz',**barrays);np.savez_compressed(out/'candidate_all_constraints.npz',**carrays)
    np.savez_compressed(out/'query_recovery.npz',source_rows=profile,numerical_root_valid=solved,station_before=station[profile],station_candidate=new_station[profile],alpha_before=alpha0[profile],alpha_candidate=alpha[profile],
      equality_px=recovered['query_equality_px'],local_transversality_sine=recovered['local_transversality_sine'],relevant_interval_certified=recovered['relevant_root_interval_certified'])
    report=dict(schema='rinkfit.full_source_root_capture_comparison.v1',model=dict(path=str(model_path),sha256=sha256_file(model_path),all_parameter_values_unchanged=True),
      source_hash=model['source_identity'],semantic_sha256=sha256_file(sem_path),query_sha256=sha256_file(run/'query_packet/manifest.json'),budget_sha256=sha256_file(packet/'budget.f64'),
      original_seed=dict(path=str(seed),station_sha256=sha256_file(seed/'station.f64'),branch_sha256=sha256_file(seed/'path.u32')),
      rows=n,frames=1248,material_labels=171,ordinary_source_order_runs=15419,order_failed_before=before_order['failed_runs'],order_failed_after=after_order['failed_runs'],
      true_RS_numerical_initial_roots=int(rs['valid'].sum()),RS_unresolved_rows=np.flatnonzero(~rs['valid']).tolist(),
      query_profiles=len(profile),query_numerical_solutions=int(solved.sum()),query_unresolved_rows=profile[~solved].tolist(),query_relevant_interval_certified=0,
      unresolved_query_policy='Retain original finite material station and source alpha; no row deleted, relabeled or widened.',
      baseline=baseline,candidate=candidate,physical_constraint_ids=pids,order_constraint_ids=oids,
      files={name:dict(path=str(out/name),sha256=sha256_file(out/name))for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64','run_constraint_seed.json']},
      unchanged_ordered_source_rows=int(rows_in_runs.sum()),changed_station_rows=int(np.count_nonzero(new_station!=station)),unchanged_physical_parameters=True,
      query_root_status='local numerical nonsingularity; full relevant-interval query certificate pending',physical_fit_success=False,elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'comparison.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--model',required=True);p.add_argument('--seed-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    report=capture_roots(a.run_dir,a.model,a.seed_dir,a.output_dir);print(json.dumps({k:report[k]for k in ['rows','query_numerical_solutions','order_failed_after','elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
