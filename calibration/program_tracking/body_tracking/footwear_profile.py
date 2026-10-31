"""Conditional native-pixel root-depth/lateral profiles; no probability claim."""
from pathlib import Path
import argparse,csv,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr
from .footwear_joint import JointFootwearProblem
from .rebuild_prepare import load,digest


def run(root,config_path,reference,output):
    root,reference,output=map(Path,[root,reference,output]);cfg=json.loads(Path(config_path).read_text(encoding='utf8'));settings=cfg['profile'];saved=load(reference);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rows=[]
    source=load(root/'input.npz');times=source['pts']/12800.;C=source['camera_centre']*float(source['metres_per_scene_unit']);ray=saved['P_world_mid']-C;ray/=np.linalg.norm(ray,axis=1)[:,None];lateral=np.c_[-ray[:,1],ray[:,0],np.zeros(len(ray))];lateral/=np.linalg.norm(lateral,axis=1)[:,None]
    with (output/'profile_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        log.write(json.dumps(dict(event='start',reference_sha256=digest(reference),config_sha256=digest(config_path),camera_frozen=True,contact_feedback=False,kind=settings['type']))+'\n')
        for center in settings['source_seconds']:
            window=(int(round((center-settings['window_half_width_s'])*12800)),int(round((center+settings['window_half_width_s'])*12800)))
            p=JointFootwearProblem(root,cfg['source_review_file'],cfg['observation_dataset'],cfg['base_body_file'],'J0',window=window);p.g=saved['shared_geometry'].copy();base=saved['leg_controls'].ravel().copy();active=(np.unique(p.bc)[:,None]*p.dof+np.arange(p.dof)).ravel();envelope=np.exp(-.5*((times-center)/settings['envelope_sigma_s'])**2)
            for axis,directions in [('depth',ray),('lateral',lateral)]:
                shift=np.stack([lsqr(p.p.Bfull,envelope*directions[:,j],atol=1e-12,btol=1e-12)[0] for j in range(3)],1)
                offsets=list(settings['depth_offsets_conditional_m'] if axis=='depth' else settings['lateral_offsets_conditional_m']);evaluated={}
                def point(offset):
                    p.fixed_root=saved['root_delta_controls']+offset*shift;p.cache_x=None;p.cache=None
                    def expand(x):
                        full=base.copy();full[active]=x;return full
                    def fun(x):return p.fun(expand(x))
                    def jac(x):return p.jac(expand(x))[:,active]
                    fit=least_squares(fun,base[active],jac=jac,bounds=(p.bounds[0][active],p.bounds[1][active]),method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-4,tr_options={'atol':1e-7,'btol':1e-7})
                    v=p.evaluate(expand(fit.x),jacobian=True);data=sum(value for key,value in v['parts'].items() if key in ['body_pixels','visible_boot_edges','visible_blade_curves','boot_direction_endpoints']);prior=sum(v['parts'].values())-data
                    row=dict(source_seconds=center,axis=axis,offset_conditional_m=offset,data_cost=data,engineering_prior_cost=prior,total_cost=float(fit.cost),optimality=float(fit.optimality),gradient_inf=float(np.max(abs(v['jac'][:,active].T@v['residual']))),termination=fit.message,nfev=fit.nfev,njev=fit.njev,local_source_exposures=p.n,active_leg_controls=len(active),parts=v['parts'])
                    rows.append(row);evaluated[offset]=row;log.write(json.dumps(dict(event='profile_point',**row,elapsed_seconds=time.perf_counter()-start))+'\n');np.savez_compressed(output/f'leg_profile_{center:.3f}_{axis}_{offset:+.3f}.npz',x=expand(fit.x),fixed_root_delta_controls=p.fixed_root,correction_knots=p.p.knots)
                for offset in offsets:point(float(offset))
                # Extend only if the observed data optimum lies at the explored
                # edge; projection-domain failure is retained as truncation.
                while axis=='depth':
                    best=min(evaluated,key=lambda value:evaluated[value]['data_cost']);lo,hi=min(evaluated),max(evaluated)
                    if best!=lo and best!=hi:break
                    next_offset=lo*2 if best==lo else hi*2
                    try:point(next_offset)
                    except ValueError as error:
                        log.write(json.dumps(dict(event='profile_domain_truncation',source_seconds=center,axis=axis,offset=next_offset,error=str(error)))+'\n');break
        result=dict(schema='rinkspace.conditional-root-profile.v1',reference=str(reference.resolve()),reference_sha256=digest(reference),camera_updated=False,contact_feedback=False,probabilistic_confidence_interval=False,rows=rows,elapsed_seconds=time.perf_counter()-start)
        (output/'profiles.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');fields=[key for key in rows[0] if key!='parts']
        with (output/'profiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({key:row[key] for key in fields} for row in rows)
        log.write(json.dumps(dict(event='complete',profile_points=len(rows),seconds=time.perf_counter()-start))+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,7),layout='constrained')
    for column,center in enumerate(settings['source_seconds']):
        for i,axis in enumerate(['depth','lateral']):
            data=sorted([r for r in rows if r['source_seconds']==center and r['axis']==axis],key=lambda r:r['offset_conditional_m']);minimum=min(r['data_cost'] for r in data);prior=min(r['engineering_prior_cost'] for r in data);x=[r['offset_conditional_m'] for r in data]
            axes[i,column].plot(x,[r['data_cost']-minimum for r in data],'o-',label='source data');axes[i,column].plot(x,[r['engineering_prior_cost']-prior for r in data],'s--',label='engineering priors');axes[i,column].set(title=f'{center:.3f}s / {axis}',xlabel='conditional metres',ylabel='cost change');axes[i,column].grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('J1 conditional root offsets with local legs reoptimized; not a confidence interval');fig.savefig(output/'depth_lateral_profiles.png',dpi=150);plt.close(fig);print(json.dumps(dict(profile_points=len(rows),elapsed_seconds=time.perf_counter()-start,output=str(output))),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.config,a.reference,a.output)
