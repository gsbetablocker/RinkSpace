"""Use the real native Jacobian to isolate shoe pressure while replaying the unchanged full objective."""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve

from calibration.program_tracking.body_tracking.ordered_native_skate import OrderedNativeSkateFactors
from calibration.program_tracking.body_tracking.programme_skate_replay import SkateReplay,save,write
from scripts.refit_programme_skates import evaluate_guard
from .solve_tracked_programme import statistics, verify_candidate

LO, HI = 79.02,95.0


def exact_source(r):
    mask=[i for i,row in enumerate(r.ordered.rows) if LO<row['pts']/12800<=HI and r.ordered.matches[i] is not None]
    records=[]
    for i in mask:
        row=r.ordered.rows[i]
        radius=.0001
        # Retain the existing measurement budget, but make the diagnostic
        # dead zone near zero. The real candidate acceptance uses the
        # original bands and exactly the same production registry.
        factor=radius/row['radius_px']
        ends=[dict(e,radius_px=e['radius_px']*factor) for e in row.get('structural_endpoints',[])]
        records.append(dict(row,radius_px=radius,structural_endpoints=ends))
    f=OrderedNativeSkateFactors(r.problem,records,mesh=r.ordered.mesh)
    f.weights=[r.ordered.weights[i]*(.0001/r.ordered.rows[i]['radius_px'])**2 for i in mask]
    for a,b,child in f.batches:
        child.weights=f.weights[a:b]
    f.matches=tuple(r.ordered.matches[i] for i in mask)
    return f


def residual_summary(f,value):
    groups={}
    for row in value['metrics'][f.name]:
        key=row['semantic']
        groups.setdefault(key,[])
        if row.get('distance_px'):
            groups[key].append(float(np.median(row['distance_px'])))
    return {k:dict(rows=len(v),median_px=float(np.median(v)),p95_px=float(np.percentile(v,95)))
            for k,v in groups.items() if v}


def project_trial(f,base,x,association,*,jacobian=False):
    f.associate_changed(x[17:],base[17:],association)
    missing=sum(m is None for m in f.matches)
    value=f.evaluate(x[17:],jacobian)
    cost=float(.5*value['residual']@value['residual'])
    return value,cost,missing


def restrict_steps(d,columns,scale):
    # Preserve the calculated direction. Retraction through the archived
    # SO(3) controls remains exact. These are diagnostic trust radii.
    v=np.zeros((int(columns[-1])-17)//3*3+20)
    v[columns]=d
    vec=v[17:].reshape(-1,3)
    cap=np.max(np.linalg.norm(vec,axis=1))
    return d*min(1.,scale/max(cap,1e-15))


def make_step(H,g,Q,v,pressure):
    M=(H+pressure*Q).tocsc()
    damping=.018*np.maximum(M.diagonal(),1.e-4)
    return spsolve(M+diags(damping,format='csc'),-g-pressure*v)


def choose_trials(r,f,J,b,cols):
    association=f.matches
    initial,b0,missing=project_trial(f,r.x,r.x,association,jacobian=True)
    if missing:
        raise ValueError('Original represented source paths were lost')
    body=(cols<17+r.problem.body.state.body_size)
    modes={'linked_leg':cols[body],'shoe_relative':cols[~body]}
    exact0=residual_summary(f,initial)
    r.emit('source_pressure_baseline',target_rows=len(f.rows),cost=b0,by_part=exact0)
    data=[]
    for name,active in modes.items():
        mask=np.isin(cols,active)
        A=J[:,mask]
        B=initial['jac'][:,active]
        H,g,Q,v=A.T@A,A.T@b,B.T@B,B.T@initial['residual']
        for pressure in ((2.,8.,32.,128.,1024.) if name=='linked_leg' else (1024.,)):
            d=restrict_steps(make_step(H,g,Q,v,pressure),active,.45 if pressure>=128 else .28)
            for frac in (1.,.5,.25) if name=='linked_leg' else (1.,):
                x=r.x.copy();x[active]+=frac*d
                pred=J[:,mask]@(frac*d)
                model_delta=float(b@pred+.5*(pred@pred))
                v1,cost,n=project_trial(f,r.x,x,association)
                info=dict(name=name,pressure=pressure,fraction=frac,raw_cost=cost,missing_paths=n,
                    approximate_production_delta=model_delta,by_part=residual_summary(f,v1),
                    peak_control_deg=float(np.rad2deg(np.max(np.abs(frac*d)))))
                r.emit('native_pixels_trial',**info)
                if n==0 and cost<b0:
                    data.append((info,x))
    write(r.args.output/'pixel_trials.json',[m for m,_ in data])
    return data,association,b0


def refine_pressure(r,f,J,b,cols,association,x,pressure,count):
    active=cols<17+r.problem.body.state.body_size
    ix=cols[active]
    A=J[:,active]
    H=A.T@A
    for iteration in range(count):
        val,cost,missing=project_trial(f,r.x,x,association,jacobian=True)
        B=val['jac'][:,ix]
        g=A.T@(b+A@(x[ix]-r.x[ix]))
        d=restrict_steps(make_step(H,g,B.T@B,B.T@val['residual'],pressure),ix,.35)
        chosen=None
        for frac in (1.,.5,.25):
            test=x.copy();test[ix]+=frac*d
            value,source,new_missing=project_trial(f,r.x,test,association)
            r.emit('source_pressure_continuation',iteration=iteration,pressure=pressure,
                   fraction=frac,raw_cost=source,previous_cost=cost,missing_paths=new_missing,
                   by_part=residual_summary(f,value))
            if not new_missing and source<cost:
                chosen=test
                break
        if chosen is None:
            break
        x=chosen
    v,c,n=project_trial(f,r.x,x,association)
    return dict(name='linked_leg_tight',pressure=pressure,raw_cost=c,missing_paths=n,by_part=residual_summary(f,v)),x


def compact_evidence(r,before,after,guard):
    keys=[k for k in before['weighted'] if (k in (
        'body_pixels','dynamic_foot_compatibility','cavity_AL','finite_ice','anatomical_joint_domain',
        'ordered_native_skate','manual_native_leg','surface_intersection','model_com_motion','full_source_skate_structure')
        or k.startswith('episode/spin_support/') or k.startswith('episode/flying_camel_path/'))]
    return dict(before_cost=before['cost'],after_cost=after['cost'],
        weighted={k:[before['weighted'][k],after['weighted'][k]] for k in keys},
        body_before=before['body'],body_after=after['body'],raw=statistics(r,before,after),
        source_conflicts=guard['source_rows'],guard=guard['failures'],
        native_anatomy=guard['pointwise_anatomical_comparison'].get('temporal_skate_s05'),
        event_cavity=guard['per_event_cavity'].get('temporal_skate_s05'))


def run(r,matrix):
    cols=matrix['columns']
    np.testing.assert_array_equal(r.x,matrix['x'])
    J=csr_matrix((matrix['data'],matrix['indices'],matrix['indptr']),shape=tuple(matrix['shape']))
    b=matrix['residual']
    f=exact_source(r)
    rows,fassoc,b0=choose_trials(r,f,J,b,cols)
    tight=min((p for p in rows if p[0]['name']=='linked_leg'),key=lambda p:p[0]['raw_cost'])
    forced=refine_pressure(r,f,J,b,cols,fassoc,tight[1],1024.,3)
    rel=[p for p in rows if p[0]['name']=='shoe_relative']
    safe=[p for p in rows if p[0]['name']=='linked_leg' and p[0]['approximate_production_delta']<0]
    safe=sorted(safe,key=lambda p:p[0]['raw_cost'])
    selected=[('forced_linked',*forced)]
    if rel:
        selected.append(('forced_shoe_relative',*rel[0]))
    if safe:
        selected.append(('predicted_feasible_linked',*safe[0]))
    save(r.args.output/'pressure_steps.npz',x=r.x,**{name:x for name,_,x in selected})
    write(r.args.output/'objective_weights.json',r.weights_record())
    r.emit('start_actual_comparison',selected=[dict(label=k,**a) for k,a,_ in selected])
    base=r.full(r.x,'same_parent_native_replay')
    parent_matches=r.ordered.matches
    reports=[]
    eligible=[]
    for label,info,x in selected:
        r.ordered.associate_changed(x[17:],r.x[17:],parent_matches)
        trial=r.full(x,label,verify_rows=False)
        guard=evaluate_guard(r,base,trial)
        compare=compact_evidence(r,base,trial,guard)
        reports.append(dict(label=label,tight_source=info,**compare))
        r.emit('actual_production_pressure_result',**reports[-1])
        if label=='predicted_feasible_linked' and not guard['failures']:
            eligible.append((x,dict(label=label,tight_source=info,**compare)))
        write(r.args.output/('replay_'+label+'.json'),reports[-1])
    write(r.args.output/'comparison.json',dict(source_start=b0,results=reports))
    if eligible:
        x,row=eligible[0]
        verify_candidate(r,x,row,scan_seed=(r.s0,r.x),initial_matches=parent_matches)
    else:
        r.emit('pressure_diagnostic_complete',promoted=False)


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','latest','latest-surface','output','matrix'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.previous=a.latest
    a.previous_surface=a.latest_surface
    a.output.mkdir()
    with (a.output/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        r=SkateReplay(a,log)
        with np.load(a.matrix,allow_pickle=False) as m:
            run(r,{k:m[k] for k in m.files})


if __name__=='__main__':
    main()
