"""Strict same-budget convex-segment restoration of a reduced W proposal.

Does not increase b or tolerate failed source rows. Finds the largest IEEE
fraction between a strictly feasible primary and the numerical W proposal for
which every original affine source norm passes. Other convex constraints are
preserved by the segment; a full legacy scene/domain check remains mandatory.
"""
from pathlib import Path
import argparse,json,hashlib,time
from copy import deepcopy
import numpy as np
from scipy import sparse
from rinkfit_start.ad_chart import PrintOnlyAdChart
from rinkfit_start.native_scene import NativeScene


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(d,base=None):
    p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
    if sha(p)!=d['sha256']:raise ValueError('Changed immutable input '+str(p))
    return p

def array(d,base):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])


def run(primary,secondary,group_chart,output):
    started=time.perf_counter();P=Path(primary).resolve();D=Path(secondary).resolve();O=Path(output).resolve();O.mkdir(exist_ok=False)
    ass=read(D/'assembly.json');CP=verify(ass['authority']['checkpoint']);cp=read(CP);model=read(verify(cp['model']))
    chart=PrintOnlyAdChart(model);pa=np.load(P/'proposal.npz');a=pa['native_primal'][:-1];n=int(ass['physical_columns']);k=int(ass['station_columns'])
    sr=read(D/'W/report.json');last=sr['history'][-1];g=last['generation'];raw=np.load(D/('W/generation_%03d/raw_solver.npz'%g))['x']
    ar=np.load(D/'primary_arrays.npz');lo=ar['lower_native'][:-1];hi=ar['upper_native'][:-1];units=ar['units'][:-1];budget=float(sr['primary_working_b'])
    eta0=a[n:];ss=np.maximum(abs(lo[n:]-eta0),abs(hi[n:]-eta0));trust=float(units[0])
    b=np.r_[a[:n]+trust*raw[:n],eta0+ss*raw[n:]]
    m=int(ass['all_active_source_SOC4']);AA=sparse.load_npz(D/'primary_A.npz').tocsr();rhs=ar['rhs'];qref=float(np.load(D/'primary_arrays.npz')['units'][-1])
    start=AA.shape[0]-4*m;uvrow=(start+4*np.arange(m)[:,None]+np.array([1,2])).ravel()
    J=(-np.sqrt(qref)/2)*(AA[uvrow,:-1]@sparse.diags(1/units));r0=(np.sqrt(qref)/2*rhs[uvrow]).reshape(m,2)
    ra=r0+(J@a).reshape(m,2);rb=r0+(J@b).reshape(m,2)
    qa=np.einsum('ij,ij->i',ra,ra);qb=np.einsum('ij,ij->i',rb,rb)
    if np.any(qa>budget):raise ValueError('Restoration anchor is not strictly within the unchanged budget')
    def q(lam):
        # Audit the actual candidate vector using the saved operator. Avoid
        # asserting IEEE linear-combination arithmetic equals its evaluation.
        r=r0+(J@(a+lam*(b-a))).reshape(m,2);return np.einsum('ij,ij->i',r,r)
    low=0.;high=1.
    if np.all(q(high)<=budget):low=high
    while np.nextafter(low,np.inf)<high:
        mid=(low+high)/2
        if np.all(q(mid)<=budget):low=mid
        else:high=mid
    restored=a+low*(b-a);qr=q(low)
    if np.any(qr>budget):raise ValueError('Restoration did not satisfy every unchanged affine norm')
    G=Path(group_chart).resolve();gm=read(G);JW=sparse.load_npz(D/'full_original_W_Jphysical.npz');JS=sparse.load_npz(verify(gm['arrays']['Jstation'],G.parent));rw=array(gm['arrays']['rW'],G.parent)
    selected=array(gm['arrays']['station_source_rows'],G.parent).astype(int)
    if not np.array_equal(selected,pa['station_rows']):raise ValueError('Station columns differ from verified full-W chart')
    def cost(v):
        r=rw+JW@v[:n]+JS@v[n:];return float(.5*r@r)
    reduced_delta=np.zeros(len(chart.free));released=ar['released_full_physical_columns'].astype(int);reduced_delta[released]=ar['reduced_physical_scale']*restored[:n]
    candidate,domain=chart.unpack(chart.pack()+reduced_delta,return_domain=True)
    mp=O/'model.json';mp.write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf8')
    stations=array(cp['row_state_files']['station'],CP.parent).copy();stations[selected]+=restored[n:]
    sp=O/'station.f64';stations.astype('<f8').tofile(sp)
    aux={key:dict(path=str(verify(desc,CP.parent)),sha256=desc['sha256'],dtype=desc['dtype'],shape=desc['shape'])for key,desc in cp['row_state_files'].items()};aux['station']=dict(path=str(sp),sha256=sha(sp),dtype='<f8',shape=[len(stations)])
    cm=dict(schema='rinkfit.physical_print_chart_trial_seed.v1',parent_checkpoint=ref(CP),model=ref(mp),row_state_files=aux,
        fraction=1.,restoration_fraction=low,requires_fresh_true_RS_profile_and_all_station_evaluation=True,accepted_nonlinear_candidate=False)
    cmP=O/'candidate_manifest.json';cmP.write_text(json.dumps(cm,indent=2),encoding='utf8')
    np.savez_compressed(O/'restored.npz',primary_native=a,W_raw_native=b,restored_native=restored,source_q=qr,active_source_rows=np.flatnonzero(array(read(verify(ass['authority']['semantic']))['active_source_mask'],verify(ass['authority']['semantic']).parent)))
    original_spec=read(verify(ass['input']));Aold=sparse.load_npz(verify(original_spec['original_A'])).tocsr();original_bounds=np.load(verify(original_spec['rhs_bounds']));original_rhs=original_bounds['rhs']
    lift=sparse.load_npz(D/'old_scaled_from_physical_scaled.npz');A=sparse.hstack([Aold[:,:-1]@lift,Aold[:,-1:]],format='csr')
    v=np.r_[restored[:n],budget];linear_violation=A@v-original_rhs;ieee=64*np.finfo(float).eps*(1+abs(original_rhs)+abs(A)@abs(v))
    sensorp=sparse.load_npz(verify(gm['arrays']['sensor_Jphysical'],G.parent))@lift;sensors=sparse.load_npz(verify(gm['arrays']['sensor_Jstation'],G.parent));sensor0=array(gm['arrays']['sensor_row0'],G.parent)
    sv=sensor0+sensorp@restored[:n]+sensors@restored[n:]
    determinants=np.asarray([b['values'][0]*b['values'][4]-b['values'][1]*b['values'][3]for b in candidate['blocks']if b['name'].startswith('print/')])
    actual_audit=dict(original_A_beyond_IEEE=int(np.count_nonzero(linear_violation>ieee)),maximum_original_A_violation=float(np.max(linear_violation,initial=0.)),
        strict_sensor_pass=bool(np.all((sv>=0)&(sv<=1079))),minimum_sensor_row=float(sv.min()),maximum_sensor_row=float(sv.max()),
        production_print_determinant_min=float(determinants.min()),production_print_determinants_pass=bool(np.all(determinants>=1e-6)),
        strict_physical_and_finite_station_box=bool(np.all(restored>=lo)&np.all(restored<=hi)))
    if actual_audit['original_A_beyond_IEEE'] or not all(actual_audit[k]for k in ['strict_sensor_pass','production_print_determinants_pass','strict_physical_and_finite_station_box']):raise ValueError('Actual restored original affine constraint audit failed')
    report=dict(schema='rinkfit.strict_same_budget_W_convex_segment_restore.v1',source=ref(__file__),primary=ref(P/'report.json'),secondary=ref(D/'W/report.json'),original_group_W_chart=ref(G),candidate_manifest=ref(cmP),
        unchanged_budget=budget,maximum_primary_q=float(qa.max()),maximum_raw_W_q=float(qb.max()),maximum_restored_q=float(qr.max()),all_active_source_count=m,violating_sources_after=int(np.count_nonzero(qr>budget)),
        restoration_fraction=low,raw_W_violating_sources=last['full_source_audit']['violating_source_rows'],
        original_parent_W_half_cost=float(.5*rw@rw),primary_W_half_cost=cost(a),raw_W_half_cost=cost(b),restored_W_half_cost=cost(restored),
        parent_W_change=cost(restored)-float(.5*rw@rw),primary_W_change=cost(restored)-cost(a),
        physical_columns=n,station_columns=k,box_pass=bool(np.all(restored>=lo)&np.all(restored<=hi)),
        all_other_convex_constraints_preserved_by_anchor_and_raw_segment=True,exact_domain=domain,
        full_decoded_scene=NativeScene(candidate).check_physics(),restored_original_constraint_audit=actual_audit,no_budget_tolerance_relaxation=True,
        accepted_nonlinear_candidate=False,elapsed_seconds=time.perf_counter()-started)
    (O/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:report[k]for k in ['schema','maximum_restored_q','unchanged_budget','restored_W_half_cost','restoration_fraction','violating_sources_after','elapsed_seconds']},ensure_ascii=False),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',required=True);p.add_argument('--secondary',required=True);p.add_argument('--group-chart',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.primary,a.secondary,a.group_chart,a.output)
