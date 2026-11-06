"""Whole-sequence geometric feasibility and attached common-state initializer.

Diagnostic only: loosen body/foot compatibility to test whether the *existing*
template and native camera can explain one observed shoe over the full action.
Reuses the same root and relative-shoe curves; no independent frame poses.
Ankle transfer creates an attached body initializer, never a free-shoe result.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,time
import numpy as np
from scipy.sparse import csr_matrix,vstack,diags,kron,eye
from scipy.optimize import least_squares
from .blade_projection import SavedBladeProjection
from .sequence_skate_state import SequenceState
from .source_skate_structure_factors import FullSkateStructureFactors
from .complete_runner_extent import CompleteRunnerExtentFactors
from .source_support_locations import SourceSupportLocationFactors
from .absolute_motion_curve import _log,greville_times
from .absolute_motion_state import evaluate_curve
from .skate_physics import fk_state
from .fullbody_round import write_json
from .rebuild_prepare import digest


def run(project,stage,parameter,document,output,foot=1,wall_seconds=180.):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.perf_counter();log_path=output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path),diagnostic_only=True,whole_seconds=[76,95],foot=foot,wall_seconds=wall_seconds)),flush=True)
    q=SavedBladeProjection(project,stage,parameter);ref=q.reference;state=SequenceState(ref,(76.,95.))
    ref.grasp_relations=dict(factor_parameters=dict(shoe_region_weight_multiplier=160.))
    body=SimpleNamespace(state=state,size=state.size,tau=float(q.camera.p['rig'][37]))
    p=SimpleNamespace(reference=ref,binding=q.binding,body=body,camera=q.camera,size=17+state.size,optimize_shared=False,factor_derivative_columns=np.arange(3))
    def filtered(key):
        d=copy.deepcopy(document[key]);d['rows']=[r for r in d['rows']if r['foot']==foot];return d
    factors=[FullSkateStructureFactors(p,filtered('full_source_skate_structures')),CompleteRunnerExtentFactors(p,filtered('complete_runner_extent')),
        SourceSupportLocationFactors(p,document['source_support_locations'],support_evidence=[r for r in document['source_support_evidence']if r['foot']==foot],**document['source_structure_budget'])]
    selection=np.r_[np.concatenate([17+i*69+np.arange(3)for i in range(state.m)]),17+state.body_size+np.concatenate([i*6+foot*3+np.arange(3)for i in range(state.shoe_m)])]
    x0=np.r_[q.binding.x,np.zeros(state.size)];ridge=np.full(len(selection),.02);Rridge=diags(ridge,format='csr')
    B2=state.basis(ref.times,2);smooth=kron(csr_matrix(B2*np.sqrt(.02)/20.),eye(3),format='csr')
    smooth=csr_matrix((smooth.data,smooth.indices,smooth.indptr),shape=(smooth.shape[0],len(selection)))
    cache={};accepted=[0];best=x0.copy();last_values=None
    with log_path.open('a',encoding='utf-8',buffering=1)as log:
        def evaluate(z,jacobian=False):
            nonlocal cache,last_values
            if cache and np.array_equal(z,cache['z'])and(not jacobian or cache['J']is not None):return cache
            x=x0.copy();x[selection]=z;values=[f.evaluate(x[17:],jacobian)for f in factors];last_values=values
            r=np.r_[*[v['residual']for v in values],ridge*z,smooth@z];J=None
            if jacobian:J=vstack([v['jac'][:,selection]for v in values]+[Rridge,smooth],format='csr')
            cache=dict(z=z.copy(),r=r,J=J,x=x,values=values)
            log.write(json.dumps(dict(event='evaluation',jacobian=jacobian,seconds=time.perf_counter()-started,cost=float(r@r/2),source_costs=[v['image_cost']for v in values]))+'\n');return cache
        first=evaluate(x0[selection],True);before={k:v for value in first['values']for k,v in value['metrics'].items()}
        def callback(z):
            nonlocal best
            accepted[0]+=1;best=evaluate(z)['x'].copy()
            np.savez_compressed(output/'last_accepted_auxiliary.npz',x=best,source_pts=ref.p['pts'],not_final_attached_result=np.array(True))
            if time.perf_counter()-started>=wall_seconds:raise StopIteration()
        fit=least_squares(lambda z:evaluate(z)['r'],x0[selection],jac=lambda z:evaluate(z,True)['J'],method='trf',tr_solver='lsmr',x_scale='jac',max_nfev=45,ftol=1e-5,xtol=1e-6,gtol=1e-4,callback=callback,tr_options=dict(atol=1e-6,btol=1e-6,maxiter=250))
        final=evaluate(fit.x);best=final['x'];after={k:v for value in final['values']for k,v in value['metrics'].items()}
    # Move the auxiliary shoe rotation into the corresponding anatomical ankle.
    # This preserves the existing attachment/ankle pivot and gives body IK a
    # physical starting point; interpolation error is measured, never hidden.
    seed=best.copy();seed[17+state.body_size:]=0.;tg=greville_times(state.curve.knots)
    rel_aux=state.shoe_evaluation(best[17:],tg)['rotations'];rel_base=state.shoe_evaluation(np.zeros(state.size),tg)['rotations']
    RFB=q.binding.parameters()[0][foot]
    delta=RFB@rel_aux[:,foot]@rel_base[:,foot].swapaxes(-1,-2)@RFB.T
    body_control=seed[17:17+state.body_size].reshape(state.m,69);joint=7+foot;start=3+joint*3;body_control[:,start:start+3]=_log(delta)
    auxiliary=fk_state(ref,ref.at(ref.times),state.increments(best[17:],ref.times));common=fk_state(ref,ref.at(ref.times),state.increments(seed[17:],ref.times))
    difference=_log(auxiliary['R'][:,foot].swapaxes(-1,-2)@common['R'][:,foot]);translation=np.linalg.norm(auxiliary['O'][:,foot]-common['O'][:,foot],axis=-1)
    np.savez_compressed(output/'common_initialization_x.npz',x=seed,parent_parameter_sha256=np.array(digest(parameter)),source_pts=ref.p['pts'],initializer_only=np.array(True))
    np.savez_compressed(output/'auxiliary_trajectory.npz',pts=ref.p['pts'],shoe_R=auxiliary['R'],shoe_O_solver=auxiliary['O'],common_seed_R=common['R'],common_seed_O_solver=common['O'],foot=np.array(foot),not_a_final_candidate=np.array(True))
    write_json(output/'geometry_feasibility.json',dict(parent_parameter=str(parameter),parent_sha256=digest(parameter),source_before=before,source_after=after,accepted_steps=accepted[0],wall_seconds=time.perf_counter()-started,solver_stop=str(fit.message),
        complete_source_only_no_body_or_cavity_quality_claim=True,shared_template_camera_and_scale_unchanged=True,whole_source_frames=len(ref.times),
        common_seed_transfer_orientation_max_deg=float(np.rad2deg(np.linalg.norm(difference,axis=-1)).max()),common_seed_transfer_origin_max_m=float(translation.max()*q.ratio),
        coordinate_layout='same 17 shared + full body + relative-shoe control vector; auxiliary selected root XYZ and one shoe SO3 only',final_geometry_unaccepted=True))
    print(json.dumps(dict(event='auxiliary_complete',accepted=accepted[0],seconds=time.perf_counter()-started,initializer=str(output/'common_initialization_x.npz'))),flush=True)


def main():
    ap=argparse.ArgumentParser()
    for name in ('project','stage','parameter','document','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--wall-seconds',type=float,default=180.);ap.add_argument('--foot',type=int,choices=(0,1),default=1);a=ap.parse_args()
    run(a.project,a.stage,a.parameter,json.loads(a.document.read_text(encoding='utf-8')),a.output,a.foot,a.wall_seconds)

if __name__=='__main__':main()
