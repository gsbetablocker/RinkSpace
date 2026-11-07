"""A coupled P/G initialization proposal on the unchanged real objective.

Uses an explicitly recorded cached Jacobian as a quasi-Newton proposal only.
Every trial is re-evaluated with the current joint body/shoe/camera factors.
There are no per-shot offsets, independent fitted windows or new observations.
"""
from pathlib import Path
import argparse,copy,json,time,shutil
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve
from .trajectory_backend import load_reference,PrimaryActions,primary_policy
from .trajectory_probe import body_groups
from .sequence_refit import SequenceJoint,SequenceActions
from .continuous_physics import ContinuousMassGeometry
from .body_execution import BodyExecution,BodyExecutionConfig
from .absolute_motion_state import curve_from_packet,absolute_reference
from .sequence_skate_state import shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_delivery import export_candidate
from .ice_soft import IceSoftPolicy
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .body_clock import source_seconds


def run(project,prepared,parent,output,cache_stage,wall_seconds=240.,*,protect_source_controls=False):
    project,prepared,parent,output,cache_stage=map(Path,(project,prepared,parent,output,cache_stage));output.mkdir(parents=True,exist_ok=False);(output/'solve').mkdir()
    started=time.perf_counter();saved=load(parent/'solve/candidate.npz');ref,camera,info,doc,seed=load_reference(project,prepared);validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    ref.absolute_curve=curve_from_packet(saved,seed=True);ref.relative_skate_curve=shoe_curve_from_packet(saved,seed=True)
    for field,name in(('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz')):
        if str(saved[field])!=digest(prepared/name):raise ValueError('Prepared identity changed')
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,gpu_batch_size=1024,local_derivatives='analytic',compatibility_backend='cuda_resident'));execution.attach(ref)
    policy=IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))
    with (output/'progress.jsonl').open('a',encoding='utf-8',buffering=1)as log,execution.limits():
        sec=source_seconds(ref.p);p=SequenceJoint(ref,camera,prepared/'observations.npz',tuple(sec[[0,-1]]),doc.get('supports',[]),output/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=policy)
        x=saved['x'].copy();mass=ContinuousMassGeometry(ref);base=PrimaryActions(p,ref,mass,doc);p.action_factors=SequenceActions(p,ref,mass,doc,np.zeros(p.body.size),base_factors=base);primary_policy(p)
        p.action_factors.evaluate(np.zeros(p.body.size),False)
        baseline=json.loads((parent/'primary_quality_baseline.json').read_text(encoding='utf-8'));first=p.evaluate(x,False);before_P=ref.at(ref.times)['P']+p.body.state.increments(x[17:],ref.times)[:,:3];before_jump=float(np.max(np.linalg.norm(np.diff(before_P,axis=0),axis=1)))*ref.binding.ratio
        cache=cache_stage/'linear_cache';identity=json.loads((cache/'identity.json').read_text(encoding='utf-8'))
        for field,name in(('prepared_input_sha256','input.npz'),('prepared_observations_sha256','observations.npz')):
            if identity[field]!=digest(prepared/name):raise ValueError('Cached J input changed')
        if identity['objective_policy_sha256']!=digest(parent/'objective_policy.json'):
            raise ValueError('Cached J uses a different actual objective policy')
        cached_weights=load(cache_stage/'objective_block_weights.npz');current_weights=load(parent/'objective_block_weights.npz')
        if set(cached_weights)!=set(current_weights):raise ValueError('Cached weighted blocks changed')
        for name in cached_weights:np.testing.assert_array_equal(cached_weights[name],current_weights[name])
        cached_associations=load(cache_stage/'source_associations.npz');current_associations=load(parent/'source_associations.npz')
        if set(cached_associations)!=set(current_associations):raise ValueError('Cached source association layout changed')
        for name in cached_associations:np.testing.assert_array_equal(cached_associations[name],current_associations[name])
        matrix=csr_matrix((np.load(cache/'data.npy',mmap_mode='r'),np.load(cache/'indices.npy',mmap_mode='r'),np.load(cache/'indptr.npy',mmap_mode='r')),shape=identity['shape'],copy=False)
        if matrix.shape!=(len(first['residual']),p.size):raise ValueError('Cached objective row/control layout changed')
        controls=np.arange(p.body.state.m);protected=np.empty(0,int);source_pts=[]
        if protect_source_controls:
            # Preserve clear shoe evidence DURING THIS initialization proposal.
            # The later joint solve has all these controls free. This avoids a
            # single shoe-region conflict throttling unrelated global P/G.
            source_pts=sorted({int(row['pts'])for row in first['action_metrics'].get('visible_runner_contours',[])})
            fi=np.searchsorted(ref.p['pts'],source_pts)
            if len(fi):
                np.testing.assert_array_equal(ref.p['pts'][fi],source_pts)
                tau=np.abs(camera.readout_times(fi));node_times=ref.times[fi,None]+tau[:,None]*np.array([-.5,.5])
                protected=np.unique(p.body.state.basis(node_times.ravel()).indices)
                controls=np.setdiff1d(controls,protected)
        if not len(controls):raise ValueError('No free global root controls in this source-protected proposal')
        ids=17+(controls[:,None]*69+np.arange(3)[None]).ravel();J=matrix[:,ids].tocsr();H=(J.T@J).tocsc();regularizer=1e-3*np.maximum(H.diagonal(),1e-12);direction=spsolve(H+diags(regularizer),-J.T@first['residual'])
        if not np.isfinite(direction).all():raise ArithmeticError('Nonfinite coupled root proposal')
        trials=[];selected=x.copy();chosen=first;alpha_used=0.
        for alpha in(1.,.5,.25):
            if time.perf_counter()-started>wall_seconds:break
            trial=x.copy();trial[ids]+=alpha*direction;value=p.evaluate(trial,False);source_fail=[]
            for row in value['action_metrics'].get('visible_runner_contours',[]):
                before,radius=baseline['source'][row['id']];after=float(np.percentile(row['outside_visible_band_px'],95))
                if after>before+.25*radius:source_fail.append(row['id'])
            torso=body_groups(p,value)['torso'];old_torso=baseline['torso']['pixel_p95'];torso_ok=torso['pixel_p95']<=old_torso+max(2.,.1*old_torso)
            P=ref.at(ref.times)['P']+p.body.state.increments(trial[17:],ref.times)[:,:3];jump=float(np.max(np.linalg.norm(np.diff(P,axis=0),axis=1)))*ref.binding.ratio
            cost=float(value['residual']@value['residual']/2);image_ok=value['image_cost']<=first['image_cost']*1.05
            accepted=cost<float(first['residual']@first['residual']/2)and not source_fail and torso_ok and image_ok and jump<before_jump
            row=dict(alpha=alpha,cost=cost,image_cost=value['image_cost'],source_rejected_ids=source_fail,torso_pixel_p95=torso['pixel_p95'],max_adjacent_P_step_m=jump,accepted=bool(accepted));trials.append(row);log.write(json.dumps(dict(event='root_initialization_trial',**row))+'\n')
            if accepted:selected=trial;chosen=value;alpha_used=alpha;break
        p.binding.x=selected[:17].copy();p.cache=p.cache_x=None;chosen=p.evaluate(selected,False)
        parent_summary=json.loads((parent/'run_summary.json').read_text(encoding='utf-8'));parent_steps=parent_summary.get('retained_step',parent_summary['accepted_steps']);count=int(alpha_used>0)
        p.checkpoint_extra={k:v for k,v in saved.items()if k.startswith('source_association_')or k in('source_evidence_sha256','programme_context_json','prepared_input_sha256','prepared_observations_sha256','objective_phase','objective_policy_sha256','objective_block_weights_sha256')};p.checkpoint_extra['parent_accepted_steps']=np.array(parent_steps)
        for name in('objective_policy.json','objective_block_weights.npz','primary_quality_baseline.json','source_associations.npz'):
            if(parent/name).exists():shutil.copyfile(parent/name,output/name)
        np.savez_compressed(output/'solve/candidate.npz',**p.checkpoint_fields(selected));write_json(output/'solve/candidate_selection.json',dict(retained_step=count,attempt_step=count,available=True,quality_failed=False))
        candidate=absolute_reference(ref,p.body.state.current_curve(selected[17:]),root=prepared);candidate.relative_skate_curve=p.body.state.shoe_current_curve(selected[17:]);candidate.binding=copy.copy(ref.binding);candidate.binding.reference=candidate
        z=export_candidate(prepared,output,candidate,camera,info,doc.get('supports',[]));C,_=mass.evaluate(candidate.at(candidate.times),np.zeros((len(candidate.times),69)))
        z.update(com_world_mid=mass.com(C),P_velocity_world_mps=candidate.root_derivative(candidate.times,1)*ref.binding.ratio,source_shot_ids=ref.p['shot_ids'],position_uncertainty_status=np.full(len(ref.times),'conditional scale/depth; development candidate'))
        np.savez_compressed(output/'fitted/result.npz',**z)
        summary=dict(parameter=str(output/'solve/candidate.npz'),result=str(output/'fitted/result.npz'),source_frames=len(ref.times),whole_seconds=sec[[0,-1]].tolist(),accepted_steps=parent_steps+count,additional_accepted_steps=count,retained_step=parent_steps+count,stop='bounded_coupled_root_initialization',wall_seconds=time.perf_counter()-started,actual_objective_unchanged=True,proposal_J_state=str(cache_stage),proposal_J_may_be_stale=True,all_trials_recomputed_current_state=True,before_max_P_step_m=before_jump,trials=trials,initialization_protection=dict(enabled=protect_source_controls,source_pts=source_pts,temporary_protected_root_controls=protected.tolist(),free_root_controls=controls.tolist(),later_joint_controls_remain_free=True,no_extra_objective_or_per_shot_offset=True),engineering_ready=True,trajectory_candidate=True,external_validation_status='FS world validation failed; conditional scale ambiguity',source_visual_review='pending',accuracy_accepted=False)
        write_json(output/'run_summary.json',summary);write_json(output/'final_metrics.json',dict(groups=body_groups(p,chosen),weighted_parts={b.name:float(b.residual@b.residual/2)for b in chosen['weighted_blocks']}));print(json.dumps(summary),flush=True)
    execution.close();return summary

def main():
    ap=argparse.ArgumentParser()
    for name in('project','prepared','parent','output','cache-stage'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--protect-source-controls',action='store_true');ap.add_argument('--wall-seconds',type=float,default=240.)
    a=ap.parse_args();run(a.project,a.prepared,a.parent,a.output,a.cache_stage,a.wall_seconds,protect_source_controls=a.protect_source_controls)
if __name__=='__main__':main()
