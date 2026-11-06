"""Finalize an existing accepted whole-S05 checkpoint after a trial exception.

No optimizer, old snapshot migration, new CASE identity or network inference.
The existing input/evidence hashes and archived seed are required to match.
"""
from pathlib import Path
import argparse,copy,json,shutil,time
import numpy as np
from .source_guided_refit import prepare,set_objective_policy,native_metrics,source_quality
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .exmr_integration import CASE_FIELD,read_case_identity,PROGRAMME,save_packet
from .absolute_motion_state import absolute_reference
from .skate_physics import fk_state
from .skate_template import profile_height


def recover(project,parent_stage,parent_parameter,stage,parameter):
    from rinkspace.path_policy import guard_project_write
    project,parent_stage,parent_parameter,stage,parameter=map(Path,(project,parent_stage,parent_parameter,stage,parameter))
    parameter=guard_project_write(parameter,project_root=project)
    if parameter.exists()or(stage/'run_summary.json').exists()or(stage/'common/exmr_hand_nofoot/fitted/result.npz').exists():raise FileExistsError('Existing result is immutable')
    checkpoint=stage/'solve/last_accepted.npz';saved=load(checkpoint);selection=json.loads((stage/'solve/candidate_selection.json').read_text(encoding='utf-8'))
    document=json.loads((stage/'source_evidence.json').read_text(encoding='utf-8'));started=time.perf_counter()
    with (stage/'checkpoint_recovery.jsonl').open('x',encoding='utf-8',buffering=1)as log:
        p,zero,execution,info,mass,original=prepare(project,parent_stage,parent_parameter,stage,log,document,reuse=True)
        try:
            if read_case_identity(saved)!=read_case_identity(p.reference.fixed_checkpoint_identity):raise ValueError('Accepted checkpoint current CASE hash mismatch')
            for key in ('absolute_seed_position_controls','absolute_seed_rotation_controls','absolute_knots','relative_skate_seed_rotation_controls','relative_skate_knots'):
                expected=p.body.state.packet(np.zeros(p.body.size))[key]
                np.testing.assert_array_equal(saved[key],expected)
            p.objective_block_weights=load(stage/'objective_block_weights.npz');p.record_compatibility_association=True;p.body.compatibility_difference_step=1e-8
            p.al.multiplier=saved['multiplier'].copy();p.al.penalty=float(saved['penalty'])
            with execution.limits():
                initializer=load(str(saved['source_initializer_path']))
                p.evaluate(initializer['x'],False) # Restore the actual fixed inner association, not a new fit.
                x=saved['x'];value=p.evaluate(x,False);cost=float(value['residual']@value['residual']/2)
                prior_cost=selection['retained_metrics']['joint_cost']
                if not np.isclose(cost,prior_cost,rtol=1e-9,atol=1e-7):raise ValueError('Accepted objective replay mismatch '+str((cost,prior_cost)))
                write_json(stage/'final_metrics.json',native_metrics(p,value));write_json(stage/'final_source_quality.json',dict(after=source_quality(value)))
                with checkpoint.open('rb')as src,parameter.open('xb')as target:shutil.copyfileobj(src,target)
                final=absolute_reference(p.reference,p.body.state.current_curve(x[17:]),root=p.reference.root);final.relative_skate_curve=p.body.state.shoe_current_curve(x[17:])
                shell=copy.copy(p);shell.reference=final;packet=save_packet(shell,info,p.evidence,document,p.reference.root,status='joint_fit')
                state=fk_state(final,final.at(final.times),np.zeros((len(final.times),69)));C,R=mass.evaluate(final.at(final.times),np.zeros((len(final.times),69)))
                packet.update(com_world_mid=mass.com(C),absolute_parameter_path=np.array(str(parameter)),**p.body.state.packet(x[17:]))
                template=final.binding.packet();dimensions=template['parameters'];rear=-dimensions.runner_rear_extension_m;front=dimensions.sole_length_m-dimensions.runner_front_inset_m;xx=np.array([rear,(rear+front)/2,front]);stations=np.c_[xx,np.zeros(3),profile_height(xx,dimensions)]
                packet['blade_material_world_mid']=(state['O'][:,:,None]+np.einsum('nfij,kj->nfki',state['R'],stations))*final.binding.ratio
                packet['runner_edges_world_mid']=(state['O'][:,:,None,None,:]+np.einsum('nfij,ekj->nfeki',state['R'],template['edges']))*final.binding.ratio
                modes=p.action_factors.modes;packet.update(sequence_contact_modes=modes.modes,sequence_contact_status=modes.status,sequence_contact_mode_names=np.asarray(modes.record['mode_names']),sequence_contact_association_json=np.array(json.dumps(modes.record,sort_keys=True)))
                counts=np.zeros((len(final.times),2),np.int32);location=np.zeros_like(counts,bool);lookup={int(pt):i for i,pt in enumerate(final.p['pts'])}
                for row in document['full_source_skate_structures']['rows']:counts[lookup[row['pts']],row['foot']]+=1
                for row in document['source_support_evidence']:location[lookup[row['pts']],row['foot']]=True
                packet.update(source_complete_structure_observation_count=counts,source_support_location_in_optimization=location,runner_trajectory_evidence_status=np.where(counts>0,'direct_source_structure_fit','inferred_between_source_exposures'))
                np.savez_compressed(p.reference.root/'fitted/result.npz',**packet)
                summary=dict(phase='source_guided',parameter=str(parameter),result=str(p.reference.root/'fitted/result.npz'),accepted_steps=selection['retained_step'],retained_step=selection['retained_step'],source_frames=len(final.times),whole_seconds=[76.,95.],final_cost=cost,
                    initial_cost=json.loads((stage/'initial_registry.json').read_text(encoding='utf-8'))['cost'],stop=dict(reason='trial runner-join coordinate assertion; accepted checkpoint recovered without fitting',trial_exception_preserved=str(stage.parent/'source_guided_seeded_v2_console.log')),
                    checkpoint_sha256=digest(checkpoint),parameter_bytes_equal_checkpoint=parameter.read_bytes()==checkpoint.read_bytes(),current_case_identity_exact=True,no_new_solve=True,mode_initialization='original pre-seed plan replayed; next outer stage reassociates at current pose',recovery_seconds=time.perf_counter()-started,accuracy_accepted=False,user_accepted=False,not_default=True)
                write_json(stage/'run_summary.json',summary);print(json.dumps(dict(event='accepted_checkpoint_recovered',parameter=str(parameter),result=summary['result'],accepted=summary['accepted_steps'])),flush=True)
        finally:execution.close()


def main():
    ap=argparse.ArgumentParser()
    for name in ('project','parent-stage','parent-parameter','stage','parameter'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();recover(a.project,a.parent_stage,a.parent_parameter,a.stage,a.parameter)

if __name__=='__main__':main()
