"""Prepare an explicit observed -> true-RS transition or exact true-RS primal restart.

No solver is launched. Both modes retain original producer model/packet inputs;
new native state is imported only after their qualification replay.
"""
from copy import deepcopy
from pathlib import Path
import argparse,json
from .observed_primal_restart import prepare as observed_prepare, read, checked, reference, write_new, semantic_config, refresh_review_binding
TRUE_RS='rinkfit.sourceUV_trueRS_finite_capture.v3'
IMPORTS=('observed_primal_restart_v1','observed_primal_warm_start_v1','capture_primal_transition_v2','capture_primal_restart_v2')


def comparable(cfg):
    cfg=semantic_config(cfg)
    for key in IMPORTS:cfg.pop(key,None)
    return cfg


def validate(cfg,model,packet):
    keys=[k for k in IMPORTS if k in cfg]
    if len(keys)!=1 or keys[0] not in IMPORTS[2:] or cfg.get('source_uv_capture_objective')!=TRUE_RS:
        raise ValueError('One explicit true-RS import required')
    doc=read(checked(cfg[keys[0]]))
    if keys[0]=='capture_primal_transition_v2':
        if doc['schema']!='rinkfit.observed_to_trueRS_primal_transition.v2':raise ValueError('Transition schema differs')
        original=read(checked(doc['observed_import_configuration']))
        from .observed_primal_restart import validate_restart_manifest
        validate_restart_manifest(original,model,packet)
        old=comparable(original);old['source_uv_capture_objective']=TRUE_RS
    else:
        if doc['schema']!='rinkfit.trueRS_complete_primal_restart.v2':raise ValueError('Restart schema differs')
        original=read(checked(doc['original_configuration']));old=comparable(original)
        if doc['input_model']!=model or doc['source_packet']!=packet:raise ValueError('Producer inputs changed')
        for name in ('checkpoint','model','state_binding','nuisance_manifest'):checked(doc[name])
        for value in doc['files'].values():checked(value)
        binding=read(doc['state_binding']['file'])
        if binding['objective_id']!=TRUE_RS or binding['state_token']!=doc['expected_state_token']:raise ValueError('State identity mismatch')
    if old!=comparable(cfg):raise ValueError('True-RS import changed source, stage, weights or numerical domain')
    return doc


def prepare(spec_path,candidate_dir,native,prepared_dir,output_dir,*,transition=False,verify_only=False,max_inner_iterations=None):
    spec=read(spec_path);prepared=Path(prepared_dir).resolve();candidate=Path(candidate_dir).resolve()
    if prepared.exists() or Path(output_dir).exists():raise FileExistsError('New preparation and output required')
    cfg=read(checked(spec['configuration']))
    if transition:
        observed_prepare(spec_path,candidate,native,prepared/'observed',prepared/'unused_observed_output',verify_only=verify_only)
        imported=read(prepared/'observed/spec.json');cfg=read(imported['configuration']['file'])
        doc=dict(schema='rinkfit.observed_to_trueRS_primal_transition.v2',observed_import_configuration=imported['configuration'],quality_accepted=False,optimizer_history_restored=False)
        for key in IMPORTS:cfg.pop(key,None)
        cfg['source_uv_capture_objective']=TRUE_RS
        cfg['capture_primal_transition_v2']=write_new(prepared/'transition.json',doc)
    else:
        if cfg.get('source_uv_capture_objective')!=TRUE_RS:raise ValueError('Producer is not true-RS v3')
        prepared.mkdir(parents=True)
        binding=read(candidate/'source_uv_capture_state_binding.json')
        doc=dict(schema='rinkfit.trueRS_complete_primal_restart.v2',original_configuration=spec['configuration'],input_model=spec['model'],source_packet=spec['packet'],
            checkpoint=reference(candidate/'engineering_checkpoint.json'),model=reference(candidate/'model.json'),state_binding=reference(candidate/'source_uv_capture_state_binding.json'),
            nuisance_manifest=reference(candidate/'explicit_nuisance/manifest.json'),expected_state_token=binding['state_token'],
            files={name:reference(candidate/'explicit_nuisance'/name)for name in ('station.f64','row.f64','alpha.f64','path.u32')},quality_accepted=False,optimizer_history_restored=False)
        for key in IMPORTS:cfg.pop(key,None)
        cfg['capture_primal_restart_v2']=write_new(prepared/'restart.json',doc)
    if max_inner_iterations is not None:
        if isinstance(max_inner_iterations,bool) or int(max_inner_iterations)!=max_inner_iterations or max_inner_iterations<1:raise ValueError('max_inner_iterations must be a positive integer')
        cfg['max_inner_iterations']=int(max_inner_iterations)
    refresh_review_binding(cfg, prepared)
    cfg['source_uv_capture_construct_problem_only']=bool(verify_only)
    validate(cfg,spec['model'],spec['packet'])
    result=deepcopy(spec);result.update(native=reference(native),configuration=write_new(prepared/'config.json',cfg),output=str(Path(output_dir).resolve()))
    ref=write_new(prepared/'spec.json',result)
    return dict(prepared_spec=ref,solver_launched=False,same_problem_restart=not transition,true_RS_objective=True,quality_accepted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--candidate',required=True);p.add_argument('--native',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);p.add_argument('--transition',action='store_true');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--max-inner-iterations',type=int)
    a=p.parse_args();print(json.dumps(prepare(a.spec,a.candidate,a.native,a.prepared,a.output,transition=a.transition,verify_only=a.verify_only,max_inner_iterations=a.max_inner_iterations)))

if __name__=='__main__':main()
