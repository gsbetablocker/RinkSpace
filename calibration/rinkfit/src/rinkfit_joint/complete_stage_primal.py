"""Prepare the exact complete current primal for continuation or observed -> true RS.

Immutable effective source authority stays fixed. A separate SHA-bound candidate
carries the complete target layout, full raw state, cells, motion and PHR duals.
No Solve is launched, and preparation is not a native state/quality certificate.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
from .observed_primal_restart import checked, read, reference, write_new, refresh_review_binding

OBSERVED = 'rinkfit.sourceUV_observed_row_finite_initialization.v1'
TRUE_RS = 'rinkfit.sourceUV_trueRS_finite_capture.v3'
IMPORTS = ('representative_continuation_v1','representative_objective_transition_v1',
    'representative_source_append_primal_v1','observed_primal_restart_v1',
    'observed_primal_warm_start_v1','capture_primal_transition_v2',
    'capture_primal_restart_v2','complete_stage_primal_v1',
    'engineering_global_phr_mode_transition_v1')
COLUMNS = ('station.f64','row.f64','alpha.f64','path.u32')


def checkpoint_header(path):
    with Path(path).open(encoding='utf-8') as stream:
        magic=stream.readline().strip();fp=shlex.split(stream.readline())
        outer=stream.readline().split();lam=stream.readline().split();nu=stream.readline().split()
        present=stream.readline().strip();token=shlex.split(stream.readline());identity=shlex.split(stream.readline())
    if (magic!='RINKFIT_AL_CHECKPOINT_V1' or len(fp)!=1 or len(outer)!=7
            or float(outer[4])!=0 or lam!=['0'] or nu!=['0'] or present!='1'
            or len(token)!=1 or len(identity)!=6 or identity[-1] not in (OBSERVED,TRUE_RS)):
        raise ValueError('Complete current engineering primal required; formal AL cannot be discarded')
    return dict(runtime_fingerprint=fp[0],state_token=token[0],source_packet_sha256=identity[0],objective_id=identity[-1])


def global_state_record(path):
    # Keep the native canonical JSON bytes for the bound digest. Re-dumping
    # through Python may use a different exponent spelling for the same double.
    prefix='"engineering_global_phr_state_v1" '
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith(prefix):
                fields=shlex.split(line)
                if len(fields)!=2:raise ValueError('Invalid complete global PHR record')
                return json.loads(fields[1]),hashlib.sha256(fields[1].encode()).hexdigest()
    return None,None


def effective_invariants(manifest):
    value=deepcopy(manifest)
    for key in ('validator_sources','validator_code_sha256','compiled_source_validator_sha256'):value.pop(key,None)
    value['qualification'].pop('validator_code_sha256',None)
    return value


def validate(cfg, model_ref, packet_ref):
    ref=cfg['complete_stage_primal_v1'];doc=read(checked(ref))
    if doc['schema']!='rinkfit.complete_stage_primal.v1' or doc.get('complete_checkpoint')is not True:
        raise ValueError('Complete stage primal manifest required')
    if doc['model']!=model_ref or doc['source_packet']!=packet_ref:
        raise ValueError('CLI complete candidate/source address differs')
    for key in ('model','checkpoint','original_configuration','explicit_nuisance_manifest','representative_source_selection_v1'):
        checked(doc[key])
    if doc['representative_source_selection_v1']!=cfg['representative_source_selection_v1']:
        raise ValueError('Stage import changed actual thin selection')
    original=read(checked(doc['original_configuration']));effective=read(checked(cfg['effective_input_v1']))
    if (Path(effective['source_packet']['file']).resolve()!=Path(packet_ref['file']).resolve()
            or effective['source_packet']['sha256']!=packet_ref['sha256']):raise ValueError('Original effective source differs')
    checked(effective['source_model'])
    if doc['effective_input_v1']!=original['effective_input_v1']:raise ValueError('Original effective authority differs from producer')
    if doc['effective_input_v1']!=cfg['effective_input_v1']:
        from .effective_input_audit_v1 import digest
        policy=doc.get('effective_source_revalidation',{});old=read(checked(doc['effective_input_v1']))
        invariant=effective_invariants(old)
        prepared_ref=cfg.get('prepared_effective_input_v1')
        validation_bound=(policy.get('new_native_qualification_reexecution_required') is True and prepared_ref is None
            or policy.get('new_native_qualification_reexecution_required') is False and prepared_ref is not None
            and policy.get('prepared_effective_input_v1')==prepared_ref)
        if (policy.get('schema')!='rinkfit.complete_primal_effective_source_revalidation.v1'
                or policy.get('from_effective_input')!=doc['effective_input_v1']
                or policy.get('to_effective_input')!=cfg['effective_input_v1']
                or policy.get('from_state_token')!=doc['state_token']
                or not validation_bound
                or not policy.get('reason') or invariant!=effective_invariants(effective)
                or policy.get('invariant_descriptor_sha256')!=digest(invariant)):
            raise ValueError('Effective ABI migration changed original mathematical source authority')
    elif 'effective_source_revalidation' in doc:raise ValueError('Unchanged ABI cannot redeclare source migration')
    if original['source_uv_capture_objective']!=cfg['source_uv_capture_objective']:
        t=doc.get('stage_transition',{})
        if (original['source_uv_capture_objective']!=OBSERVED or cfg['source_uv_capture_objective']!=TRUE_RS
                or t.get('schema')!='rinkfit.complete_primal_stage_transition.v1'
                or t.get('kind')!='observed_to_true_RS' or t.get('from_state_token')!=doc['state_token']
                or t.get('target_objective_id')!=TRUE_RS or not t.get('reason')):
            raise ValueError('Stage change requires bound observed-to-trueRS transition')
    elif 'stage_transition' in doc:raise ValueError('Same-target continuation cannot redeclare stage transition')
    h=checkpoint_header(checked(doc['checkpoint']))
    if (h['state_token']!=doc['state_token'] or h['source_packet_sha256']!=packet_ref['sha256']
            or h['objective_id']!=original['source_uv_capture_objective']):
        raise ValueError('Original complete native token/stage/source mismatch')
    raw_path=checked(doc['explicit_nuisance_manifest']);raw=read(raw_path)
    if raw['schema']!='rinkfit.joint_explicit_PHR_nuisance_state.v1' or raw['packet_sha256']!=packet_ref['sha256']:
        raise ValueError('Complete raw manifest source/schema differs')
    n=int(raw['raw_rows'])
    for name in COLUMNS:
        path=raw_path.parent/name;spec=raw['files'][name]
        if int(spec['size_bytes'])!=n*(4 if name=='path.u32' else 8) or path.stat().st_size!=int(spec['size_bytes']):
            raise ValueError('Complete raw denominator/bytes differ: '+name)
        checked(dict(file=str(path),sha256=spec['sha256']))
    return doc


def prepare(spec_path,candidate_dir,native,prepared_dir,output_dir,*,transition=False,
            verify_only=False,max_inner_iterations=None,effective_input=None,prepared_effective_input=None):
    spec=read(spec_path);candidate=Path(candidate_dir).resolve();prepared=Path(prepared_dir).resolve()
    output=Path(output_dir).resolve()
    if prepared.exists() or output.exists():raise FileExistsError('New preparation/output required')
    cfg=read(checked(spec['configuration']));objective=cfg.get('source_uv_capture_objective')
    if objective not in (OBSERVED,TRUE_RS) or 'effective_input_v1' not in cfg:
        raise ValueError('Qualified current effective observed/trueRS producer required')
    if transition and objective!=OBSERVED:raise ValueError('Transition source must be observed-row')
    cp_ref=reference(candidate/'engineering_checkpoint.json');h=checkpoint_header(checked(cp_ref))
    if h['objective_id']!=objective or h['source_packet_sha256']!=spec['packet']['sha256']:
        raise ValueError('Candidate source/stage differs from producer')
    doc=dict(schema='rinkfit.complete_stage_primal.v1',complete_checkpoint=True,
        original_configuration=spec['configuration'],source_packet=spec['packet'],
        effective_input_v1=cfg['effective_input_v1'],model=reference(candidate/'model.json'),
        checkpoint=cp_ref,explicit_nuisance_manifest=reference(candidate/'explicit_nuisance/manifest.json'),
        representative_source_selection_v1=cfg['representative_source_selection_v1'],state_token=h['state_token'],
        optimizer_history_restored=False,quality_accepted=False)
    for key in IMPORTS:cfg.pop(key,None)
    cfg['reinitialize_native_auxiliaries']=False
    cfg['source_uv_capture_reseed_new_sources']=False
    if effective_input is not None:
        cfg.pop('prepared_effective_input_v1',None)
    if prepared_effective_input is not None:
        cfg['prepared_effective_input_v1']=reference(prepared_effective_input)
    if effective_input is not None:
        from .effective_input_audit_v1 import digest
        new_ref=reference(effective_input);old=read(checked(doc['effective_input_v1']));new=read(checked(new_ref))
        invariant=effective_invariants(old)
        if invariant!=effective_invariants(new) or old['compiled_source_validator_sha256']==new['compiled_source_validator_sha256']:
            raise ValueError('Source revalidation must preserve all source/qualification/mathematical input authority and actually change ABI')
        doc['effective_source_revalidation']=dict(schema='rinkfit.complete_primal_effective_source_revalidation.v1',
            from_effective_input=doc['effective_input_v1'],to_effective_input=new_ref,from_state_token=h['state_token'],
            invariant_descriptor_sha256=digest(invariant),new_native_qualification_reexecution_required='prepared_effective_input_v1' not in cfg,
            reason='Explicit target source validator proof with identical original source, qualification, arrays, source clock/model, selection and W budget authority')
        if 'prepared_effective_input_v1' in cfg:
            doc['effective_source_revalidation']['prepared_effective_input_v1']=cfg['prepared_effective_input_v1']
        cfg['effective_input_v1']=new_ref
    if 'prepared_effective_input_v1' in cfg:
        from .prepared_effective_input_v1 import validate_configuration
        validate_configuration(cfg,read(checked(cfg['effective_input_v1'])))
    if transition:
        doc['stage_transition']=dict(schema='rinkfit.complete_primal_stage_transition.v1',
            kind='observed_to_true_RS',from_state_token=h['state_token'],target_objective_id=TRUE_RS,
            reason='Explicit complete observed primal transfer; fresh all-selected-source true RS and formal Query revalidation remain required')
        cfg['source_uv_capture_objective']=TRUE_RS
        global_state,digest=global_state_record(checked(cp_ref))
        if global_state is not None:
            cfg['engineering_global_phr_mode_transition_v1']=dict(
                schema='rinkfit.engineering_global_phr_mode_transition.v1',from_state_token=h['state_token'],
                from_global_state_sha256=digest,target_objective_id=TRUE_RS,
                action='preserve_same_global_constraints',
                reason='Identical ordered original global physical factors, policy, rho and duals; new RS row equations are revalidated separately')
    if max_inner_iterations is not None:
        if isinstance(max_inner_iterations,bool) or int(max_inner_iterations)!=max_inner_iterations or max_inner_iterations<1:
            raise ValueError('Positive integer inner iterations required')
        cfg['max_inner_iterations']=int(max_inner_iterations)
    prepared.mkdir(parents=True)
    cfg['complete_stage_primal_v1']=write_new(prepared/'primal.json',doc)
    cfg['source_uv_capture_construct_problem_only']=False
    cfg['source_uv_capture_preflight_only']=bool(verify_only)
    refresh_review_binding(cfg,prepared)
    validate(cfg,doc['model'],spec['packet'])
    result=deepcopy(spec);result.update(model=doc['model'],native=reference(native),
        configuration=write_new(prepared/'config.json',cfg),output=str(output))
    ref=write_new(prepared/'spec.json',result)
    return dict(prepared_spec=ref,solver_launched=False,source_objective_id=objective,
        target_objective_id=cfg['source_uv_capture_objective'],same_objective_continuation=not transition and effective_input is None,source_validation_ABI_changed=effective_input is not None,
        immutable_effective_source_authority_preserved=True,complete_native_restore_still_required=True,quality_accepted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','candidate','native','prepared','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--transition',action='store_true');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--effective-input');p.add_argument('--prepared-effective-input');p.add_argument('--max-inner-iterations',type=int);a=p.parse_args()
    print(json.dumps(prepare(a.spec,a.candidate,a.native,a.prepared,a.output,
        transition=a.transition,verify_only=a.verify_only,max_inner_iterations=a.max_inner_iterations,effective_input=a.effective_input,prepared_effective_input=a.prepared_effective_input)))
if __name__=='__main__':main()
