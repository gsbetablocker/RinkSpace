"""Issue an explicit native effective-input manifest from original SHA-bound bytes.

This is input packaging, not an old qualification cache or a quality certificate.
Original source files are never edited, and historical code roles retain their
original containing JSON reference and exact field address.
"""
from pathlib import Path
import argparse, hashlib, json, time, importlib.util
import numpy as np
from .effective_input_audit_v1 import CheckedBytes, resolve_packet, digest, references, require, strict_json

METADATA_LINKS={'parent_packet','dense_parent_packet','parent_qualification','source_board_qualification',
 'inherited_native_qualification','inherited_board_source_qualification','review','revision','recovery',
 'recipe','source_qualification','current_source_guard','board_material_source_qualification',
 'source_qualification_binding','source_review_evidence','qualification','dense_source_qualification'}
CODE_PURPOSES={'issuer','reader','source_profile_reader','native_frame_extractor',
 'actual_review_consumer','consumer_module_ref','issuer_module_ref','reader_module_refs',
 'recursive_parent_guard_module_ref','actual_input_dependencies','actual_dependencies',
 'source_validator_module_ref'}

def ref(path):
    p=Path(path); raw=p.read_bytes()
    return {'file':str(p.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'byte_count':len(raw)}

def pointer_refs(value, pointer=''):
    if isinstance(value,dict):
        if 'file' in value and 'sha256' in value: yield pointer,value
        for k,v in value.items():
            yield from pointer_refs(v,pointer+'/'+k.replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        for i,v in enumerate(value): yield from pointer_refs(v,pointer+'/'+str(i))

def prepare(cfg_path,head_path,model_path,out_path,root,runtime_payload_map=None):
    cfg=strict_json(Path(cfg_path).read_bytes()); store=CheckedBytes()
    head=ref(head_path); model=ref(model_path); resolved,ancestry=resolve_packet(head,store)
    selection_ref=cfg['representative_source_selection_v1']; selection=store.json(selection_ref,Path(cfg_path).parent)
    require(selection['schema']=='rinkfit.per_frame_representative_source_selection.v2','Effective fitting requires v2 training domain')
    eligibility=selection['training_eligibility']; provenance=selection['eligibility_provenance']
    require(selection['source_packet']['sha256']==head['sha256'],'Effective selection packet mismatch')
    roles=provenance['frame_roles']; verdicts=provenance['source_identity_verdicts']
    admission,_=store.array(resolved['arrays']['admission'],Path(head_path).parent)
    fit_role,_=store.array(resolved['arrays']['fit_role'],Path(head_path).parent)
    physical=((admission==1)&(fit_role==1)).astype(np.uint8)
    require(np.array_equal(physical,admission),'Composite physical mask needs an explicit typed output; no silent admission-only mask')
    physical_ref=resolved['arrays']['admission']
    for r in [model,selection_ref,roles,verdicts,eligibility['mask'],physical_ref]:
        store.read(r,Path(cfg_path).parent)
    native=Path(root)/'calibration/rinkfit/cpp'
    files=[native/'src/joint_packet.cpp',native/'include/rinkfit/joint_packet.hpp',
      *[native/'include/rinkfit'/n for n in ['joint_effective_input_v1.hpp',
       'joint_effective_source_guard_v1.hpp','joint_dense_source_guard_v1.hpp',
       'joint_printed_board_source_guard_v1.hpp','joint_current_board_source_guard_v1.hpp',
       'joint_ice_shape_source_guard_v1.hpp','joint_source_guard.hpp',
       'joint_representative_eligibility_v2.hpp','joint_dense_capture_weights_v1.hpp',
       'joint_engineering_budget_v2.hpp']]]
    build_spec=importlib.util.spec_from_file_location('rinkfit_effective_native_build',Path(root)/'calibration/rinkfit/tools/build_joint_native.py')
    build_module=importlib.util.module_from_spec(build_spec);build_spec.loader.exec_module(build_module)
    compiled_abi,compiled_sources=build_module.source_validation_identity(native)
    validators=[ref(p) for p in sorted(compiled_sources)]; validator_sha=digest(validators)
    q={'schema':'rinkfit.effective_source_qualification.v1','source_packet_sha256':head['sha256'],
       'resolved_metadata_sha256':digest(resolved),'validator_code_sha256':validator_sha,
       'original_qualification':cfg['dense_source_qualification'],'selection':selection_ref,
       'frame_roles':roles,'source_identity_verdicts':verdicts,
       'physical_mask':physical_ref,'training_mask':eligibility['mask']}
    # Follow actual referenced source JSON, once by content+address. No data is
    # classified as historical from its extension. Only an original code-purpose
    # field can produce a historical issuance entry.
    queue=[head,cfg['dense_source_qualification']]; queue.extend({k:v for k,v in x.items() if k in {'file','sha256','byte_count'}} for x in ancestry)
    seen=set(); historical={}; generations={}; scanned=0; unavailable=[]
    while queue:
        current=queue.pop(); p=store.path(current,Path(head_path).parent); key=(str(p.resolve()).casefold(),current['sha256'])
        if key in seen:continue
        seen.add(key)
        try: data=store.read(current,p.parent); doc=strict_json(data)
        except (OSError,ValueError,UnicodeError) as error:
            # Unavailable historical references are an audit result; runtime
            # required inputs remain enforced by native numerical consumers.
            unavailable.append({'reference':current,'error':str(error)});continue
        scanned+=1
        if isinstance(doc,dict) and 'qualification' in doc.get('schema',''):
            for field in ('source_packet','parent_packet'):
                r=doc.get(field)
                if isinstance(r,dict) and 'file' in r and 'sha256' in r:
                    rp=store.path(r,p.parent);generations.setdefault((str(rp.resolve()).casefold(),r['sha256']),
                        {'container':{k:current[k] for k in ('file','sha256','byte_count') if k in current},'json_pointer':'/'+field,'reference':r})
        for pointer,r in pointer_refs(doc):
            rp=store.path(r,p.parent); tokens=set(pointer.split('/'))
            if tokens & CODE_PURPOSES and rp.suffix=='.py':
                hkey=(str(rp.resolve()).casefold(),r['sha256'])
                historical.setdefault(hkey,{'role':'historical_issuance_code','used_as_runtime_data':False,
                 'container':{k:current[k] for k in ('file','sha256','byte_count') if k in current},
                 'json_pointer':pointer,'reference':r,
                 'reason':'Original Python issuance or source-reader implementation; native executes current mathematical validation, not this historical Python file.'})
            elif rp.suffix.lower()=='.json' and tokens & METADATA_LINKS:
                queue.append(r)
    manifest={'schema':'rinkfit.effective_joint_input.v1','abi':'rinkfit.effective_input_abi.v1',
      'canonical_encoding':'rinkfit.tagged_json_sha256.v1','source_packet':head,'source_model':model,
      'raw_rows':resolved['row_count'],'ancestry':[{k:x[k] for k in ('file','sha256','byte_count')} for x in ancestry],
      'arrays':resolved['arrays'],'resolved_metadata_sha256':digest(resolved),
      'validator_sources':validators,'validator_code_sha256':validator_sha,'compiled_source_validator_sha256':compiled_abi,'qualification':q,
      'engineering_support_budget_v2':cfg['engineering_support_budget_v2'],
      'historical_audit_only':list(historical.values()),'qualification_generations':list(generations.values()),'quality_accepted':False,
      'historical_issuance_replay_complete':False,'source_image_identity_proven':False}
    if runtime_payload_map:
        payload=strict_json(Path(runtime_payload_map).read_bytes())
        require(payload['schema']=='rinkfit.exact_runtime_payload_map.v1' and payload['source_packet']['sha256']==head['sha256'],'Runtime payload map binds another source generation')
        required={(str(store.path(r,Path(head_path).parent).resolve()).casefold(),r['sha256']) for _,r in references(resolved) if 'dtype' in r}
        for entry in payload['runtime_payload']:
            original=entry['original'];actual=entry['actual']
            require(entry['schema']=='rinkfit.exact_runtime_payload_file.v1' and original['sha256']==actual['sha256'] and
                    (str(Path(original['file']).resolve()).casefold(),original['sha256']) in required,'Runtime payload is not an exact original required backing')
            store.read(actual,Path(runtime_payload_map).parent)
        manifest['runtime_payload']=payload['runtime_payload']
    out=Path(out_path);require(not out.exists(),'Refuse to overwrite an existing effective manifest');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf8')
    report={'schema':'rinkfit.effective_input_preparation.v1','manifest':ref(out),
      'original_head_sha256':head['sha256'],'raw_rows':resolved['row_count'],
      'original_ancestry_layers':len(ancestry),'json_containers_scanned':scanned,
      'historical_code_references':len(historical),'qualification_packet_scopes':len(generations),'historical_index_scope':'original_packet_and_qualification_metadata_with_exact_code_purpose_fields','unavailable_referenced_json':unavailable,
      'native_qualification_executed':False,'native_input_enabled':False,'quality_accepted':False}
    out.with_name(out.stem+'_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
    return report


def actual_source_qualification(cfg):
    """Return the original proof ref without disguising the selected input ABI."""
    if 'effective_input_v1' not in cfg:
        return cfg['dense_source_qualification']
    require(not {'dense_source_qualification','engineering_dense_source_weights_v1'} & set(cfg),
            'Effective and legacy qualification/weight modes are mutually exclusive')
    m=strict_json(Path(cfg['effective_input_v1']['file']).read_bytes())
    require(hashlib.sha256(Path(cfg['effective_input_v1']['file']).read_bytes()).hexdigest()==cfg['effective_input_v1']['sha256'], 'Effective manifest SHA mismatch')
    require(m['schema']=='rinkfit.effective_joint_input.v1' and m['qualification']['schema']=='rinkfit.effective_source_qualification.v1','Effective manifest schema')
    return m['qualification']['original_qualification']


def validate_effective_configuration(cfg,packet_ref,model_ref,recipe):
    """Bind exact effective data to a completed proof or a new native source run."""
    original_qualification=actual_source_qualification(cfg)
    store=CheckedBytes();m=store.json(cfg['effective_input_v1'],Path(cfg['effective_input_v1']['file']).parent)
    require(m['abi']=='rinkfit.effective_input_abi.v1' and m['canonical_encoding']=='rinkfit.tagged_json_sha256.v1','Unsupported effective ABI')
    def same(a,b):return Path(a['file']).resolve()==Path(b['file']).resolve() and a['sha256']==b['sha256']
    if 'complete_stage_primal_v1' in cfg:
        from .complete_stage_primal import validate
        complete = validate(cfg,model_ref,packet_ref)
        require(same(complete.get('effective_source_revalidation',{}).get('to_effective_input',complete['effective_input_v1']),cfg['effective_input_v1']), 'Complete primal omitted explicit target effective authority')
        require(same(m['source_packet'],packet_ref), 'Effective original packet changed')
        store.read(m['source_model'],Path(cfg['effective_input_v1']['file']).parent)
    else:
        require(same(m['source_packet'],packet_ref) and same(m['source_model'],model_ref),'Effective original packet/model changed')
    require(m['qualification']['source_packet_sha256']==packet_ref['sha256'] and m['qualification']['resolved_metadata_sha256']==m['resolved_metadata_sha256'],'Effective qualification/payload identity')
    values={r['file']:r['sha256'] for r in m['validator_sources']}
    require(len(values)==len(m['validator_sources']) and hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()==m['compiled_source_validator_sha256'],'Effective original compiled validator collection mismatch')
    require(digest(m['validator_sources'])==m['validator_code_sha256']==m['qualification']['validator_code_sha256'],'Effective validator source collection digest')
    prepared = None
    if 'prepared_effective_input_v1' in cfg:
        from .prepared_effective_input_v1 import validate_configuration
        prepared = validate_configuration(cfg,m)
    else:
        require(m['compiled_source_validator_sha256']==recipe['source_validation_abi_hash'],'Effective manifest/native compiled validator ABI mismatch')
        for r in m['validator_sources']:store.read(r,Path(cfg['effective_input_v1']['file']).parent)
    for k in ('original_qualification','selection','frame_roles','source_identity_verdicts','physical_mask','training_mask'):
        store.read(m['qualification'][k],Path(cfg['effective_input_v1']['file']).parent)
    require(same(m['qualification']['selection'],cfg['representative_source_selection_v1']) and same(m['qualification']['source_identity_verdicts'],cfg['source_identity_verdicts_v1']),'Effective training authority differs from actual configuration')
    require(m['engineering_support_budget_v2']==cfg['engineering_support_budget_v2'],'Effective common exposure budget changed')
    for e in m.get('runtime_payload',[]):
        require(e['schema']=='rinkfit.exact_runtime_payload_file.v1' and e['original']['sha256']==e['actual']['sha256'],'Effective payload may not replace required original data bytes')
        store.read(e['actual'],Path(cfg['effective_input_v1']['file']).parent)
    return dict(schema='rinkfit.effective_input_launch_preflight.v1',passed=True,binding=cfg['effective_input_v1'],source_packet=packet_ref,
                source_model=m['source_model'],mutable_complete_primal_model=model_ref if 'complete_stage_primal_v1' in cfg else None,
                raw_rows=m['raw_rows'],compiled_source_validator_sha256=m['compiled_source_validator_sha256'],
                original_qualification=original_qualification,legacy_weight_arrays_consumed=False,
                prepared_binding=cfg.get('prepared_effective_input_v1'),
                prior_completed_native_source_proof=prepared['cold_native_proof'] if prepared else None,
                solver_compiled_source_validator_sha256=recipe['source_validation_abi_hash'],
                native_mathematical_reexecution_required=prepared is None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--packet',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(prepare(a.config,a.packet,a.model,a.out,a.root),ensure_ascii=False))
if __name__=='__main__':main()
