"""Prepare managed current input from an immutable, already executed native proof.

The original proof's validator identity is retained. Solver implementation code is
not a new issuer of that proof, and no historical source mathematics is replayed.
"""
from pathlib import Path
import argparse, hashlib, json, time
from .effective_input_audit_v1 import CheckedBytes, resolve_packet, digest, strict_json, require, references
from .effective_input_prepare_v1 import ref, pointer_refs, METADATA_LINKS

SCHEMA='rinkfit.prepared_effective_input.v1'

def same(a,b):
    return Path(a['file']).absolute()==Path(b['file']).absolute() and a['sha256']==b['sha256']

class PreparedStore(CheckedBytes):
    def __init__(self, manifest, aliases=None):
        super().__init__(); self.payload={}
        for x in manifest.get('runtime_payload',[]):
            require(x['original']['sha256']==x['actual']['sha256'],'Prepared payload changed original bytes')
            self.payload[(str(Path(x['original']['file']).absolute()).casefold(),x['original']['sha256'])]=x['actual']
        if aliases:
            for x in aliases["aliases"]:
                self.payload[(str(Path(x["legacy_path"]).absolute()).casefold(),x["sha256"])]=dict(file=x["actual_file"],sha256=x["sha256"])
        self.json_bytes={}
    def actual(self,r,base):
        p=self.path(r,base).absolute();key=(str(p).casefold(),r['sha256'])
        return self.payload.get(key,dict(r,file=str(p)))
    def read(self,r,base):
        actual=self.actual(r,base);p=Path(actual['file']);key=(str(p.absolute()),r['sha256'])
        if key in self.json_bytes:return self.json_bytes[key]
        data=super().read(actual,p.parent)
        if p.suffix.lower()=='.json':self.json_bytes[key]=data
        # resolve_packet expects verification indexed by the original address.
        op=self.path(r,base)
        self.verified[(str(op),r['sha256'])]=dict(file=str(p.absolute()),sha256=r['sha256'],byte_count=len(data),location='filesystem')
        return data
    def verify(self,r,base):
        actual=self.actual(r,base);p=Path(actual['file']).absolute();key=(str(p),r['sha256'])
        if key not in self.verified:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            require(h.hexdigest()==r['sha256'],'Prepared source dependency SHA mismatch: '+str(p))
            require('byte_count' not in r or r['byte_count']==p.stat().st_size,'Prepared byte count mismatch')
            self.verified[key]=dict(file=str(p),sha256=r['sha256'],byte_count=p.stat().st_size,location='filesystem')
        return dict(file=str(p),sha256=r['sha256'],byte_count=self.verified[key]['byte_count'])

def validate_proof(m,report,effective_binding):
    require(report.get('passed') is True and report.get('full_native_numerical_payload_consumed') is True,
            'Prepared input needs an actually completed native full-input proof')
    require(same(report['binding'],effective_binding),'Cold result belongs to another effective manifest')
    for k in ('source_packet','raw_rows','resolved_metadata_sha256','validator_code_sha256','compiled_source_validator_sha256'):
        require(report[k]==m[k],'Cold proof identity differs: '+k)
    q=report['qualification']
    require(q.get('full_original_native_source_mathematics_reexecuted') is True and q.get('legacy_validation_cache_reused') is False,
            'Cold proof did not execute original source mathematics')
    require(q['source_packet']==m['source_packet'] and q['raw_rows']==m['raw_rows'] and report['typed_rows']==m['raw_rows'],
            'Cold proof omitted full current raw domain')
    require(report.get('quality_accepted') is False and report.get('source_image_identity_proven') is False,
            'Prepared source proof must not be promoted to image acceptance')

def prepare(effective_binding,cold_report_binding,out_dir,dependency_aliases=None):
    start=time.perf_counter();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    target=out/'prepared.json';require(not target.exists(),'Refuse to overwrite an existing prepared input')
    base=Path(effective_binding['file']).parent
    initial=CheckedBytes();m=initial.json(effective_binding,base);report=initial.json(cold_report_binding,Path(cold_report_binding['file']).parent)
    validate_proof(m,report,effective_binding);aliases=initial.json(dependency_aliases,Path(dependency_aliases["file"]).parent) if dependency_aliases else None;store=PreparedStore(m,aliases)
    resolved,ancestry=resolve_packet(m['source_packet'],store)
    require(resolved['arrays']==m['arrays'] and resolved['row_count']==m['raw_rows'] and digest(resolved)==m['resolved_metadata_sha256'],
            'Prepared resolved data differ from actual full-input proof')
    require([x['sha256'] for x in ancestry]==[x['sha256'] for x in m['ancestry']],'Prepared original ancestry differs')
    historical={(str(store.path(x['reference'],Path(x['container']['file']).parent).absolute()).casefold(),x['reference']['sha256']) for x in m['historical_audit_only']}
    validator={(str(Path(x['file']).absolute()).casefold(),x['sha256']) for x in m['validator_sources']}
    dependencies={};containers={};implementation={};queue=[m['source_packet'],m['qualification']['original_qualification']]
    queue.extend(x['container'] for x in m['historical_audit_only']);queue.extend(x['container'] for x in m.get('qualification_generations',[]))
    def dependency(r,b,container=None,pointer=None):
        p=store.path(r,b).absolute();key=(str(p).casefold(),r['sha256'])
        if key in dependencies:return
        if key in historical or key in validator:return
        address=p.as_posix().casefold()
        if p.suffix==".py" and "/calibration/rinkfit/" in address and any(x in address for x in ("/src/","/tools/")):
            implementation[key]=dict(reference=dict(file=str(p),sha256=r["sha256"]),container=container,json_pointer=pointer,role="historical_repository_implementation",used_as_runtime_data=False)
            return
        actual=store.verify(r,b);dependencies[key]=dict(original=dict(file=str(p),sha256=r['sha256']),actual=actual)
    for r in [m['source_packet'],m['source_model'],*m['arrays'].values(),*m['qualification'].values()]:
        if isinstance(r,dict) and 'file'in r and 'sha256'in r:dependency(r,base)
    # Data evidence closure is enumerated once by the producer. Historical issuer
    # code stays in the old proof, while all source/data references stay exact.
    while queue:
        r=queue.pop();p=store.path(r,base).absolute();key=(str(p).casefold(),r['sha256'])
        if key in containers:continue
        raw=store.read(r,p.parent);doc=strict_json(raw);containers[key]=dict(file=str(p),sha256=r['sha256'])
        dependency(r,p.parent)
        if len(containers)%20==0:print(json.dumps(dict(prepared_metadata=len(containers),unique_source_dependencies=len(dependencies))),flush=True)
        for pointer,x in pointer_refs(doc):
            dependency(x,p.parent,r,pointer)
            xp=store.path(x,p.parent)
            if xp.suffix.lower()=='.json' and set(pointer.split('/')) & METADATA_LINKS:queue.append(x)
    # Current full W, Query, source registries and source-only refs in the resolved
    # payload are dependencies even when introduced by an append container.
    for _,r in references(resolved):dependency(r,base)
    payload=out/'resolved.json';payload.write_text(json.dumps(resolved,ensure_ascii=False,separators=(',',':'))+'\n','utf8')
    closure=out/'dependencies.json';closure.write_text(json.dumps({'schema':'rinkfit.prepared_source_dependencies.v1','dependencies':list(dependencies.values())},ensure_ascii=False,indent=2)+'\n','utf8')
    receipt={'schema':SCHEMA,'effective_input':effective_binding,'cold_native_proof':cold_report_binding,
      'resolved_payload':ref(payload),'dependency_closure':ref(closure),'source_packet':m['source_packet'],'source_model':m['source_model'],
      'raw_rows':m['raw_rows'],'resolved_metadata_sha256':m['resolved_metadata_sha256'],'arrays':m['arrays'],'qualification':m['qualification'],
      'engineering_support_budget_v2':m['engineering_support_budget_v2'],
      'proof_validator':{'validator_code_sha256':m['validator_code_sha256'],'compiled_source_validator_sha256':m['compiled_source_validator_sha256']},
      'proof_scope':{'physical_rows':report['qualification']['physical_rows'],'training_rows':report['qualification']['training_rows'],'typed_groups':report['typed_groups']},
      'original_validator_identity_preserved':True,'native_source_mathematics_reexecuted_during_preparation':False,
      'source_dependency_count':len(dependencies),'metadata_containers':len(containers),'historical_implementation_dependencies':list(implementation.values()),'quality_accepted':False,'source_image_identity_proven':False}
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf8')
    return dict(prepared=ref(target),elapsed_seconds=time.perf_counter()-start,source_dependency_count=len(dependencies),metadata_containers=len(containers),proof_validator=receipt['proof_validator'])

def validate_configuration(cfg,m):
    store=CheckedBytes();b=cfg['prepared_effective_input_v1'];p=store.json(b,Path(b['file']).parent)
    require(p['schema']==SCHEMA and same(p['effective_input'],cfg['effective_input_v1']),'Prepared effective binding differs')
    report=store.json(p['cold_native_proof'],Path(p['cold_native_proof']['file']).parent);validate_proof(m,report,cfg['effective_input_v1'])
    for k in ('source_packet','source_model','raw_rows','resolved_metadata_sha256','arrays','qualification','engineering_support_budget_v2'):
        require(p[k]==m[k],'Prepared current source field differs: '+k)
    require(p['proof_validator']=={k:m[k] for k in ('validator_code_sha256','compiled_source_validator_sha256')},'Prepared original validator identity changed')
    return p

def main():
    a=argparse.ArgumentParser();a.add_argument('--effective',required=True);a.add_argument('--proof',required=True);a.add_argument('--out',required=True);x=a.parse_args()
    print(json.dumps(prepare(ref(x.effective),ref(x.proof),x.out),ensure_ascii=False))
if __name__=='__main__':main()
