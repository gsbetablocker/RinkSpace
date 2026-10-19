"""Rebind only demonstrated arithmetic-equivalent source operator labels.

This never creates W, changes mean/covariance, or confers physical identity.
"""
from pathlib import Path
import ast,inspect,json
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_worker import _remeasure

def rebind_equivalent_statistics(registry_path,old_audit,new_audit,equivalence_path,output):
    out=Path(output)
    if out.exists():raise ValueError('new numerical-equivalence generation required')
    out.mkdir(parents=True)
    old={r['source_row_id']:r for r in read_jsonl(old_audit)};new={r['source_row_id']:r for r in read_jsonl(new_audit)}
    if set(old)!=set(new):raise ValueError('source population changed during semantic-only rebind')
    ep=Path(equivalence_path)
    if ep.name.endswith('.jsonl') or ep.name.endswith('.jsonl.gz'):evidence=list(read_jsonl(ep))
    else:evidence=json.loads(ep.read_text(encoding='utf8'))
    ids=set()
    def collect(x):
        if isinstance(x,dict):
            if 'source_row_id' in x:ids.add(x['source_row_id'])
            for v in x.values():collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    collect(evidence)
    # Current reusable source operator has exactly one load of operator:
    # its paint_center special case. All other labels use the same edge code.
    code=inspect.getsource(_remeasure);tree=ast.parse(code)
    loads=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='operator' and isinstance(n.ctx,ast.Load)]
    compares=[n for n in ast.walk(tree) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id=='operator' and len(n.ops)==1 and isinstance(n.ops[0],ast.Eq) and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value=='paint_center']
    if len(loads)!=1 or len(compares)!=1:raise ValueError('source operator dispatch changed; equivalence must be audited again')
    registry=json.loads(Path(registry_path).read_text(encoding='utf8'));rebound=[];checked={}
    for gid,group in registry.items():
        ref=group['W'];wp=Path(ref['file']);wp=wp if wp.is_absolute() else Path(registry_path).parent/wp
        if str(wp) not in checked:
            if file_sha256(wp)!=ref['sha256']:raise ValueError('numeric W changed')
            checked[str(wp)]=ref['sha256']
        group['W']={**ref,'file':str(wp.resolve())}
        for rid in group['row_ids']:
            a=old[rid];b=new[rid]
            def current(r):return object_sha256({'native_pts':r['native_pts'],'current_uv':r['current_uv'],'source_row_id':rid,'measurement_operator':r['measurement_operator_resolved']})
            old_hash=current(a);new_hash=current(b)
            if old_hash!=group['current_measurements_sha256'][rid]:raise ValueError('old W measurement hash differs before rebind: '+rid)
            if old_hash==new_hash:continue
            if rid not in ids:raise ValueError('operator equivalence proof missing source row: '+rid)
            for key in ['native_pts','source_uv','current_uv','source_family_id','point_correspondence','manual_reference_preserved']:
                if a.get(key)!=b.get(key):raise ValueError('semantic rebind changed numerical/source input: '+key)
            for key in ['current_uv','normal','profile','flank_uv','flank_profiles','moved_px','status']:
                if a['source_measurement_audit'].get(key)!=b['source_measurement_audit'].get(key):raise ValueError('native operator argument/output changed: '+key)
            ao=a['measurement_operator_resolved'];bo=b['measurement_operator_resolved']
            if ao=='paint_center' or bo=='paint_center' or a['point_correspondence'] or a['manual_reference_preserved']:raise ValueError('different native numeric operator branch cannot be rebound')
            original=object_sha256({'native_pts':b['native_pts'],'source_uv':b['source_uv'],'source_row_id':rid})
            if original!=group['source_measurements_sha256'][rid]:raise ValueError('immutable source identity changed')
            group['current_measurements_sha256'][rid]=new_hash
            rebound.append({'source_row_id':rid,'source_group_id':gid,'old_operator':ao,'new_operator':bo,'old_current_hash':old_hash,'new_current_hash':new_hash,'W_sha256':ref['sha256'],'source_mean_or_operator_arguments_changed':False,'arithmetic_branch':'unchanged _remeasure single material edge'})
        group['numeric_operator_label_equivalence_audited']=True
    write_json(out/'statistics_registry.json',registry);write_json(out/'rebound_rows.json',rebound)
    report={'schema':'rinkfit.numeric_equivalent_source_operator_rebind.v1','changed_statistical_row_labels':len(rebound),'groups':len(registry),
        'W_bytes_or_rank_changed':False,'source_mean_changed':False,'new_numeric_W_created':False,'physical_identity_admission_claimed':False,
        'old_registry':{'file':str(registry_path),'sha256':file_sha256(registry_path)},'old_audit':{'file':str(old_audit),'sha256':file_sha256(old_audit)},
        'new_audit':{'file':str(new_audit),'sha256':file_sha256(new_audit)},'source_equivalence_evidence':{'file':str(ep),'sha256':file_sha256(ep)},
        'numeric_source_operator_ast_sha256':object_sha256(ast.dump(tree,include_attributes=False)),
        'statistics_registry_sha256':file_sha256(out/'statistics_registry.json'),'formal_physical_noise_claimed':False}
    write_json(out/'manifest.json',report);return report
