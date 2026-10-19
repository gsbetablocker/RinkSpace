"""Fit-only coverage guard for the recalled native board-source issuer.

Historical replay and model-only numerical donors remain readable. A new
membership or auxiliary wrapper cannot re-authorize this source family.
"""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
from .io_utils import file_sha256,object_sha256

FAMILY_PREFIXES=('board_instance_native/','board_instance/','outer_native/')
CERTIFICATE_SCHEMA='rinkfit.board_material_fit_qualification.v1'
BOUND_ARRAYS=('uv','native_pts','shot_index','material_index','kind','admission','fit_role','profile_normal')

def _load_ref(ref,base,seen):
    path=Path(ref.get('file',ref.get('path','')))
    if not path.is_absolute():path=base/path
    # Hash the same immutable textual dependency once. Resolving this already
    # absolute evidence path for every member would reopen its directory tree
    # hundreds of thousands of times; alternate spellings only cause an extra
    # verification, never reuse authority from another dependency.
    key=(str(path),ref.get('sha256'))
    if key not in seen:
        if not ref.get('sha256')or file_sha256(path)!=ref['sha256']:raise ValueError('Changed board source qualification evidence: '+str(path))
        seen.add(key)
    return path

def assert_board_material_source_qualification(packet,packet_path,*,verified_dependency_cache=None):
    """Require new role AND frozen-operator pass for every active recalled ID."""
    ids=packet.get('source_row_ids',())
    affected=np.fromiter((i for i,s in enumerate(ids)if str(s).startswith(FAMILY_PREFIXES)),dtype=np.int64)
    if not len(affected):return
    base=Path(packet_path).resolve().parent;seen=set() if verified_dependency_cache is None else verified_dependency_cache;arrays={}
    def arr(name):
        d=packet['arrays'][name];p=Path(d['file']);p=p if p.is_absolute()else base/p
        _load_ref(dict(file=str(p),sha256=d['sha256']),base,seen)
        a=np.memmap(p,dtype=np.dtype(d['dtype']),mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
        arrays[name]=a;return a
    admission=arr('admission');role=arr('fit_role');mat=arr('material_index')
    active=affected[(admission[affected]==1)&(role[affected]==1)]
    if not len(active):return
    ref=packet.get('board_material_source_qualification')
    if not ref:raise ValueError('Recalled board source issuer: '+str(len(active))+' active native board readouts lack the new per-ID material-role and frozen-operator audit; old flags, membership edits and auxiliary wrappers cannot authorize fit')
    cert_path=_load_ref(ref,base,seen);cert=json.loads(cert_path.read_text(encoding='utf8'))
    for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
        if (cert_path.parent/name).exists():raise ValueError('Board material requalification certificate recalled: '+str(cert_path.parent/name))
    incremental = cert.get('schema') == 'rinkfit.board_material_fit_qualification.v2'
    if cert.get('schema') not in (CERTIFICATE_SCHEMA, 'rinkfit.board_material_fit_qualification.v2'):raise ValueError('Unsupported board physical requalification certificate')
    if incremental:
        from .qualified_scope import QUALIFICATION_DIMENSIONS, SOURCE_SHA, DOMAIN
        if (cert.get('qualification_dimensions') != list(QUALIFICATION_DIMENSIONS) or cert.get('source_video_sha256') != SOURCE_SHA or cert.get('time_base') != '1/12800' or cert.get('coordinate_domain') != DOMAIN or cert.get('exact_ID_scope_complete') is not True or cert.get('whole_original_scope_qualified') is not False):raise ValueError('Incremental certificate must declare exact qualified scope and complete native dimensions')
        _load_ref(cert['consumer_module_ref'], cert_path.parent, seen)
    if int(cert.get('count',-1))!=len(active)or cert.get('active_board_rows_sha256')!=hashlib.sha256(active.astype('<i8').tobytes()).hexdigest():raise ValueError('Board qualification does not cover the exact active source indices')
    for k in BOUND_ARRAYS:
        if cert.get('membership_array_sha256',{}).get(k)!=packet['arrays'][k]['sha256']:raise ValueError('Board qualification frozen source array differs: '+k)
    recipe_path=_load_ref(cert['recipe_ref'],cert_path.parent,seen);recipe=json.loads(recipe_path.read_text(encoding='utf8'))
    from .board_source_requalification import VERSION
    if recipe.get('version')!=VERSION:raise ValueError('Recalled board source qualification algorithm version')
    if recipe.get('public_function')!='rinkfit_joint.board_source_requalification.reevaluate_current_board_source_rows':raise ValueError('Unsupported physical board source qualification function')
    for module in recipe.get('frozen_module_refs',()):_load_ref(module,recipe_path.parent,seen)
    qa_path=_load_ref(recipe['counterexample_QA_ref'],recipe_path.parent,seen)
    if json.loads(qa_path.read_text(encoding='utf8')).get('passed')is not True:raise ValueError('Board source material counterexample QA failed')
    _load_ref(cert['source_scope_ref'],cert_path.parent,seen)
    current=Path(__file__).with_name('board_source_requalification.py').resolve()
    code_ref=next((r for r in recipe.get('frozen_module_refs',())if Path(r.get('path',r.get('file',''))).resolve()==current),None)
    if not incremental and (not code_ref or file_sha256(current)!=code_ref['sha256']):raise ValueError('Board material role qualification code fingerprint changed')
    if incremental and (cert.get('producer_module_is_historical_executed_archive') is not True or not recipe.get('historical_executed_recipe_ref')):raise ValueError('Incremental certificate lacks exact historical producer archive')
    for key in ('source_row_ids','source_dataset_ids'):
        digest=hashlib.sha256(json.dumps(packet.get(key,()),ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()
        if cert.get(key+'_sha256')!=digest:raise ValueError('Board qualification source-ID or dataset registry changed')
    binding=_load_ref(cert['qualified_active_board_rows_ref'],cert_path.parent,seen)
    if not np.array_equal(np.fromfile(binding,dtype='<i8'),active):raise ValueError('Board qualification exact active-row binary coverage differs')
    if int(cert.get('role_and_operator_pass_count',-1))!=len(active)or cert.get('unresolved_or_wrong_rows_consumed')!=0:raise ValueError('Board qualification cannot consume unresolved physical roles or operators')
    uv=arr('uv');pts=arr('native_pts');normals=arr('profile_normal') if incremental else None;datasets=packet.get('source_dataset_ids',())
    if len(datasets)!=len(ids):raise ValueError('Board source dataset binding missing')
    ledger=_load_ref(cert['qualified_rows_ref'],cert_path.parent,seen)
    native_ledger=_load_ref(cert['qualified_rows_native_ref'],cert_path.parent,seen)
    native_stream=native_ledger.open('r',encoding='utf8')
    opener=gzip.open if ledger.suffix=='.gz'else open;position=0
    with opener(ledger,'rt',encoding='utf8')as stream:
        for line in stream:
            r=json.loads(line)
            native_line=native_stream.readline()
            if not native_line:raise ValueError('Missing native board qualification ledger row')
            native_row=json.loads(native_line)
            for key in ('global_row','source_row_id','source_dataset_id','native_pts','current_uv','status','operator_status'):
                if native_row.get(key)!=r.get(key):raise ValueError('Native board qualification ledger differs from complete physical/operator audit: '+key)
            if position>=len(active):raise ValueError('Extra qualified board readout')
            i=int(active[position]);position+=1
            if int(r['global_row'])!=i or r.get('source_row_id')!=ids[i]or r.get('source_dataset_id')!=datasets[i]or int(r.get('native_pts',-1))!=int(pts[i]):raise ValueError('Board audit ID/dataset/clock mismatch')
            if r.get('status')!='pass'or r.get('operator_status')!='pass':raise ValueError('Unresolved or wrong material/operator cannot enter board fit')
            if incremental:
                if not np.array_equal(np.asarray(r.get('original_profile_normal'),float),normals[i]):raise ValueError('Incremental original frozen profile normal differs')
                if any(r.get('qualification_dimensions', {}).get(k) != 'pass' for k in QUALIFICATION_DIMENSIONS) or r.get('actual_readout_alias_of') is not None:raise ValueError('Incremental qualification has unknown source dimension or duplicate readout')
                if r.get('source_video_sha256') != SOURCE_SHA or r.get('time_base') != '1/12800' or r.get('coordinate_domain') != DOMAIN or int(r.get('material_index', -1)) != int(mat[i]):raise ValueError('Incremental native source/domain/material binding differs')
            if not np.array_equal(np.asarray(r.get('current_uv'),float),uv[i]):raise ValueError('Board requalification altered the original current mean')
            expected=object_sha256(dict(native_pts=int(pts[i]),current_uv=uv[i].tolist(),source_row_id=ids[i],measurement_operator=r.get('measurement_operator')))
            if expected!=r.get('current_measurement_sha256'):raise ValueError('Board material/operator evidence not bound to this actual mean')
            if not r.get('evidence_refs'):raise ValueError('Board pass has no actual material evidence')
            for evidence in r['evidence_refs']:_load_ref(evidence,ledger.parent,seen)
    if native_stream.read().strip():raise ValueError('Extra native board qualification ledger rows')
    native_stream.close()
    if position!=len(active):raise ValueError('Incomplete board source audit coverage')
