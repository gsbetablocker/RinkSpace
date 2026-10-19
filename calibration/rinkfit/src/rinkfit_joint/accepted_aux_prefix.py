"""Carry a complete engineering candidate's four auxiliaries across append only.

This numerical state binding grants no source qualification or formal acceptance.
"""
from __future__ import annotations
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
from .io_utils import file_sha256

OBJECTIVE = 'engineering_recovery_native_pixel_v1'
AUX = {'station': ('station.f64', '<f8'), 'path': ('path.u32', '<u4'),
       'row': ('row.f64', '<f8'), 'alpha': ('alpha.f64', '<f8')}

def _ref(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=file_sha256(path))

def _verified(ref, base, cache=None):
    path = Path(ref['file'])
    path = path if path.is_absolute() else Path(base)/path
    path=path.resolve();key=(str(path),ref['sha256'])
    if cache is None or key not in cache:
        if file_sha256(path) != ref['sha256']:raise ValueError('changed engineering state dependency '+str(path))
        if cache is not None:cache.add(key)
    return path

def _array(doc, name, base):
    d = doc['arrays'][name]
    path = _verified(d, base)
    return np.memmap(path, mode='r', dtype=d['dtype'], offset=d.get('offset_bytes', 0), shape=tuple(d['shape']))

def _prefix_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def _digest(value):
    """Typed JSON digest shared with the native prefix consumer; no float text ABI."""
    h = hashlib.sha256()
    def put(v):
        if v is None: h.update(b'n')
        elif isinstance(v, bool): h.update(b't' if v else b'f')
        elif isinstance(v, int): h.update(b'i'+str(v).encode()+b';')
        elif isinstance(v, float):
            if not math.isfinite(v): raise ValueError('nonfinite source-prefix JSON')
            h.update(b'd'+struct.pack('<d', v))
        elif isinstance(v, str):
            b = v.encode('utf8'); h.update(b's'+str(len(b)).encode()+b':'+b)
        elif isinstance(v, list):
            h.update(b'[')
            for x in v: put(x)
            h.update(b']')
        elif isinstance(v, dict):
            h.update(b'{')
            for k in sorted(v): put(k); put(v[k])
            h.update(b'}')
        else: raise TypeError(type(v))
    put(value)
    return h.hexdigest()

def _runtime_query_digest(queries):
    h = hashlib.sha256()
    for q in sorted(queries,key=lambda q:int(q['row_index'])):
        record = dict(row_index=int(q['row_index']),alpha=[float(v) for v in q['alpha']],
            uv=[[float(v) for v in p] for p in q['native_uv']],
            nx=[float(v) for v in q.get('numerator_x',[])],ny=[float(v) for v in q.get('numerator_y',[])],
            den=[float(v) for v in q.get('denominator',[])],origin=float(q.get('alpha_origin',0)),
            scale=float(q.get('alpha_scale',1)),normal=[float(v) for v in q['normal']])
        h.update(_digest(record).encode())
    return h.hexdigest()

def _runtime_group_digest(doc,base,verified_files=None):
    h = hashlib.sha256(); count = 0; mappings={}
    def add(g, w):
        nonlocal count
        h.update(_digest(dict(id=g['id'],rows=[int(i) for i in g['rows']],rank=int(g['rank']),
            statistic_kind=g['statistic_kind'],W_values_sha256=_prefix_hash(np.asarray(w,dtype='<f8')))).encode())
        count += 1
    for g in doc['groups']:
        if isinstance(g['W'],dict):
            d=g['W'];p=_verified(d,base,verified_files);key=(str(p),d['dtype'])
            if key not in mappings:mappings[key]=np.memmap(p,mode='r',dtype=d['dtype'])
            start=int(d.get('offset_bytes',0))//np.dtype(d['dtype']).itemsize;size=int(np.prod(d['shape']))
            w=mappings[key][start:start+size].reshape(d['shape'])
        else:w=g['W']
        add(g,w)
    if doc.get('independent_rows'):
        sigma=_array(doc,'source_sigma',base);n=int(doc['row_count'])
        for i in doc['independent_rows']:
            sx=float(sigma[i]) if sigma.size==n else float(sigma.reshape(n,2)[i,0]);sy=sx if sigma.size==n else float(sigma.reshape(n,2)[i,1])
            add(dict(id='independent/'+str(i),rows=[i],rank=2,statistic_kind='original'),[1/sx,0.,0.,1/sy])
    return h.hexdigest(),count

def _cpp_hex(value):
    # This production backend uses MSVC: std::hexfloat emits all thirteen
    # hexadecimal fraction digits, including zero (verified by native probe).
    mantissa, exponent = float(value).hex().split('p')
    integer, fraction = mantissa.split('.')
    return integer+'.'+fraction.ljust(13,'0')+'p'+exponent

def _quoted(value):
    return '"'+value.replace('\\', '\\\\').replace('"', '\\"')+'"'

def _quoted_fields(line):
    decoder = json.JSONDecoder(); result = []; tail = line.strip()
    while tail:
        value, end = decoder.raw_decode(tail)
        if not isinstance(value, str): raise ValueError('checkpoint string expected')
        result.append(value); tail = tail[end:].lstrip()
    return result

def verify_engineering_checkpoint(path, receipt, model, aux, active, query_rows, station_rows=None):
    """Stream the complete standard checkpoint; recompute its native FNV token."""
    h = 14695981039346656037
    def canonical(text):
        nonlocal h
        for b in text.encode('utf8'): h = ((h ^ b)*1099511628211) & ((1 << 64)-1)
    seen = set(); rows = set(); stations = set(); alphas = set(); blocks = model['blocks']
    with Path(path).open(encoding='utf8') as stream:
        def line():
            value = stream.readline()
            if not value: raise ValueError('truncated engineering checkpoint')
            return value.rstrip('\r\n')
        def count():
            v = int(line())
            if not 0 <= v <= 20_000_000: raise ValueError('invalid checkpoint count')
            return v
        if line() != 'RINKFIT_AL_CHECKPOINT_V1': raise ValueError('unsupported engineering checkpoint')
        fingerprint = _quoted_fields(line())
        if fingerprint != [receipt['configuration_sha256']]: raise ValueError('engineering configuration mismatch')
        outer = line().split()
        if len(outer) != 7 or int(outer[0]) != 2 or float(outer[4]) != 0:
            raise ValueError('engineering checkpoint contains an old AL phase or rho')
        if line() != '0' or line() != '0': raise ValueError('engineering checkpoint contains old AL duals')
        if count() != 1: raise ValueError('missing engineering current state')
        token = _quoted_fields(line())
        if token != [receipt['candidate_state_token']]: raise ValueError('engineering candidate token mismatch')
        identity = _quoted_fields(line())
        if len(identity) != 6 or identity[0] != receipt['source_packet_sha256'] or identity[5] != OBJECTIVE:
            raise ValueError('engineering checkpoint source or objective mismatch')
        canonical(' '.join(_quoted(v) for v in identity)+'\n')
        size = count(); canonical(str(size)+'\n')
        if size < len(blocks): raise ValueError('incomplete engineering physical blocks')
        for index in range(size):
            text = line(); decoder = json.JSONDecoder(); name, end = decoder.raw_decode(text)
            flag = text[end:].strip()
            if flag not in ('0','1') or not name or name in seen: raise ValueError('invalid engineering parameter')
            seen.add(name); values_text = line().split(); n = int(values_text[0])
            values = [float(x) for x in values_text[1:]]
            if n != len(values) or not n or not all(math.isfinite(x) for x in values): raise ValueError('incomplete engineering parameter values')
            canonical(_quoted(name)+' '+flag+' '+str(n)+''.join(' '+_cpp_hex(v) for v in values)+'\n')
            if index < len(blocks):
                if name != blocks[index]['name'] or values != blocks[index]['values'] or flag != '0':
                    raise ValueError('engineering checkpoint physical model differs')
            elif name.startswith(('station/','row/','alpha/')):
                key, raw = name.split('/'); i = int(raw)
                if i < 0 or i >= len(active) or not active[i] or n != 1 or values[0] != aux[key][i]:
                    raise ValueError('engineering checkpoint explicit auxiliary differs '+name)
                {'station': stations, 'row': rows, 'alpha': alphas}[key].add(i)
        maps = []
        for _ in range(2):
            size = count(); canonical(str(size)+'\n'); values = {}; previous = None
            for _ in range(size):
                fields = _quoted_fields(line())
                if len(fields) != 2 or fields[0] in values or (previous is not None and fields[0] <= previous):
                    raise ValueError('invalid engineering discrete state')
                previous = fields[0]; values[fields[0]] = fields[1]
                canonical(_quoted(fields[0])+' '+_quoted(fields[1])+'\n')
            maps.append(values)
        if count() != 0 or line() != 'END_RINKFIT_AL_CHECKPOINT' or stream.read().strip():
            raise ValueError('engineering checkpoint contains old best state or trailing data')
    if f'{h:016x}' != token[0]: raise ValueError('engineering complete state token is corrupted')
    active_rows = set(np.flatnonzero(active).tolist())
    if rows != active_rows or alphas != (active_rows & set(query_rows)):
        raise ValueError('engineering checkpoint omitted active row or Query auxiliaries')
    if station_rows is not None and stations != set(station_rows):
        raise ValueError('engineering checkpoint omitted active station auxiliaries')
    if maps[0].get('joint_packet_sha256') != receipt['source_packet_sha256']:
        raise ValueError('engineering discrete source generation mismatch')
    position = 0
    for run in maps[0].get('finite_material_paths_RLE_u32_v1','').split(';'):
        if not run: continue
        count_text, value = run.split(':'); count = int(count_text); value = int(value)
        if count <= 0 or position+count > len(active) or not np.all(aux['path'][position:position+count] == value):
            raise ValueError('engineering complete discrete path differs')
        position += count
    if position != len(active): raise ValueError('engineering complete discrete path omitted rows')
    return dict(state_token=token[0], objective_id=OBJECTIVE, complete_parameters=len(seen),
                runtime_fingerprint=fingerprint[0], source_packet_sha256=identity[0], AL_duals_resumed=False)

def _checkpoint_physical_prefix(path, model, receipt):
    values=[]
    with Path(path).open(encoding='utf8') as stream:
        lines=[stream.readline().strip() for _ in range(9)]
        if lines[0]!='RINKFIT_AL_CHECKPOINT_V1' or _quoted_fields(lines[1])!=[receipt['configuration_sha256']] or lines[3:6]!=['0','0','1'] or _quoted_fields(lines[6])!=[receipt['candidate_state_token']]:
            raise ValueError('incomplete native engineering checkpoint prefix')
        identity=_quoted_fields(lines[7])
        if identity[0]!=receipt['source_packet_sha256'] or identity[-1]!=OBJECTIVE or int(lines[8])<len(model['blocks']):
            raise ValueError('native engineering checkpoint source/objective prefix differs')
        for block in model['blocks']:
            text=stream.readline().strip();name,end=json.JSONDecoder().raw_decode(text)
            fields=stream.readline().split();v=[float(x)for x in fields[1:]]
            if name!=block['name'] or text[end:].strip()!='0' or int(fields[0])!=len(v) or len(v)!=len(block['values']) or not all(math.isfinite(x)for x in v):
                raise ValueError('invalid native engineering physical checkpoint prefix')
            values.append(v)
    return values

def _exact_model_from_CP(exported, cp_values):
    import copy
    aliases={};used=set();other=set()
    for shot in exported['shots'].values():
        coordinates=shot.get('control_coordinates',[dict(block=i,offset=0)for i in shot['control_blocks']])
        if len(coordinates)!=len(shot['control_blocks']):raise ValueError('ambiguous control alias allocation')
        for target,c in zip(shot['control_blocks'],coordinates):
            source=int(c['block']);offset=int(c['offset']);used.add(source)
            if target!=source or offset:
                value=dict(source_block=source,offset=offset)
                if target in aliases and aliases[target]!=value:raise ValueError('ambiguous control mirror')
                aliases[int(target)]=value
    def references(value):
        if isinstance(value,dict):
            for key,v in value.items():
                if key in ('linear_control_manifolds','control_coordinates','control_blocks'):continue
                if key in ('block','geometry_block','advertisement_block','half_width_block','head_block','optics_block','reference_angles_block','scene_block') and isinstance(v,int):other.add(v)
                else:references(v)
        elif isinstance(value,list):
            for v in value:references(v)
    references(exported)
    exact=copy.deepcopy(exported);changes=[]
    for index,(block,cp)in enumerate(zip(exported['blocks'],cp_values)):
        if block['values']==cp:continue
        alias=aliases.get(index)
        if alias is None or index in used or index in other or not block['name'].startswith('shot/') or '/control/'not in block['name']:
            raise ValueError('CP/export difference is a consumed physical block')
        source=alias['source_block'];offset=alias['offset']
        if block['values']!=cp_values[source][offset:offset+3]:raise ValueError('exported mirror does not equal the exact consumed CP source slice')
        changes.append(dict(block=index,name=block['name'],source_block=source,offset=offset,
                            exported_values=block['values'],exact_CP_values=cp))
        exact['blocks'][index]['values']=cp
    if len(cp_values)!=len(exported['blocks']):raise ValueError('incomplete exact CP model')
    return exact,changes

def repack_engineering_exact_candidate(candidate, output_root):
    """Repair only exported nonconsumed mirrors; preserve the actual CP/function."""
    import shutil
    state=Path(candidate).resolve();out=Path(output_root).resolve();target=out/'candidate'
    if out.exists():raise FileExistsError('immutable exact engineering repack exists')
    original_receipt=state.parent/'engineering_receipt.json';receipt=json.loads(original_receipt.read_text(encoding='utf8'))
    exported_path=state/'model.json';checkpoint=state/'engineering_checkpoint.json';manifest_path=state/'explicit_nuisance/manifest.json'
    if receipt['objective_id']!=OBJECTIVE or receipt.get('source_recalled',True) or receipt.get('formal_Phase_II_KKT_passed',True) or file_sha256(exported_path)!=receipt['model_sha256']:
        raise ValueError('incompatible original native engineering candidate')
    exported=json.loads(exported_path.read_text(encoding='utf8'));cp_values=_checkpoint_physical_prefix(checkpoint,exported,receipt)
    exact,changes=_exact_model_from_CP(exported,cp_values)
    if not changes:raise ValueError('candidate requires no mirror repack')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['packet_sha256']!=receipt['source_packet_sha256']:raise ValueError('repack explicit source mismatch')
    # Verify everything before creating the isolated state directory.
    for name,(filename,dtype)in AUX.items():
        p=state/'explicit_nuisance'/filename
        if file_sha256(p)!=manifest['files'][filename]['sha256'] or p.stat().st_size!=int(manifest['raw_rows'])*np.dtype(dtype).itemsize:
            raise ValueError('incomplete original explicit repack state')
    (target/'explicit_nuisance').mkdir(parents=True)
    exact_path=target/'model.json';exact_path.write_text(json.dumps(exact,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    shutil.copyfile(checkpoint,target/'engineering_checkpoint.json');shutil.copyfile(manifest_path,target/'explicit_nuisance/manifest.json')
    for filename,_ in AUX.values():shutil.copyfile(state/'explicit_nuisance'/filename,target/'explicit_nuisance'/filename)
    proof=dict(schema='rinkfit.lossless_engineering_CP_exact_model_repack.v1',
        original_native_receipt=_ref(original_receipt),original_exported_model=_ref(exported_path),original_complete_checkpoint=_ref(checkpoint),
        original_explicit_manifest=_ref(manifest_path),exact_model=_ref(exact_path),
        complete_checkpoint=_ref(target/'engineering_checkpoint.json'),explicit_manifest=_ref(target/'explicit_nuisance/manifest.json'),
        changed_nonconsumed_control_mirrors=changes,changed_mirror_blocks=len(changes),all_other_model_fields_exact=True,
        all_consumed_control_coordinates_and_master_values_exact=True,all_complete_CP_block_values_preserved=True,
        physical_projection_function_changed=False,new_optimization_performed=False,source_qualification_issued=False,formal_acceptance_claimed=False,
        candidate_state_token=receipt['candidate_state_token'],source_packet_sha256=receipt['source_packet_sha256'],objective_id=OBJECTIVE)
    proof_path=out/'lossless_CP_model_repack.json';proof_path.write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    rebound={**receipt,'model_sha256':file_sha256(exact_path),'lossless_CP_exact_model_repack':_ref(proof_path)}
    (out/'engineering_receipt.json').write_text(json.dumps(rebound,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    return dict(candidate=str(target),model=_ref(exact_path),receipt=_ref(out/'engineering_receipt.json'),
                proof=_ref(proof_path),changed_mirror_blocks=len(changes),no_new_optimization=True)

def _verify_exact_repack(state, receipt, model):
    ref=receipt.get('lossless_CP_exact_model_repack')
    if ref is None:return
    proof=json.loads(_verified(ref,state.parent).read_text(encoding='utf8'))
    if proof.get('schema')!='rinkfit.lossless_engineering_CP_exact_model_repack.v1':raise ValueError('invalid exact model repack contract')
    original_receipt=json.loads(_verified(proof['original_native_receipt'],state.parent).read_text(encoding='utf8'))
    expected={**original_receipt,'model_sha256':file_sha256(state/'model.json'),'lossless_CP_exact_model_repack':ref}
    if receipt!=expected:raise ValueError('exact model repack changed native Ceres/source/token/target metadata')
    exported=json.loads(_verified(proof['original_exported_model'],state.parent).read_text(encoding='utf8'))
    if file_sha256(_verified(proof['original_exported_model'],state.parent))!=original_receipt['model_sha256']:
        raise ValueError('exact model repack original native model mismatch')
    original_CP=_verified(proof['original_complete_checkpoint'],state.parent)
    cp_values=_checkpoint_physical_prefix(original_CP,exported,original_receipt)
    exact,changes=_exact_model_from_CP(exported,cp_values)
    if model!=exact or proof['changed_nonconsumed_control_mirrors']!=changes or proof['changed_mirror_blocks']!=len(changes):
        raise ValueError('exact model repack changed projection metadata or consumed values')
    for key in ('all_other_model_fields_exact','all_consumed_control_coordinates_and_master_values_exact','all_complete_CP_block_values_preserved'):
        if proof.get(key)is not True:raise ValueError('incomplete exact model repack proof')
    for key in ('physical_projection_function_changed','new_optimization_performed','source_qualification_issued','formal_acceptance_claimed'):
        if proof.get(key)is not False:raise ValueError('forbidden exact model repack claim')
    if _ref(state/'model.json')!=proof['exact_model'] or _ref(state/'engineering_checkpoint.json')!=proof['complete_checkpoint'] or file_sha256(state/'engineering_checkpoint.json')!=file_sha256(original_CP):
        raise ValueError('exact model repack changed full CP state')
    manifest_path=state/'explicit_nuisance/manifest.json';original_manifest=_verified(proof['original_explicit_manifest'],state.parent)
    if _ref(manifest_path)!=proof['explicit_manifest'] or file_sha256(manifest_path)!=file_sha256(original_manifest):
        raise ValueError('exact model repack changed explicit manifest')
    if proof['candidate_state_token']!=original_receipt['candidate_state_token'] or proof['source_packet_sha256']!=original_receipt['source_packet_sha256'] or proof['objective_id']!=OBJECTIVE:
        raise ValueError('exact model repack state/source/target mismatch')

def rebind_engineering_nuisance(packet_path, candidate, output_path):
    """Bind candidate/explicit_nuisance, never independent-RS reprofile output."""
    src = Path(packet_path).resolve(); state = Path(candidate).resolve(); out = Path(output_path).resolve()
    if out.exists(): raise FileExistsError('immutable engineering auxiliary packet exists')
    if state.name != 'candidate': raise ValueError('engineering binding requires the candidate explicit state, not independent reprofile')
    receipt_path = state.parent/'engineering_receipt.json'; receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('schema') != 'rinkfit.joint_engineering_recovery_run.v1' or receipt.get('objective_id') != OBJECTIVE:
        raise ValueError('not a native engineering pixel candidate')
    if receipt.get('source_recalled', True) or receipt.get('formal_Phase_II_KKT_passed', True) or receipt.get('common_worst_budget_consumed', True) or receipt.get('original_W_likelihood_consumed', True) or receipt.get('callback_error', ''):
        raise ValueError('recalled, failed, or incompatible engineering candidate')
    model_path = state/'model.json'
    if file_sha256(model_path) != receipt['model_sha256']: raise ValueError('engineering candidate physical model changed')
    model = json.loads(model_path.read_text(encoding='utf8'))
    _verify_exact_repack(state,receipt,model)
    manifest_path = state/'explicit_nuisance/manifest.json'; manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    parent = Path(receipt['source_packet']).resolve()
    if file_sha256(parent) != receipt['source_packet_sha256'] or manifest['packet_sha256'] != receipt['source_packet_sha256']:
        raise ValueError('engineering explicit state source mismatch')
    original = json.loads(parent.read_text(encoding='utf8')); packet = json.loads(src.read_text(encoding='utf8'))
    from .joint_solver import _source_membership_fingerprint
    for source,document in ((parent,original),(src,packet)):
        directories={source.parent}
        certificate=document.get('board_material_source_qualification')
        if certificate: directories.add(_verified(certificate,source.parent).parent)
        fingerprint=_source_membership_fingerprint(document)
        for ref in document.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')==fingerprint: directories.add(_verified(ref,source.parent).parent)
        for directory in directories:
            for marker in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
                if (directory/marker).exists(): raise ValueError('current engineering source qualification recalled '+str(directory/marker))
    old_n = int(manifest['raw_rows']); n = int(packet['row_count'])
    if int(original['row_count']) != old_n or n < old_n or original['shot_ids'] != packet['shot_ids']:
        raise ValueError('engineering appended scope or prefix count mismatch')
    for key in ('source_row_ids','source_dataset_ids'):
        if original[key] != packet[key][:old_n]: raise ValueError('changed engineering source ID prefix '+key)
    old_queries = [q for q in original.get('queries',[]) if int(q['row_index']) < old_n]
    new_queries = [q for q in packet.get('queries',[]) if int(q['row_index']) < old_n]
    if old_queries != new_queries: raise ValueError('changed engineering genuine Query prefix')
    for key in ('stroke_pairs','independent_rows'):
        previous = original.get(key,[])
        if packet.get(key,[])[:len(previous)] != previous: raise ValueError('changed engineering source topology prefix '+key)
    old_domains = original.get('source_station_domains',{}).get('rows',[])
    new_domains = [q for q in packet.get('source_station_domains',{}).get('rows',[]) if int(q['row_index']) < old_n]
    if old_domains != new_domains: raise ValueError('changed engineering finite source station domains')
    contract_arrays = {}; arrays = {}
    for name, old_d in original['arrays'].items():
        if name in AUX: continue
        old_a = _array(original, name, parent.parent); new_a = _array(packet, name, src.parent)
        if old_a.shape[0] != old_n or new_a.shape[0] != n or old_a.dtype != new_a.dtype or not np.array_equal(old_a,new_a[:old_n]):
            raise ValueError('changed engineering source array prefix '+name)
        contract_arrays[name] = dict(current_descriptor=packet['arrays'][name], prefix_sha256=_prefix_hash(old_a))
        arrays[name] = new_a
    verified_group_dependencies=set()
    def normalized_group(group, base):
        g = dict(group)
        if isinstance(g['W'],dict):
            d = dict(g['W']); d['file'] = str(_verified(d,base,verified_group_dependencies)); g['W'] = d
        return g
    old_groups = [normalized_group(g,parent.parent) for g in original['groups']]
    groups = [normalized_group(g,src.parent) for g in packet['groups']]
    if groups[:len(old_groups)] != old_groups or any(any(int(i)<old_n for i in g['rows']) for g in groups[len(old_groups):]):
        raise ValueError('changed engineering W group prefix or cross-prefix appended group')
    active = (arrays['admission'][:old_n] == 1)&(arrays['fit_role'][:old_n] == 1)
    aux = {}
    for name,(filename,dtype) in AUX.items():
        d = manifest['files'][filename]; p = _verified(dict(file=str(state/'explicit_nuisance'/filename),sha256=d['sha256']),state)
        a = np.memmap(p,mode='r',dtype=dtype)
        if a.shape != (old_n,) or not np.isfinite(a).all(): raise ValueError('incomplete engineering explicit '+name)
        aux[name] = a
    checkpoint = state/'engineering_checkpoint.json'
    required_station = set(np.flatnonzero(active & (arrays['kind'][:old_n] != 1)).tolist())
    checked = verify_engineering_checkpoint(checkpoint,receipt,model,aux,active,[q['row_index'] for q in old_queries],required_station)
    out.parent.mkdir(parents=True,exist_ok=True)
    checked_files = set()
    def absolute(d):
        p = Path(d['file']); p = p if p.is_absolute() else src.parent/p; p = p.resolve()
        key = (str(p),d['sha256'])
        if key not in checked_files: _verified(d,src.parent); checked_files.add(key)
        return {**d,'file':str(p)}
    packet['arrays'] = {k:absolute(d) for k,d in packet['arrays'].items()}
    for g in packet['groups']:
        if isinstance(g['W'],dict): g['W'] = absolute(g['W'])
    for d in packet.get('source_manifests',[]): d.update(absolute(d))
    for name in contract_arrays: contract_arrays[name]['current_descriptor'] = packet['arrays'][name]
    aux_contract = {}
    for name,(filename,dtype) in AUX.items():
        fresh = _array(packet,name,src.parent)
        target = out.parent/('accepted_current_'+filename)
        if target.exists(): raise FileExistsError('immutable engineering merged auxiliary exists')
        with target.open('wb') as stream:
            np.asarray(aux[name],dtype=dtype).tofile(stream)
            np.asarray(fresh[old_n:],dtype=dtype).tofile(stream)
        descriptor = dict(file=str(target),sha256=file_sha256(target),dtype=dtype,shape=[n])
        packet['arrays'][name] = descriptor
        aux_contract[name] = dict(candidate_file=_ref(state/'explicit_nuisance'/filename),current_descriptor=descriptor,prefix_sha256=_prefix_hash(aux[name]))
    runtime_groups_digest,runtime_group_count=_runtime_group_digest(original,parent.parent,verified_group_dependencies)
    contract = dict(schema='rinkfit.engineering_accepted_aux_prefix_contract.v1',old_raw_rows=old_n,new_raw_rows=n,
        old_source_packet=_ref(parent),appended_source_packet=_ref(src),source_arrays=contract_arrays,auxiliaries=aux_contract,
        source_ID_prefix_sha256=_digest([original['source_dataset_ids'],original['source_row_ids']]),
        query_prefix_sha256=_digest(old_queries),group_prefix_sha256=_digest(old_groups),old_group_count=len(old_groups),
        runtime_query_prefix_sha256=_runtime_query_digest(old_queries),runtime_group_prefix_sha256=runtime_groups_digest,
        runtime_old_group_count=runtime_group_count,shot_ids=original['shot_ids'],
        complete_Query_material_kind_normal_E_ID_PTS_UV_and_W_prefix_verified=True,
        full_candidate_explicit_arrays_preserved=True,old_active_rows=int(active.sum()),expected_station_rows=len(required_station))
    contract_path = out.parent/'accepted_aux_prefix_contract.json'
    if contract_path.exists(): raise FileExistsError('immutable engineering prefix contract exists')
    contract_path.write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    packet['accepted_explicit_state_binding'] = dict(schema='rinkfit.accepted_explicit_state_packet_binding.v2',
        state_kind='engineering_candidate',parent_packet=_ref(src),accepted_source_packet=_ref(parent),
        accepted_receipt=_ref(receipt_path),accepted_model=_ref(model_path),explicit_nuisance_manifest=_ref(manifest_path),
        complete_engineering_checkpoint=_ref(checkpoint),prefix_contract=_ref(contract_path),
        model_sha256=receipt['model_sha256'],state_token=checked['state_token'],objective_id=OBJECTIVE,
        configuration_sha256=receipt['configuration_sha256'],preserved_raw_rows=old_n,
        native_replay_pass_claimed=False,formal_acceptance_claimed=False,old_AL_duals_resumed=False,
        source_UV_W_E_admission_roles_changed=False,new_appended_rows_must_be_initialized=True)
    out.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    from .joint_solver import _inherit_source_recall_marker
    _inherit_source_recall_marker(src,out)
    return dict(packet=str(out),sha256=file_sha256(out),model=str(model_path),model_sha256=receipt['model_sha256'],
                preserved_raw_rows=old_n,fresh_appended_rows=n-old_n,state_token=checked['state_token'],objective_id=OBJECTIVE)
