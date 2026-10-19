"""Content identity checks for native candidate bundles; never grants fit acceptance.

A file hash binds saved values to their producer report. These checks do not
recompute projections or certify station optimality, derivatives, or evidence.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

STATE_FILES = ('model.json', 'candidate_predictions.npz', 'station.f64',
               'path.u32', 'initial_row.f64', 'initial_alpha.f64')
ROW_FILES = {'station.f64': ('<f8', 'stations'),
             'path.u32': ('<u4', 'branches'),
             'initial_row.f64': ('<f8', 'root_rows'),
             'initial_alpha.f64': ('<f8', 'profile_alpha')}
CURRENT_REPORT_SCHEMA = 'rinkfit.full_active_finite_RS_inner_trial.v1'
LEGACY_REPORT_SCHEMAS = {'rinkfit.same_theta_full_ordinary_complete_native_seed.v1'}
LEGACY_MANIFEST_SCHEMAS = {'rinkfit.complete_continuous_reduced_SLP_checkpoint.v1'}

class StateIntegrityError(ValueError):
    """A saved state is missing, stale, or inconsistent with its authority."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_reference(spec, base, *, expected_path=None):
    if not isinstance(spec, dict) or not isinstance(spec.get('sha256'), str):
        raise StateIntegrityError('Missing content reference/hash')
    value = spec.get('file', spec.get('path'))
    if not isinstance(value, str):
        raise StateIntegrityError('Missing content reference path')
    path = Path(value)
    path = path if path.is_absolute() else Path(base)/path
    if expected_path is not None and path.resolve() != Path(expected_path).resolve():
        raise StateIntegrityError('Reference redirects the expected state file: '+str(path))
    if not path.is_file() or sha256_file(path) != spec['sha256']:
        raise StateIntegrityError('Referenced content SHA mismatch: '+str(path))
    return path


def validate_report_files(state, report=None, *, required_files=STATE_FILES):
    """Require every producer ledger entry and its current exact bytes.

    Return the report dictionary. Passing a report does not waive any file check.
    Descriptor ledger entries may not redirect a canonical state filename.
    """
    state = Path(state)
    report = read_json(state/'report.json') if report is None else report
    ledger = report.get('files')
    if not isinstance(ledger, dict):
        raise StateIntegrityError('Evaluation report lacks its file ledger')
    for name in required_files:
        if name not in ledger:
            raise StateIntegrityError('Evaluation file ledger is incomplete: '+name)
        entry = ledger[name]
        if isinstance(entry, str):
            entry = dict(file=name, sha256=entry)
        verify_reference(entry, state, expected_path=state/name)
    return report


def validate_state_files(state, *, allow_legacy=False):
    """Validate a report ledger, or an explicitly enabled legacy checkpoint.

    Missing reports never silently permit current candidates. A legacy fallback
    requires a known checkpoint schema, its complete state token, model/auxiliary
    references, and an explicit saved-prediction reference.
    """
    state = Path(state)
    if (state/'report.json').is_file():
        return dict(authority='evaluation_report', report=validate_report_files(state))
    if not allow_legacy:
        raise StateIntegrityError('Missing evaluation report; legacy mode was not selected')
    path = state/'manifest.json'
    if not path.is_file():
        raise StateIntegrityError('Missing explicit legacy checkpoint manifest')
    manifest = read_json(path)
    if manifest.get('schema') not in LEGACY_MANIFEST_SCHEMAS:
        raise StateIntegrityError('Unsupported legacy checkpoint schema')
    body = dict(manifest)
    token = body.pop('state_token', None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if token != actual:
        raise StateIntegrityError('Legacy checkpoint state token mismatch')
    verify_reference(manifest.get('model'), state, expected_path=state/'model.json')
    rows = manifest.get('row_state_files', {})
    keys = {'station.f64':'station', 'path.u32':'branch',
            'initial_row.f64':'row', 'initial_alpha.f64':'alpha'}
    for name, key in keys.items():
        verify_reference(rows.get(key, rows.get(name)), state, expected_path=state/name)
    verify_reference(manifest.get('native_predictions'), state,
                     expected_path=state/'candidate_predictions.npz')
    return dict(authority='explicit_legacy_checkpoint', manifest=manifest)


def load_array(spec, base):
    path = verify_reference(spec, base)
    try:
        value = np.fromfile(path, spec['dtype'])
        shape = tuple(spec['shape'])
    except (KeyError, TypeError, ValueError) as exc:
        raise StateIntegrityError('Invalid typed array descriptor') from exc
    if value.size != int(np.prod(shape)):
        raise StateIntegrityError('Typed array byte count differs: '+str(path))
    return value.reshape(shape)


def _equal(actual, expected, what):
    if np.shape(actual) != np.shape(expected) or not np.array_equal(actual, expected, equal_nan=True):
        raise StateIntegrityError(what+' differs')


def validate_prediction_arrays(state, *, n, native_pts, active_mask, native_uv=None,
                               kind=None, profile_normal=None, material_index=None,
                               model=None, predictions=None):
    """Check saved NPZ and four auxiliaries, optionally reusing an open mapping.

    Return an array dictionary, owned by the caller. No optical/RS projection is
    replayed: RS_valid is a producer claim bound by the separately checked ledger.
    """
    state = Path(state)
    if type(n) is not int or n < 1:
        raise StateIntegrityError('Invalid explicit source row count')
    active = np.asarray(active_mask)
    if active.shape != (n,) or not np.isin(active, [0, 1]).all():
        raise StateIntegrityError('Invalid expected active source mask')
    active = active.astype(bool)
    if np.shape(native_pts) != (n,):
        raise StateIntegrityError('Expected native PTS shape differs')
    if predictions is None:
        with np.load(state/'candidate_predictions.npz', allow_pickle=False) as saved:
            z = {key:saved[key] for key in saved.files}
    else:
        z = {key:np.asarray(predictions[key]) for key in predictions}
    shapes = {key:(n,) for key in ['source_rows','native_pts','active_source_mask',
        'stations','branches','root_rows','profile_alpha','coverage_error_px','RS_valid']}
    shapes.update(uv=(n,2), error_native_uv=(n,2))
    for key, shape in shapes.items():
        if key not in z or z[key].shape != shape:
            raise StateIntegrityError('Saved prediction key/shape differs: '+key)
    if not np.issubdtype(z['source_rows'].dtype, np.integer):
        raise StateIntegrityError('Saved source rows must be integers')
    _equal(z['source_rows'], np.arange(n), 'Saved source row order')
    if not np.issubdtype(z['native_pts'].dtype, np.integer):
        raise StateIntegrityError('Saved native PTS must be integers')
    _equal(z['native_pts'], native_pts, 'Saved native PTS')
    _equal(z['active_source_mask'], active, 'Saved source scope')
    if not np.isin(z['RS_valid'], [0,1]).all() or not z['RS_valid'][active].all():
        raise StateIntegrityError('Active saved RS validity fails')
    for name, (dtype, key) in ROW_FILES.items():
        path = state/name
        if not path.is_file() or path.stat().st_size != n*np.dtype(dtype).itemsize:
            raise StateIntegrityError('Auxiliary byte count differs: '+name)
        aux = np.fromfile(path, dtype)
        _equal(aux, z[key], 'Auxiliary/NPZ '+name)
    for key in ['uv','error_native_uv','stations','root_rows','profile_alpha','coverage_error_px']:
        if not np.isfinite(z[key][active]).all():
            raise StateIntegrityError('Nonfinite active prediction: '+key)
    if np.any((z['stations'][active]<0)|(z['stations'][active]>1)):
        raise StateIntegrityError('Station outside finite material domain')
    if not np.issubdtype(z['branches'].dtype, np.integer):
        raise StateIntegrityError('Saved path indices must be integers')
    if native_uv is not None:
        if np.shape(native_uv) != (n,2):
            raise StateIntegrityError('Expected native UV shape differs')
        error = z['uv']-native_uv
        _equal(z['error_native_uv'][active], error[active], 'Saved native UV residual')
        if kind is not None and profile_normal is not None:
            if np.shape(kind)!=(n,) or np.shape(profile_normal)!=(n,2):
                raise StateIntegrityError('Expected semantic shapes differ')
            coverage = np.linalg.norm(error, axis=1)
            profile = np.asarray(kind)==2
            coverage[profile] = abs(np.einsum('ij,ij->i',np.asarray(profile_normal)[profile],error[profile]))
            _equal(z['coverage_error_px'][active],coverage[active],'Saved engineering error')
    if model is not None and material_index is not None:
        if np.shape(material_index)!=(n,):
            raise StateIntegrityError('Expected material index shape differs')
        mi = np.asarray(material_index)
        if not np.issubdtype(mi.dtype,np.integer) or np.any((mi[active]<0)|(mi[active]>=len(model['materials']))):
            raise StateIntegrityError('Unknown active material')
        counts = np.asarray([len(m['paths']) for m in model['materials']])
        if np.any((z['branches'][active]<0)|(z['branches'][active]>=counts[mi[active]])):
            raise StateIntegrityError('Unknown active finite path')
    return z


def _check_flattened_replacements(stats, replacements, base):
    """Reject a second, unchecked W table diverging from validated authority."""
    rows = stats.get('replacements')
    if not isinstance(rows,list) or len(rows)!=len(replacements):
        raise StateIntegrityError('Flattened statistics replacement ledger differs')
    seen = set()
    for item in rows:
        gid = item['group_id']
        if gid not in replacements or gid in seen:
            raise StateIntegrityError('Unknown/duplicate flattened W replacement')
        seen.add(gid)
        expected = replacements[gid]
        _equal(np.asarray(item['source_rows']),expected['source_rows'],'Flattened W source rows')
        _equal(load_array(item['W'],base),expected['W'],'Flattened W matrix')


def validate_candidate_state(state, packet, semantic, *, require_all_ordinary=True,
                             allow_legacy_report=False, predictions=None,
                             check_scene=True, check_statistics=True):
    """Validate one complete saved value state against explicit source authorities.

    Return report/model/predictions/packet/semantic dictionaries. The default
    checks the semantic parent chain and original W cost, but it is still a saved
    value consistency check, not independent source evidence or a new RS solve.
    A known historical report without a packet ref requires allow_legacy_report.
    """
    state, packet, semantic = Path(state), Path(packet), Path(semantic)
    rep = validate_report_files(state)
    schema = rep.get('schema')
    legacy = allow_legacy_report and schema in LEGACY_REPORT_SCHEMAS
    if schema != CURRENT_REPORT_SCHEMA and not legacy:
        raise StateIntegrityError('Unsupported complete native candidate report schema')
    if rep.get('semantic',{}).get('sha256') != sha256_file(semantic):
        raise StateIntegrityError('Evaluation semantic authority differs')
    verify_reference(rep['semantic'],state)
    if rep.get('packet',{}).get('sha256') != sha256_file(packet):
        if not legacy or 'packet' in rep:
            raise StateIntegrityError('Evaluation packet differs or is absent')
    else:
        verify_reference(rep['packet'],state)
    pk, sem, model = read_json(packet), read_json(semantic), read_json(state/'model.json')
    if not isinstance(model.get('blocks'),list) or not model['blocks'] or not isinstance(model.get('materials'),list) or not model['materials']:
        raise StateIntegrityError('Missing complete physical model blocks/materials')
    if any(not isinstance(m.get('paths'),list) or not m['paths'] for m in model['materials']):
        raise StateIntegrityError('Missing physical material paths')
    if any(not np.isfinite(np.asarray(b.get('values'),float)).all() for b in model['blocks']):
        raise StateIntegrityError('Nonfinite physical model values')
    source = pk.get('source_identity',pk.get('immutable_source_hashes',{}).get('observations.npz'))
    if source is None or model.get('source_identity') != source:
        raise StateIntegrityError('Physical model/packet source identity differs')
    if 'immutable_source_hashes' in pk and model.get('source_hashes') != pk['immutable_source_hashes']:
        raise StateIntegrityError('Physical model original source hashes differ')
    n = pk.get('observations')
    raw = {key:load_array(d,packet.parent) for key,d in pk['arrays'].items()}
    if raw['native_uv'].shape!=(n,2) or raw['source_rows'].shape!=(n,):
        raise StateIntegrityError('Packet source dimensions differ')
    _equal(raw['source_rows'],np.arange(n),'Packet source row order')
    tables = read_json(verify_reference(pk['string_tables'],packet.parent))
    signatures = sem.get('immutable_source_arrays',{})
    source_arrays = {'source_uv':raw['native_uv'],'native_pts':raw['native_pts'],
                     'frame_indices':raw['frame_index']}
    for key, table_key, index_key in [('source_point_ids','source_point_ids','point_id_index'),
                                     ('source_target_labels','material_labels','material_index')]:
        spec = signatures.get(key)
        if not isinstance(spec,dict):
            raise StateIntegrityError('Missing semantic immutable source signature: '+key)
        source_arrays[key] = np.asarray(tables[table_key],dtype=spec['dtype_str'])[raw[index_key]]
    for key, value in source_arrays.items():
        spec = signatures.get(key,{})
        if (list(value.shape)!=spec.get('shape') or value.dtype.str!=spec.get('dtype_str')
            or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()!=spec.get('sha256_c_contiguous_array_bytes')):
            raise StateIntegrityError('Packet changed immutable native source array: '+key)
    from .semantic_resolution import load_semantics
    _, arrays, _, extra = load_semantics(semantic,n,expected_sha256=sha256_file(semantic))
    active = load_array(sem['active_source_mask'],semantic.parent).astype(bool)
    kind, normal = arrays['kind'], arrays['profile_normal']
    if 'resolved_material_index' in extra:
        material = extra['resolved_material_index']
    elif 'resolved_material_index' in sem or 'original_material_index' in sem:
        material = load_array(sem.get('resolved_material_index',sem.get('original_material_index')),semantic.parent)
    else:
        material = raw['material_index']
    z = validate_prediction_arrays(state,n=n,native_pts=raw['native_pts'],active_mask=active,
        native_uv=raw['native_uv'],kind=kind,profile_normal=normal,material_index=material,
        model=model,predictions=predictions)
    height = int(pk['native_image_wh'][1])
    if np.any((z['root_rows'][active]<0)|(z['root_rows'][active]>height-1)):
        raise StateIntegrityError('Saved RS root row outside native sensor')
    expected_counts = {'raw_rows':n,'active_rows':int(active.sum()),'frames':int(pk['frame_count'])}
    if len(np.unique(raw['frame_index'])) != pk['frame_count']:
        raise StateIntegrityError('Packet frame count differs')
    if any(rep.get(k)!=v for k,v in expected_counts.items()):
        raise StateIntegrityError('Evaluation source/frame counts differ')
    best = rep.get('best',{})
    if best.get('valid') is not True or best.get('active_RS_valid_rows')!=int(active.sum()):
        raise StateIntegrityError('Evaluation active RS count differs')
    if best.get('query_numerical_roots') != int(np.sum(active&(kind==2))):
        raise StateIntegrityError('Evaluation profile root count differs')
    if require_all_ordinary and (best.get('locally_captured_ordinary_candidates')!=int(np.sum(active&(kind==0))) or best.get('inactive_under_current_worst_bound_rows')!=0):
        raise StateIntegrityError('Not every active ordinary station was recomputed')
    budget = raw['engineering_budget_px']
    if budget.shape!=(n,) or not np.isfinite(budget).all() or np.any(budget<=0):
        raise StateIntegrityError('Invalid original engineering budget')
    coverage = z['coverage_error_px'][active]
    q = (z['coverage_error_px']/budget)**2
    if not np.isclose(best.get('maximum_error_px',np.nan),coverage.max(),rtol=1e-12,atol=1e-12) or not np.isclose(best.get('max_q',np.nan),q[active].max(),rtol=1e-12,atol=1e-12):
        raise StateIntegrityError('Evaluation maximum/engineering metric differs')
    if not np.allclose(best.get('quantiles_px',[]),np.quantile(coverage,[.5,.95,.99,1]),rtol=1e-12,atol=1e-12) or best.get('over2')!=int(np.sum(coverage>2)):
        raise StateIntegrityError('Evaluation distribution metrics differ')
    scene = rep.get('whole_static_scene',{})
    if scene.get('passed') is not True or scene.get('all_material_paths_checked')!=sum(len(m['paths']) for m in model['materials']):
        raise StateIntegrityError('Evaluation physical scene guard fails/count differs')
    if check_scene:
        from .native_scene import NativeScene
        if NativeScene(model).check_physics()!=scene:
            raise StateIntegrityError('Evaluation physical scene belongs to different model')
    if check_statistics:
        stats_path = verify_reference(sem['active_statistics'],semantic.parent)
        stats = read_json(stats_path)
        replacements = extra.get('source_exclusion',{}).get('replacements',{})
        _check_flattened_replacements(stats,replacements,stats_path.parent)
        groups_path = verify_reference(sem['assets']['groups.jsonl'],semantic.parent)
        members, weights = arrays['group_rows'], arrays['weights']
        error = z['error_native_uv'];partition=np.zeros(n,np.int8);rank=0;cost=0.
        for line in groups_path.read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            group=json.loads(line);rr=members[group['member_start']:group['member_start']+group['member_count']].astype(int)
            W=weights[group['weight_start']:group['weight_start']+group['rank']*2*len(rr)].reshape(group['rank'],2*len(rr))
            if group['group_id'] in replacements:
                r=replacements[group['group_id']];rr,W=r['source_rows'],r['W']
            partition[rr]+=1;rank+=len(W);v=W@error[rr].reshape(-1);cost+=.5*float(v@v)
        ind=arrays['ordinary_rows'].astype(int);ind=ind[active[ind]];partition[ind]+=1;rank+=2*len(ind)
        cost+=.5*float(np.sum((error[ind]/arrays['source_sigma'][ind,None])**2))
        if not np.array_equal(partition,active.astype(np.int8)) or rep.get('partition_exactly_once') is not True or rank!=stats.get('formal_residual_rows') or rank!=rep.get('statistics_rank'):
            raise StateIntegrityError('Evaluation original W partition/rank differs')
        if rep.get('source_W_E_unchanged') is not True or not np.isclose(cost,best.get('whole_original_W_half_cost',np.nan),rtol=1e-11,atol=1e-8):
            raise StateIntegrityError('Evaluation original W cost differs')
    return dict(report=rep,model=model,predictions=z,packet=pk,semantic=sem)


def validate_receipt_bindings(receipt, state, *, validated=None):
    """Bind a review receipt to one validated current candidate's exact files.

    Return the validated-state dictionary. Parent/proposal lineage and visual
    acceptance remain the producer's separate, explicitly recorded obligations.
    """
    state = Path(state)
    if receipt.get('schema') != 'rinkfit.complete_recomputed_native_candidate.v1':
        raise StateIntegrityError('Unsupported native candidate receipt schema')
    packet = verify_reference(receipt.get('packet'),state)
    semantic = verify_reference(receipt.get('semantic'),state)
    value = validate_candidate_state(state,packet,semantic) if validated is None else validated
    rep, model = value['report'], value['model']
    ledger = rep['files']
    refs = {'model.json':receipt.get('model'),
            'candidate_predictions.npz':receipt.get('predictions')}
    rowrefs = receipt.get('row_state_files',{})
    if set(rowrefs)!=set(ROW_FILES):
        raise StateIntegrityError('Receipt four-auxiliary reference set differs')
    refs.update(rowrefs)
    for name,spec in refs.items():
        verify_reference(spec,state,expected_path=state/name)
        entry = ledger[name]
        digest = entry if isinstance(entry,str) else entry['sha256']
        if spec['sha256']!=digest:
            raise StateIntegrityError('Receipt/evaluation file identity differs: '+name)
    for key in ['packet','semantic']:
        if receipt[key]['sha256']!=rep[key]['sha256']:
            raise StateIntegrityError('Receipt/evaluation '+key+' identity differs')
    best = rep['best']
    counts = {'source_rows':rep['raw_rows'],'active_rows':rep['active_rows'],
        'all_frames':rep['frames'],
        'all_ordinary_stations_recomputed':best['locally_captured_ordinary_candidates'],
        'true_RS_valid_rows':best['active_RS_valid_rows'],
        'profile_new_numerical_roots':best['query_numerical_roots']}
    if any(receipt.get(k)!=v for k,v in counts.items()):
        raise StateIntegrityError('Receipt/evaluation source or recapture counts differ')
    if receipt.get('metrics')!=best or receipt.get('physical_guard')!=rep['whole_static_scene']:
        raise StateIntegrityError('Receipt/evaluation metrics or physical guard differs')
    if receipt.get('source_identity')!=model['source_identity']:
        raise StateIntegrityError('Receipt physical source identity differs')
    if receipt.get('complete_state_for_fresh_relinearization') is not True or receipt.get('physical_fit_accepted') is not False:
        raise StateIntegrityError('Receipt does not describe an unaccepted complete value state')
    if 'evaluation_report' in receipt:
        verify_reference(receipt['evaluation_report'],state,expected_path=state/'report.json')
    return value
