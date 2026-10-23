"""Compile an incremental exact-ID board qualification into an immutable packet.

Raw rows and the complete programme timeline are retained.  This compiler checks
existing RGB/operator/mask evidence; it does not infer identity from a fit, rerun
MC, change UV, or authorize a native auxiliary state.  A fresh native membership
bridge/domain check remains required before fitting the generated packet.
"""
from __future__ import annotations
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from .io_utils import file_sha256, object_sha256, write_json
from .packet_storage import write_packet
from .source_scope_repair import marginalize_group
from .board_source_qualification_guard import FAMILY_PREFIXES, BOUND_ARRAYS

SCHEMA = 'rinkfit.incremental_qualified_source_scope.v1'
CERTIFICATE_SCHEMA = 'rinkfit.board_material_fit_qualification.v2_operator_bound'
SOURCE_SHA = 'e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
DOMAIN = 'native_1920x1080_pixel_centres'
QUALIFICATION_DIMENSIONS = ('video_identity', 'native_clock_domain', 'physical_material',
                            'measurement_operator', 'original_pixel_support', 'independent_mask')
SAFE_MASK_CATEGORIES = frozenset(('actual_body_only_receipt', 'body_and_graphics_only'))


def _read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def _rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _ref(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=file_sha256(path))


class EvidenceResolver:
    """Relocations preserve historical executed bytes, never substitute new code."""
    def __init__(self, relocations=()):
        self.seen = set()
        self.relocations = {}
        self.used_relocations = {}
        for item in relocations:
            source = item['original']
            key = (str(Path(source.get('file', source.get('path'))).resolve()).lower(), source['sha256'])
            dest = item['archive']
            if dest['sha256'] != source['sha256']:
                raise ValueError('Historical dependency relocation changes bytes')
            self.relocations[key] = dest

    def __call__(self, ref, base=None):
        path = Path(ref.get('file', ref.get('path', '')))
        if not path.is_absolute():
            path = Path(base or '.') / path
        key = (str(path.resolve()).lower(), ref.get('sha256'))
        selected = ref
        if key in self.relocations:
            selected = self.relocations[key]
            self.used_relocations[key] = dict(original=ref, archive=selected)
            path = Path(selected.get('file', selected.get('path')))
        cache = (str(path.resolve()).lower(), selected.get('sha256'))
        if cache not in self.seen:
            if not cache[1] or file_sha256(path) != cache[1]:
                raise ValueError('Changed qualification dependency: ' + str(path))
            self.seen.add(cache)
        return path.resolve()

    def resolved_ref(self, ref, base=None):
        # The resolver authenticates each immutable dependency once.
        path = self(ref, base)
        return dict(file=str(path), sha256=ref['sha256'])


def select_membership(previous_active, qualified, *, alias=False, previous_inactive=False,
                      explicit_requalification=False, wrong_role=False):
    """An unresolved source is pending, an alias is not an independent reading."""
    if alias:
        return False, 'exact_actual_readout_alias'
    if wrong_role:
        return False, 'wrong_role'
    if not qualified:
        return False, 'pending_qualification'
    if previous_inactive and not explicit_requalification:
        return False, 'previous_exclusion_preserved'
    return bool(previous_active or explicit_requalification), 'active_qualified'


def select_group_statistics(group, retained_rows, load_whitener):
    """Use only declared parent supported Cov; missing statistics are explicit."""
    if not retained_rows:
        return None, None, dict(method='whole_group_retired', new_rank=0)
    try:
        if group.get('statistic_kind') != 'constraint_only_unknown_noise':
            # Check unchanged dependencies too. No stale W becomes likelihood.
            load_whitener(group['W'])
        if retained_rows != group['rows'] and group.get('statistic_kind') not in ('constraint_only_unknown_noise',):
            declared = (group.get('supported_covariance_declared') is True or
                        'supported whitening' in group.get('original_numerical_scheme', ''))
            if not declared:
                raise ValueError('Parent W has no declared supported covariance interpretation')
        return marginalize_group(group, retained_rows, load_whitener)
    except (OSError, ValueError, KeyError) as exc:
        # Engineering recovery may use a qualified source without a likelihood.
        result = {**group, 'rows': list(retained_rows), 'rank': 0, 'W': [],
                  'statistic_kind': 'constraint_only_unknown_noise',
                  'statistical_status': 'statistical_unavailable',
                  'parent_statistical_declaration': copy.deepcopy(group)}
        return result, None, dict(method='statistical_unavailable_preserve_parent_declaration',
            reason=str(exc), new_rank=0, precision_invented=False, W_columns_sliced=False)


def _verified_frame(record, item, resolver, cache, source_sha):
    ref = record['actual_current_RGB_evidence']
    path = resolver(ref)
    key = str(path)
    if key not in cache:
        frame = _read(path)
        out = frame.get('gate_output', frame)
        members = out.get('physical_membership_rows', out.get('members'))
        if not isinstance(members, list):
            raise ValueError('Actual RGB review has no physical/operator member records')
        cache[key] = (frame, {r['source_row_id']: r for r in members})
        if len(cache) > 4:
            cache.popitem(last=False)
    frame, members = cache[key]
    detail = members.get(record['source_row_id'])
    if not detail or object_sha256(detail) != record['canonical_review_record_sha256']:
        raise ValueError('Per-ID actual RGB review hash differs')
    if any(detail.get(k) != 'pass' for k in ('status', 'physical_role_status', 'operator_status')):
        raise ValueError('Summary pass is not a physical and frozen-operator pass')
    native = frame.get('native_receipt', {})
    if (native.get('source_video_sha256') != source_sha or native.get('time_base') != '1/12800'
            or int(native.get('native_pts', -1)) != int(record['native_pts'])
            or native.get('native_RGB_resampled') is not False):
        raise ValueError('Review is not bound to the official original native frame')
    out = frame.get('gate_output', frame)
    checker = out.get('frozen_original_operator_checker')
    if not checker:
        raise ValueError('Actual RGB pass lacks frozen original operator checker')
    evidence = [resolver.resolved_ref(ref), resolver.resolved_ref(checker)]
    if not detail.get('evidence_refs'):
        raise ValueError('Per-ID actual material pass has no original profile support')
    for r in detail['evidence_refs'] + record.get('evidence_refs', []):
        evidence.append(resolver.resolved_ref(r, path.parent))
    for k in ('source_frame_proof', 'current_mask'):
        if not frame.get(k):
            raise ValueError('Frame qualification lacks ' + k)
        evidence.append(resolver.resolved_ref(frame[k], path.parent))
    return evidence


def compile_active_qualified_scope(parent_packet, primary_manifest, mask_provenance,
                                   programme_scope, output, *, dependency_relocations=(), prepared_index_ref=None):
    """Consume actual closed exact-ID evidence without claiming full audit closure.

    Already inactive sources remain excluded. Later callers may pass a previous
    generated packet: pending IDs can become active only through their new real
    closure while inherited historical exclusions remain excluded. Non-recalled
    members retain their existing source admission and evidence declarations.
    """
    parent_path = Path(parent_packet).resolve()
    primary_path = Path(primary_manifest).resolve()
    mask_path = Path(mask_provenance).resolve()
    programme_path = Path(programme_scope).resolve()
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('Immutable qualified generation output exists')
    resolver = EvidenceResolver(dependency_relocations)
    packet = _read(parent_path)
    primary = _read(primary_path)
    masks = _read(mask_path)
    programme = _read(programme_path)
    if primary.get('schema') != 'rinkfit.actual_board_source_physical_requalification_run.v1':
        raise ValueError('Not a real physical and frozen-operator review run')
    if not primary.get('closed_chunks'):
        raise ValueError('No genuinely closed exact-ID review chunks')
    if masks.get('schema') != 'rinkfit.actual_source_legacy_mask_dependency_audit.v1':
        raise ValueError('No independent source mask provenance audit')
    scope_ref = primary['source_scope']
    scope_path = resolver(scope_ref)
    scope = _read(scope_path)
    if masks['source_scope']['sha256'] != scope_ref['sha256']:
        raise ValueError('Mask and physical qualification use different raw scope')
    resolver(masks['source_scope'])
    clock = programme['source']
    shots = programme['shots']
    if (clock['sha256'] != SOURCE_SHA or clock['time_base'] != '1/12800'
        or int(clock['native_pts_step']) != 256 or len(shots) != 9
        or int(programme['included_frame_count']) != 8434
        or shots[0]['first_pts'] != 256256 or shots[-1]['last_pts'] != 2415104
        or [s['id'] for s in shots] != packet['shot_ids']):
        raise ValueError('Qualified scope must preserve the actual complete nine-shot programme')
    recipe_path = resolver(primary['gate_recipe'])
    recipe = _read(recipe_path)
    for ref in recipe.get('code_refs', []):
        resolver(ref, recipe_path.parent)
    role_recipe_path = resolver(recipe['physical_role_recipe'])
    role_recipe = _read(role_recipe_path)
    for ref in role_recipe.get('frozen_module_refs', []):
        resolver(ref, role_recipe_path.parent)
    if not _read(resolver(role_recipe['counterexample_QA_ref'])).get('passed'):
        raise ValueError('Actual material role counterexample QA failed')
    arrays = {k: {**d, 'file': str(resolver(d, parent_path.parent))} for k, d in packet['arrays'].items()}
    def arr(k):
        d = arrays[k]
        return np.memmap(d['file'], dtype=d['dtype'], mode='r',
                         offset=int(d.get('offset_bytes', 0)), shape=tuple(d['shape']))
    oldad, roles, uv, pts, material, normals = (arr(k) for k in ('admission', 'fit_role', 'uv', 'native_pts', 'material_index', 'profile_normal'))
    n = int(packet['row_count'])
    ids, datasets = packet['source_row_ids'], packet['source_dataset_ids']
    if len(ids) != n or len(datasets) != n:
        raise ValueError('Raw source registry length changed')
    out.mkdir(parents=True)
    if prepared_index_ref:
        prepared_path = resolver(prepared_index_ref)
        prepared = _read(prepared_path)
        if (prepared.get('schema') != 'rinkfit.verified_partial_qualification_index.v1'
            or prepared['primary_audit']['sha256'] != file_sha256(primary_path)
            or prepared['mask_provenance']['sha256'] != file_sha256(mask_path)
            or prepared['source_scope']['sha256'] != scope_ref['sha256']):
            raise ValueError('Prepared exact source index belongs to different evidence')
        index_path = resolver(prepared['database'])
        db = sqlite3.connect('file:' + index_path.as_posix() + '?mode=ro', uri=True)
        closed_pts = set(map(int, prepared['closed_native_pts']))
        mask_frames = {int(r['native_pts']):r for r in _rows(resolver(masks['files']['frame_mask_provenance.jsonl.gz']))}
    else:
        index_path = out / 'qualification.sqlite3'
        db = sqlite3.connect(index_path)
        db.execute('CREATE TABLE rows(sid TEXT, pts INTEGER, record TEXT, inventory TEXT, mask TEXT, PRIMARY KEY(sid,pts)) WITHOUT ROWID')
        closed_pts = set()
        closed_raw_rows = 0
        for ref in primary['closed_chunks']:
            chunk_path = resolver(ref)
            chunk = _read(chunk_path)
            if (chunk.get('all_source_rows_accounted') is not True
                or chunk['source_scope']['sha256'] != scope_ref['sha256']
                or chunk['gate_recipe']['sha256'] != primary['gate_recipe']['sha256']):
                raise ValueError('Unclosed or redirected physical source chunk')
            local_pts = set(map(int, chunk['native_pts']))
            if closed_pts & local_pts or len(local_pts) != int(chunk['exposures']):
                raise ValueError('Duplicate native exposure across closed workers')
            closed_pts |= local_pts
            count = 0
            for r in _rows(resolver(chunk['files']['physical_membership_rows'], chunk_path.parent)):
                if int(r['native_pts']) not in local_pts:
                    raise ValueError('Source row outside genuinely closed exposure')
                db.execute('INSERT INTO rows(sid,pts,record) VALUES(?,?,?)',
                           (r['source_row_id'], int(r['native_pts']), json.dumps(r, separators=(',', ':'))))
                count += 1
            if count != int(chunk['source_rows']):
                raise ValueError('Closed source row accounting differs')
            closed_raw_rows += count
        if len(closed_pts) != int(primary['completed_exposures']) or closed_raw_rows != int(primary['raw_original_rows']):
            raise ValueError('Actual closed partial audit disagrees with its chunk membership')
        inventoried = 0
        for item in _rows(resolver(scope['source_row_inventory'])):
            key = (item['source_row_id'], int(item['native_pts']))
            found = db.execute('SELECT record FROM rows WHERE sid=? AND pts=?', key).fetchone()
            if found:
                r = json.loads(found[0])
                if (r['source_current_uv'] != item['current_uv']
                    or r['current_measurement_sha256'] != item['current_measurement_sha256']
                    or r['source_record_sha256'] != item['original_source_record_sha256']
                    or r.get('actual_readout_alias_of') != item.get('exact_alias_of')):
                    raise ValueError('Closed review redirects immutable raw source identity/mean/operator')
                db.execute('UPDATE rows SET inventory=? WHERE sid=? AND pts=?', (json.dumps(item, separators=(',', ':')), *key))
                inventoried += 1
        if inventoried != closed_raw_rows:
            raise ValueError('Closed review missing from actual original raw inventory')
        mask_frames = {}
        for frame in _rows(resolver(masks['files']['frame_mask_provenance.jsonl.gz'])):
            p = int(frame['native_pts'])
            if p in mask_frames:
                raise ValueError('Repeated independent frame mask provenance')
            mask_frames[p] = frame
        for r in _rows(resolver(masks['files']['source_row_mask_dependency.jsonl.gz'])):
            db.execute('UPDATE rows SET mask=? WHERE sid=? AND pts=?',
                       (json.dumps(r, separators=(',', ':')), r['source_row_id'], int(r['native_pts'])))
        db.commit()
        write_json(out/'prepared_index.json', dict(schema='rinkfit.verified_partial_qualification_index.v1',
            primary_audit=_ref(primary_path), mask_provenance=_ref(mask_path), source_scope=scope_ref,
            database=_ref(index_path), closed_native_pts=sorted(closed_pts)))
    ad = np.asarray(oldad).copy()
    active_before = (oldad == 1) & (roles == 1)
    previous_pending = set()
    prior = packet.get('qualified_scope_generation')
    if prior:
        manifest = _read(resolver(prior, parent_path.parent))
        for r in _rows(resolver(manifest['membership_decisions'])):
            if r['new_membership_status'] == 'pending_qualification' and r['previous_active']:
                previous_pending.add(int(r['global_row']))
    decisions_path = out / 'membership.jsonl.gz'
    native_path = out / 'qualified_native.jsonl'
    qualified_path = out / 'qualified.jsonl.gz'
    counts = collections.Counter()
    cache = collections.OrderedDict()
    board_active = []
    with gzip.open(decisions_path, 'wt', encoding='utf8', newline='\n') as decisions, native_path.open('w', encoding='utf8', newline='\n') as native, gzip.open(qualified_path, 'wt', encoding='utf8', newline='\n') as qualified:
        for i, sid in enumerate(ids):
            if not str(sid).startswith(FAMILY_PREFIXES):
                continue
            key = (sid, int(pts[i]))
            found = db.execute('SELECT record,inventory,mask FROM rows WHERE sid=? AND pts=?', key).fetchone()
            qualified_now = False
            record = item = mask = None
            reason = 'no_closed_original_native_review'
            evidence = []
            if found:
                record, item = json.loads(found[0]), json.loads(found[1])
                mask = json.loads(found[2]) if found[2] else None
                expected = object_sha256(dict(native_pts=int(pts[i]), current_uv=uv[i].tolist(), source_row_id=sid, measurement_operator=item['operator']))
                exact_version = (np.array_equal(uv[i], item['current_uv'])
                    and int(material[i]) == int(item['joint_material_index'])
                    and np.array_equal(normals[i], item['normal'])
                    and expected == record['current_measurement_sha256'])
                if not exact_version:
                    # A historical SID may have been reused by another journal.
                    # Its pass/wrong-role verdict cannot transfer to these UV.
                    record = None
                    found = None
                    reason = 'pending_exact_mean_operator_source_version_mismatch'
                if exact_version:
                    mask_frame = mask_frames.get(int(pts[i]))
                    mask_pass = (mask and mask['category'] in SAFE_MASK_CATEGORIES and mask_frame
                                 and mask_frame['category'] == mask['category']
                                 and mask['current_measurement_sha256'] == expected)
                    qualified_now = bool(mask_pass and all(record.get(k) == 'pass' for k in ('status', 'physical_role_status', 'operator_status')))
                    reason = 'independent_mask_pending' if not mask_pass else 'actual_physical_and_frozen_operator_' + record['status']
                    if qualified_now:
                        try:
                            evidence = _verified_frame(record, item, resolver, cache, SOURCE_SHA)
                            for k in ('source_proof_ref', 'original_mask_ref', 'body_only_receipt_ref'):
                                if mask_frame.get(k):
                                    evidence.append(resolver.resolved_ref(mask_frame[k]))
                        except (OSError, ValueError, KeyError) as exc:
                            # A summary pass cannot override an unclosed exact
                            # frame/member proof. Keep this source pending.
                            qualified_now = False
                            reason = 'pending_actual_evidence_verification: ' + str(exc)
            previous_active = bool(active_before[i] or i in previous_pending)
            selected, status = select_membership(previous_active, qualified_now,
                alias=bool(record and record.get('actual_readout_alias_of')),
                previous_inactive=not previous_active,
                wrong_role=bool(record and record.get('physical_role_status') == 'wrong_role'))
            ad[i] = int(selected and roles[i] == 1)
            if selected and roles[i] != 1:
                status = 'previous_exclusion_preserved'
            counts[status] += 1
            decision = dict(global_row=i, source_row_id=sid, source_dataset_id=datasets[i], native_pts=int(pts[i]),
                previous_active=previous_active, previous_admission=int(oldad[i]), new_membership_status=status,
                current_admission=int(ad[i]), physical_role_status=record.get('physical_role_status', 'unreviewed') if record else 'unreviewed',
                operator_status=record.get('operator_status', 'unreviewed') if record else 'unreviewed',
                reason=reason, statistical_status='separately_recorded_by_group')
            decisions.write(json.dumps(decision, ensure_ascii=False, separators=(',', ':')) + '\n')
            if ad[i] == 1:
                board_active.append(i)
                certrow = dict(global_row=i, source_row_id=sid, source_dataset_id=datasets[i], native_pts=int(pts[i]),
                    current_uv=uv[i].tolist(), material_index=int(material[i]), original_profile_normal=normals[i].tolist(), status='pass', operator_status='pass',
                    measurement_operator=item['operator'], current_measurement_sha256=expected,
                    qualification_dimensions={k:'pass' for k in QUALIFICATION_DIMENSIONS},
                    source_video_sha256=SOURCE_SHA, time_base='1/12800', coordinate_domain=DOMAIN,
                    actual_readout_alias_of=None, evidence_refs=evidence,
                    actual_review_ref=record['actual_current_RGB_evidence'],
                    canonical_review_record_sha256=record['canonical_review_record_sha256'],
                    independent_mask_category=mask['category'])
                line = json.dumps(certrow, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
                native.write(line)
                qualified.write(line)
    db.close()
    active = (ad == 1) & (roles == 1)
    admission_path = out / 'admission.bin'
    ad.tofile(admission_path)
    arrays['admission'] = {**arrays['admission'], **_ref(admission_path), 'offset_bytes':0}
    partition = np.zeros(n, np.uint8)
    independent = [int(i) for i in packet['independent_rows'] if active[int(i)]]
    partition[independent] += 1
    groups, proofs, pending = [], [], []
    bundle = out / 'selected_W.f64'
    rank = 2 * len(independent)
    with bundle.open('wb') as stream:
        for old in packet['groups']:
            retained = [int(i) for i in old['rows'] if active[int(i)]]
            def load(spec):
                path = resolver(spec, parent_path.parent)
                return np.memmap(path, dtype=spec['dtype'], mode='r', offset=int(spec.get('offset_bytes',0)), shape=tuple(spec['shape']))
            group, W, proof = select_group_statistics(old, retained, load)
            proofs.append(dict(group_id=old['id'], **proof))
            if group is None:
                continue
            if group.get('statistic_kind') == 'constraint_only_unknown_noise' and not group.get('unknown_noise_source_refs'):
                group['unknown_noise_source_refs'] = [{**_ref(parent_path), 'role':'immutable_parent_raw_source_ledger_not_source_qualification'}]
            if W is not None:
                offset = stream.tell()
                W.astype('<f8').tofile(stream)
                spec = dict(file=str(bundle), sha256='pending', dtype='<f8', shape=list(W.shape), offset_bytes=offset)
                group['W'] = spec
                pending.append(spec)
            elif group['W'] != []:
                group['W'] = {**group['W'], 'file':str(resolver(group['W'], parent_path.parent))}
            groups.append(group)
            rank += int(group['rank'])
            partition[retained] += 1
    for spec in pending:
        spec['sha256'] = file_sha256(bundle)
    if not np.array_equal(partition, active.astype(np.uint8)):
        raise ValueError('Generated statistical groups do not partition exact active sources')
    write_json(out / 'statistics.json', proofs)
    indices = np.asarray(board_active, dtype='<i8')
    indices.tofile(out / 'qualified_rows.i64')
    # The historical recipe is archived with the exact executed module bytes.
    frozen_recipe = copy.deepcopy(role_recipe)
    frozen_recipe['frozen_module_refs'] = [resolver.resolved_ref(r, role_recipe_path.parent) for r in role_recipe.get('frozen_module_refs', [])]
    frozen_recipe['counterexample_QA_ref'] = resolver.resolved_ref(role_recipe['counterexample_QA_ref'], role_recipe_path.parent)
    frozen_recipe['historical_executed_recipe_ref'] = _ref(role_recipe_path)
    write_json(out / 'executed_recipe.json', frozen_recipe)
    manifest = dict(schema=SCHEMA, status='incremental_exact_ID_membership_closed', full_original_audit_complete=False,
        parent_packet=_ref(parent_path), primary_partial_audit=_ref(primary_path), source_scope=scope_ref,
        programme_scope=_ref(programme_path), mask_provenance=_ref(mask_path),
        raw_row_count=n, raw_source_registry_sha256=object_sha256(dict(source_row_ids=ids, source_dataset_ids=datasets)),
        raw_array_sha256={k:d['sha256'] for k,d in packet['arrays'].items() if k != 'admission'},
        closed_native_exposures=len(closed_pts), full_audit_native_exposures=int(scope['development_exposures']),
        programme_native_exposures=8434, membership_status_counts=dict(counts), active_rows=int(active.sum()),
        active_qualified_board_rows=len(indices), inherited_nonrecalled_active_rows=int(active.sum())-len(indices),
        inherited_nonrecalled_source_evidence_revalidated_here=False,
        membership_decisions=_ref(decisions_path), statistics=_ref(out/'statistics.json'),
        qualifier_database=_ref(index_path), executed_dependency_relocations=list(resolver.used_relocations.values()),
        native_complete_state_membership_bridge_required=True, current_model_auxiliary_domain_checks_required=True,
        source_and_statistical_qualification_are_separate=True, old_source_ledger_or_UV_overwritten=False,
        same_problem_AL_monotonicity_claimed=False, formal_statistics_accepted=False)
    write_json(out/'generation.json', manifest)
    cert = dict(schema=CERTIFICATE_SCHEMA, count=len(indices), role_and_operator_pass_count=len(indices),
        unresolved_or_wrong_rows_consumed=0, qualification_dimensions=list(QUALIFICATION_DIMENSIONS),
        source_video_sha256=SOURCE_SHA, time_base='1/12800', coordinate_domain=DOMAIN,
        recipe_ref=_ref(out/'executed_recipe.json'), source_scope_ref=_ref(out/'generation.json'),
        qualified_rows_ref=_ref(qualified_path), qualified_rows_native_ref=_ref(native_path),
        qualified_active_board_rows_ref={**_ref(out/'qualified_rows.i64'), 'dtype':'<i8', 'shape':[len(indices)]},
        active_board_rows_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        membership_array_sha256={k:arrays[k]['sha256'] for k in BOUND_ARRAYS},
        source_row_ids_sha256=hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',',':')).encode()).hexdigest(),
        source_dataset_ids_sha256=hashlib.sha256(json.dumps(datasets, ensure_ascii=False, separators=(',',':')).encode()).hexdigest(),
        exact_ID_scope_complete=True, whole_original_scope_qualified=False,
        producer_module_is_historical_executed_archive=True,
        consumer_module_ref=_ref(Path(__file__)))
    from .measurement_operator import measurement_operator_binding
    cert['measurement_operator_binding'] = measurement_operator_binding({**packet, 'arrays': arrays})
    write_json(out/'certificate.json', cert)
    original_rank = 2*sum(i<int(packet['new_source_start']) for i in independent) + sum(int(g['rank']) for g in groups if all(int(i)<int(packet['new_source_start']) for i in g['rows']))
    result = {**packet, 'arrays':arrays, 'groups':groups, 'independent_rows':independent,
        'active_rows':int(active.sum()), 'joint_statistics_rank':rank, 'current_original_W_rank':original_rank,
        'new_source_statistics_rank':rank-original_rank, 'recapture_rows':np.flatnonzero(active).tolist(),
        'board_material_source_qualification':_ref(out/'certificate.json'),
        'qualified_scope_generation':_ref(out/'generation.json'),
        'source_membership_revision':dict(parent=_ref(parent_path), authority=_ref(out/'generation.json'),
            raw_source_rows_removed=False, requires_native_complete_state_membership_bridge=True,
            same_problem_AL_monotonicity_claimed=False),
        'constraint_only_unknown_noise_rows':sum(len(g['rows']) for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise'),
        'constraint_only_unknown_noise_groups':sum(g.get('statistic_kind')=='constraint_only_unknown_noise' for g in groups),
        'formal_new_source_statistical_optimum_claimed':False,
        'source_placeholder_station_half_is_not_solver_ready':True,
        'all_new_source_nuisance_requires_fresh_physical_capture':True}
    write_packet(out/'packet.json', result)
    from .board_source_qualification_guard import assert_board_material_source_qualification
    assert_board_material_source_qualification(result, out/'packet.json')
    report = dict(schema='rinkfit.incremental_qualified_scope_compilation.v1', packet=_ref(out/'packet.json'),
        generation=_ref(out/'generation.json'), certificate=_ref(out/'certificate.json'),
        raw_rows=n, previous_active_rows=int(active_before.sum()), current_active_rows=int(active.sum()),
        qualified_board_rows=len(indices), inherited_nonrecalled_active_rows=int(active.sum())-len(indices),
        membership_status_counts=dict(counts), source_guard_passed=True,
        numerical_membership_bridge_pending=True, model_domain_checks_pending=True,
        full_programme_native_frames=8434, programme_shots_preserved=packet['shot_ids'],
        raw_array_and_registry_unchanged=True, new_statistical_precision_invented=False)
    write_json(out/'report.json', report)
    return report
