"""Authenticate source-only UNKNOWN_DEPTH -> finite_board cap migration.

Require the actual normal startup manifest, keep the original complete state and cap packet intact,
and accepts only retained combined role children plus exact reviewed children.
The public role installer separately checks each original finite_mask=False
before installing an admitted row. No J/profile/normal reuse is authorized.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np

SCHEMA='rinkspace.authenticated-original-motion-role-correction-truth-proof.v1'
COMBINED='rinkspace.original-motion-multiple-block-far-board-role-admission.v1'
SECONDARY='rinkspace.original-secondary-motion-source-qualified-far-board-role-admission.v1'
PROGRAMME='rinkspace.original-programme-FIT-source-qualified-far-board-role-admission.v1'
LOW='rinkspace.original-low-FIT-source-qualified-far-board-role-admission.v1'
ROLE_KEY='original_motion_far_board_role_correction'
MANUAL_KEYS=('highest_direct_human_hard_truth','highest_user_confirmed_native_sources')

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked_reference(value):
    path=Path(value['path']).resolve()
    if sha(path)!=value['sha256']:raise ValueError('Typed-role proof reference bytes changed')
    return path
def target(manifest):
    return dict(parameter_count=int(manifest['full_parameters']),source_rows=int(manifest['source_rows']),
                source_contract_sha256=hashlib.sha256(canonical(manifest['source_contract']).encode('utf8')).hexdigest())
def role_target(manifest):
    role=manifest.get(ROLE_KEY)
    return None if role is None else dict(packet_path=str(Path(role['packet_path']).resolve()),
                                         packet_sha256=role['packet_sha256'],
                                         contract_sha256=hashlib.sha256(canonical(role).encode('utf8')).hexdigest())
def cap_identity(packet):
    result={}
    for family,record in packet['families'].items():
        checked_reference(record['authority']);ids=record['group_ids']
        if len(set(ids))!=len(ids):raise ValueError('Distinct original manual group IDs required')
        result[family]=dict(group_ids=ids,authority=record['authority'])
        for name in ('limits','uncertainties'):
            values=np.asarray(record[name],dtype=np.float64)
            if values.shape!=(len(ids),)or not np.isfinite(values).all()or np.any(values<0):raise ValueError('Original finite nonnegative caps required')
            result[family][name+'_float64_sha256']=hashlib.sha256(values.tobytes()).hexdigest()
        result[family]['zero_limits']=int(np.count_nonzero(np.asarray(record['limits'],float)==0))
    return result
def authenticate_state(path,q=None):
    path=Path(path).resolve();packet=read(path)
    if packet.get('schema')!='rinkspace.joint-source-truth-restoration-state.v1':raise ValueError('Original complete truth restoration packet required')
    arrays_path=checked_reference(packet['complete_state_reference']);manifest_path=checked_reference(packet['source_input_manifest_reference'])
    with np.load(arrays_path,allow_pickle=False)as arrays:
        value=arrays['q'].copy()
    if hashlib.sha256(value.tobytes()).hexdigest()!=packet['complete_q_sha256']:raise ValueError('Complete real seed identity changed')
    if q is not None and not np.array_equal(np.asarray(q,float),value):raise ValueError('Complete real seed q differs')
    original=read(manifest_path)
    if target(original)!=packet['source_target']:raise ValueError('Original mathematical source target differs')
    if packet.get('source_role_operator_target',role_target(original))!=role_target(original):raise ValueError('Original saved physical role target differs')
    caps=cap_identity(packet)
    return packet,original,caps

def append_role_contract(original,new_path,binding_factory):
    """Derive the one permitted actual top-level contract from frozen packets."""
    old_contract=original[ROLE_KEY];old_path=checked_reference(dict(path=old_contract['packet_path'],sha256=old_contract['packet_sha256']))
    old=read(old_path);new_path=Path(new_path).resolve();new=read(new_path)
    if old.get('schema')!=COMBINED or new.get('schema')!=COMBINED:raise ValueError('Retained combined physical-role packet required')
    retained=old['source_packets'];extra=new['source_packets'][len(retained):]
    if not extra or canonical(new['source_packets'][:len(retained)])!=canonical(retained):raise ValueError('All old role child references must remain in order')
    for item in retained:checked_reference(item)
    mutable={'authority','source_packets','original_unknown_motion_points_retyped','original_motion_material_role_count_by_block','planned_finite_point_count_by_block',
             'postcompaction_planned_finite_point_count_by_block','postcompaction_actual_runtime_source_role_count'}
    if any(canonical(new.get(k))!=canonical(v)for k,v in old.items()if k not in mutable):raise ValueError('An original role admission field changed')
    if not isinstance(new.get('authority'),str)or not new['authority'].strip():raise ValueError('Explicit source-only material authority required')
    records=[];child_refs=[]
    for item in extra:
        child_path=checked_reference(item);child=read(child_path)
        schema=child.get('schema')
        if (schema not in(SECONDARY,PROGRAMME,LOW) or child.get('new_source_scalar_rows')!=0
            or child.get('new_query_or_scalar_rows')!=0
            or child.get('original_UV_PTS_sigma_and_correspondence_unchanged')is not True
            or child.get('HELD_or_diagnostic_tracks_not_promoted_to_FIT')is not True
            or (schema==PROGRAMME and child.get('fixed_metric_depth_world_XYZ_or_plane_supplied')is not False)
            or (schema==LOW and child.get('source_material_qualification_no_world_XYZ_plane_or_depth_supplied')is not True)
            or child.get('fixed_metric_depth_world_XYZ_or_plane_supplied')is True):
            raise ValueError('Only actual RGB-qualified original admitted programme UNKNOWN_DEPTH roles may change')
        binding=binding_factory(child_path)
        for row in binding.records:
            sigma=np.asarray(row['source_sigma_px'],float)
            # The public programme admission binder authenticates these original
            # blocks against their exact source clock, FIT arrays and compaction
            # maps.  Cap migration must cover the same authenticated inventory.
            allowed=(('secondary',)if schema==SECONDARY else('low',)if schema==LOW
                else('main1','main7','secondary','low','main9','main4'))
            if (row['source_role_block']not in allowed or row['split']!='fit'
                or sigma.shape!=(2,)or not np.isfinite(sigma).all()or np.any(sigma<=0.)
                or(schema==SECONDARY and not np.array_equal(sigma,[2.,2.]))
                or(schema!=LOW and row.get('physical_world_point_or_measured_depth_supplied')is not False)
                or row.get('physical_world_point_or_measured_depth_supplied')is True):
                raise ValueError('Original source role, FIT sigma or non-metric physical type changed')
            records.append(deepcopy(row))
        child_refs.append(ref(child_path))
    old_by=old_contract['original_global_bundle_indices_by_block'];seen={(block,int(i))for block,ids in old_by.items()for i in ids}
    keys=[(row['source_role_block'],int(row['original_global_bundle_index']))for row in records]
    if len(set(keys))!=len(keys)or any(key in seen for key in keys):raise ValueError('Added roles must be distinct and formerly untyped, not existing roles')
    expected=deepcopy(old_contract);expected.update(packet_path=str(new_path),packet_sha256=sha(new_path),
        authenticated_component_packets=deepcopy(new['source_packets']),
        original_global_bundle_indices=old_contract['original_global_bundle_indices']+[int(r['original_global_bundle_index'])for r in records],
        original_unknown_motion_points_retyped=old_contract['original_unknown_motion_points_retyped']+len(records))
    for block in sorted({r['source_role_block']for r in records}):
        same=[r for r in records if r['source_role_block']==block]
        expected['original_global_bundle_indices_by_block'][block]=old_by.get(block,[])+[int(r['original_global_bundle_index'])for r in same]
        pairs={tuple(map(int,p))for p in old_contract['actual_source_pairs_by_block'].get(block,[])}
        pairs.update((int(r['source_pts']),int(r['target_pts']))for r in same)
        expected['actual_source_pairs_by_block'][block]=[list(p)for p in sorted(pairs)]
    counts={block:len(ids)for block,ids in expected['original_global_bundle_indices_by_block'].items()}
    if (new['original_unknown_motion_points_retyped']!=sum(counts.values())
        or ('original_motion_material_role_count_by_block'in new and new['original_motion_material_role_count_by_block']!=counts)):
        raise ValueError('Role count is not the exact admitted unique set')
    order=[b['block']for b in original['original_motion_input_compaction']['blocks']]
    planned_key='planned_finite_point_count_by_block'
    if planned_key in old:
        old_planned=old[planned_key]
        if len(order)!=len(old_planned):raise ValueError('Original programme block inventory differs')
        planned=[int(v)+sum(r['source_role_block']==name for r in records)for name,v in zip(order,old_planned)]
        if new.get(planned_key)!=planned:raise ValueError('Finite role inventory changes beyond exact admitted set')
    elif planned_key in new:raise ValueError('Role-only migration cannot introduce an unbound original inventory')
    # Role-only additions retain the existing compaction map.  Their active
    # representatives must therefore increase by the same exact admitted set;
    # introducing a newly typed removed duplicate requires source-revision
    # authentication instead of this narrower role-only migration.
    post_key='postcompaction_planned_finite_point_count_by_block'
    runtime_key='postcompaction_actual_runtime_source_role_count'
    if post_key in old:
        old_post=old[post_key]
        if len(old_post)!=len(order):raise ValueError('Original postcompaction block inventory differs')
        post=[int(v)+sum(r['source_role_block']==name for r in records)for name,v in zip(order,old_post)]
        if new.get(post_key)!=post:raise ValueError('Postcompaction finite inventory changes beyond exact admitted set')
    elif post_key in new:raise ValueError('Role-only migration cannot introduce an unbound compaction inventory')
    if runtime_key in old:
        if new.get(runtime_key)!=int(old[runtime_key])+len(records):raise ValueError('Runtime role count changes beyond exact admitted set')
    elif runtime_key in new:raise ValueError('Role-only migration cannot introduce an unbound runtime role count')
    inventory=[dict(block=r['source_role_block'],original_global_bundle_index=int(r['original_global_bundle_index']),
                    source_pts=int(r['source_pts']),target_pts=int(r['target_pts']),source_uv=r['source_uv'],target_uv=r['target_uv'],
                    source_sigma_px=r['source_sigma_px'],split=r['split'],qualified_record_sha256=hashlib.sha256(canonical(r).encode('utf8')).hexdigest())for r in records]
    return expected,inventory,child_refs

def prepare_proof(state_path,new_role_packet,binding_factory,implementation_path):
    packet,original,caps=authenticate_state(state_path)
    expected,inventory,children=append_role_contract(original,new_role_packet,binding_factory)
    return dict(schema=SCHEMA,original_truth_state=ref(state_path),complete_q_sha256=packet['complete_q_sha256'],
        original_source_target=packet['source_target'],original_source_role_target=role_target(original),
        original_role_packet=dict(path=original[ROLE_KEY]['packet_path'],sha256=original[ROLE_KEY]['packet_sha256']),
        corrected_role_packet=ref(new_role_packet),added_role_children=children,exact_added_point_inventory=inventory,
        expected_actual_role_contract=expected,original_manual_cap_identity=caps,
        existing_manual_group_count=sum(len(r['group_ids'])for r in packet['families'].values()),
        public_role_operator_implementation=ref(implementation_path),
        actual_new_manifest_required=True,original_state_and_caps_not_rewritten=True,
        cross_target_source_J_profiles_and_normals_not_authorized=True)

def authenticate_role_correction(proof_path,state_path,q,manifest,binding_factory,implementation_path):
    proof=read(proof_path)
    if proof.get('schema')!=SCHEMA:raise ValueError('Explicit typed-role truth proof required')
    if checked_reference(proof['original_truth_state'])!=Path(state_path).resolve():raise ValueError('Typed-role proof belongs to another complete checkpoint')
    if checked_reference(proof['public_role_operator_implementation'])!=Path(implementation_path).resolve():raise ValueError('Public finite-board operator identity differs')
    packet,original,caps=authenticate_state(state_path,q)
    if (proof['complete_q_sha256']!=packet['complete_q_sha256']or proof['original_source_target']!=packet['source_target']
        or proof['original_source_role_target']!=role_target(original)or proof['original_manual_cap_identity']!=caps
        or proof['existing_manual_group_count']!=sum(len(r['group_ids'])for r in packet['families'].values())
        or proof['original_role_packet']!=dict(path=original[ROLE_KEY]['packet_path'],sha256=original[ROLE_KEY]['packet_sha256'])):raise ValueError('Original seed, source role identity or exact cap bytes differ')
    corrected=checked_reference(proof['corrected_role_packet'])
    expected,inventory,children=append_role_contract(original,corrected,binding_factory)
    if canonical(expected)!=canonical(proof['expected_actual_role_contract'])or inventory!=proof['exact_added_point_inventory']or children!=proof['added_role_children']:raise ValueError('Exact reviewed point whitelist differs')
    if canonical(manifest.get(ROLE_KEY))!=canonical(expected):raise ValueError('Actual loaded physical roles differ from the exact correction')
    if role_target(manifest)==role_target(original):raise ValueError('Typed-role proof requires an actual operator target change')
    # No mathematical source change besides the separate role operator is allowed.
    if canonical(manifest['source_contract'])!=canonical(original['source_contract'])or target(manifest)!=target(original):raise ValueError('Source operators, masks, scales, slots or row count changed outside exact role correction')
    for key in MANUAL_KEYS+('original_source_role_retirement','static_motion_source_role_retirement'):
        if canonical(manifest.get(key))!=canonical(original.get(key)):raise ValueError('Original manual or source retirement contract changed: '+key)
    return dict(proof=ref(proof_path),original_truth_state=proof['original_truth_state'],
        original_source_target=packet['source_target'],actual_current_source_target=target(manifest),
        original_source_role_target=role_target(original),actual_current_source_role_target=role_target(manifest),
        exact_changed_original_point_count=len(inventory),source_rows_added_removed_or_reweighted=0,
        group_count=sum(len(r['group_ids'])for r in packet['families'].values()),
        all_caps_float64_sha256=caps,old_zero_limits_preserved_exactly=True,
        original_truth_packet_not_rewritten=True,old_source_J_profiles_and_normals_not_authorized=True)

def enforce_role_change_opt_in(original,manifest,proof_path):
    changed=canonical(original.get(ROLE_KEY))!=canonical(manifest.get(ROLE_KEY))
    if changed and proof_path is None:raise ValueError('Original source role operator changed; explicit typed-role proof required')
    if not changed and proof_path is not None:raise ValueError('Typed-role proof supplied without a source role operator change')
    return changed

import cv2
STATIC_FACTOR_KEY='additional_native_static_board_tracks'
STATIC_META='secondary_Ad22_23_absolute_source_candidate'
STATIC_FACTOR_SCHEMA='rinkspace.native-current-board-track-factor.v1'

def derive_static_append(original, new_path, receipt_path):
    factor = original[STATIC_FACTOR_KEY]
    old_path = checked_reference(dict(path=factor['packet_path'], sha256=factor['packet_sha256']))
    new_path = Path(new_path).resolve()
    old = read(old_path)
    new = read(new_path)
    receipt = read(receipt_path)
    if receipt['parent_source_packet'] != ref(old_path) or receipt['admitted_source_packet'] != ref(new_path):
        raise ValueError('Actual admitted static append receipt belongs to another packet')
    for name, expected in (('parent_all_fields_and_UV_sigma_roles_operator_arrays_exact', True), ('old_FIT_HELD_rows_or_existing_anchors_changed', False), ('old_HELD_promoted_to_FIT', False), ('original_frozen_templates_dimensions_and_material_parameters_unchanged', True), ('new_geometry_or_camera_DOF', 0), ('all_actual_RGB_file_BGR_native_clock_and_actual_exposure_certified', True), ('target_UV_are_real_mutual_detector_pixels_not_model_or_H_predictions', True)):
        if receipt.get(name) != expected:
            raise ValueError('Explicit reviewed static append semantics differ: ' + name)
    anchors = old['source_anchor_observations']
    extra = new['source_anchor_observations'][len(anchors):]
    if len(old['rows']) != 78 or len(anchors) != 17 or len(old['fit_row_order']) != 50:
        raise ValueError('Exact retained source batch required')
    if canonical(new['source_anchor_observations'][:len(anchors)]) != canonical(anchors):
        raise ValueError('An original source material anchor changed')
    if any((canonical(new.get(k)) != canonical(v) for k, v in old.items() if k not in ('source_anchor_observations', 'fit_row_order'))):
        raise ValueError('An original source field, row, template, UV or sigma changed')
    if set(new) != set(old) | {STATIC_META}:
        raise ValueError('Unreviewed static source metadata/field append')
    metadata = new[STATIC_META]
    if metadata['parent_source_packet'] != ref(old_path) or metadata.get('new_source_admitted_here') is not True or metadata.get('source_UV_model_independent') is not True or (metadata.get('source_review_pending') is not False):
        raise ValueError('Explicit actual RGB-reviewed material admission required')
    if len(extra) != 4:
        raise ValueError('Only the reviewed two FIT anchors and two development checks may be appended')
    spec = {(22, 'fit'): (229570304, 8), (22, 'holdout'): (229570304, 4), (23, 'fit'): (229584640, 4), (23, 'holdout'): (229584640, 3)}
    if {(int(r['advertisement_number']), r['split']): (int(r['actual_pts']), len(r['uv'])) for r in extra} != spec:
        raise ValueError('Reviewed material source ID/time/split inventory differs')
    parents = {r['id']: r for r in anchors}
    seen = set()
    new_seen = set()
    inventory = []
    for r in anchors:
        for uv in r['uv']:
            seen.add((int(r['actual_pts']), int(r['advertisement_number']), np.asarray(uv, np.float64).tobytes()))
    audits = {int(a['advertisement_number']): a for a in metadata['audit']}
    for row in extra:
        n = len(row['uv'])
        audit = audits[int(row['advertisement_number'])]
        if row['pts'] != row['native_pts'] or row['pts'] != row['actual_pts'] or row['time_base'] != '1/15360' or (row['source_qualification'].get('target_UV_are_actual_detector_pixels_not_H_prediction') is not True) or (row['source_qualification'].get('original_HELD_promoted_to_FIT') is not False) or (row['source_qualification'].get('model_projection_used') is not False) or (row['initialization'].get('derived_material_coordinates_are_not_world_observations') is not True) or (row['initialization'].get('new_material_geometry_or_parameters_added') is not False) or (row.get('world') is not None) or (len(row['original_source_point_ledger']) != n):
            raise ValueError('Real source UV/material semantics differ')
        raw = checked_reference(row['source_raw_frame'])
        bgr = cv2.imread(str(raw))
        if ref(raw) != audit['target_frame'] or bgr is None or hashlib.sha256(bgr.tobytes()).hexdigest() != audit['target_native_BGR_sha256']:
            raise ValueError('Actual raw target RGB differs')
        checked_reference(audit['source_frame'])
        checked_reference(audit['target_lossless_cache_manifest'])
        checked_reference(audit['review_image'])
        if audit['target_actual_exposure_pts'] != row['actual_pts'] or audit['model_camera_projection_used'] is not False or audit['original_FIT_source_pixel_binding_max_px'] != 0.0:
            raise ValueError('Actual source exposure or material-pixel binding differs')
        material = []
        for index, item in enumerate(row['original_source_point_ledger']):
            parent = parents[item['parent_anchor_id']]
            k = int(item['parent_anchor_point_index'])
            if parent['split'] != 'fit' or item['parent_role'] != 'fit' or item['parent_advertisement_number'] != parent['advertisement_number']:
                raise ValueError('An original HELD point cannot provide new FIT material')
            if not np.array_equal(item['original_source_UV'], parent['uv'][k]) or not np.array_equal(item['matched_source_SIFT_UV'], parent['uv'][k]) or (not np.array_equal(row['sigma_px'], parent['sigma_px'])) or (row['initialization']['template_id'] != parent['initialization']['template_id']) or (row['initialization']['q0_sha256'] != parent['initialization']['q0_sha256']):
                raise ValueError('Original FIT UV/sigma/template initializer changed')
            material.append(parent['initialization']['material_coordinates_normalized'][k])
            key = (int(row['actual_pts']), int(row['advertisement_number']), np.asarray(row['uv'][index], np.float64).tobytes())
            if key in seen or key in new_seen:
                raise ValueError('A target pixel is duplicated or a development check overlaps FIT')
            new_seen.add(key)
        if np.asarray(material, np.float64).tobytes() != np.asarray(row['initialization']['material_coordinates_normalized'], np.float64).tobytes():
            raise ValueError('Frozen normalized material point bytes changed')
        inventory.append(dict(id=row['id'], actual_pts=row['actual_pts'], split=row['split'], advertisement_number=row['advertisement_number'], points=n, complete_row_sha256=hashlib.sha256(canonical(row).encode('utf8')).hexdigest()))
    fit = [r for r in extra if r['split'] == 'fit']
    held = [r for r in extra if r['split'] == 'holdout']
    delta = 2 * sum((len(r['uv']) for r in fit))
    if delta != 24 or receipt['new_FIT_points'] != 12 or receipt['new_FIT_scalar_rows'] != 24 or (receipt['new_development_HELD_points'] != 7):
        raise ValueError('Only the reviewed 24 FIT scalar rows are admitted')
    if new['fit_row_order'] != old['fit_row_order'] + [r['id'] for r in fit]:
        raise ValueError('Original FIT row order changed')
    expected = deepcopy(factor)
    expected.update(packet_path=str(new_path), packet_sha256=sha(new_path), additional_source_anchor_fit_groups=factor['additional_source_anchor_fit_groups'] + len(fit), additional_source_anchor_fit_scalar_rows=factor['additional_source_anchor_fit_scalar_rows'] + delta, source_anchor_development_HELD_groups=factor['source_anchor_development_HELD_groups'] + len(held), scalar_rows=factor['scalar_rows'] + delta, frozen_print_FIT_groups=factor['frozen_print_FIT_groups'] + len(fit), fit_row_ids_in_evaluation_order=factor['fit_row_ids_in_evaluation_order'] + [r['id'] for r in fit])
    public_ref = receipt['public_factor_proof']
    public = read(checked_reference(public_ref))
    contract = deepcopy(public['public_factor_contract'])
    contract.update(packet_path=str(new_path), packet_sha256=sha(new_path))
    if canonical(contract) != canonical(expected):
        raise ValueError('Public new factor changes outside exact retained-prefix append')
    return (expected, inventory, dict(parent_packet=ref(old_path), admitted_packet=ref(new_path), admission_receipt=ref(receipt_path), public_factor_proof=public_ref, added_FIT_scalar_rows=delta))

def derive_static_source_contract(original, expected_factor, delta):
    old_factor = original[STATIC_FACTOR_KEY]
    hits = []

    def walk(value, path='source_contract'):
        if isinstance(value, list):
            return [walk(v, path + '/' + str(i)) for i, v in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if value.get('schema') == STATIC_FACTOR_SCHEMA:
            if canonical(value) != canonical(old_factor):
                raise ValueError('Original static source contract differs from actual factor')
            hits.append(path)
            return deepcopy(expected_factor)
        result = {k: walk(v, path + '/' + k) for k, v in value.items()}
        counters = {'rinkspace.original-source-canonical-complete-parameter-layout.v1': ('source_rows',), 'rinkspace.complete-source-plus-native-board-tracks.v1': ('total_rows',)}
        for key in counters.get(value.get('schema'), ()):
            if type(value.get(key)) is not int:
                raise ValueError('Exact source row counter required')
            result[key] = value[key] + delta
        return result
    expected = walk(original['source_contract'])
    if len(hits) != 1:
        raise ValueError('Exactly one original static track factor must be replaced')
    return expected

"""Authenticate exact qualified aliases without changing old source maps."""

COMPACTION_KEY='original_motion_input_compaction'
ALIAS_SCHEMA='rinkspace.source-qualified-finite-exact-duplicate-runtime-aliases.v1'
_COMPACTION_DYNAMIC_FIELDS=frozenset((
    'same_state_retained_local_r_max_difference','same_state_retained_local_J_max_difference',
    'same_state_retained_r_J_bitexact','retained_r_max_scale','retained_J_max_scale',
    'actual_retained_r_J_check_seconds','same_exposure_actual_r_max','same_exposure_coalesced_local_J_max'))

def compaction_identity(contract):
    """Retain all input/CSR identities; exclude only evaluated diagnostics."""
    result=deepcopy(contract)
    for row in result['blocks']:
        if row['removed_points']:
            if (row.get('same_state_retained_CSR_row_column_identity_bitexact')is not True
                or row.get('same_state_retained_r_J_passed_tight_compaction_numerical_tolerance')is not True):
                raise ValueError('Complete retained input/CSR value/J certification required')
            tolerance=row['compaction_numerical_tolerance']
            fraction=float(tolerance['J_max_error_fraction'])
            if not np.isfinite(fraction)or not 0.<=fraction<=1.:
                raise ValueError('Retained source numerical identity certification failed')
            tolerance.pop('J_max_error_fraction')
        for key in _COMPACTION_DYNAMIC_FIELDS:row.pop(key,None)
    return result

def derive_compaction_revision(original,role_path,inventory,fixture_path,implementation_path):
    """Bind the three reviewed history duplicates to their old map/arrays."""
    old=original[COMPACTION_KEY];expected=compaction_identity(old)
    fixture_path=Path(fixture_path).resolve();fixture=read(fixture_path)
    if (fixture.get('schema')!='rinkspace.original-main1-source-material-exact-three-history-duplicate-admitted-fixture.v1'
        or fixture.get('all_six_original_indices_have_final_actual_RGB_material_admission')is not True):
        raise ValueError('The exact finally admitted original main1 duplicate fixture is required')
    historical=read(checked_reference(fixture['original_pending_duplicate_fixture']))
    if historical.get('schema')!='rinkspace.original-main1-source-material-exact-three-history-duplicate-fixture.v1':
        raise ValueError('Original pending duplicate provenance differs')
    retained=deepcopy(historical['pairs'])
    for row in retained:row['qualified_material_still_requires_actual_RGB_review']=False
    if canonical(retained)!=canonical(fixture['pairs']):raise ValueError('An original source pair, pixel, sigma or RGB reference changed')
    main_path=checked_reference(old['main_candidate']);main=read(main_path)
    spec=next(r for r in main['classification']if r['block']=='main1')
    if (fixture['original_admission_cache']!=spec['original_admission_cache']
        or fixture['original_input_compaction_map']['path']!=spec['map']['path']
        or fixture['original_input_compaction_map']['sha256']!=spec['map']['sha256']):
        raise ValueError('Original duplicate map/admission bytes differ')
    archive_path=checked_reference(spec['original_admission_cache'])
    map_path=checked_reference(spec['map'])
    admission=fixture['source_RGB_role_admission'];admission_path=checked_reference(admission)
    new_packet=read(role_path)
    matches=[r for r in new_packet['source_packets']if Path(r['path']).resolve()==admission_path and r['sha256']==admission['sha256']]
    if len(matches)!=1:raise ValueError('Final source material admission is not an authenticated role child')
    checked_reference(matches[0])
    with np.load(map_path,allow_pickle=False)as archive:
        mapping=archive['old_point_to_compact_point'].copy()
        pairs=dict(zip(map(int,archive['removed_exact_duplicate_old_point_indices']),
                       map(int,archive['preserved_equal_source_point_for_each_removed'])))
    records={(r['block'],int(r['original_global_bundle_index'])):r for r in inventory}
    alias=[]
    for row in inventory:
        block=row['block'];i=int(row['original_global_bundle_index'])
        if block!='main1':continue
        if int(mapping[i])>=0:continue
        kept=pairs.get(i)
        if kept is None or ('main1',kept)not in records or int(mapping[kept])<0:
            raise ValueError('Every finite duplicate requires its exact newly qualified retained representative')
        peer=records[('main1',kept)]
        for key in ('source_pts','target_pts','source_uv','target_uv','source_sigma_px','split'):
            if canonical(row[key])!=canonical(peer[key]):raise ValueError('Qualified duplicate actual source signature differs')
        alias.append(dict(removed_original_global_bundle_index=i,
            duplicate_runtime_representative_original_global_bundle_index=kept,
            active_global_bundle_index=int(mapping[kept])))
    alias.sort(key=lambda r:r['removed_original_global_bundle_index'])
    wanted=sorted((int(r['removed_precompaction_bundle_index']),int(r['kept_precompaction_bundle_index']))for r in fixture['pairs'])
    if len(wanted)!=3 or [(r['removed_original_global_bundle_index'],r['duplicate_runtime_representative_original_global_bundle_index'])for r in alias]!=wanted:
        raise ValueError('Only the exact three reviewed original duplicate aliases may change')
    with np.load(archive_path,allow_pickle=False)as source:
        active=source['active_original_row_index']
        field_names=('source_uv_a','source_uv_b','endpoint_index_a','endpoint_index_b','source_original_sigma_xy')
        for drop,kept in wanted:
            a,b=int(active[drop]),int(active[kept])
            for key in field_names:
                left,right=np.asarray(source[key][a]),np.asarray(source[key][b])
                if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
                    raise ValueError('Original duplicate source pair/UV/sigma bytes are not identical')
    order=[r['block']for r in old['blocks']]
    count=int(new_packet['planned_finite_point_count_by_block'][order.index('main1')])
    alias_report=dict(schema=ALIAS_SCHEMA,source_role_binding_sha256=sha(role_path),pairs=alias,
        before_finite_points=count,after_finite_points=count-len(alias),both_source_certificates_preserved=True,
        exact_original_pair_UV_sigma_and_current_plane_value_J_checked=True,
        original_compaction_maps_unchanged=True,runtime_records_and_geometry_scatter_unique=True)
    row=next(r for r in expected['blocks']if r['block']=='main1')
    if row['all_removed_actual_finite_mask_false']is not True or 'qualified_finite_exact_duplicate_aliases'in row:
        raise ValueError('Original target must predate these exact finite aliases')
    row['all_removed_actual_finite_mask_false']=False;row['qualified_finite_exact_duplicate_aliases']=alias_report
    return dict(duplicate_source_fixture=ref(fixture_path),public_compaction_implementation=ref(implementation_path),
        exact_aliases=alias,expected_actual_compaction_identity=expected,
        old_maps_source_prefix_masks_sigma_clock_and_scalar_slots_preserved=True,
        source_qualification_records_retained=2*len(alias),runtime_physical_representatives=len(alias))

def authenticate_compaction_revision(component,original,manifest,role_path,inventory,implementation_path):
    if checked_reference(component['public_compaction_implementation'])!=Path(implementation_path).resolve():
        raise ValueError('Actual public input compaction implementation differs')
    actual=derive_compaction_revision(original,role_path,inventory,
        checked_reference(component['duplicate_source_fixture']),implementation_path)
    if canonical(component)!=canonical(actual):raise ValueError('Exact qualified source alias proof differs')
    if canonical(compaction_identity(manifest[COMPACTION_KEY]))!=canonical(actual['expected_actual_compaction_identity']):
        raise ValueError('Actual old source maps/slots/CSR identity or exact finite aliases differ')
    return actual

"""Authenticate exact reviewed source changes without expanding old caps.

Require the real new startup manifest, exact retained source fields and fresh
derivatives. The original complete checkpoint and cap packet stay intact.
"""
REVISION_SCHEMA='rinkspace.authenticated-reviewed-source-revision-truth-proof.v1'

def prepare_reviewed_source_revision(state_path,role_path,static_path,static_receipt,binding_factory,role_implementation,duplicate_fixture,compaction_implementation):
    packet,original,caps=authenticate_state(state_path)
    role,role_inventory,children=append_role_contract(original,role_path,binding_factory)
    if {row['block']for row in role_inventory}!={'main1','main4'}:raise ValueError('This reviewed revision only appends the qualified main1/main4 source roles')
    factor,anchor_inventory,static_component=derive_static_append(original,static_path,static_receipt)
    source_contract=derive_static_source_contract(original,factor,static_component['added_FIT_scalar_rows'])
    compaction=derive_compaction_revision(original,role_path,role_inventory,duplicate_fixture,compaction_implementation)
    return dict(schema=REVISION_SCHEMA,original_truth_state=ref(state_path),complete_q_sha256=packet['complete_q_sha256'],
        original_source_target=packet['source_target'],original_source_role_target=role_target(original),original_manual_cap_identity=caps,
        existing_manual_group_count=sum(len(r['group_ids'])for r in packet['families'].values()),
        corrected_role_packet=ref(role_path),added_role_children=children,exact_added_role_inventory=role_inventory,
        expected_actual_role_contract=role,public_role_operator_implementation=ref(role_implementation),
        static_track_append=static_component,exact_added_material_anchor_inventory=anchor_inventory,
        original_motion_compaction_revision=compaction,
        expected_actual_static_factor_contract=factor,
        expected_new_source_contract_sha256=hashlib.sha256(canonical(source_contract).encode('utf8')).hexdigest(),
        original_state_caps_manual_authorities_and_held_not_changed=True,
        real_new_startup_manifest_required=True,cross_target_source_J_profiles_and_normals_not_authorized=True)

def authenticate_reviewed_source_revision(proof_path,state_path,q,manifest,binding_factory,role_implementation,compaction_implementation):
    proof=read(proof_path)
    if proof.get('schema')!=REVISION_SCHEMA:raise ValueError('Explicit reviewed source revision truth proof required')
    if checked_reference(proof['original_truth_state'])!=Path(state_path).resolve():raise ValueError('Source revision proof belongs to another complete truth state')
    if checked_reference(proof['public_role_operator_implementation'])!=Path(role_implementation).resolve():raise ValueError('Public finite-board role operator identity differs')
    packet,original,caps=authenticate_state(state_path,q)
    if (proof['complete_q_sha256']!=packet['complete_q_sha256']or proof['original_source_target']!=packet['source_target']
        or proof['original_source_role_target']!=role_target(original)or proof['original_manual_cap_identity']!=caps
        or proof['existing_manual_group_count']!=sum(len(r['group_ids'])for r in packet['families'].values())):raise ValueError('Original complete q, source role or exact cap bytes differ')
    role_path=checked_reference(proof['corrected_role_packet'])
    role,role_inventory,children=append_role_contract(original,role_path,binding_factory)
    if {row['block']for row in role_inventory}!={'main1','main4'}:raise ValueError('Only reviewed main1/main4 original unknown-to-finite roles may change')
    component=proof['static_track_append']
    factor,anchor_inventory,observed_component=derive_static_append(original,checked_reference(component['admitted_packet']),checked_reference(component['admission_receipt']))
    if (canonical(role)!=canonical(proof['expected_actual_role_contract'])or role_inventory!=proof['exact_added_role_inventory']
        or children!=proof['added_role_children']or canonical(factor)!=canonical(proof['expected_actual_static_factor_contract'])
        or anchor_inventory!=proof['exact_added_material_anchor_inventory']or component!=observed_component):raise ValueError('Exact reviewed source whitelist/packet identity differs')
    expected=derive_static_source_contract(original,factor,component['added_FIT_scalar_rows'])
    compaction=authenticate_compaction_revision(proof['original_motion_compaction_revision'],original,manifest,role_path,role_inventory,compaction_implementation)
    if hashlib.sha256(canonical(expected).encode('utf8')).hexdigest()!=proof['expected_new_source_contract_sha256']:raise ValueError('Derived source revision identity differs')
    if (canonical(manifest[ROLE_KEY])!=canonical(role)or canonical(manifest[STATIC_FACTOR_KEY])!=canonical(factor)
        or canonical(manifest['source_contract'])!=canonical(expected)
        or manifest['full_parameters']!=original['full_parameters']or manifest['source_rows']!=original['source_rows']+component['added_FIT_scalar_rows']):raise ValueError('Actual loaded source changed outside exact role correction and reviewed material append')
    for key in MANUAL_KEYS+('original_source_role_retirement','static_motion_source_role_retirement'):
        if canonical(manifest.get(key))!=canonical(original.get(key)):raise ValueError('Original manual support/authority or source retirement changed: '+key)
    return dict(proof=ref(proof_path),original_truth_state=proof['original_truth_state'],
        original_source_target=packet['source_target'],actual_current_source_target=target(manifest),
        original_source_role_target=role_target(original),actual_current_source_role_target=role_target(manifest),
        exact_changed_original_role_points=len(role_inventory),exact_added_FIT_material_points=component['added_FIT_scalar_rows']//2,
        added_FIT_scalar_rows=component['added_FIT_scalar_rows'],new_development_checks_in_FIT=False,
        qualified_finite_duplicate_aliases=compaction['exact_aliases'],
        old_compaction_maps_source_prefix_masks_sigma_clock_and_scalar_slots_preserved=True,
        group_count=sum(len(r['group_ids'])for r in packet['families'].values()),all_caps_float64_sha256=caps,
        original_truth_packet_not_rewritten=True,all_old_zero_limits_preserved_exactly=True,
        old_source_J_profiles_and_normals_not_authorized=True)
