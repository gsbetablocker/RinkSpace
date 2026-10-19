"""Gate new complete-material C0 sections against one shared RGB covariance.

The original unbound/inner-edge record and previous source generations stay
unchanged. This consumes the source-qualified v3 journal and its new typed
flank IDs; it never promotes the recalled source-ellipse estimator.
"""
from pathlib import Path
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .finite_conditional_gate import SourceVerifier

ESTIMATOR = 'current_C0_full_five_profile_same_material_frozen_pair_v3'
RECALL_MARKERS = ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json', 'SOURCE_IDENTITY_EVIDENCE_RECALL.json')


def _ref(path):
    p = Path(path).resolve()
    return dict(file=str(p), sha256=file_sha256(p))


def inspect_sources(project, run, identity):
    """Validate actual source provenance without RGB decoding or creating W."""
    identity = Path(identity).resolve()
    if any((identity / name).exists() for name in RECALL_MARKERS):
        raise ValueError('Typed paired source qualification has been recalled')
    v = SourceVerifier(project, run)
    v.verify(_ref(identity / 'manifest.json'))
    manifest = read_json(identity / 'manifest.json')
    if manifest['schema'] != 'rinkfit.current_C0_full_material_typed_shared_RGB_input.v1':
        raise ValueError('Unsupported actual paired source generation')
    original_manifest_file = v.verify(manifest['source_identity_manifest'])
    if any((original_manifest_file.parent / name).exists() for name in RECALL_MARKERS):
        raise ValueError('Original material-pair source qualification has been recalled')
    original_manifest = read_json(original_manifest_file)
    source_file = v.verify(original_manifest['source_journal'])
    source = {r['source_row_id']: r for r in read_jsonl(source_file)}
    qualification_file = v.verify(manifest['source_qualification_receipts'])
    qualifications = {r['source_row_id']: r for r in read_jsonl(qualification_file)}
    pairs = read_json(v.verify(manifest['approved_pairs']))
    factors = {r['source_row_id']: r for r in read_jsonl(v.verify(manifest['factor_rows']))}
    if len(pairs) != 12 or len(factors) != 24 or len(source) != 12:
        raise ValueError('Complete v3 source accounting changed')
    expected = v.contract.expected('ice.C0:outer', 'ice')
    if expected['operator'] != 'paint_center':
        raise ValueError('C0 must remain a finite-width circular stroke')
    verified = []
    for item in pairs:
        p = item['pair']; pair_id = p['source_row_id']; old_id = p['original_source_row_id']
        r = source[old_id]; q = qualifications[old_id]; pts = int(r['native_pts'])
        v.guard.assert_training_pixels([pts], shot_id=r['shot_id'])
        if not pair_id.startswith('current_C0_material_v3/') or pair_id.startswith('restored_C0_fullwide/'):
            raise ValueError('A new material measurement needs a distinct source version')
        if not (r['source_geometry_eligible'] and r['physical_identity_passed'] and q['source_geometry_eligible']):
            raise ValueError('Actual complete-material source identity is not qualified')
        if r['estimator_kind'] != ESTIMATOR or r['joint_material_index'] != 144 or r['target_id'] != 'ice.C0:outer':
            raise ValueError('Actual C0 source definition/binding changed')
        if r['measurement_contract_sha256'] != v.contract.digest:
            raise ValueError('Actual source definition no longer matches the shared fact contract')
        original_hash = object_sha256(dict(source_row_id=old_id, native_pts=pts,
            actual_4D_UV=r['actual_pair_4D_UV'], estimator=ESTIMATOR))
        if original_hash != r['current_4D_source_mean_sha256'] or original_hash != q['new_source_mean_sha256']:
            raise ValueError('Original qualified 4D source hash changed')
        if original_hash != item['original_4D_qualification_hash']:
            raise ValueError('Typed source is not the qualified new material mean')
        op = r['current_pair_operator']; rec = r['source_measurement_audit']['source_material_receipt']
        if op['source_ellipse_used_as_material_gate'] or op['masked_dependency_pixels'] != 0:
            raise ValueError('Old ellipse selection or masked material is not admissible')
        if r['float_RGB_base_replay_max_abs_px'] != 0 or r['source_measurement_audit']['status'] != 'current_native_same_paint_two_ordered_flanks_midline':
            raise ValueError('Actual material-pair mean is not the exact frozen replay')
        v.verify(op['pure_remeasure_entry']); v.verify(rec['actual_five_profiles_reference'])
        for name, digest in op['source_implementation_sha256'].items():
            v.verify(dict(file=str(Path(project) / 'calibration/rinkfit/src/rinkfit_joint' / name), sha256=digest))
        if rec['material_signal'] != 'dark' or rec['channel'] != 0 or rec['flank_polarity'] != [-1, 1]:
            raise ValueError('Wrong actual ordered gray-stroke transition')
        if not all(rec[k] for k in ('same_material_both_flanks', 'opposite_ordered_transitions', 'material_channel_frozen_for_replay')):
            raise ValueError('Same-material two-flank proof is incomplete')
        review = v.doc(r['source_identity_ref'])
        v.doc(review['existing_C0_identity'])
        for artifact in review['actual_windows']:
            v.verify(artifact)
        fr = r['current_native_frame_source']
        v.verify(dict(file=r['native_png'], sha256=r['native_png_sha256']))
        v.verify(dict(file=fr['mask_file'], sha256=fr['mask_sha256']))
        if p['actual_flank_uv'] != r['actual_pair_4D_UV'] or p['source_material_receipt'] != rec:
            raise ValueError('Typed source redirects actual flank mean/operator')
        if p['native_pts'] != pts or p['shot_id'] != r['shot_id'] or p['joint_material_index'] != 144:
            raise ValueError('Typed paired source clock/material differs')
        if p['width_scope_id'] != r['width_scope_id'] or p['allowed_world_side_assignments'] != [[-1, 1], [1, -1]]:
            raise ValueError('Width must be local and physical side association remains unoriented')
        public_hash = object_sha256(dict(native_pts=pts, source_row_id=pair_id,
            actual_flank_uv=p['actual_flank_uv'], measurement_operator='finite_width_paint_material_flank_pair'))
        if public_hash != p['source_mean_sha256'] or public_hash != item['actual_4D_source_mean_sha256']:
            raise ValueError('Typed paired 4D covariance mean hash changed')
        n = rec['normal']; seed = rec['original_navigation_seed_uv']
        if abs(sum(x*x for x in n)-1) > 1e-9 or p['source_normal'] != n:
            raise ValueError('Actual frozen source normal changed')
        rows = []
        for side, fid in enumerate(p['factor_row_ids']):
            f = factors[fid]
            if f['source_pair_id'] != pair_id or f['flank_side_index'] != side or f['source_normal'] != n:
                raise ValueError('Two-flank covariance column order changed')
            if f['current_uv'] != p['actual_flank_uv'][side] or f['source_uv'] != f['current_uv'] or f['source_material_receipt'] != rec:
                raise ValueError('Actual 4D source has been replaced by a midpoint/old inner edge')
            lo, hi = rec['frozen_flank_windows'][2][side]['normal_window_offsets']
            mid = .5*(lo+hi); query = [seed[j]+mid*n[j] for j in range(2)]
            alpha = sum((f['current_uv'][j]-query[j])*n[j] for j in range(2)); bounds = [lo-mid, hi-mid]
            if f['source_profile_query_uv'] != query or f['source_profile_alpha_bounds'] != bounds or abs(f['source_profile_current_alpha']-alpha) > 1e-12:
                raise ValueError('Source Query window was changed for model prediction')
            if not bounds[0]-.26 <= alpha <= bounds[1]+.26:
                raise ValueError('Actual source mean is outside its frozen source window')
            rows.append(f)
        verified.append((item, r, rows))
    return v, manifest, verified


def preflight(project, run, identity):
    v, manifest, verified = inspect_sources(project, run, identity)
    v.finish()
    return dict(source_qualified_pairs=len(verified), source_qualified_flank_rows=2*len(verified),
        actual_same_material_source_operator_verified=True, noise_statistics_or_solver_admission_added=False)


def promote(project, run, identity, statistics, output):
    """Admit only both flanks in one verified full-exposure conditional group."""
    identity = Path(identity).resolve(); output = Path(output).resolve()
    if output.exists():
        raise ValueError('New immutable paired conditional admission required')
    v, manifest, verified = inspect_sources(project, run, identity)
    statsfile = Path(statistics).resolve(); statistics_ref = _ref(statsfile)
    v.verify(statistics_ref); producer_ref = _ref(__file__); v.verify(producer_ref)
    registry = read_json(statsfile)
    lookup = {}
    for gid, group in registry.items():
        for rid in group['row_ids']:
            if rid in lookup:
                raise ValueError('One actual source row occurs in multiple covariance groups')
            lookup[rid] = gid
    source_pairs = []; solver_rows = []; decisions = []; receipts = []
    for item, source, factors in verified:
        p = item['pair']; pair_id = p['source_row_id']; ids = p['factor_row_ids']; pts = int(p['native_pts'])
        gids = {lookup.get(rid) for rid in ids}
        if len(gids) != 1 or None in gids:
            decisions.append(dict(source_row_id=pair_id, source_identity_eligible=True,
                solver_eligible=False, numeric_statistics_ready=False,
                statistical_objective_eligible=False,
                pending_reason='both_real_flanks_require_one_defined_whole_exposure_covariance'))
            continue
        gid = next(iter(gids)); g = registry[gid]; native = g['native_receipt']
        if g['statistic_kind'] != 'native_joint_readout' or not g['numeric_whitening_verified']:
            raise ValueError('Missing actual native conditional whitening')
        if g.get('actual_mean_replay_max_px', g.get('actual_base_replay_max_px', 1.)) > 1e-9:
            raise ValueError('Statistics belong to another source mean/operator')
        if not g['shared_full_native_RGB_domain'] or g['physical_total_error_calibrated']:
            raise ValueError('Conditional source model cannot masquerade as total physical accuracy')
        if int(native['native_pts']) != pts or native['shot_id'] != p['shot_id'] or not native['training_role_verified'] or native.get('native_RGB_resampled', False):
            raise ValueError('Current covariance clock/role differs from actual source')
        if native['decoded_native_RGB_sha256'] != source['current_native_frame_source']['RGB_sha256']:
            raise ValueError('Current covariance is not the same original native RGB exposure')
        if g['current_pair_measurements_sha256'][pair_id] != item['actual_4D_source_mean_sha256']:
            raise ValueError('Current covariance changes the paired 4D source mean')
        for key in ('W', 'frozen_operator', 'replay_arrays'):
            v.verify(g[key])
        for f in factors:
            rid = f['source_row_id']
            sh = object_sha256(dict(native_pts=pts, source_uv=f['source_uv'], source_row_id=rid))
            ch = object_sha256(dict(native_pts=pts, current_uv=f['current_uv'], source_row_id=rid,
                measurement_operator=f['measurement_operator_resolved']))
            if g['source_measurements_sha256'][rid] != sh or g['current_measurements_sha256'][rid] != ch:
                raise ValueError('Conditional W and true source/operator hashes differ')
            solver_rows.append(dict(f, source_identity_eligible=True, solver_eligible=True,
                numeric_statistics_ready=True, statistical_objective_eligible=True,
                statistical_group_id=gid, statistic_kind='native_joint_readout',
                statistics_use='conditional_noise_objective', conditional_model=g['conditional_model'],
                physical_total_error_calibrated=False, measurement_contract_sha256=v.contract.digest,
                exact_full_exposure_covariance_marginal_required=True))
        receipt = dict(schema='rinkfit.full_material_paired_conditional_admission.v1',
            source_row_id=pair_id, native_pts=pts, shot_id=p['shot_id'],
            original_actual_4D_qualification_hash=item['original_4D_qualification_hash'],
            current_4D_source_mean_sha256=item['actual_4D_source_mean_sha256'],
            source_qualification_manifest=_ref(identity/'manifest.json'),
            source_identity_ref=source['source_identity_ref'], statistical_group_id=gid,
            parent_statistics_sha256=object_sha256(g), actual_source_mean_or_operator_changed=False,
            raw_image_midpoint_consumed=False, global_uniform_width_imposed=False,
            exact_full_exposure_covariance_marginal_required=True,
            physical_total_error_calibrated=False, measurement_contract_sha256=v.contract.digest)
        receipts.append(receipt)
        source_pairs.append(dict(item, solver_eligible=True, numeric_statistics_ready=True,
            statistical_objective_eligible=True, statistical_group_id=gid,
            source_admission_receipt_sha256=object_sha256(receipt)))
        decisions.append(dict(source_row_id=pair_id, source_identity_eligible=True,
            solver_eligible=True, numeric_statistics_ready=True, statistical_objective_eligible=True,
            statistical_group_id=gid, current_4D_source_mean_sha256=item['actual_4D_source_mean_sha256']))
    v.finish(); output.mkdir(parents=True)
    write_json(output/'approved_pairs.json', source_pairs)
    write_jsonl(output/'solver_rows.jsonl.gz', solver_rows)
    write_jsonl(output/'factor_rows.jsonl.gz', solver_rows)
    write_jsonl(output/'promotion_decisions.jsonl.gz', decisions)
    write_jsonl(output/'receipts.jsonl.gz', receipts)
    report = dict(schema='rinkfit.full_material_paired_conditional_admission.v1',
        physically_approved_pairs=len(verified), conditional_approved_pairs=len(source_pairs),
        conditional_approved_flank_rows=len(solver_rows), source_identity=_ref(identity/'manifest.json'),
        statistics_registry=statistics_ref, producer=producer_ref,
        raw_midpoint_consumed=False, source_mean_or_flank_UV_changed=False,
        mixed_whole_exposure_covariance_consumed_once=True,
        physical_total_error_calibrated=False, measurement_contract_sha256=v.contract.digest)
    write_json(output/'manifest.json', report)
    return report
