"""Audit literal original/additional curve coefficients without a fit or J."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
ORIGINAL = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl')
CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review82_systematic_20261009.json'


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def original_family(row):
    identity = row['id']
    if row.get('point_correspondence') or 'corner' in identity or re.match(r'ad[0-9]+:', identity):
        return 'advertisement_point'
    if identity in ('far_top', 'far_contact', 'board_top', 'ice_board_contact') or 'board_top' in identity:
        return 'board_curve'
    if re.match(r'[FR][1-4]', identity): return 'curling_ring'
    if identity.lower().startswith('lane'): return 'Lane_curve'
    if identity.startswith('C0') or re.match(r'A[1-4]', identity): return 'hockey_circle'
    if identity in ('C', 'B1', 'B2') or 'goal' in identity: return 'hockey_straight'
    return 'other_absolute'


def run():
    options = json.loads(CONFIG.read_text(encoding='utf8'))['source_options']
    additional = Path(options['--additional-ice-source'])
    authority = Path(options['--direct-human-truth-authority'])
    humans = {row['id'] for row in json.loads(authority.read_text(encoding='utf8'))['records']}
    original_rows = pickle.loads(ORIGINAL.read_bytes())['programme_absolute_rows']
    packet = json.loads(additional.read_text(encoding='utf8'))
    records, groups = [], defaultdict(list)
    for family, source in (('original_absolute', original_rows), ('additional_native_ice', packet['rows'])):
        for index, item in enumerate(source):
            row = item['source_row'] if family == 'original_absolute' else item
            uv = np.asarray(row['uv'], float); n = len(uv)
            sigma = np.broadcast_to(np.asarray(row.get('sigma', row.get('sigma_px')), float), (n,))
            original = family == 'original_absolute'
            role = item['collection_role'] if original else row['split'].upper()
            pts = int(item['group_actual_pts']) if original else int(row.get('actual_pts', row['pts']))
            physical = row['id'] if original else json.dumps(row['curve'], sort_keys=True, separators=(',', ':'))
            record = dict(family=family+'/'+(original_family(row) if original else row['curve']['kind']),
                collection=role, row_index=index, source_id=row['id'], actual_pts=pts,
                source_pts=int(row['pts']), N=n, W=float(np.sum(1/sigma**2)),
                sigma_minmax=[float(sigma.min()), float(sigma.max())],
                within_row_exact_UV_duplicates=n-len(np.unique(uv, axis=0)),
                physical_curve_declaration=physical,
                source_sampling_metadata={key: value for key, value in row.items()
                    if any(word in key.lower() for word in ('interpol', 'resampl', 'derived', 'dense',
                        'correlation_query', 'sigma_semantics', 'sigma_basis', 'sourceuv_origin', 'extraction_method'))})
            if original:
                record.update(block=item['block'], group_order=item['group_order'], row_order=item['row_order'], origin=str(row.get('origin', '')))
            working = row.get('source_working_sigma_before_stroke_correlation_px')
            record['explicit_pre_correlation_working_sigma'] = working is not None
            if working is not None:
                working = np.broadcast_to(np.asarray(working, float), (n,)); multiplier = sigma/working
                record.update(explicit_sampling_multiplier_minmax=[float(multiplier.min()), float(multiplier.max())],
                    explicit_working_sigma_minmax=[float(working.min()), float(working.max())],
                    prior_sampling_matches_sqrt_N=bool(np.allclose(multiplier, np.sqrt(n), rtol=1e-12, atol=1e-12)))
            centre = not original and row['id'] in humans and row['split']=='fit'
            record['enabled_direct_manual_centre'] = centre
            if centre:
                extra = np.sqrt(n) if row.get('practical_manual_location_uncertainty_native_px', 0)>0 else 1.
                record.update(direct_manual_centre_extra_sampling_multiplier=float(extra),
                    direct_manual_centre_W=float(np.sum(1/(sigma*extra)**2)))
            records.append(record)
            if role=='FIT': groups[(family, record.get('block'), pts, physical)].append((record, uv))
    families = defaultdict(list)
    for record in records: families[(record['family'], record['collection'])].append(record)
    totals = [dict(family=key[0], collection=key[1], rows=len(rows), UV=sum(row['N'] for row in rows),
        sum_inv_sigma2=sum(row['W'] for row in rows), group_W_minmax=[min(row['W'] for row in rows), max(row['W'] for row in rows)],
        N_minmax=[min(row['N'] for row in rows), max(row['N'] for row in rows)]) for key, rows in sorted(families.items())]
    multiple = []
    for key, values in groups.items():
        if len(values)<2: continue
        uv = np.concatenate([value[1] for value in values])
        multiple.append(dict(group_key=key, row_indices=[value[0]['row_index'] for value in values],
            source_ids=[value[0]['source_id'] for value in values], N=len(uv),
            exact_UV_duplicates=len(uv)-len(np.unique(uv, axis=0)), W=sum(value[0]['W'] for value in values)))
    report = dict(schema='rinkspace.actual-curve-source-sampling-weight-audit.v1', read_only=True,
        input_config=ref(CONFIG), original_absolute_packet=ref(ORIGINAL), additional_ice_packet=ref(additional), direct_human_authority=ref(authority),
        precision_definition='W=sum 1/stored_sigma_i^2 per UV; identical coefficient for both x/y. This is objective weighting, not measured covariance.',
        families=totals, multiple_same_exposure_literal_curve_rows=multiple, records=records,
        direct_centre_correction='Only practical_manual_location_uncertainty_native_px>0 rows receive another sqrt(N). New explicit pre-correlated rows69..78 do not. Their sigma already contains sqrt(N); no second normalizer was applied there.',
        examples_231678464=[row for row in records if row['actual_pts']==231678464],
        limits=['Literal source IDs are a conservative duplicate inventory; cross-family production alias must be authenticated.',
            'No interpolation conclusion from collinearity alone; explicit source lineage required.',
            'Dense hard truth vertices do not count as soft FIT observations.',
            'No packet, source weight, source eligibility, guard or solver state changed.'], accuracy_accepted=False)
    path = BASE/'actual_curve_sampling_weight_audit_20261010.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(path=str(path), families=totals, duplicate_curve_groups=len(multiple),
        groups_with_exact_UV_duplicates=sum(bool(row['exact_UV_duplicates']) for row in multiple),
        additional_pre_correlated_rows=sum(row['explicit_pre_correlation_working_sigma'] for row in records),
        direct_manual_ice_W=sum(row.get('direct_manual_centre_W', 0.) for row in records)), ensure_ascii=False))


if __name__ == '__main__': run()
