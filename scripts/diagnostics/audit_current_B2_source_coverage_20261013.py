"""Read exact v6 B2 source/index coverage without adding any fit pixels."""
from __future__ import annotations
from collections import Counter
import gc,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
J=WORK/'current_source_pool_joint_qualified_20261013_v1'
def ref(p):return dict(path=str(Path(p).resolve()),sha256=file_sha(p))
def main():
    path=J/'finite_fit_dataset_specific_inner_review_v6/fit_dataset.json'
    dataset_reference=dict(path=str(path),sha256='a835599e7d7a1a0bcf9dcd92e90fd4ad160b01e60e99975ec977d5f250f69fd8')
    authenticate_source_reference(dataset_reference);dataset=json.loads(path.read_text(encoding='utf8'))
    coverage_reference=dataset['original_native_coverage_reference'];authenticate_source_reference(coverage_reference)
    with np.load(coverage_reference['path'],allow_pickle=False)as z:
        actual=z['actual_pts'];parts=z['native_review_part_index'];metadata=json.loads(z['metadata_json'].item())
    expected={}
    for pi,source in enumerate(metadata['native_review_parts']):
        shot=Path(source['path']).name.removesuffix('_native_alias_review.npz').replace('_',':',1)
        expected.setdefault(shot,set()).update(map(int,actual[parts==pi]))
    stats={shot:dict(counts=Counter(),exposures={key:set()for key in ('source','fit','held','pending_support','pending_named')},sources=[])for shot in expected}
    for ci,cut in enumerate(dataset['cuts']):
        source=cut['source_pack'];authenticate_source_reference(source);authenticate_source_reference(cut['support_index'])
        rows,offsets=read_minimal_compact_source_headers(source['path'])
        with np.load(cut['support_index']['path'],allow_pickle=False)as saved:
            arrays={key:saved[key+'_point_indices']for key in ('fit','held','pending_support','pending_named')}
            if not np.array_equal(offsets,saved['row_offsets']):raise ValueError('Immutable source row boundaries differ')
        touched=set()
        for row in rows:
            if row['physical_material_id'].split(':',1)[0]!='B2':continue
            shot=row['shot_id'];native_actual=int(row['actual_pts']);s=stats[shot];touched.add(shot)
            ri=row['original_source_row_index'];a,b=map(int,offsets[ri:ri+2]);s['counts']['source_points']+=b-a;s['counts']['source_rows']+=1;s['exposures']['source'].add(native_actual)
            for key,selection in arrays.items():
                lo,hi=np.searchsorted(selection,[a,b]);n=int(hi-lo);s['counts'][key+'_points']+=n
                if n:s['exposures'][key].add(native_actual)
        for shot in touched:stats[shot]['sources'].append(dict(source_pack_reference=source,support_index_reference=cut['support_index']))
        print(json.dumps(dict(source_index=ci,rows=len(rows),B2_cuts=sorted(touched))),flush=True)
        del rows,offsets,arrays;gc.collect()
    records=[];total=Counter()
    for shot,s in stats.items():
        missing=sorted(expected[shot]-s['exposures']['fit'])
        unknown=sorted(expected[shot]-s['exposures']['source'])
        counts=dict(s['counts']);counts.update(expected_actual_exposures=len(expected[shot]),actual_exposures_with_finite_B2_fit=len(s['exposures']['fit']),actual_exposures_without_finite_B2_fit=len(missing),actual_exposures_without_B2_source_inventory=len(unknown))
        records.append(dict(shot_id=shot,counts=counts,actual_pts_without_finite_B2_fit=missing,actual_pts_without_any_B2_source_row=unknown,
            B2_source_evidence_actual_pts={k:sorted(v)for k,v in s['exposures'].items()},inputs=s['sources']))
        total.update(counts)
    out=J/'current_B2_source_coverage_audit_v1.json'
    if out.exists():raise FileExistsError('Keep prior current audit')
    result=dict(schema='rinkspace.current-frozen-B2-source-index-coverage-audit.v1',dataset_reference=dataset_reference,source_clock_reference=coverage_reference,counts=dict(total),cuts=records,
        no_pixels_or_sigma_changed=True,all_missing_and_named_source_inventory_retained=True,
        missing_B2_fit_is_not_a_claim_B2_is_visible_or_required=True,original_source_visibility_still_requires_native_RGB_review=True,
        source_inventory_is_not_independent_accuracy_acceptance=True,fit_admission_or_training_expanded=False,implementation_reference=ref(__file__))
    out.write_text(json.dumps(result,separators=(',',':')),encoding='utf8');print(json.dumps(dict(reference=ref(out),counts=dict(total))),flush=True)
if __name__=='__main__':main()
