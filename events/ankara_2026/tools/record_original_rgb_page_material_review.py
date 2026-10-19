"""Record explicitly viewed native RGB sheets; no automatic image classifier."""
from pathlib import Path
import argparse,hashlib,json
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--review',type=Path,required=True);ap.add_argument('--pages',required=True)
    ap.add_argument('--assigned-range',type=int,nargs=2,required=True);ap.add_argument('--reviewer',required=True)
    ap.add_argument('--unknown-prefix',type=int,action='append',default=[])
    ap.add_argument('--observation',required=True)
    a=ap.parse_args();receipt=read(a.receipt);pageids=list(map(int,a.pages.split(',')))
    scope=set(range(a.assigned_range[0],a.assigned_range[1]+1))
    assert set(pageids)<=scope and receipt['original_UV_prefix_index_sigma_PTS_bitexact'] is True
    prior=read(a.review) if a.review.exists() else dict(
        schema='rinkspace.original-low-FIT-actual-RGB-source-only-material-review.v1',
        source_RGB_receipt=dict(path=str(a.receipt),sha256=sha(a.receipt)),reviewer=a.reviewer,
        assigned_sheet_indices=sorted(scope),reviewed_sheets=[],qualified_source_only_rows=[],uncertain_source_only_rows=[],
        source_model_residual_not_used_for_material_selection=True,
        source_UV_sigma_PTS_weights_rows_and_runtime_roles_modified=False,
        exact_subpixel_matching_uncertainty_not_certified_from_blurred_RGB=True)
    assert prior['source_RGB_receipt']['sha256']==sha(a.receipt) and prior['reviewer']==a.reviewer
    reviewed={r['sheet_index']for r in prior['reviewed_sheets']};assert not reviewed.intersection(pageids)
    selected=[r for r in receipt['point_to_pair_sheet'] if r['sheet_index'] in pageids]
    assert set(a.unknown_prefix)<={r['current_original_prefix_bundle_index']for r in selected}
    for pi in pageids:
        sheet=receipt['sheets'][pi];assert pi==sheet['sheet_index'] and sha(sheet['path'])==sheet['sha256']
        rows=[r for r in selected if r['sheet_index']==pi]
        prior['reviewed_sheets'].append(dict(sheet_index=pi,path=sheet['path'],sha256=sheet['sha256'],
            actual_original_points=len(rows),both_endpoints_actually_viewed=True,
            original_prefix_indices=[r['current_original_prefix_bundle_index']for r in rows],
            observation=a.observation))
        for row in rows:
            unknown=row['current_original_prefix_bundle_index'] in a.unknown_prefix
            target=prior['uncertain_source_only_rows'] if unknown else prior['qualified_source_only_rows']
            target.append(dict(row,material_identity_review='unknown_requires_native_context' if unknown else 'actual_both_endpoints_static_far_board_printed_material',
                original_pixel_measurements_or_sigma_not_reestimated=True,material_reviewed_by=a.reviewer,
                source_paper_identity_not_inferred_from_navigation_mask=True))
    covered={r['sheet_index']for r in prior['reviewed_sheets']}
    prior.update(status='complete_source_only_material_review' if covered==scope else 'partial_source_only_material_review',
        reviewed_sheet_count=len(covered),qualified_point_count=len(prior['qualified_source_only_rows']),
        uncertain_point_count=len(prior['uncertain_source_only_rows']),remaining_sheet_indices=sorted(scope-covered))
    prior['reviewed_sheets'].sort(key=lambda r:r['sheet_index'])
    a.review.write_text(json.dumps(prior,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(a.review),sha256=sha(a.review),status=prior['status'],reviewed_sheets=len(covered),
        qualified=prior['qualified_point_count'],uncertain=prior['uncertain_point_count'])),flush=True)
if __name__=='__main__':main()
