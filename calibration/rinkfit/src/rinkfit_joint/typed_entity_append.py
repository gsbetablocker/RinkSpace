"""Validate versioned current entity measurements before a source append.

This is a separate measurement version. Existing raw rows and original source
flags stay fixed; an earlier inactive numeric row is never silently reactivated.
"""
from pathlib import Path
from .io_utils import read_jsonl,read_json,object_sha256


def load_qualification(reference,verify):
    if reference is None:return {},[]
    if isinstance(reference,(list,tuple)):
        combined={};sources=[]
        for item in reference:
            rows,refs=load_qualification(item,verify)
            if set(combined)&set(rows):raise ValueError('Repeated physical entity source revision in qualification list')
            combined.update(rows);sources.extend(refs)
        return combined,sources
    manifest_file=verify(reference);manifest=read_json(manifest_file)
    if manifest.get('schema')=='rinkfit.exact_typed_source_suffix_rebase.v4':
        from .typed_source_rebase_v4 import load_rebased_qualification
        return load_rebased_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.exact_typed_source_suffix_rebase.v3':
        from .typed_source_rebase_v3 import load_rebased_qualification
        return load_rebased_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.exact_typed_source_suffix_rebase.v2':
        from .typed_source_rebase_v2 import load_rebased_qualification
        return load_rebased_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.exact_typed_source_suffix_rebase.v1':
        from .typed_source_rebase_v1 import load_rebased_qualification
        return load_rebased_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.known_corner_source_support_qualification.v1':
        from .known_entity_tracklet_append import load_source_support_qualification
        return load_source_support_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.static_board_tracklet_point_qualification.v1':
        from .static_board_tracklet_append import load_point_qualification
        return load_point_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')=='rinkfit.static_ice_tracklet_point_qualification.v1':
        from .static_ice_tracklet_append import load_point_qualification
        return load_point_qualification(manifest_file,manifest,reference,verify)
    if manifest.get('schema')!='rinkfit.new_actual_type1_source_qualification_gate.v1':
        raise ValueError('Explicit new physical entity qualification required')
    for name in ('input_cumulative_journal','qualification_receipts','current_model','source_mean_and_floating_estimator_replay_QA'):
        verify(manifest[name])
    if not manifest['old_raw_source_flags_false_are_not_overwritten']or not manifest['new_measurement_version_geometry_qualification_only']:
        raise ValueError('Entity qualification attempts to rewrite original source history')
    receipts={r['source_row_id']:r for r in read_jsonl(verify(manifest['qualification_receipts']))}
    if len(receipts)!=manifest['qualified_new_measurement_rows']:
        raise ValueError('Entity qualification row ledger incomplete')
    for receipt in receipts.values():
        if not receipt['new_actual_source_geometry_qualified']or receipt['entity_kind']!=1 or not receipt['physical_point']or receipt['solver_eligible']or receipt['statistical_objective_eligible']:
            raise ValueError('Entity source qualification is not explicit geometry-only')
        if receipt['original_flags_preserved'].get('manual_reference_preserved'):
            raise ValueError('Automatic entity source append cannot overwrite a manual click')
    return receipts,[dict(file=str(manifest_file),sha256=reference['sha256']),manifest['qualification_receipts'],dict(manifest['input_cumulative_journal'],role='qualified_current_entity_source_journal')]


def validate_new_entity(row,receipt,material,material_index):
    if receipt is None:raise ValueError('No source qualification for physical entity measurement')
    if receipt.get('schema')=='rinkfit.source_supported_known_corner_qualification.v1':
        from .known_entity_tracklet_append import validate_source_support_receipt
        return validate_source_support_receipt(row,receipt,material,material_index)
    if receipt.get('schema')=='rinkfit.static_board_tracklet_point_receipt.v1':
        from .static_board_tracklet_append import validate_point_receipt
        return validate_point_receipt(row,receipt,material,material_index)
    if receipt.get('schema')=='rinkfit.static_ice_tracklet_point_receipt.v1':
        from .static_ice_tracklet_append import validate_point_receipt
        return validate_point_receipt(row,receipt,material,material_index)
    if not row.get('point_correspondence')or row.get('measurement_operator_resolved')!='physical_corner':
        raise ValueError('Entity adapter requires the actual corresponding-corner source operator')
    expected=object_sha256(dict(source_row_id=row['source_row_id'],native_pts=row['native_pts'],
        current_2D_UV=row['current_uv'],estimator='native_two_edge_quadratic_intersection_v1'))
    if expected!=receipt['current_2D_source_mean_sha256']or receipt['native_pts']!=row['native_pts']or receipt['original_UV_preserved']!=row['source_uv']:
        raise ValueError('Entity source receipt differs from current mean/original source UV')
    if receipt['existing_material_index']!=material_index:
        raise ValueError('Entity source redirects its fixed physical material')
    paths=[i for i,p in enumerate(material['paths'])if p['id']==receipt['existing_point_path']]
    if len(paths)!=1:raise ValueError('Entity physical point path is not uniquely present')
    p=material['paths'][paths[0]]
    if not p.get('physical_point',False)or int(p['kind'])!=int(receipt['point_path_kind']):
        raise ValueError('Entity path is not the source-qualified fixed physical point')
    # The numeric forward for kind 2 with a single vertex is a fixed point.
    if p['kind']==2 and len(p.get('points',[]))!=1 and len(p.get('geometry_indices',[]))!=2:
        raise ValueError('Entity fixed material path contains more than one physical vertex')
    return paths[0]
