"""Advertisement source selection and current-scene binding; never fitting."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ..compat.ad_assets import import_mixed_ad_template, append_source_atlas


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def import_assets(selection_path, out_path):
    """Consume the selected existing sources once, keeping their own clocks."""
    selection_path = Path(selection_path).resolve()
    selection = read(selection_path)
    if selection.get('schema')=='rinkspace.print_selection.v1':
        from ..compat.ad_assets import import_selected_parts
        result=import_selected_parts(selection)
        out_path=Path(out_path).resolve()
        if out_path.exists():raise FileExistsError('Advertisement asset output already exists: '+str(out_path))
        out_path.parent.mkdir(parents=True,exist_ok=True)
        result['selection']=reference(selection_path)
        out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        return {'assets':str(out_path),'counts':result['counts'],'fit_executed':False}
    scope = read(selection['scope'])
    source = scope['source']
    clock = scope['source_clock']
    current_source = {'source_sha256': source['sha256'],
                      'source_video_path': source['registered_path'],
                      'time_base': f"{clock['time_base_num']}/{clock['time_base_den']}",
                      'native_hw': [source['height'], source['width']]}
    result = import_mixed_ad_template(selection['mixed_template'], current_source)
    for atlas in selection.get('source_atlases', []):
        result = append_source_atlas(result, atlas['path'], current_source,
                                     frame_sha256=atlas['frame_sha256'])
    result['selection'] = reference(selection_path)
    result['current_scene'] = reference(selection['scene'])
    result['source_selection_notes'] = deepcopy(selection['source_selection_notes'])
    result['review_references'] = []
    for item in selection.get('review_references', []):
        path = Path(item['path'])
        data = read(path)
        result['review_references'].append({**deepcopy(item), **reference(path),
            'schema': data.get('schema'), 'authority_promoted': False})
    result['discovery_references'] = [reference(p) for p in selection['discovery_references']]
    result['historical_runtime_inventory'] = deepcopy(selection['historical_runtime_inventory'])
    result['limitations'] = deepcopy(selection['limitations'])
    # Material coordinates remain the recorded source charts. This import does
    # not create a new banner homography or turn an incomplete view into corners.
    result['templates'][0]['material_chart_status'] = 'recorded_initial_source_charts_not_refitted'
    result['templates'][0]['shape_review'] = deepcopy(selection.get('template_shape_review', {'status':'not_reviewed'}))
    result['templates'][0]['shared_size_instance_ids'] = deepcopy(selection.get('shared_size_instance_ids', []))
    result['templates'][0]['shared_size_relation_source'] = selection['shared_size_relation_source']
    result['instances'][0]['scene_surface_id'] = 'entity/board.continuous'
    result['instances'][0]['placement_status'] = 'D1_source_binding_only_D3_installation_pending'
    result['instances'][0]['historical_installation_ref'] = selection['historical_runtime_inventory']['model']
    if selection.get('glyph_rebuild'):
        from ..compat.ad_assets import apply_glyph_rebuild
        result=apply_glyph_rebuild(result,selection['glyph_rebuild'])
    for pair in selection.get('paired_instances',[]):
        from ..domain.print_assets import attach_paired_instance
        result=attach_paired_instance(result,pair['primary_instance_id'],pair['paired_instance_id'],
            pair['size_block_id'],authority=pair['authority'])
    for item in selection.get('installation_initializations',[]):
        from ..domain.print_assets import initialize_instance
        seed_path=Path(item['path'])
        if not seed_path.is_absolute():seed_path=selection_path.parent/seed_path
        seed=read(seed_path)
        result=initialize_instance(result,item['instance_id'],seed['native_parameters'],
            provenance={**reference(seed_path),'authority':'preserved_installation_initializer_not_new_fit'})
    out_path = Path(out_path).resolve()
    if out_path.exists():
        raise FileExistsError('Advertisement asset output already exists: '+str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return {'assets': str(out_path), 'counts': result['counts'], 'fit_executed': False}
