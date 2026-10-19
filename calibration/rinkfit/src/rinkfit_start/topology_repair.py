"""Restore explicitly recorded finite contour topology to a separate candidate.

No image, residual, endpoint-distance heuristic, historical owner or additional
measurement participates.  Every repaired flag is re-read from hashed JSON.
"""
from pathlib import Path
import copy,json,hashlib,shutil
import numpy as np
from .data import sha256_file
from .runtime import atomic_json


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def json_pointer(document,pointer):
    if not pointer.startswith('/'):raise ValueError('Absolute source JSON pointer required')
    node=document
    for token in pointer[1:].split('/'):
        token=token.replace('~1','/').replace('~0','~')
        node=node[int(token)] if isinstance(node,list) else node[token]
    return node


def restore_explicit_topology(model_path,mapping_path,output_dir):
    """Write an immutable candidate with unchanged physical parameter values."""
    original=Path(model_path);mapping_path=Path(mapping_path);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Never overwrite a model candidate')
    for name in ['model.json','topology_repair.json']:
        path=str(out/name)
        if len(path.encode('utf-16-le'))//2>=260 or len((out/name).name)>255:raise ValueError('OneDrive-compatible short absolute path required')
    model=json.loads(original.read_text(encoding='utf-8'));old=copy.deepcopy(model)
    mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    if mapping['model_sha256']!=sha256_file(original) or mapping.get('derived_from_endpoint_distance'):
        raise ValueError('Explicit mapping/model identity mismatch or inferred topology')
    source=Path(mapping['source_path'])
    if sha256_file(source)!=mapping['source_sha256'] or not mapping['source_sha_matches_startup_seed_source_scene']:
        raise ValueError('Authenticated startup scene identity mismatch')
    document=json.loads(source.read_text(encoding='utf-8'));changes=[];seen=set()
    for record in mapping['positive_closure_mismatches']:
        mi=int(record['material_index']);pi=int(record['path_index']);key=(mi,pi)
        if key in seen:raise ValueError('Duplicate topology repair')
        seen.add(key);material=model['materials'][mi];path=material['paths'][pi]
        if material['label']!=record['material_label'] or path['id']!=record['path_id'] or path['variant_id']!=record['variant_id'] or canonical_sha(path)!=record['original_path_sha256']:
            raise ValueError('Finite material/path identity mismatch')
        flag=json_pointer(document,record['source_pointer'])
        if flag is not True or record['explicit_source_closed'] is not True or path['closed'] is not False or path['kind']!=2:
            raise ValueError('Only explicit closed-source/open-export discrepancies are repairs')
        parent=json_pointer(document,record['source_pointer'].rsplit('/',1)[0])
        if parent['id']!=record['descriptor']['contour_key']:raise ValueError('Source contour identity mismatch')
        points=path['points'];old_count=len(points)
        if len(points)<2:raise ValueError('Noncurve source topology')
        if points[0]!=points[-1]:points.append(copy.deepcopy(points[0]))
        path['closed']=True
        path['topology_provenance']=dict(method='restored_explicit_named_source_closed_flag',
            source_path=str(source),source_sha256=mapping['source_sha256'],source_json_pointer=record['source_pointer'],
            original_open_path_sha256=record['original_path_sha256'],mapping_sha256=sha256_file(mapping_path),
            endpoint_distance_used=False,original_vertices_unchanged=True)
        changes.append(dict(material_index=mi,material_label=material['label'],path_index=pi,path_id=path['id'],variant_id=path['variant_id'],old_vertices=old_count,new_vertices=len(points),
            old_station_to_same_world_station_scale=(old_count-1)/(len(points)-1),source_json_pointer=record['source_pointer']))
    if not changes:raise ValueError('No authenticated topology discrepancy')
    if canonical_sha(model['blocks'])!=canonical_sha(old['blocks']) or canonical_sha(model['camera'])!=canonical_sha(old['camera']):
        raise AssertionError('Topology repair changed physical/camera values')
    model['model_id']='pixel_topology040_explicit_closed_contours'
    model['provenance']['topology_repair']=dict(original_model_path=str(original),original_model_sha256=sha256_file(original),
        mapping_path=str(mapping_path),mapping_sha256=sha256_file(mapping_path),source_sha256=mapping['source_sha256'],
        restored_closed_paths=len(changes),source_measurements_unchanged=True,physical_blocks_unchanged=True)
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'model.json',model)
    report=dict(schema='rinkfit.explicit_contour_topology_repair.v1',model=dict(path=str(out/'model.json'),sha256=sha256_file(out/'model.json')),
        original_model=dict(path=str(original),sha256=sha256_file(original)),mapping=dict(path=str(mapping_path),sha256=sha256_file(mapping_path)),
        source=dict(path=str(source),sha256=sha256_file(source)),changes=changes,
        parameter_blocks_bytewise_canonical_sha256=canonical_sha(model['blocks']),camera_bytewise_canonical_sha256=canonical_sha(model['camera']),
        parameters_and_camera_unchanged=True,all_original_path_vertices_unchanged=True,new_measurements_added=False,endpoint_distance_used=False)
    atomic_json(out/'topology_repair.json',report)
    certificate=original.parent/'Brown_domain_certificate.json'
    if certificate.exists():
        cert=json.loads(certificate.read_text(encoding='utf-8'));cert['model_sha256']=sha256_file(out/'model.json')
        cert['topology_only_rebinding']=dict(previous_certificate=dict(path=str(certificate),sha256=sha256_file(certificate)),camera_and_parameter_blocks_identical=True)
        atomic_json(out/'Brown_domain_certificate.json',cert)
    return model,report


def remap_same_world_seed(run_dir,seed_dir,output_dir,repair):
    """Change only chart coordinates when a closing segment extends a path."""
    run=Path(run_dir);seed=Path(seed_dir);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Never overwrite an initializer')
    S=np.fromfile(seed/'station.f64','<f8');B=np.fromfile(seed/'path.u32','<u4');M=np.fromfile(run/'backend_input/material.u32','<u4')
    if S.shape!=(182017,) or B.shape!=S.shape or M.shape!=S.shape:raise ValueError('Full original source required')
    before=S.copy();changed=np.zeros(len(S),bool)
    for c in repair['changes']:
        mask=(M==c['material_index'])&(B==c['path_index']);S[mask]*=c['old_station_to_same_world_station_scale'];changed|=mask
    out.mkdir(parents=True,exist_ok=True);S.astype('<f8').tofile(out/'station.f64');shutil.copyfile(seed/'path.u32',out/'path.u32')
    for name in ['initial_row.f64','initial_alpha.f64']:
        if (seed/name).exists():shutil.copyfile(seed/name,out/name)
    report=dict(schema='rinkfit.explicit_topology_same_world_chart_seed.v1',source_rows=len(S),input=str(seed),output=str(out),
        model=repair['model'],changed_chart_rows=int(np.count_nonzero(S!=before)),affected_material_rows=int(changed.sum()),
        outside_affected_material_stations_byte_unchanged=bool(np.array_equal(S[~changed],before[~changed])),
        finite_support_preserved=True,old_world_point_preserved_by_exact_affine_station_relation=True,needs_all_source_capture_and_order_repair=True,
        files={name:dict(path=str(out/name),sha256=sha256_file(out/name)) for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64'] if (out/name).exists()})
    atomic_json(out/'station_initialization.json',report);return report
