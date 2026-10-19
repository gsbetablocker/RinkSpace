"""Read-only full target producer audit without rebuilding the programme graph."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np


def audit(args):
    from calibration.solver_runtime import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle(); install_geometry_unpickle()
    from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
    from calibration.program_tracking.compact_source_json import unpack_source_json
    from calibration.solver_runtime.strict_source_dataset_adapter import StrictDatasetFactor, _key
    from calibration.solver_runtime.strict_dataset_target_bindings import apply_confirmed_v6_target_bindings
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    from structured_programme_source_candidate.paired_programme_chart import CompleteProgrammeParameterLayout
    from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
    from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor
    from structured_programme_source_candidate.course_patterns_independent_placements import IndependentlyPlacedCoursePatterns
    from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry
    from rinkspace.vision_camera_initialization import frozen_scene_curve

    manifest=json.loads(args.input_manifest.read_text(encoding='utf8'))
    private_ref=manifest['absolute_geometry_fd_parallelism']['material_owner']
    authenticate_source_reference(private_ref)
    owner=pickle.loads(Path(private_ref['path']).read_bytes())
    identity=json.loads((args.state.parent/'state_identity.json').read_text(encoding='utf8'))
    authenticate_source_reference(dict(path=str(args.state),sha256=identity['state_npz_sha256']))
    authenticate_source_reference(dict(path=str(args.scene),sha256=identity['scene_sha256']))
    scene=json.loads(args.scene.read_text(encoding='utf8'))
    with np.load(args.state,allow_pickle=False) as saved:
        canonical=saved['q'].copy(); pattern_start=int(saved['aux__shared_pattern_start'])
        canonical_world=saved['aux__current_shared_world_geometry_parameter_indices'].copy()
        canonical_placements=saved['aux__current_independent_pattern_placement_parameter_indices'].copy()
    if hashlib.sha256(canonical.tobytes()).hexdigest()!=identity['q_sha256']:
        raise ValueError('Original complete numerical state identity changed')
    layout=CompleteProgrammeParameterLayout(identity['original_complete_parameter_prefix_count'],34)
    q=layout.evaluation_q(canonical); inverse=layout.canonical_to_evaluation
    world_indices=inverse[canonical_world]
    factor=StrictDatasetFactor.__new__(StrictDatasetFactor)
    factor.parameter_count=len(q); factor.geometry=owner.current; factor.world_indices=world_indices
    factor.core_count=layout.original_complete_count-38+34
    factor.ice=CurrentIceMaterialView(owner.current); factor.pattern_source=None
    factor._world_cache_q=factor._camera_cache_q=factor._evaluation_q=None
    factor._world_cache={}; factor._derivative_cache={}; factor._camera_cache={}
    circle=NativeBoardCircleFactor.__new__(NativeBoardCircleFactor)
    circle.geometry=owner.current; circle.world_indices=world_indices; circle.unit=owner.current.scene_unit_m
    circle.radius_shared=scene['additional_board_circle_patterns']['radius_shared']
    circle.circle_indices=inverse[np.asarray(identity['independent_circle_material_parameter_indices'])]
    _,_,course_chain=owner.current.print_course_binding(q[world_indices])
    circle.active_world=world_indices[np.any(np.asarray(course_chain)!=0,axis=0)]
    factor.circle_factor=circle
    original=SharedPerimeterBoardPatternGeometry(owner.base,owner.current.scene_unit_m,
        facet_authority=scene['shared_board_patterns']['facet_authority'])
    @contextmanager
    def bind(core):
        geometry=IndependentlyPlacedCoursePatterns(original,owner.source_course.course,
            owner.current.current_course6(core[world_indices]),q[inverse[canonical_placements]])
        yield geometry
    material=NativeBoardMaterialPointFactor.__new__(NativeBoardMaterialPointFactor)
    material.core_count=factor.core_count; material.pattern_start=int(inverse[pattern_start])
    material.patterns=SimpleNamespace(bind_current_geometry=bind)
    factor.material_factor=material
    dataset_ref=dict(path=str(args.dataset),sha256=file_sha(args.dataset))
    dataset=json.loads(args.dataset.read_text(encoding='utf8'))
    targets={}; ad5_groups=[]; fit_points=held_points=0
    for cut in dataset['cuts']:
        for ref in (cut['source_pack'],cut['support_index']):authenticate_source_reference(ref)
        with np.load(cut['source_pack']['path'],allow_pickle=False) as pack, np.load(cut['support_index']['path'],allow_pickle=False) as index:
            headers=unpack_source_json(pack['rows_json']); offsets=pack['row_offsets']
            uv=sigma=None
            for role in ('fit','held'):
                stations=index[role+'_point_indices']; row_ids=np.searchsorted(offsets,stations,side='right')-1
                counts=np.bincount(row_ids,minlength=len(headers))
                if role=='fit':fit_points+=len(stations)
                else:held_points+=len(stations)
                for row_id in np.flatnonzero(counts):
                    row=headers[row_id]; target=row['model_target']
                    targets[_key(target)]=target
                    if target.get('advertisement_number')==5 and target.get('extent_scope')=='external_border':
                        if uv is None:uv=pack['uv'];sigma=pack['sigma_px']
                        selected=stations[row_ids==row_id]
                        group=dict(actual_pts=row['actual_pts'],model_target=dict(target),uv=uv[selected],
                            sigma=sigma[selected],source_locators=[dict(source_pack=cut['source_pack'],source_row_index=int(row_id),
                            original_native_pts=row['native_pts'],decoded_BGR_sha256=row['decoded_BGR_sha256'])])
                        ad5_groups.append(group)
    if (fit_points,held_points)!=(dataset['fit_points'],dataset['held_points']):raise ValueError('Full selected target inventory changed')
    binding=apply_confirmed_v6_target_bindings(ad5_groups,[],dataset_ref)
    for group in ad5_groups:
        original_target=group['original_model_target_before_confirmed_label_binding']
        targets.pop(_key(original_target),None); targets[_key(group['model_target'])]=group['model_target']
    factor.geometry.validate_parameters(q[world_indices]); factor._prepare_evaluation(q)
    records=[]; failures=[]
    for target in targets.values():
        try:
            world=factor._world(q,target,False)[0]
            if world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all():raise ValueError('No finite complete material world')
            record=dict(model_target=target,knots=len(world),current_world_producer_passed=True)
            if target['kind'] in ('advertisement_boundary','board_circle'):
                provider_target=target
                if target['kind']=='board_circle':
                    role=scene['additional_board_circle_patterns']['circles'][str(target['advertisement_number'])]['extent_scope']
                    provider_target=dict(target,target_role=role)
                    record['same_q_exported_circle_extent_role']=role
                expected=frozen_scene_curve(scene,provider_target)
                if expected.shape!=world.shape or expected.tobytes()!=world.tobytes():raise ValueError('Current target differs from original same-q exported provider')
                record['same_q_exported_provider_world_bytes_identical']=True
            records.append(record)
        except Exception as error:failures.append(dict(model_target=target,error=str(error)))
    receipt=dict(schema='rinkspace.strict-dataset-full-current-target-private-owner-audit.v1',dataset_reference=dataset_ref,
        private_material_reference=private_ref,state_q_sha256=identity['q_sha256'],fit_points=fit_points,held_points=held_points,
        target_label_binding_receipt=binding,complete_current_material_target_records=records,failures=failures,
        passed=not failures,programme_graph_rebuilt=False,solver_or_fit_started=False,source_pixels_or_roles_changed=False)
    args.report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(args.report),targets=len(targets),passed=len(records),failures=failures),ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('input-manifest','state','scene','dataset','report'):parser.add_argument('--'+name,required=True,type=Path)
    audit(parser.parse_args())
