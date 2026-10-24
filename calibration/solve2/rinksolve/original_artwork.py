"""Artwork-only reconstruction on an immutable original first-shot scene.

This data migration never imports a historical solver or transfers later camera,
trajectory, ice or board parameter values. Source contours remain static artwork
objects on the original continuous board. Numerical vertices are not landmarks.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import re
import numpy as np
from . import schema as S
from .model import (ArtworkModel, ArtworkGroup, ParameterState, block, plain,
                    load_model_bundle, save_model_bundle)

VERSION='rinksolve.original_first_shot_artwork.v1'


def ad_number(curve):
    m=re.match(r'^Ad([0-9]+)\.',curve.label)
    return int(m[1]) if m else None


def build(original, source, ad7_metadata, *, source_paths=None):
    if original.state.digest()!='4cdff29eeea87a3e2624e976876aee5d39b9b99dfb92a0e3e02fb7328a24b945':
        raise ValueError('original first-shot state does not match restored d78 reference')
    if original.scene.structural_family is not None:
        raise ValueError('original independent scene required; no later structural conversion')
    old={c.id:c for c in original.scene.curves}
    imported={c.id:c for c in source.scene.curves if ad_number(c) is not None}
    numbers=sorted({ad_number(c) for c in old.values() if ad_number(c) is not None})
    if numbers!=list(range(2,13)):
        raise ValueError('restored first-shot Ad2--Ad12 inventory changed')
    old_ad_blocks={c.parameter_block_id for c in old.values() if ad_number(c) is not None and c.parameter_block_id}
    specs=[replace(b,mode=S.VariableMode.FIXED) if b.id in old_ad_blocks else b for b in original.state.blocks]
    values=list(original.state.values)
    rebuilt={};arts=[];links=[];per_ad=[]
    for n in numbers:
        parts=[c for c in imported.values() if ad_number(c)==n]
        if not parts:
            raise ValueError(f'missing current finite shapes for Ad{n}')
        artbase=next((a for a in (ad7_metadata.artworks if n==7 else source.artworks)
                      if a.id==f'artwork_initial/Ad{n}'),None)
        if artbase is None or len(artbase.groups)!=1:
            raise ValueError(f'one static whole-artwork group required for Ad{n}')
        group=artbase.groups[0];key=group.layout_block_id
        specs.append(block(key,S.OwnerKind.ARTWORK,group.id,
            ['station_shift','height_shift','log_uniform_scale','angle'],
            [S.Unit.SCENE_UNIT,S.Unit.SCENE_UNIT,S.Unit.DIMENSIONLESS,S.Unit.RADIAN],original.state.coordinate_system_id))
        values.append(S.ParameterValue(key,(0.,0.,0.,0.)))
        anchors=[]
        for c in parts:
            d=plain(c.definition)
            if c.kind!='curve_alias':
                if c.kind!='board_material_polyline' or c.parameter_block_id is not None:
                    raise ValueError(f'nonstatic artwork source: {c.id}')
                anchors.append(tuple(d['shared_content_anchor_unwrapped']))
                d.update(shared_layout_block_id=key,shared_group_id=group.id,
                    shape_vertices_are_optimizer_variables=False,
                    original_first_shot_physics_preserved=True)
            rebuilt[c.id]=replace(c,definition=d,parameter_block_id=None)
        if len(set(anchors))!=1:
            raise ValueError(f'Ad{n} has inconsistent unfolded coordinate anchors')
        ids=tuple(c.id for c in parts)
        arts.append(replace(artbase,groups=(replace(group,shape_ids=ids),)))
        per_ad.append(dict(ad=f'Ad{n}',curve_count=len(parts),parameter_block_id=key,
            original_curve_count=sum(ad_number(c)==n for c in old.values()),
            source_templates=[t.id for t in artbase.source_templates],
            source_status=artbase.source_status,whole_shot_accepted=False,
            physical_installation_rectangle_imported=False,
            inherited_source_qualification_not_promoted=True))
    curves=tuple(rebuilt.get(c.id,c) for c in original.scene.curves)+tuple(c for k,c in rebuilt.items() if k not in old)
    for c in original.scene.curves:
        if ad_number(c) is None:
            assert next(q for q in curves if q.id==c.id) is c
        else:
            q=next(q for q in curves if q.id==c.id)
            links.append(dict(original_curve_id=c.id,current_curve_id=q.id,
                original_label=c.label,current_label=q.label,
                definition_replaced=plain(c.definition)!=plain(q.definition),
                original_geometry_parameter=c.parameter_block_id,
                original_parameter_values_retained_fixed=True,
                runtime_shape_is_static=True))
    state=ParameterState('original_first_shot_artwork_initial@1',original.state.revision+1,
        original.state.coordinate_system_id,tuple(specs),tuple(values))
    landmarks=tuple(original.landmarks)+tuple(p for p in ad7_metadata.landmarks if p.id.startswith('Ad7.'))
    provenance={**plain(original.provenance),'original_first_shot_artwork':dict(
        schema=VERSION,original_parameter_sha256=original.state.digest(),
        later_parameter_state_copied=False,source_packages=source_paths or {},
        imported_installation_rectangles=0,source_canvas_is_banner=False,
        constant_unfolded_board_shapes=True,ad3_source_bindings_applied_to_evidence=False,
        unknown_world_coordinates_not_assumed_metric=True,
        ad7_typed_content_records=plain(ad7_metadata.provenance).get('artwork_content_contracts',{}),
        ad3_typed_content_records=plain(source.provenance).get('ad3_svg_candidate',{}).get('typed_content_records',{}),
        ad3_source_geometry_initializer=plain(source.provenance).get('ad3_svg_candidate',{}),
        source_qualification_unchanged=True,whole_shot_acceptance=False)}
    candidate=replace(original,id=original.id+'/restored_first_shot_artwork',revision=original.revision+1,
        scene=replace(original.scene,curves=curves),state=state,artworks=tuple(arts),
        landmarks=landmarks,provenance=provenance)
    # All numerical original values remain available, including frozen historical
    # ad vertices; only the eleven new static object transforms are optimizer DOF.
    assert all(np.array_equal(original.state.get(b.id),state.get(b.id)) for b in original.state.blocks)
    assert candidate.rigs==original.rigs and candidate.shots==original.shots
    report=dict(schema=VERSION,original_model_id=original.id,original_model_revision=original.revision,
        original_parameter_sha256=original.state.digest(),parameter_sha256=state.digest(),
        original_blocks_bitwise_value_preserved=len(original.state.blocks),
        old_ad_vertex_blocks_retained_fixed=len(old_ad_blocks),new_artwork_blocks=len(numbers),
        new_artwork_scalar_DOF=4*len(numbers),original_curves=len(original.scene.curves),
        new_total_curves=len(curves),original_nonadvertisement_curves_preserved=sum(ad_number(c) is None for c in old.values()),
        camera_or_trajectory_or_ice_or_board_parameters_changed=0,
        imported_later_physical_parameter_values=0,physical_installation_rectangles_imported=0,
        original_path_map=links,
        current_curve_map=[dict(curve_id=c.id,label=c.label,original_curve_id=c.id if c.id in old else None,
            source_template_id=c.definition.get('source_template_id'),source_feature_id=c.definition.get('source_feature_id'),
            alias_target=c.definition.get('target_curve_id'),finite_semantic_arc=c.definition.get('finite_semantic_arc'),
            physically_static=True,vertices_are_landmarks=False,
            artwork_block_id=c.definition.get('shared_layout_block_id'),
            semantic_status=c.semantic_status) for c in curves if ad_number(c) is not None],
        ads=per_ad,whole_shot_acceptance=False)
    return candidate,report


def migrate(data_root,output):
    D=Path(data_root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    original=load_model_bundle(D/'runs/solve2_s01/ceres_regression/d78_exact')
    source=load_model_bundle(D/'runs/solve2_s01/ad3_binding/projection/model')
    metadata=load_model_bundle(D/'runs/solve2_s01/ad7_contract_candidate/model')
    candidate,report=build(original,source,metadata,source_paths=dict(
        finite_shapes=str(D/'runs/solve2_s01/ad3_binding/projection/model'),
        Ad7_typed=str(D/'runs/solve2_s01/ad7_contract_candidate'),
        Ad3_typed=str(D/'runs/solve2_s01/ad3_binding/typed_records.json'),
        current_prepared=str(D/'runs/solve2_s01/ad3_binding/prepared')))
    save_model_bundle(candidate,out/'working_model')
    (out/'migration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return candidate,report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m,r=migrate(a.data_root,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('ads','original_path_map','current_curve_map')},ensure_ascii=False))
