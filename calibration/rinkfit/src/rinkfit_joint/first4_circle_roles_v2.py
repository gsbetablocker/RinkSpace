"""Separate actual circular-paint outer edges from the preserved paint midline.

Original circle centers, midline radii, paths, source UV, and solver flags are
preserved. The outer radius is a shared suffix on the same physical circle.
"""
from __future__ import annotations
import argparse,copy,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,file_sha256,object_sha256
from rinkfit_start.native_scene import NativeScene

MODEL_SHA='67cc5e29e38acd7d2ace1333c74021feb5043e5b089dda52c2da447377709169'
MAP_SHA='c2ca5952c86ee4dea33ca9eef5d1758ed06335912930fd8f6d3d15366012ff7f'
INVENTORY_SHA='81adb350303e73afdefbfbb3b9f74ca317a7a22224a457bb8bdb37bb992e03b5'
OLD_CONFLICT='explicit_source_operator_conflicts_with_material_definition'
ROLE='outside_paint_to_ice_material_interface'
IDS=('C0','A2')


def _ref(path):
    path=Path(path);return dict(path=str(path.resolve()),sha256=file_sha256(path),bytes=path.stat().st_size)


def _ellipse_rho(points,ellipse):
    c=np.asarray(ellipse['center_uv'],float);axes=np.asarray(ellipse['axes_px'],float)/2.
    a=np.deg2rad(float(ellipse['angle_degrees']));rotation=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    q=(np.asarray(points,float)-c)@rotation.T
    return np.sqrt(np.sum((q/axes)**2,axis=-1))


def paired_outline_seed(fid,mid,rows,generation):
    """Approximate width initializer from real paired pixels, never world truth.

    A source ellipse chart is only an affine local approximation. It neither
    establishes a camera nor makes image inner/outer points equal world angles.
    """
    evidence=[];used_refs=[];ratios=[];all_pairs=0
    for logical in sorted({r['source_packet'] for r in rows}):
        ref=generation['references'][logical]
        if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('changed original circle source packet')
        packet=read_json(ref['path']);used_refs.append(ref)
        feature=next((f for f in packet['features'] if f.get('target_id')=='ice.'+fid+':outer'),None)
        if feature is None:continue
        ellipse=feature.get('geometric_interpretation') or {}
        if not all(k in ellipse for k in ('center_uv','axes_px','angle_degrees')):continue
        axes=np.asarray(ellipse['axes_px'],float)
        if axes.shape!=(2,) or np.any(axes<=0) or not np.isfinite(axes).all():continue
        for index,d in enumerate((feature.get('source_circle_fit') or {}).get('direct_readouts',[])):
            source=d.get('source_evidence') or {};pair=source.get('actual_outline_inner_outer_uv')
            if pair is None:continue
            all_pairs+=1;pair=np.asarray(pair,float)
            if pair.shape!=(2,2) or not np.isfinite(pair).all():continue
            if not source.get('material_state_order_verified_from_current_native_luminance_not_scene_style_colours'):continue
            if not source.get('actual_source_support_outside_person_and_broadcast_masks'):continue
            if not source.get('outside_material_interface_used_for_outer_circle'):continue
            rho=_ellipse_rho(pair,ellipse)
            # This is an initialization-chart sanity guard only. It does not
            # reject any source row, select fit residuals, or grant admission.
            if not (np.isfinite(rho).all() and 0<rho[0]<rho[1]):continue
            relative=(rho[1]-rho[0])/(rho[1]+rho[0])
            if not 0<relative<1:continue
            ratios.append(float(relative));evidence.append(dict(source_packet=logical,
                source_packet_sha256=ref['sha256'],source_pointer=f'/features/{packet["features"].index(feature)}/source_circle_fit/direct_readouts/{index}/source_evidence',
                source_native_pts=packet.get('source_pts'),actual_outline_inner_outer_uv=pair.tolist(),
                measured_native_full_width_px=float(np.linalg.norm(pair[1]-pair[0])),
                temporary_source_ellipse=ellipse,relative_halfwidth_numeric_chart_seed=float(relative),
                source_sigma_px=d.get('sigma_px'),observed_world_angle_correspondence_claimed=False))
    if ratios:
        h=float(mid*np.median(ratios));method='paired_actual_inner_outer_pixels_with_temporary_source_ellipse_chart_only'
        spread=np.quantile(ratios,[.25,.5,.75]).tolist()
    else:
        h=max(.005*mid,.0001);method='explicit_engineering_halfwidth_initialization_no_measured_world_width';spread=None
    return float(mid+h),dict(method=method,actual_pairs_found=all_pairs,actual_pairs_used_for_seed=len(ratios),
        initial_halfwidth_L=h,relative_halfwidth_chart_quartiles=spread,
        source_pixel_pairs=evidence,source_references=used_refs,
        old_and_new_radius_share_original_physical_center=True,measured_world_width_claimed=False,
        source_ellipse_is_projection_approximation_only=True,source_world_correspondences_generated=0,
        observed_same_station_or_same_world_angle_claimed=False,camera_calibration_gain_claimed=False,
        seed_uncertainty='source ellipse chart is approximate; outer radius remains free and must be fitted from original pixels')


def build_circle_roles(model,mapping,rows,generation,inventory,*,model_ref,map_ref,binding_ref,inventory_ref,guide_ref,contract_ref):
    result=copy.deepcopy(model);overlay={};rowmap={};corrections=[];protected=[];roles=[]
    if len({r['source_row_id'] for r in rows})!=len(rows):raise ValueError('duplicate source rows')
    for fid in IDS:
        oldtarget='ice.'+fid+':outer';newtarget='ice.'+fid+':paint_outer_boundary'
        found=[i for i,m in enumerate(result['materials']) if m['label']==oldtarget]
        if len(found)!=1:raise ValueError('old physical midline material not unique')
        oldmi=found[0];oldmaterial=result['materials'][oldmi]
        if len(oldmaterial['paths'])!=1:raise ValueError('original midline requires explicit single circle path')
        oldpath=oldmaterial['paths'][0];bi=oldpath['geometry_block'];block=result['blocks'][bi]
        if oldpath['kind']!=1 or oldpath.get('radius_index',2)!=2 or block['name']!='ice/circle/'+fid:
            raise ValueError('unexpected original shared circular line geometry')
        if len(block['values'])!=3 or any(len(block[k])!=3 for k in ('lower','upper','scales')):
            raise ValueError('this role generation requires the exact three-scalar preserved circle prefix')
        materialrows=[r for r in rows if r.get('target_id')==oldtarget and r.get('measurement_operator_resolved')=='material_interface']
        midrows=[r for r in rows if r.get('target_id')==oldtarget and r.get('measurement_operator_resolved')=='paint_center']
        expected={'C0':(147,89),'A2':(14,21)}[fid]
        if (len(materialrows),len(midrows))!=expected:raise ValueError('mixed circle operator row denominator changed')
        original=copy.deepcopy(block);mid=float(block['values'][2]);outer,seed=paired_outline_seed(fid,mid,materialrows,generation)
        if not mid<outer<.4:raise ValueError('outer radius seed does not leave a positive finite paint halfwidth')
        block['values'].append(outer);block['lower'].append(mid);block['upper'].append(.4);block['scales'].append(original['scales'][2])
        # Do not change any original fixed_indices/constant flags. The native
        # consumer protects exactly indices 0:3 and releases only this suffix.
        protected.append(dict(block_index=bi,block_name=block['name'],preserved_prefix_size=3,
            original_prefix_values=original['values'],original_prefix_lower=original['lower'],
            original_prefix_upper=original['upper'],original_prefix_scales=original['scales'],
            source_ref=model_ref,new_free_scalar_indices=[3]))
        newmi=len(result['materials']);newpath=copy.deepcopy(oldpath)
        newpath.update(id='circle_role_v2/'+fid+'/paint_outer_boundary',variant_id='circle_role_v2/'+fid+'/paint_outer_boundary',
            radius_index=3,provenance=json.dumps(dict(physical_definition=ROLE,source_role_guide=guide_ref,
                circular_line_not_target_ring=True,same_center_as_preserved_midline=True,outer_radius_is_free_suffix=True,
                initialization_only=seed['method'],source_UV_or_world_correspondence_generated=False),ensure_ascii=False))
        result['materials'].append(dict(label=newtarget,paths=[newpath]))
        newkey='|'.join((newtarget,'ice','material_interface',ROLE));oldkey='|'.join((oldtarget,'ice','material_interface','unspecified'))
        binding=dict(target=newtarget,material_index=newmi,path_index=0,geometry_ready=True,allowed_paths=[0],
            allowed_station_intervals=[[0.,1.]],measurement_operator='material_interface',boundary_role=ROLE,
            basis='original_source_outside_paint_to_ice_role_with_shared_circle_center_and_independent_outer_radius',
            radius_index=3,geometry_block=bi,source_role_authority=guide_ref,absolute_calibration_gain_claimed=False)
        overlay[newkey]=copy.deepcopy(binding);overlay[oldkey]=copy.deepcopy(binding)
        # The new target with unspecified role is also a precise outside entity;
        # it cannot alias a midpoint merely because an old role field was null.
        overlay['|'.join((newtarget,'ice','material_interface','unspecified'))]=copy.deepcopy(binding)
        definition=dict(raw_operator='material_interface',operator='material_interface',expected_operator='material_interface',
            authority='source_outside_circle_paint_boundary_role_split_v2',conflict=False,
            expected_physical_definition=dict(operator='material_interface',physical_role=ROLE,
                model_support='one_original_physical_circle_center_and_free_outer_paint_radius_suffix',
                finite_width_circular_line_classification_preserved=True,paint_center_definition_retained_on_original_target=True,
                source_role_reference=guide_ref,original_midline_contract_reference=contract_ref),
            definition_match_is_not_actual_RGB_proof=True)
        definition['measurement_contract_sha256']=object_sha256(definition['expected_physical_definition'])
        reasons=Counter()
        for row in materialrows:
            if OLD_CONFLICT not in row.get('source_review_reasons',[]):raise ValueError('expected old definition conflict missing')
            if row.get('measurement_definition',{}).get('conflict') is not True:raise ValueError('old circle source contract conflict not exact')
            pin={k:copy.deepcopy(row[k]) for k in ('source_row_id','source_packet','source_packet_sha256','shot_id','native_pts','source_uv')}
            pin['original_source_point_id']=row.get('original_source_point_id')
            remain=[x for x in row['source_review_reasons'] if x!=OLD_CONFLICT]
            for reason in remain:reasons[reason]+=1
            correction=dict(**pin,old_target_id=oldtarget,new_target_id=newtarget,
                old_joint_material_id=row['joint_material_id'],new_joint_material_id=newkey,
                old_material_id=row.get('material_id'),new_material_id=newtarget,
                old_boundary_role=row.get('boundary_role'),new_boundary_role=ROLE,
                old_typed_material_descriptor=copy.deepcopy(row['typed_material_descriptor']),
                new_typed_material_descriptor=dict(kind='ice_curve',curve_id=fid+':paint_outer_boundary',
                    residual=row['typed_material_descriptor']['residual'],physical_role=ROLE,
                    geometry_status='shared_original_circle_center_free_outer_radius_source_role_split_v2'),
                original_measurement_definition=copy.deepcopy(row['measurement_definition']),
                new_measurement_definition=copy.deepcopy(definition),original_source_review_reasons=copy.deepcopy(row['source_review_reasons']),
                remove_source_review_reasons=[OLD_CONFLICT],preserved_other_source_review_reasons=remain,
                new_source_review_reasons=remain,measurement_operator_resolved_unchanged='material_interface',
                source_flags_archived={k:copy.deepcopy(row.get(k)) for k in ('source_allowed_for_fit','original_allowed_for_fit',
                    'source_user_accepted','solver_eligible','numeric_statistics_ready','physical_identity_status','source_support_status')},
                current_uv_preserved=copy.deepcopy(row.get('current_uv')),actual_source_role_reference=guide_ref,
                corrected_role_is_not_solver_admission=True,raw_UV_ID_PTS_and_source_packet_unchanged=True,
                consumer_must_requalify_source_and_rebuild_statistics_Query=True)
            corrections.append(correction);rowmap[row['source_row_id']]={**copy.deepcopy(binding),**pin,
                'old_target_id':oldtarget,'new_target_id':newtarget,'new_joint_material_id':newkey}
        roles.append(dict(physical_circle_id=fid,original_midline_target=oldtarget,original_midline_material_index=oldmi,
            new_outer_target=newtarget,new_outer_material_index=newmi,shared_geometry_block=bi,
            original_midline_radius_index=2,new_outer_radius_index=3,paint_center_rows_unchanged=len(midrows),
            outer_material_source_rows=len(materialrows),preserved_other_reason_counts=dict(reasons),
            outer_radius_initialization=seed,old_midline_path_unchanged=True))
    # Validate the only original-state edits are suffixes in these two blocks.
    allowed={v['block_index'] for v in protected}
    for i,(old,new) in enumerate(zip(model['blocks'],result['blocks'])):
        if i in allowed:
            for key,value in old.items():
                if key in ('values','lower','upper','scales'):assert new[key][:len(value)]==value
                else:assert new[key]==value
        else:assert old==new
    assert result['materials'][:len(model['materials'])]==model['materials']
    assert len(result['blocks'])==len(model['blocks'])
    for key,value in model.items():
        if key not in ('blocks','materials'):assert result[key]==value
    refs=dict(input_model=model_ref,base_material_map=map_ref,source_binding=binding_ref,
        operator_inventory=inventory_ref,outside_role_guide=guide_ref,original_midline_contract=contract_ref)
    receipt=dict(schema='rinkfit.first4.circular_paint_operator_split.v2',roles=roles,
        protected_S01_scalar_prefixes=protected,input_references=refs,
        original_S01_scalar_values_material_paths_and_metadata_preserved=True,
        circle_centers_shared_once=True,new_backend_added=False,source_UV_ID_PTS_modified=False,
        source_flags_modified=False,source_world_truth_or_correspondence_generated=False)
    result['circle_operator_role_split_v2']=receipt
    overlay_doc=dict(schema='rinkfit.first4.circle_role_material_overlay.v2',semantic_to_material=overlay,row_to_material=rowmap,
        protected_S01_scalar_prefixes=protected,role_receipt=receipt,input_references=refs,
        application='copy base_material_map then update semantic_to_material and row_to_material with this overlay',
        paint_center_semantic_bindings_unchanged=True)
    correction_doc=dict(schema='rinkfit.first4.source_circle_role_corrections.v2',input_references=refs,corrections=corrections,
        row_corrections={c['source_row_id']:c for c in corrections},counts=dict(total=len(corrections),C0=147,A2=14),
        original_generation_modified=False,allowed_reason_removals=[OLD_CONFLICT],other_source_reasons_preserved=True,
        no_fit_admission_or_source_support_flags_promoted=True,source_and_statistics_requalification_required=True)
    return result,overlay_doc,correction_doc


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--mapping',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--inventory',required=True);p.add_argument('--project',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    if file_sha256(a.model)!=MODEL_SHA or file_sha256(a.mapping)!=MAP_SHA or file_sha256(a.inventory)!=INVENTORY_SHA:
        raise ValueError('circle operator role input pin changed')
    binding=read_json(a.binding);generation=read_json(binding['generation']['path'])
    for ref in (binding['generation'],binding['rows']):
        if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('source input generation changed')
    with gzip.open(binding['rows']['path'],'rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
    project=Path(a.project);guide=project/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot4_20261014/ice_source_guides.json'
    contract=project/'events/ankara_2026/model_constraints/paint_roles_1018.json'
    model,overlay,corrections=build_circle_roles(read_json(a.model),read_json(a.mapping),rows,generation,read_json(a.inventory),
        model_ref=_ref(a.model),map_ref=_ref(a.mapping),binding_ref=_ref(a.binding),inventory_ref=_ref(a.inventory),
        guide_ref=_ref(guide),contract_ref=_ref(contract))
    out=Path(a.out)
    for name in ('model_roles_v2.json','role_map_overlay_v2.json','source_role_corrections_v2.json'):
        if (out/name).exists():raise ValueError('immutable circle role output already exists')
    write_json(out/'model_roles_v2.json',model);overlay['output_model']=_ref(out/'model_roles_v2.json')
    corrections['output_model']=overlay['output_model'];write_json(out/'role_map_overlay_v2.json',overlay)
    write_json(out/'source_role_corrections_v2.json',corrections)
    print(json.dumps(dict(model=_ref(out/'model_roles_v2.json'),overlay=_ref(out/'role_map_overlay_v2.json'),
        corrections=_ref(out/'source_role_corrections_v2.json'),protected_S01_scalar_prefixes=overlay['protected_S01_scalar_prefixes'],
        radius_initializations=[dict(id=r['physical_circle_id'],radius=r['outer_radius_initialization']['initial_halfwidth_L'],
            pairs=r['outer_radius_initialization']['actual_pairs_used_for_seed']) for r in overlay['role_receipt']['roles']]),ensure_ascii=False))

if __name__=='__main__':main()
