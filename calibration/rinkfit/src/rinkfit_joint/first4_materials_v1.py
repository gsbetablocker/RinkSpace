"""Append first-four-shot actual materials without changing the complete prefix.

Print boundaries with repeated source exposures have one free world-chart shape
on the common continuous board. Historical coordinates only initialize that
shape; they are never a fixed template, physical survey, or camera measurement.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from rinkfit_start.native_scene import NativeScene
from .io_utils import file_sha256, object_sha256, read_json, write_json
from .material_crosswalk import _source_contours
from .shared_print import append_shared_print_components

MODEL_SHA = '5cadc6e9cc8e8d998fee169b3503f90b654351672a05d444b3e3584b3254cdef'
BINDING_SHA = 'bfd5ceb4a6600081e53f591b070d46f22b3f9587706244daf14164c0d09751ad'
# The exact-source audit proves only partial raw subarcs for these names.
# No full path alias, second copy of the same ink, or QR identity is invented.
PARTIAL_OR_CONFLICTING = {
    'Ad1.print.shot3_SKATING_T': 'only_partial_raw_subarc_with_r2_ALL_SKATING_T_00_no_full_path_coverage',
    'Ad1.print.shot3_SKATING_I': 'only_partial_raw_subarc_with_r2_ALL_SKATING_I_00_no_full_path_coverage',
    'Ad1.print.shot3_qr_right_top_outer': 'old_r2_right_top_02_contains_both_outer_and_centre_raw_subarcs',
    'Ad1.print.shot3_qr_right_top_centre': 'old_r2_right_top_02_contains_both_outer_and_centre_raw_subarcs',
}
FORBIDDEN = {'Ad4.print.current_tail002'}


def _ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=file_sha256(path), bytes=path.stat().st_size)


def _exposures(rows):
    return sorted({(r['shot_id'], int(r['native_pts'])) for r in rows
                   if r.get('source_support_status') == 'source_supported_identity_candidate'
                   and r.get('current_uv') is not None})


class SourceBoard:
    """Invert the source board itself, never an image camera or source UV."""
    def __init__(self, scene, current):
        p = scene['parameters']; b = p['bounds']; self.scale = float(b[2] - b[0])
        c = p['corner_curve']; rx, ry = c['tangent_extent_xy_scene_units']
        g = [float(b[3]-b[1])/self.scale, float(rx)/self.scale,
             float(ry)/self.scale, float(c['exponent']), float(p['board_height'])/self.scale]
        self.scene = NativeScene(dict(blocks=[dict(name='source/global32', values=g+[0.]*27)], materials=[]))
        self.current = current
        self.samples = np.linspace(0., self.scene.perimeter, 32768, endpoint=False)
        self.tree = cKDTree(self.scene.board(self.samples, 0.)[:, :2])
        self.lengths = np.array([self.scene.long, self.scene.arc, self.scene.short, self.scene.arc]*2)
        self.edges = np.r_[0., np.cumsum(self.lengths)]
        self.current_lengths = np.array([current.long, current.arc, current.short, current.arc]*2)
        self.current_edges = np.r_[0., np.cumsum(self.current_lengths)]

    def station(self, xyz):
        xyz = np.asarray(xyz, float)/self.scale
        guess = self.tree.query(xyz[:, :2])[1]
        step = self.scene.perimeter/len(self.samples); result = []
        residual = []
        for xy, i in zip(xyz[:, :2], guess):
            s0 = self.samples[int(i)]
            f = minimize_scalar(lambda s: float(np.sum((self.scene.board(s, 0.)[:2]-xy)**2)),
                                bounds=(s0-2*step, s0+2*step), method='bounded', options={'xatol':1e-13})
            result.append(float(f.x % self.scene.perimeter)); residual.append(float(np.sqrt(f.fun)))
        s = np.asarray(result)
        # Same named board segment and fractional arclength are initialization,
        # not a claim that old and current absolute stations were measured.
        part = np.minimum(np.searchsorted(self.edges[1:], s, side='left'), 7)
        frac = (s-self.edges[part])/self.lengths[part]
        current_s = self.current_edges[part]+frac*self.current_lengths[part]
        current_s = np.unwrap(current_s*2*np.pi/self.current.perimeter)*self.current.perimeter/(2*np.pi)
        z = xyz[:, 2]/self.scene.height*self.current.height
        return np.c_[current_s, z], dict(source_board_projection_max_L=float(max(residual, default=0)),
            retraction='own_source_board_segment_fraction_to_current_common_board; relative_source_H_to_current_H',
            source_scene_XYZ_is_numeric_initializer_only=True, camera_inverse_projection_used=False,
            exact_source_current_station_identity_claimed=False)


def _template_chart_vertices(model, adid, contour):
    """Use the existing physical installation as an initial chart; add none."""
    key = 'ad/'+str(adid)
    bi = next((i for i,b in enumerate(model['blocks']) if b['name']==key), None)
    if bi is None:
        raise ValueError('no_current_same_physical_ad_installation_for_material_uv_initializer')
    p = np.asarray(contour['material_uv'], float)
    if contour.get('material_coordinate_domain') != 'ad_rectangle_u_left_to_right_v_top_to_bottom':
        raise ValueError('unknown_material_coordinate_domain')
    corner = next((m for m in model['materials'] if m['label']==f'Ad{adid}.left_bottom'), None)
    if corner is None:
        raise ValueError('no_existing_source_labeled_installation_orientation')
    left_u = float(corner['paths'][0]['points'][0][0]); u = p[:,0] if left_u==0. else 1.-p[:,0]
    a = np.asarray(model['blocks'][bi]['values'], float)
    z = a[2]+(1.-p[:,1])*(a[3]-a[2]); s = a[0]+u*a[1]
    return np.c_[s,z], dict(retraction='source_material_uv_to_existing_same_physical_installation_numeric_chart',
        existing_advertisement_block=bi, source_u_left_to_right=True, source_v_top_to_bottom=True,
        source_scene_XYZ_is_numeric_initializer_only=True, camera_inverse_projection_used=False,
        exact_source_current_station_identity_claimed=False)


def compile_first4_materials(model, rows, project_root, *, input_refs=None, overlap_ref=None):
    result = copy.deepcopy(model); project = Path(project_root)
    prefix_blocks = copy.deepcopy(model['blocks']); prefix_materials = copy.deepcopy(model['materials'])
    bylabel = {m['label']:i for i,m in enumerate(result['materials'])}
    current = NativeScene(model); sources = {}; source_refs=[]
    for sid in sorted({r['shot_id'] for r in rows if r['shot_id'] in ('s02','s03','s04')}):
        nr=int(sid[1:]); dr=project/'events/ankara_2026/models'/('shot2_8a28_r2_20261014' if nr==2 else f'shot{nr}_8a28_20261014')
        scene=read_json(dr/'scene.json'); template=read_json(dr/'ad2_material_template.json')
        refs=[_ref(dr/name) for name in ('scene.json','ad2_material_template.json','feature_registry.json')]
        sources[sid]=dict(scene=scene, template=template, contours=_source_contours(scene,template),
                          board=SourceBoard(scene,current), refs=refs)
        source_refs.extend(refs)
    bytarget=defaultdict(list); semantic=defaultdict(list)
    for r in rows:
        if r.get('target_id'): bytarget[r['target_id']].append(r)
        if r.get('joint_material_id'): semantic[r['joint_material_id']].append(r)
    target_map={}; pending=[]; added=[]; components=[]; component_targets=[]
    aliases=[]; existing=[]
    for target,members in sorted(bytarget.items()):
        d=members[0].get('typed_material_descriptor') or {}; kind=d.get('kind')
        if target in FORBIDDEN:
            target_map[target]=dict(target=target, material_index=None, geometry_ready=False,
                reason='explicit_source_identity_rejection_actor_cap_or_adjacent_ad_not_board_body')
            continue
        if target in bylabel:
            mi=bylabel[target]; paths=result['materials'][mi]['paths']
            target_map[target]=dict(target=target, material_index=mi, path_index=0,
                geometry_ready=True, allowed_paths=list(range(len(paths))), basis='existing_exact_source_target_label',
                absolute_calibration_gain_claimed=False)
            existing.append(dict(target=target,material_index=mi))
            continue
        def unknown(reason):
            target_map[target]=dict(target=target,material_index=None,geometry_ready=False,reason=reason)
            pending.append(dict(target=target,reason=reason,rows=len(members),source_qualified_exposures=len(_exposures(members)),actual_source_exposures=len({(r['shot_id'],int(r['native_pts'])) for r in members}),source_review_reasons=sorted({reason for r in members for reason in r.get('source_review_reasons',[])})))
        if kind=='ice_curve':
            sid=members[0]['shot_id']; src=sources[sid]; fid,_,role=d['curve_id'].partition(':')
            feature=next((f for f in src['scene']['ice_features'] if f['id']==fid),None)
            if feature is None: unknown('source_ice_feature_missing');continue
            if role.startswith('edge') and feature['kind']=='polygon':
                edge=int(role[4:]); poly=feature['polygon_xyz']
                if edge in feature.get('open_edges',[]):unknown('unpainted_open_edge_not_physical_ice_material');continue
                xyz=[poly[edge],poly[(edge+1)%len(poly)]]
            elif role.startswith('segment') and feature['kind']=='polyline':
                xyz=feature['segments_xyz'][int(role[7:])]
            else: unknown('unsupported_actual_ice_primitive');continue
            xy=np.asarray(xyz,float)[:,:2]/src['board'].scale
            if xy.shape!=(2,2):unknown('actual_ice_segment_requires_two_source_endpoints');continue
            bi=len(result['blocks']); name='ice/first4/'+hashlib.sha256(target.encode()).hexdigest()[:20]
            result['blocks'].append(dict(name=name,values=xy.ravel().tolist(),lower=[-.5]*4,upper=[.5]*4,
                scales=[.05]*4,constant=False,fixed_indices=[]))
            mi=len(result['materials']); result['materials'].append(dict(label=target,paths=[dict(
                id='first4/'+target,variant_id='first4/'+target,kind=0,geometry_block=bi,
                physical_point=False,closed=False,provenance=json.dumps(dict(source_descriptor=d,
                    source_feature_id=fid,source_scene=src['refs'][0],source_coordinate_role='free_numeric_initializer_only',
                    physical_endpoint_measured=False,source_open_edges=feature.get('open_edges',[]),
                    lane_display_proxy_is_not_measured_endpoint=True),ensure_ascii=False))]))
            target_map[target]=dict(target=target,material_index=mi,path_index=0,geometry_ready=True,
                allowed_paths=[0],basis='actual_typed_source_ice_segment_free_initializer',absolute_calibration_gain_claimed=False)
            added.append(dict(target=target,material_index=mi,geometry_block=bi,category='ice_curve',
                source_shot=sid,source_feature_id=fid,free_numeric_initializer_only=True))
            continue
        if kind!='ad_print':unknown('missing_supported_source_geometry');continue
        if target in PARTIAL_OR_CONFLICTING:unknown(PARTIAL_OR_CONFLICTING[target]);continue
        # The earlier generic ring is superseded by thirteen actual ink shapes.
        # Neither a large ring nor all fragments are a valid whole-target alias.
        if target=='Ad24.print.outermost_visible_circular_print_stroke':
            unknown('legacy_navigation_circle_has_no_proven_alias_to_13_actual_Ad24_ink_shapes');continue
        exp=_exposures(members)
        if len(exp)<2:
            actual_exp={(r['shot_id'],int(r['native_pts'])) for r in members}
            if len(actual_exp)>1:
                unknown('repeated_local_name_without_source_proven_shape_crosswalk')
            else:unknown('single_actual_exposure_without_proven_existing_shape_identity_no_absolute_gain')
            continue
        descriptor_id={(r.get('typed_material_descriptor') or {}).get('advertisement_id') for r in members}
        if len(descriptor_id)!=1:unknown('inconsistent_actual_physical_panel_descriptor');continue
        sid=members[0]['shot_id']; src=sources[sid]; adid=int(d['advertisement_id']); key=d['contour_key']
        candidates=src['contours'].get((adid,key),[])
        if not candidates:unknown('no_actual_source_contour_or_material_template');continue
        # Prefer a native template over a historical XYZ copy of that contour.
        c=next((c for c in candidates if c.get('material_uv') and c.get('source_native_uv')),candidates[0])
        try:
            if c.get('material_uv'):
                vertices,init=_template_chart_vertices(result,adid,c)
            else:
                xyz=c.get('world_xyz_scene_units') or c.get('xyz_scene_units')
                if xyz is None:raise ValueError('no_source_world_chart_numerical_initializer')
                vertices,init=src['board'].station(xyz)
            if len(vertices)>2 and np.linalg.norm(vertices[0]-vertices[-1])<1e-11:vertices=vertices[:-1]
            if len(vertices)<2 or not np.isfinite(vertices).all():raise ValueError('invalid_source_contour_initializer')
            # Projection/retraction is only an initializer. If it leaves the
            # actual board domain, do not clip it into a fabricated observation.
            if np.min(vertices[:,1])<0 or np.max(vertices[:,1])>current.height:
                raise ValueError('source_initializer_outside_current_physical_board_height')
        except (ValueError,KeyError,IndexError) as e:unknown(str(e));continue
        refs=src['refs']+list(input_refs or [])
        packet_refs=sorted({(r['source_packet'],r['source_packet_sha256']) for r in members})
        refs.extend(dict(source_packet_logical_path=p,sha256=h) for p,h in packet_refs)
        native=src['template'].get('advertisements',{}).get(str(adid),{}).get('source',{})
        source_path=c.get('source_path') or c.get('source_frame_path') or native.get('path')
        source_hash=c.get('source_png_sha256') or c.get('source_frame_sha256') or native.get('sha256')
        if source_path:refs.append(dict(path=source_path,sha256=source_hash,role='original_source_contour_exposure'))
        uid='ankara2026/board/Ad'+str(adid)+'/'+key
        component=dict(physical_uid=uid,source_identity_qualified=True,geometry_kind='polyline',
            initial_course_height_vertices=vertices.tolist(),source_exposures=[dict(shot_id=s,native_pts=t) for s,t in exp],
            source_refs=refs,paths=[dict(id='actual_source_boundary',variant_id='actual_source_boundary',
                vertex_indices=list(range(len(vertices))),closed=bool(c.get('closed',False)),
                boundary_role=d.get('contour_role','actual_ink_boundary'),material_key=target)],
            absolute_link_to_existing_material=None)
        components.append(component);component_targets.append((target,init,sid,c,exp))
    # Use the common scene implementation. Every vertex remains shared and free;
    # adding a shape never adds a second private board or advertisement install.
    result,newmap,append_report=append_shared_print_components(result,components)
    for target,init,sid,c,exp in component_targets:
        mi=newmap[target]; target_map[target]=dict(target=target,material_index=mi,path_index=0,
            geometry_ready=True,allowed_paths=[0],basis='repeated_source_same_named_static_ink_free_common_board_shape',
            initializer_only=True,absolute_calibration_gain_claimed=False,existing_shape_identity_claimed=False)
        added.append(dict(target=target,material_index=mi,geometry_block=result['materials'][mi]['paths'][0]['geometry_block'],
            category='ad_print',source_shot=sid,actual_exposures=len(exp),source_contour_id=c.get('id'),
            shared_free_world_shape=True,new_private_advertisement_installation=False,initializer=init,
            source_native_uv_present=bool(c.get('source_native_uv')),geometry_accuracy_accepted=False))
    mapping={}
    for uid,members in sorted(semantic.items()):
        targets={r.get('target_id') for r in members}
        if len(targets)!=1:
            mapping[uid]=dict(material_index=None,geometry_ready=False,reason='semantic_id_has_conflicting_source_targets')
        else:mapping[uid]=copy.deepcopy(target_map.get(next(iter(targets)),dict(material_index=None,geometry_ready=False,
            reason='unbound_actual_material_identity')))
    if result['blocks'][:len(prefix_blocks)]!=prefix_blocks or result['materials'][:len(prefix_materials)]!=prefix_materials:
        raise AssertionError('complete original model prefix changed')
    # Metadata from the complete model must survive byte-for-value. New metadata
    # is added at one key; no rig, optical, clock, or shot container is rewritten.
    for key,value in model.items():
        if key not in ('blocks','materials') and result[key]!=value:raise AssertionError('original metadata changed: '+key)
    row_counts=Counter()
    for r in rows:
        d=mapping.get(r.get('joint_material_id'),{})
        row_counts['geometry_ready_rows' if d.get('geometry_ready') else 'unknown_geometry_rows']+=1
        if d.get('geometry_ready'):row_counts['ready_'+str(r.get('category'))]+=1
    report=dict(schema='rinkfit.first4_actual_materials.v1',semantic_to_material=mapping,row_to_material={},
        source_references=source_refs,input_references=list(input_refs or []),partial_source_overlap_reference=overlap_ref,
        complete_prefix_blocks=len(prefix_blocks),complete_prefix_materials=len(prefix_materials),
        complete_original_prefix_unchanged=True,all_original_metadata_unchanged=True,
        original_source_rows_and_UV_unchanged=True,source_camera_inverse_projection_used=False,
        source_scene_geometry_is_not_truth=True,source_scene_retracted_on_actual_common_board=True,
        source_identity_is_separate_from_numerical_geometry=True,absolute_calibration_gain_claimed=False,
        geometry_ready_is_not_solver_eligibility=True,one_common_continuous_board=True,
        new_private_advertisement_installations=0,shared_print_append=append_report,
        existing_exact_target_bindings=existing,added_materials=added,aliases=aliases,pending_geometry=pending,
        counts=dict(row_counts),total_source_rows=len(rows),total_semantic_materials=len(mapping),
        new_blocks=len(result['blocks'])-len(prefix_blocks),new_materials=len(result['materials'])-len(prefix_materials),
        all_materials_geometry_ready=not pending)
    result['first4_actual_material_compilation']=dict(schema=report['schema'],counts=report['counts'],
        source_references=source_refs,input_references=list(input_refs or []),complete_original_prefix_unchanged=True,
        new_blocks=report['new_blocks'],new_materials=report['new_materials'],absolute_calibration_gain_claimed=False,
        pending_geometry=pending)
    return result,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--binding',required=True)
    p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    model_path=Path(a.model);binding_path=Path(a.binding);out=Path(a.out)
    if file_sha256(model_path)!=MODEL_SHA or file_sha256(binding_path)!=BINDING_SHA:
        raise ValueError('first4 source/model input pin changed; review a new generation explicitly')
    binding=read_json(binding_path);rows_path=Path(binding['rows']['path'])
    if file_sha256(rows_path)!=binding['rows']['sha256']:raise ValueError('source row hash mismatch')
    with gzip.open(rows_path,'rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
    overlap=out.parent/'S03_family_v2/material_overlap_readonly.json'
    refs=[_ref(model_path),_ref(binding_path),_ref(rows_path)]
    compiled,report=compile_first4_materials(read_json(model_path),rows,a.project,input_refs=refs,
        overlap_ref=_ref(overlap) if overlap.is_file() else None)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('model_compiled.json','material_map.json'):
        if (out/name).exists():raise ValueError('immutable compilation output already exists: '+name)
    write_json(out/'model_compiled.json',compiled);write_json(out/'material_map.json',report)
    print(json.dumps(dict(model=_ref(out/'model_compiled.json'),mapping=_ref(out/'material_map.json'),
        blocks=len(compiled['blocks']),materials=len(compiled['materials']),counts=report['counts'],
        added=report['new_materials'],unknown_targets=len(report['pending_geometry'])),ensure_ascii=False))

if __name__=='__main__':main()
