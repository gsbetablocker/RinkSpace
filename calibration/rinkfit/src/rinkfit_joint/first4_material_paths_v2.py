"""Resolve source-typed finite path domains; keep frozen v1 model/map intact."""
from __future__ import annotations
import argparse
import copy
import gzip
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .io_utils import read_json,write_json,file_sha256,object_sha256
from rinkfit_start.native_scene import NativeScene

BOARD_SEGMENTS={'lower_long_side':0,'lower_right':1,'right_end':2,'upper_right':3,
                'upper_long_side':4,'upper_left':5,'left_end':6,'lower_left':7}


def _ref(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=file_sha256(path),bytes=path.stat().st_size)


def _provenance(path):
    try:return json.loads(path.get('provenance','{}'))
    except (TypeError,ValueError):return {}


def resolve_typed_paths(material,descriptors):
    """Return only branches of the actual typed entity, never all board faces."""
    if not descriptors:raise ValueError('source_typed_descriptor_missing')
    kinds={d.get('kind') for d in descriptors}
    if len(kinds)!=1:raise ValueError('conflicting_source_descriptor_kind')
    kind=next(iter(kinds));paths=material['paths']
    if kind=='board_edge':
        roles={d.get('role') for d in descriptors};parts={d.get('part') for d in descriptors}
        if len(roles)!=1 or len(parts)!=1:raise ValueError('conflicting_source_board_role_or_part')
        role=next(iter(roles));part=next(iter(parts))
        if role not in ('top','contact') or part not in BOARD_SEGMENTS:
            raise ValueError('unresolved_source_board_role_or_part')
        segment=BOARD_SEGMENTS[part];height=1. if role=='top' else 0.
        allowed=[i for i,p in enumerate(paths) if p.get('kind') in (3,'board_boundary')
                 and p.get('boundary_segment')==segment and p.get('fixed_height_fraction')==height]
        if not allowed:raise ValueError('no_finite_path_for_actual_source_board_part_and_height')
        return allowed,dict(basis='typed_source_board_part_and_role_exact_boundary_segment',
            typed_part=part,typed_role=role,boundary_segment=segment,fixed_height_fraction=height,
            complete_perimeter_unlocked=False,screen_nearest_branch_used=False)
    if len(paths)==1:
        return [0],dict(basis='single_finite_path_of_source_typed_material',screen_nearest_branch_used=False)
    if kind=='ice_curve':
        curves={d.get('curve_id') for d in descriptors}
        if len(curves)!=1:raise ValueError('conflicting_source_ice_curve')
        # A named half-circle boundary is its explicit arc plus closing diameter.
        # This union has one common geometry block and remains entirely on ice.
        curve=next(iter(curves));pdesc=[_provenance(p).get('source_descriptor',{}) for p in paths]
        if {p.get('kind') for p in paths}=={4,5} and len({p.get('geometry_block') for p in paths})==1 \
                and all(d.get('curve_id')==curve for d in pdesc):
            return list(range(len(paths))),dict(basis='explicit_source_half_circle_boundary_arc_and_diameter_union',
                curve_id=curve,common_geometry_block=paths[0]['geometry_block'],screen_nearest_branch_used=False)
        raise ValueError('multi_path_ice_curve_has_no_explicit_source_union')
    if kind in ('ad_print','ad_print_curve'):
        keys={d.get('contour_key') for d in descriptors}
        if len(keys)!=1:raise ValueError('conflicting_source_ink_contour_key')
        key=next(iter(keys));proof=[_provenance(p) for p in paths]
        # The established source crosswalk explicitly maps one source contour to
        # these artwork paths, e.g. Ad7 r2_ISU_04 -> logo_part04 + logo_part05.
        if all(p.get('current_source_contour')==key and p.get('existing_actual_paths') for p in proof):
            return list(range(len(paths))),dict(basis='existing_source_crosswalk_explicit_actual_ink_path_union',
                contour_key=key,source_actual_path_ids=sorted({v for p in proof for v in p['existing_actual_paths']}),
                screen_nearest_branch_used=False)
        # Otherwise require a source topology path list. Internal holes are not
        # silently included in an exterior contour just because a label matches.
        explicit={v for d in descriptors for v in (d.get('curve_topology') or {}).get('path_ids',[])}
        if explicit:
            allowed=[i for i,p in enumerate(paths) if p.get('id','').split('/')[-1] in explicit]
            if allowed:return allowed,dict(basis='source_explicit_ink_topology_path_ids',source_path_ids=sorted(explicit),
                screen_nearest_branch_used=False)
        same=[i for i,p in enumerate(paths) if p.get('source_contour_id')==key]
        if same:return same,dict(basis='exact_source_contour_id',contour_key=key,screen_nearest_branch_used=False)
        raise ValueError('multiple_print_paths_without_source_explicit_topology_domain')
    raise ValueError('multiple_paths_without_source_semantic_domain')


def repair_material_map(model,mapping,rows):
    result=copy.deepcopy(mapping);groups=defaultdict(list)
    for row in rows:
        if row.get('joint_material_id'):groups[row['joint_material_id']].append(row)
    changes=[];checks=[];failures=[];counts=Counter();scene=NativeScene(model)
    for uid,desc in sorted(result['semantic_to_material'].items()):
        if not desc.get('geometry_ready'):continue
        members=groups[uid];mi=int(desc['material_index']);mat=model['materials'][mi]
        descriptors=[r.get('typed_material_descriptor') or {} for r in members]
        try:
            allowed,proof=resolve_typed_paths(mat,descriptors)
            if not allowed or any(i<0 or i>=len(mat['paths']) for i in allowed):raise ValueError('illegal_finite_path_indices')
            point=any(r.get('point_correspondence') or r.get('category')=='ad_corner' for r in members)
            if point and any(not mat['paths'][i].get('physical_point') for i in allowed):
                raise ValueError('point_correspondence_is_not_a_fixed_physical_point')
            old=dict(path_index=desc.get('path_index'),allowed_paths=desc.get('allowed_paths'))
            # Initialization selects a legal branch only. When the physical
            # source entity is an explicit union, the solver may evaluate that
            # finite union; it may never change the source surface/board part.
            seed=desc.get('path_index',allowed[0]);seed=seed if seed in allowed else allowed[0]
            desc.update(path_index=seed,allowed_paths=allowed,path_domain_proof=proof,
                        finite_station_interval=[0.,1.],path_domain_version=2)
            finite=[]
            for pi in allowed:
                path=mat['paths'][pi]
                if path['kind'] in (4,5):
                    # NativeScene currently omits these evaluator cases. Validate
                    # the common mathematical shape directly without changing it.
                    q=np.asarray(model['blocks'][path['geometry_block']]['values'],float)
                    t=np.linspace(0.,1.,129);angles=q[3]+np.pi*t
                    if path['kind']==4:xy=q[:2]+q[2]*np.c_[np.cos(angles),np.sin(angles)]
                    else:xy=q[:2]+q[2]*(1.-2.*t[:,None])*np.array([np.cos(q[3]),np.sin(q[3])])
                    xyz=np.c_[xy,np.zeros(len(xy))]
                else:xyz=scene.world(mi,pi,np.linspace(0.,1.,129))
                if not np.isfinite(xyz).all():raise ValueError('nonfinite_finite_path_world_evaluation')
                finite.append(dict(path_index=pi,path_id=path['id'],kind=path['kind'],samples=129,finite=True,
                    boundary_segment=path.get('boundary_segment'),fixed_height_fraction=path.get('fixed_height_fraction')))
            rec=dict(semantic_id=uid,target=desc.get('target'),material_index=mi,rows=len(members),
                source_descriptor_sha256=object_sha256(descriptors),old=old,new=dict(path_index=seed,allowed_paths=allowed),
                proof=proof,finite_paths=finite,point_correspondence=point)
            checks.append(rec);counts['checked_geometry_ready_semantic_ids']+=1
            if old!=rec['new']:changes.append(rec);counts['changed_semantic_ids']+=1;counts['changed_source_rows']+=len(members)
            if len(mat['paths'])>1:counts['consumed_multi_path_semantic_ids']+=1
        except (ValueError,IndexError,KeyError) as e:
            desc.update(geometry_ready=False,path_index=None,allowed_paths=[],reason='finite_path_domain_unresolved:'+str(e))
            failures.append(dict(semantic_id=uid,reason=str(e),rows=len(members)));counts['unresolved_semantic_ids']+=1
    catalog=[]
    consumed={v['material_index'] for v in checks}
    for mi,mat in enumerate(model['materials']):
        if len(mat['paths'])>1:catalog.append(dict(material_index=mi,label=mat['label'],consumed_by_source14=mi in consumed,
            paths=[dict(index=i,id=p['id'],kind=p['kind'],boundary_segment=p.get('boundary_segment'),
                fixed_height_fraction=p.get('fixed_height_fraction'),physical_point=p.get('physical_point'),
                source_contour_id=p.get('source_contour_id'),material_role=p.get('material_role')) for i,p in enumerate(mat['paths'])]))
    report=dict(schema='rinkfit.first4.source_typed_path_domains.v2',counts=dict(counts),changes=changes,
        unresolved=failures,all_consumed_path_checks=checks,multi_path_model_catalog=catalog,
        source_UV_ID_PTS_changed=False,source_surfaces_changed=False,model_changed=False,
        branches_selected_from_source_descriptor=True,screen_nearest_branch_used=False,
        whole_perimeter_unlocked=False,finite_station_interval=[0.,1.],rootW_or_query_observations_rewritten=False)
    result['schema']='rinkfit.first4_actual_materials.path_domains_v2'
    result['finite_path_domain_repair_v2']=report
    return result,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--mapping',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    model=read_json(a.model);base=read_json(a.mapping);binding=read_json(a.binding);rowsref=binding['rows']
    if file_sha256(rowsref['path'])!=rowsref['sha256']:raise ValueError('source row identity changed')
    with gzip.open(rowsref['path'],'rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
    result,report=repair_material_map(model,base,rows)
    refs=dict(model=_ref(a.model),frozen_v1_map=_ref(a.mapping),source_binding=_ref(a.binding),source_rows=rowsref)
    result['finite_path_domain_repair_v2']['input_references']=refs
    out=Path(a.out);audit=out.with_name('material_path_audit_v2.json')
    if out.exists() or audit.exists():raise ValueError('immutable v2 material map output already exists')
    write_json(out,result);write_json(audit,report)
    print(json.dumps(dict(mapping=_ref(out),audit=_ref(audit),counts=report['counts'],unresolved=report['unresolved']),ensure_ascii=False))

if __name__=='__main__':main()
