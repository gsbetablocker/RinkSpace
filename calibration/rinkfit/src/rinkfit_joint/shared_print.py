"""Source-identified print components on the one continuous unfolded board.
All vertices are world chart parameters shared by the physical UID, never per exposure.
Source image contours and inverse projection are numerical initialization only.
"""
from __future__ import annotations
import copy,hashlib
import numpy as np

def append_shared_print_components(model,components):
    result=copy.deepcopy(model);records={d['physical_uid']:d for d in result.get('shared_print_components',[])};mapping={};reports=[];height=float(result['blocks'][result['scene_block']]['values'][4])
    for c in components:
        uid=str(c['physical_uid']);exposures={(str(d['shot_id']),int(d['native_pts'])) for d in c['source_exposures']}
        if not c.get('source_identity_qualified') or len(exposures)<2 or not c.get('source_refs'):raise ValueError('shared print component requires qualified same-ink identity at two actual exposures and source references: '+uid)
        kind=c.get('geometry_kind','polyline')
        if uid in records:
            old=records[uid]
            if old['geometry_kind']!=kind:raise ValueError('physical UID shape family changed: '+uid)
            mapping.update(old['material_keys']);reports.append(dict(physical_uid=uid,idempotent=True,existing_numeric_state_preserved=True));continue
        tag=hashlib.sha256(uid.encode()).hexdigest()[:24];bi=len(result['blocks']);material_keys={}
        if kind=='polyline':
            vertices=np.asarray(c['initial_course_height_vertices'],float)
            if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<2 or not np.isfinite(vertices).all():raise ValueError('finite worldchart initialization vertices required')
            if np.min(vertices[:,1])<0 or np.max(vertices[:,1])>height:raise ValueError('numeric print initialization leaves actual board height domain')
            values=vertices.ravel().tolist();lower=[v for _ in vertices for v in (-16.,0.)];upper=[v for _ in vertices for v in (16.,.1)]
            paths=c.get('paths') or [dict(id='outer',vertex_indices=list(range(len(vertices))),closed=True,boundary_role='outer')]
            for d in paths:
                indices=list(map(int,d['vertex_indices']))
                if len(indices)<2 or min(indices)<0 or max(indices)>=len(vertices):raise ValueError('finite shared vertex index domain')
                key=d.get('material_key','print_component.'+uid+'|board|ink_boundary|'+d.get('boundary_role',d['id']));pi=dict(id=d['id'],variant_id=d.get('variant_id',d['id']),kind=6,geometry_block=bi,geometry_indices=[j for i in indices for j in (2*i,2*i+1)],closed=bool(d.get('closed',False)),physical_point=False,provenance='source-confirmed shared physical UID; source contour only numerical initialization');mi=len(result['materials']);result['materials'].append(dict(label=key,paths=[pi]));material_keys[key]=mi
        elif kind=='ellipse':
            if not c.get('ellipse_geometry_confirmed') or not c.get('shape_authority_refs'):raise ValueError('ellipse requires actual/catalog authority; apparent image roundness is not a world shape fact')
            values=np.asarray(c['initial_ellipse'],float).tolist()
            if len(values)!=5 or not np.isfinite(values).all() or min(values[2],values[4])<=0 or values[1]-values[4]<0 or values[1]+values[4]>height:raise ValueError('finite positive ellipse initialization within actual board height required')
            lower=[-16.,0.,1e-12,-16.,1e-12];upper=[16.,.1,16.,16.,.1];key=c.get('material_key','print_component.'+uid+'|board|ink_boundary|outer');mi=len(result['materials']);result['materials'].append(dict(label=key,paths=[dict(id='outer',variant_id='outer',kind=7,geometry_block=bi,closed=True,physical_point=False,start_angle=0.,end_angle=2*np.pi,provenance='source/catalog-confirmed ellipse boundary; five world parameters without phase gauge')]));material_keys[key]=mi
        else:raise ValueError('unknown shared print geometry kind')
        result['blocks'].append(dict(name='print_component/'+kind+'/'+tag,values=values,lower=lower,upper=upper,scales=[max(abs(v),.01) for v in values],constant=False,fixed_indices=[]));record=dict(physical_uid=uid,geometry_kind=kind,geometry_block=bi,material_keys=material_keys,source_exposures=[dict(shot_id=s,native_pts=t) for s,t in sorted(exposures)],source_refs=c['source_refs'],shape_authority_refs=c.get('shape_authority_refs',[]),absolute_link_to_existing_material=c.get('absolute_link_to_existing_material'),one_physical_UID_has_one_shared_world_shape=True,source_atlas_is_not_a_fixed_world_template=True,absolute_camera_calibration_gain_claimed=False);records[uid]=record;mapping.update(material_keys);reports.append(dict(physical_uid=uid,geometry_block=bi,material_keys=material_keys,source_exposures=len(exposures),absolute_camera_calibration_gain_claimed=False))
    result['shared_print_components']=list(records.values());report=dict(schema='rinkfit.shared_physical_print_append.v1',components=reports,old_numeric_blocks_and_materials_unchanged=result['blocks'][:len(model['blocks'])]==model['blocks'] and result['materials'][:len(model['materials'])]==model['materials'],new_source_UV_or_shape_regularization_added=False,no_Ad_number_or_brand_binding_inferred=True,absolute_camera_calibration_gain_claimed=False)
    return result,mapping,report
