"""Reviewable numerical donors without source, topology, or optimizer-state mutation.

These constructors propose initial values only. They do not authorize recalled
observations, accept an AL step, or import any checkpoint counters/duals.
"""
from __future__ import annotations
import copy
import numpy as np


def _by_name(model):
    result={b['name']:i for i,b in enumerate(model['blocks'])}
    if len(result)!=len(model['blocks']):
        raise ValueError('ambiguous named physical block')
    return result


def _set_values(model,index,values):
    block=model['blocks'][index]
    x=np.asarray(values,dtype=float)
    if x.shape!=np.asarray(block['values']).shape or not np.isfinite(x).all():
        raise ValueError('invalid donor block '+block['name'])
    if np.any(x<np.asarray(block['lower'])) or np.any(x>np.asarray(block['upper'])):
        raise ValueError('donor outside current box '+block['name'])
    block['values']=x.tolist()


def print_physical_coefficients(ad,affine):
    a=np.asarray(ad,float);q=np.asarray(affine,float);height=a[3]-a[2]
    if a.shape!=(4,) or q.shape!=(6,) or a[1]<=0 or height<=0:
        raise ValueError('invalid physical board affine chart')
    return np.r_[a[1]*q[:2],a[0]+a[1]*q[2],height*q[3:5],a[2]+height*q[5]]


def rebase_print_physical_coefficients(physical,ad):
    v=np.asarray(physical,float);a=np.asarray(ad,float);height=a[3]-a[2]
    if v.shape!=(6,) or a[1]<=0 or height<=0:
        raise ValueError('invalid target board affine chart')
    return np.r_[v[:2]/a[1],(v[2]-a[0])/a[1],v[3:5]/height,(v[5]-a[2])/height]


def _linear_membership(model):
    records=[]
    for chart in model.get('linear_control_manifolds',[]):
        v=np.asarray(model['blocks'][chart['block']]['values'])
        C=np.asarray(chart['constraint_matrix'])
        error=float(np.max(np.abs(C@v))) if C.size else 0.
        if error>2e-10:
            raise ValueError('donor leaves existing exact control subspace')
        records.append({'block':chart['block'],'ambient_size':len(v),'membership_error':error})
    return records


def _changes(before,after):
    return [dict(block=i,name=a['name'],before=a['values'],after=b['values'])
            for i,(a,b) in enumerate(zip(before['blocks'],after['blocks']))
            if a['values']!=b['values']]


def original_s01_physical_motion_donor(current,original,*,shot_id='s01'):
    """Restore one complete exported motion and old physical composites.

    Current material/path definitions, domains, freedoms, new geometry blocks,
    measurement widths, other cameras, and print-only chart anchors stay intact.
    Constant print-only anchors are coordinates, so their original redundant
    values are never copied; each old affine is rebuilt from its physical
    unfolded coefficients in the current anchor.
    """
    if current['schema']!='rinkfit.joint_event_model.v1' or original['schema']!='rinkfit.event_model.v1.1':
        raise ValueError('explicit joint/event model adapters required')
    q=copy.deepcopy(current);oi=_by_name(original);ci=_by_name(current)
    mapping={int(k):int(v) for k,v in current['s01_block_map'].items()}
    if len(set(mapping.values()))!=len(mapping):raise ValueError('nonunique original physical mapping')
    chart_blocks=set()
    for ai,b in enumerate(current['blocks']):
        if not b['name'].startswith('ad/') or not b.get('constant',False):continue
        paths=[(m,p) for m in current['materials'] for p in m['paths'] if p.get('advertisement_block')==ai]
        if paths and all(p['kind']==2 and p.get('geometry_block',-1)>=0 and '.print.' in m['label'] for m,p in paths):
            chart_blocks.add(ai)
    _set_values(q,current['scene_block'],original['blocks'][0]['values'][:5])
    coordinate_maps=[]
    for source,target in mapping.items():
        a=original['blocks'][source];b=q['blocks'][target]
        if a['name']!=b['name']:raise ValueError('physical name mismatch in original map')
        if target in chart_blocks:continue
        if len(a['values'])==len(b['values']):
            _set_values(q,target,a['values'])
            coordinate_maps.append(dict(name=b['name'],source_block=source,current_block=target,source_to_current_slots={str(i):i for i in range(len(a['values']))}))
        elif a['name'].startswith('ice/circle/'):
            # New measured interfaces may have inserted radii. Match named
            # physical material roles, never radius storage positions.
            slots={0:0,1:1}
            old_materials={m['label']:m for m in original['materials']}
            for material in current['materials']:
                om=old_materials.get(material['label'])
                if om is None:continue
                for path in material['paths']:
                    if path.get('geometry_block')!=target or path['kind']!=1:continue
                    op=next((p for p in om['paths'] if p['id']==path['id'] and p['kind']==1),None)
                    if op is None or op.get('geometry_block')!=source:continue
                    old_slot=int(op.get('radius_index',2));new_slot=int(path.get('radius_index',2))
                    if old_slot in slots and slots[old_slot]!=new_slot:raise ValueError('ambiguous physical radius role')
                    slots[old_slot]=new_slot
            if set(slots)!=set(range(len(a['values']))):raise ValueError('incomplete original radius-role map')
            value=list(b['values'])
            for source_slot,target_slot in slots.items():value[target_slot]=a['values'][source_slot]
            _set_values(q,target,value)
            coordinate_maps.append(dict(name=b['name'],source_block=source,current_block=target,source_to_current_slots={str(k):v for k,v in slots.items()},new_current_slots_kept=sorted(set(range(len(value)))-set(slots.values()))))
        else:raise ValueError('unmapped changed old physical block shape '+a['name'])
    # Original geometry ownership is proved from the original path table.
    print_owners={}
    for material in original['materials']:
        for path in material['paths']:
            gi=path.get('geometry_block',-1);ai=path.get('advertisement_block',-1)
            if path['kind']==2 and gi>=0:
                print_owners.setdefault(gi,set()).add(ai)
    proofs=[]
    for old_gi,owners in print_owners.items():
        if len(owners)!=1 or next(iter(owners))<0:raise ValueError('ambiguous original print chart')
        old_ai=next(iter(owners));gi=mapping[old_gi];ai=mapping[old_ai]
        new_owners={p.get('advertisement_block',-1) for m in current['materials'] for p in m['paths'] if p.get('geometry_block')==gi}
        if new_owners!={ai}:raise ValueError('current print owner differs from original mapping')
        physical=print_physical_coefficients(original['blocks'][old_ai]['values'],original['blocks'][old_gi]['values'])
        affine=rebase_print_physical_coefficients(physical,q['blocks'][ai]['values'])
        _set_values(q,gi,affine)
        error=float(np.max(np.abs(print_physical_coefficients(q['blocks'][ai]['values'],affine)-physical)))
        if error>1e-13:raise ValueError('physical affine rebase did not preserve original composite')
        proofs.append(dict(block=gi,name=q['blocks'][gi]['name'],physical_coefficients=physical.tolist(),max_delta=error,current_chart_anchor_kept=ai in chart_blocks))
    shot=q['shots'][shot_id];cam=original['camera']
    rig=current['rigs'][shot['rig_id']]
    if np.any(np.asarray(current['blocks'][shot['reference_angles_block']]['values'])!=0):raise ValueError('original reference motion needs its exact zero-offset gauge')
    for key in ('R0_seed','n_seed','b_seed','n_tangent1','n_tangent2','f_ref','tau_bound'):
        if rig['camera'][key]!=cam[key]:raise ValueError('original camera coordinate seed differs: '+key)
    keys=('dt','first_control_time','n_controls','dependent_control','reference_indices','reference_weights')
    if any(shot['camera'][key]!=cam[key] for key in keys):raise ValueError('original motion basis/gauge differs')
    entries=shot.get('control_coordinates',[dict(block=i,offset=0) for i in shot['control_blocks']])
    if len(entries)!=cam['n_controls']:raise ValueError('partial motion donor forbidden')
    begin=cam.get('control_block_begin',1)
    for j,(entry,storage) in enumerate(zip(entries,shot['control_blocks'])):
        value=original['blocks'][begin+j]['values'];bi=entry['block'];off=entry['offset']
        v=list(q['blocks'][bi]['values']);v[off:off+3]=value;_set_values(q,bi,v)
        if storage!=bi:_set_values(q,storage,value)
    # All ownership flags and all non-value metadata are inherited literally.
    changed=_changes(current,q);allowed=set(mapping.values())|{current['scene_block']}|set(shot['control_blocks'])|{e['block'] for e in entries}
    if any(x['block'] not in allowed for x in changed):raise ValueError('unrequested new parameter changed')
    for ai in chart_blocks:
        if q['blocks'][ai]!=current['blocks'][ai]:raise ValueError('print-only redundant anchor was overwritten')
    for a,b in zip(q['blocks'],current['blocks']):
        if {k:v for k,v in a.items() if k!='values'}!={k:v for k,v in b.items() if k!='values'}:
            raise ValueError('donor changed a parameter domain/freedom')
    report=dict(schema='rinkfit.complete_original_s01_numeric_donor.v1',changes=changed,
                complete_motion_controls=len(entries),old_physical_blocks=len(mapping),
                old_physical_coordinate_maps=coordinate_maps,
                original_motion_clock=dict(time_origin_pts=shot['time_origin_pts'],scope=copy.deepcopy(shot['scope']),basis=copy.deepcopy(shot['camera']),mathematical_local_domain=[cam['first_control_time']+cam['dt'],cam['first_control_time']+(cam['n_controls']-2)*cam['dt']],allowed_readout_half_seconds=rig['camera']['tau_bound']/2),
                original_print_physical_coefficients=proofs,print_only_current_anchor_blocks=sorted(chart_blocks),
                linear_control_membership=_linear_membership(q),
                material_topology_domains_and_measurement_widths_unchanged=True,
                new_independent_parameter_values_unchanged=True,shared_heads_optics_other_shot_motion_unchanged=True,
                source_or_checkpoint_not_written=True,full_AL_acceptance_performed=False,
                source_qualification_not_granted=True)
    return q,report


def all_camera_numeric_donor(current,donor):
    """Transfer whole rig/shot numeric values only under identical clock/bases."""
    q=copy.deepcopy(current);di=_by_name(donor);changed_names=[]
    if set(q['shots'])!=set(donor['shots']) or set(q['rigs'])!=set(donor['rigs']):
        raise ValueError('partial camera donor forbidden')
    for sid,s in q['shots'].items():
        d=donor['shots'][sid]
        for key in ('rig_id','scope','time_origin_pts','camera'):
            if s[key]!=d[key]:raise ValueError('camera donor basis or source clock differs: '+sid+'/'+key)
    for rid,rig in q['rigs'].items():
        if rig['camera']!=donor['rigs'][rid]['camera']:raise ValueError('rig seed/allowed optical domain differs')
    for i,b in enumerate(q['blocks']):
        if not b['name'].startswith(('rig/','shot/')):continue
        if b['name'] not in di:raise ValueError('missing complete camera block '+b['name'])
        d=donor['blocks'][di[b['name']]]
        _set_values(q,i,d['values']);changed_names.append(b['name'])
    return q,dict(schema='rinkfit.complete_all_camera_numeric_donor.v1',
                  changes=_changes(current,q),complete_camera_block_count=len(changed_names),
                  complete_shots=sorted(q['shots']),complete_rigs=sorted(q['rigs']),
                  linear_control_membership=_linear_membership(q),
                  camera_bases_seeds_clock_and_allowed_domains_unchanged=True,
                  current_geometry_measurement_widths_and_source_structure_unchanged=True,
                  checkpoint_metadata_and_duals_not_imported=True,full_AL_acceptance_performed=False)
