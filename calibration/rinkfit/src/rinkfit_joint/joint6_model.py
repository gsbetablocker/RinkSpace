"""Extend a sealed first-four physical candidate to a six-shot activity view.

The added motion uses semantic block names in the existing complete container.
This changes no source UV, no shared rig/scene values, and no first-four pose.
The S01 stage sidecar freezes current projection dependencies; it does not claim
that this candidate is numerically equivalent to the earlier d78 baseline.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from .io_utils import read_json, write_json, file_sha256, object_sha256

SCOPE = ('s01','s02','s03','s04','s05','s06')

def _ref(p):
    p=Path(p).resolve()
    return dict(file=str(p),sha256=file_sha256(p),bytes=p.stat().st_size)

def _names(model):
    out={b['name']:i for i,b in enumerate(model['blocks'])}
    if len(out)!=len(model['blocks']):
        raise ValueError('Nonunique semantic physical block names')
    return out

def _motion(shot):
    return {int(shot['reference_angles_block']),*map(int,shot['control_blocks']),
            *(int(c['block']) for c in shot.get('control_coordinates',[]))}

def build_joint6_model(parent_model_file, shot_metadata_file, s01_model_file, output,
                       *, screen_state_file, baseline_identity_file):
    parent=read_json(parent_model_file); donor=read_json(shot_metadata_file)
    s01=read_json(s01_model_file); screen=read_json(screen_state_file)
    baseline=read_json(baseline_identity_file); out=Path(output).resolve()
    if out.exists():
        raise FileExistsError('Immutable joint6 activity view output exists')
    if tuple(parent['shots'])!=SCOPE[:4] or set(donor['shots'])!=set('s%02d'%i for i in range(1,10)):
        raise ValueError('Unexpected parent or complete donor shot scope')
    donor_ref=parent['first4_source_generation_activity_view']['base_model']
    if file_sha256(shot_metadata_file)!=donor_ref['sha256']:
        raise ValueError('Complete shot-metadata donor differs from recorded parent')
    if file_sha256(s01_model_file)!=parent['s01_model_sha256']:
        raise ValueError('S01 material identity source changed')
    if parent['source_clock']!=donor['source_clock']:
        raise ValueError('Donor source clock differs')
    clock=parent['source_clock']
    if clock['time_base']!=[1,12800] or clock['frame_step_pts']!=256 or clock['image_size']!=[1920,1080]:
        raise ValueError('Joint6 requires original 50 fps programme PTS')
    if screen['source_video_sha256']!=clock['source_video_sha256']:
        raise ValueError('Screen/source identity differs')
    if baseline['q_sha256']!='8a28afed8f9f896112e5b2f64be515b2bc02019e10e09c7e0eda2598579d4891':
        raise ValueError('Unexpected original baseline identity')
    if set(parent['rigs'])!={'main','secondary','end'}:
        raise ValueError('Parent is not a single three-rig scene')
    model=copy.deepcopy(parent); target=_names(model); source=_names(donor)
    if model['blocks'][model['scene_block']]['name']!='scene/rink5':
        raise ValueError('Shared scene identity changed')
    mapped=[]; activated=[]
    for sid in SCOPE[4:]:
        shot=copy.deepcopy(donor['shots'][sid]); rid=shot['rig_id']
        # The inherited controls are in this exact rig parameter chart. Shared
        # physical values may have improved, but fixed coordinate gauges must match.
        for key in ('R0_seed','n_seed','n_tangent1','n_tangent2','b_seed','f_ref'):
            if donor['rigs'][rid]['camera'][key]!=parent['rigs'][rid]['camera'][key]:
                raise ValueError('Incompatible inherited rig coordinate chart: '+rid+'/'+key)
        mapping={}
        for old in sorted(_motion(shot)):
            name=donor['blocks'][old]['name']
            if name not in target:
                raise ValueError('Retained complete motion block missing: '+name)
            new=target[name]; mapping[old]=new
            a=donor['blocks'][old]; b=model['blocks'][new]
            if any(a.get(k)!=b.get(k) for k in ('values','lower','upper','scales')):
                raise ValueError('Retained motion numeric content changed: '+name)
            if not name.startswith('shot/'+sid+'/'):
                raise ValueError('Added motion would overwrite a shared physical block')
            before=dict(constant=b.get('constant',False),fixed_indices=b.get('fixed_indices',[]))
            for key in ('constant','fixed_indices'):
                if key in a:b[key]=copy.deepcopy(a[key])
                else:b.pop(key,None)
            after=dict(constant=b.get('constant',False),fixed_indices=b.get('fixed_indices',[]))
            activated.append(dict(name=name,block=new,previous=before,restored=after))
            mapped.append(dict(shot=sid,name=name,source_index=old,target_index=new))
        shot['reference_angles_block']=mapping[int(shot['reference_angles_block'])]
        shot['control_blocks']=[mapping[int(i)] for i in shot['control_blocks']]
        for coordinate in shot.get('control_coordinates',[]):coordinate['block']=mapping[int(coordinate['block'])]
        model['shots'][sid]=shot
    if set(model['shots'])!=set(SCOPE):raise ValueError('Unexpected expanded shot scope')
    # Construct the complete current S01 projection closure from all original
    # d78 materials, shared camera/scene, and every effective/storage control.
    material_map={x['label']:x for x in model['materials']}
    if len(material_map)!=len(model['materials']):raise ValueError('Ambiguous material labels')
    protected={int(model['scene_block']),int(model['rigs']['main']['head_block']),
               int(model['rigs']['main']['optics_block']),*_motion(model['shots']['s01'])}
    material_names=[]
    for original in s01['materials']:
        label=original['label']
        if label not in material_map:raise ValueError('S01 material lost: '+label)
        material_names.append(label)
        for path in material_map[label]['paths']:
            for key,value in path.items():
                if key.endswith('_block') and isinstance(value,int) and value>=0:
                    if value>=len(model['blocks']):raise ValueError('Dangling S01 path block')
                    protected.add(value)
    # Invariant proof is exact on source-independent numeric projection inputs.
    if model['rigs']!=parent['rigs'] or model['materials']!=parent['materials']:
        raise ValueError('Shared rig or material content changed')
    if model['linear_control_manifolds']!=parent['linear_control_manifolds']:
        raise ValueError('Original S01 time subspace changed')
    for sid in SCOPE[:4]:
        if model['shots'][sid]!=parent['shots'][sid]:raise ValueError('First-four shot metadata changed')
    for i,block in enumerate(parent['blocks']):
        if block['values']!=model['blocks'][i]['values']:raise ValueError('Existing physical value changed')
    projection_hash=object_sha256(dict(scene=model['scene_block'],rig=model['rigs']['main'],
        shot=model['shots']['s01'],blocks=[dict(index=i,values=model['blocks'][i]['values']) for i in sorted(protected)],
        materials=[material_map[x] for x in material_names],manifolds=model['linear_control_manifolds']))
    protection=dict(schema='rinkfit.joint6_initial_S01_projection_protection.v1',
        temporary_runtime_fixed_physical_blocks=[model['blocks'][i]['name'] for i in sorted(protected)],
        temporary_runtime_fixed_nuisance_shots=['s01'],complete_material_labels=material_names,
        protected_blocks=len(protected),projection_input_sha256=projection_hash,
        protects_current_parent_projection=True,original_d78_forward_equivalence_claimed=False,
        production_registered_dependency_audit_still_required=True,
        unlock_policy='Candidate copies only, with same-source S01/full-six comparison; this is an initial stage sidecar.')
    provenance=dict(schema='rinkfit.joint6_complete_container_activity_view.v1',
        parent=_ref(parent_model_file),complete_motion_donor=_ref(shot_metadata_file),
        s01_material_identity=_ref(s01_model_file),screen_navigation=_ref(screen_state_file),
        original_8a28_baseline_identity=_ref(baseline_identity_file),
        source_clock=clock,shots=list(SCOPE),rig_count=3,scene_count=1,
        added_motion_semantic_mapping=mapped,restored_motion_activity=activated,
        all_existing_physical_values_exact=True,all_first4_projection_inputs_exact=True,
        inherited_s05_s06_motion_requires_new_source_capture=True,
        no_old_nuisance_Query_W_AL_state_resumed=True,source_UV_changed=False,
        source_generation_or_packet_created=False,physical_quality_accepted=False)
    model['joint6_source_activity_view']={k:copy.deepcopy(v) for k,v in provenance.items()
        if k in ('schema','parent','complete_motion_donor','source_clock','shots','inherited_s05_s06_motion_requires_new_source_capture')}
    out.mkdir(parents=True)
    write_json(out/'model6.json',model)
    write_json(out/'s01_initial_protection.json',protection)
    provenance['model']=_ref(out/'model6.json');provenance['initial_S01_protection']=_ref(out/'s01_initial_protection.json')
    provenance['implementation']=_ref(__file__)
    write_json(out/'model6_receipt.json',provenance)
    return provenance

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--complete-motion-donor',required=True)
    p.add_argument('--s01-model',required=True);p.add_argument('--screen',required=True);p.add_argument('--baseline-identity',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(build_joint6_model(a.parent,a.complete_motion_donor,a.s01_model,a.output,
        screen_state_file=a.screen,baseline_identity_file=a.baseline_identity),ensure_ascii=False))

if __name__=='__main__':main()
