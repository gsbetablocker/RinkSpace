"""Restore explicit continuous-board candidate domains without changing source rows.

v2 mistook a source guide's contextual part for a measured finite-branch identity.
This opt-in successor consumes only the candidate union already present in each
material's frozen association policy. It never grows a curve or uses a residual
as an identity measurement. Other material maps and the S01 packet are preserved.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from rinkfit_start.native_scene import NativeScene
from .first4_material_paths_v2 import BOARD_SEGMENTS, _provenance, _ref
from .io_utils import read_json, write_json, object_sha256
from .source_generation_input_v2 import QualifiedSourceGenerationInput


def context_board_union(material, descriptors, scene, material_index):
    if not descriptors or any(d.get('kind') != 'board_edge' for d in descriptors):
        raise ValueError('source_board_descriptor_missing')
    roles = {d.get('role') for d in descriptors}
    if len(roles) != 1 or not roles <= {'top', 'contact'}:
        raise ValueError('source_board_role_conflict')
    role = next(iter(roles)); height = 1. if role == 'top' else 0.
    paths = material['paths']; policies = []
    for p in paths:
        pol = _provenance(p).get('descriptor', {}).get('association_policy')
        if pol: policies.append(pol)
    if not policies:
        return None, dict(basis='no_existing_explicit_context_candidate_policy',
                          existing_path_count=len(paths), typed_parts=sorted({d.get('part') for d in descriptors}))
    if len(policies) != len(paths) or len({object_sha256(p) for p in policies}) != 1:
        raise ValueError('inconsistent_frozen_board_candidate_policy')
    policy = policies[0]
    if policy.get('original_part_is_context_not_a_hard_correspondence') is not True:
        return None, dict(basis='source_part_has_no_explicit_context_policy')
    if policy.get('edge_role_is_preserved') is not True or policy.get('candidate_parts_are_not_new_source_observations') is not True:
        raise ValueError('unqualified_board_context_policy')
    parts = policy['candidate_parts']; excluded = set(policy.get('excluded_opposite_parts', []))
    if not parts or len(parts) != len(set(parts)) or set(parts) & excluded:
        raise ValueError('invalid_existing_candidate_part_domain')
    segments = [BOARD_SEGMENTS[part] for part in parts]
    if len(segments) == 8 or any((b-a) % 8 != 1 for a,b in zip(segments,segments[1:])):
        raise ValueError('candidate_domain_is_not_an_open_connected_branch_chain')
    allowed = []
    for part, seg in zip(parts, segments):
        matches = [i for i,p in enumerate(paths) if p.get('kind') in (3,'board_boundary')
                   and p.get('boundary_segment') == seg and p.get('fixed_height_fraction') == height]
        if len(matches) != 1:
            raise ValueError('existing_candidate_branch_missing_or_wrong_height')
        allowed.extend(matches)
    if set(allowed) != set(range(len(paths))):
        raise ValueError('material_contains_paths_outside_its_explicit_policy')
    xyz = [np.asarray(scene.world(material_index, pi, np.array([0.,1.])),float) for pi in allowed]
    if not all(x.shape == (2,3) and np.isfinite(x).all() for x in xyz):
        raise ValueError('nonfinite_existing_board_branch')
    gaps = [float(np.linalg.norm(a[-1]-b[0])) for a,b in zip(xyz,xyz[1:])]
    if max(gaps, default=0.) > 1e-9:
        raise ValueError('existing_candidate_branches_do_not_share_physical_endpoints')
    proof = dict(basis='frozen_material_continuous_board_context_candidate_union',
        source_part_is_context=True, typed_parts=sorted({d.get('part') for d in descriptors}),
        typed_role=role, fixed_height_fraction=height, candidate_parts=parts,
        boundary_segments=segments, excluded_opposite_parts=sorted(excluded),
        complete_perimeter_unlocked=False, candidate_parts_are_new_source_observations=False,
        screen_nearest_branch_used=False, source_UV_used_to_choose_domain=False,
        all_paths_preexisting=True, finite_station_interval=[0.,1.],
        connected_model_endpoint_gaps_L=gaps, frozen_association_policy=policy,
        historical_policy_frame_range_is_not_new_shot_pixel_evidence=True)
    return allowed, proof


def repair_context_board_map(model, mapping, rows):
    rows=list(rows); result=copy.deepcopy(mapping); groups=defaultdict(list)
    for r in rows:
        if (r.get('typed_material_descriptor') or {}).get('kind') == 'board_edge':
            groups[r['joint_material_id']].append(r)
    scene=NativeScene(model); checks=[]; delta=[]; counts=Counter()
    for uid,members in sorted(groups.items()):
        desc=result['semantic_to_material'][uid]
        if not desc.get('geometry_ready'):raise ValueError('existing_board_material_not_ready:'+uid)
        mi=int(desc['material_index']); material=model['materials'][mi]
        if any(r['physical_surface'] != 'board' for r in members):
            raise ValueError('board_context_union_changes_source_surface')
        allowed,proof=context_board_union(material,[r['typed_material_descriptor'] for r in members],scene,mi)
        rec=dict(semantic_id=uid,target=desc['target'],material_index=mi,rows=len(members),
                 admitted=sum(r.get('source_admitted') is True for r in members),
                 shots=dict(Counter(r['shot_id'] for r in members)),proof=proof)
        counts['board_semantic_ids_checked']+=1; counts['board_rows_checked']+=len(members)
        if allowed is None:
            rec['changed']=False; checks.append(rec); counts['single_path_without_context_policy_rows']+=len(members)
            continue
        old=copy.deepcopy(desc); seed=desc.get('path_index',allowed[0])
        if seed not in allowed:seed=allowed[0]
        desc.update(allowed_paths=allowed,path_index=seed,path_domain_proof=proof,
                    finite_station_interval=[0.,1.],path_domain_version=3)
        rec.update(changed=old.get('allowed_paths') != allowed,old=old,new=copy.deepcopy(desc)); checks.append(rec)
        for r in members:
            rid=r['source_row_id']; previous=copy.deepcopy(mapping.get('row_to_material',{}).get(rid))
            override=copy.deepcopy(previous or desc)
            if previous and previous.get('material_index') != mi:
                raise ValueError('board_context_conflicts_with_actual_row_material_override')
            override.update(material_index=mi,path_index=seed,allowed_paths=allowed,
                geometry_ready=True,path_domain_proof=proof,path_domain_version=3,
                source_row_id=rid,source_uv=r['source_uv'],native_pts=r['native_pts'],shot_id=r['shot_id'],
                source_packet_sha256=r['source_packet_sha256'])
            result.setdefault('row_to_material',{})[rid]=override
            delta.append(dict(source_row_id=rid,joint_material_id=uid,shot_id=r['shot_id'],
                native_pts=r['native_pts'],source_admitted=r.get('source_admitted'),source_uv=r['source_uv'],
                current_uv=r.get('current_uv'),source_packet=r['source_packet'],
                source_packet_sha256=r['source_packet_sha256'],old_row_override=previous,
                old_semantic_path_index=old.get('path_index'),old_allowed_paths=old.get('allowed_paths'),
                new_material_index=mi,new_path_index=seed,new_allowed_paths=allowed))
        counts['context_semantic_ids_restored']+=1;counts['context_rows_mapped']+=len(members)
        counts['context_rows_admitted']+=rec['admitted'];counts['context_rows_not_admitted_retained']+=len(members)-rec['admitted']
    changed_uids={x['semantic_id'] for x in checks if x['changed']}
    for uid,v in mapping['semantic_to_material'].items():
        if uid not in changed_uids and result['semantic_to_material'][uid] != v:
            raise ValueError('noncontext_material_binding_changed')
    report=dict(schema='rinkfit.first4.board_context_path_domains.v3',counts=dict(counts),
        all_board_checks=checks,row_binding_delta=delta,
        source_rows_UV_ID_PTS_status_unchanged=True,model_geometry_and_paths_unchanged=True,
        original_part_never_used_as_hard_branch_in_context_policy=True,
        numerical_residual_used_to_change_domain=False,finite_branches_not_extended=True,
        source_qualification_not_relaxed=True,S01_packet_and_original_prefix_unchanged=True,
        unresolved_single_path_without_policy=[x for x in checks if not x['changed']])
    result['schema']='rinkfit.first4_actual_materials.path_domains_v3'
    result['board_context_domain_repair_v3']={k:v for k,v in report.items() if k!='row_binding_delta'}
    return result,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--mapping',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);audit=out.with_name('board_context_path_audit_v3.json')
    if out.exists() or audit.exists():raise ValueError('immutable context map output exists')
    q=QualifiedSourceGenerationInput(a.binding)
    result,report=repair_context_board_map(read_json(a.model),read_json(a.mapping),q.rows())
    refs=dict(model=_ref(a.model),base_map=_ref(a.mapping),source_binding=_ref(a.binding),compiler=_ref(__file__))
    report['input_references']=refs;result['board_context_domain_repair_v3']['input_references']=refs
    write_json(out,result);write_json(audit,report)
    print(__import__('json').dumps(dict(mapping=_ref(out),audit=_ref(audit),counts=report['counts'])))

if __name__=='__main__':main()
