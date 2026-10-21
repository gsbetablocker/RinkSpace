from pathlib import Path
import json,hashlib
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'c_local_v3';O.mkdir(exist_ok=True)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ref(f):return dict(file=str(f),sha256=sha(f))
B=R/'gpu_recovery_v6c/candidate/model.json';A=R/'accepted_C_operator_prefix_v2/model.json';b=json.loads(B.read_text(encoding='utf8'));a=json.loads(A.read_text(encoding='utf8'));assert sha(A)=='a37fdc1f3dd19834b73d7ce466938befe1dec47dc431ba3112f5cc69c113e06e'
keys=['schema','source_clock','scene_block','rigs','shots','linear_control_manifolds','physical_equalities','coordinate_gauge','path_domain_semantics','joint_ray_domain_extension','physical_parameter_ownership']
checks={k:dict(base_present=k in b,current_present=k in a,equal=b.get(k)==a.get(k))for k in keys};assert all(x['equal']and x['base_present']==x['current_present']for x in checks.values())
assert len(a['blocks'])==len(b['blocks'])+13 and b['blocks']==a['blocks'][:len(b['blocks'])]
assert b['measurement_widths']==a['measurement_widths'][:len(b['measurement_widths'])] and b['stroke_pair_model_bindings']==a['stroke_pair_model_bindings'][:len(b['stroke_pair_model_bindings'])]
ids={b['scene_block']}
for r in b['rigs'].values():ids.update([r['head_block'],r['optics_block']])
mirrors=set()
for s in b['shots'].values():
 ids.add(s['reference_angles_block']);controls=[x['block']for x in s['control_coordinates']]if 'control_coordinates'in s else s['control_blocks'];ids.update(controls);mirrors.update(set(s['control_blocks'])-set(controls))
for m in b['materials']:
 for p in m['paths']:
  for k in ['geometry_block','advertisement_block']:
   if p.get(k,-1)>=0:ids.add(p[k])
ids.update(x['half_width_block']for x in b['measurement_widths']);ids.update(x['block']for x in b.get('linear_control_manifolds',[]));mirrors-=ids
f=dict(schema='rinkfit.C_source_navigation_complete_projection_freeze.v3',registry=ref(R/'c_local_v1/source_branch_registry_v2.json'),parent_source_gate=ref(R/'c_local_v1/source_gate_v2.json'),base_model=ref(B),approved_V9_successor_model=ref(A),required_exact_metadata_keys=keys,metadata_presence={k:k in b for k in keys},metadata_values={k:b[k]for k in keys if k in b},base_blocks=len(b['blocks']),allowed_suffix_blocks=a['blocks'][len(b['blocks']):],base_measurement_widths=b['measurement_widths'],allowed_suffix_measurement_widths=a['measurement_widths'][len(b['measurement_widths']):],base_stroke_pair_model_bindings=b['stroke_pair_model_bindings'],allowed_suffix_stroke_pair_model_bindings=a['stroke_pair_model_bindings'][len(b['stroke_pair_model_bindings']):],base_consumed_physical_block_indices=sorted(ids),unconsumed_control_mirror_blocks_value_exempt=sorted(mirrors),scope='Source navigation seeds require entire immutable source clock, rig calibration/ray domain, shot motion/control-coordinate layout, control manifolds/equalities, scene and material/width projection definitions. Runtime typed definitions are independently compared with base typed load. Unconsumed control mirror numeric values are excluded; source helpers/journals not used in projection are not treated as rig metadata.',allow_only_exact_actual_V9_13_width_suffix=True,source_UV_Query_rank_or_parent_CP_changed=False,statistical_rank=223719,source_builder=ref(Path(__file__)))
F=O/'source_freeze_v3.json'
if F.exists():raise FileExistsError(F)
F.write_text(json.dumps(f,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');report=dict(schema='rinkfit.actual_a37_projection_metadata_against_v6c.v3',base_model=ref(B),actual_R33_model=ref(A),nonblock_changed_keys=[k for k in set(b)|set(a) if k!='blocks'and b.get(k)!=a.get(k)],required_metadata=checks,all_5143_old_block_JSON_exact=True,only_13_real_V9_width_suffix=True,measurement_width_old_prefix_exact=True,stroke_pair_old_prefix_exact=True,consumed_old_physical_blocks=len(ids),unconsumed_control_mirror_value_exempt_count=len(mirrors),source_UV_Query_rank_unchanged=True,freeze=ref(F),Solve_started=False)
(O/'actual_metadata_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(report,ensure_ascii=False))
