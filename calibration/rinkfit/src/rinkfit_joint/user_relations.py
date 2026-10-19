"""Compile existing user-confirmed ice relations into exact physical equations.

No dimensions, pixel observations, standard ring ratios or statistical weights
are introduced. Source endpoints remain coverage proxies unless actually read.
"""
from pathlib import Path
import argparse,copy
import numpy as np
from .io_utils import read_json,write_json,file_sha256,object_sha256

def compile_user_relations(model,project):
 selection=Path(project)/'events/ankara_2026/programmes/okada_mei_sp/fit_joint9/selection.json'
 if selection.is_file() and read_json(selection).get('legacy_structure_migration_enabled') is False:raise ValueError('Current user declined legacy structural geometry migration for this joint9 run; retain the repaired Shot1 baseline')
 result=copy.deepcopy(model);base=Path(project)/'events/ankara_2026/model_constraints';references=[dict(path=str(base/n),sha256=file_sha256(base/n)) for n in ('geometry_relations_current.json','user_corrections_current.json','paint_roles_1018.json')];facts=read_json(references[0]['path']);lookup={x['label']:x for x in result['materials']};rows=[];missing=[]
 def slot(label,component):
  material=lookup.get(label)
  if material is None:missing.append(dict(material=label,reason='no corresponding actual finite primitive in current model'));return None
  path=material['paths'][0];block=int(path['geometry_block']);indices=path.get('geometry_indices') or [];index=indices[component] if indices else int(path.get('geometry_offset',0))+component
  if index>=len(result['blocks'][block]['values']):raise ValueError('Physical relation component exceeds current geometry block')
  return (block,index)
 def circle(feature,role,coord):
  label='ice.'+feature+':'+role
  if label not in lookup:missing.append(dict(material=label,reason='ring role not present; no unseen UV or contour is invented'));return None
  path=lookup[label]['paths'][0]
  if path['kind']!=1:raise ValueError('Confirmed circle relation bound to another primitive: '+label)
  return slot(label,int(path.get('radius_index',2)) if coord=='radius' else {'x':0,'y':1}[coord])
 def line(feature,component):
  label='ice.'+feature+':segment0'
  if label in lookup and lookup[label]['paths'][0]['kind']!=0:raise ValueError('Confirmed straight carrier bound to another primitive: '+label)
  return slot(label,component)
 def equation(id,terms,rhs=0.):
  if any(s is None for s,c in terms):return
  combined={}
  for s,c in terms:combined[s]=combined.get(s,0.)+float(c)
  combined={s:c for s,c in combined.items() if c}
  if not combined:return
  rows.append(dict(id='user/'+id,terms=[dict(block=b,index=i,coefficient=c) for (b,i),c in sorted(combined.items())],rhs=float(rhs),source_refs=references))
 def equal(id,a,b):equation(id,[(a,1),(b,-1)])
 # Complete F/R columns and paired rows; the single row pitch is estimated.
 for col in 'FR':
  ref=circle(col+'1','outer','x')
  for i in range(2,5):equal(col+'_column_X/'+str(i),circle(col+str(i),'outer','x'),ref)
 for i in range(1,5):equal('paired_FR_row_Y/'+str(i),circle('F'+str(i),'outer','y'),circle('R'+str(i),'outer','y'))
 for i in (1,2):equation('equal_row_pitch/'+str(i),[(circle('F'+str(i),'outer','y'),1),(circle('F'+str(i+1),'outer','y'),-2),(circle('F'+str(i+2),'outer','y'),1)])
 for role in ('outer','middle','inner','center'):
  present=[feature for feature in [c+str(i) for c in 'FR' for i in range(1,5)] if 'ice.'+feature+':'+role in lookup]
  if present:
   ref=circle(present[0],role,'radius')
   for feature in present[1:]:equal('shared_house_radius/'+role+'/'+feature,circle(feature,role,'radius'),ref)
  for feature in [c+str(i) for c in 'FR' for i in range(1,5)]:
   if 'ice.'+feature+':'+role not in lookup:missing.append(dict(material='ice.'+feature+':'+role,reason='no observed/modelled ring path; equality only applies to currently defined same physical ring role'))
 # The rink origin is fixed at its centre in the L=1 shared model.
 equation('C0_at_rink_centre_X',[(circle('C0','outer','x'),1)])
 equation('C0_at_rink_centre_Y',[(circle('C0','outer','y'),1)])
 for feature in ('A1','A2','A3','A4'):equal('common_hockey_circle_radius/'+feature,circle(feature,'outer','radius'),circle('C0','outer','radius'))
 for a,b in (('A1','A2'),('A3','A4')):
  equal('same_end_faceoff_X/'+a+'_'+b,circle(a,'outer','x'),circle(b,'outer','x'))
  equation('same_end_faceoff_Y_mirror/'+a+'_'+b,[(circle(a,'outer','y'),1),(circle(b,'outer','y'),1)])
 for feature in ('C','B1','B2'):equal('parallel_transverse_carrier/'+feature,line(feature,0),line(feature,2))
 equation('C_is_rink_midline',[(line('C',0),1)])
 equation('B1_B2_mirror_about_C',[(line('B1',0),1),(line('B2',0),1)])
 # Four real user-confirmed house connections are independent finite ranges
 # on the row centre carrier. Their x extents are not set to circle centres,
 # lane right ends, standard lengths or invented measured endpoints.
 for i in range(1,5):
  feature=f'house_connection_F{i}_R{i}';equal('house_carrier_horizontal/'+str(i),line(feature,1),line(feature,3));equal('house_carrier_through_row/'+str(i),line(feature,1),circle('F'+str(i),'outer','y'))
 prior=result.get('physical_equalities',[])
 if any(x.get('id') in {r['id'] for r in rows} for x in prior):raise ValueError('User equality generation already compiled; do not duplicate constraints')
 result['physical_equalities']=prior+rows;result['existing_user_geometry_relations']=dict(source_refs=references,exact_equalities_added=len(rows),sigma_or_likelihood_weight_introduced=False,unseen_geometry_or_UV_created=False,extent_policy='finite source coverage proxies stay separately estimable, never promoted to painted endpoints')
 slots=sorted({(t['block'],t['index']) for r in rows for t in r['terms']});index={s:i for i,s in enumerate(slots)};A=np.zeros((len(rows),len(slots)));rhs=np.asarray([r['rhs'] for r in rows]);x=np.asarray([result['blocks'][b]['values'][i] for b,i in slots])
 for j,r in enumerate(rows):
  for t in r['terms']:A[j,index[(t['block'],t['index'])]]=t['coefficient']
 residual=A@x-rhs;rank=int(np.linalg.matrix_rank(A));augmented=int(np.linalg.matrix_rank(np.c_[A,rhs]));projection=x+np.linalg.lstsq(A,rhs-A@x,rcond=None)[0];projection_error=float(np.max(abs(A@projection-rhs),initial=0.));bounds=[]
 for (b,i),v in zip(slots,projection):
  q=result['blocks'][b]
  if v<q['lower'][i] or v>q['upper'][i]:bounds.append(dict(block=b,index=i,proposed=float(v),lower=q['lower'][i],upper=q['upper'][i]))
 report=dict(schema='rinkfit.joint9.existing_user_physical_equalities.v1',source_refs=references,equality_count=len(rows),distinct_scalar_slots=len(slots),matrix_rank=rank,augmented_matrix_rank=augmented,linear_relation_system_consistent=rank==augmented,current_maximum_relation_violation_L=float(np.max(abs(residual),initial=0.)),current_relation_residuals=[dict(id=r['id'],residual_L=float(v)) for r,v in zip(rows,residual)],minimum_change_projection_diagnostic=dict(applied=False,maximum_relation_error_L=projection_error,maximum_parameter_change_L=float(np.max(abs(projection-x),initial=0.)),original_parameter_bound_violations=bounds),missing_current_primitive_roles=missing,not_compiled_here=['Lane common halfwidth/rightX carrier handled by joint kernel','same-content physical ad size requires canonical unwrapped print/installation coordinates, never print-only witness box equality','source-specific actual visibility, topology and material RGB remain separately audited'],source_pixels_or_W_or_E_modified=False,parameter_values_modified=False)
 return result,report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('New immutable relation compilation generation required')
 model,report=compile_user_relations(read_json(a.model),a.project);out.mkdir(parents=True);write_json(out/'model.json',model);report['source_model']=dict(path=str(Path(a.model).resolve()),sha256=file_sha256(a.model));report['output_model']=dict(path=str(out/'model.json'),sha256=file_sha256(out/'model.json'));write_json(out/'report.json',report);print(__import__('json').dumps({k:report[k] for k in ('equality_count','distinct_scalar_slots','matrix_rank','linear_relation_system_consistent','current_maximum_relation_violation_L','minimum_change_projection_diagnostic')}))
