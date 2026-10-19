"""Evidence based physical-material identities, keeping local component versions.

A typed name never overrides conflicting source contours.  Two different
sponsor panels remain separate even if their artwork is identical.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,argparse
import numpy as np
from PIL import Image
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import _channels,_sample,_points

def _source_contours(scene,template):
 out=defaultdict(list)
 for ad in scene.get('advertisements',[]):
  nr=ad.get('user_number',ad.get('advertisement_id'))
  for c in ad.get('external_contours',[]):
   if c.get('id'):out[(int(nr),c['id'])].append(c)
 for nr,ad in template.get('advertisements',{}).items():
  for key,c in ad.get('contours',{}).items():out[(int(nr),key)].append(c)
 return out

def _audit_contour(contour,ad,cache):
 uv=_points(contour.get('source_native_uv'))
 path=contour.get('source_path') or contour.get('source_frame_path');expected=contour.get('source_png_sha256') or contour.get('source_frame_sha256')
 if not len(uv):return {'verified':False,'reason':'template_has_no_original_native_contour','closed':bool(contour.get('closed'))}
 if path:
  path=Path(path)
  if path.is_file():
   digest=file_sha256(path)
   if expected and digest!=expected:return {'verified':False,'reason':'template_source_png_hash_mismatch','path':str(path)}
   if str(path) not in cache:
    if len(cache)>=4:cache.pop(next(iter(cache)))
    cache[str(path)]=_channels(np.asarray(Image.open(path).convert('RGB')))
   channels=cache[str(path)];h,w=channels.shape[1:]
   inside=(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
   tests=[]
   count=len(uv) if contour.get('closed') else len(uv)-1
   for i in range(max(0,count)):
    j=(i+1)%len(uv);d=uv[j]-uv[i]
    if not inside[i] or not inside[j] or np.linalg.norm(d)<1e-8:tests.append(False);continue
    normal=np.array([-d[1],d[0]])/np.linalg.norm(d);q=.5*(uv[i]+uv[j]);samples=_sample(channels,q+np.arange(-2,2.01,.5)[:,None]*normal);tests.append(float(np.ptp(samples,axis=0).max())>=4.)
   footprint={'physical_advertisement_number':ad,'source_png_sha256':digest,'source_frame_index':contour.get('source_frame_index'),'source_native_uv':uv.tolist(),'closed':bool(contour.get('closed'))}
   return {'verified':bool(inside.all() and tests and all(tests)),'reason':'all_template_edges_current_source_supported' if inside.all() and tests and all(tests) else 'some_template_edges_missing_or_weak_source','source_footprint_sha256':object_sha256(footprint),'source_path':str(path),'source_png_sha256':digest,'vertices':len(uv),'vertices_in_image':int(inside.sum()),'segments':len(tests),'pixel_supported_segments':sum(tests),'segment_support':tests,'closed':bool(contour.get('closed')),'source_frame_index':contour.get('source_frame_index')}
 return {'verified':False,'reason':'original_template_png_path_not_resolved','source_frame_index':contour.get('source_frame_index'),'native_contour_vertices':len(uv)}

def build_crosswalk(source_audit_dir,project_root,output_dir):
 audit=Path(source_audit_dir);project=Path(project_root);out=Path(output_dir)
 if out.exists():raise ValueError('A new immutable crosswalk generation is required')
 rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));materials=read_json(audit/'joint_material_registry.json')['materials'];by=defaultdict(list)
 for r in rows:by[r.get('joint_material_id') or 'unbound:'+r['shot_id']].append(r)
 caches={};refs=[];contours={};regs={}
 for sid in sorted({r['shot_id'] for r in rows}):
  nr=int(sid[1:]);dr=project/'events/ankara_2026/models'/('shot2_8a28_r2_20261014' if nr==2 else f'shot{nr}_8a28_20261014');sp=dr/'scene.json';tp=dr/'ad2_material_template.json';rp=dr/'feature_registry.json';contours[sid]=_source_contours(read_json(sp),read_json(tp));regs[sid]=read_json(rp)['targets'];refs.extend({'path':str(x),'sha256':file_sha256(x)} for x in (sp,tp,rp))
 counts=Counter();registry=[];decisions=[]
 for mat in materials:
  members=by[mat['id']];target=mat.get('source_target');good=[r for r in members if r['source_support_status']=='source_supported_identity_candidate'];cat=members[0]['category'];audits=[];footprints=set();reasons=[]
  if not target:reasons.append('unbound_identity_must_be_resolved_from_actual_material_topology')
  elif cat=='ad_print':
   for sid in mat['shots']:
    d=regs[sid].get(target,{})
    ad=d.get('advertisement_id');key=d.get('contour_key')
    found=contours[sid].get((int(ad),key),[]) if ad is not None else []
    if not found:audits.append({'shot_id':sid,'verified':False,'reason':'typed_print_contour_has_no_resolved_actual_source_template','contour_key':key});continue
    for c in found:
     a=_audit_contour(c,int(ad),caches);audits.append({'shot_id':sid,'contour_key':key,**a})
     if a.get('verified'):footprints.add(a['source_footprint_sha256'])
   if not footprints:reasons.append('no_fully_source_supported_print_contour')
   if len(footprints)>1:reasons.append('same_local_target_has_different_source_footprints_requires_shape_correspondence')
   if any(not a['verified'] for a in audits):reasons.append('incomplete_or_conflicting_print_template_source')
  elif cat not in {'board_edge','ad_edge','ad_corner','ice_curve'}:reasons.append('outside_supported_static_material_classes')
  # Nonprint shared identity is the typed physical descriptor, material role,
  # user advertisement/ice naming and present native edge readout together.
  # Geometry/accuracy are not accepted by this identity decision.
  accepted=bool(good and not reasons)
  if not good:reasons.append('no_admissible_current_native_material_readout')
  identity_basis=('exact_same_actual_source_contour_footprint_in_same_physical_ad_panel' if cat=='ad_print' else 'typed_physical_material_definition_and_current_native_transition')
  result={**mat,'identity_accepted':accepted,'identity_status':'source_identity_accepted' if accepted else 'automatic_identity_followup','identity_basis':identity_basis,'identity_reasons':reasons,'template_source_audits':audits,'canonical_source_footprint_sha256':next(iter(footprints)) if len(footprints)==1 else None,'geometry_frozen':False,'source_accuracy_or_geometry_accepted':False}
  registry.append(result);counts['accepted_materials' if accepted else 'pending_materials']+=1
  for row in members:
   eligible_identity=accepted and row['source_support_status']=='source_supported_identity_candidate'
   decisions.append({'source_row_id':row['source_row_id'],'joint_material_id':mat['id'],'physical_identity_passed':eligible_identity,'current_native_source_support_passed':row['source_measurement_audit']['current_uv'] is not None,'identity_evidence_sha256':object_sha256(result),'source_identity_reasons':reasons,'source_row_reasons':row['source_review_reasons'],'source_family_id':row['source_family_id'],'solver_eligible':False,'statistics_promotion_pending':True})
   counts['identity_source_ready_rows' if eligible_identity else 'pending_rows']+=1
 out.mkdir(parents=True);write_json(out/'joint_material_registry.json',{'schema':'rinkfit.joint9.verified_material_registry.v1','materials':registry,'references':refs,'source_audit_summary_sha256':file_sha256(audit/'summary.json'),'raw_flags_changed':False,'numeric_geometry_fitted':False});write_jsonl(out/'identity_decisions.jsonl.gz',decisions)
 summary={'schema':'rinkfit.joint9.material_identity_crosswalk.v1','counts':dict(counts),'source_rows':len(rows),'physical_panel_separation_preserved':True,'fit_residuals_used':False,'local_component_names_alone_accepted':False,'source_audit_dir':str(audit)};write_json(out/'summary.json',summary);return summary

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--audit',required=True);a.add_argument('--project',required=True);a.add_argument('--out',required=True);x=a.parse_args();print(json.dumps(build_crosswalk(x.audit,x.project,x.out)))
