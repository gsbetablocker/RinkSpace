"""Compile source-confirmed materials into one mutable physical joint scene.

Historical world coordinates are converted only into normalized initializers.
No observation, free depth, camera, source flag or old material index is changed.
"""
from pathlib import Path
from collections import defaultdict
import argparse,copy,json,re
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from .io_utils import read_json,write_json,file_sha256,object_sha256
from rinkfit_start.native_scene import NativeScene

BOARD_SEGMENTS={'lower_long_side':0,'lower_right':1,'right_end':2,'upper_right':3,'upper_long_side':4,'upper_left':5,'left_end':6,'lower_left':7}
ROLE_POINTS={'bottom_edge':[[0,0],[1,0]],'top_edge':[[0,1],[1,1]],'left_bottom':[[1,0]],'left_top':[[1,1]],'right_bottom':[[0,0]],'right_top':[[0,1]],'left_vertical':[[1,0],[1,1]],'right_vertical':[[0,0],[0,1]]}

def compile_materials(model,registry,project_root):
 result=copy.deepcopy(model);blocks=result['blocks'];materials=result['materials'];bylabel={m['label']:i for i,m in enumerate(materials)};byblock={b['name']:i for i,b in enumerate(blocks)};accepted=[m for m in registry['materials'] if m.get('identity_accepted')];targets=sorted({m['source_target'] for m in accepted});targetmeta={m['source_target']:m for m in accepted};project=Path(project_root);refs=[];models={};typed={};native={};adseed={};audit=[];pending=[];radius_revisions=[]
 scene_index=result.get('scene_block',0);g=blocks[scene_index]['values'];height=float(g[4]);current_perimeter=None
 for sid in sorted({s for m in accepted for s in m['shots']}):
  nr=int(sid[1:]);dr=project/'events/ankara_2026/models'/('shot2_8a28_r2_20261014' if nr==2 else f'shot{nr}_8a28_20261014');sp=dr/'scene.json';rp=dr/'feature_registry.json';scene=read_json(sp);models[sid]=scene;typed[sid]=read_json(rp)['targets'];refs.extend({'path':str(x),'sha256':file_sha256(x)} for x in (sp,rp))
 def source(target):
  meta=targetmeta[target]
  for sid in meta['shots']:
   if target in typed[sid]:return sid,models[sid],typed[sid][target]
  raise ValueError('Source descriptor not found '+target)
 def addblock(name,values,lower,upper,scales,constant=False):
  if name in byblock:return byblock[name]
  values=np.asarray(values,dtype=float);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
  if not(np.isfinite(values).all() and np.all(values>=lower) and np.all(values<=upper)):raise ValueError('Illegal shared material initializer '+name)
  i=len(blocks);blocks.append(dict(name=name,values=values.tolist(),lower=lower.tolist(),upper=upper.tolist(),scales=list(scales),constant=bool(constant)));byblock[name]=i;return i
 def source_scene(sid):
  if sid in native:return native[sid]
  scene=models[sid];p=scene['parameters'];bounds=p['bounds'];scale=float(bounds[2]-bounds[0]);W=float(bounds[3]-bounds[1])/scale;cc=p['corner_curve'];rx=float(cc['tangent_extent_xy_scene_units'][0])/scale;ry=float(cc['tangent_extent_xy_scene_units'][1])/scale;gv=[W,rx,ry,float(cc['exponent']),float(p['board_height'])/scale]+[0.]*27
  ns=NativeScene({'blocks':[{'name':'global32','values':gv}],'materials':[]});ss=np.linspace(0,ns.perimeter,65536,endpoint=False);tree=cKDTree(ns.board(ss,0)[:,:2]);native[sid]=(ns,ss,tree,scale);return native[sid]
 def station(sid,xyz):
  ns,ss,tree,scale=source_scene(sid);a=np.asarray(xyz,float)/scale;index=tree.query(a[:,:2])[1];step=ns.perimeter/len(ss);out=[]
  for xy,i in zip(a[:,:2],index):
   guess=ss[int(i)];fit=minimize_scalar(lambda s:float(np.sum((ns.board(s,0)[:2]-xy)**2)),bounds=(guess-step*2,guess+step*2),method='bounded',options={'xatol':1e-13});out.append(fit.x)
  return np.asarray(out),a[:,2]
 def installation(sid,adid):
  key=(sid,adid)
  if key in adseed:return adseed[key]
  scene=models[sid];ad=next((a for a in scene['advertisements'] if int(a.get('user_number',-1))==adid),None)
  if ad is None:raise ValueError('Missing actual physical ad '+str(adid))
  corners=ad.get('corners_xyz_scene_units')
  if corners:
   xyz=[corners[k] for k in ('left_bottom','right_bottom','left_top','right_top')]
  else:
   # Pure coordinate witness for real irregular print contours, never an
   # invented physical external rectangle.  Four witness scalars stay fixed.
   xyz=[point for c in ad.get('external_contours',[]) for point in (c.get('world_xyz_scene_units') or c.get('xyz_scene_units') or [])]
   if len(xyz)<2:raise ValueError('Missing unwrapped actual print initializer '+str(adid))
  s,z=station(sid,xyz);ns=source_scene(sid)[0];s=np.unwrap(s*2*np.pi/ns.perimeter)*ns.perimeter/(2*np.pi);start=float(min(s));span=float(max(s)-start);bottom=float(min(z));top=float(max(z));q=[start,span,bottom,top]
  if not(0<span<ns.perimeter and 0<=bottom<top<=height):raise ValueError('Physical ad seed is outside current board domain '+str(adid))
  external=any(t.startswith(f'Ad{adid}.') and '.print.' not in t for t in targets);bi=addblock(f'ad/{adid}',q,[-10,1e-6,0,0],[10,3,.1,.1],[1,.1,.02,.02],constant=not external)
  adseed[key]=(bi,np.asarray(q),ad,1 if corners and s[0]>s[1] else 0);return adseed[key]
 # Add all missing circle radii in one sorted block; old path indices are remapped
 # rather than copying centers or changing any preexisting world coordinates.
 families=defaultdict(list)
 for target in targets:
  if target in bylabel or not target.startswith('ice.'):continue
  sid,scene,d=source(target);curve=d.get('curve_id','');fid,_,role=curve.partition(':');feature=next((x for x in scene['ice_features'] if x['id']==fid),None)
  if feature and feature['kind']=='circle':families[fid].append((target,sid,feature,role or 'outer'))
 circle_paths={}
 for fid,items in families.items():
  name='ice/circle/'+fid;bi=byblock.get(name);old_values=None
  if bi is None:
   sid=items[0][1];feature=items[0][2];scale=source_scene(sid)[3];center=np.asarray(feature['center_xyz'])[:2]/scale;bi=addblock(name,center.tolist()+[float(feature['radius'])/scale],[-.5,-.5,1e-5],[.5,.5,.4],[.5,.5,.05]);old_values=[]
  else:old_values=list(blocks[bi]['values'][2:])
  b=blocks[bi];requests=[]
  for target,sid,f,role in items:
   ring=next((r for r in f.get('rings',[]) if r['role']==role),None);radius=float(ring['radius'] if ring else f['radius'])/source_scene(sid)[3];requests.append((target,radius))
  radii=list(b['values'][2:])
  for target,radius in requests:
   if all(abs(radius-r)>1e-12 for r in radii):radii.append(radius)
  radii.sort();oldr=list(b['values'][2:]);indexmap={2+j:2+radii.index(r) for j,r in enumerate(oldr)}
  for material in materials:
   for path in material['paths']:
    if path.get('geometry_block')==bi and path['kind'] in (1,'ice_circle'):path['radius_index']=indexmap[path.get('radius_index',2)]
  if radii!=oldr:radius_revisions.append({'physical_circle':fid,'block':bi,'old_radius_indices_to_new':indexmap,'old_center_unchanged':True,'old_radii_unchanged':True,'all_preexisting_circle_paths_remapped':True})
  b['values']=b['values'][:2]+radii;b['lower']=b['lower'][:2]+[1e-5]*len(radii);b['upper']=b['upper'][:2]+[.4]*len(radii);b['scales']=b['scales'][:2]+[.05]*len(radii)
  for target,r in requests:circle_paths[target]=dict(kind=1,geometry_block=bi,radius_index=2+radii.index(r),closed=True,start_angle=0.,end_angle=2*np.pi,physical_point=False)
 for target in targets:
  if target in bylabel:continue
  sid,scene,d=source(target);provenance=json.dumps({'source_descriptor':d,'source_scene_role':'normalized initializer only; not a physical measurement','source_scene_sha256':next(x['sha256'] for x in refs if x['path'].endswith('scene.json') and ('shot'+str(int(sid[1:]))+'_') in x['path']),'source_identity_registry':targetmeta[target]['id']},ensure_ascii=False);paths=[]
  try:
   if target in circle_paths:paths=[circle_paths[target]]
   elif target.startswith('board.'):
    role=d.get('role') or target.split('.')[1];part=d.get('part');paths=[dict(kind=3,boundary_segment=BOARD_SEGMENTS.get(part,-1),fixed_height_fraction=1. if role=='top' else 0.,closed=False,physical_point=False)]
   elif target.startswith('ice.'):
    curve=d['curve_id'];fid,_,role=curve.partition(':');f=next(x for x in scene['ice_features'] if x['id']==fid);scale=source_scene(sid)[3]
    if f['kind']=='polyline':vertices=np.asarray(f['segments_xyz'][int(role.replace('segment',''))],float)[:,:2]/scale;closed=False
    elif f['kind']=='polygon':
     if (f.get('current_crease_structure') or {}).get('kind')=='user_confirmed_half_circle':
      xyz=np.asarray(f['polygon_xyz'],float)[:,:2]/scale;center=.5*(xyz[0]+xyz[-1]);radius=.5*np.linalg.norm(xyz[0]-xyz[-1]);angles=[np.arctan2(v[1],v[0]) for v in (xyz[0]-center,xyz[-1]-center)];middle=xyz[len(xyz)//2]-center;orientation=min(angles,key=lambda a:np.linalg.norm(radius*np.array([np.cos(a+np.pi/2),np.sin(a+np.pi/2)])-middle));bi=addblock('ice/half_circle/'+fid,list(center)+[float(radius),float(orientation)],[-.5,-.5,1e-5,-20],[.5,.5,.4,20],[.5,.5,.05,1]);paths=[dict(kind=4,geometry_block=bi,closed=False,physical_point=False),dict(kind=5,geometry_block=bi,closed=False,physical_point=False)]
      vertices=None;closed=False
     else:vertices=np.asarray(f['polygon_xyz'],float)[:,:2]/scale;closed=bool(f.get('closed',True))
    else:raise ValueError('Unsupported source ice primitive')
    if vertices is not None:
     if len(vertices)>1 and np.linalg.norm(vertices[-1]-vertices[0])<1e-12:vertices=vertices[:-1]
     bi=addblock('ice/polyline/'+fid,vertices.ravel(),[-.5]*(2*len(vertices)),[.5]*(2*len(vertices)),[.05]*(2*len(vertices)))
     for j in range(len(vertices) if closed else len(vertices)-1):k=(j+1)%len(vertices);paths.append(dict(kind=0,geometry_block=bi,geometry_indices=[2*j,2*j+1,2*k,2*k+1],closed=False,physical_point=False))
   elif d['kind'] in {'ad_edge','ad_corner'}:
    adid=int(d['advertisement_id']);bi,seed,ad,left=installation(sid,adid);role=d['role'];points=copy.deepcopy(ROLE_POINTS[role]);
    if left==0:points=[[1-u,v] for u,v in points]
    paths=[dict(kind=2,advertisement_block=bi,points=points,closed=False,physical_point=d['kind']=='ad_corner')]
   elif d['kind']=='ad_print':
    adid=int(d['advertisement_id']);key=d['contour_key']
    # A new local contour on an already modeled print panel needs a real
    # crosswalk to the old actual ink before creating a second shared shape.
    if any(m['label'].startswith(f'Ad{adid}.print.') for m in model['materials']):raise ValueError('existing_s01_same_panel_print_requires_actual_contour_crosswalk')
    bi,seed,ad,left=installation(sid,adid);contour=next(c for c in ad.get('external_contours',[]) if c['id']==key);xyz=contour.get('world_xyz_scene_units') or contour.get('xyz_scene_units')
    if xyz is None:raise ValueError('No source-normalized contour initializer')
    ss,zz=station(sid,xyz);ns=source_scene(sid)[0];ss=seed[0]+np.mod(ss-seed[0]+ns.perimeter/2,ns.perimeter)-ns.perimeter/2;points=np.c_[(ss-seed[0])/seed[1],(zz-seed[2])/(seed[3]-seed[2])]
    shape=addblock('print/joint/'+object_sha256(target)[:20],[1,0,0,0,1,0],[-10]*6,[10]*6,[1,1,.1,1,1,.1])
    paths=[dict(kind=2,advertisement_block=bi,geometry_block=shape,points=points.tolist(),closed=bool(contour.get('closed')),physical_point=False)]
   else:raise ValueError('Unsupported confirmed primitive')
   for j,path in enumerate(paths):path.update(id='joint/'+target+'/'+str(j),variant_id='joint/'+target,provenance=provenance)
   bylabel[target]=len(materials);materials.append({'label':target,'paths':paths});audit.append({'target':target,'material_index':bylabel[target],'paths':len(paths),'source_shot':sid,'initializer_only':True})
  except (ValueError,KeyError,StopIteration) as e:pending.append({'target':target,'reason':str(e),'solver_rows_must_remain_ineligible':True})
 mapping={m['id']:{'target':m['source_target'],'material_index':bylabel.get(m['source_target']),'geometry_ready':m['source_target'] in bylabel} for m in accepted}
 report={'schema':'rinkfit.joint9.material_compiler.v1','source_references':refs,'coordinate_normalization':'each typed scene XYZ divided by exact full X extent; L=1 full length; initializers only','added_materials':audit,'circle_radius_insertions':radius_revisions,'pending_geometry':pending,'semantic_to_material':mapping,'s01_material_indices_preserved':True,'unknown_depth_parameters_added':0,'shared_scene_count':1,'all_confirmed_source_materials_geometry_ready':not pending}
 result['joint_material_compilation']=report;return result,report

def append_user_outer_perimeter(model,*,source_authority_references):
 """Add aliases of the existing common perimeter at H and zero, no new DOF.

 Use on the actual current candidate. It does not load any old numeric model,
 change prior paths/parameters, or reuse the withdrawn interior-band heights.
 """
 result=copy.deepcopy(model);before=object_sha256(model);mapping={}
 for target,operator,role,height in [('board.top','board_upper_material_edge','top',1.),('board.contact','ice_board_interface','contact',0.)]:
  found=next((i for i,m in enumerate(result['materials']) if m['label']==target),None)
  if found is None:
   found=len(result['materials']);result['materials'].append(dict(label=target,paths=[dict(id=f'outer/{role}/{j}',variant_id=f'outer/{role}',kind=3,boundary_segment=j,fixed_height_fraction=height,geometry_block=-1,physical_point=False,closed=False,provenance=json.dumps(dict(authority_references=source_authority_references,physical_definition='blue strip upper edge' if role=='top' else 'yellow strip lower edge at actual ice; not reflection',common_continuous_perimeter=True,physical_height='H' if role=='top' else '0',new_numeric_degrees_of_freedom=0),ensure_ascii=False)) for j in range(8)]))
  else:
   paths=result['materials'][found]['paths']
   if {p.get('boundary_segment') for p in paths}!={*range(8)} or any(p.get('kind')!=3 or p.get('geometry_block',-1)>=0 or p.get('fixed_height_fraction')!=height for p in paths):raise ValueError('Existing outer alias differs from the common H/0 perimeter')
  mapping['|'.join((target,'board',operator,role))]=dict(target=target,material_index=found,geometry_ready=True)
 if result['blocks']!=model['blocks'] or result['materials'][:len(model['materials'])]!=model['materials']:raise ValueError('Appending physical perimeter aliases changed existing numeric state')
 return result,dict(schema='rinkfit.user_outer_perimeter_aliases.v1',source_model_object_sha256=before,semantic_to_material=mapping,existing_blocks_and_materials_unchanged=True,new_parameter_blocks=0,new_height_parameters=0,authority_references=source_authority_references)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--registry',required=True);p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('New immutable material compilation generation required')
 model,report=compile_materials(read_json(a.model),read_json(a.registry),a.project);out.mkdir(parents=True);write_json(out/'model.json',model);write_json(out/'material_map.json',report);write_json(out/'manifest.json',{'schema':'rinkfit.joint9.compiled_material_state.v1','model':{'path':str(out/'model.json'),'sha256':file_sha256(out/'model.json')},'source_model':{'path':str(Path(a.model).resolve()),'sha256':file_sha256(a.model)},'registry':{'path':str(Path(a.registry).resolve()),'sha256':file_sha256(a.registry)},'materials':len(model['materials']),'blocks':len(model['blocks']),'geometry_pending':len(report['pending_geometry'])});print(json.dumps({'materials':len(model['materials']),'added_materials':len(report['added_materials']),'pending':len(report['pending_geometry'])}))
if __name__=='__main__':main()
