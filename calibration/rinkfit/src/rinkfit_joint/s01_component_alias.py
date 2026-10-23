"""Source-only print crosswalk for s01 Ad7/8 and current native contours.

No camera/model residual chooses a physical identity. Old paths and parameters
remain unchanged; current source additions share each actual artwork chart.
"""
from pathlib import Path
from collections import defaultdict
import argparse,copy,json,re
import cv2,numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from .io_utils import read_json,write_json,file_sha256,object_sha256
from .material_path import new_board_path_points

def dense(q,step=.25):
 q=np.asarray(q,float);out=[]
 for a,b in zip(q,np.roll(q,-1,axis=0)):
  n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)));out.append(a+np.arange(n)[:,None]/n*(b-a))
 return np.concatenate(out)
def _bbox(q):q=np.asarray(q);return np.r_[q.min(0),q.max(0)]
def _newblock(model,name,initial):
 b=copy.deepcopy(initial);b['name']=name;b['constant']=False;b['fixed_indices']=[];index=len(model['blocks']);model['blocks'].append(b);return index

def compile_aliases(model,material_map,registry,project):
 result=copy.deepcopy(model);mapping=copy.deepcopy(material_map);project=Path(project);scene_path=project/'events/ankara_2026/models/shot2_8a28_r2_20261014/scene.json';scene=read_json(scene_path);spp=scene['source_conditioned_board_prints'];bylabel={m['label']:i for i,m in enumerate(result['materials'])};pending={m['target'] for m in mapping['pending_geometry']};accepted={m['source_target']:m for m in registry['materials'] if m.get('identity_accepted')};receipts=[];aliases=[];additions=[];diagnostics=[];refs=[dict(path=str(scene_path),sha256=file_sha256(scene_path))]
 def add(target,paths,basis):
  if target not in pending or target not in accepted:return
  for j,p in enumerate(paths):
   if p['kind']==2:p['points']=new_board_path_points(p['points'],p.get('closed',False))
   p['id']='crosswalk/'+target+'/'+str(j);p['variant_id']='crosswalk/'+target;p['provenance']=json.dumps(basis,ensure_ascii=False)
  index=len(result['materials']);result['materials'].append(dict(label=target,paths=paths));bylabel[target]=index
  m=accepted[target];mapping['semantic_to_material'][m['id']]={'target':target,'material_index':index,'geometry_ready':True,'source_identity_basis':basis}
  aliases.append(dict(target=target,material_index=index,paths=len(paths),physical_panel=m['physical_entity'],basis=basis))
 # The current Ad7 source spells both complete words in actual native pixels.
 # Character order plus actual counter topology identifies official subpaths;
 # the old s01 logo chart already uses exactly this SHA-verified artwork.
 ap=project/'events/ankara_2026/models/ad7_shape_1016.json';a7=read_json(ap);refs.append(dict(path=str(ap),sha256=file_sha256(ap)))
 if file_sha256(a7['source_svg']['path'])!=a7['source_svg']['sha256']:raise ValueError('Original Ad7 SVG source changed')
 refs.append(a7['source_svg']);ad7=next(a for a in scene['advertisements'] if int(a['user_number'])==7);new7={c['id']:c for c in ad7['external_contours'] if c.get('source_native_uv')};old7={}
 for m in result['materials']:
  if m['label'].startswith('Ad7.print.'):
   for p in m['paths']:
    name=p['id'].split('/')[-1];old7[name]=(m,p)
 primary=copy.deepcopy(old7['logo_part00'][1]);source7=new7['r2_ISU_00'];sp=Path(source7['source_path'])
 if file_sha256(sp)!=source7['source_png_sha256']:raise ValueError('Current native Ad7 source changed')
 refs.append(dict(path=str(sp),sha256=file_sha256(sp)));base7=dict(method='same actual Ad7 panel; current native ink profiles + official artwork character order/counter topology',source_native_frame_index=source7['source_frame_index'],source_native_png_sha256=source7['source_png_sha256'],official_shape_sha256=file_sha256(ap),source_svg_sha256=a7['source_svg']['sha256'],new_physical_parameters=0,model_residual_used_for_identity=False)
 logo={'r2_ISU_00':['logo_part01'],'r2_ISU_01':['logo_part02'],'r2_ISU_02':['logo_part03'],'r2_ISU_03':['logo_part00'],'r2_ISU_04':['logo_part04','logo_part05']}
 for key,names in logo.items():add('Ad7.print.'+key,[copy.deepcopy(old7[x][1]) for x in names],{**base7,'current_source_contour':key,'existing_actual_paths':names,'source_support_registry':accepted.get('Ad7.print.'+key,{}).get('canonical_source_footprint_sha256')})
 glyphmap={};top=sorted([c for k,c in new7.items() if k.startswith('r2_INTERNATIONAL_') and 'inner' not in k],key=lambda c:np.mean(c['source_native_uv'],0)[0]);bot=sorted([c for k,c in new7.items() if k.startswith('r2_SKATING_UNION_') and 'inner' not in k],key=lambda c:np.mean(c['source_native_uv'],0)[0]);corrA=[];corrB=[]
 for prefix,word,curves in [('top','INTERNATIONAL',top),('bottom','SKATINGUNION',bot)]:
  if len(curves)!=len(word):raise ValueError('Current native character order incomplete '+word)
  for j,(char,c) in enumerate(zip(word,curves)):
   key=f'{prefix}_{j:02d}_{char}_part00';glyphmap[c['id']]=key;ca=_bbox(a7['contours'][key]['material_uv']);cb=_bbox(c['source_native_uv']);corrA.append(.5*(ca[:2]+ca[2:]));corrB.append(.5*(cb[:2]+cb[2:]))
 for key,c in new7.items():
  if 'inner' not in key:continue
  point=np.mean(c['source_native_uv'],0);outers=top if key.startswith('r2_INTERNATIONAL') else bot;possible=[o for o in outers if cv2.pointPolygonTest(np.asarray(o['source_native_uv'],np.float32),tuple(point.astype(float)),False)>=0]
  if len(possible)!=1:diagnostics.append(dict(target=key,reason='source_counter_not_in_unique_character'));continue
  parent=glyphmap[possible[0]['id']];candidate=parent.replace('part00','part01')
  if candidate not in a7['contours']:raise ValueError('Current source counter lacks official contour '+key)
  glyphmap[key]=candidate
 H7,inl=cv2.findHomography(np.asarray(corrA),np.asarray(corrB),cv2.RANSAC,1.5)
 if H7 is None or int(inl.sum())<20:raise ValueError('Ad7 current-source text topology not matched '+json.dumps({'correspondences':len(corrA),'inliers':None if inl is None else int(inl.sum()),'canonical_centers':np.asarray(corrA).tolist(),'native_centers':np.asarray(corrB).tolist()}))
 word_witness=dict(source_center_correspondences=len(corrA),inliers=int(inl.sum()),artwork_to_current_source_navigation_H=H7.tolist(),H_not_final_geometry=True)
 for key,shape in glyphmap.items():
  p=copy.deepcopy(primary);definition=a7['contours'][shape];p['points']=definition['material_uv'];p['closed']=definition['closed'];p['physical_point']=False
  qq=cv2.perspectiveTransform(np.asarray(definition['material_uv'])[None],H7)[0];dist=cKDTree(dense(new7[key]['source_native_uv'])).query(qq)[0];support=dict(median_current_outline_distance_px=float(np.median(dist)),p95_current_outline_distance_px=float(np.quantile(dist,.95)),official_subpath=shape,current_source_contour=key)
  if support['median_current_outline_distance_px']>2.5:diagnostics.append(dict(target='Ad7.print.'+key,reason='official_shape_current_source_support_not_matched',**support));continue
  add('Ad7.print.'+key,[p],{**base7,**word_witness,**support,'existing_shared_affine_block':primary['geometry_block'],'shape_expanded_only_to_currently_visible_glyph':True})
 # Ad8 old prototype and current native picture are matched in real RGB. They
 # share artwork, while each physical installation remains Ad8 (never Ad21).
 ad8=next(a for a in scene['advertisements'] if int(a['user_number'])==8);new8={c['id']:c for c in ad8['external_contours'] if c.get('source_native_uv')};old8=next(x for x in spp['patterns'] if int(x['advertisement_id'])==21)['external_contours'];old8={x['id']:x for x in old8};sourceold=spp['shared_templates'][0]['canonical_anchor_frame'];sourcenew=new8['r2_WYNDHAM_00'];pairs=[];kps=[];des=[]
 for ref,items in [(sourceold,list(old8.values())),(dict(path=sourcenew['source_path'],sha256=sourcenew['source_png_sha256']),list(new8.values()))]:
  if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Ad8 source prototype changed')
  refs.append(ref);im=cv2.imread(ref['path'],0);points=np.concatenate([c['source_native_uv'] for c in items]);a=np.floor(points.min(0)-20).astype(int);b=np.ceil(points.max(0)+20).astype(int);crop=cv2.resize(im[a[1]:b[1],a[0]:b[0]],None,fx=3,fy=3);kp,d=cv2.SIFT_create(5000,contrastThreshold=.008).detectAndCompute(crop,None);kps.append(np.asarray([k.pt for k in kp])/3+a);des.append(d)
 raw=cv2.BFMatcher().knnMatch(des[0],des[1],k=2);matches=[a for a,b in raw if a.distance<.72*b.distance];A=np.asarray([kps[0][m.queryIdx] for m in matches]);B=np.asarray([kps[1][m.trainIdx] for m in matches]);H,inside=cv2.findHomography(A,B,cv2.RANSAC,1.5)
 if H is None or inside.sum()<20:raise ValueError('Real Ad8 print RGB does not have a unique source match')
 base8=dict(method='actual native RGB SIFT correspondence plus ordered glyph/counter and contour footprint',source_prototype=sourceold,current_native_source=dict(path=sourcenew['source_path'],sha256=sourcenew['source_png_sha256'],programme_frame_index_1based=sourcenew['source_frame_index']),real_RGB_matches=len(matches),real_RGB_inliers=int(inside.sum()),source_pixel_H_navigation_only=H.tolist(),physical_installation='Ad8',same_template_Ad21_location_not_merged=True,model_geometry_used_for_identity=False)
 correspond={'r2_WYNDHAM_02':'Ad8:component13','r2_WYNDHAM_03':'Ad8:component11','r2_WYNDHAM_05':'Ad8:component10','r2_WYNDHAM_06':'Ad8:component8','r2_WYNDHAM_08':'Ad8:component7'}
 for key,oldkey in correspond.items():
  pa=result['materials'][bylabel['Ad8.print.'+oldkey]]['paths'][0];q=cv2.perspectiveTransform(np.asarray(old8[oldkey]['source_native_uv'],float)[None],H)[0];distance=cKDTree(dense(q)).query(dense(new8[key]['source_native_uv']))[0]
  if np.median(distance)>2.5:diagnostics.append(dict(target=key,reason='actual_template_contour_mismatch',median_distance=float(np.median(distance))));continue
  add('Ad8.print.'+key,[copy.deepcopy(pa)],{**base8,'existing_source_contour':oldkey,'current_source_contour':key,'matched_source_median_distance_px':float(np.median(distance)),'new_physical_parameters':0})
 # Transfer current measured outlines into the existing fixed local print chart
 # only as a source-informed initializer. This is not camera-derived UV truth.
 U=[];V=[]
 for label,mi in bylabel.items():
  if not label.startswith('Ad8.print.Ad8:component'):continue
  oldkey=label[len('Ad8.print.'):];q=result['materials'][mi]['paths'][0]['points'];u=old8[oldkey]['source_native_uv']
  if len(q)==len(u)+1:q=q[:-1]
  if len(q)!=len(u):raise ValueError('Original Ad8 contour vertex/source order changed')
  U.extend(u);V.extend(q)
 unique={}
 for u,v in zip(U,V):unique[tuple(u)]=v
 transfer=RBFInterpolator(np.asarray(list(unique)),np.asarray(list(unique.values())),kernel='thin_plate_spline',degree=1);primary8=copy.deepcopy(result['materials'][bylabel['Ad8.print.Ad8:component14']]['paths'][0]);word_block=None
 for key,c in new8.items():
  target='Ad8.print.'+key
  if target not in pending or target in bylabel:continue
  old_uv=cv2.perspectiveTransform(np.asarray(c['source_native_uv'],float)[None],np.linalg.inv(H))[0];local=transfer(old_uv);p=copy.deepcopy(primary8);p['points']=local.tolist();p['closed']=bool(c.get('closed',True));p['physical_point']=False
  if key=='r2_WYNDHAM_04_inner':p['geometry_block']=result['materials'][bylabel['Ad8.print.Ad8:component11']]['paths'][0]['geometry_block']
  elif key=='r2_WYNDHAM_07_inner':p['geometry_block']=result['materials'][bylabel['Ad8.print.Ad8:component8']]['paths'][0]['geometry_block']
  elif key=='r2_Wyndham_icon_00_inner':p['geometry_block']=result['materials'][bylabel['Ad8.print.Ad8:component15']]['paths'][0]['geometry_block']
  elif key.startswith('r2_Ankara_'):
   if word_block is None:word_block=_newblock(result,'print/Ad8/current_Ankara_word',result['blocks'][primary8['geometry_block']])
   p['geometry_block']=word_block
  add(target,[p],{**base8,'current_source_contour':key,'source_footprint_sha256':accepted[target]['canonical_source_footprint_sha256'],'new_shape_is_actual_visible_native_ink':True,'shared_existing_affine_block':p['geometry_block'] if p['geometry_block']!=word_block else None,'shared_new_word_affine_block':word_block if p['geometry_block']==word_block else None,'initializer_transfer':'current UV -> actual source RGB correspondence -> original normalized source print chart; neither UV observation nor physical truth is generated','topology_note':'W/Y are separately visible current contours; the legacy joined outline is retained as historical modeled support pending explicit s01 topology rebase' if key in ('r2_WYNDHAM_00','r2_WYNDHAM_01') else 'separate actual ink boundary/counter'})
 done={a['target'] for a in aliases};mapping['pending_geometry']=[x for x in mapping['pending_geometry'] if x['target']not in done];mapping['all_confirmed_source_materials_geometry_ready']=not mapping['pending_geometry'];report=dict(schema='rinkfit.joint9.s01_print_crosswalk.v1',source_references=refs,source_aliases=aliases,diagnostics=diagnostics,new_materials=len(aliases),new_physical_blocks=len(result['blocks'])-len(model['blocks']),old_materials_and_paths_unchanged=result['materials'][:len(model['materials'])]==model['materials'],old_parameter_values_unchanged=result['blocks'][:len(model['blocks'])]==model['blocks'],pending= mapping['pending_geometry'],original_UV_or_flags_changed=False,source_geometry_initializer_not_fit_accuracy=True,legacy_connected_WY_source_topology_followup_required=True);result['current_s01_print_crosswalk']=report;return result,mapping,report

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--map',required=True);p.add_argument('--registry',required=True);p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('A new immutable source crosswalk generation is required')
 m,mp,r=compile_aliases(read_json(a.model),read_json(a.map),read_json(a.registry),a.project);out.mkdir(parents=True);write_json(out/'model.json',m);write_json(out/'material_map.json',mp);write_json(out/'report.json',r);print(json.dumps(dict(added=r['new_materials'],blocks=r['new_physical_blocks'],pending=len(r['pending']),diagnostics=len(r['diagnostics']),out=str(out))))
if __name__=='__main__':main()
