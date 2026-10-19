from pathlib import Path
import sys,json,hashlib,time
from collections import Counter
ROOT=Path(r'D:/OneDrive/codex_projs/RinkSpace');DATA=Path(r'D:/CodexData/RinkSpace_Data');W=DATA/'runs/okada_rec_1019';RESULT=DATA/'events/ankara_2026/programmes/okada_mei_sp/results'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.render_result import fixed_scene_paths,SceneRenderer
from calibration.program_tracking.demo_camera_patch import apply_patch,world_plane_homography,project_xy

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):
 with Path(p).open('x',encoding='utf8')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write(chr(10))
def provenance_inventory(scene):
 result=[]
 def add(kind,xyz,label,pointers,feature=None):result.append(dict(kind=kind,xyz=np.asarray(xyz,float),label=label,source_json_pointers=pointers,feature_id=feature))
 for kind,points,pointer in [('contact',scene['ice_boundary']['points_xyz'],'/ice_boundary/points_xyz'),('board_top',scene['board_surface']['top_edge_xyz'],'/board_surface/top_edge_xyz')]:
  a=np.asarray(points,float)
  if not np.array_equal(a[0],a[-1]):a=np.vstack([a,a[0]])
  add(kind,a,kind,[pointer])
 for fi,f in enumerate(scene['ice_features']):
  if f['kind']=='circle':
   rings=f.get('rings',[dict(center_xyz=f['center_xyz'],radius=f['radius'],role='outer')])
   for ri,ring in enumerate(rings):
    if ring['role']!='outer':continue
    t=np.linspace(0,2*np.pi,181);xyz=np.asarray(ring['center_xyz'])+np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]*float(ring['radius']);add('marking',xyz,f['id']+':outer',[f'/ice_features/{fi}/rings/{ri}'if 'rings'in f else f'/ice_features/{fi}'],f['id'])
  elif f['kind']=='polyline':
   for si,points in enumerate(f['segments_xyz']):
    a=np.asarray(points,float);xyz=np.vstack([np.linspace(x,y,32)for x,y in zip(a[:-1],a[1:])])if len(a)>1 else a;add('marking',xyz,f['id']+':segment'+str(si),[f'/ice_features/{fi}/segments_xyz/{si}'],f['id'])
 edges={}
 for ci,cell in enumerate(scene['ice_grid']['cells']):
  a=np.asarray(cell['polygon_xyz'],float)
  for vi,(x,y)in enumerate(zip(a,np.roll(a,-1,axis=0))):
   if np.linalg.norm(x-y)<=1e-10:continue
   key=tuple(sorted((tuple(np.round(x,9)),tuple(np.round(y,9)))));old=edges.get(key);origins=[]if old is None else old[1];origins.append(dict(cell_index=ci,edge_start_vertex=vi,edge_end_vertex=(vi+1)%len(a),source_json_pointer=f'/ice_grid/cells/{ci}/polygon_xyz'));edges[key]=(np.linspace(x,y,9),origins)
 for ei,(key,(xyz,origins))in enumerate(edges.items()):
  add('grid',xyz,'grid/'+str(ei),[o['source_json_pointer']for o in origins]);result[-1].update(grid_edge_key_scene_round9=[list(x)for x in key],grid_cell_edge_origins=origins)
 for fi,f in enumerate(scene['ice_features']):
  if f['kind']=='polygon':
   a=np.asarray(f['polygon_xyz'],float);omit=set(f.get('open_edges',[]))
   if not omit:add('marking',np.vstack([a,a[:1]]),f['id']+':boundary',[f'/ice_features/{fi}/polygon_xyz'],f['id'])
   else:
    for ei,(x,y)in enumerate(zip(a,np.roll(a,-1,axis=0))):
     if ei not in omit:add('marking',np.linspace(x,y,32),f['id']+':edge'+str(ei),[f'/ice_features/{fi}/polygon_xyz',f'/ice_features/{fi}/open_edges'],f['id'])
   continue
  if f['kind']!='circle':continue
  for ri,ring in enumerate(f.get('rings',[])):
   if ring['role']=='outer':continue
   t=np.linspace(0,2*np.pi,181);xyz=np.asarray(ring['center_xyz'])+np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]*float(ring['radius']);add('marking',xyz,f['id']+':'+ring['role'],[f'/ice_features/{fi}/rings/{ri}'],f['id'])
 return result

def main():
 begin=time.perf_counter();out=W/'screen_grid_demo50_s2_v1'
 if out.exists():raise FileExistsError('New immutable direct initial state required')
 src=RESULT/'demo_last_shot_20261016/source_packet.npz';srcdoc=read(src.with_suffix('.json'));patchfile=RESULT/'demo_full_20261016/v3/camera_patch.npz';patchdoc=read(patchfile.with_suffix('.json'))
 if sha(src)!='4e5cb848873df4cb43f0a78f43deda17a445c6e60219111b9aa32d884c02a3ee'or sha(patchfile)!='e28e8b191c8eeb53d4bb1819ca1a1b3eb9f360d2f0540b8b6eea7f1994d22ccf':raise ValueError('Actual Demo packet or V3 patch changed')
 if srcdoc['output']['sha256']!=sha(src)or patchdoc['source_packet']['sha256']!=sha(src):raise ValueError('Demo provenance mismatch')
 with np.load(src,allow_pickle=False)as a:p={k:a[k].copy()for k in a.files}
 fullpts=np.arange(256256,2415104+256,256,dtype=np.int64)
 if not(np.array_equal(p['pts'],fullpts)and np.array_equal(p['native_pts'],fullpts)and str(p['time_base'])=='1/12800'and str(p['native_fps'])=='50/1'):raise ValueError('Official source clock mismatch')
 patch=apply_patch(p,patchfile);covered=np.zeros(len(fullpts),'u1');changed=np.zeros(len(fullpts),'u1');covered[patch['source_packet_index']]=1;changed[patch['source_packet_index']]=patch['changed']
 scope_path=W/'programme_scope.json';scope=read(scope_path)
 if scope['included_frame_count']!=8434 or scope['source']['sha256']!=srcdoc['source']['sha256']:raise ValueError('Programme source identity/scope mismatch')
 selected=np.flatnonzero(p['shot_index']!=1);pts=p['pts'][selected];shot=p['shot_index'][selected].astype('u1')
 if len(selected)!=7186 or int(pts[0])!=575744 or np.any(shot==1):raise ValueError('S01 must never occur in direct Demo initial state')
 s01dir=DATA/'runs/okada_s1_r11/temporal_1018/local_m4_iter3_scope_full2';s01manifest=s01dir/'complete_state_manifest.json';s01doc=read(s01manifest);s01model=s01dir/'model.json'
 if sha(s01model)!='d78b7012c80664131f88134be0f9f868c1d6742ad7516155fc131b72ca3110e9'or sha(s01model)!=s01doc['model']['sha256']:raise ValueError('Preserved complete S01 baseline identity mismatch')
 for r in s01doc['row_state_files'].values():
  if sha(r['path'])!=r['sha256']:raise ValueError('Preserved S01 auxiliary state identity mismatch')
 s01pres=dict(state_kind='preserved_complete_physical_baseline',model=ref(s01model),complete_state_manifest=ref(s01manifest),row_state_files=s01doc['row_state_files'],parent_complete_checkpoint=s01doc['parent_complete_checkpoint'],native_frames=1248,first_pts=256256,last_pts=575488,Demo_values_present=False,physical_values_or_auxiliaries_modified=False)
 oldpath=Path(srcdoc['inputs']['scene']['path'])
 if sha(oldpath)!=srcdoc['inputs']['scene']['sha256']:raise ValueError('Frozen Demo scene identity mismatch')
 scene=read(oldpath);paths=fixed_scene_paths(scene,draw_grid=True);prov=provenance_inventory(scene)
 if len(paths)!=len(prov):raise ValueError('Original SceneRenderer topology/provenance count mismatch')
 for i,((kind,xyz),origin)in enumerate(zip(paths,prov)):
  if kind!=origin['kind']or not np.array_equal(xyz,origin['xyz']):raise ValueError('Original fixed SceneRenderer path order changed '+str(i))
 alloffset=np.r_[0,np.cumsum([len(x)for k,x in paths])];ice_indices=[i for i,(k,x)in enumerate(paths)if k in('grid','marking')];ice=[paths[i]for i in ice_indices];xyz=np.concatenate([x for k,x in ice]);V=len(xyz);F=len(selected);offsets=np.r_[0,np.cumsum([len(x)for k,x in ice])].astype('<i8')
 if len(ice)!=2130 or V!=26238 or np.max(np.abs(xyz[:,2]))>1e-12:raise ValueError('Complete original ice renderer inventory changed')
 segments=np.concatenate([np.c_[np.arange(a,b-1),np.arange(a+1,b)]for a,b in zip(offsets[:-1],offsets[1:])]).astype('<i4');kinds=np.asarray([0if k=='grid'else 1for k,x in ice],dtype='u1');records=[]
 for oi,source_index in enumerate(ice_indices):
  q={k:v for k,v in prov[source_index].items()if k!='xyz'};kind,a=paths[source_index];q.update(path_id='SceneRenderer/path/'+str(source_index),original_renderer_path_index=source_index,output_path_index=oi,vertex_begin=int(offsets[oi]),vertex_end=int(offsets[oi+1]),original_renderer_vertex_begin=int(alloffset[source_index]),original_renderer_vertex_end=int(alloffset[source_index+1]),vertex_count=len(a),explicit_closed=bool(np.array_equal(a[0],a[-1])),color_bgr=list(SceneRenderer.COLORS[kind]),original_world_vertex_bytes_sha256=hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest());records.append(q)
 out.mkdir();topology=out/'topology.json';write(topology,dict(schema='rinkfit.original_screen_grid_topology.v1',source_scene=ref(oldpath),original_inventory_producer='calibration.program_tracking.render_result.fixed_scene_paths(scene,draw_grid=True)',original_full_renderer_paths=len(paths),original_full_renderer_vertices=int(alloffset[-1]),included_ice_paths=len(ice),included_ice_vertices=V,original_ice_grid_paths=2065,original_ice_marking_paths=65,original_ice_vertices_all_preserved=True,producer_does_not_resample_filter_crosswalk_or_deduplicate=True,grid_edge_deduplication_is_original_fixed_scene_paths_behavior=True,nonice_not_in_direct_ice_state=[dict(original_renderer_path_index=i,kind=k,vertices=len(a))for i,(k,a)in enumerate(paths)if k not in('grid','marking')],segment_rule='only consecutive original vertices inside each path; repeated closing endpoint retained, never connect paths or invent additional closure',paths=records))
 small=dict(path_offsets=offsets,path_kind=kinds,path_original_renderer_index=np.asarray(ice_indices,dtype='<i4'),segment_vertices=segments,frame_native_pts=pts.astype('<i8'),frame_shot_number=shot,frame_demo_bridge_mode=p['bridge_mode'][selected].astype('u1'),frame_demo_direct_mapping=p['direct_mapping'][selected].astype('u1'),frame_demo_inferred=p['inferred'][selected].astype('u1'),frame_demo_valid_flag=p['valid'][selected].astype('u1'),frame_demo_patch_covered=covered[selected],frame_demo_patch_changed=changed[selected],frame_demo_anchor_left_official_pts=p['anchor_left_official_pts'][selected],frame_demo_anchor_right_official_pts=p['anchor_right_official_pts'][selected],frame_demo_anchor_left_legacy_pts=p['anchor_left_legacy_pts'][selected],frame_demo_anchor_right_legacy_pts=p['anchor_right_legacy_pts'][selected])
 fields={}
 for name,a in small.items():
  file=out/(name+'.npy')
  with file.open('xb')as f:np.save(f,a,allow_pickle=False)
  fields[name]=dict(dtype=str(a.dtype),shape=list(a.shape),storage='npy_memmap',**ref(file))
 spec=dict(screen_uv=('<f8',[F,V,2],'screen_uv.f64'),depth=('<f8',[F,V],'depth.f64'),vertex_mask_bits=('u1',[F,V],'vertex_mask_bits.u8'));files={k:(out/v[2]).open('xb')for k,v in spec.items()};hashes={k:hashlib.sha256()for k in spec};finite_total=front_total=inside_total=0;framecounts=[]
 try:
  for fi,pi in enumerate(selected):
   c=p['camera25'][pi];h=world_plane_homography(c)
   with np.errstate(divide='ignore',invalid='ignore',over='ignore'):uv=project_xy(h,xyz[:,:2])
   R=Rotation.from_rotvec(c[3:6]).as_matrix();depth=(xyz-c[:3])@R[2];finite=np.isfinite(uv).all(1)&np.isfinite(depth);front=finite&(depth>.05);inside=finite&(uv[:,0]>=0)&(uv[:,0]<=1919)&(uv[:,1]>=0)&(uv[:,1]<=1079);flags=finite.astype('u1')|(front.astype('u1')<<1)|(inside.astype('u1')<<2)
   for key,a in [('screen_uv',uv),('depth',depth),('vertex_mask_bits',flags)]:
    view=memoryview(np.ascontiguousarray(a,dtype=spec[key][0])).cast('B');files[key].write(view);hashes[key].update(view)
   finite_total+=int(finite.sum());front_total+=int(front.sum());inside_total+=int((inside&front).sum());framecounts.append([int(finite.sum()),int(front.sum()),int((inside&front).sum())])
   if fi%1000==0:print(json.dumps(dict(completed_frames=fi+1,total_frames=F,vertices_per_frame=V,seconds=time.perf_counter()-begin)),flush=True)
 finally:
  for f in files.values():f.close()
 for key,(dt,shape,name)in spec.items():
  file=out/name;expected=int(np.prod(shape,dtype=np.int64))*np.dtype(dt).itemsize
  if file.stat().st_size!=expected:raise ValueError('Incomplete direct state array '+key)
  fields[key]=dict(dtype=dt,shape=shape,storage='raw_binary_C_order',file=str(file.resolve()),sha256=hashes[key].hexdigest(),capacity_bytes=expected)
 countfile=out/'frame_vertex_counts.npy'
 with countfile.open('xb')as f:np.save(f,np.asarray(framecounts,'<i4'),allow_pickle=False)
 fields['frame_vertex_counts']=dict(dtype='<i4',shape=[F,3],storage='npy_memmap',**ref(countfile));shots={}
 for s in scope['shots']:
  if s['id']=='s01':shots['s01']=s01pres;continue
  mask=shot==int(s['id'][1:]);rows=np.flatnonzero(mask);shots[s['id']]=dict(state_kind='direct_screen_grid',frame_start=int(rows[0]),frame_stop_exclusive=int(rows[-1]+1),native_frames=len(rows),first_pts=s['first_pts'],last_pts=s['last_pts'],bridge_mode_counts=dict(Counter(map(int,p['bridge_mode'][selected][mask]))),projection_materialization_only=True,observations_created=False)
 modelref=DATA/'runs/okada_full9_recovery_1019/baseline_import_gpu32_v1/navigation_model.json'
 doc=dict(schema='rinkfit.direct_screen_grid_initial_state.v1',state_kind='mixed_preserved_S01_and_direct_screen_grid',status='complete',scope=ref(scope_path),source_identity=scope['source'],source_packet=ref(src),source_packet_receipt=ref(src.with_suffix('.json')),V3_camera_patch=ref(patchfile),V3_camera_patch_receipt=ref(patchfile.with_suffix('.json')),old_scene_reference=ref(oldpath),current_physical_reference=ref(modelref),s01_preservation=s01pres,shots=shots,fields=fields,topology=ref(topology),materializer=ref(__file__),projection_implementation=ref(ROOT/'calibration/program_tracking/demo_camera_patch.py'),inventory_implementation=ref(ROOT/'calibration/program_tracking/render_result.py'),original_path_implementation=ref(ROOT/'calibration/program_tracking/projection.py'),frames=F,vertices_per_frame=V,total_screen_vertices=F*V,paths_per_frame=len(ice),segments_per_frame=len(segments),native_time_base='1/12800',native_fps='50/1',native_pts_step=256,all9_scope_frames=8434,Demo_S01_rows_written=0,screen_grid_shots=['s%02d'%i for i in range(2,10)],operation='deterministic materialization of complete existing Demo no-Brown native screen layout; no fitting, optimization, calibration or new clock conversion',screen_UV_are_direct_initial_state=True,render_contract='render only stored screen_uv plus stored depth/mask/topology; do not reconstruct through any current camera, project_rs or model; S01 uses original preserved baseline route',vertex_mask_bits={0:'finite(UV,depth)',1:'depth > 0.05 old conditional scene units',2:'finite UV inside [0,1919]x[0,1079]'},mask_is_visibility_only_original_vertices_not_removed=True,raw_UV_unclamped_and_nonfinite_values_preserved=True,depth_use='allows original finite-segment screen and near-plane clipping directly from homogeneous screen UV*depth/depth without reconstructing camera',source_bridge_modes=srcdoc['bridge_mode'],counts=dict(finite_vertex_rows=finite_total,forward_vertex_rows=front_total,onscreen_forward_vertex_rows=inside_total),array_bytes=sum(Path(r['file']).stat().st_size for r in fields.values()),elapsed_seconds=time.perf_counter()-begin,source_observations_W_Query_qualification_changed=False,old_camera25_values_copied_into_initial_state=False,current_physical_camera_fitted_or_modified=False,current_shared_geometry_or_S01_values_changed=False,curve_crosswalk_or_sampling_used=False,source_accuracy_accepted=False)
 write(out/'manifest.json',doc);print(json.dumps(dict(manifest=str(out/'manifest.json'),frames=F,vertices_per_frame=V,paths=len(ice),segments=len(segments),total_screen_vertices=F*V,bytes=doc['array_bytes'],seconds=doc['elapsed_seconds'],Demo_S01_rows_written=0,fields={k:v for k,v in fields.items()if k in spec})),flush=True)
if __name__=='__main__':main()
