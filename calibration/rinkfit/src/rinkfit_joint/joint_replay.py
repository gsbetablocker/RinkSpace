"""Immutable 8434-frame camera ledger and same-state native review rendering."""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json,time,re
from fractions import Fraction
import av,cv2,numpy as np
from .joint_model import as_event_view,graph_from_joint,validate_physical_domain
from .camera_graph import CameraGraph
from .io_utils import read_json,write_json,file_sha256
from rinkfit_start.native_scene import NativeScene
from rinkfit_start.native_camera import rotation

def camera_ledger(model_file,scope_file,output):
 out=Path(output)
 if out.exists():raise ValueError('A new immutable replay generation is required')
 model=read_json(model_file);scope=read_json(scope_file);graph=graph_from_joint(model);cg=CameraGraph(graph);validation=cg.validate(require_ready=True);physical=validate_physical_domain(model);out.mkdir(parents=True)
 values=[];pts=[];sids=[];rows=[];certs=[];model_sha=file_sha256(model_file)
 with gzip.open(out/'camera_rows.jsonl.gz','wt',encoding='utf8') as f:
  for si,s in enumerate(scope['shots']):
   sid=s['id'];camera=cg.native_camera(sid);stamp=np.arange(s['first_pts'],s['last_pts']+1,256,dtype=np.int64);local=cg.local_seconds(sid,stamp);abf=camera.spline(local);alpha,beta,logf=abf.T;R=rotation(camera.b,beta)@camera.R0@rotation(camera.n,alpha);z=logf-np.log(float(camera.optics['f_ref']));w=np.einsum('...ji,j->...i',R,camera.b);h=np.cross(camera.n,w);h/=np.linalg.norm(h,axis=1)[:,None];D=np.broadcast_to(camera.d,(len(stamp),3)).copy();D[:,2]+=camera.head.get('pupil_linear',0)*z+camera.head.get('pupil_quadratic',0)*z*z;C=camera.P+np.einsum('...ji,...j->...i',R,D)+camera.a*h;db=camera.derivative_bounds(local)
   full_tau=float(graph['rigs'][s['rig_id']].get('tau_bound',.02));duration=(s['last_pts']-s['first_pts'])/12800.;halo=bool(camera.domain[0]<=-full_tau/2+1e-12 and camera.domain[1]>=duration+full_tau/2-1e-12);opt=camera.optical_domain_certificate();certs.append(dict(shot_id=sid,optical=opt,full_allowed_tau_halo=halo,current_tau=camera.tau,allowed_tau_bound=full_tau,cubic_domain=camera.domain))
   for j,p in enumerate(stamp):
    row=dict(shot_id=sid,rig_id=s['rig_id'],native_pts=int(p),programme_frame_index_1based=s['programme_frames'][0]+j,time_base='1/12800',source_seconds=int(p)/12800.,local_seconds=float(local[j]),alpha_beta_logf=abf[j].tolist(),camera_center_L1=C[j].tolist(),world_to_camera_rotation=R[j].tolist(),focal_native_px=float(np.exp(logf[j])),signed_readout_seconds=camera.tau,readout_time_interval_seconds=[float(local[j]-abs(camera.tau)/2),float(local[j]+abs(camera.tau)/2)],whole_allowed_readout_halo_covered=halo,optical_domain_certified=bool(opt['passed']),mid_exposure_state_is_not_pixel_accuracy_claim=True,model_sha256=model_sha)
    f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
   values.append(np.c_[local,abf,C,R.reshape(-1,9),db]);pts.append(stamp);sids.append(np.full(len(stamp),si,np.uint8))
 np.savez_compressed(out/'cameras.npz',native_pts=np.concatenate(pts),shot_index=np.concatenate(sids),camera_values=np.concatenate(values),columns=np.asarray(['local_seconds','alpha','beta','logf','Cx','Cy','Cz',*[f'R{i}{j}' for i in range(3) for j in range(3)],'yaw_derivative_bound','pitch_derivative_bound','logf_derivative_bound']))
 total=sum(len(x) for x in pts)
 if total!=8434 or len(set(np.concatenate(pts).tolist()))!=8434:raise ValueError('Nine-shot camera ledger is incomplete or duplicated')
 receipt=dict(schema='rinkfit.joint9.full_camera_replay.v1',model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),scope=dict(path=str(Path(scope_file).resolve()),sha256=file_sha256(scope_file)),graph_schema=validation,physical_domain=physical,frames=total,all8434_camera_states_present=True,per_shot=certs,all_optical_domains_passed=all(c['optical']['passed'] for c in certs),all_full_tau_halos_covered=all(c['full_allowed_tau_halo'] for c in certs),source_pixel_accuracy_verified=False,independent_metre_scale=False,coordinate_gauge='L=1 full rink length',arrays_sha256=file_sha256(out/'cameras.npz'),ledger_sha256=file_sha256(out/'camera_rows.jsonl.gz'))
 write_json(out/'manifest.json',receipt);return receipt

def _display_board_closure(scene,path):
 """Sample a declared closing ink edge on the actual curved board surface.

 kind=2's observation station remains the original finite N-1-edge path.
 Display must still honour closed=True when the first vertex is not repeated.
 """
 points=np.asarray(path['points'],float)
 uv=np.linspace(points[-1],points[0],33)[1:]
 block=int(path.get('geometry_block',-1))
 if block>=0:
  q=scene.values[block];uv=uv@np.array([[q[0],q[3]],[q[1],q[4]]])+q[[2,5]]
 ad=scene.values[int(path['advertisement_block'])]
 return scene.board(ad[0]+ad[1]*uv[:,0],ad[2]+(ad[3]-ad[2])*uv[:,1])

def _source_track_unknown_point(path):
 """Latent track coordinates are not current-exposure source dots or grid ink."""
 if not path.get('physical_point'):return False
 provenance=path.get('provenance',{})
 if isinstance(provenance,str):
  try:provenance=json.loads(provenance)
  except (ValueError,TypeError):return False
 return bool(isinstance(provenance,dict) and provenance.get('track_id') and provenance.get('geometry_role') in ('shared_unknown_ice_XY_z0','unknown_shared_material_uv_on_existing_continuous_board_ad'))

def inventory(model):
 scene=NativeScene(as_event_view(model,'s01'));xyz=[];records=[]
 for mi,m in enumerate(scene.materials):
  for pi,pa in enumerate(m['paths']):
   kind=pa['kind'];n=1 if pa.get('physical_point') else (257 if kind in (1,7) else max(33,min(1025,4*len(pa.get('points',[])))) if kind==2 else 129)
   if kind==6 and not pa.get('physical_point'):
    block=scene.values[pa['geometry_block']];indices=pa.get('geometry_indices');coordinates=len(indices) if indices else len(block)-pa.get('geometry_offset',0);vertices=coordinates//2;edges=vertices if pa.get('closed',False) else vertices-1
    if coordinates%2 or edges<1:raise ValueError('Shared board print path lacks complete finite vertices')
    n=max(4,int(np.ceil(128/edges)))*edges+1
   grid=np.linspace(0,1,n)
   if kind==2 and len(pa.get('points',[]))>1:grid=np.unique(np.r_[grid,np.arange(len(pa['points']))/(len(pa['points'])-1)])
   intervals=pa.get('allowed_station_intervals',[[0.,1.]])
   if 'allowed_station_intervals'in pa:
    if model.get('path_domain_semantics',{}).get('schema')!='rinkfit.source_finite_station_domain.v1':raise ValueError('Source finite station domain semantics missing')
    refs=pa.get('station_domain_source_refs',[])
    if not refs:raise ValueError('Restricted material station domain requires actual source proof')
    for ref in refs:
     source=ref.get('file',ref.get('path'))
     if file_sha256(source)!=ref['sha256']:raise ValueError('Material station domain source changed')
   previous=-np.inf
   for interval in intervals:
    lo,hi=map(float,interval)
    if not np.isfinite([lo,hi]).all() or not 0.<=lo<=hi<=1. or lo<previous:raise ValueError('Invalid material finite station interval union')
    previous=hi;stations=np.array([lo]) if pa.get('physical_point') else np.unique(np.r_[lo,grid[(grid>=lo)&(grid<=hi)],hi]);q=scene.world(mi,pi,stations)
    if kind==2 and pa.get('closed',False) and lo==0. and hi==1. and len(pa['points'])>1 and not np.array_equal(pa['points'][0],pa['points'][-1]):q=np.concatenate((q,_display_board_closure(scene,pa)))
    label=m['label'];color=(40,225,240) if label.startswith(('Ad','print_component.')) else (80,230,80) if label.startswith('board') else (255,140,50);start=sum(len(x) for x in xyz);xyz.append(q);records.append(dict(material_index=mi,path=pi,label=label,begin=start,end=start+len(q),closed=bool(pa.get('closed',False) and lo==0. and hi==1.),original_path_closed=pa.get('closed',False),station_domain_interval=[lo,hi],station_samples=stations.tolist() if 'allowed_station_intervals'in pa else None,source_station_domain_applied='allowed_station_intervals'in pa,latent_source_track_point=_source_track_unknown_point(pa),board_material_surface=bool(kind in (2,3,6,7) or label.startswith(('Ad','board.','print_component.'))),color=color))
 # Actual ad labels follow the solved surfaces, once per physical panel.
 byad={};allxyz=np.concatenate(xyz,axis=0)
 for r in records:
  ad=re.match(r'^Ad(\d+)[.]',r['label'])
  if ad:byad.setdefault(ad.group(1),[]).append(allxyz[r['begin']:r['end']])
 for ad,points in byad.items():
  q=np.concatenate(points);centre=np.mean(q,axis=0,keepdims=True);start=sum(len(x) for x in xyz);xyz.append(centre);records.append(dict(material_index=-1,path=-1,label='Ad'+ad,begin=start,end=start+1,closed=False,color=(0,255,255)))
 # Ice identifiers use only the currently solved material paths; this adds no geometry.
 byice={}
 for r in records:
  if r['label'].startswith('ice.') and '.tracklet.' not in r['label'] and r['material_index']>=0:
   feature=r['label'][4:].split(':')[0];byice.setdefault(feature,[]).append(allxyz[r['begin']:r['end']])
 for feature,points in byice.items():
  q=np.concatenate(points);centre=((q.min(axis=0)+q.max(axis=0))/2)[None];start=sum(len(x) for x in xyz);xyz.append(centre);label=feature.replace('house_connection_','').replace('_','-');records.append(dict(material_index=-1,path=-1,label=label,label_kind='ice',begin=start,end=start+1,closed=False,color=(255,190,90)))
 return np.concatenate(xyz),records

def fast_native_project(camera,world,local):
 """Same physical evaluator, with certified fixed-point roots then bisection.

No off-sensor root is certified; finite clipping only draws legal subsegments.
"""
 mid=camera.at(world,local);v=np.clip(mid['uv'][:,1],0,1079);cert=camera.rs_certificate(world,local)
 for _ in range(12):
  a=camera.at(world,local+camera.tau*(v/1079-.5));nv=np.clip(a['uv'][:,1],0,1079)
  if np.max(np.abs(nv-v),initial=0)<1e-9:v=nv;break
  v=nv
 a=camera.at(world,local+camera.tau*(v/1079-.5));eq=v-a['uv'][:,1];good=a['valid']&cert['certified']&(np.abs(eq)<2e-8)
 unresolved=a['valid']&cert['certified']&(~good)&(mid['uv'][:,1]>=0)&(mid['uv'][:,1]<=1079)
 if unresolved.any():
  ids=np.flatnonzero(unresolved);b=camera.project_rs(world[ids],local);a['uv'][ids]=b['uv'];good[ids]=b['valid'];eq[ids]=b['root_equality_px']
 return a['uv'],good,eq

def render_visibility(camera,scene,world,records,local,uv):
 """Cull hidden inner-board faces and rays blocked by the finite near board.

This changes display visibility only, never source identity or fit admission.
"""
 row=np.clip(np.nan_to_num(uv[:,1],nan=539.5,posinf=1079.,neginf=0.),0,1079);tm=local+camera.tau*(row/1079-.5);a=camera.spline(tm);R=rotation(camera.b,a[:,1])@camera.R0@rotation(camera.n,a[:,0]);z=a[:,2]-np.log(float(camera.optics['f_ref']));wb=np.einsum('...ji,j->...i',R,camera.b);h=np.cross(camera.n,wb);h/=np.linalg.norm(h,axis=1)[:,None];D=np.broadcast_to(camera.d,(len(world),3)).copy();D[:,2]+=camera.head.get('pupil_linear',0)*z+camera.head.get('pupil_quadratic',0)*z*z;C=camera.P+np.einsum('...ji,...j->...i',R,D)+camera.a*h
 u=np.maximum((np.abs(world[:,0])-(.5-scene.rx))/scene.rx,0);v=np.maximum((np.abs(world[:,1])-(scene.width/2-scene.ry))/scene.ry,0);normal=-np.c_[np.sign(world[:,0])*u**(scene.p-1)/scene.rx,np.sign(world[:,1])*v**(scene.p-1)/scene.ry]
 board=np.zeros(len(world),bool)
 for r in records:
  if r.get('board_material_surface',r['label'].startswith(('Ad','board.','print_component.'))):board[r['begin']:r['end']]=True
 facing=np.sum(normal*(C[:,:2]-world[:,:2]),axis=1)>=-1e-12;visible=(~board)|facing
 outside=scene.containment(C[:,:2])>1e-10
 if outside.any():
  lo=np.zeros(len(world));hi=np.ones(len(world));direction=world-C
  for _ in range(36):
   mid=(lo+hi)/2;q=C+direction*mid[:,None];inside=scene.containment(q[:,:2])<=0;hi=np.where(inside,mid,hi);lo=np.where(inside,lo,mid)
  entry=C+direction*hi[:,None];blocked=outside&(hi<1-1e-7)&(entry[:,2]<scene.height-1e-10)&(entry[:,2]>=0);visible&=~blocked
 return visible

def draw(bgr,uv,valid,records,title):
 image=bgr.copy();paths=0;labels=0
 for r in records:
  if r.get('latent_source_track_point',False):continue
  q=uv[r['begin']:r['end']];g=valid[r['begin']:r['end']]
  if r['material_index']<0:
   if g[0] and 0<=q[0,0]<1920 and 0<=q[0,1]<1080:
    cv2.putText(image,r['label'],tuple(np.rint(q[0]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA);cv2.putText(image,r['label'],tuple(np.rint(q[0]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.65,r['color'],1,cv2.LINE_AA);labels+=int(r['label'].startswith('Ad'))
   continue
  if len(q)==1:
   if g[0] and (q[0]>=0).all() and (q[0]<[1920,1080]).all():cv2.circle(image,tuple(np.rint(q[0]).astype(int)),3,r['color'],1,cv2.LINE_AA)
   continue
  for a,b,yes in zip(q[:-1],q[1:],g[:-1]&g[1:]):
   if yes and np.max(np.abs([a,b]))<1e7:
    ok,x,y=cv2.clipLine((0,0,1920,1080),tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
    if ok:cv2.line(image,x,y,r['color'],1,cv2.LINE_AA);paths+=1
 cv2.rectangle(image,(0,1038),(1920,1080),(12,23,28),-1);cv2.putText(image,title,(18,1067),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,245,245),1,cv2.LINE_AA)
 return image,dict(visible_subsegments=paths,visible_ad_labels=labels)


def draw_fast(bgr,uv,valid,records,title):
 """Draw the same finite runs with OpenCV polylines instead of per-edge calls."""
 image=bgr.copy();paths=0;labels=0
 for r in records:
  if r.get('latent_source_track_point',False):continue
  q=uv[r['begin']:r['end']];g=valid[r['begin']:r['end']].copy()
  g &= np.isfinite(q).all(axis=1)&(np.max(np.abs(q),axis=1)<1e7)
  if r['material_index']<0:
   if g[0] and 0<=q[0,0]<1920 and 0<=q[0,1]<1080:
    at=tuple(np.rint(q[0]).astype(int));cv2.putText(image,r['label'],at,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA);cv2.putText(image,r['label'],at,cv2.FONT_HERSHEY_SIMPLEX,.65,r['color'],1,cv2.LINE_AA);labels+=int(r['label'].startswith('Ad'))
   continue
  if len(q)==1:
   if g[0] and (q[0]>=0).all() and (q[0]<[1920,1080]).all():cv2.circle(image,tuple(np.rint(q[0]).astype(int)),3,r['color'],1,cv2.LINE_AA)
   continue
  # Split only at an invalid physical projection; no line crosses a failed root.
  edge=np.diff(np.r_[False,g,False].astype(np.int8));beg=np.flatnonzero(edge==1);end=np.flatnonzero(edge==-1)
  pieces=[np.rint(q[a:b]).astype(np.int32).reshape(-1,1,2) for a,b in zip(beg,end) if b-a>=2]
  if pieces:cv2.polylines(image,pieces,False,r['color'],1,cv2.LINE_AA)
  both=g[:-1]&g[1:];lo=np.minimum(q[:-1],q[1:]);hi=np.maximum(q[:-1],q[1:]);paths+=int(np.count_nonzero(both&(hi[:,0]>=0)&(lo[:,0]<=1919)&(hi[:,1]>=0)&(lo[:,1]<=1079)))
 cv2.rectangle(image,(0,1038),(1920,1080),(12,23,28),-1);cv2.putText(image,title,(18,1067),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,245,245),1,cv2.LINE_AA)
 return image,dict(visible_subsegments_bbox_intersecting=paths,visible_ad_labels=labels,physical_runs_drawn_with_clipped_OpenCV_polylines=True)

def render_snapshots(model_file,scope_file,output,pick_pts=None,*,shots=None,compiled_dir=None,gpu_dll=None,gpu_validation=None,packet_file=None,top_view=False,only_pick_pts=False):
 out=Path(output)
 if out.exists():raise ValueError('New immutable review generation required')
 model=read_json(model_file);scope=read_json(scope_file);cg=CameraGraph(graph_from_joint(model));world,records=inventory(model);scene=NativeScene(as_event_view(model,'s01'));sha=file_sha256(model_file);wanted={}
 for s in scope['shots']:
  if shots is not None and s['id'] not in shots:continue
  pp=np.arange(s['first_pts'],s['last_pts']+1,256,dtype=np.int64)
  if not only_pick_pts:
   for k in np.unique(np.linspace(0,len(pp)-1,5).round().astype(int)):wanted[int(pp[k])]=s['id']
 if pick_pts:
  for p in pick_pts:
   sid=next(s['id'] for s in scope['shots'] if s['first_pts']<=p<=s['last_pts'])
   if shots is not None and sid not in shots:raise ValueError('Requested snapshot is outside selected shots')
   cg.local_seconds(sid,np.asarray([p],np.int64));wanted[p]=sid
 if not wanted:raise ValueError('Snapshot selection is empty')
 if file_sha256(scope['source']['path'])!=scope['source']['sha256']:raise ValueError('Source SHA changed')
 out.mkdir(parents=True);frames=[];tiles=[]
 from .joint_display import CurrentSourcePoints,SharedModelTopView
 points=CurrentSourcePoints(packet_file,model,scope) if packet_file else None;top=SharedModelTopView(world,records,scene) if top_view else None
 if compiled_dir:
  from .joint_acceleration import install
  install(cg,model_file,out/'acceleration',world,compiled_dir,gpu_dll=gpu_dll,gpu_validation=gpu_validation,shots=list(dict.fromkeys(wanted.values())))
 with av.open(scope['source']['path']) as cont:
  st=cont.streams.video[0]
  if Fraction(st.time_base)!=Fraction(1,12800):raise ValueError('Native time base changed')
  for f in cont.decode(st):
   pts=int(f.pts)
   if pts not in wanted:
    if pts>max(wanted):break
    continue
   sid=wanted[pts];bgr=f.to_ndarray(format='bgr24');local=float(cg.local_seconds(sid,np.array([pts],np.int64))[0]);uv,good,eq=fast_native_project(cg.native_camera(sid),world,local);good&=render_visibility(cg.native_camera(sid),scene,world,records,local,uv);title=f'CANDIDATE {sid} source={pts/12800:.3f}s PTS={pts} model={sha[:12]} L=1';im,metrics=draw_fast(bgr,uv,good,records,title)
   if points is not None:metrics.update(points.draw(im,pts))
   if top is not None:im=np.concatenate([im,top.frame(cg.native_camera(sid),local,sid)],axis=1)
   fn=out/f'{sid}_{pts}.png';cv2.imwrite(str(fn),im,[cv2.IMWRITE_PNG_COMPRESSION,2]);tile=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA);tiles.append(tile);frames.append(dict(shot_id=sid,native_pts=pts,path=str(fn),sha256=file_sha256(fn),decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),same_model_sha256=sha,projected_path_samples=len(world),valid_path_samples=int(good.sum()),maximum_certified_root_equality_px=float(np.max(np.abs(eq[good]),initial=0)),**metrics))
 for i in range(0,len(tiles),12):
  canvas=np.zeros((810,1920,3),np.uint8)
  for j,t in enumerate(tiles[i:i+12]):y,x=divmod(j,4);canvas[y*270:(y+1)*270,x*480:(x+1)*480]=t
  cv2.imwrite(str(out/f'page_{i//12+1:02d}.png'),canvas)
 if len(frames)!=len(wanted):raise ValueError('Review missing a requested native exposure')
 receipt=dict(schema='rinkfit.joint9.same_model_native_snapshot_review.v1',reviewed_shot_ids=list(dict.fromkeys(wanted.values())),snapshot_selection_only=True,model=dict(path=str(Path(model_file).resolve()),sha256=sha),source=scope['source'],frames=frames,all_material_paths_enabled=True,latent_track_predictions_drawn=False,current_dots_only_from_source_packet=True,all_ad_labels_enabled=True,all_ice_lines_and_board_paths_enabled=True,physical_state_shared=True,current_actual_source_points_enabled=points is not None,current_actual_source_packet=points.packet_ref if points is not None else None,same_world_top_view_enabled=top is not None,inner_board_backfaces_and_occluding_near_board_culled=True,metric_scale_verified=False,visual_review_complete=False,video_generated=False);write_json(out/'manifest.json',receipt);return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['cameras','snapshots']);p.add_argument('--model',required=True);p.add_argument('--scope',required=True);p.add_argument('--out',required=True);p.add_argument('--pts',type=int,nargs='*');p.add_argument('--shots',nargs='+');p.add_argument('--compiled-dir');p.add_argument('--gpu-dll');p.add_argument('--gpu-validation');a=p.parse_args();print(json.dumps(camera_ledger(a.model,a.scope,a.out) if a.command=='cameras' else render_snapshots(a.model,a.scope,a.out,a.pts,shots=a.shots,compiled_dir=a.compiled_dir,gpu_dll=a.gpu_dll,gpu_validation=a.gpu_validation),ensure_ascii=False))
