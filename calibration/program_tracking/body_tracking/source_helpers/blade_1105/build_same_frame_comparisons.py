from pathlib import Path
import sys, os, json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');N=D/'runs/body_s01_20261029/blade_support_profile_1105';O=N.parent/'blade_orientation_1105'
sys.path[:0]=[str(P),str(P/'src')];os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_template import profile_height
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from rinkspace.viewer3d.source import SourceVideo
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
summary=read(N/'runner_support_run/run_summary.json');cfg=read(P/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_boot_axis_1105.json')
defs={'previous':(O/'entry_81_2_82_0',P/'events/ankara_2026/programmes/okada_mei_sp/spin_boot_axis_entry_1105.npz'),'runner_candidate':(N/'runner_support_run',Path(summary['parameter']))}
queries={k:SavedBladeProjection(P,*v) for k,v in defs.items()};points={};solids={};profiles={};body={}
selected=np.array([1121792,1135104,1160960,1173504,1183488],np.int64)
for key,m in queries.items():
 packet=m.binding.packet();p=packet['parameters'];xx=np.linspace(-p.runner_rear_extension_m,p.sole_length_m-p.runner_front_inset_m,65);curve=np.c_[xx,np.zeros(len(xx)),profile_height(xx,p)]
 station=np.array([xx[0],.5*(xx[0]+xx[-1]),xx[-1]]);q=np.c_[station,np.zeros(3),profile_height(station,p)];q=np.r_[q,packet['adapter'][[0,7]].mean(0)[None],packet['adapter'][[1,8]].mean(0)[None]]
 points[key]=m.project_points(selected,1,q)['uv'];solids[key]=m.project_points(selected,1,curve)['uv']
 times=m.times[np.searchsorted(m.pts,selected)];s=fk_state(m.reference,m.reference.at(times),np.zeros((len(times),69)));f=batch_features(m.binding,s['R'],s['O']);ids=[i for i,n in enumerate(f['ids']) if n in packet['template']['support_feature_ids']['glide']]
 profiles[key]=dict(R=s['R'][:,1],O=s['O'][:,1],curve=curve,height=f['height'][:,1,ids].min(1),ratio=m.ratio,lower=packet['template']['vertices'])
 with np.load(defs[key][0]/'common/exmr_hand_nofoot/fitted/result.npz',allow_pickle=False) as z:body[key]=z['COCO_uv'][np.searchsorted(z['source_pts'],selected)].copy()
manifest=read(N.parent/'blade_observability_1105/dual_source_manifest.json');existing={r['pts']:r for r in manifest['frames']};source=SourceVideo(cfg['source_video'],[1,12800]);frames_dir=N/'source_frames';frames_dir.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17);large=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',22);artifacts=[]
for j,pt in enumerate(selected):
 if int(pt) in existing:
  source_path=Path(existing[int(pt)]['source_frame']);im=Image.open(source_path).convert('RGB');origin='reused_exact_native_source'
 else:
  im=Image.fromarray(source.read(int(pt)));source_path=frames_dir/(str(pt)+'.png');im.save(source_path);origin='decoded_exact_native_source_no_interpolation'
 alluv=np.concatenate([points[k][j] for k in queries]);lo=alluv.min(0)-[110,90];hi=alluv.max(0)+[110,110];centre=(lo+hi)/2;w=max(400.,hi[0]-lo[0]);h=max(250.,hi[1]-lo[1]);roi=(int(centre[0]-w/2),int(centre[1]-h/2),int(centre[0]+w/2),int(centre[1]+h/2))
 sheet=Image.new('RGB',(1920,780),'white');draw=ImageDraw.Draw(sheet)
 for c,key in enumerate(['source','previous','runner_candidate']):
  panel=im.copy()
  if key!='source':
   d=ImageDraw.Draw(panel);uv=points[key][j]
   for a,b in [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:d.line([tuple(body[key][j,a]),tuple(body[key][j,b])],fill=(225,140,0),width=2)
   d.line([tuple(v) for v in solids[key][j]],fill=(40,220,255),width=3)
   d.line([tuple(v) for v in uv[3:]],fill=(255,150,20),width=2)
   for v,color,label in zip(uv[:3],[(55,85,255),(255,220,0),(255,255,255)],['R','M','F']):
    x,y=v;d.ellipse((x-4,y-4,x+4,y+4),fill=color,outline='black');d.text((x+5,y-6),label,font=font,fill=color,stroke_width=1,stroke_fill='black')
  sheet.paste(panel.resize((640,360)),(640*c,30));sheet.paste(panel.crop(roi).resize((640,350)),(640*c,395));draw.text((640*c+8,4),key+' '+str(pt/12800.)+'s',font=large,fill='black')
 draw.text((10,753),'Same native exposure/camera. Orange: body/boot. Cyan: exact curved runner. R/M/F: fixed material stations, NOT contact.',font=font,fill='black')
 output=N/('native_runner_compare_'+str(pt)+'.png');sheet.save(output)
 # A fixed longitudinal foot-side view of the same saved rigid steel, Z=0 shown.
 side=Image.new('RGB',(1200,540),'white');sd=ImageDraw.Draw(side);sd.text((12,8),'Saved foot-side geometry '+str(pt/12800.)+'s (conditional metric frame)',font=large,fill='black')
 side_heights=[((profiles[k]['O'][j]+profiles[k]['lower']@profiles[k]['R'][j].T)*profiles[k]['ratio'])[:,2] for k in queries]
 shared_zmin=min(-.025,min(float(v.min()) for v in side_heights)-.02);shared_zmax=max(.25,max(float(v.max()) for v in side_heights)+.02)
 for c,key in enumerate(queries):
  v=profiles[key];R=v['R'][j];Oworld=v['O'][j]*v['ratio'];forward=R[:,0].copy();forward[2]=0;forward/=np.linalg.norm(forward)
  local=v['lower'];world=(v['O'][j]+local@R.T)*v['ratio'];x=(world-Oworld)@forward;z=world[:,2]
  zmin=shared_zmin;zmax=shared_zmax;xmin=-.06;xmax=.36
  def xy(a,b):return (600*c+40+(a-xmin)/(xmax-xmin)*520,480-(b-zmin)/(zmax-zmin)*390)
  sd.line([xy(xmin,0),xy(xmax,0)],fill=(30,140,80),width=3)
  for px,pz in zip(x,z):
   a,b=xy(float(px),float(pz));sd.ellipse((a-1,b-1,a+1,b+1),fill=(170,180,185))
  curve=(v['O'][j]+v['curve']@R.T)*v['ratio'];cx=(curve-Oworld)@forward
  sd.line([xy(float(a),float(b)) for a,b in zip(cx,curve[:,2])],fill=(20,80,200),width=3)
  sd.text((600*c+12,45),key,font=large,fill='black');sd.text((600*c+12,70),'runner min Z %.1f cm'%(v['height'][j]*100),font=font,fill='black')
 sd.text((12,510),'Z=0 is the existing nominal ice reference. Raw height is NOT calibrated real contact truth.',font=font,fill='black')
 sidepath=N/('side_runner_compare_'+str(pt)+'.png');side.save(sidepath)
 artifacts.append(dict(pts=int(pt),seconds=float(pt/12800.),source_frame=str(source_path),source_frame_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),source_origin=origin,comparison=str(output),side_view=str(sidepath),source_pixel_observations_added=0,source_contact_material='unknown'))
source.close();(N/'same_frame_comparison_manifest.json').write_text(json.dumps(dict(schema='rinkspace.runner-same-frame-comparison.v1',time_base=[1,12800],frames=artifacts,no_video_generated=True,no_source_observations_inferred_from_model=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(comparisons=len(artifacts),manifest=str(N/'same_frame_comparison_manifest.json'))))
