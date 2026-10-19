"""Original-PTS static-boundary diagnostics; aggregates are never new observations."""
from pathlib import Path
import argparse,csv,json,numpy as np
from rinkfit_start.native_scene import camera_from_event
from rinkfit_start.candidate_receipt import read,resolved,ref

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--semantic',required=True);p.add_argument('--baseline',required=True);p.add_argument('--state',required=True);p.add_argument('--output',required=True);p.add_argument('--source');p.add_argument('--frame-window',type=int,nargs=2,required=True);p.add_argument('--corner-label',required=True);p.add_argument('--board-prefix',required=True);p.add_argument('--ring-prefix',required=True);p.add_argument('--raw-frames',type=int,nargs='*',default=[]);p.add_argument('--diagnostic-u',type=float);a=p.parse_args(argv)
 PK=Path(a.packet);SP=Path(a.semantic);CP=Path(a.baseline);D=Path(a.state);O=Path(a.output)
 from .state_integrity import validate_report_files,sha256_file
 from .semantic_resolution import load_semantics
 validate_report_files(D);validate_report_files(CP)
 if O.exists():raise ValueError('Refuse overwriting evidence')
 O.mkdir(parents=True);packet=read(PK);sem=read(SP);model=read(CP/'model.json');newmodel=read(D/'model.json');cam=camera_from_event(model);newcam=camera_from_event(newmodel)
 def array(d,base):return np.fromfile(resolved(base,d),d['dtype']).reshape(d['shape'])
 _,verified_arrays,_,semantic_extra=load_semantics(SP,int(packet['observations']),expected_sha256=sha256_file(SP));require_active=array(sem['active_source_mask'],SP.parent).astype(bool)
 if not np.array_equal(verified_arrays['kind'],array(sem['arrays']['kind'],SP.parent)):raise ValueError('Selected semantic kind differs from verified parent chain')
 if not np.array_equal(semantic_extra['source_exclusion']['active_mask'],require_active):raise ValueError('Selected active scope differs from verified source authority')
 frames=array(packet['arrays']['frame_index'],PK.parent);pts=array(packet['arrays']['native_pts'],PK.parent);uv=array(packet['arrays']['native_uv'],PK.parent);active=array(sem['active_source_mask'],SP.parent).astype(bool);mi=np.asarray(semantic_extra['resolved_material_index']);labels=np.array([model['materials'][int(i)]['label']for i in mi]);old=np.load(CP/'candidate_predictions.npz');new=np.load(D/'candidate_predictions.npz');oldUV=old['uv'];newUV=new['uv'];N=len(uv)
 if not(len(oldUV)==len(newUV)==len(active)==N):raise ValueError('Source row count differs')
 if not np.array_equal(new['native_pts'],pts)or not np.array_equal(new['active_source_mask'],active):raise ValueError('Candidate source identity differs')
 if 'native_pts'in old and not np.array_equal(old['native_pts'],pts):raise ValueError('Baseline PTS differs')
 F=np.arange(a.frame_window[0],a.frame_window[1]+1);frame_pts=[]
 for f in F:
  t=np.unique(pts[frames==f])
  if len(t)!=1:raise ValueError('Frame has no unique native PTS: '+str(f))
  frame_pts.append(int(t[0]))
 tb=float(packet['native_time_base'][0])/float(packet['native_time_base'][1]);center=(np.array(frame_pts)-int(packet['pts_origin']))*tb;diagnostic_u=float(a.diagnostic_u)if a.diagnostic_u is not None else float(packet['native_image_wh'][0])/2
 masks=[labels==a.corner_label,labels==a.corner_label,np.char.startswith(labels,a.board_prefix),np.char.startswith(labels,a.ring_prefix)];cols={'Original UV':'#18252c','baseline':'#d76718','candidate':'#0c829b'};panels=[]
 for coordinate in [0,1]:
  vals={k:[]for k in cols}
  for f in F:
   jj=np.flatnonzero(active&(frames==f)&masks[coordinate])
   for key,arr in [('Original UV',uv),('baseline',oldUV),('candidate',newUV)]:vals[key].append(float(np.mean(arr[jj,coordinate]))if len(jj)else np.nan)
  panels.append((a.corner_label+' '+('u'if coordinate==0 else'v')+' (native px)',vals))
 vals={k:[]for k in cols}
 for f in F:
  jj=np.flatnonzero(active&(frames==f)&masks[2])
  for key,arr in [('Original UV',uv),('baseline',oldUV),('candidate',newUV)]:
   q=arr[jj];vals[key].append(float(np.polyval(np.polyfit(q[:,0],q[:,1],1),diagnostic_u))if len(jj)>2 else np.nan)
 panels.append(('board top fitted v at u='+str(diagnostic_u)+' (px)',vals));vals={k:[]for k in cols}
 for f in F:
  jj=np.flatnonzero(active&(frames==f)&masks[3]);vals['Original UV'].append(0. if len(jj)else np.nan)
  for key,arr in [('baseline',oldUV),('candidate',newUV)]:vals[key].append(float(np.mean(arr[jj,1]-uv[jj,1]))if len(jj)else np.nan)
 panels.append((a.ring_prefix+' mean predicted v - Original UV (px)',vals));oldt=cam.spline(center);newt=newcam.spline(center);panels.append(('focal length (native px)',{'baseline':np.exp(oldt[:,2]),'candidate':np.exp(newt[:,2])}));panels.append(('beta angle (degrees)',{'baseline':np.rad2deg(oldt[:,1]),'candidate':np.rad2deg(newt[:,1])}))
 records=[];series=[]
 for i,(title,data)in enumerate(panels):
  label=[a.corner_label,a.corner_label,a.board_prefix+'*',a.ring_prefix+'*','camera.focal','camera.beta'][i];counts=[]
  for j,f in enumerate(F):
   ids=np.flatnonzero(active&(frames==f)&masks[i])if i<4 else np.array([],int);counts.append(len(ids));row=dict(frame=int(f),source_pts=frame_pts[j],source_seconds=frame_pts[j]*tb,panel=i,label=label,quantity=title,source_rows=len(ids))
   for key in cols:
    v=data.get(key,[np.nan]*len(F))[j];row[key]=float(v)if np.isfinite(v)else None
   records.append(row)
  derivation='Mean direct original material-point coordinate, identity fixed by source label'if i<2 else'Least-squares line v(u) evaluated at diagnostic u using current frame board top rows; diagnostic aggregate, not an independent actual annotation or new observation'if i==2 else'Mean predicted v minus original v over current frame ring rows; diagnostic aggregate, not an independent annotation or new observation'if i==3 else'Camera function at original exposure-center native PTS; model state, not an observed hardware measurement'
  series.append(dict(panel=i,label=label,quantity=title,source_row_counts=counts,values={k:[float(v)if np.isfinite(v)else None for v in values]for k,values in data.items()},derivation=derivation))
 output=dict(schema='rinkfit.native_temporal_panel_numeric_series.v1',state=str(D),source_scope=int(active.sum()),packet=ref(PK),semantic=ref(SP),baseline_model=ref(CP/'model.json'),candidate_model=ref(D/'model.json'),source_time_base=packet['native_time_base'],pts_origin=packet['pts_origin'],frames=F.tolist(),source_pts=frame_pts,panels=series,summary_rows_are_not_new_observations=True,source_UV_and_W_E_unchanged=True)
 (O/'temporal_series.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 with(O/'temporal_series.csv').open('w',encoding='utf-8-sig',newline='')as file:
  w=csv.DictWriter(file,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 from PIL import Image,ImageDraw,ImageFont
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);tiny=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
 im=Image.new('RGB',(1800,1520),'white');draw=ImageDraw.Draw(im);draw.text((32,16),f'Native PTS source frames {F[0]}-{F[-1]} = {frame_pts[0]*tb:.3f}-{frame_pts[-1]*tb:.3f}s; raw rows kept',font=font,fill='#142b36')
 for i,(title,data)in enumerate(panels):
  x0=90+(i%2)*900;y0=100+(i//2)*470;w=740;h=330;finite=[np.asarray(v)[np.isfinite(v)]for v in data.values()];allv=np.concatenate(finite);lo=float(allv.min());hi=float(allv.max());span=max(hi-lo,.1);lo-=.08*span;hi+=.08*span;draw.text((x0-40,y0-38),title,font=font,fill='#142b36');draw.rectangle((x0,y0,x0+w,y0+h),outline='#aebdc5',width=1)
  for v in np.linspace(lo,hi,5):
   yy=y0+h-(v-lo)/(hi-lo)*h;draw.line((x0,yy,x0+w,yy),fill='#e5ecef');draw.text((x0-80,yy-8),f'{v:.2f}',font=tiny,fill='#526a77')
  for tick in np.linspace(int(F[0]),int(F[-1]),6):
   xx=x0+(tick-F[0])/(F[-1]-F[0])*w;draw.text((xx-12,y0+h+8),str(int(round(tick))),font=tiny,fill='#526a77')
  for key,values in data.items():
   points=[]
   for f,v in zip(F,values):
    if not np.isfinite(v):
     if len(points)>1:draw.line(points,fill=cols[key],width=3)
     points=[];continue
    xx=x0+(f-F[0])/(F[-1]-F[0])*w;yy=y0+h-(v-lo)/(hi-lo)*h;points.append((xx,yy));draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=cols[key])
   if len(points)>1:draw.line(points,fill=cols[key],width=3)
  for j,key in enumerate(data):draw.text((x0+j*220,y0+h+42),key,font=small,fill=cols[key])
 im.save(O/'temporal_boundaries.png');receipts=[]
 if a.raw_frames:
  if not a.source:raise ValueError('--source is required for raw source-frame overlays')
  import av
  wanted={}
  for f in a.raw_frames:
   t=np.unique(pts[frames==f])
   if len(t)!=1:raise ValueError('Raw frame has no unique PTS')
   wanted[int(t[0])]=int(f)
  with av.open(a.source)as container:
   stream=container.streams.video[0]
   if float(stream.time_base)!=tb:raise ValueError('Original video PTS timebase differs from packet')
   container.seek(min(wanted),stream=stream,backward=True)
   for frame in container.decode(stream):
    if frame.pts in wanted:
     f=wanted[frame.pts];image=frame.to_image();dr=ImageDraw.Draw(image);jj=np.flatnonzero(active&(frames==f));width,height=image.size
     for key,arr,color,radius in [('baseline',oldUV,'#ff7a1c',3),('candidate',newUV,'#00def5',3),('Original UV',uv,'#1e44ff',2)]:
      for x,y in arr[jj]:
       if np.isfinite(x+y)and -20<=x<=width+20 and -20<=y<=height+20:dr.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
     dr.rectangle((0,0,width,66),fill='white');dr.text((18,8),f'Frame {f}; PTS={frame.pts}; source {frame.pts*tb:.3f}s | blue original UV | orange baseline | cyan candidate',font=font,fill='#122f3d');image.save(O/f'raw_compare_f{f:04d}.png');receipts.append(dict(frame=f,actual_pts=frame.pts,time_base=str(frame.time_base),rows=len(jj),file=f'raw_compare_f{f:04d}.png'))
    if frame.pts>max(wanted):break
  if len(receipts)!=len(wanted):raise ValueError('Requested native PTS frame missing')
 (O/'evidence.json').write_text(json.dumps(dict(schema='rinkfit.native_static_boundary_comparison.v1',implementation_source=ref(Path(__file__)),source=a.source,source_time_base=packet['native_time_base'],same_original_UV=True,selected_active_rows=int(active.sum()),packet=ref(PK),semantic=ref(SP),baseline_model=ref(CP/'model.json'),model=ref(D/'model.json'),diagnostic_only_not_full_shot_visual_acceptance=True,raw_frames=receipts),ensure_ascii=False,indent=2),encoding='utf-8');print(str(O))
if __name__=='__main__':main()
