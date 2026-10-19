"""Native first-frame/pulse boundary observations and same-state true-RS curves."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .joint_metrics import PacketArrays
from .joint_model import graph_from_joint,as_event_view
from .camera_graph import CameraGraph
from .joint_replay import fast_native_project,render_visibility
from rinkfit_start.native_scene import NativeScene

PLOT_SCRIPT="import sys\nimport numpy as np\nfrom reportlab.graphics.shapes import Drawing,String,Line,Rect\nfrom reportlab.graphics.charts.lineplots import LinePlot\nfrom reportlab.graphics.widgets.markers import makeMarker\nfrom reportlab.graphics import renderSVG\nfrom reportlab.lib import colors\nz=np.load(sys.argv[1]);frames=z['frames'];x=z['x'];observed=z['observed_y'];predicted=z['predicted_y'];d=Drawing(1600,1000);d.add(Rect(0,0,1600,1000,fillColor=colors.white,strokeColor=None));palette=[colors.HexColor(s)for s in ('#00669c','#d36b0c','#178953','#a13881','#6149ac')]\nd.add(String(800,969,'Actual immutable UV (solid dots) and same-model true-RS curve (dashed)',fontSize=22,textAnchor='middle'));d.add(String(800,940,'Model '+sys.argv[3][:16]+' | source first frame 20.020s; native PTS /12800; actual per-frame x retained',fontSize=14,textAnchor='middle'))\nfor column,(lo,hi,title)in enumerate(((0,10,'First native frames 1-10'),(10,16,'Frames 135-140'))):\n for j,role in enumerate(('Blue UPPER','Yellow LOWER')):\n  bx=column*790+85;by=550-j*420;lp=LinePlot();lp.x=bx;lp.y=by;lp.width=650;lp.height=290;series=[];styles=[]\n  for k,xx in enumerate(x):\n   for arr,pred in ((observed,False),(predicted,True)):\n    good=np.isfinite(arr[lo:hi,j,k]);edges=np.diff(np.r_[False,good,False].astype(int));beg=np.flatnonzero(edges==1);end=np.flatnonzero(edges==-1)\n    for a,b in zip(beg,end):\n     series.append([(float(frames[lo+q]),float(arr[lo+q,j,k]))for q in range(a,b)]);styles.append((k,pred))\n  if not series:d.add(String(bx,by+140,'No supported current samples',fontSize=18));continue\n  lp.data=series;lp.xValueAxis.valueMin=float(frames[lo]);lp.xValueAxis.valueMax=float(frames[hi-1]);lp.xValueAxis.valueStep=1;lp.xValueAxis.labels.fontSize=12;lp.yValueAxis.labels.fontSize=12;lp.yValueAxis.labelTextFormat='%.2f';lp.yValueAxis.visibleGrid=True;lp.yValueAxis.gridStrokeColor=colors.HexColor('#dddddd')\n  for index,(k,pred)in enumerate(styles):\n   lp.lines[index].strokeColor=palette[k];lp.lines[index].strokeWidth=1.8\n   if pred:lp.lines[index].strokeDashArray=[5,3]\n   else:lp.lines[index].symbol=makeMarker('FilledCircle',size=4,fillColor=palette[k],strokeColor=palette[k])\n  d.add(lp);d.add(String(bx,by+325,title+' / '+role,fontSize=19));d.add(String(bx,by-42,'Native programme frame (50 fps)',fontSize=14));d.add(String(bx-64,by+305,'y (px)',fontSize=13))\nfor k,xx in enumerate(x):\n xx0=160+k*280;d.add(Line(xx0,44,xx0+30,44,strokeColor=palette[k],strokeWidth=2));d.add(String(xx0+40,39,'actual source track '+str(int(xx)),fontSize=13))\nrenderSVG.drawToFile(d,sys.argv[2])\n"

def _at_x(points,x,maximum_gap):
 p=np.asarray(points,float);result=np.full(len(x),np.nan)
 if len(p)<2:return result
 p=p[np.argsort(p[:,0])];p=p[np.r_[True,np.diff(p[:,0])>1e-9]]
 for i,v in enumerate(x):
  k=int(np.searchsorted(p[:,0],v));k=max(1,min(k,len(p)-1));a,b=p[k-1:k+1]
  if a[0]<=v<=b[0] and b[0]-a[0]<=maximum_gap:result[i]=np.interp(v,[a[0],b[0]],[a[1],b[1]])
 return result

def audit(model_file,packet_file,output,*,plot_python=None):
 out=Path(output)
 if out.exists():raise ValueError('New temporal audit generation required')
 model=read_json(model_file);cg=CameraGraph(graph_from_joint(model));camera=cg.native_camera('s01');scene=NativeScene(as_event_view(model,'s01'));a=PacketArrays(packet_file);packet=a.packet;uv=a.load('uv');pts=a.load('native_pts');shot=a.load('shot_index');mat=a.load('material_index');active=(a.load('admission')==1)&(a.load('fit_role')==1);s01=shot==packet['shot_ids'].index('s01');labels=np.asarray([m['label']for m in model['materials']]);target=labels[mat];frames=np.r_[np.arange(1,11),np.arange(135,141)];stamps=cg.graph['shots']['s01']['scope']['first_pts']+(frames-1)*256;x=np.arange(1.,6.);anchors={};source_x=np.full((len(frames),2,len(x)),np.nan);source_row_indices=np.full((len(frames),2,len(x)),-1,dtype=np.int64);course=np.linspace(0,scene.perimeter,32769);observed=np.full((len(frames),2,len(x)),np.nan);predicted=observed.copy();rows=[]
 for i,(frame,stamp) in enumerate(zip(frames,stamps)):
  local=float(cg.local_seconds('s01',np.array([stamp],np.int64))[0]);counts=[]
  for j,(role,h)in enumerate((('board.top',scene.height),('board.contact',0.))):
   select=active&s01&(pts==stamp)&np.char.startswith(target,role);ids=np.flatnonzero(select);counts.append(len(ids));key=(i>=10,j)
   if key not in anchors and len(ids):
    ordered=ids[np.argsort(uv[ids,0])];anchors[key]=uv[ordered[np.linspace(0,len(ordered)-1,len(x)).round().astype(int)],0]
   if key in anchors and len(ids):
    for k,anchor in enumerate(anchors[key]):
     idx=ids[np.argmin(np.abs(uv[ids,0]-anchor))]
     if abs(uv[idx,0]-anchor)<=40.:observed[i,j,k]=uv[idx,1];source_x[i,j,k]=uv[idx,0];source_row_indices[i,j,k]=idx
   world=scene.board(course,h);q,good,equality=fast_native_project(camera,world,local);good&=render_visibility(camera,scene,world,[dict(label=role,begin=0,end=len(world))],local,q);good&=np.isfinite(q).all(1);predicted[i,j]=_at_x(q[good],source_x[i,j],64.)
  rows.append(dict(programme_frame=int(frame),native_pts=int(stamp),source_seconds=float(stamp/12800),local_seconds=local,actual_top_rows=counts[0],actual_contact_rows=counts[1]))
 out.mkdir(parents=True);np.savez_compressed(out/'boundary_series.npz',frames=frames,native_pts=stamps,x=x,observed_y=observed,predicted_y=predicted,source_x=source_x,source_row_indices=source_row_indices)
 subprocess.run([str(plot_python or sys.executable),"-c",PLOT_SCRIPT,str(out/"boundary_series.npz"),str(out/"actual_vs_prediction.svg"),file_sha256(model_file)],check=True)
 def summary(lo,hi):
  obs=observed[lo:hi];pre=predicted[lo:hi];finite=np.isfinite(obs)&np.isfinite(pre);dif=np.abs(obs-pre);exc=[]
  for j in range(2):
   for k in range(len(x)):
    q=obs[:,j,k];q=q[np.isfinite(q)]
    if len(q)>1:exc.append(float(np.ptp(q)))
  return dict(source_and_model_comparable_samples=int(finite.sum()),missing_source_or_projection_samples=int(finite.size-finite.sum()),maximum_sampled_vertical_difference_px=float(dif[finite].max())if finite.any()else None,maximum_observed_temporal_excursion_px=max(exc,default=None))
 tau_bound=float(cg.graph['rigs'][cg.graph['shots']['s01']['rig_id']].get('tau_bound',.02));support=list(map(float,camera.domain));full_readout_support=bool(support[0]<=-tau_bound/2 and support[1]>=tau_bound/2)
 report=dict(schema='rinkfit.joint9.native_temporal_boundary_audit.v1',model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),packet=dict(path=str(Path(packet_file).resolve()),sha256=file_sha256(packet_file)),frames=rows,source_track_ids=x.tolist(),source_x_varies_and_is_used_for_each_prediction=True,first_native_frame_local_seconds=rows[0]['local_seconds'],first_frame_RS_required_time_interval=[-tau_bound/2,tau_bound/2],allocated_cubic_domain=support,first_frame_entire_readout_inside_allocated_domain=full_readout_support,source_padding_exposures_created=False,camera_time_clamped=False,first_complete_control_values=camera.controls[:4].tolist(),opening=summary(0,10),pulse=summary(10,16),interpolation='Raw unmodified source UV tracked by nearest x within 40px of the window first-source anchor. Only predicted finite curve is interpolated at each actual source x; no source interpolation/extrapolation.',source_RGB_newly_consumed=False,full_active_source_error_replay_replaced=False,temporal_conflict_resolved=False,accuracy_certified=False,files={name:dict(path=str(out/name),sha256=file_sha256(out/name))for name in ('actual_vs_prediction.svg','boundary_series.npz')});write_json(out/'report.json',report);return report
