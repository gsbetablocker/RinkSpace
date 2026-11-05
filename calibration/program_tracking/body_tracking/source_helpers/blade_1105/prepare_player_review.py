from pathlib import Path
import sys,os,json,csv,copy,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');O=D/'runs/body_s01_20261029/blade_orientation_1105';sys.path[:0]=[str(P),str(P/'src')];os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_template import profile_height
from rinkspace.viewer3d.blade_review import BladeReview,overlay_primitives

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-16),-1,1))))
audit=read(O.parent/'blade_observability_1105/blade_observability_audit.json');assembly=audit['factor_assembly'];doc=read(O/'source_boot_axis_document.json');geo=read(O/'source_axis_geometry_review.json');old=read(O/'current_projection_details.json');wide=old['wide_rows'].copy();allpts=np.arange(972800,1216001,256,dtype=np.int64)
defs={}
for label,vid in [('A','continuous_source'),('B','continuous_flight'),('C6','continuous_refined'),('local_281','local_281')]:
 c=assembly['candidates'][label];defs[vid]=dict(prefix=label,stage=c['stage_root'],parameter=c['input_files'][-1]['path'],result=c['result']['path'])
for folder,vid,prefix in [('entry_81_2_82_0','boot_axis_entry','boot_entry'),('support_87_6_90_0','boot_axis_support','boot_support')]:
 s=read(O/folder/'run_summary.json');defs[vid]=dict(prefix=prefix,stage=str(O/folder),parameter=s['parameter'],result=s['result'])
arrays={'pts':allpts};metrics=[];lookups={}
for vid,d in defs.items():
 m=SavedBladeProjection(P,d['stage'],d['parameter']);packet=m.binding.packet();p=packet['parameters'];xx=np.array([-p.runner_rear_extension_m,.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),p.sole_length_m-p.runner_front_inset_m]);q=np.r_[np.c_[xx,np.zeros(3),profile_height(xx,p)],packet['adapter'][[0,7]].mean(0)[None],packet['adapter'][[1,8]].mean(0)[None]]
 uv=np.stack([m.project_points(allpts,foot,q)['uv'] for foot in [0,1]],1);arrays[d['prefix']+'_blade_uv']=uv[:,:,:3];arrays[d['prefix']+'_boot_uv']=uv[:,:,3:]
 with np.load(d['result'],allow_pickle=False) as z:b=z['blade_material_world_mid'];localclear=z['skate_clearance_m_mid'];qworld=z['Q_world_mid']
 axis=b[:,:,2]-b[:,:,0];axis/=np.linalg.norm(axis,axis=-1,keepdims=True);dot=np.sum(axis[1:]*axis[:-1],axis=-1);rate=np.full((951,2),np.nan);rate[1:]=np.arccos(np.clip(dot,-1,1))/.02;arrays[d['prefix']+'_axis_rate_rad_s']=rate
 for g in geo['items']:
  regions=g.get('boot_contour_regions') or {}
  if not regions.get('heel') or not regions.get('toe'):continue
  i=int((g['pts']-972800)//256);f=g['foot_proposal'];source=np.asarray(regions['toe']['center_native_xy_px'])-regions['heel']['center_native_xy_px'];error=angle(uv[i,f,4]-uv[i,f,3],source)
  metrics.append(dict(variant=vid,pts=g['pts'],foot=f,boot_axis_error_deg=error,conditional_blade_axis_error_deg=None,source_role=g['role'],source_uncertainty='finite boot contours; not exact blade material'))
 # Keep the old source-region-centre blade direction estimates explicitly conditional.
 parentlabel=d['prefix'] if d['prefix'] in ['A','B','C6','local_281'] else 'local_281'
 for oldrow in old['wide_rows']:
  if oldrow['candidate']!=parentlabel:continue
  record=copy.deepcopy(oldrow);pt=record['pts'];f=record['foot'];i=(pt-972800)//256
  if vid in ['boot_axis_entry','boot_axis_support']:
   record['candidate']=vid
   for j,name in enumerate(['rear','middle','front']):record['blade_'+name+'_u']=float(uv[i,f,j,0]);record['blade_'+name+'_v']=float(uv[i,f,j,1])
   record['boot_heel_u']=float(uv[i,f,3,0]);record['boot_heel_v']=float(uv[i,f,3,1]);record['boot_toe_u']=float(uv[i,f,4,0]);record['boot_toe_v']=float(uv[i,f,4,1])
   # Old conditional axes outside the repaired windows are unchanged. Recompute whenever present.
   sr=[r for r in doc['observations'] if r['pts']==pt and r['foot']==f and r.get('kind')=='blade_endpoint_region' and r.get('optimization_enabled',True)]
   ep={r['blade_endpoint']:r for r in sr}
   for endpoint,j in [('rear',0),('front',2)]:
    if endpoint in ep:
     r=ep[endpoint];off=uv[i,f,j]-r['centre_uv'];rad=np.sqrt(off@np.linalg.solve(r['region_shape_px2'],off))
     record[endpoint+'_region_centre_px']=float(np.linalg.norm(off));record[endpoint+'_region_outside_px']=float(np.linalg.norm(off)*max(0,1-1/max(rad,1e-30)));record[endpoint+'_region_dx']=float(off[0]);record[endpoint+'_region_dy']=float(off[1])
   if record.get('boot_contour_axis_error_deg') is not None:
    br=[r for r in doc['observations'] if r['pts']==pt and r['foot']==f and r.get('kind')=='boot_marker_region']
    bh=next((r for r in br if r.get('feature')=='heel'),None);bt=next((r for r in br if r.get('feature')=='toe'),None)
    if bh and bt:record['boot_contour_axis_error_deg']=angle(uv[i,f,4]-uv[i,f,3],np.asarray(bt['centre_uv'])-bh['centre_uv'])
   if set(ep)=={'front','rear'} and record['blade_axis_visibility']=='visible':
    s=np.asarray(ep['front']['centre_uv'])-ep['rear']['centre_uv'];record['blade_axis_error_deg']=angle(uv[i,f,2]-uv[i,f,0],s);record['conditional_endpoint_regions_axis_error_deg']=record['blade_axis_error_deg']
    for name,j in [('rear',0),('front',2)]:
     r=ep[name];off=uv[i,f,j]-r['centre_uv'];rad=np.sqrt(off@np.linalg.solve(r['region_shape_px2'],off));record[name+'_region_centre_px']=float(np.linalg.norm(off));record[name+'_region_outside_px']=float(np.linalg.norm(off)*max(0,1-1/max(rad,1e-30)))
   gm=next((g for g in geo['items'] if g['pts']==pt and g['foot_proposal']==f),None)
   if gm and gm.get('visible_steel_line_segment') and gm['axis_direction_usable']:
    sg=gm['visible_steel_line_segment'];record['source_segment_to_model_whole_axis_deg']=angle(uv[i,f,2]-uv[i,f,0],np.asarray(sg['point_b_native_xy_px'])-sg['point_a_native_xy_px'])
   if gm and gm.get('boot_contour_regions',{}).get('heel'):
    bc=gm['boot_contour_regions'];record['new_source_boot_contour_axis_error_deg']=angle(uv[i,f,4]-uv[i,f,3],np.asarray(bc['toe']['center_native_xy_px'])-bc['heel']['center_native_xy_px'])
   wide.append(record)
  if record.get('blade_axis_error_deg') is not None:
   existing=next((r for r in metrics if r['variant']==vid and r['pts']==pt and r['foot']==f),None)
   if existing:existing['conditional_blade_axis_error_deg']=record['blade_axis_error_deg']
   else:metrics.append(dict(variant=vid,pts=pt,foot=f,boot_axis_error_deg=record.get('boot_contour_axis_error_deg'),conditional_blade_axis_error_deg=record['blade_axis_error_deg'],source_uncertainty='conditional finite endpoint-region centres; not certified material'))
 lookups[vid]=uv
 print(json.dumps(dict(event='player_projection',variant=vid,frames=951,max_3D_axis_rate_rad_s=float(np.nanmax(rate)),lower_foot_81_54_81_68=np.argmin(localclear[277:285],axis=1).tolist())),flush=True)
np.savez_compressed(O/'player_blade_projection.npz',**arrays)
visibility=[]
for e in audit['reviewed_source_exposures']:
 for foot in e['feet']:
  v=foot['visual_review'];visibility.append(dict(pts=e['pts'],foot=foot['foot'],blade_axis_observable=v.get('blade_axis_observable')=='visible',orientation_unknown_reason=v.get('orientation_unknown_reason'),anatomical_identity=v.get('anatomical_identity')))
# New gap reads have explicitly unassigned objects: keep both model feet unknown.
for r in read(O/'source_gap_review/manifest.json')['frames']:
 for f in [0,1]:
  visibility=[v for v in visibility if not(v['pts']==r['pts'] and v['foot']==f)];visibility.append(dict(pts=r['pts'],foot=f,blade_axis_observable=False,orientation_unknown_reason='New source-only gap review could not bind a reliable foot identity and full axis; no new direction factor.',anatomical_identity='unknown'))
asset=dict(schema='rinkspace.blade-player-review.v1',time_base=[1,12800],projection_npz='player_blade_projection.npz',variants={vid:d['prefix'] for vid,d in defs.items()},source_boot_axes=doc['boot_axis_observations'],source_regions=[r for r in doc['observations'] if r['visibility']=='visible' and (r.get('optimization_enabled',True) or r['role']=='review') and r.get('image_footprint')],source_visibility=visibility,source_steel_segments=[dict(pts=g['pts'],foot=g['foot_proposal'],points=[g['visible_steel_line_segment']['point_a_native_xy_px'],g['visible_steel_line_segment']['point_b_native_xy_px']],uncertainty='Visible subsegment chord only; not a whole-blade axis or fixed material endpoints. '+g['axis_usable_scope']) for g in geo['items'] if g.get('visible_steel_line_segment') and g['axis_direction_usable']],axis_metrics=metrics,interpretation='All model projections use saved current body pose, fixed shoes and native five-node rolling camera; all source observations exactPTS. NaN rate at first frame remains unknown. No source geometry is synthesized from model.',source_foot_identity='conditional episode; not independent anatomical truth')
(O/'player_blade_review.json').write_text(json.dumps(asset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
fields=sorted(set(k for r in wide for k in r));fields=['candidate','pts','seconds','foot']+[k for k in fields if k not in ['candidate','pts','seconds','foot']]
with (O/'blade_axis_projection_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(wide)
# Clone the existing player; retain every old variant and the old default.
cfgpath=P/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_spin_281_repaired_1105.json';cfg=read(cfgpath)
for vid,intervals in [('boot_axis_entry',[[81.2,82.0]]),('boot_axis_support',[[81.2,82.0],[87.6,90.0]])]:
 cfg['variants'].append(dict(id=vid,label=('靴轮廓／轴线局部修正（8步，待审核）' if vid=='boot_axis_entry' else '再修87.6–90秒（另8步，足鞋结构退化待审）'),kind='fullbody',result=defs[vid]['result'],shell_topology='saved',body_reference_label='exmr_hand_nofoot',required_saved_diagnostics=['com_world_mid','blade_material_world_mid','skate_witness_world_mid'],body_refinement_intervals_seconds=intervals))
cfg.update(blade_review_asset=str(O/'player_blade_review.json'),comparison_variants=['local_281','boot_axis_entry'],default_variant='local_281',initial_source_pts=1044480,support_observations=str(O/'source_boot_axis_document.json'),review_status='Two bounded boot-axis candidates, no user/physical acceptance; original default preserved',session_id='blade_boot_1105')
newcfg=P/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_boot_axis_1105.json';assert len(str(newcfg).encode('utf-16-le'))//2<260
if newcfg.exists():assert read(newcfg)==cfg
else:newcfg.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=BladeReview(O/'player_blade_review.json',time_base=[1,12800])
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
for pt,sourcepath,box in [(1044480,O.parent/'spin_local_280_1105/source_review/1044480.png',(960,710,1290,950)),(1044736,O.parent/'spin_local_280_1105/source_review/1044736.png',(960,710,1290,950)),(1121792,Path(next(g['source_frame'] for g in geo['items'] if g['pts']==1121792)),(980,690,1300,910))]:
 source=Image.open(sourcepath).convert('RGB');sheet=Image.new('RGB',(2640,520),'white')
 for i,vid in enumerate(['source','local_281','boot_axis_entry','boot_axis_support']):
  im=source.copy()
  if vid!='source':
   draw=ImageDraw.Draw(im)
   for primitive in overlay_primitives(review.at(vid,pt)):
    color=tuple(round(255*c) for c in primitive['color'][:3]);points=[tuple(x) for x in primitive['points']]
    if primitive['kind']=='line' and len(points)>1:draw.line(points,fill=color,width=2)
    elif primitive['kind']=='point':
     x,y=points[0];draw.ellipse((x-4,y-4,x+4,y+4),fill=color,outline='black')
   # Add sparse same-frame body outline from the actual exported native projection.
   with np.load(defs[vid]['result'],allow_pickle=False) as z:coco=z['COCO_uv'][(pt-972800)//256]
   for a,b in [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:draw.line([tuple(coco[a]),tuple(coco[b])],fill=(220,110,0),width=2)
  cropped=im.crop(box).resize((660,480));sheet.paste(cropped,(660*i,40));sd=ImageDraw.Draw(sheet);sd.text((660*i+8,4),vid+'  '+str(pt/12800)+'s',font=font,fill='black')
 path=O/f'boot_native_compare_{pt}.png'
 finalsheet=Image.new('RGB',(2640,555),'white');finalsheet.paste(sheet,(0,0));ImageDraw.Draw(finalsheet).text((8,526),'Green: source boot contours / visible steel SUBSEGMENT. Orange: model body / boot axis. Blue / yellow / white: template blade rear / middle / front.',font=font,fill='black');finalsheet.save(path)
print(json.dumps(dict(player_config=str(newcfg),player_asset=str(O/'player_blade_review.json'),projection_comparison_rows=len(wide),overlays=3,old_default_preserved=cfg['default_variant']),ensure_ascii=False))
