"""Build the isolated Ad4 source-gap revision; never alter its parent state."""
from pathlib import Path
import argparse, copy, hashlib, json, sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from rinkfit_start.source_contour_revision import apply_source_contour_revision,map_retained_stations
from rinkfit_start.native_scene import NativeScene

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def transform(q,H):return cv2.perspectiveTransform(np.asarray(q,float).reshape(1,-1,2),np.asarray(H,float))[0]
def check_path(p):
 p=str(Path(p).resolve());parts=Path(p).parts
 if len(p.encode('utf-16-le'))//2>=260 or any(len(v.encode('utf-16-le'))//2>255 for v in parts):raise ValueError('unsafe OneDrive path: '+p)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--run',type=Path,default=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_s1_r11'))
 ap.add_argument('--revision',type=Path,default=ROOT/'events/ankara_2026/programmes/okada_mei_sp/src_rev1018/ad4_v1.json')
 args=ap.parse_args();R=args.run;O=R/'source_audit_1018';D=R/'ad4_source_v1';check_path(args.revision)
 parent=R/'physical_i20_full_capture_main/checkpoints/i00020_1792282587350451200/model.json'
 parent_hash_before=sha(parent)
 J=json.loads((O/'inputs.json').read_text(encoding='utf8'));audit=json.loads((O/'source_report.json').read_text(encoding='utf8'));A=J['ad4'];model=json.loads(parent.read_text(encoding='utf8'))
 evs={e['frame_index']:e for e in A['source_evidence']};images={};masks={};warps={};inverses={}
 atlas=cv2.imread(A['atlas']['path']);assert sha(A['atlas']['path'])==A['atlas']['sha256']
 for n in (400,440):
  e=evs[n];s=e['source'];assert sha(s['image_path'])==s['png_sha256'];assert sha(s['mask_path'])==s['mask_sha256']
  images[n]=cv2.imread(s['image_path']);masks[n]=cv2.imread(s['mask_path'],0);H=np.array(e['source_to_atlas_homography']);inverses[n]=np.linalg.inv(H);warps[n]=cv2.warpPerspective(images[n],H,(3000,180))
 source_refs={str(n):copy.deepcopy(evs[n]) for n in (400,440)}
 assert sha(J['actual_scene_path'])==audit['actual_scene_sha256']
 source_scene=json.loads(Path(J['actual_scene_path']).read_text(encoding='utf8'))
 for key in ('word_ANKARA_K_arc1','word_ANKARA_R_arc1'):
  assert source_scene['payload']['templates']['advertisements']['4']['contours'][key]==A['contours'][key]
 K=A['contours']['word_ANKARA_K_arc1'];Q=np.array(K['source_atlas_uv']);old=1-np.array(K['material_uv'])
 parts=[];template_contours={}
 for branch,(a,b,n) in enumerate(((0,12,440),(16,19,400))):
  name='word_ANKARA_K_arc1_src'+str(n);ix=list(range(a,b+1));source={'frame_index':n,'native_pts':evs[n]['source']['native_pts'],'source_seconds':evs[n]['source']['source_time_seconds'],'source_key':str(n)}
  parts.append({'new_path_index':branch,'path_suffix':'src'+str(n),'old_vertex_indices':ix,'closed':False,'source_evidence':source})
  c={k:copy.deepcopy(v)for k,v in K.items()if k not in ('material_uv','prototype_material_uv','source_atlas_uv')};c.update(id=name,closed=False,source_contour_key=K['id'],source_old_vertex_indices=ix,source_native_uv=transform(Q[ix],inverses[n]).tolist(),source_frame_index=n,source_atlas_uv=Q[ix].tolist(),material_uv=np.array(K['material_uv'])[ix].tolist(),prototype_material_uv=np.array(K['prototype_material_uv'])[ix].tolist(),genuine_contiguous_outer_boundary_only=True,physical_identity_verified_on_native_crop=True,artificial_letter_cut_edges_added=False,original_template_accepted=False,independent_metrology_accepted=False)
  template_contours[name]=c
 seam=[]
 for y in (115,116,118,120,121):
  seam.append({'atlas_y':y,'atlas_x':list(range(474,479)),'stored_atlas_BGR':atlas[y,474:479].tolist(),'source440_only_warp_BGR':warps[440][y,474:479].tolist(),'source400_only_warp_BGR':warps[400][y,474:479].tolist(),'source400_native_uv':transform([[x,y]for x in range(474,479)],inverses[400]).tolist(),'source400_native_image_valid_mask':((transform([[x,y]for x in range(474,479)],inverses[400])[:,0]>=0)&(transform([[x,y]for x in range(474,479)],inverses[400])[:,0]<=1919)).tolist()})
 revision={'schema':'rinkfit.source-contour-path-revision.v1','revision_id':'ankara_2026.okada_mei_sp.Ad4.K_arc1.source_gap.20261018.v1','status':'independent_source_template_patch_not_adopted_into_main_state','not_a_direct_human_confirmation':True,'source_scene':ref(J['actual_scene_path']),'parent_source_audit':ref(O/'source_report.json'),'parent_inputs':ref(O/'inputs.json'),'source_atlas':A['atlas'],'source_evidence':source_refs,'scope':'删除由拼图图外插值制造的缺口两侧伪边；只保留原顶点，不新建形状或源观测。','seam_evidence':{'removed_old_edges':[12,13,14,15],'old_edge_count':19,'native_finding':'edge12和edge15是拼图第476列暗缝两侧，不是原片印刷材料边界。f400反投影第476列x约-0.29，黑色border值参与双线性插值；第475列来自f440，第477列来自f400。同位置f440原片连续，无该暗缝。删除依据是源像素与图像有效域，不是拟合形状或SVD条件数。','threshold_context':'原生成器在此区要求min(BGR)>135且min/max>.83；第476列y116..121降至min115..131，而f440同位置min163..184。','pixel_witnesses':seam,'diagnostic_paths':[str(O/'word_ANKARA_K_arc1_raw_f440.png'),str(O/'word_ANKARA_K_arc1_raw_f400.png'),str(O/'K_edge_native_f440.png'),str(O/'K_edge_native_f400.png')],'foreground_masks_not_a_material_edge_or_padding_certificate':True},'template_revision':{'old_contour_key':K['id'],'replacement_contours':template_contours,'closed':False,'new_vertices_added':0,'unseen_gap_reconstructed':False,'material_uv_definition':'原模板Ad4展开面u从左到右/v从上到下；求解现有点为[1-u,1-v]。','R_arc1_action':'unchanged_true_visible_lower_open_arc_from_f400','inherited_template_accuracy_accepted':False},'template_path_patch':{'material_label':'Ad4.print.word_ANKARA_K_arc1','source_template_contour':K['id'],'old_path_index':0,'old_path_closed':False,'expected_old_points':old.tolist(),'expected_point_relation_to_template':'[1-material_u,1-material_v]','retained_paths':parts,'removed_old_edge_indices':[12,13,14,15],'old_station_intervals_retained':[[0.,12/19],[16/19,1.]],'old_station_interval_deleted_open':[12/19,16/19],'parameter_blocks_changed':False,'geometry_block_and_advertisement_block_references_preserved':True,'affine6_changed':False,'new_degrees_of_freedom':0,'materials_labels_and_indices_changed':False,'path_count_delta':1,'station_transport_formulas':['old_s in [0,12/19]: new_path=0; new_s=19*old_s/12','old_s in [16/19,1]: new_path=1; new_s=(19*old_s-16)/3','12/19<old_s<16/19: no equivalent retained station; recapture original UV against legal residual paths'],'original_source_UV_labels_W_E_unchanged':True,'deleted_interval_is_not_source_row_exclusion':True,'fresh_RS_S_path_selection_Q_and_derivatives_required':True,'old_solver_multipliers_or_linearization_resume_permitted':False},'limits':['仅证明删段和来源；未证明剩余模板的亚像素轮廓精度、真实尺寸或最终贴合。','两条剩余弧仍共享原来的一个affine6块；R_arc1及全部其他材料、安装和相机参数不变。','原片修订已用于开发，不能继续作为独立留出通过证据。','保留源、原模板、旧模型与失败；候选不覆盖主状态。']}
 write(args.revision,revision)
 candidate=apply_source_contour_revision(model,revision);mi=next(i for i,m in enumerate(model['materials'])if m['label']==revision['template_path_patch']['material_label'])
 assert candidate['blocks']==model['blocks'];assert len(candidate['materials'])==len(model['materials']);assert len(candidate['materials'][mi]['paths'])==2
 write(D/'model_candidate.json',candidate)
 material=np.fromfile(R/'backend_input/material.u32','<u4');station=np.fromfile(parent.parent/'station.f64','<f8');path=np.fromfile(parent.parent/'path.u32','<u4');pts=np.fromfile(R/'backend_input/pts.i64','<i8');uv=np.fromfile(R/'backend_input/uv.f64','<f8').reshape(-1,2);rows=np.flatnonzero(material==mi)
 branch,newstation,valid=map_retained_stations(path[rows],station[rows],revision);transport=[]
 for k,r in enumerate(rows):
  transport.append({'source_row':int(r),'native_pts':int(pts[r]),'source_seconds':float(pts[r]/12800),'original_source_uv':uv[r].tolist(),'old_path':int(path[r]),'old_station':float(station[r]),'old_edge':int(min(np.floor(station[r]*19),18)),'retained_position_transport_valid':bool(valid[k]),'new_path_seed':int(branch[k])if valid[k]else None,'new_station_seed':float(newstation[k])if valid[k]else None,'fresh_RS_S_Q_required':True})
 write(D/'i20_station_transport.json',transport)
 edge_sources=[]
 for key in ('word_ANKARA_K_arc1','word_ANKARA_R_arc1'):
  co=A['contours'][key];q=np.array(co['source_atlas_uv'])
  for edge,(a,b)in enumerate(zip(q[:-1],q[1:])):
   original=audit['Ad4'][key]['edges'][edge];n=440 if key.endswith('K_arc1') and edge<=12 else 400;keep=key.endswith('R_arc1')or edge not in (12,13,14,15);dense=a+np.linspace(0,1,65)[:,None]*(b-a);native=transform(dense,inverses[n]);inside=(native[:,0]>=0)&(native[:,0]<=1919)&(native[:,1]>=0)&(native[:,1]<=1079);z=np.rint(np.clip(native,[0,0],[1919,1079])).astype(int);fg=masks[n][z[:,1],z[:,0]]>0
   edge_sources.append({'contour_key':key,'old_edge_index':edge,'old_vertices':[edge,edge+1],'action':'retain'if keep else'remove_whole_edge','provenance_exact_RGB_source_runs':original['source_runs'],'visible_native_frame_for_retained_edge':n if keep else None,'native_pts':evs[n]['source']['native_pts'],'native_uv_start_end':transform([a,b],inverses[n]).tolist(),'dense_samples':65,'native_in_image_samples':int(inside.sum()),'foreground_mask_samples':int((fg&inside).sum()),'source_png_sha256':evs[n]['source']['png_sha256'],'source_mask_sha256':evs[n]['source']['mask_sha256'],'assessment':'unchanged visible native lower open arc'if key.endswith('R_arc1')else'retained native material arc; atlas initializer accuracy not accepted'if keep else'padding-induced stitching column edge; no source material edge; no connector added'})
 write(D/'segment_sources.json',edge_sources)
 revision['segment_sources']=edge_sources
 write(args.revision,revision)
 oldscene=NativeScene(model);newscene=NativeScene(candidate);error=[]
 for part in parts:
  a,b=part['old_vertex_indices'][0],part['old_vertex_indices'][-1];s=np.linspace(a/19,b/19,257);p,t,ok=map_retained_stations(np.zeros(len(s),int),s,revision);assert ok.all();v=oldscene.world(mi,0,s);w=newscene.world(mi,part['new_path_index'],t);error.append(float(np.max(np.abs(v-w))))
 untouched=copy.deepcopy(candidate);untouched['materials'][mi]['paths']=copy.deepcopy(model['materials'][mi]['paths']);untouched.pop('source_contour_revisions');assert untouched==model
 report={'schema':'rinkfit.ad4-source-gap-validation.v1','passed':True,'revision':ref(args.revision),'parent_model':ref(parent),'candidate_model':ref(D/'model_candidate.json'),'candidate_is_not_resumed_solver_state':True,'parent_model_bytes_unchanged':sha(parent)==parent_hash_before,'all_parameter_blocks_unchanged':True,'all_other_material_paths_unchanged':True,'material_count':len(model['materials']),'path_count_before':sum(len(m['paths'])for m in model['materials']),'path_count_after':sum(len(m['paths'])for m in candidate['materials']),'K_old_vertices':20,'K_new_path_vertices':[13,4],'removed_old_edges':[12,13,14,15],'K_source_rows':len(rows),'transportable_rows':int(valid.sum()),'rows_requiring_finite_path_recapture':[r for r in transport if not r['retained_position_transport_valid']],'retained_same_world_max_absolute_error':max(error),'physical_parameters_refit':False,'R_arc1_changed':False,'shape_degeneracy_ratios_unchanged':{k:v['singular_ratio']for k,v in audit['Ad4'].items()},'source_segments':ref(D/'segment_sources.json'),'all_rows_require_fresh_RS_S_Q_before_adoption':True,'source_repair_is_not_final_accuracy_acceptance':True}
 write(D/'validation.json',report)
 html='<!doctype html><meta charset="utf-8"><title>Ad4 源轮廓独立修订</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}img{max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Ad4 源轮廓独立修订 v1</h1><p>仅删 K_arc1 拼缝伪边12–15，保留0..12、16..19两条开放弧，不桥接。R_arc1不动；原模型、源UV与参数块未修改。该候选未接入主状态。</p><p>原片：f400 PTS358400 / 28.000s；f440 PTS368640 / 28.800s，time_base=1/12800。官方50fps单节目源；图片路径与sha256见修订JSON。</p><h2>原片局部</h2>'
 for f in ('word_ANKARA_K_arc1_raw_f440.png','word_ANKARA_K_arc1_raw_f400.png','word_ANKARA_R_arc1_raw_f400.png','K_edge_native_f440.png','K_edge_native_f400.png'):
  html+='<p>'+f+'</p><img src="'+(O/f).as_uri()+'">'
 html+='<h2>验证与限制</h2><pre>'+json.dumps(report,ensure_ascii=False,indent=2)+'</pre>'
 (D/'review.html').write_text(html,encoding='utf8')
 print(json.dumps({k:report[k]for k in ('passed','revision','candidate_model','path_count_before','path_count_after','transportable_rows','rows_requiring_finite_path_recapture','retained_same_world_max_absolute_error')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
