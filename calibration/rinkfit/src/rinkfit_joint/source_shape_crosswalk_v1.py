"""Consume an explicitly reviewed print-shape identity and replay native pixels.

The semantic identity is an agent visual-review fact, not inferred from a local
component number. Pixel, mask, source-row and material-path checks are performed
here. This issues geometry candidates only; it supplies no noise likelihood.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import cv2
from .io_utils import file_sha256, write_json, write_jsonl
from .r_inner_prefix_v1 import array
from rinkspace.vision_source_shapes import read_source_material_transition_profiles


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(record):
    p=Path(record['file'])
    if file_sha256(p)!=record['sha256']:
        raise ValueError('Changed source-shape dependency: '+str(p))
    return p


def validate_rows(rows, expected):
    if len(rows)!=len(expected):raise ValueError('Reviewed shape source coverage differs')
    fields=('source_row_id','reuses_original_source_row_id','native_pts','shot_id',
            'source_uv','current_uv','joint_material_index','target_id','kind')
    for a,b in zip(rows,expected):
        if any(a.get(k)!=b.get(k)for k in fields):
            raise ValueError('Reviewed shape source, UV or physical material differs')
        if a.get('point_correspondence')is not False or a.get('entity_station_fixed')is not False:
            raise ValueError('A shape crosswalk does not establish a fixed world point')


def qualify(packet,model,witness,output):
    pp=Path(packet).resolve();mp=Path(model).resolve();wp=Path(witness).resolve();out=Path(output)
    if out.exists():raise ValueError('New immutable shape qualification required')
    p=read(pp);m=read(mp);w=read(wp)
    if w['schema']!='rinkfit.source_visual_semantic_shape_crosswalk.v1' or w.get('human_review_claimed')is not False:
        raise ValueError('Explicit agent source-semantic review required')
    for flag in ('component_number_alone_used','camera_or_model_projection_used_for_identity',
                 'world_point_or_same_station_correspondence_claimed','source_UV_moved'):
        if w.get(flag)is not False:raise ValueError('Unsupported source identity method')
    source=read(checked(w['source']));s=source['source'];pts=int(source['source_pts'])
    if pts!=w['source']['native_pts']or s['time_base']!=w['source']['time_base']or s['source_video_sha256']!=w['source']['video_sha256']:
        raise ValueError('Reviewed source clock differs')
    bgr=cv2.imread(s['image_path']);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if file_sha256(s['image_path'])!=s['png_sha256']or hashlib.sha256(rgb.tobytes()).hexdigest()!=s['decoded_RGB_sha256']or s['decoded_RGB_sha256']!=w['source']['RGB_sha256']:
        raise ValueError('Reviewed source pixels differ')
    if file_sha256(s['mask_path'])!=s['mask_sha256']or s['mask_sha256']!=w['source']['mask_sha256']:
        raise ValueError('Reviewed source exclusion differs')
    mask=cv2.imread(s['mask_path'],0)
    outer=source['features'][w['source_feature_index']];hole=source['features'][w['source_hole_feature_index']]
    if outer['id']!=w['source_feature_id']or hole['id']!=w['source_hole_feature_id']or outer.get('physical_surface')!='board' or hole.get('physical_surface')!='board':
        raise ValueError('Reviewed print entity differs')
    if len(outer['segments'])!=1 or outer['segments'][0].get('closed')is not True or len(hole['segments'])!=1 or hole['segments'][0].get('closed')is not True:
        raise ValueError('This reviewed annular logo requires complete outer and inner topology')
    uv=np.asarray(outer['segments'][0]['uv'],float);inside=np.asarray(hole['segments'][0]['uv'],float)
    if uv.tolist()!=w['source_outer_UV']or inside.tolist()!=w['source_hole_UV']:
        raise ValueError('Original reviewed shape UV changed')
    if not all(cv2.pointPolygonTest(uv.astype(np.float32),tuple(map(float,q)),False)>0 for q in inside):
        raise ValueError('Reviewed logo hole is not inside its actual outer boundary')
    tp=checked(w['template']);template=read(tp);ad=template['advertisements'][str(w['template']['advertisement_number'])]
    old=ad['contours'][w['template']['contour_key']]
    if old['role']!='printed_ink_external_boundary'or not old['closed']or w['template']['hole_contour_key']not in ad['contours']:
        raise ValueError('Registered canonical printed boundary role differs')
    if file_sha256(w['template']['source_file'])!=w['template']['source_sha256']:
        raise ValueError('Canonical historical print source changed')
    mi=int(w['canonical_material_index']);material=m['materials'][mi];paths=material['paths']
    if material['label']!=w['canonical_material_label']or len(paths)!=1:
        raise ValueError('Canonical finite material differs')
    path=paths[0]
    if path['id']!=w['canonical_path_id']or path.get('physical_point')is not False or path['advertisement_block']!=w['canonical_parent_advertisement_block']or path['geometry_block']!=w['canonical_geometry_block']:
        raise ValueError('Canonical continuous-board shape path differs')
    if m['blocks'][path['advertisement_block']]['name']!='ad/'+w['physical_panel_id'][2:]:
        raise ValueError('Physical advertisement installation differs')
    profile=read(checked(w['source_position_review']));checked(profile['source_frame']);checked(profile['source_RGB']);checked(profile['actual_mask']);checked(profile['reader'])
    records=read(checked(profile['source_readings']))
    if len(records)!=len(uv)or any(r['original_UV']!=q.tolist()for r,q in zip(records,uv)):
        raise ValueError('Source-profile original row order differs')
    normals=np.asarray([r['current_source_profile']['source_search_normal']for r in records],float)
    binding=dict(source_sha256=s['source_video_sha256'],source_pts=pts,shot_id=w['source']['shot_id'],time_base=s['time_base'],native_hw=[1080,1920],decoded_native_rgb_sha256=s['decoded_RGB_sha256'])
    actual=read_source_material_transition_profiles(rgb,binding=binding,search_points_uv=uv,normals=normals,search_intervals_px=np.tile([-24.,24.],(len(uv),1)),transition='dark_to_light',keep_profiles=True)
    seen=set();A={k:array(p,k,pp.parent,seen)for k in ('uv','native_pts','admission','fit_role')};ids={}
    for i,rid in enumerate(p['source_row_ids']):ids.setdefault(rid,[]).append(i)
    qualified=[];unknown=[];details=[]
    for k,(record,reading,q,n)in enumerate(zip(records,actual['readings'],uv,normals)):
        matches=[i for i in ids.get(record['source_row_id'],[])if int(A['native_pts'][i])==pts]
        if len(matches)!=1:raise ValueError('Original source identity is not unique')
        row=matches[0]
        if not np.array_equal(A['uv'][row],q)or (A['admission'][row]==1 and A['fit_role'][row]==1):
            raise ValueError('Original UV changed or the restored source would duplicate an active row')
        for field in ('source_q10_q50_q90_normal_offsets_px','actual_source_luminance_profile','actual_source_signed_luminance_gradient'):
            if not np.array_equal(np.asarray(reading[field]),np.asarray(record['current_source_profile'][field])):
                raise ValueError('Source profile does not replay exactly')
        tangent=np.array([-n[1],n[0]]);samples=q+np.arange(-26,27)[None,:,None]*n+np.arange(-2,3)[:,None,None]*tangent
        z=np.rint(samples).astype(int);valid=(z[:,:,0]>=0)&(z[:,:,0]<1920)&(z[:,:,1]>=0)&(z[:,:,1]<1080)
        clear=bool(valid.all()and not(mask[np.clip(z[:,:,1],0,1079),np.clip(z[:,:,0],0,1919)]>0).any())
        q10,q50,q90=reading['source_q10_q50_q90_normal_offsets_px'];supported=bool(q10-2<=0<=q90+2)
        ok=reading['disposition']=='measured_material_transition'and clear and supported
        value=dict(source_row_id='source_shape_crosswalk_v1/'+w['source']['shot_id']+'/'+str(pts)+'/Ad3.logo_outer/'+str(k),reuses_original_source_row_id=record['source_row_id'],original_raw_row=row,source_family_id='source_shape_crosswalk_v1:'+w['source']['shot_id']+':'+str(pts)+':'+w['physical_panel_id']+'.logo_outer',shot_id=w['source']['shot_id'],native_pts=pts,source_uv=q.tolist(),current_uv=q.tolist(),normal=n.tolist(),joint_material_id=w['source_feature_id']+'|board|ink_boundary|unspecified',joint_material_index=mi,target_id=material['label'],kind='finite_material_curve',point_correspondence=False,entity_station_fixed=False,source_geometry_eligible=ok,statistical_objective_eligible=False,numeric_statistics_ready=False,solver_eligible=False,solver_eligible_for_statistical_objective=False,receipt_sha256=file_sha256(wp))
        (qualified if ok else unknown).append(value);details.append(dict(original_raw_row=row,original_UV=q.tolist(),actual_current_q50_candidate=reading['actual_source_transition_candidate_uv'].tolist(),mask_halo_clear=clear,source_position_qualified=ok,original_UV_preserved=True))
    out.mkdir(parents=True);write_jsonl(out/'qualified_rows.jsonl',qualified);write_jsonl(out/'unknown_rows.jsonl',unknown);write_json(out/'actual_consumer_rows.json',details)
    report=dict(schema='rinkfit.actual_source_reviewed_shape_consumer.v1',packet=dict(file=str(pp),sha256=file_sha256(pp)),model=dict(file=str(mp),sha256=file_sha256(mp)),witness=dict(file=str(wp),sha256=file_sha256(wp)),rows=len(uv),qualified_rows=len(qualified),unknown_rows=len(unknown),actual_native_profiles_replayed=True,original_UV_preserved=True,old_raw_history_and_other_pending_rows_unchanged=True,semantic_identity_is_explicit_agent_visual_review=True,automatic_semantic_generalization_claimed=False,world_point_or_station_correspondence_claimed=False,source_noise_W_or_sigma_issued=False,quality_accepted=False,producer=dict(file=str(Path(__file__).resolve()),sha256=file_sha256(__file__)))
    write_json(out/'report.json',report);return report
