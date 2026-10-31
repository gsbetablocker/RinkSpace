"""One pending Ad4 JUNIOR source-glyph probe from the independent S09 bank.

Native glyph geometry locates a source-reference search crop only. Actual white
source components form a navigation atlas; unchanged PrintPattern registration,
separated-current-glyph checks and actual RGB transition profiles supply UV.
"""
from pathlib import Path
import argparse,hashlib,json,sys
from collections import Counter
import cv2,numpy as np
import rinkspace.session.print_patterns as pp
from rinkspace.vision_print_glyphs import extract_glyphs
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from rinkspace.core.native import NativeCore
from capture_independent_badge import exact_frames,sha,write
from capture_independent_materials import source_exclusion,spaced_indices,plain
from capture_independent_prints import source_patch_check

PTS=2179328


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path'])
    if sha(video)!=index['source']['actual_sha256']:raise ValueError('Actual original source differs')
    bank=root/'print_references/references.json';meta=next(m for m in json.loads(bank.read_text('utf8'))['references']if m['ad_id']=='Ad4'and m['shot_id']=='s09');z=dict(np.load(meta['descriptor_npz'],allow_pickle=False));ref_path=root/'s09/originals'/f's09_pts{meta["pts"]}.png';ref=cv2.imread(str(ref_path))
    if hashlib.sha256(ref.tobytes()).hexdigest()!=meta['source_BGR_sha256']:raise ValueError('Independent Ad4 reference RGB differs')
    z['source_image']=ref;z['reference_mask']=cv2.imread(meta['descriptor_mask'],0)>0
    image=exact_frames(video,[PTS])[PTS];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,'s09',PTS);rex,rmp=source_exclusion(root,'s09',meta['pts'])
    matched=match_features((z['keypoints'],z['descriptors']),descriptors(image,(~ex).astype('uint8')*255),ReuseConfig(min_matches=6,ransac_px=3.))
    if matched is None:raise ValueError('No unchanged public RootSIFT initialization; preserve source gap')
    initial_H=matched[2];banner_check=source_patch_check(image,ex,meta,z,initial_H)
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts'];library=json.loads((root.parent/'prepared_joint_ice/joint_manifest.json').read_text('utf8'))['native_library'];core=NativeCore(library);guides=[];native_map={}
    for c in cuts:
        if not c['id'].startswith('print_instance/Ad4::Ad4.print.shot4_JUNIOR_'):continue
        projection=core.project_motion(world['rig'][0],np.tile(meta['navigation_pose'],(6,1)),-2.,1.,1080,0.,world['xyz'][c['begin']:c['end']])
        if np.any(projection['status'])or not np.isfinite(projection['uv']).all():raise ValueError('Known native glyph reference navigation unavailable')
        stem=c['id'].split('::',1)[1].removesuffix('/path0');native_map['print_instance/Ad4::'+stem]=c['id'];guides.append(projection['uv'])
    uv=np.vstack(guides);lo=np.maximum(np.floor(uv.min(0)-20).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)+20).astype(int),[1920,1080]);roi=[*lo.tolist(),*hi.tolist()]
    values=ref[lo[1]:hi[1],lo[0]:hi[0]].min(2);available=~rex[lo[1]:hi[1],lo[0]:hi[0]];threshold,_=cv2.threshold(values[available].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    extracted=extract_glyphs(ref,roi,list('JUNIOR'),threshold=threshold)
    paths=[]
    for key,glyph in extracted['paths'].items():
        name='Ad4.print.shot4_JUNIOR_'+key
        if 'print_instance/Ad4::'+name not in native_map:raise ValueError('Extracted semantic glyph has no exact current native path')
        paths.append(dict(path_id=name,pixels=glyph['template_source_uv'],source_native_contour=glyph['source_native_uv'],native_path_id=native_map['print_instance/Ad4::'+name]))
    atlas=dict(paths=paths,source_image=str(ref_path),source_BGR_sha256=meta['source_BGR_sha256'],known_word='JUNIOR',
        component_order_basis='Known native JUNIOR labels and source left-to-right order; assistant inference pending primary review',
        source_component_contours_used_as_navigation_only=True,contour_vertices_are_not_observations=True,extraction=extracted['extraction'])
    pattern=pp.PrintPattern(atlas,ref);reference_resolved,reference_structure=pattern.resolved_glyphs(ref,rex.astype('uint8'),np.eye(3))
    registration=pattern.register(image,ex.astype('uint8'),initial_H)
    captured={};original=pp.read_source_material_transition_profiles
    def telemetry(source_rgb,**kwargs):
        caller=sys._getframe(1);captured['query_path_labels']=list(caller.f_locals.get('labels',[]));kwargs['keep_profiles']=True
        result=original(source_rgb,**kwargs);captured['public_readings']=result['readings'];captured['search_uv']=np.array(kwargs['search_points_uv']);captured['normals']=np.array(kwargs['normals']);captured['intervals']=np.array(kwargs['search_intervals_px']);return result
    pp.read_source_material_transition_profiles=telemetry
    try:unthinned,readout=pattern.read_paths(image,ex.astype('uint8'),registration,dict(source_sha256=index['source']['actual_sha256'],source_pts=PTS,time_base='1/12800',shot_id='s09',decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),spacing=8.)
    finally:pp.read_source_material_transition_profiles=original
    out.mkdir(parents=True);source_image=out/f's09_pts{PTS}_original.png';cv2.imwrite(str(source_image),image);cv2.imwrite(str(out/'applied_SAM_exclusion.png'),ex.astype('uint8')*255)
    for row in unthinned:
        row['atlas_path_id']=row['path_id'];row['path_id']=native_map[row['path_id']];row.update(fit_eligible=False,review_state='pending_primary_source_glyph_and_instance_review',measurement_operator='actual_RGB_white_ink_interface_q10_q50_q90',model_UV_synthesized=False,source_quantile_semantics='q10/q50/q90, not q25/q50/q75',native_or_H_vertices_used_as_observations=False)
        reading=captured['public_readings'][row['candidate']];np.testing.assert_array_equal(row['uv'],reading['actual_source_uv']);np.testing.assert_array_equal(row['actual_normal_quantiles'],reading['source_q10_q50_q90_normal_offsets_px'])
    rows=[]
    for path in sorted({r['path_id']for r in unthinned}):
        group=[r for r in unthinned if r['path_id']==path];rows.extend(group[i]for i in spaced_indices([r['uv']for r in group],spacing=48.))
    write(out/'reference_atlas.json',atlas);write(out/'reference_structure.json',dict(resolved=sorted(reference_resolved),structure=reference_structure));write(out/'word_registration.json',registration);write(out/'all_source_transition_profiles.json',captured)
    record=dict(pts=PTS,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_image=str(source_image),mask_path=str(mp),mask_sha256=sha(mp),rows=rows,all_unthinned_public_reader_rows=unthinned,fit_eligible=False,review_state='pending_primary_source_glyph_and_instance_review',source_readout=readout,source_profiles=str(out/'all_source_transition_profiles.json'),reference_atlas=str(out/'reference_atlas.json'),reference_atlas_semantics_pending_review=True,physical_instance_scope='Ad4 at this one PTS as declared by primary task; no propagation to other shots or frames')
    (out/'source_rows.jsonl').write_text(json.dumps(plain(record),ensure_ascii=False)+'\n',encoding='utf8')
    proof=image.copy()
    if registration['status']=='source_pattern_registered':
        for path in paths:
            q=pp.project(np.array(registration['H']),path['pixels']);cv2.polylines(proof,[np.rint(q).astype('int32')],True,(0,150,200),1,cv2.LINE_AA)
    for row in unthinned:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),1,(0,140,0),-1)
    for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(proof,f'Ad4 JUNIOR | S09 PTS {PTS} | {len(rows)} actual RGB q50 PENDING | NO FIT',(20,35),0,.65,(0,220,255),2,cv2.LINE_AA)
    proof_path=out/f'{PTS}_actual_white_ink_pending.jpg';cv2.imwrite(str(proof_path),proof)
    if rows:
        q=np.array([r['uv']for r in rows]);lo=np.maximum(np.floor(q.min(0)-25).astype(int),[0,0]);hi=np.minimum(np.ceil(q.max(0)+26).astype(int),[1920,1080]);cv2.imwrite(str(out/f'{PTS}_actual_white_ink_crop.png'),proof[lo[1]:hi[1],lo[0]:hi[0]])
    files=[Path(__file__),Path(pp.__file__),Path(extract_glyphs.__code__.co_filename)]
    report=dict(pts=PTS,native_source_time_seconds=PTS/12800.,known_native_glyph_paths=sorted(native_map.values()),pending_observed_path_ids=sorted({r['path_id']for r in rows}),pending_rows=len(rows),unthinned_actual_RGB_rows=len(unthinned),actual_queries=readout.get('queries',0),public_readout=readout,RootSIFT_matches=len(matched[0]),banner_source_pattern_check=banner_check,word_source_pattern_registration=registration,reference_source=str(ref_path),reference_source_sha256=sha(ref_path),reference_glyph_search_roi=roi,reference_material_threshold_navigation_only=float(threshold),source_rows=str(out/'source_rows.jsonl'),source_point_proof=str(proof_path),source_crop_proof=str(out/f'{PTS}_actual_white_ink_crop.png'),all_profiles=str(out/'all_source_transition_profiles.json'),glyph_atlas_component_names='known native word order, assistant source-component inference pending review',public_qualification_thresholds_changed=False,no_fit_package_created=True,fit_executed=False,other_PTS_processed=False,implementation_sha256={str(f.resolve()):sha(f)for f in files})
    write(out/'report.json',report);print(json.dumps({k:report[k]for k in['pts','pending_rows','unthinned_actual_RGB_rows','actual_queries','pending_observed_path_ids','source_point_proof','source_crop_proof']}),flush=True)


if __name__=='__main__':main()