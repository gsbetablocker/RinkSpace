"""Prepare source-only Ad4 GRAND/PRIX outer-glyph navigation from the approved bank.

Known current material coordinates delimit broad source searches and exact native
identities. Original photographed components form navigation contours. They are
never observations; every current UV must still be read by public RGB profiles.
"""
from pathlib import Path
import argparse,hashlib,json
import cv2,numpy as np
from rinkspace.geometry import project
from rinkspace.vision_print_glyphs import extract_glyphs
from rinkspace.session.print_patterns import PrintPattern
from capture_independent_badge import sha
from capture_independent_materials import source_exclusion
from capture_independent_ad4_glyph_gap import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();cv2.setNumThreads(2)
    if out.exists():raise FileExistsError(out)
    bank_path=root/'print_references/references.json';meta=next(v for v in json.loads(bank_path.read_text('utf8'))['references']if v['ad_id']=='Ad4');source_path=root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png';image=cv2.imread(str(source_path))
    if hashlib.sha256(image.tobytes()).hexdigest()!=meta['source_BGR_sha256']:raise ValueError('Actual approved reference pixels differ')
    exclusion,mask_path=source_exclusion(root,meta['shot_id'],meta['pts']);bank=np.load(meta['descriptor_npz'],allow_pickle=False);offset=bank['edge_offsets'];uv=bank['projected_uv'];quad=np.array([uv[offset[0]+(offset[1]-offset[0])-1],uv[offset[0]],uv[offset[2]],uv[offset[3]]],np.float32);chart=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),quad)
    scene=json.loads(a.scene.read_text('utf8'));asset=next(v for v in scene['advertisement_assets']if v.get('advertisement_number')==4);asset_path=Path(asset['path']);
    if sha(asset_path)!=asset['sha256']:raise ValueError('Current native Ad4 template differs')
    model=json.loads(asset_path.read_text('utf8'));template=next(v for v in model['templates']if v['id']==asset['template_id']);native={v['id']:v for v in template['paths']};paths=[];extractions=[];attempts=[]
    for word,letters in [('GRAND',list('GRAND')),('PRIX',list('PRIX'))]:
        native_paths=[v for v in template['paths']if f'.shot4_{word}_'in v['id']and v['role']=='printed_ink_external_boundary'];guides=np.vstack([project(chart,np.asarray(v['material_uv']))for v in native_paths]);lo=guides.min(0);hi=guides.max(0);extracted=None;roi=None
        for margin in [8,12,16,20]:
            lower=np.maximum(np.floor(lo-margin).astype(int),[0,0]);upper=np.minimum(np.ceil(hi+margin).astype(int),[1920,1080]);region=[*lower.tolist(),*upper.tolist()];signal=image[lower[1]:upper[1],lower[0]:upper[0]].min(2);available=~exclusion[lower[1]:upper[1],lower[0]:upper[0]]
            if not available.any():attempts.append(dict(word=word,roi=region,status='no_unmasked_reference_material'));continue
            threshold,_=cv2.threshold(signal[available].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            try:extracted=extract_glyphs(image,region,letters,threshold=threshold)
            except ValueError as error:attempts.append(dict(word=word,roi=region,status='source_components_unresolved',reason=str(error),actual_reference_Otsu=float(threshold)));continue
            roi=region;attempts.append(dict(word=word,roi=region,status='actual_reference_word_components_separated',actual_reference_Otsu=float(threshold)));break
        if extracted is None:raise ValueError('Actual original word not separated: '+word+'; '+str(attempts))
        extractions.append(dict(word=word,roi=roi,extraction=extracted['extraction']))
        for key,glyph in extracted['paths'].items():
            if key.endswith('_inner')or key=='N':continue
            stem=f'Ad4.print.shot4_{word}_{key}';native_id=stem+'/path0'
            if native_id not in native or native[native_id]['role']!='printed_ink_external_boundary':raise ValueError('No exact current native exterior: '+native_id)
            current=template['paths'][next(i for i,v in enumerate(template['paths'])if v['id']==native_id)];source=np.asarray(glyph['template_source_uv']);xy=np.rint(source).astype(int)
            if np.any(xy<0)or np.any(xy>[1919,1079])or exclusion[xy[:,1],xy[:,0]].any():raise ValueError('Reference glyph intersects current source exclusion: '+stem)
            paths.append(dict(path_id=stem,pixels=glyph['template_source_uv'],source_native_contour=glyph['source_native_uv'],native_path_id='print_instance/Ad4::'+native_id,native_role=current['role'],native_material_uv=current['material_uv'],fit_eligible=True,admission_pending=False,known_word=word))
    atlas=dict(paths=paths,source_image=str(source_path),source_BGR_sha256=meta['source_BGR_sha256'],source_video_sha256=meta['source_video_sha256'],known_word='GRAND PRIX',component_order_basis='Current primary-session independently identified Ad4 GRAND PRIX content, exact current normalized native template labels and photographed source-component word order',source_component_contours_used_as_navigation_only=True,contour_vertices_are_not_observations=True,native_projection_called=False,reference_panel_projection_metadata_used_only_to_delimit_search=True,source_extraction=extractions,source_search_attempts=attempts,native_template=dict(path=str(asset_path),sha256=asset['sha256'],template_id=template['id']),GRAND_N_not_admitted='Current native N has ambiguous __part0/__part1 outer identities; no new N path or UV assignment',all_glyph_holes_not_admitted=True,public_current_readout_thresholds_changed=False)
    pattern=PrintPattern(atlas,image);resolved,structure=pattern.resolved_glyphs(image,exclusion.astype('uint8'),np.eye(3))
    if len(resolved)!=len(paths):raise ValueError('Reference outer components did not pass unchanged public separation')
    out.mkdir(parents=True);write(out/'reference_atlas.json',atlas);write(out/'reference_structure.json',dict(resolved=sorted(resolved),structure=structure));proof=image.copy()
    for p in paths:cv2.polylines(proof,[np.rint(p['pixels']).astype('int32')],True,(0,0,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out/'reference_source_navigation_overlay.png'),proof);write(out/'binding.json',dict(reference_atlas_sha256=sha(out/'reference_atlas.json'),source_image=str(source_path),source_BGR_sha256=meta['source_BGR_sha256'],source_mask_path=str(mask_path),source_mask_sha256=sha(mask_path),source_bank=str(bank_path),source_bank_sha256=sha(bank_path),native_source_material_semantics_only=True,no_world_camera_state_or_old_observation_file_read=True,no_observations_created=True,script_sha256=sha(__file__)));print(json.dumps(dict(atlas=str(out/'reference_atlas.json'),paths=len(paths),native_paths=[p['native_path_id']for p in paths],attempts=attempts)),flush=True)


if __name__=='__main__':main()
