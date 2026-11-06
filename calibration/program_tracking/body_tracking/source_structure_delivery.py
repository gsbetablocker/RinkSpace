"""Source structures, real-fit bindings and per-exposure review assets."""
from pathlib import Path
import argparse,copy,hashlib,html,json
from .source_skate_structure_review import render_frame,render_ice_map


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def build(structures,contacts,integration,upstream,parameter,result,output,diagnostics=None):
    structures,contacts,integration,upstream,parameter,result,output=map(Path,(structures,contacts,integration,upstream,parameter,result,output))
    for p in [parameter,result]:
        if not p.is_file():raise FileNotFoundError('Real fitted candidate required: '+str(p))
    output.mkdir(parents=True,exist_ok=True);source=read(structures);mapped=read(contacts);fit=read(integration);asset=read(upstream)
    asset['projection_npz']=str((upstream.parent/asset['projection_npz']).resolve())
    all_frames=source['frames']+source.get('unassigned_identity_observations',[])
    all_frames.sort(key=lambda r:(r['pts'],str(r.get('source_track_id')),str(r.get('foot'))))
    asset['source_structures']=all_frames;asset['source_structure_document']=dict(path=str(structures),sha256=sha(structures),source_only=True)
    asset['source_plane_locations']=[];asset['source_ice_grid']=[]
    ml={(r['pts'],r['foot']):r for r in mapped['frames']};admitted={(r['pts'],r['foot'])for r in fit['source_support_evidence']}
    for key,row in ml.items():
        if key not in admitted:continue
        feature=row['features']['ice_interface'];m=feature['auxiliary_representative']['mapping']
        asset['source_plane_locations'].append(dict(pts=row['pts'],foot=row['foot'],xy_conditional_m=m['xyz_conditional_m'][0][:2],source_status=feature['source_status'],pixel_radius_max_offset_m=m['sampled_xy_max_offset_conditional_m'][0],actual_contact_confirmed=False,source_pixel_representative=m['points_uv'][0]))
        if 'auxiliary_grid'in row:asset['source_ice_grid'].append(dict(pts=row['pts'],foot=row['foot'],**row['auxiliary_grid']))
    asset['source_structure_fit_variants']={'sequence_source_structure':dict(description_zh='此候选实际使用本轮9类源结构及有证据的条件支撑XY；未匹配的斜撑、开孔和靴上部保留作源图显示，未作为材料点强配。',
        feature_rows=[{k:r[k]for k in ['pts','foot','structure_feature']}for r in fit['full_source_skate_structures']['rows']],support_rows=fit['source_support_evidence'],
        unassigned_source_sets=[dict(pts=g['pts'],feature_rows_count=len(g['branches'][0]['rows']),source_anatomical_identity='unassigned',role='latent_one_to_one_source_set_fit')for g in fit.get('unassigned_source_skate_sets',[])],
        parameter=str(parameter),parameter_sha256=sha(parameter),result=str(result),result_sha256=sha(result),integration_sha256=sha(integration))}
    if fit.get('complete_runner_extent',{}).get('rows'):
        asset['source_structure_fit_variants']['sequence_source_structure']['description_zh']+=' 完整可见的钢刃下缘还使用双向覆盖残差，已与正向项平分原预算。'
    if diagnostics is not None:
        measured=read(diagnostics)['candidates']['current']
        asset['source_structure_fit_variants']['sequence_source_structure']['support_diagnostics']=measured['support_rows']
        asset['source_structure_fit_variants']['sequence_source_structure']['diagnostics_provenance']=dict(path=str(diagnostics),sha256=sha(diagnostics))
    asset['source_plane_document']=dict(path=str(contacts),sha256=sha(contacts),no_body_pose_used=True)
    asset_path=output/'player_source_review.json';write(asset_path,asset)
    pages=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S05源鞋刀观测</title><style>body{font:17px system-ui;background:#20252d;color:#eef0f3;margin:24px}a{color:#8bceff}section{padding:15px;margin:20px 0;background:#303741}img{width:100%;height:auto}summary{cursor:pointer}nav{line-height:2;position:sticky;top:0;background:#20252d}</style><h1>S05源鞋刀观测与实际拟合绑定</h1><p>原片 / 源结构描线 / 源图线稿。实线直接描线；虚线保留局部分界、模糊或推断；遮挡处不补材料身份。这里显示源图读数，不是模型线。</p><p>同帧9类可匹配结构进入唯一人体鞋刀状态；支撑XY按现有相机和冰面回算，与2D源像素共享预算。转出遮挡后的未分左右轮廓以可见集合参与，仍不声称已确认解剖侧别。</p><nav>']
    records=[]
    for i,row in enumerate(all_frames):pages.append(f'<a href="#s{i}">{row["pts"]/12800.:.2f}s/{html.escape(str(row.get("foot")))}</a> · ')
    pages.append('</nav>')
    for i,row in enumerate(all_frames):
        crop=Path(row['source_crop_path']);from PIL import Image
        with Image.open(crop)as image:size=image.size
        entry=dict(pts=row['pts'],source_seconds=row['pts']/12800.,native_crop=str(crop),native_crop_sha256=row['source_crop_sha256'],native_bounds_xyxy=row['native_bounds_xyxy'])
        picture=output/f'source_{row["pts"]}_{i}.png';render_frame(entry,row,picture)
        location=ml.get((row['pts'],row.get('foot')));map_path=None
        if location is not None and(row['pts'],row.get('foot'))in admitted:
            map_path=output/f'ice_{row["pts"]}_{i}.png';render_ice_map(entry,row,location,map_path)
        pages.append(f'<section id="s{i}"><h2>{row["pts"]/12800.:.2f}s · PTS {row["pts"]} · 源foot={html.escape(str(row.get("foot")))}</h2><a href="{picture.name}"><img loading="lazy" src="{picture.name}"></a>')
        if map_path:pages.append(f'<details><summary>源图冰面回投</summary><a href="{map_path.name}"><img loading="lazy" src="{map_path.name}"></a></details>')
        pages.append('</section>');records.append(dict(pts=row['pts'],foot=row.get('foot'),source_track_id=row.get('source_track_id'),source_image=str(picture),ice_image=str(map_path)if map_path else None))
    pages.append('</html>');(output/'index.html').write_text(''.join(pages),encoding='utf-8')
    report=dict(source_exposures=len(set(r['pts']for r in all_frames)),source_skate_contours=len(all_frames),source_associated_foot_exposures=len(source['frames']),unassigned_source_contours=len(source.get('unassigned_identity_observations',[])),support_location_rows=len(admitted),fit_feature_rows=len(fit['full_source_skate_structures']['rows']),player_asset=str(asset_path),gallery=str(output/'index.html'),records=records,real_parameter_sha256=sha(parameter),real_result_sha256=sha(result),gui_opened=False,accuracy_accepted=False)
    write(output/'review.json',report);return report


def main():
    p=argparse.ArgumentParser()
    for key in ['structures','contacts','integration','upstream','parameter','result','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--diagnostics',type=Path)
    a=p.parse_args();r=build(**vars(a));print(json.dumps({k:v for k,v in r.items()if k!='records'}))
if __name__=='__main__':main()
