"""Deliver saved whole-S05 source-guided/fusion candidates in the existing player.

Read-only model replay, native projection and headless mesh preparation only.
The user's old configuration and default launcher are never replaced.
"""
from pathlib import Path
import argparse,copy,json,os
import numpy as np
from . import sequence_delivery as delivery
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .exmr_integration import PROGRAMME

NAV=(76.,78.4,80.4,81.02,81.04,81.62,87.64,91.2,92.8,94.4,94.8,95.)


def deliver(project,root,selection,*,prepare=True,frames=True,revision=None):
    from rinkspace.storage import AssetStore
    from rinkspace.path_policy import guard_project_write
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import FullSurface,prepare_variant
    project,root=map(lambda x:Path(x).resolve(),(project,root));store=AssetStore(project);sequence=root.parent/'body_sequence_1105'
    if revision is not None and (not revision or not all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in revision)):
        raise ValueError('Revision must be a non-empty lowercase filename component')
    suffix='_'+revision if revision else ''
    selected=json.loads(Path(selection).read_text(encoding='utf-8'));out=root/('delivery'+suffix);out.mkdir(exist_ok=True)
    config_name='viewer3d_source_fusion'+suffix+'_1106.json'
    launcher_name='START_3D_SOURCE_FUSION'+suffix.upper()+'.cmd'
    before_path=project/PROGRAMME/'viewer3d_sequence_source_structure_1106.json';before=json.loads(before_path.read_text(encoding='utf-8'));old_sha=digest(before_path)
    delivery.PARENT='sequence_source_structure';delivery.CURRENT=selected['final_variant'];delivery.KEY_SECONDS=NAV
    original_stage=sequence/'full_source_extent_fit_1106';original_parameter=project/PROGRAMME/'spin_sequence_source_structure_1106.npz'
    definitions={delivery.PARENT:dict(stage=original_stage,parameter=original_parameter,label='审核基线 9c00203f')}
    definitions.update({r['variant']:dict(stage=Path(r['stage']),parameter=Path(r['parameter']),label=r['label'])for r in selected['candidates']})
    pts=np.arange(972800,1216001,256,dtype=np.int64)
    for vid,d in definitions.items():
        d['raw_result']=d['stage']/'common/exmr_hand_nofoot/fitted/result.npz';d['result']=d['raw_result'];d['summary']=json.loads((d['stage']/'run_summary.json').read_text(encoding='utf-8'));d['saved']=load(d['raw_result'])
        if Path(d['summary']['parameter']).resolve()!=d['parameter'].resolve():raise ValueError('Candidate parameter/summary mismatch: '+vid)
        np.testing.assert_array_equal(d['saved']['pts'],pts);np.testing.assert_array_equal(d['saved']['source_pts'],pts)
        np.testing.assert_array_equal(d['saved']['time_base'],[1,12800])
    loc=dict(project=project,store=store,sequence=sequence,out=out,family='source_structure',asset_variants=list(definitions),stage=definitions[delivery.CURRENT]['stage'],source_asset=Path(before['blade_review_asset']))
    asset,projection=delivery.blade_assets(loc,pts,definitions,before)
    review=json.loads(asset.read_text(encoding='utf-8'));fit_variants=review.setdefault('source_structure_fit_variants',{})
    for r in selected['candidates']:
        d=definitions[r['variant']];source=json.loads((d['stage']/'source_evidence.json').read_text(encoding='utf-8'));metrics=json.loads((d['stage']/'final_metrics.json').read_text(encoding='utf-8'))
        fit_variants[r['variant']]=dict(description_zh=r['label']+'；同一951曝光人体—附着鞋刀状态。源可见性与完整范围独立记录，物理准确性未验收。',
            feature_rows=[{k:v for k,v in row.items()if k not in ('source_annotation','polyline_uv','source_range','source_frame_metadata')}for row in source['full_source_skate_structures']['rows']],
            support_rows=source['source_support_evidence'],support_diagnostics=metrics['source_support_locations'],
            unassigned_source_sets=[dict(pts=g['pts'],feature_rows_count=len(g['branches'][0]['rows']))for g in source['unassigned_source_skate_sets']],
            source_contour_contract_provenance=source['source_contour_contract_provenance'],source_episode_provenance=source['source_episode_provenance'])
        fit_variants[r['variant']].update(support_location_policy=source.get('support_location_policy'),root_prior_policy=source.get('root_prior_policy'),
            native_source_support_contacts=metrics.get('native_source_support_contacts'),complete_visible_structure_extents=metrics.get('complete_visible_structure_extents'),source_pixel_edge_profiles=metrics.get('source_pixel_edge_profiles'),
            source_pixel_edge_revision=source.get('source_pixel_edge_revision'),support_expectation_policy=source.get('support_expectation_policy'),body_proxy_policy=source.get('body_proxy_policy'))
    write_json(asset,review)
    config=copy.deepcopy(before)
    for r in selected['candidates']:
        vid=r['variant'];d=definitions[vid]
        if any(v['id']==vid for v in config['variants']):raise ValueError('Preserve pre-existing variant '+vid)
        config['variants'].append(dict(id=vid,label=r['label'],kind='fullbody',result=str(d['result']),shell_topology='saved',body_reference_label='exmr_hand_nofoot',required_saved_diagnostics=delivery.REQUIRED,body_refinement_intervals_seconds=[[76.,95.]]))
    final_source=json.loads((definitions[delivery.CURRENT]['stage']/'source_evidence.json').read_text(encoding='utf-8'))
    episode=final_source['source_episode_provenance'];episode_path=Path(episode['path'])
    if digest(episode_path)!=episode['sha256']:raise ValueError('Source episode ledger changed')
    config.update(session_id='source_fusion'+suffix+'_1106',comparison_variants=['sequence_source_structure',selected['source_guided_variant'],selected['final_variant']],blade_review_asset=str(asset),source_structure_review_asset=str(asset),initial_source_pts=1121792,
        initial_view='side',review_status=selected.get('review_status_zh','整段源引导与证据/物理融合候选；所有失败与unknown保留，待用户视觉验收'),source_episode_ledger=str(episode_path))
    # The copied config retains the user's old default. The new launcher opts
    # explicitly into the final candidate; existing launchers stay unchanged.
    assert config['default_variant']==before['default_variant']
    formal=guard_project_write(project/PROGRAMME/config_name,project_root=project)
    if formal.exists():raise FileExistsError(formal)
    write_json(formal,config);loc.update(formal=formal,staged=out/formal.name,clean_source_column=True,comparison_titles=[('clean original video',None),('9c00203f baseline',delivery.PARENT),('source-guided evidence + physics',delivery.CURRENT)])
    launcher=guard_project_write(project/'launchers'/launcher_name,project_root=project)
    if launcher.exists():raise FileExistsError(launcher)
    config_arg='%~dp0..\\events\\ankara_2026\\programmes\\okada_mei_sp\\'+config_name
    launcher.write_bytes(('@echo off\r\nif "%~1"=="" (\r\n  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "'+config_arg+'" -Variant '+selected['final_variant']+' -Fresh\r\n) else (\r\n  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "'+config_arg+'" %*\r\n)\r\nexit /b %ERRORLEVEL%\r\n').encode('utf-8'))
    caches={};catalog=ResultCatalog(project=project,data_root=store.data,config=formal)
    if prepare:
        surface=None
        for row in selected['candidates']:
            vid=row['variant']
            if surface is None:surface=FullSurface(catalog)
            path=prepare_variant(catalog,vid,surface);state=load(path/'state.npz');np.testing.assert_array_equal(state['pts'],pts)
            caches[vid]=dict(path=str(path),metadata=json.loads((path/'metadata.json').read_text(encoding='utf-8')),source_frames=len(state['pts']))
    comparisons=[]
    if frames:comparisons=delivery.static_frames(loc,prepared=(pts,definitions,before))
    if digest(before_path)!=old_sha:raise ValueError('Original player configuration changed')
    report=dict(config=str(formal),launcher=str(launcher),variants={vid:dict(stage=str(d['stage']),parameter=str(d['parameter']),parameter_sha256=digest(d['parameter']),source_result=str(d['raw_result']),source_result_sha256=digest(d['raw_result']),display_result=str(d['result']),projection_verification=d.get('projection_verification'))for vid,d in definitions.items()},
        exact_source_pts=pts.tolist(),source_frames=951,whole_seconds=[76.,95.],native_projection=projection,display_caches=caches,same_frame_count=len(comparisons),original_config_sha256=old_sha,original_default_preserved=True,no_gui_opened=True,no_video_generated=True,accuracy_accepted=False)
    write_json(out/'delivery.json',report);print(json.dumps(dict(event='source_fusion_delivery',config=str(formal),launcher=str(launcher),same_frames=len(comparisons))),flush=True);return report


def main():
    ap=argparse.ArgumentParser()
    for name in ('project','root','selection'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--skip-caches',action='store_true');ap.add_argument('--skip-frames',action='store_true');ap.add_argument('--revision');a=ap.parse_args()
    deliver(a.project,a.root,a.selection,prepare=not a.skip_caches,frames=not a.skip_frames,revision=a.revision)

if __name__=='__main__':main()
