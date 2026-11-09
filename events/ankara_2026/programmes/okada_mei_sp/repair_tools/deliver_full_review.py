"""Install a separate whole-programme review, reusing the source player and exporter."""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.programme_projection import read, render_comparison
from calibration.program_tracking.body_tracking.programme_skate_replay import write
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.storage import AssetStore
from rinkspace.viewer3d.blade_review import BladeReview, overlay_primitives
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.source import SourceVideo
from scripts.deliver_programme_skate_review import (
    PROGRAMME, RATE, binding, copy_checked, export, failures, metrics_rows,
    projection, register_json, replay_fit, review_summary, source_shapes, temporal_review,
)

PARENT='programme_skate_evidence_1109'
CANDIDATE='programme_full_review_1109'


def main():
    parser=ArgumentParser()
    for key in ('project','prepared','replay','output','visual-review'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    programme=a.project/PROGRAMME
    parent_record=read(programme/(PARENT+'.json'))
    parent_config=programme/('viewer3d_'+PARENT+'.json')
    initial_config_sha=digest(parent_config)
    cfg=read(parent_config)
    parent_result=Path(parent_record['result']['path'])
    parent_param=Path(parent_record['parameter']['path'])
    receipt=read(a.replay/'acceptance.json')
    parameter=a.replay/'candidate.npz'
    source=Path(receipt['source_evidence'])
    evidence=read(source)['programme_skate_evidence']
    summary=read(a.visual_review)
    if (digest(parent_param)!=receipt['parent_sha256'] or digest(parameter)!=receipt['parameter_sha256']
            or digest(source)!=receipt['source_evidence_sha256'] or receipt['coverage']['native_exposures']!=8329
            or summary['source_video_sha256']!=evidence['source_video_sha256']):
        raise ValueError('Candidate/parent/visual source binding changed')
    a.output.mkdir(exist_ok=True)
    saved_export=a.output/'final/export_receipt.json'
    if saved_export.is_file():
        exported=read(saved_export)
        if (exported['parameter']['sha256']!=digest(parameter)
                or exported['result']['sha256']!=digest(exported['result']['path'])):
            raise ValueError('Existing export differs from requested result')
    else:
        exported=export(a.project,a.prepared,source,parameter,parent_param,parent_result,a.output/'final')
    result=Path(exported['result']['path'])
    support=np.asarray(exported['affected_intervals_pts'])
    if np.any((support[:,0]<=2123776)&(support[:,1]>=2001408)):
        raise ValueError('Complete LSp interval received changed spline support')
    comparison=read(a.replay/'source_comparison.json')
    new_projection=a.output/'projection/programme_blade_review.json'
    if new_projection.is_file():
        bound=read(new_projection)['prediction_binding']
        if bound['parameter_sha256']!=digest(parameter):raise ValueError('Wrong projection state')
    else:
        new_projection=projection(a.project,a.prepared,parameter,result,a.output/'projection',CANDIDATE,
                                  replay_fit(evidence,receipt,comparison),source,source)
    new_doc=read(new_projection)
    old_report=read(Path(cfg['blade_review_asset']))
    report=deepcopy(old_report)
    report['projection_layers'].append(dict(path=str(new_projection.parent/new_doc['projection_npz']),variants=[CANDIDATE]))
    report['variants'].update(new_doc['variants'])
    report['source_structure_fit_variants'].update(new_doc['source_structure_fit_variants'])
    report['prediction_bindings'][CANDIDATE]=new_doc['prediction_binding']
    amend_ids={name for amendment in evidence['amendments'] for name in amendment['event_ids']}
    new_source={key:[r for r in evidence[key] if r['event_id'] in amend_ids]
        for key in ('ordered_contours','manual_landmarks','native_support_bands')}
    report['source_structures'].extend(source_shapes(new_source))
    report['source_overlay_revision_variant']=CANDIDATE
    report['source_evidence_binding']['programme_skate_ledger']=binding(source)
    for label,tag in [('parent',PARENT),('candidate',CANDIDATE)]:
        contours=[{k:r[k] for k in ('id','pts','foot','semantic','role','model_uv')}
            for _,family,r in metrics_rows(comparison,label)
            if family=='ordered_native_skate' and 'model_uv' in r and (label=='candidate' or r['event_id'] in amend_ids)]
        if label=='parent':
            report['reviewed_model_contours'].setdefault(tag,[]).extend(contours)
        else:
            report['reviewed_model_contours'][tag]=contours
    asset=a.output/'review.json'
    register_json(asset,report)
    temporal=temporal_review(result,parent_result,parent_result,evidence)
    register_json(a.output/'time_identity.json',temporal)
    register_json(a.output/'source_by_part.json',review_summary(evidence,comparison))
    base_nav=read(Path(cfg['source_navigation']))
    base_nav['events']=[e for e in base_nav['events'] if e.get('kind') not in
        ('source_residual','model_surface','joint_domain','source_identity_review')]
    surface=read(Path(receipt['surface_comparison']))
    nav=failures(base_nav,evidence,comparison,dict(results=dict(candidate=surface['candidate'])),
        temporal,dict(final_guard=receipt['trial']['guard']))
    for row in summary['priority_anchors']:
        if row['event'] not in receipt['selected_events']:
            nav['events'].append(dict(label=row['label'],start_pts=row['start_pts'],end_pts=row['end_pts'],
                    worst_pts=row['worst_pts'],kind='chronological_source_review',reason_zh=row['note_zh']))
    nav['programme_visual_review']=binding(a.visual_review)
    nav_path=a.output/'navigation.json'
    register_json(nav_path,nav)

    store=AssetStore(a.project)
    reviewed=BladeReview(asset,assets=store,time_base=(1,RATE))
    pts=np.asarray(list(reviewed.index))
    if len(pts)!=8329 or any(v not in reviewed.models for v in (PARENT,CANDIDATE)):
        raise ValueError('Full source projection or comparison variant missing')
    stamps=set(pts.tolist())
    for row in nav['events']:
        for k in ('start_pts','end_pts','worst_pts'):
            if k in row and row[k] not in stamps:
                raise ValueError('Navigation points outside exact source PTS')
    source_video=Path(evidence['source_video'])
    if digest(source_video)!=evidence['source_video_sha256']:
        raise ValueError('Source video differs from the reviewed evidence')
    times=[53.58,66.82,97.66,110.32,151.28,179.52]
    selected=[round(t*RATE) for t in times]
    reader=SourceVideo(source_video,(1,RATE))
    try:
        for stamp in [int(pts[0]),*selected,int(pts[-1])]:
            assert reader.read(stamp).shape==(1080,1920,3)
            for variant in (PARENT,CANDIDATE):
                state=reviewed.at(variant,stamp)
                assert state['pts']==stamp and state['model'] is not None
                assert overlay_primitives(state,include_model_contours=True,include_model_body=True,include_trajectory=True)
    finally:
        reader.close()
    del reviewed
    comparison_dir=a.output/'same_frame'
    panel=comparison_dir/'same_frame_comparison.jpg'
    if not panel.is_file():
        panel=render_comparison(a.project,asset,source_video,[PARENT,CANDIDATE],selected,comparison_dir)

    if digest(parent_config)!=initial_config_sha:
        raise ValueError('Parent player configuration changed during delivery')
    final_param=programme/(CANDIDATE+'.npz')
    record_path=programme/(CANDIDATE+'.json')
    new_config=programme/('viewer3d_'+CANDIDATE+'.json')
    copy_checked(parameter,final_param)
    record=dict(schema='rinkspace.programme-footwork-review.v1',candidate_id=CANDIDATE,
        parameter=binding(final_param),result=binding(result),source=binding(source),
        review=binding(asset),navigation=binding(nav_path),source_seconds=[float(pts[0]/RATE),float(pts[-1]/RATE)],
        coverage=receipt['coverage'],acceptance=receipt['acceptance'],user_accepted=False,accuracy_accepted=False,
        previous=parent_record['parameter'],replay=binding(a.replay/'acceptance.json'),
        visual_review=binding(a.visual_review),source_comparison=binding(a.replay/'source_comparison.json'),
        same_frame=binding(panel),export=binding(saved_export),selected_events=receipt['selected_events'],
        camera_sha256=exported['camera_sha256'])
    register_json(record_path,record,project_file=True)
    parent_view=next(v for v in cfg['variants'] if v['id']==PARENT)
    view=deepcopy(parent_view)
    view.update(id=CANDIDATE,candidate_id=CANDIDATE,label='全节目复核后的步法与鞋刀修正',
        parameter=str(final_param),parameter_sha256=digest(final_param),result=str(result),result_sha256=digest(result),
        diagnostic_record=str(record_path),diagnostic_record_sha256=digest(record_path),body_reference_label=PARENT,
        body_refinement_intervals_seconds=(support/RATE).tolist())
    cfg['variants'].append(view)
    cfg.update(session_id=CANDIDATE,latest_review_variant=CANDIDATE,
        comparison_variants=[PARENT,CANDIDATE],blade_review_asset=str(asset),source_navigation=str(nav_path),
        initial_source_pts=round(97.66*RATE),source_review_model_trajectory_default=True)
    register_json(new_config,cfg,project_file=True)
    catalog=ResultCatalog(project=a.project,config=new_config)
    assert len(catalog.navigation_points('cuts'))==8
    assert len(catalog.navigation_points('events'))==len(nav['events'])
    assert catalog.config['default_variant']==read(parent_config)['default_variant']
    delivery=dict(config=binding(new_config),candidate=binding(record_path),comparison=binding(panel),
                  navigation=binding(nav_path),source_exposures=len(pts),source_video_decode_verified=True)
    register_json(a.output/'delivery.json',delivery)
    print(json.dumps(delivery,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
