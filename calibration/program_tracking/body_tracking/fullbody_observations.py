"""Reversible, source-PTS observation associations; no detector inference."""
from pathlib import Path
import json
import numpy as np
from .rebuild_prepare import load, digest

ISSUES = ((320512, (10,), 'right_wrist'), (332288, (9,), 'left_wrist'),
          (521728, (13, 14, 15, 16), 'knee_ankle_identity'))


def source_diagnostics(root, project, output):
    """Decode exact source PTS, preserving context and original point labels."""
    import av
    import cv2
    root, project, output = Path(root), Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((project/'events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json').read_text(encoding='utf8'))['primary_source']
    old = load(root/'input.npz')
    whole = load(root/'next_round_20261029/wholebody/wholebody_coco23.npz')
    np.testing.assert_array_equal(old['pts'], whole['pts'])
    records = []
    with av.open(source['path']) as container:
        stream = container.streams.video[0]
        assert stream.time_base.numerator == 1 and stream.time_base.denominator == 12800
        for pts, joints, name in ISSUES:
            targets = [pts-256, pts, pts+256]
            container.seek(targets[0], stream=stream, backward=True)
            panels = []
            for frame in container.decode(stream):
                if frame.pts not in targets:
                    if frame.pts > targets[-1]: break
                    continue
                fi = int(np.searchsorted(old['pts'], frame.pts))
                raw = frame.to_ndarray(format='bgr24')
                box = old['source_bbox'][fi]
                x0, y0 = np.maximum(np.floor(box[:2]-40), 0).astype(int)
                x1, y1 = np.minimum(np.ceil(box[2:]+40), [1920,1080]).astype(int)
                for kind, color in [('raw', None), ('COCO', (30,210,255)), ('RTMW', (255,200,20))]:
                    image = raw.copy()
                    if color is not None:
                        points = old['source_uv'][fi] if kind == 'COCO' else whole['raw_wholebody133'][fi,:,:2]
                        for j in joints:
                            xy = tuple(np.rint(points[j]).astype(int))
                            cv2.circle(image, xy, 7, color, 2, cv2.LINE_AA)
                            cv2.putText(image, str(j), (xy[0]+9, xy[1]), cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
                    crop = image[y0:y1,x0:x1]
                    scale = min(420/crop.shape[1], 650/crop.shape[0])
                    crop = cv2.resize(crop,None,fx=scale,fy=scale)
                    panel = np.full((690,440,3),25,np.uint8)
                    panel[35:35+crop.shape[0],:crop.shape[1]]=crop
                    cv2.putText(panel,f'{frame.pts/12800:.2f}s {kind} PTS {frame.pts}',(7,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
                    panels.append(panel)
                records.append(dict(pts=int(frame.pts),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],
                                    source_file=source['path'],source_sha256=source['sha256']))
            if len(panels)!=9:raise ValueError(f'Missing exact source frames for {pts}')
            cv2.imwrite(str(output/(name+'.jpg')),np.vstack([np.hstack(panels[i:i+3]) for i in range(0,9,3)]))
    (output/'source_frames.json').write_text(json.dumps(records,indent=2),encoding='utf8')


def build_ledger(root, output, decisions):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    old_path=root/'input.npz';whole_path=root/'next_round_20261029/wholebody/wholebody_coco23.npz'
    old,whole=load(old_path),load(whole_path)
    np.testing.assert_array_equal(old['pts'],whole['pts'])
    n=len(old['pts']);uv=old['source_uv'].copy();use=old['source_qualified'].copy()
    source=np.zeros((n,17),np.int8);semantic=np.broadcast_to(np.arange(17),(n,17)).copy()
    # Engineering pixel scales; never normalize disparate raw detector scores.
    sigma=np.broadcast_to(np.array([6]*5+[5,5,7,7,8,8,6,6,6,6,7,7],float),(n,17)).copy()
    changes=[];associations=[]
    for pts,joints,name in ISSUES:
        decision=decisions.get(str(pts),{})
        for j in joints:
            k=({13:14,14:13,15:16,16:15}[j] if pts==521728 else j)
            sid=0 if pts==521728 else 1
            associations.append(dict(pts=pts,joint=j,index=k,source_id=sid,name=name,decision=decision))
    for decision in decisions.get('additional_associations',[]):
        operation=decision['operation']
        if operation not in ('use_rtmw','use_original_index','exclude'):
            raise ValueError(f'Unknown observation operation: {operation}')
        associations.append(dict(pts=int(decision['pts']),joint=int(decision['joint']),
            index=int(decision.get('index',decision['joint'])),source_id=int(operation=='use_rtmw'),
            name=decision['hypothesis_id'],decision=decision,exclude=operation=='exclude'))
    seen=set()
    for association in associations:
            pts,j=association['pts'],association['joint'];decision=association['decision']
            if (pts,j) in seen:raise ValueError(f'Duplicate observation association: {(pts,j)}')
            seen.add((pts,j))
            fi=int(np.searchsorted(old['pts'],pts))
            if fi>=n or int(old['pts'][fi])!=pts or not 0<=j<17:raise ValueError('Association must name an exact source PTS and COCO joint')
            k,sid,name=association['index'],association['source_id'],association['name']
            if not 0<=k<(17 if sid==0 else 133):raise ValueError('Invalid alternative source index')
            accepted=decision.get('status')=='assistant_source_reviewed' and not association.get('exclude',False)
            alternative=old['source_uv'][fi,k] if sid==0 else whole['raw_wholebody133'][fi,k,:2]
            if accepted:
                uv[fi,j]=alternative;source[fi,j]=sid;semantic[fi,j]=k
            else:use[fi,j]=False
            changes.append(dict(shot_id='s01',pts=pts,time_base=[1,12800],joint=j,
                hypothesis_id=name,original_source_id='COCO17',original_semantic_index=j,
                original_uv=old['source_uv'][fi,j].tolist(),original_raw_score=float(old['source_score'][fi,j]),
                alternative_source_id=('COCO17','RTMW133')[sid],alternative_semantic_index=k,
                alternative_uv=alternative.tolist(),alternative_raw_score=float(old['source_score'][fi,k] if sid==0 else whole['raw_wholebody133'][fi,k,2]),
                adoption='selected_local_overlay' if accepted else 'excluded_pending_review',
                physical_left_right_status='local_semantic_association_reviewed' if accepted else 'unconfirmed',
                visibility=decision.get('visibility','reviewed_with_blur; see evidence' if accepted else 'unconfirmed'),uncertainty=dict(sigma_px=float(sigma[fi,j]),calibrated=False),
                evidence=decision,reason='source image and adjacent identity; no selection by model residual'))
    use &= np.isfinite(uv).all(2)
    disagreement=np.linalg.norm(old['source_uv']-whole['raw_wholebody133'][:,:17,:2],axis=2)
    packet=dict(pts=old['pts'],time_base=np.array([1,12800]),uv=uv,use=use,sigma_px=sigma,
                source_id=source,semantic_index=semantic,detector_disagreement_px=disagreement,
                raw_coco_uv=old['source_uv'],raw_coco_score=old['source_score'],raw_rtmw133=whole['raw_wholebody133'],
                source_crop_affine=old['source_crop_affine'],lower_body_inferred=old['lower_body_inferred'])
    np.savez_compressed(output/'observations.npz',**packet)
    ledger=dict(schema='rinkspace.observation-overlay.v1',shot_id='s01',time_base=[1,12800],
        coordinate_space='native full image 1920x1080; source crop affine preserved, not applied twice',
        sources=[dict(id='COCO17',path=str(old_path),sha256=digest(old_path),raw_field='source_uv'),
                 dict(id='RTMW133',path=str(whole_path),sha256=digest(whole_path),raw_field='raw_wholebody133')],
        array_index_contract='Each [frame,joint] maps through pts, source_id, semantic_index; untouched rows use original COCO17 identity. Raw scores are source-specific, not probabilities.',
        reversibility='Select original source/index to undo; original files unchanged.',
        visibility_default='unreviewed; lower_body_inferred and existing source_qualified exclusions retained, not relabelled occluded',
        uncertainty='sigma_px is an engineering scale, no calibrated probability claim',
        correlated_sources='Exactly one selected source per joint; no duplicate weighting.',
        changes=changes,changed_points=sum(c['adoption']=='selected_local_overlay' for c in changes),
        excluded_points=sum(c['adoption']=='excluded_pending_review' for c in changes),
        source_image_review='assistant only; not user acceptance',
        equipment='Existing source-only footwear factors retain their original review/admission and unknown physical blade side. RTMW feet and local blade A/B tracks remain diagnostics, not anatomical heel/toe or confirmed physical identity.',
        rejected_proposals=[dict(pts=568320,source='blade_local_tracks',slot='B',state='old_proximity_gate_rejected',physical_identity='unconfirmed',used=False,reason='preserved conflict; not certified wrong or invisible')],
        evaluation_roles='All five requested windows are development. No independent final validation set exists.',
        observation_array_sha256=digest(output/'observations.npz'))
    (output/'ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf8')
    return ledger
