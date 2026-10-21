"""Current-packet native readers and exact per-subject review dependencies.

The packet is read-only. Broad material readings are candidates; source class,
physical instance, operator, visibility and position remain separate outcomes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .io_utils import read_json, read_jsonl, file_sha256, object_sha256, write_json
from .feedback_checkpoint import PROJECT

CONFIG = dict(version='current_packet_native_readers_v1', navigation='unchanged source UV and original source operator only',
              model_navigation=False, current_packet_scope=True, source_UV_changed_by_review=False)
CODE = ('source_recovery_review.py', 'native_source.py', 'material_profile_candidates.py',
        'board_source_requalification.py', 'entity_corners.py', 'physical_source_operator.py', 'source_worker.py','feedback_material_guard.py')
FACTS = ('ice_contact_material_roles_user_confirmed_20261006.json', 'user_board_edge_semantics_20261010.json')


def reference(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=file_sha256(p))


def checked(ref):
    p=Path(ref.get('path',ref.get('file'))).resolve()
    if file_sha256(p)!=ref['sha256']:raise ValueError('Changed native review dependency: '+str(p))
    return p


def reader_dependencies():
    root=Path(__file__).parent
    code=[reference(root/n) for n in CODE]
    code += [reference(PROJECT/'src/rinkspace'/n) for n in ('vision_source_topology.py','vision_source_shapes.py','vision_programme_ice_review.py','vision_programme_board_review.py','vision_source_fit.py')]
    code += [reference(PROJECT/'src'/n) for n in ('rinkspace/feedback_runtime.py','rinkspace/feedback_outbox.py','rink_feedback/core.py','rink_feedback/cycle.py','rink_feedback/adapters.py')]
    facts=[reference(PROJECT/'events/ankara_2026/model_constraints'/n) for n in FACTS]
    return dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(CONFIG),
                effective_facts_sha256=object_sha256(facts)), code, facts


def effective_case_refs(source, material_label, shot_id):
    if not material_label.startswith('board.'):return []
    from rinkspace.feedback_runtime import SourceFeedbackRead
    binding=dict(source_sha256=source['video_sha256'],source_pts=source['pts'],time_base=source['time_base'],
                 native_hw=[1080,1920],shot_id=shot_id)
    feedback=SourceFeedbackRead(binding,project=PROJECT,entity_ids=['board.contact','board.bottom_strip','board.top'],
        tags=['ice_contact','board_foot','material_boundary'])
    return sorted([{k:m[k]for k in ('case_id','version','content_sha256')}for m in feedback.applicable],key=lambda r:r['case_id'])


def observation_content(packet, arrays, index):
    keys=('native_pts','uv','profile_normal','kind','E','material_index','shot_index')
    return dict(source_row_id=packet['source_row_ids'][index],source_dataset_id=packet['source_dataset_ids'][index],
                values={k:np.asarray(arrays[k][index]).tolist() for k in keys})


def cache_path(source, dataset, row_id):
    from rinkspace.storage import AssetStore
    key=object_sha256(dict(source=source,dataset=dataset,row_id=row_id))
    return AssetStore(PROJECT).data/'events/ankara_2026/feedback/review_cache'/(key+'.json')


def cached_dependencies(source, packet, arrays, index):
    p=cache_path(source,packet['source_dataset_ids'][index],packet['source_row_ids'][index])
    if not p.is_file():return None
    cached=read_json(p)
    if not all(k in cached for k in ('review_dependencies','material_label','shot_id')):return None
    deps=cached['review_dependencies'];current,_,facts=reader_dependencies()
    currentcases=effective_case_refs(source,cached['material_label'],cached['shot_id'])
    current['effective_facts_sha256']=object_sha256(dict(refs=facts+cached.get('extra_fact_refs',[]),case_versions=currentcases))
    if any(deps[k]!=v for k,v in current.items()):return None
    if deps['observation_snapshot_sha256']!=object_sha256(observation_content(packet,arrays,index)):return None
    try:
        for ref in cached['evidence_refs']:checked(ref)
    except (OSError, ValueError):return None
    return deps


class CurrentPacketReview:
    def __init__(self, packet_path, config_path, *, requested_row_ids=None):
        self.packet_path=Path(packet_path).resolve();self.packet=read_json(self.packet_path)
        self.config_path=Path(config_path).resolve();self.config=read_json(config_path)
        if checked(self.config['packet'])!=self.packet_path:raise ValueError('Review config belongs to another packet')
        self.arrays={}
        for k in ('native_pts','uv','profile_normal','kind','E','material_index','shot_index','admission','fit_role'):
            desc=self.packet['arrays'][k];path=Path(desc['file']);path=path if path.is_absolute() else self.packet_path.parent/path
            if file_sha256(path)!=desc['sha256']:raise ValueError('Changed current-packet source array')
            self.arrays[k]=np.memmap(path,dtype=desc['dtype'],mode='r',shape=tuple(desc['shape']))
        self.journals={};self.journal_refs_by_id={}
        wanted=None if requested_row_ids is None else set(requested_row_ids)
        refs=list(self.config.get('source_journals',[]))
        if self.config.get('resolve_current_journals'):
            refs+=list(reversed(self.packet.get('source_family_journal_refs',[])))
        unique={str(Path(r.get('path',r.get('file'))).resolve()):r for r in refs}
        for ref in unique.values():
            path=checked(ref)
            for row in read_jsonl(path):
                rid=row['source_row_id']
                if wanted is None or rid in wanted:
                    self.journals[rid]=row;self.journal_refs_by_id[rid]=ref
            if wanted is not None and wanted.issubset(self.journals):break
        self.scope=checked(self.config['programme_scope']);self.roles=checked(self.config['frame_roles'])
        self.depbase,self.code,self.facts=reader_dependencies()
        self.board=None

    def read_frame(self, pts, tasks, output, *, device='cuda'):
        from PIL import Image
        from .native_source import iter_scope_native_rgb
        output=Path(output).resolve()
        for ref in self.code+self.facts:checked(ref)
        if output.exists():raise FileExistsError('Immutable current native review already exists')
        _,rgb,decode=next(iter_scope_native_rgb(self.scope,[int(pts)],roles_path=self.roles))
        cfg=self.config.get('frames',{}).get(str(pts),{})
        maskref=cfg.get('mask')
        mask=np.ones(rgb.shape[:2],bool) if maskref is None else np.asarray(Image.open(checked(maskref)).convert('L'))>0
        if mask.shape!=rgb.shape[:2]:raise ValueError('Native source mask differs')
        output.mkdir(parents=True)
        existing_native=cfg.get('native_image')
        if existing_native is not None:
            imagepath=checked(existing_native)
            if not np.array_equal(np.asarray(Image.open(imagepath).convert('RGB')),rgb):raise ValueError('Referenced native image differs from the fresh exact-PTS decode')
            native_ref=reference(imagepath)
        else:
            Image.fromarray(rgb).save(output/'native.png');native_ref=reference(output/'native.png')
        indices=[];labels={}
        for task in tasks:
            i=int(task['source_row_index']);context=task['context']
            if int(self.arrays['native_pts'][i])!=int(pts):raise ValueError('Review subject belongs to another PTS')
            expected=task.get('source_row_id')
            if expected and self.packet['source_row_ids'][i]!=expected:raise ValueError('Current source-row locator differs')
            if context['source']['video_sha256']!=decode['source_video_sha256']:raise ValueError('Review task belongs to another video')
            indices.append(i);labels[i]=context['entity_ids'][0]
        indices=sorted(set(indices));rows=[]
        for i in indices:
            rid=self.packet['source_row_ids'][i];journal=self.journals.get(rid)
            row=dict(journal or {},source_row_id=rid,shot_id=decode['shot_id'],native_pts=int(pts),
                     current_uv=self.arrays['uv'][i].tolist(),material_role=labels[i],target_id=labels[i],
                     current_packet_row_index=i,original_source_journal_found=journal is not None)
            rows.append(row)
        context=dict(native_video_decode_receipt=decode,current_mask_ref=maskref,
                     evidence_directory=str(output/'profiles'),feedback_run_id=output.parent.name+'_'+output.name)
        boardrows=[r for r in rows if r['material_role'].startswith('board.')]
        result={};outcomes=[]
        if boardrows:
            from .board_source_requalification import BoardMaterialRoleRequalifier
            if self.board is None:self.board=BoardMaterialRoleRequalifier(PROJECT,device=device,use_frozen_visual_model=False)
            result['board']=self.board.evaluate(rgb,mask,boardrows,context)
            outcomes.extend(result['board']['physical_membership_rows'])
            from .material_profile_candidates import read_candidates,ORDERED_POLICY
            columns=sorted({int(round(r['current_uv'][0]))+dx for r in boardrows for dx in (-8,0,8)})
            source=dict(video_sha256=decode['source_video_sha256'],native_pts=int(pts),time_base=decode['time_base'],
                        decoded_native_RGB_sha256=decode['decoded_native_RGB_sha256'])
            roi=cfg.get('board_vertical_roi',[0,rgb.shape[0]])
            result['board_broad_candidates']=read_candidates(rgb,source=source,columns=columns,vertical_roi=roi,
                                                            policy=ORDERED_POLICY,mask=mask)
        from .source_worker import _channels
        from .entity_corners import remeasure_corner
        from .physical_source_operator import measure_paint_midline
        channels=None
        for row in rows:
            label=row['material_role']
            if label.startswith('board.'):continue
            rid=row['source_row_id'];journal=self.journals.get(rid);i=row['current_packet_row_index']
            if label.startswith('Ad'):
                op=(journal or {}).get('current_corner_operator')
                if op:
                    if channels is None:channels=_channels(rgb)
                    reading=remeasure_corner(channels,op['original_corner_seed_native_uv'],op['edge_tangents'],mask)
                    outcomes.append(dict(source_row_id=rid,material_role=label,reader='entity_corners.remeasure_corner',
                        result=reading,status='source_UV_retained',original_UV_unchanged=True,
                        stored_UV=row['current_uv'],remeasurement_delta_px=None if not reading['valid'] else float(np.linalg.norm(np.asarray(reading['current_intersection_uv'])-row['current_uv'])),
                        instance_evidence=(journal or {}).get('physical_identity_status'),
                        reason='Residual never retracts or moves a current RGB-supported fixed corner'))
                else:outcomes.append(dict(source_row_id=rid,material_role=label,status='unknown_original_corner_operator_missing'))
            elif label.startswith('ice.'):
                signal='blue' if any(t in label for t in ('B1','B2')) else 'red' if label.startswith(('ice.C:','ice.C.')) else None
                op=(journal or {}).get('measurement_operator_resolved')
                if op=='paint_center' or signal is not None:
                    reading=measure_paint_midline(rgb,mask,row['current_uv'],self.arrays['profile_normal'][i],confirmed_signal=signal)
                    outcomes.append(dict(source_row_id=rid,material_role=label,reader='physical_source_operator.measure_paint_midline',
                        status='actual_native_material_read',result=reading,physical_instance=(journal or {}).get('physical_identity_status','unknown'),installed=False))
                else:
                    from rinkspace.vision_source_shapes import read_connected_source_material_profiles
                    binding=dict(source_sha256=decode['source_video_sha256'],source_pts=int(pts),time_base=decode['time_base'],shot_id=decode['shot_id'],native_hw=list(rgb.shape[:2]))
                    normal=np.asarray(self.arrays['profile_normal'][i]);radius=max(24.,float(self.arrays['E'][i])*4)
                    reading={direction:read_connected_source_material_profiles(rgb,binding=binding,search_points_uv=[row['current_uv']],normals=[normal],
                        search_intervals_px=[[-radius,radius]],transition=direction,keep_profiles=False)
                        for direction in ('dark_to_light','light_to_dark')}
                    outcomes.append(dict(source_row_id=rid,material_role=label,reader='vision_source_shapes.read_connected_source_material_profiles',
                        status='actual_native_material_candidates_instance_pending',result=reading,installed=False))
            else:outcomes.append(dict(source_row_id=rid,material_role=label,status='unknown_material_operator'))
        # Production objects use numpy scalars/arrays. Serialize actual values,
        # never a projection or an exception string in place of a source read.
        from .board_source_requalification import _plain
        resultref=output/'review.json';write_json(resultref,_plain(dict(source=decode,results=result,outcomes=outcomes)))
        cache_entries=[];resolutions=[];depbase,code,facts=self.depbase,self.code,self.facts
        depbase=dict(depbase)
        for i in indices:
            rid=self.packet['source_row_ids'][i]
            extra_facts=[] if rid not in self.journal_refs_by_id else [self.journal_refs_by_id[rid]]
            source=dict(video_sha256=decode['source_video_sha256'],time_base=decode['time_base'],coordinate_domain='native_pixels',pts=int(pts))
            deps=dict(depbase,effective_facts_sha256=object_sha256(dict(refs=facts+extra_facts,case_versions=effective_case_refs(source,labels[i],decode['shot_id']))),source_binding_sha256=object_sha256(dict(source=source,RGB_sha256=decode['decoded_native_RGB_sha256'],
                      mask_sha256=None if maskref is None else maskref['sha256'])),
                      observation_snapshot_sha256=object_sha256(observation_content(self.packet,self.arrays,i)),navigation_state_sha256='not_used')
            refs=[native_ref,reference(resultref),*code,*facts,*extra_facts]+([] if maskref is None else [maskref])
            cache=cache_path(source,self.packet['source_dataset_ids'][i],self.packet['source_row_ids'][i]);cache.parent.mkdir(parents=True,exist_ok=True)
            write_json(cache,dict(review_dependencies=deps,evidence_refs=refs,extra_fact_refs=extra_facts,material_label=labels[i],shot_id=decode['shot_id'],source_row_id=self.packet['source_row_ids'][i],review=str(resultref)))
            cache_entries.append(dict(source_row_index=i,review_dependencies=deps))
            # Link an executed review to the real checkpoint subject. A review
            # cannot certify a new observation; this only reuses its source facts.
            from rinkspace.feedback_runtime import feedback_store
            from rink_feedback.cycle import FeedbackCycle
            task=next(t for t in tasks if int(t['source_row_index'])==i)
            subject=self.packet['source_dataset_ids'][i]+'::'+self.packet['source_row_ids'][i]
            with feedback_store(PROJECT,'ankara_2026') as store:
                try:oldfinding=store.get_finding(task['finding_id'])
                except Exception:oldfinding={}
                finding=dict(source=source,event_id='ankara_2026',programme_id='okada_mei_sp',camera_id=oldfinding.get('camera_id'),
                    entity_ids=[labels[i]],observation_ids=[subject],
                    signal_type=oldfinding.get('signal_type','source_material_review'),
                    problem_key=oldfinding.get('problem_key','current_native_source_material_review'),
                    context_tags=oldfinding.get('context_tags',['native_source_recheck']),
                    evidence_refs=[native_ref,reference(resultref)],review_dependencies=deps)
                cycle=FeedbackCycle(store);job=cycle.checkpoint(output.name,[finding]);fid=job['tasks'][0]['finding_id']
                native=dict(native_ref,kind='native_pixels',source=source,independent_of_model=True,current_frame=True)
                outcome=next((o for o in outcomes if o.get('source_row_id')==self.packet['source_row_ids'][i]),{})
                good=(outcome.get('reader')=='entity_corners.remeasure_corner'and outcome.get('result',{}).get('valid')is True
                      and outcome.get('remeasurement_delta_px')==0. and bool(outcome.get('instance_evidence')))
                checks={name:dict(status='verified'if good else'unknown',evidence_refs=[native_ref['sha256']])
                        for name in ('semantic_class','physical_instance','measurement_operator','visibility','position')}
                if good:
                    checks['visibility']['visible']=True
                    checks['position'].update(value='same current native finite-edge intersection reproduced; geometric subpixel truth not asserted')
                elif outcome.get('status')=='wrong_role':
                    checks['measurement_operator']=dict(status='verified',value='registered material role conflicts with actual RGB plateau',evidence_refs=[native_ref['sha256']])
                resolution=cycle.learn(dict(finding_id=fid,source=source,resolution_type='model_mismatch'if good else'operator_error'if outcome.get('status')=='wrong_role'else'unresolved',
                    checks=checks,evidence_refs=[native,reference(resultref),*extra_facts],review_dependencies=deps,request_recheck=True),lesson=None)
                resolutions.append(resolution['resolution'])
        receipt=dict(schema='rinkfit.current_packet_native_source_review.v1',native_pts=int(pts),decode=decode,
          source_packet=reference(self.packet_path),native_image=native_ref,mask=maskref,
          review=reference(resultref),review_dependencies=cache_entries,
          counts=dict(subjects=len(indices),board=len(boardrows),other_outcomes=len(outcomes)),
          source_feedback=result.get('board',{}).get('source_feedback'),resolutions=resolutions,
          old_source_UV_unchanged=True,new_observation_installed=False,
          mask_missing=maskref is None,visual_accuracy_improvement_claimed=False)
        write_json(output/'receipt.json',receipt);return receipt
