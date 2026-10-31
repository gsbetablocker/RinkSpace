"""Read new-shot ice from current native search geometry and original RGB.

No historical point packages or trajectories are opened. The current native
state only locates source searches; original RGB supplies every candidate UV.
Candidates stay pending unless explicit source-hashed anchor approval and actual
RGB connectivity uniquely establish a whole-qualified source fragment identity.
"""
from pathlib import Path
import argparse,hashlib,json,time,re
from collections import defaultdict
import cv2,numpy as np
from capture_independent_materials import native_ice_inventory,source_exclusion,spaced_indices,plain,write
from prepare_independent_navigation import frames
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.session.material_tracks import (qualify_visible_line_fragments,source_nodes,same_frame_conflicts,
    link_fragments,propagate_identities,source_line_reference_support)
from rinkspace.path_policy import guard_project_write
from rinkspace.session.chromatic_line_navigation import source_board_floor_limit


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def approval_sources(anchor_capture, approval_path, shots, source_sha256):
    """Explicit group allowlist; pending reference captures never approve themselves."""
    if approval_path is None:return [],dict(status='no_explicit_anchor_approval',inputs={})
    if anchor_capture is None:raise ValueError('--anchor-approval requires --anchor-capture')
    approval_path=Path(approval_path).resolve(strict=True)
    document=json.loads(approval_path.read_text('utf8'))
    if document.get('schema')!='rinkspace.independent_line_anchor_approval.v1':raise ValueError('Unsupported anchor approval schema')
    if document.get('source_video_sha256')!=source_sha256:
        raise ValueError('Anchor approval belongs to a different original video')
    if not isinstance(document.get('groups'),list):raise ValueError('Anchor approval needs groups')
    inputs={str(approval_path):digest(approval_path)};groups=[];seen=set()
    for number,group in enumerate(document['groups']):
        if group.get('shot') not in shots:continue
        if group.get('approved') is not True:continue
        sid,pts,identity=group['shot'],group.get('pts'),group.get('path_id')
        evidence=group.get('evidence')
        if (isinstance(pts,bool) or not isinstance(pts,int) or not isinstance(identity,str) or not identity
                or not isinstance(evidence,(str,list,dict)) or not evidence):
            raise ValueError('Approved anchor needs shot/PTS/path_id and explicit evidence')
        sha=group.get('source_rows_file_sha256')
        if not isinstance(sha,str) or re.fullmatch('[0-9a-f]{64}',sha) is None:
            raise ValueError('Approved anchor needs source_rows_file_sha256')
        key=(sid,pts,identity)
        if key in seen:raise ValueError('Duplicate approved anchor group')
        seen.add(key)
        source=Path(anchor_capture).resolve(strict=True)/sid/'source_rows.jsonl'
        if str(source) not in inputs:inputs[str(source)]=digest(source)
        if inputs[str(source)]!=sha:raise ValueError('Approved anchor source_rows SHA changed: '+str(source))
        groups.append(dict(group,approval_group=number,source_rows_file=str(source),approval=str(approval_path),
                           approval_sha256=inputs[str(approval_path)]))
    return groups,dict(status='explicit_group_approval',inputs=inputs,groups=len(groups))


def anchor_records(groups, source_sha256):
    """Load only approved current-shot files; retain exact reference UV and proof."""
    records={};result=[]
    for group in groups:
        path=group['source_rows_file']
        if path not in records:
            records[path]=[]
            for line_number,line in enumerate(Path(path).read_bytes().splitlines(keepends=True),1):
                record=json.loads(line)
                records[path].append((record,line_number,hashlib.sha256(line).hexdigest()))
        matches=[item for item in records[path] if item[0].get('pts')==group['pts']]
        if len(matches)!=1:raise ValueError('Approved reference PTS must occur exactly once')
        record,line_number,line_sha=matches[0]
        if record.get('source_video_sha256')!=source_sha256:
            raise ValueError('Reference group source video differs')
        bgr=record.get('source_BGR_sha256')
        if not isinstance(bgr,str) or re.fullmatch('[0-9a-f]{64}',bgr) is None:
            raise ValueError('Reference group lacks exact source BGR proof')
        rows=[r for r in record.get('rows',[]) if r.get('path_id')==group['path_id']]
        uv=np.asarray([r.get('uv') for r in rows],float).reshape(-1,2)
        if (not len(uv) or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>[1919,1079])
                or any(r.get('model_UV_synthesized') is True for r in rows)):
            raise ValueError('Approved reference needs unchanged actual native source UV')
        signals={r['material_signal'] for r in rows if r.get('material_signal') is not None}
        if len(signals)>1 or any(signal not in ('red','blue','dark') for signal in signals):
            raise ValueError('Approved source readout material_signal must be explicit and consistent')
        result.append(dict(group,uv=uv,material_signal=next(iter(signals)) if signals else None,
                           source_BGR_sha256=bgr,source_line=line_number,source_line_sha256=line_sha,
                           reference_fit_eligible=record.get('fit_eligible') is True))
    return result


def load_navigation_graph(root, shot):
    path=Path(root)/shot['shot_id']/'navigation/source_graph.npz'
    note=dict(path=str(path),sha256=None,status='missing_source_graph')
    if not path.is_file():return None,note
    note.update(sha256=digest(path),status='current_source_graph')
    with np.load(path,allow_pickle=False) as z:graph={k:z[k] for k in z.files}
    expected=np.arange(shot['processing']['first_pts'],shot['processing']['stop_pts_exclusive'],256,dtype=np.int64)
    pts,supported,steps=graph['pts'],graph['supported'],graph['adjacent_h']
    if (pts.dtype.kind not in 'iu' or not np.array_equal(pts,expected) or supported.dtype.kind!='b'
            or supported.shape!=(len(pts),) or steps.shape!=(len(pts)-1,3,3)):
        raise ValueError('Source navigation graph must match registered actual shot PTS')
    cumulative=np.tile(np.eye(3),(len(pts),1,1));segments=np.zeros(len(pts),int)
    for i,step in enumerate(steps,1):
        # supported[i] qualifies only the incoming (i-1 -> i) source edge.
        usable=bool(supported[i] and np.isfinite(step).all())
        if usable:
            current=step@cumulative[i-1]
            try:np.linalg.inv(current)
            except np.linalg.LinAlgError:usable=False
            if usable and np.isfinite(current).all() and current[2,2]!=0:
                cumulative[i]=current/current[2,2]
            else:usable=False
        segments[i]=segments[i-1]+int(not usable)
    return dict(graph,index={int(p):i for i,p in enumerate(pts)},cumulative=cumulative,segments=segments),note


def navigation_between(graph, previous_pts, current_pts):
    if graph is None:return None,'missing_source_graph'
    a,b=graph['index'].get(int(previous_pts)),graph['index'].get(int(current_pts))
    if a is None or b is None:return None,'PTS_absent_from_source_graph'
    if a==b:return np.eye(3),'same_actual_source_PTS'
    if graph['segments'][a]!=graph['segments'][b]:
        return None,'unsupported_source_graph_interval'
    h=graph['cumulative'][b]@np.linalg.inv(graph['cumulative'][a])
    if not np.isfinite(h).all() or h[2,2]==0:return None,'invalid_source_graph_transform'
    return h/h[2,2],'continuous_supported_source_navigation'



def line_search_guide(uv, width=1920, height=1080):
    uv=np.asarray(uv,float).reshape(-1,2)
    if not np.isfinite(uv).all() or len(np.unique(uv,axis=0))<3 or np.linalg.norm(np.ptp(uv,axis=0))<56.:return None
    centre=uv.mean(0);_,_,axes=np.linalg.svd(uv-centre,full_matrices=False);direction=axes[0]
    bounds=[]
    for axis,limit in ((0,width-1),(1,height-1)):
        if direction[axis]==0:continue
        for boundary in (0.,float(limit)):
            t=(boundary-centre[axis])/direction[axis];point=centre+t*direction
            if np.all(point>=-1e-6) and np.all(point<=[width-1+1e-6,height-1+1e-6]):bounds.append((t,point))
    if len(bounds)<2:return None
    return np.asarray([min(bounds,key=lambda p:p[0])[1],max(bounds,key=lambda p:p[0])[1]])


def source_search_records(capture, shots, source_sha256, center_path):
    if capture is None:return {},dict(inputs={},status='no_optional_source_search_capture')
    binding=Path(capture)/'binding.json';document=json.loads(binding.read_text('utf8'))
    if document.get('source_video_sha256')!=source_sha256:raise ValueError('Source search capture video differs')
    notes=dict(inputs={str(binding):digest(binding)},status='same_frame_C_search_only',missing_shots=[]);records={}
    for sid in shots:
        source=Path(capture)/sid/'source_rows.jsonl'
        if not source.is_file():notes['missing_shots'].append(sid);continue
        sha=digest(source);notes['inputs'][str(source)]=sha;seen=set()
        for line_number,line in enumerate(source.read_bytes().splitlines(keepends=True),1):
            record=json.loads(line);pts=record.get('pts')
            if isinstance(pts,bool) or not isinstance(pts,int) or pts in seen:raise ValueError('Source search PTS must be distinct actual integers')
            seen.add(pts)
            if record.get('source_video_sha256')!=source_sha256:raise ValueError('Source search frame video differs')
            bgr=record.get('source_BGR_sha256')
            if not isinstance(bgr,str) or re.fullmatch('[0-9a-f]{64}',bgr) is None:raise ValueError('Source search frame lacks BGR proof')
            rows=[row for row in record.get('rows',[]) if row.get('path_id')==center_path]
            if not rows:continue
            uv=np.asarray([row.get('uv') for row in rows],float).reshape(-1,2)
            if (not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>[1919,1079])
                    or any(row.get('model_UV_synthesized') is True for row in rows)):
                raise ValueError('Source search guide needs actual unchanged source UV')
            records[(sid,pts)]=dict(uv=uv,source_BGR_sha256=bgr,path=str(source),sha256=sha,
                source_line=line_number,source_line_sha256=hashlib.sha256(line).hexdigest(),
                inherited_fit_eligibility=False,inherited_identity=False)
    return records,notes


def same_frame_center_search(curves, center_path, record, source_bgr_sha256, width=1920, height=1080):
    if record is None:return curves,dict(status='same_frame_C_search_missing',search_only=True)
    if record['source_BGR_sha256']!=source_bgr_sha256:
        return curves,dict(status='same_frame_C_search_RGB_mismatch',search_only=True)
    guide=line_search_guide(record['uv'],width,height)
    if guide is None:return curves,dict(status='insufficient_same_frame_C_search_support',search_only=True)
    result=dict(curves);result[center_path]=dict(uv=guide,valid=np.ones(len(guide),bool))
    note={k:v for k,v in record.items() if k!='uv'}
    note.update(status='same_frame_actual_C_UV_search_guide',path_id=center_path,guide_UV=guide.tolist(),
        search_only=True,observation_UV_created=False,identity_established=False)
    return result,note


def ice_source_mask(rgb, exclusion):
    limit,receipt=source_board_floor_limit(rgb,exclusion)
    mask=np.asarray(exclusion,bool).copy()
    if limit is not None:mask[:max(0,int(np.ceil(limit)))]=True
    return mask,receipt


def resolve_same_frame_identity_conflicts(frames, labels):
    """A contradictory same-source duplicate keeps both RGB components pending."""
    lookup={(int(label['frame']),int(label['fragment'])):label for label in labels};bad={}
    for fi,frame in enumerate(frames):
        for conflict in frame['cross_identity_conflicts']:
            first=lookup.get((fi,int(conflict['a'])));second=lookup.get((fi,int(conflict['b'])))
            if first and second and first['identities'] and second['identities']:
                identities=sorted(set(first['identities']+second['identities']))
                if len(identities)>1:
                    for key in set(first['proofs'])|set(second['proofs']):bad[key]=set(identities)|bad.get(key,set())
    result=[]
    for label in labels:
        rivals=set().union(*(bad.get(key,set()) for key in label['proofs']))
        result.append(dict(label,status='identity_conflict',identities=sorted(set(label['identities'])|rivals),
                           same_frame_source_conflict=True) if rivals else label)
    return result


def source_reference_curves(curves, paths, groups, pts, graph, width=1920, height=1080):
    """A transferred/fitted line is search only, never an observation or identity."""
    result=dict(curves);notes=[]
    for path in paths:
        references=[g for g in groups if g['path_id']==path['native_path_id']]
        for reference in sorted(references,key=lambda g:abs(pts-g['pts'])):
            h,reason=navigation_between(graph,reference['pts'],pts)
            if h is None:continue
            homogeneous=np.c_[reference['uv'],np.ones(len(reference['uv']))]@h.T
            if not np.isfinite(homogeneous).all() or np.any(homogeneous[:,2]==0):continue
            uv=homogeneous[:,:2]/homogeneous[:,2:]
            endpoints=line_search_guide(uv,width,height)
            if endpoints is None:continue
            result[path['native_path_id']]=dict(uv=endpoints,valid=np.ones(2,bool))
            notes.append(dict(path_id=path['native_path_id'],reference_pts=reference['pts'],navigation=reason,
                approval_group=reference['approval_group'],source_rows_file_sha256=reference['source_rows_file_sha256'],
                H_current_from_reference=h.tolist(),guide_UV=endpoints.tolist(),search_only=True,observation_UV_created=False))
            break
    return result,notes


def read_material_groups(readers, rgb, pts, mask, sid, curves, inventory):
    parts=[];offset=0
    for reader,global_curves in readers:
        part=reader.read(rgb,pts,None,mask,shot_id=sid,native_projections=curves)
        mapped=dict(part)
        mapped['query_curve']=np.asarray(global_curves,dtype=np.int16)[part['query_curve']]
        mapped['candidate_query']=part['candidate_query']+offset
        parts.append(mapped);offset+=len(part['query_uv'])
    result=dict(parts[0])
    for key,value in parts[0].items():
        if isinstance(value,np.ndarray) and key!='curve_status':result[key]=np.concatenate([part[key] for part in parts])
    result['curve_status']=np.zeros(len(inventory),dtype=np.uint8)
    for (reader,global_curves),part in zip(readers,parts):result['curve_status'][global_curves]=part['curve_status']
    result['resolved_measurement_roles']=[{k:v for k,v in p.items() if k!='xyz'} for p in inventory]
    return result


def match_anchor_fragments(groups, pts, measurement, fragments, source_bgr_sha256, supported):
    anchors=[];attempts=[]
    for group in groups:
        if group['pts']!=pts:continue
        # Incoming navigation support is not validity of the current RGB.
        reason='source_BGR_sha256_mismatch' if group['source_BGR_sha256']!=source_bgr_sha256 else None
        matched=[]
        if reason is None:
            for fi,fragment in enumerate(fragments):
                ids=np.asarray(fragment['candidate_ids'],int)
                support=source_line_reference_support(group['uv'],measurement['candidate_uv'][ids],measurement['candidate_query'][ids])
                if support['qualified']:
                    anchors.append(dict(pts=pts,fragment=fi,identity=group['path_id'],approval_group=group['approval_group'],
                        approval_sha256=group['approval_sha256'],source_rows_file_sha256=group['source_rows_file_sha256'],
                        source_line_sha256=group['source_line_sha256'],support=support,evidence=group['evidence']))
                    matched.append(fi)
        attempts.append(dict(pts=pts,identity=group['path_id'],approval_group=group['approval_group'],
            matched_fragments=matched,status='matched_actual_source_contour' if matched else (reason or 'no_actual_source_contour_support')))
    return anchors,attempts


def measured_row(measurement, candidate, fragment, path, *, identity=None):
    i=int(candidate)
    return dict(path_id=identity or path['native_path_id'],entity=path['curve_id'] if identity is None else identity,
        query_path_id=path['native_path_id'],material_signal=path['material_signal'],uv=measurement['candidate_uv'][i].tolist(),
        actual_source_sides=measurement['candidate_sides'][i].tolist(),
        actual_source_quantiles=measurement['candidate_side_quantiles_uv'][i].tolist(),
        measurement_operator='paired_material_boundary_midline',model_UV_synthesized=False,candidate=i,fragment=fragment,
        source_identity_basis='approved source contour anchor and unique bidirectional RGB fragment graph' if identity else
                              'pending whole-scene identity and competing-path qualification')


def identity_rows(measurement, fragments, paths, labels, conflicts, supported=True):
    """Choose existing whole-qualified fragment readouts; merge duplicate queries."""
    by_fragment={int(label['fragment']):label for label in labels};identities={};pending=[]
    for fi in range(len(fragments)):
        label=by_fragment.get(fi)
        if label and label['status']=='anchored' and len(label['identities'])==1:
            identities[fi]=label['identities'][0]
    blocked=set()
    for conflict in conflicts:
        a,b=int(conflict['a']),int(conflict['b'])
        if a in identities and b in identities and identities[a]!=identities[b]:blocked.update((a,b))
    grouped=defaultdict(list)
    for fi,fragment in enumerate(fragments):
        if fi in identities and fi not in blocked:grouped[identities[fi]].append(fi)
        else:
            pending.append(dict(fragment=fi,candidate_ids=fragment['candidate_ids'],
                reason='same_frame_identity_conflict' if fi in blocked else
                       'identity_conflict' if by_fragment.get(fi,{}).get('status')=='identity_conflict' else 'no_unique_approved_anchor_connection',
                identities=by_fragment.get(fi,{}).get('identities',[])))
    rows=[];merges=[]
    for identity,fragment_ids in grouped.items():
        candidates=np.unique([i for fi in fragment_ids for i in fragments[fi]['candidate_ids']]).astype(int)
        uv=measurement['candidate_uv'][candidates]
        # Ordering and thinning choose original candidate indices only. No
        # fitted point, midpoint between queries or gap pixel is introduced.
        _,first=np.unique(uv,axis=0,return_index=True);candidates=candidates[np.sort(first)];uv=measurement['candidate_uv'][candidates]
        if len(uv)>1:
            _,_,axes=np.linalg.svd(uv-uv.mean(0),full_matrices=False)
            candidates=candidates[np.argsort(uv@axes[0],kind='stable')]
        retained=candidates[spaced_indices(measurement['candidate_uv'][candidates],48.)]
        for candidate in retained:
            origin=next(fi for fi in fragment_ids if int(candidate) in fragments[fi]['candidate_ids'])
            row=measured_row(measurement,int(candidate),origin,paths[fragments[origin]['curve']],identity=identity)
            row['identity_fragments']=fragment_ids;row['source_shape_qualified_whole_fragment']=True
            rows.append(row)
        merges.append(dict(identity=identity,fragments=fragment_ids,actual_candidate_ids=candidates.tolist(),
            retained_candidate_ids=retained.tolist(),spacing_px=48.,source_UV_changed=False,interpolated_points_created=0))
    return rows,pending,merges


def finalize_identity_capture(folder, frame_graph, edge_sets, anchors, paths, provenance):
    labels=resolve_same_frame_identity_conflicts(frame_graph,propagate_identities(frame_graph,edge_sets,anchors))
    labels_by_frame=defaultdict(list)
    for label in labels:labels_by_frame[int(label['frame'])].append(label)
    links_path=folder/'source_links.jsonl';labels_path=folder/'source_labels.json'
    write(labels_path,dict(labels=labels,anchors=anchors,source_UV_synthesized=False))
    proof=dict(links=str(links_path),links_sha256=digest(links_path),labels=str(labels_path),
               labels_sha256=digest(labels_path),**provenance)
    qualified=0;pending_count=0;counts=defaultdict(int)
    with (folder/'candidate_source_rows.jsonl').open(encoding='utf8') as source, \
         (folder/'source_rows.jsonl').open('x',encoding='utf8') as output, \
         (folder/'pending_source_rows.jsonl').open('x',encoding='utf8') as pending_output:
        for fi,(line,frame) in enumerate(zip(source,frame_graph,strict=True)):
            record=json.loads(line)
            if record['pts']!=frame['pts'] or digest(record['raw_source_measurements'])!=frame['raw_sha256']:
                raise ValueError('Current raw capture changed before identity selection')
            with np.load(record['raw_source_measurements'],allow_pickle=False) as z:measurement={k:z[k] for k in z.files}
            rows,pending,merges=identity_rows(measurement,frame['fragments'],paths,labels_by_frame[fi],
                                            frame['cross_identity_conflicts'],frame['navigation_supported'])
            review_image=None
            if frame.get('source_image'):
                image=cv2.imread(frame['source_image'])
                if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=record['source_BGR_sha256']:
                    raise ValueError('Current source proof PNG changed')
                for row in rows:cv2.circle(image,tuple(np.rint(row['uv']).astype(int)),3,(0,230,20),-1)
                for identity in sorted({row['path_id'] for row in rows}):
                    uv=np.asarray([row['uv'] for row in rows if row['path_id']==identity]);at=np.rint(np.median(uv,axis=0)).astype(int)
                    cv2.putText(image,identity,tuple(at+[0,-10]),0,.5,(0,210,20),1,cv2.LINE_AA)
                cv2.putText(image,f'PTS {record["pts"]} | {len(rows)} source-qualified points | {len(pending)} pending fragments',(15,35),0,.65,(0,220,255),2,cv2.LINE_AA)
                review_image=folder/f'{record["pts"]}_source_qualified.png'
                if not cv2.imwrite(str(review_image),image):raise OSError('Qualified proof PNG write failed')
            qualified_record=dict(record,rows=rows,fit_eligible=bool(rows),source_identity_proof=proof,
                identity_review_image=str(review_image) if review_image else None,
                review_state='approved_anchor_and_bidirectional_RGB_source_qualified' if rows else 'pending_independent_identity',
                pending_fragments=pending,duplicate_query_merge=merges)
            output.write(json.dumps(plain(qualified_record),ensure_ascii=False)+'\n')
            pending_ids={p['fragment'] for p in pending}
            pending_output.write(json.dumps(plain(dict(record,rows=[r for r in record['rows'] if r['fragment'] in pending_ids],
                fit_eligible=False,pending_fragments=pending,source_identity_proof=proof)),ensure_ascii=False)+'\n')
            qualified+=len(rows);pending_count+=len(pending)
            for row in rows:counts[row['path_id']]+=1
    write(folder/'identity_graph.json',dict(frames=frame_graph,anchors=anchors,**provenance,
        native_or_H_coordinates_used_as_observations=False,identity_assumed_from_curve_names=False))
    return dict(qualified_rows=qualified,pending_fragments=pending_count,by_identity=dict(counts),
                source_uv_synthesized=False,source_links=str(links_path),source_labels=str(labels_path))



def anchor_signal_paths(paths, groups):
    result=[];notes=[]
    for path in paths:
        references=[g for g in groups if g['path_id']==path['native_path_id'] and g.get('material_signal') is not None]
        signals={g['material_signal'] for g in references}
        if len(signals)>1:raise ValueError('Approved source readout signals disagree for '+path['native_path_id'])
        selected=dict(path)
        if signals:
            selected['material_signal']=next(iter(signals))
            notes.append(dict(path_id=path['native_path_id'],material_signal=selected['material_signal'],
                source_rows_file_sha256=sorted({g['source_rows_file_sha256'] for g in references}),
                approval_groups=[g['approval_group'] for g in references],native_material_changed=False,identity_established=False))
        result.append(selected)
    return result,notes


def capture_frames(index, source, selected_pts):
    if not selected_pts:
        yield from enumerate(frames(index,source));return
    window=source['processing'];pts=sorted({p for p in selected_pts if window['first_pts']<=p<=window['last_pts']})
    if any((p-window['first_pts'])%256 for p in pts):raise ValueError('Explicit capture PTS must be actual native exposures')
    if not pts:return
    groups=np.split(np.asarray(pts,np.int64),np.flatnonzero(np.diff(pts)!=256)+1)
    for group in groups:
        partial=dict(source,processing=dict(window,first_pts=int(group[0]),last_pts=int(group[-1]),stop_pts_exclusive=int(group[-1])+256))
        for pts,image in frames(index,partial):yield (pts-window['first_pts'])//256,(pts,image)


def timed_capture_frames(index, source, selected_pts):
    iterator=iter(capture_frames(index,source,selected_pts))
    while True:
        started=time.perf_counter()
        try:fi,pair=next(iterator)
        except StopIteration:return
        yield fi,pair,time.perf_counter()-started


def capture(args):
    args.output=guard_project_write(args.output)
    if args.output.exists():raise FileExistsError(args.output)
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'))
    manifest=json.loads((args.run_root/'joint_manifest.json').read_text('utf8'))
    if not args.shots or len(set(args.shots))!=len(args.shots):raise ValueError('Select distinct shots')
    sources={s['shot_id']:s for s in index['shots']};shots={s['id']:s for s in manifest['shots']}
    if any(sid not in sources or sid not in shots for sid in args.shots):raise ValueError('Selected shot missing from current bootstrap/state')
    anchor_capture=getattr(args,'anchor_capture',None);approval_path=getattr(args,'anchor_approval',None)
    reference_navigation=getattr(args,'source_reference_navigation',False)
    search_capture=getattr(args,'source_search_capture',None)
    anchored_only=getattr(args,'anchored_path_search_only',False)
    if (approval_path or reference_navigation or anchored_only) and not anchor_capture:raise ValueError('Anchor approval/navigation requires --anchor-capture')
    if anchored_only and not approval_path:raise ValueError('--anchored-path-search-only requires explicit approved groups')
    declarations,approval=approval_sources(anchor_capture,approval_path,args.shots,index['source']['actual_sha256'])
    groups=anchor_records(declarations,index['source']['actual_sha256'])
    navigation={sid:load_navigation_graph(root,sources[sid]) for sid in args.shots} if anchor_capture else {}
    if (anchor_capture or search_capture) and digest(index['source']['resolved_path'])!=index['source']['actual_sha256']:
        raise ValueError('Actual original video SHA changed')
    state=dict(np.load(args.state,allow_pickle=False))
    library=args.library or Path(manifest['native_library'])
    if digest(library)!=str(state['native_library_provenance_sha256']):
        raise ValueError('Source search must use the same native library provenance as its state')
    scene_path=Path(manifest['scene_path'])
    if digest(scene_path)!=manifest['scene_file_sha256']:raise ValueError('Scene changed')
    raw=json.loads(scene_path.read_text('utf8'));model=SceneParameterModel(raw,scene_path,optimize_scene=True)
    parameter_state=ParameterState(model.initial_state.layout,state['scene_values']);core=NativeCore(library)
    sample=sample_entities(parameter_state.apply_scene(raw),core);world=sample['xyz'];cuts=sample['cuts']
    paths=native_ice_inventory(cuts,world);lookup={c['id']:c for c in cuts}
    valid_ids={p['native_path_id'] for p in paths}
    center_ids=[p['native_path_id'] for p in paths if p['feature_id']=='C']
    if search_capture and len(center_ids)!=1:raise ValueError('Optional source-search capture supports one C path')
    search_records,search_binding=source_search_records(search_capture,args.shots,index['source']['actual_sha256'],center_ids[0] if center_ids else None)
    if any(g['path_id'] not in valid_ids for g in groups):raise ValueError('Approved anchor identity is not in the current native ice inventory')
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    def make_reader(selected,radius):
        return IceProgrammeSourceReader(None,source_sha256=index['source']['actual_sha256'],time_base='1/12800',
            native_paths=selected,sample_spacing_px=24.,search_radius_px=radius,separate_bands=True,
            prepared_source_context=dict(source_sha256=index['source']['actual_sha256'],time_base='1/12800',
                authority=[dict(path=str(scene_path),sha256=manifest['scene_file_sha256'])]))
    base_paths=paths
    active_by_shot={sid:[i for i,p in enumerate(base_paths) if not anchored_only or any(
        g['shot']==sid and g['path_id']==p['native_path_id'] for g in groups)] for sid in args.shots}
    if any(not ids for ids in active_by_shot.values()):raise ValueError('Selected shot has no approved physical paths to search')
    signal_notes={sid:anchor_signal_paths(base_paths,[g for g in groups if g['shot']==sid])[1] for sid in args.shots}
    args.output.mkdir(parents=True);cv2.setNumThreads(2);started=time.monotonic()
    write(args.output/'binding.json',dict(schema='rinkspace.independent_ice_capture.v1',
        source_video_sha256=index['source']['actual_sha256'],source_index_sha256=digest(root/'index.json'),
        scene_path=str(scene_path),scene_sha256=manifest['scene_file_sha256'],manifest_sha256=digest(args.run_root/'joint_manifest.json'),
        adapter_sha256=digest(__file__),source_qualification_sha256=digest(qualify_visible_line_fragments.__code__.co_filename),
        source_floor_guard_sha256=digest(source_board_floor_limit.__code__.co_filename),
        state=str(args.state),state_sha256=digest(args.state),
        native_library=str(library),native_library_sha256=digest(library),independent_source_capture=True,
        legacy_new_shot_inputs_read=False,source_UV_synthesized=False,query_spacing_px=24.,fit_spacing_px=48.,
        search_radius_px=args.search_radius,native_inventory=[{k:v for k,v in p.items() if k!='xyz'} for p in paths],
        anchor_capture=str(anchor_capture) if anchor_capture else None,anchor_approval=approval,
        source_readout_overrides=signal_notes,anchored_path_search_only=anchored_only,
        searched_path_ids={sid:[base_paths[i]['native_path_id'] for i in ids] for sid,ids in active_by_shot.items()},
        explicit_PTS_decode='selected_native_contiguous_runs_only' if args.pts else 'complete_source_shot',
        source_search_capture=search_binding,source_reference_navigation=reference_navigation,center_search_radius_px=getattr(args,'center_search_radius',128.) if reference_navigation or search_capture else args.search_radius,
        source_graphs={sid:note for sid,(_,note) in navigation.items()},fit_eligible=False))
    for sid in args.shots:
        source,shot=sources[sid],shots[sid]
        paths,_=anchor_signal_paths(base_paths,[g for g in groups if g['shot']==sid])
        active_indices=active_by_shot[sid];readers=[]
        if reference_navigation or search_capture:
            for central in (False,True):
                ids=[i for i in active_indices if (paths[i]['feature_id']=='C')==central]
                if ids:readers.append((make_reader([paths[i] for i in ids],getattr(args,'center_search_radius',128.) if central else args.search_radius),ids))
        else:readers=[(make_reader([paths[i] for i in active_indices],args.search_radius),active_indices)]
        folder=guard_project_write(args.output/sid)
        for name in ('candidate_source_rows.jsonl','source_rows.jsonl','pending_source_rows.jsonl','identity_graph.json',
                     'source_links.jsonl','source_labels.json','anchor_matches.json','coverage.json','frame_timings.jsonl'):
            guard_project_write(folder/name)
        folder.mkdir();total=0;frame_count=0;counts={}
        graph,graph_note=navigation[sid] if anchor_capture else (None,{})
        references=[group for group in groups if group['shot']==sid]
        representatives={f['actual_pts']for f in source['frames']}
        frame_graph=[];edge_sets=[];anchors=[];anchor_attempts=[];previous=None
        rowfile=folder/('candidate_source_rows.jsonl' if anchor_capture else 'source_rows.jsonl')
        stage_totals=defaultdict(float)
        with rowfile.open('w',encoding='utf8',buffering=1)as output, (folder/'source_links.jsonl').open('x',encoding='utf8') as links,              (folder/'frame_timings.jsonl').open('x',encoding='utf8',buffering=1) as timing_log:
            for fi,(pts,image),decode_s in timed_capture_frames(index,source,args.pts):
                if args.pts and pts not in args.pts:continue
                if not args.pts and fi%args.stride and pts not in representatives:continue
                timing={'decode_s':decode_s};tick=time.perf_counter()
                def mark(stage):
                    nonlocal tick
                    now=time.perf_counter();timing[stage]=now-tick;tick=now
                mask,mask_path=source_exclusion(root,sid,pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                mask,floor_receipt=ice_source_mask(rgb,mask)
                source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest();mark('mask_floor_s')
                clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
                projection=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,world)
                projected=projection['uv'].copy();projected[projection['status']!=0]=np.nan
                curves={p['native_path_id']:dict(uv=projected[lookup[p['native_path_id']]['begin']:lookup[p['native_path_id']]['end']],valid=projection['status'][lookup[p['native_path_id']]['begin']:lookup[p['native_path_id']]['end']]==0)for p in paths}
                search_notes=[]
                if reference_navigation:curves,search_notes=source_reference_curves(curves,paths,references,pts,graph)
                if search_capture:
                    curves,search_receipt=same_frame_center_search(curves,center_ids[0],search_records.get((sid,pts)),source_bgr_sha256)
                    search_notes.append(search_receipt)
                mark('search_navigation_s')
                measured=read_material_groups(readers,rgb,pts,mask,sid,curves,paths);mark('measure_s')
                measured['all_modeled_ice_paths_processed']=len(active_indices)==len(paths)
                fragments,report=qualify_visible_line_fragments(measured,active_indices,mask);mark('qualify_s')
                nodes=source_nodes(rgb,measured,fragments,paths);conflicts=same_frame_conflicts(nodes,fragments);mark('nodes_s')
                stem=folder/str(pts)
                for ending in ('.npz','.json','_nodes.npz','_source.png','_source_qualified.png','_native_search.jpg','_source_candidates.jpg'):
                    guard_project_write(folder/(str(pts)+ending))
                np.savez_compressed(stem.with_suffix('.npz'),**{k:v for k,v in measured.items()if isinstance(v,np.ndarray)})
                write(stem.with_suffix('.json'),dict(pts=pts,fragments=fragments,reports=report,cross_identity_conflicts=conflicts,
                    source_reference_search=search_notes,floor_receipt=floor_receipt))
                rows=[]
                for j,fragment in enumerate(fragments):
                    ids=np.asarray(fragment['candidate_ids'],int);path=paths[fragment['curve']]
                    for k in spaced_indices(measured['candidate_uv'][ids]):
                        rows.append(measured_row(measured,int(ids[k]),j,path))
                        counts[path['curve_id']]=counts.get(path['curve_id'],0)+1
                record=dict(pts=pts,source_BGR_sha256=source_bgr_sha256,source_video_sha256=index['source']['actual_sha256'],
                    mask_path=str(mask_path),mask_sha256=digest(mask_path),fit_eligible=False,review_state='pending_independent_identity',rows=rows,
                    raw_source_measurements=str(stem.with_suffix('.npz')),cross_identity_conflicts=conflicts,
                    floor_receipt=floor_receipt,ice_search_exclusion_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    source_search_geometry=search_notes,source_readout_overrides=signal_notes[sid])
                output.write(json.dumps(plain(record),ensure_ascii=False)+'\n')
                mark('serialize_raw_s')
                if anchor_capture:
                    h,nav_reason=navigation_between(graph,pts,pts);supported=h is not None
                    np.savez_compressed(folder/f'{pts}_nodes.npz',**nodes)
                    source_image=None
                    if args.pts or pts in representatives:
                        source_image=folder/f'{pts}_source.png'
                        if not cv2.imwrite(str(source_image),image):raise OSError('Original source proof PNG write failed')
                    frame_graph.append(dict(pts=pts,fragments=fragments,cross_identity_conflicts=conflicts,
                        source_image=str(source_image) if source_image else None,
                        source_BGR_sha256=source_bgr_sha256,raw=str(stem.with_suffix('.npz')),raw_sha256=digest(stem.with_suffix('.npz')),
                        nodes=str(folder/f'{pts}_nodes.npz'),nodes_sha256=digest(folder/f'{pts}_nodes.npz'),
                        navigation_supported=supported,navigation_status=nav_reason,
                        incoming_navigation_supported=bool(graph['supported'][graph['index'][pts]]) if graph is not None else None))
                    seeded,attempts=match_anchor_fragments(references,pts,measured,fragments,source_bgr_sha256,supported)
                    anchors.extend(seeded);anchor_attempts.extend(attempts)
                    if previous is not None:
                        h,reason=navigation_between(graph,previous['pts'],pts)
                        edges=[] if h is None else link_fragments(previous['rgb'],rgb,previous['nodes'],nodes,previous['mask'],mask,h)
                        edge_sets.append(edges)
                        links.write(json.dumps(plain(dict(previous_pts=previous['pts'],current_pts=pts,
                            navigation_status=reason,navigation_H=None if h is None else h,edges=edges,
                            source_graph_sha256=graph_note.get('sha256'),native_or_H_UV_used_as_measurement=False)))+'\n')
                    previous=dict(pts=pts,rgb=rgb,nodes=nodes,mask=mask)
                mark('anchor_link_s')
                if args.pts or pts in representatives:
                    grid=image.copy();proof=image.copy()
                    for path in paths:
                        uv=curves[path['native_path_id']]['uv'];valid=np.isfinite(uv).all(1)
                        valid&=(uv>[-1920,-1080]).all(1)&(uv<[3840,2160]).all(1)
                        for a,b in np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2):
                            if b-a>1:cv2.polylines(grid,[np.rint(uv[a:b]).astype(np.int32)],False,(0,230,230),1,cv2.LINE_AA)
                        visible=uv[valid&(uv>[0,0]).all(1)&(uv<[1920,1080]).all(1)]
                        if len(visible):cv2.putText(grid,path['curve_id'],tuple(np.rint(visible[len(visible)//2]).astype(int)),0,.45,(0,180,180),1,cv2.LINE_AA)
                    for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,230,20),-1)
                    for fragment in fragments:
                        ids=fragment['candidate_ids'];xy=np.rint(measured['candidate_uv'][ids[len(ids)//2]]).astype(int)
                        cv2.putText(proof,paths[fragment['curve']]['curve_id'],tuple(xy+[0,-8]),0,.5,(0,160,10),1,cv2.LINE_AA)
                    cv2.imwrite(str(folder/f'{pts}_native_search.jpg'),grid);cv2.imwrite(str(folder/f'{pts}_source_candidates.jpg'),proof)
                mark('proof_serialization_s')
                for stage,value in timing.items():stage_totals[stage]+=value
                timing_log.write(json.dumps(dict(pts=pts,**timing))+'\n')
                total+=len(rows);frame_count+=1
                print(json.dumps(dict(shot=sid,pts=pts,frames=frame_count,rows=total,current_rows=len(rows),fragments=len(fragments),conflicts=len(conflicts),elapsed_s=time.monotonic()-started)),flush=True)
        coverage=dict(frames=frame_count,rows=total,by_path=counts,fit_eligible=False,timings_s=dict(stage_totals),
            searched_path_ids=[paths[i]['native_path_id'] for i in active_indices],
            unsearched_path_ids=[p['native_path_id'] for i,p in enumerate(paths) if i not in active_indices],
            partial_native_path_search=len(active_indices)!=len(paths),all_native_features_covered=False)
        if anchor_capture:
            write(folder/'anchor_matches.json',dict(attempts=anchor_attempts,anchors=anchors,anchor_approval=approval))
            for path,sha in {**approval['inputs'],**search_binding['inputs']}.items():
                if digest(path)!=sha:raise ValueError('Current source reference changed before qualification: '+path)
            provenance=dict(source_graph=graph_note,anchor_approval=approval,source_search_capture=search_binding,
                            source_video_sha256=index['source']['actual_sha256'])
            finalized=time.perf_counter()
            coverage.update(finalize_identity_capture(folder,frame_graph,edge_sets,anchors,paths,provenance))
            coverage['identity_finalize_s']=time.perf_counter()-finalized
            if graph_note.get('sha256') and digest(graph_note['path'])!=graph_note['sha256']:raise ValueError('Source graph changed')
        write(folder/'coverage.json',coverage)
    for path,sha in {**approval['inputs'],**search_binding['inputs']}.items():
        if digest(path)!=sha:raise ValueError('Approved reference input changed during capture: '+path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','run-root','state','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--library',type=Path);p.add_argument('--shots',nargs='+',default=['s07','s08','s09']);p.add_argument('--pts',type=int,nargs='+');p.add_argument('--stride',type=int,default=1);p.add_argument('--search-radius',type=float,default=40.)
    p.add_argument('--anchor-capture',type=Path,help='Current independent reference capture only')
    p.add_argument('--anchor-approval',type=Path,help='Explicit source-hashed per-shot/PTS/path approval groups')
    p.add_argument('--source-reference-navigation',action='store_true',help='Supported H transfers reference UV to search guides only')
    p.add_argument('--source-search-capture',type=Path,help='Current same-frame pending C UV supplies search only, never identity or eligibility')
    p.add_argument('--anchored-path-search-only',action='store_true',help='Search only this shot approved physical paths; coverage gaps remain explicit')
    p.add_argument('--center-search-radius',type=float,default=128.,help='C search radius in optional source-reference navigation mode')
    capture(p.parse_args())
if __name__=='__main__':main()
