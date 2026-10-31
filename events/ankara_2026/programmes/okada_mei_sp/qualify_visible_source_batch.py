"""Assemble reviewed source families, retaining unremeasured parent observations.

All positive coordinates come from archived current-RGB readings. Replacement
IDs are limited to actually sampled profile footprints of the same path/PTS.
No camera or model residual participates in admission, replacement or density.
"""
from pathlib import Path
from collections import defaultdict,Counter
from contextlib import ExitStack
import argparse,hashlib,json,time
import re
import numpy as np
from rinkspace.path_policy import guard_project_write
from prepare_native_source_update import _source_hash

def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def contour_pieces(rows,gap=24.):
    if not rows:return []
    pieces=[];piece=[]
    for row in rows:
        if piece and np.linalg.norm(np.asarray(row['uv'])-piece[-1]['uv'])>gap:
            pieces.append(piece);piece=[]
        piece.append(row)
    if piece:pieces.append(piece)
    return pieces

def thin(rows,spacing=48.):
    result=[]
    for piece in contour_pieces(rows):
        if len(piece)<=2:result.extend(piece);continue
        uv=np.array([r['uv']for r in piece]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
        count=max(2,int(np.ceil(arc[-1]/spacing))+1)
        chosen=np.unique(np.argmin(abs(arc[:,None]-np.linspace(0,arc[-1],count)[None]),axis=0))
        result.extend(piece[i]for i in chosen)
    # A closed short glyph must keep multiple positions/directions; split
    # fragments already keep their actual endpoints without a per-piece quota.
    if len(result)<min(4,len(rows)):
        for i in np.linspace(0,len(rows)-1,min(4,len(rows))).round().astype(int):
            if not any(r is rows[i]for r in result):result.append(rows[i])
    return result

def supported_length(rows):
    return sum(float(np.linalg.norm(np.diff(np.array([r['uv']for r in piece]),axis=0),axis=1).sum())
               for piece in contour_pieces(rows)if len(piece)>1)

def footprint_members(old_uv,rows):
    """Membership in archived sampled strips, never a curve extrapolation."""
    keep=np.zeros(len(old_uv),bool)
    for row in rows:
        if not row.get('_parent_replacement_allowed',True):continue
        n=np.asarray(row['normal'],float);t=np.array([-n[1],n[0]])
        origin=np.asarray(row.get('search_uv',row['uv']),float)
        delta=old_uv-origin;along=abs(delta@t);across=delta@n
        radius=row['_profile_radius'];half=row['_tangent_half_width']
        keep|=(along<=half)&(across>=-radius)&(across<=radius)
    return keep


def validate_capture(binding,summary,sid,frames):
    schemas={'rinkspace.current_near_blue_cap_capture.v1','rinkspace.current_native_whole_print_RGB_capture.v1'}
    if binding.get('schema')not in schemas:raise ValueError('Unsupported source capture schema')
    if sid not in binding.get('shots',[]):raise ValueError('Shot absent from source capture binding')
    if summary.get('changed_dependencies'):raise ValueError('Source capture reports changed dependencies')
    if summary.get('shots',{}).get(sid,{}).get('frames')!=frames:
        raise ValueError('Source capture completion count differs from full shot')
    return binding['schema']=='rinkspace.current_near_blue_cap_capture.v1'


def validate_frame(record,detail,pt,expected_source,frame_base):
    if isinstance(record.get('pts'),bool)or record.get('pts')!=pt or detail.get('pts')!=pt:
        raise ValueError('Capture exposure ledger differs')
    if record.get('source_video_sha256')!=expected_source:raise ValueError('Capture source video identity differs')
    for key in ('source_BGR_sha256','source_exclusion_sha256','mask_sha256'):
        if not isinstance(record.get(key),str)or re.fullmatch('[0-9a-f]{64}',record[key])is None:
            raise ValueError('Capture needs a source SHA256: '+key)
        if frame_base is not None and record[key]!=frame_base[key]:
            raise ValueError('Source frame/mask identity differs: '+key)
    for component in detail.get('components',[]):
        if component.get('source_BGR_sha256')!=record['source_BGR_sha256']:
            raise ValueError('Near component source pixels differ')


def validate_profile_row(row):
    uv=np.asarray(row.get('uv'),float);normal=np.asarray(row.get('normal'),float)
    if uv.shape!=(2,)or not np.isfinite(uv).all()or np.any(uv<0)or np.any(uv>[1919,1079]):
        raise ValueError('Invalid native current UV')
    if normal.shape!=(2,)or not np.isfinite(normal).all()or not np.isclose(np.linalg.norm(normal),1.,rtol=1e-5,atol=1e-6):
        raise ValueError('Source profile needs a finite unit normal')
    if row.get('kind',1)!=1:raise ValueError('Source contour requires kind=1')
    if 'search_uv'in row:
        anchor=np.asarray(row['search_uv'],float)
        if anchor.shape!=(2,)or not np.isfinite(anchor).all():raise ValueError('Invalid source profile search anchor')

def capture_folders(roots,sid):
    folders=[]
    for root in roots:
        candidates=[root]if(root/'binding.json').is_file()else sorted(p for p in root.iterdir()if p.is_dir()and(p/'binding.json').is_file())
        for p in candidates:
            if (p/sid/'source_rows.jsonl').is_file():folders.append(p)
    return folders

def run(args):
    out=guard_project_write(args.output.resolve())
    if out.exists():raise FileExistsError(out)
    manifest=read(args.prepared/'joint_manifest.json');mapping=read(args.prepared/'mapping.json')
    input_sha=sha(args.prepared/'point_observations.npz')
    if input_sha!=manifest['point_input_sha256']:raise ValueError('Parent input changed')
    points=dict(np.load(args.prepared/'point_observations.npz',allow_pickle=False));out.mkdir(parents=True)
    protected={str(args.prepared/'point_observations.npz'):input_sha}
    protected.update({str(args.prepared/name):sha(args.prepared/name)for name in ('joint_manifest.json','mapping.json')})
    summary={};started=time.monotonic()
    for sid in args.shots:
        shot=next(s for s in manifest['shots']if s['id']==sid);folder=out/sid;folder.mkdir()
        expected_pts=range(shot['first_pts'],shot['stop_pts_exclusive'],256)
        if not len(expected_pts)or expected_pts[-1]!=shot['last_pts']:raise ValueError('Parent shot source clock is inconsistent')
        expected_source=_source_hash(manifest,mapping,sid)
        captures=capture_folders(args.capture_roots,sid)
        if not captures:raise ValueError('No complete capture for '+sid)
        count=Counter();replacement=[];by_path=Counter();frame_coverage=[]
        indices=np.arange(shot['row_begin'],shot['row_stop']);order=indices[np.argsort(points['pts'][indices],kind='stable')]
        p=points['pts'][order];unique,starts=np.unique(p,return_index=True);stops=np.r_[starts[1:],len(order)];parent_by_pts={int(pt):order[a:b]for pt,a,b in zip(unique,starts,stops)}
        with ExitStack()as stack:
            sources=[]
            for capture in captures:
                if not (capture/'summary.json').is_file():raise ValueError('Capture has not completed: '+str(capture))
                binding=read(capture/'binding.json')
                near=validate_capture(binding,read(capture/'summary.json'),sid,len(expected_pts))
                data=capture/sid/'source_rows.jsonl';raw=capture/sid/('readouts.jsonl'if near else'registration.jsonl')
                for dep in [capture/'binding.json',capture/'summary.json',data,raw]:protected[str(dep)]=sha(dep)
                sources.append(dict(root=capture,binding=binding,near=near,data=stack.enter_context(data.open(encoding='utf8')),
                    raw=stack.enter_context(raw.open(encoding='utf8')),path=data,raw_path=raw))
            dest=stack.enter_context((folder/'source_rows.jsonl').open('x',encoding='utf8'));audit=stack.enter_context((folder/'source_support_audit.jsonl').open('x',encoding='utf8'));pending=stack.enter_context((folder/'pending.jsonl').open('x',encoding='utf8'))
            for fi,pt in enumerate(expected_pts):
                candidates=defaultdict(list);frame_base=None;raw_candidates=[];rejected=[]
                for source in sources:
                    data_line=next(source['data'],None);raw_line=next(source['raw'],None)
                    if data_line is None or raw_line is None:raise ValueError('Capture source ledger is incomplete')
                    record=json.loads(data_line);detail=json.loads(raw_line)
                    validate_frame(record,detail,pt,expected_source,frame_base)
                    if frame_base is None:frame_base=record
                    if source['near']and sid not in args.near_reviewed_shots:
                        if record['rows']:rejected.append(dict(capture=str(source['root']),reason='near_board_physical_identity_pending',rows=record['rows']))
                        continue
                    groups=defaultdict(list)
                    notes={(str(n['atlas_index'])+'/'+n['instance']):n for n in detail.get('patterns',[])}
                    if len(notes)!=len(detail.get('patterns',[])):raise ValueError('Duplicate source pattern identities')
                    for ri,original in enumerate(detail['all_actual_readouts']):
                        row=dict(original);pid=row['path_id']
                        validate_profile_row(row)
                        if row.get('current_source_RGB_measured')is not True or row.get('model_UV_synthesized')is True or row.get('model_UV_used_as_measurements')is True:raise ValueError('Only actual current RGB readings allowed')
                        if source['near']:
                            if pid!='board/continuous/top' or row['source_view']!='top_face_only':raise ValueError('Near cap role differs')
                            row['_parent_replacement_allowed']=False
                            row['parent_replacement_scope']='append_only_original_profile_search_anchor_not_archived'
                        else:
                            if any(row['source_profile_flags'].values()):raise ValueError('Unresolved source profile')
                            note=notes[row['source_registration_label']];reg=note['registration']
                            if reg['status']!='source_pattern_registered':raise ValueError('Unregistered pattern supplied positive UV')
                            blur=reg['shared_horizontal_blur_px']
                            if not np.isfinite(blur)or blur<1:raise ValueError('Invalid registered source blur width')
                            if not pid.startswith('print_instance/'+note['instance']+'::'):
                                raise ValueError('Source print instance differs from registration')
                            row['_profile_radius']=max(4.,reg['shared_horizontal_blur_px']*abs(row['normal'][0])/2+3)
                            row['_tangent_half_width']=1.
                            row['_parent_replacement_allowed']=True
                            row['measurement_operator']='registered_whole_pattern_original_RGB_signed_transition_q50'
                        row['source_family_review']=dict(capture_root=str(source['root']),
                            binding_sha256=protected[str(source['root']/'binding.json')],
                            state='root_reviewed_not_independent_pixel_acceptance',
                            physical_role_reviewed_shot=sid if source['near']else None,
                            user_visual_acceptance=False,pixel_accuracy_accepted=False)
                        row['source_readout_locator']=dict(path=str(source['raw_path']),sha256=protected[str(source['raw_path'])],line=fi+1,array='all_actual_readouts',index=ri)
                        row['fit_eligible']=True;row['model_UV_synthesized']=False
                        groups[pid].append(row)
                    for pid,rows in groups.items():candidates[pid].append(dict(rows=rows,source=source,score=supported_length(rows)))
                old_indices=parent_by_pts.get(pt,np.empty(0,dtype=int));incoming=[];audit_groups=[]
                for pid,choices in candidates.items():
                    choice=max(choices,key=lambda c:(c['score'],len(c['rows'])));dense=choice['rows'];selected=thin(dense)
                    local=old_indices[np.array([mapping['path_ids'][int(i)]==pid for i in points['path_indices'][old_indices]],bool)]
                    covered=footprint_members(points['uv'][local],dense)&(points['kind'][local]==1);retired=local[covered]
                    replacement.append(dict(pts=pt,path_id=pid,parent_observation_ids=points['observation_ids'][retired].astype(str).tolist()))
                    for row in selected:
                        if row.pop('_parent_replacement_allowed'):
                            row['sampled_source_profile_radius_px']=row.pop('_profile_radius')
                            row['sampled_source_tangent_half_width_px']=row.pop('_tangent_half_width')
                    incoming.extend(selected);by_path[pid]+=len(selected)
                    audit_groups.append(dict(path_id=pid,selected_capture=str(choice['source']['root']),source_supported_length_px=choice['score'],
                        actual_readouts=len(dense),retained_rows=len(selected),replaced_parent_rows=len(retired),
                        same_group_parent_rows_preserved=int((~covered).sum()),alternative_captures=len(choices)-1))
                    audit_groups[-1]['parent_replacement_scope']=('append_only_no_archived_search_anchor'if choice['source']['near']else'actual_sampled_profile_strips')
                    count['retired_parent_rows']+=len(retired);count['same_group_parent_rows_preserved']+=int((~covered).sum())
                    count['raw_selected_readouts']+=len(dense);count['new_path_pts_groups']+=not len(local)
                output={k:v for k,v in frame_base.items()if k!='rows'};output.update(rows=incoming,fit_eligible=True,
                    review_state='root_reviewed_source_families_not_independent_pixel_acceptance',user_visual_acceptance=False,pixel_accuracy_accepted=False)
                dest.write(json.dumps(output,ensure_ascii=False,allow_nan=False)+'\n');audit.write(json.dumps(dict(pts=pt,groups=audit_groups))+'\n')
                if rejected:pending.write(json.dumps(dict(pts=pt,candidates=rejected))+'\n')
                count['frames']+=1;count['frames_with_rows']+=bool(incoming);count['rows']+=len(incoming)
                frame_coverage.append(dict(pts=pt,rows=len(incoming),paths=len(candidates)))
                if fi%250==0:print(json.dumps(dict(shot=sid,frames=fi+1,rows=count['rows'],seconds=time.monotonic()-started)),flush=True)
            for source in sources:
                if next(source['data'],None)is not None or next(source['raw'],None)is not None:raise ValueError('Capture extends beyond selected shot')
        write(folder/'source_replacement.json',dict(policy='explicit_parent_observation_ids',parent_point_input_sha256=input_sha,groups=replacement))
        write(folder/'coverage.json',dict(counts=dict(count),by_path=dict(by_path),frames=frame_coverage))
        summary[sid]=dict(count)
    changed=[p for p,s in protected.items()if sha(p)!=s]
    if changed:
        write(out/'qualification_failure.json',dict(changed_dependencies=changed,fit_eligible=False,fit_executed=False))
        raise RuntimeError('Source dependency changed during assembly')
    write(out/'qualification_summary.json',dict(shots=summary,parent_point_input_sha256=input_sha,source_dependencies=protected,
        changed_dependencies=changed,near_reviewed_shots=args.near_reviewed_shots,
        qualification='Main-session root-reviewed source families, registered pattern structure and signed actual RGB profiles; near role only for explicitly reviewed shots. Not independent pixel acceptance.',
        replaced_only_explicit_parent_ids_in_measured_same_path_profiles=True,UV_modified=False,
        retained_spacing_px=48.,actual_fragment_endpoints_preserved=True,weights_added=False,
        user_visual_acceptance=False,pixel_accuracy_accepted=False,fit_executed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture-roots',nargs='+',type=Path,required=True);p.add_argument('--shots',nargs='+',required=True)
    p.add_argument('--near-reviewed-shots',nargs='*',default=[]);run(p.parse_args())
