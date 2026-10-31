"""Apply source-shape precision checks to cached, identity-linked RGB readouts."""
from pathlib import Path
import argparse,collections,json,time
import cv2,numpy as np
import run_native_programme as run
from rinkspace.session import material_tracks
from rinkspace.session.material_tracks import qualify_visible_line_fragments


def _qualifier_sha256():
    return run.digest(material_tracks.__file__)


def _row_identity(row):
    """Return only complete, exact candidate/source identities for reuse."""
    if not isinstance(row,dict):return None
    source,identity=row.get('source_reference'),row.get('identity')
    candidate,pts=row.get('candidate'),row.get('pts')
    if not isinstance(source,str) or not source or not isinstance(identity,str) or not identity:return None
    if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) for value in (candidate,pts)):return None
    return source,int(candidate),int(pts),identity


def _exact_row_key(row):
    identity=_row_identity(row)
    if identity is None:return None
    try:uv=np.asarray(row.get('uv',[]))
    except (TypeError,ValueError):return None
    if uv.shape!=(2,) or uv.dtype.kind not in 'iuf' or not np.isfinite(uv).all():return None
    return (*identity,*(float(value).hex() for value in uv))


def _dependency_path(value,packet):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (packet.parent/path).resolve()


def _reusable_decisions(packet,sid):
    """Authenticate an explicit prior source; no RGB or masks are reopened."""
    packet=Path(packet).resolve();data=run.read(packet)
    if data.get('fit_eligible') is not True:raise ValueError('Reusable source is not explicitly fit_eligible')
    if data.get('shot')!=sid:raise ValueError('Reusable source belongs to another shot')
    dependencies={}
    for name in ('source_packet','source_links'):
        path=_dependency_path(data[name],packet);actual=run.digest(path)
        if actual!=data[name+'_sha256']:raise ValueError('Reusable source dependency changed: '+str(path))
        dependencies[name]=dict(path=str(path),sha256=actual)
    try:original=run.read(dependencies['source_packet']['path'])
    except json.JSONDecodeError:original={}
    try:qualification=run.read(dependencies['source_links']['path'])
    except json.JSONDecodeError:qualification={}
    prior_excluded=qualification.get('excluded',[]) if isinstance(qualification,dict) else []
    if not isinstance(prior_excluded,list):prior_excluded=[]
    prior_qualifier=qualification.get('qualifier_sha256') if isinstance(qualification,dict) else None
    current_qualifier=_qualifier_sha256()
    reference=dict(packet=str(packet),packet_sha256=run.digest(packet),dependencies=dependencies,
        prior_excluded_count=len(prior_excluded),ambiguous_decisions_not_reused=0,
        prior_excluded_preserved_in_source_links=True,
        prior_qualifier_sha256=prior_qualifier,current_qualifier_sha256=current_qualifier,
        decisions_reusable=prior_qualifier==current_qualifier,
        decision_reuse_reason='matching_qualifier_sha256' if prior_qualifier==current_qualifier else
            'qualifier_sha256_missing_requalification_required' if prior_qualifier is None else
            'qualifier_sha256_changed_requalification_required')
    if prior_qualifier!=current_qualifier:return {},{},reference
    decisions={};conflicts=set();excluded_records={}
    def remember(key,decision,record=None):
        if key is None or key in conflicts:return
        if key in decisions and decisions[key]!=decision:
            conflicts.add(key);decisions.pop(key);excluded_records.pop(key,None);return
        decisions[key]=decision
        if record is not None:excluded_records[key]=record
    for row in data.get('accepted',[]):remember(_exact_row_key(row),'accepted')
    source_keys=collections.defaultdict(set)
    for row in original.get('accepted',[]) if isinstance(original,dict) else []:
        key=_exact_row_key(row)
        if key is not None:source_keys[key[:4]].add(key)
    for row in prior_excluded:
        key=_exact_row_key(row)
        if key is None:
            if not isinstance(row,dict) or 'uv' in row:continue
            possible=source_keys.get(_row_identity(row),set())
            if len(possible)!=1:continue
            key=next(iter(possible))
        remember(key,'excluded',row)
    reference['ambiguous_decisions_not_reused']=len(conflicts)
    return decisions,excluded_records,reference


def qualify(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.monotonic();cv2.setNumThreads(2)
    binding_path=args.source/'binding.json';binding=run.read(binding_path)
    run.write_json(args.output/'binding.json',dict(binding,qualification_parent=dict(
        source=str(args.source.resolve()),binding_path=str(binding_path.resolve()),binding_sha256=run.digest(binding_path))))
    for sid in args.shots:
        packet=args.source/sid/'source_verified.json';data=run.read(packet)
        reuse=getattr(args,'reuse_qualified_source',None);decisions={};old_excluded={};reuse_reference=None
        if reuse is not None:
            reuse=Path(reuse);prior=reuse if reuse.is_file() else reuse/sid/'source_verified.json'
            if prior.is_file():decisions,old_excluded,reuse_reference=_reusable_decisions(prior,sid)
        groups=collections.defaultdict(list)
        for row in data['accepted']:groups[row['source_reference']].append(row)
        pending_pts={row['pts'] for row in data['accepted'] if _exact_row_key(row) not in decisions}
        if reuse is None:_,frames=run.source_frames(sid)
        elif pending_pts:_,frames=run.source_frames(sid,selected_pts=pending_pts)
        else:frames=[]
        frames={f['pts']:f for f in frames}
        retained=[];excluded=[];reused=collections.Counter();checked_frames=set()
        for fi,(source,rows) in enumerate(groups.items()):
            pending=[row for row in rows if _exact_row_key(row) not in decisions];selected=set()
            if pending:
                z=dict(np.load(source));frame=frames[pending[0]['pts']];image=cv2.imread(frame['image']);masks,_=run.surfaces(frame,image)
                fragments,_=qualify_visible_line_fragments(z,list(range(len(binding['inventory']))),~masks['ice'])
                selected={cid for f in fragments for cid in f['candidate_ids']};checked_frames.add(pending[0]['pts'])
            for row in rows:
                key=_exact_row_key(row);decision=decisions.get(key)
                if decision is not None:reused[decision]+=1
                if decision=='accepted' or decision is None and row['candidate'] in selected:retained.append(row)
                elif decision=='excluded':excluded.append(dict(old_excluded[key]))
                else:excluded.append(dict(pts=row['pts'],candidate=row['candidate'],source_reference=source,identity=row['identity']))
            if fi%50==0:print(json.dumps(dict(shot=sid,frames=fi+1,total=len(groups),retained=len(retained),excluded=len(excluded),seconds=time.monotonic()-started)),flush=True)
        folder=args.output/sid;folder.mkdir()
        receipt=folder/'source_qualification.json'
        run.write_json(receipt,dict(parent_packet=str(packet.resolve()),parent_packet_sha256=run.digest(packet),
            inherited_links=str(_dependency_path(data['source_links'],packet)),inherited_links_sha256=data['source_links_sha256'],
            qualifier_sha256=_qualifier_sha256(),
            center_role='native-curvature-rectified source line; midpoint tolerance independent of boundary blur width',
            old_rows_preserved_in_parent=True,source_material_profiles_reread=False,source_images_read_for_masks=bool(checked_frames),
            source_frames_checked=len(checked_frames),reused_accepted_rows=reused['accepted'],reused_excluded_rows=reused['excluded'],
            reuse_qualified_source=reuse_reference,UV_changed=False,excluded=excluded))
        verified=dict(data);verified.update(shot=sid,accepted=retained,
            source_packet=str(_dependency_path(data['source_packet'],packet)),source_links=str(receipt),
            source_links_sha256=run.digest(receipt),fit_eligible=True,user_visual_acceptance=False)
        run.write_json(folder/'source_verified.json',verified)
        grouped=collections.defaultdict(set)
        for row in retained:grouped[row['identity']].add(row['pts'])
        print(json.dumps(dict(shot=sid,retained=len(retained),excluded=len(excluded),frame_coverage={k:len(v)for k,v in grouped.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shots',nargs='+',required=True)
    p.add_argument('--reuse-qualified-source',type=Path,help='Reuse only authenticated exact prior candidate/source/PTS/identity/UV decisions')
    qualify(p.parse_args())
