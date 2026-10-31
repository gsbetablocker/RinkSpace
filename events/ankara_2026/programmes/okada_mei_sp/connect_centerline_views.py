"""Connect cached measured line fragments across shots using actual RGB matches.

Native plane geometry proposes cross-view searches. Bidirectional source profile
matches, competing-fragment checks and anchored components decide connectivity.
No source coordinate is generated or changed by a cross-view transform.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,time
import cv2,numpy as np
import run_native_programme as run
from capture_native_centerlines import plain
from rinkspace.session.material_tracks import source_nodes,link_fragments
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.compat.d78_entities import sample_entities
from rinkspace.session.native_shape_guidance import point_to_partial_polyline


def connect(args):
    if args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic();cv2.setNumThreads(2)
    m=run.read(args.run_root/'joint_manifest.json');v=dict(np.load(args.state))
    binding=run.read(args.source/'binding.json');inventory=binding['inventory'];shots={s['id']:s for s in m['shots']}
    if binding['state_sha256']!=run.digest(args.state):raise ValueError('Search state differs from source capture')
    model=SceneParameterModel(run.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);core=NativeCore(m['native_library'])
    samples=sample_entities(ParameterState(model.initial_state.layout,v['scene_values']).apply_scene(model.scene),core)
    world={c['id']:samples['xyz'][c['begin']:c['end']] for c in samples['cuts']}
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    parent={};records={};frame_index={};frame_raw={};fragments={};seeds=set();points={};links=[]
    def find(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    for sid in binding['shots']:
        _,frames=run.source_frames(sid);frame_index.update({(sid,f['pts']):f for f in frames})
        packet=run.read(args.source/sid/'source_pixels.json');points[sid]=packet['accepted']+packet['pending']
        for r in points[sid]:
            key=(sid,r['pts'],r['fragment']);parent[key]=key;records[key]=r
            frame_raw[(sid,r['pts'])]=Path(r['source_reference'])
        for r in packet['accepted']:seeds.add((sid,r['pts'],r['fragment']))
        edgefile=run.read(args.source/sid/'source_links.json')
        for e in edgefile['edges']:
            if e['unambiguous']:
                a=(sid,e['from_pts'],e['a']);b=(sid,e['to_pts'],e['b'])
                if a in parent and b in parent:parent[find(a)]=find(b)
    grid_x,grid_y=np.meshgrid(np.linspace(-.5,.5,33),np.linspace(-.25,.25,17))
    navigation_grid=np.c_[grid_x.ravel(),grid_y.ravel(),np.zeros(grid_x.size)]
    @lru_cache(maxsize=None)
    def geometry(sid,pts):
        shot=shots[sid];clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot);projected=[];valid=[];pieces=[]
        for p in inventory:
            result=core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,world[p['native_path_id']])
            projected.extend(result['uv']);valid.extend(result['status']==0)
            piece=result['uv'].copy();piece[result['status']!=0]=np.nan;pieces.append(piece)
        result=core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,clock,navigation_grid)
        projected.extend(result['uv']);valid.extend(result['status']==0)
        projected=np.asarray(projected);valid=np.asarray(valid)&np.isfinite(projected).all(1)
        valid&=(projected>[-320,-180]).all(1)&(projected<[2240,1260]).all(1)
        return dict(projection=projected,valid=valid,pieces=pieces)
    @lru_cache(maxsize=24)
    def frame_data(sid,pts):
        frame=frame_index[(sid,pts)];image=cv2.imread(frame['image']);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        masks,_=run.surfaces(frame,image);raw=frame_raw[(sid,pts)]
        z=dict(np.load(raw));parts=run.read(raw.with_suffix('.json'))['fragments'];nodes=source_nodes(rgb,z,parts,inventory)
        return dict(rgb=rgb,mask=~masks['ice'],nodes=nodes,fragments=parts,z=z,**geometry(sid,pts))
    components={}
    for key in parent:components.setdefault(find(key),[]).append(key)
    # One representative per already connected component; longest measured span.
    representatives={}
    for root,members in components.items():
        for key in members:
            raw=Path(records[key]['source_reference'])
            if raw not in fragments:fragments[raw]=run.read(raw.with_suffix('.json'))['fragments']
        representatives[root]=max(members,key=lambda k:fragments[Path(records[k]['source_reference'])][k[2]]['span_px'])
    if args.inherited_links:
        old_binding=run.read(args.inherited_links.parent/'binding.json')
        if old_binding['parent_binding_sha256']!=run.digest(args.source/'binding.json'):
            raise ValueError('Inherited links belong to a different source capture')
        for line in args.inherited_links.read_text().splitlines():
            edge=json.loads(line);ka=tuple(edge['from_fragment']);kb=tuple(edge['to_fragment'])
            if not edge['unambiguous'] or ka not in parent or kb not in parent:continue
            if records[ka]['identity']!=records[kb]['identity']:raise ValueError('Inherited identity conflict')
            parent[find(kb)]=find(ka);links.append(edge)
    anchored={find(k) for k in seeds};by_identity={}
    for root,key in representatives.items():by_identity.setdefault(records[key]['identity'],[]).append(root)
    args.output.mkdir(parents=True)
    run.write_json(args.output/'binding.json',dict(parent_source=str(args.source),parent_binding_sha256=run.digest(args.source/'binding.json'),
        state_sha256=run.digest(args.state),adapter_sha256=run.digest(__file__),RGB_matching_required=True,model_used_only_for_search=True,
        inventory=inventory,shots=binding['shots'],target_shots=args.shots,target_identities=args.identities,
        inherited_links=str(args.inherited_links) if args.inherited_links else None,
        inherited_links_sha256=run.digest(args.inherited_links) if args.inherited_links else None,
        search_order='overlapping native material support; same camera; overlap fraction; temporal distance; measured span'))
    print(json.dumps(dict(stage='source_components',components=len(components),anchored_components=len(anchored),
        inherited_cross_links=len(links),source_frames=len(frame_index),seconds=time.monotonic()-started)),flush=True)
    # Geometric support is a search index only. It is cheap and independent of
    # full RGB frames, so disjoint materials never trigger image matching.
    support={}
    for ri,(root,key) in enumerate(representatives.items()):
        if args.identities and records[key]['identity'] not in args.identities:continue
        raw=frame_raw[key[:2]];part=fragments[raw][key[2]];ci=part['curve']
        curve=geometry(*key[:2])['pieces'][ci];finite=np.isfinite(curve).all(1)
        if len(curve)<2 or not np.any(finite[:-1]&finite[1:]):continue
        with np.load(raw) as z:
            query=z['candidate_query'][part['candidate_ids']];uv=z['query_uv'][query]
        match=point_to_partial_polyline(uv,curve);path_world=world[inventory[ci]['native_path_id']]
        cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path_world,axis=0),axis=1))]
        station=cumulative[match.segment_indices]+match.segment_fractions*np.diff(cumulative)[match.segment_indices]
        spacing=np.median(np.diff(np.unique(station))) if len(np.unique(station))>1 else 0.
        support[root]=dict(low=float(station.min()-spacing),high=float(station.max()+spacing),
            span=part['span_px'],camera=shots[key[0]]['camera_index'])
        if len(support)%200==0:print(json.dumps(dict(stage='support_index',indexed_components=len(support),
            visited_components=ri+1,seconds=time.monotonic()-started)),flush=True)
    start=time.monotonic();last_progress=start;attempted=set();rgb_attempts=0;stopped=False
    print(json.dumps(dict(stage='source_matching',indexed_components=len(support),seconds=time.monotonic()-started)),flush=True)
    with (args.output/'connections.jsonl').open('w',encoding='utf8',buffering=1) as log:
      for edge in links:log.write(json.dumps(edge)+'\n')
      changed=True
      while changed:
        changed=False
        for root,target in representatives.items():
            if (args.output/'stop_requested').exists():stopped=True;break
            if find(root) in anchored:continue
            name=records[target]['identity']
            if args.shots and target[0] not in args.shots:continue
            if args.identities and name not in args.identities:continue
            if root not in support:continue
            def overlap(r):
                a,b=support[root],support[r]
                return max(0.,min(a['high'],b['high'])-max(a['low'],b['low']))
            donors=[r for r in by_identity[name] if r in support and find(r) in anchored
                    and representatives[r][:2]!=target[:2] and overlap(r)>0]
            donors.sort(key=lambda r:(support[r]['camera']!=support[root]['camera'],
                -overlap(r)/max(support[root]['high']-support[root]['low'],1e-12),
                abs(representatives[r][1]-target[1]),-support[r]['span']))
            for donor in donors:
                if (args.output/'stop_requested').exists():stopped=True;break
                reference=representatives[donor];pair=(reference[:2],target[:2])
                if pair in attempted:continue
                attempted.add(pair);ga=geometry(*reference[:2]);gb=geometry(*target[:2]);common=ga['valid']&gb['valid']
                if time.monotonic()-last_progress>=10.:
                    print(json.dumps(dict(attempted_frame_pairs=len(attempted),RGB_pairs=rgb_attempts,
                        cross_links=len(links),seconds=time.monotonic()-start)),flush=True);last_progress=time.monotonic()
                if common.sum()<4:continue
                H,_=cv2.findHomography(ga['projection'][common],gb['projection'][common],0)
                if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:continue
                a=frame_data(*reference[:2]);b=frame_data(*target[:2])
                from scipy.spatial import cKDTree
                moved=cv2.perspectiveTransform(a['nodes']['uv'][None].astype(float),H)[0]
                target_rows=b['nodes']['fragment']==target[2]
                donor_rows=a['nodes']['fragment']==reference[2]
                if not donor_rows.any() or not target_rows.any():continue
                possible=cKDTree(moved[donor_rows]).query(b['nodes']['uv'][target_rows])[0]<64.
                if possible.sum()<3:continue
                rgb_attempts+=1
                edges=link_fragments(a['rgb'],b['rgb'],a['nodes'],b['nodes'],a['mask'],b['mask'],H)
                for e in edges:
                    ka=(*reference[:2],e['a']);kb=(*target[:2],e['b'])
                    if not e['unambiguous'] or ka not in parent or kb not in parent:continue
                    if records[ka]['identity']!=records[kb]['identity']:continue
                    ra,rb=find(ka),find(kb)
                    if ra==rb:continue
                    was=ra in anchored or rb in anchored;parent[rb]=ra
                    if was:anchored.discard(rb);anchored.add(ra);changed=True
                    record=dict(from_fragment=ka,to_fragment=kb,**e);links.append(record);log.write(json.dumps(plain(record))+'\n')
                if len(attempted)%25==0:
                    print(json.dumps(dict(attempted_frame_pairs=len(attempted),cross_links=len(links),seconds=time.monotonic()-start)),flush=True)
                if find(root) in anchored:break
            if len(attempted) and len(attempted)%100==0:
                print(json.dumps(dict(attempted_frame_pairs=len(attempted),cross_links=len(links),seconds=time.monotonic()-start)),flush=True)
        print(json.dumps(dict(attempted_frame_pairs=len(attempted),cross_links=len(links),seconds=time.monotonic()-start)),flush=True)
        if stopped:break
    for sid,rows in points.items():
        selected=[r for r in rows if find((sid,r['pts'],r['fragment'])) in anchored]
        folder=args.output/sid;folder.mkdir();run.write_json(folder/'source_pixels.json',dict(shot=sid,accepted=selected,source_UV_changed=False))
        run.write_json(folder/'source_verified.json',dict(shot=sid,accepted=selected,source_packet=str(folder/'source_pixels.json'),source_packet_sha256=run.digest(folder/'source_pixels.json'),
            source_links=str(args.output/'connections.jsonl'),source_links_sha256=run.digest(args.output/'connections.jsonl'),
            inherited_source=str(args.source/sid/'source_links.json'),inherited_source_sha256=run.digest(args.source/sid/'source_links.json'),
            fit_eligible=True,user_visual_acceptance=False))
        print(json.dumps(dict(shot=sid,accepted_rows=len(selected))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inherited-links',type=Path)
    parser.add_argument('--shots',nargs='+')
    parser.add_argument('--identities',nargs='+')
    connect(parser.parse_args())
