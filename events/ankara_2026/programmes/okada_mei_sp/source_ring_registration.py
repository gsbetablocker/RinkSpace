"""Propagate verified ring interfaces through the complete actual-source graph."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from rinkspace.session.material_tracks import propagate_identities
from rinkspace.session.material_ring_qualification import matched_complete_ring_consensus


def complete_stacks(args):
    import run_native_programme as run
    from rinkspace.session import material_ring_qualification
    root=args.run_root/'registration/ring_materials_v1'
    for shot in args.shots:
        source=root/args.link_name/shot/'source_verified.json';packet=run.read(source)
        grouped=defaultdict(list)
        for row in packet['accepted']:grouped[(row['pts'],row['identity'].split(':')[0])].append(row)
        qualified=[];reports=[]
        for (pts,house),rows in grouped.items():
            hypotheses=run.read(root/shot/f'{pts}.json')['houses'].get(house)
            if hypotheses is None:continue
            matched=defaultdict(set)
            for row in rows:matched[row['identity'].split(':')[1]].add(row['candidate'])
            result=matched_complete_ring_consensus(hypotheses['hypotheses'],hypotheses['stack'],matched)
            qualified.extend(row for row in rows if row['candidate'] in result['selected'].get(row['identity'].split(':')[1],[]))
            reports.append(dict(pts=pts,house=house,**result))
        target=source.parent/'complete_stack_consensus.json'
        report=dict(shot=shot,qualified=qualified,frames=reports,source_packet=str(source),source_sha256=run.digest(source),
            qualification_code_sha256=run.digest(material_ring_qualification.__file__),fit_observations_changed=False)
        if target.exists() and run.read(target)!=report:raise ValueError('Source consensus cache differs; choose another source version')
        run.write_json(target,report)
        print(json.dumps(dict(shot=shot,matched_rows=len(packet['accepted']),qualified_rows=len(qualified),
            qualified_frames=len({r['pts'] for r in qualified}))),flush=True)


def propagate(args):
    import run_native_programme as run
    root=args.run_root
    inventory=run.read(root/'registration/materials/inventory.json')
    paths={p['curve_id']:p['native_path_id'] for p in inventory}
    for shot in args.shots:
        atlas=root/'registration/materials'/shot/args.atlas_name
        source_packet=root/'registration/ring_materials_v1'/args.link_name/shot/'complete_stack_consensus.json'
        packet=run.read(source_packet);graph=run.read(atlas/'graph.json')
        grouped=defaultdict(list)
        for row in packet['qualified']:grouped[row['pts']].append(row)
        anchors=[]
        for pts,rows in grouped.items():
            with np.load(atlas/f'{pts}.npz') as z:nodes=dict(z)
            by_candidate={int(c):i for i,c in enumerate(nodes['ids'])}
            candidates=defaultdict(set)
            for row in rows:
                ni=by_candidate.get(row['candidate'])
                if ni is not None:candidates[(row['identity'],int(nodes['fragment'][ni]))].add(row['candidate'])
            for (identity,fragment),ids in candidates.items():
                ni=[by_candidate[i] for i in ids]
                if len(ids)<3 or np.linalg.norm(np.ptp(nodes['uv'][ni],axis=0))<56.:continue
                anchors.append(dict(pts=pts,fragment=fragment,identity=identity,observed_candidate_ids=sorted(ids),
                    source_reference=str(source_packet),model_UV_as_measurements=False))
        labels=propagate_identities(graph['frames'],graph['edges'],anchors)
        by_pts=defaultdict(list)
        for label in labels:
            if label['status']=='anchored':by_pts[label['pts']].append(label)
        accepted=[];frame_reports=[]
        _,frames=run.source_frames(shot);sources={f['pts']:f for f in frames}
        for pts,local in sorted(by_pts.items()):
            with np.load(atlas/f'{pts}.npz') as z:nodes=dict(z)
            with np.load(root/'registration/materials'/shot/f'{pts}.npz') as z:measured=dict(z)
            by_house=defaultdict(lambda:defaultdict(set))
            for label in local:
                identity=label['identities'][0];house,role=identity.split(':')
                selected=set()
                for ai,proof in label['proofs'].items():
                    if proof is None:selected.update(anchors[int(ai)]['observed_candidate_ids']);continue
                    edge_index=proof['edge_frame']
                    if not isinstance(edge_index,int):raise ValueError('Unexpected source link')
                    edge=graph['edges'][edge_index][proof['edge']]
                    axis=0 if label['frame']==edge_index else 1
                    ni=np.array([p[axis] for p in edge['pairs']],int)
                    if not np.all(nodes['fragment'][ni]==label['fragment']):raise ValueError('Source fragment changed')
                    selected.update(map(int,nodes['ids'][ni]))
                by_house[house][role].update(selected)
            source=run.read(root/'registration/ring_materials_v1'/shot/f'{pts}.json')
            for house,matched in by_house.items():
                if house not in source['houses']:continue
                candidate=source['houses'][house]
                result=matched_complete_ring_consensus(candidate['hypotheses'],candidate['stack'],matched)
                retained=0
                for role,ids in result['selected'].items():
                    for cid in ids:
                        quantiles=measured['candidate_side_quantiles_uv'][cid]
                        uv=quantiles[:,1].mean(0)
                        accepted.append(dict(pts=pts,identity=f'{house}:{role}',path_id=paths[f'{house}:{role}'],candidate=cid,
                            uv=uv.tolist(),actual_source_quantiles=quantiles.tolist(),evidence_image=sources[pts]['image'],
                            source_BGR_sha256=sources[pts]['source_BGR_sha256'],source_anchor_packet=str(source_packet),
                            source_graph=str(atlas/'graph.json'),whole_source_stack_consensus=True,
                            actual_bidirectional_temporal_material_correspondence=True,source_UV_synthesized=False,
                            physical_instance_inference='cross_view_source_reference_not_new_human_acceptance'))
                        retained+=1
                frame_reports.append(dict(pts=pts,house=house,matched={r:len(ids) for r,ids in matched.items()},retained=retained,
                    status=result['status'],plausible=result['plausible']))
        out=(args.output or root/'registration/ring_materials_v1/linked_temporal')/shot;out.mkdir(parents=True,exist_ok=True)
        if (out/'source_verified.json').exists():raise FileExistsError('Preserve the existing source-graph result and choose a new output')
        run.write_json(out/'identity_graph.json',dict(anchors=anchors,labels=labels,source_graph_sha256=run.digest(atlas/'graph.json'),
            source_packet_sha256=run.digest(source_packet)))
        run.write_json(out/'source_verified.json',dict(shot=shot,accepted=accepted,frames=frame_reports,
            source_reference=str(source_packet),source_reference_sha256=run.digest(source_packet),
            source_graph=str(atlas/'graph.json'),source_graph_sha256=run.digest(atlas/'graph.json'),
            composite_pixels_used_as_measurements=False,model_pixels_used_as_measurements=False,
            fit_observations_changed=False,user_visual_acceptance=False))
        print(json.dumps(dict(shot=shot,seed_readouts=len(packet['qualified']),seed_anchors=len(anchors),
            seed_frames=len(grouped),propagated_frames=len(by_pts),qualified_frames=len({r['pts'] for r in accepted}),
            qualified_readouts=len(accepted),identity_conflicts=sum(r['status']=='identity_conflict' for r in labels))),flush=True)
