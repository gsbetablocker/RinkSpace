"""Current RGB material references located by preserved human source ranges.

The hand drawing carries identity and a search interval. Only fresh current
RGB q50 readings become candidate observations; drawings never supply UV.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from rinkspace.session.material_tracks import assemble_fragments, source_nodes, qualify_reference_links
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader


def _digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _trace(record):
    scope = record['full_drawn_scope']
    ref = scope['reference']
    if _digest(ref['path']) != ref['sha256']:
        raise ValueError('Protected human reference changed')
    if 'source_trace_arrays' in scope:
        arrays = scope['source_trace_arrays']
        if _digest(arrays['path']) != arrays['sha256']:
            raise ValueError('Protected human trace changed')
        return np.asarray(np.load(arrays['path'])['uv'], float)
    return np.asarray(_read(ref['path'])[scope['native_coordinates_field']], float)


def register_source_reference(old, current):
    """Verify existing clock correspondence using real photo features."""
    sift = cv2.SIFT_create(nfeatures=3000)
    k0, d0 = sift.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(d0, d1, k=2)
    good = [m for m, n in matches if m.distance < .7*n.distance]
    before = np.array([k0[m.queryIdx].pt for m in good])
    after = np.array([k1[m.trainIdx].pt for m in good])
    affine, inside = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=1.)
    if affine is None or int(inside.sum()) < 30:
        raise ValueError('Old source image cannot be independently registered')
    error = np.linalg.norm(cv2.transform(before[:, None], affine)[:, 0]-after, axis=1)
    p95 = float(np.percentile(error[inside.ravel().astype(bool)], 95))
    return affine, dict(method='SIFT_photo_similarity_RANSAC', matches=len(good),
        inliers=int(inside.sum()), p95_px=p95, current_from_original=affine.tolist(),
        native_projection_used=False)


def prepare(args):
    """Read the full preserved scope, retaining unresolved competing branches."""
    import run_native_programme as run
    cv2.setNumThreads(2)
    root = args.run_root
    protected = root/'source_material_checks/protected_manual_sources.json'
    report = _read(protected)
    material_root = run.material_directory(args)
    inventory = _read(material_root/'inventory.json')
    paths = [dict(p, xyz=np.zeros((2, 3))) for p in inventory]
    by_path = {p['native_path_id']: i for i, p in enumerate(paths)}
    groups = defaultdict(list)
    for record in report['protected_human_ranges']:
        if record['current_path_correspondence'] not in by_path:continue
        groups[(record['mapped_official50']['shot'], record['mapped_official50']['pts'])].append(record)
    out = args.output or root/'registration/manual_materials'
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(protected_source_report=str(protected), protected_source_sha256=_digest(protected),
        source_UV_from_current_RGB_only=True, hand_drawings_used_for_identity_and_search_range=True,
        exact_human_pixel_acceptance=False, input_observations_changed=False)
    receipt['material_inventory'] = dict(path=str(material_root/'inventory.json'),sha256=_digest(material_root/'inventory.json'))
    _write(out/'identity.json', receipt)
    summaries = []
    for shot_id in sorted({key[0] for key in groups}):
        visibility, frames = run.source_frames(shot_id)
        by_pts = {f['pts']: f for f in frames}
        for (shot, pts), records in groups.items():
            if shot != shot_id:
                continue
            frame = by_pts[pts]
            image = cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest() != frame['source_BGR_sha256']:
                raise ValueError('Current source image changed')
            old = cv2.imread(records[0]['original_source']['source_frame'])
            affine, registration = register_source_reference(old, image)
            # Empty paths keep the global measurement-family numbering stable.
            projections = {p['native_path_id']: dict(uv=np.empty((0, 2)), valid=np.empty(0, bool)) for p in paths}
            local_paths = [dict(p) for p in paths]
            traces, limits, record_by_curve = {}, {}, {}
            for record in records:
                ci = by_path[record['current_path_correspondence']]
                trace = cv2.transform(_trace(record)[:, None], affine)[:, 0]
                traces[ci] = trace
                limits[ci] = max(abs(float(x)) for x in record['uncertainty']['normal_interval_min_max_px'])+registration['p95_px']
                record_by_curve[ci] = record
                # Smoothing defines search normals, never changes the source.
                smooth = gaussian_filter1d(trace, 5., axis=0, mode='nearest')
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
                query_arc = np.r_[np.arange(0., arc[-1], 14.), arc[-1]]
                query = np.c_[np.interp(query_arc, arc, smooth[:, 0]), np.interp(query_arc, arc, smooth[:, 1])]
                projection = dict(uv=query, valid=np.ones(len(query), bool))
                if local_paths[ci]['operator'] == 'material_interface':
                    # Drawing conic determines outward normal sign only.
                    ellipse = cv2.fitEllipse(trace.astype(np.float32))
                    local_paths[ci]['center_xyz'] = np.zeros(3)
                    projection['centre_uv'] = np.asarray(ellipse[0])
                projections[local_paths[ci]['native_path_id']] = projection
            reader = IceProgrammeSourceReader(None, source_sha256=visibility['source_video_sha256'],
                time_base='1/12800', sample_spacing_px=28., search_radius_px=48., native_paths=local_paths,
                prepared_source_context=dict(source_sha256=visibility['source_video_sha256'],time_base='1/12800',
                    authority=[dict(path=str(protected),sha256=_digest(protected)),receipt['material_inventory']])
                    if getattr(args,'material_root',None) is not None else None)
            masks, _ = run.surfaces(frame, image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            measured = reader.read(rgb, pts, None, ~masks['ice'], shot_id=shot, native_projections=projections)
            flags = measured['candidate_flags'].copy()
            distance = np.full(len(flags), np.inf)
            curve = measured['query_curve'][measured['candidate_query']]
            for ci, trace in traces.items():
                ids = np.flatnonzero(curve == ci)
                distance[ids] = cKDTree(trace).query(measured['candidate_uv'][ids])[0]
                flags[ids[distance[ids] > limits[ci]]] |= 2
            filtered = dict(measured, candidate_flags=flags)
            fragments = assemble_fragments(filtered, inventory)
            # Multiple branches sharing the same query range remain pending.
            for fi, fragment in enumerate(fragments):
                fragment['reference_id'] = record_by_curve[fragment['curve']]['scope_id']
                fragment['identity'] = inventory[fragment['curve']]['curve_id']
                own = set(measured['candidate_query'][fragment['candidate_ids']].tolist())
                rivals = []
                for fj, other in enumerate(fragments):
                    if fi == fj or fragment['curve'] != other['curve']:
                        continue
                    theirs = set(measured['candidate_query'][other['candidate_ids']].tolist())
                    if len(own & theirs) >= 3:
                        rivals.append(fj)
                fragment['competing_fragments'] = rivals
                fragment['source_branch_unique_in_human_range'] = not rivals
            nodes = source_nodes(rgb, measured, fragments, inventory)
            folder = out/shot/str(pts)
            folder.mkdir(parents=True, exist_ok=True)
            arrays = {k: v for k, v in measured.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(folder/'readouts.npz', **arrays, manual_range_distance=distance, range_filtered_flags=flags)
            np.savez_compressed(folder/'nodes.npz', **nodes)
            metadata = dict(shot=shot, pts=pts, source_frame=frame['image'], source_BGR_sha256=frame['source_BGR_sha256'],
                registration=registration, fragments=fragments, source_references=records,
                user_pixel_acceptance=False, current_RGB_measured=True, measurements_admitted=False,
                manual_range_px={str(k): v for k, v in limits.items()})
            _write(folder/'references.json', metadata)
            proof = image.copy()
            for ci, trace in traces.items():
                visible = (trace >= 0).all(1)&(trace < [image.shape[1], image.shape[0]]).all(1)
                for p in trace[visible][::8]: cv2.circle(proof, tuple(np.rint(p).astype(int)), 1, (0, 180, 180), -1)
            for fi, fragment in enumerate(fragments):
                color = (0, 255, 0) if fragment['source_branch_unique_in_human_range'] else (0, 80, 255)
                uv = measured['candidate_uv'][fragment['candidate_ids']]
                for p in uv: cv2.circle(proof, tuple(np.rint(p).astype(int)), 3, color, -1)
                at = tuple(np.rint(uv[len(uv)//2]).astype(int))
                cv2.putText(proof, str(fi), at, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(proof, str(fi), at, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
            cv2.imwrite(str(folder/'source_proof.jpg'), np.vstack([image, proof]), [cv2.IMWRITE_JPEG_QUALITY, 94])
            summary = dict(shot=shot, pts=pts, protected_ranges=len(records), candidates=len(measured['candidate_uv']),
                fragments=len(fragments), unambiguous_fragments=sum(f['source_branch_unique_in_human_range'] for f in fragments),
                report=str(folder/'references.json'), proof=str(folder/'source_proof.jpg'))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    _write(out/'references.json', dict(receipt=receipt, frames=summaries))


def anchor_graph(args):
    """Connect fresh manual-range RGB references to the complete source graph."""
    import run_native_programme as run
    from rinkspace.session.material_tracks import link_fragments, propagate_identities
    cv2.setNumThreads(2)
    root = args.run_root
    refs = args.output or root/'registration/manual_materials'
    references=getattr(args,'reference_directory',None) or refs
    refs.mkdir(parents=True,exist_ok=True)
    reference_summary = _read(references/'references.json')
    all_reports = []
    for shot in sorted({r['shot'] for r in reference_summary['frames']}):
        _, frames = run.source_frames(shot)
        by_pts = {f['pts']: f for f in frames}
        atlas = run.material_directory(args)/shot/args.atlas_name
        graph_path=atlas/getattr(args,'atlas_graph_name','graph.json')
        graph = _read(graph_path)
        anchors = []
        attachments = []
        for summary in reference_summary['frames']:
            if summary['shot'] != shot:
                continue
            pts = summary['pts']
            folder = references/shot/str(pts)
            info = _read(folder/'references.json')
            with np.load(folder/'nodes.npz') as z:
                reference = dict(z)
            chosen = [i for i, f in enumerate(info['fragments']) if f['source_branch_unique_in_human_range']]
            # Different source fragments of the same human-identified curve
            # support one identity; they are not competing physical instances.
            identities = sorted({info['fragments'][i]['identity'] for i in chosen})
            keep = np.isin(reference['fragment'], chosen)
            old = {k: v[keep] for k, v in reference.items()}
            old['fragment'] = np.array([identities.index(info['fragments'][i]['identity']) for i in old['fragment']], np.int32)
            frame = by_pts[pts]
            image = cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest() != frame['source_BGR_sha256']:
                raise ValueError('Current source changed')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks, _ = run.surfaces(frame, image)
            with np.load(atlas/f'{pts}.npz') as z:
                now = dict(z)
            links = link_fragments(rgb, rgb, old, now, ~masks['ice'], ~masks['ice'], np.eye(3))
            for edge in links:
                if edge['unambiguous']:
                    anchors.append(dict(pts=pts, fragment=edge['b'], identity=identities[edge['a']],
                        source_reference=str(folder/'references.json'), current_RGB_verified=True,
                        hand_drawn_UV_used_as_observations=False, link=edge))
            attachment = dict(pts=pts, identities=identities, edges=links, source_reference=summary['report'])
            attachments.append(attachment)
            print(json.dumps(dict(event='manual_reference_graph_attachment',shot=shot,pts=pts,
                available_identities=len(identities),links=len(links),unambiguous=sum(e['unambiguous'] for e in links))),flush=True)
        labels = propagate_identities(graph['frames'], graph['edges'], anchors,graph.get('reference_edges',[]))
        accepted = [r for r in labels if r['status'] == 'anchored']
        report = dict(shot=shot, source_graph=str(graph_path), source_graph_sha256=_digest(graph_path),
            references_sha256=_digest(references/'references.json'), anchors=anchors, attachments=attachments,labels=labels,
            anchored_fragments=len(accepted), anchored_frames=len({r['pts'] for r in accepted}),
            identity_conflicts=sum(r['status']=='identity_conflict' for r in labels),
            full_source_exposures=len(frames), fit_observations_changed=False)
        (refs/shot).mkdir(parents=True,exist_ok=True)
        _write(refs/shot/'identity_graph.json', report)
        all_reports.append({k:v for k,v in report.items() if k not in ('anchors','attachments','labels')})
        print(json.dumps(all_reports[-1]),flush=True)
    _write(refs/'identity_graph_summary.json',all_reports)


def direct_links(args):
    """Try independently registered reference RGB against every exposure."""
    import run_native_programme as run
    from rinkspace.session.material_tracks import link_fragments, propagate_identities
    cv2.setNumThreads(2)
    root = args.run_root
    refs = args.output or root/'registration/manual_materials'
    reference_summary = _read(refs/'references.json')
    out = refs/'direct_links'
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    log_path = out/'read.jsonl'
    print('MAIN LOG: '+str(log_path),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        for shot in sorted({r['shot'] for r in reference_summary['frames']}):
            _, frames = run.source_frames(shot)
            by_pts = {f['pts']: i for i, f in enumerate(frames)}
            atlas = run.material_directory(args)/shot/args.atlas_name
            graph = _read(atlas/'graph.json')
            with np.load(run.material_navigation_file(args,shot)) as z:
                projected = dict(z)
            references = []
            for summary in reference_summary['frames']:
                if summary['shot'] != shot: continue
                pts = summary['pts']
                folder = refs/shot/str(pts)
                info = _read(folder/'references.json')
                with np.load(folder/'nodes.npz') as z: nodes = dict(z)
                chosen = [i for i, f in enumerate(info['fragments']) if f['source_branch_unique_in_human_range']]
                identities = sorted({info['fragments'][i]['identity'] for i in chosen})
                keep = np.isin(nodes['fragment'], chosen)
                old = {k: v[keep] for k, v in nodes.items()}
                old['fragment'] = np.array([identities.index(info['fragments'][i]['identity']) for i in old['fragment']], np.int32)
                frame = frames[by_pts[pts]]
                image = cv2.imread(frame['image']);masks,_ = run.surfaces(frame,image)
                references.append(dict(pts=pts,index=by_pts[pts],rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                    mask=~masks['ice'],nodes=old,identities=identities,source_reference=str(folder/'references.json')))
            anchors = []
            folder = out/shot
            folder.mkdir(exist_ok=True)
            for fi, frame in enumerate(frames):
                target = folder/f'{frame["pts"]}.json'
                if target.exists(): record = _read(target)
                else:
                    image = cv2.imread(frame['image'])
                    if hashlib.sha256(image.tobytes()).hexdigest() != frame['source_BGR_sha256']:
                        raise ValueError('Current image changed')
                    masks,_ = run.surfaces(frame,image);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    with np.load(atlas/f'{frame["pts"]}.npz') as z: now=dict(z)
                    record = dict(pts=frame['pts'],matches=[])
                    for reference in references:
                        ri=reference['index']
                        pa,pb=projected['uv'][ri],projected['uv'][fi]
                        visible=(projected['status'][ri]==0)&(projected['status'][fi]==0)
                        visible &= np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
                        for xy in (pa,pb):
                            visible &= (xy[:,0]>-150)&(xy[:,0]<2070)&(xy[:,1]>100)&(xy[:,1]<1230)
                        if visible.sum()<12: continue
                        h,_=cv2.findHomography(pa[visible],pb[visible],0)
                        edges=link_fragments(reference['rgb'],rgb,reference['nodes'],now,reference['mask'],~masks['ice'],h)
                        qualify_reference_links(edges,reference['identities'])
                        record['matches'].append(dict(reference_pts=reference['pts'],identities=reference['identities'],
                            source_reference=reference['source_reference'],edges=edges,initial_h=h.tolist(),
                            native_map_supplies_no_measurement=True))
                    _write(target,record)
                for matched in record['matches']:
                    for edge in matched['edges']:
                        if edge['identity_unambiguous']:
                            anchors.append(dict(pts=frame['pts'],fragment=edge['b'],identity=matched['identities'][edge['a']],
                                source_reference=matched['source_reference'],reference_pts=matched['reference_pts'],
                                evidence_link=str(target),current_RGB_verified=True,link=edge,
                                hand_drawn_UV_used_as_observations=False))
                if fi%50==0 or fi==len(frames)-1:
                    progress=dict(event='manual_reference_direct_links',shot=shot,frames=fi+1,total=len(frames),
                        reference_links=len(anchors),directly_supported_frames=len({a['pts'] for a in anchors}),elapsed_s=time.monotonic()-started)
                    line=json.dumps(progress);log.write(line+'\n');print(line,flush=True)
            labels=propagate_identities(graph['frames'],graph['edges'],anchors)
            supported=[r for r in labels if r['status']=='anchored']
            report=dict(shot=shot,source_graph=str(atlas/'graph.json'),source_graph_sha256=_digest(atlas/'graph.json'),
                anchors=anchors,labels=labels,anchored_fragments=len(supported),anchored_frames=len({r['pts'] for r in supported}),
                identity_conflicts=sum(r['status']=='identity_conflict' for r in labels),full_source_exposures=len(frames),
                fit_observations_changed=False,current_RGB_only=True,manual_scope_protected=True)
            _write(folder/'identity_graph.json',report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('anchors','labels')}),flush=True)


def export_source_readouts(args):
    """Export actual matched nodes; retain whole unresolved material families."""
    import run_native_programme as run
    root=args.run_root
    refs=args.output or root/'registration/manual_materials'
    material_root=run.material_directory(args)
    inventory=_read(material_root/'inventory.json')
    paths={p['curve_id']:p['native_path_id'] for p in inventory}
    # These reference materials cannot yet be separated from reflection in
    # the actual source image. Preserve the complete families for further work.
    pending={'R1:outer':'Reference RGB cannot distinguish this branch from reflection near the skater.',
             'house_connection_F1_R1:segment0':'Reference RGB does not independently resolve the thin line from vertical reflection.'}
    for shot in args.shots:
        report_path=getattr(args,'source_identity_graph',None) or refs/'direct_links'/shot/'identity_graph.json'
        if not report_path.exists(): continue
        report=_read(report_path)
        if report['shot']!=shot:raise ValueError('Source identity graph belongs to another shot')
        if _digest(report['source_graph'])!=report['source_graph_sha256']:
            raise ValueError('Source material graph changed after identity qualification')
        graph=_read(report['source_graph'])
        atlas=Path(report['source_graph']).parent
        _,frames=run.source_frames(shot)
        by_pts={f['pts']:f for f in frames}
        groups=defaultdict(list)
        for label in report['labels']:
            if label['status']=='anchored':groups[label['pts']].append(label)
        accepted=[];pending_rows=[];counts=[]
        for pts,labels in sorted(groups.items()):
            with np.load(atlas/f'{pts}.npz') as z:nodes=dict(z)
            with np.load(material_root/shot/f'{pts}.npz') as z:measured=dict(z)
            frame_rows=[];seen=set()
            for label in labels:
                identity=label['identities'][0]
                selected=set()
                for anchor_index,proof in label['proofs'].items():
                    if proof is None:
                        edge=report['anchors'][int(anchor_index)]['link'];axis=1
                    else:
                        ei=proof['edge_frame']
                        if isinstance(ei,int):
                            edge=graph['edges'][ei][proof['edge']]
                            axis=0 if label['frame']==ei else 1
                        elif isinstance(ei,str) and ei.startswith('reference:'):
                            record=graph['reference_edges'][int(ei.split(':',1)[1])]
                            edge=record['edges'][proof['edge']]
                            if label['frame'] not in (record['reference'],record['current']):
                                raise ValueError('Source proof frame is not an endpoint')
                            axis=0 if label['frame']==record['reference'] else 1
                        else:raise ValueError('Unexpected graph reference-edge kind')
                    selected.update(int(pair[axis]) for pair in edge['pairs'])
                for ni in sorted(selected):
                    if int(nodes['fragment'][ni])!=label['fragment']:raise ValueError('Source proof points to another fragment')
                    candidate=int(nodes['ids'][ni]);key=(identity,candidate)
                    if key in seen:continue
                    seen.add(key)
                    quantiles=measured['candidate_side_quantiles_uv'][candidate]
                    if not np.isfinite(quantiles).all():continue
                    uv=quantiles[:,1].mean(0)
                    row=dict(pts=pts,identity=identity,path_id=paths[identity],candidate=candidate,uv=uv.tolist(),
                        actual_source_quantiles=quantiles.tolist(),evidence_image=by_pts[pts]['image'],
                        source_BGR_sha256=by_pts[pts]['source_BGR_sha256'],source_identity_graph=str(report_path),
                        identity_graph_fragment=label['fragment'],human_identity_scope_used=True,
                        human_precise_pixel_acceptance=False,current_source_RGB_measured=True,
                        all_source_coordinates_from_readouts=True,model_UV_used_as_measurements=False)
                    if identity in pending:pending_rows.append(dict(row,pending_reason=pending[identity]))
                    else:accepted.append(row);frame_rows.append(row)
            counts.append(dict(pts=pts,readouts=len(frame_rows),identities=sorted({r['identity'] for r in frame_rows})))
        output=report_path.parent
        _write(output/'source_verified.json',dict(shot=shot,accepted=accepted,pending=pending_rows,frames=counts,
            source_packet=report['source_graph'],source_packet_sha256=report['source_graph_sha256'],
            source_links=str(report_path),source_links_sha256=_digest(report_path),
            source_graph_sha256=report['source_graph_sha256'],identity_graph_sha256=_digest(report_path),
            identity_review=str(refs/'image_review.json'),
            pending_material_families=pending,native_fit_residual_not_used_for_selection=True,
            reference_pixel_uncertainty_remains=True,user_visual_acceptance=False,fit_observations_changed=False))
        proofs=output/'source_proofs';proofs.mkdir(exist_ok=True)
        chosen=[1014272,1071616,1152000,1198080]
        color_names=sorted({r['identity'] for r in accepted})
        palette=[(0,255,0),(0,128,255),(255,0,255),(255,255,0),(0,255,255),(255,100,50),(50,180,255),(255,0,0)]
        for pts in chosen:
            rows=[r for r in accepted if r['pts']==pts]
            if not rows:continue
            source=cv2.imread(by_pts[pts]['image']);annotated=source.copy()
            for r in rows:
                col=palette[color_names.index(r['identity'])%len(palette)]
                cv2.circle(annotated,tuple(np.rint(r['uv']).astype(int)),3,col,-1)
            y=35
            for name in sorted({r['identity'] for r in rows}):
                col=palette[color_names.index(name)%len(palette)]
                cv2.putText(annotated,name,(1150,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3)
                cv2.putText(annotated,name,(1150,y),cv2.FONT_HERSHEY_SIMPLEX,.7,col,1);y+=30
            cv2.imwrite(str(proofs/f'{pts}.jpg'),np.vstack([source,annotated]),[cv2.IMWRITE_JPEG_QUALITY,95])
        print(json.dumps(dict(event='manual_material_source_packet',shot=shot,readouts=len(accepted),
            supported_frames=len({r['pts'] for r in accepted}),pending_rows=len(pending_rows),
            identities={n:sum(r['identity']==n for r in accepted) for n in color_names})),flush=True)


def prepare_revision(args):
    """Make one preserved whole-material revision for common native validation."""
    import os
    import run_native_programme as run
    import apply_native_multicamera_review as review
    root=args.run_root
    refs=args.output or root/'registration/manual_materials'
    seed=root/'whole_material_seed.npz'
    if seed.exists():raise FileExistsError('Whole-material candidate already prepared')
    manifest=_read(root/'joint_manifest.json');mapping=_read(root/'mapping.json')
    before=_digest(root/'point_observations.npz')
    if before!=manifest['point_input_sha256']:raise ValueError('Native input changed')
    protected=_read(root/'source_material_checks/protected_manual_sources.json')
    if protected['protected_exact_or_fixed_current_UV']:
        raise ValueError('Explicit human pixel observations require individual preservation')
    with np.load(root/'point_observations.npz',allow_pickle=False) as z:original=dict(z)
    ice_paths={p['native_path_id'] for p in _read(root/'registration/materials/inventory.json')}
    ice_indices=[i for i,p in enumerate(mapping['path_ids']) if p in ice_paths]
    decisions=[];pending_by_shot={}
    for shot in manifest['shots']:
        if shot['id'] not in ('s03','s05'):continue
        _,frames=run.source_frames(shot['id']);images={f['pts']:f['image'] for f in frames}
        rows=np.arange(shot['row_begin'],shot['row_stop'])
        rows=rows[np.isin(original['path_indices'][rows],ice_indices)]
        pending_by_shot[shot['id']]=len(rows)
        for i in rows:
            decisions.append(dict(observation_id=str(original['observation_ids'][i]),pts=int(original['pts'][i]),
                expected_path_id=mapping['path_ids'][original['path_indices'][i]],uv=original['uv'][i].tolist(),
                decision='exclude_preserving_source',evidence_image=images[int(original['pts'][i])],
                reason='Whole S03/S05 machine ice package is retained pending source-material identity re-registration. Replaced by a current RGB material graph where established; uncovered entities and exposures remain explicitly missing. Not selected by fitting residual.'))
    sources=[('s03',root/'registration/ring_materials_v1/linked_cross_view/s03/complete_stack_consensus.json','qualified'),
             ('s05',refs/'direct_links/s05/source_verified.json','accepted')]
    additions=[];packets=[];seen=set()
    for shot,path,field in sources:
        report=_read(path);packets.append(dict(path=str(path),sha256=_digest(path),rows=len(report[field])))
        for row in report[field]:
            key=(row['pts'],row['path_id'],*row['uv'])
            if key in seen:continue
            seen.add(key)
            additions.append(dict(row,observation_id=f'whole_material:{shot}:{row["pts"]}:{mapping["path_ids"].index(row["path_id"])}:{row["candidate"]}',kind=1,
                reason='Current RGB gradient readout connected to a preserved human material scope or complete cross-view ring stack by actual bidirectional source profiles. Numeric joint validation remains pending; this is not human pixel acceptance.'))
    if not additions:raise ValueError('No current source material available')
    archive=root/'input_before_whole_material';archive.mkdir()
    os.link(root/'point_observations.npz',archive/'point_observations.npz')
    _write(archive/'joint_manifest.json',manifest);_write(archive/'mapping.json',mapping)
    coverage={}
    for shot in manifest['shots']:
        if shot['id'] not in pending_by_shot:continue
        selected=[r for r in additions if shot['first_pts']<=r['pts']<=shot['last_pts']]
        have={r['pts'] for r in selected}
        coverage[shot['id']]=dict(readouts=len(selected),frames=len(have),
            missing_ice_source_pts=[pts for pts in range(shot['first_pts'],shot['last_pts']+1,256) if pts not in have],
            identities=sorted({r['identity'] for r in selected}))
    report=dict(input_sha256=before,decisions=decisions,added_observations=additions,source_packets=packets,
        pending_machine_ice_rows=pending_by_shot,protected_human_scope_report=str(root/'source_material_checks/protected_manual_sources.json'),
        protected_human_scopes=protected['protected_human_scope_count'],coverage=coverage,
        purpose='common_geometry_validation_candidate_not_completed_six_shot_result',
        native_fit_residual_used_for_selection=False,old_four_shot_rows_unchanged=True,
        old_source_UV_and_identity_retained=True,source_UV_synthesized=False,uncovered_constraints_not_declared_solved=True)
    path=root/'whole_material_revision.json';_write(path,report)
    review.p.RUN=root
    review.apply(path,root/'source_pending_seed.npz',seed,root/'whole_material_receipt')
    print(json.dumps(dict(event='whole_material_revision_prepared',old_rows=len(original['pts']),pending=pending_by_shot,
        new_readouts=len(additions),new_input_sha256=_digest(root/'point_observations.npz'),seed=str(seed))),flush=True)


def readout_graph(args):
    """Unify verified whole lines with actual point matches through weak frames."""
    import run_native_programme as run
    from functools import lru_cache
    from rinkspace.session.material_tracks import link_fragments,propagate_readout_identities
    cv2.setNumThreads(2)
    material=run.material_directory(args);inventory=_read(material/'inventory.json')
    report_path=args.source_identity_graph
    if report_path is None or args.output is None or not args.material_role_name:
        raise ValueError('Explicit whole-line identity, role revision and new output required')
    strong=_read(report_path);shot=strong['shot']
    if shot not in args.shots:raise ValueError('Source shot is outside the requested sequence')
    strong_path=Path(strong['source_graph']);strong_graph=_read(strong_path)
    if _digest(strong_path)!=strong['source_graph_sha256']:raise ValueError('Whole-line source graph changed')
    raw_path=material/shot/args.atlas_name/'graph.json';raw_graph=_read(raw_path)
    _,frames=run.source_frames(shot);bypts={f['pts']:i for i,f in enumerate(frames)}
    for g in (raw_graph,strong_graph):
        if [f['pts'] for f in g['frames']]!=[f['pts'] for f in frames]:raise ValueError('Source graph clock mismatch')
    output=args.output/shot;output.mkdir(parents=True,exist_ok=True)
    receipt=dict(source_identity_sha256=_digest(material/'identity.json'),
        strong_identity=str(report_path),strong_identity_sha256=_digest(report_path),
        strong_graph_sha256=_digest(strong_path),raw_graph=str(raw_path),raw_graph_sha256=_digest(raw_path),
        role_identity_sha256=_digest(material/shot/args.material_role_name/'identity.json'),
        implementation_sha256=_digest(Path(__file__)),
        propagation_sha256=_digest(Path(propagate_readout_identities.__code__.co_filename)),
        local_fragment_membership_does_not_propagate_identity=True)
    if (output/'receipt.json').exists() and _read(output/'receipt.json')!=receipt:
        raise ValueError('Readout graph dependencies changed; preserve the previous result')
    _write(output/'receipt.json',receipt)
    measurements=[];roles=[];eligibility=[];groups=[];counts=[];raw_ids=[];strong_ids=[]
    for fi,f in enumerate(frames):
        source=material/shot/f'{f["pts"]}.npz'
        with np.load(source) as z:m=dict(z)
        role_path=material/shot/args.material_role_name/f'{f["pts"]}.json';role=_read(role_path)
        if role['source_readout_sha256']!=_digest(source):raise ValueError('Qualified source readouts changed')
        ok=(m['candidate_flags']&3)==0;curve=m['query_curve'][m['candidate_query']]
        for c in role['curves']:
            if c.get('source_line_support_qualified'):
                ids=c['selected_candidate_ids'];ok[curve==c['curve']]=False;ok[ids]=True
                groups.append(dict(frame=fi,candidate_ids=ids,source_line_support_qualified=True,
                    source_role=str(role_path),source_role_sha256=_digest(role_path)))
        measurements.append(m);roles.append(role);eligibility.append(ok);counts.append(len(ok))
        with np.load(raw_path.parent/f'{f["pts"]}.npz') as z:raw_ids.append(z['ids'])
        with np.load(strong_path.parent/f'{f["pts"]}.npz') as z:strong_ids.append(z['ids'])
    links=[]
    def collect(previous,current,edges,old_ids,new_ids,evidence):
        for ei,edge in enumerate(edges):
            if not edge.get('unambiguous') or not edge.get('source_only'):continue
            pairs=[[int(old_ids[a]),int(new_ids[b])] for a,b in edge['pairs']
                if eligibility[previous][old_ids[a]] and eligibility[current][new_ids[b]]]
            if not pairs:continue
            a,b=np.asarray(pairs).T
            if min(len(np.unique(a)),len(np.unique(b)))<3:continue
            if np.linalg.norm(np.ptp(measurements[current]['candidate_uv'][np.unique(b)],axis=0))<56.:continue
            links.append(dict(previous=previous,current=current,pairs=pairs,source_only=True,unambiguous=True,
                evidence=evidence,source_edge=ei,identity_from_query_name=False))
    for graph,path,ids in ((raw_graph,raw_path,raw_ids),(strong_graph,strong_path,strong_ids)):
        for fi,edges in enumerate(graph['edges']):collect(fi,fi+1,edges,ids[fi],ids[fi+1],dict(graph=str(path),edge_frame=fi))
        for ri,record in enumerate(graph.get('reference_edges',[])):
            a,b=record['reference'],record['current']
            collect(a,b,record['edges'],ids[a],ids[b],dict(graph=str(path),reference_edge=ri))
    anchors=[]
    for ai,anchor in enumerate(strong['anchors']):
        fi=bypts[anchor['pts']]
        for _,ni in anchor['link']['pairs']:
            ci=int(strong_ids[fi][ni])
            if eligibility[fi][ci]:anchors.append(dict(frame=fi,candidate=ci,identity=anchor['identity'],
                source_reference=anchor['source_reference'],source_identity_report=str(report_path),source_anchor=ai))
    # A qualified nearby whole line supplies a source reference. Its name only
    # schedules an image comparison; only actual matching nodes inherit it.
    trusted=defaultdict(set)
    for label in strong['labels']:
        if label['status']=='anchored':
            fi=label['frame'];curve=strong_graph['frames'][fi]['fragments'][label['fragment']]['curve']
            trusted[curve].add(fi)
    trusted={c:np.asarray(sorted(v)) for c,v in trusted.items()};scheduled=set()
    for fi,role in enumerate(roles):
        m=measurements[fi];curve=m['query_curve'][m['candidate_query']]
        for c in role['curves']:
            if c.get('source_line_support_qualified') or c['curve'] not in trusted:continue
            if np.count_nonzero(eligibility[fi]&(curve==c['curve']))<3:continue
            available=trusted[c['curve']];at=np.searchsorted(available,fi)
            for j in (at-1,at):
                if 0<=j<len(available) and int(available[j])!=fi:scheduled.add((int(available[j]),fi))
    with np.load(run.material_navigation_file(args,shot)) as z:projection=dict(z)
    @lru_cache(maxsize=6)
    def pixels(fi):
        f=frames[fi];image=cv2.imread(f['image'])
        if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=f['source_BGR_sha256']:
            raise ValueError('Readout correspondence source pixels changed')
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),~run.surfaces(f,image)[0]['ice']
    @lru_cache(maxsize=32)
    def candidates(fi):
        m=measurements[fi];curve=m['query_curve'][m['candidate_query']];ok=eligibility[fi]
        pieces=[dict(curve=int(c),candidate_ids=np.flatnonzero(ok&(curve==c)).tolist()) for c in np.unique(curve[ok])]
        return source_nodes(pixels(fi)[0],m,pieces,inventory)
    direct=output/'direct_links';direct.mkdir(exist_ok=True);started=time.monotonic()
    for ordinal,(ri,fi) in enumerate(sorted(scheduled,key=lambda x:(x[1],x[0]))):
        target=direct/f'{frames[ri]["pts"]}_{frames[fi]["pts"]}.json'
        if target.exists():record=_read(target)
        else:
            rgb,mask=pixels(fi);ref,rm=pixels(ri)
            with np.load(strong_path.parent/f'{frames[ri]["pts"]}.npz') as z:old=dict(z)
            now=candidates(fi);pa,pb=projection['uv'][ri],projection['uv'][fi]
            visible=(projection['status'][ri]==0)&(projection['status'][fi]==0)&np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
            for xy in (pa,pb):visible&=(xy[:,0]>-150)&(xy[:,0]<2070)&(xy[:,1]>100)&(xy[:,1]<1230)
            edges=[];h=None
            if visible.sum()>=12:
                h,_=cv2.findHomography(pa[visible],pb[visible],0)
                edges=link_fragments(ref,rgb,old,now,rm,mask,h)
            record=dict(previous=ri,current=fi,reference_pts=frames[ri]['pts'],pts=frames[fi]['pts'],
                old_ids=old['ids'].tolist(),current_ids=now['ids'].tolist(),edges=edges,
                initial_h=h.tolist() if h is not None else None,model_supplies_search_only=True,
                local_query_groups_do_not_share_identity=True)
            _write(target,record)
        collect(ri,fi,record['edges'],record['old_ids'],record['current_ids'],dict(source_links=str(target),sha256=_digest(target)))
        if ordinal%50==0 or ordinal==len(scheduled)-1:
            print(json.dumps(dict(event='weak_frame_actual_node_links',pairs=ordinal+1,total=len(scheduled),
                accepted_links=len(links),elapsed_s=time.monotonic()-started)),flush=True)
    graph_path=output/'graph.json'
    _write(graph_path,dict(shot=shot,source_frames=[dict(pts=f['pts'],source_BGR_sha256=f['source_BGR_sha256']) for f in frames],
        candidate_counts=counts,qualified_groups=groups,links=links,anchors=anchors,dependencies=receipt,
        identity_transfer='actual_matched_nodes_or_source_qualified_whole_line_only',native_fit_residual_used=False,direct_source_pairs=len(scheduled)))
    _export_readout_packet(args,inventory,frames,measurements,graph_path)


def _export_readout_packet(args,inventory,frames,measurements,graph_path):
    from rinkspace.session.material_tracks import propagate_readout_identities
    graph=_read(graph_path);output=graph_path.parent;shot=graph['shot']
    counts=graph['candidate_counts'];links=graph['links'];anchors=graph['anchors'];groups=graph['qualified_groups']
    bypts={f['pts']:i for i,f in enumerate(frames)}
    offsets,components,labels=propagate_readout_identities(counts,links,anchors,groups)
    np.savez_compressed(output/'node_components.npz',offsets=offsets,component=components)
    identity={r['component']:r['identities'][0] for r in labels if r['status']=='anchored'}
    identity_path=output/'identity_graph.json'
    _write(identity_path,dict(shot=shot,source_graph=str(graph_path),source_graph_sha256=_digest(graph_path),
        node_components=str(output/'node_components.npz'),node_components_sha256=_digest(output/'node_components.npz'),
        component_labels=labels,conflicts=sum(r['status']=='identity_conflict' for r in labels),
        original_UV_unchanged=True,local_fragments_not_used_as_identity_equivalence=True))
    paths={p['curve_id']:p['native_path_id'] for p in inventory};accepted=[];coverage=[]
    for fi,f in enumerate(frames):
        m=measurements[fi];names=[]
        for ci in range(counts[fi]):
            component=int(components[offsets[fi]+ci]);name=identity.get(component)
            if name is None:continue
            quantiles=m['candidate_side_quantiles_uv'][ci]
            if not np.isfinite(quantiles).all():continue
            uv=quantiles[:,1].mean(0);names.append(name)
            accepted.append(dict(pts=f['pts'],identity=name,path_id=paths[name],candidate=int(ci),uv=uv.tolist(),
                actual_source_quantiles=quantiles.tolist(),evidence_image=f['image'],source_BGR_sha256=f['source_BGR_sha256'],
                source_identity_graph=str(identity_path),identity_component=component,human_identity_scope_used=True,
                human_precise_pixel_acceptance=False,current_source_RGB_measured=True,
                all_source_coordinates_from_readouts=True,model_UV_used_as_measurements=False))
        coverage.append(dict(pts=f['pts'],readouts=len(names),identities=sorted(set(names))))
    _write(output/'source_verified.json',dict(shot=shot,accepted=accepted,pending=[],frames=coverage,
        source_packet=str(graph_path),source_packet_sha256=_digest(graph_path),
        source_links=str(identity_path),source_links_sha256=_digest(identity_path),identity_review=str(args.output/'image_review.json'),
        native_fit_residual_not_used_for_selection=True,fit_eligible=False,review_state='awaiting_source_image_review',
        user_visual_acceptance=False,fit_observations_changed=False))
    proofs=output/'source_proofs';proofs.mkdir(exist_ok=True)
    colors={name:col for name,col in zip(paths,[(0,255,0),(0,128,255),(255,0,255)])}
    for pts in [948992,1004032,1008640,1010688,1014272,1041152,1218304,1300992]:
        rows=[r for r in accepted if r['pts']==pts]
        if not rows:continue
        source=cv2.imread(frames[bypts[pts]]['image']);annotated=source.copy()
        for row in rows:cv2.circle(annotated,tuple(np.rint(row['uv']).astype(int)),3,colors[row['identity']],-1)
        cv2.imwrite(str(proofs/f'{pts}.jpg'),np.vstack([source,annotated]),[cv2.IMWRITE_JPEG_QUALITY,95])
    summary=dict(source_rows=len(accepted),source_frames=sum(c['readouts']>0 for c in coverage),
        full_frames=len(frames),identity_conflicts=sum(r['status']=='identity_conflict' for r in labels),
        direct_source_pairs=graph.get('direct_source_pairs'),output=str(output/'source_verified.json'),fit_eligible=False)
    _write(output/'summary.json',summary);print(json.dumps(summary),flush=True)


def refine_readout_geometry(args):
    """Apply actual-source shape checks without redoing unchanged RGB matching."""
    import run_native_programme as run
    from rinkspace.session.material_tracks import check_matched_line_geometry
    material=run.material_directory(args);inventory=_read(material/'inventory.json')
    parent=_read(args.source_identity_graph);path=Path(parent['source_graph']);graph=_read(path)
    if _digest(path)!=parent['source_graph_sha256']:raise ValueError('Source graph changed')
    if _digest(Path(parent['node_components']))!=parent['node_components_sha256']:raise ValueError('Source labels changed')
    shot=graph['shot'];_,frames=run.source_frames(shot)
    if [f['pts'] for f in frames]!=[f['pts'] for f in graph['source_frames']]:raise ValueError('Source clock changed')
    output=args.output/shot
    if output.exists():raise FileExistsError('Preserve the prior source consistency revision')
    output.mkdir(parents=True)
    with np.load(parent['node_components']) as z:offsets=z['offsets'];components=z['component']
    identities={r['component']:r['identities'][0] for r in parent['component_labels'] if r['status']=='anchored'}
    measurements=[];allowed=[];reviews=[]
    for fi,f in enumerate(frames):
        source=material/shot/f'{f["pts"]}.npz'
        with np.load(source) as z:m=dict(z)
        role=_read(material/shot/args.material_role_name/f'{f["pts"]}.json')
        if role['source_readout_sha256']!=_digest(source):raise ValueError('Source readouts changed')
        measurements.append(m);groups=defaultdict(list)
        for ci in range(len(m['candidate_uv'])):
            identity=identities.get(int(components[offsets[fi]+ci]))
            if identity is not None:groups[identity].append(ci)
        ok=np.ones(len(m['candidate_uv']),bool);reports=[]
        for identity,ids in groups.items():
            selected,diagnostics=check_matched_line_geometry(m,ids,role['curves'])
            ok[ids]=False;ok[selected]=True
            reports.append(dict(identity=identity,input_candidates=len(ids),retained_candidates=len(selected),curves=diagnostics))
        allowed.append(ok);reviews.append(dict(pts=f['pts'],source_readout_sha256=_digest(source),identities=reports))
    links=[]
    for link in graph['links']:
        a,b=link['previous'],link['current']
        pairs=[(i,j) for i,j in link['pairs'] if allowed[a][i] and allowed[b][j]]
        if not pairs:continue
        left,right=np.asarray(pairs).T
        if min(len(np.unique(left)),len(np.unique(right)))<3:continue
        if np.linalg.norm(np.ptp(measurements[b]['candidate_uv'][np.unique(right)],axis=0))<56.:continue
        links.append(dict(link,pairs=pairs))
    groups=[dict(g,candidate_ids=[i for i in g['candidate_ids'] if allowed[g['frame']][i]]) for g in graph['qualified_groups']]
    anchors=[a for a in graph['anchors'] if allowed[a['frame']][a['candidate']]]
    review_path=output/'source_geometry.json';_write(review_path,json.loads(json.dumps(reviews,default=lambda x:x.tolist())))
    revised=dict(graph,links=links,qualified_groups=groups,anchors=anchors,
        parent_source_graph=str(path),parent_source_graph_sha256=_digest(path),
        source_geometry_review=str(review_path),source_geometry_review_sha256=_digest(review_path),
        sparse_matches_not_promoted_to_whole_line_qualifications=True,
        implementation_sha256=_digest(Path(__file__)),
        geometry_check_sha256=_digest(Path(check_matched_line_geometry.__code__.co_filename)))
    graph_path=output/'graph.json';_write(graph_path,revised)
    _export_readout_packet(args,inventory,frames,measurements,graph_path)
