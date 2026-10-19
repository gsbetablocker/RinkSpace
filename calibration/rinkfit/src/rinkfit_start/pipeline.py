"""Executable S0/S1/S2 orchestration over the complete reviewed shot.

Local H results are initialization candidates. They never certify a physical
camera, a metre scale, a whole-shot pixel budget, or a rejected observation.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from dataclasses import fields
import hashlib
import json
import os
import time
import numpy as np

from .data import Observations, SeedCatalog, sha256_file
from .runtime import atomic_json, atomic_npz, canonical_json, Journal, run_lock, validate_output_path
from .registration import (MaterialSupport, RegistrationObservations, RegistrationOptions,
                           RegistrationError, fit_registration, normalized_dlt, project_points)

PACK_ROOT = Path(__file__).resolve().parents[2]
_WORKER = None


def load_config(path=None):
    config=json.loads((Path(path) if path else PACK_ROOT/'configs/shot1.json').read_text(encoding='utf-8'))
    if config['scope'] != 'all_1248_native_frames_all_182017_reviewed_observations':
        raise ValueError('This event launch configuration must preserve the full source scope')
    if config['engineering_budget_native_px'] <= 0:
        raise ValueError('Native engineering budget must be positive')
    known={f.name for f in fields(RegistrationOptions)}
    if set(config['registration_options'])-known: raise ValueError('Unknown registration option')
    return config


def build_material(catalog, label):
    """Use ordered MODEL vertices only; never infer an observed run order."""
    entry=catalog.materials[label]; variants=entry['variants']
    if len(variants)!=1:
        return None, 'multiple_declared_model_branches_need_original_branch_binding'
    variant=catalog.variants[variants[0]]; plane=variant['plane_candidate']
    if plane is None: return None, 'material_requires_shared_3d_board_surface'
    geometry=variant['geometry']; kind=geometry['kind']
    provenance=f"conditional legacy world seed {variants[0]}; source scene {catalog.catalog['source_scene_sha256']}; not a fixed scene constraint"
    kw={'provenance':provenance, 'branch_id':variants[0]}
    if kind=='circle':
        center=catalog.plane_coordinates(plane,np.array([geometry['center_xyz']]))[0]
        return MaterialSupport.arc(label,center,geometry['radius'],*geometry['angle_interval'],**kw), None
    if kind=='segment':
        ends=catalog.plane_coordinates(plane,np.array([geometry['a_xyz'],geometry['b_xyz']]))
        return MaterialSupport.segment(label,*ends,**kw), None
    points=catalog.plane_coordinates(plane,catalog.world[variant['world_array']])
    if kind=='point': return MaterialSupport.point(label,points[0],**kw),None
    if 'path_offsets' in geometry:
        offsets=geometry['path_offsets']
        if len(offsets)!=2:
            return None,'disconnected_model_paths_need_finite_union_adapter_no_false_connecting_segment'
        points=points[offsets[0]:offsets[1]]
        if geometry.get('path_closed',[False])[0] and not np.array_equal(points[0],points[-1]):
            points=np.vstack([points,points[0]])
    # Consecutive duplicate model vertices have zero support length. Removing
    # those vertices is an exact representation change, not a source UV edit.
    keep=np.r_[True,np.any(np.diff(points,axis=0)!=0,axis=1)]
    points=points[keep]
    if len(points)<2: return None,'degenerate_conditional_model_seed'
    return MaterialSupport.polyline(label,points,connectivity_evidence=variant['connectivity_provenance'],**kw),None


def load_context(data_dir):
    obs=Observations.load(data_dir); seeds=SeedCatalog(data_dir); seeds.validate_source(obs)
    supports={}; unsupported={}
    for label in seeds.materials:
        try: support,reason=build_material(seeds,label)
        except (RegistrationError,ValueError) as exc:
            support,reason=None,'conditional_seed_not_representable: '+str(exc)
        if support is None: unsupported[label]=reason
        else: supports[label]=support
    return obs,seeds,supports,unsupported


def source_fingerprint(data_dir):
    names=['input_manifest.json','observations.npz','seeds.json','world_seeds.npz','camera_seeds.npz',
           'bootstrap_state.json','source_sigma.npz','media.json']
    return {name:sha256_file(Path(data_dir)/name) for name in names}


def prepare(data_dir, output, config_path=None):
    data_dir=Path(data_dir).resolve(); output=validate_output_path(output)
    config=load_config(config_path); obs,seeds,supports,unsupported=load_context(data_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'run.json').exists():
        old=json.loads((output/'run.json').read_text(encoding='utf-8'))
        if old['input_hashes']!=source_fingerprint(data_dir) or old['config']!=config:
            raise ValueError('Existing run has different inputs or options; choose a new run directory')
        return {'run_dir':str(output),'status':'existing_compatible_run','source':obs.summary()}
    with run_lock(output):
        plane_names=list(seeds.planes); ids=[]; jobs=[]; rows=[]
        for frame in range(obs.manifest['frame_min'],obs.manifest['frame_max']+1):
            idx=obs.indices_for_frame(frame); frame_jobs=[]
            for plane in plane_names:
                selected=[int(i) for i in idx if str(obs.source_target_labels[i]) in supports
                          and seeds.plane_for_material(str(obs.source_target_labels[i]))==plane]
                if not selected: continue
                key=f'f{frame:04d}_{plane}'
                jobs.append({'key':key,'frame':frame,'plane':plane,'observations':len(selected),
                             'materials':len(set(obs.source_target_labels[selected].tolist()))})
                frame_jobs.append(key)
            unresolved=[int(i) for i in idx if str(obs.source_target_labels[i]) in unsupported]
            rows.extend(unresolved)
            ids.append({'frame':frame,'native_pts':int(obs.native_pts[idx[0]]),'source_rows':len(idx),
                        'local_jobs':frame_jobs,'requires_physical_or_branch_binding_rows':len(unresolved),
                        'final_physical_verification':'pending'})
        run={'schema':'rinkfit.run.v1','data_dir':str(data_dir),'input_hashes':source_fingerprint(data_dir),
             'config':config,'source':obs.summary(),'complete_shot_ledger':'frame_ledger.json',
             'physical_fit_completed':False,'local_H_is_final_camera':False}
        atomic_json(output/'run.json',run)
        atomic_json(output/'frame_ledger.json',ids)
        atomic_json(output/'jobs.json',jobs)
        atomic_npz(output/'source_obligations.npz',{'source_row':np.arange(len(obs.source_uv),dtype=np.int64),
                    'pending_3d_or_branch_source_rows':np.array(rows,dtype=np.int64)})
        atomic_json(output/'input_coverage.json',{'source':obs.summary(),'conditional_planes':seeds.coverage(obs),
                    'S1_unsupported_materials':unsupported,'S1_jobs':len(jobs),
                    'S1_scheduled_source_rows':len(obs.source_uv)-len(rows),
                    'S1_pending_3d_or_branch_source_rows':len(rows),'S6_source_rows_required':len(obs.source_uv),
                    'not_applicable_to_H_does_not_remove_a_physical_obligation':True})
        Journal(output).emit('prepared',source=obs.summary(),jobs=len(jobs),physical_fit_completed=False)
    return {'run_dir':str(output),'status':'prepared','source':obs.summary(),'registration_jobs':len(jobs)}


def read_run(run_dir):
    run_dir=Path(run_dir).resolve(); run=json.loads((run_dir/'run.json').read_text(encoding='utf-8'))
    if source_fingerprint(run['data_dir'])!=run['input_hashes']:
        raise ValueError('A frozen input or seed file changed after prepare')
    return run_dir,run


def _worker_init(data_dir, config):
    global _WORKER
    _WORKER=(*load_context(data_dir),config)


def _bbox_seed(materials, native_uv):
    samples=np.concatenate([m.atlas_points(np.linspace(*m.station_interval,33)) for m in materials])
    lo=samples.min(0); span=np.maximum(np.ptp(samples,axis=0),1e-5)
    vmin=native_uv.min(0); vspan=np.maximum(np.ptp(native_uv,axis=0),8.)
    scale=vspan/span
    H=np.array([[scale[0],0,vmin[0]-scale[0]*lo[0]],
                [0,-scale[1],vmin[1]+vspan[1]+scale[1]*lo[1]],[0,0,1.]])
    return H


def _save_candidate(directory, name, candidate, source_rows):
    # Full local state in compact arrays. Input and seed catalog hashes in
    # run.json identify all source data and geometry; their values are frozen.
    arrays={'source_rows':source_rows,'H_atlas_to_native':candidate.H_atlas_to_native,
            'stations':candidate.replay.stations,'predicted_native_uv':candidate.replay.predicted_native_uv,
            'error_native_uv':candidate.replay.error_native_uv,
            'at_support_boundary':candidate.replay.at_support_boundary,
            'ambiguous_minimum':candidate.replay.ambiguous_minimum}
    quadrature=[]
    for j,q in enumerate(candidate.quadrature):
        arrays[f'q{j}_stations']=q.stations; arrays[f'q{j}_weights']=q.weights
        quadrature.append({'index':j,'material_id':q.material_id,'measure':q.measure,
                           'native_fill_bound':q.native_fill_bound(candidate.H_atlas_to_native)})
    atomic_npz(directory/f'{name}.npz',arrays)
    record=candidate.to_dict(include_observations=False)
    for key in ['materials','stations','predicted_native_uv','error_native_uv','at_support_boundary','ambiguous_minimum','quadrature']:
        record.pop(key,None)
    record['material_ids']=[m.material_id for m in candidate.materials]
    record['quadrature_manifest']=quadrature
    record['arrays_sha256']=sha256_file(directory/f'{name}.npz')
    atomic_json(directory/f'{name}.json',record)
    return record


def _run_job(job, root, diagnostic=False):
    obs,seeds,supports,unsupported,config=_WORKER
    frame,plane=job['frame'],job['plane']; frame_rows=obs.indices_for_frame(frame)
    rows=np.array([i for i in frame_rows if str(obs.source_target_labels[i]) in supports
                   and seeds.plane_for_material(str(obs.source_target_labels[i]))==plane],dtype=np.int64)
    labels=tuple(str(x) for x in obs.source_target_labels[rows]); local={k:supports[k] for k in sorted(set(labels))}
    observations=RegistrationObservations(obs.source_uv[rows],labels,obs.record_keys(rows))
    options_dict=dict(config['registration_options'])
    if diagnostic: options_dict.update(config['diagnostic_options'])
    options=RegistrationOptions(**options_dict)
    candidates=[('historical_GS_no_distortion_seed',seeds.historical_H(plane,frame))]
    entities=[(i,local[label]) for i,label in enumerate(labels) if local[label].kind=='point']
    if len(entities)>=4:
        try:
            dlt=normalized_dlt(np.array([m.coordinates[0] for _,m in entities]),
                     observations.native_uv[[i for i,_ in entities]],
                     entity_evidence=[f'original physical ad_corner descriptor: {m.material_id}' for _,m in entities])
            candidates.append(('reviewed_physical_corner_DLT',dlt.matrix))
        except RegistrationError: pass  # Failed initialization rank; all rows remain.
    candidates.append(('independent_bbox_affine_hypothesis_no_claim_of_point_correspondence',
                       _bbox_seed(tuple(local.values()),observations.native_uv)))
    directory=Path(root)/('diagnostics' if diagnostic else 'registration')/job['key']
    directory.mkdir(parents=True,exist_ok=True)
    errors=[]; alternatives=[]; winner=None; started=time.monotonic()
    for candidate_index,(name,H) in enumerate(candidates):
        try:
            result=fit_registration(local,observations,H,frame_id=frame,plane_id=plane,options=options,
                                    initial_provenance=name)
            best=result.best_native
            alternatives.append({'seed':name,'max_error_native_px':best.replay.max_error_native_px,
                'rms_error_native_px':best.replay.rms_error_native_px,'rank':best.rank.rank if best.rank else None,
                'H_atlas_to_native':best.H_atlas_to_native.tolist(),'status':result.status,
                'complete_best_snapshot':f'candidate_{candidate_index}.json'})
            _save_candidate(directory,f'candidate_{candidate_index}',best,rows)
            atomic_json(directory/f'candidate_{candidate_index}_log.json',result.optimization_log)
            if winner is None or (best.replay.max_error_native_px,best.replay.rms_error_native_px)<winner[0]:
                winner=((best.replay.max_error_native_px,best.replay.rms_error_native_px),name,result)
        except (RegistrationError,ValueError,np.linalg.LinAlgError) as exc:
            errors.append({'seed':name,'initializer_failure':str(exc),'observation_error_claim':False})
    record={'schema':'rinkfit.local_job.v1','job':job,'diagnostic_only':diagnostic,'observations':len(rows),
            'source_rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'candidate_failures':errors,
            'alternatives':alternatives,'elapsed_seconds':time.monotonic()-started,
            'final_physical_accuracy_accepted':False,'all_local_original_rows_retained':True}
    if winner:
        _,name,result=winner
        record.update(status='local_candidate_available',chosen_seed=name,
            best=_save_candidate(directory,'best',result.best_native,rows),
            final=_save_candidate(directory,'final',result.final,rows),limitations=list(result.limitations))
        atomic_json(directory/'optimization_log.json',result.optimization_log)
    else:
        record.update(status='initializer_domain_or_rank_unresolved',physical_obligation='all rows still required')
    atomic_json(directory/'result.json',record)  # Commit job last.
    return {'key':job['key'],'status':record['status'],'observations':len(rows),
            'elapsed_seconds':record['elapsed_seconds'],
            'local_max_native_px':None if not winner else winner[0][0]}


def diagnose(run_dir, frame, plane):
    root,run=read_run(run_dir)
    jobs=json.loads((root/'jobs.json').read_text(encoding='utf-8'))
    matches=[j for j in jobs if j['frame']==frame and j['plane']==plane]
    if not matches: raise ValueError('No applicable planar observations in that diagnostic job')
    _worker_init(run['data_dir'],run['config'])
    with run_lock(root):
        Journal(root).emit('diagnostic_started',frame=frame,plane=plane,whole_shot_scope_unchanged=True)
        result=_run_job(matches[0],root,True)
        Journal(root).emit('diagnostic_finished',**result,physical_fit_completed=False)
    return result


def register(run_dir, workers=None):
    root,run=read_run(run_dir); config=run['config']
    workers=int(workers or config['workers'])
    if workers<1: raise ValueError('workers must be positive')
    jobs=json.loads((root/'jobs.json').read_text(encoding='utf-8'))
    # Priority changes scheduling only; every frame/row was registered in S0.
    # Broad material support is useful for early initializers across the shot.
    jobs.sort(key=lambda j:(-j['materials'],-j['observations'],j['frame'],j['plane']))
    pending=[j for j in jobs if not (root/'registration'/j['key']/'result.json').exists()]
    done=len(jobs)-len(pending); journal=Journal(root)
    with run_lock(root):
        journal.emit('registration_started',jobs=len(jobs),completed=done,workers=workers,
                     all_frames_in_scope=1248,production_physical_fit=False)
        iterator=iter(pending); futures={}; exhausted=False; paused=False
        with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init,
                                 initargs=(run['data_dir'],config)) as pool:
            while futures or not exhausted:
                paused=(root/'STOP_REQUESTED').exists()
                while not paused and not exhausted and len(futures)<2*workers:
                    job=next(iterator,None)
                    if job is None: exhausted=True; break
                    futures[pool.submit(_run_job,job,str(root))]=job
                if paused and not futures: break
                ready,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                if not ready:
                    journal.emit('registration_heartbeat',completed=done,jobs=len(jobs),running=len(futures),
                                 stop_requested=paused,physical_fit_completed=False)
                for future in ready:
                    job=futures.pop(future)
                    try: result=future.result()
                    except Exception as exc:
                        journal.emit('worker_failure',key=job['key'],error=repr(exc),
                                     source_obligation_retained=True)
                        raise
                    done+=1; journal.emit('registration_job_finished',completed=done,jobs=len(jobs),**result)
        status='registration_paused' if paused and done<len(jobs) else 'registration_frontend_finished'
        journal.emit(status,completed=done,jobs=len(jobs),physical_fit_completed=False)
    return {'status':status,'completed_jobs':done,'total_jobs':len(jobs),'physical_fit_completed':False}


def build_graphs(run_dir):
    from .graph import PairwiseEdge, GraphAnchor, RegistrationGraph
    root,run=read_run(run_dir); obs,seeds,supports,_=load_context(run['data_dir'])
    report={}; all_frames=range(1,1249)
    with run_lock(root):
        journal=Journal(root)
        for plane in seeds.planes:
            corners={}
            for frame in all_frames:
                available={}
                for row in obs.indices_for_frame(frame):
                    label=str(obs.source_target_labels[row]); m=supports.get(label)
                    if m is not None and m.kind=='point' and seeds.plane_for_material(label)==plane:
                        available.setdefault(label,[]).append(int(row))
                corners[frame]={label:idx[0] for label,idx in available.items() if len(idx)==1}
            edges=[]; rejected=0; pairs=set()
            for i in all_frames:
                for offset in run['config']['graph_pair_offsets_frames']:
                    j=i+offset
                    if j<=1248: pairs.add((i,j))
            # A revisit edge still needs actual shared corner entities in both
            # native frames. It is never synthesized from H_j H_i^-1.
            by_signature={}
            for frame,entities in corners.items():
                if len(entities)>=4: by_signature.setdefault(tuple(sorted(entities)),[]).append(frame)
            for frames_in in by_signature.values():
                first=frames_in[0]
                pairs.update((first,j) for j in frames_in[1:])
            for i,j in sorted(pairs):
                shared=sorted(set(corners[i])&set(corners[j]))
                if len(shared)<4: continue
                ri=[corners[i][k] for k in shared]; rj=[corners[j][k] for k in shared]
                try:
                    edges.append(PairwiseEdge.from_entity_correspondences(i,j,obs.source_uv[ri],obs.source_uv[rj],
                        plane_id=plane,entity_evidence=[f'physical ad_corner descriptor {k}' for k in shared],
                        provenance='independent DLT of reviewed physical corner entities in original native frames',
                        source_record_keys=obs.record_keys(ri),target_record_keys=obs.record_keys(rj)))
                except RegistrationError: rejected+=1
            graph=RegistrationGraph(all_frames,plane,edges)
            anchors=[]
            for component in graph.connected_components():
                candidates=[]
                for frame in component:
                    path=root/'registration'/f'f{frame:04d}_{plane}'/'best.json'
                    if path.exists():
                        item=json.loads(path.read_text(encoding='utf-8'))
                        if item['rank'] and item['rank']['rank']==8:
                            candidates.append((item['max_error_native_px'],frame,item))
                if candidates:
                    # Distinct temporal support, with all alternatives retaining
                    # the SAME conditional template; these are not hard truths.
                    candidates.sort(key=lambda x:x[1])
                    for part in np.array_split(np.arange(len(candidates)),min(3,len(candidates))):
                        _,frame,item=min((candidates[k] for k in part),key=lambda x:(x[0],x[1]))
                        anchors.append(GraphAnchor(frame,np.array(item['H_atlas_to_native']),f'conditional_legacy_plane:{plane}',
                              'rank-8 S1 finite-material candidate; conditional legacy metric template, not measured geometry'))
            initial=graph.initialize(anchors,gauge_mode='component_projective')
            out=root/'graph';out.mkdir(exist_ok=True)
            atomic_json(out/f'{plane}.json',initial.to_dict())
            refined=graph.refine(initial,max_iterations=run['config']['graph_refinement_iterations'])
            atomic_json(out/f'{plane}_refined.json',refined.to_dict())
            original_anchors={a.frame_id:a for a in anchors}
            upgrade_charts=[]
            for component in initial.components:
                reference=original_anchors.get(component.gauge_frame)
                upgrade_charts.append({'component_id':component.component_id,
                    'reference_frame':component.gauge_frame,
                    'Q_metric_to_component_atlas':None if reference is None else reference.H_atlas_to_native,
                    'metric_status':'unknown' if reference is None else 'conditional_legacy_world_seed',
                    'physical_plane_M':seeds.embedding(plane),
                    'Q_direction':'q_component=Q xi_metric; H_component Q ~ P M'})
            atomic_json(out/f'{plane}_charts.json',upgrade_charts)
            cycles=graph.fundamental_cycles(max_cycles=run['config']['graph_diagnostic_cycle_limit'])
            report[plane]={'frames':1248,'independent_corner_edges':len(edges),
                'degenerate_entity_edge_hypotheses':rejected,'components':len(initial.components),
                'conditional_metric_anchors':len(anchors),'actual_cycle_checks':len(cycles),
                'cycle_max_native_px':max((c.max_displacement_native_px for c in cycles),default=None),
                'joint_refinement_status':refined.status,
                'initial_native_witness_rms_px':refined.initial_native_witness_rms_px,
                'final_native_witness_rms_px':refined.final_native_witness_rms_px,
                'curve_pair_edges_bound':False,'global_physical_verification':False}
            journal.emit('graph_plane_initialized',plane=plane,**report[plane])
        atomic_json(root/'graph/summary.json',report)
    return report


def metric_initializers(run_dir):
    from .metric_upgrade import PlaneHomographyInput, common_projection_from_planes
    root,run=read_run(run_dir); seeds=SeedCatalog(run['data_dir']); records=[]; projections=[]; ready_frames=[]
    with run_lock(root):
        for frame in range(1,1249):
            inputs=[]
            for plane in seeds.planes:
                path=root/'registration'/f'f{frame:04d}_{plane}'/'best.json'
                if not path.exists():continue
                item=json.loads(path.read_text(encoding='utf-8'))
                if not item['rank'] or item['rank']['rank']<8:continue
                inputs.append(PlaneHomographyInput(plane_id=plane,frame_key=f'frame:{frame}',
                    M=seeds.embedding(plane),H_atlas_to_native=np.array(item['H_atlas_to_native']),
                    Q_metric_to_atlas=np.eye(3),metric_chart_provenance=
                    'S1 explicitly used the same legacy plane chart; Q=I is conditional old-world initialization only',
                    genuine_plane=True,metric_chart_kind='conditional_world_seed'))
            result=common_projection_from_planes(inputs)
            # IEEE infinity is a diagnostic, never a JSON numeric value.
            from .runtime import json_value
            record=json_value(result)
            for key in ['nullspace_gap','identifiable_subspace_condition']:
                if record[key] is not None and not np.isfinite(record[key]):record[key]='infinity'
            record['frame']=frame;record['rank_8_plane_count']=len(inputs)
            records.append(record)
            if result.ready:ready_frames.append(frame);projections.append(result.P)
        out=root/'metric';out.mkdir(exist_ok=True)
        atomic_json(out/'frame_ledger.json',records)
        atomic_npz(out/'conditional_P.npz',{'frame_indices':np.array(ready_frames,dtype=np.int32),
                    'P':np.array(projections,dtype=float).reshape((-1,3,4))})
        summary={'frames_in_ledger':1248,'conditional_P_initializers':len(ready_frames),
                 'fixed_head_recovered':False,'unknown_atlas_Q_fitted':False,'physical_fit_completed':False}
        Journal(root).emit('metric_initializers_prepared',**summary)
    return summary
