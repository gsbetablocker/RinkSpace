"""Empirical current-profile geometry statistics for dense initialization.

Five actually observed neighboring profiles are paired source realizations.
A source-only quadratic removes tangent curvature; retained residual support is
correlated across the whole frame-local family. This is a geometry surrogate,
not a calibrated noise likelihood. No old selected exposure is counted twice.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip
import numpy as np
from .io_utils import read_jsonl,file_sha256,write_json,object_sha256
from .source_statistics import supported_factor_whitening,StatisticsUnavailable
from .role_guard import FrameRoleGuard

def _shifts(intensity,offsets):
    y=np.asarray(intensity);g=abs(np.gradient(y,float(offsets[1]-offsets[0]),axis=1));k=g.argmax(axis=1)
    if np.any((k==0)|(k==len(offsets)-1)):raise StatisticsUnavailable('profile realization touches finite source window')
    v=np.take_along_axis(g,k[:,None]+np.arange(-1,2)[None,:],axis=1);den=v[:,0]-2*v[:,1]+v[:,2]
    frac=np.divide(.5*(v[:,0]-v[:,2]),den,out=np.zeros(5),where=abs(den)>1e-9)
    return offsets[k]+(offsets[1]-offsets[0])*np.clip(frac,-1,1)

def compile_dense_statistics(dense_dirs,roles,output,*,selected_exposure_journal=None):
    out=Path(output)
    if out.exists():raise ValueError('new dense statistics generation required')
    out.mkdir(parents=True);guard=FrameRoleGuard(roles);counts=Counter();registry={};decisions=[];parents=[];source_rows=0;ordinal=0
    selected_pts={r['native_pts'] for r in read_jsonl(selected_exposure_journal)} if selected_exposure_journal is not None else None
    for dirname in dense_dirs:
        parent=Path(dirname);mp=parent/'manifest.json'
        if not mp.exists():mp=parent/'progress.json'
        raw_doc=mp.read_bytes();doc=json.loads(raw_doc);snapshot=out/f'input_snapshot_{len(parents):03d}.json';snapshot.write_bytes(raw_doc)
        parents.append({'file':str(snapshot.resolve()),'sha256':file_sha256(snapshot),'published_source_path':str(mp),'snapshot_immutable':True})
        chunks=doc.get('chunks')
        if chunks is None:
            chunks=[{'chunk':0,'files':[{'path':str(parent/name),'sha256':file_sha256(parent/name)} for name in ['rows.jsonl.gz','profiles.npz','profile_index.json']]}]
            write_json(out/f'monolithic_input_{len(parents)-1:03d}.json',chunks)
        for chunk in chunks:
            refs={Path(x.get('path',x.get('file'))).name:x for x in chunk['files']}
            paths={}
            for name in ['rows.jsonl.gz','profiles.npz','profile_index.json']:
                ref=refs[name];p=Path(ref.get('path',ref.get('file')));p=p if p.is_absolute() else parent/p
                if file_sha256(p)!=ref['sha256']:raise ValueError('immutable dense source chunk changed')
                paths[name]=p
            rows=list(read_jsonl(paths['rows.jsonl.gz']));source_rows+=len(rows)
            with np.load(paths['profiles.npz'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            profiles=defaultdict(list)
            for x in json.loads(paths['profile_index.json'].read_text(encoding='utf8')):profiles[x['source_row_id']].append(x)
            families=defaultdict(list)
            for r in rows:
                rid=r['source_row_id'];rec={'source_row_id':rid,'source_family_id':r['source_family_id'],'native_pts':r['native_pts']}
                guard.assert_training_pixels([r['native_pts']],shot_id=r['shot_id'])
                duplicate_selected=(r['native_pts'] in selected_pts) if selected_pts is not None else (r.get('anchor_source_row_id') is not None and r['native_pts']==r.get('anchor_pts'))
                if duplicate_selected:
                    rec.update(status='pending_shared_selected_exposure',reason='same native exposure as selected anchor; must be joint source union, not an extra independent family');decisions.append(rec);counts['selected_exposure_joint_covariance_pending']+=1;continue
                if r.get('point_correspondence'):
                    rec.update(status='pending_two_direction_entity_covariance');decisions.append(rec);counts['entity_two_direction_pending']+=1;continue
                pieces=profiles.get(rid,[]);operator=r['measurement_operator'];normal=np.asarray(r['source_measurement_audit'].get('normal'),float)
                if not pieces or normal.shape!=(2,) or not np.isfinite(normal).all():
                    rec.update(status='pending_missing_actual_profile');decisions.append(rec);counts['actual_profile_missing']+=1;continue
                try:
                    errors=[]
                    for p in pieces:
                        i=p['profile_index'];shift=_shifts(arrays['intensity'][i],arrays['normal_offsets']);t=arrays['tangent_offsets'];A=np.c_[np.ones(5),t,t*t]
                        errors.append(shift-A@np.linalg.lstsq(A,shift,rcond=None)[0])
                    if operator=='paint_center' and len(errors)!=2:raise StatisticsUnavailable('width center needs both actual material flank profiles')
                    e=np.mean(errors,axis=0);normal/=np.linalg.norm(normal)
                except (ValueError,StatisticsUnavailable) as exc:
                    rec.update(status='pending_profile_support',reason=str(exc));decisions.append(rec);counts['actual_profile_domain_pending']+=1;continue
                families[r['source_family_id']].append((r,e,normal,pieces))
            for family,members in families.items():
                rr=[x[0] for x in members];ids=[r['source_row_id'] for r in rr];normals=np.asarray([x[2] for x in members]);F=np.asarray([x[1] for x in members])/np.sqrt(2.)
                rec={'source_group_id':family,'row_ids':ids,'native_pts':rr[0]['native_pts'],'shot_id':rr[0]['shot_id']}
                try:Wn,proof=supported_factor_whitening(F)
                except (ValueError,StatisticsUnavailable) as exc:
                    rec.update(status='pending_no_supported_empirical_profile_covariance',reason=str(exc));decisions.append(rec);counts['empirical_covariance_pending']+=1;continue
                W=np.empty((len(Wn),2*len(rr)));W[:,0::2]=Wn*normals[:,0];W[:,1::2]=Wn*normals[:,1]
                path=out/f'group_{ordinal:06d}_W.f64';W.astype('<f8').tofile(path);ordinal+=1
                current={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'current_uv':r['current_uv'],'source_row_id':r['source_row_id'],'measurement_operator':r['measurement_operator']}) for r in rr}
                rec.update(status='empirical_profile_geometry_initialization_W_ready',rank=len(W),W={'file':str(path.resolve()),'sha256':file_sha256(path),'dtype':'<f8','shape':list(W.shape)},
                    statistic_kind='initialization_surrogate',original_numerical_scheme='paired_actual_five_tangent_profiles_empirical_source_geometry',numeric_whitening_verified=True,
                    current_measurements_sha256=current,
                    source_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'source_uv':r['current_uv'],'source_row_id':r['source_row_id']}) for r in rr},
                    original_source_family_sha256=object_sha256([{'native_pts':r['native_pts'],'source_uv':r['current_uv'],'source_row_id':r['source_row_id']} for r in rr]),
                    source_uv_definition='first actual current native profile measurement in this dense generation; not anchor click; navigation seed independent',
                    proof=proof,noise_calibration_status='empirical source-profile geometry only; physical noise and cross-family/source-selection nuisance pending',
                    current_source_mean_changed=False,representatives_assumed_independent=False,PSD_floor_or_ridge_used=False,
                    actual_profiles={'file':str(paths['profiles.npz']),'sha256':refs['profiles.npz']['sha256'],'profile_indices_by_row':{r['source_row_id']:[p['profile_index'] for p in x[3]] for r,x in zip(rr,members)}},
                    raw_dense_rows={'file':str(paths['rows.jsonl.gz']),'sha256':refs['rows.jsonl.gz']['sha256']},
                    assumptions={'five_profiles':'paired actual native tangent offsets [-2,-1,0,1,2]; quadratic source-only curvature removed, 2 empirical residual degrees',
                        'within_family_correlation':'whole source family uses the same five actual tangent translations; unsupported null modes receive no information',
                        'source_operator':'frozen recorded channel; finite normal peak window; no image world model read',
                        'not_full_physical_noise_likelihood':True},anchor_source_row_ids=[r.get('anchor_source_row_id') for r in rr])
                registry[family]=rec;decisions.append(rec);counts['groups_ready']+=1;counts['represented_rows']+=len(rr);counts['rank']+=len(W)
            write_json(out/'progress.json',{'source_rows':source_rows,'counts':dict(counts),'immutable_chunks':len(parents),'last_chunk':chunk['chunk']})
    write_json(out/'statistics_registry.json',registry);write_json(out/'decisions.json',decisions)
    result={'schema':'rinkfit.dense_empirical_profile_geometry_statistics.v1','source_rows':source_rows,'counts':dict(counts),'source_generations':parents,
        'source_mean_changed':False,'selected_anchor_exposures_recounted':False,
        'selected_exposure_journal':None if selected_exposure_journal is None else {'file':str(selected_exposure_journal),'sha256':file_sha256(selected_exposure_journal)},'formal_noise_or_accuracy_claimed':False,'solver_admission_claimed':False,'PSD_floor_or_ridge_used':False}
    write_json(out/'manifest.json',result);return result
