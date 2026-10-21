"""Restore only the original S01 source/complete primal from a verified full CP.

No old AL state, later-shot source policy, or later-shot nuisance is resumed.
The existing native baseline auxiliary consumer installs the exact active rows.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
from .accepted_aux_prefix import AUX,_prefix_hash,_digest,_runtime_query_digest,_runtime_group_digest
from .enriched_aux_prefix_v1 import _metadata,objective_config_identity
from .enriched_checkpoint_v1 import verify_source_enriched_checkpoint
from .r_inner_prefix_v1 import array,checked
from .source_math_identity_v1 import require
from .io_utils import file_sha256
from .packet_storage import write_packet
from .source_quality_compare_v1 import _camera
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene

N_S01=182017

def ref(p):
    return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def extract_s01_prefix(source_packet,state_dir,metadata_file,configuration_path,output):
    src=Path(source_packet).resolve(); state=Path(state_dir).resolve();out=Path(output).resolve()
    require(not out.exists(),'immutable S01 extraction output exists')
    seen=set(); old=read(src); n=old['row_count']; N=N_S01
    meta,cfg=_metadata(metadata_file,'native_initialization_snapshot',configuration_path,
                       state_dir=state,source_packet=src)
    require(objective_config_identity(configuration_path,old,src.parent,seen)==meta['configuration_sha256'],
            'actual source objective config digest')
    modelpath=state/'model.json'; model=read(modelpath)
    require(file_sha256(modelpath)==meta['model_sha256'],'full CP model hash')
    manifest=read(state/'explicit_nuisance/manifest.json')
    require(manifest['raw_rows']==n and manifest['packet_sha256']==file_sha256(src),'full explicit CP source rows')
    aux={}
    for k,(name,dtype) in AUX.items():
        f=state/'explicit_nuisance'/name
        require(file_sha256(f)==manifest['files'][name]['sha256'] and f.stat().st_size==n*np.dtype(dtype).itemsize,
                'complete CP auxiliary '+k)
        aux[k]=np.memmap(f,mode='r',dtype=dtype,shape=(n,))
        require(np.isfinite(aux[k]).all(),'finite CP auxiliary '+k)
    A={k:array(old,k,src.parent,seen) for k in old['arrays']}
    active=(A['admission']==1)&(A['fit_role']==1)
    require(old['shot_ids'][0]=='s01' and np.all(A['shot_index'][:N]==0)
        and np.all(A['shot_index'][N:]!=0),'exact contiguous original S01 cohort')
    require(set(old['source_dataset_ids'][:N])=={'s01_original'},'original S01 dataset identity')
    require(int(active[:N].sum())==181840,'original active S01 source membership changed')
    print('verified arrays; checking complete source-enriched CP',flush=True)
    proof=verify_source_enriched_checkpoint(state/'engineering_checkpoint.json',meta,model,old,aux,active,
        [q['row_index']for q in old['queries']],
        [int(i)for i in np.flatnonzero(active)if A['kind'][i]!=1],
        base=src.parent,verified_dependency_cache=seen)
    print('complete CP token verified; extracting S01',flush=True)
    out.mkdir(parents=True); arrays={}; hashes={}
    for k,a in A.items():
        require(a.shape[0]==n,'non-row source array requires explicit extraction: '+k)
        v=aux[k][:N] if k in AUX else a[:N]
        f=out/(k+'.bin');np.ascontiguousarray(v).tofile(f)
        arrays[k]=dict(file=str(f),sha256=file_sha256(f),dtype=v.dtype.str,shape=list(v.shape),layout='C')
        hashes[k]=_prefix_hash(v)
    groups=[]
    for g in old['groups']:
        members=g['rows'];inside=[i<N for i in members]
        if not any(inside):continue
        require(all(inside),'W group crosses S01/other-shot boundary')
        q=copy.deepcopy(g)
        if isinstance(q['W'],dict):q['W']['file']=str(checked(q['W'],src.parent,seen))
        groups.append(q)
    queries=[q for q in old['queries']if q['row_index']<N]
    strokes=[]
    for q in old.get('stroke_pairs',[]):
        inside=[i<N for i in q['rows']]
        if any(inside):require(all(inside),'width pair crosses S01 boundary');strokes.append(q)
    domains=copy.deepcopy(old.get('source_station_domains',{}))
    if 'rows'in domains:domains['rows']=[q for q in domains['rows']if q['row_index']<N]
    # Only the consumed source/operator ABI is carried into the new packet.
    keep=('schema','independent_statistic_kind','material_binding')
    packet={k:copy.deepcopy(old[k])for k in keep if k in old}
    packet.update(row_count=N,shot_ids=old['shot_ids'][:4],arrays=arrays,
        source_dataset_ids=old['source_dataset_ids'][:N],source_row_ids=old['source_row_ids'][:N],
        source_family_ids=old['source_family_ids'][:N],groups=groups,queries=queries,
        independent_rows=[i for i in old.get('independent_rows',[])if i<N],stroke_pairs=strokes,
        allowed_paths={str(k):v for k,v in old.get('allowed_paths',{}).items()if int(k)<N},
        source_station_domains=domains,active_rows=int(active[:N].sum()),
        new_source_start=N,current_original_W_rank=sum(int(g['rank'])for g in groups)+2*len([i for i in old.get('independent_rows',[])if i<N]),
        source_manifests=[ref(src)],source_clock=copy.deepcopy(model['source_clock']),
        physical_source_audit_complete=False,formal_new_source_statistical_optimum_claimed=False,
        initialization_objective_only=True,source_noise_statistics_supplied_for_appended_rows=False,
        accepted_physical_baseline=False)
    packet['joint_statistics_rank']=packet['current_original_W_rank']
    packet['new_source_statistics_rank']=0
    packet['first4_s01_complete_cp_provenance']=dict(full_source_packet=ref(src),full_model=ref(modelpath),
        full_checkpoint=ref(state/'engineering_checkpoint.json'),metadata=ref(metadata_file),configuration=cfg,
        state_token=proof['state_token'],old_AL_duals_or_objective_resumed=False,
        later_shot_source_policies_inherited=False)
    partition=np.zeros(N,np.int16)
    partition[packet['independent_rows']]+=1
    for g in groups:partition[g['rows']]+=1
    require(np.array_equal(partition,active[:N].astype(np.int16)),'complete original active W partition')
    write_packet(out/'packet.json',packet)
    proof.update(source_packet_ref=ref(src),model_ref=ref(modelpath),checkpoint_ref=ref(state/'engineering_checkpoint.json'),
                 metadata_ref=ref(metadata_file),configuration_ref=cfg)
    write(out/'full_checkpoint_verification.json',proof)
    group_hash,group_count=_runtime_group_digest(packet,out,seen)
    extraction=dict(schema='rinkfit.first4_original_s01_complete_cp_prefix.v1',packet=ref(out/'packet.json'),
        source_packet=ref(src),source_model=ref(modelpath),source_checkpoint=ref(state/'engineering_checkpoint.json'),
        full_checkpoint_verification=ref(out/'full_checkpoint_verification.json'),
        original_rows=N,active_rows=int(active[:N].sum()),inactive_rows=N-int(active[:N].sum()),
        four_auxiliaries_from_explicit_CP=True,source_arrays_sha256=hashes,
        original_queries_sha256=_runtime_query_digest(queries),source_group_digest=group_hash,source_group_count=group_count,
        current_original_W_rank=packet['current_original_W_rank'],original_group_count=len(groups),
        original_independent_rows=len(packet['independent_rows']),
        original_UV_ID_PTS_E_and_W_unchanged=True,all_source_indices_identity_map=True,
        old_AL_duals_resumed=False,later_shot_source_policies_inherited=False,physical_accuracy_accepted=False)
    write(out/'extraction.json',extraction)
    return extraction


def bind_s01_aux_import(final_packet,final_model,output,*,prefix_dir=None):
    src=Path(final_packet).resolve();mp=Path(final_model).resolve();out=Path(output).resolve()
    pref=Path(prefix_dir).resolve()if prefix_dir else src.parent/'s01_prefix'
    require(not out.exists(),'immutable S01 auxiliary binding output exists')
    ep=pref/'extraction_v2.json' if (pref/'extraction_v2.json').exists() else pref/'extraction.json'
    extraction=read(ep);proof=read(checked(extraction['full_checkpoint_verification'],pref,set()))
    baseline=read(checked(extraction['source_model'],pref,set())); model=read(mp)
    old=read(checked(extraction['packet'],pref,set())); packet=read(src);N=N_S01; seen=set()
    require(packet['row_count']>=N,'new packet omits original S01 source')
    for k in ('source_dataset_ids','source_row_ids','source_family_ids'):
        require(packet[k][:N]==old[k],'same-ID original source prefix '+k)
    A={k:array(packet,k,src.parent,seen)for k in packet['arrays']}
    for k in old['arrays']:
        require(_prefix_hash(A[k][:N])==extraction['source_arrays_sha256'][k],
                'original S01 source or CP auxiliary changed: '+k)
    require([q for q in packet['queries']if q['row_index']<N]==old['queries'],'original finite Query exact')
    gs=[g for g in packet['groups']if any(i<N for i in g['rows'])]
    gdoc=dict(old,groups=gs,independent_rows=[i for i in packet.get('independent_rows',[])if i<N])
    require(_runtime_group_digest(gdoc,src.parent,seen)==(extraction['source_group_digest'],extraction['source_group_count']),
            'original exact W/group partition changed')
    require({k:v for k,v in packet.get('allowed_paths',{}).items()if int(k)<N}==old.get('allowed_paths',{}),'original finite paths')
    domains=copy.deepcopy(packet.get('source_station_domains',{}))
    if 'rows' in domains: domains['rows']=[q for q in domains['rows']if q['row_index']<N]
    old_domains=old.get('source_station_domains',{})
    require({k:v for k,v in domains.items()if k!='source_refs'}=={k:v for k,v in old_domains.items()if k!='source_refs'},'original finite station domains')
    old_refs=old_domains.get('source_refs',[])
    require(domains.get('source_refs',[])[:len(old_refs)]==old_refs,'original finite-domain source references')
    require([q for q in packet.get('stroke_pairs',[])if any(i<N for i in q['rows'])]==old.get('stroke_pairs',[]),'original width pair topology')
    active=(A['admission'][:N]==1)&(A['fit_role'][:N]==1);rows=np.flatnonzero(active)
    require(len(rows)==extraction['active_rows'],'original active source membership changed')
    # Re-evaluate all actually used materials/camera dependencies, with the
    # serialized packed PTZ chart and actual reference angles in both models.
    oldA={k:array(old,k,pref,seen)for k in ('material_index','path','station','row','uv','profile_normal','kind','native_pts')}
    bscene=NativeScene(as_event_view(baseline,'s01'));scene=NativeScene(as_event_view(model,'s01'))
    bm=oldA['material_index'][rows];cm=A['material_index'][rows];paths=A['path'][rows];stations=A['station'][rows]
    for a,b,p in zip(bm,cm,paths):
        require(baseline['materials'][int(a)]['label']==model['materials'][int(b)]['label'] and
                baseline['materials'][int(a)]['paths'][int(p)]['id']==model['materials'][int(b)]['paths'][int(p)]['id'],
                'same-ID original material/path identity changed')
    X0,ok0,_=bscene.world_rows(bm,paths,stations);X,ok,_=scene.world_rows(cm,paths,stations)
    require(ok0.all()and ok.all()and np.array_equal(X,X0),'S01 source material XYZ changed')
    bcam=_camera(baseline,'s01');cam=_camera(model,'s01');rr=A['row'][rows]
    t=(A['native_pts'][rows]-model['shots']['s01']['time_origin_pts'])/12800.+cam.tau*(rr/1079.-.5)
    t0=(oldA['native_pts'][rows]-baseline['shots']['s01']['time_origin_pts'])/12800.+bcam.tau*(rr/1079.-.5)
    p0=bcam.at(X0,t0);pred=cam.at(X,t)
    require(p0['valid'].all()and pred['valid'].all(),'S01 restored projection domain failed')
    maximum=float(np.max(np.abs(pred['uv']-p0['uv'])))
    require(maximum<=2e-8,'original S01 complete CP prediction changed')
    error=pred['uv']-A['uv'][rows];euclid=np.linalg.norm(error,axis=1);op=euclid.copy()
    qset={q['row_index']for q in old['queries']};q=np.asarray([int(i)in qset for i in rows])
    op[q]=np.abs(np.sum(error[q]*A['profile_normal'][rows[q]],axis=1))
    labels=np.asarray([model['materials'][int(i)]['label']for i in cm]);a2=(np.char.startswith(labels,'Ad2.'))&(A['kind'][rows]==1)
    e2=euclid[a2]
    out.mkdir(parents=True);files={}
    vals={'rows.u64':rows.astype('<u8'),'original_rows.u64':rows.astype('<u8'),
          'station.f64':A['station'][rows].astype('<f8'),'path.u32':paths.astype('<u4'),
          'row.f64':rr.astype('<f8'),'alpha.f64':A['alpha'][rows].astype('<f8')}
    for name,v in vals.items():
        f=out/name;v.tofile(f);files[name]=dict(ref(f),dtype=v.dtype.str,shape=list(v.shape))
    report=dict(schema='rinkfit.exact_native_baseline_auxiliary_import.v1',current_packet=ref(src),model=ref(mp),
        baseline_model=extraction['source_model'],complete_baseline_state=extraction['full_checkpoint_verification'],
        source_key_fields=['source_dataset_id','shot_id','native_pts','source_row_id'],original_rows=N,
        bijective_original_prefix_rows=N,preserved_active_rows=len(rows),files=files,
        all_original_UV_ID_PTS_material_kind_normal_Query_exact=True,
        maximum_saved_d78_prediction_change_px=maximum,
        RS_equality_max_px=float(np.max(np.abs(rr-pred['uv'][:,1]))),
        overall_original_operator_RMS_px=float(np.sqrt(np.mean(op*op))),
        overall_original_euclidean_RMS_px=float(np.sqrt(np.mean(euclid*euclid))),
        Ad2_same623=dict(n=len(e2),RMS_px=float(np.sqrt(np.mean(e2*e2)))if len(e2)else None),
        source_membership_UV_W_E_or_qualification_changed=False,v98_auxiliary_state_loaded=False,
        original_source_and_explicit_full_cp_proof=ref(ep),
        fresh_recapture_must_skip_preserved_rows=True,old_AL_duals_resumed=False,
        source_qualification_issued=False,physical_accuracy_accepted=False)
    write(out/'manifest.json',report);return report
