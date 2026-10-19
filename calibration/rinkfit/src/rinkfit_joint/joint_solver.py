"""Reproducible CPU joint physical backend build and immutable run driver.
The full-W nonlinear initializer is explicit about missing formal Phase I/II certificates.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def _source_membership_fingerprint(packet):
    value='rinkfit.joint_active_source_watch.v1|'+str(packet.get('row_count',0))
    for key in ('uv','native_pts','shot_index','kind','material_index','admission','fit_role'):
        value+='|'+key+'='+packet.get('arrays',{}).get(key,{}).get('sha256','missing')
    for ref in packet.get('source_manifests',[]):value+='|source_manifest='+ref.get('sha256','missing')
    return hashlib.sha256(value.encode()).hexdigest()

def assert_source_identity_current(packet,*,return_verified_packet=False):
    """Check this source generation and explicitly same-membership aux wrappers.

    Historical parent references do not participate. Changed membership or a new
    source certificate invalidates stale inherited watch entries.
    """
    path=Path(packet).resolve()
    # The direct generation recall is known before any large packet hydration.
    # Same-membership auxiliary watches below still require their exact hashes.
    for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
        marker=path.parent/name
        if marker.exists():
            try:reason=json.loads(marker.read_text(encoding='utf8'))
            except (ValueError,OSError):reason='source recall marker exists; see the named file'
            raise ValueError('Source physical identity certificate has been recalled: '+str(marker)+'; '+json.dumps(reason,ensure_ascii=False))
    doc=json.loads(path.read_text(encoding='utf8'));fingerprint=_source_membership_fingerprint(doc);paths=[path]
    for ref in doc.get('source_identity_watchlist',[]):
        if ref.get('source_membership_fingerprint')!=fingerprint:continue
        parent=Path(ref['file']).resolve()
        if sha(parent)!=ref['sha256']:raise ValueError('changed same-source qualification dependency '+str(parent))
        paths.append(parent)
    for source in paths:
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            marker=source.parent/name
            if marker.exists():
                try:reason=json.loads(marker.read_text(encoding='utf8'))
                except (ValueError,OSError):reason='source recall marker exists; see the named file'
                raise ValueError('Source physical identity certificate has been recalled: '+str(marker)+'; '+json.dumps(reason,ensure_ascii=False))
    from .board_source_qualification_guard import assert_board_material_source_qualification
    assert_board_material_source_qualification(doc,path)
    return doc if return_verified_packet else None

def _inherit_source_recall_marker(source_packet,destination_packet):
    source=Path(source_packet).resolve();destination=Path(destination_packet).resolve();parent=json.loads(source.read_text(encoding='utf8'));doc=json.loads(destination.read_text(encoding='utf8'));fingerprint=_source_membership_fingerprint(doc)
    if fingerprint!=_source_membership_fingerprint(parent):raise ValueError('auxiliary-only rebind changed source qualification fingerprint')
    watches=[dict(file=str(source),sha256=sha(source),source_membership_fingerprint=fingerprint,reason='identical source membership/UV/material/clock/certificates; only numerical auxiliaries rebound')]
    watches += [r for r in parent.get('source_identity_watchlist',[]) if r.get('source_membership_fingerprint')==fingerprint]
    unique={}
    for ref in watches:unique[(ref['file'],ref['sha256'])]=ref
    doc['source_identity_watchlist']=list(unique.values());destination.write_text(json.dumps(doc,indent=2),encoding='utf8')
    for ref in doc['source_identity_watchlist']:
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            marker=Path(ref['file']).parent/name
            if marker.exists():
                dest=destination.parent/name
                if not dest.exists():
                    data=json.loads(marker.read_text(encoding='utf8'));data['direct_recalled_parent_packet']=ref;data['packet_sha256']=sha(destination);data['state_rebind_does_not_reaudit_source_identity']=True
                    dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def build(output, *, ceres_install=None):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('immutable build exists: '+str(out))
    out.mkdir(parents=True);base=Path(__file__).resolve().parents[2];cpp=base/'cpp'
    for folder in ('include','src'):(out/folder).mkdir()
    shutil.copytree(cpp/'include/rinkfit',out/'include/rinkfit')
    sources=[]
    for name in ('event_model.cpp','joint_model.cpp','joint_packet.cpp','joint_builder.cpp','joint_capture.cpp','joint_al.cpp','joint_recovery.cpp','joint_gpu_cache.cpp','joint_gpu_driver.cpp','augmented_lagrangian.cpp','ceres_backend.cpp','checkpoint.cpp','joint_fit.cpp'):
        q=out/'src'/name;shutil.copyfile(cpp/'src'/name,q);sources.append(q)
    # Preserve the original Windows toolchain setup in the event-owned source tree.
    driver=Path(__file__).resolve().parents[4]/'events/ankara_2026/tools/build_event_backend.py'
    setup={};save=sys.argv
    try:
        sys.argv=[str(driver),str(out)]
        exec(compile(driver.read_text(encoding='utf8').split('sources=')[0],str(driver),'exec'),setup)
    finally:
        sys.argv=save
    flags=['/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES'];inc=[out/'include',*setup['inc'][1:]];libs=[setup['deps']/(x+'/Library/lib/'+y+'.lib') for x,y in [('ceres-solver','ceres'),('glog','glog'),('gflags','gflags')]]
    candidate=None
    if ceres_install is not None:
        candidate=Path(ceres_install).resolve()
        for f in (candidate/'include/ceres/solver.h',candidate/'include/ceres/internal/config.h',candidate/'lib/ceres.lib',candidate/'bin/ceres.dll'):
            if not f.is_file():raise FileNotFoundError(f)
        config=(candidate/'include/ceres/internal/config.h').read_text(encoding='utf8')
        if any(line.strip().startswith('#define CERES_NO_CUDA') for line in config.splitlines()):raise ValueError('explicit candidate library was built without CUDA')
        inc=[candidate/'include',*[q for q in inc if 'ceres-solver' not in str(q)]]
        libs[0]=candidate/'lib/ceres.lib'
        cuda_runtime=Path(r'D:/Program Files/CUDA/bin/x64')
        if not cuda_runtime.is_dir():raise FileNotFoundError(cuda_runtime)
        setup['bins']=[candidate/'bin',cuda_runtime,*setup['bins']]
        setup['env']['PATH']=';'.join(map(str,setup['bins']))+';'+setup['env'].get('PATH','')
    command=[str(setup['cl']),*flags,*['/I'+str(x) for x in inc],*map(str,sources),'/Fo:'+str(out)+os.sep,'/Fe:'+str(out/'joint_fit.exe'),*map(str,libs)]
    r=subprocess.run(command,cwd=out,env=setup['env'],capture_output=True,text=True,errors='replace');(out/'build.log').write_text(r.stdout+r.stderr,encoding='utf8')
    recipe=dict(schema='rinkfit.joint_backend_build.v1',command=command,source_files={str(q.relative_to(out)):sha(q) for q in [*sources,*list((out/'include').rglob('*.hpp'))]},runtime_bin_paths=list(map(str,setup['bins'])),returncode=r.returncode)
    if candidate is not None:recipe['optional_CUDA_Ceres_install']=dict(path=str(candidate),library_sha256=sha(candidate/'lib/ceres.lib'),runtime_DLL_sha256=sha(candidate/'bin/ceres.dll'),configuration_sha256=sha(candidate/'include/ceres/internal/config.h'),build_provenance_reference=str(candidate/'build_recipe.json'))
    if not r.returncode:recipe['executable_sha256']=sha(out/'joint_fit.exe')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf8')
    if r.returncode:raise RuntimeError('joint backend build failed: '+str(out/'build.log'))
    print(json.dumps(dict(executable=str(out/'joint_fit.exe'),sha256=recipe['executable_sha256'])))
    return out/'joint_fit.exe'
def execute(executable,model,packet,output,iterations=0,threads=8,fixed_prefixes=None,fixed_shot_nuisance=None,recapture_new_sources=False,observed_row_time_new_sources=False):
    if iterations or recapture_new_sources:assert_source_identity_current(packet)
    exe=Path(executable).resolve();recipe=json.loads((exe.parent/'recipe.json').read_text(encoding='utf8'))
    if sha(exe)!=recipe['executable_sha256']:raise ValueError('compiled backend SHA changed')
    env=os.environ.copy();env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','');env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('immutable generation exists')
    log=out.parent/(out.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    command=[str(exe),str(Path(model).resolve()),str(Path(packet).resolve()),str(out),str(iterations),str(threads)]
    if fixed_prefixes or fixed_shot_nuisance or recapture_new_sources or observed_row_time_new_sources:
        stage=out.parent/(out.name+'.stage.json');stage.write_text(json.dumps(dict(fix_prefixes=fixed_prefixes or [],fix_shot_nuisance=fixed_shot_nuisance or [],recapture_new_sources=recapture_new_sources,observed_row_time_new_sources=observed_row_time_new_sources),indent=2),encoding='utf8');command.append(str(stage))
    with log.open('w',encoding='utf8') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    result=dict(schema='rinkfit.joint_backend_execution.v1',command=command,backend_sha256=sha(exe),model_sha256=sha(model),packet_sha256=sha(packet),returncode=r.returncode,log=str(log),formal_Phase_I_II_passed=False)
    if out.exists():(out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result));return r.returncode

def execute_explicit(executable,model,packet,output,*,config=None,geometry_initializer=False,audit_only=False,engineering_recovery=False):
    """Run the exact shared explicit RS/query graph; certification comes from receipts."""
    if not audit_only:assert_source_identity_current(packet)
    exe=Path(executable).resolve();recipe=json.loads((exe.parent/'recipe.json').read_text(encoding='utf8'))
    if sha(exe)!=recipe['executable_sha256']:raise ValueError('compiled backend SHA changed')
    env=os.environ.copy();env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','');env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('immutable explicit generation exists')
    cfg=json.loads(Path(config).read_text(encoding='utf8')) if config else {}
    if 'screen_grid_initialization_only' in cfg and not isinstance(cfg['screen_grid_initialization_only'],bool):
        raise ValueError('screen_grid_initialization_only must be an explicit JSON boolean')
    if cfg.get('screen_grid_manifest') and cfg.get('screen_grid_initialization_only') is not True:
        raise ValueError('direct screen-grid input requires screen_grid_initialization_only=true; no optimizer is started implicitly')
    if cfg.get('screen_grid_initialization_only') is True:
        if not engineering_recovery or geometry_initializer or audit_only:
            raise ValueError('screen-grid initialization requires its explicit engineering-recovery stage')
        if not cfg.get('screen_grid_manifest'):
            raise ValueError('screen_grid_initialization_only requires screen_grid_manifest')
        from .screen_grid_state import initialize_screen_grid
        return initialize_screen_grid(cfg['screen_grid_manifest'],model,packet,out,config_file=config)
    log=out.parent/(out.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    if engineering_recovery and (geometry_initializer or audit_only):raise ValueError('engineering recovery is an independent production objective')
    if audit_only and geometry_initializer:raise ValueError('audit mode reads native state and does not run an initialization objective')
    mode='--engineering-recovery' if engineering_recovery else '--formal-audit' if audit_only else ('--geometry-phr' if geometry_initializer else '--formal')
    command=[str(exe),mode,str(Path(model).resolve()),str(Path(packet).resolve()),str(out)]
    if config:command.append(str(Path(config).resolve()))
    from .feedback_checkpoint import CheckpointObserver
    with CheckpointObserver(out, packet) as observer:
        with log.open('w',encoding='utf8') as stream:r=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if engineering_recovery and config:
        cfg=json.loads(Path(config).read_text(encoding='utf8'))
        if cfg.get('checkpoint_source_review') and observer.consumed_task_documents:
            from .feedback_dispatch import dispatch_checkpoint_reviews
            review=cfg['checkpoint_source_review']
            dispatch_checkpoint_reviews(observer.consumed_task_documents,review['run'],out/'source_reviews',device=review.get('device','cuda'))
    result=dict(schema='rinkfit.joint_explicit_backend_execution.v1',command=command,backend_sha256=sha(exe),model_sha256=sha(model),packet_sha256=sha(packet),returncode=r.returncode,log=str(log),geometry_initialization_only=bool(geometry_initializer),engineering_recovery=bool(engineering_recovery),objective_id='engineering_recovery_native_pixel_v1' if engineering_recovery else None,certificate_in_native_and_formal_receipts=not engineering_recovery)
    if out.exists():(out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result));return r.returncode

def rebind_current_auxiliaries(packet_path,state_path,output_path,*,state_packet_path=None):
    """Version a full packet with all four current nuisances, keeping source UV/W."""
    import numpy as np
    src=Path(packet_path).resolve();state=Path(state_path).resolve();out=Path(output_path).resolve()
    if out.exists():raise FileExistsError('immutable packet generation exists: '+str(out))
    packet=json.loads(src.read_text(encoding='utf8'));report=json.loads((state/'value_report.json').read_text(encoding='utf8'));n=int(packet['row_count'])
    checked_dependencies=set()
    def absolute(d):
        q=Path(d['file']);q=q if q.is_absolute() else src.parent/q;q=q.resolve()
        key=(str(q),d['sha256'])
        if key not in checked_dependencies:
            if sha(q)!=d['sha256']:raise ValueError('changed parent packet dependency: '+str(q))
            checked_dependencies.add(key)
        return {**d,'file':str(q)}
    packet['arrays']={k:absolute(d) for k,d in packet['arrays'].items()}
    for g in packet['groups']:
        if isinstance(g['W'],dict):g['W']=absolute(g['W'])
    for ref in packet.get('source_manifests',[]):ref.update(absolute(ref))
    state_n=int(report['files']['station.f64']['size_bytes'])//8
    if state_n>n:raise ValueError('current state is longer than packet')
    new_query_rows=sorted(int(q['row_index'])for q in packet.get('queries',[])if int(q['row_index'])>=state_n)
    prefix_proof=None
    if state_n<n:
        if state_packet_path is None:raise ValueError('appended packet requires the exact previous state packet for prefix verification')
        old_path=Path(state_packet_path).resolve();old=json.loads(old_path.read_text(encoding='utf8'))
        if int(old['row_count'])!=state_n:raise ValueError('state packet raw row count mismatch')
        for name in ('source_dataset_ids','source_row_ids'):
            if packet[name][:state_n]!=old[name]:raise ValueError('changed raw prefix source identity '+name)
        for name in ('native_pts','shot_index','uv','kind','material_index','E'):
            d=old['arrays'][name];q=Path(d['file']);q=q if q.is_absolute() else old_path.parent/q
            if sha(q)!=d['sha256']:raise ValueError('changed previous source prefix dependency')
            a=np.fromfile(q,d['dtype'],count=int(np.prod(d['shape'])),offset=d.get('offset_bytes',0)).reshape(d['shape']);dnew=packet['arrays'][name];b=np.fromfile(dnew['file'],dnew['dtype'],count=int(np.prod(dnew['shape'])),offset=dnew.get('offset_bytes',0)).reshape(dnew['shape'])
            if not np.array_equal(a,b[:state_n]):raise ValueError('changed raw prefix source array '+name)
        prefix_proof=dict(previous_packet=dict(file=str(old_path),sha256=sha(old_path)),inherited_rows=state_n,new_appended_rows=n-state_n,raw_source_identity_UV_E_material_clock_prefix_verified=True)
    files={}
    for name,file,dtype in (('station','station.f64','<f8'),('path','path.u32','<u4'),('row','row.f64','<f8'),('alpha','alpha.f64','<f8')):
        q=state/file;expected=report['files'][file]['sha256']
        if sha(q)!=expected:raise ValueError('changed current full-state auxiliary: '+str(q))
        v=np.fromfile(q,dtype)
        if v.shape!=(state_n,) or not np.isfinite(v).all():raise ValueError('incomplete/nonfinite current auxiliary '+name)
        if state_n<n:
            d=packet['arrays'][name];combined=np.fromfile(d['file'],d['dtype'],count=n,offset=d.get('offset_bytes',0));combined[:state_n]=v;out.parent.mkdir(parents=True,exist_ok=True);q=out.parent/('current_'+file)
            if q.exists():raise FileExistsError('immutable appended auxiliary already exists')
            combined.astype(dtype,copy=False).tofile(q);expected=sha(q)
        files[name]=dict(file=str(q),sha256=expected,dtype=dtype,shape=[n])
        packet['arrays'][name]=files[name]
    packet['current_full_state_auxiliary_binding']=dict(schema='rinkfit.joint_current_full_state_auxiliary_binding.v1',parent_packet=dict(file=str(src),sha256=sha(src)),state_report=dict(file=str(state/'value_report.json'),sha256=sha(state/'value_report.json')),state_model_sha256=report['model_sha256'],arrays=files,source_UV_W_admission_roles_changed=False,appended_prefix_verification=prefix_proof,new_queries_keep_actual_source_window_alpha_initialization=new_query_rows)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf8')
    _inherit_source_recall_marker(src,out)
    return dict(packet=str(out),sha256=sha(out),current_state=str(state),row_count=n)

def rebind_explicit_nuisance(packet_path,accepted_state,output_path):
    """Bind the complete accepted explicit state without claiming a native replay."""
    import numpy as np
    src=Path(packet_path).resolve();state=Path(accepted_state).resolve();out=Path(output_path).resolve()
    if out.exists():raise FileExistsError('immutable explicit state packet exists')
    if (state.parent/'engineering_receipt.json').exists() and not (state/'accepted_state_receipt.json').exists() and not (state/'formal_receipt.json').exists():
        from .accepted_aux_prefix import rebind_engineering_nuisance
        return rebind_engineering_nuisance(src,state,out)
    packet=json.loads(src.read_text(encoding='utf8'));receipt_path=state/'accepted_state_receipt.json';complete_checkpoint=None
    if not receipt_path.exists():
        receipt_path=state/'formal_receipt.json';value_report=json.loads((state/'value_report.json').read_text(encoding='utf8'));formal_checkpoint=state/'formal_checkpoint.json'
        if not formal_checkpoint.exists():raise ValueError('explicit nuisance binding requires an accepted receipt or a complete PHR checkpoint')
        complete_checkpoint=dict(file=str(formal_checkpoint),sha256=sha(formal_checkpoint),source_generation_receipt=dict(file=str(receipt_path),sha256=sha(receipt_path)))
    receipt=json.loads(receipt_path.read_text(encoding='utf8'))
    if complete_checkpoint is not None:
        if not receipt.get('state_token') or value_report['files']['model.json']['sha256']!=value_report['model_sha256']:raise ValueError('incomplete PHR model/receipt binding')
        receipt['model_sha256']=value_report['model_sha256']
    manifest=json.loads((state/'explicit_nuisance/manifest.json').read_text(encoding='utf8'))
    parent_sha=receipt['source_packet_sha256']
    if manifest['packet_sha256']!=parent_sha:raise ValueError('explicit nuisance and receipt belong to different source packets')
    n=int(packet['row_count']);old_n=int(manifest['raw_rows']);prefix_proof=None;new_query_rows=[]
    if n<old_n or sha(state/'model.json')!=receipt['model_sha256']:raise ValueError('incomplete or changed physical model/nuisance binding')
    if sha(src)!=parent_sha:
        parent=Path(receipt['source_packet']).resolve()
        if sha(parent)!=parent_sha:raise ValueError('changed accepted source parent packet')
        original=json.loads(parent.read_text(encoding='utf8'))
        if int(original['row_count'])!=old_n:raise ValueError('accepted parent row count mismatch')
        old_queries={int(q['row_index']):q for q in original.get('queries',[])};new_queries={int(q['row_index']):q for q in packet.get('queries',[])}
        for index,q in old_queries.items():
            if new_queries.get(index)!=q:raise ValueError('accepted genuine Query changed '+str(index))
        new_query_rows=sorted(set(new_queries)-set(old_queries))
        for name in ('source_dataset_ids','source_row_ids'):
            if packet[name][:old_n]!=original[name]:raise ValueError('changed accepted raw prefix '+name)
        for name in ('native_pts','shot_index','uv','kind','material_index','E'):
            arrays=[]
            for doc,base in ((original,parent.parent),(packet,src.parent)):
                d=doc['arrays'][name];q=Path(d['file']);q=q if q.is_absolute() else base/q
                if sha(q)!=d['sha256']:raise ValueError('changed accepted/new raw dependency '+name)
                arrays.append(np.fromfile(q,d['dtype'],count=int(np.prod(d['shape'])),offset=d.get('offset_bytes',0)).reshape(d['shape']))
            if not np.array_equal(arrays[0],arrays[1][:old_n]):raise ValueError('changed accepted raw prefix array '+name)
        prefix_proof=dict(previous_packet=dict(file=str(parent),sha256=parent_sha),inherited_rows=old_n,new_appended_rows=n-old_n,raw_source_identity_UV_E_material_clock_prefix_verified=True,membership_or_weights_are_from_new_scope=True)
    checked=set()
    def absolute(d):
        p=Path(d['file']);p=p if p.is_absolute() else src.parent/p;p=p.resolve();key=(str(p),d['sha256'])
        if key not in checked:
            if sha(p)!=d['sha256']:raise ValueError('changed source/statistical dependency')
            checked.add(key)
        return {**d,'file':str(p)}
    packet['arrays']={k:absolute(d) for k,d in packet['arrays'].items()}
    for g in packet['groups']:
        if isinstance(g['W'],dict):g['W']=absolute(g['W'])
    for ref in packet.get('source_manifests',[]):ref.update(absolute(ref))
    for name,file,dtype in (('station','station.f64','<f8'),('path','path.u32','<u4'),('row','row.f64','<f8'),('alpha','alpha.f64','<f8')):
        p=state/'explicit_nuisance'/file;d=manifest['files'][file]
        if sha(p)!=d['sha256']:raise ValueError('accepted explicit nuisance changed')
        v=np.fromfile(p,dtype)
        if v.shape!=(old_n,) or not np.isfinite(v).all():raise ValueError('incomplete explicit '+name)
        if old_n<n or (name=='alpha' and new_query_rows):
            a=packet['arrays'][name];combined=np.fromfile(a['file'],a['dtype'],count=n,offset=a.get('offset_bytes',0));new_alpha=combined[new_query_rows].copy() if name=='alpha' else None;combined[:old_n]=v
            if name=='alpha':combined[new_query_rows]=new_alpha
            out.parent.mkdir(parents=True,exist_ok=True);p=out.parent/('accepted_current_'+file)
            if p.exists():raise FileExistsError('immutable appended explicit auxiliary already exists')
            combined.astype(dtype,copy=False).tofile(p);d={'sha256':sha(p)}
        packet['arrays'][name]=dict(file=str(p),sha256=d['sha256'],dtype=dtype,shape=[n])
    packet['accepted_explicit_state_binding']=dict(schema='rinkfit.accepted_explicit_state_packet_binding.v1',parent_packet=dict(file=str(src),sha256=sha(src)),accepted_receipt=dict(file=str(receipt_path),sha256=sha(receipt_path)),complete_PHR_checkpoint=complete_checkpoint,model_sha256=receipt['model_sha256'],native_replay_pass_claimed=False,source_UV_W_E_admission_roles_changed=False,appended_prefix_verification=prefix_proof)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,indent=2),encoding='utf8')
    _inherit_source_recall_marker(src,out)
    return dict(packet=str(out),sha256=sha(out),model=str(state/'model.json'),model_sha256=receipt['model_sha256'])

def independent_replay(model_path,packet_path,state_path,*,output=None):
    """All admitted UV use independent NativeScene/NativeCamera and fresh roots."""
    import numpy as np
    from .joint_model import as_event_view,graph_from_joint,validate_physical_domain
    from .camera_graph import CameraGraph
    from rinkfit_start.native_scene import NativeScene
    model=json.loads(Path(model_path).read_text(encoding='utf8'));packet=json.loads(Path(packet_path).read_text(encoding='utf8'));state=Path(state_path);base=Path(packet_path).parent
    def ar(name,dtype):
        d=packet['arrays'][name];q=base/d['file']
        if sha(q)!=d['sha256']:raise ValueError('packet array hash mismatch '+name)
        return np.fromfile(q,dtype=dtype,offset=d.get('offset_bytes',0),count=int(np.prod(d['shape']))).reshape(d['shape'])
    if 'arrays' in packet:
        pts=ar('native_pts','<i8');uv=ar('uv','<f8');mat=ar('material_index','<u4');branch=ar('path','<u4');admit=ar('admission','u1');role=ar('fit_role','u1');active=(admit==1)&(role==1);sids=np.asarray(packet['shot_ids'])[ar('shot_index','<u2')]
    else:
        rows=packet['rows'];pts=np.asarray([r['native_pts'] for r in rows],np.int64);uv=np.asarray([r['source_uv'] for r in rows]);mat=np.asarray([r['material_index'] for r in rows]);branch=np.asarray([r['path'] for r in rows]);active=np.asarray([r.get('active',False) for r in rows]);sids=np.asarray([r['shot_id'] for r in rows])
    n=len(pts);s=np.fromfile(state/'station.f64','<f8');path=np.fromfile(state/'path.u32','<u4');cpp=np.fromfile(state/'predicted_uv.f64','<f8').reshape(n,2)
    if s.shape!=(n,) or path.shape!=(n,):raise ValueError('incomplete current nuisance arrays')
    scene=NativeScene(as_event_view(model,next(iter(model['shots']))));world=np.full((n,3),np.nan)
    for mi,pi in sorted(set(zip(mat[active].tolist(),path[active].tolist()))):
        rr=np.flatnonzero(active&(mat==mi)&(path==pi));world[rr]=scene.world(int(mi),int(pi),s[rr])
    from .measurement_widths import apply_source_paint_interfaces
    paint_rows=apply_source_paint_interfaces(model,packet,world,active,mat,path,s,pair_index=ar('stroke_pair_index','<i4')if 'stroke_pair_index'in packet.get('arrays',{})else None,source_side=ar('stroke_source_side','i1')if 'stroke_source_side'in packet.get('arrays',{})else None)
    camera=CameraGraph(graph_from_joint(model));native=camera.project_rows(sids[active],pts[active],world[active]);valid=np.asarray(native['valid'],bool);diff=np.linalg.norm(native['uv']-cpp[active],axis=1);domain=validate_physical_domain(model)
    report=dict(schema='rinkfit.joint_independent_native_replay.v1',rows=n,actual_paint_interface_rows=len(paint_rows),paint_interface_world_definition='Exact model local halfwidth and recorded world-side binding; ordinary mesh centreline unchanged',active_rows=int(active.sum()),valid_rows=int(valid.sum()),invalid_rows=int((~valid).sum()),max_CPP_native_difference_px=float(np.max(diff)) if np.isfinite(diff).all() else None,max_RS_root_equality_px=float(np.max(native['root_equality_px'])) if len(diff) else None,geometry_domain=domain,whole_RS_certificates_passed=bool(valid.all()),passed=bool(valid.all() and np.isfinite(diff).all() and diff.max(initial=0)<1e-6 and domain['passed']),model_sha256=sha(model_path),packet_sha256=sha(packet_path))
    if output:Path(output).write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

def main(argv=None):
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);b=sub.add_parser('build');b.add_argument('--out',required=True)
    v=sub.add_parser('independent-replay');v.add_argument('--model',required=True);v.add_argument('--packet',required=True);v.add_argument('--state',required=True);v.add_argument('--output',required=True)
    for name in ('phr-fit','phr-audit','engineering-recovery'):
        q=sub.add_parser(name);q.add_argument('--executable',required=True);q.add_argument('--model',required=True);q.add_argument('--packet',required=True);q.add_argument('--out',required=True);q.add_argument('--config');q.add_argument('--initializer',action='store_true')
    for name in ('replay','fit'):
        q=sub.add_parser(name);q.add_argument('--executable',required=True);q.add_argument('--model',required=True);q.add_argument('--packet',required=True);q.add_argument('--out',required=True);q.add_argument('--iterations',type=int,default=0 if name=='replay' else 8);q.add_argument('--threads',type=int,default=8);q.add_argument('--fix-prefix',action='append',default=[]);q.add_argument('--fix-shot-nuisance',action='append',default=[]);q.add_argument('--recapture-new-sources',action='store_true');q.add_argument('--observed-row-time-new-sources',action='store_true')
    a=ap.parse_args(argv)
    if a.command=='build':build(a.out)
    elif a.command in ('phr-fit','phr-audit','engineering-recovery'):raise SystemExit(execute_explicit(a.executable,a.model,a.packet,a.out,config=a.config,geometry_initializer=a.initializer,audit_only=a.command=='phr-audit',engineering_recovery=a.command=='engineering-recovery'))
    elif a.command=='independent-replay':print(json.dumps(independent_replay(a.model,a.packet,a.state,output=a.output)))
    else:raise SystemExit(execute(a.executable,a.model,a.packet,a.out,a.iterations,a.threads,a.fix_prefix,a.fix_shot_nuisance,a.recapture_new_sources,a.observed_row_time_new_sources))
if __name__=='__main__':main()

def bind_captured_nuisance(packet_path,capture_state,source_packet,output_path,*,recapture_rows=None,topology_recapture_rows=None):
    """Reuse actual full capture, initialize only genuinely new query alphas.

    Source identities/UV/E/material/time must match exactly. Statistics and
    membership are supplied by the named packet, never inferred from residuals.
    """
    import numpy as np
    src=Path(packet_path).resolve();state=Path(capture_state).resolve();parent=Path(source_packet).resolve();out=Path(output_path).resolve()
    if out.exists():raise FileExistsError('immutable captured state binding exists')
    packet=json.loads(src.read_text(encoding='utf8'));original=json.loads(parent.read_text(encoding='utf8'));manifest=json.loads((state/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema')!='rinkfit.joint_initial_nuisance_state.v1' or manifest['input_packet_sha256']!=sha(parent):raise ValueError('capture manifest source mismatch')
    n=int(packet['row_count']);old_n=int(manifest['raw_rows'])
    if n<old_n or int(original['row_count'])!=old_n:raise ValueError('capture raw prefix size mismatch')
    checked=set()
    def absolute(d,folder):
        q=Path(d['file']);q=(q if q.is_absolute() else folder/q).resolve();key=(str(q),d['sha256'])
        if key not in checked:
            if sha(q)!=d['sha256']:raise ValueError('immutable capture dependency changed '+str(q))
            checked.add(key)
        return {**d,'file':str(q)}
    def array(doc,folder,key):
        d=absolute(doc['arrays'][key],folder);return np.fromfile(d['file'],dtype=d['dtype'],count=int(np.prod(d['shape'])),offset=d.get('offset_bytes',0)).reshape(d['shape'])
    for name in ('source_dataset_ids','source_row_ids'):
        if packet[name][:old_n]!=original[name]:raise ValueError('changed captured raw identity '+name)
    for name in ('native_pts','shot_index','uv','kind','material_index','E'):
        if not np.array_equal(array(original,parent.parent,name),array(packet,src.parent,name)[:old_n]):raise ValueError('changed captured raw prefix '+name)
    packet['arrays']={k:absolute(d,src.parent) for k,d in packet['arrays'].items()}
    for g in packet['groups']:
        if isinstance(g['W'],dict):g['W']=absolute(g['W'],src.parent)
    for ref in packet.get('source_manifests',[]):ref.update(absolute(ref,src.parent))
    old_queries={int(q['row_index']):q for q in original.get('queries',[])};new_queries={int(q['row_index']):q for q in packet.get('queries',[])}
    for index,q in old_queries.items():
        if new_queries.get(index)!=q:raise ValueError('existing genuine query changed during capture binding '+str(index))
    new_query_rows=sorted(set(new_queries)-set(old_queries));out.parent.mkdir(parents=True,exist_ok=True)
    for name,file,dtype in (('station','station.f64','<f8'),('path','path.u32','<u4'),('row','row.f64','<f8'),('alpha','alpha.f64','<f8')):
        q=state/file;d=manifest['files'][file]
        if sha(q)!=d['sha256']:raise ValueError('captured auxiliary hash changed')
        v=np.fromfile(q,dtype)
        if v.shape!=(old_n,) or not np.isfinite(v).all():raise ValueError('incomplete capture '+name)
        if n!=old_n or (name=='alpha' and new_query_rows):
            combined=array(packet,src.parent,name).copy();keep_new=combined[new_query_rows].copy() if name=='alpha' else None;combined[:old_n]=v
            if name=='alpha':combined[new_query_rows]=keep_new
            q=out.parent/('captured_current_'+file)
            if q.exists():raise FileExistsError('immutable captured auxiliary exists')
            combined.astype(dtype,copy=False).tofile(q);d={'sha256':sha(q)}
        packet['arrays'][name]=dict(file=str(q),sha256=d['sha256'],dtype=dtype,shape=[n])
    packet['captured_nuisance_binding']=dict(schema='rinkfit.actual_captured_nuisance_binding.v1',captured_manifest=dict(file=str(state/'manifest.json'),sha256=sha(state/'manifest.json')),captured_source_packet=dict(file=str(parent),sha256=sha(parent)),new_queries_initialized_from_original_actual_source_windows=new_query_rows,source_means_E_material_clock_changed=False,native_RS_or_query_closure_claimed=False)
    if recapture_rows is not None:packet['recapture_rows']=sorted(set(map(int,recapture_rows))|set(new_query_rows))
    if topology_recapture_rows is not None:packet['source_topology_recapture_rows']=list(map(int,topology_recapture_rows));packet['recapture_rows']=sorted(set(packet.get('recapture_rows',[]))|set(map(int,topology_recapture_rows)))
    out.write_text(json.dumps(packet,indent=2),encoding='utf8');_inherit_source_recall_marker(src,out)
    return dict(packet=str(out),sha256=sha(out),new_query_rows=len(new_query_rows),reused_raw_rows=old_n)
