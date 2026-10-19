"""Persistent owner of the verified-data worker; numerical results are fresh per request."""
from pathlib import Path
import hashlib,json,os,queue,subprocess,threading,time,struct

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _ref(path):return dict(path=str(Path(path)),sha256=_sha(path))
def _verified(desc,base=None):
    p=Path(desc['path'] if 'path'in desc else desc['file'])
    if not p.is_absolute():p=Path(base)/p
    if _sha(p)!=desc['sha256']:raise ValueError('Frozen identity changed: '+str(p))
    return p

def _atomic(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    if path.exists() or tmp.exists():raise FileExistsError(str(path))
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

class ResidentTransportClient:
    def __init__(self,build_dir,run_dir,stop_file=None):
        self.build_dir=Path(build_dir).resolve();self.run_dir=Path(run_dir).resolve()/'resident_client'
        self.stop_file=Path(stop_file).resolve() if stop_file else None
        self.worker=None;self.authority=None;self.generation=0;self.closed=False
        self._queue=queue.Queue();self._lock=threading.Lock();self._recipe=None

    def start(self):
        if self.closed:raise RuntimeError('Cannot restart a closed worker; create a new client')
        if self.worker is not None:return self
        recipe_path=self.build_dir/'recipe.json';recipe=_read(recipe_path)
        gpu=recipe.get('schema')=='rinkfit.resident_verified_GPU_transport.v1'
        if gpu:
            if not recipe.get('CPU_primal_RS_profile_W') or not recipe.get('raw_UV_GPU') or not recipe.get('same_exact_projection_RS_profile_W_math'):raise ValueError('GPU transport changed the original primal/RS/profile/W contract')
            validation=_read(_verified(recipe['backend_validation']))
            if validation.get('build')!=recipe.get('validated_backend_build'):raise ValueError('GPU QA and exact backend build differ')
            verified_build=_read(_verified(validation['build']))
            for key in ['original_whole_UV','original_transport_math']:
                origin=recipe[key];_verified(origin)
                if verified_build['source_inputs'].get(origin['path'])!=origin['sha256']:raise ValueError('GPU QA covers different transport source: '+key)
            for file,digest in verified_build['source_inputs'].items():
                if file in recipe['sources'] and recipe['sources'][file]!=digest:raise ValueError('GPU QA/worker source or native library mismatch: '+file)
            if (validation.get('raw_rows'),validation.get('active_rows'),validation.get('profiles'),validation.get('original_W_rank'))!=(182017,182016,961,214112):raise ValueError('Incomplete GPU validation source coverage')
            if not validation.get('passed') or not validation.get('complete_state_unchanged') or validation.get('end_to_end_speedup',0)<=1:raise ValueError('GPU whole transport requires a passed complete fixed-state comparison and measured benefit')
        else:
            if recipe.get('schema')!='rinkfit.resident_verified_transport_prototype.v3':raise ValueError('An approved transport build is required')
            if not recipe.get('no_kernel_changed') or not recipe.get('same_exact_projection_RS_profile_W_math'):raise ValueError('Unverified transport arithmetic identity')
        if not {'whole_uv','profile_implicit'}.issubset(set(recipe.get('supported_ops',[]))):raise ValueError('Transport worker operations unavailable')
        for p,digest in recipe['sources'].items():
            if _sha(p)!=digest:raise ValueError('Frozen worker source changed: '+p)
        exe=_verified(recipe['executable']);self._recipe=recipe;self._recipe_ref=_ref(recipe_path)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):raise FileExistsError(str(self.run_dir))
        self.run_dir.mkdir(parents=True,exist_ok=True)
        env={k.upper():v for k,v in os.environ.items()}
        env['PATH']=os.pathsep.join(recipe['runtime_bin_paths'])+os.pathsep+env.get('PATH','')
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        self._log=(self.run_dir/'worker_stdout.log').open('x',encoding='utf-8')
        try:
            self.worker=subprocess.Popen([str(exe)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict',bufsize=1)
        except BaseException:
            self._log.close();self.closed=True;raise
        def pump():
            try:
                for line in self.worker.stdout:self._queue.put(line)
            except Exception as exc:self._queue.put(exc)
            finally:self._queue.put(None)
        self._reader=threading.Thread(target=pump,name='resident-source-output',daemon=True)
        try:
            self._reader.start()
            _atomic(self.run_dir/'worker_identity.json',dict(schema='rinkfit.resident_client_owner.v1',pid=self.worker.pid,executable=_ref(exe),recipe=self._recipe_ref,client_source=_ref(__file__),root_J_result_cache=False))
        except BaseException:
            if self.worker.poll() is None:self.worker.terminate()
            self.worker.wait()
            if self._reader.is_alive():self._reader.join()
            self._log.close();self.closed=True
            raise
        return self

    def _exchange(self,request):
        self.generation+=1;index=self.generation;begin=time.monotonic()
        _atomic(self.run_dir/f'request_{index:06d}.json',request)
        if self.worker.poll() is not None:raise RuntimeError('Resident worker already exited: '+str(self.worker.returncode))
        self.worker.stdin.write(json.dumps(request,ensure_ascii=False,separators=(',',':'))+'\n');self.worker.stdin.flush()
        while True:
            try:line=self._queue.get(timeout=.25)
            except queue.Empty:
                if self.worker.poll() is not None:raise RuntimeError('Worker exited without response')
                continue
            if line is None:raise RuntimeError('Resident worker output closed')
            if isinstance(line,Exception):raise line
            self._log.write(line);self._log.flush()
            if line.startswith('RESIDENT_RESPONSE '):
                reply=json.loads(line[len('RESIDENT_RESPONSE '):]);reply['client_wall_seconds']=time.monotonic()-begin
                _atomic(self.run_dir/f'response_{index:06d}.json',reply)
                if not reply.get('ok'):raise RuntimeError('Resident request rejected: '+reply.get('error','unknown'))
                return reply

    def _transport_state(self,path):
        doc=_read(path)
        if doc.get('schema')!='rinkfit.complete_continuous_reduced_SLP_checkpoint.v1':raise ValueError('A complete transport CP is required')
        if doc.get('raw_rows')!=182017 or doc.get('active_rows')!=182016:raise ValueError('Transport CP must retain the whole raw and active ledger')
        _verified(doc['model']);linearization=_verified(doc['linearization']);lm=_read(linearization)
        for key in ['packet','semantic']:_verified(lm[key])
        data=dict(model=doc['model'],packet=lm['packet'],semantic=lm['semantic'])
        ident=dict(checkpoint_sha256=_sha(path),model_sha256=doc['model']['sha256'],
                   model=doc['model'],packet=lm['packet'],semantic=lm['semantic'],linearization=doc['linearization'],arrays={})
        for key,dt in [('station','<f8'),('branch','<u4'),('row','<f8'),('alpha','<f8')]:
            desc=doc['row_state_files'][key];file=_verified(desc,path.parent)
            if desc['shape']!=[182017] or desc['dtype']!=dt:raise ValueError('Incomplete transport raw array '+key)
            if file.stat().st_size!=182017*(4 if key=='branch'else 8):raise ValueError('Wrong transport array byte count '+key)
            ident['arrays'][key]=desc['sha256']
        # CP.linearization may be an older numerical linearization. It is used
        # only for its immutable source authority, never as the new theta/J.
        return ident,data

    def _check_profile_rows(self,result,report,data):
        sem_path=_verified(data['semantic']);sem=_read(sem_path)
        kd=sem['arrays']['kind'];ad=sem['active_source_mask']
        for d in [kd,ad]:
            if d['dtype']!='|u1'or d['shape']!=[182017]:raise ValueError('Profile authority mask/kind schema')
        kind=_verified(kd,sem_path.parent).read_bytes();active=_verified(ad,sem_path.parent).read_bytes()
        if len(kind)!=182017 or len(active)!=182017:raise ValueError('Profile authority byte count')
        rows=[i for i,(k,a)in enumerate(zip(kind,active))if k==2 and a]
        desc=result['profile_rows'];file=_verified(desc,report.parent)
        if len(rows)!=961 or desc['dtype']!='<u8'or desc['shape']!=[961]:raise ValueError('Profile source coverage')
        if file.read_bytes()!=struct.pack('<961Q',*rows):raise ValueError('Profile output source identities differ from immutable source')

    def transport(self,role,checkpoint,out_dir):
        if role not in ['whole_uv','profile_implicit']:raise ValueError('Unknown transport role')
        return self.request(role,checkpoint,out_dir)

    def _state(self,role,path):
        if role in ['whole_uv','profile_implicit']:return self._transport_state(path)
        doc=_read(path);inp=_verified(doc['input']) if role=='values' else path
        data=_read(inp)
        if data['schema']!='rinkfit.qbar_sparse_linearization_input.v1':raise ValueError('Not a complete reduced input')
        for key in ['model','packet','semantic']:
            _verified(data[key])
            if role=='values':
                _verified(doc[key])
                if doc[key]!=data[key]:raise ValueError('Qbar manifest and its parent input disagree: '+key)
        ident=dict(input_sha256=_sha(inp),model_sha256=data['model']['sha256'],arrays={})
        dtypes={'station':'<f8','path':'<u4','row':'<f8','alpha':'<f8','mode':'|u1'}
        for key,dt in dtypes.items():
            desc=data['arrays'][key];file=_verified(desc,inp.parent)
            if desc['shape']!=[182017] or desc['dtype']!=dt:raise ValueError('Incomplete raw state array '+key)
            itemsize=1 if key=='mode' else 4 if key=='path' else 8
            if file.stat().st_size!=182017*itemsize:raise ValueError('Wrong raw state array byte count '+key)
            ident['arrays'][key]=desc['sha256']
        selected=data.get('source_rows')
        if selected is None:ident['selected_rows_sha256']='all_active'
        elif isinstance(selected,list):ident['selected_rows_sha256']=hashlib.sha256(json.dumps(selected,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        else:_verified(selected,inp.parent);ident['selected_rows_sha256']=selected['sha256']
        if role=='values':
            for key in ['linearization_station','linearization_row','source_rows']:
                desc=doc[key];_verified(desc,path.parent);ident['qbar_'+key+'_sha256']=desc['sha256']
            ident['qbar_manifest_sha256']=_sha(path)
        return ident,data

    def request(self,role,input_manifest,out_dir):
        if role not in ['qbar','values','whole_uv','profile_implicit']:raise ValueError('Unknown resident role')
        with self._lock:
            if self.stop_file and self.stop_file.exists():raise InterruptedError('Stop requested before a new resident evaluation')
            self.start();path=Path(input_manifest).resolve();out_dir=Path(out_dir).resolve()
            if out_dir.exists() and any(out_dir.iterdir()):raise FileExistsError(str(out_dir))
            expected,data=self._state(role,path)
            if self.authority is None:
                packet=_read(_verified(data['packet']));qd=packet['query_packet'];query={'path':qd['manifest'],'sha256':qd['sha256']};_verified(query)
                reply=self._exchange(dict(op='init',packet=data['packet'],semantic=data['semantic'],query=query,reader_policy_sha256=self._recipe['original_reader_policy_sha256'],output=str(self.run_dir/'dataset')))
                result=reply['result']
                if result['raw_rows']!=182017 or result['active_rows']!=182016 or not result['complete_source_validation_performed']:raise ValueError('Dataset initialization was incomplete')
                self.authority=result['authority'];_atomic(self.run_dir/'initialization.json',reply)
            for key in ['packet','semantic']:
                if data[key]['sha256']!=self.authority[key+'_sha256']:raise ValueError('New source authority requires a new fully verified worker')
            reply=self._exchange(dict(op=role,input=_ref(path),output=str(out_dir),authority=self.authority,expected_state=expected));r=reply['result']
            if r['state']!=expected or r['authority']!=self.authority:raise ValueError('Resident reply does not match the requested full state')
            report=_verified(r['report'])
            if report.resolve()!=(out_dir/'manifest.json').resolve():raise ValueError('Resident wrote a different report path')
            result=_read(report)
            for key in ['model','packet','semantic']:
                if result.get(key)!=data[key]:raise ValueError('Evaluated report identity differs from the requested state: '+key)
                _verified(result[key])
            parentkey='checkpoint' if role in ['whole_uv','profile_implicit']else ('input' if role=='qbar'else 'qbar_linearization')
            if result.get(parentkey)!=_ref(path):raise ValueError('Evaluated report uses a different request parent')
            rawkey,activekey=('raw_source_rows','active_source_rows') if role=='qbar' else ('raw_rows','active_rows')
            if role=='profile_implicit':
                if result.get('raw_rows')!=182017 or result.get('profiles')!=961:raise ValueError('Profile transport coverage incomplete')
                if not result.get('no_Solve')or not result.get('original_finite_S_alpha_domain_preserved'):raise ValueError('Profile transport scope changed')
                self._check_profile_rows(result,report,data)
            elif result.get(rawkey)!=182017 or result.get(activekey)!=182016:raise ValueError('Evaluated report is not whole-source')
            if role=='whole_uv':
                if result.get('full_W_rank')!=214112 or not result.get('statistical_partition_exactly_once')or result.get('profile_count')!=961:raise ValueError('Whole-W transport source/statistical coverage changed')
            if role in ['whole_uv','profile_implicit']:
                after,_=self._state(role,path)
                if after!=expected:raise ValueError('CP/model/four auxiliaries changed during transport')
                canonical=json.dumps(expected,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')
                if hashlib.sha256(canonical).hexdigest()!=r['state_sha256']:raise ValueError('Transport state digest mismatch')
                disk=_read(_verified(_ref(out_dir/'resident_receipt.json')))
                if disk!=r:raise ValueError('Transport on-disk receipt differs from worker response')
            if role=='values' and result.get('physical_model_snapshot',{}).get('sha256')!=data['model']['sha256']:raise ValueError('Value-state model snapshot differs')
            if not r.get('no_root_or_J_result_cache') or not r.get('parameter_model_fresh_per_request'):raise ValueError('Numerical-state reuse is forbidden')
            return dict(manifest_path=str(report),manifest=r['report'],role=role,worker_pid=self.worker.pid,authority=self.authority,expected_state=expected,state_sha256=r['state_sha256'],request_seconds=r['request_seconds'],client_wall_seconds=reply['client_wall_seconds'],receipt=_ref(out_dir/'resident_receipt.json'),
                        operator_identity=dict(protocol=('rinkfit.resident.GPU_transport.v1' if self._recipe.get('raw_UV_GPU') else 'rinkfit.resident.transport.v3'),role=role,worker_executable=self._recipe['executable'],recipe=self._recipe_ref,client_source=_ref(__file__),authority_closure_sha256=self.authority['closure_sha256']),
                        target_identity=dict(model=data['model'],packet=data['packet'],semantic=data['semantic'],complete_request=_ref(path),expected_state_sha256=r['state_sha256']))

    def close(self):
        if self.closed:return
        with self._lock:
            try:
                if self.worker is not None and self.worker.poll() is None:
                    self._exchange(dict(op='close'));self.worker.stdin.close();self.worker.wait()
            finally:
                if self.worker is not None and self.worker.poll() is None:self.worker.terminate();self.worker.wait()
                if self.worker is not None:
                    self._reader.join();self._log.close()
                self.closed=True
    def __enter__(self):return self.start()
    def __exit__(self,*args):self.close()

# Retains the established owner import while exposing the explicit v3 name.
ResidentReducedClient = ResidentTransportClient
