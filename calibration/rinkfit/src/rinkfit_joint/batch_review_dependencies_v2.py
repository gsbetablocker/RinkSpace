"""Batch exact-source review reuse, without changing the frozen pixel readers.

Caller supplies the same SHA-validated packet/arrays as collect_findings. All
subjects remain represented. A result is publishable only after validate().
The batch owns no solver data and never mutates a source observation.
"""
from pathlib import Path
import hashlib,json,sqlite3,time
from copy import deepcopy
import numpy as np
from .io_utils import read_json,read_jsonl,file_sha256,object_sha256,write_json
from . import packet_feedback_facade_v2 as facade
from . import source_recovery_review as legacy
from . import static_board_feedback_review as texture
from . import static_board_feedback_review_v2 as tailtexture

SCHEMA='rinkfit.exact_source_review_dependency_batch.v1'

class BatchInputChanged(ValueError):pass


def _path(ref):return Path(ref.get('path',ref.get('file'))).resolve()
def _key(source,rid,uv):return (rid,int(source['pts']),tuple(uv))


class VerifiedReferences:
    """Fresh unique SHA checks for one batch, including archive identity mapping."""
    def __init__(self):self.refs={};self.declarations=[];self.hash_count=0
    def check(self,ref):
        p=_path(ref);old=facade._archive()['original_module']
        if p==_path(old)and ref['sha256']==old['sha256']:
            p=Path(facade.__file__).with_name('packet_feedback_reader_e712.py').resolve()
        key=str(p).casefold();wanted=ref['sha256'];prior=self.refs.get(key)
        if prior is not None and prior['sha256']!=wanted:raise BatchInputChanged('conflicting SHA declarations for '+str(p))
        if prior is None:
            actual=file_sha256(p);self.hash_count+=1
            if actual!=wanted:raise BatchInputChanged('changed dependency '+str(p))
            self.refs[key]=dict(path=str(p),sha256=wanted)
        self.declarations.append(dict(declared=ref,verified_path=str(p)))
        return p
    def validate(self):
        for ref in self.refs.values():
            self.hash_count+=1
            if file_sha256(ref['path'])!=ref['sha256']:raise BatchInputChanged('dependency changed before publication '+ref['path'])
    def read(self,ref):return read_json(self.check(ref))


def _durable_inputs(project,event):
    root=Path(project);paths=set((root/'events'/event/'feedback').rglob('*.json'))
    paths.update((root/'events').glob('*/programmes/*/primary_source_selection.json'))
    paths.update((root/'events').glob('*/model_constraints/source_clock_registry.json'))
    return [dict(path=str(p.resolve()),sha256=file_sha256(p))for p in sorted(paths,key=str)]


def _case_rows(connection):
    return [dict(r)for r in connection.execute('SELECT case_id,version,status,content_sha256,body FROM cases ORDER BY case_id,version').fetchall()]


class AuthoritySnapshot:
    """Restore all durable cases once; keep every case version and retraction."""
    def __init__(self,project,event,verifier):
        from rinkspace.feedback_runtime import feedback_store
        from rinkspace.storage import AssetStore
        self.project=Path(project);self.event=event;self.verifier=verifier
        self.database=AssetStore(project).data/'events'/event/'feedback/feedback.sqlite'
        before=_durable_inputs(project,event)
        with feedback_store(project,event)as store:self.rows=_case_rows(store.db)
        self.durable=_durable_inputs(project,event)
        if self.durable!=before:raise BatchInputChanged('durable authority changed during restore')
        for ref in self.durable:verifier.check(ref)
        self.digest=object_sha256(self.rows);latest={}
        for row in self.rows:
            if row['status']in('active','retracted'):
                previous=latest.get(row['case_id'])
                if previous is None or row['version']>previous['version']:latest[row['case_id']]=row
        self.published=list(latest.values());self.authenticated={};self.scope_cache={};self.query_cache={}
    def validate(self):
        if _durable_inputs(self.project,self.event)!=self.durable:raise BatchInputChanged('durable authority added/changed/removed before publication')
        with sqlite3.connect('file:'+self.database.as_posix()+'?mode=ro',uri=True)as db:
            db.row_factory=sqlite3.Row
            if object_sha256(_case_rows(db))!=self.digest:raise BatchInputChanged('case version/status/body set changed before publication')
    def query(self,context):
        from rink_feedback.core import _context,_match
        context=_context(context);digest=object_sha256(context)
        if digest not in self.query_cache:
            matches=[]
            for row in self.published:
                if row['status']!='active':continue
                case=json.loads(row['body']);match=_match(case,context)
                if match:matches.append(dict(case_id=row['case_id'],version=row['version'],content_sha256=row['content_sha256'],case=case,**match))
            self.query_cache[digest]=sorted(matches,key=lambda item:(not item['case'].get('mandatory',False),-len(item['matched_tags']),item['case_id']))
        return deepcopy(self.query_cache[digest])
    def _authenticate(self,match):
        key=(match['case_id'],match['version'],match['content_sha256'])
        if key not in self.authenticated:
            try:
                for ref in match['case'].get('evidence_refs',[]):
                    self.verifier.check(ref);original=ref.get('original_address_metadata_only')
                    if original:
                        p=Path(original)
                        if p.is_relative_to(self.project)and p.is_file():self.verifier.check(dict(path=str(p),sha256=ref['sha256']))
                self.authenticated[key]=True
            except (OSError,ValueError):self.authenticated[key]=False
        return self.authenticated[key]
    def effective_case_refs(self,source,label,shot):
        if not label.startswith('board.'):return []
        from rinkspace.feedback_runtime import native_source,resolve_scope
        binding=dict(source_sha256=source['video_sha256'],source_pts=int(source['pts']),time_base=source['time_base'],native_hw=[1080,1920],shot_id=shot)
        native=native_source(binding);scopekey=(native['video_sha256'],native['time_base'])
        if scopekey not in self.scope_cache:self.scope_cache[scopekey]=resolve_scope(native,self.project)
        event,programme=self.scope_cache[scopekey]
        if event!=self.event:raise BatchInputChanged('source authority event differs')
        entities=['board.contact','board.bottom_strip','board.top'];tags=['ice_contact','board_foot','material_boundary']
        context=dict(source=native,event_id=event,programme_id=programme,shot_id=shot,camera_id=None,entity_ids=sorted(entities),tags=sorted(tags),include_related_cases=True,split_context=dict(mode='unknown'))
        applicable=[]
        for match in self.query(context):
            if match['match_status']!='applicable':
                case=match['case'];scope=case.get('scope',{})
                same=(scope.get('event_id')==event and scope.get('source_video_sha256')==native['video_sha256']and scope.get('camera_id')in(None,None)and bool(set(scope.get('entity_ids',[]))&set(entities)))
                if not(case.get('kind')=='recipe'and same and case.get('assertions',{}).get('lesson',{}).get('reader_check')in {'ordered_board_material_flanks','single_actual_stroke_two_flanks_topology','declared_target_material_interface'}):continue
            if self._authenticate(match):applicable.append({k:match[k]for k in('case_id','version','content_sha256')})
        return sorted(applicable,key=lambda r:r['case_id'])
    def store_view(self,store):
        snapshot=self
        class View:
            def query(self,context):return snapshot.query(context)
            def __getattr__(self,name):return getattr(store,name)
        return View()


class ExactReviewDependencyBatch:
    """Use resolve_all() after preparing the complete checkpoint subject list.

    subjects: [{row_index, source}], one item for every actual representative.
    The supplied packet/arrays must be validated by the caller's checkpoint
    boundary, exactly as for the scalar frozen cached_dependencies interface.
    """
    def __init__(self,packet,arrays,review_config,subjects,*,project=None,event='ankara_2026'):
        from rinkspace.storage import AssetStore
        self.packet=packet;self.arrays=arrays;self.project=Path(project or legacy.PROJECT);self.event=event
        self.verifier=VerifiedReferences();self.config_ref=review_config if isinstance(review_config,dict)else dict(path=str(Path(review_config).resolve()),sha256=file_sha256(review_config));self.cfg=self.verifier.read(self.config_ref)
        facade._validate_config_reader_refs(self.cfg)
        for ref in self.cfg.get('actual_reader_module_refs',[]):self.verifier.check(ref)
        optional=self.cfg.get('actual_batch_dependency_helper_ref')
        if optional and self.verifier.check(optional)!=Path(__file__).resolve():raise BatchInputChanged('batch helper module identity differs')
        self.subjects=list(subjects);self.started=time.perf_counter();self.records=[];self.prepared=False;self.validated=False;self.json_cache={};self.journal_stats=[]
        self.cache_root=AssetStore(self.project).data/'events'/event/'feedback/review_cache'
        self.authority=AuthoritySnapshot(self.project,event,self.verifier)
        self.frozen=facade._frozen_reader();self.old_base,self.old_code,self.facts=self.frozen.reader_dependencies()
        for ref in self.old_code+self.facts:self.verifier.check(ref)
        self.texture_base,self.texture_code=texture._dependencies(self.cfg)
        self.tail_texture_base,self.tail_texture_code=tailtexture._dependencies(self.cfg)
        for ref in self.tail_texture_code:self.verifier.check(ref)
        for ref in self.texture_code:self.verifier.check(ref)
        self.ordered=facade._ordered_journals(packet,self.cfg)
        self.order_keys=[(str(_path(r)).casefold(),r['sha256'])for r in self.ordered]
    def _json(self,ref):
        p=self.verifier.check(ref);key=(str(p).casefold(),ref['sha256'])
        if key not in self.json_cache:self.json_cache[key]=read_json(p)
        return self.json_cache[key]
    def _cache(self,source,index,texture_row):
        tail=self.packet['source_row_ids'][index].startswith(tailtexture.PREFIX)
        code=(self.tail_texture_base if tail else self.texture_base if texture_row else self.old_base)['reader_code_sha256'];namespace=('btx_'if tail else'bt_'if texture_row else'pb_')+code[:16]
        key=object_sha256(dict(source=source,dataset=self.packet['source_dataset_ids'][index],row_id=self.packet['source_row_ids'][index]))
        return self.cache_root/namespace/(key+'.json')
    def _one(self,item):
        i=int(item['row_index']);source=item['source'];rid=self.packet['source_row_ids'][i];is_texture=rid.startswith(texture.PREFIX)
        texture_module=tailtexture if rid.startswith(tailtexture.PREFIX)else texture
        texture_base=self.tail_texture_base if texture_module is tailtexture else self.texture_base
        if rid.startswith('native_tail_C_v1/'):
            return dict(row_index=i,source=source,source_row_id=rid,status='unknown',dependencies=None,reason='actual_C_paired_reader_pending')
        record=dict(row_index=i,source=source,source_row_id=rid,status='unknown',dependencies=None);path=self._cache(source,i,is_texture)
        if not path.is_file():record['reason']='no_completed_exact_reader_cache';return record
        cache=read_json(path);record['cache']=dict(path=str(path),sha256=file_sha256(path));self.verifier.check(record['cache']);deps=cache.get('review_dependencies',{});obs=legacy.observation_content(self.packet,self.arrays,i)
        if is_texture:
            if any(deps.get(k)!=v for k,v in texture_base.items()):record['reason']='texture_reader_or_policy_changed';return record
            inputs=texture_module._inputs(source,self.packet,self.arrays,i,self.cfg)
            if deps.get('actual_source_inputs_sha256')!=object_sha256(inputs):record['reason']='texture_actual_inputs_changed';return record
            if self._json(cache['input_binding'])!=inputs:record['reason']='texture_binding_changed';return record
        else:
            required=('material_label','shot_id','extra_fact_refs','evidence_refs')
            if not all(k in cache for k in required):record['reason']='legacy_cache_incomplete';return record
            current=dict(self.old_base);cases=self.authority.effective_case_refs(source,cache['material_label'],cache['shot_id']);current['effective_facts_sha256']=object_sha256(dict(refs=self.facts+cache['extra_fact_refs'],case_versions=cases))
            if any(deps.get(k)!=v for k,v in current.items()):record['reason']='legacy_reader_policy_or_facts_changed';return record
            if deps.get('observation_snapshot_sha256')!=object_sha256(obs):record['reason']='legacy_observation_changed';return record
            binding=self._json(cache['extra_fact_refs'][0])
            if binding.get('schema')!='rinkfit.exact_packet_native_review_inputs.v1'or binding['observation']!=obs or binding['source']!=source:record['reason']='legacy_binding_changed';return record
            for ref in binding['input_dependencies']:self.verifier.check(ref)
            if self.cfg.get('frames',{}).get(str(source['pts']),{})!=binding['frame_inputs']or self.frozen.source_reader_policy(self.cfg)!=binding.get('source_reader_policy'):record['reason']='legacy_actual_frame_or_policy_changed';return record
            selected=binding.get('source_journal')
            if selected is None:record['reason']='unknown_current_journal_ownership';return record
            selectedkey=(str(_path(selected)).casefold(),selected['sha256'])
            if selectedkey not in self.order_keys:record['reason']='selected_journal_not_current';return record
            record['selected_journal']=selected;record['selected_order']=self.order_keys.index(selectedkey);record['exact_subject_key']=_key(source,rid,self.arrays['uv'][i].tolist())
        for ref in cache['evidence_refs']:self.verifier.check(ref)
        record['candidate_dependencies']=deps;record['observation']=obs;record['status']='pending_batch_source_index'if not is_texture else'exact_source_review_reusable'
        if is_texture:record['dependencies']=deps
        return record
    def prepare(self):
        if self.prepared:return self
        for item in self.subjects:
            try:r=self._one(item)
            except (OSError,ValueError,KeyError,IndexError)as error:r=dict(row_index=int(item['row_index']),source=item['source'],status='unknown',dependencies=None,reason='actual_input_rejected',error=str(error))
            self.records.append(r)
        candidates=[r for r in self.records if r['status']=='pending_batch_source_index'];wanted={r['exact_subject_key']for r in candidates};ids={k[0]for k in wanted};maxorder=max((r['selected_order']for r in candidates),default=-1);presence={};memo={}
        for order,ref in enumerate(self.ordered[:maxorder+1]):
            digest=ref['sha256'];p=self.verifier.check(ref)
            if digest not in memo:
                start=time.perf_counter();found={};lines=0
                for row in read_jsonl(p):
                    lines+=1;rid=row.get('source_row_id');uv=row.get('current_uv')
                    if rid not in ids or not isinstance(uv,list):continue
                    key=(rid,int(row.get('native_pts',-1)),tuple(uv))
                    if key in wanted:found[key]=lines
                memo[digest]=found;self.journal_stats.append(dict(reference=ref,rows_streamed=lines,exact_requested_matches=len(found),seconds=time.perf_counter()-start))
            presence[order]=memo[digest]
        for r in candidates:
            key=r.pop('exact_subject_key');selected=r['selected_order'];earlier=next((j for j in range(selected)if key in presence[j]),None)
            if earlier is not None:r.update(status='unknown',reason='higher_priority_exact_journal_changed',higher_priority_journal=self.ordered[earlier])
            elif key not in presence[selected]:r.update(status='unknown',reason='exact_mean_not_in_selected_journal')
            else:r.update(status='exact_source_review_reusable',dependencies=r['candidate_dependencies'],selected_source_line_1based=presence[selected][key])
            r.pop('candidate_dependencies',None)
        self.prepared=True;return self
    def validate(self):
        self.validated=False
        if not self.prepared:raise ValueError('prepare complete actual subject list first')
        self.verifier.validate();self.authority.validate();self.validated=True;return self
    def resolve_all(self):
        if not self.prepared:self.prepare()
        self.validate();return [r['dependencies']for r in self.records]
    def dependencies_for(self,source,row_index):
        if not self.validated:raise ValueError('batch has not passed publication validation')
        matches=[r for r in self.records if r['row_index']==int(row_index)and r['source']==source]
        if len(matches)!=1:raise ValueError('subject absent or duplicated in complete batch')
        r=matches[0]
        if r.get('observation')!=legacy.observation_content(self.packet,self.arrays,row_index):return None
        return r['dependencies']
    def checkpoint(self,store,run_id,findings):
        """Frozen FeedbackCycle with all current cases, no repeated restore/query IO."""
        from rink_feedback.cycle import FeedbackCycle
        if not self.validated:self.resolve_all()
        supplied=[]
        for finding in findings:
            f=dict(finding);deps=self.dependencies_for(f['source'],f['source_row_index'])
            if deps is not None:f['review_dependencies']=deps
            else:f.pop('review_dependencies',None)
            supplied.append(f)
        result=FeedbackCycle(self.authority.store_view(store)).checkpoint(run_id,supplied)
        self.validate();return result
    def receipt(self):
        return dict(schema=SCHEMA,helper=dict(path=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),subjects=len(self.subjects),resolved_subjects=len(self.records),reusable=sum(r['dependencies']is not None for r in self.records),unknown=sum(r['dependencies']is None for r in self.records),all_subjects_preserved=len(self.records)==len(self.subjects),authority_case_versions=len(self.authority.rows),authority_version_set_sha256=self.authority.digest,durable_input_refs=self.authority.durable,unique_verified_dependencies=list(self.verifier.refs.values()),hash_operations=self.verifier.hash_count,journals=self.journal_stats,source_results=self.records,validated_before_publication=self.validated,seconds=time.perf_counter()-self.started,source_UV_or_packet_mutated=False)
