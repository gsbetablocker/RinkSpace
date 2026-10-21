"""Current-packet feedback reads with exact journal ownership and new caches.

The frozen native RGB operators remain in source_recovery_review. Its function
is reused with an explicit versioned cache sink, without changing its globals,
historical source proofs or historical cache entries.
"""
from pathlib import Path
import json
import types
import numpy as np
from . import source_recovery_review as legacy
from .io_utils import read_json,read_jsonl,file_sha256,object_sha256,write_json
from .board_source_qualification_guard import _load_ref
from .packet_bound_source_review import PacketBoundSourceReview,reference

VERSION='exact_packet_native_feedback_v1'
CONFIG=dict(legacy.CONFIG,journal_selection='exact_packet_row_ID_PTS_current_mean_qualified_original_UV',
            old_journal_may_overwrite_exact_match=False,cache_namespace=VERSION)


def reader_dependencies():
    _,code,facts=legacy.reader_dependencies()
    root=Path(__file__).parent
    code=code+[reference(root/name)for name in ('packet_bound_source_review.py','packet_bound_feedback_review.py','feedback_dispatch.py','feedback_checkpoint.py')]
    return dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(CONFIG),
                effective_facts_sha256=object_sha256(facts)),code,facts


def cache_path(source,dataset,row_id):
    from rinkspace.storage import AssetStore
    key=object_sha256(dict(source=source,dataset=dataset,row_id=row_id))
    version=reader_dependencies()[0]['reader_code_sha256'][:16]
    return AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/review_cache'/('pb_'+version)/(key+'.json')


def source_reader_policy(config):
    return {key:config.get(key)for key in ('programme_scope','frame_roles','source_journals','resolve_current_journals')}


def _exact_journal_corner_operator(journal):
    if journal.get('current_corner_operator'):return journal
    audit=journal.get('source_measurement_audit',{});op=audit.get('current_two_material_edges',{})
    if (op.get('valid')is not True or op.get('current_intersection_uv')!=journal.get('current_uv')
        or len(op.get('edge_fits',[]))!=2 or not op.get('intersection_seed_uv')):return journal
    fits=op['edge_fits'];tangents=[edge['tangent']for edge in fits]
    if np.asarray(tangents).shape!=(2,2)or not np.isfinite(tangents).all():return journal
    row=dict(journal,current_corner_operator=dict(original_corner_seed_native_uv=op['intersection_seed_uv'],edge_tangents=tangents),
             current_corner_operator_origin='exact_current_packet_mean_source_journal_two_material_edges')
    if not row.get('physical_identity_status')and row.get('physical_identity_continuation')and row.get('source_geometry_review_pending')is False:
        row['physical_identity_status']=row['physical_identity_continuation']
    return row


def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    path=cache_path(source,packet['source_dataset_ids'][index],packet['source_row_ids'][index])
    if not path.is_file():return None
    cached=read_json(path);deps=cached.get('review_dependencies',{})
    if not all(k in cached for k in ('material_label','shot_id','extra_fact_refs','evidence_refs')):return None
    current,_,facts=reader_dependencies()
    cases=legacy.effective_case_refs(source,cached['material_label'],cached['shot_id'])
    current['effective_facts_sha256']=object_sha256(dict(refs=facts+cached['extra_fact_refs'],case_versions=cases))
    if any(deps.get(k)!=v for k,v in current.items()):return None
    observation=legacy.observation_content(packet,arrays,index)
    if deps.get('observation_snapshot_sha256')!=object_sha256(observation):return None
    try:
        for ref in cached['evidence_refs']:legacy.checked(ref)
        binding=read_json(legacy.checked(cached['extra_fact_refs'][0]))
        if binding.get('schema')!='rinkfit.exact_packet_native_review_inputs.v1' or binding['observation']!=observation:return None
        for ref in binding['input_dependencies']:legacy.checked(ref)
        if binding['source']!=source:return None
        if review_config is not None:
            cfg=read_json(legacy.checked(review_config)if isinstance(review_config,dict)else review_config)
            if cfg.get('frames',{}).get(str(source['pts']),{})!=binding['frame_inputs']:return None
            if source_reader_policy(cfg)!=binding.get('source_reader_policy'):return None
    except (KeyError,ValueError,OSError):return None
    return deps


def _frozen_native_reader():
    original=legacy.CurrentPacketReview.read_frame
    # Function globals are copied, not monkey-patched. The mature pixel readers,
    # source feedback and resolution implementation remain exactly the same.
    function=types.FunctionType(original.__code__,dict(original.__globals__,cache_path=cache_path),
                               name='read_exact_packet_native_frame',argdefs=original.__defaults__,closure=original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return function


_READ_NATIVE_FRAME=_frozen_native_reader()


class PacketBoundCurrentReview:
    def __init__(self,packet_path,config_path,*,requested_row_indices):
        self.packet_path=Path(packet_path).resolve();self.packet=read_json(self.packet_path)
        self.config_path=Path(config_path).resolve();self.config=read_json(self.config_path)
        if legacy.checked(self.config['packet'])!=self.packet_path:raise ValueError('Review config belongs to another exact packet')
        self.requested=sorted(set(map(int,requested_row_indices)));self.arrays={};seen=set()
        for name in ('native_pts','uv','profile_normal','kind','E','material_index','shot_index','admission','fit_role'):
            d=self.packet['arrays'][name];p=_load_ref(d,self.packet_path.parent,seen)
            self.arrays[name]=np.memmap(p,dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
        self.scope=legacy.checked(self.config['programme_scope']);self.roles=legacy.checked(self.config['frame_roles'])
        self.depbase,self.code,self.facts=reader_dependencies();self.board=None
        by_id={}
        for i in self.requested:
            if i<0 or i>=int(self.packet['row_count']):raise ValueError('Review row index outside exact packet')
            by_id.setdefault(self.packet['source_row_ids'][i],[]).append(i)
        self._by_index={};self._raw_refs={};self._qualified={}
        refs=list(self.config.get('source_journals',[]))
        if self.config.get('resolve_current_journals'):refs+=list(reversed(self.packet.get('source_family_journal_refs',[])))
        unique={str(Path(r.get('path',r.get('file'))).resolve()):r for r in refs}
        gates={}
        for ref in unique.values():
            path=_load_ref(ref,self.config_path.parent,seen)
            gatepath=path.parent/'source_qualification_manifest.json'
            if gatepath.is_file():
                g=read_json(gatepath)
                if g.get('schema')=='rinkfit.new_actual_type1_source_qualification_gate.v1' and g.get('input_cumulative_journal',{}).get('sha256')==ref['sha256']:
                    gates[str(gatepath)]=reference(gatepath)
        for ref in unique.values():
            path=_load_ref(ref,self.config_path.parent,seen)
            for j in read_jsonl(path):
                for i in by_id.get(j['source_row_id'],[]):
                    if i in self._by_index:continue
                    if int(j.get('native_pts',-1))==int(self.arrays['native_pts'][i]) and j.get('current_uv')==self.arrays['uv'][i].tolist():
                        self._by_index[i]=j;self._raw_refs[i]=ref
            # Generic exact matches are never overwritten by older journals.
            if len(self._by_index)==len(self.requested):break
        point_rows=[i for i in self.requested if self.arrays['kind'][i]==1 and self.arrays['admission'][i]==1 and self.arrays['fit_role'][i]==1]
        if point_rows and gates:
            qualified=PacketBoundSourceReview(self.packet_path,list(gates.values()),requested_row_indices=point_rows)
            for i,row in qualified.rows.items():
                self._by_index[i]=row['journal'];self._raw_refs[i]=row['journal_ref'];self._qualified[i]=row
        self._by_index={i:_exact_journal_corner_operator(row)for i,row in self._by_index.items()}
        self.journals={};self.journal_refs_by_id={}
        self._input_refs={}
        self.reader_version=VERSION;self.reader_module_ref=reference(Path(__file__))
        from rinkspace.storage import AssetStore
        folder=AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/review_inputs'/VERSION
        for i in self.requested:
            pts=int(self.arrays['native_pts'][i]);rid=self.packet['source_row_ids'][i]
            source=dict(video_sha256=read_json(self.scope)['source']['sha256'],time_base='1/12800',coordinate_domain='native_pixels',pts=pts)
            frame=self.config.get('frames',{}).get(str(pts),{});dependencies=[self.config['programme_scope'],self.config['frame_roles']]
            dependencies.extend(frame[k]for k in ('native_image','mask','mask_source_proof')if frame.get(k))
            if i in self._raw_refs:dependencies.append(self._raw_refs[i])
            if i in self._qualified:
                q=self._qualified[i];dependencies.extend([q['qualification_gate'],q['qualification_receipt']['ledger'],q['physical_instance_review']])
            binding=dict(schema='rinkfit.exact_packet_native_review_inputs.v1',reader_version=VERSION,
                source=source,observation=legacy.observation_content(self.packet,self.arrays,i),frame_inputs=frame,source_reader_policy=source_reader_policy(self.config),
                source_journal=self._raw_refs.get(i),selected_journal_current_uv=self._by_index.get(i,{}).get('current_uv'),
                source_journal_exact_current_mean=i in self._by_index,qualified_point_original_UV_exact=i in self._qualified,
                current_corner_operator_origin=self._by_index.get(i,{}).get('current_corner_operator_origin'),
                input_dependencies=dependencies)
            digest=object_sha256(binding);path=folder/(digest+'.json')
            if path.exists():
                if read_json(path)!=binding:raise ValueError('Immutable exact source input identity collision')
            else:folder.mkdir(parents=True,exist_ok=True);write_json(path,binding)
            self._input_refs[i]=reference(path)

    def has_current_cached_review(self,task):
        i=int(task['source_row_index']);source=task['context']['source']
        return cached_dependencies(source,self.packet,self.arrays,i,review_config=self.config_path)is not None

    def read_frame(self,pts,tasks,output,*,device='cuda'):
        self.journals={};self.journal_refs_by_id={}
        for task in tasks:
            i=int(task['source_row_index']);rid=self.packet['source_row_ids'][i]
            if rid in self.journals and self.journals[rid].get('current_uv')!=self._by_index.get(i,{}).get('current_uv'):
                raise ValueError('Distinct active measurement versions require separate row reviews')
            if i in self._by_index:self.journals[rid]=self._by_index[i]
            self.journal_refs_by_id[rid]=self._input_refs[i]
        return _READ_NATIVE_FRAME(self,pts,tasks,output,device=device)


# Production family routing facade. Historical pixel-reader bytes are archived
# and executed unchanged; texture source points never enter a corner operator.
def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    from .packet_feedback_facade import cached_dependencies as dispatch_cached
    return dispatch_cached(source,packet,arrays,index,review_config=review_config)

def PacketBoundCurrentReview(packet_path,config_path,*,requested_row_indices):
    from .packet_feedback_facade import PacketFeedbackReview
    return PacketFeedbackReview(packet_path,config_path,requested_row_indices=requested_row_indices)
