"""Next-boundary exact-source batch index including actual paint interfaces."""
from pathlib import Path
import time
from . import batch_review_dependencies_v2 as frozen
from . import packet_feedback_facade_v5 as facade
from . import paint_pair_feedback_review_v1 as paint
from . import original_paint_pair_feedback_review_v1 as original
from . import batch_review_dependencies_v4 as previous
from .io_utils import read_json,file_sha256,object_sha256
from .packet_bound_source_review import reference
from . import source_recovery_review as legacy
from . import r_inner_feedback_review_v1 as Rinner

class ExactReviewDependencyBatch(previous.ExactReviewDependencyBatch):
    def __init__(self,packet,arrays,review_config,subjects,*,project=None,event='ankara_2026'):
        from rinkspace.storage import AssetStore
        self.packet=packet;self.arrays=arrays;self.project=Path(project or legacy.PROJECT);self.event=event
        self.verifier=frozen.VerifiedReferences();self.config_ref=review_config if isinstance(review_config,dict)else dict(path=str(Path(review_config).resolve()),sha256=file_sha256(review_config));self.cfg=self.verifier.read(self.config_ref)
        facade._validate_config_reader_refs(self.cfg)
        for ref in self.cfg.get('actual_reader_module_refs',[]):self.verifier.check(ref)
        optional=self.cfg.get('actual_batch_dependency_helper_ref')
        if optional and self.verifier.check(optional)!=Path(__file__).resolve():raise frozen.BatchInputChanged('batch helper module identity differs')
        self.subjects=list(subjects);self.started=time.perf_counter();self.records=[];self.prepared=False;self.validated=False;self.json_cache={};self.journal_stats=[]
        self.cache_root=AssetStore(self.project).data/'events'/event/'feedback/review_cache';self.authority=frozen.AuthoritySnapshot(self.project,event,self.verifier)
        self.frozen=facade._frozen_reader();self.old_base,self.old_code,self.facts=self.frozen.reader_dependencies()
        for ref in self.old_code+self.facts:self.verifier.check(ref)
        self.texture_base,self.texture_code=frozen.texture._dependencies(self.cfg);self.tail_texture_base,self.tail_texture_code=frozen.tailtexture._dependencies(self.cfg)
        for ref in self.texture_code+self.tail_texture_code:self.verifier.check(ref)
        self.ordered=facade._ordered_journals(packet,self.cfg);self.order_keys=[(str(frozen._path(x)).casefold(),x['sha256'])for x in self.ordered]
        self.paint_base,self.paint_code=paint._dependencies(self.cfg)
        for ref in self.paint_code:self.verifier.check(ref)
        self.original_base,self.original_code=original._dependencies(self.cfg)
        for ref in self.original_code:self.verifier.check(ref)
        self.Rinner_base,self.Rinner_code=Rinner._dependencies(self.cfg)
        for ref in self.Rinner_code:self.verifier.check(ref)
    def _one(self,item):
        i=int(item['row_index']);source=item['source'];rid=self.packet['source_row_ids'][i]
        if not rid.startswith(Rinner.PREFIX):return super()._one(item)
        record=dict(row_index=i,source=source,source_row_id=rid,status='unknown',dependencies=None)
        path=Rinner._cache(source,self.packet,i,self.Rinner_base['reader_code_sha256'])
        if not path.is_file():record['reason']='no_completed_exact_R_inner_reader_cache';return record
        cache=read_json(path);record['cache']=dict(path=str(path),sha256=file_sha256(path));self.verifier.check(record['cache']);deps=cache.get('review_dependencies',{})
        if any(deps.get(k)!=v for k,v in self.Rinner_base.items()):record['reason']='R_inner_reader_or_policy_changed';return record
        inputs=Rinner._inputs(source,self.packet,self.arrays,i,self.cfg)
        if deps.get('actual_source_inputs_sha256')!=object_sha256(inputs):record['reason']='R_inner_actual_source_inputs_changed';return record
        if self._json(cache['input_binding'])!=inputs:record['reason']='R_inner_binding_changed';return record
        for ref in cache['evidence_refs']:self.verifier.check(ref)
        record.update(dependencies=deps,candidate_dependencies=deps,observation=legacy.observation_content(self.packet,self.arrays,i),status='exact_source_review_reusable');return record
    def receipt(self):
        result=super().receipt();result['helper']=reference(__file__);result['delegated_existing_source_helper']=reference(previous.__file__);result['paint_interface_source_inputs_are_model_independent']=True;return result
