"""Route unknown board textures separately; execute the unchanged old reader.

The historical pixel reader is byte-archived, not reimplemented. Its logical
cache identity survives an unrelated packet append or numerical auxiliary state.
"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from . import source_recovery_review as legacy
from .io_utils import read_json,read_jsonl,file_sha256,object_sha256
from .packet_bound_source_review import reference

TEXTURE_PREFIX='static_board_texture/'
_FROZEN=None
_ARCHIVE=None
_MATCHES={}


def _archive():
    global _ARCHIVE
    if _ARCHIVE is None:
        from rinkspace.storage import AssetStore
        p=AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/reader_archives/exact_packet_native_feedback_e7127975/manifest.json'
        _ARCHIVE=read_json(p)
        expected='e7127975bc13a57c062e031e5e102381d8beec8aef7b03a6cd99625c0eb7f4be'
        canonical='a94ed687946a937cd15d8717cdb17c53506f06afb01f4b1954d22a80b523830a'
        if (_ARCHIVE.get('schema')!='rinkfit.executed_native_feedback_reader_archive.v1'
            or _ARCHIVE['original_module']['sha256']!=expected
            or _ARCHIVE['archive_module']['sha256']!=expected
            or object_sha256(_ARCHIVE['canonical_code'])!=canonical
            or _ARCHIVE['canonical_dependencies']['reader_code_sha256']!=canonical):
            raise ValueError('historical native reader archive identity differs')
        legacy.checked(_ARCHIVE['archive_module'])
    return _ARCHIVE


def _checked(ref):
    a=_archive();old=a['original_module']
    path=Path(ref.get('file',ref.get('path'))).resolve()
    if path==Path(old['file']).resolve()and ref['sha256']==old['sha256']:
        return legacy.checked(dict(file=str(Path(__file__).with_name('packet_feedback_reader_e712.py')),sha256=old['sha256']))
    return legacy.checked(ref)


def _frozen_reader():
    global _FROZEN
    if _FROZEN is not None:return _FROZEN
    a=_archive();archive=legacy.checked(dict(file=str(Path(__file__).with_name('packet_feedback_reader_e712.py')),sha256=a['archive_module']['sha256']));name='rinkfit_joint._native_feedback_e7127975'
    spec=importlib.util.spec_from_file_location(name,archive);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    def dependencies():
        # Code identity describes the actual historical pixel operator. The
        # original logical module bytes now live in the verified archive.
        code=[]
        for ref in a['canonical_code']:
            p=_checked(ref);code.append(reference(p))
        _,_,facts=legacy.reader_dependencies()
        return dict(a['canonical_dependencies'],effective_facts_sha256=object_sha256(facts)),code,facts
    m.reader_dependencies=dependencies
    m.cached_dependencies=_legacy_cached_dependencies
    _FROZEN=m
    return m


def _ordered_journals(packet,cfg):
    refs=list(cfg.get('source_journals',[]))
    if cfg.get('resolve_current_journals'):refs+=list(reversed(packet.get('source_family_journal_refs',[])))
    return list({str(Path(r.get('file',r.get('path'))).resolve()):r for r in refs}.values())


def _matches(ref,rid,pts,uv):
    key=(ref['sha256'],rid,int(pts),object_sha256(uv))
    if key not in _MATCHES:
        _MATCHES[key]=any(r.get('source_row_id')==rid and int(r.get('native_pts',-1))==int(pts)and r.get('current_uv')==uv for r in read_jsonl(_checked(ref)))
    return _MATCHES[key]


def _legacy_cached_dependencies(source,packet,arrays,index,*,review_config=None):
    m=_frozen_reader();path=m.cache_path(source,packet['source_dataset_ids'][index],packet['source_row_ids'][index])
    if not path.is_file():return None
    cached=read_json(path);deps=cached.get('review_dependencies',{})
    if not all(k in cached for k in ('material_label','shot_id','extra_fact_refs','evidence_refs')):return None
    current,_,facts=m.reader_dependencies();cases=legacy.effective_case_refs(source,cached['material_label'],cached['shot_id'])
    current['effective_facts_sha256']=object_sha256(dict(refs=facts+cached['extra_fact_refs'],case_versions=cases))
    if any(deps.get(k)!=v for k,v in current.items()):return None
    observation=legacy.observation_content(packet,arrays,index)
    if deps.get('observation_snapshot_sha256')!=object_sha256(observation):return None
    try:
        for ref in cached['evidence_refs']:_checked(ref)
        binding=read_json(_checked(cached['extra_fact_refs'][0]))
        if binding.get('schema')!='rinkfit.exact_packet_native_review_inputs.v1'or binding['observation']!=observation or binding['source']!=source:return None
        for ref in binding['input_dependencies']:_checked(ref)
        if review_config is not None:
            cfg=read_json(_checked(review_config)if isinstance(review_config,dict)else review_config)
            if cfg.get('frames',{}).get(str(source['pts']),{})!=binding['frame_inputs']or m.source_reader_policy(cfg)!=binding.get('source_reader_policy'):return None
            selected=binding.get('source_journal')
            if selected is None:return None # no current exact journal ownership proof
            rid=packet['source_row_ids'][index];uv=arrays['uv'][index].tolist();matched=False
            for ref in _ordered_journals(packet,cfg):
                same=(Path(ref.get('file',ref.get('path'))).resolve()==Path(selected.get('file',selected.get('path'))).resolve()and ref['sha256']==selected['sha256'])
                if same:matched=_matches(ref,rid,source['pts'],uv);break
                if _matches(ref,rid,source['pts'],uv):return None # a newer actual input takes priority
            if not matched:return None
    except (KeyError,ValueError,OSError):return None
    return deps


def _validate_config_reader_refs(cfg):
    for ref in cfg.get('actual_reader_module_refs',[]):legacy.checked(ref)
    contract=cfg.get('actual_reader_contract')
    if contract is not None and (contract.get('schema')!='rinkfit.exact_source_review_dispatch_contract.v2'
        or contract.get('mask_nonzero_means')!='excluded'
        or contract.get('board_texture_world_position')!='unknown_shared_2D_per_track'
        or contract.get('board_texture_known_world_corner')is not False):
        raise ValueError('unsupported exact native reader semantic contract')


def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    if review_config is not None:
        try:_validate_config_reader_refs(read_json(_checked(review_config)if isinstance(review_config,dict)else review_config))
        except (OSError,ValueError,KeyError):return None
    if packet['source_row_ids'][index].startswith(TEXTURE_PREFIX):
        from .static_board_feedback_review import cached_dependencies as texture_cached
        return texture_cached(source,packet,arrays,index,review_config=review_config)
    return _legacy_cached_dependencies(source,packet,arrays,index,review_config=review_config)


class PacketFeedbackReview:
    def __init__(self,packet_path,config_path,*,requested_row_indices):
        _validate_config_reader_refs(read_json(config_path))
        self.legacy=_frozen_reader().PacketBoundCurrentReview(packet_path,config_path,requested_row_indices=requested_row_indices)
        self.packet=self.legacy.packet;self.arrays=self.legacy.arrays
        texture=[i for i in self.legacy.requested if self.packet['source_row_ids'][i].startswith(TEXTURE_PREFIX)]
        self.texture=None
        if texture:
            from .static_board_feedback_review import StaticBoardTextureReview
            self.texture=StaticBoardTextureReview(self.legacy,requested_row_indices=texture)
        self.reader_version='exact_frozen_native_e712_and_static_board_texture_v1';self.reader_module_ref=reference(__file__)
    def has_current_cached_review(self,task):
        return cached_dependencies(task['context']['source'],self.packet,self.arrays,int(task['source_row_index']),review_config=self.legacy.config_path)is not None
    def read_frame(self,pts,tasks,output,*,device='cuda'):
        from .io_utils import write_json
        out=Path(output).resolve();old=[t for t in tasks if not self.packet['source_row_ids'][int(t['source_row_index'])].startswith(TEXTURE_PREFIX)];new=[t for t in tasks if self.packet['source_row_ids'][int(t['source_row_index'])].startswith(TEXTURE_PREFIX)]
        if not new:return self.legacy.read_frame(pts,old,out,device=device)
        if not old:return self.texture.read_frame(pts,new,out)
        if out.exists():raise FileExistsError('immutable mixed native review exists')
        out.mkdir(parents=True);a=self.legacy.read_frame(pts,old,out/'known_operators',device=device);b=self.texture.read_frame(pts,new,out/'board_textures')
        result=dict(schema='rinkfit.current_native_feedback_family_facade.v1',native_pts=int(pts),counts=dict(subjects=len(tasks),board_texture=len(new),known_operators=len(old)),resolutions=a['resolutions']+b['resolutions'],family_receipts=[reference(out/'known_operators/receipt.json'),reference(out/'board_textures/receipt.json')],source_UV_changed=False,new_observation_installed=False)
        write_json(out/'receipt.json',result);return result
