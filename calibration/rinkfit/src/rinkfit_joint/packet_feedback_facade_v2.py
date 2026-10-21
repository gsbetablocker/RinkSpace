"""Versioned exact-source dispatch including qualified S09 unknown board texture.

The frozen old reader and S02 texture namespace remain independently replayable.
No static-board texture is routed through a known-world-corner estimator.
"""
from pathlib import Path
from . import packet_feedback_facade as frozen
from . import static_board_feedback_review as oldtexture
from . import static_board_feedback_review_v2 as tailtexture
from .io_utils import read_json,write_json
from .packet_bound_source_review import reference
_archive=frozen._archive
_checked=frozen._checked
_frozen_reader=frozen._frozen_reader
_ordered_journals=frozen._ordered_journals
_validate_config_reader_refs=frozen._validate_config_reader_refs

def texture_reader(row_id):
    if row_id.startswith(tailtexture.PREFIX):return tailtexture
    if row_id.startswith(oldtexture.PREFIX):return oldtexture
    return None

def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    if review_config is not None:
        try:_validate_config_reader_refs(read_json(_checked(review_config)if isinstance(review_config,dict)else review_config))
        except (OSError,ValueError,KeyError):return None
    texture=texture_reader(packet['source_row_ids'][index])
    if texture is not None:return texture.cached_dependencies(source,packet,arrays,index,review_config=review_config)
    if packet['source_row_ids'][index].startswith('native_tail_C_v1/'):
        return None # exact paired C source reader is not yet installed; never corner
    return frozen._legacy_cached_dependencies(source,packet,arrays,index,review_config=review_config)

class PacketFeedbackReview:
    def __init__(self,packet_path,config_path,*,requested_row_indices):
        _validate_config_reader_refs(read_json(config_path))
        self.legacy=_frozen_reader().PacketBoundCurrentReview(packet_path,config_path,requested_row_indices=requested_row_indices)
        self.packet=self.legacy.packet;self.arrays=self.legacy.arrays;self.readers={}
        for module in (oldtexture,tailtexture):
            indices=[i for i in self.legacy.requested if texture_reader(self.packet['source_row_ids'][i])is module]
            if indices:self.readers[module.VERSION]=module.StaticBoardTextureReview(self.legacy,requested_row_indices=indices)
        self.reader_version='exact_native_family_dispatch_v2';self.reader_module_ref=reference(__file__)
    def has_current_cached_review(self,task):
        return cached_dependencies(task['context']['source'],self.packet,self.arrays,int(task['source_row_index']),review_config=self.legacy.config_path)is not None
    def read_frame(self,pts,tasks,output,*,device='cuda'):
        out=Path(output).resolve();groups={};receipts=[];resolutions=[]
        for task in tasks:
            rid=self.packet['source_row_ids'][int(task['source_row_index'])];module=texture_reader(rid)
            if module is None and rid.startswith('native_tail_C_v1/'):raise ValueError('C actual paired source reader pending; cannot use known-corner strategy')
            groups.setdefault(None if module is None else module.VERSION,[]).append(task)
        if len(groups)==1:
            key=next(iter(groups));return self.legacy.read_frame(pts,tasks,out,device=device)if key is None else self.readers[key].read_frame(pts,tasks,out)
        if out.exists():raise FileExistsError('immutable mixed native source review exists')
        out.mkdir(parents=True)
        for key,items in groups.items():
            folder=out/('known_operators'if key is None else key)
            result=self.legacy.read_frame(pts,items,folder,device=device)if key is None else self.readers[key].read_frame(pts,items,folder)
            resolutions+=result['resolutions'];receipts.append(reference(folder/'receipt.json'))
        result=dict(schema='rinkfit.current_native_feedback_family_facade.v2',native_pts=int(pts),counts=dict(subjects=len(tasks)),resolutions=resolutions,family_receipts=receipts,source_UV_changed=False,new_observation_installed=False)
        write_json(out/'receipt.json',result);return result

PacketBoundCurrentReview=PacketFeedbackReview
