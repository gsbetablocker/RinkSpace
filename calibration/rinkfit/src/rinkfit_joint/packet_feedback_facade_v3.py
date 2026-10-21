"""Next-boundary dispatch for the actual two C paint material interfaces."""
from pathlib import Path
from . import packet_feedback_facade_v2 as frozen
from . import paint_pair_feedback_review_v1 as paint
from .io_utils import read_json,write_json
from .packet_bound_source_review import reference
_archive=frozen._archive
_checked=frozen._checked
_frozen_reader=frozen._frozen_reader
_ordered_journals=frozen._ordered_journals
_validate_config_reader_refs=frozen._validate_config_reader_refs

def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    if packet['source_row_ids'][index].startswith(paint.PREFIX):
        if review_config is not None:
            try:_validate_config_reader_refs(read_json(_checked(review_config)if isinstance(review_config,dict)else review_config))
            except (OSError,KeyError,ValueError):return None
        return paint.cached_dependencies(source,packet,arrays,index,review_config=review_config)
    return frozen.cached_dependencies(source,packet,arrays,index,review_config=review_config)

class PacketFeedbackReview:
    def __init__(self,packet_path,config_path,*,requested_row_indices):
        doc=read_json(packet_path);paired={i for i in requested_row_indices if doc['source_row_ids'][i].startswith(paint.PREFIX)};del doc
        self.other=frozen.PacketFeedbackReview(packet_path,config_path,requested_row_indices=set(requested_row_indices)-paired)
        self.legacy=self.other.legacy;self.packet=self.other.packet;self.arrays=self.other.arrays
        self.paint=paint.PaintPairSourceReview(self.legacy,requested_row_indices=paired)if paired else None
        self.reader_version='exact_native_family_dispatch_with_paint_interfaces_v3';self.reader_module_ref=reference(__file__)
    def has_current_cached_review(self,task):
        return cached_dependencies(task['context']['source'],self.packet,self.arrays,int(task['source_row_index']),review_config=self.legacy.config_path)is not None
    def read_frame(self,pts,tasks,output,*,device='cuda'):
        out=Path(output).resolve();new=[t for t in tasks if self.packet['source_row_ids'][int(t['source_row_index'])].startswith(paint.PREFIX)];old=[t for t in tasks if not self.packet['source_row_ids'][int(t['source_row_index'])].startswith(paint.PREFIX)]
        if not new:return self.other.read_frame(pts,old,out,device=device)
        if not old:return self.paint.read_frame(pts,new,out)
        if out.exists():raise FileExistsError('Immutable mixed source family review exists')
        out.mkdir(parents=True);a=self.other.read_frame(pts,old,out/'existing_source_operators',device=device);b=self.paint.read_frame(pts,new,out/'paint_interfaces')
        result=dict(schema='rinkfit.current_native_feedback_family_facade.v3',native_pts=int(pts),counts=dict(subjects=len(tasks)),resolutions=a['resolutions']+b['resolutions'],family_receipts=[reference(out/'existing_source_operators/receipt.json'),reference(out/'paint_interfaces/receipt.json')],source_UV_changed=False,new_observation_installed=False)
        write_json(out/'receipt.json',result);return result

PacketBoundCurrentReview=PacketFeedbackReview
