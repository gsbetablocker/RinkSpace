"""Review-only reverse source continuity with the same actual RGB reader."""
from __future__ import annotations
import numpy as np
from .vision_board_context_seed_reader import ContextTemplateBoardSourceReader

class BackwardContextTemplateBoardSourceReader(ContextTemplateBoardSourceReader):
    def __init__(self,*args,actual_exposure_pts=None,**kwargs):
        if actual_exposure_pts is None:raise ValueError('Verified source exposure clock required')
        verified=np.asarray(actual_exposure_pts,np.int64)
        super().__init__(*args,actual_exposure_pts=verified,**kwargs)
        self.actual_successor=dict(zip(map(int,verified[1:]),map(int,verified[:-1])))

    def read(self,*args,**kwargs):
        binding=args[2] if len(args)>2 else kwargs['binding']
        native=int(binding.get('source_pts',binding.get('native_pts')))
        previous=getattr(self,'last_backward_native',None)
        if previous is not None and native>=previous:
            raise ValueError('Reverse board review requires strictly descending original native PTS')
        # The inherited guard enforces an ascending traversal. The independent
        # guard above enforces this traversal instead. All source bindings and
        # previous RGB/actual exposure evidence remain their exact originals.
        self.previous_pts=None
        result=super().read(*args,**kwargs)
        self.last_backward_native=native
        result['source_review_direction']='backward'
        result['source_clock_not_reversed_or_relabelled']=True
        for row in result['advertisement_rows']:
            temporal=row['temporal_evidence']
            if temporal['kind']=='same_shot_actual_RGB_temporal_identity':
                if temporal['previous_actual_pts']<=row['actual_pts']:
                    raise ValueError('Reverse continuity must bind a later exact source exposure')
                temporal['direction']='backward'
            row['source_review_layer_only']=True
        return result
