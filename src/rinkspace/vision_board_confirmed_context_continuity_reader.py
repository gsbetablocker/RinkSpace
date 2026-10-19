"""Keep authenticated actual LK identity when an edge profile is still absent."""
from __future__ import annotations
import numpy as np
from .vision_board_verified_context_interface_reader import VerifiedContextInterfaceBoardSourceReader

class ConfirmedContextContinuityBoardSourceReader(VerifiedContextInterfaceBoardSourceReader):
    def _template_matches(self,gray,exclusion,bands,shot_id,*,tracked_names=()):
        found=super()._template_matches(gray,exclusion,bands,shot_id,tracked_names=tracked_names)
        clearance=getattr(self,'box_clearance',None);current=clearance.current if clearance is not None else None
        if (current is None or self.previous_gray is None or self.previous_shot!=shot_id
                or self.actual_successor.get(self.previous_actual_pts)!=int(current['binding']['actual_pts'])):return found
        continued=self._track(gray,current['hard'])
        for name,row in list(found.items()):
            tracked=continued.get(name);anchor=row['template']['anchor']
            if (row.get('current_instance_identity_confirmed') is True or tracked is None
                    or tracked.get('current_instance_identity_confirmed') is not True
                    or tracked['template']['anchor']['anchor_id']!=anchor['anchor_id']
                    or not self._reviewed_full_entity(anchor)):continue
            edge=np.asarray(row['search_uv'],float);centre=edge.mean(0)
            _,_,axes=np.linalg.svd(edge-centre,full_matrices=False)
            mismatch=np.median(np.abs((tracked['search_uv']-centre)@axes[-1]))
            # This is the inherited discovery/continuation conflict check,
            # not a source edge-location or scene fit accuracy threshold.
            if mismatch<=8.:found.pop(name)
        return found
