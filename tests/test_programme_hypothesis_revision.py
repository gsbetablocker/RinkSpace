"""Scale/depth input revisions retain a track choice, not stale geometry masks."""
import numpy as np
from calibration.program_tracking.body_tracking.programme_continue import inherited_episode_hypotheses


def test_geometry_associations_are_rebuilt_without_losing_track_hypothesis():
    prefix='source_association_extra_track_'
    parent={'source_evidence_sha256':np.array('same'),prefix+'choice':np.array(1),
        prefix+'fixed_choice':np.array([4,5]),prefix+'visible_mask':np.array([True,False]),
        prefix+'candidate_costs':np.array([2.,3.])}
    child=inherited_episode_hypotheses(parent,'same')
    assert set(child)=={prefix+'choice'}
    assert int(child[prefix+'choice'])==1
    assert inherited_episode_hypotheses(parent,'different evidence')=={}
