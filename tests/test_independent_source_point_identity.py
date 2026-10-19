from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'calibration/program_tracking'))
from calibration.program_tracking.independent_source_point_identity import source_point_identity_closure
from calibration.program_tracking.far_rail_source_identity_flow import expand_between_reviewed_seeds

def test_individual_blue_seeds_do_not_promote_entire_component_or_cross_blur_break():
    uv=np.column_stack([np.arange(7)*20,np.ones(7)*100.])
    bands=np.tile([-2.,2.],(7,1));bands[3]=[-30,30]
    labels=np.array([0,1,1,0,1,1,0],np.uint8)
    actual,steps=source_point_identity_closure(uv,bands,np.ones(7)*20,[0,7],np.empty((0,2),int),labels,expand_between_reviewed_seeds)
    assert actual.tolist()==[0,1,1,0,1,1,0]

def test_conflicting_actual_point_chain_does_not_silently_resolve_identity():
    with pytest.raises(ValueError,match='conflicting reviewed identities'):
        source_point_identity_closure([[0,0],[20,0]], [[-2,2],[-2,2]], [20,20], [0,2], [[0,1]], [1,2], expand_between_reviewed_seeds)
