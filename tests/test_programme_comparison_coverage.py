import numpy as np
from run_native_programme import comparison_exposure_coverage


def test_missing_exposure_stays_in_the_complete_comparison():
    manifest={'shots':[dict(id='s03',first_pts=720896,last_pts=721152,row_begin=0,row_stop=2)]}
    mapping={'path_ids':['ice','print']}
    observations={'pts':np.array([720896,720896]),'path_indices':np.array([0,1])}
    errors=[np.array([2.,4.]),np.array([1.,4.])]
    result=comparison_exposure_coverage(manifest,mapping,observations,errors,{'ice':'ice','print':'advertisement'})
    assert result['source_exposures']==2
    assert result['frames'][1]['before']['rows']==0
    assert result['frames'][1]['ice_after']['rms'] is None
    assert result['frames'][0]['ice_before']['rms']==2.
    assert result['frames'][0]['ice_after']['rms']==1.
    assert result['shots']['s03']['layers']['advertisement']['after']['rms']==4.
