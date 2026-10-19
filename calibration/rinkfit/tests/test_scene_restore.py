from copy import deepcopy
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from rinkfit_start.native_scene import NativeScene

_spec=importlib.util.spec_from_file_location('rinkfit_conditional_w_restore',Path(__file__).resolve().parents[1]/'tools/reduced/conditional_W_ad_chart.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
restore=_module.original_ice_endpoint_corrections


def model(endpoint):
    g=np.zeros(32);g[:5]=[.5,.12,.08,1.9051596,.02]
    return {'blocks':[{'name':'global','values':g.tolist()},
        {'name':'ice/line/example','values':[-.3,0.,*endpoint],'constant':False}],
        'materials':[{'label':'ice.example:segment0','paths':[{'kind':0,'geometry_block':1}]}]}


@pytest.mark.parametrize('endpoint',[(.499,.23),(-.499,-.23),(.501,.02)])
def test_original_rounded_rink_restoration_changes_only_existing_endpoint(endpoint):
    source=model(endpoint);before=deepcopy(source);assert not NativeScene(source).check_physics()['passed']
    corrections=restore(source);assert corrections
    assert source==before
    repaired=deepcopy(source)
    for change in corrections:
        assert change['block']==1 and change['component']in(2,3)
        repaired['blocks'][1]['values'][change['component']]=change['after']
    assert NativeScene(repaired).check_physics(tolerance=1e-12)['passed']
    assert repaired['blocks'][0]==before['blocks'][0]
    assert repaired['blocks'][1]['values'][:2]==before['blocks'][1]['values'][:2]
    assert len(repaired['blocks'][1]['values'])==4
    if endpoint[1]==.02:assert repaired['blocks'][1]['values'][3]==.02


def test_legal_endpoint_is_bitwise_unchanged_and_shared_path_is_repaired_once():
    legal=model((.45,.12));assert restore(legal)==[]
    outside=model((.499,.23));outside['materials'][0]['paths']*=2
    changes=restore(outside)
    assert len(changes)==2
    assert len({(x['block'],x['component'])for x in changes})==2


def test_fixed_existing_endpoint_cannot_be_silently_released():
    source=model((.499,.23));source['blocks'][1]['constant']=True
    with pytest.raises(ValueError,match='fixed endpoint'):restore(source)
    source['blocks'][1]['constant']=False;source['blocks'][1]['fixed_indices']=[2]
    with pytest.raises(ValueError,match='fixed endpoint component'):restore(source)
