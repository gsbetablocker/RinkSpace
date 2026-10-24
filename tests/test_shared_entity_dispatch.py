"""Current-scene dispatch must retain hard geometry and physical curve kinds."""
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore, NativeError
from rinkspace.compat.d78_entities import native_definition, sample_path, sample_entities
from rinkspace.domain.entity_graph import resolve_path, fixed_node_graph
from rinkspace.domain.field_model import sample_model

pytestmark = pytest.mark.skipif(not os.environ.get('RINKSPACE_CORE_LIBRARY'), reason='explicit native library required')


@pytest.fixture
def model():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/'events/ankara_2026/scene/field/model.json').read_text(encoding='utf-8'))


@pytest.fixture
def core():
    return NativeCore(os.environ['RINKSPACE_CORE_LIBRARY'])


def path_for(model,code):
    return next(p for p in model['paths'] if p.get('generated_code')==code)


def test_generic_resolver_preserves_four_way_faceoff_symmetry(model,core):
    q=model['shared_layout']
    expected=[[q[25],-q[27],0.],[q[25],q[27],0.],[-q[25],q[27],0.],[-q[25],-q[27],0.]]
    centres=[]
    for code in range(406,410):
        p=resolve_path(model,path_for(model,code),core)
        centres.append(p['parameters'][:3])
        assert p['parameters'][3]==q[20]
        sampled=sample_path(model,path_for(model,code),core,[0.])[0]
        np.testing.assert_allclose(sampled,np.array(expected[code-406])+[q[20],0.,0.],atol=1e-15)
    np.testing.assert_array_equal(centres,expected)


def test_generic_resolver_uses_current_carriers_and_centred_rows(model,core):
    changed=deepcopy(model)
    changed['shared_layout'][2]=.9  # inactive old y0 slot
    changed['shared_layout'][18]=.3  # inactive old paint rotation
    for code in range(400,405):
        p=resolve_path(changed,path_for(changed,code),core)['parameters']
        assert p[0]==p[3]
        assert p[1]==-p[4]
        assert p[1]>0
    q=changed['shared_layout']
    for row in range(4):
        p=resolve_path(changed,path_for(changed,103+row*4),core)['parameters']
        assert p[1]==pytest.approx((-1.5+row)*q[3],abs=1e-15)


@pytest.mark.parametrize('code',[410,411])
def test_half_ellipse_is_not_misclassified_as_circle(model,core,code):
    path=path_for(model,code)
    resolved=resolve_path(model,path,core)
    assert resolved['kind']=='ice_ellipse' and resolved['native_kind']==6
    p=resolved['parameters'];xyz=sample_path(model,path,core,np.linspace(0.,1.,25))
    np.testing.assert_allclose(((xyz[:,0]-p[0])/p[3])**2+((xyz[:,1]-p[1])/p[4])**2,1.,atol=1e-13)
    np.testing.assert_allclose(xyz[[0,-1],0],p[0],atol=1e-15)


def test_generic_inventory_keeps_all_paths_and_samples_entire_board_course(model,core):
    before=deepcopy(model)
    sampled=sample_entities(model,core,samples=65)
    assert sampled['inventory']['path_count']==66
    assert set(sampled['inventory']['path_ids'])=={p['id'] for p in model['paths']}
    assert sum(c['kind']=='ice_circle' for c in sampled['cuts'])==37
    assert sum(c['kind']=='ice_ellipse' for c in sampled['cuts'])==2
    boards=[c for c in sampled['cuts'] if c['kind']=='board_course']
    assert len(boards)==2
    for cut in boards:
        assert cut['support'] is None
        assert cut['sampling_interval']==[0.,core.board_perimeter(model['board'])]
        xyz=sampled['xyz'][cut['begin']:cut['end']]
        np.testing.assert_allclose(xyz[0],xyz[-1],atol=1e-14)
        assert np.ptp(xyz[:,0])>.95
        assert np.ptp(xyz[:,1])>.45
    assert model==before


def test_fixed_node_graph_uses_same_current_definitions_without_new_observations(model,core):
    expected=deepcopy(model)
    expected['paths']=[resolve_path(model,p,core) for p in model['paths']]
    for path in expected['paths']:path.pop('generated_code',None)
    actual=fixed_node_graph(model,core)
    assert actual==fixed_node_graph(expected,core)
    assert actual['fixed_junctions']>0
    assert actual['new_source_measurements']==0
    assert all(n['source_uv'] is None and not n['per_frame_station_parameter'] for n in actual['nodes'])


@pytest.mark.parametrize('endpoint_kind',['before_R','at_R','at_N','past_N'])
def test_bad_connections_fail_all_current_entry_points(model,core,endpoint_kind):
    q=model['shared_layout']
    q[15]={'before_R':q[1]-.01,'at_R':q[1],'at_N':q[8],'past_N':q[8]+.01}[endpoint_kind]
    path=path_for(model,201)
    for call in [lambda:native_definition(model,path,core),lambda:resolve_path(model,path,core),
                 lambda:sample_path(model,path,core,[0.,1.]),lambda:sample_model(model,core),
                 lambda:sample_entities(model,core),lambda:fixed_node_graph(model,core)]:
        with pytest.raises(NativeError):call()


def test_absent_options_preserve_explicit_legacy_replay(model,core):
    legacy=deepcopy(model);legacy.pop('shared_geometry_options')
    with pytest.raises(ValueError):native_definition(legacy,path_for(legacy,406),core)
    legacy['schema']='rinkspace.scene.shared.v1'
    path=path_for(legacy,406)
    kind,parameters=native_definition(legacy,path,core)
    assert (kind,parameters)==core.shared_geometry(legacy['board'],legacy['shared_layout'],406,options=0)
    assert resolve_path(legacy,path,core)['parameters']==parameters
