"""Preparation guards for changed physical meaning and periodic station bounds."""
import numpy as np
import pytest
import json
from copy import deepcopy
from pathlib import Path
from rinkspace.session.prepare import validate_current_feature_contracts
from rinkspace.session.dataset import DatasetError, load_prepared, write_array, write_json, iter_current_observations
from rinkspace.domain.measurement import POLICY_ID, MeasurementPolicyError
from rinkspace.domain.print_assets import PrintBindingError


def fixture():
    features=[dict(id='c0',label='ice.C0:centreline',kind='finite_curve',measurement='centreline')]
    arrays=dict(pts=np.array([256256],dtype='<i8'),active=np.array([1],dtype='u1'),feature_index=np.array([0],dtype='<i4'),
                kind=np.array([1],dtype='u1'),local_index=np.array([0],dtype='<i8'),support=np.array([[0.,1.]],dtype='<f8'))
    return features,arrays,np.array([0],dtype='<i8')


def test_original_outer_edge_cannot_enter_current_centreline_model():
    f,a,rows=fixture();f[0]['original_measurement']='outer_boundary'
    with pytest.raises(MeasurementPolicyError):
        validate_current_feature_contracts({},dict(measurement_policy_id=POLICY_ID),f,a,rows)


def test_continuous_board_requires_explicit_nonbinding_support_kind():
    f,a,rows=fixture();f[0].update(label='board.bottom',native_kind=7,
        station_domain=dict(kind='unbounded_periodic',hard_segment_bounds=False))
    with pytest.raises(DatasetError):validate_current_feature_contracts({}, {}, f,a,rows)
    a['support_kind']=np.array([1],dtype='u1');a['support'][:]=0
    result=validate_current_feature_contracts({}, {}, f,a,rows)
    assert result['periodic_station_rows']==1
    a['support'][:]=[0.,1.]
    with pytest.raises(DatasetError):validate_current_feature_contracts({}, {}, f,a,rows)


def test_current_policy_is_mandatory_without_opt_in():
    f,a,rows=fixture();f[0]['measurement']='outer_boundary'
    with pytest.raises(MeasurementPolicyError):
        validate_current_feature_contracts({}, {}, f,a,rows)


@pytest.mark.parametrize('active', [0, 1])
def test_obsolete_scene_path_rejected_even_when_feature_registry_is_canonical(active):
    f,a,rows=fixture();a['active'][:]=active
    scene={'paths':[dict(id='unused-C0-path',label='ice.C0:outer',kind='ice_circle',measurement='outer_boundary')]}
    before=deepcopy(scene)
    with pytest.raises(MeasurementPolicyError):validate_current_feature_contracts({},scene,f,a,rows)
    assert scene==before


@pytest.mark.parametrize('field,value', [('original_measurement','outer_boundary'),('model_line_width',.01)])
def test_scene_rejects_alternate_definition_and_nonzero_midline_width(field,value):
    f,a,rows=fixture()
    path=dict(id='C0-path',label='ice.C0:centreline',kind='ice_circle',measurement='centreline')
    path[field]=value
    with pytest.raises(MeasurementPolicyError):validate_current_feature_contracts({}, {'paths':[path]},f,a,rows)


def test_all_current_ring_layers_and_ellipse_exceptions_remain_admissible():
    root=Path(__file__).resolve().parents[1]
    scene=json.loads((root/'events/ankara_2026/scene/field/model.json').read_text(encoding='utf-8'))
    f,a,rows=fixture()
    result=validate_current_feature_contracts({},scene,f,a,rows,
        scene_path=root/'events/ankara_2026/scene/field/model.json')
    assert result['scene_measurement']['features']==66
    assert sum(100<=p.get('generated_code',-1)<132 for p in scene['paths'])==32
    assert [p['measurement'] for p in scene['paths'] if p.get('generated_code') in (410,411)]==['contour']*2


@pytest.mark.parametrize('active', [0, 1])
def test_segment_board_cannot_omit_periodic_declaration(active):
    f,a,rows=fixture();a['active'][:]=active
    f[0].update(label='board.bottom',surface='board',native_kind=3,parameters=[0.,0.],finite_support=[0.,1.])
    with pytest.raises(DatasetError):validate_current_feature_contracts({}, {}, f,a,rows)


def test_boundary_identity_from_scene_cannot_be_hidden_by_feature_metadata():
    f,a,rows=fixture()
    scene={'paths':[dict(id='c0',label='board.bottom',kind='board_boundary',measurement='centreline',parameters=[0.,0.])]}
    with pytest.raises(DatasetError):validate_current_feature_contracts({},scene,f,a,rows)
    scene['paths'][0].update(native_kind=7,parameters=[0.],station_domain={'kind':'unbounded_periodic','hard_segment_bounds':False})
    with pytest.raises(DatasetError):validate_current_feature_contracts({},scene,f,a,rows)


def test_finite_board_artwork_and_fixed_points_are_not_reclassified_as_boundaries():
    f,a,rows=fixture()
    f[0].update(label='Ad2.print',surface='board',native_kind=5,finite_support=[0.,1.])
    # A bare kind5 declaration cannot supply a material identity. Actual bound
    # print curves are covered by test_print_measurement, without kind7 stations.
    with pytest.raises(PrintBindingError):validate_current_feature_contracts({}, {}, f,a,rows)
    f[0].pop('native_kind')
    f[0].update(label='board.top',kind='material_point',measurement='material_point')
    a['kind'][:]=0;a['local_index'][:]=-1;a['support'][:]=0
    assert validate_current_feature_contracts({}, {}, f,a,rows)['periodic_station_rows']==0


def prepared_fixture(tmp_path, records, *, active=1):
    f,a,_=fixture();a['active'][:]=active
    a.update(uv=np.array([[501.,603.]],dtype='<f8'),noise_group_weights=np.array([1.,0.],dtype='<f8'))
    refs={k:write_array(tmp_path,k,v) for k,v in a.items()}
    write_json(tmp_path/'scene.json',{'paths':[dict(id='c0',label='ice.C0:centreline',kind='ice_circle',measurement='centreline')]})
    text=''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records)
    (tmp_path/'observations.jsonl').write_text(text,encoding='utf-8')
    write_json(tmp_path/'manifest.json',dict(schema='rinkspace.prepared.v1',scene_ref='scene.json',arrays=refs,features=f,observations_ref='observations.jsonl'))
    return a


@pytest.mark.parametrize('active', [0, 1])
@pytest.mark.parametrize('change', [dict(measurement='outer_boundary'),
    dict(source_measurement_evidence={'measurement':'outer_boundary'}),dict(original_measurement='outer_boundary'),dict(feature_id='wrong')])
def test_fit_loader_rejects_conflicting_current_row_metadata(tmp_path,active,change):
    record={'feature_id':'c0','measurement':'centreline',**change}
    prepared_fixture(tmp_path,[record],active=active)
    before=(tmp_path/'observations.jsonl').read_bytes()
    with pytest.raises(MeasurementPolicyError):load_prepared(tmp_path,purpose='fit')
    assert (tmp_path/'observations.jsonl').read_bytes()==before
    # Explicit historical review stays read-only and can inspect failures.
    load_prepared(tmp_path,purpose='review')


def test_fit_loader_preserves_current_pixels_pts_weights_and_does_not_replay_provenance(tmp_path):
    record={'feature_id':'c0','measurement':'centreline','source_measurement_evidence':{'measurement':'centreline'},
            'provenance':{'file':'missing_historical_package.json'},'source_point_id':'original-17','noise_group_id':'W17'}
    expected=prepared_fixture(tmp_path,[record])
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    _,actual=load_prepared(tmp_path,purpose='fit')
    for key,value in expected.items():np.testing.assert_array_equal(actual[key],value)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('count',[0,2])
def test_fit_loader_rejects_truncated_or_extra_row_metadata(tmp_path,count):
    prepared_fixture(tmp_path,[dict(feature_id='c0',measurement='centreline')]*count)
    with pytest.raises(DatasetError):load_prepared(tmp_path,purpose='fit')


def test_prepare_metadata_selection_preserves_exact_rows_without_rechecking_unselected_sources(tmp_path):
    records=[dict(feature_id='c0',measurement='outer_boundary'),dict(feature_id='c0',measurement='centreline')]
    path=tmp_path/'rows.jsonl'
    lines=[json.dumps(r)+'\n' for r in records];path.write_text(''.join(lines),encoding='utf-8')
    f,_,_=fixture()
    assert list(iter_current_observations(path,f,np.array([0,0]),mask=np.array([False,True])))==lines[1:]
