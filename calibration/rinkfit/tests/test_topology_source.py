import importlib.util,json
from pathlib import Path
import pytest
from rinkfit_start.data import sha256_file
from rinkfit_start.topology_repair import canonical_sha,restore_explicit_topology


spec=importlib.util.spec_from_file_location('topology_exporter',Path(__file__).parents[1]/'tools/build_inputs.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)


def test_explicit_closed_contour_need_not_repeat_endpoint():
    desc=dict(kind='ad_print',advertisement_id=10,contour_key='ink',geometry_source='scene_existing_external_contour')
    contour=dict(id='ink',closed=True,xyz_scene_units=[[0,0,0],[1,0,0],[0,1,0]])
    scene=dict(source_conditioned_board_prints=dict(patterns=[dict(advertisement_id=10,external_contours=[contour])]))
    result=exporter.explicit_contour_topology(scene,desc,3,'sourcehash')
    assert result['path_closed']==[True] and result['path_offsets']==[0,3]
    assert not result['topology_provenance']['endpoint_distance_used']
    # A repeated endpoint alone does not authenticate closed source topology.
    contour.pop('closed');contour['xyz_scene_units'][-1]=[0,0,0]
    assert exporter.explicit_contour_topology(scene,desc,3,'sourcehash') is None
    contour['closed']=False
    assert exporter.explicit_contour_topology(scene,desc,3,'sourcehash')['path_closed']==[False]
    scene['advertisements']=[dict(user_number=10,external_contours=[dict(id='ink',closed=True)])]
    with pytest.raises(ValueError,match='Conflicting'):
        exporter.explicit_contour_topology(scene,desc,3,'sourcehash')


def test_named_template_explicit_flag_and_escaped_pointer():
    desc=dict(kind='ad_print',advertisement_id=3,contour_key='a/b',template='material.json')
    templates=dict(advertisements={'3':dict(contours={'a/b':dict(closed=True)})})
    topology=exporter.explicit_contour_topology({},desc,3,'sourcehash',templates)
    assert topology['path_closed']==[True]
    assert topology['topology_provenance']['json_pointers']==['/payload/templates/advertisements/3/contours/a~1b/closed']
    # An explicitly named complete circle retains its known closed primitive;
    # no repeated sampled endpoint is needed to recover its topology.
    circle=dict(kind='ad_print',advertisement_id=11,contour_key='ring',physical_primitive='circle_on_continuous_unwrapped_board')
    assert exporter.explicit_contour_topology({},circle,100,'sourcehash')['path_closed']==[True]


def test_repair_requires_hashed_explicit_flag_and_preserves_parameters(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps(dict(contour=dict(id='ink',closed=True))))
    path=dict(id='w0/path0',variant_id='w0',kind=2,closed=False,points=[[0,0],[1,0],[0,1]])
    model=dict(blocks=[dict(name='global',values=[1.,2.])],camera=dict(radius_max=.4),materials=[dict(label='Ad10.print.ink',paths=[path])],provenance={})
    original=tmp_path/'original.json';original.write_text(json.dumps(model))
    mapping=dict(model_sha256=sha256_file(original),source_path=str(source),source_sha256=sha256_file(source),source_sha_matches_startup_seed_source_scene=True,derived_from_endpoint_distance=False,
        positive_closure_mismatches=[dict(material_index=0,path_index=0,material_label='Ad10.print.ink',path_id='w0/path0',variant_id='w0',original_path_sha256=canonical_sha(path),explicit_source_closed=True,source_pointer='/contour/closed',descriptor=dict(contour_key='ink'))])
    catalog=tmp_path/'mapping.json';catalog.write_text(json.dumps(mapping))
    repaired,report=restore_explicit_topology(original,catalog,tmp_path/'new')
    p=repaired['materials'][0]['paths'][0]
    assert p['closed'] and p['points']==[[0,0],[1,0],[0,1],[0,0]]
    assert repaired['blocks']==model['blocks'] and repaired['camera']==model['camera']
    assert report['changes'][0]['old_station_to_same_world_station_scale']==2/3
    # Source flag changes are rejected even if someone updates the outer hash.
    source.write_text(json.dumps(dict(contour=dict(id='ink',closed=False))))
    mapping['source_sha256']=sha256_file(source);catalog.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='Only explicit'):
        restore_explicit_topology(original,catalog,tmp_path/'invalid')
