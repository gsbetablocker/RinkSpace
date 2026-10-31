import numpy as np
import json
from types import SimpleNamespace
import pytest
import qualify_visible_source_batch as qualifier
from qualify_visible_source_batch import thin,footprint_members
from prepare_native_source_update import authenticate_parent_observation_ids

def rows(xs):
    return [dict(uv=[float(x),100.],normal=[0.,1.],search_uv=[float(x),100.],
        _profile_radius=10.,_tangent_half_width=1.)for x in xs]

def test_sampling_keeps_real_fragment_endpoints_without_filling_gap():
    source=rows(list(range(0,201,8))+list(range(400,601,8)))
    selected=thin(source)
    assert all(any(r is s for s in source)for r in selected)
    assert {0.,200.,400.,600.}<={r['uv'][0]for r in selected}
    assert len(selected)<len(source)/2
    assert not any(200<r['uv'][0]<400 for r in selected)

def test_short_glyph_is_retained_without_repeating_points():
    source=rows(range(0,33,8));selected=thin(source)
    assert 4<=len(selected)<=len(source)
    assert len({id(r)for r in selected})==len(selected)

def test_same_named_remote_board_edge_is_outside_current_profile():
    source=rows([20,28,36])
    old=np.array([[20,101],[28,92],[20,20],[200,100],[24,100]],float)
    assert footprint_members(old,source).tolist()==[True,True,False,False,False]


@pytest.fixture
def source_batch(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    old=dict(pts=np.full(5,1000,np.int64),uv=np.array([[20.,101.],[20.,20.],[20.,100.],[24.,100.],[200.,100.]]),
        kind=np.ones(5,np.int64),path_indices=np.array([0,0,1,1,1],np.int32),
        observation_ids=np.array(['near','far','sampled-print','unsampled-gap','remote-print'],dtype='U96'),
        source_rows=np.arange(5,dtype=np.int64),stations=np.arange(5,dtype=float)/10)
    np.savez(prepared/'point_observations.npz',**old)
    mapping=dict(path_ids=['board/continuous/top','print_instance/Ad7::Ad7.A'],source_video_sha256='a'*64)
    manifest=dict(point_input_sha256=qualifier.sha(prepared/'point_observations.npz'),source_video_sha256='a'*64,
        shots=[dict(id='s07',first_pts=1000,last_pts=1000,stop_pts_exclusive=1256,row_begin=0,row_stop=5)])
    qualifier.write(prepared/'mapping.json',mapping);qualifier.write(prepared/'joint_manifest.json',manifest)
    roots=[]
    for family in ('print','near'):
        root=tmp_path/family;(root/'s07').mkdir(parents=True);roots.append(root)
        near=family=='near'
        binding=dict(schema='rinkspace.current_near_blue_cap_capture.v1'if near else'rinkspace.current_native_whole_print_RGB_capture.v1',
            shots=['s07'],query_spacing_px=8.)
        actual=[]
        for x in (20.,28.):
            row=dict(path_id=mapping['path_ids'][0 if near else 1],uv=[x,100.],normal=[0.,1.],
                kind=1,current_source_RGB_measured=True,model_UV_synthesized=False,
                measurement_operator='actual_source_profile')
            if near:row.update(source_view='top_face_only',normal_interval_px=[-1.,1.])
            else:row.update(search_uv=[x,100.],source_profile_flags=dict(unresolved=False),source_registration_label='1/Ad7')
            actual.append(row)
        record=dict(pts=1000,source_video_sha256='a'*64,source_BGR_sha256='b'*64,
            source_exclusion_sha256='c'*64,mask_sha256='d'*64,rows=actual,fit_eligible=False)
        detail=dict(pts=1000,all_actual_readouts=actual)
        if near:detail['components']=[dict(source_BGR_sha256='b'*64)]
        else:detail['patterns']=[dict(atlas_index=1,instance='Ad7',registration=dict(
            status='source_pattern_registered',shared_horizontal_blur_px=1))]
        qualifier.write(root/'binding.json',binding)
        qualifier.write(root/'summary.json',dict(shots=dict(s07=dict(frames=1)),changed_dependencies=[]))
        for filename,value in [('source_rows',record),('readouts'if near else'registration',detail)]:
            (root/'s07'/(filename+'.jsonl')).write_text(json.dumps(value)+'\n',encoding='utf8')
    args=SimpleNamespace(prepared=prepared,output=tmp_path/'qualified',capture_roots=roots,
        shots=['s07'],near_reviewed_shots=['s07'])
    return args,old,manifest,mapping


def test_print_retires_only_actual_one_pixel_strip_and_near_appends_without_array_mutation(source_batch):
    args,old,manifest,mapping=source_batch
    originals={path:qualifier.sha(path)for root in [args.prepared,*args.capture_roots]for path in root.rglob('*')if path.is_file()}
    qualifier.run(args)
    declaration=qualifier.read(args.output/'s07/source_replacement.json')
    groups={group['path_id']:group['parent_observation_ids']for group in declaration['groups']}
    assert groups[mapping['path_ids'][0]]==[]
    assert groups[mapping['path_ids'][1]]==['sampled-print']
    frame=json.loads((args.output/'s07/source_rows.jsonl').read_text())
    assert all(row['source_family_review']['state']=='root_reviewed_not_independent_pixel_acceptance'for row in frame['rows'])
    assert all(not row['source_family_review']['pixel_accuracy_accepted']for row in frame['rows'])
    near=[row for row in frame['rows']if row['path_id']==mapping['path_ids'][0]]
    assert all('sampled_source_profile_radius_px'not in row for row in near)
    print_rows=[row for row in frame['rows']if row['path_id']==mapping['path_ids'][1]]
    assert all(row['sampled_source_tangent_half_width_px']==1. for row in print_rows)
    paths=list(groups);packet_mapping=dict(path_ids=paths,source_replacement=declaration)
    incoming=dict(pts=np.full(len(frame['rows']),1000,np.int64),kind=np.ones(len(frame['rows']),np.int64),
        path_indices=np.array([paths.index(row['path_id'])for row in frame['rows']]))
    authenticated=authenticate_parent_observation_ids(packet_mapping,incoming,manifest,mapping,old,'s07')
    assert authenticated[1000,mapping['path_ids'][0]]==[]
    assert all(qualifier.sha(path)==digest for path,digest in originals.items())


def test_near_family_has_no_automatic_admission_without_explicit_shot_review(source_batch):
    args,_,_,mapping=source_batch;args.near_reviewed_shots=[]
    qualifier.run(args)
    frame=json.loads((args.output/'s07/source_rows.jsonl').read_text())
    assert all(row['path_id']!=mapping['path_ids'][0]for row in frame['rows'])
    pending=json.loads((args.output/'s07/pending.jsonl').read_text())
    assert pending['candidates'][0]['reason']=='near_board_physical_identity_pending'


@pytest.mark.parametrize('invalid',['video','BGR','mask','exclusion','schema','normal','uv','summary','truncated'])
def test_bad_source_identity_or_geometry_cannot_produce_completed_qualification(source_batch,invalid):
    args,_,_,_=source_batch;root=args.capture_roots[0]
    data=root/'s07/source_rows.jsonl';detail=root/'s07/registration.jsonl'
    if invalid in ('video','BGR','mask','exclusion'):
        frame=json.loads(data.read_text());key=dict(video='source_video_sha256',BGR='source_BGR_sha256',
            mask='mask_sha256',exclusion='source_exclusion_sha256')[invalid]
        frame[key]='f'*64;data.write_text(json.dumps(frame)+'\n',encoding='utf8')
    elif invalid=='schema':
        binding=qualifier.read(root/'binding.json');binding['schema']='unknown';qualifier.write(root/'binding.json',binding)
    elif invalid=='summary':
        qualifier.write(root/'summary.json',dict(shots=dict(s07=dict(frames=0))))
    elif invalid=='truncated':detail.write_text('',encoding='utf8')
    else:
        record=json.loads(detail.read_text());record['all_actual_readouts'][0][invalid]=[0.,0.]if invalid=='normal'else[1920.,100.]
        detail.write_text(json.dumps(record)+'\n',encoding='utf8')
    with pytest.raises(ValueError):qualifier.run(args)
    assert not (args.output/'qualification_summary.json').exists()
