from copy import deepcopy
import numpy as np

from rinkspace.vision_material_reader import retain_material_identity, _tracked_material_identity
from rinkspace.vision_navigation import _uncovered_query


def _source():
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s02')
    scene=dict(path='native.json',sha256='c'*64)
    proof=dict(kind='current_RGB_material_graph_identity',source_binding=binding,
        native_path_id='native/line',source_seed_authority=scene,current_RGB_measured=True,
        identity_from_projection_alone=False,decoded_native_RGB_sha256='d'*64)
    guide=dict(id='native/line',points_uv=np.array([[x,10.] for x in range(0,100,10)]),
        segment_ids=np.zeros(10,int),source_binding=binding,native_geometry=dict(scene=scene,path_id='native/line'),
        identity_confirmed=False,identity_authority=None)
    row=dict(native_path_id='native/line',query_id='original-query',source_indices=[1,6,7,8],
        actual_source_uv=[[10.,10.],[60.,10.],[70.,10.],[80.,10.]],support=dict(actual_readout_evidence=dict(
            identity_authorities={'proof':proof},identity_by_source_index={str(i):dict(
                authority_key='proof',actual_source_uv=[i*10.,10.]) for i in (6,7,8)})))
    guide['material_identity_continuation']=retain_material_identity(guide,row,binding,'d'*64)
    return binding,guide,row


def test_only_exact_measured_identity_members_continue_through_valid_flow():
    binding,guide,row=_source();saved=deepcopy(row)
    assert len(guide['material_identity_continuation']['authorities'])==1
    continued=dict(guide,source_binding=dict(binding,source_pts=512),origin='temporal_source_search_hint',
        hint_previous_source_binding=binding,temporal_flow_supported=np.array([True]*7+[False,True,True]))
    indices,authority=_tracked_material_identity(continued,continued['source_binding'])
    assert indices.tolist()==[6,8]
    assert guide['identity_confirmed'] is False and guide['identity_authority'] is None
    assert authority['human_pixel_acceptance'] is False and authority['current_RGB_reread_required'] is True
    assert authority['original_readout_proof_references']['0']['original_query_indices']==[6,8]
    assert row==saved
    assert _tracked_material_identity(continued,dict(continued['source_binding'],shot_id='s03'))[0].size==0
    assert _tracked_material_identity(dict(continued,hint_previous_source_binding=dict(binding,source_pts=0)),
        continued['source_binding'])[0].size==0


def test_query_gap_slicing_remaps_continuation_members_without_spreading_identity():
    binding,guide,_=_source()
    incoming=dict(guide,origin='temporal_source_search_hint',source_binding=dict(binding,source_pts=512),
        hint_previous_source_binding=binding,temporal_flow_supported=np.ones(10,bool))
    prior=dict(id='native/line',points_uv=[[0.,10.],[50.,10.]],segment_ids=[0,0])
    clipped=_uncovered_query(incoming,[prior])
    indices,authority=_tracked_material_identity(clipped,clipped['source_binding'])
    np.testing.assert_array_equal(clipped['points_uv'][indices],[[60.,10.],[70.,10.],[80.,10.]])
    assert authority['original_readout_proof_references']['0']['original_query_indices']==[6,7,8]
    assert guide['material_identity_continuation']['members'].keys()=={'6','7','8'}


def test_mismatched_readout_pixel_cannot_become_a_temporal_identity_anchor():
    binding,guide,row=_source();row['actual_source_uv'][1][1]+=1
    support=retain_material_identity(guide,row,binding,'d'*64)
    assert support['members'].keys()=={'7','8'}


def test_scoped_withdrawal_does_not_revoke_independently_read_points_on_the_same_path():
    from rinkspace.session.source_observations import SourceCandidate
    from rinkspace.session.source_programme import qualify_source_identities
    def point(name,identity='confirmed',**extra):
        return SourceCandidate(name,'s02',256,{},[10.,12.],'line',actual_uv=[10.,12.],
            native_path_id='native/line',identity=identity,localization='measured',source_group_id='original-group',**extra)
    old=point('withdrawn',withdrawal=dict(reason='wrong point',scope='exact original point'))
    independent=point('current',evidence=dict(identity_authority=dict(
        kind='current_RGB_material_graph_identity',source_records=[dict(id='independent-source')])))
    dependent=point('dependent',evidence=dict(identity_authority=dict(
        source_seed_authority=dict(source_records=[dict(id='withdrawn')]))))
    before=deepcopy(old)
    qualify_source_identities([old],[independent,dependent],[])
    assert independent.identity=='confirmed' and independent.actual_uv==[10.,12.]
    assert dependent.identity=='unresolved' and dependent.actual_uv==[10.,12.]
    assert dependent.evidence['withdrawn_identity_support_ids']==['withdrawn']
    assert old==before


def test_disjoint_source_queries_on_one_native_path_keep_each_exact_identity_scope():
    from rinkspace.session.source_observations import SourceCandidate
    from rinkspace.session.source_programme import qualify_source_identities
    rows=[SourceCandidate(str(i),'s02',256,{},[float(i),12.],'line',actual_uv=[float(i),12.],
        native_path_id='native/line',identity='named_source_candidate',localization='measured') for i in range(3)]
    queries=[dict(id='native/line',identity_confirmed=True,identity_authority=dict(source_records=[dict(id=str(i))])) for i in range(2)]
    qualify_source_identities(rows,[],queries)
    assert [r.identity for r in rows]==['confirmed','confirmed','named_source_candidate']
    assert rows[0].evidence['identity_authority']['source_records']==[dict(id='0')]
    assert [r.actual_uv for r in rows]==[[0.,12.],[1.,12.],[2.,12.]]
