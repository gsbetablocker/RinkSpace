import hashlib,json
from types import SimpleNamespace
import numpy as np
from rinkspace.vision_native_references import _selected_external_rows,REGISTERED_SOURCE_IDENTITY
from rinkspace.vision_reidentify import AcceptedSourceReidentifier


def test_external_source_store_uses_current_registry_and_preserves_original_uv(tmp_path):
    arrays=tmp_path/'rows.npz';correspondence=tmp_path/'correspondence.json'
    uv=np.array([[10.125,20.5],[11.25,22.5],[40.,50.],[42.,52.]])
    np.savez(arrays,row=np.arange(4),stable_id=np.array(['a/vertex/0','a/vertex/1','b/vertex/0','b/vertex/1']),
        pts=np.full(4,256),uv=uv,exclusion_bits=np.zeros(4,int),curve_index=np.array([0,0,1,1]),
        historical_curve_index=np.full(4,999))
    correspondence.write_text('{}',encoding='utf8')
    ref=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assets=dict(sources={'video':dict(sha256='a'*64,video_path='source.mp4',time_base='1/12800')},
        templates=[dict(paths=[dict(id='Ad3.retained')])],observation_store=dict(
            arrays=ref(arrays),source_correspondence=ref(correspondence),source_id='video',time_base='1/12800',
            instance_id='print_instance/Ad3',curve_registry=[dict(template_path_id='Ad3.retained'),dict(template_path_id='Ad3.removed')]))
    rows=_selected_external_rows(assets,tmp_path/'assets.json')
    assert len(rows)==1 and rows[0]['feature_id']=='Ad3.retained'
    np.testing.assert_array_equal(rows[0]['source_native_uv'],uv[:2])
    assert rows[0]['pts']==256 and rows[0]['active_for_fit'] is False
    assert rows[0]['human_pixel_acceptance'] is False
    assert rows[0]['original_record']['source_array_indices']==[0,1]


def test_ad_reidentification_merges_native_paths_from_distinct_reference_frames():
    reader=AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    def lookup(identity):
        name=identity.rsplit('::',1)[-1]
        if name not in ('Ad2.outer','Ad2.hole'):return None
        return dict(id='print_instance/Ad2::'+name,layer='advertisement',instance_id='print_instance/Ad2',
            template_id='template/Ad2',template_path_id=name)
    reader.geometry=SimpleNamespace(lookup=lookup)
    binding=dict(source_pts=256)
    def guide(name):return dict(id=name,points_uv=[[10.,20.],[20.,30.]],segment_ids=[0,0],closed=False,
        identity_authority={'kind':REGISTERED_SOURCE_IDENTITY})
    reader.references={
        'outer':dict(guides=[guide('Ad2.outer')],reference={'path':'outer'},source_binding={'source_pts':0}),
        'hole':dict(guides=[guide('Ad2.hole'),guide('Ad2.removed')],reference={'path':'hole'},source_binding={'source_pts':128})}
    rows=[dict(physical_id='Ad2',identity_authority={'kind':'direct_user_confirmation','reference':'accepted'},
        source_pixel_to_expected_identity_verified=True,source_packet_sha256=key,inlier_count=n,
        H_query_native_from_reference_native=np.eye(3)) for key,n in [('outer',100),('hole',25)]]
    guides,identities=reader.advertisement_guides(np.zeros((64,64,3),np.uint8),binding,{'matches':rows},native_only=True)
    assert {g['id'] for g in guides}=={'print_instance/Ad2::Ad2.outer','print_instance/Ad2::Ad2.hole'}
    assert all(g['reference_UV_used_as_measurement'] is False for g in guides)


def test_continuous_reference_flow_never_crosses_actual_shot_cut():
    reader=AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    source=dict(source_sha256='a'*64,time_base='1/12800',shot_id='s01',source_pts=256)
    reader.references={'one':dict(source_binding=source)}
    result,receipt=reader.continuous_reference_ice(np.zeros((64,64,3),np.uint8),dict(source,source_pts=512,shot_id='s02'),
        {'one':{}},{},flow_engine=SimpleNamespace(),excluded=np.zeros((64,64),bool))
    assert result==[] and receipt['cross_cut_flow_used'] is False
