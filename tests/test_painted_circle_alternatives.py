import numpy as np

from capture_native_centerlines import qualify_painted_paths


def test_two_coherent_circle_bands_remain_separate_unidentified_alternatives():
    angle=np.linspace(.3,2.8,73)
    guide=np.c_[400*np.cos(angle),60*np.sin(angle)]+[900.,500.]
    normal=(guide-[900.,500.])/np.array([400.**2,60.**2])
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    query=np.tile(np.arange(len(guide)),2)
    offsets=np.r_[np.full(len(guide),-9.),np.full(len(guide),9.)]
    uv=guide[query]+offsets[:,None]*normal[query]
    sides=uv[:,None]+np.array([-2.,2.])[None,:,None]*normal[query,None]
    quantiles=sides[:,:,None]+np.array([-.6,0.,.6])[None,None,:,None]*normal[query,None,None]
    measured=dict(query_curve=np.zeros(len(guide),np.int16),query_uv=guide,
                  query_normal=normal,query_interval=np.tile([-48.,48.],(len(guide),1)),
                  candidate_query=query,candidate_uv=uv,candidate_sides=sides,
                  candidate_side_quantiles_uv=quantiles,candidate_flags=np.zeros(len(uv),np.uint8))
    original=uv.copy()
    fragments,reports=qualify_painted_paths(measured,[dict(closed=True)],np.zeros((1080,1920),bool))
    selected=[i for f in fragments for i in f['candidate_ids']]
    assert len(set(selected))==len(selected)
    assert len(set(selected)&set(range(73)))>=30
    assert len(set(selected)&set(range(73,146)))>=30
    assert all(max(f['candidate_ids'])<73 or min(f['candidate_ids'])>=73 for f in fragments)
    assert all(f['physical_identity'] is None for f in fragments)
    assert all(not f['unseen_source_pixels_created'] for f in fragments)
    np.testing.assert_array_equal(measured['candidate_uv'],original)
