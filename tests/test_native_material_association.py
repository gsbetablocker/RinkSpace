import numpy as np

from rinkspace.session.native_material_association import associate_annulus_interfaces


def _inventory():
    return [dict(curve_id=f'ring:{role}',feature_id='ring',world_curve_role=role,
                 native_path_id=f'native/ring:{role}',operator='material_interface')
            for role in ('middle','outer')]


def _measurement(query_uv, query_curve, candidate_uv, candidate_query, flags=None):
    uv=np.asarray(candidate_uv,float).reshape(-1,2)
    return dict(query_uv=np.asarray(query_uv,float).reshape(-1,2),
        query_curve=np.asarray(query_curve,int),query_normal=np.tile([0.,1.],(len(query_uv),1)),
        candidate_uv=uv,candidate_query=np.asarray(candidate_query,int),
        candidate_flags=np.zeros(len(uv),np.uint8) if flags is None else np.asarray(flags,np.uint8),
        candidate_sides=uv[:,None]+np.array([[0.,-.5],[0.,.5]]))


def test_native_roles_internal_ripples_and_local_band_domain_choose_original_rgb_ids():
    curves=[np.array([[0.,0.],[90.,0.]]),np.array([[0.,20.],[90.,20.]])]
    queries=np.array([[x,y] for y in (0.,20.) for x in (0.,30.,60.,90.)])
    candidate_uv=[];candidate_query=[];expected=[]
    for qi,(x,y) in enumerate(queries):
        # Opposite boundary, internal midpoint, beyond the local band domain,
        # farther legal readout, then the closest actual boundary readout.
        delta=1. if y==0. else -1.
        candidate_uv.extend([[x,20.-y],[x,10.],[x,y-25.*delta],
                             [x,y+2.*delta],[x,y+delta]])
        candidate_query.extend([qi]*5);expected.append(len(candidate_uv)-1)
    measured=_measurement(queries,np.repeat([0,1],4),candidate_uv,candidate_query)
    originals={key:value.copy() for key,value in measured.items()}
    selected,fragments,receipt=associate_annulus_interfaces(measured,_inventory(),curves)
    assert selected==expected
    assert {cid for f in fragments for cid in f['candidate_ids']}==set(expected)
    assert receipt['rejected_native_role_or_tie']==16
    assert receipt['rejected_local_domain']==8
    assert receipt['selected_by_role']=={'middle':4,'outer':4}
    for key,value in measured.items():np.testing.assert_array_equal(value,originals[key])


def test_partial_arc_missing_outer_readouts_and_source_masks_do_not_create_points():
    theta=np.linspace(0.,2*np.pi,257)
    curves=[radius*np.c_[np.cos(theta),np.sin(theta)] for radius in (120.,150.)]
    angle=np.linspace(.15,1.4,12)
    middle=120.*np.c_[np.cos(angle),np.sin(angle)]
    outer=150.*np.c_[np.cos(angle),np.sin(angle)]
    query=np.r_[middle,outer]
    measured=_measurement(query,np.repeat([0,1],12),middle,np.arange(12))
    measured['query_normal']=query/np.linalg.norm(query,axis=1)[:,None]
    measured['candidate_flags'][3]=1
    measured['candidate_flags'][5]=2
    measured['candidate_flags'][7]=8  # Unrelated flag bits do not ban a clean source reading.
    eligible=np.ones(12,bool);eligible[9]=False
    original=measured['candidate_uv'].copy()
    selected,fragments,receipt=associate_annulus_interfaces(measured,_inventory(),curves,eligible)
    assert selected==[0,1,2,4,6,7,8,10,11]
    assert fragments and all(fragment['curve']==0 for fragment in fragments)
    assert receipt['clean_and_user_eligible']==9
    assert receipt['selected_by_role']=={'middle':9,'outer':0}
    assert not receipt['physical_identity_confirmed']
    np.testing.assert_array_equal(measured['candidate_uv'],original)
    masked,masked_fragments,masked_receipt=associate_annulus_interfaces(
        measured,_inventory(),curves,np.zeros(12,bool))
    assert masked==masked_fragments==[]
    assert masked_receipt['clean_and_user_eligible']==masked_receipt['selected']==0


def test_real_rgb_annulus_interfaces_beat_internal_color_texture_and_opposite_ring():
    import cv2
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
    from rinkspace.session.native_shape_guidance import point_to_polyline
    center=np.array([400.,240.]);angle=np.linspace(0.,2*np.pi,257)
    axes=[(220,60),(300,82)]
    curves=[center+np.c_[a*np.cos(angle),b*np.sin(angle)] for a,b in axes]
    rgb=np.full((480,800,3),210,np.uint8)
    cv2.ellipse(rgb,tuple(center.astype(int)),axes[1],0,0,360,(90,90,90),-1)
    cv2.ellipse(rgb,tuple(center.astype(int)),(270,74),0,0,360,(140,140,140),-1)
    cv2.ellipse(rgb,tuple(center.astype(int)),(250,68),0,0,360,(90,90,90),-1)
    cv2.ellipse(rgb,tuple(center.astype(int)),axes[0],0,0,360,(210,210,210),-1)
    rgb=cv2.GaussianBlur(rgb,(7,7),1.5)
    inventory=_inventory();projections={}
    for path,curve in zip(inventory,curves):
        path.update(xyz=np.c_[curve,np.zeros(len(curve))],center_xyz=np.r_[center,0.],
            material_signal='dark',material_transition='light_to_dark' if path['world_curve_role']=='middle' else 'dark_to_light',
            profile_direction='outward_normal')
        projections[path['native_path_id']]=dict(uv=curve,valid=np.ones(len(curve),bool),centre_uv=center)
    reader=IceProgrammeSourceReader(None,source_sha256='a'*64,time_base='1/12800',
        native_paths=inventory,sample_spacing_px=24.,search_radius_px=80.)
    measured=reader.read(rgb,256,None,np.zeros(rgb.shape[:2],bool),
        shot_id='synthetic',native_projections=projections)
    original=measured['candidate_uv'].copy()
    selected,fragments,receipt=associate_annulus_interfaces(measured,inventory,curves)
    expected=measured['query_curve'][measured['candidate_query']]
    distances=np.column_stack([point_to_polyline(original,curve).distance_px for curve in curves])
    own=distances[np.arange(len(original)),expected]
    assert np.any(own>10.)  # Actual source-reader alternatives include internal texture/other ring.
    assert selected and fragments and receipt['selected_by_role']['middle']>=12
    assert receipt['selected_by_role']['outer']>=12
    assert np.percentile(own[selected],95)<2.
    assert all(expected[cid]==int(distances[cid].argmin()) for cid in selected)
    np.testing.assert_array_equal(measured['candidate_uv'],original)


def test_empty_source_candidates_return_no_synthetic_or_projected_observations():
    curves=[np.array([[0.,0.],[100.,0.]]),np.array([[0.,20.],[100.,20.]])]
    selected,fragments,receipt=associate_annulus_interfaces(
        {'candidate_uv':np.empty((0,2))},_inventory(),curves)
    assert selected==fragments==[]
    assert receipt['status']=='no_source_candidates' and receipt['selected']==0


def test_common_translation_selects_real_interface_instead_of_texture_near_old_query():
    import cv2
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
    from rinkspace.session.native_shape_guidance import point_to_polyline
    rgb=np.full((200,320,3),210,np.uint8)
    rgb[88:129]=90
    rgb[100:105]=150  # An internal material ripple near the old middle query.
    rgb=cv2.GaussianBlur(rgb,(7,7),1.2)
    old_curves=[np.array([[20.,y],[300.,y]]) for y in (100.,140.)]
    aligned_curves=[curve+[0.,-12.] for curve in old_curves]
    inventory=_inventory();projections={}
    for path,curve in zip(inventory,old_curves):
        path.update(xyz=np.c_[curve,np.zeros(len(curve))],center_xyz=np.array([160.,0.,0.]),
            material_signal='dark',material_transition='light_to_dark' if path['world_curve_role']=='middle' else 'dark_to_light',
            profile_direction='outward_normal')
        projections[path['native_path_id']]=dict(uv=curve,valid=np.ones(len(curve),bool),centre_uv=np.array([160.,0.]))
    reader=IceProgrammeSourceReader(None,source_sha256='a'*64,time_base='1/12800',
        native_paths=inventory,sample_spacing_px=24.,search_radius_px=48.)
    measured=reader.read(rgb,256,None,np.zeros(rgb.shape[:2],bool),
        shot_id='translated_source',native_projections=projections)
    original_uv=measured['candidate_uv'].copy();original_queries=measured['query_uv'].copy()
    query=measured['candidate_query'];role=measured['query_curve'][query]
    own=point_to_polyline(original_uv,aligned_curves[0]).distance_px
    near_old=np.linalg.norm(original_uv-original_queries[query],axis=1)<6.
    near_aligned=own<2.
    competing_queries=set(query[(role==0)&near_old])&set(query[(role==0)&near_aligned])
    assert len(competing_queries)>=4
    selected,fragments,receipt=associate_annulus_interfaces(measured,inventory,aligned_curves)
    middle=np.array([cid for cid in selected if role[cid]==0],int)
    assert len(middle)>=4 and fragments and receipt['selected_by_role']['outer']>=4
    assert own[middle].max()<2.
    assert all(query[cid] in competing_queries for cid in middle)
    np.testing.assert_array_equal(measured['candidate_uv'],original_uv)
    np.testing.assert_array_equal(measured['query_uv'],original_queries)
