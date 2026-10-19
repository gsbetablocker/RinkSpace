"""Source-material errors must remain observable while physical roles separate."""
import unittest
import numpy as np
import torch
from rinkspace.vision_source_topology import (_ordered_foot_material_pairs, read_board_foot_material_profiles,
    _ordered_board_foot_profile_batches,_native_material_query_strip,source_graphic_rectangle_candidates,
    source_foot_material_role_families,source_graphic_material_support_conflicts,
    source_nested_material_band_conflicts,source_material_transition_support,
    source_local_material_direction_evidence,source_board_material_search_navigation,
    source_board_top_foot_role_pairs,read_source_board_rail_material_profiles,observed_signed_material_transition)


class SignedMaterialIntervalTests(unittest.TestCase):
    def test_native_material_staircase_preserves_zero_gradients_inside_one_observed_transition(self):
        values=np.array([0.,.01,.01,.025,.025,.04,.04,.03],np.float32)
        signed=np.diff(values);before=signed.copy()
        interval=observed_signed_material_transition(values,signed,np.ones(len(signed),bool),2,1)
        np.testing.assert_array_equal(interval['source_indices'],np.arange(5))
        np.testing.assert_array_equal(interval['zero_gradient_source_indices'],[1,3])
        np.testing.assert_array_equal(interval['original_signed_gradient_samples'],signed[:5])
        np.testing.assert_array_equal(signed,before)
        self.assertFalse(interval['original_gradient_samples_interpolated'])
        self.assertTrue(interval['interval_is_observed_optical_support_not_exact_physical_boundary'])

    def test_actual_colour_reversal_separates_competing_steps_without_merging_or_deleting_peaks(self):
        values=np.array([0.,.02,.02,.015,.035,.035,.055],np.float32)
        signed=np.diff(values);before=signed.copy()
        interval=observed_signed_material_transition(values,signed,np.ones(len(signed),bool),0,1)
        np.testing.assert_array_equal(interval['source_indices'],[0])
        self.assertGreater(signed[3],0)
        np.testing.assert_array_equal(signed,before)


class BoardRailMaterialRoleTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.rgb=np.full((140,200,3),160,np.uint8)
        self.rgb[42:108]=225
        self.rgb[30:42]=[45,65,130]
        self.rgb[70:95,40:160]=[60,100,180]
        self.binding=dict(source_sha256='a'*64,source_pts=8,time_base='1/15360',native_hw=[140,200])
        x=np.arange(200)
        top=np.concatenate((np.column_stack((x,np.full(200,30))),np.column_stack((x,np.full(200,42)))))
        self.packet=dict(source_binding=self.binding,RGB_sha256=hashlib.sha256(self.rgb.tobytes()).hexdigest(),
            components=[dict(typed_class_candidate='board_top',candidate_kind='straight_edge_fragment',source_support_uv=top)])
        self.foot=dict(source_binding=self.binding,readings=[dict(primary_source_uv=
            np.column_stack((np.arange(0,200,4),np.full(50,116))).astype(np.float32))])
        p=np.concatenate((np.column_stack((np.arange(40,160),np.full(120,70))),
            np.column_stack((np.arange(40,160),np.full(120,94)))))
        self.ads=[dict(kind='advertisement_panel',object_id=2,source_binding=self.binding,
            boundary_readings=[dict(boundary_role='printed_pattern_external_contour',actual_source_uv=p)])]

    def test_blue_band_entry_is_upper_role_and_white_exit_is_internal_with_raw_pool_retained(self):
        paired=source_board_top_foot_role_pairs(self.rgb,self.packet,self.foot,advertisement_instances=self.ads)
        self.assertIsNotNone(paired['selected_hypothesis'])
        material=read_source_board_rail_material_profiles(self.rgb,paired)
        entry=material['actual_source_blue_rail_entry_uv'];exit=material['actual_source_blue_to_white_exit_uv']
        self.assertGreater(len(entry),30)
        self.assertTrue(((entry[:,1]>=29)&(entry[:,1]<=30)).all())
        self.assertTrue(((exit[:,1]>=41)&(exit[:,1]<=42)).all())
        self.assertEqual(material['source_material_role_evidence'][0]['exit_role_hypothesis'],
            'board_internal_blue_to_white_material_interface')
        self.assertFalse(material['fitted_navigation_UV_used_as_measurement'])
        self.assertTrue(material['all_observed_signed_optical_peak_pool'])
        self.assertFalse(material['source_pixel_accuracy_accepted'])

    def test_top_only_fragments_and_global_colored_regions_cannot_substitute_contact_or_print(self):
        missing=dict(self.foot,readings=[dict(primary_source_uv=np.empty((0,2),np.float32))])
        no_foot=source_board_top_foot_role_pairs(self.rgb,self.packet,missing,advertisement_instances=self.ads)
        no_print=source_board_top_foot_role_pairs(self.rgb,self.packet,self.foot,advertisement_instances=())
        self.assertIsNone(no_foot['selected_hypothesis'])
        self.assertIsNone(no_print['selected_hypothesis'])
        material=read_source_board_rail_material_profiles(self.rgb,no_print)
        self.assertEqual(material['actual_source_blue_rail_entry_uv'].shape,(0,2))
        self.assertFalse(no_print['top_fragment_allowed_as_contact'])

    def test_current_person_prediction_competes_for_role_but_retains_original_RGB_edges(self):
        paired=source_board_top_foot_role_pairs(self.rgb,self.packet,self.foot,advertisement_instances=self.ads)
        first=read_source_board_rail_material_profiles(self.rgb,paired)
        mask=np.zeros((140,200),bool);mask[27:44,40:100]=True
        conflict=read_source_board_rail_material_profiles(self.rgb,paired,source_instances=[
            dict(kind='occluder',object_id=1000,source_binding=self.binding,region_mask=mask)])
        np.testing.assert_array_equal(first['actual_source_blue_rail_entry_uv'],conflict['actual_source_blue_rail_entry_uv'])
        np.testing.assert_array_equal(first['actual_source_blue_to_white_exit_uv'],conflict['actual_source_blue_to_white_exit_uv'])
        self.assertTrue(any(e['competing_current_source_person_support']for e in conflict['source_material_role_evidence']))
        self.assertTrue(conflict['current_source_person_regions_are_competing_roles_not_raw_RGB_veto'])


class BoardMaterialSearchNavigationTests(unittest.TestCase):
    def setUp(self):
        self.binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/15360',native_hw=[60,100])
        self.top=np.array([[10.,15.25],[14.,15.75],[18.,16.25],[70.,17.]],np.float32)
        self.foot=np.array([[10.,45.5],[14.,46.],[18.,46.5],[70.,48.]],np.float32)
        self.upper=dict(source_binding=self.binding,RGB_sha256='b'*64,
            actual_source_blue_rail_entry_uv=self.top)
        self.lower=dict(source_binding=self.binding,RGB_sha256='b'*64,
            readings=[dict(primary_source_uv=self.foot)])

    def test_wide_mask_never_invents_actual_UV_or_bridges_long_missing_gap(self):
        before_top,before_foot=self.top.copy(),self.foot.copy()
        result=source_board_material_search_navigation(self.upper,self.lower,native_hw=(60,100),
            neighbor_radius_px=4,boundary_uncertainty_margin_px=3)
        self.assertTrue(result['search_mask'][25,14])
        self.assertFalse(result['search_mask'][:,40].any())
        self.assertIn(40,result['missing_source_support_columns'])
        self.assertFalse(result['RGB_hard_veto_allowed'])
        self.assertFalse(result['navigation_region_independently_verifies_board_role'])
        self.assertEqual(result['top_actual_source_uv'].dtype,np.float32)
        np.testing.assert_array_equal(result['top_actual_source_uv'],before_top)
        np.testing.assert_array_equal(result['foot_actual_source_uv'],before_foot)
        np.testing.assert_array_equal(self.top,before_top)
        np.testing.assert_array_equal(self.foot,before_foot)

    def test_unresolved_material_or_person_evidence_keeps_search_with_explicit_conflict(self):
        self.upper['source_material_role_evidence']=[dict(actual_source_material_entry_uv=self.top[1],
            source_role_status='source_role_conflict_or_multiple_optical_transitions')]
        result=source_board_material_search_navigation(self.upper,self.lower,native_hw=(60,100))
        self.assertIn(14,result['uncertain_search_columns'])
        self.assertTrue(result['search_mask'][30,14])
        self.assertFalse(result['pixels_are_RGB_measurements'])

    def test_wrong_PTS_clock_RGB_and_native_domain_cannot_supply_board_search(self):
        for key,value in (('source_pts',8),('time_base','1/12800'),('source_sha256','c'*64)):
            wrong=dict(self.lower,source_binding=dict(self.binding,**{key:value}))
            with self.assertRaisesRegex(ValueError,'mix source'):
                source_board_material_search_navigation(self.upper,wrong,native_hw=(60,100))
        with self.assertRaisesRegex(ValueError,'different original RGB'):
            source_board_material_search_navigation(self.upper,dict(self.lower,RGB_sha256='c'*64),native_hw=(60,100))
        with self.assertRaisesRegex(ValueError,'different native'):
            source_board_material_search_navigation(self.upper,self.lower,native_hw=(61,100))


class BoardFootMaterialEvidenceTests(unittest.TestCase):
    def test_all_source_contact_components_and_same_column_competitors_are_queried(self):
        height,width=90,120
        classes=['background','ice_contact','board_top','ice_paint','ad_boundary','shadow_or_reflection']
        probabilities=torch.zeros((1,6,height,width));probabilities[:,0]=1.
        for ys in (slice(15,18),slice(40,44)):
            probabilities[:,0,ys]=.01;probabilities[:,1,ys]=.99
        probabilities[:,0,4,7]=.01;probabilities[:,1,4,7]=.99
        rgb=np.full((height,width,3),180,np.uint8)
        points,normals,heights,evidence=_native_material_query_strip(rgb,
            dict(class_probabilities=probabilities),classes,dict(selected_pair=None),4.,'cpu')
        self.assertTrue(evidence['all_predicted_components_queried'])
        self.assertEqual(set(evidence['query_component_labels']),{1,2,3})
        self.assertTrue(any((points==[7,4]).all(1)), 'An unsampled one-pixel component must remain queryable')
        self.assertGreaterEqual(sum(points[:,0]==6),2,'A false upper candidate must not suppress the lower one')
        self.assertFalse(evidence['SAM_envelope_used_to_decide_visibility'])
        self.assertFalse(evidence['all_native_columns_queried'])
        self.assertEqual(evidence['native_probability_column_domain_scanned'],[0,width])
        self.assertEqual(evidence['profile_sample_spacing_native_px'],4.)

    def test_graphic_outside_exit_still_conflicts_with_selected_material_band(self):
        y=torch.arange(140,dtype=torch.float32)
        coordinates=torch.stack((torch.full_like(y,75),y),-1)[None].repeat(2,1,1)
        original=coordinates.clone()
        profiles=dict(sampled_native_uv=coordinates,valid_mask=torch.ones((2,140),dtype=torch.bool))
        pairs=dict(entry_index=torch.tensor([30,90]),exit_index=torch.tensor([87,120]),
            missing_same_band=torch.zeros(2,dtype=torch.bool))
        graphics=dict(candidates=[dict(box_xyxy=[50,40,120,84])])
        evidence=source_graphic_material_support_conflicts(profiles,pairs,graphics)
        self.assertTrue(evidence['selected_material_support_overlaps_source_graphic'][0],
            'Exit below the rectangle cannot make a graphic-contaminated band into board material')
        self.assertFalse(evidence['selected_material_support_overlaps_source_graphic'][1],
            'A graphic elsewhere in the search window cannot veto the actual selected material')
        self.assertEqual(int(evidence['actual_sampled_material_support_inside_source_graphic'][0]),44)
        torch.testing.assert_close(coordinates,original,rtol=0,atol=0)

    def test_disconnected_warm_foreground_cannot_inherit_main_foot_gap(self):
        left=np.arange(2,43,4.);right=np.arange(82,119,4.)
        main=np.column_stack((np.r_[left,right],40+np.r_[left,right]*.08))
        branch=np.column_stack((np.arange(54,67,4.),np.arange(54,67,4.)*.4+60))
        corner=np.column_stack((np.arange(124,137,4.),np.arange(124,137,4.)*.7-31.8))
        uv=np.concatenate((main,branch,corner));original=uv.copy()
        evidence=dict(selected_inside_warm_gamma=np.full(len(uv),.07),
            selected_source_model_contact_type=np.full(len(uv),.999),
            selected_band_width_px=np.full(len(uv),8.))
        result=source_foot_material_role_families(uv,np.ones(len(uv),bool),evidence,native_hw=(140,200))
        self.assertTrue(result['primary_role_samples'][:len(main)].all())
        self.assertFalse(result['primary_role_samples'][len(main):len(main)+len(branch)].any(),
            'Four high-confidence warm points do not establish the board foot through an unrelated source gap')
        self.assertTrue(result['primary_role_samples'][-len(corner):].all(),
            'Short source material around a corner cannot be removed by a full-width/minspan condition')
        branch_family=next(f for f in result['families'] if len(main) in f['original_RGB_query_indices'])
        self.assertFalse(branch_family['navigation_links'],'A foreground family must not inherit the board-foot links')
        self.assertTrue(result['unresolved_optical_samples'][len(main):len(main)+len(branch)].all())
        np.testing.assert_array_equal(uv,original)

    def test_warm_foot_family_does_not_promote_white_cap_with_identical_head_confidence(self):
        x=np.arange(2,118,4.)
        cap=np.column_stack((x,15+x*.05));foot=np.column_stack((x,40+x*.2))
        uv=np.concatenate((cap,foot));preserved=uv.copy()
        evidence=dict(selected_inside_warm_gamma=np.r_[np.full(len(x),.016),np.full(len(x),.08)],
            selected_source_model_contact_type=np.ones(len(uv)))
        assigned=source_foot_material_role_families(uv,np.ones(len(uv),bool),evidence,native_hw=(90,120))
        self.assertFalse(assigned['primary_role_samples'][:len(x)].any())
        self.assertTrue(assigned['primary_role_samples'][len(x):].all())
        self.assertTrue(assigned['unresolved_optical_samples'][:len(x)].all())
        np.testing.assert_array_equal(uv,preserved)

    def test_graphic_rectangle_is_read_from_RGB_at_any_source_location(self):
        import cv2
        boxes=[]
        for offset in (0,40):
            rgb=np.full((160,300,3),150,np.uint8)
            cv2.rectangle(rgb,(50+offset,40),(200+offset,84),(210,210,200),-1)
            cv2.putText(rgb,'1:54',(70+offset,73),cv2.FONT_HERSHEY_SIMPLEX,.8,(60,60,55),2)
            result=source_graphic_rectangle_candidates(rgb,dict(native_hw=[160,300],source_pts=123))
            self.assertEqual(len(result['candidates']),1)
            boxes.append(result['candidates'][0]['box_xyxy'])
            self.assertFalse(result['candidates'][0]['PTS_or_human_box_coordinates_used'])
        np.testing.assert_array_equal(np.array(boxes[1])-boxes[0],[40,0,40,0])

    def test_warm_ice_reflection_tail_does_not_replace_supported_native_RGB_transition(self):
        axis=torch.arange(193,dtype=torch.float32)*.5
        gamma=torch.zeros_like(axis)
        gamma[(axis>=20)&(axis<40)]=.08
        gamma[(axis>=40)&(axis<60)]=.045
        gamma[(axis>=60)&(axis<70)]=.045*(70-axis[(axis>=60)&(axis<70)])/10
        light=torch.full_like(axis,.65)
        light[(axis>=20)&(axis<40)]=.62
        light[(axis>=40)&(axis<43)]=.55+.02*(axis[(axis>=40)&(axis<43)]-40)
        rgb=torch.stack((light+gamma/3,light+gamma/3,light-2*gamma/3),-1)[None]
        coordinates=torch.stack((torch.full_like(axis,4),axis),-1)[None]
        profiles=dict(rgb_profile=rgb,valid_mask=torch.ones((1,len(axis)),dtype=torch.bool),
            rgb_gradient_valid_mask=torch.ones((1,len(axis)-1),dtype=torch.bool),
            sampled_native_uv=coordinates,metadata=dict(sample_step_native_px=.5))
        background=(axis<40).float()[None]
        contact=((axis>=40)&(axis<55)).float()[None]
        semantic=dict(background=background,ice_contact=contact,ice_paint=(axis>=55).float()[None])
        ice=torch.ones((100,10),dtype=torch.bool)
        result=_ordered_foot_material_pairs(profiles,ice,[300],semantic_search_profiles=semantic)
        self.assertTrue(result['usable'][0])
        self.assertLessEqual(abs(float(result['exit_index'][0])*.5-40),2)
        self.assertTrue((result['all_ordered_pair_indices_profile_entry_exit'][:,2]*.5>60).any(),
            'The wrong diffuse tail must remain in the full optical candidate ledger')
        self.assertTrue(result['source_model_probabilities_used_only_to_rank_actual_RGB_optical_candidates'])
        repeated={key:value.expand((5,)+value.shape[1:]).clone() if isinstance(value,torch.Tensor) else value
                  for key,value in profiles.items()}
        semantics={key:value.expand(5,-1).clone() for key,value in semantic.items()}
        full=_ordered_foot_material_pairs(repeated,ice,[300]*5,semantics)
        batched=_ordered_board_foot_profile_batches(repeated,ice,[300]*5,semantics,batch_size=2)
        for key,value in full.items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,batched[key],rtol=0,atol=0,equal_nan=True,
                    msg='Batching must retain every point, flag and candidate pair index')

    def test_profile_window_start_is_not_a_measured_material_entry(self):
        axis=torch.arange(193,dtype=torch.float32)*.5
        gamma=torch.where(axis<30,.08,0.)
        rgb=torch.stack((.6+gamma/3,.6+gamma/3,.6-2*gamma/3),-1)[None]
        profiles=dict(rgb_profile=rgb,valid_mask=torch.ones((1,193),dtype=torch.bool),
            rgb_gradient_valid_mask=torch.ones((1,192),dtype=torch.bool),
            sampled_native_uv=torch.stack((torch.full_like(axis,4),axis),-1)[None],
            metadata=dict(sample_step_native_px=.5))
        result=_ordered_foot_material_pairs(profiles,torch.ones((100,10),dtype=torch.bool),[100])
        self.assertTrue(result['missing_same_band'][0])
        self.assertEqual(len(result['all_ordered_pair_indices_profile_entry_exit']),0)

    def test_source_RGB_exit_inside_contact_plateau_remains_observed(self):
        axis=torch.arange(193,dtype=torch.float32)*.5
        gamma=torch.where((axis>=20)&(axis<40),.08,0.)
        rgb=torch.stack((.65+gamma/3,.65+gamma/3,.65-2*gamma/3),-1)[None]
        profiles=dict(rgb_profile=rgb,valid_mask=torch.ones((1,193),dtype=torch.bool),
            rgb_gradient_valid_mask=torch.ones((1,192),dtype=torch.bool),
            sampled_native_uv=torch.stack((torch.full_like(axis,4),axis),-1)[None],
            metadata=dict(sample_step_native_px=.5))
        semantic=dict(background=(axis<10).float()[None],
            ice_contact=(axis>=10).float()[None],ice_paint=torch.zeros((1,193)))
        result=_ordered_foot_material_pairs(profiles,torch.ones((100,10),dtype=torch.bool),[300],semantic)
        self.assertTrue(result['weak_source_model_contact_type'][0],
            'The class band need not rise again at the original RGB material exit')
        self.assertFalse(result['missing_same_band'][0])
        self.assertTrue(result['usable'][0],
            'A class plateau cannot erase a uniquely observed original RGB interface')
        self.assertLessEqual(abs(float(result['exit_index'][0])*.5-40),1)

    def test_confirmed_warm_material_exit_cannot_disappear_due_to_absolute_chroma_cap(self):
        axis=torch.arange(193,dtype=torch.float32)*.5
        gamma=torch.where((axis>=20)&(axis<40),.22,0.)
        rgb=torch.stack((.65+gamma/3,.65+gamma/3,.65-2*gamma/3),-1)[None]
        profiles=dict(rgb_profile=rgb,valid_mask=torch.ones((1,193),dtype=torch.bool),
            rgb_gradient_valid_mask=torch.ones((1,192),dtype=torch.bool),
            sampled_native_uv=torch.stack((torch.full_like(axis,4),axis),-1)[None],
            metadata=dict(sample_step_native_px=.5))
        result=_ordered_foot_material_pairs(profiles,torch.ones((100,10),dtype=torch.bool),[300])
        self.assertTrue(result['usable'][0])
        self.assertLessEqual(abs(float(result['exit_index'][0])*.5-40),1)
        self.assertLess(float(result['selected_inside_pale_fraction'][0]),.7)
        self.assertGreaterEqual(float(result['selected_inside_source_warm_fraction'][0]),.7)
        self.assertFalse(result['maximum_absolute_yellow_chroma_used_as_RGB_candidate_veto'])

    def test_distinct_source_query_windows_cannot_promote_nested_internal_band_as_foot(self):
        uv=np.array([[50.,86.],[50.,102.],[70.,85.],[70.,100.],[90.,120.]])
        widths=np.array([4.,30.,20.,20.,5.])
        preserved=uv.copy()
        result=source_nested_material_band_conflicts(uv,np.ones(5,bool),
            dict(selected_band_width_px=widths),np.tile([0.,1.],(5,1)))
        np.testing.assert_array_equal(result['internal_source_material_band_role_conflict'],[True,False,False,False,False])
        np.testing.assert_array_equal(uv,preserved)
        self.assertFalse(result['nested_interval_establishes_precise_physical_boundary'])
        self.assertEqual(result['source_observed_nested_intervals'][0]['enclosing_original_query_index'],1)

    def test_blurred_material_fall_preserves_its_support_and_parallel_peaks(self):
        gamma=np.array([[0.,.1,.2,.18,.15,.15,.09,.1,.01,.02]],np.float32)
        uv=np.stack((np.zeros(10),np.arange(10)),axis=-1)[None].astype(np.float32)
        source=dict(source_sha256='a'*64,source_pts=5,time_base='1/60')
        profiles=dict(valid_mask=np.ones((1,10),bool),sampled_native_uv=uv,source_binding=source)
        pairs=dict(gamma=gamma,entry_index=np.array([1]),exit_index=np.array([6]),missing_same_band=np.array([False]))
        before=gamma.copy()
        result=source_material_transition_support(profiles,pairs)
        np.testing.assert_array_equal(result['source_profile_sample_index_bounds'],[[2,8]])
        np.testing.assert_array_equal(result['observed_native_warm_colour_fall_support_endpoints_uv'],[[[0,2],[0,8]]])
        self.assertGreater(len(result['actual_parallel_falling_colour_peaks_profile_gradient_index']),1)
        self.assertFalse(result['pixel_precision_or_physical_contact_interval_accepted'])
        np.testing.assert_array_equal(gamma,before)

    def test_adjacent_material_intervals_survive_lighting_change_but_upper_wall_does_not(self):
        left=np.column_stack((np.arange(10.,26.,4.),35.2+np.arange(4)*1.6))
        right=np.column_stack((np.arange(38.,54.,4.),41.-np.arange(4)*1.6))
        wall=np.column_stack((np.arange(27.,39.,4.),np.full(3,12.)))
        uv=np.r_[left,right,wall];preserved=uv.copy()
        evidence=dict(selected_inside_warm_gamma=np.r_[np.full(4,.15),np.full(4,.05),np.full(3,.02)],
            selected_band_width_px=np.full(11,8.),selected_source_model_contact_type=np.ones(11))
        intervals=np.stack((uv-np.array([0.,5.]),uv+np.array([0.,3.])),axis=1)
        result=source_foot_material_role_families(uv,np.ones(11,bool),evidence,native_hw=(90,100),
            source_transition_support=dict(observed_support=np.ones(11,bool),
                observed_native_warm_colour_fall_support_endpoints_uv=intervals))
        self.assertTrue(result['primary_role_samples'][:8].all(),
            'Actual neighbouring paired source intervals can establish continuation under changed lighting')
        self.assertFalse(result['primary_role_samples'][8:].any(),
            'The nearby upper curve has no matching native material intervals')
        np.testing.assert_array_equal(uv,preserved)
        self.assertTrue(any(row['accepted_source_material_curve_continuation'] for row in
            result['source_material_curve_continuation_evidence']))

    def test_source_material_authority_cannot_be_applied_to_another_video(self):
        binding=dict(source_sha256='a'*64,source_pts=1,time_base='1/100',native_hw=[4,4],
                     shot_id='s',camera_id='c')
        authority=dict(kind='direct_user_drawing',reference='actual_user_contract.json',
                       source_sha256='b'*64,board_foot_material='warm_yellow_strip_belongs_to_board')
        with self.assertRaisesRegex(ValueError,'source-bound'):
            read_board_foot_material_profiles(np.zeros((4,4,3),np.uint8),
                dict(source_binding=binding,instances=[]),dict(source_binding=binding),
                material_role_authority=authority,device='cpu')

    def test_original_RGB_yellow_band_is_not_lost_to_absolute_red_green_limit(self):
        rgb=torch.full((1,80,3),.75)
        rgb[:,20:40]=torch.tensor([.85,.75,.56])
        y=torch.arange(80,dtype=torch.float32)*.5
        profiles=dict(rgb_profile=rgb,valid_mask=torch.ones((1,80),dtype=torch.bool),
            rgb_gradient_valid_mask=torch.ones((1,79),dtype=torch.bool),
            sampled_native_uv=torch.stack((torch.full_like(y,50),y),1)[None],
            metadata=dict(sample_step_native_px=.5))
        result=_ordered_foot_material_pairs(profiles,torch.zeros((80,100),dtype=torch.bool),np.array([80.]))
        self.assertFalse(bool(result['missing_same_band'][0]))
        self.assertGreater(len(result['all_ordered_pair_indices_profile_entry_exit']),0)
        self.assertFalse(result['absolute_R_minus_G_used_as_RGB_candidate_veto'])
        self.assertAlmostEqual(float(result['selected_original_R_G_colour_reference_fraction'][0]),0.)


class LocalMaterialDirectionRoleTests(unittest.TestCase):
    def source(self, *, wider=True, different=True, warmer_yellow=False, count=31):
        rgb=np.full((count,40,3),.75,np.float32)
        entry=np.full(count,10,np.int64);exit=np.full(count,18,np.int64)
        foot=np.array([.85,.75,.56],np.float32)
        rgb[:,11:19]=foot
        own=count//2
        if wider:
            entry[own],exit[own]=4,28
        selected=foot if not different else np.array([.60,.49,.485],np.float32)
        if warmer_yellow:
            selected=.75+(foot-.75)*.45
        rgb[own]=.75;rgb[own,entry[own]+1:exit[own]+1]=selected
        uv=np.column_stack((np.arange(count)*4,np.full(count,25))).astype(np.float32)
        profiles=dict(rgb_profile=rgb,valid_mask=np.ones((count,40),bool),
            source_binding=dict(source_sha256='a'*64,source_pts=1,time_base='1/60',shot_id='s'),
            metadata=dict(sample_step_native_px=.5))
        pair=dict(entry_index=entry,exit_index=exit,selected_band_width_px=(exit-entry)*.5)
        family=dict(families=[dict(family_id=0,original_RGB_query_indices=list(range(count)))])
        return profiles,pair,uv,family,own

    def test_foreground_colour_and_wider_band_withhold_role_without_deleting_RGB(self):
        profiles,pair,uv,family,own=self.source()
        preserved_rgb=profiles['rgb_profile'].copy();preserved_uv=uv.copy()
        result=source_local_material_direction_evidence(profiles,pair,uv,np.ones(len(uv),bool),family)
        self.assertTrue(result['source_material_direction_role_conflict'][own])
        self.assertFalse(result['raw_RGB_candidate_veto'])
        np.testing.assert_array_equal(profiles['rgb_profile'],preserved_rgb)
        np.testing.assert_array_equal(uv,preserved_uv)
        reference=next(r for r in result['source_reference_evidence'] if r['source_query_index']==own)
        self.assertGreater(len(reference['source_reference_query_indices']),8)
        self.assertEqual(reference['source_reference_actual_uv'].dtype,uv.dtype)

    def test_colour_difference_alone_is_not_a_material_role_veto(self):
        profiles,pair,uv,family,own=self.source(wider=False)
        result=source_local_material_direction_evidence(profiles,pair,uv,np.ones(len(uv),bool),family)
        self.assertTrue(result['source_material_direction_disagreement'][own])
        self.assertFalse(result['source_material_width_disagreement'][own])
        self.assertFalse(result['source_material_direction_role_conflict'][own])

    def test_actual_yellow_direction_survives_illumination_and_band_width_change(self):
        profiles,pair,uv,family,own=self.source(warmer_yellow=True)
        result=source_local_material_direction_evidence(profiles,pair,uv,np.ones(len(uv),bool),family)
        self.assertFalse(result['source_material_direction_role_conflict'][own])

    def test_missing_neighbourhood_does_not_invent_a_global_hue_reference(self):
        profiles,pair,uv,family,own=self.source(count=5)
        result=source_local_material_direction_evidence(profiles,pair,uv,np.ones(len(uv),bool),family)
        self.assertFalse(result['source_reference_observed'].any())
        self.assertFalse(result['source_material_direction_role_conflict'].any())
        self.assertTrue(np.isnan(result['source_reference_direction_rad']).all())


if __name__=='__main__':
    unittest.main()
