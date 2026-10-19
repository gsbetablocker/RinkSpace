from rinkspace.strict_source_training_schedule import source_crop_schedule


def positive(feature, uv):
    return dict(feature_id=feature, source_curve_id=feature+':segment0', kind='ice_paint',
        boundary_role='paint_centerline', state='source_reviewed_interval', uv=uv)


def test_late_actual_negative_is_not_lost_when_prefix_exceeds_step_budget():
    catalog=[(i,[positive('C',[[100,600]])],[]) for i in range(100)]
    catalog[80]=(80,[positive('C',[[100,600]])],[dict(uv=[188,838])])
    schedule,receipt=source_crop_schedule(catalog,crop_size=512,native_hw=(1080,1920),optimizer_steps=4)
    assert 80 in schedule and (0,512) in schedule[80]
    assert 0 in schedule and 99 in schedule
    assert sum(map(len,schedule.values()))==4
    assert receipt['all_actual_negative_crops_scheduled'] is True


def test_new_source_entity_and_pending_regions_have_distinct_scheduling_roles():
    pending=positive('unknown',[[1500,600]]);pending['state']='pending'
    catalog=[(0,[positive('C',[[100,600]]),pending],[]),
        (10,[positive('B1',[[1700,800]])],[])]
    schedule,receipt=source_crop_schedule(catalog,crop_size=512,native_hw=(1080,1920),optimizer_steps=2)
    assert schedule=={0:[(0,512)],10:[(1408,512)]}
    assert receipt['required_source_entity_count']==2


def test_late_known_reflection_role_is_scheduled_without_becoming_a_line_target():
    catalog=[(i,[positive('C',[[100,600]])],[]) for i in range(100)]
    role=dict(state='human_confirmed_semantic_role_at_existing_source_uv',
        semantic_supervision_only=True,position_supervision_allowed=False,
        fitting_admissible=False,vision_training_role='train',kind='shadow_or_reflection',
        uv=[[188,838],[185,940]])
    schedule,receipt=source_crop_schedule(catalog,crop_size=512,native_hw=(1080,1920),
        optimizer_steps=4,semantic_catalog=[(80,[role])])
    assert (0,512) in schedule[80]
    assert receipt['mandatory_known_semantic_crop_count']==1
    assert receipt['all_known_semantic_crops_scheduled'] is True
    assert receipt['required_source_entity_count']==1
