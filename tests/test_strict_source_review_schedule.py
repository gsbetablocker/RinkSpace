from rinkspace.strict_source_review_schedule import reviewed_source_crop_schedule


def _annotation(authority,point):
    return dict(state='source_reviewed_interval',kind='ice_paint',boundary_role='paint_centerline',
        source_curve_id='B1:segment0',physical_material_id='B1:segment0',uv=[point],authority=dict(sha256=authority))


def test_current_corrected_source_crops_remain_required_when_entity_was_already_seen():
    catalog=[(n,[_annotation('old',[100,700])],[]) for n in range(20)]
    catalog.extend([(25,[_annotation('current',[100,700]),_annotation('current',[900,700])],[]),
        (26,[_annotation('current',[100,700])],[])])
    schedule,record=reviewed_source_crop_schedule(catalog,crop_size=512,native_hw=(1080,1920),
        optimizer_steps=6,reviewed_source_sha256s=['current'])
    assert schedule[25]==[(0,512),(512,512)]
    assert schedule[26]==[(0,512)]
    assert 0 in schedule
    assert record['all_current_review_crops_scheduled'] is True
    assert record['scheduled_crop_count']==6
