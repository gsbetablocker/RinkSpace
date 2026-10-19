import numpy as np
import pytest
from pathlib import Path

from scripts.diagnostics.verify_review85_startup_binding_from_receipts import (
    BINDING,actual_receipt_carriers,invoke,read,
)

pytestmark=pytest.mark.skipif(not BINDING.exists(),reason='Actual saved event source audit receipts unavailable')

def test_actual85_source_receipts_preserve_exact82_caps_and_wrong_q_rejects(tmp_path):
    f=actual_receipt_carriers(tmp_path);packet,_=invoke(f)
    protected=read(f.scope['protected_same_q_caps']['path'])
    assert sum(len(v['group_ids'])for v in packet['families'].values())==85
    for family,values in protected['families'].items():
        n=len(values['group_ids'])
        for key in ('group_ids','limits','uncertainties'):assert packet['families'][family][key][:n]==values[key]
    f.q[0]=np.nextafter(f.q[0],np.inf)
    with pytest.raises(ValueError,match='complete q differs'):invoke(f)

def test_actual_source_identity_and_current_violation_must_match_saved_certificate(tmp_path):
    f=actual_receipt_carriers(tmp_path)
    f.manifest['additional_real_native_ice_curves']['packet_sha256']='0'*64
    with pytest.raises(ValueError,match='qualified native source differs'):invoke(f)
    f=actual_receipt_carriers(tmp_path);f.factors['native'].violations[-1]+=.01
    with pytest.raises(ValueError,match='violations differ'):invoke(f)


CURRENT_BINDING=BINDING.parent/'same_qbf860d13_truth85_ice109_board108_current_target_startup_binding.json'
CURRENT_LAYOUT=BINDING.parent/'original_absolute109_16_active_compaction_complete_layout_verification_20261010.json'


@pytest.mark.skipif(not CURRENT_BINDING.exists() or not CURRENT_LAYOUT.exists(),reason='Current bf source transaction not assembled')
def test_current_real_bf85_caps_continue_through_physical_source_layout_revision(tmp_path):
    f=actual_receipt_carriers(tmp_path,CURRENT_BINDING,CURRENT_LAYOUT);packet,_=invoke(f)
    assert packet['complete_q_sha256'].startswith('bf860d13')
    assert packet['source_target']['source_rows']==4501702
    for family,parent in f.parent['families'].items():
        for key in ('group_ids','limits','uncertainties'):
            assert packet['families'][family][key]==parent[key]
    f.q[0]=np.nextafter(f.q[0],np.inf)
    with pytest.raises(ValueError,match='complete q differs'):invoke(f)
    f=actual_receipt_carriers(tmp_path,CURRENT_BINDING,CURRENT_LAYOUT)
    f.manifest['original_absolute_input_compaction']['original_retirement_low_block_binding_composition']['before']['original_start']+=2
    with pytest.raises(ValueError,match='UI source qualification differs'):invoke(f)
    f=actual_receipt_carriers(tmp_path,CURRENT_BINDING,CURRENT_LAYOUT)
    f.manifest['original_source_point_qualification']['packet_sha256']='0'*64
    with pytest.raises(ValueError,match='Actual reviewed role source differs'):invoke(f)


CURRENT86_BINDING=BINDING.parent/'q1a4d97d9_truth86_TOP_zero_extra_current_target_startup_binding.json'
CURRENT86_LAYOUT=BINDING.parent/'original_absolute109_16_near49_physical_sigma4_same_q1a4_active_compaction_verification_20261010.json'


@pytest.mark.skipif(not CURRENT86_BINDING.exists()or not CURRENT86_LAYOUT.exists(),reason='Actual current-user TOP86 transaction unavailable')
def test_actual_current_TOP86_preserves_parent85_and_rejects_q_or_authority_change(tmp_path):
    # This replays the saved *actual public factor* receipt through the startup
    # gate. The production gate separately recomputes every source violation.
    f=actual_receipt_carriers(tmp_path,CURRENT86_BINDING,CURRENT86_LAYOUT);packet,_=invoke(f)
    assert packet['complete_q_sha256'].startswith('1a4d97d9')
    assert sum(len(v['group_ids'])for v in packet['families'].values())==86
    for family,parent in f.parent['families'].items():
        n=len(parent['group_ids'])
        for key in ('group_ids','limits','uncertainties'):
            assert packet['families'][family][key][:n]==parent[key]
    last=f.values['families']['native'];n=last['old_count']
    assert last['limits'][n:]==[last['violations_px'][-1]+last['uncertainties'][-1]]
    f.q[0]=np.nextafter(f.q[0],np.inf)
    with pytest.raises(ValueError,match='complete q differs'):invoke(f)
    f=actual_receipt_carriers(tmp_path,CURRENT86_BINDING,CURRENT86_LAYOUT)
    f.factors['native'].contract['authority']=f.parent['families']['native']['authority']
    with pytest.raises(ValueError,match='Actual manual authority differs'):invoke(f)


ZERO_TOP_BINDING=BINDING.parent/'q1a4d97d9_truth86_TOP_zero_extra_current_target_startup_binding.json'


@pytest.mark.skipif(not ZERO_TOP_BINDING.exists()or not CURRENT86_LAYOUT.exists(),reason='Actual zero-extra TOP receipt unavailable')
def test_new_TOP_keeps_only_measured_raster_registration_with_no_engineering_expansion(tmp_path):
    f=actual_receipt_carriers(tmp_path,ZERO_TOP_BINDING,CURRENT86_LAYOUT);packet,_=invoke(f)
    for family,parent in f.parent['families'].items():
        n=len(parent['group_ids'])
        for key in ('group_ids','limits','uncertainties'):
            assert packet['families'][family][key][:n]==parent[key]
    native=packet['families']['native'];frozen=f.values['families']['native']
    assert native['uncertainties'][-1]==0.
    assert native['limits'][-1]==frozen['violations_px'][-1]
    assert frozen['violations_px'][-1]>6.
    location=read(f.audit['new_source_options']['--practical-human-location-envelope']['path'])
    row=location['records'][-1]
    for key in ('normal_expansion_px','tangent_expansion_px','engineering_uncorrelated_location_scale_px'):
        assert row[key]==0.
