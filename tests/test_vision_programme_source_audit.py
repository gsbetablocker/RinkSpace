"""The posterior checker must not turn old corners or expected IDs into truth."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

spec = spec_from_file_location("programme_audit", Path(__file__).parents[1] / "scripts/audit_vision_programme_sources.py")
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


def test_original_corner_array_is_not_transposed_or_expanded_to_rectangle():
    uv = np.array([[545., 865.], [545., 230.]], dtype=np.float64)
    group = dict(group_id="old", uv=uv, original_point_physical_ids=["ad2:left_bottom", "ad2:left_top"],
                 feature_id="known_ad_corners", geometry_role=None)
    rows = audit.point_role_groups(group)
    assert [x["geometry_role"] for x in rows] == ["corner:left_bottom", "corner:left_top"]
    assert all(x["feature_id"] == "Ad2" for x in rows)
    assert np.array_equal(np.concatenate([x["uv"] for x in rows]), uv)
    assert rows[0]["uv"].dtype == uv.dtype
    with pytest.raises(ValueError, match="Nx2"):
        audit.original_array(np.zeros((2, 3)))


def test_expected_identity_cannot_satisfy_actual_confirmed_reference():
    binding = dict(source_sha256="a"*64, time_base="1/15360", source_pts=256)
    reference = dict(group_id="human", source_binding=binding, feature_id="Ad24",
                     geometry_role="outermost_visible_circular_print_stroke", authority_role="direct_user_confirmed_native_UV",
                     uv=np.array([[10., 20.]]), active_reference=True)
    row = dict(source_binding=binding, named_readings=[dict(expected_id="Ad24", physical_id=None,
        physical_role=reference["geometry_role"], actual_source_uv=[[10., 20.]], hint_pixels_used_as_output=False)])
    result = audit.audit_frame(row, [reference])
    assert result["source_crosscheck"]["comparisons"][0]["status"] == "missing_same_identity_and_role_source_measurement"
    assert not result["fitting_qualified_measurements"]
    assert result["expected_named_ID_does_not_establish_physical_identity"]


def test_retired_reference_is_preserved_but_never_compared_as_active():
    binding = dict(source_sha256="a"*64, time_base="1/15360", source_pts=256)
    reference = dict(group_id="old_wrong_ad7", source_binding=binding, feature_id="Ad7", geometry_role="corner:right_top",
                     authority_role="historical_assistant_source_measurement", uv=np.array([[10., 20.]]), active_reference=False)
    result = audit.audit_frame(dict(source_binding=binding, named_readings=[]), [reference])
    assert result["retired_original_reference_groups"] == ["old_wrong_ad7"]
    assert result["source_crosscheck"]["comparisons"] == []


def test_missing_raw_candidate_is_not_fabricated_and_primary_must_be_actual_subset():
    binding = dict(source_sha256="a"*64, time_base="1/15360", source_pts=256)
    candidate = np.array([[10.,20.],[np.nan,np.nan]],dtype=np.float32)
    reading = dict(candidate_source_uv=candidate,candidate_usable=np.array([True,False]),
        primary_role_samples=np.array([True,False]),primary_source_uv=candidate[:1],
        mask_or_navigation_UV_used_as_output=False,physical_role="ice_contact_board_foot_lower_edge")
    row = dict(source_binding=binding,named_readings=[],source_board_foot_material_reading=dict(readings=[reading]))
    result = audit.audit_frame(row,[])
    assert not result["failures"]
    assert result["material_readings"][0]["nonfinite_raw_candidate_count"] == 1
    reading["primary_source_uv"] = np.array([[10.,21.]],dtype=np.float32)
    assert result["fitting_qualified_measurements"] == []
    assert audit.audit_frame(row,[])["failures"] == ["material_0_primary_is_not_existing_actual_candidate_subset"]


def compact_fixture():
    binding = dict(source_sha256="a"*64, time_base="1/15360", source_pts=256)
    uv = np.array([[10.,20.],[12.,20.25]],dtype=np.float32)
    group = dict(actual_source_uv=uv,source_indices=np.array([5,9],np.int64),expected_id="Ad24",
        physical_id="Ad24",identity_authority=dict(kind="original_source_confirmed_identity",reference="actualsource"),
        geometry_role="outermost_visible_circular_print_stroke",RGB_recheck_executed=True,
        observed_uncertainty=dict(original_interval=[[-2.,2.],[-3.,3.]]),measurement_method="original_RGB_peak_recheck")
    row = dict(schema="rinkspace.application-source-observations.v1",source_binding=binding,measurement_groups=[group])
    ref = dict(source_binding=binding,group_id="old",feature_id="Ad24",geometry_role=group["geometry_role"],
        authority_role="historical_assistant_source_measurement",uv=uv.copy(),active_reference=True)
    return row,ref


def test_compact_keeps_original_arrays_and_saves_pointer_not_duplicate_uv():
    row,ref=compact_fixture();original=row["measurement_groups"][0]["actual_source_uv"].copy()
    checked=audit.audit_compact_application_frame(row,[ref])
    metadata=checked["application_measurement_groups"][0]
    assert metadata["original_array"]["dtype"]=="<f4"
    assert metadata["source_measurement_pointer"]=="/measurement_groups/0"
    assert "actual_source_uv" not in metadata
    assert checked["source_crosscheck"]["comparisons"][0]["identity_and_physical_role_matched"]
    assert np.array_equal(row["measurement_groups"][0]["actual_source_uv"],original)


def test_flow_only_coordinates_cannot_pass_as_new_RGB_measurement():
    row,ref=compact_fixture();row["measurement_groups"][0]["RGB_recheck_executed"]=False
    checked=audit.audit_compact_application_frame(row,[ref])
    assert checked["propagated_without_RGB_recheck_point_count"]==2
    assert checked["final_actual_RGB_point_count"]==0
    assert not checked["source_crosscheck"]["comparisons"][0]["identity_and_physical_role_matched"]


def test_compact_foreign_frame_or_hidden_raw_pool_fails():
    row,ref=compact_fixture();row["measurement_groups"][0]["source_binding"]={**row["source_binding"],"source_pts":512}
    with pytest.raises(ValueError,match="another native"):
        audit.audit_compact_application_frame(row,[ref])
    row,_=compact_fixture();row["profiles"]={}
    with pytest.raises(ValueError,match="full raw resource"):
        audit.audit_compact_application_frame(row,[])


def test_missing_original_indices_and_uncertainty_are_reported_without_invention():
    row,ref=compact_fixture();group=row["measurement_groups"][0]
    group["source_indices"]=np.array([],dtype=np.int64)
    group["observed_uncertainty"]={"localization_interval_not_available":True}
    group["source_flags"]={"missing":np.array([False,True])}
    result=audit.audit_compact_application_frame(row,[ref]);metadata=result["application_measurement_groups"][0]
    assert metadata["original_source_indices_missing_not_fabricated"]
    assert metadata["uncertainty_missing_not_fabricated"]
    assert metadata["named_fitting_qualification_pending"]
    assert metadata["source_flags"]=={"missing":1}


def test_known_instance_does_not_promote_conflicting_material_role_to_fitting():
    row,ref=compact_fixture();row["measurement_groups"][0]["geometry_role"]="printed_pattern_source_role_conflict"
    checked=audit.audit_compact_application_frame(row,[ref]);metadata=checked["application_measurement_groups"][0]
    assert metadata["physical_id"]=="Ad24"
    assert metadata["role_unresolved_or_conflicting"] and metadata["named_fitting_qualification_pending"]
    assert "identity_authority" not in metadata
    assert metadata["identity_authority_source_pointer"]=="/measurement_groups/0/identity_authority"


def test_whole_native_aggregate_keeps_authenticated_resource_pointer_not_fragment_arrays():
    row,ref=compact_fixture();checked=audit.audit_compact_application_frame(row,[ref])
    entry=dict(frame_ordinal=1,record_member_offset=20,record_member_sha256="b"*64,uncompressed_UTF8_sha256="c"*64)
    compact=audit.aggregate_application_check(checked,entry)
    assert "application_measurement_groups" not in compact
    assert compact["measurement_group_count"]==1
    assert compact["distinct_confirmed_physical_instance_count"]==1
    assert compact["source_measurement_record"]["record_member_offset"]==20
    assert compact["source_crosscheck"]["comparisons"]==checked["source_crosscheck"]["comparisons"]
