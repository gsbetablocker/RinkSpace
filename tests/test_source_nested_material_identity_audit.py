from calibration.program_tracking.source_nested_material_identity_audit import (
    inherited_nested_material_review_problem, NESTED_GRADIENT_BASIS,
)


def inherited_row(material='F3:inner'):
    return dict(physical_material_id=material,actual_pts=229333248,native_pts=229333248,
        decoded_BGR_sha256='a'*64,source_seed_reference=dict(
            physical_identity_basis=NESTED_GRADIENT_BASIS,
            source_annotation_id='user_complete_position_F3_outer_native229332992'))


def direct_proof(material='F3:inner',uv=None):
    return dict(original_source_image_actually_reviewed=True,decision='allow',
        source_feature_physical_identity_confirmed=True,physical_material_id=material,
        actual_pts=229333248,native_pts=229333248,decoded_BGR_sha256='a'*64,
        uv=uv or [158.5,867.1],model_projection_or_fit_residual_used_for_identity=False)


def test_confirmed_outer_does_not_certify_a_named_inner_material():
    assert inherited_nested_material_review_problem(inherited_row(),
        direct_proof('F3:outer'),point_uv=[158.5,867.1])
    assert inherited_nested_material_review_problem(inherited_row('F3:outer')) is None


def test_actual_material_review_cannot_be_borrowed_from_another_point_or_native_frame():
    row=inherited_row();proof=direct_proof()
    assert inherited_nested_material_review_problem(row,proof,point_uv=[158.5,867.1]) is None
    assert inherited_nested_material_review_problem(row,proof,point_uv=[158.5,867.2])
    proof['native_pts']-=256
    assert inherited_nested_material_review_problem(row,proof,point_uv=[158.5,867.1])
