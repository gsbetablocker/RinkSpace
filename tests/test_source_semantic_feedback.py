from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from calibration.program_tracking.test_strict_absolute_source import fixture_inputs
from calibration.program_tracking.strict_absolute_source import normalize_selected_ice_rows
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, file_sha
from rinkspace.strict_source_supervision import write_strict_negative_source_pack
from rinkspace.source_semantic_feedback import (
    KnownSourceSemanticFeedback, write_known_source_role_feedback,
    known_source_semantic_point_loss,
)


CLASSES = ['background', 'ice_paint', 'ad_boundary', 'shadow_or_reflection']


def source_feedback(tmp_path):
    binding, row, measurement, inventory = fixture_inputs(tmp_path)
    bgr = np.zeros((1080, 1920, 3), np.uint8)
    binding['decoded_native_BGR_sha256'] = hashlib.sha256(bgr.tobytes()).hexdigest()
    binding['shot_id'] = 'main1:0'
    measurement['source_binding'] = deepcopy(binding)
    normalized = normalize_selected_ice_rows([row], measurement,
        source_binding=binding, curve_inventory=inventory)
    source = normalized['review'].source
    png = tmp_path/'actual_native.png'
    cv2.imwrite(str(png), bgr)
    png_reference = dict(path=str(png), sha256=file_sha(png))
    frozen = tmp_path/'original_uv.npz'
    points = np.array([[3.25, 4.75], [6., 7.]])
    np.savez(frozen, uv=points)
    proof = deepcopy(next(iter(normalized['review'].records.values())))
    proof.update(decision='exclude', scope='outside_model', static_material_confirmed=False,
        physical_material_id='Ad7.ice_reflection', reject_only_authority='actual_visual_role_absence',
        actual_source_RGB_role_absence_confirmed=True, confidence_used_as_rejection_evidence=False,
        model_probability_used_as_rejection_evidence=False, excluded_kinds=['ice_paint', 'ad_boundary'],
        human_region_authority=dict(png_reference, statement='Explicit synthetic reflection fixture'),
        original_pixel_inventory_reference=dict(path=str(frozen), sha256=file_sha(frozen)),
        shot_id='main1:0')
    review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA, source=source,
        reviewed_endpoints=[dict(proof, uv=p.tolist(), original_frozen_pixel_index=i)
            for i, p in enumerate(points)]))
    negative = write_strict_negative_source_pack(tmp_path/'negative.npz', review)
    negative_sha = file_sha(negative['path'])
    reference = write_known_source_role_feedback(tmp_path/'semantic.npz',
        original_negative_reference=negative, source_png_reference=png_reference,
        human_role_confirmation=dict(authority_kind='human_current_conversation_role_statement',
            statement='Explicit synthetic known reflection role', kind='shadow_or_reflection',
            physical_material_id='Ad7.ice_reflection', model_probability_used_for_role=False))
    assert file_sha(negative['path']) == negative_sha
    reader = KnownSourceSemanticFeedback(reference, classes=CLASSES)
    plan = dict(shots=[dict(id='main1:0', source_sha256=binding['source_sha256'],
        time_base=binding['time_base'], whole_shot=True, split='train')])
    return reader, bgr, binding, plan


def test_known_roles_only_plan_on_original_whole_training_shot(tmp_path):
    reader, bgr, binding, plan = source_feedback(tmp_path)
    annotations = reader.frame_supervision(bgr, binding, shot_plan=plan)
    assert len(annotations) == 1 and annotations[0]['uv'] == [[3.25, 4.75], [6., 7.]]
    assert reader.supervision_catalog(110, shot_plan=plan) == annotations
    assert reader.source_metadata == reader.source
    assert annotations[0]['fitting_admissible'] is False
    assert not any(k in annotations[0] for k in ('normal', 'interval_px', 'primitive', 'model_target'))
    held = deepcopy(plan); held['shots'][0]['split'] = 'development'
    assert reader.frame_supervision(bgr, binding, shot_plan=held) == []
    invalid = deepcopy(plan); invalid['shots'][0]['whole_shot'] = False
    with pytest.raises(ValueError, match='complete-shot'):
        reader.frame_supervision(bgr, binding, shot_plan=invalid)


def test_actual_rgb_source_clock_and_native_cannot_be_substituted(tmp_path):
    reader, bgr, binding, plan = source_feedback(tmp_path)
    changed = bgr.copy(); changed[0, 0, 0] = 1
    with pytest.raises(ValueError, match='actual native RGB'):
        reader.frame_supervision(changed, binding, shot_plan=plan)
    with pytest.raises(ValueError, match='actual native frame'):
        reader.frame_supervision(bgr, dict(binding, actual_pts=101), shot_plan=plan)
    with pytest.raises(ValueError, match='exact source clock'):
        reader.frame_supervision(bgr, dict(binding, time_base='2/30720'), shot_plan=plan)
    with pytest.raises(ValueError, match='exact integers'):
        reader.frame_supervision(bgr, dict(binding, native_pts=110.1), shot_plan=plan)
    with pytest.raises(ValueError, match='hash aliases'):
        reader.frame_supervision(bgr, dict(binding, video_sha256='a'*64), shot_plan=plan)


def test_semantic_ce_targets_known_class_only_and_protects_direct_human_uv(tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(1)
    reader, bgr, binding, plan = source_feedback(tmp_path)
    annotations = reader.frame_supervision(bgr, binding, shot_plan=plan)
    logits = torch.zeros((1, len(CLASSES), 8, 8), requires_grad=True)
    loss, count = known_source_semantic_point_loss(logits, annotations, CLASSES, source_binding=binding)
    assert count == 2
    assert float(loss.detach()) == pytest.approx(np.log(len(CLASSES)))
    loss.backward()
    assert logits.grad[0, 3, 4, 3] < 0 and logits.grad[0, 1, 4, 3] > 0
    human = [dict(state='user_confirmed_visual_supervision', kind='ice_paint', uv=[[3.25, 4.75]])]
    with pytest.raises(ValueError, match='same original UV'):
        reader.frame_supervision(bgr, binding, shot_plan=plan, human_annotations=human)
    with pytest.raises(ValueError, match='same original UV'):
        known_source_semantic_point_loss(logits, annotations, CLASSES,
            source_binding=binding, human_annotations=human)
    retired = deepcopy(human); retired[0]['source_active_supervision_mask'] = [False]
    assert reader.frame_supervision(bgr, binding, shot_plan=plan, human_annotations=retired) == annotations


def test_loss_rejects_changed_uv_duplicated_indices_or_position_targets(tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(1)
    reader, bgr, binding, plan = source_feedback(tmp_path)
    annotations = reader.frame_supervision(bgr, binding, shot_plan=plan)
    logits = torch.zeros((1, len(CLASSES), 8, 8))
    changed = deepcopy(annotations); changed[0]['uv'][0][0] += .01
    duplicate = deepcopy(annotations); duplicate[0]['original_point_indices'][1] = 0
    position = deepcopy(annotations); position[0]['normal'] = [[0., 1.], [0., 1.]]
    wrong_role = deepcopy(annotations); wrong_role[0]['vision_training_role'] = 'held'
    for rows in (changed, duplicate, position, wrong_role):
        with pytest.raises(ValueError, match='alter source roles'):
            known_source_semantic_point_loss(logits, rows, CLASSES, source_binding=binding)
