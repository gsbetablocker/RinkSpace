from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.test_strict_absolute_source import fixture_inputs
from calibration.program_tracking.strict_absolute_source import StrictAbsolutePackWriter, normalize_selected_ice_rows
from calibration.program_tracking.strict_model_scope import file_sha
from rinkspace.strict_source_dataset_training import DatasetSourceTrainingInputs


def _fixture(tmp_path, role='train', held=False):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    row=normalized['rows'][0]
    row['vision_training_role']=role
    row['source_readout_uncertainty']=dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True,paint_width_used_as_center_tolerance=False,
        measurement_method='Explicit isolated test fixture, not source evidence',quantization_step_px=.125,
        actual_readout_blur_interval_px=[-.125,.125],normal_interval_px=[-.25,.25])
    writer=StrictAbsolutePackWriter(tmp_path/'source.npz',source=normalized['review'].source,expected_actual_pts=[100])
    writer.add_row(row,normalized['review'])
    if held:
        guard=deepcopy(row);guard['split']='held'
        writer.add_row(guard,normalized['review'])
    writer.mark_exposure_processed(100)
    ref=writer.finish()
    index=tmp_path/'indices.npz'
    np.savez_compressed(index,row_offsets=np.asarray([0,1,2]if held else [0,1],np.int64),
        fit_point_indices=np.asarray([0],np.int64),held_point_indices=np.asarray([1]if held else [],np.int64))
    manifest=tmp_path/'dataset.json'
    manifest.write_text(json.dumps(dict(schema='rinkspace.immutable-reviewed-finite-source-fit-dataset.v1',
        source=normalized['review'].source,cuts=[dict(source_pack=dict(path=ref['path'],sha256=ref['sha256']),
            support_index=dict(path=str(index),sha256=file_sha(index)))])),encoding='utf8')
    plan=dict(shots=[dict(id='main1:0',split='train',whole_shot=True,source_sha256=binding['source_sha256'],
        time_base=binding['time_base'],start_pts=100,end_pts=200)])
    return dict(path=str(manifest),sha256=file_sha(manifest)),plan,row


def test_existing_row_and_whole_shot_roles_both_required(tmp_path):
    ref,plan,_=_fixture(tmp_path,role='unlisted_not_training')
    with pytest.raises(ValueError,match='explicit whole-shot and row'):
        DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'])


def test_current_actual_sides_and_measured_intervals_survive_bridge(tmp_path):
    ref,plan,row=_fixture(tmp_path)
    inputs=DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'])
    annotation=inputs.supervision_catalog(110)['annotations'][0]
    assert annotation['uv']==np.asarray(row['source_sides_uv'],float).mean(axis=1).tolist()
    assert annotation['interval_px']==[[-.25,.25]]
    assert annotation['model_projection_used_to_generate_or_select_uv'] is False
    assert inputs.supervision_catalog(110)['negatives']==[]
    assert inputs.counts['supervision_points']==1


def test_complete_HELD_exposure_and_same_entity_protection_precedes_labels(tmp_path):
    ref,plan,_=_fixture(tmp_path,held=True)
    with pytest.raises(ValueError,match='explicit whole-shot and row'):
        DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'])


@pytest.mark.parametrize('material_matches', [True, False])
def test_exact_current_pending_filter_blocks_training_and_checks_identity(tmp_path, material_matches):
    ref,plan,row=_fixture(tmp_path)
    manifest=json.loads((tmp_path/'dataset.json').read_text())
    fact=dict(schema='rinkspace.exact-source-named-identity-pending-filter.v1',
        retain_pending_source_readouts_full_literal_evidence=True,source_pack_groups=[dict(
            source_pack_reference=manifest['cuts'][0]['source_pack'],pending_named_point_count=1,
            source_review_witnesses=[],entries=[dict(pack_row_index=0,
                physical_material_id=row['physical_material_id'] if material_matches else 'wrong-material',
                source_binding=dict(source_sha256=manifest['source']['video_sha256'],
                    time_base=manifest['source']['time_base'],actual_pts=row['actual_pts'],
                    native_pts=row['native_pts'],decoded_native_BGR_sha256=row['decoded_BGR_sha256']),
                pack_point_indices=[0],exclude_from_final_named_FIT=True,
                status='source_material_measured_named_identity_pending')])])
    path=tmp_path/'pending.json';path.write_text(json.dumps(fact),encoding='utf8')
    expected='explicit whole-shot and row' if material_matches else 'Exact named exclusion differs'
    with pytest.raises(ValueError,match=expected):
        DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'],
            pending_filters=[dict(path=str(path),sha256=file_sha(path))])


def test_new_review_requires_exact_admission_without_changing_original_None_role(tmp_path):
    ref,plan,row=_fixture(tmp_path,role=None)
    manifest=json.loads((tmp_path/'dataset.json').read_text())
    source=manifest['cuts'][0]['source_pack']
    with np.load(source['path'],allow_pickle=False) as z:
        candidates=z['original_point_indices'].tolist()
    receipt=dict(schema='rinkspace.current-reviewed-exact-source-vision-training-admission.v1',
        source_pack_reference=source,source_review_witnesses=[],fit_dataset_reference=ref,
        vision_training_role='train',source_pack_roles_unchanged=True,model_projection_used_as_labels=False,
        unknown_pending_used_as_positive_or_negative=False,other_unlisted_source_roles_promoted=False,
        points=1,entries=[dict(pack_row_index=0,pack_point_indices=[0],original_candidate_indices=candidates,
            **{k:row[k] for k in ('physical_material_id','feature_id','actual_pts','native_pts','decoded_BGR_sha256')})])
    path=tmp_path/'admission.json';path.write_text(json.dumps(receipt),encoding='utf8')
    inputs=DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'],
        training_admissions=[dict(path=str(path),sha256=file_sha(path))])
    assert inputs.counts['new_exact_review_training_admitted_points']==1
    assert inputs.counts['supervision_points']==1
    assert file_sha(source['path'])==source['sha256']
    assert row['vision_training_role'] is None


def test_cached_source_labels_reuse_only_the_same_indices_and_protection_gate(tmp_path):
    ref,plan,_=_fixture(tmp_path)
    cache=tmp_path/'cache'
    first=DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'],
        supervision_cache_directory=cache)
    second=DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'],
        supervision_cache_directory=cache)
    assert second.counts['supervision_cache_packets_reused']==1
    assert first.supervision_catalog(110)==second.supervision_catalog(110)
    assert first.frame_bindings==second.frame_bindings


def test_explicit_original_video_clock_binding_is_used_with_source_snapshot_evidence(tmp_path):
    ref,plan,_=_fixture(tmp_path)
    # The original media identity stays explicit even when point-proof image
    # evidence is supplied separately by a source review.
    manifest=json.loads((tmp_path/'dataset.json').read_text())
    from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
    source=manifest['cuts'][0]['source_pack']
    pack=StrictAbsoluteSourcePack(source['path'],expected_sha256=source['sha256'])
    video=pack.proof_templates[0]['source_image_evidence']['path']
    inputs=DatasetSourceTrainingInputs(ref,shot_plan=plan,classes=['background','ice_paint'],
        source_video_reference=dict(path=video,sha256=manifest['source']['video_sha256']))
    assert inputs.exact_frame_binding(110)['source_path']==video
    assert inputs.supervision_catalog(110)['annotations'][0]['actual_source_RGB_locations_used'] is True
