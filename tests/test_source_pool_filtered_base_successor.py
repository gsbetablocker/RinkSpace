from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, file_sha
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import (
    current_row_source_binding, export_exact_named_identity_pending_pack,
)
from calibration.program_tracking.test_strict_source_named_identity_filter_with_exclusions import _fixture
from scripts.diagnostics.assemble_full_programme_source_pool_20261013 import assemble_cut, require_exact_base_successor


def _reference(path):
    return dict(path=str(path), sha256=file_sha(path))


def _fit_base(tmp_path):
    fixture = tmp_path/'fixture'
    fixture.mkdir()
    _, original, fact = _fixture(fixture)
    writer = StrictAbsolutePackWriter(tmp_path/'fit_base.npz', source=original.metadata['source'],
        expected_actual_pts=original.metadata['expected_actual_pts'],
        original_references=original.metadata['original_references'], legacy_pixel_families=['ice'])
    row = original.fit_rows[0]
    proofs = [dict(deepcopy(original.proof_templates[int(code)]), uv=point.tolist())
        for code, point in zip(original.values['source_proof_template_index'][:3], row['uv'])]
    review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA, source=original.metadata['source'], reviewed_endpoints=proofs))
    assert writer.add_row(row, review) == 3
    writer.mark_exposure_processed(100)
    receipt = writer.finish()
    base_ref = dict(path=receipt['path'], sha256=receipt['sha256'])
    base = StrictAbsoluteSourcePack(base_ref['path'], expected_sha256=base_ref['sha256'])
    group = fact['source_pack_groups'][0]
    group['source_pack_reference'] = base_ref
    group['entries'][0]['source_binding'] = current_row_source_binding(base, 0)
    filter_path = tmp_path/'pending.json'
    filter_path.write_text(json.dumps(fact), encoding='utf8')
    return base_ref, base, _reference(filter_path)


def _empty_delta(tmp_path, base, refs=()):
    writer = StrictAbsolutePackWriter(tmp_path/'empty_delta.npz', source=base.metadata['source'],
        expected_actual_pts=base.metadata['expected_actual_pts'], original_references=list(refs),
        legacy_pixel_families=['ice'])
    for pts, binding in base.frame_bindings.items():
        writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'])
        writer.mark_exposure_processed(pts)
    receipt = writer.finish()
    return dict(path=receipt['path'], sha256=receipt['sha256'])


def test_replacing_base_rebuilds_cut_without_resurrecting_pending_pixels(tmp_path):
    base_ref, base, filter_ref = _fit_base(tmp_path)
    delta_ref = _empty_delta(tmp_path, base)
    output = tmp_path/'assembly'
    entry = dict(shot_id='100:R0402_034', absolute_inputs=base_ref)
    first_ref, first_record = assemble_cut(entry, [delta_ref], output)
    original_receipt = output/'100_R0402_034_assembly_receipt.json'
    original_receipt_bytes = original_receipt.read_bytes()
    filtered = export_exact_named_identity_pending_pack(base_ref, filter_ref, tmp_path/'filtered.npz')
    filtered_ref = dict(path=filtered['path'], sha256=filtered['sha256'])
    entry['absolute_inputs'] = filtered_ref
    second_ref, second_record = assemble_cut(entry, [delta_ref], output)
    second = StrictAbsoluteSourcePack(second_ref['path'], expected_sha256=second_ref['sha256'])
    assert second.values['uv'].tobytes() == base.values['uv'][[0, 2]].tobytes()
    assert second_record['previous_merged_pixels_reused'] is False
    assert second_record['earlier_original_cut_reference'] == base_ref
    assert original_receipt.read_bytes() == original_receipt_bytes
    assert file_sha(first_ref['path']) == first_ref['sha256']
    reused_ref, _ = assemble_cut(entry, [delta_ref], output)
    assert reused_ref == second_ref
    assert second.exclusions[-1]['pack_point_indices'] == [1]


def test_replacement_requires_original_authority_and_identical_clock(tmp_path):
    base_ref, base, _ = _fit_base(tmp_path)
    unrelated = _empty_delta(tmp_path, base)
    with pytest.raises(ValueError, match='original source authority'):
        require_exact_base_successor(base_ref, unrelated)
