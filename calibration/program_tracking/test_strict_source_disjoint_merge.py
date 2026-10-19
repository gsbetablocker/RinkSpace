from copy import deepcopy
import json
import weakref

import numpy as np
import pytest

from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack,
    normalize_selected_ice_rows, normalize_selected_board_rows,
)
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.test_strict_absolute_source import fixture_inputs
import calibration.program_tracking.strict_source_disjoint_merge as disjoint


FIELDS = ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
    'source_side_quantiles_uv', 'original_point_indices')
EXPECTED = [100, 200, 300]


def _fixture_pack(directory, *, pts=100, split='fit', board=False, empty=False,
        processed_pts=None, extra_frame=None):
    directory.mkdir()
    binding, row, measurement, inventory = fixture_inputs(directory)
    binding.update(actual_pts=pts, native_pts=pts+10)
    row['actual_pts'] = pts
    row['temporal_evidence']['current_actual_pts'] = pts
    measurement['source_binding'] = deepcopy(binding)
    if board:
        row.update(scope='board', feature_id='board.upper', physical_material_id='far_board.upper',
            model_target=dict(kind='current_scene_board_edge', edge='upper'),
            observation_semantics='board_physical_boundary',
            admission_status='admitted_source_physical_boundary_temporal_identity',
            uv=np.asarray([[311.,410.], [315.,410.], [319.,410.]]), normal=[[0.,1.]]*3,
            normal_from_actual_source_neighbour_pixels=True, source_neighbour_adjacency_confirmed=True,
            source_readout_uncertainty=dict(actual_source_RGB_measured=True,
                paint_width_used_as_center_tolerance=False, source_native_pixel_quantization_step_px=1.,
                source_interpolation_grid_px=1./32., actual_readout_blur_interval_px=[[-.3,.4]]*3))
        row['temporal_evidence']['physical_material_id']=row['physical_material_id']
        normalized=normalize_selected_board_rows([row], dict(source=binding), source_binding=binding)
        actual=normalized['rows'][0]
        # A saved missing weight must remain missing even with the immutable
        # support ingredients used by ordinary merge's optional reconstruction.
        actual['sigma_px']=np.full(3,np.nan)
    else:
        measurement['candidate_sides']=np.repeat(measurement['candidate_sides'],3,axis=0)
        measurement['candidate_sides'] += np.arange(3,dtype=np.float32)[:,None,None]*4
        measurement['candidate_uv']=measurement['candidate_sides'].astype(float).mean(1).astype(np.float32)
        measurement['candidate_query']=np.arange(3)
        measurement['query_curve']=np.zeros(3,int)
        measurement['candidate_flags']=np.full(3,8,np.uint8)
        row['original_candidate_indices']=[0,1,2]
        normalized=normalize_selected_ice_rows([row],measurement,
            source_binding=binding,curve_inventory=inventory,split=split)
        actual=normalized['rows'][0]
        actual['sigma_px']=np.asarray([2.25,np.asarray([0x7ff8000000000042],np.uint64).view(np.float64)[0],3.5])
    actual['split']=split
    actual['strict_pack_origin']=dict(existing_original_source_row=17, preserve_this=True)
    actual['source_side_quantiles_uv']=np.full((3,2,3,2),
        np.asarray([0xfff8000000000013],np.uint64).view(np.float64)[0])
    actual['source_side_quantiles_uv'][0,0,0,0]=-0.
    writer=StrictAbsolutePackWriter(directory/'pack.npz',source=normalized['review'].source,
        expected_actual_pts=EXPECTED,legacy_pixel_families=['ice','ads'])
    if not empty:
        assert writer.add_row(actual,normalized['review'])==3
    if empty and extra_frame is None:
        writer._register_frame(pts,pts+10,'b'*64)
    if extra_frame is not None:
        writer._register_frame(extra_frame,extra_frame+10,'c'*64)
    writer.exclusions.append(dict(reason='historical_failed_readout_preserved', actual_pts=pts))
    writer.mark_exposure_processed(pts if processed_pts is None else processed_pts)
    return writer.finish()


def _rewrite_metadata(reference, path, update):
    with np.load(reference['path'],allow_pickle=False) as z:
        values={k:z[k].copy() for k in z.files}
    metadata=json.loads(str(values['metadata_json'].item()))
    update(metadata)
    values['metadata_json']=json.dumps(metadata)
    np.savez_compressed(path,**values)
    return dict(path=str(path),sha256=file_sha(path))


def test_disjoint_payload_proofs_held_missing_board_support_and_empty_cut_are_literal(tmp_path):
    first=_fixture_pack(tmp_path/'first')
    held=_fixture_pack(tmp_path/'held',pts=200,split='held',board=True)
    empty=_fixture_pack(tmp_path/'empty',pts=300,empty=True)
    before=[StrictAbsoluteSourcePack(r['path'],expected_sha256=r['sha256']) for r in (first,held,empty)]
    receipt=disjoint.merge_disjoint_strict_absolute_source_packs([first,held,empty],tmp_path/'merged.npz')
    result=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert receipt['points']==6 and receipt['rows']==2
    assert receipt['all_exposures_processed'] is True and receipt['pixels_or_time_subsampled'] is False
    assert receipt['missing_source_sigma_reconstructed'] is False
    for name in FIELDS:
        expected=np.concatenate([p.values[name] for p in before])
        assert result.values[name].dtype==expected.dtype
        assert result.values[name].tobytes()==expected.tobytes(),name
    assert result.values['row_offsets'].tolist()==[0,3,6]
    assert [h['split'] for h in result.rows]==['fit','held']
    assert np.isnan(result.held_rows[0]['sigma_px']).all()
    assert result.metadata['source_uncertainty_ready_for_fit'] is False
    assert result.metadata['expected_actual_pts']==EXPECTED
    assert result.metadata['processed_actual_pts']==EXPECTED
    assert result.values['source_frame_native_pts'].tolist()==[110,210,310]
    assert result.metadata['all_exposures_exact_native_BGR_bound'] is True
    assert result.exclusions==[x for p in before for x in p.exclusions]
    proof_expected=[p.proof_templates[int(code)] for p in before for code in p.values['source_proof_template_index']]
    proof_actual=[result.proof_templates[int(code)] for code in result.values['source_proof_template_index']]
    assert proof_actual==proof_expected
    for i,h in enumerate(result.rows):
        assert h['strict_pack_origin']==before[i].rows[0]['strict_pack_origin']
        assert h['strict_disjoint_pack_origin']['source_row_index']==0
        assert h['strict_disjoint_pack_origin']['source_pack']['sha256']==(first,held)[i]['sha256']
        for k,v in before[i].rows[0].items():
            assert h[k]==v,k


def test_decoder_releases_each_input_pack_before_constructing_next(tmp_path,monkeypatch):
    refs=[_fixture_pack(tmp_path/'one'),_fixture_pack(tmp_path/'two',pts=200)]
    live=maximum=0
    original=disjoint.StrictAbsoluteSourcePack
    def release():
        nonlocal live
        live-=1
    def track(*args,**kwargs):
        nonlocal live,maximum
        pack=original(*args,**kwargs)
        live+=1
        maximum=max(maximum,live)
        weakref.finalize(pack,release)
        return pack
    monkeypatch.setattr(disjoint,'StrictAbsoluteSourcePack',track)
    disjoint.merge_disjoint_strict_absolute_source_packs(refs,tmp_path/'sequential.npz')
    assert maximum==1 and live==0


def test_processed_exposure_overlap_is_rejected_before_any_output(tmp_path):
    first=_fixture_pack(tmp_path/'first')
    second=_fixture_pack(tmp_path/'second')
    out=tmp_path/'rejected.npz'
    with pytest.raises(ValueError,match='Overlapping exposures'):
        disjoint.merge_disjoint_strict_absolute_source_packs([first,second],out)
    assert not out.exists()


def test_conflicting_original_reference_bytes_are_rejected_before_output(tmp_path):
    first=_fixture_pack(tmp_path/'first')
    second=_fixture_pack(tmp_path/'second',pts=200)
    authority=tmp_path/'shared_authority.json';authority.write_text('{"fixture":true}',encoding='utf8')
    common=dict(path=str(authority),sha256=file_sha(authority))
    one=_rewrite_metadata(first,tmp_path/'first_conflict.npz',lambda m:m.update(original_references=[common]))
    two=_rewrite_metadata(second,tmp_path/'second_conflict.npz',lambda m:m.update(
        original_references=[dict(common,sha256='0'*64)]))
    out=tmp_path/'rejected.npz'
    with pytest.raises(ValueError,match='conflicting bytes'):
        disjoint.merge_disjoint_strict_absolute_source_packs([one,two],out)
    assert not out.exists()


def test_source_row_outside_its_processed_cut_is_rejected(tmp_path):
    outside=_fixture_pack(tmp_path/'outside',pts=200,processed_pts=100)
    out=tmp_path/'rejected.npz'
    with pytest.raises(ValueError,match='row belongs outside'):
        disjoint.merge_disjoint_strict_absolute_source_packs([outside],out)
    assert not out.exists()


def test_native_frame_without_processed_cut_membership_is_rejected(tmp_path):
    outside=_fixture_pack(tmp_path/'outside',empty=True,extra_frame=200)
    out=tmp_path/'rejected.npz'
    with pytest.raises(ValueError,match='binds pixels outside'):
        disjoint.merge_disjoint_strict_absolute_source_packs([outside],out)
    assert not out.exists()


def test_invalid_processed_clock_does_not_become_complete(tmp_path):
    first=_fixture_pack(tmp_path/'first')
    forged=_rewrite_metadata(first,tmp_path/'bad_clock.npz',lambda m:m.update(processed_actual_pts=[999]))
    with pytest.raises(ValueError,match='processing clock is invalid'):
        disjoint.merge_disjoint_strict_absolute_source_packs([forged],tmp_path/'rejected.npz')
    assert not (tmp_path/'rejected.npz').exists()
