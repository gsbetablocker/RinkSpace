import json

import numpy as np
import pytest

from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_review_coverage import consolidate_source_review_coverage


def _reference(path):
    return dict(path=str(path), sha256=file_sha(path))


def _fixtures(tmp_path):
    source = dict(video_sha256='a'*64, time_base='1/15360', pixel_domain='native1920x1080')
    camera = tmp_path/'clock.npz'
    np.savez(camera, pts=[100,110,200], exposure_representative_pts=[100,100,200],
        metadata_json=json.dumps(dict(source_sha256='a'*64, source_time_base='1/15360')))
    identity = tmp_path/'identity.json'
    identity.write_text(json.dumps(dict(camera_path=str(camera), camera_sha256=file_sha(camera))), encoding='utf8')
    identity_ref = _reference(identity)
    original = tmp_path/'original.npz'
    np.savez(original, actual_pts=[100,200], offsets=[0,2,4], uv=np.zeros((4,2)))
    def native(name, pts, actual, sha, *, status=(0,1), board=False):
        aliases = [(i,p) for i,p in enumerate(pts) if p != actual[i]]
        metadata = dict(schema='rinkspace.strict-source-native-alias-material-review.v1',
            source=source, new_FIT_observations_created=False, expected_native_pts_reference=identity_ref)
        path = tmp_path/(name+'.npz')
        np.savez(path, native_pts=pts, actual_pts=actual, BGR_sha256=np.asarray(sha, dtype='S64'),
            representative=np.asarray(pts)==actual, native_material_checked=np.ones(len(pts),bool),
            alias_native_pts=[p for _,p in aliases], alias_candidate_offsets=[0,len(status)] if aliases else [0],
            canonical_candidate_status=np.asarray(status if aliases else (),np.uint8),
            alias_board_checks_json=json.dumps([dict(native_pts=p, actual_pts=actual[i],
                board_current_native_material_checked=board) for i,p in aliases]),
            metadata_json=json.dumps(metadata))
        return _reference(path)
    def retirement(name, status):
        path = tmp_path/(name+'.npz')
        metadata = dict(schema='rinkspace.strict-source-old-pixel-retirement.v1', source=source,
            default_legacy_admission='forbidden', unknown_is_not_negative_training_truth=True,
            original_pixel_reference=_reference(original), legacy_pixel_families=['ice','board'])
        np.savez(path, old_pixel_disposition=np.asarray(status,np.uint8), exposure_pts=[100,200],
            exposure_offsets=[0,2,4], metadata_json=json.dumps(metadata))
        return _reference(path)
    return identity_ref, native, retirement


def test_full_RGB_clock_index_keeps_pending_and_explicit_role_absence(tmp_path):
    identity, native, retired = _fixtures(tmp_path)
    first = native('first', [100,110], [100,100], ['a'*64,'b'*64])
    later = native('later', [200], [200], ['c'*64])
    out = tmp_path/'coverage.npz'
    receipt = consolidate_source_review_coverage([first,later],
        [retired('retired_first',[1,4,0,0]), retired('retired_later',[0,2,3,1])],
        identity_reference=identity, output_path=out)
    assert receipt['processed_native_frames'] == 3
    assert receipt['pending_alias_identity_checks'] == 1
    with np.load(out,allow_pickle=False) as data:
        assert data['old_pixel_disposition'].tolist() == [1,4,3,1]
        assert data['native_review_part_index'].tolist() == [0,0,1]
        assert 'measured_native_uv' not in data.files
        metadata=json.loads(str(data['metadata_json'].item()))
        assert metadata['all_native_frames_RGB_processed'] is True
        assert metadata['all_old_points_received_source_disposition'] is True
        assert metadata['all_native_aliases_admitted_advertisements_material_checked'] is False
        assert metadata['source_processing_is_not_manual_visual_completion'] is True


def test_repeated_native_RGB_hash_cannot_be_silently_replaced(tmp_path):
    identity, native, retired = _fixtures(tmp_path)
    first = native('first', [100,110], [100,100], ['a'*64,'b'*64])
    changed = native('changed', [110], [100], ['d'*64], status=(1,1), board=True)
    out = tmp_path/'rejected.npz'
    with pytest.raises(ValueError,match='competing actual BGR'):
        consolidate_source_review_coverage([first,changed], [retired('retired',[1,2,0,0])],
            identity_reference=identity, output_path=out)
    assert not out.exists()


def test_new_source_delta_retains_every_previous_native_row_locator(tmp_path):
    identity,native,retired=_fixtures(tmp_path)
    first=native('first',[100,110],[100,100],['a'*64,'b'*64])
    newer=native('newer',[110],[100],['b'*64],status=(1,1),board=True)
    out=tmp_path/'history.npz'
    consolidate_source_review_coverage([first,newer],
        [retired('retired',[1,4,0,0])],identity_reference=identity,output_path=out)
    with np.load(out,allow_pickle=False) as values:
        assert values['native_review_history_offsets'].tolist()==[0,1,3,3]
        assert values['native_review_history_part_index'].tolist()==[0,0,1]
        assert values['native_review_history_local_index'].tolist()==[0,1,0]
        metadata=json.loads(str(values['metadata_json'].item()))
        assert metadata['native_review_parts']==[first,newer]
        assert metadata['all_previous_native_review_rows_preserved'] is True
