import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime import static_role_qualification_cache_runtime as candidate


@pytest.fixture
def role_fixture(tmp_path, monkeypatch):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import original_motion_far_board_roles as roles
    original_finish = roles.OriginalMotionFarBoardRoleBinding._finish_certificate
    calls, sha_calls = [], []

    def original_sha(path):
        sha_calls.append(str(Path(path).resolve()))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    class Role:
        _finish_certificate = original_finish

        def __init__(self, packet_path):
            calls.append(str(Path(packet_path).resolve()))
            self.packet_path = Path(packet_path).resolve()
            self.packet = json.loads(self.packet_path.read_text(encoding='utf8'))
            self.records = copy.deepcopy(self.packet['records'])
            for reference in self.packet.get('source_packets', []):
                if roles._sha(reference['path']) != reference['sha256']:
                    raise ValueError('Original child packet changed')
                child = roles.OriginalMotionFarBoardRoleBinding(reference['path'])
                self.records.extend(copy.deepcopy(child.records))
            for reference in self.packet.get('proofs', []):
                if roles._sha(reference['path']) != reference['sha256']:
                    raise ValueError('Original proof changed')
            self._finish_certificate()

    monkeypatch.setattr(roles, 'OriginalMotionFarBoardRoleBinding', Role)
    monkeypatch.setattr(roles, '_sha', original_sha)
    proof = tmp_path/'proof.bin'
    proof.write_bytes(b'abcd')
    row = dict(original_global_bundle_index=3, source_role_block='main1',
               source_pts=11, target_pts=12, source_uv=[1., 2.], target_uv=[2., 3.])
    child = tmp_path/'child.json'
    child.write_text(json.dumps(dict(schema=roles.PROGRAMME_REVIEW_SCHEMA, records=[row],
        proofs=[dict(path=str(proof), sha256=original_sha(proof))])), encoding='utf8')
    outer = tmp_path/'outer.json'
    outer.write_text(json.dumps(dict(schema=roles.COMBINED_SCHEMA, records=[],
        source_packets=[dict(path=str(child), sha256=original_sha(child))])), encoding='utf8')
    controller = candidate._install_on(roles, cache_directory=tmp_path/'cache',
        report_directory=None, producer_identity={'fixture_producer': 'original-finish'})
    return roles, outer, proof, calls, sha_calls, controller


def _equal(left, right):
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if type(left) is dict:
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if type(left) in (tuple, list):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


def test_recursive_miss_then_hit_original_finish_byteexact(role_fixture):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    first = roles.OriginalMotionFarBoardRoleBinding(packet)
    assert len(calls) == 2
    assert controller.events[-1]['cache_saved']
    receipts = list(controller.directory.rglob('qualification_receipt.json'))
    assert len(receipts) == 1  # No repeated recursive producer/payload saves.
    receipt = json.loads(receipts[0].read_text(encoding='utf8'))
    assert {p for p,digest in receipt['original_dependency_SHA256']} == {
        str(packet), str(packet.parent/'child.json'), str(proof)}
    second = roles.OriginalMotionFarBoardRoleBinding(packet)
    assert len(calls) == 2
    assert controller.events[-1]['action'] == 'authenticated_hit'
    assert _equal(first.__dict__, second.__dict__)
    second.source_cameras[11] = np.ones(15)
    second._current_binding_cache = ('different-current-q', {})
    second.records[0]['active_global_bundle_index'] = 9
    third = roles.OriginalMotionFarBoardRoleBinding(packet)
    assert third.source_cameras == {} and third._current_binding_cache is None
    assert 'active_global_bundle_index' not in third.records[0]
    assert np.shares_memory(first.source_uv, third.source_uv) is False
    assert _equal(first.__dict__, third.__dict__)


def test_same_size_mtime_changed_source_runs_original_error(role_fixture):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    roles.OriginalMotionFarBoardRoleBinding(packet)
    before = proof.stat()
    proof.write_bytes(b'efgh')
    os.utime(proof, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='^Original proof changed$'):
        roles.OriginalMotionFarBoardRoleBinding(packet)
    assert len(calls) == 4


def test_corrupt_payload_falls_back_to_original(role_fixture):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    first = roles.OriginalMotionFarBoardRoleBinding(packet)
    payload = next(controller.directory.rglob('static_certificate.pickle'))
    payload.write_bytes(b'corrupt')
    second = roles.OriginalMotionFarBoardRoleBinding(packet)
    assert len(calls) == 4
    assert _equal(first.__dict__, second.__dict__)
    assert controller.events[-1]['action'] == 'original_constructor_miss'
    assert controller.events[-1]['cache_saved'] is False


def test_nonstatic_graph_field_never_saved(role_fixture):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    role = roles.OriginalMotionFarBoardRoleBinding(packet)
    role.geometry = object()
    with pytest.raises(TypeError, match='constructor fields changed'):
        candidate._static_certificate(role)
    role.__dict__.pop('geometry')
    role.source_cameras[11] = np.zeros(15)
    with pytest.raises(TypeError, match='Current cameras present'):
        candidate._static_certificate(role)


def test_sha_capture_delegates_existing_scope_chain(role_fixture):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    sha_calls.clear()
    roles.OriginalMotionFarBoardRoleBinding(packet)
    assert set(sha_calls) == {str(packet), str(packet.parent/'child.json'), str(proof)}
    sha_calls.clear()
    roles.OriginalMotionFarBoardRoleBinding(packet)
    assert sha_calls == []  # The hit uses independent real byte SHA, not a scope hit.


def test_reporting_failure_does_not_change_original_pass(role_fixture, monkeypatch):
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    blocker = packet.parent/'not-a-directory'
    blocker.write_bytes(b'x')
    controller.report_directory = blocker/'reports'
    first = roles.OriginalMotionFarBoardRoleBinding(packet)
    second = roles.OriginalMotionFarBoardRoleBinding(packet)
    assert _equal(first.__dict__, second.__dict__)
    assert controller.events[-1]['action'] == 'authenticated_hit'
    assert 'report_write_error' in controller.events[-1]


def test_cache_hit_preserves_existing_scope_original_exit_rehash(role_fixture, monkeypatch):
    from calibration.solver_runtime.startup_source_auth_reuse_runtime import SourceAuthenticationScope, _source_sha
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    roles.OriginalMotionFarBoardRoleBinding(packet)
    original = candidate._ORIGINAL_SHA
    monkeypatch.setattr(candidate, '_ORIGINAL_SHA', lambda path: _source_sha(path, original))
    sha_calls.clear()
    with SourceAuthenticationScope(packet.parent/'run-output') as scope:
        roles.OriginalMotionFarBoardRoleBinding(packet)
        assert len(scope.files) == 3
        assert all(entry['hits'] > 0 for entry in scope.files.values())
        assert controller.events[-1]['original_startup_auth_exit_rehash_preserved']
    assert scope.revalidated_files == 3 and scope.closed
    assert len(sha_calls) == 6  # Original first registration and original exit read.


def test_only_reviewed_predecessor_hit_and_source_failure_preserve_receipt(role_fixture, monkeypatch):
    from calibration.solver_runtime.startup_source_auth_reuse_runtime import SourceAuthenticationScope, _source_sha
    roles, packet, proof, calls, sha_calls, controller = role_fixture
    controller.producer_identity.update(qualification_cache_helper_SHA256=candidate._REVIEWED_PREDECESSOR_HELPER_SHA256,
        actual_qualification_function_code_SHA256=[['original_finish', 'same-original-finish-code']])
    first = roles.OriginalMotionFarBoardRoleBinding(packet)
    receipt_path = next(controller.directory.rglob('qualification_receipt.json'))
    receipt_bytes = receipt_path.read_bytes()
    controller.producer_identity['qualification_cache_helper_SHA256'] = '7f14ed88f9416ad7feab5dba74d37b6101fc008b03d47c700e12b8f663dfbabb'
    original = candidate._ORIGINAL_SHA
    monkeypatch.setattr(candidate, '_ORIGINAL_SHA', lambda path: _source_sha(path, original))
    with SourceAuthenticationScope(packet.parent/'run-output') as scope:
        second = roles.OriginalMotionFarBoardRoleBinding(packet)
        assert controller.events[-1]['compatible_predecessor_helper_sha256'] == candidate._REVIEWED_PREDECESSOR_HELPER_SHA256
    assert scope.revalidated_files == 3 and len(calls) == 2
    assert _equal(first.__dict__, second.__dict__)
    assert receipt_path.read_bytes() == receipt_bytes
    assert len(list(controller.directory.rglob('qualification_receipt.json'))) == 1
    before = proof.stat()
    proof.write_bytes(b'efgh')
    os.utime(proof, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='^Original proof changed$'):
        roles.OriginalMotionFarBoardRoleBinding(packet)
    assert receipt_path.read_bytes() == receipt_bytes and len(calls) == 4
