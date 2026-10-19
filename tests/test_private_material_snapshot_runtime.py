"""Small pickle ownership proof; no full graph, worker or 360 MB owner load."""
from collections import OrderedDict
from functools import partial
import hashlib
import pickle

import numpy as np
import pytest

from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
from calibration.solver_runtime.private_material_snapshot_runtime import (
    PRIVATE_ONLY_UNUSED_FIELDS, install_private_material_snapshot_runtime,
)


def tiny_finite_records(owner, records, q, **kwargs):
    return owner._original_motion_far_board_role_binding


class TinyOwner:
    parameter_count = 100

    def __init__(self):
        self.bindings = {key: dict(contract=dict(id=key), retained_value=np.array([i, i + .25]))
                         for i, key in enumerate(('used0', 'used1', 'unused'))}
        self.registration_ledger = [value['contract'] for value in self.bindings.values()]
        self._qualification_prefix = dict(unneeded_registration='retained only in live binder')
        self._scene_cache = OrderedDict()
        self._value_cache = OrderedDict()
        self._successful_world_validation = OrderedDict()
        self._companion_alias_certificates = {('used0', 'world_xyz', 'sha'): dict(scale=1.),
                                               ('unused', 'world_xyz', 'sha'): dict(scale=1.)}
        self.companion_alias_ledger = list(self._companion_alias_certificates.values())
        self._original_motion_far_board_role_binding = dict(unused_motion_J=np.arange(10000, dtype=float))
        self._existing_finite_perimeter_binding = dict(unused_motion_records=['same original UV'])
        self._original_motion_far_board_records_method = tiny_finite_records
        self.finite_board_records = partial(tiny_finite_records, self)

    def _key(self, row):
        return row['id']

    def __getstate__(self):
        return {key: value for key, value in vars(self).items() if key not in ('_scene_cache', '_value_cache')}

    def __setstate__(self, state):
        vars(self).update(state)
        self._scene_cache = OrderedDict()
        self._value_cache = OrderedDict()


def test_private_snapshot_drops_only_unused_self_backreferences_and_keeps_live_owner(tmp_path, monkeypatch):
    original = source.freeze_private_material_owner
    owner = TinyOwner()
    rows = [dict(id='used0', uv=[[1., 2.]], world=[[3., 4., 5.]]),
            dict(id='used1', uv=[[6., 7.]], world=[[8., 9., 10.]])]
    groups = [(123, rows)]
    original_live_fields = dict(vars(owner))
    original_rows = pickle.dumps(rows, protocol=5)
    original_snapshot = original(owner, groups, tmp_path / 'old_immutable.pkl')
    original_bytes = (tmp_path / 'old_immutable.pkl').read_bytes()
    old_owner = pickle.loads(original_bytes)
    assert old_owner.finite_board_records.args[0] is not old_owner
    assert old_owner.finite_board_records.args[0]._qualification_prefix is not None

    monkeypatch.setattr(source, 'freeze_private_material_owner', original)
    monkeypatch.setattr(source, '_rinkspace_private_material_snapshot_runtime', None, raising=False)
    monkeypatch.setattr(source, '_rinkspace_original_freeze_private_material_owner', None, raising=False)
    report = tmp_path / 'receipt'
    receipt = install_private_material_snapshot_runtime(report_directory=report)
    actual = source.freeze_private_material_owner(geometry=owner, groups=groups, destination=tmp_path / 'new_immutable.pkl')
    private = pickle.loads((tmp_path / 'new_immutable.pkl').read_bytes())
    assert (tmp_path / 'old_immutable.pkl').read_bytes() == original_bytes
    assert actual['sha256'] == hashlib.sha256((tmp_path / 'new_immutable.pkl').read_bytes()).hexdigest()
    assert actual['serialized_bytes'] < original_snapshot['serialized_bytes']
    assert actual['registered_material_count'] == original_snapshot['registered_material_count'] == 2
    assert actual['source_row_count'] == original_snapshot['source_row_count'] == 2
    assert actual['world_parameter_count'] == original_snapshot['world_parameter_count'] == 100
    assert pickle.dumps(rows, protocol=5) == original_rows
    assert list(private.bindings) == ['used0', 'used1'] or set(private.bindings) == {'used0', 'used1'}
    for key in private.bindings:
        assert private.bindings[key]['contract'] == old_owner.bindings[key]['contract']
        assert private.bindings[key]['retained_value'].tobytes() == old_owner.bindings[key]['retained_value'].tobytes()
    assert private._companion_alias_certificates == old_owner._companion_alias_certificates
    for field in PRIVATE_ONLY_UNUSED_FIELDS:
        assert field not in vars(private)
    assert '_qualification_prefix' not in vars(private)
    assert set(vars(owner)) == set(original_live_fields)
    for field, value in original_live_fields.items():
        assert vars(owner)[field] is value
    assert owner.finite_board_records.args[0] is owner
    assert actual['private_snapshot_execution']['removed_private_instance_fields'] == list(PRIVATE_ONLY_UNUSED_FIELDS)
    assert receipt['actual_same_q_all_32_source_J_byte_equivalence_pending'] is True
    assert install_private_material_snapshot_runtime(report_directory=report) is receipt
    with pytest.raises(FileExistsError, match='immutable'):
        source.freeze_private_material_owner(owner, groups, tmp_path / 'new_immutable.pkl')
