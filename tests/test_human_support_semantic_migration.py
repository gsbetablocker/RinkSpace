"""Replay the actual q757 all86 certificate and reject an altered startup."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics.prepare_q757_finite_support_revision import (
    SYS,PREFIX,actual_receipt_carriers,invoke,read,write,ref,
)

pytestmark=pytest.mark.skipif(not(SYS/(PREFIX+'_startup_binding.json')).exists(),
                            reason='Actual all86 same-q semantic audit is unavailable')


def test_actual_q757_all86_caps_only_tighten_and_other_q_rejects(tmp_path):
    f=actual_receipt_carriers(tmp_path);packet,_=invoke(f)
    assert sum(len(v['limits'])for v in packet['families'].values())==86
    for family,row in packet['families'].items():
        old=f.parent['families'][family]
        assert row['group_ids']==old['group_ids'] and row['uncertainties']==old['uncertainties']
        np.testing.assert_array_less(np.asarray(row['limits']),np.asarray(old['limits'])+1e-12)
        assert np.all(np.asarray(row['limits'])[np.asarray(old['limits'])==0.] == 0.)
    f.q[0]=np.nextafter(f.q[0],np.inf)
    with pytest.raises(ValueError,match='complete q differs'):invoke(f)


def test_semantic_revision_requires_explicit_actual_certificate(tmp_path):
    f=actual_receipt_carriers(tmp_path);scope=deepcopy(f.scope)
    del scope['admitted_truth_support_operator_revision']
    binding=tmp_path/'MISSING_REVISION_REJECTED.json';write(binding,scope);f.binding=binding
    with pytest.raises(ValueError,match='explicit same-q semantic migration'):invoke(f)


def test_semantic_revision_rejects_expanded_cap_and_changed_actual_operator_value(tmp_path):
    f=actual_receipt_carriers(tmp_path)
    values=read(f.scope['exact_caps_values']['path']);values['families']['native']['limits'][44]+=.001
    path=tmp_path/'EXPANDED_CAP_REJECTED.json';write(path,values)
    scope=deepcopy(f.scope);scope['exact_caps_values']=ref(path)
    binding=tmp_path/'EXPANDED_CAP_BINDING_REJECTED.json';write(binding,scope);f.binding=binding
    with pytest.raises(ValueError,match='cap prefix differs'):invoke(f)
    f=actual_receipt_carriers(tmp_path);f.factors['native'].violations[44]+=.001
    with pytest.raises(ValueError,match='violations differ'):invoke(f)
