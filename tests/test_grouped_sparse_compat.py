"""Real legacy metadata and active solver type boundaries, no source/state."""
from pathlib import Path
from types import MethodType,SimpleNamespace
import json,os,sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix,issparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'events/ankara_2026/programmes/okada_mei_sp')]
import fit_time_grouped_compat as entry
from calibration.program_tracking.first_shot_source_transition import reference
import z190_global_schur_linear_step as qp
RESULTS={}


class Owner:
    def __init__(self):
        names=('bundle','start','state','shared','fixed','scales','quadratic','formal_offsets',
            'source_producers','recovery_references','worker_receipts','evaluation_receipts',
            'transverse_world_cache','guard_cache','raw_retention_receipt','seed_receipt')
        self.programme=SimpleNamespace(**dict.fromkeys(names))
        self.start=np.zeros(3);self.shared=np.array([0,1]);self.fixed=np.array([2])
        self.matrix=csr_matrix([[1.,0.],[0.,1.]]);self.lower=np.array([-.1,-.2]);self.invalid=False
        def constraints(this,q,selected):
            return dict(matrix=this.matrix,lower=this.lower,report=dict(original_halfspaces=True))
        self.linear_constraints=MethodType(constraints,self)
    def checkpoint_arrays(self):return dict(history=np.array([1.,2.]))
    def checkpoint_identity(self):return dict(protocol='three-parameter type boundary')
    def complete_state_payload(self,q):return dict(q_augmented=q.copy(),aux__history=np.array([1.,2.]))
    def solver_kwargs(self):
        return dict(global_schur_count=len(self.shared)+int(self.invalid),global_schur_indices=self.shared.copy(),
            fixed_parameter_indices=self.fixed.copy(),shared_linear_constraints=self.linear_constraints,
            checkpoint_identity=self.checkpoint_identity())


def test_actual_legacy_preflight_only_gets_temporary_dense(tmp_path):
    owner=Owner();original=owner.linear_constraints
    with pytest.raises((TypeError,ValueError)):
        entry._METADATA_ORIGINAL(owner,tmp_path/'reproduce.json')
    entry.install_solver_delivery(owner)
    receipt=entry.metadata_preflight(owner,tmp_path/'metadata.json')
    record=json.loads((tmp_path/'metadata.json').read_text(encoding='utf-8'))
    assert record['passed'] and record['actual_shared_halfspace_rows']==2 and record['actual_shared_halfspace_columns']==2
    assert owner.__dict__['linear_constraints']is original
    assert owner.linear_constraints(owner.start,owner.shared)['matrix']is owner.matrix
    solver_record=owner.solver_kwargs()['shared_linear_constraints'](owner.start,owner.shared)
    np.testing.assert_array_equal(solver_record['matrix'],owner.matrix.toarray())
    assert solver_record['lower']is owner.lower
    RESULTS['success']=dict(original_preflight_rejection_reproduced=True,legacy_metadata_passed=True,
        owner_callback_same_instance=True,owner_matrix_still_same_CSR=True,lower_object_unchanged=True)


def test_original_method_is_restored_when_old_check_fails(tmp_path):
    owner=Owner();original=owner.linear_constraints;owner.invalid=True
    with pytest.raises(ValueError,match='Complete actual solver metadata'):
        entry.metadata_preflight(owner,tmp_path/'failure.json')
    assert owner.__dict__['linear_constraints']is original
    assert issparse(owner.linear_constraints(owner.start,owner.shared)['matrix'])
    RESULTS['failure']=dict(original_ValueError_preserved=True,owner_callback_same_instance_after_failure=True)


def test_active_schur_solver_receives_same_dense_halfspaces():
    normal=csr_matrix(np.diag([4.,3.,2.]));gradient=np.array([-3.,1.,.4])
    owner=Owner();original=owner.linear_constraints
    options=dict(global_count=2,global_step_constraint_matrix=owner.matrix,global_step_constraint_lower=owner.lower)
    with pytest.raises((TypeError,ValueError)):
        qp.solve_global_schur_step(normal,gradient,**options)
    standard=qp.solve_global_schur_step(normal,gradient,**dict(options,global_step_constraint_matrix=owner.matrix.toarray()))
    entry.install_solver_delivery(owner)
    domain=owner.solver_kwargs()['shared_linear_constraints'](owner.start,owner.shared)
    adapted=qp.solve_global_schur_step(normal,gradient,global_count=2,
        global_step_constraint_matrix=domain['matrix'],global_step_constraint_lower=domain['lower'])
    assert standard.step.dtype==adapted.step.dtype and standard.step.tobytes()==adapted.step.tobytes()
    assert owner.__dict__['linear_constraints']is original and owner.linear_constraints(owner.start,owner.shared)['matrix']is owner.matrix
    RESULTS['solver']=dict(active_solver_CSR_rejection=True,dense_step_byte_exact=True,step=adapted.step.tolist(),
        active_solver=reference(qp.__file__),same_all_halfspaces_and_lower_values=True)
    if os.environ.get('RINKSPACE_COMPAT_QA_OUTPUT'):
        assert set(RESULTS)=={'success','failure','solver'}
        output=Path(os.environ['RINKSPACE_COMPAT_QA_OUTPUT'])
        assert len(str(output.resolve()).encode('utf-16-le'))//2<260
        value=dict(schema='rinkspace.legacy-sparse-dense-boundary-protocol.v1',passed=True,
            entry=reference(entry.__file__),tests=reference(__file__),original_metadata=reference(entry._METADATA_MODULE.__file__),
            active_solver=reference(qp.__file__),temporary_method_restored_on_success_and_failure=True,
            original_owner_CSR_callback_unchanged=True,active_solver_CSR_rejection_reproduced=True,
            active_solver_dense_step_byte_exact=True,details=RESULTS,no_full_source_or_physical_state_constructed=True,
            source_math_or_guard_or_threshold_changed=False)
        with output.open('x',encoding='utf-8')as handle:json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')
