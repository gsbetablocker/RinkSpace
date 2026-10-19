"""Rebuild only saved q091 truth tangents and inspect the actual phase-I failure.

Reuses the existing all-face builder. No source J, fit, source images or report.
The small sparse numeric diagnostic is kept in the failed run's bootstrap.
"""
from pathlib import Path
import ast
import copy
import inspect
import sys
import time

import audit_q8a15_complete_truth_phase_I_20261011 as common

np = common.np
R = common.RUN.parent
RUN = R / 'okada_review86_q8_all_saved_location_estimators_fix_20261011'
Q = '091f7278c5c601208c168fdbf9b560f227503d2d2dfd6fd0f3a10a7568fb16c8'
FAILED = R / 'okada_review86_q091f_saved_location_semantics_20261011'
CONFIG = common.ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q091f_original_saved_locations_and_F3_fix_20261011.json'


def assemble():
    # Execute the unchanged existing reconstruction, stopping before any save
    # or numerical solver. All dependency columns and all 86 groups remain.
    source = ast.parse(inspect.getsource(common.main))
    nodes = source.body[0].body
    start = next(i for i, n in enumerate(nodes)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'options')
    stop = next(i for i, n in enumerate(nodes)
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'matrix')
    body = ast.parse('began = time.perf_counter()').body + copy.deepcopy(nodes[start:stop])
    body += ast.parse('return locals()').body
    function = ast.FunctionDef(name='build', args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(common.__dict__, RUN=RUN, Q=Q, DIRECTORY=RUN/'accepted_states'/Q,
        CONFIG=CONFIG, DEST=Path(str(FAILED)+'.bootstrap'))
    exec(compile(tree, __file__+'::<existing-all-face-builder>', 'exec'), namespace)
    return namespace['build']()


def main():
    from scipy.sparse import csr_matrix
    from calibration.solver_runtime.priority_feasibility_witness_runtime import install_priority_feasibility_witness_runtime
    v = assemble()
    before = {k: common.ah(v[k]) for k in ('A', 'b', 'lo', 'hi', 'owned', 'v')}
    common.emit(stage='complete_original_domain_rebuilt', q=Q, rows=len(v['b']),
                columns=len(v['lo']), groups=len(v['v']))
    install_priority_feasibility_witness_runtime()
    captured = {}

    def trace(frame, event, arg):
        if event == 'exception' and frame.f_code.co_name == '_verified_same_original_conic_phaseI':
            captured.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = common.priority.contract_original_support_bounds(
            v['lo'], v['hi'], v['A'], v['b'], v['owned'], v['v'])
        common.emit(stage='phase_I_passed', common_fraction=result['common_fraction'])
    except ValueError as error:
        common.emit(stage='actual_phase_I_failure_reproduced', error=str(error))
    finally:
        sys.settrace(old_trace)
    assert before == {k: common.ah(v[k]) for k in before}
    if not captured:
        return
    for key in ('B', 'expansion', 'rhs', 'norm', 'x', 'dual', 'slack'):
        if key not in captured:
            raise RuntimeError('Missing actual failure field: '+key)
    c = captured
    common.emit(stage='actual_failed_candidate', status=str(c['solved'].status),
        original_failure=c['original_failure'], original_t=c['original_t'], clipped_t=float(c['x'][-1]),
        min_slack=float(np.min(c['slack'])), min_dual=float(np.min(c['dual'])),
        bad_primal_count=int(np.count_nonzero(c['slack'] < -1e-9)),
        bad_dual_count=int(np.count_nonzero(c['dual'] < -1e-10)))
    # Reconstruct exact preconditioner row mask for names of failed faces.
    n=len(v['lo']);il=np.flatnonzero(np.isfinite(v['lo']));iu=np.flatnonzero(np.isfinite(v['hi']))
    rows=np.vstack((v['A'], np.eye(n)[il], -np.eye(n)[iu]))
    rhs=np.r_[v['b'],v['lo'][il],-v['hi'][iu]]
    expansion=np.r_[v['coeff'],np.zeros(len(il)+len(iu))]
    cn=np.max(abs(rows),axis=0);D=np.divide(1.,cn,out=np.ones(n),where=cn>0.)
    B=rows*D[None,:];norm=np.sqrt(np.einsum('ij,ij->i',B,B)+expansion*expansion)
    mask=norm!=0.;original=np.flatnonzero(mask)
    for key, expected in [('B',B[mask]),('rhs',rhs[mask]),('norm',norm[mask]),('expansion',expansion[mask])]:
        np.testing.assert_array_equal(c[key], expected)
    worst=np.argsort(c['slack'])[:12]
    for index in worst:
        row=int(original[index]) if index<len(original) else None
        desc=(common.row_description(row,v['physical'],v['blocks'],v['groups'])
              if row is not None and row<len(v['b']) else dict(family='box', row=row))
        common.emit(stage='worst_actual_phaseI_face', row=row, normalized_slack=float(c['slack'][index]),
                    dual=float(c['dual'][index]), description=desc)
    sparse=csr_matrix(v['A'])
    target=Path(str(FAILED)+'.bootstrap')/'q091_original_priority_failure.npz'
    if target.exists():
        raise FileExistsError('Preserved failure capture already exists')
    np.savez_compressed(target,A_data=sparse.data,A_indices=sparse.indices,A_indptr=sparse.indptr,
        A_shape=np.array(sparse.shape),lower=v['lo'],upper=v['hi'],original_constraint_lower=v['b'],
        row_group_index=v['owned'],current_group_violations=v['v'],row_norm=norm,column_scale=D,
        nonconstant_rows=mask,failed_coordinate=c['x'],failed_dual=c['dual'],failed_slack=c['slack'],
        selected_canonical_columns=v['selected_c'],canonical_q=v['canonical'])
    common.emit(stage='small_sparse_failure_saved', path=str(target), bytes=target.stat().st_size,
                source_images_written=0, fit_or_source_J=False)


if __name__ == '__main__':
    main()
