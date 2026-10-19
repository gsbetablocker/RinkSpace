"""Capture two real production SOC witnesses, then unwind without a trial.

All driver options, normal construction, Schur backsolve and full KKT checks
remain the production route. This process-local observer is diagnostic only.
"""
from __future__ import annotations
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ActualSOCWitnessCaptured(BaseException):
    pass


def sha_array(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def original_closure_bindings(callback):
    """Read through execution-only callback wrappers without altering them."""
    found={};seen=set();pending=[callback]
    while pending:
        current=pending.pop()
        if id(current)in seen:continue
        seen.add(id(current))
        if hasattr(current,'__wrapped__'):pending.append(current.__wrapped__)
        if not inspect.isfunction(current):continue
        values=inspect.getclosurevars(current).nonlocals
        found.update(values)
        pending.extend(v for v in values.values()if inspect.isfunction(v))
    return found


def main():
    import calibration.solver_runtime.__main__ as entry
    from calibration.solver_runtime import execution_acceleration as execution
    original_install = execution.install_named66_execution
    destination = None

    @functools.wraps(original_install)
    def install_capture(driver, directory):
        nonlocal destination
        receipt = original_install(driver, directory)
        import numpy as np
        from scipy.sparse import issparse
        from calibration.solver_runtime import optional_soc_runtime as soc
        from structured_programme_source_candidate import source_step_curvature_tightening as curves
        destination = Path(directory).resolve() / 'actual_SOC_complete_witness_capture'
        if not destination.is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
            raise ValueError('Capture must stay in the existing Data root')
        destination.mkdir(exist_ok=False)
        original_progress = soc._with_outer_nonlinear_progress
        original_measure = curves.measured_outward_step_tightening
        response_inputs = {}

        @functools.wraps(original_measure)
        def observe(matrix, lower, shared_step, responses, actual):
            response_inputs.clear()
            response_inputs.update(matrix=matrix, lower=lower, shared_step=shared_step,
                                   responses=responses, actual=actual)
            return original_measure(matrix, lower, shared_step, responses, actual)
        curves.measured_outward_step_tightening = observe

        @functools.wraps(original_progress)
        def capture(original_SOC, q, linear, selected, domain, options,
                    normal, gradient, diagonal, tightening, solve_linear):
            current = linear
            previous_step = None
            records = []
            closure = original_closure_bindings(tightening)
            evaluation_q = closure['evaluation_q']

            def solve(*args, **kwargs):
                nonlocal current
                current = solve_linear(*args, **kwargs)
                return current

            def source_response(center, displacement, family, row):
                response_inputs.clear()
                tightening(center, displacement, selected, domain)
                if not response_inputs:
                    raise ValueError('Actual production source response unavailable')
                return float(curves.actual_carrier_step_response(
                    domain['curvature_responses'][family], response_inputs['actual'][family])[row])

            def measured(value, step, columns, actual_domain):
                nonlocal previous_step
                correction = tightening(value, step, columns, actual_domain)
                if not response_inputs:
                    raise ValueError('Actual production SOC source measurement unavailable')
                actual = {key: np.asarray(array).copy() for key, array in response_inputs['actual'].items()}
                response = actual_domain['curvature_responses']
                A = actual_domain['matrix']; lower = actual_domain['lower']
                slack = A @ step[columns] - lower
                for family, metadata in response.items():
                    begin, end = metadata['row_start'], metadata['row_stop']
                    slack[begin:end] = curves.actual_carrier_step_response(metadata, actual[family]) - lower[begin:end]
                face = int(np.argmin(slack))
                family = next(key for key, metadata in response.items()
                              if metadata['row_start'] <= face < metadata['row_stop'])
                local_face = face - response[family]['row_start']
                metadata = response[family]
                identity = dict(family=family, original_step_face=face, family_face=local_face,
                                original_slack=float(slack[face]))
                if family == 'native':
                    columns_map = metadata['row_carrier_projection'].getrow(local_face).indices
                    human = closure['human_truth']
                    offsets = np.cumsum([0] + [2 * len(record['uv']) for record in human.full_stroke_records])
                    for carrier in columns_map:
                        if carrier < offsets[-1]:
                            group = int(np.searchsorted(offsets, carrier, side='right') - 1)
                            vertex = int((carrier - offsets[group]) // 2)
                            record = human.full_stroke_records[group]
                            identity.update(group=group, source_id=record['row']['id'],
                                curve=record['row']['curve'], source_vertex=vertex,
                                axis='normal' if (carrier-offsets[group]) % 2 == 0 else 'tangent',
                                source_uv=np.asarray(record['uv'][vertex]).tolist())
                            break
                ordinal = len(records)
                prefix = destination / ('witness_' + str(ordinal))
                np.savez(str(prefix)+'.npz', q=value, step=step, selected=columns,
                    evaluation_q=evaluation_q(value), evaluation_step=evaluation_q(step),
                    normal_diagonal=diagonal, gradient=gradient,
                    full_q_sha256=np.asarray(sha_array(value)), full_step_sha256=np.asarray(sha_array(step)),
                    source_manifest_path=np.asarray(str(Path(directory).resolve().with_suffix('')/'input_manifest.json')))
                merit = soc.original_nonlinear_step_face_feasibility(A, lower, step[columns], response, actual)
                derivatives=[]
                anchors=[('nominal', value, step), ('full_trial', value+step, step)]
                if previous_step is not None:
                    anchors.append(('prior_trial_SOC_change', value+previous_step, step-previous_step))
                for label, center, direction in anchors:
                    for h in (1e-5, 1e-6):
                        plus=source_response(center, h*direction, family, local_face)
                        minus=source_response(center, -h*direction, family, local_face)
                        derivatives.append(dict(anchor=label, h=h, actual_FD=(plus-minus)/(2*h),
                            nominal_frozen_Jv=float(A[face]@direction[columns])))
                # Restore the same nominal source bindings after the observer.
                tightening(value, np.zeros_like(step), columns, actual_domain)
                context=soc._outer_soc_measurement.get()
                if context is not None:context['measurement']=merit
                row=dict(ordinal=ordinal, full_q_sha256=sha_array(value), full_step_sha256=sha_array(step),
                    complete_original_KKT_verified=soc._previous_full_KKT_verified(current),
                    merit=merit, source_identity=identity, directional_derivatives=derivatives,
                    linear_report=current.report, wall_time_ns=time.time_ns(),
                    production_linear_options={key: value for key,value in options.items()
                        if isinstance(value,(bool,float,int,str))})
                (Path(str(prefix)+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2,
                    default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else str(value))+'\n',encoding='utf8')
                if ordinal==0:
                    arrays=dict(matrix=A,lower=lower,selected=columns,q=value)
                    for key,r in response.items():
                        arrays[key+'_nominal_carrier']=r['nominal_carrier']
                        for label in ('retained_mask','row_start','row_stop'):
                            if label in r: arrays[key+'_'+label]=np.asarray(r[label])
                        projection=r.get('row_carrier_projection')
                        if projection is not None:
                            arrays.update({key+'_projection_'+name: getattr(projection,name) for name in ('data','indices','indptr')})
                            arrays[key+'_projection_shape']=np.asarray(projection.shape)
                    np.savez(destination/'actual_production_step_faces.npz',**arrays)
                np.savez(str(prefix)+'_actual_carriers.npz',**actual)
                previous_step=step.copy(); records.append(row)
                print(json.dumps(dict(stage='actual_production_SOC_witness_captured', ordinal=ordinal,
                    max_violation=merit['maximum_violation_px'],total_violation=merit['total_violation_px'],
                    source_identity=identity, full_KKT=row['complete_original_KKT_verified'])),flush=True)
                # Two witnesses diagnose one real correction; this is not a
                # numerical iteration limit and never creates an accepted trial.
                if len(records)==2:
                    raise ActualSOCWitnessCaptured(str(destination))
                return correction
            return original_progress(original_SOC,q,linear,selected,domain,options,
                                     normal,gradient,diagonal,measured,solve)
        soc._with_outer_nonlinear_progress = capture
        return receipt

    execution.install_named66_execution = install_capture
    try:
        entry.main()
    except ActualSOCWitnessCaptured as signal:
        print(json.dumps(dict(stage='actual_production_SOC_capture_completed',path=str(signal),
            diagnostic_only=True,trial_evaluated=False,candidate_accepted=False,
            original_driver_cleanup_unwound=True)),flush=True)
        return 0
    finally:
        execution.install_named66_execution = original_install
    raise RuntimeError('Production did not reach two actual SOC witnesses')


if __name__=='__main__':
    raise SystemExit(main())
