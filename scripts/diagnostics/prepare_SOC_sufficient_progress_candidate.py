"""Prepare, without installing, the optional SOC sufficient-progress control."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
SOURCE=ROOT/'calibration/solver_runtime/optional_soc_runtime.py'
OUTPUT=SYS/'optional_SOC_sufficient_actual_predicted_progress_EXECUTION_CANDIDATE_20261011.py'

HELPER='''def _sufficient_SOC_feasibility_reduction(agreement, maximum_roundoff, total_roundoff):
    """Require a useful fraction of each promised actual feasibility reduction.

    The one-quarter sufficient-progress fraction is an algorithmic condition
    for another optional correction. It is distinct from the existing one-
    quarter symmetric prediction-error rule that controls damping reduction.
    A better verified direction is retained before this check; all final
    physical, truth and objective gates remain authoritative.
    """
    fraction=.25
    if agreement is None:
        return dict(sufficient=True, first_measurement=True,
                    minimum_actual_to_predicted_reduction_fraction=fraction, merits={})
    merits={};promises=0
    for key,roundoff in [('maximum',maximum_roundoff),('total',total_roundoff)]:
        roundoff=float(roundoff)
        entry=agreement[key]
        predicted=float(entry['predicted_source_reduction'])
        actual=float(entry['actual_source_reduction'])
        if not np.isfinite([predicted,actual,roundoff]).all() or roundoff<0.:
            raise ValueError('Finite same-merit SOC predicted/actual reductions required')
        promised=predicted>roundoff
        promises+=int(promised)
        sufficient=(not promised or actual+roundoff>=fraction*predicted)
        merits[key]=dict(predicted_reduction_px=predicted,actual_reduction_px=actual,
            numerical_merit_roundoff_px=roundoff,predicted_reduction_resolvable=promised,
            actual_to_predicted_reduction=None if not promised else actual/predicted,
            sufficient=bool(sufficient))
    return dict(sufficient=bool(promises and all(v['sufficient']for v in merits.values())),
        first_measurement=False,minimum_actual_to_predicted_reduction_fraction=fraction,
        compared_quantities='Same original-face maximum and total violations before/after current candidate',
        no_source_support_or_acceptance_tolerance_change=True,merits=merits)


'''


def main():
    text=SOURCE.read_text(encoding='utf8')
    text=text.replace('def _with_outer_nonlinear_progress(',HELPER+'def _with_outer_nonlinear_progress(',1)
    text=text.replace("            entry.update(improves_best_observed_actual_feasibility=bool(progress),",
        "            sufficient=_sufficient_SOC_feasibility_reduction(agreement,maximum_roundoff,total_roundoff)\n"
        "            entry.update(sufficient_actual_to_predicted_feasibility_reduction=sufficient,\n"
        "                improves_best_observed_actual_feasibility=bool(progress),",1)
    text=text.replace("            if retained is not None and (not progress or not verified):",
        "            if retained is not None and (not progress or not verified or not sufficient['sufficient']):",1)
    text=text.replace("            'optional_actual_original_nonlinear_feasibility_nonprogress')",
        "            ('optional_actual_original_nonlinear_feasibility_insufficient_model_reduction'\n"
        "             if stalled['improves_best_observed_actual_feasibility']else\n"
        "             'optional_actual_original_nonlinear_feasibility_nonprogress'))",1)
    text=text.replace("schema='rinkspace.optional-SOC-actual-original-nonlinear-face-progress.v3'",
        "schema='rinkspace.optional-SOC-actual-original-nonlinear-face-progress.v4'",1)
    text=text.replace("        no_iteration_or_time_limit=True,no_original_support_or_target_relaxation=True,",
        "        no_iteration_or_time_limit=True,no_original_support_or_target_relaxation=True,\n"
        "        minimum_actual_to_predicted_feasibility_reduction_fraction=.25,\n"
        "        sufficient_progress_is_distinct_from_symmetric_prediction_error_trust=True,",1)
    text=text.replace("    lambda reduction, not a further correction with verified actual progress.",
        "    lambda reduction. Another correction requires sufficient actual\n"
        "    reduction of both promised feasibility merits.",1)
    text=text.replace("                  optional_outer_SOC_prediction_disagreement_only_holds_lambda=True,",
        "                  optional_outer_SOC_prediction_disagreement_only_holds_lambda=True,\n"
        "                  optional_outer_SOC_requires_quarter_of_each_resolvable_predicted_merit_reduction=True,",1)
    ast.parse(text)
    OUTPUT.write_text(text,encoding='utf8')
    receipt=dict(candidate=str(OUTPUT),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),installed=False,
        original_lambda_prediction_error_criterion_unchanged=True,retains_better_full_KKT_before_stopping=True,
        mathematical_target_and_final_acceptance_unchanged=True,no_iteration_or_time_limit=True)
    OUTPUT.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
