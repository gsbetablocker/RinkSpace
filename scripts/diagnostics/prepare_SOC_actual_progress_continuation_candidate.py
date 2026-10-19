"""Prepare an uninstalled execution-control candidate without touching a live owner."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'calibration/solver_runtime/optional_soc_runtime.py'
out=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
candidate=out/'optional_SOC_continue_on_actual_progress_EXECUTION_CANDIDATE_20261011.py'
text=source.read_text(encoding='utf8')
pairs=[
 ('current=linear;retained=None;retained_lower=None;best=None;previous=None;records=[];stalled=None',
  'current=linear;retained=None;retained_lower=None;best=None;previous=None;records=[];stalled=None;unreliable_retained=None'),
 ('nonlocal retained,retained_lower,best,previous,stalled',
  'nonlocal retained,retained_lower,best,previous,stalled,unreliable_retained'),
 ("entry['current_complete_KKT_direction_retained']=True", 
  "entry['current_complete_KKT_direction_retained']=True\n                if not reliable:unreliable_retained=entry\n                entry['prediction_disagreement_only_holds_damping']=not reliable"),
 ('if retained is not None and (not progress or not verified or not reliable):',
  'if retained is not None and (not progress or not verified):'),
 ("            'optional_actual_original_nonlinear_feasibility_model_disagreement'\n            if not stalled['original_face_reduction_model_is_reliable']else\n",''),
 ("schema='rinkspace.optional-SOC-actual-original-nonlinear-face-progress.v2',",
  "schema='rinkspace.optional-SOC-actual-original-nonlinear-face-progress.v3',"),
 ("if stalled is not None and not stalled['original_face_reduction_model_is_reliable']:",
  'if unreliable_retained is not None:'),
 ("original_face_reduction_model_agreement=stalled['original_face_reduction_model_agreement'],",
  "original_face_reduction_model_agreement=unreliable_retained['original_face_reduction_model_agreement'],"),
 ("current_candidate_retained=not stalled['current_candidate_discarded'],",
  "current_candidate_retained=True,\n            later_improved_candidates_may_have_been_retained=True,"),
 ('further cuts and lambda reduction, not retention of that better candidate.',
  'lambda reduction, not a further correction with verified actual progress.'),
 ("optional_outer_SOC_requires_actual_original_nonlinear_face_progress=True,",
  "optional_outer_SOC_requires_actual_original_nonlinear_face_progress=True,\n                  optional_outer_SOC_prediction_disagreement_only_holds_lambda=True,"),
]
for before,after in pairs:
    if text.count(before)!=1:raise ValueError('Expected exact one execution-control splice: '+before)
    text=text.replace(before,after)
compile(text,str(candidate),'exec');candidate.write_text(text,encoding='utf8')
receipt=out/'optional_SOC_continue_on_actual_progress_EXECUTION_CANDIDATE_receipt_20261011.json'
receipt.write_text(json.dumps(dict(schema='rinkspace.uninstalled-execution-SOC-actual-progress-candidate.v1',
    original_source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    candidate=dict(path=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest()),
    uninstalled=True,live_production_source_unmodified=True,
    source_H_g_rows_sigma_supports_operator_and_final_guards_unchanged=True),indent=2),encoding='utf8')
print(json.dumps(dict(candidate=str(candidate),receipt=str(receipt))))
