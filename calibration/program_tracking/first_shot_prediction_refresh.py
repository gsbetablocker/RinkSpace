"""Opt-in public prediction-quality refresh with a new full-step search.

The frozen solver and historical checkpoint producers stay byte unchanged.
The execution policy refreshes inaccurate derivatives, resets their numerical
search seed, and retries rejected exact-source Schur directions with the
existing LM rejection rule. Source costs, physical guards and acceptance stay
public.
"""
from copy import deepcopy
from types import MethodType
import ast
import inspect
import textwrap

from .official_static_surface_dataset import reference
import __future__


def refreshed_fraction(model, fraction, adaptive):
    """A rejected derivative prediction must not retain its tiny step habit."""
    if not adaptive or not model.force_fresh:
        return fraction
    model.current_report['fresh_J_backtracking_reset'] = dict(
        previous_fraction=float(fraction), next_fraction=1.,
        reason='The accepted source prediction requested an exact current J; search its new direction from the full step',
        next_true_source_cost_and_all_original_guards_required=True)
    return 1.


def compile_quality_solver(original, *, retry_rejected_schur=False):
    """Add the numerical seed reset and optional exact-source LM rejection retry."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    original_ast = ast.dump(tree, include_attributes=False)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) or node.test.id != 'accepted':
            continue
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and ast.unparse(node.body[0].value.func) == 'derivative_model.accepted'):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError('Expected the unique original accepted derivative-model update')
    node = matches[0]
    if len(node.body) != 1:
        raise ValueError('Original accepted derivative-model body changed')
    added = ast.parse('next_line_search_fraction = _prediction_refresh_fraction(derivative_model, next_line_search_fraction, adaptive_source_backtracking)').body[0]
    node.body.append(added)
    node.body.pop()
    if ast.dump(tree, include_attributes=False) != original_ast:
        raise ValueError('Unexpected solver AST change outside the numerical seed reset')
    node.body.append(added)
    if retry_rejected_schur:
        from calibration.solver_runtime.schur_rejection_retry import augment_rejected_schur_retry
        tree = augment_rejected_schur_retry(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    if retry_rejected_schur:
        from calibration.solver_runtime.schur_rejection_retry import rejected_direction_damping
        namespace['_rejected_direction_damping'] = rejected_direction_damping
    namespace['_prediction_refresh_fraction'] = refreshed_fraction
    flags = original.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(tree, __file__ + '::public_solver_prediction_refresh', 'exec', flags=flags), namespace)
    result = namespace[original.__name__]
    result.__module__ = __name__
    return result


def install_prediction_refresh(owner, *, driver, entry_reference=None, minimum_prediction_ratio=.25,
                               reset_initial_fresh_source_fraction=False, retry_rejected_schur=True):
    """Install after the ordinary source/J wrappers, before solver startup."""
    ratio = float(minimum_prediction_ratio)
    if not 0. < ratio <= 1.:
        raise ValueError('Explicit finite positive source-prediction agreement ratio required')
    if hasattr(owner, 'prediction_refresh_receipt'):
        raise ValueError('Prediction-quality execution policy already installed')
    original_solver = driver.solve_joint_source
    if original_solver is not driver.solver_runtime.solve_joint_source:
        raise ValueError('Expected the original public solver entry before opt-in')
    refs = [reference(__file__), reference(inspect.getsourcefile(original_solver))]
    if retry_rejected_schur:
        from calibration.solver_runtime import schur_rejection_retry
        refs.append(reference(schur_rejection_retry.__file__))
    if entry_reference is not None:
        refs.append(deepcopy(entry_reference))
    receipt = dict(schema='rinkspace.prediction-quality-source-J-refresh.v1',
        producers=refs, minimum_actual_to_predicted_reduction=ratio,
        original_public_refresh_ratio_option_used=True,
        accepted_refresh_resets_next_line_search_fraction_to_one=True,
        old_checkpoint_q_aux_damping_and_fraction_loaded_unchanged=True,
        new_source_initial_fresh_J_search_seed_reset_requested=bool(reset_initial_fresh_source_fraction),
        saved_canonical_and_damping_modified=False,
        subsequent_reset_only_after_an_accepted_trial_requests_fresh_J=True,
        original_source_values_rows_noise_Huber_profiles_guards_acceptance_unchanged=True,
        no_iteration_time_or_J_age_cap=True, frozen_producer_files_modified=False,
        rejected_exact_Schur_direction_retries_existing_LM=bool(retry_rejected_schur),
        rejected_LM_reuses_same_q_r_J_H_profiles_and_original_acceptance_precision=True)
    previous_kwargs, previous_identity = owner.solver_kwargs, owner.checkpoint_identity
    policy = deepcopy(owner.source_jacobian_reuse_policy)
    policy.update(existing_default_prediction_refresh_ratio_unchanged=False,
        selected_prediction_refresh_ratio=ratio, prediction_quality_execution=deepcopy(receipt))
    for ref in refs:
        if ref not in owner.source_producers:
            owner.source_producers.append(deepcopy(ref))
        if ref not in owner.recovery_references:
            owner.recovery_references.append(deepcopy(ref))

    def check():
        for ref in refs:
            if reference(ref['path']) != ref:
                raise ValueError('Frozen prediction-refresh producer changed')

    def identity(adapter):
        result = previous_identity()
        result['source_J_execution_policy'] = deepcopy(policy)
        result['prediction_quality_source_J_refresh'] = deepcopy(receipt)
        return result

    def kwargs(adapter):
        check()
        result = previous_kwargs()
        if not result.get('reuse_source_jacobian') or not result.get('adaptive_source_backtracking'):
            raise ValueError('Quality refresh requires existing modified Newton and adaptive source backtracking')
        if reset_initial_fresh_source_fraction:
            if getattr(owner, '_initial_complete_J_reuse', None) is not False:
                raise ValueError('An initial fresh current-source J is required for the new-source search seed reset')
            result['checkpoint_identity']['fresh_source_initial_search_seed_reset'] = dict(
                saved_fraction=float(result['initial_line_search_fraction']), execution_fraction=1.,
                q_aux_and_damping_unchanged=True, saved_canonical_modified=False,
                numerical_trial_search_heuristic_only=True,
                exact_new_source_J_and_complete_true_cost_and_all_guards_required=True)
            result['initial_line_search_fraction'] = 1.
        result['source_jacobian_refresh_ratio'] = ratio
        result['checkpoint_identity']['source_J_execution_policy'] = deepcopy(policy)
        result['checkpoint_identity']['prediction_quality_source_J_refresh'] = deepcopy(receipt)
        return result

    driver.solve_joint_source = compile_quality_solver(original_solver, retry_rejected_schur=retry_rejected_schur)
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.solver_kwargs = MethodType(kwargs, owner)
    owner.source_jacobian_reuse_policy = policy
    owner.prediction_refresh_receipt = deepcopy(receipt)
    return deepcopy(receipt)
