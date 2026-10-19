"""Private numeric scene for the audited complete CurrentWideSource consumer.

The original wide factor, qualification, sampling and finite differences run
unchanged. Its internal course scene does not export panels or meshes, and the
audited curve/top readers do not consume those fields. Public course.scene and
all projection functions (including their __globals__) remain untouched.
"""
from __future__ import annotations

import ast
import atexit
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


def _copy_numeric_scene(base):
    # Copy each branch the ORIGINAL scene body writes; static fields are only
    # read by the admitted qualification/curve chain, never returned publicly.
    result = dict(base)
    parameters = result['parameters'] = dict(base['parameters'])
    parameters['corner_curve'] = dict(base['parameters']['corner_curve'])
    result['ice_boundary'] = dict(base['ice_boundary'])
    result['board_surface'] = dict(base['board_surface'])
    if 'shared_geometry_checks' in base:
        checks = result['shared_geometry_checks'] = dict(base['shared_geometry_checks'])
        for key in ('checks', 'unknown_check_reasons'):
            if key in checks:
                checks[key] = dict(checks[key])
    return result


def _clone_scene(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.decorator_list = []
    copied = panel_loops = mesh_loops = perimeter = cumulative = 0
    statements = []
    for statement in function.body:
        rendered = ast.unparse(statement)
        if rendered == 'result = deepcopy(self.base)':
            statement.value.func = ast.Name('_rinkspace_copy_numeric_scene', ast.Load())
            copied += 1
        elif isinstance(statement, ast.For) and ast.unparse(statement.iter) == "enumerate(surf['panels'])":
            panel_loops += 1
            continue
        elif isinstance(statement, ast.For) and ast.unparse(statement.iter) == "result['meshes']":
            mesh_loops += 1
            continue
        elif rendered == "surf['perimeter_length'] = cum":
            perimeter += 1
            continue
        elif rendered == 'cum = 0.0':
            cumulative += 1
            continue
        statements.append(statement)
    if (copied, panel_loops, mesh_loops, perimeter, cumulative) != (1, 1, 1, 1, 1):
        raise RuntimeError('Original course scene structure changed; private wide scene not installed')
    function.body = statements
    namespace = dict(original.__globals__)
    namespace['_rinkspace_copy_numeric_scene'] = _copy_numeric_scene
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + ':wide_numeric_scene', 'exec'), namespace)
    return namespace[function.name]


def _audited_getter_chain(geometry, allowed, terminal):
    seen = set()
    while id(geometry) not in seen:
        seen.add(id(geometry))
        getter = getattr(geometry, 'one_scene_source_world_bundle', None)
        function = getattr(getter, '__func__', None)
        if function not in allowed:
            return False
        if function in terminal:
            return True
        geometry = getattr(geometry, 'parent', None)
        if geometry is None:
            return False
    return False


def _identity(value):
    function = getattr(value, '__func__', value)
    return (getattr(function, '__module__', type(function).__module__) + '.'
            + getattr(function, '__qualname__', type(function).__qualname__))


def _consumer_identity(factor, course):
    return dict(factor_type=_identity(type(factor)), geometry_type=_identity(type(factor.geometry)),
                course_type=_identity(type(course)), course_scene=_identity(course.scene),
                venue_qualifier=_identity(factor.qualified_scene),
                current_scene_reader=_identity(getattr(factor.geometry, 'one_scene_source_world_bundle', None)))


def _clone_curve_worlds(original, numeric_scene):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.decorator_list = []
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and ast.unparse(node) == 'course.scene(cq, rebuild_grid=False)']
    if len(calls) != 1:
        raise RuntimeError('Original complete wide scene call changed')
    call = calls[0]
    call.func = ast.Name('_rinkspace_wide_numeric_scene', ast.Load())
    call.args = [ast.Name('self', ast.Load()), ast.Name('course', ast.Load()), ast.Name('cq', ast.Load())]
    call.keywords = []
    namespace = dict(original.__globals__)
    namespace['_rinkspace_wide_numeric_scene'] = numeric_scene
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + ':wide_numeric_scene', 'exec'), namespace)
    return namespace[function.name]


def install_wide_source_scene_runtime(driver=None, *, report_directory=None):
    """Install one audited _curve_worlds call adaptation; unknown paths stay original.

    No cache of q, qualification PASS, residual or Jacobian is introduced.
    Counters/identity are returned in the receipt and written once at ordinary
    process exit when report_directory is supplied. There is no per-world IO.
    """
    from . import install
    install()
    from structured_programme_source_candidate.current_wide_source import CurrentWideSource
    from structured_programme_source_candidate.paired_location_programme_geometry import PairedLocationProgrammeGeometry
    from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry
    from simple_wide_perspective_reset_candidate.differentiable_material_course_domain import DifferentiableBoardCourseGeometry
    from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry
    from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry import SharedUnmeasuredAdvertisementGeometry
    from complete_ice_template_candidate.full_free_boundary_geometry import (
        CurrentSharedGeometryWithFullFreeBoundary, StructuredSharedIceTemplateFullBoundaryGeometry,
    )
    previous = getattr(CurrentWideSource, '_rinkspace_wide_numeric_scene_receipt', None)
    if previous is not None:
        return previous
    original = CurrentWideSource._curve_worlds
    original_scene = SharedBoardCourseNumericGeometry.scene
    private_scene = _clone_scene(original_scene)
    terminal = {CurrentSharedGeometryWithFullFreeBoundary.one_scene_source_world_bundle,
                StructuredSharedIceTemplateFullBoundaryGeometry.one_scene_source_world_bundle}
    allowed = terminal | {SharedUnmeasuredAdvertisementGeometry.one_scene_source_world_bundle}
    receipt = dict(
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_curve_worlds_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_scene_sha256=hashlib.sha256(inspect.getsource(original_scene).encode()).hexdigest(),
        admitted_course_types=[_identity(SharedBoardCourseNumericGeometry), _identity(DifferentiableBoardCourseGeometry)],
        differentiable_course_original_z_sha256=hashlib.sha256(
            inspect.getsource(DifferentiableBoardCourseGeometry._z).encode()).hexdigest(),
        source_qualification_validators_curves_ad_corners_and_FD_stencils_retained=True,
        public_scene_and_projection_function_globals_untouched=True,
        omitted_internal_only=['complete_scene_deepcopy', 'panel_rebuild', 'perimeter_length_rebuild', 'mesh_rebuild'],
        numerical_or_qualification_cache=False, optimized_scene_calls=0, original_scene_calls=0,
        fallback_reasons={}, first_fallback_by_reason={}, first_admitted_consumer=None,
        accuracy_accepted=False,
    )

    def numeric_scene(factor, course, q):
        # QualifiedStructuredFullBoundaryGeometry explicitly installs this
        # registered differentiable course subclass. It inherits the EXACT
        # scene body; its _z delegates original contact/top perimeter policy.
        reason = None
        if type(factor) is not CurrentWideSource:
            reason = 'factor_type_not_audited'
        elif type(factor.geometry) is not PairedLocationsSharedSizesGeometry:
            reason = 'geometry_type_not_audited'
        elif type(course) not in (SharedBoardCourseNumericGeometry, DifferentiableBoardCourseGeometry):
            reason = 'course_type_not_audited'
        elif getattr(course.scene, '__func__', None) is not original_scene:
            reason = 'course_scene_function_changed'
        elif factor.qualified_scene is not PairedLocationProgrammeGeometry.qualified_scene:
            reason = 'venue_qualifier_changed'
        elif not _audited_getter_chain(factor.geometry, allowed, terminal):
            reason = 'current_scene_reader_chain_not_audited'
        elif not all(isinstance(row, dict) and isinstance(row.get('target'), dict)
                     and row['target'].get('kind') in (
                         'board_side', 'advertisement_corner', 'circle_outline', 'line_segment',
                         'polygon_supporting_line', 'polygon_outline') for row in factor.observations):
            reason = 'observation_target_kind_not_audited'
        if reason is not None:
            receipt['original_scene_calls'] += 1
            receipt['fallback_reasons'][reason] = receipt['fallback_reasons'].get(reason, 0) + 1
            if reason not in receipt['first_fallback_by_reason']:
                receipt['first_fallback_by_reason'][reason] = _consumer_identity(factor, course)
            return course.scene(q, rebuild_grid=False)
        receipt['optimized_scene_calls'] += 1
        if receipt['first_admitted_consumer'] is None:
            receipt['first_admitted_consumer'] = _consumer_identity(factor, course)
        return private_scene(course, q, rebuild_grid=False)

    adapted = _clone_curve_worlds(original, numeric_scene)

    @functools.wraps(original)
    def curve_worlds(self, world):
        return adapted(self, world)

    CurrentWideSource._curve_worlds = curve_worlds
    CurrentWideSource._rinkspace_wide_numeric_scene_receipt = receipt
    if report_directory is not None:
        destination = Path(report_directory)

        def write_summary():
            try:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'wide_numeric_source_scene_summary.json').write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            except (OSError, TypeError, ValueError) as error:
                # Diagnostic IO never replaces the original source/math error.
                receipt['summary_io_error'] = type(error).__name__ + ': ' + str(error)

        atexit.register(write_summary)
    return receipt
