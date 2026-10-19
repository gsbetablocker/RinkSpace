"""Omit unused motion-role backreferences only from private absolute-FD owners.

The original snapshot producer still selects the identical registered source
materials, clears its original disposable caches and writes an immutable pickle.
The one AST insertion drops four instance fields after making its private dict;
it never modifies the live binder, source rows or material/geometry producers.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


PRIVATE_ONLY_UNUSED_FIELDS = (
    'finite_board_records',
    '_original_motion_far_board_role_binding',
    '_existing_finite_perimeter_binding',
    '_original_motion_far_board_records_method',
)


def _private_only_function(original):
    source_text = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source_text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError('The original private owner snapshot function is required')
    function = tree.body[0]
    matches = []
    for index, statement in enumerate(function.body):
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Attribute)
                and statement.targets[0].attr == '__dict__'
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id == 'private'
                and ast.dump(statement.value) == ast.dump(ast.parse('dict(vars(geometry))', mode='eval').body)):
            matches.append(index)
    if len(matches) != 1:
        raise TypeError('Original private dict copy must occur exactly once')
    # Literal, private-only dictionary operations; no supplied source text.
    inserted = [ast.Expr(ast.Call(
        func=ast.Attribute(ast.Attribute(ast.Name('private', ast.Load()), '__dict__', ast.Load()), 'pop', ast.Load()),
        args=[ast.Constant(field), ast.Constant(None)], keywords=[])) for field in PRIVATE_ONLY_UNUSED_FIELDS]
    function.body[matches[0] + 1:matches[0] + 1] = inserted
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    return namespace[function.name], source_text


def install_private_material_snapshot_runtime(*, report_directory=None):
    """Install before driver.main creates its new immutable private snapshot."""
    from . import absolute_geometry_fd_pool_candidate as source
    report_destination = None if report_directory is None else Path(report_directory).resolve()
    previous = getattr(source, '_rinkspace_private_material_snapshot_runtime', None)
    if previous is not None:
        if previous['report_directory'] != (None if report_destination is None else str(report_destination)):
            raise RuntimeError('Private owner snapshot helper already installed for another run')
        return previous
    original = source.freeze_private_material_owner
    if original.__module__ != source.__name__ or original.__name__ != 'freeze_private_material_owner':
        raise TypeError('Original source-owned private snapshot producer required')
    private_function, original_text = _private_only_function(original)
    helper = Path(__file__).resolve()
    receipt = dict(schema='rinkspace.private-material-owner-unused-role-backreferences.v1',
        original_freeze_private_material_owner_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        report_directory=None if report_destination is None else str(report_destination),
        removed_private_instance_fields=list(PRIVATE_ONLY_UNUSED_FIELDS),
        ast_change='Only four dict.pop calls immediately after original private.__dict__ copy',
        original_material_bindings_source_rows_certificates_and_numeric_producers_retained=True,
        source_ice_material_view_current_and_producer_retained=True,
        live_binder_and_existing_immutable_snapshots_modified=False,
        actual_same_q_all_32_source_J_byte_equivalence_pending=True,
        actual_fresh_J_speedup_pending=True, accuracy_accepted=False)
    if report_destination is not None:
        report_destination.mkdir(parents=True, exist_ok=True)

    def freeze_private_material_owner(geometry, groups, destination):
        existing = [field for field in PRIVATE_ONLY_UNUSED_FIELDS if field in vars(geometry)]
        result = private_function(geometry, groups, destination)
        # Original worker checks still use original path/SHA/serialized_bytes.
        result['private_snapshot_execution'] = dict(
            schema=receipt['schema'], helper_sha256=receipt['helper_sha256'],
            original_freeze_private_material_owner_sha256=receipt['original_freeze_private_material_owner_sha256'],
            removed_private_instance_fields=existing,
            unused_original_binder_partial_backreference_removed='finite_board_records' in existing,
            original_registered_material_count=result['registered_material_count'],
            original_source_row_count=result['source_row_count'],
            serialized_bytes=result['serialized_bytes'], live_owner_modified=False,
            original_numerical_producer_and_source_registration_retained=True,
            actual_same_q_all_32_source_J_byte_equivalence_pending=True)
        if report_destination is not None:
            (report_destination / 'actual_private_material_snapshot.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        return result

    # Match the original public call signature, including destination= callers.
    freeze_private_material_owner.__signature__ = inspect.signature(original)
    source._rinkspace_original_freeze_private_material_owner = original
    source.freeze_private_material_owner = freeze_private_material_owner
    source._rinkspace_private_material_snapshot_runtime = receipt
    if report_destination is not None:
        (report_destination / 'private_material_snapshot_runtime_receipt.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return receipt
