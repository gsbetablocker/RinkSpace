"""Linear-time indexing of the original authenticated material records.

No source record, geometry calculation, derivative stencil or acceptance rule
changes. The original role function keeps its identity for captured aliases.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


_RECEIPT = None


def _first_index(records):
    # The original next(...) returns the FIRST equal record. Keep that rule
    # even before the original duplicate-record rejection at the end.
    if not all(type(r) is dict and type(r.get('original_global_bundle_index')) is int
               for r in records):
        return None
    result = {}
    for row in records:
        result.setdefault(row['original_global_bundle_index'], row)
    return result


def _patch(function, mode):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    changes = []
    class Index(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            target = ast.unparse(node.targets[0])
            if mode == 'finite' and target == 'old':
                if ast.unparse(node.value) != '[i for i in range(len(records)) if i not in revised]':
                    raise ValueError('Original ordered role complement changed')
                changes.append('linear_role_complement')
                return ast.parse('_revised_set = set(revised)\nold = [i for i in range(len(records)) if i not in _revised_set]').body
            if mode == 'qualification' and target == 'records':
                if ast.unparse(node.value) != "deepcopy(packet['admitted_source_only_rows'])":
                    raise ValueError('Original qualified records changed')
                changes.append('first_record_index')
                return [node, ast.parse('_first_records = _rinkspace_first_source_record_index(records)').body[0]]
            if mode == 'qualification' and target == 'peer':
                if ast.unparse(node.value) != "next((r for r in records if r['original_global_bundle_index'] == representative), None)":
                    raise ValueError('Original duplicate keeper search changed')
                changes.append('indexed_first_duplicate_keeper')
                node.value = ast.IfExp(test=ast.parse('_first_records is not None', mode='eval').body,
                    body=ast.parse('_first_records.get(representative)', mode='eval').body, orelse=node.value)
            return node

    tree = Index().visit(tree)
    expected = ['linear_role_complement'] if mode == 'finite' else ['first_record_index', 'indexed_first_duplicate_keeper']
    if changes != expected or function.__code__.co_freevars:
        raise ValueError('Original closure-free role indexing sites changed: ' + str(changes))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    function.__globals__['_rinkspace_first_source_record_index'] = _first_index
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(function), 'exec'), function.__globals__, namespace)
    function.__code__ = namespace[function.__name__].__code__
    return dict(function=function.__name__, original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                changes=changes)


def install_role_record_index_runtime(*, report_directory=None):
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from structured_programme_source_candidate import original_motion_far_board_roles as roles
    changes = [_patch(roles._role_finite_board_records, 'finite'),
               _patch(roles._programme_review_records, 'qualification')]
    path = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.original-material-role-linear-index.v1',
        helper_path=str(path), helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_functions=changes, all_original_record_validation_retained=True,
        first_duplicate_keeper_and_original_record_order_retained=True,
        original_geometry_numerics_and_stencils_unchanged=True, cross_call_cache=False,
        registered_source_files_modified=False, accuracy_accepted=False)
    if report_directory is not None:
        destination = Path(report_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'role_record_index_receipt.json').write_text(json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT
