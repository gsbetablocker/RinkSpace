"""Inventory exact current source operators before a complete graph migration.

The source pack and index remain immutable. Saved support arrays are recorded
without inferring permission to change the original likelihood. This is a
read-only routing inventory; it cannot open the solver startup gate.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


EXTRA_FIELDS = ('source_optical_normal_support_authorized', 'boundary_location_observation',
                'center_definition', 'normal_interval_operator_authorized',
                'source_support_observation_operator', 'working_sigma_status',
                'source_uncertainty_kind', 'source_integration_mass_rule')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def operator_headers(path):
    """Reuse the parser with a private scalar-field view, never global mutation."""
    reader = read_minimal_compact_source_headers
    namespace = dict(reader.__globals__)
    namespace['_ROW_FIELDS'] = (*namespace['_ROW_FIELDS'], *EXTRA_FIELDS)
    local_reader = FunctionType(reader.__code__, namespace, reader.__name__, reader.__defaults__,
                                reader.__closure__)
    return local_reader(path)


def operator_identity(header):
    return dict(target=header['model_target'], scope=header['scope'],
                observation_semantics=header['observation_semantics'],
                fields={name: header[name] for name in (
                    'curve_observation_kind', 'point_correspondence', 'must_preserve_upper_branch',
                    'scanline_branch', 'scanline_fixed_axis', 'extent_side', 'clip_boundary_x',
                    'clip_blur_allowance_px', *EXTRA_FIELDS) if name in header},
                saved_operator_array_fields=header['source_operator_array_fields_present'])


def inventory(dataset_reference):
    authenticate_source_reference(dataset_reference)
    dataset = json.loads(Path(dataset_reference['path']).read_text(encoding='utf8'))
    if dataset['schema'] != 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1':
        raise ValueError('Current immutable source dataset required')
    records, target_counts, operator_counts, totals = {}, Counter(), Counter(), Counter()
    for source_number, cut in enumerate(dataset['cuts']):
        ref = cut['source_pack']
        authenticate_source_reference(ref)
        authenticate_source_reference(cut['support_index'])
        rows, offsets = operator_headers(ref['path'])
        with np.load(cut['support_index']['path'], allow_pickle=False) as saved:
            if not np.array_equal(offsets, saved['row_offsets']):
                raise ValueError('Original source row boundaries changed')
            selected = {role: saved[role + '_point_indices'] for role in ('fit', 'held')}
            promoted = set(saved['promoted_held_rows'].tolist())
        amounts = Counter()
        for row_number, header in enumerate(rows):
            lo, hi = map(int, offsets[row_number:row_number + 2])
            for role, indices in selected.items():
                left, right = np.searchsorted(indices, [lo, hi])
                count = int(right - left)
                if not count:
                    continue
                effective_role = 'held' if row_number in promoted else header['split']
                if header['scope'] not in ('ice', 'board') or effective_role != role:
                    raise ValueError('Model-external or wrong-role source entered the native routing plan')
                declaration = operator_identity(header)
                key = json.dumps(dict(split=role, **declaration), sort_keys=True,
                                 separators=(',', ':'), allow_nan=False)
                record = records.setdefault(key, dict(split=role, **declaration,
                    finite_points=0, source_rows=0, actual_exposures=set(), sources=set(),
                    original_row_locators=[]))
                record['finite_points'] += count
                record['source_rows'] += 1
                record['actual_exposures'].add(header['actual_pts'])
                record['sources'].add(source_number)
                # Keep every original row locator, never an arbitrary row sample.
                record['original_row_locators'].append([source_number, row_number, count])
                amounts[role + '_points'] += count
                target_counts[role + ':' + header['model_target']['kind']] += count
                operator_counts[role + ':' + str(header.get('curve_observation_kind', '<absent>'))] += count
        for role in selected:
            field = role + '_points'
            if amounts[field] != cut[field]:
                raise ValueError('Native routing inventory lost admitted exact source points')
        totals.update(amounts)
    for field, count in totals.items():
        if dataset[field] != count:
            raise ValueError('Whole source operator inventory lost exact FIT/HELD points')
    result = []
    for key, record in sorted(records.items()):
        record['actual_exposures'] = sorted(record['actual_exposures'])
        record['sources'] = sorted(record['sources'])
        result.append(record)
    return dict(schema='rinkspace.strict-source-native-operator-routing-inventory.v1',
                dataset_reference=dataset_reference, counts=dict(totals),
                finite_points_by_target_kind=dict(sorted(target_counts.items())),
                finite_points_by_declared_curve_operator=dict(sorted(operator_counts.items())),
                original_source_packs=[c['source_pack'] for c in dataset['cuts']],
                records=result, support_arrays_not_reinterpreted_as_loss_permission=True,
                original_source_pixels_weights_paths_HELD_and_model_targets_unchanged=True,
                full_solver_graph_adapter_installed=False, solver_started=False,
                source_header_parser_reference=reference(
                    read_minimal_compact_source_headers.__globals__['__file__']),
                implementation_reference=reference(__file__))
