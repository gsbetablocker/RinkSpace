"""Select the actual source reader, without promoting historical admission.

Historical input bindings remain readable for audit. Fresh dense packet creation
requires a binding issued by the positive finite-instance factory. A typed
binding is always read by TypedDenseSourceFitInput, including during replay.
"""
from __future__ import annotations
from pathlib import Path
from .io_utils import read_json


def source_reader_kind(binding, *, purpose):
    if purpose not in ('new_packet', 'historical_audit'):
        raise ValueError('An explicit source-consumer purpose is required')
    if binding.get('schema') != 'rinkfit.dense_source_fit_input.v1':
        raise ValueError('Unsupported dense source input schema')
    typed = binding.get('positive_finite_instance_required') is True
    if typed:
        if not binding.get('instance_support_gate') or not binding.get('legacy_material_geometry_reader'):
            raise ValueError('Typed input is missing its pinned source-instance gate')
        if purpose=='new_packet' and not binding.get('frame_role_authority'):
            raise ValueError('Fresh dense packets require pinned native frame roles')
        return 'typed'
    if purpose == 'new_packet':
        raise ValueError('Fresh dense packets require positive finite source-instance evidence; compile the complete source ledger before building. Historical q7 remains available for audit only.')
    return 'legacy'


def open_dense_source_input(binding_path, *, purpose):
    path = Path(binding_path).resolve()
    kind = source_reader_kind(read_json(path), purpose=purpose)
    if kind == 'typed':
        from .dense_typed_source_fit import TypedDenseSourceFitInput
        reader = TypedDenseSourceFitInput(path, require_frame_roles=purpose=='new_packet')
        reader.reader_qualification_status = 'typed_dense_suffix_instance_gate_required'
        return reader
    from .dense_source_fit import DenseSourceFitInput
    reader = DenseSourceFitInput(path)
    reader.reader_qualification_status = 'legacy_qualification_not_full_positive'
    return reader
