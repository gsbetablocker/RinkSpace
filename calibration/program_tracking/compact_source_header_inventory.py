"""Stream minimal row identities from immutable shared source JSON in NPZ.

This read-only inventory does not decode or qualify source proofs and cannot
authorize FIT. Original row numbers point back to the complete source pack.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .compact_source_json import ENCODING, REFERENCE_KEY


class _JSONStream:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.decoder = json.JSONDecoder()

    def _read(self):
        text = self.stream.read(1 << 16)
        if not text:
            raise ValueError('Incomplete compact source JSON')
        self.buffer += text

    def whitespace(self):
        while True:
            self.buffer = self.buffer.lstrip()
            if self.buffer:
                return
            self._read()

    def take(self, character):
        self.whitespace()
        if not self.buffer.startswith(character):
            raise ValueError('Unexpected source JSON structural character')
        self.buffer = self.buffer[len(character):]

    def value(self):
        self.whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                self._read()
                continue
            # A number at the read boundary may be incomplete even when it
            # forms a valid shorter JSON number. Require its following syntax.
            if end == len(self.buffer):
                self._read()
                continue
            self.buffer = self.buffer[end:]
            return value


_ROW_FIELDS = ('shot_id', 'actual_pts', 'native_pts', 'scope', 'physical_material_id', 'feature_id',
    'model_target', 'observation_semantics', 'split', 'point_count',
    'decoded_BGR_sha256', 'curve_observation_kind', 'point_correspondence',
    'must_preserve_upper_branch', 'scanline_branch', 'extent_side',
    'source_query_indices', 'anchor_point_indices', 'source_visible_paths',
    'source_neighbour_adjacency_confirmed', 'final_current_RGB_pixel_precision_checked',
    'scanline_fixed_axis', 'clip_boundary_x', 'clip_blur_allowance_px',
    'source_working_support_reconstructed_from_saved_actual_readout',
    'family', 'vision_training_role')


def read_minimal_compact_source_headers(path):
    """Return original ordered row identities with bounded JSON parsing heap."""
    slots = []
    with zipfile.ZipFile(Path(path)) as archive:
        with archive.open('rows_json.npy') as binary:
            version = np.lib.format.read_magic(binary)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(binary)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(binary)
            else:
                raise ValueError('Supported numeric NPY source-header version required')
            if len(shape) != 1 or fortran or dtype != np.dtype('uint8'):
                raise ValueError('Shared UTF-8 source header format required')
            parser = _JSONStream(io.TextIOWrapper(binary, encoding='utf8'))
            parser.take('{')
            if parser.value() != 'encoding':
                raise ValueError('Source JSON envelope field order differs')
            parser.take(':')
            if parser.value() != ENCODING:
                raise ValueError('Unknown source JSON evidence encoding')
            parser.take(',')
            if parser.value() != 'values':
                raise ValueError('Shared source evidence table required')
            parser.take(':')
            parser.take('[')
            def resolve(value):
                if isinstance(value, dict) and set(value)=={REFERENCE_KEY}:
                    index = value[REFERENCE_KEY]
                    if type(index) is not int or not 0 <= index < len(slots):
                        raise ValueError('Invalid or forward compact source reference')
                    return slots[index]
                return value
            while True:
                parser.whitespace()
                if parser.buffer.startswith(']'):
                    parser.take(']')
                    break
                value = parser.value()
                kept = None
                if isinstance(value, dict):
                    if all(k in value for k in ('physical_material_id', 'model_target', 'scope', 'split', 'actual_pts')):
                        kept = {k:resolve(value[k]) for k in _ROW_FIELDS if k in value}
                        kept['source_operator_array_fields_present'] = [k for k in
                            ('normal_interval', 'point_support_bounds_xy_px', 'source_tangent_interval_px') if k in value]
                        if not isinstance(kept.get('model_target'), dict):
                            raise ValueError('Model target was not retained as an exact minimal descriptor')
                        kept['__ordered_source_row_header__'] = True
                    elif ('curve_id' in value or 'advertisement_number' in value or 'kind' in value):
                        # Model targets consist of scalar identity descriptors;
                        # preserve every exact field rather than simplify them.
                        trial = {k:resolve(v) for k,v in value.items()}
                        if all(type(v) in (str, int, float, bool, type(None))
                               or isinstance(v, list) and all(type(item) is str for item in v)
                               for v in trial.values()):
                            kept = trial
                elif isinstance(value, list):
                    if not value:
                        kept = []
                    elif all(type(v) is int for v in value):
                        kept = np.asarray(value, np.int64)
                    elif all(type(v) is str for v in value):
                        # Ordered named material roles are a model descriptor,
                        # not a large numeric observation or proof payload.
                        kept = list(value)
                    else:
                        children = [resolve(v) for v in value]
                        if all(isinstance(v,dict) and v.get('__ordered_source_row_header__') is True for v in children):
                            kept = children
                        elif all(isinstance(v,np.ndarray) and v.dtype==np.int64 for v in children):
                            kept = children
                slots.append(kept)
                parser.whitespace()
                if parser.buffer.startswith(','):
                    parser.take(',')
                elif not parser.buffer.startswith(']'):
                    raise ValueError('Unexpected shared source table separator')
            parser.take(',')
            if parser.value() != 'root':
                raise ValueError('Compact source root required')
            parser.take(':')
            rows = resolve(parser.value())
            parser.take('}')
            if not isinstance(rows,list) or not all(isinstance(r,dict) and r.get('__ordered_source_row_header__') is True for r in rows):
                raise ValueError('Original ordered source row table required')
    with np.load(path, allow_pickle=False) as values:
        offsets = values['row_offsets'].copy()
    if len(offsets) != len(rows)+1:
        raise ValueError('Minimal source rows and actual point offsets disagree')
    for i, row in enumerate(rows):
        row.pop('__ordered_source_row_header__')
        for name in ('source_query_indices', 'anchor_point_indices', 'source_visible_paths'):
            if name in row:
                value = row[name]
                if isinstance(value,np.ndarray):
                    row[name] = value.tolist()
                elif isinstance(value,list):
                    row[name] = [v.tolist() if isinstance(v,np.ndarray) else v for v in value]
        n = int(offsets[i+1]-offsets[i])
        if row.get('point_count') != n:
            raise ValueError('Source row count differs from immutable actual arrays')
        row['original_source_row_index'] = i
    return rows, offsets
