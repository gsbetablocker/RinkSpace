"""Safe selected original source rows for exact native-camera diagnostics.

Only authenticated JSON and non-object NPZ arrays are used. The stored rows
retain the original native profile query/branch semantics and W membership;
loading this packet does not construct a whole SourceBundle or evaluate a fit.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .first_shot_source_transition import reference, read

SCHEMA = 'rinkspace.safe-selected-native-source-frames.v1'


def _encode(value, arrays):
    from .source_evidence_whitening import SourceWhiteningOperator
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are forbidden in safe source export')
        key = 'array_' + str(len(arrays))
        arrays[key] = value.copy()
        return {'_safe_npz_array': key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, SourceWhiteningOperator):
        return {'_safe_whitening_operator': _encode(value.to_record(), arrays)}
    if isinstance(value, tuple):
        return {'_safe_tuple': [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError('Safe source metadata requires string dictionary keys')
        return {key: _encode(item, arrays) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported safe source metadata type: ' + type(value).__name__)


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'_safe_npz_array'}:
            return arrays[value['_safe_npz_array']].copy()
        if set(value) == {'_safe_tuple'}:
            return tuple(_decode(item, arrays) for item in value['_safe_tuple'])
        if set(value) == {'_safe_whitening_operator'}:
            from .source_evidence_whitening import SourceWhiteningOperator
            return SourceWhiteningOperator.from_record(_decode(value['_safe_whitening_operator'], arrays))
        return {key: _decode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


def export_source_frames(owner, *, output_directory, frame_indices, q=None):
    from .first_shot_fit import _CameraState
    from .first_shot_row_source import row_timed_raw
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'frames.json').exists() or (output / 'arrays.npz').exists():
        raise ValueError('Selected source export already exists')
    requested = tuple(int(index) for index in frame_indices)
    if len(set(requested)) != len(requested):
        raise ValueError('Distinct explicit programme frames required')
    q = owner.start.copy() if q is None else np.asarray(q, dtype=np.float64).copy()
    if q.shape != owner.start.shape or not np.isfinite(q).all():
        raise ValueError('A finite complete current state is required for source export')
    worlds, _ = owner.world(q, False)
    arrays, frames = {}, []
    for index in requested:
        matches = [i for i, frame in enumerate(owner.bundle.frames) if frame['index'] == index]
        if len(matches) != 1:
            raise ValueError('Selected frame is missing or duplicated')
        position = matches[0]
        frame = owner.bundle.frames[position]
        local_worlds = {row['_target_key']: worlds[row['_target_key']].copy() for row in frame['rows']}
        raw = row_timed_raw(owner.state, q, frame, local_worlds)
        whitening = owner.bundle.whitening[position]
        first, stop = owner.formal_offsets[position:position+2]
        frames.append(dict(frame=frame, worlds=local_worlds, raw_pixel_residual=raw,
            whitened_residual=whitening.apply(raw), formal_scales=owner.scales[first:stop].copy(),
            formal_quadratic=owner.quadratic[first:stop].copy(),
            global_formal_slice=[int(first), int(stop)], source_ledger=owner.bundle.ledgers[position]))
    # This is the exact geometry-free input used by the existing source worker.
    camera = _CameraState(owner.bundle.state).__getstate__()
    payload = dict(q=q, frames=frames, camera_state=camera,
        model_reference=owner.model_reference, binding=owner.bundle.state.binding_reference,
        association=owner.association_reference, producers=owner.source_producers,
        world_parameter_indices=owner.state.world_parameter_indices.copy(),
        complete_state_arrays=owner.checkpoint_arrays(), checkpoint_identity=owner.checkpoint_identity())
    if hasattr(owner.state, 'scene'):
        payload['scene'] = owner.state.scene(q)
    encoded = _encode(payload, arrays)
    np.savez_compressed(output / 'arrays.npz', **arrays)
    receipt = dict(schema=SCHEMA, arrays=reference(output / 'arrays.npz'), payload=encoded,
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(), frame_indices=list(requested),
        source_rows_preserve_all_native_queries_and_original_branch_semantics=True,
        correlated_W_matrix_membership_and_formal_order_preserved=True,
        no_source_bundle_reconstruction_required=True, pickle_deserialization=False,
        exact_raw_entry='first_shot_row_source.row_timed_raw',
        exact_whitened_entry='first_shot_row_source.evaluate_row_timed_frame',
        producer=reference(__file__))
    (output / 'frames.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf8')
    return dict(manifest=reference(output / 'frames.json'), arrays=receipt['arrays'],
        q_sha256=receipt['q_sha256'], frame_indices=list(requested))


def load_source_frames(path):
    receipt = read(reference(path))
    if receipt.get('schema') != SCHEMA or receipt.get('pickle_deserialization') is not False:
        raise ValueError('Safe selected native source packet required')
    ref = receipt['arrays']
    if reference(ref['path']) != ref:
        raise ValueError('Selected source arrays changed')
    with np.load(ref['path'], allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    payload = _decode(receipt['payload'], arrays)
    if hashlib.sha256(payload['q'].tobytes()).hexdigest() != receipt['q_sha256']:
        raise ValueError('Selected source parameter identity changed')
    return payload, receipt
