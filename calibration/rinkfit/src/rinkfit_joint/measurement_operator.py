"""Versioned, language-independent identity of a frozen source operator.

Numbers use their IEEE-754 binary64 bits, not a language's JSON spelling.
station/row/alpha (and captured path) are solver auxiliaries: their legal
continuation is governed by finite domains, never by reissuing source truth.
"""
import hashlib
import math
import struct

SCHEMA = 'rinkfit.measurement_operator_binding.v1'
MEASUREMENT_ARRAYS = ('uv', 'native_pts', 'shot_index', 'kind', 'material_index',
                      'admission', 'fit_role', 'profile_normal', 'query_uv',
                      'alpha_bounds', 'stroke_pair_index', 'stroke_source_side')

def canonical_tree_sha256(value):
    """Hash typed JSON trees with an identical C++ implementation.

    Each node hashes its type tag and payload; containers contain child hex
    digests. Object keys are sorted in Unicode/UTF-8 order. Finite JSON numbers
    are binary64 values; integers outside exact binary64 range are rejected.
    """
    if value is None:
        data = b'n'
    elif isinstance(value, bool):
        data = b'b1' if value else b'b0'
    elif isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number) or (isinstance(value, int) and int(number) != value):
            raise ValueError('Operator identity requires finite exact binary64 numbers')
        data = b'd' + struct.pack('>d', number).hex().encode('ascii')
    elif isinstance(value, str):
        encoded = value.encode('utf8')
        data = b's' + str(len(encoded)).encode('ascii') + b':' + encoded
    elif isinstance(value, (list, tuple)):
        data = b'a' + str(len(value)).encode('ascii') + b':' + ''.join(
            canonical_tree_sha256(v) for v in value).encode('ascii')
    elif isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('Operator identity requires string object keys')
        data = b'o' + str(len(value)).encode('ascii') + b':' + ''.join(
            canonical_tree_sha256(k) + canonical_tree_sha256(value[k])
            for k in sorted(value)).encode('ascii')
    else:
        raise ValueError('Unsupported operator identity JSON node')
    return hashlib.sha256(data).hexdigest()

def dependency_identity(value):
    """Moving exact SHA-bound dependencies changes location, not measurement."""
    if isinstance(value, dict):
        return {k: dependency_identity(v) for k, v in value.items()
                if not ('sha256' in value and k in ('file', 'path'))}
    if isinstance(value, list):
        return [dependency_identity(v) for v in value]
    return value

def measurement_operator_descriptor(packet):
    queries = []
    seen = set()
    n = int(packet['row_count'])
    for q in packet.get('queries', []):
        i = int(q['row_index'])
        if i < 0 or i >= n or i in seen:
            raise ValueError('Duplicate or out-of-range measurement Query')
        seen.add(i)
        queries.append({k: v for k, v in q.items() if k != 'alpha0'})
    arrays = packet.get('arrays', {})
    prefix = int(packet.get('dense_source_binding', {}).get('row_start', n))
    def registry(name):
        values = packet.get(name)
        return values[:prefix] if values is not None else None
    return dict(schema=SCHEMA, row_count=n,
        arrays={k: dependency_identity(arrays.get(k)) for k in MEASUREMENT_ARRAYS},
        source_row_ids=registry('source_row_ids'),
        source_dataset_ids=registry('source_dataset_ids'),
        source_family_ids=registry('source_family_ids'),
        dense_source_binding=dependency_identity(packet.get('dense_source_binding')),
        queries=queries, allowed_paths=packet.get('allowed_paths', {}),
        source_station_domains=dependency_identity(packet.get('source_station_domains')),
        material_binding=dependency_identity(packet.get('material_binding')),
        stroke_pairs=dependency_identity(packet.get('stroke_pairs')))

def measurement_operator_binding(packet):
    return dict(schema=SCHEMA, sha256=canonical_tree_sha256(measurement_operator_descriptor(packet)))

def assert_measurement_operator_binding(packet, certificate):
    binding = certificate.get('measurement_operator_binding')
    if not binding or binding.get('schema') != SCHEMA:
        raise ValueError('Legacy source qualification has no frozen measurement-operator binding; explicit source measurement revision is required before new fitting')
    if binding != measurement_operator_binding(packet):
        raise ValueError('Source qualification measurement operator differs: Query/window/normal/source/material/finite path domain')


def assert_append_operator_prefix(parent, result, count):
    """An append may add measurements, never recapture or widen old authority."""
    for name in ('source_row_ids', 'source_dataset_ids'):
        if result[name][:count] != parent[name]:
            raise ValueError('Append changes inherited source order: ' + name)
    old_queries = [{k: v for k, v in q.items() if k != 'alpha0'} for q in parent.get('queries', [])]
    new_queries = [{k: v for k, v in q.items() if k != 'alpha0'} for q in result.get('queries', []) if int(q['row_index']) < count]
    if canonical_tree_sha256(old_queries) != canonical_tree_sha256(new_queries):
        raise ValueError('Append changes inherited Query operator')
    old = parent.get('allowed_paths', {})
    current = {k: v for k, v in result.get('allowed_paths', {}).items() if int(k) < count}
    if old != current:
        raise ValueError('Append drops or widens inherited finite path authority')
    for packet in (parent, result):
        for key, paths in packet.get('allowed_paths', {}).items():
            if int(key) < 0 or int(key) >= int(packet['row_count']) or not paths or len(set(paths)) != len(paths) or any(type(p) is not int or p < 0 for p in paths):
                raise ValueError('Invalid explicit finite path authority')
    a, b = parent.get('source_station_domains'), result.get('source_station_domains')
    if a is None and any(int(q['row_index']) < count for q in (b or {}).get('rows', [])):
        raise ValueError('Append creates a new inherited finite station source domain')
    if a is not None:
        rows = [q for q in (b or {}).get('rows', []) if int(q['row_index']) < count]
        if b is None or a.get('schema') != b.get('schema') or a.get('rows') != rows or any(ref not in b.get('source_refs', []) for ref in a.get('source_refs', [])):
            raise ValueError('Append changes inherited finite station source domain')
