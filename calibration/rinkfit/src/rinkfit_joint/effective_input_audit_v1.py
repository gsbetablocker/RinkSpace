"""Independent effective-domain inspection; never issues a native qualification.

Canonical encoding rinkfit.tagged_json_sha256.v1 has these byte tags: N, T, F;
I + u64LE(ASCII length) + decimal integer; D + IEEE754 binary64 LE;
S + u64LE(UTF8 length) + UTF8; A/O + u64LE(item count) + children.
Object keys are UTF8-byte sorted and encoded as S; no Unicode normalization.
A reference object's file is replaced with 'sha256:'+its SHA only for the
portable digest. Its original bytes and address remain in the ancestry log.
This encoding must be implemented and cross-tested in native before use as ABI.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import time
import numpy as np

APPENDABLE = {'groups', 'queries', 'source_manifests', 'recapture_rows',
              'engineering_pixel_dependency_groups.groups'}
REVIEW_CHAINS = {'native_material_identity_revision': 'resolved_native_identity_revision_chain',
                 'native_print_identity_rebind': 'resolved_native_print_identity_revision_chain',
                 'native_source_measurement_revision': 'resolved_native_source_measurement_revision_chain'}
DTYPES = {'int64': '<i8', 'i64': '<i8', 'uint16': '<u2', 'u16': '<u2',
          'float64': '<f8', 'f64': '<f8', 'uint8': '|u1', 'u8': '|u1',
          'uint32': '<u4', 'u32': '<u4', 'int32': '<i4', 'i32': '<i4',
          'int8': '|i1', 'i8': '|i1'}


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=nonfinite)


def digest(value, portable=True):
    h = hashlib.sha256()
    def length(n):
        h.update(struct.pack('<Q', n))
    def visit(v):
        if v is None: h.update(b'N')
        elif isinstance(v, bool): h.update(b'T' if v else b'F')
        elif isinstance(v, int):
            b = str(v).encode('ascii'); h.update(b'I'); length(len(b)); h.update(b)
        elif isinstance(v, float):
            require(math.isfinite(v), 'Nonfinite canonical double')
            h.update(b'D'); h.update(struct.pack('<d', v))
        elif isinstance(v, str):
            b = v.encode('utf8'); h.update(b'S'); length(len(b)); h.update(b)
        elif isinstance(v, list):
            h.update(b'A'); length(len(v))
            for x in v: visit(x)
        elif isinstance(v, dict):
            h.update(b'O'); length(len(v))
            for key in sorted(v, key=lambda k: k.encode('utf8')):
                visit(key)
                visit('sha256:' + v['sha256'] if portable and key == 'file' and 'sha256' in v else v[key])
        else: raise ValueError('Unsupported canonical JSON type')
    visit(value)
    return h.hexdigest()


class CheckedBytes:
    """Only invocation-local deduplication; no pre-existing validation cache."""
    def __init__(self, bundle=None, bundle_only=False):
        self.bundle, self.bundle_only = bundle, bundle_only
        self.verified = {}

    def path(self, ref, base):
        p = Path(ref['file'])
        return p if p.is_absolute() else Path(base) / p

    def read(self, ref, base):
        p = self.path(ref, base)
        if not self.bundle_only and p.is_file():
            data = p.read_bytes(); location = 'filesystem'
        else:
            require(self.bundle is not None, 'Missing required bytes: ' + str(p))
            data = self.bundle.read(dict(file=str(p), sha256=ref['sha256'])); location = 'complete_zip'
        actual = hashlib.sha256(data).hexdigest()
        require(actual == ref['sha256'], 'SHA mismatch: ' + str(p))
        if 'byte_count' in ref:
            require(len(data) == ref['byte_count'], 'Byte count mismatch: ' + str(p))
        self.verified[(str(p), actual)] = dict(file=str(p), sha256=actual, byte_count=len(data), location=location)
        return data

    def json(self, ref, base):
        return strict_json(self.read(ref, base))

    def array(self, ref, base):
        require(ref.get('byte_order', 'little') == 'little' and ref.get('layout', 'C') == 'C', 'Array byte/layout')
        dtype = np.dtype(DTYPES.get(ref['dtype'], ref['dtype']))
        require(dtype.str in {'<f8', '<i8', '<u8', '<u4', '<i4', '<u2', '|u1', '|i1'}, 'Array dtype outside ABI')
        shape = ref['shape']
        require(shape and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape), 'Invalid array shape')
        n = math.prod(shape); offset = ref.get('offset_bytes', 0)
        require(isinstance(offset, int) and offset >= 0 and offset % dtype.itemsize == 0, 'Misaligned array offset')
        p = self.path(ref, base); key = (str(p), ref['sha256'])
        if not self.bundle_only and p.is_file():
            if key not in self.verified:
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
                require(h.hexdigest() == ref['sha256'], 'Array SHA mismatch: ' + str(p))
                self.verified[key] = dict(file=str(p), sha256=h.hexdigest(), byte_count=p.stat().st_size, location='filesystem')
            size = self.verified[key]['byte_count']
            data = np.memmap(p, mode='r', dtype=dtype, offset=offset, shape=tuple(shape))
        else:
            raw = self.read(ref, base); size = len(raw)
            require(offset + n * dtype.itemsize <= size, 'Array slice outside backing bytes')
            data = np.frombuffer(raw, dtype=dtype, count=n, offset=offset).reshape(shape)
        require(offset + n * dtype.itemsize <= size and ('offset_bytes' in ref or size == n * dtype.itemsize), 'Array exact byte count')
        require('byte_count' not in ref or ref['byte_count'] == size, 'Array declared byte count')
        slice_sha = hashlib.sha256(memoryview(np.asarray(data)).cast('B')).hexdigest()
        require('slice_sha256' not in ref or ref['slice_sha256'] == slice_sha, 'Array slice SHA mismatch')
        return data, dict(**self.verified[key], dtype=dtype.str, shape=shape, offset_bytes=offset,
                          slice_byte_count=n*dtype.itemsize, slice_sha256=slice_sha)


def resolve_packet(head, store):
    """Same key/append semantics as cpp/src/joint_packet.cpp, without expansion."""
    ancestry = []
    def resolve(ref, base, seen):
        p = store.path(ref, base); key = str(p.resolve()).casefold()
        require(key not in seen, 'Cyclic dense packet parent')
        patch = store.json(ref, base)
        require(patch.get('schema') == 'rinkfit.joint_packet.v1', 'Packet schema')
        require(not set(REVIEW_CHAINS.values()) & set(patch), 'Derived review chain supplied by input')
        if 'dense_parent_packet' not in patch:
            ancestry.append(store.verified[(str(p), ref['sha256'])]); return patch
        out = resolve(patch['dense_parent_packet'], p.parent, seen | {key})
        for current, chain in REVIEW_CHAINS.items():
            if current in out and chain not in out: out[chain] = [out[current]]
        for key, tail in patch.get('dense_parent_append', {}).items():
            require(key in APPENDABLE and isinstance(tail, list), 'Unsupported append path')
            if key == 'engineering_pixel_dependency_groups.groups':
                require('engineering_pixel_dependency_groups' not in patch, 'Dependency append/override conflict')
                dst = out['engineering_pixel_dependency_groups']['groups']
            else:
                require(key not in patch, 'Append/override conflict')
                dst = out.setdefault(key, [])
            require(isinstance(dst, list), 'Parent append field is not array'); dst.extend(tail)
        for key, val in patch.items():
            if key == 'dense_parent_append': continue
            require(key not in {'source_row_ids', 'source_dataset_ids', 'source_family_ids'}, 'Overridden immutable prefix registry')
            out[key] = val
        for current, chain in REVIEW_CHAINS.items():
            if current in out and (not out.get(chain) or out[chain][-1] != out[current]):
                out.setdefault(chain, []).append(out[current])
        ancestry.append(store.verified[(str(p), ref['sha256'])]); return out
    return resolve(head, Path(head['file']).parent, set()), ancestry


def references(value, prefix=''):
    if isinstance(value, dict):
        if 'file' in value and 'sha256' in value: yield prefix, value
        for key, child in value.items(): yield from references(child, prefix + '/' + key)
    elif isinstance(value, list):
        for i, child in enumerate(value): yield from references(child, prefix + '/' + str(i))


def inspect_payload(head, model_ref, store):
    start = time.perf_counter(); packet, ancestry = resolve_packet(head, store)
    base = Path(head['file']).parent; model = store.json(model_ref, base); n = packet['row_count']
    arrays, checked_arrays = {}, {}
    for key, ref in packet['arrays'].items():
        a, check = store.array(ref, base)
        require(a.shape[0] == n, 'Raw array denominator: ' + key)
        require(a.dtype.kind not in 'fc' or np.isfinite(a).all(), 'Nonfinite array: ' + key)
        arrays[key] = a; checked_arrays[key] = check
    require(np.isin(arrays['admission'], [0, 1]).all(), 'Nonbinary physical admission')
    physical = (arrays['admission'] == 1) & (arrays['fit_role'] == 1)
    require(int(physical.sum()) == packet['active_rows'], 'Active denominator')
    require(np.isin(arrays['kind'], [0, 1, 2]).all(), 'Unknown source operator kind')
    require((arrays['E'] > 0).all(), 'Nonpositive source engineering E')
    require(((arrays['station'] >= 0) & (arrays['station'] <= 1)).all(), 'Raw source station range')
    require(((arrays['row'] >= 0) & (arrays['row'] <= 1079)).all(), 'Raw observed row range')
    shot_ids = packet['shot_ids']; shot = arrays['shot_index']; pts = arrays['native_pts']
    require((shot < len(shot_ids)).all(), 'Shot dictionary range')
    materials = model['materials']; mat = arrays['material_index']; path = arrays['path']
    require((mat < len(materials)).all(), 'Material dictionary range')
    for mi, material in enumerate(materials):
        rr = mat == mi
        require((path[rr] < len(material['paths'])).all(), 'Material path range')
    for si, shot_id in enumerate(shot_ids):
        rr = shot == si; scope = model['shots'][shot_id]['scope']
        require(((pts[rr] >= scope['first_pts']) & (pts[rr] <= scope['last_pts'])).all(), 'Source PTS outside shot')
    prefix_ids = packet.get('source_row_ids', []); datasets = packet.get('source_dataset_ids', ['joint'] * len(prefix_ids))
    require(len(prefix_ids) == len(datasets), 'Prefix dataset denominator')
    hashes = np.empty(n, dtype='V32'); source_digest = hashlib.sha256(); cursor = 0
    def add(row, dataset, identity):
        require(isinstance(identity, str) and identity != '', 'Empty/invalid source ID')
        fields = (dataset.encode('utf8'), identity.encode('utf8'), shot_ids[int(shot[row])].encode('utf8'))
        h = hashlib.sha256()
        for field in fields:
            value = struct.pack('<Q', len(field)) + field; h.update(value); source_digest.update(value)
        value = struct.pack('<q', int(pts[row])); h.update(value); source_digest.update(value)
        hashes[row] = h.digest()
    for i, identity in enumerate(prefix_ids): add(i, datasets[i], identity)
    cursor = len(prefix_ids)
    registries = ([packet['dense_source_binding']] if 'dense_source_binding' in packet else [])
    registries += packet.get('native_ad_source_bindings', [packet['native_ad_source_binding']] if 'native_ad_source_binding' in packet else [])
    summaries = []
    for reg in registries:
        require(reg['schema'] == 'rinkfit.dense_source_suffix_registry.v1' and reg['row_start'] == cursor, 'Suffix registry discontinuity')
        count = reg['count']; off, oc = store.array(reg['source_id_offsets'], base); buf, bc = store.array(reg['source_id_utf8'], base)
        family, fc = store.array(reg['source_family_index'], base); ordinal, sc = store.array(reg['source_store_ordinal'], base)
        require(off.shape == (count + 1,) and off[0] == 0 and off[-1] == len(buf) and (off[1:] > off[:-1]).all(), 'UTF8 offsets')
        require(family.shape == ordinal.shape == (count,) and (family < len(reg['source_family_ids'])).all(), 'Source family/ordinal denominator')
        require(len(set(reg['source_family_ids'])) == len(reg['source_family_ids']), 'Duplicate family dictionary')
        raw = memoryview(np.asarray(buf)).cast('B')
        for k in range(count): add(cursor+k, reg['source_dataset_id'], bytes(raw[int(off[k]):int(off[k+1])]).decode('utf8'))
        summaries.append(dict(row_start=cursor, count=count, dataset=reg['source_dataset_id'],
                              family_count=len(reg['source_family_ids']), ordinal_sha256=sc['slice_sha256'],
                              ordinal_min=int(ordinal.min()) if count else None, ordinal_max=int(ordinal.max()) if count else None))
        cursor += count
    require(cursor == n, 'Incomplete raw source registry')
    order = np.argsort(hashes); collisions = order[1:][hashes[order[1:]] == hashes[order[:-1]]]
    # Equal immutable keys necessarily have equal hashes. Fail closed on even a
    # hypothetical distinct-key collision; never accept probabilistic uniqueness.
    require(not len(collisions), 'Repeated/colliding immutable source identity')
    partition = np.zeros(n, np.uint8); rank = known = unknown = 0
    for group in packet['groups']:
        rr = np.asarray(group['rows'], np.int64); gr = group['rank']
        require(len(rr) and rr.min() >= 0 and rr.max() < n and len(np.unique(rr)) == len(rr), 'Invalid statistical source unit')
        require(not partition[rr].any() and 0 <= gr <= 2*len(rr), 'Overlapping statistical source units')
        partition[rr] = 1
        if group['statistic_kind'] == 'constraint_only_unknown_noise':
            require(gr == 0 and group['W'] == [] and group.get('unknown_noise_source_refs'), 'Unknown-noise W fabrication'); unknown += 1
        else:
            require(gr > 0, 'Positive statistical unit lost rank'); known += 1
            w = np.asarray(group['W'], float) if isinstance(group['W'], list) else store.array(group['W'], base)[0]
            require(w.size == gr*2*len(rr) and np.isfinite(w).all(), 'Full W dimensions/finiteness')
        rank += gr
    independent = np.asarray(packet.get('independent_rows', []), np.int64)
    if len(independent):
        require(independent.min() >= 0 and independent.max() < n and len(np.unique(independent)) == len(independent) and not partition[independent].any(), 'Independent source partition')
        require((arrays['source_sigma'][independent] > 0).all(), 'Original sigma positivity')
        partition[independent] = 1; rank += 2*len(independent)
    require((partition[physical] == 1).all(), 'Physical source has no statistical/constraint unit')
    query_seen = set()
    for q in packet.get('queries', []):
        row = q['row_index']; aa = np.asarray(q['alpha']); uv = np.asarray(q['native_uv']); normal = np.asarray(q['normal'])
        require(0 <= row < n and row not in query_seen, 'Duplicate/outside Query row'); query_seen.add(row)
        require(aa.ndim == 1 and len(aa) >= 2 and uv.shape == (len(aa), 2) and np.isfinite(uv).all() and np.isfinite(aa).all() and (np.diff(aa) > 0).all(), 'Frozen Query samples')
        require(q.get('alpha_scale', 1) > 0 and normal.shape == (2,) and np.isfinite(normal).all() and np.any(normal), 'Frozen Query scale/normal')
    ref_index = [dict(field=field, **ref) for field, ref in references(packet)]
    return dict(schema='rinkfit.effective_payload_inspection.v1', passed=True, validator=dict(file=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                source_packet=head, model=model_ref, raw_rows=n, parent_layers=len(ancestry), ancestry=ancestry,
                canonical_encoding='rinkfit.tagged_json_sha256.v1', resolved_metadata_sha256=digest(packet),
                arrays=checked_arrays, physical_admitted_rows=int(physical.sum()), source_registries=summaries,
                complete_identity_tuple_stream_sha256=source_digest.hexdigest(), source_keys_unique=True,
                groups=dict(positive_rank=known, rank_zero=unknown, independent_rows=len(independent), original_rank=rank),
                queries=len(query_seen), legacy_reference_index=ref_index,
                input_mode='inspection_only_no_native_abi', effective_payload_complete=False,
                missing_native_proofs=['cross_language_resolved_digest', 'source_qualification_evidence_consumer', 'material_path_and_dependency_math', 'actual_training_selection_budget_and_motion', 'checkpoint_input_abi'],
                historical_issuance_replay_complete=False, source_image_identity_proven=False,
                native_input_enabled=False, quality_accepted=False, seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--packet', required=True); parser.add_argument('--model', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    def ref(path):
        p=Path(path).resolve(); return dict(file=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    output = Path(args.output); require(not output.exists(), 'Immutable output exists')
    report = inspect_payload(ref(args.packet), ref(args.model), CheckedBytes())
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({k:report[k] for k in ['passed','raw_rows','parent_layers','resolved_metadata_sha256','physical_admitted_rows','groups','queries','native_input_enabled','seconds']}))


if __name__ == '__main__': main()
