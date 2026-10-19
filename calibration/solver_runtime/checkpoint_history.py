"""Complete solver history without growing NumPy Unicode checkpoint scalars.

Numerical checkpoint members retain their original names and values.  The
history_json member and iterations.json accept either the historical JSON list
or a small, authenticated reference to immutable record versions.
"""
from __future__ import annotations

import codecs
import hashlib
import io
import json
from pathlib import Path
import shutil
import uuid
import zipfile

import numpy as np

REFERENCE_SCHEMA = 'rinkspace.solver-history-reference.v1'
SEGMENT_SCHEMA = 'rinkspace.solver-history-segment.v1'


def _default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_default)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _reference(head, count):
    value = dict(schema=REFERENCE_SCHEMA, record_count=count, head=head)
    value['reference_sha256'] = _digest(_text(value).encode('utf8'))
    return value


def _validate_reference(value):
    if not isinstance(value, dict) or value.get('schema') != REFERENCE_SCHEMA:
        raise ValueError('Explicit solver history reference schema required')
    if set(value) != {'schema', 'record_count', 'head', 'reference_sha256'}:
        raise ValueError('Solver history reference field inventory changed')
    payload = {key: item for key, item in value.items() if key != 'reference_sha256'}
    if _digest(_text(payload).encode('utf8')) != value['reference_sha256']:
        raise ValueError('Solver history reference hash differs')
    count = value['record_count']
    if type(count) is not int or count < 0 or (value['head'] is None) != (count == 0):
        raise ValueError('Solver history reference count/head differs')
    if count and (not isinstance(value['head'], dict) or value['head'].get('record_count') != count):
        raise ValueError('Solver history head count differs')
    return value


def _read_segment(link):
    if not isinstance(link, dict) or set(link) != {'path', 'sha256', 'record_count'}:
        raise ValueError('Solver history segment link inventory differs')
    path = Path(link['path'])
    if not path.is_absolute() or path.name != link['sha256'] + '.json':
        raise ValueError('Absolute content-addressed solver history segment required')
    raw = path.read_bytes()
    if _digest(raw) != link['sha256']:
        raise ValueError('Solver history segment hash differs: ' + str(path))
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) !=
            {'schema', 'previous', 'index', 'record_count', 'record'}
            or value['schema'] != SEGMENT_SCHEMA):
        raise ValueError('Solver history segment schema/inventory differs')
    if value['record_count'] != link['record_count'] or not isinstance(value['record'], dict):
        raise ValueError('Solver history segment count/record differs')
    previous = value['previous']
    before = 0 if previous is None else previous.get('record_count')
    index, count = value['index'], value['record_count']
    if (type(before) is not int or type(index) is not int or type(count) is not int
            or index not in (before, before - 1) or index < 0 or count != index + 1):
        raise ValueError('Solver history segment append/tail-version contract differs')
    return value


def iter_history_reference(reference):
    """Verify every link and yield only the latest version of each record."""
    reference = _validate_reference(reference)
    count = reference['record_count']
    selected = {}
    link = reference['head']
    seen = set()
    while link is not None:
        if link['sha256'] in seen:
            raise ValueError('Cyclic solver history reference')
        seen.add(link['sha256'])
        segment = _read_segment(link)
        if segment['index'] >= count:
            raise ValueError('Solver history reference truncates a recorded state')
        selected.setdefault(segment['index'], link)
        link = segment['previous']
    if set(selected) != set(range(count)):
        raise ValueError('Solver history reference has missing records')
    for index in range(count):
        yield _read_segment(selected[index])['record']


class HistorySegmentWriter:
    """Append new records or a mutable tail; the caller seals older records."""
    def __init__(self, directory, *, reference=None, history=()):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.head = None
        self.count = 0
        self.last_text = None
        if reference is not None:
            value = _validate_reference(reference)
            # Resume's reader already verified the complete immutable chain.
            if len(history) != value['record_count']:
                raise ValueError('Restored solver history count differs')
            self.head, self.count = value['head'], value['record_count']
            self.last_text = None if not history else _text(history[-1])

    def append_record(self, record, *, index=None):
        index = self.count if index is None else index
        if index not in (self.count, self.count - 1) or index < 0:
            raise ValueError('Only append or update the unsealed history tail')
        encoded = _text(record)
        if index == self.count - 1 and encoded == self.last_text:
            return
        prefix = dict(schema=SEGMENT_SCHEMA, previous=self.head, index=index,
                      record_count=index + 1)
        # Each record is encoded once; no old-history JSON or Unicode scalar.
        raw = (_text(prefix)[:-1] + ', "record": ' + encoded + '}').encode('utf8')
        digest = _digest(raw)
        path = self.directory / (digest + '.json')
        if path.exists():
            if _digest(path.read_bytes()) != digest:
                raise ValueError('Existing immutable history segment differs')
        else:
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(raw)
            temporary.replace(path)
        self.head = dict(path=str(path), sha256=digest, record_count=index + 1)
        self.count, self.last_text = index + 1, encoded

    def commit(self, history):
        if len(history) < self.count:
            raise ValueError('Complete solver history cannot shrink')
        # The prior tail can receive its final checkpoint/review timing before
        # the next record is appended.  All older records are detached/sealed.
        start = max(0, self.count - 1)
        for index in range(start, len(history)):
            self.append_record(history[index], index=index)
        return _reference(self.head, self.count)


def _iter_json_records(chunks):
    """Decode a legacy JSON list one record at a time (bounded by one record)."""
    decoder = json.JSONDecoder()
    source = iter(chunks)
    buffer = ''
    ended = False
    def more():
        nonlocal buffer, ended
        try:
            buffer += next(source)
        except StopIteration:
            ended = True
    more()
    while not buffer.lstrip() and not ended:
        more()
    buffer = buffer.lstrip()
    if not buffer.startswith('['):
        for chunk in source:
            buffer += chunk
        value = json.loads(buffer)
        yield from iter_history_reference(value)
        return
    buffer = buffer[1:]
    need_value = True
    had_record = False
    while True:
        buffer = buffer.lstrip()
        while not buffer and not ended:
            more(); buffer = buffer.lstrip()
        if buffer.startswith(']'):
            if not need_value:
                buffer = buffer[1:]
                break
            # Empty lists are legal; a comma followed by ] is not.
            if had_record:
                raise ValueError('Trailing comma in legacy solver history')
            buffer = buffer[1:]
            break
        if not need_value:
            if not buffer.startswith(','):
                raise ValueError('Legacy solver history separator differs')
            buffer = buffer[1:]; need_value = True
            continue
        try:
            record, stop = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if ended:
                raise
            # A large record must not be reparsed from its beginning for each
            # tiny chunk. Double the buffered length before the next attempt;
            # total failed parse work stays linear in the record's size.
            target = max(1, len(buffer) * 2)
            while len(buffer) < target and not ended:
                more()
            continue
        if not isinstance(record, dict):
            raise ValueError('Solver history record must be an object')
        yield record
        had_record = True
        buffer = buffer[stop:]; need_value = False
    for chunk in source:
        buffer += chunk
    if buffer.strip():
        raise ValueError('Unexpected data after solver history')


def iter_solver_history(path):
    with Path(path).open('r', encoding='utf8') as stream:
        yield from _iter_json_records(iter(lambda: stream.read(1024 * 1024), ''))


def read_solver_history(path):
    return list(iter_solver_history(path))


def _checkpoint_history_chunks(path, field='history_json'):
    with zipfile.ZipFile(path) as archive, archive.open(field + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported history NPY header version')
        if shape != () or dtype.kind != 'U':
            raise ValueError('Historical scalar Unicode history member required')
        encoding = 'utf-32-be' if dtype.byteorder == '>' else 'utf-32-le'
        decoder = codecs.getincrementaldecoder(encoding)()
        remaining = dtype.itemsize
        while remaining:
            chunk = stream.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError('Truncated checkpoint history member')
            remaining -= len(chunk)
            yield decoder.decode(chunk)
        tail = decoder.decode(b'', final=True)
        if tail:
            yield tail
        if stream.read(1):
            raise ValueError('Unexpected data after checkpoint history member')


def iter_checkpoint_history(checkpoint_path, *, field='history_json'):
    yield from _iter_json_records(_checkpoint_history_chunks(checkpoint_path, field))


def read_checkpoint_history(stored, checkpoint_path=None, *, field='history_json'):
    if checkpoint_path is not None:
        return list(iter_checkpoint_history(checkpoint_path, field=field))
    value = json.loads(str(np.asarray(stored[field]).item()))
    return value if isinstance(value, list) else list(iter_history_reference(value))


def read_checkpoint_reviews(stored, checkpoint_path=None):
    return read_checkpoint_history(stored, checkpoint_path, field='reviews_json')


def read_solver_reviews(path):
    return read_solver_history(path)


def checkpoint_history_reference(stored, checkpoint_path=None, *, field='history_json'):
    """Read only new small metadata; legacy migration uses the streaming API."""
    if checkpoint_path is not None:
        chunks = _checkpoint_history_chunks(checkpoint_path, field)
        try:
            first = next(chunks).lstrip()
            if first.startswith('['):
                return None
            value = json.loads(first + ''.join(chunks))
        finally:
            chunks.close()
    else:
        value = json.loads(str(np.asarray(stored[field]).item()))
    return None if isinstance(value, list) else _validate_reference(value)


def migrate_legacy_checkpoint_history(source, destination, *, history_directory=None):
    """Stream authoritative NPZ history into segments and copy all other NPYs.

Never materializes the old Unicode scalar or all records.  The destination is
new; the original checkpoint/archive and its published identity are untouched.
"""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history_directory = Path(history_directory or destination.parent / 'history_segments')
    references = {}
    semantic_digests = {}
    for field, directory in (('history_json', history_directory),
                             ('reviews_json', destination.parent / 'review_segments')):
        writer = HistorySegmentWriter(directory)
        semantic = hashlib.sha256()
        for record in iter_checkpoint_history(source, field=field):
            writer.append_record(record)
            semantic.update((writer.last_text + '\n').encode('utf8'))
        references[field] = _reference(writer.head, writer.count)
        semantic_digests[field] = dict(records=writer.count, sha256=semantic.hexdigest(),
                                      encoding='checkpoint_history._text(record) plus LF, UTF-8')
    temporary = destination.with_name(destination.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as new:
        for entry in old.infolist():
            if entry.filename in ('history_json.npy', 'reviews_json.npy'):
                array = io.BytesIO()
                np.lib.format.write_array(array, np.asarray(_text(references[entry.filename[:-4]])), allow_pickle=False)
                new.writestr(entry.filename, array.getvalue())
            else:
                with old.open(entry) as incoming, new.open(entry.filename, 'w', force_zip64=True) as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    temporary.replace(destination)
    return dict(schema='rinkspace.checkpoint-history-migration.v1', source=str(source.resolve()),
                destination=str(destination.resolve()), history_reference=references['history_json'],
                reviews_reference=references['reviews_json'],
                record_semantic_digests=semantic_digests,
                numerical_members_copied_without_numpy_decode=True,
                authoritative_history_source='original NPZ history_json.npy', original_checkpoint_preserved=True)
