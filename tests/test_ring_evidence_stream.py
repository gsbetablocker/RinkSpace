"""Evidence files may exceed memory; corruption must not become missing detections."""
import gzip
import json

import pytest

from rinkspace.ring_evidence import iter_ring_frames


@pytest.mark.parametrize('compressed', [False, True])
def test_stream_keeps_large_cross_chunk_profiles_and_exact_pts(tmp_path, compressed):
    frames = [{'pts': 238955520, 'profile': 'a' * 1100000},
              {'pts': 238955776, 'profile': 'b' * 2100000}]
    path = tmp_path / ('observations.json.gz' if compressed else 'observations.json')
    opener = gzip.open if compressed else open
    with opener(path, 'wt', encoding='utf-8') as handle:
        json.dump({'source_id': 'actual-clock', 'frames': frames}, handle)
    assert list(iter_ring_frames(path)) == frames


@pytest.mark.parametrize('document', [
    '{"frames": [{"pts": 1} {"pts": 2}]}',
    '{"frames": [{"pts": 1},]}',
    '{"frames": [{"pts": 1}, {"pts":',
    '{"frames": [{"pts": 1}]',
    '{"frames": [{"pts": 1}], "silently_ignored": true}',
    '{"nested": {"frames": [{"pts": 1}]}}',
    '{"frames": [42]}',
])
def test_corrupt_or_unsupported_evidence_is_not_silently_accepted(tmp_path, document):
    path = tmp_path / 'evidence.json'
    path.write_text(document, encoding='utf-8')
    with pytest.raises(ValueError):
        list(iter_ring_frames(path))


def test_empty_evidence_and_gzip_truncation_remain_distinct(tmp_path):
    path = tmp_path / 'evidence.json.gz'
    data = gzip.compress(b'{"frames": []}')
    path.write_bytes(data)
    assert list(iter_ring_frames(path)) == []
    path.write_bytes(data[:-4])
    with pytest.raises(EOFError):
        list(iter_ring_frames(path))
