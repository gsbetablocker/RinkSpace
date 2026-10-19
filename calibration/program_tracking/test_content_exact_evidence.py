import json

import pytest

from calibration.program_tracking.compact_source_json import (
    ContentExactEvidenceCanonicalizer, ExactEvidenceInterner,
    pack_source_json, unpack_source_json,
)


def literal(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':'))


def test_content_canonicalizer_shares_complex_source_evidence_without_encoded_bank():
    interner = ContentExactEvidenceCanonicalizer()
    evidence = {'source': {'path': '真实原片/' + 'x'*160, 'sha256': '1'*64},
        'uv': [[1.25, 2.5], [3.75, -0.0]], 'counts': [1, 2, 3]}
    first = interner.canonicalize(evidence)
    second = interner.canonicalize(json.loads(literal(evidence)))
    assert second is first
    assert literal(first) == literal(evidence)
    assert not hasattr(interner, 'values')
    assert not hasattr(interner, 'objects')
    assert all(type(key) is bytes and len(key) == 32 for key in interner.lookup)
    assert literal(unpack_source_json(pack_source_json([first, second]))) == literal([evidence, evidence])


def test_primitive_source_lists_do_not_add_cache_entries():
    interner = ContentExactEvidenceCanonicalizer()
    for index in range(100):
        value = [index, float(index), -0.0, False, None, '源']
        assert literal(interner.canonicalize(value)) == literal(value)
    assert interner.lookup == {}


def test_forced_digest_collisions_keep_types_signed_zero_and_distinct_source_proofs(monkeypatch):
    interner = ContentExactEvidenceCanonicalizer()
    monkeypatch.setattr(interner, '_digest', lambda _: b'forced-collision')
    inputs = [{'pixel': [value], 'proof': {'source': '同一原片'}}
              for value in (True, 1, 1.0, 0.0, -0.0, None, '1')]
    canonical = [interner.canonicalize(value) for value in inputs]
    assert len({id(value) for value in canonical}) == len(inputs)
    assert [literal(value) for value in canonical] == [literal(value) for value in inputs]
    for value, expected in zip(inputs, canonical):
        assert interner.canonicalize(json.loads(literal(value))) is expected
    assert len(interner.lookup) == 1


def test_existing_source_json_literal_equality_and_invalid_values_stay_exact():
    values = [{'source': {'authority': 'a'*150}, 'uv': [[0.0, -0.0], [1, 1.0]],
               'confirmed': True, 'missing': None} for _ in range(3)]
    old = ExactEvidenceInterner()
    new = ContentExactEvidenceCanonicalizer()
    assert literal([new.canonicalize(value) for value in values]) == literal(
        [old.canonicalize(value) for value in values])
    with pytest.raises(ValueError, match='JSON compliant'):
        new.canonicalize([float('nan')])
    with pytest.raises(ValueError, match='Reserved'):
        new.canonicalize({'__rinkspace_exact_source_json_ref__': 0})
    with pytest.raises(TypeError, match='keys'):
        new.canonicalize({1: 'source'})
