"""Privacy scanner fixtures: no production secrets or private values."""
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import codecs

import pytest

SPEC = spec_from_file_location('privacy_audit', Path(__file__).resolve().parents[1] / 'scripts' / 'audit_worktree_privacy.py')
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def token() -> str:
    return 'ghp_' + ''.join(chr(65 + i % 26) for i in range(40))


@pytest.mark.parametrize('encoding,bom', [
    ('utf-16-le', codecs.BOM_UTF16_LE),
    ('utf-16-be', codecs.BOM_UTF16_BE),
    ('utf-16-le', b''),
    ('utf-16-be', b''),
])
def test_utf16_secret_and_home_keep_original_offsets(encoding, bom):
    prefix = 'fixture text before key: '
    home = 'C:' + chr(92) + 'Users' + chr(92) + 'fixture-user'
    text = prefix + token() + '\n' + home
    payload = bom + text.encode(encoding)
    findings, count = audit.scan(BytesIO(payload), 'fixture')
    keys = [x for x in findings if x['category'] == 'github_token']
    homes = [x for x in findings if x['category'] == 'user_home_path']
    assert count == len(payload)
    assert len(keys) == len(homes) == 1
    assert keys[0]['source_encoding'] == encoding
    assert keys[0]['byte_offset'] == len(bom) + len(prefix.encode(encoding))
    assert homes[0]['byte_offset'] == len(bom) + len((prefix + token() + '\n').encode(encoding))
    assert token() not in str(findings)
    assert home not in str(findings)


@pytest.mark.parametrize('encoding', ['utf8', 'utf-16-le', 'utf-16-be'])
def test_tokens_and_home_cross_stream_chunks(monkeypatch, encoding):
    monkeypatch.setattr(audit, 'CHUNK_BYTES', 31)
    prefix = 'fixture text before key: '
    home = 'C:' + chr(92) + 'Users' + chr(92) + 'fixture-user'
    text = prefix + token() + '\n' + home + '\n' + 'x' * 100
    bom = b'' if encoding == 'utf8' else (codecs.BOM_UTF16_LE if encoding.endswith('le') else codecs.BOM_UTF16_BE)
    payload = bom + text.encode(encoding)
    findings, count = audit.scan(BytesIO(payload), 'fixture')
    assert count == len(payload)
    assert sum(x['category'] == 'github_token' for x in findings) == 1
    assert sum(x['category'] == 'user_home_path' for x in findings) == 1
    assert next(x for x in findings if x['category'] == 'github_token')['byte_offset'] == len(bom) + len(prefix.encode(encoding))


def test_utf16_multibyte_prefix_and_surrogate_chunk(monkeypatch):
    monkeypatch.setattr(audit, 'CHUNK_BYTES', 19)
    prefix = '数据🙂 local key: '
    payload = codecs.BOM_UTF16_LE + (prefix + token()).encode('utf-16-le')
    findings, count = audit.scan(BytesIO(payload), 'fixture')
    key = next(x for x in findings if x['category'] == 'github_token')
    assert key['byte_offset'] == 2 + len(prefix.encode('utf-16-le'))
    assert count == len(payload)


@pytest.mark.parametrize('name', ['.env', '.env.local', 'credentials.json', 'token.json', 'id_rsa', 'id_ed25519', '.netrc', '.npmrc', 'client.pem', 'signing.pfx'])
def test_sensitive_filenames(name):
    assert audit.sensitive_filename(name)
    assert audit.is_text_candidate(name)


@pytest.mark.parametrize('name', ['.env.example', '.env.sample', '.env.production.template', 'id_ed25519.pub', 'README', 'CMakeLists.txt', 'Dockerfile', 'model.json'])
def test_safe_names_and_extensionless_text(name):
    assert not audit.sensitive_filename(name)
    assert audit.is_text_candidate(name)


def test_malformed_utf16_is_reported_without_values(monkeypatch):
    monkeypatch.setattr(audit, 'CHUNK_BYTES', 9)
    payload = codecs.BOM_UTF16_LE + b'a\x00x'
    findings, count = audit.scan(BytesIO(payload), 'fixture')
    assert count == len(payload)
    assert findings == [{'path': 'fixture', 'category': 'text_encoding_error', 'byte_offset': 4, 'source_encoding': 'utf-16-le'}]


def test_empty_stream():
    assert audit.scan(BytesIO(b''), 'fixture') == ([], 0)


@pytest.mark.parametrize('prefix', [b'{"referenced_threads": ["', b"confirmation_received_in_root_task='"])
def test_private_identifier_without_known_field_name(prefix):
    identifier = ('01abcdef' + '-1234-1234-1234-123456789abc').encode('ascii')
    findings, count = audit.scan(BytesIO(prefix + identifier + b"'"), 'fixture.txt')
    assert count == len(prefix) + len(identifier) + 1
    assert [(row['category'], row['byte_offset']) for row in findings] == [('private_codex_identifier', len(prefix))]
