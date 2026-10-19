"""Stream all current Git candidates for privacy locations, without printing values.

This is a local pattern audit. Binary media/archive payloads are inventoried for
separate metadata/archive inspection; no third-party service receives content.
"""
from __future__ import annotations
import argparse
import codecs
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEXT = {'.py','.ps1','.cmd','.bat','.c','.cpp','.h','.hpp','.cu','.cuh','.md','.rst','.txt','.json','.jsonl','.toml','.yml','.yaml','.html','.htm','.css','.js','.ts','.xml','.svg','.ini','.cfg','.env','.pem','.key','.lock','.patch','.log','.csv','.tsv','.spec','.pub','.ipynb'}
PATTERNS = {
 'private_codex_identifier': re.compile(rb'\b01[0-9a-f]{6}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|codex-clipboard-[0-9a-f-]{36}', re.I),
 'oauth_token_format': re.compile(rb'\bya29\.[A-Za-z0-9_-]{24,255}'),
 'jwt_candidate': re.compile(rb'\beyJ[A-Za-z0-9_-]{8,2048}\.[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,1024}\b'),
 'database_credential_url': re.compile(rb'(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)://[^\s/@\x22\x27:]{1,128}:[^\s/@\x22\x27]{1,256}@'),
 'private_ipv4': re.compile(rb'(?<![A-Za-z0-9_.])(?:10\.[0-9]{1,3}|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?![A-Za-z0-9_.])'),
 'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
 'github_token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255})\b'),
 'openai_or_anthropic_key': re.compile(rb'\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{24,255}\b'),
 'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'google_api_key': re.compile(rb'\bAIza[0-9A-Za-z_-]{35}\b'),
 'slack_token': re.compile(rb'\bxox[baprs]-[0-9A-Za-z-]{20,255}\b'),
 'credential_url': re.compile(rb'https?://[^\s/\x22\x27<>:@]{1,128}:[^\s/\x22\x27<>@]{1,256}@'),
 'literal_secret_assignment': re.compile(rb'(?i)[\x22\x27]?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret[_-]?key)[\x22\x27]?\s{0,16}[:=]\s{0,16}[\x22\x27]([^\x22\x27\r\n]{12,256})[\x22\x27]'),
 'email': re.compile(rb'(?<![A-Za-z0-9_.+%-])[A-Za-z0-9_.+%-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}(?![A-Za-z])'),
 'user_home_path': re.compile(rb'(?i)(?:[A-Z]:[\\/]{1,4}Users[\\/]{1,4}|/(?:home|Users)/)[A-Za-z0-9_.-]{1,64}'),
 'china_phone_candidate': re.compile(rb'(?<![A-Za-z0-9_.])(?:\+?86[- ]?)?1[3-9][0-9]{9}(?![A-Za-z0-9_.])'),
 'chat_or_session_reference': re.compile(rb'(?i)(?:thread://|codex://threads/|chatgpt\.com/(?:c|share)/|\x22(?:threadId|conversation_id|session_id)\x22\s*:)'),
 'signed_download_url': re.compile(rb'(?i)(?:[?&](?:X-Amz-Signature|X-Goog-Signature|sig|access_token|token)=)[^\s\x22\x27&<>]{12,512}'),
}
PLACEHOLDER = re.compile(rb'(?i)(?:example|placeholder|your[_ -]|replace[_ -]|changeme|redacted|dummy|test[_-]|<|\{|\$)')


CHUNK_BYTES = 1024 * 1024
# Includes a byte of left context, and exceeds the longest token rule above.
OVERLAP_BYTES = 8193


def sensitive_filename(path: str | Path) -> bool:
    """Flag credential-file names for review; do not read or print their values."""
    name = Path(path).name.casefold()
    if name == '.env' or name.startswith('.env.'):
        return re.search(r'(?:^|[._-])(?:example|sample|template)(?:$|[._-])', name) is None
    if name in {'.git-credentials', '.netrc', '.npmrc', '.pypirc'}:
        return True
    if re.fullmatch(r'id_(?:rsa|ed25519|dsa|ecdsa)(?:\..*)?', name):
        return not name.endswith('.pub')
    if re.fullmatch(r'(?:credentials?|tokens?|secrets?)(?:\.(?:json|ya?ml|toml|ini|cfg|txt))?', name):
        return True
    return Path(name).suffix in {'.pem', '.key', '.p12', '.pfx', '.jks', '.keystore'}


def is_text_candidate(path: str | Path) -> bool:
    path = Path(path)
    return (path.suffix.casefold() in TEXT or not path.suffix
            or path.name.casefold().startswith('.env') or sensitive_filename(path))


def detect_text_encoding(head: bytes) -> tuple[str | None, int]:
    """Recognize UTF-16 without mistaking ordinary UTF-8 text for it."""
    if head.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le', 2
    if head.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be', 2
    sample = head[:4096]
    pairs = len(sample) // 2
    if pairs >= 8:
        even = sum(sample[i] == 0 for i in range(0, pairs * 2, 2)) / pairs
        odd = sum(sample[i] == 0 for i in range(1, pairs * 2, 2)) / pairs
        if odd >= 0.6 and even <= 0.1:
            return 'utf-16-le', 0
        if even >= 0.6 and odd <= 0.1:
            return 'utf-16-be', 0
    return None, 0


def scan(stream, label: str) -> tuple[list[dict], int]:
    """Stream original bytes; UTF-16 findings retain original byte offsets."""
    findings, seen, carry, total = [], set(), b'', 0
    first = stream.read(CHUNK_BYTES)
    encoding, bom_bytes = detect_text_encoding(first)
    decoder = codecs.getincrementaldecoder(encoding)(errors='strict') if encoding else None
    raw_consumed, carry_start = bom_bytes, 0
    chunk = first[bom_bytes:]
    total += len(first)
    while True:
        # Decoder state may hold a split code unit or high surrogate.
        decoded_start = raw_consumed - (len(decoder.getstate()[0]) if decoder else 0)
        raw_consumed += len(chunk)
        final = not chunk
        try:
            normalized = decoder.decode(chunk, final=final).encode('utf8') if decoder else chunk
        except UnicodeError:
            findings.append({'path': label, 'category': 'text_encoding_error',
                             'byte_offset': decoded_start, 'source_encoding': encoding})
            # A malformed text file needs inspection; still count every source byte.
            while True:
                remainder = stream.read(CHUNK_BYTES)
                total += len(remainder)
                if not remainder:
                    return findings, total
        block = carry + normalized
        start = carry_start if carry else decoded_start

        def source_offset(index: int) -> int:
            if not encoding:
                return start + index
            return start + len(block[:index].decode('utf8').encode(encoding))

        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(block):
                if carry and match.start() == 0:
                    continue  # Left overlap context may begin inside a longer token.
                at = source_offset(match.start())
                key = (category, at)
                if key in seen:
                    continue
                if category == 'literal_secret_assignment':
                    literal = match.group(1)
                    if PLACEHOLDER.search(literal) or len(set(literal)) < 8 or re.search(rb'\s', literal):
                        continue
                if category in ('github_token', 'openai_or_anthropic_key') and PLACEHOLDER.search(match.group()):
                    continue
                if category == 'email' and re.search(rb'@(?:example\.(?:com|org|net)|test\.(?:com|org)|localhost)$', match.group(), re.I):
                    continue
                seen.add(key)
                item = {'path': label, 'category': category, 'byte_offset': at}
                if encoding:
                    item['source_encoding'] = encoding
                findings.append(item)
        begin = max(0, len(block) - OVERLAP_BYTES)
        if encoding:
            # Keep the overlap on a UTF-8 character boundary for offset conversion.
            while begin < len(block) and block[begin] & 0xC0 == 0x80:
                begin += 1
        carry_start = source_offset(begin)
        carry = block[begin:]
        if final:
            return findings, total
        chunk = stream.read(CHUNK_BYTES)
        total += len(chunk)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if output.is_relative_to(ROOT):
        p.error('Write diagnostics outside the project')
    raw = subprocess.check_output(['git','--no-optional-locks','ls-files','-c','-o','--exclude-standard','-z'],cwd=ROOT)
    names = sorted(set(x.decode('utf8') for x in raw.split(b'\0') if x))
    findings, others, missing, scanned, byte_count = [], [], [], 0, 0
    for name in names:
        source = ROOT/name
        if not source.is_file() or source.is_symlink():
            missing.append(name); continue
        if sensitive_filename(name):
            findings.append({'path': name, 'category': 'sensitive_filename', 'byte_offset': 0})
        if not is_text_candidate(name):
            others.append({'path':name,'bytes':source.stat().st_size}); continue
        before = source.stat()
        with source.open('rb') as stream:
            found, count = scan(stream,name)
        if (source.stat().st_size,source.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
            findings.append({'path':name,'category':'changed_during_scan','byte_offset':0})
        findings.extend(found);scanned += 1;byte_count += count
        if scanned % 500 == 0:
            print(json.dumps({'progress_files':scanned,'bytes':byte_count}),flush=True)
    result = {'schema':'rinkspace.worktree-privacy.v1','finished_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'All current tracked and unignored untracked text candidates; streaming entire contents, 8193-byte overlap; UTF-16 normalized with original byte offsets; extensionless and sensitive filenames included; no text size truncation.',
       'summary':{'text_files':scanned,'text_bytes':byte_count,'findings':len(findings),'categories':dict(Counter(x['category'] for x in findings))},
       'findings':findings,'nontext_for_separate_inspection':others,'missing_or_linked':missing,
       'limitations':['Pattern candidates require contextual review. No matched values are stored.','Binary media/archive metadata and Git history are audited separately.','Ignored local files are outside Git candidate scope; sensitive filenames are candidates, not confirmation of a secret.','Public athlete names and official URLs are project facts, not automatically private data.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
