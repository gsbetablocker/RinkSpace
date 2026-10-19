"""Preserve a declared technical review as compact, byte-exact ZIP assets.

The input plan is explicit. No directory discovery, Git mutation or solver execution.
Build into external staging first; publish only archives below GitHub's ordinary
100 MiB file boundary. Original JSON paths remain provenance, resolved by the reader.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from rinkspace.path_policy import check_project_path
from audit_public_repository import scan_text, text_file


def build(plan_path: Path, output: Path, staging: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    output, staging = output.resolve(), staging.resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError('Review destination must be inside this project')
    if staging.is_relative_to(PROJECT) or not staging.is_absolute():
        raise ValueError('Staging must be outside the project')
    if output.exists() or staging.exists():
        raise FileExistsError('Use new review and staging directories; no overwrite')
    groups = sorted({row['group'] for row in plan['assets']})
    targets = [output / 'manifest.json', *(output / (g + '.zip') for g in groups)]
    for target in targets:
        if check_project_path(target, project_root=PROJECT):
            raise ValueError('OneDrive path limit: ' + str(target))
    staging.mkdir(parents=True)
    handles = {g: zipfile.ZipFile(staging / (g + '.zip'), 'x', zipfile.ZIP_DEFLATED,
                                  compresslevel=1, allowZip64=True) for g in groups}
    unique, assets, scanned, skipped = {}, [], 0, []
    try:
        for row in plan['assets']:
            source = Path(row['original_path'])
            if not source.is_file() or source.is_symlink():
                raise ValueError('Missing or linked input: ' + str(source))
            before = source.stat()
            if before.st_size != row['bytes']:
                raise ValueError('Size changed since declared selection: ' + str(source))
            expected = row.get('sha256')
            key = expected or str(source.resolve()).lower()
            if key not in unique:
                group = row['group']
                suffix = source.suffix.lower()
                if len(suffix) > 10 or not suffix.replace('.', '').isalnum():
                    suffix = '.bin'
                member = 'files/' + (expected or hashlib.sha256(key.encode()).hexdigest()) + suffix
                digest = hashlib.sha256()
                text = bytearray() if text_file(str(source)) and before.st_size <= 4 * 1024 * 1024 else None
                with source.open('rb') as inp, handles[group].open(member, 'w', force_zip64=True) as dest:
                    while chunk := inp.read(1024 * 1024):
                        digest.update(chunk)
                        dest.write(chunk)
                        if text is not None:
                            text.extend(chunk)
                actual = digest.hexdigest()
                if expected and actual != expected:
                    raise ValueError('Frozen input identity changed: ' + str(source))
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Input changed during preservation: ' + str(source))
                if text is not None:
                    findings, _ = scan_text(bytes(text), row['logical_path'], 'review_archive')
                    if findings:
                        raise ValueError('Credential candidate locations require review: ' + json.dumps(findings))
                    scanned += 1
                elif text_file(str(source)):
                    skipped.append(row['logical_path'])
                unique[key] = (group + '.zip', member, actual)
            artifact, member, actual = unique[key]
            assets.append({k: v for k, v in row.items() if k != 'group'} |
                          {'sha256': actual, 'artifact': artifact, 'member': member})
    finally:
        for handle in handles.values():
            handle.close()
    archives = []
    for group in groups:
        candidate = staging / (group + '.zip')
        if candidate.stat().st_size > 100 * 1024 * 1024:
            raise ValueError('Archive exceeds GitHub ordinary-file boundary; refine groups: ' + str(candidate))
        with zipfile.ZipFile(candidate) as z:
            archives.append({'file': candidate.name, 'bytes': candidate.stat().st_size,
                             'members': len(z.infolist())})
    manifest = {'schema': 'rinkspace.offline_review.v1', 'snapshot': plan['snapshot'],
                'status': plan['status'], 'scope': plan['scope'], 'archives': archives,
                'assets': assets, 'reused_code': plan.get('reused_code', []),
                'text_credential_scan': {'scanned_files': scanned, 'skipped_over_4MiB': skipped,
                                         'candidate_count': 0, 'not_a_security_certificate': True},
                'source_array_and_W_unique_references': plan.get('source_array_and_W_unique_references'),
                'excluded': plan.get('excluded', [])}
    output.mkdir(parents=True)
    for info in archives:
        shutil.copyfile(staging / info['file'], output / info['file'])
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'assets': len(assets), 'unique_members': len(unique), 'archives': archives,
                      'output': str(output)}, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--staging', required=True, type=Path)
    args = p.parse_args()
    build(args.plan, args.output, args.staging)


if __name__ == '__main__':
    main()
