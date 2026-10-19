"""Recover checkpoint history storage without loading a model or fitting.

Preserve the source archive. Verify every other NPY member byte for byte and
round-trip both complete record streams through the public authenticated reader.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from calibration.solver_runtime.checkpoint_history import (
    iter_checkpoint_history,
    migrate_legacy_checkpoint_history,
)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def member_sha(archive, name):
    with archive.open(name) as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record_digest(path, field):
    digest = hashlib.sha256()
    count = 0
    for record in iter_checkpoint_history(path, field=field):
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        digest.update((text + '\n').encode('utf8'))
        count += 1
    return dict(records=count, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-identity', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.report.exists():
        raise FileExistsError('Preserve existing recovery output and report')
    identity = json.loads(args.source_identity.read_text(encoding='utf-8-sig'))
    source_sha = file_sha(args.source)
    if source_sha != identity['state_npz_sha256']:
        raise ValueError('Source accepted checkpoint identity differs')
    print(json.dumps(dict(stage='migrating_authoritative_checkpoint_record_streams')), flush=True)
    result = migrate_legacy_checkpoint_history(args.source, args.destination)
    print(json.dumps(dict(stage='verifying_unchanged_NPY_members_and_complete_record_streams')), flush=True)
    members = []
    with zipfile.ZipFile(args.source) as old, zipfile.ZipFile(args.destination) as new:
        if old.namelist() != new.namelist():
            raise ValueError('Checkpoint member inventory or order differs')
        for name in old.namelist():
            if name in ('history_json.npy', 'reviews_json.npy'):
                continue
            before, after = member_sha(old, name), member_sha(new, name)
            if before != after:
                raise ValueError('Original checkpoint member differs: ' + name)
            members.append(dict(name=name, sha256=before, byte_exact=True))
    streams = {}
    for field in ('history_json', 'reviews_json'):
        restored = record_digest(args.destination, field)
        expected = result['record_semantic_digests'][field]
        if any(restored[key] != expected[key] for key in ('records', 'sha256')):
            raise ValueError('Complete record stream differs: ' + field)
        streams[field] = dict(**restored, semantic_roundtrip=True)
    with np.load(args.destination, allow_pickle=False) as saved:
        q_sha = hashlib.sha256(np.ascontiguousarray(saved['q']).tobytes()).hexdigest()
        if q_sha != identity['q_sha256']:
            raise ValueError('Recovered q differs from accepted identity')
        profiles = [key for key in saved.files if key.startswith('profile__')]
    result.update(source_sha256=source_sha, destination_sha256=file_sha(args.destination),
                  source_identity=str(args.source_identity.resolve()),
                  source_bytes=args.source.stat().st_size,
                  destination_bytes=args.destination.stat().st_size,
                  non_history_members=members, complete_record_stream_verification=streams,
                  q_sha256=q_sha, profile_fields=profiles,
                  source_graph_or_residual_or_J_loaded=False, new_fit_started=False,
                  verification_passed=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(stage='recovered_and_verified', report=str(args.report.resolve()),
                         q_sha256=q_sha, preserved_members=len(members), streams=streams,
                         source_bytes=result['source_bytes'], destination_bytes=result['destination_bytes'])),
          flush=True)


if __name__ == '__main__':
    main()
