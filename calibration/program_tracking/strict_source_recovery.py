"""One append-only recovery journal for an actual source replay.

Committed records contain the already-checked source rows, exact proof
templates and numeric readouts. They permit saving after an interrupted final
serialization; they never admit an unknown candidate or replace source RGB.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np


class SourceAdmissionRecoveryJournal:
    def __init__(self, path, writer):
        self.path = Path(path).resolve()
        if self.path.exists():
            raise ValueError('Preserve an earlier source recovery journal')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL)')
        self.row_count = self.proof_count = self.exclusion_count = 0
        self.processed = set()
        self.frame_bindings = {}
        self._record('manifest', dict(source=writer.source,
            expected_actual_pts=writer.expected_pts.tolist(),
            original_references=writer.original_references,
            legacy_pixel_families=writer.legacy_pixel_families))
        self.connection.commit()

    def _record(self, kind, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
            separators=(',', ':')).encode('utf8')
        compressed = zlib.compress(payload, 1)
        self.connection.execute('INSERT INTO records (kind,payload,sha256) VALUES (?,?,?)',
            (kind,compressed,hashlib.sha256(compressed).hexdigest()))

    def checkpoint(self, writer):
        # Numeric arrays remain binary; NaN uncertainty is preserved literally
        # rather than becoming a fabricated finite solver weight.
        for index in range(self.row_count, len(writer.rows)):
            values = BytesIO()
            np.savez_compressed(values, uv=writer.uv[index], sigma_px=writer.sigma[index],
                source_sides_uv=writer.sides[index], raw_candidate_uv=writer.raw_uv[index],
                source_side_quantiles_uv=writer.side_quantiles[index],
                original_point_indices=writer.original_indices[index],
                source_proof_template_index=writer.proof_indices[index])
            payload = values.getvalue()
            self.connection.execute('INSERT INTO records (kind,payload,sha256) VALUES (?,?,?)',
                ('arrays',payload,hashlib.sha256(payload).hexdigest()))
            self._record('row', writer.rows[index])
        for proof in writer.proof_templates[self.proof_count:]:
            self._record('proof', proof)
        for exclusion in writer.exclusions[self.exclusion_count:]:
            self._record('exclusion', exclusion)
        frames = {str(pts):value for pts,value in writer.frame_bindings.items()
            if self.frame_bindings.get(pts) != value}
        processed = sorted(writer.processed_pts-self.processed)
        self._record('checkpoint', dict(frame_bindings=frames, processed_actual_pts=processed,
            original_references=writer.original_references,
            row_count=len(writer.rows), proof_count=len(writer.proof_templates)))
        self.connection.commit()
        self.row_count = len(writer.rows)
        self.proof_count = len(writer.proof_templates)
        self.exclusion_count = len(writer.exclusions)
        self.processed = set(writer.processed_pts)
        self.frame_bindings = dict(writer.frame_bindings)

    def close(self):
        self.connection.close()


def restore_source_admission_journal(path, output_path):
    """Restore only committed, byte-checked rows through the public FIT gate."""
    from .strict_absolute_source import StrictAbsolutePackWriter
    from .strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA
    connection = sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro', uri=True)
    rows, arrays, proofs, exclusions = [], [], [], []
    frames, processed, references = {}, set(), []
    manifest = checkpoint = None
    for _,kind,payload,sha in connection.execute('SELECT id,kind,payload,sha256 FROM records ORDER BY id'):
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Source recovery journal record changed')
        if kind == 'arrays':
            with np.load(BytesIO(payload), allow_pickle=False) as values:
                arrays.append({key:values[key].copy() for key in values.files})
            continue
        value = json.loads(zlib.decompress(payload).decode('utf8'))
        if kind == 'manifest':
            if manifest is not None:raise ValueError('Competing source recovery manifests')
            manifest = value
        elif kind == 'row':rows.append(value)
        elif kind == 'proof':proofs.append(value)
        elif kind == 'exclusion':exclusions.append(value)
        elif kind == 'checkpoint':
            checkpoint = value
            frames.update({int(pts):binding for pts,binding in value['frame_bindings'].items()})
            processed.update(value['processed_actual_pts'])
            references = value['original_references']
        else:raise ValueError('Unknown source recovery record')
    connection.close()
    if (manifest is None or checkpoint is None or len(arrays) != len(rows)
            or len(rows) != checkpoint['row_count'] or len(proofs) != checkpoint['proof_count']):
        raise ValueError('Incomplete committed source admission checkpoint')
    writer = StrictAbsolutePackWriter(output_path, **dict(manifest,
        original_references=references))
    for pts,binding in frames.items():
        writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'], binding.get('shot_id'))
    for row,values in zip(rows,arrays):
        ids = values.pop('source_proof_template_index')
        endpoints = [dict(proofs[int(code)], uv=point.tolist()) for code,point in zip(ids,values['uv'])]
        review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
            source=manifest['source'], reviewed_endpoints=endpoints))
        if writer.add_row(dict(row, **values), review) != len(values['uv']):
            raise ValueError('Committed source row lost its exact admission proof')
    writer.exclusions.extend(exclusions)
    for pts in sorted(processed):writer.mark_exposure_processed(pts)
    return writer
