"""Adapt a partially processed delta to its literal processed source scope."""
from __future__ import annotations

import json
from pathlib import Path
from types import FunctionType

from . import strict_source_dataset_append as common
from .strict_absolute_source import StrictAbsoluteSourcePack
from .strict_source_fit_dataset import reference


def append_partial_reviewed_deltas(base_reference, delta_references, audit_reference, output_directory):
    """Reuse the frozen append validator with an explicit processed-scope view.

    A delta may declare the whole programme but have read only one segment.
    The original global flags remain false in its immutable file. Only its
    exactly bound, actually processed frames form this local view; the parent
    supplies the complete programme clock. No unprocessed frame gets a label.
    """
    base = json.loads(Path(base_reference['path']).read_text(encoding='utf8'))
    full_clock = set(base['expected_actual_pts'])
    scopes = []

    class ProcessedDeltaView(StrictAbsoluteSourcePack):
        def __init__(self, path, *, expected_sha256):
            super().__init__(path, expected_sha256=expected_sha256)
            original = self.metadata
            processed = original['processed_actual_pts']
            if (not set(original['expected_actual_pts']).issubset(full_clock)
                    or set(processed) != set(self.frame_bindings)
                    or not processed or any(row['actual_pts'] not in self.frame_bindings for row in self.rows)):
                raise ValueError('Delta processed scope must have exact native RGB bindings for every actual row')
            scopes.append(dict(source_pack=dict(path=str(Path(path).resolve()), sha256=expected_sha256),
                original_declared_actual_exposures=len(original['expected_actual_pts']),
                original_all_exposures_processed=original['all_exposures_processed'],
                original_all_exposures_exact_native_BGR_bound=original['all_exposures_exact_native_BGR_bound'],
                local_processed_actual_pts=list(processed), local_scope_complete_and_exactly_bound=True))
            # These booleans describe only the explicitly listed local domain.
            # The original file, original metadata and full parent stay intact.
            self.metadata = dict(original, expected_actual_pts=list(processed),
                all_exposures_processed=True, all_exposures_exact_native_BGR_bound=True)

    # Clone the unchanged validator's namespace, rather than monkey-patching a
    # module or relaxing its clock/RGB/HELD/preflight checks for other callers.
    namespace = dict(common.append_reviewed_deltas.__globals__,
                     StrictAbsoluteSourcePack=ProcessedDeltaView)
    append = FunctionType(common.append_reviewed_deltas.__code__, namespace)
    result = append(base_reference, delta_references, audit_reference, output_directory)
    dataset_path = Path(result['dataset_reference']['path'])
    manifest = json.loads(dataset_path.read_text(encoding='utf8'))
    manifest['source_review_authorities'].append(reference(__file__))
    manifest['appended_delta_processed_scope_views'] = scopes
    manifest['partial_delta_is_not_full_programme_processing_claim'] = True
    dataset_path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':'),
                                       allow_nan=False), encoding='utf8')
    receipt_path = Path(result['append_receipt_reference']['path'])
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    receipt.update(revised_dataset_reference=reference(dataset_path),
        partial_original_delta_processed_scope_views=scopes,
        original_pack_global_processing_flags_unchanged=True,
        full_clock_and_actual_RGB_bindings_still_checked_against_parent=True,
        partial_scope_adapter_reference=reference(__file__))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, separators=(',', ':'),
                                      allow_nan=False), encoding='utf8')
    return dict(dataset_reference=reference(dataset_path), append_receipt_reference=reference(receipt_path))
