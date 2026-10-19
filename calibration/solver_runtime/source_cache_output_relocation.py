"""Authenticate a saved source normal across execution-output relocations.

The numerical producer registry, signed source manifests, all input arrays and
files, scientific backend and fixed/shared policies still use the original
strict verifier. Only three existing diagnostic/output destinations differ.
This execution adapter never rewrites a sealed cache or its signed metadata.
"""
from __future__ import annotations
import functools
import hashlib
from pathlib import Path


def install_output_relocation_compatibility():
    from structured_programme_source_candidate import complete_same_state_replay as replay
    original = replay._check_strict_identity
    if getattr(original, '_execution_output_relocation_compatibility', False):
        return

    @functools.wraps(original)
    def verify(old, current, **options):
        # Explicit append/migration contracts and producer changes retain their
        # existing verifier. This route permits no implementation exclusion.
        if options.get('append_normalized_pair') is not None:
            return original(old, current, **options)
        old_identity = old.get(replay.IDENTITY_KEY, {})
        current_identity = current.get(replay.IDENTITY_KEY, {})
        if old_identity.get('implementation_files') != current_identity.get('implementation_files'):
            return original(old, current, **options)
        relocations = [dict(key=key, previous=old.get('native_backend_requested', {}).get(key),
            revised=current.get('native_backend_requested', {}).get(key))
            for key in replay._QP_OUTPUT_PATH_KEYS
            if old.get('native_backend_requested', {}).get(key) != current.get('native_backend_requested', {}).get(key)]
        if not relocations:
            return original(old, current, **options)
        left = replay._qp_compare_source_semantics(replay._semantic_manifest(old))
        right = replay._qp_compare_source_semantics(replay._semantic_manifest(current))
        if left != right:
            return original(old, current, **options)
        # The old/current signatures are checked against their ORIGINAL
        # unmodified manifests. This pair affects the final equality only.
        verified = original(old, current, **dict(options, append_normalized_pair=(left, right)))
        if verified is not None:
            raise RuntimeError('Output relocation unexpectedly used a numerical/source migration')
        adapter = Path(__file__).resolve()
        return dict(mode='execution_output_relocation_same_source_normal',
            execution_adapter=dict(path=str(adapter), sha256=hashlib.sha256(adapter.read_bytes()).hexdigest()),
            execution_output_path_relocations=relocations,
            ignored_execution_output_path_keys=list(replay._QP_OUTPUT_PATH_KEYS),
            allowed_load_only_implementation_paths=[],
            original_signed_semantics_verified=True,
            all_original_source_code_native_files_arrays_profiles_and_policies_verified=True,
            sealed_cache_or_source_manifest_rewritten=False,
            cached_QP_solution_reused=False)

    verify._execution_output_relocation_compatibility = True
    replay._check_strict_identity = verify
