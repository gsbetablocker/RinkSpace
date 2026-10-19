"""Exact source pairs whose numerical model is unchanged by solver repairs.

This allowlist does not waive warm-start geometry/provenance checks. Every
other source change, including a later edit to either endpoint, is rejected.
Evidence and before-source snapshots are in solver_acceleration/round9.
"""
import hashlib


EQUIVALENT_SOURCES = {
    "surface_pair.py": (
        "334be777dfd1d6741a557632d8d9c47aaa23db8220cd516ada18bd2d7834cf6d",
        "8aa7c051f8e4f0da117c65353072226da992a5d0bd887d9855cbc4b6e42ae931",
        "Same fixed-plane pixel residual; analytic Brown15 derivatives on a "
        "verified stable inverse branch, original numerical derivative fallback",
    ),
    "projection.py": (
        "0d3912f58eba72e302fb0eb1e1d01f7cf39ee20c12a55d34440179345ae768f8",
        "ae6261d26704b0732712a4002e987455f8959e8f414362b5ea3b17abd39715c4",
        "Exact-input, verification-scoped reuse of original OpenCV projections; "
        "camera/state definitions and ordinary projection arithmetic unchanged",
    ),
}


def reviewed_source_change(name, saved, current, expected_hash):
    pair = EQUIVALENT_SOURCES.get(name)
    if pair is None or expected_hash != pair[0]:
        return None
    saved_hash = hashlib.sha256(saved.read_bytes()).hexdigest()
    current_hash = hashlib.sha256(current.read_bytes()).hexdigest()
    if saved_hash != pair[0] or current_hash != pair[1]:
        return None
    return dict(saved_source=str(saved), saved_sha256=saved_hash,
                current_sha256=current_hash, equivalence=pair[2],
                evidence="solver_acceleration/round9; frozen residual/Jacobian and solve comparisons")
