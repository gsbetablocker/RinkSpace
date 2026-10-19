"""Read-only source-domain audit for the current physical-wide proposals.

The canonical implementation distinguishes the seven-sample masked median from
B1 entry/exit's single physical section. It reconstructs actual source samples,
checks all four bilinear contributors, and reports available excluded context
separately from source contributions actually used.

Report: repair_20261014/wide_quantitative_diagnostics/
physical_wide_bilinear_context_audit.json. A valid source sampling domain alone
never proves the named physical material identity.
"""
from .repair_physical_wide_sampling_context import audit


if __name__ == '__main__':
    audit()
