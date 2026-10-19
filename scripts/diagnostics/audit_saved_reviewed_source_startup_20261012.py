"""Audit a failed startup's saved inputs; never reconstruct or evaluate its graph."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime import truth_source_location_revision as common
from calibration.solver_runtime.reviewed_source_continuation import authenticate_reviewed_source_revision
import numpy as np


def main():
    manifest_path, proof_path, output_path = map(Path, sys.argv[1:])
    manifest = common._read(manifest_path); proof = common._read(proof_path)
    state_path = common._checked(proof['original_truth_state']); state = common._read(state_path)
    contracts = dict(native=manifest['highest_direct_human_hard_truth']['native'],
        wide=manifest['highest_direct_human_hard_truth']['wide'],
        confirmed=manifest['highest_user_confirmed_native_sources'])
    # The real startup already constructed these factors before saving this
    # manifest. This audit supplies only their saved authority/ID inventory;
    # it does not claim to rerun their physical guards or source residuals.
    inventory = {}
    for family, contract in contracts.items():
        authority = common._read(common._checked(contract['authority']))
        records = [dict(**{'record' if family == 'wide' else 'row': dict(id=r['id'])})
            for r in authority['records']]
        inventory[family] = SimpleNamespace(contract=contract, records=records)
    with np.load(common._checked(state['complete_state_reference']), allow_pickle=False) as z:
        q = z['q'].copy()
    try:
        result = authenticate_reviewed_source_revision(proof_path, state_path, q, manifest, inventory)
    except Exception as error:
        tb = error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name == 'authenticate_reviewed_source_revision':
                local = tb.tb_frame.f_locals
                original = local['original']['source_contract']
                actual = local.get('normalized', local.get('current'))
                differences = []
                def compare(a, b, path=()):
                    if common._same(a, b):return
                    if isinstance(a, dict) and isinstance(b, dict):
                        for key in sorted(a.keys() | b.keys()):compare(a.get(key), b.get(key), path + (key,))
                    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                        for i, (x, y) in enumerate(zip(a, b)):compare(x, y, path + (i,))
                    else:
                        differences.append(dict(path=list(path), original=a, actual=b))
                if actual is not None:compare(original, actual)
                common._write(output_path, dict(schema='rinkspace.saved-reviewed-source-startup-data-audit.v1',
                    status='rejected', error=str(error), manifest_reference=common.reference(manifest_path),
                    proof_reference=common.reference(proof_path), source_contract_differences=differences,
                    no_graph_residual_J_or_physical_guard_evaluated=True))
            tb = tb.tb_next
        raise
    report = dict(schema='rinkspace.saved-reviewed-source-startup-data-audit.v1', status='passed',
        manifest_reference=common.reference(manifest_path), proof_reference=common.reference(proof_path),
        actual_saved_contract_and_authority_inventory_authenticated=True,
        runtime_factor_constructors_already_passed_in_this_saved_startup=True,
        no_graph_residual_J_or_physical_guard_evaluated=True,
        full_runtime_authentication_still_required_on_next_startup=True,
        migration=result['migration'])
    reference = common._write(output_path, report)
    print(json.dumps(reference, ensure_ascii=False))


if __name__ == '__main__':main()
