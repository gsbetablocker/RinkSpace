"""Replay the startup certificate gate on saved actual85 source audit values.

No source operator is numerically reevaluated here. The calling production
startup must still supply its newly loaded actual factors and complete target.
"""
from copy import deepcopy
import hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
from structured_programme_source_candidate.reviewed_manual_append_restore import prepare_reviewed_manual_append_state

SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
BINDING=SYS/'same_q27fa_review85_final87_compact_startup_binding.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def target(manifest):
    raw=json.dumps(manifest['source_contract'],sort_keys=True,separators=(',',':'),allow_nan=False)
    return dict(parameter_count=manifest['full_parameters'],source_rows=manifest['source_rows'],
        source_contract_sha256=hashlib.sha256(raw.encode()).hexdigest())

def actual_receipt_carriers(directory, binding=BINDING, layout_path=None):
    """Small binding fixture; all identity/cap numbers come from real receipts."""
    scope=read(binding);audit=read(scope['same_q_public_audit']['path']);values=read(scope['exact_caps_values']['path'])
    parent=read(audit['old_caps_reference']['path']);old=read(audit['parent_manifest_reference']['path'])
    with np.load(audit['complete_state_reference']['path'],allow_pickle=False)as z:q=z['q'].copy()
    manifest=deepcopy(old);manifest.update(source_rows=audit['new_complete_source_scalar_count'],
        physical_domain_initial_q_sha256=audit['complete_q_sha256'])
    native_ref=scope['source_qualified_native_packet_sources']['additional_real_native_ice_curves']
    qualified=load_source_point_qualification(native_ref['path'],read(native_ref['path']),source_family='additional_native_ice_curves')
    manifest['additional_real_native_ice_curves']=dict(packet_path=native_ref['path'],packet_sha256=native_ref['sha256'],
        source_only_review_qualification=qualified.contract)
    board=audit['new_source_options']['--additional-board-source']
    manifest['additional_real_native_board_edges']=dict(packet_path=board['path'],packet_sha256=board['sha256'])
    layout=read(layout_path or SYS/'original_absolute_active_compaction_complete_layout_verification_20261010.json')
    for field,entry in scope['admitted_operator_sources'].items():
        reference=entry['reference'];bound=dict(packet_path=reference['path'],packet_sha256=reference['sha256'])
        if field=='static_motion_source_role_retirement':
            bound=deepcopy(layout['static_motion_contract'])
        elif field=='original_absolute_input_compaction':bound=deepcopy(layout['source_compaction'])
        manifest[field]=bound
    direct=manifest['source_contract']['direct_manual_centre_secondary']
    direct['authority']=audit['new_source_options']['--direct-human-truth-authority']
    direct['added_scalar_rows']+=audit['added_direct_manual_centre_scalars']
    if scope.get('admitted_engineering_prior_append')is not None:
        prior=read(scope['admitted_engineering_prior_append']['actual_same_q_public_audit']['path'])['contract']
        manifest['principal_point_engineering_prior']=deepcopy(prior)
        manifest['source_contract'].update(principal_point_engineering_prior=deepcopy(prior),
            original_source_scalar_prefix_preserved=True,original_source_scalars_before_PP6=manifest['source_rows']-6,
            source_rows=manifest['source_rows'])
    factors={}
    for family,rows in values['families'].items():
        authority=parent['families'][family]['authority'] if family=='wide' else audit['new_source_options'][
            '--direct-human-truth-authority' if family=='native' else '--user-confirmed-source-truth-authority']
        contract=dict(authority=authority)
        if family=='native':
            for field,option in [('full_stroke_guard','--direct-human-full-stroke-guard'),
                    ('source_2D_stroke_cap_support','--direct-human-stroke-cap-support'),
                    ('practical_manual_location_envelope','--practical-human-location-envelope')]:
                contract[field]=audit['new_source_options'][option]
        factors[family]=SimpleNamespace(contract=contract,ids=rows['group_ids'],
            uncertainties=np.asarray(rows['uncertainties'],float),violations=np.asarray(rows['violations_px'],float))
    manifest_path=Path(directory)/'receipt_carrier_binding_manifest_DIAGNOSTIC_ONLY.json';write(manifest_path,manifest)
    return SimpleNamespace(binding=binding,scope=scope,audit=audit,values=values,parent=parent,manifest=manifest,
        manifest_path=manifest_path,q=q,factors=factors,parent_path=Path(audit['old_caps_reference']['path']))

def invoke(fixture):
    captured={}
    def capture(path,packet):captured['packet']=packet
    output=prepare_reviewed_manual_append_state(fixture.binding,fixture.parent_path,fixture.manifest_path,
        fixture.manifest,fixture.q,fixture.factors,group_ids=lambda f,family:f.ids,
        violation_buffer=lambda f,q,family:f.violations,uncertainties=lambda f,family:f.uncertainties,
        source_target=target,role_target=lambda manifest:dict(receipt_gate_test_only=True),write=capture)
    return captured['packet'],output

def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='review85_binding_',dir=SYS)as directory:
        f=actual_receipt_carriers(directory);packet,_=invoke(f);protected=read(f.scope['protected_same_q_caps']['path'])
        for family,values in protected['families'].items():
            n=len(values['group_ids'])
            if any(packet['families'][family][k][:n]!=values[k]for k in ('group_ids','limits','uncertainties')):
                raise ValueError('Protected82 output prefix changed')
        checks['actual85_saved_values_bind_standard_packet_with_exact_protected82']=True
        for name,change in [
            ('different_complete_q_rejected',lambda f:f.q.__setitem__(0,np.nextafter(f.q[0],np.inf))),
            ('different_admitted_source_bytes_rejected',lambda f:f.manifest['additional_real_native_ice_curves'].__setitem__('packet_sha256','0'*64)),
            ('different_native_group_identity_rejected',lambda f:f.factors['native'].ids.__setitem__(0,'another_physical_source')),
            ('different_actual_source_violation_rejected',lambda f:f.factors['native'].violations.__setitem__(-1,f.factors['native'].violations[-1]+.01)),
            ('different_actual_uncertainty_rejected',lambda f:f.factors['native'].uncertainties.__setitem__(0,99.)),
        ]:
            f=actual_receipt_carriers(directory);change(f)
            try:invoke(f)
            except ValueError as error:checks[name]=str(error)
            else:raise ValueError('Startup gate did not reject '+name)
    report=dict(schema='rinkspace.actual-review85-startup-binding-certificate-replay.v1',passed=True,
        complete_q_sha256=packet['complete_q_sha256'],group_count=sum(len(v['group_ids'])for v in packet['families'].values()),
        protected_group_count=82,actual_source_rows=packet['source_target']['source_rows'],checks=checks,
        actual_saved_source_audits_used=True,numerical_factors_recomputed=False,
        complete_startup_manifest_fixture_is_not_the_actual_production_source_contract=True,
        production_startup_must_recompute_actual_factor_violations_and_bind_its_full_manifest=True,
        production_restoration_packet_written=False,solver_started=False)
    write(SYS/'actual_review85_startup_binding_certificate_replay.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
