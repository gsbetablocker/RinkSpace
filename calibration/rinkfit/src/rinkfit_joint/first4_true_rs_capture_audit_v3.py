"""Independent strict-RS v3 consumer; reuses the unchanged v1 numerical audit.

The isolated import adds the exact v3 identity, never widens a sensor interval.
Certified zoom and one-scene restrictions are verified against the frozen g0
parent. Numeric envelopes remain engineering certificates, not source coverage.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from .io_utils import file_sha256

BASE_SHA256 = '0b373951e7a1f3e75a0d979a51d5f1d2201faa194e2981f0c5a92531c7d6efca'
OBJECTIVE = 'rinkfit.sourceUV_trueRS_finite_capture.v3'
POLICY = 'sourceUV_trueRS_capture_v3: strict sensor bracket; certified zoom hard domain; scene fixed; derived station/row; original Query not imposed'

def _read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def _ref(p):
    return dict(file=str(Path(p).resolve()), sha256=file_sha256(p))

def _checked(ref):
    p = Path(ref['file'])
    if not p.is_absolute() or file_sha256(p) != ref['sha256']:
        raise ValueError('v3 immutable dependency changed: '+str(p))
    return p

def _base():
    path = Path(__file__).with_name('first4_true_rs_capture_audit_v1.py')
    if file_sha256(path) != BASE_SHA256:
        raise ValueError('Independent numerical audit version changed')
    spec = importlib.util.spec_from_file_location('rinkfit_joint._strict_capture_v3_independent_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OBJECTIVE_POLICIES = {**module.OBJECTIVE_POLICIES, OBJECTIVE: POLICY}
    return module

def parent_enclosures(cp):
    # Parent bytes are authenticated first. Parse only the existing numeric
    # hulls; child CP FNV and all values are independently checked by base.audit.
    with Path(cp).open(encoding='utf8') as f:
        if f.readline().strip() != 'RINKFIT_AL_CHECKPOINT_V1':
            raise ValueError('Parent complete checkpoint header')
        for _ in range(5):
            f.readline()
        token = json.loads(f.readline())
        f.readline()
        count = int(f.readline())
        values = {}
        for _ in range(count):
            name, end = json.JSONDecoder().raw_decode(f.readline())
            row = f.readline().split()
            if name.startswith('rig/') and name.endswith('/numerical_zoom_enclosure'):
                if int(row[0]) != 2:
                    raise ValueError('Parent zoom hull dimension')
                values[name.split('/')[1]] = list(map(float, row[1:]))
    return token, values

def verify_certified_model(model, parent, ranges):
    if set(ranges) != {'main', 'secondary', 'end'}:
        raise ValueError('All three parent zoom hulls required')
    for key in ('source_clock','rigs','shots','linear_control_manifolds','physical_equalities'):
        if model.get(key) != parent.get(key):
            raise ValueError('Frozen projection layout differs: '+key)
    bi = model['scene_block']
    if bi != parent['scene_block'] or model['blocks'][bi]['values'] != parent['blocks'][bi]['values']:
        raise ValueError('The one g0 scene values changed')
    fixed = set(model['shots']['s01']['control_blocks'])
    fixed.update(x['block'] for x in model['shots']['s01'].get('control_coordinates', []))
    for i in fixed:
        if not model['blocks'][i]['constant'] or model['blocks'][i]['values'] != parent['blocks'][i]['values']:
            raise ValueError('S01 authoritative motion changed: '+str(i))
    controls = 0
    for sid in ('s02','s03','s04'):
        shot = model['shots'][sid]
        if 'control_coordinates' in shot:
            raise ValueError('Unqualified direct control mapping: '+sid)
        lo, hi = ranges[shot['rig_id']]
        for j, i in enumerate(shot['control_blocks']):
            b = model['blocks'][i]
            if b['name'] != 'shot/'+sid+'/control/'+str(j) or len(b['values']) != 3:
                raise ValueError('Direct logf coordinate identity changed')
            lower, upper = max(lo,b['lower'][2]), min(hi,b['upper'][2])
            if not lower <= b['values'][2] <= upper:
                raise ValueError('Direct logf exceeds frozen certified enclosure: '+b['name'])
            controls += 1
    return dict(scene_values_unchanged=True, scene_serialized_constant=model['blocks'][bi]['constant'], S01_authoritative_blocks_verified=len(fixed),
                direct_logf_controls_verified=controls, frozen_numeric_enclosures=ranges,
                source_optical_calibration_claimed=False)

def audit(packet_path, candidate, output, *, certified_domain, graph_domain_receipt, baseline_import=None, native_domain_receipt=None):
    base = _base()
    cpdir = Path(candidate)
    binding = _read(cpdir/'source_uv_capture_state_binding.json')
    if binding['objective_id'] != OBJECTIVE:
        raise ValueError('Exact v3 objective identity required')
    domain = _read(certified_domain)
    if domain['schema'] != 'rinkfit.capture_certified_numeric_domain.v1' or domain['scene_policy'] != 'fixed_to_same_g0_scene' or domain['direct_logf_shots'] != ['s02','s03','s04']:
        raise ValueError('Unsupported certified domain policy')
    pm = _checked(domain['parent_model'])
    pc = _checked(domain['parent_complete_checkpoint'])
    pg = _checked(domain['parent_native_g0_receipt'])
    token, ranges = parent_enclosures(pc)
    receipt = _read(pg)
    debt = receipt['true_RS_constraint_debt']
    if token != domain['parent_state_token'] or receipt['complete_state_token'] != token or debt['state_token'] != token or debt['g_max_positive'] != 0 or debt['h_max_abs'] != 0 or receipt['model_sha256'] != domain['parent_model']['sha256']:
        raise ValueError('Same-token authenticated g0 parent required')
    child_token, child_ranges = parent_enclosures(cpdir/'engineering_checkpoint.json')
    if child_token != binding['state_token'] or child_ranges != ranges:
        raise ValueError('Stored certified zoom auxiliary changed')
    domain_result = verify_certified_model(_read(cpdir/'model.json'), _read(pm), ranges)
    graph = _read(graph_domain_receipt)
    if graph['schema'] != 'rinkfit.actual_capture_certified_numeric_domain.v1' or graph['policy_identity'] != file_sha256(certified_domain) or graph['parent_state_token'] != token or not graph['scene_constant'] or graph['scene_values'] != _read(pm)['blocks'][_read(pm)['scene_block']]['values']:
        raise ValueError('Actual native Ceres scene-domain receipt differs')
    hulls = {r['rig']:r for r in graph['fixed_certified_zoom_enclosures']}
    if set(hulls) != set(ranges) or any(not hulls[k]['actual_Ceres_constant'] or hulls[k]['range'] != ranges[k] for k in ranges):
        raise ValueError('Actual native Ceres certified zoom constants differ')
    if graph.get('certificate_expanded') is not False or graph.get('source_UV_or_model_parameter_boxes_changed') is not False:
        raise ValueError('Numeric certificate must not expand source or model domains')
    domain_result['actual_Ceres_domain_receipt'] = _ref(graph_domain_receipt)
    domain_result['Ceres_constness_independently_reimplemented'] = False
    report = base.audit(packet_path, candidate, output, baseline_import=baseline_import, native_domain_receipt=native_domain_receipt)
    if report['complete_CP']['numerical_zoom_enclosures'] != ranges:
        raise ValueError('Stored certified zoom auxiliary changed')
    report.update(schema='rinkfit.first4_true_RS_capture_independent_audit.v3',
                  independent_certified_capture_domain=dict(**domain_result,parent_state_token=token,certificate=_ref(certified_domain)),
                  producer=_ref(__file__), unchanged_numerical_consumer=_ref(Path(base.__file__)))
    report['dependencies'].extend([_ref(certified_domain), _ref(pm), _ref(pc), _ref(pg), _ref(graph_domain_receipt), _ref(__file__)])
    (Path(output)/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report

def main():
    p = argparse.ArgumentParser()
    for key in ('packet','candidate','output','certified-domain','graph-domain-receipt'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--baseline-import')
    p.add_argument('--native-domain-receipt')
    a = p.parse_args()
    r = audit(a.packet,a.candidate,a.output,certified_domain=a.certified_domain,graph_domain_receipt=a.graph_domain_receipt,baseline_import=a.baseline_import,native_domain_receipt=a.native_domain_receipt)
    print(json.dumps(dict(report=str(Path(a.output)/'report.json'),active_rows=r['active_rows'],valid=r['independent_valid_rows'],full_RMS=r['all']['raw_source_2D_UV']['full_RMS_px']),ensure_ascii=False))

if __name__ == '__main__':
    main()
