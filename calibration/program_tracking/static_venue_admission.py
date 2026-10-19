"""Explicit reviewed material extension of an unchanged static control packet.

The caller must supply an independently authorized, exact admission reference.
Hashes authenticate evidence bytes; they do not create material admission.
The old37 parser and every world/camera/source/J implementation stay unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.solver_runtime.source_loss import SourceLossWeights
from .static_venue_source import AdmittedStaticPacket, _sha, _json_sha


def _verified_bytes(reference):
    if not isinstance(reference,dict)or set(reference)!={'path','sha256'}:
        raise ValueError('An explicit exact path/SHA256 evidence reference is required')
    path=Path(reference['path']);expected=reference['sha256']
    if not path.is_absolute()or not isinstance(expected,str)or len(expected)!=64:
        raise ValueError('Absolute evidence path and SHA256 required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Reviewed static evidence bytes changed: '+str(path))
    return raw


def _read(reference):
    return json.loads(_verified_bytes(reference).decode('utf-8-sig'))


def _supported_material(target,operator):
    # These checks only restrict an explicitly reviewed proof; they never
    # infer admission from an entity name, historical use flag or residual.
    kind=target.get('kind');feature=target.get('feature_id')
    return ((kind=='circle_outline'and feature in('A1','A2','A3','A4')and
             operator=='thin circular paint centre, not a filled-house interface or an inferred outer paint side')
        or(kind=='line_segment'and feature=='goal_line_right'and operator=='red goal-line material middle')
        or(kind=='polygon_outline'and feature=='goal_crease_right'and operator=='blue filled-crease outer transition against light ice'))


class VerifiedStaticMaterialPacket:
    """Opt-in proof-bound extension, with the existing factor's packet protocol.

    No default admission is expanded. The base37 must be present unchanged;
    only additional original source IDs individually bound by the approved
    proof may enter. No source values, roles, order, clock or loss are inferred.
    """
    def __init__(self,path,*,expected_sha256,admitted_control_ids,reviewed_admission):
        admission_ref=deepcopy(reviewed_admission);admission=_read(admission_ref)
        if admission.get('schema')!='rinkspace.explicit-static-venue-next44-admission.v4':
            raise ValueError('Explicit next material-source transition schema required')
        source_ref=dict(path=str(Path(path).resolve()),sha256=expected_sha256)
        if admission.get('source_packet')!=source_ref:raise ValueError('Admission belongs to another source packet')
        base_ref=admission['base37_admission'];base_admission=_read(base_ref)
        if base_admission.get('source_packet')!=source_ref:raise ValueError('Base source identity changed')
        base_ids=tuple(base_admission['admitted_control_ids']);requested=tuple(admitted_control_ids)
        if requested!=tuple(admission['admitted_control_ids']):raise ValueError('Explicit admitted row order/set changed')
        if len(set(requested))!=len(requested)or not set(base_ids)<=set(requested):
            raise ValueError('Original base rows must remain present without duplication')
        # Reuse authentication and all original non-A material checks.
        base=AdmittedStaticPacket(path,expected_sha256=expected_sha256,admitted_control_ids=base_ids)
        packet=json.loads(_verified_bytes(source_ref).decode('utf-8-sig'))
        incoming=packet['original_v5_observations'];by_id={r['id']:r for r in incoming}
        ordered=[r['id']for r in incoming if r['id']in set(requested)]
        if ordered!=list(requested):raise ValueError('Original source row order must be retained')
        proof_ref=admission['new_material_operator_proof'];proof=_read(proof_ref)
        if proof.get('schema')!='rinkspace.next-static-wide-A-goal-operator-admission-proof.v1':
            raise ValueError('Current source/material/operator proof required')
        if proof.get('source_packet')!=source_ref or proof.get('base37_admission')!=base_ref:
            raise ValueError('Operator proof source/base identity changed')
        for reference in(proof['actual_C0_operator_closure'],proof['actual_current_safe_frames'],proof['actual_array_reference'],proof['current_geometry_producer']):
            _verified_bytes(reference)
        inventory=_read(proof['RGB_inventory'])
        if inventory.get('source_packet')!=source_ref:raise ValueError('RGB review belongs to another source')
        _verified_bytes(inventory['raw_RGB'])
        extra=set(requested)-set(base_ids);proof_rows=proof['rows'];proof_ids=[r['id']for r in proof_rows]
        if len(set(proof_ids))!=len(proof_ids)or set(proof_ids)!=extra:
            raise ValueError('Every additional ID needs exactly one individual material proof')
        declared=admission['added_group_measurement_operators']
        if {r['id']for r in declared}!=extra or len(declared)!=len(extra):raise ValueError('Admission delta proof changed')
        declared_by_id={r['id']:r for r in declared};bindings=[]
        for evidence in proof_rows:
            identity=evidence['id'];row=by_id[identity];uv=np.asarray(row['pixels_uv'],float)
            if(evidence.get('agent_RGB_material_identity_review')!='completed_original_raw_no_model_overlay'
                    or evidence.get('source_UV_sigma_PTS_unchanged')is not True
                    or not _supported_material(row['target'],evidence.get('current_measurement_operator'))):
                raise ValueError('Reviewed material identity/operator is not closed: '+identity)
            if(row.get('synthetic')is not False or evidence['original_target']!=row['target']
                    or evidence['points']!=len(uv)or evidence['UV_float64_sha256']!=_sha(uv)
                    or evidence['original_sigma_px']!=row['sigma_px']
                    or evidence['literal_weight']!=row.get('weight',1.)):
                raise ValueError('A material proof cannot change original source values: '+identity)
            sigma=np.asarray(row['sigma_px'],float);weight=float(row.get('weight',1.))
            if(uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all()
                    or not np.isfinite(sigma).all()or np.any(sigma<=0)
                    or not np.isfinite(weight)or weight<=0):raise ValueError('Finite unchanged native measurement required')
            declared_row=declared_by_id[identity]
            for field in('points','current_measurement_operator','additional_geometry_dependencies','original_RGB_UV_crop'):
                if declared_row[field]!=evidence[field]:raise ValueError('Admission/proof per-group binding changed: '+identity)
            _verified_bytes(evidence['original_RGB_UV_crop'])
            bindings.append(dict(id=identity,operator=evidence['current_measurement_operator'],UV_float64_sha256=evidence['UV_float64_sha256'],
                geometry_dependencies=evidence['additional_geometry_dependencies'],RGB_crop=evidence['original_RGB_UV_crop']))
        rows=[deepcopy(by_id[identity])for identity in requested]
        points=sum(len(r['pixels_uv'])for r in rows)
        if(len(rows)!=admission['control_group_count']or points!=admission['control_UV_count']
                or 2*points!=admission['control_scalar_rows']):raise ValueError('Declared source delta inventory changed')
        self._rows_json=tuple(json.dumps(r,ensure_ascii=False,allow_nan=False)for r in rows)
        self.reference=base.reference;self.contract=deepcopy(base.contract);self.frame=deepcopy(base.frame)
        self.holdout_ids=tuple(r['id']for r in incoming if r['id']not in set(requested))
        self.advertisement_holdout_count=base.advertisement_holdout_count
        scales=np.concatenate([np.repeat(np.broadcast_to(np.asarray(r['sigma_px'],float),(len(r['pixels_uv']),)),2)for r in rows])
        self.scales=np.frombuffer(scales.tobytes(),dtype=np.float64)
        self.quadratic=np.frombuffer(np.zeros(len(scales),bool).tobytes(),dtype=bool)
        self.loss_weights=SourceLossWeights(np.concatenate([np.full(2*len(r['pixels_uv']),float(r.get('weight',1.)))for r in rows]))
        self.ledger=deepcopy(base.ledger)
        self.ledger.update(ordered_rows=[dict(id=r['id'],target=r['target'],points=len(r['pixels_uv']),UV_float64_sha256=_sha(np.asarray(r['pixels_uv'],float)),sigma_px=r['sigma_px'],literal_Huber2_outer_weight=r.get('weight',1.))for r in rows],
            held_curve_ids=self.holdout_ids,reviewed_material_extension=dict(admission=admission_ref,base37_admission=base_ref,
                proof=proof_ref,bindings=bindings,original_base_identity_sha256=base.identity_sha256,
                actual_C0_operator_closure=proof['actual_C0_operator_closure'],actual_current_safe_frames=proof['actual_current_safe_frames'],
                actual_array_reference=proof['actual_array_reference'],current_geometry_producer=proof['current_geometry_producer'],RGB_inventory=proof['RGB_inventory']))
        self.identity_sha256=_json_sha(self.ledger)
    @property
    def rows(self):
        return tuple(json.loads(row)for row in self._rows_json)


def static_material_packet(path,*,expected_sha256,admitted_control_ids,reviewed_admission=None):
    """Default stays strict legacy37; extension requires an explicit proof ref."""
    if reviewed_admission is None:
        return AdmittedStaticPacket(path,expected_sha256=expected_sha256,admitted_control_ids=admitted_control_ids)
    return VerifiedStaticMaterialPacket(path,expected_sha256=expected_sha256,
        admitted_control_ids=admitted_control_ids,reviewed_admission=reviewed_admission)
