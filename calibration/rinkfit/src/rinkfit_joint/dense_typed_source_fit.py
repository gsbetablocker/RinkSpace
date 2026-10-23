"""Future dense inputs require positive finite source-instance evidence.

Historical immutable bindings keep their original reader. New callers use this
factory; current pixel/profile success never grants a physical entity identity.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,json,types
import numpy as np
from . import dense_source_fit as legacy
from .io_utils import read_json,write_json,file_sha256,object_sha256
from .dense_joint_packet import _array
from .dense_joint_run import bind_current_generation_rows
from .dense_fit_feedback import _one_frame

REASONS=dict(legacy.REASONS,**{})
REASONS.update({10:'current_source_instance_explicitly_unknown',11:'positive_finite_source_instance_support_not_issued',12:'protected_or_unresolved_exposure_role'})
POSITIVE_KINDS={1:'actual_source_connected_instance',2:'verified_static_track_to_confirmed_instance',3:'human_source_identity'}


def _ref(path):
    path=Path(path).resolve();return dict(file=str(path),sha256=file_sha256(path))


def _pin(ref):
    from .exact_dependency_aliases import resolve_dependency
    return resolve_dependency(ref)


def _raw_identity_unchanged(current,prior):
    for key in ('uv','native_pts','shot_index','material_index'):
        if current['arrays'][key]!=prior['arrays'][key]:raise ValueError('Typed instance evidence raw identity arrays differ')
    if current['dense_source_binding']!=prior['dense_source_binding']:raise ValueError('Typed source registry differs from current original identities')


def build_instance_registry(packet,qualification,reviews,output):
    """Bind small reviewed finite scopes to current immutable raw IDs and UVs.

    This is deliberately partial: unreviewed rows get no positive permission.
    Source evidence is pinned; original source/packet arrays are referenced only.
    """
    packet=Path(packet).resolve();qualification=Path(qualification).resolve();doc=read_json(packet);qual=read_json(qualification)
    if _pin(qual['source_packet'])!=packet:raise ValueError('Current qualification binds another packet')
    a={key:_array(doc['arrays'][key],packet.parent)for key in ('uv','native_pts','shot_index','material_index','admission','fit_role')}
    n0=doc['dense_source_binding']['row_start'];ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],packet.parent)
    rows={};refs=[];scopes=[]
    for value in reviews:
        rp=Path(value).resolve();r=read_json(rp);refs.append(_ref(rp));prior=read_json(_pin(r['source_packet']));_raw_identity_unchanged(doc,prior)
        if r['schema']=='rinkfit.typed_dense_source_instance_support_audit.v1':
            if r.get('paired_material_profile_can_grant_world_identity')is not False or r.get('unreviewed_source_rows_default_to_confirmed')is not False:raise ValueError('Instance support must be explicit and finite')
            cols={key:_array(r['arrays'][key],rp.parent)for key in ('packet_row','source_store_ordinal','instance_state','instance_support_kind')}
            refs.extend([r['actual_review_consumer'],r['actual_source_review_report']]);_pin(r['actual_review_consumer']);_pin(r['actual_source_review_report'])
            records=[]
            for i,ri in enumerate(cols['packet_row']):
                ri=int(ri)
                if ri<n0 or int(ordinal[ri-n0])!=int(cols['source_store_ordinal'][i]):raise ValueError('Reviewed source ordinal changed')
                records.append(dict(packet_row=ri,instance_state=int(cols['instance_state'][i]),instance_support_kind=int(cols['instance_support_kind'][i])))
            scope=r['audited_scope'];scopes.append(dict(review=_ref(rp),scope=scope))
        elif r['schema']=='rinkfit.actual_finite_source_instance_row_review.v1':
            if r.get('old_guides_or_8a_not_positive_identity_authority')is not True:raise ValueError('Per-row instance review must reject guide authority')
            records=r['per_row'];scope=dict(shot_id=r['shot_id'],native_pts=r['native_pts'],raw_rows=r['raw_scope_rows']);scopes.append(dict(review=_ref(rp),scope=scope))
        else:raise ValueError('Unsupported actual finite instance evidence schema')
        for v in records:
            ri=int(v['packet_row']);state=int(v['instance_state']);kind=int(v['instance_support_kind']);sid=doc['source_row_ids'][ri];uv=a['uv'][ri].tolist();shot=doc['shot_ids'][int(a['shot_index'][ri])];pts=int(a['native_pts'][ri])
            if shot!=scope['shot_id']or pts!=int(scope['native_pts']):raise ValueError('Evidence is outside its finite audited exposure')
            if state not in (0,1)or(state==1 and kind not in POSITIVE_KINDS)or(state==0 and kind!=0):raise ValueError('Instance support state/kind conflict')
            if state==1 and not(a['admission'][ri]and a['fit_role'][ri]):raise ValueError('Positive evidence cannot silently revive current unknown')
            if 'selected_row_id'in v and(v['selected_row_id']!=sid or v['source_uv']!=uv):raise ValueError('Per-row review changes original source identity/UV')
            item=dict(selected_row_id=sid,packet_row=ri,source_store_ordinal=int(ordinal[ri-n0]),shot_id=shot,native_pts=pts,source_uv=uv,material_index=int(a['material_index'][ri]),instance_state=state,instance_support_kind=kind,evidence=_ref(rp))
            if sid in rows and rows[sid]!=item:raise ValueError('Conflicting positive instance evidence')
            rows[sid]=item
    qualified=read_json(_pin(doc['dense_qualification_binding']));parts=qualified['qualified_source_parts']
    links=bind_current_generation_rows(packet,[v['packet_row']for v in rows.values()]);frames={}
    for item in rows.values():
        link=links[str(item['packet_row'])];part=next(i for i,p in enumerate(parts)if p['reference']==link['rows_ref']);key=(part,item['shot_id'],item['native_pts'])
        if key not in frames:frames[key]=_one_frame(link['rows_ref'],item['shot_id'],item['native_pts'])[0]
        current=frames[key][item['selected_row_id']]
        if current['uv']!=item['source_uv']or current['shot_id']!=item['shot_id']or current['native_pts']!=item['native_pts']:raise ValueError('Reviewed current source-part semantics redirect original UV or clock')
        item.update(source_part_index=part,source_point_id=current['source_point_id'],feature_id=current['feature_id'],measurement_definition=current['measurement_definition'],physical_surface=current['physical_surface'])
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Immutable typed instance registry already exists')
    result=dict(schema='rinkfit.dense_positive_finite_source_instance_registry.v1',source_packet=_ref(packet),source_geometry_qualification=_ref(qualification),actual_consumer=_ref(__file__),current_source_parts=parts,current_source_part_semantics_verified=True,audited_scopes=scopes,evidence_refs=refs,rows=list(rows.values()),reviewed_rows=len(rows),source_confirmed_rows=sum(v['instance_state']==1 for v in rows.values()),world_instance_unknown_rows=sum(v['instance_state']==0 for v in rows.values()),unreviewed_rows_default_to_confirmed=False,paired_profile_or_repeated_pixels_grant_identity=False,source_UV_or_current_packet_modified=False,source_pixels_decoded=0,statistical_precision_issued=False)
    write_json(out,result);return _ref(out)


class SourceInstanceGate:
    def __init__(self,registry):
        self.path=Path(registry).resolve();self.ref=_ref(self.path);self.doc=read_json(self.path)
        if self.doc.get('schema')!='rinkfit.dense_positive_finite_source_instance_registry.v1'or self.doc.get('unreviewed_rows_default_to_confirmed')is not False or self.doc.get('paired_profile_or_repeated_pixels_grant_identity')is not False:raise ValueError('Positive finite instance registry required')
        _pin(self.doc['actual_consumer']);_pin(self.doc['source_packet']);_pin(self.doc['source_geometry_qualification'])
        for ref in self.doc['evidence_refs']:_pin(ref)
        self.rows={v['selected_row_id']:v for v in self.doc['rows']}
        self.source_parts=self.doc.get('current_source_parts',[]);self._frames={}
        if len(self.rows)!=self.doc['reviewed_rows']:raise ValueError('Ambiguous reviewed source identities')

    def check(self,row,material_index):
        qualification=row.get('physical_instance_qualification')
        if row.get('source_fit_eligible')is False or (isinstance(qualification,str)and qualification in ('unknown','pending','unverified')):return 10
        support=self.rows.get(row['selected_row_id'])
        if support is None:return 11
        if support['instance_state']!=1:return 10
        if support['instance_support_kind']not in POSITIVE_KINDS:raise ValueError('Unrecognized positive instance authority')
        if (support['shot_id'],support['native_pts'],support['source_uv'],support['material_index'])!=(row['shot_id'],int(row['native_pts']),row['uv'],int(material_index)):raise ValueError('Positive evidence redirects a source row')
        if self.source_parts:
            part=int(support['source_part_index']);key=(part,support['shot_id'],support['native_pts'])
            if key not in self._frames:self._frames[key]=_one_frame(self.source_parts[part]['reference'],support['shot_id'],support['native_pts'])[0]
            current=self._frames[key][row['selected_row_id']]
            if current.get('source_fit_eligible')is False or current.get('physical_instance_qualification')=='unknown':return 10
            for field in ('source_point_id','uv','feature_id','measurement_definition','physical_surface','shot_id','native_pts'):
                if current[field]!=row[field]:raise ValueError('Candidate differs from the current reviewed source-part record: '+field)
        return 0

    def decision(self,row,crosswalk,parent_keys=None):
        code,d=legacy._decision(row,crosswalk,parent_keys)
        if code:return code,d
        code=self.check(row,d['material_index']);return code,d


def build_typed_dense_input(storage_bindings,model_path,out,*,instance_registry,project_root,frame_roles,parent_packet=None):
    """Write the real existing columnar input ABI with positive-only eligibility.

    Every source row retains a reason in the full denominator. No UV table,
    fabricated covariance, or source pixel decoder is created here.
    """
    from .role_guard import FrameRoleGuard,DENY_TRAIN
    role_guard=FrameRoleGuard(frame_roles);role_ref=_ref(frame_roles)
    if role_guard.document.get("schema")!="rinkfit.joint_source_frame_protection.v1" or role_guard.document["source_video_sha256"]!=read_json(model_path)["source_clock"]["source_video_sha256"]:raise ValueError("Frame roles are not bound to the current source video")
    if parent_packet:
        parent=read_json(parent_packet);pa={k:_array(parent['arrays'][k],Path(parent_packet).parent)for k in ('admission','fit_role','native_pts','shot_index')}
        for sid,pts in set((parent['shot_ids'][int(sh)],int(pts))for ad,fr,pts,sh in zip(pa['admission'],pa['fit_role'],pa['native_pts'],pa['shot_index'])if ad and fr):
            role_guard.assert_training_pixels([pts],shot_id=sid)
    gate=SourceInstanceGate(instance_registry)
    if gate.doc.get('current_source_part_semantics_verified')is not True or not gate.source_parts:raise ValueError('Future production input requires current qualified source-part semantics')
    expected={object_sha256(p['reference'])for p in gate.source_parts};actual={object_sha256(g['selected_rows'])for g in read_json(storage_bindings)['generations'].values()}
    if actual!=expected:raise ValueError('Future candidate source bindings are not the current qualified semantic parts')
    def decision(row,crosswalk,parent_keys=None):
        code,descriptor=gate.decision(row,crosswalk,parent_keys)
        if code:return code,descriptor
        r=role_guard.roles.get(int(row['native_pts']))
        if r is None or r['shot_id']!=row['shot_id'] or r['role']in DENY_TRAIN:return 12,descriptor
        return 0,descriptor
    def write(path,value):
        if value.get('schema')=='rinkfit.dense_source_fit_input.v1':
            value=dict(value,instance_support_gate=gate.ref,positive_finite_instance_required=True,legacy_material_geometry_reader=_ref(legacy.__file__),frame_role_authority=role_ref)
        elif value.get('schema')=='rinkfit.dense_finite_material_source_admission.v1':
            value=dict(value,instance_support_gate=gate.ref,positive_finite_instance_required=True,unreviewed_instance_is_unknown=True,profile_or_repeat_pixel_is_not_instance_authority=True,frame_role_authority=role_ref)
        return write_json(path,value)
    namespace=dict(legacy.__dict__,_decision=decision,REASONS=REASONS,write_json=write,__file__=__file__)
    function=types.FunctionType(legacy.build_dense_input.__code__,namespace,legacy.build_dense_input.__name__,legacy.build_dense_input.__defaults__,legacy.build_dense_input.__closure__)
    function.__kwdefaults__=legacy.build_dense_input.__kwdefaults__
    return function(storage_bindings,model_path,out,project_root=project_root,parent_packet=parent_packet)


class TypedDenseSourceFitInput(legacy.DenseSourceFitInput):
    def __init__(self,binding_path,*,require_frame_roles=True):
        super().__init__(binding_path)
        from .role_guard import FrameRoleGuard
        role=self.binding.get("frame_role_authority")
        if require_frame_roles and role is None:raise ValueError("Fresh typed input requires a pinned frame-role authority")
        if role:self._pin(role)
        self.frame_role_guard=FrameRoleGuard(role['file'])if role else None
        if self.binding.get('positive_finite_instance_required')is not True:raise ValueError('This future reader requires an explicit positive finite instance gate')
        self._pin(self.binding['legacy_material_geometry_reader']);self._pin(self.binding['instance_support_gate']);self.instance_gate=SourceInstanceGate(self.binding['instance_support_gate']['file'])

    def rows(self):
        for row in super().rows():
            if self.frame_role_guard:self.frame_role_guard.assert_training_pixels([int(row["native_pts"])],shot_id=row["shot_id"])
            support=self.instance_gate.rows.get(row['selected_row_id'])
            if support is None or support['instance_state']!=1 or support['instance_support_kind']not in POSITIVE_KINDS:raise ValueError('Future physical consumer lacks positive finite instance authority')
            if (support['shot_id'],support['native_pts'],support['source_uv'],support['material_index'])!=(row['shot_id'],row['native_pts'],row['uv'],row['material_index']):raise ValueError('Future positive instance support changes source identity')
            probe=dict(selected_row_id=row['selected_row_id'],shot_id=row['shot_id'],native_pts=row['native_pts'],uv=row['uv'],source_point_id=row['reuses_original_source_point_id'],feature_id=row['target_id'],physical_surface=row['physical_surface'],measurement_definition=row['measurement_definition'])
            if self.instance_gate.check(probe,row['material_index'])!=0:raise ValueError('Future source row is not positive in the current qualified semantic part')
            if self.instance_gate.source_parts and row['source_store_ref']!=self.instance_gate.source_parts[int(support['source_part_index'])]['reference']:raise ValueError('Future source row comes from a different semantic source part')
            row.update(physical_identity_status='positive_finite_source_instance_supported',source_support_status=POSITIVE_KINDS[support['instance_support_kind']],instance_support_kind=support['instance_support_kind'],instance_support_evidence=support['evidence'],profile_success_grants_physical_identity=False)
            yield row


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    r=sub.add_parser('registry');r.add_argument('--packet',required=True);r.add_argument('--qualification',required=True);r.add_argument('--review',action='append',required=True);r.add_argument('--out',required=True)
    b=sub.add_parser('build');b.add_argument('--bindings',required=True);b.add_argument('--model',required=True);b.add_argument('--out',required=True);b.add_argument('--project',required=True);b.add_argument('--instance-registry',required=True);b.add_argument('--parent-packet');b.add_argument('--frame-roles',required=True)
    p.add_argument('--code-aliases',help='Explicit append-only exact historical Python source registry')
    a=p.parse_args()
    from .exact_dependency_aliases import dependency_alias_context
    with dependency_alias_context(_ref(a.code_aliases)if a.code_aliases else None):
        return _main_operation(a)

def _main_operation(a):
    if a.operation=='registry':result=build_instance_registry(a.packet,a.qualification,a.review,a.out)
    else:result=build_typed_dense_input(a.bindings,a.model,a.out,instance_registry=a.instance_registry,project_root=a.project,frame_roles=a.frame_roles,parent_packet=a.parent_packet)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
