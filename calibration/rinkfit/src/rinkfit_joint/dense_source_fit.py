"""Stream qualified dense source readings into finite physical-material candidates.

The source store remains authoritative for UV, IDs, masks and source provenance.
This bridge issues curve identity candidates, never a fixed world station, a
noise covariance, or a pixel-profile Query that was not actually measured.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import hashlib
import re
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, file_sha256, object_sha256
from .source_record_store import SourceRecordStore

REASONS = {0:'finite_material_geometry_candidate',1:'inferred_not_current_direct_measurement',2:'outside_confirmed_static_surface',3:'unbound_material_identity',4:'no_matching_finite_model_material',5:'measurement_definition_not_compatible',6:'invalid_source_clock_or_UV',7:'print_requires_instance_shape_or_fixed_texture_crosswalk',8:'fixed_world_point_correspondence_not_issued',9:'duplicate_parent_same_exposure_UV_material'}
WIDTH = 'width_midpoint_between_both_material_edges'
RING = 'declared_target_ring_material_boundary'
BOARD_DEFS = {'board.top':{'upper edge of blue rim at top of vertical board','ice_facing_boundary_of_gray_board_top_surface_same_zero_thickness_board_upper_boundary_as_far_blue_upper_edge','retained_parent_material_or_manual_definition'},'board.contact':{'lower edge of yellow kickplate at actual ice interface','retained_parent_material_or_manual_definition'}}


def _ref(path):
    path=Path(path).resolve()
    return dict(file=str(path),sha256=file_sha256(path))


def _materials(model):
    result={}
    for i,m in enumerate(model['materials']):
        label=m['label']; paths=m['paths']
        if label in ('board.top','board.contact'):
            h=1. if label=='board.top' else 0.
            if len(paths)!=8 or sorted(p.get('boundary_segment') for p in paths)!=list(range(8)) or any(p.get('kind')!=3 or p.get('fixed_height_fraction')!=h or p.get('physical_point')is not False for p in paths):
                raise ValueError('Dense board correspondence needs the actual eight connected perimeter branches')
            result[label]=dict(material_index=i,path_index=0,allowed_paths=list(range(8)),physical_point=False,kind='board_edge',measurement_definitions=sorted(BOARD_DEFS[label]))
        elif label.startswith('ice.') and not label.startswith('ice.tracklet.'):
            if not paths or any(p.get('physical_point')is not False for p in paths):continue
            if any(p.get('kind') not in (0,1,4,5) for p in paths):continue
            defs=[WIDTH]
            if re.fullmatch(r'ice\.[FR][1-4]:(inner|middle|outer)',label):defs=[RING]
            elif ':center' in label:continue
            elif 'goal_crease_' in label:defs=['filled_blue_goal_area_material_boundary']
            result[label]=dict(material_index=i,path_index=0,allowed_paths=list(range(len(paths))),physical_point=False,kind='ice_curve',measurement_definitions=defs)
    return result


def _decision(row,crosswalk,parent_keys=None):
    try:uv=np.asarray(row['uv'],dtype=np.float64)
    except (KeyError,TypeError,ValueError):return 6,None
    if row.get('time_base')!='1/12800' or row.get('shot_id')not in ('s02','s03','s04','s05','s06') or uv.shape!=(2,) or not np.isfinite(uv).all() or not(0<=uv[0]<=1919 and 0<=uv[1]<=1079):return 6,None
    if row.get('physical_surface')not in ('ice','board'):return 2,None
    if row.get('source_UV_is_direct_measurement')is not True:return 1,None
    fid=row['feature_id']
    if fid.startswith('unbound.'):return 3,None
    if row.get('display_class')=='print':return 7,None
    d=crosswalk.get(fid)
    if d is None:return 4,None
    if d['physical_point']:return 8,None
    definition=row.get('measurement_definition')
    # A user-confirmed original native click names a curve, not a fixed point.
    manual=row.get('user_confirmed')is True and row.get('source_kind')=='preserved_user_manual_native_image_click'
    if definition not in d['measurement_definitions'] and not manual:return 5,None
    if fid.startswith('board.') != (row['physical_surface']=='board'):return 2,None
    if parent_keys is not None and _key(row['shot_id'],row['native_pts'],d['material_index'],uv)in parent_keys:return 9,d
    return 0,d


def _key(shot,pts,mi,uv):
    return (shot,int(pts),int(mi),np.asarray(uv,dtype='<f8').tobytes())


def _parent_keys(packet_path):
    if not packet_path:return set()
    pp=Path(packet_path).resolve();p=read_json(pp);arrays={}
    for k in ('uv','native_pts','shot_index','material_index','admission','fit_role'):
        r=p['arrays'][k];f=Path(r['file']);f=f if f.is_absolute()else pp.parent/f
        if file_sha256(f)!=r['sha256']:raise ValueError('Parent numeric source array changed')
        arrays[k]=np.fromfile(f,dtype=r['dtype'],offset=int(r.get('offset_bytes',0)),count=int(np.prod(r['shape']))).reshape(r['shape'])
    a=arrays;active=np.flatnonzero((a['admission']==1)&(a['fit_role']==1))
    return {_key('s%02d'%(int(a['shot_index'][i])+1),a['native_pts'][i],a['material_index'][i],a['uv'][i])for i in active}


def _policy_refs(project):
    root=Path(project)/'events/ankara_2026/model_constraints'
    return [_ref(root/name)for name in ('user_board_edge_semantics_20261010.json','paint_roles_1018.json','source_material_operator_corrections_20261012.json','source_material_operator_corrections_right_end_20261012.json','source_material_operator_corrections_straight_connections_20261012.json','lane2_user_confirmed_20261007.json')]


def build_dense_input(storage_bindings,model_path,out,*,project_root,parent_packet=None):
    out=Path(out).resolve()
    if out.exists():raise ValueError('New immutable dense geometry bridge output required')
    storage=read_json(storage_bindings);model=read_json(model_path);crosswalk=_materials(model);parent=_parent_keys(parent_packet)
    out.mkdir(parents=True);material_file=out/'material_map.json'
    mapping=dict(schema='rinkfit.dense_named_finite_source_material_map.v1',semantic_to_material={},row_to_material={},model_materials_sha256=object_sha256(model['materials']),source_identity_is_finite_curve_not_fixed_world_point=True,source_station_not_measured=True)
    for label,d in crosswalk.items():mapping['semantic_to_material'][label]=dict(d,geometry_ready=True,target=label,basis='existing_named_model_material_and_explicit_source_measurement_definition',source_accuracy_certified=False)
    write_json(material_file,mapping)
    totals=Counter();reason_shot=Counter();eligible=Counter();source_parts=[];ranges=[];ordinal=0;eligible_count=0;reasons=[];materials=[];frames={};normalizer=hashlib.sha256()
    for name,g in storage['generations'].items():
        source_ref=g['selected_rows'];start=ordinal;part_count=0;part_eligible=0
        for r in read_jsonl(source_ref):
            code,d=_decision(r,crosswalk,parent);reasons.append(code);materials.append(d['material_index']if d else -1);totals[REASONS[code]]+=1;reason_shot[(r['shot_id'],REASONS[code])]+=1
            if code==0:
                eligible[(r['shot_id'],r['feature_id'])]+=1;eligible_count+=1;part_eligible+=1
                key=(r['shot_id'],int(r['native_pts']));frames[key]=frames.get(key,0)+1
                normalizer.update(json.dumps([r['selected_row_id'],r['shot_id'],r['native_pts'],r['uv'],d['material_index']],separators=(',',':'),ensure_ascii=False).encode('utf8')+b'\n')
            ordinal+=1;part_count+=1
        source_parts.append(dict(name=name,reference=source_ref,ordinal_start=start,row_count=part_count,eligible_count=part_eligible));ranges.append([start,ordinal])
    index=out/'eligibility.npz'
    np.savez_compressed(index,reason=np.asarray(reasons,dtype=np.uint8),material_index=np.asarray(materials,dtype=np.int32))
    report=dict(schema='rinkfit.dense_finite_material_source_admission.v1',display_source_rows=ordinal,eligible_geometry_rows=eligible_count,reason_codes=REASONS,reason_counts=dict(totals),reason_counts_by_shot=[dict(shot_id=k[0],reason=k[1],rows=v)for k,v in sorted(reason_shot.items())],eligible_counts_byshot_byfeature=[dict(shot_id=k[0],feature_id=k[1],rows=v)for k,v in sorted(eligible.items())],covered_frames=len(frames),eligible_rows_normalized_sha256=normalizer.hexdigest(),real_source_Query_rows=0,fixed_world_point_rows=0,no_Query_original_sourceUV_rows=eligible_count,source_UV_ID_and_native_clock_unchanged=True,source_pixels_decoded=0,full_source_table_copy_written=False,noise_covariance_issued=False,physical_model_accuracy_accepted=False,known_limits=['Direct source curves are geometry candidates; fit residual cannot certify or change their physical identity.','Print boundaries and 2D image texture without reviewed world/material correspondence are excluded here; retain existing qualified parent packet anchors.','Parent same-exposure exact UV/material duplicates are excluded; other correlated observations remain in one named-material dependency family.','All eight physical board branches remain candidates; nearest branch is a numerical auxiliary, never source identity truth.','No row is promoted to a fixed world point or fixed station. Unknown W is retained explicitly.'])
    write_json(out/'admission.json',report)
    binding=dict(schema='rinkfit.dense_source_fit_input.v1',source_storage_bindings=_ref(storage_bindings),source_parts=source_parts,model_ref=_ref(model_path),model_materials_sha256=object_sha256(model['materials']),material_map=_ref(material_file),eligibility_index=_ref(index),admission=_ref(out/'admission.json'),parent_packet=_ref(parent_packet)if parent_packet else None,policy_refs=_policy_refs(project_root),reader=_ref(__file__),total_rows=eligible_count,total_display_rows=ordinal,eligible_rows_normalized_sha256=normalizer.hexdigest(),shot_ids=['s02','s03','s04','s05','s06'],all_missing_rows_accounted_by_columnar_reason_index=True,source_accuracy_accepted=False)
    write_json(out/'source_binding.json',binding)
    return dict(binding=_ref(out/'source_binding.json'),report=_ref(out/'admission.json'),rows=eligible_count,display_rows=ordinal,index_bytes=index.stat().st_size)


class DenseSourceFitInput:
    def __init__(self,binding_path):
        self.path=Path(binding_path).resolve();self.identity=file_sha256(self.path);self.binding=read_json(self.path)
        if self.binding.get('schema')!='rinkfit.dense_source_fit_input.v1':raise ValueError('Unsupported dense source geometry binding')
        self._refs={};self._pin(_ref(self.path))
        for key in ('source_storage_bindings','model_ref','material_map','eligibility_index','admission','reader'):self._pin(self.binding[key])
        for ref in self.binding['policy_refs']:self._pin(ref)
        if self.binding.get('parent_packet'):self._pin(self.binding['parent_packet'])
        self.mapping=read_json(self.binding['material_map']['file']);self.model=read_json(self.binding['model_ref']['file'])
        if object_sha256(self.model['materials'])!=self.binding['model_materials_sha256']:raise ValueError('Dense source material topology changed')
        self.crosswalk=_materials(self.model);self.row_count=int(self.binding['total_rows'])
        with np.load(self.binding['eligibility_index']['file'],allow_pickle=False)as a:self.reason=a['reason'].copy();self.material_index=a['material_index'].copy()
        if len(self.reason)!=self.binding['total_display_rows']or len(self.reason)!=len(self.material_index)or int(np.sum(self.reason==0))!=self.row_count:raise ValueError('Dense source qualification denominator changed')

    def _pin(self,ref):
        p=Path(ref['file']).resolve();sha=ref['sha256']
        if p in self._refs and self._refs[p]!=sha:raise ValueError('Conflicting dense source dependency pin')
        if file_sha256(p)!=sha:raise ValueError('Changed dense source dependency: '+str(p))
        self._refs[p]=sha

    def assert_dependencies(self):
        for p,sha in self._refs.items():
            if file_sha256(p)!=sha:raise ValueError('Changed dense source dependency: '+str(p))
        for part in self.binding['source_parts']:
            ref=part['reference']
            with SourceRecordStore(ref['store'],readonly=True)as store:
                info=store.stream_info(ref['stream'])
                if info['semantic_sha256']!=ref['stream_semantic_sha256']or info.get('sealed')!=1:raise ValueError('Changed sealed source stream')

    def rows(self):
        count=0;ordinal=0;proof=hashlib.sha256()
        for part in self.binding['source_parts']:
            part_n=0
            for r in read_jsonl(part['reference']):
                code=int(self.reason[ordinal]);mi=int(self.material_index[ordinal]);original_ordinal=ordinal;ordinal+=1;part_n+=1
                if code:continue
                check,d=_decision(r,self.crosswalk)
                if check!=0 or d['material_index']!=mi:raise ValueError('Stored qualification redirects actual source row')
                rid=r['selected_row_id'];uv=r['uv'];fid=r['feature_id'];sid=r['shot_id'];pts=int(r['native_pts']);family='dense_named_material:'+sid+':'+fid
                proof.update(json.dumps([rid,sid,pts,uv,mi],separators=(',',':'),ensure_ascii=False).encode('utf8')+b'\n')
                count+=1
                yield dict(source_row_id=rid,reuses_original_source_point_id=r['source_point_id'],selected_row_id=rid,source_uv=uv,current_uv=uv,uv=uv,shot_id=sid,native_pts=pts,time_base=r['time_base'],target_id=fid,joint_material_id=fid,joint_material_index=mi,material_index=mi,path_index=d['path_index'],allowed_paths=d['allowed_paths'],physical_point=False,source_family_id=family,dependency_component_id=family,physical_surface=r['physical_surface'],measurement_definition=r['measurement_definition'],measurement_operator_resolved='point_to_named_finite_curve',point_correspondence=False,entity_station_fixed=False,source_admitted=True,source_support_status='source_supported_identity_candidate',physical_identity_status='typed_identity_current_pixel_supported',source_review_reasons=[],source_geometry_eligible=True,source_noise_W_supplied=False,solver_eligible_for_statistical_objective=False,query_available=False,query_unavailable_reason='source_pixel_profile_Query_not_issued_by_this_display_generation',engineering_pixel_uv=uv,engineering_pixel_basis='unchanged_original_source_UV',source_UV_is_direct_measurement=True,user_confirmed=r.get('user_confirmed',False),source_packet=part['reference']['stream'],source_packet_sha256=part['reference']['stream_semantic_sha256'],source_packet_hash_kind='recovered_records_semantic_SHA256',source_store_ref=part['reference'],source_stream_ordinal=part_n-1,source_global_ordinal=original_ordinal,source_pixel_or_annotation_ref=r['source_pixel_or_annotation_ref'],source_branch_id=r.get('source_branch_id'),source_kind=r['source_kind'],typed_material_descriptor=dict(kind=d['kind'],curve_id=fid.removeprefix('ice.')))
            if part_n!=part['row_count']:raise ValueError('Dense source stream full denominator changed')
        if ordinal!=len(self.reason)or count!=self.row_count or proof.hexdigest()!=self.binding['eligible_rows_normalized_sha256']:raise ValueError('Actual dense source geometry rows changed')

    def supports(self):
        # Absence of source profile statistics stays explicit; do not invent W.
        return iter(())

    def material_map(self):
        return self.mapping

    def verify_model_materials(self,path):
        if object_sha256(read_json(path)['materials'])!=self.binding['model_materials_sha256']:raise ValueError('Rebound model changes source material topology')
        return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--bindings',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--project',required=True);p.add_argument('--parent-packet');a=p.parse_args()
    print(json.dumps(build_dense_input(a.bindings,a.model,a.out,project_root=a.project,parent_packet=a.parent_packet)))

if __name__=='__main__':main()
