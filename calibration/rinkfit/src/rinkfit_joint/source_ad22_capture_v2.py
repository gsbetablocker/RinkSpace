"""Conditional Ad22 print capture on the existing common board, camera held fixed.

The fitted four numbers are a compact shared motion of the current twenty
print affines, not a measured outer panel. The output re-encodes that physical
composite into the original fixed ad/22 witness and changes only its existing
six-coefficient print blocks. Original source identities, UV and templates stay
unchanged. The result initializes geometry; it supplies no W or survey truth.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,copy,json,time
import numpy as np
from scipy.optimize import least_squares
from rinkfit_start.native_scene import NativeScene
from rinkfit_start.ad_chart import effective_affine,local_affine
from .joint_model import as_event_view,validate_physical_domain
from .print_chart_domain import decode_common_witness
from .source_quality_compare_v1 import _camera
from .source_generation_input_v2 import QualifiedSourceGenerationInput
from .io_utils import read_json,write_json,file_sha256


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha256(path))

def arrays(packet):
    return {k:np.fromfile(v.get('file',v.get('path')),dtype=v['dtype']).reshape(v['shape']) for k,v in packet['arrays'].items() if k in ('material_index','path','admission','E','native_pts')}

def closest(uv,curve,valid):
    a=curve[:-1];d=curve[1:]-a;den=np.sum(d*d,axis=1);keep=valid[:-1]&valid[1:]&(den>1e-20)&np.isfinite(a).all(1)&np.isfinite(d).all(1);a=a[keep];d=d[keep];den=den[keep]
    if not len(a):return np.full_like(uv,10000.),np.zeros(len(uv),bool)
    u=np.clip(np.einsum('ijk,jk->ij',uv[:,None]-a[None],d)/den,0.,1.);feet=a[None]+u[:,:,None]*d[None];error=uv[:,None]-feet;chosen=np.argmin(np.sum(error*error,axis=2),axis=1)
    return error[np.arange(len(uv)),chosen],np.ones(len(uv),bool)


def stats(errors,valid):
    distance=np.linalg.norm(errors,axis=1);good=np.asarray(valid,bool)&np.isfinite(distance)
    return dict(rows=len(distance),valid=int(good.sum()),failed=int((~good).sum()),full_rms_px=float(np.sqrt(np.mean(distance*distance))) if good.all() else None,available_rms_px=float(np.sqrt(np.mean(distance[good]**2))) if good.any() else None,p95_px=float(np.quantile(distance[good],.95)) if good.any() else None,max_px=float(distance[good].max()) if good.any() else None)


class Capture:
    def __init__(self,model,rows,ad_index):
        self.model=model;self.ad_index=ad_index;self.scene=NativeScene(as_event_view(model,'s02'));self.camera=_camera(model,'s02');labels={x['label']:i for i,x in enumerate(model['materials'])};self.groups=[]
        for pts,target in sorted({(r['native_pts'],r['target_id']) for r in rows}):
            selected=[r for r in rows if r['native_pts']==pts and r['target_id']==target];mi=labels[target]
            self.groups.append(dict(native_pts=int(pts),target_id=target,material_index=mi,rows=selected,UV=np.asarray([r['source_uv'] for r in selected],float),admitted=np.asarray([r.get('source_admitted') is True for r in selected],bool)))
    def evaluate(self,installation,subdivisions=8,detail=False):
        self.scene.values[self.ad_index]=np.asarray(installation,float)
        parameters=[np.linspace(0.,1.,(len(self.scene.path(g['material_index'],0)['points'])-1)*subdivisions+1) for g in self.groups]
        xyz=[self.scene.world(g['material_index'],0,s) for g,s in zip(self.groups,parameters)];ts=[np.full(len(s),(g['native_pts']-self.model['shots']['s02']['time_origin_pts'])/12800.) for g,s in zip(self.groups,parameters)]
        p=self.camera.project_rs(np.concatenate(xyz),np.concatenate(ts));offset=0;all_errors=[];all_valid=[];group_stats=[]
        for g,s in zip(self.groups,parameters):
            curve=p['uv'][offset:offset+len(s)];valid=p['valid'][offset:offset+len(s)];offset+=len(s);e,v=closest(g['UV'],curve,valid);all_errors.append(e);all_valid.append(v);group_stats.append(dict(native_pts=g['native_pts'],target_id=g['target_id'],**stats(e,v)))
        error=np.concatenate(all_errors);valid=np.concatenate(all_valid)
        if detail:return dict(**stats(error,valid),by_exposure={str(pts):stats(np.concatenate([e for g,e in zip(self.groups,all_errors) if g['native_pts']==pts]),np.concatenate([v for g,v in zip(self.groups,all_valid) if g['native_pts']==pts])) for pts in sorted({g['native_pts'] for g in self.groups})},by_target_exposure=group_stats),error,valid
        return error[np.concatenate([g['admitted'] for g in self.groups])].ravel()
    def converged(self,installation,tolerance=.001):
        sub=8;previous=self.evaluate(installation,sub,True)
        while True:
            sub*=2;now=self.evaluate(installation,sub,True);delta=float(np.max(np.abs(np.linalg.norm(previous[1],axis=1)-np.linalg.norm(now[1],axis=1))))
            if np.array_equal(previous[2],now[2]) and delta<=tolerance:
                return now[0],dict(subdivisions_per_original_segment=sub,previous_subdivisions=sub//2,successive_distance_change_max_px=delta,tolerance_px=tolerance),now[1]
            previous=now


def apply_overlay(model,overlay):
    result=copy.deepcopy(model);names={b['name']:i for i,b in enumerate(result['blocks'])}
    for entry in overlay['block_updates']:
        i=names[entry['block_name']]
        if result['blocks'][i]['values']!=entry['expected_values']:raise ValueError('capture overlay starting composite changed: '+entry['block_name'])
        result['blocks'][i]['values']=entry['values']
    return result


def run(binding,model_file,s01_packet,packet_file,material_map,source_scene,output):
    start=time.perf_counter();out=Path(output)
    if out.exists():raise ValueError('immutable capture output already exists')
    model=read_json(model_file);q=QualifiedSourceGenerationInput(binding);all_rows=[r for r in q.rows() if (r.get('target_id') or '').startswith('Ad22.')];labels={m['label']:i for i,m in enumerate(model['materials'])};ai=next(i for i,b in enumerate(model['blocks']) if b['name']=='ad/22');affected=[i for i,m in enumerate(model['materials']) if any(p.get('advertisement_block')==ai for p in m['paths'])];qids=sorted({int(p['geometry_block']) for i in affected for p in model['materials'][i]['paths']});rows=[r for r in all_rows if r['target_id'] in labels and labels[r['target_id']] in affected]
    if len(qids)!=20 or any(model['materials'][i]['paths'][0]['kind']!=2 for i in affected):raise ValueError('unexpected same Ad22 composite topology')
    if any(p.get('geometry_block') in qids and p.get('advertisement_block')!=ai for m in model['materials'] for p in m['paths']):raise ValueError('shared affine is used by a different physical ad')
    if any(r['shot_id']!='s02' for r in rows):raise ValueError('different source camera needs explicit frozen capture setup')
    cap=Capture(model,rows,ai);old=np.asarray(model['blocks'][ai]['values']);height=cap.scene.height;perimeter=cap.scene.perimeter;before,bproof,berror=cap.converged(old)
    # Lattice is only a numerical search for an initial basin, not observed
    # station correspondence or a statement of global physical optimality.
    seeds=[]
    for station in np.linspace(0.,perimeter,385,endpoint=False):
        value=old.copy();value[0]=station;cost=float(np.mean(cap.evaluate(value)**2));seeds.append((cost,value))
    seeds.sort(key=lambda v:v[0]);seed=seeds[0][1];x0=np.r_[seed[:3],(seed[3]-seed[2])/(height-seed[2])]
    def install(x):return np.r_[x[:3],x[2]+(height-x[2])*x[3]]
    fit=least_squares(lambda x:cap.evaluate(install(x)),x0,bounds=([-perimeter,1e-6,0.,1e-6],[2*perimeter,perimeter,height-1e-9,1.]),x_scale=[.1,.05,.01,1.],ftol=1e-10,xtol=1e-10,gtol=1e-8)
    candidate=install(fit.x);after,aproof,aerror=cap.converged(candidate)
    if not fit.success or after['failed']:raise ValueError('conditional Ad22 capture did not finish with all original observations')
    updates=[]
    for qi in qids:
        block=model['blocks'][qi];physical=effective_affine(candidate,block['values']);new=local_affine(old,physical)
        if np.any(new<np.asarray(block['lower'])) or np.any(new>np.asarray(block['upper'])) or new[0]*new[4]-new[1]*new[3]<1e-6:raise ValueError('captured composite has no legal original fixed witness: '+block['name'])
        updates.append(dict(block_name=block['name'],block_index=qi,expected_values=block['values'],values=new.tolist(),physical_composite=physical.tolist()))
    overlay=dict(schema='rinkfit.source_conditioned_print_composite_capture.v1',base_model=ref(model_file),input_binding=ref(binding),ad_block_name='ad/22',ad_block_index=ai,original_fixed_witness=old.tolist(),conditional_compact_installation=candidate.tolist(),original_ad_witness_and_all_flags_bounds_unchanged=True,block_updates=updates,source_UV_ID_templates_unchanged=True,scene_camera_unchanged=True,additional_private_panel=False,geometry_accuracy_accepted=False,statistical_W_or_world_truth_created=False)
    result=apply_overlay(model,overlay);view=as_event_view(result,'s02');res=NativeScene(view);original=NativeScene(as_event_view(model,'s02'));werr=[]
    cap.scene.values[ai]=candidate
    for mi in affected:
        s=np.linspace(0.,1.,513);werr.append(np.max(np.abs(res.world(mi,0,s)-cap.scene.world(mi,0,s))))
        path=res.path(mi,0);pp=np.asarray(path['points']);qq=res.values[path['geometry_block']];uv=pp@np.array([[qq[0],qq[3]],[qq[1],qq[4]]])+qq[[2,5]];heights=old[2]+(old[3]-old[2])*uv[:,1]
        if heights.min()<-1e-10 or heights.max()>height+1e-10:raise ValueError('captured complete finite print support leaves actual board')
    world_error=float(max(werr));assert world_error<2e-14
    domain_before=validate_physical_domain(model);domain_after=validate_physical_domain(result)
    decoded,chart_domain=decode_common_witness(result,advertisement_ids=(22,))
    if not chart_domain['local_original_domain_equivalence_certified']:raise ValueError('capture has no strict production original print chart witness')
    if not domain_after['passed']:raise ValueError('captured model fails production validate_physical_domain: '+','.join(domain_after['violations']))
    sp=read_json(s01_packet);sa=arrays(sp);s01_affected=np.flatnonzero(np.isin(sa['material_index'],affected))
    if len(s01_affected):raise ValueError('capture changes a material used by complete S01; independent full projection proof required before use')
    packet=read_json(packet_file);pa=arrays(packet);indices={rid:i for i,rid in enumerate(packet['source_row_ids'])};unknown=[r for r in all_rows if r not in rows];unknown_details=[]
    for r in unknown:
        i=indices[r['source_row_id']];entry=read_json(material_map)['semantic_to_material'].get(r.get('joint_material_id'),{})
        unknown_details.append(dict(source_row_id=r['source_row_id'],target_id=r['target_id'],source_uv=r['source_uv'],native_pts=r['native_pts'],packet_row=i,packet_material_index=int(pa['material_index'][i]),packet_admission=int(pa['admission'][i]),material_binding=entry,quality_comparison_available=False,reason='no proven finite material path for this original identity; placeholder projection is not Ad22 evidence'))
    out.mkdir(parents=True);write_json(out/'parameter_overlay.json',overlay);write_json(out/'model_captured.json',result);write_json(out/'all_Ad22_original_source_rows.json',all_rows)
    write_json(out/'unknown_88.json',unknown_details)
    write_json(out/'production_physical_domain_before.json',domain_before);write_json(out/'production_physical_domain_after.json',domain_after);write_json(out/'production_print_chart_domain_22.json',chart_domain)
    # Independent fixed-witness consumer reads the newly written model file.
    consumed=read_json(out/'model_captured.json');cc=Capture(consumed,rows,ai);consumer,consumer_proof,cerror=cc.converged(old)
    if abs(consumer['full_rms_px']-after['full_rms_px'])>1e-8:raise ValueError('actual model consumer does not reproduce candidate composite')
    overlays=[]
    import cv2
    for pts in sorted({r['native_pts'] for r in all_rows}):
        rr=[r for r in all_rows if r['native_pts']==pts];png=Path(rr[0]['native_png'])
        if file_sha256(png)!=rr[0]['native_png_sha256']:raise ValueError('frozen original native PNG changed')
        image=cv2.imread(str(png));panels=[]
        for scene_value,title in [(original,'BEFORE: fixed current camera, original print composite'),(res,'AFTER: same camera + fixed ad witness, captured print composite')]:
            canvas=image.copy()
            for mi in affected:
                xyz=scene_value.world(mi,0,np.linspace(0,1,257));pr=cap.camera.project_rs(xyz,np.full(257,(pts-model['shots']['s02']['time_origin_pts'])/12800.));valid=pr['valid']&np.isfinite(pr['uv']).all(1)
                for k in range(256):
                    if valid[k] and valid[k+1]:cv2.line(canvas,tuple(np.rint(pr['uv'][k]).astype(int)),tuple(np.rint(pr['uv'][k+1]).astype(int)),(255,0,255),1,cv2.LINE_AA)
            for r in rr:
                xy=tuple(np.rint(r['source_uv']).astype(int));color=(0,255,0) if r in rows else (0,165,255);cv2.circle(canvas,xy,2,color,1,cv2.LINE_AA)
            cv2.putText(canvas,title,(20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA);cv2.putText(canvas,'Ad22 capture diagnostic | green: 306 r2 source UV | orange: 88 unknown original rows',(20,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);panels.append(canvas)
        dest=out/('nativePTS_'+str(pts)+'_before_after.png');cv2.imwrite(str(dest),np.concatenate(panels,axis=1));overlays.append(dict(source_native_png=ref(png),overlay=ref(dest),native_pts=pts))
    changed=[i for i,(a,b) in enumerate(zip(model['blocks'],result['blocks'])) if a!=b]
    assert changed==qids and result['materials']==model['materials']
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    report=dict(schema='rinkfit.source_conditioned_Ad22_capture_evidence.v1',model_before=ref(model_file),model_after=ref(out/'model_captured.json'),overlay=ref(out/'parameter_overlay.json'),input_binding=ref(binding),source_scene=ref(source_scene),producer=ref(__file__),all_Ad22_source_rows=len(all_rows),matched_actual_same_installation_rows=len(rows),fitted_source_admitted_rows=sum(r.get('source_admitted') is True for r in rows),source_failed_rows_never_admitted_by_capture=sum(r.get('source_admitted') is not True for r in rows),all_same_installation_rows_evaluated_including_source_failures=True,unavailable_identity_rows=len(unknown),all_original_rows_retained=True,source_current_support_failure_rows=sum(r.get('current_uv') is None for r in all_rows),matched_source_current_support_failure_rows=sum(r.get('current_uv') is None for r in rows),before=before,after=after,model_file_consumer=consumer,numerical_curve_convergence=dict(before=bproof,after=aproof,actual_model_consumer=consumer_proof),initialization_only_lattice_best=[dict(mean_coordinate_squared_error=c,installation=v.tolist()) for c,v in seeds[:4]],fit=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,conditional_J_singular_values=singular.tolist(),covariance='unknown; conditional capture is not calibrated source noise or parameter certainty'),production_validate_physical_domain_passed=domain_after['passed'],production_print_chart_domain_22_strict_witness=chart_domain['local_original_domain_equivalence_certified'],physical_domain_report=ref(out/'production_physical_domain_after.json'),print_chart_domain_report=ref(out/'production_print_chart_domain_22.json'),fixed_ad_witness_preserved=True,old_domain_witness_ad_ids=[a['advertisement_id'] for a in model.get('print_only_domain_witness',{}).get('advertisements',[])],rechart_max_full_curve_XYZ_difference_L=world_error,changed_block_indices=changed,changed_blocks=[model['blocks'][i]['name'] for i in changed],all_other_blocks_materials_topology_flags_bounds_unchanged=True,s01_complete_prefix=dict(packet=ref(s01_packet),rows=len(sa['material_index']),ad22_dependent_rows=int(len(s01_affected)),projection_change_from_parameter_dependency=0.,full_S01_projection_replay_owner='root'),unknown_original_rows=ref(out/'unknown_88.json'),unknown_current_packet_projection_unchanged=all(int(pa['material_index'][indices[r['source_row_id']]]) not in affected for r in unknown),unknown_quality_delta='unavailable: no material identity; unchanged placeholder is not finite Ad22 geometry evidence',source_Gaussian_query_covariance_or_ids_changed=False,source_conditioned_installation_is_not_world_truth=True,global_physical_optimum_claimed=False,full_shot_review_or_independent_validation_claimed=False,overlays=overlays,limitations=['Capture conditions on the frozen current scene and all-time camera; their errors remain possible.','Remaining finite curve residuals are retained and must be handled by the normal joint geometry solve.','Only explicit source_admitted rows enter the geometry capture objective; all source failures remain in comparison denominators.','Twenty existing affine blocks move coherently; their original material contours and source identities are unchanged.','The 88 old whole-component rows remain unknown; no alias to the opposite Ad9 physical instance is inferred.'],seconds=time.perf_counter()-start)
    write_json(out/'capture_evidence.json',report);print(json.dumps(dict(output=str(out),before=before['full_rms_px'],after=after['full_rms_px'],max_after=after['max_px'],changed_blocks=len(changed),unknown=len(unknown),seconds=report['seconds'])),flush=True);return report


def main():
    p=argparse.ArgumentParser()
    for k in ('binding','model','s01-packet','packet','material-map','source-scene','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.binding,a.model,a.s01_packet,a.packet,a.material_map,a.source_scene,a.out)
if __name__=='__main__':main()
