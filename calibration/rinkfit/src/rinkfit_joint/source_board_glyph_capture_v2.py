"""Conditional shared placement of new finite board glyphs, with fixed cameras.

One low-dimensional course/height affine acts on all new kind6 contours of a
physical advertisement. No vertex has an independent update and no source UV is
inverted into a world measurement. Existing S01/old print blocks are untouched.
"""
from __future__ import annotations
import argparse,copy,json,time,gzip
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from rinkfit_start.native_scene import NativeScene
from .source_ad22_capture_v2 import closest,stats as _stats,apply_overlay,ref
from .source_quality_compare_v1 import _camera
from .joint_model import as_event_view,validate_physical_domain
from .source_generation_input_v2 import QualifiedSourceGenerationInput
from .io_utils import read_json,write_json,file_sha256


def stats(errors,valid):
    if not len(errors):
        return dict(rows=0,valid=0,failed=0,full_rms_px=None,available_rms_px=None,p95_px=None,max_px=None)
    return _stats(errors,valid)


def load_arrays(packet):
    return {k:np.fromfile(v.get('file',v.get('path')),dtype=v['dtype']).reshape(v['shape'])for k,v in packet['arrays'].items()}


def production_weights(packet,A,ledger):
    dependencies={g['id']:g['rows']for g in packet['engineering_pixel_dependency_groups']['groups']}
    weights={}
    for g in ledger['groups']:
        if g['family']!='advertisement':continue
        for i in dependencies[g['source_dependency_component']]:
            if int(A['native_pts'][i])==g['native_pts'] and int(A['shot_index'][i])+1==int(g['shot'][1:]):
                if i in weights:raise ValueError('ambiguous_production_engineering_pixel_weight')
                weights[i]=float(g['per_row_weight'])/float(A['E'][i])**2
    return weights


class Capture:
    def __init__(self,model,entries,weights,indices):
        self.model=model;self.scene=NativeScene(as_event_view(model,entries[0]['row']['shot_id']))
        self.entries=sorted(entries,key=lambda x:(x['row']['shot_id'],x['row']['native_pts'],x['binding']['material_index'],x['row']['source_row_id']))
        self.rows=[e['row']for e in self.entries];self.mi=sorted({e['binding']['material_index']for e in entries})
        self.blocks=sorted({model['materials'][mi]['paths'][0]['geometry_block']for mi in self.mi})
        self.base={bi:np.asarray(model['blocks'][bi]['values'],float).reshape(-1,2)for bi in self.blocks}
        pts=np.concatenate(list(self.base.values()));self.anchor=(pts.min(0)+pts.max(0))/2;self.scale=np.maximum(np.ptp(pts,axis=0),1e-6)
        self.admitted=np.array([r.get('source_admitted')is True for r in self.rows]);self.weight=np.array([weights[indices[r['source_row_id']]]if r.get('source_admitted')is True else 0. for r in self.rows]);self.weight/=np.mean(self.weight[self.admitted])
        self.cameras={sid:_camera(model,sid)for sid in {r['shot_id']for r in self.rows}}
        self.groups=[]
        for key in sorted({(r['shot_id'],r['native_pts'],e['binding']['material_index'])for r,e in zip(self.rows,self.entries)}):
            ix=np.array([i for i,(r,e) in enumerate(zip(self.rows,self.entries))if (r['shot_id'],r['native_pts'],e['binding']['material_index'])==key]);sid,pts,mi=key;p=model['materials'][mi]['paths'][0]
            self.groups.append(dict(shot=sid,pts=pts,mi=mi,bi=p['geometry_block'],closed=p.get('closed',False),indices=ix,UV=np.asarray([self.rows[i]['source_uv']for i in ix],float)))
        self.cache={}
    def matrix(self,x):
        if len(x)==2:return np.eye(2)
        return np.array([[np.exp(x[2]),x[4]if len(x)==6 else 0.],[x[5]if len(x)==6 else 0.,np.exp(x[3])]])
    def transform(self,points,x):
        return self.anchor+(np.asarray(points)-self.anchor)/self.scale@self.matrix(x).T*self.scale+np.asarray(x[:2])*self.scale
    def sampled(self,sub):
        if sub not in self.cache:
            result=[]
            for g in self.groups:
                p=self.base[g['bi']];n=len(p);edges=n if g['closed']else n-1;t=np.linspace(0.,edges,edges*sub+1);i=np.minimum(np.floor(t).astype(int),edges-1);u=t-i
                result.append(p[i]*(1-u[:,None])+p[(i+1)%n]*u[:,None])
            self.cache[sub]=result
        return self.cache[sub]
    def evaluate(self,x,sub=8):
        error=np.empty((len(self.rows),2));valid=np.zeros(len(self.rows),bool);samples=self.sampled(sub)
        for sid,cam in self.cameras.items():
            selected=[(g,p)for g,p in zip(self.groups,samples)if g['shot']==sid];world=[];times=[]
            for g,p in selected:
                uv=self.transform(p,x);world.append(self.scene.board(uv[:,0],uv[:,1]));times.append(np.full(len(p),(g['pts']-self.model['shots'][sid]['time_origin_pts'])/12800.))
            pr=cam.project_rs(np.concatenate(world),np.concatenate(times));offset=0
            for g,p in selected:
                n=len(p);e,v=closest(g['UV'],pr['uv'][offset:offset+n],pr['valid'][offset:offset+n]);offset+=n;error[g['indices']]=e;valid[g['indices']]=v
        return error,valid
    def objective(self,x,sub=8,mask=None):
        e,v=self.evaluate(x,sub);take=self.admitted if mask is None else self.admitted&mask
        residual=(e[take]*np.sqrt(self.weight[take,None])).ravel()
        p=self.transform(np.concatenate(list(self.base.values())),x)
        # Numerical domain barrier only, never an additional source measurement.
        barrier=np.r_[np.minimum(p[:,1],0.),np.maximum(p[:,1]-self.scene.height,0.)]*1e6
        det=np.linalg.det(self.matrix(x));barrier=np.r_[barrier,min(det-1e-4,0.)*1e4]
        return np.r_[residual,barrier]
    def report(self,x,sub=8):
        e,v=self.evaluate(x,sub);s=stats(e,v);s['admitted']=stats(e[self.admitted],v[self.admitted]);s['not_admitted']=stats(e[~self.admitted],v[~self.admitted])
        s['by_shot']={sid:stats(e[ix],v[ix])for sid in self.cameras for ix in [np.array([r['shot_id']==sid for r in self.rows])]}
        s['by_exposure']={sid+':'+str(pts):dict(all=stats(e[ix],v[ix]),admitted=stats(e[ix&self.admitted],v[ix&self.admitted]),not_admitted=stats(e[ix&~self.admitted],v[ix&~self.admitted]))for sid,pts in sorted({(r['shot_id'],r['native_pts'])for r in self.rows})for ix in [np.array([r['shot_id']==sid and r['native_pts']==pts for r in self.rows])]}
        s['by_target']={target:stats(e[ix],v[ix])for target in sorted({r['target_id']for r in self.rows})for ix in [np.array([r['target_id']==target for r in self.rows])]}
        return s,e,v
    def converged(self,x):
        sub=8;old=self.report(x,sub)
        while True:
            sub*=2;new=self.report(x,sub);delta=float(np.max(np.abs(np.linalg.norm(old[1],axis=1)-np.linalg.norm(new[1],axis=1))))
            if np.array_equal(old[2],new[2])and delta<=.001:return new,dict(subdivisions_per_original_segment=sub,max_successive_distance_change_px=delta,tolerance_px=.001)
            old=new


def run(binding,model_file,s01_packet,packet_file,material_map,pixel_groups,ad,output,affine=True):
    start=time.perf_counter();out=Path(output)
    if out.exists():raise ValueError('immutable glyph capture output exists')
    model=read_json(model_file);mapping=read_json(material_map);packet=read_json(packet_file);A=load_arrays(packet);indices={rid:i for i,rid in enumerate(packet['source_row_ids'])};q=QualifiedSourceGenerationInput(binding);q.assert_dependencies();all_rows=[r for r in q.rows()if (r.get('target_id')or'').startswith('Ad'+str(ad)+'.')];entries=[];other=[]
    for r in all_rows:
        d=mapping.get('row_to_material',{}).get(r['source_row_id'],mapping['semantic_to_material'].get(r.get('joint_material_id'),{}));mi=d.get('material_index');paths=model['materials'][mi]['paths']if mi is not None else []
        if len(paths)==1 and paths[0]['kind']==6:entries.append(dict(row=r,binding=d))
        else:other.append(dict(row=r,binding=d,path_kinds=[p['kind']for p in paths]))
    if not entries:raise ValueError('same_physical_ad_has_no_new_kind6_source')
    weights=production_weights(packet,A,read_json(pixel_groups));cap=Capture(model,entries,weights,indices)
    for e in cap.entries:
        i=indices[e['row']['source_row_id']]
        if bool(A['admission'][i])!=bool(e['row']['source_admitted']):raise ValueError('new_glyph_production_admission_differs')
    for mi,m in enumerate(model['materials']):
        for p in m['paths']:
            if p.get('geometry_block')in cap.blocks and mi not in cap.mi:raise ValueError('glyph_block_is_shared_outside_same_physical_ad')
    baseline,bproof=cap.converged(np.zeros(2));stages=[];x=np.zeros(2)
    for n in ([2,4,6]if affine else[2,4]):
        seed=np.r_[x,np.zeros(n-len(x))];lower=[-cap.scene.perimeter/cap.scale[0],-cap.scene.height/cap.scale[1]]+([-2.,-2.]if n>=4 else[])+([-1.,-1.]if n==6 else[]);upper=[cap.scene.perimeter/cap.scale[0],cap.scene.height/cap.scale[1]]+([2.,2.]if n>=4 else[])+([1.,1.]if n==6 else[])
        fit=least_squares(cap.objective,seed,bounds=(lower,upper),x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-7)
        x=fit.x;detail,proof=cap.converged(x);stages.append(dict(parameters=n,x=x.tolist(),success=bool(fit.success),message=fit.message,nfev=fit.nfev,cost=float(fit.cost),summary=detail[0],finite_curve_convergence=proof,J_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist()))
        print(json.dumps(dict(ad=ad,parameters=n,RMS=detail[0]['full_rms_px'],admitted=detail[0]['admitted']['full_rms_px'],nfev=fit.nfev)),flush=True)
    updates=[]
    for bi,p in cap.base.items():
        values=cap.transform(p,x).ravel();b=model['blocks'][bi]
        if np.any(values<np.asarray(b['lower']))or np.any(values>np.asarray(b['upper'])):raise ValueError('conditional_glyph_capture_violates_original_block_bounds')
        updates.append(dict(block_name=b['name'],block_index=bi,expected_values=b['values'],values=values.tolist()))
    overlay=dict(schema='rinkfit.source_conditioned_board_glyph_capture.v1',base_model=ref(model_file),input_binding=ref(binding),physical_advertisement=ad,conditional_shared_parameters=x.tolist(),parameters=len(x),anchor=cap.anchor.tolist(),coordinate_scale=cap.scale.tolist(),normalized_chart_affine=cap.matrix(x).tolist(),block_updates=updates,scene_camera_old_print_S01_unchanged=True,relative_glyph_topology_fixed_under_one_shared_affine=True,source_UV_ID_PTS_qualification_unchanged=True,world_truth_or_statistical_W_created=False,conditional_estimate_not_calibration=True)
    result=apply_overlay(model,overlay);domain=validate_physical_domain(result)
    if not domain['passed']:raise ValueError('conditional_glyph_capture_fails_production_physical_domain:'+','.join(domain['violations']))
    after,aproof=cap.converged(x)
    # Read the updated blocks through the ordinary production scene consumer.
    test=Capture(result,entries,weights,indices);consumer,cproof=test.converged(np.zeros(2))
    if np.max(np.abs(np.linalg.norm(after[1],axis=1)-np.linalg.norm(consumer[1],axis=1)))>1e-7:raise ValueError('written_kind6_model_consumer_does_not_reproduce_shared_affine')
    sp=read_json(s01_packet);SA=load_arrays(sp);used=set(np.unique(SA['material_index']).tolist());dependencies={p.get('geometry_block')for mi in used for p in model['materials'][mi]['paths']}
    if dependencies.intersection(cap.blocks):raise ValueError('conditional_glyph_capture_changes_S01_geometry_dependency')
    if result['materials']!=model['materials']or any(a!=b for i,(a,b)in enumerate(zip(model['blocks'],result['blocks']))if i not in cap.blocks):raise ValueError('conditional_glyph_capture_changed_other_geometry_or_camera')
    out.mkdir(parents=True);write_json(out/'parameter_overlay.json',overlay);write_json(out/'model_captured.json',result);write_json(out/'production_physical_domain.json',domain);write_json(out/'all_original_ad_rows.json',all_rows);write_json(out/'unmodified_other_identities.json',other)
    with gzip.open(out/'source_distances.jsonl.gz','wt',encoding='utf8')as f:
        for r,e0,e1,v0,v1 in zip(cap.rows,baseline[1],after[1],baseline[2],after[2]):f.write(json.dumps(dict(source_row_id=r['source_row_id'],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r['target_id'],source_uv=r['source_uv'],source_admitted=r['source_admitted'],current_uv=r.get('current_uv'),before_error=e0.tolist(),after_error=e1.tolist(),before_valid=bool(v0),after_valid=bool(v1)),ensure_ascii=False)+'\n')
    report=dict(schema='rinkfit.source_conditioned_board_glyph_capture_evidence.v1',physical_advertisement=ad,input_references=dict(model=ref(model_file),binding=ref(binding),packet=ref(packet_file),material_map=ref(material_map),engineering_pixel_weights=ref(pixel_groups),producer=ref(__file__)),all_ad_source_rows=len(all_rows),new_kind6_source_rows=len(cap.rows),fitted_admitted_rows=int(cap.admitted.sum()),not_admitted_rows_retained=int((~cap.admitted).sum()),unknown_or_other_unchanged_rows=len(other),new_materials=len(cap.mi),new_geometry_blocks=len(cap.blocks),before=baseline[0],after=after[0],stages=stages,numerical_convergence=dict(before=bproof,after=aproof,actual_model_consumer=cproof),source_RMS_is_not_formal_statistics=True,objective='actual_production_engineering_pixel_dependency_weights_with_original_source_UV; no_robust_row_deletion',covariance='unknown: conditional numerical_J_singular_values_are_not_statistical_parameter_uncertainty',parameters=len(x),source_pixels_and_status_never_changed=True,relative_new_glyph_shape_and_topology_fixed_under_shared_affine=True,all_old_print_scene_camera_and_model_topology_unchanged=True,model=ref(out/'model_captured.json'),overlay=ref(out/'parameter_overlay.json'),physical_domain=ref(out/'production_physical_domain.json'),production_physical_domain_passed=domain['passed'],S01_dependency_proof=dict(rows=len(SA['material_index']),active=int(SA['admission'].sum()),modified_geometry_dependency_rows=0,source_packet=ref(s01_packet)),new_contours_do_not_use_ad_local_affine_witness=True,seconds=time.perf_counter()-start,limitations=['Conditional on frozen legal05ef scene and all-time cameras; no source world coordinates were measured.','The same complete affine acts on every new contour of this physical ad, across all its source exposures/shots.','All source failures and unmatched identities are retained; source_admitted alone selects the capture objective.','Camera/scene error and source template shape mismatch can remain; this is a placement initializer, not a physical optimum or full-shot acceptance.','Native exact visibility/finite-foot capture is to be checked by the existing C++ consumer before joint Solve.'])
    write_json(out/'capture_evidence.json',report);q.assert_dependencies();print(json.dumps(dict(ad=ad,out=str(out),before=baseline[0]['full_rms_px'],after=after[0]['full_rms_px'],seconds=report['seconds'])),flush=True);return report


def main():
    p=argparse.ArgumentParser()
    for k in ('binding','model','s01-packet','packet','material-map','pixel-groups','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--ad',type=int,required=True);p.add_argument('--no-affine',action='store_true');a=p.parse_args()
    run(a.binding,a.model,a.s01_packet,a.packet,a.material_map,a.pixel_groups,a.ad,a.out,not a.no_affine)
if __name__=='__main__':main()
