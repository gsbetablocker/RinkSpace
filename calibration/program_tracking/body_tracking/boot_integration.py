"""Cached-data foot/boot integration on the existing continuous body state."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix,coo_matrix,hstack,vstack,block_diag
from rinkspace.viewer3d.data import ResultCatalog
from rinkspace.viewer3d.prepare import FullSurface
from .footwear_binding import FootwearBinding,IntegratedBodyState
from .footwear_observations import visible_shell_edges,audit_observation_semantics
from .footwear_pixel_factors import pixel_factors
from .fullbody_fit import FullBodyProblem,footwear_rows
from .fullbody_state import camera_binding
from .fullbody_round import write_json
from .rebuild_prepare import load,digest
from .rebuild_exposure_body import moving_projection
from .boot_delivery import check_inputs,deliver


def edges_for(world,frames,camera,binding):
    pose=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],float(camera.p['first_control_time']),
        float(camera.p['control_dt']),camera.p['motion_times'][frames])
    centre=pose['centre']*camera.scale
    return {(i,f):visible_shell_edges(world[i,f*48:f*48+14],binding.faces,centre[i])
            for i in range(len(frames)) for f in (0,1)}


def save_baseline_diagnostics(reference,binding,info,output):
    path=Path(output)/'baseline_diagnostics.npz'
    if path.is_file():return
    x=binding.x.copy();enabled=binding.deformation_enabled
    binding.x=np.zeros(14);binding.deformation_enabled=False
    fields={k:[] for k in ('cavity_penetration_mid','heel_drift_mid','forefoot_error_mid',
        'upper_deformation_mid','effective_ankle_relative_rotation_mid','blade_tilt_rad_mid')}
    for start in range(0,len(reference.times),64):
        s=reference.world(reference.at(reference.times[start:start+64]));d=binding.diagnostics(s)
        for k in fields:fields[k].append(d[k])
    packet={k:np.concatenate(v) for k,v in fields.items()}
    for key in ('cavity_penetration_mid','heel_drift_mid','forefoot_error_mid'):packet[key]*=info['length_export_ratio']
    packet.update(pts=reference.p['pts'],time_base=np.array([1,12800]),binding_x0=np.zeros(14),
        metres_per_scene_unit=np.array(info['metres_per_scene_unit']),
        interpretation=np.array('same newly defined approximate cavity/footbed diagnostics applied to legacy geometry; not a measured historical inner shoe'),
        parent_result_sha256=np.array(digest(reference.baseline/'full/result.npz')))
    np.savez_compressed(path,**packet);binding.x=x;binding.deformation_enabled=enabled


class Registration:
    def __init__(self,reference,camera,binding,output,exclude_windows=()):
        self.ref,self.camera,self.binding,self.output=reference,camera,binding,Path(output)
        # All existing reviewed heel/toe + blade source frames, plus the stated
        # deep-flexion samples; no rerun of detection or anatomy relabelling.
        allgear,_=footwear_rows(reference.root,np.arange(len(reference.times)),reference.p['pts'])
        wanted=sorted({r['pts'] for r in allgear.landmark_rows+allgear.blade_rows}|{336128,421888,461056,489216,521728})
        wanted=[p for p in wanted if not any(a-1e-9<=p/12800<=b+1e-9 for a,b in exclude_windows)]
        self.frames=np.asarray([int(np.searchsorted(reference.p['pts'],p)) for p in wanted])
        self.gear,self.sources=footwear_rows(reference.root,self.frames,reference.p['pts'])
        t=reference.times[self.frames]
        self.times=(t[:,None]+np.linspace(-.5,.5,5)*float(camera.p['rig'][37])).ravel()
        saved=reference.binding;reference.binding=None
        self.state=reference.world(reference.at(self.times));reference.binding=saved
        j=np.einsum('nji,nkj->nki',self.state['Q'],self.state['native']-self.state['P'][:,None])
        self.local=j;self.mid=np.arange(len(self.frames))*5+2
        self.samples=binding.surface.sample_vertices(j[self.mid],self.state['frames'][self.mid],binding.sample_ids,double_precision=True)
        self.count=0;self.start=time.perf_counter();self.last_parts={}

    def residual(self,x,log=False):
        b=self.binding;s=self.state
        gear=b.evaluate(s['P'],s['Q'],self.local,s['frames'],x=x)
        W=np.concatenate((gear['gear'].reshape(-1,96,3),s['coco']),1).reshape(len(self.frames),5,113,3)
        uv=moving_projection(self.camera,W,self.frames)['uv']
        edge=edges_for(W[:,2],self.frames,self.camera,b)
        pixels,_,parts,_=pixel_factors(uv,self.gear,body=False,boot_edges=edge)
        neutral=b.compatibility_rows(b.neutral_native,b.neutral_frames,b.neutral_samples,x=x).ravel()*2
        dynamic=b.compatibility_rows(self.local[self.mid],s['frames'][self.mid],self.samples,x=x).ravel()/np.sqrt(len(self.frames))*2
        prior=np.r_[x[:12].reshape(2,6)[:,:3].ravel()/.08,x[:12].reshape(2,6)[:,3:].ravel()/.5,x[12:]/.2]
        result=np.r_[pixels,neutral,dynamic,prior]
        self.last_parts={**parts,'neutral_foot_fit':float(neutral@neutral/2),'dynamic_foot_fit':float(dynamic@dynamic/2),'registration_prior':float(prior@prior/2)}
        if log:
            self.count+=1
            with (self.output/'fit_progress.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(dict(stage='shared_registration',evaluation=self.count,cost=float(result@result/2),parts=self.last_parts,seconds=time.perf_counter()-self.start))+'\n')
        return result

    def solve(self):
        x0=self.binding.x.copy();r0=self.residual(x0)
        lower=np.r_[np.tile(np.r_[[-.2]*3,[-.8]*3],2),np.log([.75,.7])]
        upper=np.r_[np.tile(np.r_[[.2]*3,[.8]*3],2),np.log([1.35,1.65])]
        def accepted(x):
            np.savez_compressed(self.output/'registration_last_accepted.npz',x=x)
        fit=least_squares(lambda x:self.residual(x,True),x0,jac='3-point',method='trf',bounds=(lower,upper),
            x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-5,callback=accepted)
        self.binding.x=fit.x.copy();r=self.residual(fit.x)
        np.savez_compressed(self.output/'registration.npz',x=fit.x,initial_x=x0)
        report=dict(initial_cost=float(r0@r0/2),final_cost=float(r@r/2),parts=self.last_parts,
            source_pts=self.ref.p['pts'][self.frames].tolist(),shared_parameters=14,per_frame_equipment_parameters=0,
            stopping_reason=fit.message,gradient_inf=float(fit.optimality),nfev=fit.nfev,
            geometry_assumptions='shared-beta neutral + frozen dynamic full surface; finite lining/bed priors, no inferred internal measurements')
        write_json(self.output/'registration.json',report);return report


class BootWindowProblem(FullBodyProblem):
    def __init__(self,reference,camera,ledger,seconds,log=None):
        super().__init__(reference,camera,ledger,seconds,'full',log)
        self.binding=reference.binding;self.mid_times=reference.times[self.frames]
        self.mid_ref=reference.at(self.mid_times);self.binding_key=None

    def visible_boot_edges(self,world):
        return edges_for(world,self.frames,self.projector,self.binding)

    def compatibility(self,delta):
        s=self.reference.world(self.mid_ref,delta)
        j=np.einsum('nji,nkj->nki',s['Q'],s['native']-s['P'][:,None])
        return self.binding.compatibility_rows(j,s['frames'])/np.sqrt(2)

    def evaluate(self,x,jacobian=False):
        key=self.binding.x.tobytes()
        if self.binding_key!=key:self.cache=None;self.cache_x=None;self.binding_key=key
        base=super().evaluate(x,jacobian)
        if base.get('foot_compatibility_added'):return base
        d=self.state.increments(x,self.mid_times);extra=self.compatibility(d);rows=extra.size;blocks=[]
        if jacobian:
            derivatives=[];h=1e-5
            for active in self.state.active:
                if active<6:derivatives.append(np.zeros_like(extra));continue
                a=d.copy();b=d.copy();a[:,active]+=h;b[:,active]-=h
                derivatives.append((self.compatibility(a)-self.compatibility(b))/(2*h))
            local=np.stack(derivatives,-1);B=self.state.basis(self.mid_times)
            rr=[];cc=[];vv=[];fr,kn=np.nonzero(B)
            val=local[fr]*B[fr,kn,None,None]
            rr=np.broadcast_to(fr[:,None,None]*extra.shape[1]+np.arange(extra.shape[1])[None,:,None],val.shape).ravel()
            cc=np.broadcast_to(kn[:,None,None]*self.state.d+np.arange(self.state.d),val.shape).ravel()
            J=coo_matrix((val.ravel(),(rr,cc)),shape=(rows,self.size)).tocsr();J.eliminate_zeros()
            base['jac']=vstack((base['jac'],J),format='csr')
        base['residual']=np.r_[base['residual'],extra.ravel()]
        base['parts']['foot_cavity_support']=float(np.sum(extra**2)/2);base['foot_compatibility_added']=True
        return base


class JointBootProblem:
    def __init__(self,registration,windows,output):
        # Every source observation has one current state owner. The static
        # calibration block keeps only frames OUTSIDE the adjustable windows.
        self.registration=Registration(registration.ref,registration.camera,registration.binding,output,
            exclude_windows=[w.state.seconds for w in windows])
        self.binding=registration.binding;self.windows=windows;self.output=Path(output)
        self.sizes=[w.size for w in windows];self.offsets=np.r_[14,14+np.cumsum(self.sizes)]
        self.x0=np.r_[self.binding.x,*[np.zeros(n) for n in self.sizes]]
        self.cached=None;self.cached_x=None;self.calls=0;self.start=time.perf_counter()

    def evaluate(self,x,jacobian=False):
        if self.cached_x is not None and np.array_equal(x,self.cached_x) and (not jacobian or self.cached['jac'] is not None):return self.cached
        self.binding.x=x[:14].copy();static=self.registration.residual(x[:14]);parts=dict(registration=self.registration.last_parts)
        values=[];localJ=[]
        for i,w in enumerate(self.windows):
            v=w.evaluate(x[self.offsets[i]:self.offsets[i+1]],jacobian);values.append(v['residual']);parts[f'window_{i+1}']=v['parts'].copy()
            if jacobian:localJ.append(v['jac'])
        residual=np.r_[static,*values];J=None
        if jacobian:
            # Only 14 newly shared registration/size variables are differenced
            # as global columns. Body columns reuse local FK/exposure chains.
            cols=[];h=1e-6
            for k in range(14):
                side=[]
                for sign in (1,-1):
                    y=x[:14].copy();y[k]+=sign*h;self.binding.x=y
                    rs=[self.registration.residual(y)]
                    for i,w in enumerate(self.windows):rs.append(w.evaluate(x[self.offsets[i]:self.offsets[i+1]],False)['residual'])
                    side.append(np.concatenate(rs))
                cols.append((side[0]-side[1])/(2*h))
            self.binding.x=x[:14].copy()
            bodyJ=vstack([csr_matrix((len(static),sum(self.sizes))),block_diag(localJ,format='csr')],format='csr')
            J=hstack((csr_matrix(np.stack(cols,1)),bodyJ),format='csr');J.eliminate_zeros()
        self.calls+=1
        with (self.output/'fit_progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(stage='shared_registration_and_body',evaluation=self.calls,cost=float(residual@residual/2),parts=parts,
                variables=len(x),jacobian_nnz=None if J is None else J.nnz,seconds=time.perf_counter()-self.start))+'\n')
        self.cached_x=x.copy();self.cached=dict(residual=residual,jac=J,parts=parts);return self.cached

    def solve(self):
        first=self.evaluate(self.x0,True)
        def accepted(x):np.savez_compressed(self.output/'joint_last_accepted.npz',x=x)
        lower=np.r_[np.tile(np.r_[[-.2]*3,[-.8]*3],2),np.log([.75,.7]),np.full(sum(self.sizes),-np.pi)]
        upper=np.r_[np.tile(np.r_[[.2]*3,[.8]*3],2),np.log([1.35,1.65]),np.full(sum(self.sizes),np.pi)]
        for i,w in enumerate(self.windows):
            index=self.offsets[i]+np.flatnonzero(np.tile(w.state.active<3,w.state.m));lower[index]=-np.inf;upper[index]=np.inf
        fit=least_squares(lambda x:self.evaluate(x)['residual'],self.x0,jac=lambda x:self.evaluate(x,True)['jac'],
            method='trf',tr_solver='lsmr',bounds=(lower,upper),x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-4,
            callback=accepted,tr_options=dict(atol=1e-7,btol=1e-7))
        final=self.evaluate(fit.x,True);self.binding.x=fit.x[:14].copy()
        np.savez_compressed(self.output/'joint_result.npz',x=fit.x,offsets=self.offsets)
        report=dict(initial_cost=float(first['residual']@first['residual']/2),final_cost=float(fit.cost),parts=final['parts'],
            stopping_reason=fit.message,optimality=float(fit.optimality),nfev=fit.nfev,variables=len(fit.x),
            shared_registration_parameters=14,per_frame_shoe_pose_parameters=0,source_windows=[w.state.seconds for w in self.windows])
        write_json(self.output/'joint_result.json',report)
        for i,w in enumerate(self.windows):
            dest=self.output/f'w{i+1}';dest.mkdir(exist_ok=True)
            x=fit.x[self.offsets[i]:self.offsets[i+1]]
            np.savez_compressed(dest/'result.npz',x=x,knots=w.state.knots,active=w.state.active,seconds=w.state.seconds)
            self.registration.ref.refinements.append((w.state,x.copy()))
        return report

    def verify_new_dependencies(self):
        x=self.x0.copy();x[14:]=.0003
        # Evaluate away from fitted nearest-edge/containment switching points.
        x[:14]+=np.array([.001,-.001,.0007,.007,-.004,.005,-.001,.0008,-.0006,-.006,.005,.004,.008,-.007])
        value=self.evaluate(x,True);rng=np.random.default_rng(1102);checks=[]
        for name in ('shared_registration','upper_contact_driver'):
            direction=np.zeros_like(x)
            if name=='shared_registration':direction[:14]=rng.normal(size=14)
            else:
                w=self.windows[0];active=np.flatnonzero(np.isin(w.state.active,[6+(j-1)*3+a for j in (7,8) for a in range(3)]))
                view=direction[self.offsets[0]:self.offsets[1]].reshape(w.state.m,w.state.d)
                view[:,active]=rng.normal(size=(w.state.m,len(active)))
            direction/=np.linalg.norm(direction);h=1e-6
            predicted=value['jac']@direction
            numeric=(self.evaluate(x+h*direction)['residual']-self.evaluate(x-h*direction)['residual'])/(2*h)
            relative=float(np.linalg.norm(predicted-numeric)/max(np.linalg.norm(numeric),1e-12))
            checks.append(dict(block=name,relative_norm=relative,max_absolute=float(np.max(abs(predicted-numeric)))))
            write_json(self.output/'derivative_checks.json',checks)
            if relative>3e-3:raise ValueError('New footwear dependency derivative mismatch: '+str(checks[-1]))
        self.binding.x=self.x0[:14].copy();write_json(self.output/'derivative_checks.json',checks)
        return checks


def export(reference,camera,info,output,*,source_path=None):
    output=Path(output);b=reference.binding;n=len(reference.times);ratio=info['length_export_ratio']
    source=load(source_path or reference.baseline/'full/result.npz');array_names=('P','Q','local_R','native','coco','gear')
    result={k:[] for k in array_names};diagnostics={};body_uv=[];gear_uv=[];row=[];inside=[];root_uv=[];row_time=[]
    for start in range(0,n,32):
        frames=np.arange(start,min(start+32,n));t=reference.times[frames]
        ts=(t[:,None]+np.linspace(-.5,.5,5)*float(camera.p['rig'][37])).ravel()
        s=reference.world(reference.at(ts));m=reference.world(reference.at(t))
        for k in result:result[k].append(m[k])
        d=b.diagnostics(m)
        for k,v in d.items():
            if k.endswith('_mid'):diagnostics.setdefault(k,[]).append(v)
            else:diagnostics[k]=v
        p=moving_projection(camera,s['world'].reshape(len(frames),5,113,3),frames)
        body_uv.append(p['uv'][:,96:]);gear_uv.append(p['uv'][:,:96].reshape(-1,2,48,2));row.append(p['world']*ratio);inside.append(p['inside_exposure_rows'])
        root_uv.append(moving_projection(camera,s['P'].reshape(len(frames),5,1,3),frames)['uv'][:,0])
        row_time.append(reference.p['pts'][frames,None]/12800+float(camera.p['rig'][37])*(np.clip(p['uv'][...,1],0,1079)/1079-.5))
    r={k:np.concatenate(v) for k,v in result.items()}
    for k,v in diagnostics.items():
        if isinstance(v,list):diagnostics[k]=np.concatenate(v)
    for k in ('boot_surface_world_mid','footbed_world_mid','cavity_world_mid','anchors_world_mid','axes_world_mid',
              'cavity_penetration_mid','heel_drift_mid','forefoot_error_mid'):
        diagnostics[k]*=ratio
    diagnostics['blade_pose_world_mid'][...,-1]*=ratio
    # upper_deformation_mid reports normalized mode activity, not length.
    diagnostics['upper_deformation_mid']/=b.maximum_modes
    P=r['P']*ratio;G=P.copy();G[:,2]=0
    packet=dict(source,**diagnostics)
    for key in ('solver_config_json','active_blocks_json','stopping_reason_json','costs_json'):
        if key in source:packet['parent_'+key]=source[key]
    current_report=json.loads((output/'joint_result.json').read_text(encoding='utf8')) if (output/'joint_result.json').is_file() else {}
    refinement_mask=np.zeros(n,bool)
    for ws,_ in reference.refinements:refinement_mask|=(reference.p['pts']/12800>=ws.seconds[0]-1e-9)&(reference.p['pts']/12800<=ws.seconds[1]+1e-9)
    packet.update(P_world_mid=P,Q_world=r['Q'],Q_world_mid=r['Q'],local_R_mid=r['local_R'],native22_world_mid=r['native']*ratio,
        COCO_world_mid=r['coco']*ratio,gear_world_mid=r['gear']*ratio,boot_world_mid=r['gear'][:,:,:14]*ratio,
        blade_world_mid=r['gear'][:,:,14:]*ratio,G_world_mid=G,COCO_uv=np.concatenate(body_uv),equipment_uv=np.concatenate(gear_uv),
        P_uv_dynamic=np.concatenate(root_uv),world_at_row=np.concatenate(row),source_row_time=np.concatenate(row_time),
        inside_exposure_rows=np.concatenate(inside),footwear_binding_json=np.array(json.dumps(b.record(),sort_keys=True)),
        observation_ledger_id=np.array(digest(output/'ledger.json')),
        footwear_integration=np.array('shared T_FB, cavity, contact-driven three upper modes; unchanged body topology'),
        footwear_binding_applied_mask=np.ones(n,bool),body_refinement_window_mask=refinement_mask,
        solver_config_json=np.array(json.dumps(dict(registration='registration.json',joint='joint_result.json',ledger='ledger.json'))),
        active_blocks_json=np.array(json.dumps(dict(shared_binding=14,body_columns=[ws.active.tolist() for ws,_ in reference.refinements]))),
        stopping_reason_json=np.array(json.dumps(current_report.get('stopping_reason','shared registration only'))),
        costs_json=np.array(json.dumps(current_report.get('parts',{}))))
    dest=output/'fitted';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'result.npz',**packet)
    return packet


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--project',default=str(Path(__file__).resolve().parents[3]));parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['prepare','register','joint','export','run'],default='run');a=parser.parse_args(argv)
    project,root,out=Path(a.project),Path(a.root),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'))),flush=True)
    ref=IntegratedBodyState(root,root/'fullbody_1102');camera,info,_,_=camera_binding(project,ref)
    catalog=ResultCatalog(project=project,data_root=root.parents[1],config=project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json')
    binding=FootwearBinding(ref,FullSurface(catalog),info)
    frames=np.array([146,207,695,932]);old=ref.world(ref.at(ref.times[frames]));local=np.einsum('nji,nkj->nki',old['Q'],old['native']-old['P'][:,None])
    migrated=binding.evaluate(old['P'],old['Q'],local,old['frames'],deformation=False)
    error=float(np.max(abs(old['gear']-migrated['gear'])));assert error<1e-11
    write_json(out/'migration_check.json',dict(zero_registration_old_geometry_max_m=error,body_PQ_unchanged=True,boot_align_applied_once=True))
    ref.binding=binding
    save_baseline_diagnostics(ref,binding,info,out)
    observations=load(root/'fullbody_1102/observations.npz');observations['sigma_px'][:,15:17]=24.
    np.savez_compressed(out/'observations.npz',**observations)
    ledger=json.loads((root/'fullbody_1102/ledger.json').read_text(encoding='utf8'))
    ledger.update(parent_ledger=str(root/'fullbody_1102/ledger.json'),parent_sha256=digest(root/'fullbody_1102/ledger.json'),
        ankle_semantics='detector proxy under clothing/boot occlusion, not an observed bony centre',ankle_sigma_px=24.,
        ankle_scale_provenance='existing body solver engineering ankle scale; not calibrated uncertainty',observation_array_sha256=digest(out/'observations.npz'))
    write_json(out/'ledger.json',ledger);audit_observation_semantics(root,output=out/'observation_semantics.json')
    check_inputs(ref,binding,info,out)
    registration=Registration(ref,camera,binding,out)
    if (out/'registration.npz').is_file():binding.x=load(out/'registration.npz')['x']
    elif a.stage in ('register','run'):print(json.dumps(registration.solve()),flush=True)
    elif a.stage in ('joint','export'):raise FileNotFoundError('Shared registration is missing; run --stage register first')
    if a.stage=='prepare':return
    if a.stage in ('joint','run','export'):
        windows=[BootWindowProblem(ref,camera,out/'observations.npz',seconds) for seconds in ((35.2,36.8),(40.,41.5))]
        if (out/'joint_result.npz').is_file():
            z=load(out/'joint_result.npz');binding.x=z['x'][:14]
            for i,w in enumerate(windows):ref.refinements.append((w.state,z['x'][z['offsets'][i]:z['offsets'][i+1]]))
        elif a.stage in ('joint','run'):
            joint=JointBootProblem(registration,windows,out)
            print(json.dumps(dict(new_dependency_checks=joint.verify_new_dependencies())),flush=True)
            print(json.dumps(joint.solve()),flush=True)
        elif a.stage=='export':raise FileNotFoundError('Joint result is missing; export will not substitute an unfitted state')
        if a.stage in ('export','run'):
            packet=export(ref,camera,info,out);write_json(out/'binding.json',binding.record())
            print(json.dumps(dict(export=str(out/'fitted/result.npz'),frames=len(packet['pts']))),flush=True)
            from .boot_review import run as review_boot
            review_boot(root,out)
            from rinkspace.viewer3d.prepare import prepare_variant
            viewer=ResultCatalog(project=project,data_root=root.parents[1],
                config=project/'events/ankara_2026/programmes/okada_mei_sp/viewer3d_boot_1102.json')
            for candidate in viewer.variants:prepare_variant(viewer,candidate,binding.surface)
            print(json.dumps(deliver(project,out)),flush=True)


if __name__=='__main__':main()
