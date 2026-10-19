"""Original raster cells are hard supports; sigma0 is never divided/softened."""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import sys,os,json,hashlib,time
os.environ['OPENBLAS_NUM_THREADS']='1'
BASE=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
import numpy as np
from simple_wide_perspective_reset_candidate.audit_simple_wide_perspective import CENTER,inverse_map_forward,sha
PACKET=BASE/'user_exact_trapezoid_candidate/user_exact_trapezoid_native_pixels_and_raster_cells_v2.json'
EXPECTED='0a36c29310e2ec46f3d460a437720571f70972735327dcd4f70a5258e28cc7a9'
def corrected(native,state,with_jac=False):
    c=1000*state[:2];v=(native-c)/1000.;rr=np.sum(v*v,axis=-1);g=1+state[2]*rr+state[3]*rr*rr
    out=c+1000*v*g[...,None]
    if not with_jac:return out
    D=g[...,None,None]*np.eye(2)+2*(state[2]+2*state[3]*rr)[...,None,None]*v[..., :,None]*v[...,None,:]
    J=np.zeros((*native.shape,4));J[...,:2]=1000*(np.eye(2)-D);J[...,2]=(native-c)*rr[...,None];J[...,3]=(native-c)*(rr*rr)[...,None]
    return out,J
def radial_disk_guard(state,native_points):
    c=1000*state[:2];u=float(np.max(np.sum(((native_points-c)/1000)**2,axis=-1)))
    def minimum(a,b):
        xs=[0.,u]
        if b>0 and 0<-a/(2*b)<u:xs.append(-a/(2*b))
        return min(1+a*x+b*x*x for x in xs)
    m=min(minimum(state[2],state[3]),minimum(3*state[2],5*state[3]))
    if not np.isfinite(m)or m<=1e-8:raise ValueError('Radial axis-connected one-to-one disk lost')
    return m
def corners(cells):
    cells=np.asarray(cells,float)
    return np.stack([cells[:,0],np.c_[cells[:,1,0],cells[:,0,1]],cells[:,1],np.c_[cells[:,0,0],cells[:,1,1]]],axis=1)
class ExactUserRasterShape:
    """map4 + two unknown undistorted lines(alpha,d) =8 coordinates.

    A sign crossing between two native cell corners proves the continuous
    forward curve intersects the exact original cell. Unproved intersections
    are not treated as covered. No engineering width, sigma or source weight.
    """
    def __init__(self,packet=PACKET,*,support_mode='center_cell'):
        if sha(packet)!=EXPECTED:raise ValueError('Exact authoritative raster packet changed')
        self.packet=json.loads(Path(packet).read_text(encoding='utf8'));self.groups=self.packet['rows']
        self.support_mode=support_mode;self.center_cells=[np.asarray(g['exact_center_raster_cells_native'],float)for g in self.groups];self.cells=[];self.region_counts=[]
        if support_mode not in ('center_cell','original_pen_footprint_at_same_original_x_cell'):raise ValueError('Explicit original support mode required')
        for group,center_cells in zip(self.groups,self.center_cells):
            lists=[]
            for center in center_cells:
                if support_mode=='center_cell':regions=center[None,:,:].copy()
                else:
                    original=np.asarray(group['exact_original_pen_footprint_raster_cells_native'],float)
                    eligible=(original[:,0,0]<=center[1,0]+1e-12)&(original[:,1,0]>=center[0,0]-1e-12)
                    regions=original[eligible].copy();regions[:,0,0]=np.maximum(regions[:,0,0],center[0,0]);regions[:,1,0]=np.minimum(regions[:,1,0],center[1,0])
                    if not len(regions):raise ValueError('Original center x-cell has no original pen footprint; cannot fabricate support')
                lists.append(regions)
            counts=np.array([len(x)for x in lists]);size=int(counts.max());padded=np.empty((len(lists),size,2,2))
            for i,regions in enumerate(lists):padded[i,:len(regions)]=regions;padded[i,len(regions):]=regions[0]
            self.cells.append(padded);self.region_counts.append(counts)
        self.corner_arrays=[corners(c.reshape(-1,2,2)).reshape(len(c),c.shape[1],4,2)for c in self.cells]
        assert [len(c)for c in self.cells]==[102,166]and all(g['source_sigma_px']==0 for g in self.groups)
        self.all_native=np.concatenate([c.reshape(-1,2)for c in self.corner_arrays])
    def initial(self,map4):
        out=list(map4)
        for group in self.groups:
            uv=np.asarray(group['native_source_UV'],float);xy=(corrected(uv,map4)-CENTER)/1000.;center=xy.mean(axis=0);_,_,vt=np.linalg.svd(xy-center,full_matrices=False);n=vt[-1]
            if n[1]<0:n=-n
            out.extend([np.arctan2(n[1],n[0]),n@center])
        return np.asarray(out)
    def evaluate(self,q,jac=True,report=False):
        q=np.asarray(q,float);margin=radial_disk_guard(q[:4],self.all_native);res=[];js=[];records=[]
        for group_index,cs in enumerate(self.corner_arrays):
            alpha,d=q[4+2*group_index:6+2*group_index];n=np.array([np.cos(alpha),np.sin(alpha)]);dn=np.array([-np.sin(alpha),np.cos(alpha)])
            if jac:u,ju=corrected(cs,q[:4],True)
            else:u=corrected(cs,q[:4])
            z=(u-CENTER)/1000.;F=z@n-d;mins=np.argmin(F,axis=2);maxs=np.argmax(F,axis=2);ii=np.arange(len(cs));rr=np.arange(cs.shape[1]);fmin=np.take_along_axis(F,mins[:,:,None],axis=2)[:,:,0];fmax=np.take_along_axis(F,maxs[:,:,None],axis=2)[:,:,0]
            each_covered=(fmin<=0)&(fmax>=0);each_value=np.where(each_covered,0.,np.where(fmin>0,fmin,fmax));region=np.argmin(abs(each_value),axis=1)
            covered=each_covered[ii,region];chosen=np.where(fmin[ii,region]>0,mins[ii,region],maxs[ii,region]);r=each_value[ii,region]*1000.;res.append(r)
            if jac:
                J=np.zeros((len(cs),8));J[:,:4]=np.einsum('nij,i->nj',ju[ii,region,chosen],n)
                J[:,4+2*group_index]=z[ii,region,chosen]@dn*1000.;J[:,5+2*group_index]=-1000.;J[covered]=0.;js.append(J)
            if report:
                roots=[]
                for i in range(len(cs)):
                    ri=region[i];lo=cs[i,ri,mins[i,ri]].copy();hi=cs[i,ri,maxs[i,ri]].copy()
                    if covered[i]:
                        for _ in range(55):
                            middle=.5*(lo+hi);f=float(((corrected(middle[None,:],q[:4])[0]-CENTER)/1000.)@n-d)
                            if f<=0:lo=middle
                            else:hi=middle
                        native=.5*(lo+hi);und=corrected(native[None,:],q[:4]);native_forward=inverse_map_forward(und,q[:4])[0][0]
                        cell=self.cells[group_index][i,ri];inside=bool(np.all(native_forward>=cell[0]-1e-9)&np.all(native_forward<=cell[1]+1e-9))
                        if not inside:raise ValueError('Claimed forward curve missed original exact cell')
                        annotated=(native_forward+np.array([0.,65.])+.5)/np.array([2580/1500,1160/675])-.5
                        roots.append(dict(original_point_index=i,original_native_UV=self.groups[group_index]['native_source_UV'][i],exact_native_cell=cell.tolist(),selected_original_region_index=int(ri),original_region_count=int(self.region_counts[group_index][i]),native_forward_intersection=native_forward.tolist(),annotation_forward_intersection=annotated.tolist(),equation=float(((und[0]-CENTER)/1000.)@n-d),covered=True))
                    else:roots.append(dict(original_point_index=i,original_native_UV=self.groups[group_index]['native_source_UV'][i],exact_native_cell=self.cells[group_index][i,ri].tolist(),covered=False,positive_corner_equation=fmin[i,ri],negative_corner_equation=fmax[i,ri]))
                records.append(dict(id=self.groups[group_index]['id'],support_mode=self.support_mode,original_points=len(cs),proved_original_support_intersections=int(covered.sum()),all_original_supports_covered=bool(covered.all()),points=roots))
        return np.concatenate(res),np.vstack(js)if jac else None,dict(radial_disk_minimum=margin,groups=records)if report else None
    def fit_feasibility(self,start,*,active_indices=None):
        q=np.array(start,float);damping=1e-3;history=[]
        active=np.arange(8)if active_indices is None else np.asarray(active_indices,int)
        for iteration in range(100):
            r,J,_=self.evaluate(q);value=.5*r@r
            if np.max(abs(r))<1e-9:break
            Ja=J[:,active];scale=np.maximum(np.linalg.norm(Ja,axis=0),1e-12);A=Ja/scale;H=A.T@A;g=A.T@r;accepted=False
            for attempt in range(16):
                step=np.zeros(8);step[active]=-np.linalg.solve(H+damping*np.eye(len(active)),g)/scale
                try:candidate=q+step;rr=self.evaluate(candidate,False)[0];trial=.5*rr@rr
                except ValueError:trial=float('inf')
                if trial<value:q=candidate;damping=max(1e-10,damping/3);accepted=True;break
                damping*=5
            history.append(dict(iteration=iteration,hard_feasibility_merit=float(value),maximum_signed_equation_violation=float(abs(r).max()),accepted=accepted))
            if not accepted or value-trial<1e-13*max(1.,value):break
        return q,history
def main():
    started=time.perf_counter();p=ExactUserRasterShape();root=BASE/'simple_wide_physical_reset_candidate/raw_image_straight_lines_and_inverse_radial.json';map4=np.asarray(json.loads(root.read_text(encoding='utf8'))['state']);q0=p.initial(map4)
    r0,J0,initial=p.evaluate(q0,True,True);q,history=p.fit_feasibility(q0);r,J,final=p.evaluate(q,True,True)
    footprint=ExactUserRasterShape(support_mode='original_pen_footprint_at_same_original_x_cell');fq0=footprint.initial(map4);fq,fhistory=footprint.fit_feasibility(fq0);fr,fj,freport=footprint.evaluate(fq,True,True)
    output=dict(schema='exact-user-raster-support-independent-map4-line4-feasibility.v1',packet=dict(path=str(PACKET),sha256=sha(PACKET)),all268authoritativecenters_preserved=True,
        source_sigma_px0_not_divided_or_replaced=True,engineering_tolerance_added=False,source_constraints_are_exact_original_raster_cells=True,
        original_automatic_paint_bands_not_loaded_or_fitted=True,world_coordinates_or_endpoints_not_inferred=True,
        map4_initial=map4.tolist(),state8_initial=q0.tolist(),state8_final=q.tolist(),initial=initial,final=final,
        all268_strict_center_cells_proved=all(x['all_original_supports_covered']for x in final['groups']),strict_cell_feasibility_merit=float(.5*r@r),maximum_line_equation_violation=float(abs(r).max()),
        not_a_global_infeasibility_certificate_if_cells_unproved=True,complete_original_pen_footprint_used_as_separate_contract=True,
        original_pen_footprint_alternative=dict(state8_final=fq.tolist(),support_mode=footprint.support_mode,all268_original_supports_proved=all(x['all_original_supports_covered']for x in freport['groups']),report=freport,optimizer_history=fhistory,no_new_region_or_engineering_tolerance=True),
        optimizer_history=history,production_adopted=False,seconds=time.perf_counter()-started)
    dest=Path(__file__).with_name('actual_user268_strict_center_cell_feasibility.json');dest.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(path=str(dest),coverage=[(x['proved_original_support_intersections'],x['original_points'])for x in final['groups']],footprint=[(x['proved_original_support_intersections'],x['original_points'])for x in freport['groups']],max_equation=output['maximum_line_equation_violation'],state=q.tolist(),footprint_state=fq.tolist(),seconds=output['seconds'])),flush=True)
if __name__=='__main__':main()
