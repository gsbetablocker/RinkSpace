"""Common original-box witness for a print-only board affine quotient.

This decoder only changes chart coordinates. It never fits pixels, invents an
external rectangle, or equates separate advertisement panels. A strict witness
proves that the artificial chart boxes impose no local physical restriction.
"""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

def decode_common_witness(model,advertisement_ids=(8,9,10,11,12)):
    q=copy.deepcopy(model);report=[];seen=set()
    scene=np.asarray(q['blocks'][q['scene_block']]['values']);H=float(scene[4])
    from .joint_model import as_event_view
    from rinkfit_start.native_scene import NativeScene
    native_before=NativeScene(as_event_view(q,next(iter(q['shots']))))
    math=native_before.model if hasattr(native_before,'model') else None
    # Same analytic course integral as the event model; no scene dimensions move.
    from scipy.integrate import quad
    aspect,rx,ry,power=scene[:4]
    def speed(a):
        c,s=np.cos(a),np.sin(a);f=c**power+s**power;d=f**(-1/power);v=(c**(power-1)*s-s**(power-1)*c)/f
        return float(np.hypot(rx*d*(-s+c*v),ry*d*(c+s*v)))
    P=float(2*(1-2*rx+aspect-2*ry)+4*quad(speed,0,np.pi/2,epsabs=1e-12,epsrel=1e-12)[0])
    for adno in advertisement_ids:
        name=f'ad/{adno}';matches=[i for i,b in enumerate(q['blocks']) if b['name']==name]
        if len(matches)!=1:raise ValueError('missing unique chart '+name)
        ai=matches[0];ad=q['blocks'][ai]
        paths=[(mi,pi,p) for mi,m in enumerate(q['materials']) for pi,p in enumerate(m['paths']) if p.get('advertisement_block')==ai]
        if not ad.get('constant',False):raise ValueError('only already-quotiented print-only chart may be decoded')
        if any(p.get('geometry_block',-1)<0 or not q['materials'][mi]['label'].startswith(f'Ad{adno}.print.') for mi,pi,p in paths):raise ValueError('actual external/physical entity invalidates print-only gauge')
        gis=sorted({p['geometry_block'] for mi,pi,p in paths})
        if not gis:raise ValueError('empty print-only chart')
        if any(len(q['blocks'][gi]['values'])!=6 or q['blocks'][gi].get('constant',False) or q['blocks'][gi].get('fixed_indices',[]) for gi in gis):raise ValueError('fixed/internal constrained affine invalidates the full four-coordinate print gauge')
        if any(p.get('advertisement_block')!=ai for m in q['materials'] for p in m['paths'] if p.get('geometry_block') in gis):raise ValueError('a shape affine is shared across different physical advertisement charts')
        if seen.intersection(gis):raise ValueError('affine has multiple physical ad owners')
        seen.update(gis);a=np.asarray(ad['values'],float);dz=a[3]-a[2]
        if a[1]<=0 or dz<=0:raise ValueError('nonpositive original witness scale')
        physical={gi:np.r_[a[1]*np.asarray(q['blocks'][gi]['values'][:2]),a[0]+a[1]*q['blocks'][gi]['values'][2],dz*np.asarray(q['blocks'][gi]['values'][3:5]),a[2]+dz*q['blocks'][gi]['values'][5]] for gi in gis}
        def axis_poly(axis,margin):
            A=[];b=[]
            for gi in gis:
                block=q['blocks'][gi];lo=np.asarray(block['lower'][axis*3:axis*3+3],float);hi=np.asarray(block['upper'][axis*3:axis*3+3],float);lo,hi=lo+margin*(hi-lo),hi-margin*(hi-lo);v=physical[gi][axis*3:axis*3+3]
                for j in (0,1):A.extend([[0,lo[j]],[0,-hi[j]]]);b.extend([v[j],-v[j]])
                A.extend([[1,lo[2]],[-1,-hi[2]]]);b.extend([v[2],-v[2]])
            low=np.asarray(ad['lower'],float);up=np.asarray(ad['upper'],float)
            if axis==0:
                wlo=low[1]+margin*(min(up[1],P)-low[1]);whi=min(up[1],P)-margin*(min(up[1],P)-low[1]);slo=low[0]+margin*(up[0]-low[0]);shi=up[0]-margin*(up[0]-low[0]);bounds=[(slo,shi),(wlo,whi)]
            else:
                zlo=max(low[2],0);zhi=min(up[2],H);tlo=max(low[3],0);thi=min(up[3],H)
                A.extend([[1,1],[-1,-1]]);b.extend([thi-margin*(thi-tlo),-(tlo+margin*(thi-tlo))]);bounds=[(zlo+margin*(zhi-zlo),zhi-margin*(zhi-zlo)),(1e-6+margin*H,None)]
            r=linprog([0,1],A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=bounds,method='highs')
            if not r.success:return None
            return r.x
        def feasible(margin):
            x,z=axis_poly(0,margin),axis_poly(1,margin)
            if x is None or z is None:return None
            for v in physical.values():
                D=v[0]*v[4]-v[1]*v[3]
                if D/(x[1]*z[1])<1e-6*(1+margin):return None
            return x,z
        if feasible(0) is None:raise ValueError('no common original-domain witness '+name)
        lo,hi=0.,.499999
        for _ in range(32):
            m=(lo+hi)/2
            if feasible(m) is None:hi=m
            else:lo=m
        margin=.5*lo;decoded=feasible(margin)
        if decoded is None:raise ValueError('witness bisection inconsistency')
        x,z=decoded;new=np.array([x[0],x[1],z[0],z[0]+z[1]]);before={gi:list(q['blocks'][gi]['values']) for gi in gis};ad['values']=new.tolist()
        maximum=0.
        for gi,v in physical.items():
            t=np.r_[v[:2]/new[1],(v[2]-new[0])/new[1],v[3:5]/z[1],(v[5]-new[2])/z[1]];q['blocks'][gi]['values']=t.tolist()
            after=np.r_[new[1]*t[:2],new[0]+new[1]*t[2],z[1]*t[3:5],new[2]+z[1]*t[5]];maximum=max(maximum,float(np.max(np.abs(after-v))))
            block=q['blocks'][gi]
            if np.any(t<np.asarray(block['lower'])-1e-9) or np.any(t>np.asarray(block['upper'])+1e-9):raise ValueError('decoded affine left original domain')
        report.append({'advertisement_id':adno,'print_affines':len(gis),'path_count':len(paths),'original_witness':a.tolist(),'decoded_witness':new.tolist(),'maximum_feasible_relative_interior_margin':lo,'chosen_relative_interior_margin':margin,'strict_original_box_witness_found':margin>1e-8,'max_composite_coefficient_change':maximum,'affine_changes':[{'block':q['blocks'][gi]['name'],'before':before[gi],'after':q['blocks'][gi]['values']} for gi in gis]})
    native_after=NativeScene(as_event_view(q,next(iter(q['shots']))));delta=0.;count=0
    for mi,m in enumerate(model['materials']):
        for pi,path in enumerate(m['paths']):
            if path.get('geometry_block') not in seen:continue
            stations=np.linspace(0,1,max(33,len(path.get('points',[]))*2));old=native_before.world(mi,pi,stations);new=native_after.world(mi,pi,stations);delta=max(delta,float(np.max(np.abs(old-new))));count+=len(stations)
    if delta>1e-11:raise ValueError('common witness changed finite physical world curves')
    result={'schema':'rinkfit.print_only_common_domain_witness.v1','purpose':'coordinate decode only; no source data, physical geometry, camera, or newly requested legacy relation changed','advertisements':report,'finite_world_points_compared':count,'max_world_coordinate_change_L':delta,'local_original_domain_equivalence_certified':all(r['strict_original_box_witness_found'] for r in report),'global_search_optimality_claimed':False,'chart_original_box_union_accessibility':'each decoded physical affine has an explicit common original-domain witness; fixed current witness alone is not the entire original union'}
    q['print_only_chart_policy']='original-domain common witness decoded; physical affines invariant; see print_only_domain_witness'
    q['print_only_domain_witness']=result
    return q,result
