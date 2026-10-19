"""Fail-closed exclusion of every other finite profile-query root.

At a root the native row is exactly Q_y(alpha). Substitution reduces the
three original equations to two: camera(s, row_time(Q_y(alpha)))-Q(alpha).
The entire original [0,1] x [alpha_min,alpha_max] rectangle is covered by a
branch-and-bound tree. Leaves must exclude roots by outward interval values,
exclude the complete physical domain, or place every possible root inside a
separately certified three-variable local uniqueness box. Sampling and finite
differences never certify exclusion. Resource/precision limits are unresolved.

All NativeCamera fields are frozen binary64 constants, as in the local proof.
The solver-parameter-to-camera floating reconstruction is outside this proof.
The frozen replay's whole-readout RS certificate remains a separate gate.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
from pathlib import Path
import time

import numpy as np

from .native_query_certificate import (Interval, _AD, _dot, _cross,
    _matrix_multiply, _rotate, _polynomial, _log_positive_point,
    certify_profile_roots)

VERSION = 'native_profile_full_finite_interval_exclusion_v1'


def pack_proof_tree(nodes):
    """Lossless numeric proof graph for NPZ, including unbounded intervals.

    IEEE infinity bounds are retained as bounds, rather than converted to
    JSON nulls or mistaken for finite exclusions. Missing Krawczyk entries
    are NaN and carry an explicit non-Krawczyk node result code.
    """
    codes=['pending','split','contracted_krawczyk','excluded_interval_range',
           'excluded_physical_domain','excluded_krawczyk',
           'all_roots_in_certified_local_box','unresolved_box_budget',
           'unresolved_binary64_subdivision']
    n=len(nodes)
    arrays={'profile_row':np.asarray([v['row'] for v in nodes],dtype='<u4'),
        'parent_node':np.asarray([v['parent'] for v in nodes],dtype='<i8'),
        'result_code':np.asarray([codes.index(v['result']) for v in nodes],dtype='u1'),
        'cubic_cells':np.asarray([v.get('cubic_cells',0) for v in nodes],dtype='<u4'),
        'split_dimension':np.asarray([v.get('split_dimension',-1) for v in nodes],dtype='i1')}
    for key in ['lower','upper','F_lower','F_upper','krawczyk_lower','krawczyk_upper']:
        arrays[key]=np.asarray([v.get(key,[np.nan,np.nan]) for v in nodes],dtype='<f8').reshape(n,2)
    for key in ['depth_upper','disk_delta_lower']:
        arrays[key]=np.asarray([v.get(key,np.nan) for v in nodes],dtype='<f8')
    return arrays,{'result_codes':codes,'root_parent':-1,'profile_row_is_local_input_index':True,
        'infinite_interval_bounds_preserved':True,
        'verification_scope':'Replay every node with the hashed producer and fixed inputs; split children cover the parent rectangle, Krawczyk children contain all parent zeros, and no unresolved leaf is accepted.'}


def _query(curves, alpha):
    denominator = _polynomial(np.asarray(curves['denominator']).T, alpha)
    return (_polynomial(np.asarray(curves['numerator_u']).T, alpha)/denominator,
            _polynomial(np.asarray(curves['numerator_v']).T, alpha)/denominator,
            denominator)


def _cell_projection(camera, world, x, cell):
    """One cubic cell, including interval derivatives of the original clock."""
    u = x-cell
    weights = [(1-u)*(1-u)*(1-u)/6,
        (3*u*u*u-6*u*u+4)/6, (-3*u*u*u+3*u*u+3*u+1)/6, u*u*u/6]
    angles = [sum((weights[k]*camera.controls[cell+k-1,j] for k in range(4)),
                  _AD(0.)) for j in range(3)]
    alpha,beta,logf = angles
    supported = (alpha.v.absolute_upper()<=4)&(beta.v.absolute_upper()<=4)&(logf.v.absolute_upper()<=64)
    R = _matrix_multiply(_matrix_multiply(_rotate(camera.b,beta),camera.R0.tolist()),
                         _rotate(camera.n,alpha))
    wb = [_dot([R[k][j] for k in range(3)],camera.b) for j in range(3)]
    h = _cross(camera.n,wb); hn = _dot(h,h).sqrt(); h = [v/hn for v in h]
    z = logf-_AD(_log_positive_point(float(camera.optics['f_ref'])))
    ell = camera.head.get('pupil_linear',0.)*z+camera.head.get('pupil_quadratic',0.)*z*z
    D = [_AD(camera.d[0]),_AD(camera.d[1]),camera.d[2]+ell]
    C = [camera.P[j]+_dot([R[k][j] for k in range(3)],D)+camera.a*h[j] for j in range(3)]
    Y = [_dot(R[j],[world[k]-C[k] for k in range(3)]) for j in range(3)]
    xx,yy = Y[0]/Y[2],Y[1]/Y[2]; r2 = xx.square()+yy.square()
    k1 = _polynomial(camera.optics['k1'],z); k2 = _polynomial(camera.optics['k2'],z)
    radial = 1+k1*r2+k2*r2*r2; p1,p2 = camera.optics['p1'],camera.optics['p2']
    xd = xx*radial+2*p1*xx*yy+p2*(r2+2*xx.square())
    yd = yy*radial+p1*(r2+2*yy.square())+2*p2*xx*yy
    eta = float(camera.optics.get('eta',0.)); pp = camera.optics['principal_point']
    pl = camera.optics.get('principal_linear',[0.,0.])
    U = pp[0]+z*pl[0]+(logf+eta/2).exp()*xd
    V = pp[1]+z*pl[1]+(logf-eta/2).exp()*yd
    # Cross-multiplied disk exclusion is valid even if the depth interval
    # crosses zero. No division by an uncertain depth certifies this test.
    radius_squared = _AD(Interval(float(camera.optics['radius_max']))**2)
    disk_delta = (Y[0].square()+Y[1].square()
                  -radius_squared*Y[2].square()).v
    physical_excluded = (Y[2].v.hi<=0)|(disk_delta.lo>0)
    smooth = supported&(hn.v.lo>1e-10)&((Y[2].v.lo>0)|(Y[2].v.hi<0))
    return U,V,smooth,physical_excluded,Y[2].v.hi,disk_delta.lo,supported


def reduced_system(scene,camera,material,branch,frame_time,curves,lower,upper):
    """Outward F/J enclosure; every crossed cubic cell is evaluated separately.

    A cell restricts only the time value enclosure, retaining the complete
    analytic derivative enclosure. Taking the hull of the per-cell bounds is
    valid across cubic knots because the same uniform cubic spline is C2.
    """
    lower=np.asarray(lower,float); upper=np.asarray(upper,float); n=len(lower)
    s=_AD.variable(lower[:,0],upper[:,0],0); a=_AD.variable(lower[:,1],upper[:,1],1)
    qu,qv,den=_query(curves,a)
    denominator_good=(den.v.lo>0)|(den.v.hi<0)
    time_value=frame_time+camera.tau*(qv/1079-.5)
    x=(time_value-camera.first)/camera.dt
    finite_time=np.isfinite(x.v.lo)&np.isfinite(x.v.hi)
    # Dummy cell indices only avoid invalid integer conversion; their rows
    # remain unresolved, never accepted as an evaluation of the true clock.
    first=np.floor(np.where(finite_time,x.v.lo,0.)).astype(np.int64)
    last=np.floor(np.where(finite_time,x.v.hi,0.)).astype(np.int64)
    last=np.where(x.v.hi==last,last-1,last)
    last=np.maximum(last,first)
    span=last-first+1
    allocated=finite_time&(first>=1)&(last<=len(camera.controls)-3)&(span<=8)
    q=np.empty((n,4)); primitive=np.ones(n,bool)
    for j,(mi,bi) in enumerate(zip(material,branch)):
        p=scene.path(mi,bi)
        if p['kind'] not in (0,'ice_segment'):primitive[j]=False;q[j]=0.
        else:q[j]=scene.values[int(p['geometry_block'])]
    world=[q[:,0]*(1-s)+q[:,2]*s,q[:,1]*(1-s)+q[:,3]*s,_AD(np.zeros(n))]
    flo=np.full((n,2),np.inf); fhi=np.full((n,2),-np.inf)
    jlo=np.full((n,2,2),np.inf); jhi=np.full((n,2,2),-np.inf)
    smooth=allocated&denominator_good&primitive
    all_excluded=allocated&denominator_good&primitive
    all_physical_excluded=allocated&primitive
    depth_upper=np.full(n,-np.inf); disk_delta_lower=np.full(n,np.inf)
    any_active=np.zeros(n,bool); cells_evaluated=np.zeros(n,np.int64)
    with np.errstate(all='ignore'):
        for offset in range(int(min(8,np.max(span[allocated],initial=0)))):
            actual=first+offset; active=allocated&(actual<=last)
            if not np.any(active):continue
            cell=np.clip(actual,1,len(camera.controls)-3)
            lx=np.maximum(x.v.lo,actual); hx=np.minimum(x.v.hi,actual+1.)
            # Inactive rows carry a harmless allocated interval so that a
            # vectorized array call does not poison active arithmetic.
            xc=_AD(Interval(np.where(active,lx,cell),np.where(active,hx,cell)),x.d)
            U,V,good,outside,depth,disk,trig_good=_cell_projection(camera,world,xc,cell)
            F=[U-qu,V-qv]
            fl=np.stack([f.v.lo+np.zeros(n) for f in F],axis=1)
            fh=np.stack([f.v.hi+np.zeros(n) for f in F],axis=1)
            jl=np.stack([np.stack([f.d[k].lo+np.zeros(n) for k in range(2)],axis=1) for f in F],axis=1)
            jh=np.stack([np.stack([f.d[k].hi+np.zeros(n) for k in range(2)],axis=1) for f in F],axis=1)
            arithmetic=np.isfinite(fl).all(axis=1)&np.isfinite(fh).all(axis=1)&np.isfinite(jl).all(axis=(1,2))&np.isfinite(jh).all(axis=(1,2))
            well_defined=good&denominator_good&arithmetic
            no_zero=np.any((fl>0)|(fh<0),axis=1)&well_defined
            # Domain exclusion needs a certified elementary-function chart
            # even when division by depth is undefined in this wide box.
            outside &= trig_good&np.isfinite(depth)&np.isfinite(disk)
            all_excluded &= ~active|outside|no_zero
            all_physical_excluded &= ~active|outside
            smooth &= ~active|well_defined
            flo[active]=np.minimum(flo[active],fl[active]);fhi[active]=np.maximum(fhi[active],fh[active])
            jlo[active]=np.minimum(jlo[active],jl[active]);jhi[active]=np.maximum(jhi[active],jh[active])
            depth_upper[active]=np.maximum(depth_upper[active],depth[active])
            disk_delta_lower[active]=np.minimum(disk_delta_lower[active],disk[active])
            any_active |= active;cells_evaluated[active]+=1
    image_excluded=denominator_good&((qv.v.hi<0)|(qv.v.lo>1079))
    all_excluded=(all_excluded&any_active)|image_excluded
    all_physical_excluded=(all_physical_excluded&any_active)|image_excluded
    smooth &= any_active
    reason=np.where(~primitive,'unsupported_material',np.where(~denominator_good,'query_denominator_interval_contains_zero',
        np.where(~allocated,'allocated_clock_or_wide_query_time_unresolved',np.where(~smooth,'wide_depth_head_or_elementary_interval',''))))
    return {'F_lower':flo,'F_upper':fhi,'J_lower':jlo,'J_upper':jhi,'smooth':smooth,
        'excluded':all_excluded,'physical_excluded':all_physical_excluded,'reason':reason,
        'query_row_lower':qv.v.lo,'query_row_upper':qv.v.hi,'cells_evaluated':cells_evaluated,
        'depth_upper':depth_upper,'disk_delta_lower':disk_delta_lower}


def _krawczyk(lower,upper,ranges,point):
    n=len(lower);center=lower/2+upper/2;C=np.zeros((n,2,2));good=ranges['smooth']&point['smooth']
    for j in np.flatnonzero(good):
        try:C[j]=np.linalg.inv(ranges['J_lower'][j]/2+ranges['J_upper'][j]/2)
        except np.linalg.LinAlgError:good[j]=False
    B=[[Interval(float(i==j)) for j in range(2)] for i in range(2)]
    base=[Interval(center[:,i]) for i in range(2)]
    for i in range(2):
        for k in range(2):base[i]=base[i]-Interval(C[:,i,k])*Interval(point['F_lower'][:,k],point['F_upper'][:,k])
        for j in range(2):
            for k in range(2):B[i][j]=B[i][j]-Interval(C[:,i,k])*Interval(ranges['J_lower'][:,k,j],ranges['J_upper'][:,k,j])
    K=[]
    for i in range(2):
        value=base[i]
        for j in range(2):value=value+B[i][j]*(Interval(lower[:,j],upper[:,j])-center[:,j])
        K.append(value)
    lo=np.stack([k.lo for k in K],axis=1);hi=np.stack([k.hi for k in K],axis=1)
    good &= np.isfinite(lo).all(axis=1)&np.isfinite(hi).all(axis=1)
    return lo,hi,good


def certify_global_profiles(scene,camera,material,branch,station,frame_time,curves,
                            alpha,root_rows,*,numerical_valid=None,
                            max_boxes_per_profile=16384,batch_size=1024):
    """Prove all physical finite roots lie in the same certified local box.

    Each tree node is retained, including Krawczyk contractions and splits.
    A limit returns unresolved, with the untouched remaining domain recorded.
    This does not modify any numerical root, source bound or model parameter.
    """
    started=time.perf_counter();material=np.asarray(material);branch=np.asarray(branch)
    frame_time=np.asarray(frame_time,float);n=len(material)
    local=certify_profile_roots(scene,camera,material,branch,station,frame_time,curves,
        alpha,root_rows,numerical_valid=numerical_valid)
    bounds=np.asarray(curves['bounds']);initial_lo=np.c_[np.zeros(n),bounds[:,0]]
    initial_hi=np.c_[np.ones(n),bounds[:,1]];scale=initial_hi-initial_lo
    eligible=local['certified'].copy();counts=np.zeros(n,int);unresolved=np.zeros(n,int)
    covered=np.zeros(n,int);queue=deque();nodes=[];reasons=[Counter()for _ in range(n)]
    def add(row,lo,hi,parent=-1):
        node={'row':int(row),'parent':int(parent),'lower':np.asarray(lo,float).tolist(),
              'upper':np.asarray(hi,float).tolist(),'result':'pending'}
        k=len(nodes);nodes.append(node);queue.append(k)
    for i in np.flatnonzero(eligible):add(i,initial_lo[i],initial_hi[i])
    while queue:
        ids=[]
        while queue and len(ids)<batch_size:
            k=queue.popleft();i=nodes[k]['row']
            if counts[i]>=max_boxes_per_profile:
                nodes[k]['result']='unresolved_box_budget';unresolved[i]+=1;reasons[i]['box_budget']+=1
            else:ids.append(k);counts[i]+=1
        if not ids:continue
        ii=np.asarray([nodes[k]['row']for k in ids]);lo=np.asarray([nodes[k]['lower']for k in ids]);hi=np.asarray([nodes[k]['upper']for k in ids])
        sub={k:np.asarray(curves[k])[ii]for k in ['numerator_u','numerator_v','denominator','bounds']}
        ranges=reduced_system(scene,camera,material[ii],branch[ii],frame_time[ii],sub,lo,hi)
        center=lo/2+hi/2;point=reduced_system(scene,camera,material[ii],branch[ii],frame_time[ii],sub,center,center)
        kl,kh,kgood=_krawczyk(lo,hi,ranges,point)
        with np.errstate(all='ignore'):
            av=_AD(Interval(kl[:,1],kh[:,1]));_,qvk,_=_query(sub,av)
        for j,k in enumerate(ids):
            node=nodes[k];i=int(ii[j]);node.update(cubic_cells=int(ranges['cells_evaluated'][j]),
                F_lower=ranges['F_lower'][j].tolist(),F_upper=ranges['F_upper'][j].tolist(),
                depth_upper=float(ranges['depth_upper'][j]),disk_delta_lower=float(ranges['disk_delta_lower'][j]))
            if ranges['excluded'][j]:
                node['result']='excluded_physical_domain'if ranges['physical_excluded'][j]else'excluded_interval_range'
                reasons[i][node['result']]+=1;continue
            if kgood[j]:
                node.update(krawczyk_lower=kl[j].tolist(),krawczyk_upper=kh[j].tolist())
                if np.any(kh[j]<lo[j])or np.any(kl[j]>hi[j]):
                    node['result']='excluded_krawczyk';reasons[i][node['result']]+=1;continue
                ll,uu=local['box_lower'][i],local['box_upper'][i]
                if (np.all(kl[j]>=ll[:2])and np.all(kh[j]<=uu[:2])
                        and qvk.v.finite()[j]and qvk.v.lo[j]>=ll[2]and qvk.v.hi[j]<=uu[2]):
                    node['result']='all_roots_in_certified_local_box';covered[i]+=1;reasons[i][node['result']]+=1;continue
                newlo=np.maximum(lo[j],kl[j]);newhi=np.minimum(hi[j],kh[j]);width=hi[j]-lo[j]
                if np.any((newhi-newlo)<.75*width):
                    node['result']='contracted_krawczyk';add(i,newlo,newhi,k);continue
            dimension=int(np.argmax((hi[j]-lo[j])/scale[i]));middle=float(center[j,dimension])
            if not lo[j,dimension]<middle<hi[j,dimension]:
                node['result']='unresolved_binary64_subdivision';unresolved[i]+=1;reasons[i][node['result']]+=1;continue
            left=hi[j].copy();left[dimension]=middle;right=lo[j].copy();right[dimension]=middle
            node['result']='split';node['split_dimension']=dimension
            add(i,lo[j],left,k);add(i,right,hi[j],k)
    passed=eligible&(unresolved==0)&(covered>0)
    status=np.where(~eligible,'no_certified_local_root',np.where(passed,'certified_unique_on_original_finite_physical_support','unresolved_other_root_domains'))
    return {'version':VERSION,'global_unique_certified':passed,'status':status,
        'processed_boxes':counts,'unresolved_boxes':unresolved,'covered_local_root_leaves':covered,
        'original_finite_lower':initial_lo,'original_finite_upper':initial_hi,
        'local_certificate':local,'proof_tree':nodes,'reason_counts':[dict(c)for c in reasons],
        'original_station_and_alpha_domains_unchanged':True,'query_row_elimination':'v=Q_y(original_alpha)',
        'crossed_cubic_cells':'separate exact cubic expressions with interval derivative hulls; no polynomial is extended across another cell',
        'sampled_or_finite_difference_derivatives_used':False,
        'scope':'All physical profile roots on the original finite station x original alpha domain lie in one separately certified three-variable unique-root box. Whole-readout RS uniqueness and reconstruction-to-solver correspondence remain separate.',
        'fixed_camera_constants_scope':local['fixed_camera_constants_scope'],
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interval_core_sha256':hashlib.sha256(Path(__file__).with_name('native_query_certificate.py').read_bytes()).hexdigest(),
        'max_boxes_per_profile':max_boxes_per_profile,'batch_size':batch_size,
        'elapsed_seconds':time.perf_counter()-started}
