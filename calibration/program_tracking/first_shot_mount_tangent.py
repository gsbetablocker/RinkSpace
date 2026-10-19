"""Pure-array complete first-order derivative of the existing mount envelope.

No canonical slot, parameter count, scale convention, source or physical state
is hard coded. The caller supplies all chart derivatives in one common column
order; the exact current halfspaces and their lower margins are preserved.
The row contract is the existing extended_mount_halfspaces order: eight x/y
rows, alternating signed z support-corner rows, then six internal-lever rows.
This function does not install a solver callback. Its output includes temporal
columns and must be consumed by a complete coupled direction interface; slicing
it to shared columns is not a complete physical linearization.
"""
import itertools
import numpy as np
from scipy.sparse import csr_matrix, vstack


def full_mount_tangent(*, value, value_jacobian, matrix, absolute_lower,
                       log_scale_gradient, support, tilt, tilt_jacobian,
                       driver, driver_jacobian, interval_limits,
                       vector_a, vector_b, vector_a_jacobian,
                       vector_b_jacobian, square_interval_bounds):
    value=np.asarray(value);M=np.asarray(matrix);b=np.asarray(absolute_lower)
    tilt=np.asarray(tilt);driver=np.asarray(driver);limits=np.asarray(interval_limits)
    VJ=csr_matrix(value_jacobian);TJ=csr_matrix(tilt_jacobian);DJ=csr_matrix(driver_jacobian)
    AJ=csr_matrix(vector_a_jacobian);BJ=csr_matrix(vector_b_jacobian)
    a=np.asarray(vector_a);bv=np.asarray(vector_b);ls=csr_matrix(np.asarray(log_scale_gradient)[None,:])
    n=VJ.shape[1]
    if not (VJ.shape==(6,n) and TJ.shape==(len(tilt),n) and DJ.shape==TJ.shape
            and AJ.shape==BJ.shape==(3,n) and limits.shape==(len(tilt)-1,2)
            and M.shape==(len(b),6) and ls.shape==(1,n)):
        raise ValueError('Complete mount values, support and common-column chart derivatives required')

    def square_interval(low,high):
        return square_interval_bounds(low,high)

    points={}
    for interval in support['intervals']:
        low,high=interval['driver'];sq=square_interval(low,high);rho=np.asarray(interval['v_bounds'])[2]
        for iz,ir,ip in itertools.product(range(2),repeat=3):
            points.setdefault(((low,high)[iz],rho[ir],sq[ip]),[]).append((interval['left'],'corner',iz,ir,ip))
        for ir in range(2):points.setdefault((0.,rho[ir],0.),[]).append((interval['left'],'zero',ir))
    descriptions=[]
    for row in range(len(M)):
        if row<8:
            axis=row//4;extreme=(row%4)//2;target=support['v_min'if extreme==0 else'v_max'][axis]
            origins=[(item['left'],'xy',axis,extreme)for item in support['intervals']
                     if np.asarray(item['v_bounds'])[axis,extreme]==target]
        elif row>=len(M)-6:origins=[(-1,'internal',(row-(len(M)-6))%3)]
        else:origins=points[tuple((M[row]if row%2==0 else-M[row])[3:])]
        descriptions.extend((row,origin)for origin in origins)

    nonsmooth=[]
    def rho_jacobians(i,axis,extreme):
        left,right=limits[i];bet=tilt[i:i+2];delta=bet[1]-bet[0]
        times=[left,right];low,high=sorted((bet[0]+delta*left,bet[0]+delta*right))
        phase=float(np.arctan2(bv[axis],a[axis]))
        if delta!=0.:
            for k in range(int(np.ceil((low-phase)/np.pi)),int(np.floor((high-phase)/np.pi))+1):
                times.append((phase+k*np.pi-bet[0])/delta)
        betas=np.asarray([bet[0]+delta*t for t in times]);vals=a[axis]*np.cos(betas)+bv[axis]*np.sin(betas)
        target=np.min(vals)if extreme==0 else np.max(vals);tied=np.flatnonzero(vals==target)
        result=[]
        for index in tied:
            t=times[index];beta=betas[index]
            grad=(-a[axis]*np.sin(beta)+bv[axis]*np.cos(beta))*((1-t)*TJ.getrow(i)+t*TJ.getrow(i+1))
            grad+=np.cos(beta)*AJ.getrow(axis)+np.sin(beta)*BJ.getrow(axis)
            result.append(grad)
        if len(result)>1:nonsmooth.append(dict(kind='rho_exact_extremum_tie',interval=i,axis=axis,extreme=extreme,branches=len(result)))
        return result

    def driver_endpoints(i):
        left,right=limits[i];z=np.asarray([(1-t)*driver[i]+t*driver[i+1]for t in(left,right)])
        jac=[(1-t)*DJ.getrow(i)+t*DJ.getrow(i+1)for t in(left,right)]
        return z,jac

    rows=[];rowids=[];origins_out=[]
    for row,origin in descriptions:
        kind=origin[1]
        sign=1. if ((row<8 and row%2==0)or(8<=row<len(M)-6 and row%2==0)
                    or(len(M)-6<=row<len(M)-3))else-1.
        base=csr_matrix(M[row:row+1])@VJ+b[row]*ls
        if kind=='internal':variants=[base]
        elif kind=='xy':variants=[base+sign*value[4]*rj for rj in rho_jacobians(origin[0],origin[2],origin[3])]
        else:
            i=origin[0];rho_jacs=rho_jacobians(i,2,origin[3]if kind=='corner'else origin[2])
            if kind=='zero':variants=[base+sign*value[4]*rj for rj in rho_jacs]
            else:
                z,zj=driver_endpoints(i);iz,ir,ip=origin[2:];zvalue=(np.min(z)if iz==0 else np.max(z))
                driver_jacs=[zj[k]for k in np.flatnonzero(z==zvalue)]
                if ip==0 and np.min(z)<=0<=np.max(z):square_jacs=[csr_matrix((1,n))]
                else:
                    zz=z*z;target=np.min(zz)if ip==0 else np.max(zz)
                    square_jacs=[2*z[k]*zj[k]for k in np.flatnonzero(zz==target)]
                variants=[base+sign*(value[4]*rj+value[3]*dj+value[5]*sj)
                          for rj,dj,sj in itertools.product(rho_jacs,driver_jacs,square_jacs)]
                if len(driver_jacs)>1 or len(square_jacs)>1:
                    nonsmooth.append(dict(kind='driver_or_square_exact_extremum_tie',interval=i,driver_branches=len(driver_jacs),square_branches=len(square_jacs)))
        for result in variants:
            rows.append(result);rowids.append(row);origins_out.append(origin)
    output=vstack(rows,format='csr');output.eliminate_zeros();rowids=np.asarray(rowids,dtype=int)
    lower=(b-M@value)[rowids]
    if set(rowids)!=set(range(len(M))):
        raise ValueError('Every original physical mount halfspace must retain a tangent')
    return dict(matrix=output,lower=lower,original_mount_row_ids=rowids,
                interval_origins=origins_out,
                report=dict(original_halfspaces=len(M),differentiated_alias_rows=len(rowids),
                    complete_original_current_margins_unchanged=True,
                    exact_tie_branches_retained=nonsmooth,
                    formula='d(M(q)v(q)-b(q))=M dv + absolute_lower*dlog(scale) + continuous-pose support derivative',
                    source_observations_weights_bounds_and_parameter_count_unchanged=True))
