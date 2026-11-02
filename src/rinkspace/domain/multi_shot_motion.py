"""Independent shot trajectories sharing one native physical camera.

Packing is only a storage translation. No motion variable, exposure halo or
physical trajectory certificate connects the two sides of an edit.
"""
from dataclasses import dataclass
import numpy as np
from .motion_chart import UniformMotionChart, cubic_samples


def refine_uniform_controls(controls,first,dt,target_dt):
    controls=np.asarray(controls,float)
    ratio=dt/target_dt;m=int(round(ratio))
    if m<1 or not np.isclose(m,ratio,rtol=0,atol=1e-10):
        raise ValueError('Storage-grid refinement requires an integer interval ratio')
    if m==1:return controls.copy(),float(first)
    mask=np.ones(1)
    for _ in range(4):mask=np.convolve(mask,np.ones(m))
    mask/=m**3;radius=2*(m-1)
    count=(len(controls)-3)*m+3;offset=m-1
    k=np.arange(count)[:,None]+offset-np.arange(len(controls))[None,:]*m+radius
    valid=(k>=0)&(k<len(mask));matrix=np.zeros(k.shape);matrix[valid]=mask[k[valid]]
    return matrix@controls,float(first+dt-target_dt)


def continue_exposure_halos(controls,first,dt,frame_interval):
    """Continue endpoint cubics in controls or a tensor of control directions.

    Continue each end's interior cubic into its exposure halo. Every value
    between the first and last frame centres is unchanged, including endpoint
    derivatives through order two. Rolling-shutter rows outside that interval
    can change and require evaluation. A state is changed only by an explicit
    call. Applying the same linear map to update directions preserves this
    continuation throughout fitting, without changing the interior trajectory
    space. No neighbouring shot supplies the continuation.
    """
    result=np.asarray(controls,float).copy();span=np.asarray(frame_interval,float)
    if result.ndim not in (2,3) or result.shape[1]!=3 or span.shape!=(2,) or not np.isfinite(result).all() or not np.isfinite([first,dt,*span]).all() or dt<=0:
        raise ValueError('Invalid uniform trajectory/interval')
    x=(span-first)/dt;k=np.rint(x).astype(int);a,b=map(int,k)
    if np.max(abs(x-k))>1e-8 or not 1<=a<=b<=len(result)-2 or a+2>=len(result) or b<2:
        raise ValueError('Frame endpoints must lie on the existing native knot grid')
    def extend(values,offsets):
        j=np.asarray(offsets).reshape((-1,)+(1,)*(values.ndim-1));d1=np.diff(values,axis=0);d2=np.diff(d1,axis=0);d3=np.diff(d2,axis=0)
        return values[0]+j*d1[0]+j*(j-1)/2*d2[0]+j*(j-1)*(j-2)/6*d3[0]
    result[:a-1]=extend(result[a-1:a+3],np.arange(a-1)-(a-1))
    result[b+2:]=extend(result[b-2:b+2],np.arange(b+2,len(result))-(b-2))
    return result



def exposure_offset_seconds(record,pts):
    """Explicit fixed delivery-to-exposure steps; original integer PTS stay intact."""
    pts=np.asarray(pts);result=np.zeros(pts.shape,float);previous=None
    for step in record.get('exposure_pts_offsets',[]):
        first=step['first_pts'];offset=step['offset_pts']
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in (first,offset)) or offset<=0 or (previous is not None and first<=previous):
            raise ValueError('Exposure steps require increasing original PTS and positive integer offsets')
        if 'first_pts' in record and not record['first_pts']<=first<=record['last_pts']:raise ValueError('Exposure step outside shot')
        result[pts>=first]+=int(offset)*int(record['time_base_num'])/int(record['time_base_den']);previous=first
    return result


@dataclass
class ShotMotionTimeline:
    shots: list
    dt: float
    first: float=0.

    @property
    def count(self):return self.shots[-1]['control_stop']

    def row_times(self,pts):
        pts=np.asarray(pts)
        if pts.ndim!=1 or pts.dtype.kind not in "iu":
            raise ValueError('Original PTS must be a one-dimensional integer array')
        out=np.empty(len(pts));covered=np.zeros(len(pts),bool)
        for shot in self.shots:
            sl=slice(shot['row_begin'],shot['row_stop']);p=pts[sl]
            if np.any(covered[sl]) or not len(p):raise ValueError('Invalid shot row partition')
            # Inspect integer extrema without unsigned/signed promotion.
            low=int(p.min());high=int(p.max())
            if low<shot['first_pts'] or high>=shot['stop_pts_exclusive']:
                raise ValueError('Original PTS outside its shot')
            for name in ('reference_pts','time_base_num','time_base_den'):
                if isinstance(shot[name],(bool,np.bool_)) or not isinstance(shot[name],(int,np.integer)):
                    raise ValueError('Native clock metadata must use integer values')
            reference=int(shot['reference_pts'])
            numerator=int(shot['time_base_num']);denominator=int(shot['time_base_den'])
            if numerator<=0 or denominator<=0:
                raise ValueError('Native time base must be positive')
            # Retain the vectorized path for normal native clocks. Only cases
            # that cannot safely use int64 arithmetic need Python's wide ints.
            lim=np.iinfo(np.int64);delta_low=low-reference;delta_high=high-reference
            safe=(lim.min<=low<=high<=lim.max and lim.min<=reference<=lim.max
                  and numerator<=lim.max and denominator<=lim.max
                  and lim.min<=delta_low<=delta_high<=lim.max
                  and lim.min<=delta_low*numerator<=delta_high*numerator<=lim.max)
            if safe:
                delta=p.astype(np.int64,copy=False)-np.int64(reference)
                relative=(delta*numerator)/denominator
            else:
                relative=np.fromiter(((int(value)-reference)*numerator/denominator for value in p),
                                     dtype=np.float64,count=len(p))
            if shot.get('exposure_pts_offsets'):relative=relative+exposure_offset_seconds(shot,p)
            out[sl]=relative+self.offset(shot)
            covered[sl]=True
        if not covered.all():raise ValueError('Unassigned original observation')
        return out

    def offset(self,shot):
        return self.first+shot['control_begin']*self.dt-shot['first_control_time']

    def exposure_time(self,shot,pts):
        source=np.asarray(pts);scalar=source.ndim==0;values=np.atleast_1d(source)
        if values.dtype.kind not in 'iu' or values.ndim!=1:raise ValueError('Original exposure PTS must be integers')
        relative=np.fromiter(((int(p)-int(shot['reference_pts']))*int(shot['time_base_num'])/int(shot['time_base_den']) for p in values),dtype=float,count=len(values))
        if shot.get('exposure_pts_offsets'):relative+=exposure_offset_seconds(shot,values)
        result=relative+self.offset(shot)
        return float(result[0]) if scalar else result

    def local_exposure_time(self,shot,pts):
        return self.exposure_time(shot,pts)-self.offset(shot)

    def source_seconds_at(self,shot,native_times):
        """Display labels only: interpolate recorded frame centres, not capture facts."""
        if not shot.get('exposure_pts_offsets'):
            result=np.asarray(native_times,float)+shot['reference_pts']*shot['time_base_num']/shot['time_base_den']-self.offset(shot)
            return float(result) if result.ndim==0 else result
        base=[int(shot['first_pts']),int(shot['last_pts'])]
        for step in shot.get('exposure_pts_offsets',[]):
            # Working maps declare their registered source exposure step.
            stride=int(step.get('source_step_pts',1));base.extend([int(step['first_pts'])-stride,int(step['first_pts'])])
        pts=np.array(sorted(set(p for p in base if shot['first_pts']<=p<=shot['last_pts'])),np.int64)
        native=self.exposure_time(shot,pts);seconds=pts*shot['time_base_num']/shot['time_base_den'];x=np.asarray(native_times,float);result=np.interp(x,native,seconds)
        result=np.where(x<native[0],seconds[0]+x-native[0],result);result=np.where(x>native[-1],seconds[-1]+x-native[-1],result)
        return float(result) if result.ndim==0 else result

    def frame_windows(self):
        return np.array([self.exposure_time(s,np.asarray([s['first_pts'],s['last_pts']])) for s in self.shots])

    def max_readout(self):
        margins=[]
        for s,w in zip(self.shots,self.frame_windows()):
            begin=self.first+(s['control_begin']+1)*self.dt
            end=self.first+(s['control_stop']-2)*self.dt
            margins.extend([w[0]-begin,end-w[1]])
        # Leave a floating-point guard at each packed block's open endpoint.
        # One nextafter of the duration can be lost when added to a later
        # block's much larger storage-clock origin.
        rounding=64*np.finfo(float).eps*max(1.,abs(self.first),self.count*self.dt)
        value=2*(min(margins)-rounding)
        if value<=0:raise ValueError('A shot has no independent readout halo')
        return float(np.nextafter(value,0.))

    def _local_refinement_directions(self,shot,block,labels,*,reference_angle_gauge,closed):
        """Add only independent fine cubic directions with local source support.

        Existing coefficients and directions are retained exactly. QR chooses
        extra coordinates of the enlarged spline space; it never weights the
        pixel objective or changes a starting trajectory.
        """
        from scipy.linalg import qr
        count=len(block);first=shot['first_control_time']
        interval=self.local_exposure_time(shot,np.asarray([shot['first_pts'],shot['last_pts']]))
        frames=int(round((interval[1]-interval[0])/self.dt))
        sample_times=np.linspace(interval[0],interval[1],max(1,2*frames+1))
        ids,weights=cubic_samples(count,first,self.dt,sample_times)
        def samples(d):
            return np.einsum('ij,ijck->ick',weights,d[ids]).reshape(-1,d.shape[2])
        extra=[];extra_labels=[];seen=set();fine_charts={}
        offset=max((label[0] for label in labels),default=-1)+1
        for window in shot['local_motion_refinement']:
            a,b,stride=(window[k] for k in ('first_pts','last_pts','target_stride'))
            if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in (a,b,stride)):
                raise ValueError('Local motion windows require integer original PTS and stride')
            if not shot['first_pts']<=a<b<=shot['last_pts'] or stride<1 or stride>=shot['knot_stride'] or shot['knot_stride']%stride:
                raise ValueError('Local motion refinement must be a finer nested shot interval')
            channels=window.get('channels',['pan','tilt','log_f'])
            names=('pan','tilt','log_f')
            if not isinstance(channels,list) or not channels or any(not isinstance(c,str) or c not in names for c in channels) or len(set(channels))!=len(channels):
                raise ValueError('Local motion refinement channels must name distinct existing motion axes')
            allowed={names.index(c) for c in channels}
            bounds=self.local_exposure_time(shot,np.asarray([a,b]))
            if stride not in fine_charts:
                chart=UniformMotionChart.create(count,first,self.dt,stride,shot['motion_reference_time'])
                fine,flabel=chart.directions(reference_angle_gauge=reference_angle_gauge)
                if closed:fine=continue_exposure_halos(fine,first,self.dt,interval)
                fine_charts[stride]=(fine,flabel,offset)
                offset+=chart.refinement.shape[1]
            fine,flabel,label_offset=fine_charts[stride]
            for k,label in enumerate(flabel):
                key=(stride,*label)
                if label[1] not in allowed or key in seen:continue
                nonzero=np.flatnonzero(np.any(fine[:,:,k]!=0.,axis=1))
                if not len(nonzero):continue
                support=[max(interval[0],first+(nonzero[0]-2)*self.dt),
                         min(interval[1],first+(nonzero[-1]+2)*self.dt)]
                if support[0]<bounds[0]-1e-10 or support[1]>bounds[1]+1e-10:continue
                seen.add(key);extra.append(fine[:,:,k]);extra_labels.append((label_offset+label[0],label[1]))
        if not extra:raise ValueError('Local motion window has no finer observable cubic support')
        fine=np.stack(extra,axis=2);A=samples(block);B=samples(fine)
        Q,R,_=qr(A,mode='economic',pivoting=True)
        diagonal=np.abs(np.diag(R));tol=128*np.finfo(float).eps*max(A.shape)*max(1.,float(diagonal.max(initial=0.)))
        rank=int(np.count_nonzero(diagonal>tol));Q=Q[:,:rank]
        residual=B-Q@(Q.T@B)
        _,R,pivots=qr(residual,mode='economic',pivoting=True)
        diagonal=np.abs(np.diag(R));tol=128*np.finfo(float).eps*max(B.shape)*max(1.,float(np.linalg.norm(B,ord=2)))
        selected=sorted(map(int,pivots[:np.count_nonzero(diagonal>tol)]))
        if not selected:raise ValueError('Local motion refinement adds no independent trajectory directions')
        return np.concatenate([block,fine[:,:,selected]],axis=2),labels+[extra_labels[i] for i in selected]

    def directions(self,*,continue_endpoint_cubics=False,sparse=False):
        """Native update columns, optionally closing each shot's exposure tails.

        The closure preserves every direction on the whole frame-centre
        interval. Pure outside-tail columns become exactly zero and are
        omitted; no observable interior motion column is thresholded away.
        The supplied starting controls must already use the same closure.
        """
        blocks=[];labels=[];gauged_cameras=set()
        for i,s in enumerate(self.shots):
            count=s['control_stop']-s['control_begin']
            chart=UniformMotionChart.create(count,s['first_control_time'],self.dt,s['knot_stride'],s['motion_reference_time'])
            camera=s.get('camera_index',0)
            block,local=chart.directions(reference_angle_gauge=(camera not in gauged_cameras))
            if continue_endpoint_cubics:
                interval=self.local_exposure_time(s,np.asarray([s['first_pts'],s['last_pts']]))
                block=continue_exposure_halos(block,s['first_control_time'],self.dt,interval)
                keep=np.any(block!=0.,axis=(0,1))
                block=block[:,:,keep];local=[v for v,k in zip(local,keep) if k]
            if s.get('local_motion_refinement'):
                block,local=self._local_refinement_directions(s,block,local,
                    reference_angle_gauge=(camera not in gauged_cameras),closed=continue_endpoint_cubics)
            gauged_cameras.add(camera)
            if sparse:
                from scipy.sparse import csr_matrix
                blocks.append(csr_matrix(block.reshape(count*3,-1)))
            else:blocks.append(block)
            labels.extend([(s['id'],*v) for v in local])
        if sparse:
            from scipy.sparse import block_diag
            return block_diag(blocks,format='csr'),labels
        out=np.zeros((self.count,3,len(labels)));j=0
        for s,b in zip(self.shots,blocks):
            out[s['control_begin']:s['control_stop'],:,j:j+b.shape[2]]=b;j+=b.shape[2]
        return out,labels

    def assert_support_separation(self,pts,max_readout):
        times=self.row_times(pts)
        for s in self.shots:
            t=np.unique(times[s['row_begin']:s['row_stop']])
            for sign in (-1,1):
                ids,_=cubic_samples(self.count,self.first,self.dt,t+sign*max_readout/2)
                if ids.min()<s['control_begin'] or ids.max()>=s['control_stop']:
                    raise ValueError('Exposure support crosses a shot boundary')
