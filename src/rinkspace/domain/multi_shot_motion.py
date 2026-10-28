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


@dataclass
class ShotMotionTimeline:
    shots: list
    dt: float
    first: float=0.

    @property
    def count(self):return self.shots[-1]['control_stop']

    def row_times(self,pts):
        pts=np.asarray(pts);out=np.empty(len(pts));covered=np.zeros(len(pts),bool)
        for shot in self.shots:
            sl=slice(shot['row_begin'],shot['row_stop']);p=pts[sl]
            if np.any(covered[sl]) or not len(p):raise ValueError('Invalid shot row partition')
            if np.any(p<shot['first_pts']) or np.any(p>=shot['stop_pts_exclusive']):
                raise ValueError('Original PTS outside its shot')
            out[sl]=(p-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']+self.offset(shot)
            covered[sl]=True
        if not covered.all():raise ValueError('Unassigned original observation')
        return out

    def offset(self,shot):
        return self.first+shot['control_begin']*self.dt-shot['first_control_time']

    def frame_windows(self):
        return np.array([[(s[k]-s['reference_pts'])*s['time_base_num']/s['time_base_den']+self.offset(s)
                          for k in ('first_pts','last_pts')] for s in self.shots])

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

    def directions(self,*,continue_endpoint_cubics=False):
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
                interval=[(s[k]-s['reference_pts'])*s['time_base_num']/s['time_base_den'] for k in ('first_pts','last_pts')]
                block=continue_exposure_halos(block,s['first_control_time'],self.dt,interval)
                keep=np.any(block!=0.,axis=(0,1))
                block=block[:,:,keep];local=[v for v,k in zip(local,keep) if k]
            gauged_cameras.add(camera)
            blocks.append(block);labels.extend([(s['id'],*v) for v in local])
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
