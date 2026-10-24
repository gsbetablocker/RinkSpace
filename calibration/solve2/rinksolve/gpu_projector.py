"""Resident batch implementation of the sole new operator's project_at ABI.

Consumes ProjectionOperator camera_definition/controls directly. It does not
invent geometry or independently choose rolling-shutter roots/certificates.
"""
import ctypes as C
from pathlib import Path
import numpy as np
from .compute import CudaDevice
from .projector import so3_left_jacobian

CAMERA_FIELDS=(('R0',9),('pan_axis_world',3),('tilt_axis_camera',3),('pivot_world',3),
 ('lever_camera',3),('axis_spacing',1),('pupil_polynomial',2),('f_ref',1),('radius_max',1),
 ('eta',1),('principal_point',2),('principal_zoom_slope',2),('k1_coefficients',2),
 ('k2_coefficients',2),('tangential',2))

def pack_camera(operator):
    definition=operator.camera_definition;parts=[]
    for key,count in CAMERA_FIELDS:
        value=np.asarray(definition[key],dtype=np.float64).reshape(-1)
        if len(value)!=count or not np.isfinite(value).all():raise ValueError('camera ABI field '+key)
        parts.extend(value)
    parts.extend((operator.shot.control_dt_seconds,operator.shot.first_control_time_seconds,
                  definition['signed_readout_seconds'],operator.rig.image_width,operator.rig.image_height))
    parts.extend(so3_left_jacobian(operator.model.state.get(operator.rig.key('reference_rotation_vector'))).reshape(-1))
    result=np.array(parts,dtype=np.float64)
    if result.shape!=(51,):raise ValueError('camera ABI length')
    return result

class CudaProjectionBatch:
    def __init__(self,operator,xyz,time,*,device=None,generation=1):
        self.device=device or CudaDevice();self.owns_device=device is None
        self.operator=operator;self.generation=generation
        xyz=np.asarray(xyz);time=np.asarray(time)
        if xyz.dtype!=np.dtype('float64') or time.dtype!=np.dtype('float64') or xyz.ndim!=2 or xyz.shape[1]!=3 or time.shape!=(len(xyz),):
            raise ValueError('project_at CUDA ABI requires explicit float64 Nx3/float64 N arrays')
        self.count=len(xyz);self.control_count=len(operator.controls)
        self.xyz=self.device.upload(xyz,generation=generation,role='working')
        self.time=self.device.upload(time,generation=generation,role='working')
        self.controls=self.device.upload(np.ascontiguousarray(operator.controls),generation=generation,role='parameter')
        self.camera=self.device.upload(pack_camera(operator),generation=generation,role='parameter')
        self.output=self.device.allocate((self.count,69),np.float64,generation=generation)
        self.uv_output=self.device.allocate((self.count,2),np.float64,generation=generation)
        self.status=self.device.allocate((self.count,),np.uint8,generation=generation)
        try:
            kernels,self.compilation=self.device.compile(Path(__file__).with_name('project_at.cu'),('project_at',))
        except Exception:
            self.close()
            raise
        self.kernel=kernels['project_at'];self.kernel_milliseconds=None
    def set_state(self,operator):
        if len(operator.controls)!=self.control_count or operator.shot.id!=self.operator.shot.id:
            raise ValueError('changed shot/control layout requires a new resident batch')
        self.device.update(self.controls,np.ascontiguousarray(operator.controls),generation=self.generation)
        self.device.update(self.camera,pack_camera(operator),generation=self.generation)
        self.operator=operator
    def set_rows(self,*,xyz=None,time=None):
        if xyz is not None:self.device.update(self.xyz,np.asarray(xyz),generation=self.generation)
        if time is not None:self.device.update(self.time,np.asarray(time),generation=self.generation)
    def launch(self,full=True):
        self.device.launch(self.kernel,self.count,(self.xyz,self.time,self.controls,self.camera,
            C.c_int(self.count),C.c_int(self.control_count),C.c_int(int(full)),self.output if full else self.uv_output,self.status))
    def evaluate(self,*,measure=False):
        if measure:self.kernel_milliseconds=self.device.timed(self.launch)
        else:self.launch()
        value=self.device.download(self.output);status=self.device.download(self.status)
        valid=status==0
        result={'uv':value[:,:2],'depth':value[:,2],'ray_radius':value[:,3],'valid':valid,'status':status,
                'jacobian_world':value[:,4:10].reshape(-1,2,3),'jacobian_motion':value[:,10:16].reshape(-1,2,3),
                'cubic_weights':value[:,16:20],'motion_rates':value[:,20:23],
                'jacobian_pivot':value[:,23:29].reshape(-1,2,3),
                'jacobian_lever':value[:,29:35].reshape(-1,2,3),
                'jacobian_optics_at_fixed_time':value[:,35:59].reshape(-1,2,12),
                'jacobian_reference_rotation':value[:,59:65].reshape(-1,2,3),
                'control_indices':np.where(np.isfinite(value[:,65:69]),value[:,65:69],-1).astype(np.int64)}
        result['jacobian_time']=np.einsum('nij,nj->ni',result['jacobian_motion'],result['motion_rates'])
        return result
    def evaluate_uv(self):
        self.launch(full=False)
        uv=self.device.download(self.uv_output);status=self.device.download(self.status)
        return {"uv":uv,"status":status,"valid":status==0}
    def close(self):
        for b in (self.xyz,self.time,self.controls,self.camera,self.output,self.uv_output,self.status):b.close()
        if self.owns_device:self.device.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
