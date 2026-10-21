"""Explicit 56-byte kind0-width raw-UV GPU consumer; old 48-byte consumer stays unchanged."""
from pathlib import Path
import ctypes as C,time
import numpy as np
from .joint_gpu_raw import JointRawGPU,GPUMeta,ROW,JOB,PARTIAL
from .joint_gpu_line_width_math import KERNEL_ABI,KERNEL_SYMBOL
from .io_utils import read_json,file_sha256
LAYOUT_WIDTH_V1=np.dtype([(n,'<i4')for n in ('global','geom','ad','ref','path','block_begin','block_count','control_begin','control_count','geometry_begin','geometry_count','geometry_offset','width_slot','width_sign')])
assert LAYOUT_WIDTH_V1.itemsize==56 and C.sizeof(GPUMeta)==304

class JointRawLineWidthGPU(JointRawGPU):
    def __init__(self,manifest_file,*,ptx,contract_file):
        manifest=read_json(manifest_file);contract=read_json(contract_file)
        if manifest.get('kernel_abi')!=KERNEL_ABI or contract.get('kernel_abi')!=KERNEL_ABI or contract.get('kernel_symbol')!=KERNEL_SYMBOL or contract.get('UVLayoutWidthV1_bytes')!=56 or contract.get('GPUMeta_bytes')!=304 or contract.get('ptx_sha256')!=file_sha256(ptx):raise ValueError('Explicit line-width GPU ABI/PTX contract mismatch')
        layouts=read_json(manifest['layout']['file']);model=read_json(manifest['model']['file'])
        for l in (layouts.values()if isinstance(layouts,dict)else layouts):
            if not l['GPU_raw_math_eligible_after_joint_ABI_adaptation']:continue
            ws=l.get('width_slot',-1)
            if ws>=0:
                gs=l['geometry_slot'];ids=l['block_ids']
                if gs<0:raise ValueError('Line-width layout has no carrier geometry')
                count=len(l.get('geometry_indices',[])) or len(model['blocks'][ids[gs]]['values'])-l.get('geometry_offset',0)
                if l['primitive_kind']!=0 or l.get('width_sign')not in (-1,1)or gs<0 or ws>=len(ids)or len(model['blocks'][ids[ws]]['values'])!=1 or count!=4:raise ValueError('Unsupported or malformed line-width layout')
        super().__init__(manifest_file,ptx=ptx)
        # Resolve now, so an old 48-byte PTX cannot be launched with this ABI.
        try:self.module.get_function(KERNEL_SYMBOL)
        except Exception:
            self.close();raise
    def _batch(self,sk,md,ids,times):
        if not len(ids):return
        layouts=[];bids=[];ci=[];cs=[];co=[];gi=[];counts=[]
        for idx in ids:
            original=int(self.a['layout_index.u32'][idx]);l,local_bids,remap,nc=self._layout_maps[original];map_slot=lambda slot:-1 if slot<0 else remap[slot];layouts.append((0,map_slot(l['geometry_slot']),map_slot(l['advertisement_slot']),map_slot(l['reference_angles_slot']),self.path_map[(l['material'],l['path'])],len(bids),len(local_bids),len(ci),len(l['control_ids']),len(gi),len(l['geometry_indices']),l.get('geometry_offset',0),map_slot(l.get('width_slot',-1)),l.get('width_sign',0)));bids.extend(local_bids);ci.extend(l['control_ids']);cs.extend(map_slot(q)for q in l['control_slots']);co.extend(l['control_offsets']);gi.extend(l['geometry_indices']);counts.append(nc+1)
        layouts=np.asarray(layouts,dtype=LAYOUT_WIDTH_V1);counts=np.asarray(counts,'<i8');starts=np.r_[0,np.cumsum(counts)];jobs=np.empty(int(starts[-1]),dtype=JOB);jobs['row']=np.repeat(np.arange(len(ids),dtype='<i4'),counts);jobs['column']=np.arange(len(jobs),dtype='<i4')-np.repeat(starts[:-1].astype('<i4'),counts)-1;rows=np.zeros(len(ids),dtype=ROW);rows['t']=times[ids]
        rm=md['rig_camera_meta'];sm=md['shot_camera_meta'];meta=GPUMeta((C.c_double*9)(*rm['R0_seed']),(C.c_double*3)(*rm['n_seed']),(C.c_double*3)(*rm['b_seed']),(C.c_double*3)(*rm['n_tangent1']),(C.c_double*3)(*rm['n_tangent2']),rm['f_ref'],rm['radius_max'],rm.get('brown_wide_extension',{}).get('activation_log_f_native',0.),(C.c_double*4)(*rm.get('brown_wide_extension',{}).get('coefficients',[0.,0.,0.,0.])),sm['dt'],sm['first_control_time'],(C.c_double*4)(*sm['reference_weights']),sm['n_controls'],0,sm['dependent_control'],(C.c_int*4)(*sm['reference_indices']))
        buffers={k:self.module.upload(np.asarray(v,dtype='<i4')if k not in ('layouts','jobs','rows')else v)for k,v in dict(layouts=layouts,jobs=jobs,rows=rows,bids=bids,ci=ci,cs=cs,co=co,gi=gi if gi else [0]).items()};buffers['output']=self.module.alloc(len(jobs)*PARTIAL.itemsize);self.batches[md['shot_id']]=dict(ids=ids,meta=meta,rows=rows,jobs=jobs,starts=starts,buffers=buffers,layouts=layouts)
    def evaluate(self,sid,*,begin=0,end=None,download=True):
        batch=self.batches[sid];q=batch['buffers'];end=len(batch['jobs'])if end is None else int(end);U=C.c_uint64;I=C.c_int;b=self.buffers
        args=[batch['meta'],U(b['offsets']),U(b['values']),U(b['dims']),U(b['paths']),U(b['points']),U(q['rows']),U(q['bids']),U(q['ci']),U(q['cs']),U(q['co']),U(q['gi']),U(q['layouts']),U(q['jobs']),I(begin),I(end),U(q['output'])]
        started=time.perf_counter();self.module.launch(KERNEL_SYMBOL,(end-begin+127)//128,128,args,synchronize=True);seconds=time.perf_counter()-started
        if not download:return dict(kernel_seconds=seconds,job_count=end-begin)
        result=self.module.download(q['output'],(len(batch['jobs']),),dtype=PARTIAL)
        return result,dict(kernel_seconds=seconds,job_count=end-begin)
