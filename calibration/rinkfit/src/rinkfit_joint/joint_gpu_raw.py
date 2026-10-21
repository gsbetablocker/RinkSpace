"""Resident per-shot raw-UV CUDA cache for joint-layout shadow and integration.
Shared physical blocks are uploaded once. Original UV, W, PHR and CPU primal stay unchanged.
"""
from pathlib import Path
import ctypes as C,hashlib,json,time
import numpy as np
from .io_utils import read_json,file_sha256
from .nvrtc_module import NVRTCModule

class GPUMeta(C.Structure):
    _fields_=[('R0',C.c_double*9),('n',C.c_double*3),('b',C.c_double*3),('t1',C.c_double*3),('t2',C.c_double*3),('fref',C.c_double),('Rmax',C.c_double),('wide_boundary',C.c_double),('wide_coefficients',C.c_double*4),('dt',C.c_double),('first',C.c_double),('refw',C.c_double*4),('nc',C.c_int),('cb',C.c_int),('dep',C.c_int),('ref',C.c_int*4)]
PATH=np.dtype([('kind','<i4'),('geom','<i4'),('ad','<i4'),('ri','<i4'),('bs','<i4'),('begin','<i4'),('count','<i4'),('a0','<f8'),('a1','<f8'),('h','<f8')],align=True)
LAYOUT=np.dtype([(n,'<i4')for n in ('global','geom','ad','ref','path','block_begin','block_count','control_begin','control_count','geometry_begin','geometry_count','geometry_offset')])
ROW=np.dtype([(n,'<f8')for n in ('t','s','v','alpha','budget')])
JOB=np.dtype([('row','<i4'),('column','<i4')]);PARTIAL=np.dtype([('u','<f8'),('v','<f8'),('arithmetic','<i4'),('status','<i4')])

class JointRawGPU:
    def __init__(self,manifest_file,*,ptx,source=None,build_output=None):
        self.manifest_path=Path(manifest_file);self.manifest=read_json(manifest_file);self.m=self.manifest;self.model=self._json(self.m['model']);self.layouts=self._json(self.m['layout']);self.metadata=self._json(self.m['shot_metadata']);self.physical=self._json(self.m['physical_shared_blocks_once']);self.a={k:self._array(d)for k,d in self.m['arrays'].items()};self.batches={};self.module=NVRTCModule(ptx=ptx)if ptx else NVRTCModule(source,output=build_output)
        self.layouts={str(i):x for i,x in enumerate(self.layouts)}if isinstance(self.layouts,list)else self.layouts;self.metadata={str(i):x for i,x in enumerate(self.metadata)}if isinstance(self.metadata,list)else self.metadata;self.physical={str(i):x for i,x in enumerate(self.physical)}if isinstance(self.physical,list)else self.physical
        self.nb=max(map(int,self.physical))+1;self.dims=[int(self.physical[str(i)]['ambient_size'])for i in range(self.nb)];self.offsets=np.r_[0,np.cumsum(self.dims)[:-1]].astype('<i4');self.original_value_count=sum(self.dims);self.values=np.concatenate([np.asarray(self.physical[str(i)]['values'],float)for i in range(self.nb)]);self.rig_globals={};self.global_sources={}
        for md in self.metadata.values():
            rid=md['rig_id']
            if rid in self.rig_globals:continue
            self.rig_globals[rid]=len(self.dims);self.global_sources[rid]=md['synthetic_global32_column_sources'];self.offsets=np.r_[self.offsets,len(self.values)].astype('<i4');self.dims.append(32);self.values=np.r_[self.values,np.concatenate([np.asarray(self.physical[str(x['shared_physical_block'])]['values'],float)for x in self.global_sources[rid]])]
        self.dim_array=np.asarray(self.dims,'<i4');self.buffers=dict(offsets=self.module.upload(self.offsets),dims=self.module.upload(self.dim_array),values=self.module.upload(self.values));self.paths=[];self.path_map={};points=[]
        for mi,mat in enumerate(self.model['materials']):
            for pi,pa in enumerate(mat['paths']):
                self.path_map[(mi,pi)]=len(self.paths);pp=pa.get('points',[]);self.paths.append((int(pa['kind']),pa.get('geometry_block',-1),pa.get('advertisement_block',-1),pa.get('radius_index',2),pa.get('boundary_segment',-1),len(points)//2,len(pp),pa.get('start_angle',0.),pa.get('end_angle',2*np.pi),pa.get('fixed_height_fraction',0.)));points.extend(np.asarray(pp,float).reshape(-1).tolist())
        self.path_array=np.array(self.paths,dtype=PATH);self.points=np.asarray(points if points else [0.,0.],'<f8');self.buffers['paths']=self.module.upload(self.path_array);self.buffers['points']=self.module.upload(self.points)
        self._layout_maps={};self._seed_maps={}
        for key,l in self.layouts.items():
            if not l['GPU_raw_math_eligible_after_joint_ABI_adaptation']:continue
            oldids=l['block_ids'];removed={l['scene_slot'],l['rig_slot'],l['optics_slot']};keep=[i for i in range(len(oldids))if i not in removed];remap={old:new+1 for new,old in enumerate(keep)};md=self.metadata[str(l['shot_index'])];bids=[self.rig_globals[md['rig_id']]]+[oldids[i]for i in keep];mapping=[]
            for entry in md['synthetic_global32_column_sources']:
                mapping.extend((entry['shared_physical_block'],j,0)for j in range(entry['size']))
            for slot in keep:mapping.extend((oldids[slot],j,0)for j in range(self.dims[oldids[slot]]))
            mapping.extend((-1,-1,i)for i in (1,2,3,4));columns=len(mapping)
            if columns>160 or len(bids)+4>32:raise ValueError('Joint layout exceeds existing proven GPU row storage')
            if len(l.get('geometry_indices',[]))>16:raise ValueError('Unsupported compact primitive geometry')
            self._seed_maps[int(key)]=np.asarray([(-1,-1,0)]+mapping,'<i4');self._layout_maps[int(key)]=(l,bids,remap,columns)
        li=self.a['layout_index.u32'];times=self.a['frame_time.f64'];eligible=np.asarray([int(x)in self._layout_maps for x in li])
        for sk,md in self.metadata.items():
            ids=np.flatnonzero(eligible&np.asarray([self.layouts[str(int(x))]['shot_index']==int(sk)for x in li]));self._batch(sk,md,ids,times)
        self.update(self.values[:self.original_value_count],self.a['station.f64'],self.a['row.f64'],self.a['alpha.f64'],0.)
    def _json(self,d):
        p=Path(d.get('file',d.get('path')))
        if file_sha256(p)!=d['sha256']:raise ValueError('GPU layout dependency changed: '+str(p))
        return read_json(p)
    def _array(self,d):
        p=Path(d['file'])
        if file_sha256(p)!=d['sha256']:raise ValueError('GPU source-state array changed')
        return np.fromfile(p,dtype=d['dtype'])
    def _batch(self,sk,md,ids,times):
        if not len(ids):return
        layouts=[];bids=[];ci=[];cs=[];co=[];gi=[];counts=[]
        for idx in ids:
            original=int(self.a['layout_index.u32'][idx]);l,local_bids,remap,nc=self._layout_maps[original];map_slot=lambda slot:-1 if slot<0 else remap[slot];layouts.append((0,map_slot(l['geometry_slot']),map_slot(l['advertisement_slot']),map_slot(l['reference_angles_slot']),self.path_map[(l['material'],l['path'])],len(bids),len(local_bids),len(ci),len(l['control_ids']),len(gi),len(l['geometry_indices']),l.get('geometry_offset',0)));bids.extend(local_bids);ci.extend(l['control_ids']);cs.extend(map_slot(q)for q in l['control_slots']);co.extend(l['control_offsets']);gi.extend(l['geometry_indices']);counts.append(nc+1)
        layouts=np.asarray(layouts,dtype=LAYOUT);counts=np.asarray(counts,'<i8');starts=np.r_[0,np.cumsum(counts)];jobs=np.empty(int(starts[-1]),dtype=JOB);jobs['row']=np.repeat(np.arange(len(ids),dtype='<i4'),counts);jobs['column']=np.arange(len(jobs),dtype='<i4')-np.repeat(starts[:-1].astype('<i4'),counts)-1;rows=np.zeros(len(ids),dtype=ROW);rows['t']=times[ids]
        rm=md['rig_camera_meta'];sm=md['shot_camera_meta'];meta=GPUMeta((C.c_double*9)(*rm['R0_seed']),(C.c_double*3)(*rm['n_seed']),(C.c_double*3)(*rm['b_seed']),(C.c_double*3)(*rm['n_tangent1']),(C.c_double*3)(*rm['n_tangent2']),rm['f_ref'],rm['radius_max'],rm.get('brown_wide_extension',{}).get('activation_log_f_native',0.),(C.c_double*4)(*rm.get('brown_wide_extension',{}).get('coefficients',[0.,0.,0.,0.])),sm['dt'],sm['first_control_time'],(C.c_double*4)(*sm['reference_weights']),sm['n_controls'],0,sm['dependent_control'],(C.c_int*4)(*sm['reference_indices']))
        buffers={k:self.module.upload(np.asarray(v,dtype='<i4')if k not in ('layouts','jobs','rows')else v)for k,v in dict(layouts=layouts,jobs=jobs,rows=rows,bids=bids,ci=ci,cs=cs,co=co,gi=gi if gi else [0]).items()};buffers['output']=self.module.alloc(len(jobs)*PARTIAL.itemsize);self.batches[md['shot_id']]=dict(ids=ids,meta=meta,rows=rows,jobs=jobs,starts=starts,buffers=buffers,layouts=layouts)
    def update(self,physical_values,station,row,alpha,budget):
        pv=np.asarray(physical_values,float)
        if pv.shape!=(self.original_value_count,):raise ValueError('Physical shared block layout changed')
        self.values[:self.original_value_count]=pv
        for rid,entries in self.global_sources.items():
            at=int(self.offsets[self.rig_globals[rid]])
            for e in entries:
                bi=e['shared_physical_block'];o=int(self.offsets[bi]);self.values[at+e['offset']:at+e['offset']+e['size']]=pv[o:o+e['size']]
        self.module.upload(self.values,ptr=self.buffers['values'])
        for batch in self.batches.values():
            ids=batch['ids'];rows=batch['rows'];rows['s']=np.asarray(station)[ids];rows['v']=np.asarray(row)[ids];rows['alpha']=np.asarray(alpha)[ids];rows['budget']=budget;self.module.upload(rows,ptr=batch['buffers']['rows'])
    def evaluate(self,sid,*,begin=0,end=None,download=True):
        batch=self.batches[sid];q=batch['buffers'];end=len(batch['jobs'])if end is None else int(end);U=C.c_uint64;I=C.c_int;b=self.buffers
        args=[batch['meta'],U(b['offsets']),U(b['values']),U(b['dims']),U(b['paths']),U(b['points']),U(q['rows']),U(q['bids']),U(q['ci']),U(q['cs']),U(q['co']),U(q['gi']),U(q['layouts']),U(q['jobs']),I(begin),I(end),U(q['output'])]
        started=time.perf_counter();self.module.launch('joint_raw_uv',(end-begin+127)//128,128,args,synchronize=True);seconds=time.perf_counter()-started
        if not download:return dict(kernel_seconds=seconds,job_count=end-begin)
        result=self.module.download(q['output'],(len(batch['jobs']),),dtype=PARTIAL)
        return result,dict(kernel_seconds=seconds,job_count=end-begin)
    def oracle_inputs(self,sid,positions):
        batch=self.batches[sid];jobs=batch['jobs'][positions];r=jobs['row'];source=batch['ids'][r];layout=self.a['layout_index.u32'][source];seed=np.empty((len(jobs),3),'<i4')
        for l in np.unique(layout):
            take=layout==l;seed[take]=self._seed_maps[int(l)][jobs['column'][take]+1]
        return dict(layout_index=layout,frame_time=batch['rows']['t'][r],station=batch['rows']['s'][r],row=batch['rows']['v'][r],seed_block=seed[:,0],seed_component=seed[:,1],seed_aux=seed[:,2])
    def close(self):self.module.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
