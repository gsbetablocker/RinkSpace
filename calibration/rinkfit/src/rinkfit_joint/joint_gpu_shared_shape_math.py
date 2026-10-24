"""Versioned two-vertex shared contour evaluator; archived GPU ABIs stay unchanged."""
from pathlib import Path
import hashlib
from .joint_gpu_line_width_math import source_from_existing as _base_source
KERNEL_ABI = 'shared_print_shape_v1'
KERNEL_SYMBOL = 'joint_raw_uv_shared_shape_v1'
UV_LAYOUT_BYTES = 56
GPU_META_BYTES = 304

def source_from_existing(path):
    text, provenance = _base_source(path)
    def once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Reviewed shared-shape CUDA anchor changed: ' + old[:100])
        text = text.replace(old, new, 1)
    once('void joint_raw_uv_width_v1(', 'void ' + KERNEL_SYMBOL + '(')
    once('if(rawq){if(!l.geometry_indices.empty())', 'if(rawq&&p.kind!=6){if(!l.geometry_indices.empty())')
    once('    if(p.points.empty())gpu_abort(__LINE__);', r'''
    if(p.kind==6){
      if(!q||p.points.size()<1)gpu_abort(__LINE__);
      const int n=int(p.points.size()),segments=p.start_angle!=0?n:n-1;
      T fraction=T(0);
      if(n>1){const T coordinate=station*T(segments);const int first=minimum(segments-1,int(::floor(event_scalar(coordinate))));fraction=coordinate-T(first);}
      const T u=q[0]*(T(1)-fraction)+q[2]*fraction,v=q[1]*(T(1)-fraction)+q[3]*fraction;
      if(l.advertisement_slot<0)return board_point(u,v,g);
      const T* ad=pars[l.advertisement_slot]+p.advertisement_offset;
      return board_point(ad[0]+ad[1]*u,ad[2]+(ad[3]-ad[2])*v,g);
    }
    if(p.points.empty())gpu_abort(__LINE__);''')
    old = ''' for(int k=0;k<l.block_count;++k){int b=block_ids[l.block_begin+k];ptrs[k]=storage+cursor;for(int j=0;j<dims[b];++j){storage[cursor]=ge::Jet(values[offsets[b]+j],cursor==job.column?1.:0.);++cursor;}}
 for(int k=0;k<4;++k){ptrs[l.block_count+k]=storage+cursor;storage[cursor]=ge::Jet(k==0?row.s:k==1?row.v:k==2?row.alpha:row.budget,cursor==job.column?1.:0.);++cursor;}
 const auto&src=paths[l.path];'''
    new = r''' const auto&src=paths[l.path];int logical=0,selected[4]={0,0,0,0};
 if(src.kind==6){
   if(l.geom<0||src.count<1||row.s<0||row.s>1)ge::gpu_abort(__LINE__);
   const int segments=src.a0!=0?src.count:src.count-1;
   const int first=src.count==1?0:ge::minimum(segments-1,int(::floor(row.s*segments)));
   const int next=src.count==1?0:(first+1)%src.count;
   const int ids[4]={2*first,2*first+1,2*next,2*next+1};
   for(int j=0;j<4;++j)selected[j]=l.geometry_count?geometry_indices[l.geometry_begin+ids[j]]:l.geometry_offset+ids[j];
 }
 for(int k=0;k<l.block_count;++k){
   const int b=block_ids[l.block_begin+k];ptrs[k]=storage+cursor;
   if(src.kind==6&&k==l.geom){
     if(cursor+4>160)ge::gpu_abort(__LINE__);
     const int seed=job.column<=-2&&job.column>=-5?selected[-job.column-2]:-1;
     for(int j=0;j<4;++j){if(selected[j]<0||selected[j]>=dims[b])ge::gpu_abort(__LINE__);storage[cursor++]=ge::Jet(values[offsets[b]+selected[j]],selected[j]==seed?1.:0.);}
     logical+=dims[b];
   }else{
     if(cursor+dims[b]>160)ge::gpu_abort(__LINE__);
     for(int j=0;j<dims[b];++j){storage[cursor++]=ge::Jet(values[offsets[b]+j],logical==job.column?1.:0.);++logical;}
   }
 }
 if(cursor+4>160)ge::gpu_abort(__LINE__);
 for(int k=0;k<4;++k){ptrs[l.block_count+k]=storage+cursor;storage[cursor++]=ge::Jet(k==0?row.s:k==1?row.v:k==2?row.alpha:row.budget,logical==job.column?1.:0.);++logical;}
'''
    once(old, new)
    once('path.advertisement_offset=(src.kind==2&&src.bs>=0)?src.bs:0;', 'path.advertisement_offset=((src.kind==2||src.kind==6)&&src.bs>=0)?src.bs:0;')
    # -1 is primal; -2..-5 seed the four selected vertex components. Duplicate
    # indices seed every alias, so the returned derivative is the full chain.
    if text.count('job.column<0?') != 2:
        raise ValueError('Reviewed output column sentinel changed')
    text = text.replace('job.column<0?', 'job.column==-1?')
    provenance.update(kernel_abi=KERNEL_ABI,kernel_symbol=KERNEL_SYMBOL,
        GPUPath_kind6_count='vertex count',GPUPath_kind6_a0='closed flag',
        GPUPath_kind6_bs='advertisement offset',shape_jobs=[-2,-3,-4,-5],
        shape_storage='four selected values; complete shared block stays resident once',
        shape_station_semantics='native open/closed uniform vertex segment interpolation',
        adapter_file=str(Path(__file__).resolve()),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return text, provenance
