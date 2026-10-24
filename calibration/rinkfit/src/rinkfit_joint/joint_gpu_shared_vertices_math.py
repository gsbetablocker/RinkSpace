"""Local two-vertex blocks with explicit one-sided station cells, versioned ABI."""
from pathlib import Path
import hashlib
from .joint_gpu_shared_shape_math import source_from_existing as _base_source
KERNEL_ABI='shared_print_vertices_v2'
KERNEL_SYMBOL='joint_raw_uv_shared_vertices_v2'
UV_LAYOUT_BYTES=68
GPU_META_BYTES=304

def source_from_existing(path):
    text, provenance = _base_source(path)
    def once(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('Reviewed local-vertex CUDA anchor changed: '+old[:100])
        text=text.replace(old,new,1)
    once('void joint_raw_uv_shared_shape_v1(', 'void '+KERNEL_SYMBOL+'(')
    once('geometry_offset,width_slot,width_sign;};','geometry_offset,width_slot,width_sign,shape_segment,shape_vertex0,shape_vertex1;};')
    once('flatpath=0,geometry_offset=0;','flatpath=0,geometry_offset=0,shape_segment=-1;')
    once('if(event_scalar(station)<0||event_scalar(station)>1)gpu_abort(__LINE__);', 'if(l.shape_segment<0&&(event_scalar(station)<0||event_scalar(station)>1))gpu_abort(__LINE__);')
    once('const int first=minimum(segments-1,int(::floor(event_scalar(coordinate))));','const int first=l.shape_segment>=0?l.shape_segment:minimum(segments-1,int(::floor(event_scalar(coordinate))));')
    once('if(src.kind==6){\n   if(l.geom<0', 'if(src.kind==6&&l.shape_segment<0){\n   if(l.geom<0')
    once('if(src.kind==6&&k==l.geom){','if(src.kind==6&&l.shape_segment<0&&k==l.geom){')
    once('const auto&src=paths[l.path];int logical=0,selected[4]={0,0,0,0};',r'''const auto&src=paths[l.path];int logical=0,selected[4]={0,0,0,0};
 if(l.shape_segment>=0){
  const int segments=src.count==1?1:(src.a0!=0?src.count:src.count-1);
  if(src.kind!=6||src.count<1||l.shape_segment>=segments||l.shape_vertex0<0||l.shape_vertex1<0||l.shape_vertex0>=l.block_count||l.shape_vertex1>=l.block_count)ge::gpu_abort(__LINE__);
  const double coordinate=row.s*segments-l.shape_segment;
  if(!::isfinite(coordinate)||coordinate<-1e-12||coordinate>1+1e-12){out[index]=UVPartial{};return;}
 }
''')
    anchor=' const auto pred=model.predict_for_al(local,ptrs[l.block_count][0],ptrs[l.block_count+1][0],ptrs);'
    once(anchor,r'''
 ge::Jet local_shape_geometry[4];
 if(l.shape_segment>=0){
  const int a=block_ids[l.block_begin+l.shape_vertex0],b=block_ids[l.block_begin+l.shape_vertex1];
  if(dims[a]!=2||dims[b]!=2)ge::gpu_abort(__LINE__);
  local_shape_geometry[0]=ptrs[l.shape_vertex0][0];local_shape_geometry[1]=ptrs[l.shape_vertex0][1];
  local_shape_geometry[2]=ptrs[l.shape_vertex1][0];local_shape_geometry[3]=ptrs[l.shape_vertex1][1];
  ptrs[l.shape_vertex0]=local_shape_geometry;local.geometry_slot=l.shape_vertex0;local.geometry_offset=0;
  local.geometry_indices={geometry_indices,0};local.shape_segment=l.shape_segment;
 }
'''+anchor)
    provenance.update(kernel_abi=KERNEL_ABI,kernel_symbol=KERNEL_SYMBOL,
        UVLayoutSharedVerticesV2_bytes=UV_LAYOUT_BYTES,
        shape_storage='two real 2D shared blocks; complete one-sided local cell',
        shape_station_semantics='fixed declared segment; global source station retained; left cell at upper knot retains its left derivative; no implicit parameter-list swap',
        adapter_file=str(Path(__file__).resolve()),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return text,provenance
