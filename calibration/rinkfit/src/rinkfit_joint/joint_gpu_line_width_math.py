"""Versioned kind0 line-width CUDA adapter; existing fcff math is unchanged."""
from pathlib import Path
import hashlib
from .joint_gpu_raw_math import source_from_existing as _base_source

KERNEL_ABI='ice_line_width_v1'
KERNEL_SYMBOL='joint_raw_uv_width_v1'
UV_LAYOUT_BYTES=56
GPU_META_BYTES=304

def source_from_existing(path):
    text,provenance=_base_source(path)
    old='struct UVLayout {int global,geom,ad,ref,path,block_begin,block_count,control_begin,control_count,geometry_begin,geometry_count,geometry_offset;};'
    new=old.replace('geometry_offset;','geometry_offset,width_slot,width_sign;')
    if text.count(old)!=1:raise ValueError('Reviewed UVLayout48 anchor changed')
    text=text.replace(old,new).replace('void joint_raw_uv(','void '+KERNEL_SYMBOL+'(')
    anchor=' const auto pred=model.predict_for_al(local,ptrs[l.block_count][0],ptrs[l.block_count+1][0],ptrs);'
    if text.count(anchor)!=1:raise ValueError('Reviewed raw projection anchor changed')
    shift=r"""
 ge::Jet width_geometry[4];
 if(l.width_slot>=0){
  if(src.kind!=0||l.geom<0||(l.width_sign!=-1&&l.width_sign!=1))ge::gpu_abort(__LINE__);
  const auto*raw=ptrs[l.geom];
  if(l.geometry_count!=0&&l.geometry_count!=4)ge::gpu_abort(__LINE__);
  for(int k=0;k<4;++k)width_geometry[k]=raw[l.geometry_count?geometry_indices[l.geometry_begin+k]:l.geometry_offset+k];
  const auto dx=width_geometry[2]-width_geometry[0],dy=width_geometry[3]-width_geometry[1];
  const auto length=ge::sqrt(dx*dx+dy*dy);
  const auto shift=ge::Jet(l.width_sign)*ptrs[l.width_slot][0];
  for(int k=0;k<4;k+=2){width_geometry[k]-=shift*dy/length;width_geometry[k+1]+=shift*dx/length;}
  ptrs[l.geom]=width_geometry;local.geometry_indices={geometry_indices,0};local.geometry_offset=0;
 }
"""
    text=text.replace(anchor,shift+anchor)
    provenance.update(kernel_abi=KERNEL_ABI,kernel_symbol=KERNEL_SYMBOL,UVLayoutWidthV1_bytes=UV_LAYOUT_BYTES,GPUMeta_bytes=GPU_META_BYTES,width_primitive_kinds=[0],width_signs=[-1,1],native_math='JointEventModel.legacy_layout kind0 normal shift with normalized tangent, Jet width and carrier chain',adapter_file=str(Path(__file__).resolve()),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),old_source_or_PTX_modified=False)
    return text,provenance
