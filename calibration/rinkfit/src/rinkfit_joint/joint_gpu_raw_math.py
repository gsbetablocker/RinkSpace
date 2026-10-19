"""Device view of the existing FP64 row math, with joint layout additions only.
The old single-shot CUDA source/ABI is never edited. CPU primal/domain/PHR stay authoritative.
"""
from pathlib import Path
import hashlib,re

def source_from_existing(path):
    original=Path(path).read_bytes();text=original.decode('utf8').replace('\r\n','\n')
    def once(a,b):
        nonlocal text
        if text.count(a)!=1:raise ValueError('GPU joint math source anchor changed: '+a[:90])
        text=text.replace(a,b,1)
    once('if(l.control_ids[k]==j)return pars[l.control_slots[k]][channel];','if(l.control_ids[k]==j)return pars[l.control_slots[k]][l.control_offsets[k]+channel];')
    begin=text.index('template<class T> HD std::array<T,3> trajectory(');end=text.index('template<class T> HD FixedHead<T> head(',begin)
    tr=text[begin:end]
    if tr.count('return out;')!=1:raise ValueError('trajectory return changed')
    text=text[:begin]+tr.replace('return out;','if(l.reference_angles_slot>=0){out[0]+=pars[l.reference_angles_slot][0];out[1]+=pars[l.reference_angles_slot][1];}return out;')+text[end:]
    once('const T* g=pars[l.global_slot];const T* q=l.geometry_slot<0?nullptr:pars[l.geometry_slot];','const T* g=pars[l.global_slot];const T* rawq=l.geometry_slot<0?nullptr:pars[l.geometry_slot];T compactq[16];const T* q=rawq;if(rawq){if(!l.geometry_indices.empty()){if(l.geometry_indices.size()>16)gpu_abort(__LINE__);for(int i=0;i<l.geometry_indices.size();++i)compactq[i]=rawq[l.geometry_indices[i]];q=compactq;}else q=rawq+l.geometry_offset;}')
    once('return board_point(position,g[4]*T(p.fixed_height_fraction),g);','const T height=q?q[0]:T(p.fixed_height_fraction);if(event_scalar(height)<0||event_scalar(height)>1)gpu_abort(__LINE__);return board_point(position,g[4]*height,g);')
    once('struct Layout {int global_slot=0,geometry_slot=-1,advertisement_slot=-1,flatpath=0;double frame_time=0;Span<int>control_ids,control_slots;};','struct Layout {int global_slot=0,geometry_slot=-1,advertisement_slot=-1,reference_angles_slot=-1,flatpath=0,geometry_offset=0;double frame_time=0;Span<int>control_ids,control_slots,control_offsets,geometry_indices;};')
    text=re.sub(r'^#include.*\n','',text,flags=re.M)
    text=text.replace('#define HD __host__ __device__','#define HD __device__')
    begin=text.index('[[noreturn]] HD inline void gpu_abort(');end=text.index('struct Jet',begin)
    text=text[:begin]+'[[noreturn]] HD inline void gpu_abort(int line){atomicCAS(&gpu_first_math_failure,0,line);asm("trap;");__builtin_unreachable();}\n'+text[end:]
    text=text.replace('namespace gpu_exact {','namespace gpu_joint_exact {')
    text=text.replace('std::array','Array').replace('std::size_t','unsigned long long')
    text=text.replace('std::numeric_limits<double>::quiet_NaN()','::nan("")')
    text=text.replace('std::is_arithmetic','Arithmetic')
    for name in ['isfinite','abs','sin','cos','sqrt','exp','log','pow','floor']:
        text=text.replace('std::'+name,'::'+name)
    text=text.replace('std::max','maximum').replace('std::min','minimum')
    at=text.index('namespace gpu_joint_exact {')
    pre='''
namespace gpu_joint_exact {
template<class T,unsigned long long N>struct Array {T data[N];HD T&operator[](unsigned long long i){return data[i];}HD const T&operator[](unsigned long long i)const{return data[i];}HD T*begin(){return data;}HD const T*begin()const{return data;}HD T*end(){return data+N;}HD const T*end()const{return data+N;}HD unsigned long long size()const{return N;}HD T&at(unsigned long long i){return data[i];}HD const T&at(unsigned long long i)const{return data[i];}};
template<class T>struct Arithmetic {static constexpr bool value=false;};template<>struct Arithmetic<double>{static constexpr bool value=true;};template<>struct Arithmetic<float>{static constexpr bool value=true;};
template<class T>HD T maximum(T a,T b){return a<b?b:a;}template<class T>HD T minimum(T a,T b){return b<a?b:a;}
}
'''
    text=text[:at]+pre+text[at:]
    abi='''
struct GPUMeta {double R0[9],n[3],b[3],t1[3],t2[3],fref,Rmax,dt,first,refw[4];int nc,cb,dep,ref[4];};
struct GPUPath {int kind,geom,ad,ri,bs,begin,count;double a0,a1,h;};
struct UVRow {double t,s,v,alpha,budget;};
struct UVLayout {int global,geom,ad,ref,path,block_begin,block_count,control_begin,control_count,geometry_begin,geometry_count,geometry_offset;};
struct UVJob {int row,column;};struct UVPartial {double u,v;int arithmetic,status;};
'''
    kernel='''
namespace ge=gpu_joint_exact;
extern "C" __global__ void joint_raw_uv(GPUMeta meta,const int*offsets,const double*values,const int*dims,const GPUPath*paths,const double*points,const UVRow*rows,const int*block_ids,const int*controls,const int*slots,const int*control_offsets,const int*geometry_indices,const UVLayout*layouts,const UVJob*jobs,int begin,int end,UVPartial*out){
 int index=begin+blockIdx.x*blockDim.x+threadIdx.x;if(index>=end)return;const auto job=jobs[index];const auto row=rows[job.row];const auto l=layouts[job.row];ge::Jet storage[160];const ge::Jet*ptrs[32];int cursor=0;
 for(int k=0;k<l.block_count;++k){int b=block_ids[l.block_begin+k];ptrs[k]=storage+cursor;for(int j=0;j<dims[b];++j){storage[cursor]=ge::Jet(values[offsets[b]+j],cursor==job.column?1.:0.);++cursor;}}
 for(int k=0;k<4;++k){ptrs[l.block_count+k]=storage+cursor;storage[cursor]=ge::Jet(k==0?row.s:k==1?row.v:k==2?row.alpha:row.budget,cursor==job.column?1.:0.);++cursor;}
 const auto&src=paths[l.path];ge::Path path{};path.kind=src.kind;path.radius_index=src.ri;path.boundary_segment=src.bs;path.start_angle=src.a0;path.end_angle=src.a1;path.fixed_height_fraction=src.h;path.points={reinterpret_cast<const ge::Vec2<double>*>(points+2*src.begin),src.count};
 ge::Model model{};for(int i=0;i<9;++i)model.R0_seed[i]=meta.R0[i];for(int i=0;i<3;++i){model.n_seed[i]=meta.n[i];model.b_seed[i]=meta.b[i];model.n_tangent1[i]=meta.t1[i];model.n_tangent2[i]=meta.t2[i];}for(int i=0;i<4;++i){model.reference_indices[i]=meta.ref[i];model.reference_weights[i]=meta.refw[i];}model.f_ref=meta.fref;model.radius_max=meta.Rmax;model.dt=meta.dt;model.first_control_time=meta.first;model.n_controls=meta.nc;model.control_block_begin=meta.cb;model.dependent_control=meta.dep;model.paths=&path;
 ge::Layout local{};local.global_slot=l.global;local.geometry_slot=l.geom;local.advertisement_slot=l.ad;local.reference_angles_slot=l.ref;local.flatpath=0;local.frame_time=row.t;local.control_ids={controls+l.control_begin,l.control_count};local.control_slots={slots+l.control_begin,l.control_count};local.control_offsets={control_offsets+l.control_begin,l.control_count};local.geometry_indices={geometry_indices+l.geometry_begin,l.geometry_count};local.geometry_offset=l.geometry_offset;
 const auto pred=model.predict_for_al(local,ptrs[l.block_count][0],ptrs[l.block_count+1][0],ptrs);UVPartial r{};r.arithmetic=pred.arithmetic_valid;r.status=int(pred.physical_status);if(pred.arithmetic_valid){r.u=job.column<0?pred.uv[0].a:pred.uv[0].v;r.v=job.column<0?pred.uv[1].a:pred.uv[1].v;if(!::isfinite(r.u)||!::isfinite(r.v))r.arithmetic=0;}out[index]=r;
}
'''
    if 'std::' in text:raise ValueError('Unadapted standard library operation')
    return abi+text+'\n'+kernel,dict(original_math_file=str(Path(path).resolve()),original_math_sha256=hashlib.sha256(original).hexdigest(),adaptations=['joint reference_angles trajectory addition','shared-control offsets','finite geometry indices/offset','optional shared board-height path','device array view; scalar math operation order retained'],old_CUDA_ABI_or_source_modified=False,raw_UV_J_only=True,CPU_primal_domain_W_PHR_authoritative=True)
