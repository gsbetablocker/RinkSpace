"""Experimental same-row derivative tiles; per-job UV ABI is unchanged.

The scalar implementation is retained for mixed-row tiles and exceptional
nonfinite derivative lanes. A tile never borrows a row from another job/chunk.
No production ABI/default is selected by this generator.
"""
from pathlib import Path
import hashlib

JET=r'''
struct Jet {double a,v[WIDTH];unsigned*failure=nullptr;
HD Jet(double x=0):a(x){for(int k=0;k<WIDTH;++k)v[k]=0;}
HD Jet(double x,const int*columns,int coordinate,unsigned*f):a(x),failure(f){for(int k=0;k<WIDTH;++k)v[k]=columns[k]==coordinate?1.:0.;}
HD Jet& operator+=(Jet b){if(!failure)failure=b.failure;a+=b.a;for(int k=0;k<WIDTH;++k)v[k]+=b.v[k];return *this;}
HD Jet& operator-=(Jet b){if(!failure)failure=b.failure;a-=b.a;for(int k=0;k<WIDTH;++k)v[k]-=b.v[k];return *this;}
HD Jet& operator*=(Jet b){if(!failure)failure=b.failure;for(int k=0;k<WIDTH;++k)v[k]=v[k]*b.a+a*b.v[k];a*=b.a;return *this;}
HD Jet& operator/=(Jet b){if(!failure)failure=b.failure;const double inverse=1./b.a,ratio=a*inverse;for(int k=0;k<WIDTH;++k)v[k]=(v[k]-ratio*b.v[k])*inverse;a=ratio;return *this;}};
HD inline Jet operator+(Jet a,Jet b){return a+=b;}HD inline Jet operator-(Jet a,Jet b){return a-=b;}
HD inline Jet operator*(Jet a,Jet b){return a*=b;}HD inline Jet operator/(Jet a,Jet b){return a/=b;}
HD inline Jet operator-(Jet a){a.a=-a.a;for(int k=0;k<WIDTH;++k)a.v[k]=-a.v[k];return a;}
HD inline bool operator<(Jet a,Jet b){return a.a<b.a;}HD inline bool operator>(Jet a,Jet b){return a.a>b.a;}
HD inline bool operator<=(Jet a,Jet b){return a.a<=b.a;}HD inline bool operator>=(Jet a,Jet b){return a.a>=b.a;}
HD inline Jet unary(Jet x,double value,double scale){Jet y(value);y.failure=x.failure;for(int k=0;k<WIDTH;++k)y.v[k]=x.v[k]*scale;return y;}
HD inline Jet sin(Jet x){return unary(x,::sin(x.a),::cos(x.a));}HD inline Jet cos(Jet x){return unary(x,::cos(x.a),-::sin(x.a));}
HD inline Jet sqrt(Jet x){double y=::sqrt(x.a);return unary(x,y,.5/y);}HD inline Jet exp(Jet x){double y=::exp(x.a);return unary(x,y,y);}
HD inline Jet pow(Jet x,Jet p){double y=::pow(x.a,p.a),dx=p.a*::pow(x.a,p.a-1.),dp=y*::log(x.a);Jet z(y);z.failure=x.failure?x.failure:p.failure;for(int k=0;k<WIDTH;++k)z.v[k]=dx*x.v[k]+dp*p.v[k];return z;}
HD inline Jet abs(Jet x){return x.a<0?-x:x;}
// A derivative failure belongs to its original job. Remember it and replay that
// lane through the retained scalar implementation; never invalidate other jobs.
HD inline bool isfinite(Jet x){if(x.failure)for(int k=0;k<WIDTH;++k)if(!::isfinite(x.v[k]))*x.failure|=1u<<k;return ::isfinite(x.a);}
'''

def source_from_existing(path,width=2):
    if width not in (2,4):raise ValueError('direction width must be 2 or 4')
    path=Path(path);text=path.read_text('utf8')
    symbol='joint_raw_uv_shared_vertices_tile'+str(width)+'_v1'
    abi='shared_print_vertices_directions'+str(width)+'_v1'
    marker='namespace ge=gpu_joint_exact;'
    before,entry=text.split(marker,1)
    head,body=entry.split('{',1)
    if 'void joint_raw_uv_shared_vertices_v2(' not in head:raise ValueError('requires the current vertices-v2 mathematical kernel')
    init='\n int index=begin+blockIdx.x*blockDim.x+threadIdx.x;if(index>=end)return;'
    if not body.startswith(init):raise ValueError('scalar job entry changed')
    scalar_head=head.replace('extern "C" __global__ void joint_raw_uv_shared_vertices_v2','__device__ UVPartial joint_direction_scalar_v1').replace('int begin,int end,UVPartial*out)','int index)')
    scalar_body=body.replace(init,'\n',1).replace('out[index]=UVPartial{};return;','return UVPartial{};').replace('out[index]=r;','return r;')
    scalar=marker+'\n'+scalar_head+'{'+scalar_body
    # Reuse the exact current mathematical templates with a small vector Jet.
    start=before.index('namespace gpu_joint_exact {')
    math=before[start:].replace('gpu_joint_exact','gpu_joint_tile')
    a=math.index('struct Jet {');b=math.index('template<class T>struct Span',a)
    math=math[:a]+JET.replace('WIDTH',str(width))+'\n'+math[b:]
    tiled=body.replace('ge::','gt::')
    launch='\n int index=begin+(blockIdx.x*blockDim.x+threadIdx.x)*'+str(width)+';if(index>=end)return;const int count=gt::minimum('+str(width)+',end-index);bool same=true;for(int k=1;k<count;++k)same&=jobs[index+k].row==jobs[index].row;if(!same){for(int k=0;k<count;++k)out[index+k]=joint_direction_scalar_v1(meta,offsets,values,dims,paths,points,rows,block_ids,controls,slots,control_offsets,geometry_indices,layouts,jobs,index+k);return;}int columns['+str(width)+'];for(int k=0;k<'+str(width)+';++k)columns[k]=k<count?jobs[index+k].column:-999;unsigned failed_lanes=0;'
    tiled=tiled.replace(init,launch,1)
    old='const int seed=job.column<=-2&&job.column>=-5?selected[-job.column-2]:-1;'
    new='int seeds['+str(width)+'];for(int lane=0;lane<'+str(width)+';++lane)seeds[lane]=columns[lane]<=-2&&columns[lane]>=-5?selected[-columns[lane]-2]:-1;'
    if tiled.count(old)!=1:raise ValueError('whole-shape seed mapping changed')
    tiled=tiled.replace(old,new)
    tiled=tiled.replace('gt::Jet(values[offsets[b]+selected[j]],selected[j]==seed?1.:0.)','gt::Jet(values[offsets[b]+selected[j]],seeds,selected[j],&failed_lanes)')
    tiled=tiled.replace('gt::Jet(values[offsets[b]+j],logical==job.column?1.:0.)','gt::Jet(values[offsets[b]+j],columns,logical,&failed_lanes)')
    tiled=tiled.replace('logical==job.column?1.:0.)','columns,logical,&failed_lanes)')
    tiled=tiled.replace('out[index]=UVPartial{};return;','for(int lane=0;lane<count;++lane)out[index+lane]=UVPartial{};return;')
    tail='UVPartial r{};r.arithmetic=pred.arithmetic_valid;r.status=int(pred.physical_status);if(pred.arithmetic_valid){r.u=job.column==-1?pred.uv[0].a:pred.uv[0].v;r.v=job.column==-1?pred.uv[1].a:pred.uv[1].v;if(!::isfinite(r.u)||!::isfinite(r.v))r.arithmetic=0;}out[index]=r;'
    replacement='for(int lane=0;lane<count;++lane){if(failed_lanes&(1u<<lane)){out[index+lane]=joint_direction_scalar_v1(meta,offsets,values,dims,paths,points,rows,block_ids,controls,slots,control_offsets,geometry_indices,layouts,jobs,index+lane);continue;}UVPartial r{};r.arithmetic=pred.arithmetic_valid;r.status=int(pred.physical_status);if(pred.arithmetic_valid){r.u=columns[lane]==-1?pred.uv[0].a:pred.uv[0].v[lane];r.v=columns[lane]==-1?pred.uv[1].a:pred.uv[1].v[lane];if(!::isfinite(r.u)||!::isfinite(r.v))r.arithmetic=0;}out[index+lane]=r;}'
    if tiled.count(tail)!=1:raise ValueError('scalar output mapping changed')
    tiled=tiled.replace(tail,replacement)
    if 'logical==job.column' in tiled or 'selected[j]==seed' in tiled:raise ValueError('unconverted scalar seed')
    source=before+'\n'+scalar+'\n'+math+'\nnamespace gt=gpu_joint_tile;\n'+head.replace('joint_raw_uv_shared_vertices_v2',symbol)+'{'+tiled
    return source,dict(kernel_abi=abi,kernel_symbol=symbol,directions_per_thread=width,UVLayout_bytes=68,UVJob_bytes=8,UVPartial_bytes=24,
        source_kernel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),adapter_file=str(Path(__file__).resolve()),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mixed_row_tiles='scalar per original job',exceptional_derivative_lanes='scalar replay for exact per-job arithmetic/status',production_default=False)
