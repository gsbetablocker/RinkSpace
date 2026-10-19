"""Generate an experimental HD math view of frozen CPU templates, never edit them."""
from pathlib import Path
import re,json,hashlib
HERE=Path(__file__).parent
ROOT=HERE.parent/'include/rinkfit'
physical=(ROOT/'physical.hpp').read_text(encoding='utf-8')
event=(ROOT/'event_model.hpp').read_text(encoding='utf-8')
def extract(text,name):
    m=re.search(r'template\s*<[^\n]*>[^\n]*\b'+name+r'\s*\(',text)
    if not m:raise ValueError(name)
    start=m.start();left=text.index('{',m.end());depth=1;end=left+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    body=text[start:end]
    # Keep the CPU templates' scalar branches and exact implicit derivatives.
    body=re.sub(r'throw std::(?:domain_error|out_of_range)\("[^"]*"\)', 'gpu_abort(__LINE__)',body)
    body=re.sub(r'(template\s*<[^>\n]*>)',r'\1 HD',body,count=1)
    return body
physical=physical[physical.index('namespace rinkfit {'):physical.rindex('}  // namespace rinkfit')]
physical=physical.replace('namespace rinkfit {','namespace gpu_exact {',1)
physical=physical.replace('inline ','HD inline ').replace('bool valid() const','HD bool valid() const')
physical=re.sub(r'throw std::(?:domain_error|out_of_range)\("[^"]*"\)', 'gpu_abort(__LINE__)',physical)
physical=physical.replace('std::isfinite(optics.f_ref)','::isfinite(optics.f_ref)')
helpers='\n'.join(extract(event,n)for n in ['event_scalar','event_pow_nonnegative','event_normalized','event_exp_rotation'])
projection=event[event.index('template<class T> struct EventALProjection'):event.index('template<class T> inline double event_scalar')]
projection=projection.replace('bool physical_valid() const','HD bool physical_valid() const')
methods='\n'.join(extract(event,n)for n in ['control','trajectory','head','optics','quarter','quarter_speed','quarter_arclength','board_point','world_point','predict_for_al'])
methods=methods.replace('const EventPath& p','const Path& p').replace('const EventRowLayout& l','const Layout& l')
methods=methods.replace('materials.at(l.material).paths.at(l.path)','paths[l.flatpath]')
prefix='''#pragma once
#include "abi.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#define HD __host__ __device__
namespace gpu_exact {
__device__ int gpu_first_math_failure=0;
[[noreturn]] HD inline void gpu_abort(int line){
#ifdef __CUDA_ARCH__
  if(atomicCAS(&gpu_first_math_failure,0,line)==0)printf("GPU math guard line %d\\n",line);
  asm("trap;");
#else
  throw std::domain_error("GPU experiment unsupported arithmetic-domain exception");
#endif
}
struct Jet {double a,v;HD Jet(double x=0):a(x),v(0){}HD Jet(double x,double d):a(x),v(d){}
HD Jet& operator+=(Jet b){a+=b.a;v+=b.v;return *this;}
HD Jet& operator-=(Jet b){a-=b.a;v-=b.v;return *this;}
HD Jet& operator*=(Jet b){v=v*b.a+a*b.v;a*=b.a;return *this;}
HD Jet& operator/=(Jet b){const double inverse=1./b.a,ratio=a*inverse;v=(v-ratio*b.v)*inverse;a=ratio;return *this;}};
HD inline Jet operator+(Jet a,Jet b){return a+=b;}HD inline Jet operator-(Jet a,Jet b){return a-=b;}
HD inline Jet operator*(Jet a,Jet b){return a*=b;}HD inline Jet operator/(Jet a,Jet b){return a/=b;}
HD inline Jet operator-(Jet a){return Jet(-a.a,-a.v);}
HD inline bool operator<(Jet a,Jet b){return a.a<b.a;}HD inline bool operator>(Jet a,Jet b){return a.a>b.a;}
HD inline bool operator<=(Jet a,Jet b){return a.a<=b.a;}HD inline bool operator>=(Jet a,Jet b){return a.a>=b.a;}
HD inline Jet sin(Jet x){return Jet(::sin(x.a),::cos(x.a)*x.v);}HD inline Jet cos(Jet x){return Jet(::cos(x.a),-::sin(x.a)*x.v);}
HD inline Jet sqrt(Jet x){double y=::sqrt(x.a),scale=.5/y;return Jet(y,x.v*scale);}HD inline Jet exp(Jet x){double y=::exp(x.a);return Jet(y,y*x.v);}
HD inline Jet pow(Jet x,Jet p){double y=::pow(x.a,p.a),dx=p.a*::pow(x.a,p.a-1.),dp=y*::log(x.a);return Jet(y,dx*x.v+dp*p.v);}
HD inline Jet abs(Jet x){return x.a<0?-x:x;}HD inline bool isfinite(Jet x){return ::isfinite(x.a)&&::isfinite(x.v);}
template<class T>struct Span {const T* data=nullptr;int count=0;HD const T& operator[](int i)const{return data[i];}HD const T& at(int i)const{return data[i];}HD int size()const{return count;}HD bool empty()const{return count==0;}};
} // namespace gpu_exact
'''
suffix='''
constexpr int kEventGlobalSize=32;
struct Path {int kind,radius_index,boundary_segment;double start_angle,end_angle,fixed_height_fraction;Span<Vec2<double>> points;};
struct Layout {int global_slot=0,geometry_slot=-1,advertisement_slot=-1,flatpath=0;double frame_time=0;Span<int>control_ids,control_slots;};
struct Model {
Mat3<double> R0_seed;Vec3<double> n_seed,b_seed,n_tangent1,n_tangent2;
double f_ref,radius_max,dt,first_control_time;int n_controls,control_block_begin,dependent_control;
std::array<int,4>reference_indices;std::array<double,4>reference_weights;const Path* paths;
'''
out=prefix+physical+'\n'+helpers+'\n'+projection+'\n'+suffix+methods+'\n};\n} // namespace gpu_exact\n'
(HERE/'exact_math.cuh').write_text(out,encoding='utf-8')
(Path(__import__('os').environ.get('RINKFIT_GPU_ROWS_BUILD',r'D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/gpu_accel/rows'))/'exact_math_generation.json').write_text(json.dumps({'frozen_sources':{str(ROOT/x):hashlib.sha256((ROOT/x).read_bytes()).hexdigest()for x in ['physical.hpp','event_model.hpp']},'generated_sha256':hashlib.sha256(out.encode()).hexdigest(),'methods_preserved':['control','trajectory','head','optics','quarter','quarter_speed','quarter_arclength','board_point','world_point','predict_for_al'],'arith_exception_scope':'Experimental device trap, not a production per-row recovery path. Current full-state comparisons must have no traps.'},indent=2)+'\n')
