"""CPU checks of shared native arithmetic and an invalid-envelope C ABI path.

The projection kernel body is compiled as ordinary C++ to check termination
and arithmetic. This is not a CUDA execution or a footage-accuracy test.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _compile(tmp_path, source, *, core=False, asan=False):
    compiler = shutil.which("g++")
    msvc = shutil.which("cl") if compiler is None else None
    if compiler is None and msvc is None:
        pytest.skip("C++ compiler required for shared native arithmetic")
    if compiler is None and asan:
        pytest.skip("GCC ASan/UBSan probe; the native API is also tested directly")
    cpp = tmp_path / "audit.cpp"
    executable = tmp_path / "audit"
    cpp.write_text(source, encoding="utf-8")
    if msvc:
        executable = executable.with_suffix('.exe')
        command = [msvc, '/nologo', '/std:c++17', '/EHsc', '/O1',
                   '/I'+str(ROOT/'native/include'), str(cpp), '/Fe:'+str(executable)]
        if core:
            command += [str(ROOT/'native/src/core_api.cpp')]
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
        return executable
    command = [compiler, "-std=c++17", "-O1", "-I" + str(ROOT / "native/include")]
    if asan:
        command += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    command.append(str(cpp))
    if core:
        command.append(str(ROOT / "native/src/core_api.cpp"))
    command += ["-o", str(executable)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return executable


def _projection_source():
    code = (ROOT / "native/src/gpu_solver.cuh").read_text(encoding="utf-8")
    body = code[code.index("__device__ bool lm_halfspace"):
                code.index("__global__ void lm_feasible_init")]
    return r'''
#include "rinkspace/core_types.hpp"
#include <algorithm>
#include <cassert>
#define __device__
#define __global__
using namespace rinkspace;
struct { int x=0; } threadIdx,blockIdx;
struct View { const double* rig; const double* layout; size_t nr; unsigned options; size_t nrig=1; };
struct LinearView {
    const double *scene_steps,*rig_directions,*rig_lower,*rig_upper;
    int q; double max_readout;
};
struct SolverView { const int* rig_bound_columns; double *step,*scratch; };
// This CPU arithmetic probe has one physical rig. The production kernel now
// selects its per-camera views; joint indexing is tested by the GPU suite.
View camera_view(View v,size_t camera){assert(camera==0);return v;}
LinearView camera_linear_view(LinearView l,size_t,size_t camera){assert(camera==0);return l;}
SolverView camera_solver_view(SolverView s,size_t,size_t camera){assert(camera==0);return s;}
''' + body


def test_shared_projection_preserves_a_legal_narrow_coupled_bound(tmp_path):
    """Padding both opposing faces used to make the GPU loop nonterminating."""
    executable = _compile(tmp_path, _projection_source() + r'''
int main(){
    double rig[38]{},layout[35]{},directions[27*2]{},lower[38],upper[38];
    int columns[38];
    for(int k=0;k<38;++k){lower[k]=-INFINITY;upper[k]=INFINITY;columns[k]=-1;}
    // cy = base + d0 + d1 is a legal two-column physical parameter map.
    directions[16*2]=directions[16*2+1]=1.;
    lower[27]=-1e-15;upper[27]=1e-15;
    double step[2]{},scratch[2]{};
    View v{rig,layout,38,0};LinearView l{nullptr,directions,lower,upper,2,1.};
    SolverView s{columns,step,scratch};
    lm_project_shared_step(v,l,s);
    if(step[0]!=0.||step[1]!=0.||scratch[1]!=0.)return 1;
    step[0]=step[1]=1.;
    lm_project_shared_step(v,l,s);
    return scratch[1]==0.&&step[0]+step[1]>=lower[27]&&step[0]+step[1]<=upper[27]?0:2;
}
''')
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_shared_projection_cannot_loop_forever_on_inconsistent_halfspaces(tmp_path):
    """An inner numerical failure must return to the host's safe boundary."""
    executable = _compile(tmp_path, _projection_source() + r'''
int main(){
    double rig[38]{},layout[35]{},directions[27*2]{},lower[38],upper[38];
    int columns[38];
    for(int k=0;k<38;++k){lower[k]=-INFINITY;upper[k]=INFINITY;columns[k]=-1;}
    // Deliberately inconsistent low-level constraints test failure handling.
    // The public attachment rejects an invalid starting state separately.
    directions[15*2]=directions[15*2+1]=1.;
    directions[16*2]=directions[16*2+1]=1.;
    lower[26]=1.;upper[27]=-1.;
    double step[2]{},scratch[2]{};
    View v{rig,layout,38,0};LinearView l{nullptr,directions,lower,upper,2,1.};
    SolverView s{columns,step,scratch};
    lm_project_shared_step(v,l,s);
    return scratch[1]==1.?0:1;
}
''')
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_readout_outside_spline_is_not_a_fourth_derivative_or_memory_read(tmp_path):
    executable = _compile(tmp_path, r'''
#include "rinkspace/core_api.h"
#include <cmath>
#include <limits>
int main(){
    double rig[38]{1,0,0,0,0,-1,0,1,0,0,0,1,1,0,0,0,-3,1.5,0,-.15,.1,0,0,0,
                   100,1,50,50,0,0,0,0,0,0,0,0,0,.02};
    double controls[24];
    for(int k=0;k<8;++k){controls[3*k]=0;controls[3*k+1]=.15;controls[3*k+2]=std::log(100.);}
    const double radians=3.14159265358979323846/180.;
    double limits[21]{-.1,.3,-.1,.1,3*radians,5*radians,-20*radians,5*radians,-3*radians,3*radians};
    for(int k=10;k<19;++k)limits[k]=std::numeric_limits<double>::infinity();
    limits[19]=1.;limits[20]=6.;
    double output[8]{};
    const int code=rs_camera_envelope_evaluate(rig,38,controls,8,1.,limits,21,output,8);
    if(code!=0||output[0]!=-std::numeric_limits<double>::infinity()||output[1]!=10.)return 1;
    return output[4]==-1.&&output[5]==0.&&output[7]==0.?0:2;
}
''', core=True, asan=True)
    # Only address/undefined accesses are at issue. LeakSanitizer cannot inspect
    # threads in some managed runtimes, independently of the native code.
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10,
                            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"})
    assert result.returncode == 0, result.stderr


def _station_source():
    """Use the actual board geometry, rolling projection and analytic tangent."""
    gpu = (ROOT / "native/src/core_gpu.cu").read_text(encoding="utf-8")
    station = (ROOT / "native/src/gpu_station_refinement.cuh").read_text(encoding="utf-8")
    view = gpu[gpu.index("struct View {"):gpu.index("RS_HD bool valid_installation")]
    geometry = gpu[gpu.index("RS_HD Status geometry_at"):
                   gpu.index("RS_HD Status scene_geometry_direction")]
    projection = gpu[gpu.index("RS_HD Status profile_projection"):
                     gpu.index("RS_HD Status resolve_profile")]
    body = station[:station.index("struct StationRefinement")]
    return r'''
#include "rinkspace/motion_derivatives.hpp"
#include "rinkspace/scene_derivatives.hpp"
#include <cstdio>
#define __global__
using namespace rinkspace;
struct { int x=0; } threadIdx,blockIdx;
struct { int x=1; } blockDim;
void atomicAdd(double* p,double x){*p+=x;}
void atomic_max_nonnegative(double* p,double x){*p=std::max(*p,x);}
''' + view + geometry + projection + body


def test_periodic_station_refinement_keeps_the_original_unwrapped_support(tmp_path):
    executable = _compile(tmp_path, _station_source() + r'''
int main(){
    double board[6]{.5,.25,.1,.1,2.,.02};BoardCourse course({.5,.25,.1,.1,2.,.02});
    const double period=course.perimeter();
    double rig[38]{1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,-3,0,0,0,0,0,0,
                   100,1,50,50,0,0,0,0,0,0,0,0,0,.02};
    double controls[24]{};for(int i=0;i<8;++i)controls[3*i+2]=std::log(100.);
    int codes[1]{},paths[7]{7,0,1,0,0,0,0};double path_data[1]{.5},times[1]{},stations[1]{},observed[2]{};
    View v{};v.board=board;v.rig=rig;v.controls=controls;v.stations=stations;v.times=times;
    v.observed=observed;v.codes=codes;v.paths=paths;v.path_data=path_data;v.course=&course;
    v.npaths=1;v.nc=8;v.n=1;v.nr=38;v.first=-2.;v.dt=1.;v.height=101;
    Vec2 target{};if(station_uv(v,0,.9*period,target)!=Status::ok)return 1;
    observed[0]=target.x;observed[1]=target.y;
    constexpr int count=257;StationSample samples[count];Vec2 grid[count];
    for(int k=0;k<count;++k){samples[k]={0,1,double(k)/(count-1)};threadIdx.x=k;
        station_sample_kernel(v,samples,count,grid);}
    threadIdx.x=0;
    for(int case_id=0;case_id<4;++case_id){
        StationSpan span{};span.begin=0;span.count=count;span.periodic=1;span.lo=0.;span.hi=1.;
        const double offset=case_id==0?0.:period;
        span.lower=case_id==2?-INFINITY:offset+.4*period;
        span.upper=case_id>=2?INFINITY:offset+.5*period;
        const double original=offset+.45*period;stations[0]=original;
        Vec2 before{};if(station_uv(v,0,original,before)!=Status::ok)return 2;
        const double cost_before=station_cost(before,target);double stats[4]{};
        station_refine_kernel(v,&span,samples,grid,stats);
        Vec2 after{};if(station_uv(v,0,stations[0],after)!=Status::ok)return 3;
        if(stations[0]<span.lower||stations[0]>span.upper)return 4;
        if(!(station_cost(after,target)<cost_before)||stats[0]!=1.)return 5;
        if(case_id>=2&&std::abs(stations[0]-1.9*period)>1e-7)return 6;
        std::printf("case %d: [%.17g, %.17g], %.17g -> %.17g, cost %.17g -> %.17g\n",
                    case_id,span.lower,span.upper,original,stations[0],cost_before,station_cost(after,target));
    }
}
''')
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
