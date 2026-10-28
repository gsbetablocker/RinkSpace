"""Large native array offsets must not wrap at INT_MAX.

Reserve virtual address space and touch two pages, not a multi-GB allocation.
Compile the actual CPU spline and host/device whitening helper on the CPU;
this checks integer/address arithmetic and does not claim CUDA validation.
"""
from pathlib import Path
import os
import shutil
import struct
import subprocess

import pytest


@pytest.mark.skipif(os.name != "posix" or struct.calcsize("P") != 8,
                    reason="requires a 64-bit POSIX virtual address space")
def test_large_offsets_in_native_whitening_and_spline(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/src/gpu_linearization.cuh").read_text()
    # Isolate the actual shared arithmetic from CUDA-only resource wrappers.
    structs = source[source.index("struct GroupPlan"):source.index("inline void cuda_check")]
    view = source[source.index("struct LinearView"):source.index("RS_HD inline void linearize_row")]
    whiten = source[source.index("RS_HD inline double whiten_element"):
                    source.index("RS_HD inline void group_loss")]
    header = ('#include "rinkspace/camera_envelope.hpp"\n'
              '#include "rinkspace/scene_derivatives.hpp"\n#include <sys/mman.h>\n'
              '#ifndef MAP_NORESERVE\n#define MAP_NORESERVE 0\n#endif\n'
              'using namespace rinkspace;\n')
    main = r"""
int main(){
    constexpr int row=1072670,q=1000;
    const size_t offset=size_t(row)*2*(q+1),size=(offset+q+2)*sizeof(double);
    auto* jac=static_cast<double*>(mmap(nullptr,size,PROT_READ|PROT_WRITE,
        MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0));
    if(jac==MAP_FAILED)return 77;
    jac[offset]=3.;jac[offset+q+1]=4.;
    GroupPlan group{0,1,1,0,0,0};int member=0,columns[q];for(int j=0;j<q;++j)columns[j]=j;double weights[2]{.5,2.};
    ComponentPlan component{};component.q_count=1;component.dim=1;
    RowPlan row_plan{0,q,offset};
    LinearView l{};l.groups=&group;l.members=&member;l.weights=weights;l.q=q;
    l.components=&component;l.row_plan=&row_plan;l.columns=columns;
    double value=whiten_element(l,0,0,1,nullptr,jac);
    munmap(jac,size);if(value!=9.5)return 1;
    const size_t count=size_t(std::numeric_limits<int>::max())/3+5;
    const size_t bytes=3*count*sizeof(double);
    auto* controls=static_cast<double*>(mmap(nullptr,bytes,PROT_READ|PROT_WRITE,
        MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0));
    if(controls==MAP_FAILED)return 77;
    const int cell=int(count)-3;
    for(int j=0;j<4;++j)for(int c=0;c<3;++c)
        controls[3*size_t(cell-1+j)+c]=c+2.;
    MotionSpline spline{controls,count,0.,1.};double evaluated[3];
    auto status=spline.evaluate(cell+.25,evaluated);
    munmap(controls,bytes);
    return status==Status::ok&&std::abs(evaluated[0]-2.)<1e-14?0:2;
}
"""
    cpp = tmp_path / "indexing.cpp"
    executable = tmp_path / "indexing"
    cpp.write_text(header + structs + view + whiten + main)
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=undefined",
                    "-fno-sanitize-recover=undefined", "-I" + str(root / "native/include"),
                    str(cpp), "-o", str(executable)], check=True, capture_output=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("OS denied the virtual address reservation")
    assert result.returncode == 0, result.stderr
