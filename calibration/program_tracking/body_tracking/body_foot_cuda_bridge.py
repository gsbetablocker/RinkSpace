"""Thin device-pointer C ABI bridge and optional standalone DLL build.

Build with the existing environment, for example::
    python -m calibration.program_tracking.body_tracking.body_foot_cuda_bridge --build

No compiler, package, or environment is installed. Outputs stay in Data/build.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

REASONS = (
    '', 'body:no_triangle_intersection', 'body:vertex_on_section',
    'body:open_or_branched_section', 'body:ambiguous_triangle_section',
    'body:branched_section', 'body:multiple_section_loops',
    'body:self_intersecting_or_degenerate_section',
    'inner:self_intersecting_or_degenerate_section', 'inner:section_height_changed',
    'input:nonfinite_samples',
)


def default_library():
    data = Path(os.environ.get('RINKSPACE_DATA_ROOT', r'D:\CodexData\RinkSpace_Data'))
    return data / 'build/body_foot_gpu' / ('body_foot_cuda.dll' if os.name == 'nt' else 'libbody_foot_cuda.so')


class CollarKernel:
    def __init__(self, library=None):
        path = Path(library) if library else default_library()
        if not path.is_file():
            raise FileNotFoundError(f'{path}: build body_foot_cuda_bridge --build first')
        # Keep dependent CUDA directories registered for this object's life.
        self._dll_dirs = []
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            cuda = os.environ.get('CUDA_PATH')
            if cuda and (Path(cuda)/'bin').is_dir():
                self._dll_dirs.append(os.add_dll_directory(str(Path(cuda)/'bin')))
        self.library = ctypes.CDLL(str(path))
        self.launch = self.library.body_foot_collar
        p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        self.launch.argtypes = [p,i,i,p,i,p,i,p,p,p,i,p,d,d,d,p,p,p,p,p,p,p,p,p,p,p,p]
        self.launch.restype = i

    def __call__(self, vertices, topology, polygon, z_changed, height, tolerance, domain, stream):
        import torch
        n, nv = vertices.shape[:2]
        ne, nf = topology['edges'].shape[0], topology['triangle_edges'].shape[0]
        kw = dict(device=vertices.device)
        xy = torch.empty((n,ne,2),dtype=torch.float64,**kw)
        crossed = torch.empty((n,ne),dtype=torch.uint8,**kw)
        visited = torch.empty_like(crossed)
        order = torch.empty((n,ne),dtype=torch.int64,**kw)
        number = torch.empty((n,nf),dtype=torch.uint8,**kw)
        sums = torch.empty((n,nf),dtype=torch.int64,**kw)
        residual = torch.empty((n,),dtype=torch.float64,**kw)
        breach, minimum = torch.empty_like(residual), torch.empty_like(residual)
        count = torch.empty((n,),dtype=torch.int64,**kw)
        reason = torch.empty((n,),dtype=torch.int32,**kw)
        ptr = lambda v: ctypes.c_void_p(v.data_ptr())
        error = self.launch(ptr(vertices),n,nv,ptr(topology['edges']),ne,ptr(topology['triangle_edges']),nf,
            ptr(topology['edge_faces']),ptr(topology['edge_face_counts']),ptr(polygon),polygon.shape[1],ptr(z_changed),
            height,tolerance,domain,ptr(xy),ptr(crossed),ptr(visited),ptr(order),ptr(number),ptr(sums),
            ptr(residual),ptr(breach),ptr(minimum),ptr(count),ptr(reason),ctypes.c_void_p(stream.cuda_stream))
        if error:
            raise RuntimeError(f'body_foot_collar CUDA launch error {error}')
        # Arrays were created and consumed on the same current stream. Torch's
        # allocator safely delays their reuse until that stream reaches here.
        return dict(residual_breach_solver=residual, breach_solver=breach,
                    minimum_sample_gap_solver=minimum, sample_count=count,
                    unknown_reason_code=reason, unknown=reason!=0)


def build(output=None):
    import argparse
    import json
    import subprocess
    from scripts.build_core import prepare_toolchain
    project = Path(__file__).resolve().parents[3]
    target = Path(output).resolve() if output else default_library().resolve()
    if target == project or project in target.parents:
        raise ValueError('Build output must stay outside OneDrive source')
    target.parent.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(cuda=True,vcvars=None,generator=None,cmake=None)
    env, info = prepare_toolchain(args,project,target.parent)
    command = [info['nvcc'],'-shared','-O3','-std=c++17','-arch=sm_120','--fmad=false',
               '-o',str(target),str(Path(__file__).with_name('body_foot_cuda.cu'))]
    if os.name=='nt':command += ['-Xcompiler','/MD']
    result = subprocess.run(command,cwd=target.parent,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (target.parent/'build.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    (target.parent/'toolchain.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'CUDA build failed ({result.returncode}); see {target.parent / "build.log"}')
    return target


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true',required=True)
    parser.add_argument('--output',type=Path)
    arguments=parser.parse_args()
    print(build(arguments.output))
