"""Optional original-setup strict Brown4 value path; OpenCV Jacobians stay."""
from calibration.solver_runtime.config import native_build_directory
import ctypes,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'native'/'projection_value.c'
DEST = native_build_directory('projection_value', SOURCE)
_kernel=None;_library=None;_error=None;_manifest=None
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
try:
    _manifest=json.loads((DEST/'build.json').read_text(encoding='utf8'))
    binary=DEST/_manifest.get('binary',_manifest.get('binary_name','projection_value.dll'))
    if _manifest['source_sha256']!=sha(SOURCE)or _manifest['dll_sha256']!=sha(binary):raise ValueError('Native projection value source/binary mismatch')
    _library=ctypes.CDLL(str(binary));_kernel=_library.project_brown4_value
    _kernel.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int64,ctypes.c_void_p];_kernel.restype=ctypes.c_int
except(OSError,ValueError,KeyError,AttributeError)as exc:_error=str(exc)


def metadata():return dict(available=_kernel is not None,load_error=_error,native_build=_manifest,
    original_cv2_camera_setup=True,original_OpenCV_Jacobians=True,experimental_value_path=True)


def project(camera,rotation,translation,intrinsic,world):
    if _kernel is None:raise RuntimeError('Build optional native projection value kernel: '+str(_error))
    if np.any(camera[15:]!=0.):return None
    uv=np.empty((len(world),2),np.float64)
    arrays=[np.ascontiguousarray(a,dtype=np.float64)for a in(camera,rotation,translation,intrinsic,world)]
    status=_kernel(*[a.ctypes.data for a in arrays],len(world),uv.ctypes.data)
    return None if status else uv
