"""ctypes boundary only: all geometry and projection math lives in C++.

Loading requires an explicit built-library path. It does not install packages,
compile code, load a historical state, or start an optimizer.
"""
from __future__ import annotations

import ctypes as c
from pathlib import Path
from typing import Sequence


D = c.POINTER(c.c_double)
I = c.POINTER(c.c_int)
N = c.c_size_t


def _d(values: Sequence[float]):
    return (c.c_double * len(values))(*values)


def _points(values) -> list[tuple[float, float, float]]:
    return [tuple(values[i:i+3]) for i in range(0, len(values), 3)]


class NativeError(ValueError):
    def __init__(self, operation: str, status: int, message: str):
        super().__init__(f"{operation}: {message} (status {status})")
        self.status = status


class NativeCore:
    ABI_VERSION = 2

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path).resolve(strict=True)
        self._dll_dirs = []
        if __import__("os").name == "nt":
            import os, json
            project = Path(__file__).resolve().parents[3]
            directories = []
            cuda = os.environ.get("CUDA_PATH")
            if cuda: directories.extend([Path(cuda)/"bin"/"x64",Path(cuda)/"bin"])
            local = project/"environments/cudss-local.json"
            if local.is_file():
                root = (project/json.loads(local.read_text(encoding="utf-8"))["root"]).resolve()
                if not root.is_relative_to(project):
                    raise ValueError("cuDSS runtime must remain inside this project")
                directories.append(root/"bin")
            for directory in directories:
                if directory.is_dir():
                    self._dll_dirs.append(os.add_dll_directory(str(directory)))
        self._lib = c.CDLL(str(self.library_path))
        self._lib.rs_core_abi_version.argtypes = []
        self._lib.rs_core_abi_version.restype = c.c_int
        version = self._lib.rs_core_abi_version()
        if version != self.ABI_VERSION:
            raise ValueError(f"native ABI {version}; wrapper requires {self.ABI_VERSION}")
        self._lib.rs_status_message.argtypes = [c.c_int]
        self._lib.rs_status_message.restype = c.c_char_p
        signatures = {
            "rs_sample_entity_path": [D,N,c.c_int,D,N,D,N,D,N,D,N,I,N],
            "rs_shared_entity_definition": [D,N,D,N,c.c_int,I,D,N],
            "rs_decode_d78_camera": [D, N, D, N],
            "rs_project_motion": [D, N, D, N, c.c_double, c.c_double, c.c_int, D, N, D, N, D, N, I, N],
            "rs_curling_centres": [D, N, N, D, N],
            "rs_curling_ring": [D, N, N, c.c_int, c.c_int, c.c_double, D, N],
            "rs_curling_lanes": [D, N, N, D, N, D, N],
            "rs_centre_circle": [c.c_double, c.c_double, D, N],
            "rs_line_line": [D, N, D, N, D, N],
            "rs_line_circle": [D, N, D, N, c.c_int, D, N],
            "rs_project_points": [D, N, D, N, D, N, I, N],
            "rs_project_jacobian": [D, N, D, N, D, N],
            "rs_pivot_pose": [D, N, D, N, D, N, D, N, c.c_double, D, N],
            "rs_board_perimeter": [D, N, D, N],
            "rs_board_point": [D, N, c.c_double, c.c_double, D, N],
            "rs_board_points": [D, N, D, N, D, N, D, N, I, N],
            "rs_board_ad_polyline": [D, N, c.c_double, c.c_double, c.c_double, c.c_double, N, D, N],
        }
        for name, args in signatures.items():
            fn = getattr(self._lib, name)
            fn.argtypes = args
            fn.restype = c.c_int

        # Additive capabilities preserve loading of historical ABI2 libraries.
        optional = {
            "rs_evaluate_motion_camera": [D,N,D,N,c.c_double,c.c_double,D,N,D,N,I,N],
            "rs_evaluate_motion_kinematics": [D,N,c.c_double,c.c_double,D,N,D,N,I,N],
            "rs_motion_derivative_extrema": [D,N,c.c_double,c.c_double,c.c_double,c.c_double,D,N],
            "rs_camera_envelope_evaluate": [D,N,D,N,c.c_double,D,N,D,N],
            "rs_sample_print_deformed_material_path": [D,N,D,N,D,N,D,N,D,N,D,N,D,N,I,N],
            "rs_sample_print_analytic_material_path": [D,N,c.c_int,c.c_int,c.c_int,D,N,D,N,D,N,D,N,I,N],
            "rs_ice_board_junction": [D,N,D,N,c.c_double,c.c_int,D,N],
            "rs_sample_print_analytic_path": [D,N,D,N,c.c_int,c.c_int,c.c_int,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,I,N],
            "rs_deform_print_points": [D,N,D,N,D,N,D,N,D,N,I,N],
            "rs_sample_print_deformed_path": [D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,D,N,I,N],
            "rs_print_square_vertices": [D,N,D,N,c.c_double,D,N],
            "rs_sample_print_chart": [D,N,D,N,D,N,D,N,I,N],
            "rs_shared_entity_definition_with_options": [D,N,D,N,c.c_int,c.c_uint,I,D,N],
            "rs_board_course_chart": [D,N,D,N],
            "rs_board_stations_from_legacy": [D,N,c.c_int,D,N,D,N],
        }
        for name, args in optional.items():
            if hasattr(self._lib, name):
                fn=getattr(self._lib,name);fn.argtypes=args;fn.restype=c.c_int

    def _call(self, operation, *args) -> None:
        status = getattr(self._lib, operation)(*args)
        if status:
            raise NativeError(operation, status, self.status_message(status))

    def status_message(self, status: int) -> str:
        return self._lib.rs_status_message(status).decode("utf-8")

    def centres(self, parameters: Sequence[float], rows: int) -> list[tuple[float, float, float]]:
        if type(rows) is not int or rows <= 0:
            raise ValueError("row count must be positive")
        out = (c.c_double * (rows*6))()
        self._call("rs_curling_centres", _d(parameters), len(parameters), rows, out, len(out))
        return _points(out)

    def ring(self, parameters, rows: int, house: int, ring: int, theta: float) -> tuple[float, float, float]:
        out = (c.c_double * 3)()
        self._call("rs_curling_ring", _d(parameters), len(parameters), rows, house, ring, theta, out, 3)
        return tuple(out)

    def lanes(self, parameters, rows: int, left_support) -> list[list[list[tuple[float, float, float]]]]:
        if type(rows) is not int or rows <= 0:
            raise ValueError("row count must be positive")
        out = (c.c_double * (rows*18))()
        self._call("rs_curling_lanes", _d(parameters), len(parameters), rows, _d(left_support), len(left_support), out, len(out))
        points = _points(out)
        return [[points[row*6+edge*2:row*6+edge*2+2] for edge in range(3)] for row in range(rows)]

    def centre_circle(self, radius: float, theta: float) -> tuple[float, float, float]:
        out = (c.c_double * 3)()
        self._call("rs_centre_circle", radius, theta, out, 3)
        return tuple(out)

    def line_line(self, a, b) -> tuple[float, float, float]:
        out = (c.c_double * 3)()
        self._call("rs_line_line", _d(a), len(a), _d(b), len(b), out, 3)
        return tuple(out)

    def line_circle(self, line, circle, branch: int) -> tuple[float, float, float]:
        out = (c.c_double * 3)()
        self._call("rs_line_circle", _d(line), len(line), _d(circle), len(circle), branch, out, 3)
        return tuple(out)

    def ice_board_junction(self, board, line, height_fraction, branch: int):
        if type(branch) is not int:
            raise ValueError("junction branch must be an explicit integer")
        if not hasattr(self._lib, 'rs_ice_board_junction'):
            raise ValueError("native library lacks ice/board fixed-junction capability")
        out = (c.c_double * 3)()
        self._call('rs_ice_board_junction', _d(board), len(board), _d(line), len(line),
                   height_fraction, branch, out, 3)
        return tuple(out)

    def project(self, camera, xyz) -> tuple[list[tuple[float, float]], list[int]]:
        if any(len(point) != 3 for point in xyz):
            raise ValueError("expected XYZ points")
        flat = [value for point in xyz for value in point]
        out = (c.c_double * (len(xyz)*2))()
        status = (c.c_int * len(xyz))()
        self._call("rs_project_points", _d(camera), len(camera), _d(flat), len(xyz), out, len(out), status, len(status))
        return [tuple(out[i:i+2]) for i in range(0, len(out), 2)], list(status)

    def project_jacobian(self, camera, xyz) -> tuple[tuple[float, float], tuple[float, ...]]:
        out = (c.c_double * 8)()
        self._call("rs_project_jacobian", _d(camera), len(camera), _d(xyz), len(xyz), out, 8)
        return tuple(out[:2]), tuple(out[2:])

    def pivot_pose(self, rotation, pivot, lever, axis_world, axis_spacing: float) -> tuple[float, ...]:
        out = (c.c_double * 12)()
        self._call("rs_pivot_pose", _d(rotation), len(rotation), _d(pivot), len(pivot), _d(lever), len(lever),
                   _d(axis_world), len(axis_world), axis_spacing, out, 12)
        return tuple(out)

    def board_perimeter(self, board) -> float:
        out = (c.c_double * 1)()
        self._call("rs_board_perimeter", _d(board), len(board), out, 1)
        return float(out[0])

    def board_point(self, board, station: float, z: float) -> tuple[float, ...]:
        out = (c.c_double * 6)()
        self._call("rs_board_point", _d(board), len(board), station, z, out, 6)
        return tuple(out)

    def board_points(self, board, stations, heights) -> list[tuple[float, float, float]]:
        out = (c.c_double * (len(stations)*3))()
        status = (c.c_int * len(stations))()
        self._call("rs_board_points", _d(board), len(board), _d(stations), len(stations), _d(heights), len(heights),
                   out, len(out), status, len(status))
        failures = [(i, value) for i, value in enumerate(status) if value]
        if failures:
            i, value = failures[0]
            raise NativeError(f"rs_board_points row {i}", value, self.status_message(value))
        return _points(out)

    def ad_polyline(self, board, s0: float, width: float, z0: float, height: float, samples: int = 24):
        if samples < 2:
            raise ValueError("at least two samples per edge")
        out = (c.c_double * (4*samples*3))()
        self._call("rs_board_ad_polyline", _d(board), len(board), s0, width, z0, height, samples, out, len(out))
        points = _points(out)
        return [points[i*samples:(i+1)*samples] for i in range(4)]

    def decode_d78_camera(self, chart):
        out = (c.c_double * 38)()
        self._call("rs_decode_d78_camera", _d(chart), len(chart), out, 38)
        return list(out)

    def evaluate_camera_envelope(self, rig, controls, first_control_time, control_dt, limits):
        """Exact native continuous envelope diagnostic; this never changes state."""
        import numpy as np
        physical=np.ascontiguousarray(rig,np.float64);coefficients=np.ascontiguousarray(controls,np.float64)
        bounds=np.ascontiguousarray(limits,np.float64)
        valid_bounds=bounds.ndim==1 and (bounds.size in (10,19) or (bounds.size>=21 and (bounds.size-19)%2==0))
        if physical.shape not in ((38,),(43,)) or coefficients.ndim!=2 or coefficients.shape[1]!=3 or not valid_bounds:
            raise ValueError("envelope requires rig[38/43], controls[M,3], limits[10/19/19+2*shot_count]")
        out=np.empty(8,np.float64)
        self._call("rs_camera_envelope_evaluate",physical.ctypes.data_as(D),physical.size,
            coefficients.ctypes.data_as(D),len(coefficients),float(control_dt),bounds.ctypes.data_as(D),bounds.size,out.ctypes.data_as(D),out.size)
        return dict(margin=float(out[0]),kind=int(out[1]),cell=int(out[2]),u=float(out[3]),
            time=float(first_control_time+(out[2]+out[3])*control_dt),channel=int(out[4]),
            derivative_order=int(out[5]),limit=float(out[6]),derivative=float(out[7]))

    def motion_derivative_extrema(self, controls, first_control_time, control_dt, interval):
        """Exact continuous speed/acceleration/jerk maxima, with native times and cells."""
        import numpy as np
        coefficients=np.ascontiguousarray(controls,np.float64);span=np.asarray(interval,np.float64)
        if coefficients.ndim!=2 or coefficients.shape[1]!=3 or span.shape!=(2,):
            raise ValueError("derivative extrema require controls[M,3] and interval[2]")
        out=np.full((3,3,3),np.nan)
        self._call("rs_motion_derivative_extrema",coefficients.ctypes.data_as(D),len(coefficients),
            float(first_control_time),float(control_dt),float(span[0]),float(span[1]),out.ctypes.data_as(D),out.size)
        return dict(value=out[:,:,0],time=out[:,:,1],cell=out[:,:,2].astype(np.int32))

    def evaluate_motion_kinematics(self, controls, first_control_time, control_dt, times):
        """Read the native trajectory and its analytic time derivatives, without smoothing."""
        import numpy as np
        if not hasattr(self._lib, "rs_evaluate_motion_kinematics"):
            raise ValueError("native library lacks trajectory derivative diagnostics")
        coefficients=np.ascontiguousarray(controls,dtype=np.float64)
        requested=np.atleast_1d(np.asarray(times,dtype=np.float64))
        if coefficients.ndim!=2 or coefficients.shape[1]!=3 or requested.ndim!=1:
            raise ValueError("kinematics requires controls[M,3] and scalar or times[N]")
        requested=np.ascontiguousarray(requested);out=np.full((len(requested),4,3),np.nan,np.float64)
        status=np.empty(len(requested),np.int32)
        self._call("rs_evaluate_motion_kinematics",coefficients.ctypes.data_as(D),len(coefficients),
            float(first_control_time),float(control_dt),requested.ctypes.data_as(D),len(requested),
            out.ctypes.data_as(D),out.size,status.ctypes.data_as(I),len(status))
        return dict(values=out,position=out[:,0],velocity=out[:,1],acceleration=out[:,2],jerk=out[:,3],status=status)

    def evaluate_motion_camera(self, rig, controls, first_control_time, control_dt, times):
        """Evaluate the same native camera at explicit spline times.

        An instant is not a rolling-shutter frame. The original signed readout
        and predicted-row equation remain necessary for frame projection.
        Failed time rows retain NaNs and their native status.
        """
        import numpy as np
        if not hasattr(self._lib, "rs_evaluate_motion_camera"):
            raise ValueError("native library lacks instantaneous motion-camera snapshot capability")
        physical = np.ascontiguousarray(rig, dtype=np.float64)
        coefficients = np.ascontiguousarray(controls, dtype=np.float64)
        requested = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if physical.shape not in ((38,), (43,)) or coefficients.ndim != 2 or coefficients.shape[1] != 3 or requested.ndim != 1:
            raise ValueError("camera evaluation requires rig[38], controls[M,3] and scalar or times[N]")
        requested = np.ascontiguousarray(requested)
        out = np.full((len(requested), 24), np.nan, dtype=np.float64)
        status = np.empty(len(requested), dtype=np.int32)
        self._call("rs_evaluate_motion_camera", physical.ctypes.data_as(D), physical.size,
                   coefficients.ctypes.data_as(D), len(coefficients), float(first_control_time), float(control_dt),
                   requested.ctypes.data_as(D), len(requested), out.ctypes.data_as(D), out.size,
                   status.ctypes.data_as(I), len(status))
        return {"values": out, "pan": out[:, 0], "tilt": out[:, 1], "log_f": out[:, 2], "zoom_log_ratio": out[:, 3],
                "rotation": out[:, 4:13].reshape(-1, 3, 3), "centre": out[:, 13:16],
                "fx": out[:, 16], "fy": out[:, 17], "cx": out[:, 18], "cy": out[:, 19],
                "k1": out[:, 20], "k2": out[:, 21], "p1": out[:, 22], "p2": out[:, 23],
                "status": status}

    def project_motion(self, rig, controls, first_control_time, control_dt, image_height, frame_times, xyz):
        import numpy as np
        points = np.ascontiguousarray(xyz, dtype=np.float64)
        coefficients = np.ascontiguousarray(controls, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or coefficients.ndim != 2 or coefficients.shape[1] != 3:
            raise ValueError("motion projection requires XYZ[N,3] and controls[M,3]")
        times = np.ascontiguousarray(np.broadcast_to(np.asarray(frame_times, dtype=np.float64), (len(points),)))
        physical = np.ascontiguousarray(rig, dtype=np.float64)
        out = np.full((len(points), 6), np.nan, dtype=np.float64)
        status = np.empty(len(points), dtype=np.int32)
        self._call("rs_project_motion", physical.ctypes.data_as(D), physical.size,
                   coefficients.ctypes.data_as(D), len(coefficients), float(first_control_time), float(control_dt), int(image_height),
                   times.ctypes.data_as(D), len(times), points.ctypes.data_as(D), len(points),
                   out.ctypes.data_as(D), out.size, status.ctypes.data_as(I), len(status))
        return {"uv": out[:, :2], "depth": out[:, 2], "row": out[:, 3], "root_equality_px": out[:, 4],
                "contraction_bound": out[:, 5], "status": status}

    def sample_path(self, board, kind, parameters, vertices, stations):
        flat = [value for point in vertices for value in point]
        out = (c.c_double * (len(stations)*3))()
        status = (c.c_int * len(stations))()
        self._call("rs_sample_entity_path", _d(board), len(board), kind,
                   _d(parameters), len(parameters), _d(flat), len(flat),
                   _d(stations), len(stations), out, len(out), status, len(status))
        failures = [(i, value) for i, value in enumerate(status) if value]
        if failures:
            i, value = failures[0]
            raise NativeError(f"rs_sample_entity_path row {i}", value, self.status_message(value))
        return _points(out)

    def deform_print_points(self, vertices, bounds, shape_values, *, jacobian=True):
        """Native shared-group material deformation; failures retain NaN rows."""
        import numpy as np
        if not hasattr(self._lib, "rs_deform_print_points"):
            raise ValueError("native library lacks adjustable print-shape capability")
        points = np.ascontiguousarray(vertices, dtype=np.float64)
        box = np.ascontiguousarray(bounds, dtype=np.float64)
        shape = np.ascontiguousarray(shape_values, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or box.shape != (4,) or shape.shape != (10,):
            raise ValueError("print deformation requires material UV[N,2], bounds[4] and shape[10]")
        uv = np.full(points.shape, np.nan, dtype=np.float64)
        deriv = np.full((len(points), 2, 10), np.nan, dtype=np.float64) if jacobian else None
        status = np.empty(len(points), dtype=np.int32)
        self._call("rs_deform_print_points", points.ctypes.data_as(D), points.size,
                   box.ctypes.data_as(D), box.size, shape.ctypes.data_as(D), shape.size,
                   uv.ctypes.data_as(D), uv.size,
                   None if deriv is None else deriv.ctypes.data_as(D), 0 if deriv is None else deriv.size,
                   status.ctypes.data_as(I), status.size)
        return {"material_uv": uv, "d_uv_d_shape": deriv, "status": status}

    def sample_deformed_print_path(self, board, parameters, vertices, stations, bounds,
                                   shape_values, *, jacobian=True):
        """Same native kind5 chart/course, with an active material shape group.

        Shape columns are [tx,ty,log_sx,log_sy,xlogw0..2,ylogw0..2].
        Installation columns use physical [s,z,rotation,width,height]. No source
        pixels, fit status or camera parameters are changed by this evaluation.
        board=None, jacobian=False explicitly requests only material UV/SZ;
        that result has no XYZ or world derivatives and needs the additive
        material-only native capability.
        """
        import numpy as np
        material_only = board is None
        if material_only and jacobian:
            raise ValueError("material-only print evaluation has no world Jacobian")
        operation = "rs_sample_print_deformed_material_path" if material_only else "rs_sample_print_deformed_path"
        if not hasattr(self._lib, operation):
            raise ValueError("native library lacks "+("material-only " if material_only else "")+"adjustable print-path capability")
        points = np.ascontiguousarray(vertices, dtype=np.float64)
        course = None if material_only else np.ascontiguousarray(board, dtype=np.float64)
        installation = np.ascontiguousarray(parameters, dtype=np.float64)
        coordinates = np.ascontiguousarray(stations, dtype=np.float64)
        box = np.ascontiguousarray(bounds, dtype=np.float64)
        shape = np.ascontiguousarray(shape_values, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or (course is not None and course.shape != (6,))
                or installation.shape != (5,) or coordinates.ndim != 1
                or box.shape != (4,) or shape.shape != (10,)):
            raise ValueError("print path requires board[6], installation[5], UV[N,2], stations[M], bounds[4], shape[10]")
        n = len(coordinates)
        result = {"material_uv": np.full((n, 2), np.nan), "sz": np.full((n, 2), np.nan),
                  "status": np.empty(n, dtype=np.int32)}
        if not material_only:
            result["xyz"] = np.full((n, 3), np.nan)
        derivative_shapes = {} if material_only else {"d_uv_d_shape": (n, 2, 10), "d_sz_d_shape": (n, 2, 10),
                             "d_xyz_d_shape": (n, 3, 10), "d_xyz_d_installation": (n, 3, 5),
                             "d_xyz_d_station": (n, 3)}
        result.update({key: np.full(size, np.nan) if jacobian else None
                       for key, size in derivative_shapes.items()})
        args = [] if material_only else [course.ctypes.data_as(D), course.size]
        args.extend([installation.ctypes.data_as(D), installation.size,
                points.ctypes.data_as(D), points.size, coordinates.ctypes.data_as(D), coordinates.size,
                box.ctypes.data_as(D), box.size, shape.ctypes.data_as(D), shape.size])
        outputs = ("material_uv", "sz") if material_only else ("material_uv", "sz", "xyz", *derivative_shapes)
        for key in outputs:
            value = result[key]
            args.extend((None if value is None else value.ctypes.data_as(D), 0 if value is None else value.size))
        args.extend((result["status"].ctypes.data_as(I), n))
        self._call(operation, *args)
        return result

    def _sample_analytic_print_path(self, board, parameters, values, stations, *,
                                    family, side=0, reverse=False, jacobian=True):
        import numpy as np
        material_only = board is None
        if material_only and jacobian:
            raise ValueError("material-only print evaluation has no world Jacobian")
        operation = "rs_sample_print_analytic_material_path" if material_only else "rs_sample_print_analytic_path"
        if not hasattr(self._lib, operation):
            raise ValueError("native library lacks "+("material-only " if material_only else "")+"adjustable analytic print capability")
        course = None if material_only else np.ascontiguousarray(board, dtype=np.float64)
        installation = np.ascontiguousarray(parameters, dtype=np.float64)
        shape = np.ascontiguousarray(values, dtype=np.float64)
        coordinates = np.ascontiguousarray(stations, dtype=np.float64)
        width = 3 if family == 0 else 4
        if ((course is not None and course.shape != (6,)) or installation.shape != (5,)
                or shape.shape != (width,) or coordinates.ndim != 1):
            raise ValueError("invalid analytic print array shapes")
        if type(side) is not int or type(reverse) is not bool:
            raise ValueError("analytic print side/reverse must be integer/boolean")
        n = len(coordinates)
        name = "square" if family == 0 else "rectangle"
        result = {"material_uv": np.full((n, 2), np.nan), "sz": np.full((n, 2), np.nan),
                  "status": np.empty(n, dtype=np.int32)}
        if not material_only:
            result["xyz"] = np.full((n, 3), np.nan)
        derivative_shapes = {} if material_only else {"d_uv_d_"+name: (n, 2, width), "d_sz_d_"+name: (n, 2, width),
                             "d_xyz_d_"+name: (n, 3, width), "d_xyz_d_installation": (n, 3, 5),
                             "d_xyz_d_station": (n, 3)}
        result.update({key: np.full(size, np.nan) if jacobian else None
                       for key, size in derivative_shapes.items()})
        args = [] if material_only else [course.ctypes.data_as(D), course.size]
        args.extend([installation.ctypes.data_as(D), installation.size,
                family, side, int(reverse), shape.ctypes.data_as(D), shape.size,
                coordinates.ctypes.data_as(D), coordinates.size])
        outputs = ("material_uv", "sz") if material_only else ("material_uv", "sz", "xyz", *derivative_shapes)
        for key in outputs:
            value = result[key]
            args.extend((None if value is None else value.ctypes.data_as(D), 0 if value is None else value.size))
        args.extend((result["status"].ctypes.data_as(I), n))
        self._call(operation, *args)
        return result

    def sample_print_square_path(self, board, parameters, square_values, stations, *, jacobian=True):
        """Native metric square: values=[centre_u,centre_v,log_side_over_height]."""
        return self._sample_analytic_print_path(board, parameters, square_values, stations,
                                               family=0, jacobian=jacobian)

    def sample_print_rectangle_path(self, board, parameters, rectangle_values, side, stations, *,
                                    reverse=False, jacobian=True):
        """Native shared rectangle edge: values=[centre_u,centre_v,log_w_uv,log_h_uv]."""
        return self._sample_analytic_print_path(board, parameters, rectangle_values, stations,
                                               family=1, side=side, reverse=reverse, jacobian=jacobian)

    def print_square_vertices(self, parameters, center_uv, side_over_panel_height) -> list[tuple[float, float]]:
        """Four regular unfolded-metric square corners plus closure; native math."""
        if not hasattr(self._lib, "rs_print_square_vertices"):
            raise ValueError("native library lacks unfolded print-square capability")
        out=(c.c_double*10)()
        self._call("rs_print_square_vertices",_d(parameters),len(parameters),
                   _d(center_uv),len(center_uv),float(side_over_panel_height),out,len(out))
        return [tuple(out[i:i+2]) for i in range(0,10,2)]

    def sample_print_chart(self, parameters, vertices, stations) -> list[tuple[float, float]]:
        if not hasattr(self._lib, "rs_sample_print_chart"):
            raise ValueError("native library lacks installed print-chart sampling capability")
        if any(len(point) != 2 for point in vertices):
            raise ValueError("expected material XY points")
        flat = [value for point in vertices for value in point]
        out = (c.c_double * (len(stations)*2))()
        status = (c.c_int * len(stations))()
        self._call("rs_sample_print_chart", _d(parameters), len(parameters), _d(flat), len(flat),
                   _d(stations), len(stations), out, len(out), status, len(status))
        failures = [(i, value) for i, value in enumerate(status) if value]
        if failures:
            i, value = failures[0]
            raise NativeError(f"rs_sample_print_chart row {i}", value, self.status_message(value))
        return [tuple(out[i:i+2]) for i in range(0, len(out), 2)]

    def shared_geometry(self, board, layout, feature_code, options=0):
        kind = c.c_int()
        # Geometry6 is unchanged for legacy33 and all line/circle entities.
        size = 8 if len(layout) in (35, 39, 146) and feature_code in (410, 411) else 6
        out = (c.c_double * size)()
        if options:
            if not hasattr(self._lib, "rs_shared_entity_definition_with_options"):
                raise ValueError("native library lacks requested shared-geometry options")
            self._call("rs_shared_entity_definition_with_options", _d(board), len(board), _d(layout), len(layout), feature_code,
                       options, c.byref(kind), out, len(out))
        else:
            self._call("rs_shared_entity_definition", _d(board), len(board), _d(layout), len(layout), feature_code,
                       c.byref(kind), out, len(out))
        return int(kind.value), list(out)

    def board_course_chart(self, board):
        if not hasattr(self._lib, "rs_board_course_chart"):
            raise ValueError("native library lacks continuous board-course chart")
        out=(c.c_double*9)()
        self._call("rs_board_course_chart", _d(board),len(board),out,9)
        return list(out)

    def board_stations_from_legacy(self, board, segment, local):
        if not hasattr(self._lib, "rs_board_stations_from_legacy"):
            raise ValueError("native library lacks legacy-to-global board-station import")
        out=(c.c_double*len(local))()
        self._call("rs_board_stations_from_legacy",_d(board),len(board),int(segment),_d(local),len(local),out,len(out))
        return list(out)
