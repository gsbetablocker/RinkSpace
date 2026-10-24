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
        size = 8 if len(layout) == 35 and feature_code in (410, 411) else 6
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
