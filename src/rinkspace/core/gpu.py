"""Thin binding for the resident native GPU solver. No geometry math here."""
from __future__ import annotations
import ctypes as c
import time
import numpy as np
from .native import D, I, N, NativeCore
L = c.POINTER(c.c_int64)


class ResidentResidualProblem:
    """Current native geometry, physical motion camera, and raw UV factors.

    The native handle owns both CPU reference inputs and resident CUDA buffers.
    Native LM owns state updates and acceptance. Unsupported rows keep their status.
    """

    def __init__(self, core: NativeCore, *, board, layout, options, rig, controls,
                 first_control_time, control_dt, image_height, codes, stations, times, observed, camera_ids=None):
        self._lib = core._lib
        if not hasattr(self._lib, "rs_g1_create"):
            raise RuntimeError("native library was built without G1 CUDA capability")
        self._lib.rs_g1_create.argtypes = [D,N,D,N,c.c_uint,D,N,D,N,c.c_double,c.c_double,c.c_int,
                                           I,D,D,D,N,c.POINTER(c.c_void_p),D,N]
        self._lib.rs_g1_create.restype = c.c_int
        self._lib.rs_g1_evaluate.argtypes = [c.c_void_p,c.c_int,D,N,I,N,D,N]
        self._lib.rs_g1_evaluate.restype = c.c_int
        self._lib.rs_g1_destroy.argtypes = [c.c_void_p]
        self._lib.rs_g1_destroy.restype = None
        self._lib.rs_g1_last_error.argtypes = []
        self._lib.rs_g1_last_error.restype = c.c_char_p
        board,layout,rig,controls,stations,times,observed = [
            np.ascontiguousarray(a,dtype=np.float64)
            for a in (board,layout,rig,controls,stations,times,observed)]
        raw_codes = np.asarray(codes)
        if raw_codes.dtype.kind not in "iu" or np.any(raw_codes>np.iinfo(np.int32).max) or np.any(raw_codes<np.iinfo(np.int32).min):
            raise ValueError("feature codes must be int32 native identities")
        codes=np.ascontiguousarray(raw_codes,dtype=np.int32)
        self.count=len(codes)
        self.control_count=len(controls)
        self.layout_count=layout.size
        self.rig_count=rig.size
        if rig.ndim not in (1,2) or not rig.size:raise ValueError("invalid native camera array")
        self.camera_count=1 if rig.ndim==1 else len(rig)
        self.rig_size=rig.size//self.camera_count
        self._camera_ids=np.zeros(self.count,np.int32) if camera_ids is None else np.asarray(camera_ids)
        if (self._camera_ids.shape!=(self.count,) or self._camera_ids.dtype.kind not in "iu"
                or np.any(self._camera_ids<0) or np.any(self._camera_ids>=self.camera_count)):
            raise ValueError("One declared physical camera index is required per source row")
        self._camera_ids=np.ascontiguousarray(self._camera_ids,np.int32)
        self._motion_first=float(first_control_time)
        self._motion_dt=float(control_dt)
        self._frame_time_bounds=(float(times.min()),float(times.max()))
        self._frame_times=np.unique(times)
        self._frame_times_by_camera=[times[self._camera_ids==i] for i in range(self.camera_count)]
        self.diagnostic_download_bytes=0
        if codes.shape!=(self.count,) or stations.shape!=(self.count,) or times.shape!=(self.count,) or observed.shape!=(self.count,2):
            raise ValueError("G1 row arrays must preserve one-to-one binding")
        if board.shape!=(6,) or (rig.ndim not in (1,2) or self.rig_size not in (38,43) or (rig.ndim==2 and rig.shape[1]!=self.rig_size)) or layout.ndim!=1 or controls.ndim!=2 or controls.shape[1]!=3:
            raise ValueError("invalid native model or camera shape")
        self._handle=c.c_void_p()
        metrics=np.zeros(10,np.float64)
        self._check(self._lib.rs_g1_create(board.ctypes.data_as(D),board.size,
            layout.ctypes.data_as(D),layout.size,int(options),rig.ctypes.data_as(D),self.rig_size,
            controls.ctypes.data_as(D),len(controls),float(first_control_time),float(control_dt),int(image_height),
            codes.ctypes.data_as(I),stations.ctypes.data_as(D),times.ctypes.data_as(D),
            observed.ctypes.data_as(D),self.count,c.byref(self._handle),metrics.ctypes.data_as(D),metrics.size))
        if self.camera_count>1:
            try:
                fn=getattr(self._lib,"rs_gpu_set_camera_groups",None)
                if fn is None:raise RuntimeError("Native library has no multi-camera resident capability")
                fn.argtypes=[c.c_void_p,D,N,I,N,D];fn.restype=c.c_int
                extra=np.zeros(1,np.float64)
                self._check(fn(self._handle,rig.ctypes.data_as(D),rig.size,self._camera_ids.ctypes.data_as(I),self.count,extra.ctypes.data_as(D)))
                metrics[3]+=extra[0];metrics[5]-=extra[0]
            except BaseException:
                self._lib.rs_g1_destroy(self._handle);self._handle=c.c_void_p();raise
        names=("wall_ms","context_init_ms","h2d_event_ms","owned_device_bytes","free_before_alloc_bytes",
               "free_after_alloc_bytes","total_device_bytes","device_index","compute_major","compute_minor")
        self.creation=dict(zip(names,metrics.tolist()))
        # Reused host outputs exclude repeated host allocation from measured calls.
        self._out=np.empty((self.count,11),np.float64)
        self._status=np.empty(self.count,np.int32)
        # Retain only the immutable row identities needed to bind scene support.
        self._codes=codes.copy()
        self._initial_stations=stations.copy()

    def configure_geometry(self, paths):
        """Attach frozen CURRENT native path definitions before linearization."""
        meta=[];values=[];supports=[]
        for path in paths:
            kind=int(path["kind"]);parts=[]
            if kind==7:
                support=(-np.inf,np.inf)
            else:
                support=np.asarray(path.get("support",path.get("finite_support",(0.,1.))),np.float64)
                if (support.shape!=(2,) or not np.isfinite(support).all()
                        or not 0.<=support[0]<=support[1]<=1.):
                    raise ValueError("invalid finite native path support")
            supports.append(support)
            for key in ("parameters","vertices","knots"):
                data=np.asarray(path.get(key,[]),dtype=np.float64).ravel()
                parts.extend((len(values),data.size));values.extend(data.tolist())
            meta.append([kind,*parts])
        records=np.ascontiguousarray(meta,np.int32);data=np.ascontiguousarray(values,np.float64)
        if records.ndim!=2 or records.shape[1]!=7:raise ValueError("native geometry plan shape")
        supports=np.asarray(supports,np.float64)
        if np.any(self._codes<0) or np.any(self._codes>=len(paths)):
            raise ValueError("row references missing native path")
        row_support=supports[self._codes]
        if (not np.isfinite(self._initial_stations).all()
                or np.any(self._initial_stations<row_support[:,0])
                or np.any(self._initial_stations>row_support[:,1])):
            raise ValueError("original station outside selected native scene support")
        fn=self._lib.rs_gpu_attach_paths
        fn.argtypes=[c.c_void_p,I,N,D,N,D];fn.restype=c.c_int
        size=np.zeros(1,np.float64)
        self._check(fn(self._handle,records.ctypes.data_as(I),records.size,data.ctypes.data_as(D),data.size,size.ctypes.data_as(D)))
        self._path_supports=supports
        self.geometry_device_bytes=int(size[0])
        return {"paths":len(paths),"owned_device_bytes":self.geometry_device_bytes}

    def _check(self,status):
        if status:
            detail=self._lib.rs_g1_last_error().decode("utf-8",errors="replace")
            raise RuntimeError(f"native GPU status {status}: {detail}")

    def evaluate(self, backend):
        if backend not in ("cpu","cuda"):
            raise ValueError("backend must be cpu or cuda")
        if not self._handle.value:
            raise RuntimeError("G1 resident problem is closed")
        metrics=np.zeros(3,np.float64)
        self._check(self._lib.rs_g1_evaluate(self._handle,int(backend=="cuda"),
            self._out.ctypes.data_as(D),self._out.size,self._status.ctypes.data_as(I),self.count,
            metrics.ctypes.data_as(D),metrics.size))
        if backend=="cuda":self.diagnostic_download_bytes+=self._out.nbytes+self._status.nbytes
        return {"values":self._out.copy(),"status":self._status.copy(),
                "timings":dict(zip(("wall_ms","kernel_event_ms","d2h_event_ms"),metrics.tolist()))}


    def configure_profiles(self, *, row_ids, numerator_u, numerator_v, denominator, alpha_bounds, station_bounds, initial_alpha, mode="exact"):
        """Attach immutable source rational cubics; query alpha is not a fitted variable."""
        ids=np.asarray(row_ids)
        if ids.shape!=(self.count,) or ids.dtype.kind not in "iu":raise ValueError("one profile identity per source row is required")
        ids=np.ascontiguousarray(ids,np.int32)
        a,b,d=[np.asarray(v,np.float64) for v in (numerator_u,numerator_v,denominator)]
        count=len(a)
        if a.shape!=(count,4) or b.shape!=a.shape or d.shape!=a.shape:raise ValueError("rational cubic coefficient shape")
        ab=np.asarray(alpha_bounds,np.float64);sb=np.asarray(station_bounds,np.float64);seed=np.asarray(initial_alpha,np.float64)
        if ab.shape!=(count,2) or sb.shape!=(count,2) or seed.shape!=(count,):raise ValueError("profile finite domain shape")
        data=np.ascontiguousarray(np.c_[a,b,d,ab,sb,seed],np.float64)
        fn=self._lib.rs_gpu_attach_profiles;fn.argtypes=[c.c_void_p,I,N,D,N];fn.restype=c.c_int
        self._check(fn(self._handle,ids.ctypes.data_as(I),ids.size,data.ctypes.data_as(D),data.size))
        if mode not in ("exact","feasibility_restoration"):raise ValueError("unknown profile factor mode")
        fn=self._lib.rs_gpu_set_profile_mode;fn.argtypes=[c.c_void_p,c.c_int];fn.restype=c.c_int
        self._check(fn(self._handle,int(mode=="exact")))
        self.profile_count=count;self.profile_mode=mode

    def resolve_profile_roots(self, backend="cuda"):
        if backend not in ("cpu","cuda"):raise ValueError("backend must be cpu or cuda")
        roots=np.empty((self.count,2),np.float64);status=np.empty(self.count,np.int32)
        fn=self._lib.rs_gpu_profile_roots;fn.argtypes=[c.c_void_p,c.c_int,D,N,I,N];fn.restype=c.c_int
        self._check(fn(self._handle,int(backend=="cuda"),roots.ctypes.data_as(D),roots.size,status.ctypes.data_as(I),status.size))
        self.diagnostic_download_bytes+=roots.nbytes+status.nbytes
        return {"station":roots[:,0],"alpha":roots[:,1],"status":status}

    def configure_scene_parameters(self, *, values, records, data, directions, lower=None, upper=None, charts=None):
        """Bind current scene state and immutable path/block references, then activate directions."""
        values,data,directions=[np.ascontiguousarray(a,np.float64) for a in (values,data,directions)]
        records=np.ascontiguousarray(records,np.int32)
        if values.ndim!=1 or data.ndim!=1 or records.ndim!=2 or records.shape[1]!=12:
            raise ValueError("common native scene shape")
        if directions.ndim!=2 or directions.shape[0]!=len(values) or directions.shape[1]<=0:
            raise ValueError("scene directions must map current common values to retained columns")
        lower=np.full(values.size,-np.inf) if lower is None else np.ascontiguousarray(lower,np.float64)
        upper=np.full(values.size,np.inf) if upper is None else np.ascontiguousarray(upper,np.float64)
        if lower.shape!=values.shape or upper.shape!=values.shape:raise ValueError("scene physical bound shape")
        fn=self._lib.rs_gpu_attach_scene
        fn.argtypes=[c.c_void_p,D,N,I,N,D,N,c.c_int,D,N,D,D];fn.restype=c.c_int
        self._check(fn(self._handle,values.ctypes.data_as(D),values.size,records.ctypes.data_as(I),records.size,
            data.ctypes.data_as(D),data.size,directions.shape[1],directions.ctypes.data_as(D),directions.size,
            lower.ctypes.data_as(D),upper.ctypes.data_as(D)))
        if charts is not None:
            charts=np.ascontiguousarray(charts,np.int32)
            if charts.shape!=values.shape:raise ValueError("scene chart shape")
            fn=self._lib.rs_gpu_set_scene_chart;fn.argtypes=[c.c_void_p,I,N];fn.restype=c.c_int
            self._check(fn(self._handle,charts.ctypes.data_as(I),charts.size))
        self.scene_parameter_count=values.size
        return {"scene_parameters":values.size,"paths":len(records)}

    def export_scene_parameters(self, *, trial=False):
        values=np.empty(self.scene_parameter_count,np.float64);fn=self._lib.rs_gpu_export_scene
        fn.argtypes=[c.c_void_p,c.c_int,D,N];fn.restype=c.c_int
        self._check(fn(self._handle,int(trial),values.ctypes.data_as(D),values.size))
        self.diagnostic_download_bytes+=values.nbytes
        return values

    def configure_mount_constraints(self, *, side, camera_index=None):
        """Require the confirmed fixed pivot outside the rink, above ice, with a nonzero lever."""
        sides={"outside":1,"negative_y":2,"positive_y":3,"negative_x_end":4,"positive_x_end":5}
        if side not in sides:raise ValueError("unknown physical installation side")
        if camera_index is None:
            fn=self._lib.rs_gpu_set_mount_constraints;fn.argtypes=[c.c_void_p,c.c_int];fn.restype=c.c_int
            self._check(fn(self._handle,sides[side]))
        else:
            fn=self._lib.rs_gpu_set_mount_constraints_for_camera;fn.argtypes=[c.c_void_p,c.c_int,c.c_int];fn.restype=c.c_int
            self._check(fn(self._handle,int(camera_index),sides[side]))

    def configure_camera_envelope(self, *, forward_bounds, pupil_bounds, max_pan_tilt,
                                  max_tilt_deviation, elevation_bounds, roll_bounds,
                                  max_speed=None, max_acceleration=None, max_jerk=None,
                                  frame_time_bounds=None,frame_time_intervals=None,camera_index=None):
        """Constrain the complete centre offset and continuous physical attitude.

        Distances use the current scene gauge; angles are radians. Unknown rig
        and motion coordinates stay in the same retained system. Checks cover
        every allocated cubic cell by default. Explicit frame_time_bounds
        select a shot interval containing every source exposure; each trial's
        own readout expands it to all rows. Complete intersecting cubic cells
        are certified, including up to one extra storage cell at each end.
        frame_time_intervals selects an ordered union of disjoint shot spans;
        it leaves edit gaps outside the physical trajectory certificate.
        Optional derivative maxima are triples in pan/tilt/log(f) order, with
        time measured in seconds. None/+inf leaves a limit unset. No limits are
        inferred from data, no source residuals are changed or weighted.
        """
        limits=np.ascontiguousarray([*forward_bounds,*pupil_bounds,max_pan_tilt,
            max_tilt_deviation,*elevation_bounds,*roll_bounds],np.float64)
        if limits.shape!=(10,):raise ValueError("camera envelope requires ten limits")
        if frame_time_bounds is not None and frame_time_intervals is not None:
            raise ValueError('Specify one interval or disjoint shot intervals, not both')
        windows=frame_time_intervals
        if frame_time_bounds is not None:windows=[frame_time_bounds]
        motion=(max_speed,max_acceleration,max_jerk)
        if windows is not None or any(value is not None for value in motion):
            rows=[np.full(3,np.inf) if value is None else np.asarray(value,np.float64) for value in motion]
            if any(row.shape!=(3,) or np.any(~(row>0)) for row in rows):
                raise ValueError("motion maxima require positive pan/tilt/log(f) triples; +inf leaves a limit unset")
            limits=np.concatenate((limits,*rows))
        if windows is not None:
            span=np.asarray(windows,np.float64)
            if (span.ndim!=2 or span.shape[1]!=2 or not len(span) or not np.isfinite(span).all()
                    or np.any(span[:,1]<span[:,0]) or np.any(span[1:,0]<=span[:-1,1])):
                raise ValueError('Camera envelope intervals must be finite, ordered and disjoint')
            frame_times=self._frame_times if camera_index is None else np.unique(self._frame_times_by_camera[int(camera_index)])
            covered=np.zeros(len(frame_times),bool)
            for lo,hi in span:covered|=(frame_times>=lo)&(frame_times<=hi)
            if not covered.all():
                raise ValueError('Camera envelope interval must contain every source exposure')
            limits=np.r_[limits,((span-self._motion_first)/self._motion_dt).ravel()]
        if camera_index is None:
            fn=self._lib.rs_gpu_set_camera_envelope;fn.argtypes=[c.c_void_p,D,N];fn.restype=c.c_int
            self._check(fn(self._handle,limits.ctypes.data_as(D),limits.size))
        else:
            fn=self._lib.rs_gpu_set_camera_envelope_for_camera;fn.argtypes=[c.c_void_p,c.c_int,D,N];fn.restype=c.c_int
            self._check(fn(self._handle,int(camera_index),limits.ctypes.data_as(D),limits.size))
        self.camera_envelope_limits=limits

    def configure_projection_time_support(self, *, half_seconds):
        """Declare a finite row-time interval for a sparse fixed-rig plan.

        This is computational time support, not an image rectangle or point
        tolerance. By default the full native spline support is available.
        Configure before linearization. Out-of-support roots fail explicitly.
        Joint camera plans already reserve max_readout / 2 on each side.
        """
        fn=getattr(self._lib, 'rs_gpu_set_projection_time_support', None)
        if fn is None:
            raise ValueError('Native library lacks explicit projection time support')
        fn.argtypes=[c.c_void_p,c.c_double];fn.restype=c.c_int
        self._check(fn(self._handle,float(half_seconds)))

    def configure_camera_parameters(self, *, directions, max_readout, lower=None, upper=None):
        """Enable physical camera unknowns in the SAME native retained system.

        Rows: R0 rotation (3), pan tangent (2), tilt axis (1), pivot (3),
        lever (3), axis spacing (1), pupil coefficients (2), optics (12).
        Bounds are on the 38 physical rig components, never inferred from data.
        max_readout reserves spline support for every admissible candidate and
        for finite off-image predictions within that declared time interval.
        """
        directions=np.ascontiguousarray(directions,np.float64)
        if directions.ndim!=2 or directions.shape[0]!=self.camera_count*(31 if self.rig_size==43 else 27) or directions.shape[1]<=0:
            raise ValueError("physical camera directions must have shape (27, retained_columns)")
        lower=np.full(self.rig_count,-np.inf) if lower is None else np.ascontiguousarray(lower,np.float64)
        upper=np.full(self.rig_count,np.inf) if upper is None else np.ascontiguousarray(upper,np.float64)
        if lower.shape!=(self.rig_count,) or upper.shape!=(self.rig_count,):raise ValueError("physical rig bound shape")
        fn=self._lib.rs_gpu_configure_camera
        fn.argtypes=[c.c_void_p,c.c_int,D,N,D,D,N,c.c_double];fn.restype=c.c_int
        self._check(fn(self._handle,directions.shape[1],directions.ctypes.data_as(D),directions.size,
            lower.ctypes.data_as(D),upper.ctypes.data_as(D),self.rig_count,float(max_readout)))
        self.camera_parameter_columns=tuple(np.flatnonzero(np.any(directions!=0,axis=0)).tolist())
        return {"chart_size":directions.shape[0],"active_columns":self.camera_parameter_columns,"max_readout":float(max_readout)}

    def export_model(self, *, trial=False):
        """Download accepted (or last trial) native geometry and complete rig."""
        board=np.empty(6,np.float64);layout=np.empty(self.layout_count,np.float64);rig=np.empty(self.rig_count,np.float64)
        fn=self._lib.rs_gpu_export_model
        fn.argtypes=[c.c_void_p,c.c_int,D,N,D,N,D,N];fn.restype=c.c_int
        self._check(fn(self._handle,int(trial),board.ctypes.data_as(D),board.size,
            layout.ctypes.data_as(D),layout.size,rig.ctypes.data_as(D),rig.size))
        self.diagnostic_download_bytes+=board.nbytes+layout.nbytes+rig.nbytes
        return {"board":board,"layout":layout,"rig":rig if self.camera_count==1 else rig.reshape(self.camera_count,self.rig_size)}

    def configure_pixel_linearization(self, *, directions, local_ids, local_scales):
        """Unweighted raw UV distances; no statistical or pixel tolerance input.

        local_scales changes numerical coordinates only, never the residual.
        A contour station identifies a point on its current native geometry.
        """
        n=self.count
        return self.configure_linearization(directions=directions,local_ids=local_ids,local_scales=local_scales,
            group_offsets=np.arange(n+1,dtype=np.int32),members=np.arange(n,dtype=np.int32),
            ranks=np.full(n,2,np.int32),weight_offsets=np.zeros(n+1,np.int32),weights=np.zeros(0),
            huber_delta=0.,_pixel_residuals=True)

    def configure_linearization(self, *, directions, local_ids, local_scales, group_offsets,
                                members, ranks, weight_offsets, weights, huber_delta,_pixel_residuals=False):
        """Attach activity and complete original whitening; does not update state."""
        if not self._handle.value:
            raise RuntimeError("resident problem is closed")
        if not hasattr(self._lib,"rs_g2_attach"):
            raise RuntimeError("native library has no G2 capability")
        signatures={
            "rs_g2_attach":[c.c_void_p,c.c_int,D,N,I,N,D,N,I,c.c_int,I,N,I,I,D,N,c.c_double,D,N],
            "rs_g2_layout":[c.c_void_p,I,N,I,N,I,N],
            "rs_g2_linearize":[c.c_void_p,c.c_int,D,N,D,N],
            "rs_g2_export_blocks":[c.c_void_p,c.c_int,D,N,D,N,D,N],
            "rs_g2_probe_rows":[c.c_void_p,c.c_int,I,N,D,N],
            "rs_g2_probe_group":[c.c_void_p,c.c_int,c.c_int,D,N,D,N]}
        for name,signature in signatures.items():
            fn=getattr(self._lib,name);fn.argtypes=signature;fn.restype=c.c_int
        if _pixel_residuals:
            fn=self._lib.rs_g2_attach_pixels
            fn.argtypes=[c.c_void_p,c.c_int,D,N,I,N,D,N,D,N];fn.restype=c.c_int
        def ints(value):
            a=np.asarray(value)
            if a.ndim!=1 or a.dtype.kind not in "iu" or np.any(a<np.iinfo(np.int32).min) or np.any(a>np.iinfo(np.int32).max):
                raise ValueError("G2 indices require native int32 vectors")
            return np.ascontiguousarray(a,np.int32)
        directions=np.ascontiguousarray(directions,np.float64)
        local_scales=np.ascontiguousarray(local_scales,np.float64)
        weights=np.ascontiguousarray(weights,np.float64)
        local_ids,group_offsets,members,ranks,weight_offsets=map(ints,(local_ids,group_offsets,members,ranks,weight_offsets))
        if directions.ndim!=3 or directions.shape[:2]!=(self.control_count,3) or directions.shape[2]<=0:
            raise ValueError("G2 directions must map the existing controls")
        if local_scales.ndim!=1 or weights.ndim!=1 or local_ids.shape!=(self.count,) or members.shape!=(self.count,):
            raise ValueError("G2 row binding or local scales shape")
        if len(group_offsets)!=len(ranks)+1 or len(weight_offsets)!=len(ranks)+1:
            raise ValueError("G2 complete group offset shape")
        info=np.zeros(12,np.float64)
        if _pixel_residuals:
            self._check(self._lib.rs_g2_attach_pixels(self._handle,directions.shape[2],directions.ctypes.data_as(D),directions.size,
                local_ids.ctypes.data_as(I),local_ids.size,local_scales.ctypes.data_as(D),local_scales.size,
                info.ctypes.data_as(D),info.size))
        else:
            self._check(self._lib.rs_g2_attach(self._handle,directions.shape[2],directions.ctypes.data_as(D),directions.size,
                local_ids.ctypes.data_as(I),local_ids.size,local_scales.ctypes.data_as(D),local_scales.size,
                group_offsets.ctypes.data_as(I),len(ranks),members.ctypes.data_as(I),members.size,
                ranks.ctypes.data_as(I),weight_offsets.ctypes.data_as(I),weights.ctypes.data_as(D),weights.size,
                float(huber_delta),info.ctypes.data_as(D),info.size))
        self.pixel_residuals=bool(_pixel_residuals)
        names=("q","local_count","group_count","component_count","white_value_count","H_value_count","g_value_count",
               "device_bytes","setup_wall_ms","setup_event_span_ms","max_local_component","whitened_rank")
        self.linearization_info=dict(zip(names,info.tolist()))
        self.linearization_info['objective']='unweighted_squared_pixel_distance' if _pixel_residuals else 'correlated_rank_normalized_huber'
        self._q=int(info[0])
        sparse=hasattr(self._lib,"rs_g2_sparse_info")
        index_type=np.int64 if sparse else np.int32
        groups=np.empty((int(info[2]),6),index_type)
        components=np.empty((int(info[3]),7),index_type)
        locals_=np.empty(int(info[1]),np.int32)
        if sparse:
            fn=self._lib.rs_g2_layout64
            fn.argtypes=[c.c_void_p,L,N,L,N,I,N];fn.restype=c.c_int
            self._check(fn(self._handle,groups.ctypes.data_as(L),groups.size,
                components.ctypes.data_as(L),components.size,locals_.ctypes.data_as(I),locals_.size))
        else:
            self._check(self._lib.rs_g2_layout(self._handle,groups.ctypes.data_as(I),groups.size,
                components.ctypes.data_as(I),components.size,locals_.ctypes.data_as(I),locals_.size))
        self.layout={"groups":groups,"components":components,"locals":locals_}
        self._local_ids=local_ids.copy()
        if sparse:
            values=np.zeros(6,np.int64);fn=self._lib.rs_g2_sparse_info
            fn.argtypes=[c.c_void_p,L,N];fn.restype=c.c_int
            self._check(fn(self._handle,values.ctypes.data_as(L),values.size))
            self.sparsity=dict(zip(("retained_nnz","column_map_entries","scatter_entries",
                "jacobian_values","max_row_q","max_component_q"),map(int,values)))
            ccols=np.empty((len(components),2),np.int64)
            columns=np.empty(int(values[1]),np.int32)
            offsets=np.empty(self._q+1,np.int32);indices=np.empty(int(values[0]),np.int32)
            fn=self._lib.rs_g2_sparse_layout
            fn.argtypes=[c.c_void_p,L,N,I,N,I,N,I,N];fn.restype=c.c_int
            self._check(fn(self._handle,ccols.ctypes.data_as(L),ccols.size,
                columns.ctypes.data_as(I),columns.size,offsets.ctypes.data_as(I),offsets.size,
                indices.ctypes.data_as(I),indices.size))
            self.layout.update(component_columns=ccols,columns=columns,
                csr_offsets=offsets,csr_columns=indices)
            self.linearization_info.update(self.sparsity)
        return self.linearization_info

    def _g2_backend(self,backend):
        if backend not in ("cpu","cuda"):
            raise ValueError("backend must be cpu or cuda")
        if not self._handle.value:
            raise RuntimeError("resident problem is closed")
        if not hasattr(self,"linearization_info"):
            raise RuntimeError("G2 activity/noise plan is not configured")
        return int(backend=="cuda")

    def linearize(self,backend="cuda"):
        summary=np.zeros(4,np.float64);timings=np.zeros(8,np.float64)
        self._check(self._lib.rs_g2_linearize(self._handle,self._g2_backend(backend),
            summary.ctypes.data_as(D),summary.size,timings.ctypes.data_as(D),timings.size))
        return {"summary":dict(zip(("failed_rows","whitened_quadratic_cost","huber_cost","downweighted_groups"),summary.tolist())),
                "timings":dict(zip(("wall_ms","raw_jacobian_event_ms","whitening_event_ms","loss_event_ms",
                    "block_reduce_event_ms","summary_d2h_event_ms","owned_linearization_device_bytes","summary_d2h_bytes"),timings.tolist()))}

    def export_blocks(self,backend="cuda"):
        """Explicit engineering/checkpoint export. Never called by linearize."""
        h=np.empty(int(self.linearization_info["H_value_count"]),np.float64)
        g=np.empty(int(self.linearization_info["g_value_count"]),np.float64)
        shared=np.empty(self._q*self._q+self._q,np.float64)
        self._check(self._lib.rs_g2_export_blocks(self._handle,self._g2_backend(backend),
            h.ctypes.data_as(D),h.size,g.ctypes.data_as(D),g.size,shared.ctypes.data_as(D),shared.size))
        if backend=="cuda":self.diagnostic_download_bytes+=h.nbytes+g.nbytes+shared.nbytes
        return {"H":h,"g":g,"shared_H":shared[:self._q*self._q].reshape(self._q,self._q),"shared_g":shared[-self._q:]}

    def export_sparse_blocks(self,backend="cuda"):
        """Explicit packed component H/g and retained upper CSR diagnostics.

        Each component's first columns are layout['columns'] at its
        component_columns offset/count; remaining columns are its local IDs.
        """
        self._g2_backend(backend)
        h=np.empty(int(self.linearization_info["H_value_count"]),np.float64)
        g=np.empty(int(self.linearization_info["g_value_count"]),np.float64)
        nnz=self.sparsity["retained_nnz"];shared=np.empty(nnz+self._q,np.float64)
        fn=self._lib.rs_g2_export_sparse_blocks
        fn.argtypes=[c.c_void_p,c.c_int,D,N,D,N,D,N];fn.restype=c.c_int
        self._check(fn(self._handle,int(backend=="cuda"),h.ctypes.data_as(D),h.size,
            g.ctypes.data_as(D),g.size,shared.ctypes.data_as(D),shared.size))
        if backend=="cuda":self.diagnostic_download_bytes+=h.nbytes+g.nbytes+shared.nbytes
        return {"H":h,"g":g,"shared_values":shared[:nnz],"shared_g":shared[nnz:],
            "csr_offsets":self.layout["csr_offsets"],"csr_columns":self.layout["csr_columns"]}

    def probe_rows(self,rows,backend="cuda"):
        """Explicit local-J sample; caller selects source rows for diagnostics."""
        raw=np.asarray(rows)
        if raw.ndim!=1 or raw.dtype.kind not in "iu" or np.any(raw<0) or np.any(raw>=self.count):
            raise ValueError("row probe indices outside the resident problem")
        rows=np.ascontiguousarray(raw,np.int32)
        jac=np.empty((len(rows),2,self._q+1),np.float64)
        self._check(self._lib.rs_g2_probe_rows(self._handle,self._g2_backend(backend),
            rows.ctypes.data_as(I),len(rows),jac.ctypes.data_as(D),jac.size))
        if backend=="cuda":self.diagnostic_download_bytes+=jac.nbytes
        return jac

    def probe_group(self,group,backend="cuda"):
        group=int(group)
        if group<0 or group>=len(self.layout["groups"]):
            raise ValueError("group probe outside the resident plan")
        record=self.layout["groups"][group];dim=int(self.layout["components"][record[4],4])
        white=np.empty((int(record[2]),dim+1),np.float64);loss=np.empty(3,np.float64)
        self._check(self._lib.rs_g2_probe_group(self._handle,self._g2_backend(backend),group,
            white.ctypes.data_as(D),white.size,loss.ctypes.data_as(D),loss.size))
        if backend=="cuda":self.diagnostic_download_bytes+=white.nbytes+loss.nbytes
        return {"white_r":white[:,0],"white_J":white[:,1:],"loss":loss}

    def configure_solver(self, *, lower, upper, device_budget_bytes, host_budget_bytes,
                         initial_lambda_relative=1e-3, initial_lambda=0.0,
                         acceptance_eta=1e-4, linear_tolerance=1e-8, damping_metric="identity",
                         profile_trial_stations=False):
        """Attach native one-candidate LM. No Python numerical solve or update."""
        self._g2_backend("cuda")
        if not hasattr(self._lib,"rs_g3_attach"):
            raise RuntimeError("native library was built without project cuDSS/G3")
        signatures={
            "rs_g3_attach":[c.c_void_p,D,D,N,D,N,D,N],
            "rs_g3_step":[c.c_void_p,D,N,D,N],
            "rs_g3_export_state":[c.c_void_p,c.c_int,D,N,D,N],
            "rs_g3_export_system":[c.c_void_p,D,N,I,N]}
        for name,sig in signatures.items():
            fn=getattr(self._lib,name);fn.argtypes=sig;fn.restype=c.c_int
        lower=np.ascontiguousarray(lower,np.float64);upper=np.ascontiguousarray(upper,np.float64)
        nl=int(self.linearization_info["local_count"])
        if lower.shape!=(nl,) or upper.shape!=(nl,):raise ValueError("station support shape")
        if hasattr(self,"_path_supports"):
            active=self._local_ids>=0
            ids=self._local_ids[active]
            support=self._path_supports[self._codes[active]]
            if np.any(lower[ids]<support[:,0]) or np.any(upper[ids]>support[:,1]):
                raise ValueError("solver station bounds extend outside selected native scene support")
        options=np.array([initial_lambda_relative,initial_lambda,acceptance_eta,linear_tolerance,
                          device_budget_bytes,host_budget_bytes],np.float64)
        if damping_metric not in ("identity","jacobi","jacobi_image_station","jacobi_image"):raise ValueError("unknown LM damping metric")
        if not isinstance(profile_trial_stations,bool):raise ValueError("profile_trial_stations must be boolean")
        if profile_trial_stations and self.linearization_info.get('objective')!='unweighted_squared_pixel_distance':
            raise ValueError("trial station profiling requires independent raw pixel factors")
        if damping_metric!="identity" or profile_trial_stations:
            options=np.r_[options,{"identity":0.,"jacobi":1.,"jacobi_image_station":2.,"jacobi_image":3.}[damping_metric]]
        if profile_trial_stations:options=np.r_[options,1.]
        info=np.zeros(8,np.float64)
        self._check(self._lib.rs_g3_attach(self._handle,lower.ctypes.data_as(D),upper.ctypes.data_as(D),nl,
            options.ctypes.data_as(D),options.size,info.ctypes.data_as(D),info.size))
        self.solver_info=dict(zip(("setup_wall_ms","owned_device_bytes","max_local_dimension","reduced_CSR_nnz",
                                  "cudss_major","cudss_minor","cudss_patch","local_matrix_values"),info.tolist()))
        self.solver_info["damping_metric"]=damping_metric
        self.solver_info["profile_trial_stations"]=profile_trial_stations
        return self.solver_info

    def evaluate_errors(self):
        """All original pixel errors with an eight-byte-per-row CUDA download."""
        if not self._handle.value:raise RuntimeError("resident problem is closed")
        fn=getattr(self._lib,'rs_g1_evaluate_errors',None)
        if fn is None:
            evaluated=self.evaluate('cuda');errors=np.linalg.norm(evaluated['values'][:,9:11],axis=1)
            errors[evaluated['status']!=0]=np.inf
            return errors
        fn.argtypes=[c.c_void_p,D,N];fn.restype=c.c_int;errors=np.empty(self.count)
        self._check(fn(self._handle,errors.ctypes.data_as(D),errors.size));self.diagnostic_download_bytes+=errors.nbytes
        return errors

    def evaluate_rows(self, rows):
        """Evaluate selected source rows using the same current native model."""
        if not self._handle.value:raise RuntimeError("resident problem is closed")
        raw=np.asarray(rows)
        if raw.ndim!=1 or raw.dtype.kind not in 'iu' or np.any(raw<0) or np.any(raw>=self.count):
            raise ValueError('evaluation rows outside the resident problem')
        rows=np.ascontiguousarray(raw,np.int32);fn=getattr(self._lib,'rs_g1_evaluate_rows',None)
        if fn is None:
            evaluated=self.evaluate('cuda')
            return dict(rows=rows,values=evaluated['values'][rows].copy(),status=evaluated['status'][rows].copy())
        values=np.empty((len(rows),11));status=np.empty(len(rows),np.int32)
        fn.argtypes=[c.c_void_p,I,N,D,N,I,N];fn.restype=c.c_int
        self._check(fn(self._handle,rows.ctypes.data_as(I),len(rows),values.ctypes.data_as(D),values.size,status.ctypes.data_as(I),status.size))
        self.diagnostic_download_bytes+=values.nbytes+status.nbytes
        return dict(rows=rows,values=values,status=status)

    def _objective_timings(self):
        fn=getattr(self._lib,'rs_g3_objective_timings',None)
        if fn is None:return {}
        fn.argtypes=[c.c_void_p,D,N];fn.restype=c.c_int;values=np.empty(6)
        self._check(fn(self._handle,values.ctypes.data_as(D),values.size))
        return dict(zip(('pixel_loss','rig_domain','print_domain','scene_domain','camera_envelope','reduce_and_decide'),values.tolist()))

    def refine_curve_stations(self, *, mode="global"):
        """Lower each ordinary point's exact raw pixel distance on its native curve."""
        if not getattr(self,'pixel_residuals',False) or not hasattr(self,'solver_info'):
            raise RuntimeError('independent raw pixel solver required for curve refinement')
        if mode not in ("global","local"):raise ValueError('station refinement mode must be global or local')
        info=np.zeros(6)
        if mode=="local":
            fn=getattr(self._lib,'rs_g3_refine_stations_ex',None)
            if fn is None:raise RuntimeError('native library lacks warm station refinement')
            fn.argtypes=[c.c_void_p,c.c_int,D,N];fn.restype=c.c_int
            self._check(fn(self._handle,0,info.ctypes.data_as(D),info.size))
        else:
            fn=self._lib.rs_g3_refine_stations;fn.argtypes=[c.c_void_p,D,N];fn.restype=c.c_int
            self._check(fn(self._handle,info.ctypes.data_as(D),info.size))
        self.diagnostic_download_bytes+=info.nbytes
        return dict(zip(('changed_rows','pixel_cost_decrease','failed_rows','maximum_station_change','wall_ms','native_curve_samples'),info.tolist()))

    def stationarity_diagnostics(self, *, tolerance=1e-7, include_constraints=False):
        """Read original gradients and declared constraint normals without a step."""
        fn=getattr(self._lib,'rs_g4_stationarity_diagnostics',None)
        if fn is None:raise RuntimeError('native library lacks current-state constraint diagnostics')
        fn.argtypes=[c.c_void_p,c.c_double,D,N];fn.restype=c.c_int;values=np.empty(12)
        self._check(fn(self._handle,tolerance,values.ctypes.data_as(D),values.size));self.diagnostic_download_bytes+=values.nbytes
        result=dict(zip(('raw_retained_gradient_inf','raw_station_gradient_inf','box_retained_gradient_inf','box_station_gradient_inf',
            'cone_primal_residual','cone_complementarity_residual','cone_stationarity_residual','physical_feasibility_violation',
            'objective','constraint_projected_gradient_inf','active_constraint_count','invalid'),values.tolist()))
        result.update(stationarity_scope='declared_active_physical_tangent_cone',full_nonlinear_kkt_checked=True,
            original_undamped_objective=True,normal_coordinates='retained solver chart; each constraint row has Euclidean unit norm')
        if include_constraints:
            count=int(values[10]);q=int(self.linearization_info['q']);faces=np.empty((count,24));jac=np.empty((count,q));gradient=np.empty((2,q))
            export=self._lib.rs_g4_stationarity_export;export.argtypes=[c.c_void_p,D,N,D,N,D,N];export.restype=c.c_int
            self._check(export(self._handle,faces.ctypes.data_as(D),faces.size,jac.ctypes.data_as(D),jac.size,gradient.ctypes.data_as(D),gradient.size))
            self.diagnostic_download_bytes+=faces.nbytes+jac.nbytes+gradient.nbytes
            result.update(constraint_records=faces,normalized_constraint_jacobian=jac,raw_retained_gradient=gradient[0],projected_retained_descent=gradient[1],
                constraint_record_fields=['family','index','other','side','equality','sign','station','margin','active_scale','normalized_multiplier','gradient_norm',
                    'camera_kind','camera_cell','camera_channel','camera_u','camera_sign','camera_minimum','camera_limit','nx','ny','nz','rate_u0','rate_u1','rate_sign_code'])
        return result

    def lm_step(self):
        """Exactly one candidate: native linearize/eliminate/solve/retract/decide."""
        self._g2_backend("cuda")
        if not hasattr(self,"solver_info"):raise RuntimeError("G3 solver not attached")
        summary=np.zeros(32,np.float64);timings=np.zeros(12,np.float64)
        self._check(self._lib.rs_g3_step(self._handle,summary.ctypes.data_as(D),summary.size,
                                       timings.ctypes.data_as(D),timings.size))
        record=self._step_record(summary,timings);record["trial_objective_components_ms"]=self._objective_timings();return record

    @staticmethod
    def _step_record(summary,timings):
        names=("lambda_used","old_cost","maximum_H_diagonal","feasible_alpha","reduced_relative_residual",
               "new_blocked_in_final_pass","blocked_station_count","nonfinite_step_count","trial_domain_failures",
               "first_failed_row","trial_cost","predicted_decrease","actual_decrease","gain_ratio","applied_step_norm",
               "accepted","reason_code","lambda_next","nu_next","bound_passes","analysis_count","trial_count",
               "accepted_count","explicit_application_d2h_bytes","solver_device_bytes",
               "cudss_device_stable_estimate","cudss_device_peak_estimate","cudss_host_stable_estimate",
               "cudss_host_peak_estimate","factor_nnz","acceptance_eta","applied_step_inf")
        metrics=dict(zip(names,summary.tolist()))
        metrics["accepted"]=bool(metrics["accepted"])
        metrics["reason"]={0:"accepted_decrease_and_gain",1:"insufficient_actual_decrease_or_gain",
                           2:"nonpositive_prediction_or_no_feasible_step",3:"trial_domain_failure",
                           4:"repeated_active_set_retry_with_damping",
                           5:"accepted_decrease_after_conditional_local_completion",
                           6:"accepted_decrease_after_bounded_joint_refinement"}[int(metrics["reason_code"])]
        return {"summary":metrics,"timings":dict(zip(("wall_ms","bounds_setup_event_ms","damp_local_eliminate_event_ms",
            "cudss_analysis_event_ms","cudss_factor_event_ms","cudss_solve_event_ms","backsub_bounds_event_ms",
            "retract_prediction_event_ms","trial_objective_event_ms","summary_d2h_event_ms","accept_commit_event_ms",
            "linearization_wall_ms"),timings.tolist()))}

    def solve(self, *, gradient_tolerance=1e-7, step_tolerance=1e-6,
              relative_cost_tolerance=1e-10, progress=None, numerical_progress=None):
        """Native LM; callback records telemetry and may return False to stop.

        A stopped handle retains accepted state and damping for the next solve.
        Callback failures request a safe stop before propagating the exception.
        """
        self._g2_backend("cuda")
        if not hasattr(self,"solver_info"):raise RuntimeError("G3 solver not attached")
        callback_type=c.CFUNCTYPE(None,D,D,c.c_double,c.c_double,c.c_void_p)
        version_fn=getattr(self._lib,"rs_g4_stationarity_version",None)
        version=0
        if version_fn is not None:
            version_fn.argtypes=[];version_fn.restype=c.c_int;version=int(version_fn())
        stationarity_scope=("declared_active_physical_tangent_cone" if version==2 else
                            "single_column_rig_boxes_and_station_boxes" if version==1 else
                            "legacy_raw_retained_gradient_and_station_boxes")
        callback_errors=[]
        callback_wall_ms=0.
        diagnostic_bytes_before=self.diagnostic_download_bytes
        def report(summary,timings,gradient,numerical_rejections,context):
            nonlocal callback_wall_ms
            if progress is None or callback_errors:return
            callback_started=time.perf_counter()
            try:
                record=self._step_record(np.ctypeslib.as_array(summary,shape=(32,)).copy(),
                                         np.ctypeslib.as_array(timings,shape=(12,)).copy())
                record["trial_objective_components_ms"]=self._objective_timings()
                record["projected_gradient_before"]=float(gradient) # legacy compatible alias
                record["stationarity_scope"]=stationarity_scope
                record["full_nonlinear_kkt_checked"]=version==2
                if version==2:record["constraint_projected_gradient_before"]=float(gradient)
                elif version==1:record["box_projected_gradient_before"]=float(gradient)
                else:record["legacy_gradient_diagnostic_before"]=float(gradient)
                record["numerical_factorization_rejections"]=int(numerical_rejections)
                if progress(record) is False:self.request_stop()
            except BaseException as exc:
                callback_errors.append(exc);self.request_stop()
            finally:
                callback_wall_ms+=1000*(time.perf_counter()-callback_started)
        numeric_type=c.CFUNCTYPE(None,c.c_int,c.c_double,c.c_double,c.c_void_p)
        def numeric_report(count,lam,nu,context):
            if numerical_progress is None or callback_errors:return
            try:
                if numerical_progress(dict(numerical_factorization_rejections=count, lambda_next=lam, nu_next=nu)) is False:
                    self.request_stop()
            except BaseException as exc:
                callback_errors.append(exc);self.request_stop()
        numeric_callback=numeric_type(numeric_report)
        set_numeric=getattr(self._lib,"rs_g4_set_numeric_progress",None)
        if numerical_progress is not None and set_numeric is None:
            raise RuntimeError("Native library lacks numerical retry telemetry")
        if set_numeric is not None:
            set_numeric.argtypes=[c.c_void_p,numeric_type,c.c_void_p];set_numeric.restype=c.c_int
            self._check(set_numeric(self._handle,numeric_callback,None))
        callback=callback_type(report)
        fn=self._lib.rs_g4_solve
        fn.argtypes=[c.c_void_p,D,N,callback_type,c.c_void_p,D,N];fn.restype=c.c_int
        tol=np.array([gradient_tolerance,step_tolerance,relative_cost_tolerance],np.float64)
        out=np.zeros(17,np.float64)
        try:
            self._check(fn(self._handle,tol.ctypes.data_as(D),tol.size,callback,None,out.ctypes.data_as(D),out.size))
        finally:
            if set_numeric is not None:self._check(set_numeric(self._handle,numeric_type(),None))
        result=dict(zip(("stop_code","trials","accepted","rejected","final_cost","projected_gradient_inf",
            "last_step_inf","last_relative_cost_change","lambda_next","nu_next","wall_ms","linearizations",
            "symbolic_analyses","explicit_application_d2h_bytes","linearization_wall_ms","candidate_wall_ms","numerical_factorization_rejections"),out.tolist()))
        result["stop_reason"]={0:"projected_gradient_small",1:"accepted_step_and_cost_small",
                               2:"floating_point_stagnation_without_convergence",3:"stopped_by_caller",
                               4:"cost_progress_stagnation_without_convergence"}[int(out[0])]
        result["numerically_converged"]=out[0] in (0,1)
        result["stationarity_version"]=version
        result["stationarity_scope"]=stationarity_scope
        result["full_nonlinear_kkt_checked"]=version==2
        result["gradient_diagnostic_note"]=(
            "projected_gradient_* are legacy aliases; coupled and nonlinear constraint "
            "normals are not projected. A large diagnostic does not prove feasible descent.")
        if version==2:
            result["gradient_diagnostic_note"]="Original undamped objective; current declared physical constraint Jacobian and station tangent cone. No global optimum or camera calibration claim."
            diagnostic=self.stationarity_diagnostics()
            # A caller stop can interrupt the controller before its cone
            # projection, leaving the raw gradient in that scratch slot.
            # The completed, read-only check describes the returned state.
            result["controller_reported_gradient_inf"]=result["projected_gradient_inf"]
            result["constraint_projected_gradient_inf"]=result["projected_gradient_inf"]=diagnostic["constraint_projected_gradient_inf"]
            result["final_cost"]=diagnostic.get("objective",result["final_cost"])
            result["stationarity_diagnostics"]=diagnostic
        elif version==1:result["box_projected_gradient_inf"]=result["projected_gradient_inf"]
        else:result["legacy_gradient_diagnostic_inf"]=result["projected_gradient_inf"]
        result["callback_wall_ms"]=callback_wall_ms
        result["diagnostic_d2h_bytes_during_solve"]=self.diagnostic_download_bytes-diagnostic_bytes_before
        result["instrumented_d2h_bytes_during_solve"]=(
            result["explicit_application_d2h_bytes"]+result["diagnostic_d2h_bytes_during_solve"])
        result["timing_scope"]={"wall_ms_includes_callbacks":True,
                                "candidate_wall_ms_excludes_callbacks":True,
                                "transfer_count_excludes_library_internal_transfers":True}
        result["tolerances"]={"gradient":gradient_tolerance,"step":step_tolerance,"relative_cost":relative_cost_tolerance}
        if callback_errors:raise callback_errors[0]
        return result

    def request_stop(self):
        """Thread-safe request; native returns at the next completed stream boundary.

        The next solve call on this live handle continues its accepted state.
        """
        if not self._handle.value:raise RuntimeError("resident problem is closed")
        fn=self._lib.rs_g4_request_stop
        fn.argtypes=[c.c_void_p];fn.restype=c.c_int
        self._check(fn(self._handle))

    def export_state(self, *, trial=False):
        """Explicit state copy for saved result/review. Does not change acceptance."""
        self._g2_backend("cuda")
        if not hasattr(self,"solver_info"):raise RuntimeError("G3 solver not attached")
        controls=np.empty((self.control_count,3),np.float64);stations=np.empty(self.count,np.float64)
        self._check(self._lib.rs_g3_export_state(self._handle,int(trial),controls.ctypes.data_as(D),controls.size,
                                               stations.ctypes.data_as(D),stations.size))
        self.diagnostic_download_bytes+=controls.nbytes+stations.nbytes
        model=self.export_model(trial=trial) if hasattr(self._lib,"rs_gpu_export_model") else {}
        if hasattr(self,"scene_parameter_count"):model["scene_values"]=self.export_scene_parameters(trial=trial)
        return {"controls":controls,"stations":stations,**model}

    def probe_step_changes(self):
        """Diagnostic immediate last step: before/trial UV and packed J times applied step."""
        fn=self._lib.rs_g3_probe_changes;fn.argtypes=[c.c_void_p,D,N];fn.restype=c.c_int
        data=np.empty((self.count,6),np.float64)
        self._check(fn(self._handle,data.ctypes.data_as(D),data.size));self.diagnostic_download_bytes+=data.nbytes
        return data

    def export_step_system(self, *, sparse=False):
        """Explicit engineering snapshot; not called by lm_step."""
        self._g2_backend("cuda")
        q=self._q;nl=int(self.linearization_info["local_count"]);n=q+nl
        matrix_size=self.sparsity["retained_nnz"] if sparse else q*q
        data=np.empty(matrix_size+q+2*n,np.float64);blocked=np.empty(nl,np.int32)
        fn=self._lib.rs_g3_export_sparse_system if sparse else self._lib.rs_g3_export_system
        fn.argtypes=[c.c_void_p,D,N,I,N];fn.restype=c.c_int
        self._check(fn(self._handle,data.ctypes.data_as(D),data.size,blocked.ctypes.data_as(I),blocked.size))
        self.diagnostic_download_bytes+=data.nbytes+blocked.nbytes
        result={"rhs":data[matrix_size:matrix_size+q],"step":data[matrix_size+q:matrix_size+q+n],
                "applied":data[-n:],"blocked":blocked}
        if sparse:
            result.update(csr_values=data[:matrix_size],csr_offsets=self.layout["csr_offsets"],
                          csr_columns=self.layout["csr_columns"])
        else:result["S"]=data[:matrix_size].reshape(q,q)
        return result

    def close(self):
        if self._handle.value:
            self._lib.rs_g1_destroy(self._handle)
            self._handle=c.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
