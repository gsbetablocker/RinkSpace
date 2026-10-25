"""Thin binding for the resident native GPU solver. No geometry math here."""
from __future__ import annotations
import ctypes as c
import numpy as np
from .native import D, I, N, NativeCore


class ResidentResidualProblem:
    """Current native geometry, physical motion camera, and raw UV factors.

    The native handle owns both CPU reference inputs and resident CUDA buffers.
    Native LM owns state updates and acceptance. Unsupported rows keep their status.
    """

    def __init__(self, core: NativeCore, *, board, layout, options, rig, controls,
                 first_control_time, control_dt, image_height, codes, stations, times, observed):
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
        self.diagnostic_download_bytes=0
        if codes.shape!=(self.count,) or stations.shape!=(self.count,) or times.shape!=(self.count,) or observed.shape!=(self.count,2):
            raise ValueError("G1 row arrays must preserve one-to-one binding")
        if board.shape!=(6,) or rig.shape!=(38,) or layout.ndim!=1 or controls.ndim!=2 or controls.shape[1]!=3:
            raise ValueError("invalid native model or camera shape")
        self._handle=c.c_void_p()
        metrics=np.zeros(10,np.float64)
        self._check(self._lib.rs_g1_create(board.ctypes.data_as(D),board.size,
            layout.ctypes.data_as(D),layout.size,int(options),rig.ctypes.data_as(D),rig.size,
            controls.ctypes.data_as(D),len(controls),float(first_control_time),float(control_dt),int(image_height),
            codes.ctypes.data_as(I),stations.ctypes.data_as(D),times.ctypes.data_as(D),
            observed.ctypes.data_as(D),self.count,c.byref(self._handle),metrics.ctypes.data_as(D),metrics.size))
        names=("wall_ms","context_init_ms","h2d_event_ms","owned_device_bytes","free_before_alloc_bytes",
               "free_after_alloc_bytes","total_device_bytes","device_index","compute_major","compute_minor")
        self.creation=dict(zip(names,metrics.tolist()))
        # Reused host outputs exclude repeated host allocation from measured calls.
        self._out=np.empty((self.count,11),np.float64)
        self._status=np.empty(self.count,np.int32)

    def configure_geometry(self, paths):
        """Attach frozen CURRENT native path definitions before linearization."""
        meta=[];values=[]
        for path in paths:
            kind=int(path["kind"]);parts=[]
            for key in ("parameters","vertices","knots"):
                data=np.asarray(path.get(key,[]),dtype=np.float64).ravel()
                parts.extend((len(values),data.size));values.extend(data.tolist())
            meta.append([kind,*parts])
        records=np.ascontiguousarray(meta,np.int32);data=np.ascontiguousarray(values,np.float64)
        if records.ndim!=2 or records.shape[1]!=7:raise ValueError("native geometry plan shape")
        fn=self._lib.rs_gpu_attach_paths
        fn.argtypes=[c.c_void_p,I,N,D,N,D];fn.restype=c.c_int
        size=np.zeros(1,np.float64)
        self._check(fn(self._handle,records.ctypes.data_as(I),records.size,data.ctypes.data_as(D),data.size,size.ctypes.data_as(D)))
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
        return {"values":self._out.copy(),"status":self._status.copy(),
                "timings":dict(zip(("wall_ms","kernel_event_ms","d2h_event_ms"),metrics.tolist()))}


    def configure_linearization(self, *, directions, local_ids, local_scales, group_offsets,
                                members, ranks, weight_offsets, weights, huber_delta):
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
        self._check(self._lib.rs_g2_attach(self._handle,directions.shape[2],directions.ctypes.data_as(D),directions.size,
            local_ids.ctypes.data_as(I),local_ids.size,local_scales.ctypes.data_as(D),local_scales.size,
            group_offsets.ctypes.data_as(I),len(ranks),members.ctypes.data_as(I),members.size,
            ranks.ctypes.data_as(I),weight_offsets.ctypes.data_as(I),weights.ctypes.data_as(D),weights.size,
            float(huber_delta),info.ctypes.data_as(D),info.size))
        names=("q","local_count","group_count","component_count","white_value_count","H_value_count","g_value_count",
               "device_bytes","setup_wall_ms","setup_event_span_ms","max_local_component","whitened_rank")
        self.linearization_info=dict(zip(names,info.tolist()))
        self._q=int(info[0])
        groups=np.empty((int(info[2]),6),np.int32)
        components=np.empty((int(info[3]),7),np.int32)
        locals_=np.empty(int(info[1]),np.int32)
        self._check(self._lib.rs_g2_layout(self._handle,groups.ctypes.data_as(I),groups.size,
            components.ctypes.data_as(I),components.size,locals_.ctypes.data_as(I),locals_.size))
        self.layout={"groups":groups,"components":components,"locals":locals_}
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
                         acceptance_eta=1e-4, linear_tolerance=1e-8):
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
        options=np.array([initial_lambda_relative,initial_lambda,acceptance_eta,linear_tolerance,
                          device_budget_bytes,host_budget_bytes],np.float64)
        info=np.zeros(8,np.float64)
        self._check(self._lib.rs_g3_attach(self._handle,lower.ctypes.data_as(D),upper.ctypes.data_as(D),nl,
            options.ctypes.data_as(D),options.size,info.ctypes.data_as(D),info.size))
        self.solver_info=dict(zip(("setup_wall_ms","owned_device_bytes","padded_local_dimension","reduced_CSR_nnz",
                                  "cudss_major","cudss_minor","cudss_patch","reserved"),info.tolist()))
        return self.solver_info

    def lm_step(self):
        """Exactly one candidate: native linearize/eliminate/solve/retract/decide."""
        self._g2_backend("cuda")
        if not hasattr(self,"solver_info"):raise RuntimeError("G3 solver not attached")
        summary=np.zeros(32,np.float64);timings=np.zeros(12,np.float64)
        self._check(self._lib.rs_g3_step(self._handle,summary.ctypes.data_as(D),summary.size,
                                       timings.ctypes.data_as(D),timings.size))
        return self._step_record(summary,timings)

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
                           2:"nonpositive_prediction_or_no_feasible_step",3:"trial_domain_failure"}[int(metrics["reason_code"])]
        return {"summary":metrics,"timings":dict(zip(("wall_ms","bounds_setup_event_ms","damp_local_eliminate_event_ms",
            "cudss_analysis_event_ms","cudss_factor_event_ms","cudss_solve_event_ms","backsub_bounds_event_ms",
            "retract_prediction_event_ms","trial_objective_event_ms","summary_d2h_event_ms","accept_commit_event_ms",
            "linearization_wall_ms"),timings.tolist()))}

    def solve(self, *, gradient_tolerance=1e-7, step_tolerance=1e-6,
              relative_cost_tolerance=1e-10, progress=None):
        """Native LM; callback records telemetry and may return False to stop.

        A stopped handle retains accepted state and damping for the next solve.
        Callback failures request a safe stop before propagating the exception.
        """
        self._g2_backend("cuda")
        if not hasattr(self,"solver_info"):raise RuntimeError("G3 solver not attached")
        callback_type=c.CFUNCTYPE(None,D,D,c.c_double,c.c_double,c.c_void_p)
        callback_errors=[]
        def report(summary,timings,gradient,numerical_rejections,context):
            if progress is None or callback_errors:return
            try:
                record=self._step_record(np.ctypeslib.as_array(summary,shape=(32,)).copy(),
                                         np.ctypeslib.as_array(timings,shape=(12,)).copy())
                record["projected_gradient_before"]=float(gradient)
                record["numerical_factorization_rejections"]=int(numerical_rejections)
                if progress(record) is False:self.request_stop()
            except BaseException as exc:
                callback_errors.append(exc);self.request_stop()
        callback=callback_type(report)
        fn=self._lib.rs_g4_solve
        fn.argtypes=[c.c_void_p,D,N,callback_type,c.c_void_p,D,N];fn.restype=c.c_int
        tol=np.array([gradient_tolerance,step_tolerance,relative_cost_tolerance],np.float64)
        out=np.zeros(17,np.float64)
        self._check(fn(self._handle,tol.ctypes.data_as(D),tol.size,callback,None,out.ctypes.data_as(D),out.size))
        result=dict(zip(("stop_code","trials","accepted","rejected","final_cost","projected_gradient_inf",
            "last_step_inf","last_relative_cost_change","lambda_next","nu_next","wall_ms","linearizations",
            "symbolic_analyses","explicit_application_d2h_bytes","linearization_wall_ms","candidate_wall_ms","numerical_factorization_rejections"),out.tolist()))
        result["stop_reason"]={0:"projected_gradient_small",1:"accepted_step_and_cost_small",
                               2:"floating_point_stagnation_without_convergence",3:"stopped_by_caller"}[int(out[0])]
        result["numerically_converged"]=out[0] in (0,1)
        result["tolerances"]={"gradient":gradient_tolerance,"step":step_tolerance,"relative_cost":relative_cost_tolerance}
        if callback_errors:raise callback_errors[0]
        return result

    def request_stop(self):
        """Thread-safe request; native returns after the current candidate.

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
        return {"controls":controls,"stations":stations}

    def export_step_system(self):
        """Explicit engineering snapshot; not called by lm_step."""
        self._g2_backend("cuda")
        q=self._q;nl=int(self.linearization_info["local_count"]);n=q+nl
        data=np.empty(q*q+q+2*n,np.float64);blocked=np.empty(nl,np.int32)
        self._check(self._lib.rs_g3_export_system(self._handle,data.ctypes.data_as(D),data.size,
                                                blocked.ctypes.data_as(I),blocked.size))
        self.diagnostic_download_bytes+=data.nbytes+blocked.nbytes
        return {"S":data[:q*q].reshape(q,q),"rhs":data[q*q:q*q+q],"step":data[q*q+q:q*q+q+n],
                "applied":data[-n:],"blocked":blocked}

    def close(self):
        if self._handle.value:
            self._lib.rs_g1_destroy(self._handle)
            self._handle=c.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
