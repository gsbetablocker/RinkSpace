"""Loopback-only standard-library control server. It never imports or launches a solver."""
from __future__ import annotations

from functools import partial, lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from .hardware import HardwareMonitor
from .model_view import inspection_snapshot
from .runtime_state import model_store
from .run_monitor import current_run
from .store import CurrentStore, StoreError

LOG = logging.getLogger("rinksolve.control")


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, store, ui_dir, hardware=None):
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("控制服务仅允许绑定本机127.0.0.1")
        self.store = store
        self.ui_dir = Path(ui_dir).resolve()
        self.hardware = hardware or HardwareMonitor()
        self._selection_lock = threading.RLock()
        self._selection_cache = None
        self._projection_lock = threading.Lock()
        self._projection_cache = {}
        super().__init__(address, ControlHandler)
        self.hardware.start()

    def server_close(self):
        super().server_close()
        self.hardware.close()

    def runtime_model(self):
        binding=self.store.programme_input().get("inspection_model",{})
        if binding.get("kind")!="runtime_model":
            raise StoreError("当前尚未绑定可编辑物理模型",409,"runtime_model_not_selected")
        return model_store(binding["package"],self.store.data)

    def selection(self):
        from .preparation import load_prepared
        from .observation_edits import ObservationSelection
        from .store import read_object
        programme=self.store.programme_input()
        delivery_path=Path(programme.get('preparation_delivery',self.store.data/'runs/solve2_s01/preparation/delivery.json')).resolve()
        try:delivery_path.relative_to(self.store.data)
        except ValueError:raise StoreError('当前准备交付必须位于Data',503,'preparation_location')
        delivery=read_object(delivery_path)
        prepared_path=Path(delivery['prepared']).resolve()
        selection_path=Path(programme.get('observation_selection',self.store.data/'runs/solve2_s01/selection.json')).resolve()
        key=(str(prepared_path),str(selection_path))
        with self._selection_lock:
            if self._selection_cache is None or self._selection_cache[0]!=key:
                prepared=load_prepared(prepared_path,data_root=self.store.data)
                self._selection_cache=(key,ObservationSelection(prepared,selection_path,data_root=self.store.data))
            return self._selection_cache[1]

    @lru_cache(maxsize=4)
    def native_frame_png(self,pts):
        from .schema import SourceClock,SourceFrame
        from .sources import NativeVideoReader
        import cv2
        if type(pts) is not int or pts<256256 or pts>575488 or pts%256:
            raise StoreError('必须使用S01原生整数PTS')
        source=self.store.programme_input()['source']
        tnum,tden=map(int,source['time_base'].split('/'))
        fnum,fden=map(int,source['native_fps'].split('/'))
        clock=SourceClock(source['id']+'.native_clock',tnum,tden,'current_official_video')
        frame=SourceFrame(source['id']+':'+str(pts),clock.id,pts,source['width'],source['height'],'current_official_video')
        image=NativeVideoReader(source['path'],self.store.data,clock,source['width'],source['height'],fnum,fden).frame(frame)
        ok,png=cv2.imencode('.png',cv2.cvtColor(image.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
        if not ok:raise StoreError('当前原生曝光图像编码失败',503,'source_frame_encode')
        return png.tobytes()


    def projection_preview(self,pts):
        # The bundle remains frozen after load; newer numerical publications
        # cannot mix the source exposure with another parameter generation.
        if type(pts) is not int or pts<256256 or pts>575488 or pts%256:
            raise StoreError('必须使用S01原生整数PTS',422,'native_pts_scope')
        import cv2
        import numpy as np
        from .renderer import native_projection_preview
        with self._projection_lock:
            model=self.runtime_model().load()
            key=(model.id,model.revision,model.state.digest(),pts)
            if key in self._projection_cache:return self._projection_cache[key]
            bgr=cv2.imdecode(np.frombuffer(self.native_frame_png(pts),dtype=np.uint8),cv2.IMREAD_COLOR)
            overlay,identity=native_projection_preview(model,pts,bgr)
            binding=self.store.programme_input().get('inspection_model',{})
            identity.update(restored_original=bool(binding.get('restored_original')),read_only=bool(binding.get('read_only')),
                reference_label=binding.get('label'),original_parameter_sha256=binding.get('original_parameter_sha256'))
            ok,png=cv2.imencode('.png',overlay,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:raise StoreError('当前网格预览编码失败',503,'projection_encode')
            result=(png.tobytes(),identity)
            if len(self._projection_cache)>=2:self._projection_cache.pop(next(iter(self._projection_cache)))
            self._projection_cache[key]=result
            return result

    def state(self):
        programme=self.store.programme_input()
        recovery=programme.get('original_s01_recovery',{})
        if recovery.get('active'):
            return {"phase":"restoring_original_s01","message":"当前显示本来的第一镜头：当时冰场与相机参数原样保留。原代码、配置、日志与完整状态正在整理；当前不运行拟合，不用更差的新结果替换原版。",
                "run":None,"hardware":self.hardware.snapshot(),"events":self.store.events(),"sequence":self.store.sequence(),
                "engine":{"available":False,"legacy_solver_enabled":False},"original_s01_recovery":recovery}
        active=current_run(self.store.programme_input(),self.store.data)
        if active is not None:
            events=list(active.get("events",[]))+self.store.events()
            return {**active,"hardware":self.hardware.snapshot(),"events":events,"sequence":self.store.sequence(active)}
        return {"phase": "assembling_current_problem", "message": "S01真实输入、同一完整物理状态、新CPU/GPU前向与稀疏数值桥已交付；正在接入实体关联和联合残差。参数可编辑，S01数值拟合尚未启动，后续镜头锁定。",
                "run": None, "hardware": self.hardware.snapshot(), "events": self.store.events(),
                "sequence": self.store.sequence(), "engine": {"available": False, "legacy_solver_enabled": False}}


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "RinkSolve2/0.1"

    def log_message(self, format, *args):
        LOG.info("%s %s", self.client_address[0], format % args)

    def _json(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, exc):
        self._json(exc.status, {"error": exc.code, "message": str(exc)})

    def _path(self):
        return unquote(urlsplit(self.path).path)

    def _model_snapshot(self):
        query=parse_qs(urlsplit(self.path).query)
        value=query.get("native_pts")
        pts=None
        if value is not None:
            if len(value)!=1 or not value[0].isdigit():raise StoreError("必须提供一个原生整数PTS",422,"native_pts_type")
            pts=int(value[0])
        return inspection_snapshot(self.server.store.programme_input(),self.server.store.data,pts)

    def _observations(self):
        query=parse_qs(urlsplit(self.path).query)
        def one(key,default=None):
            value=query.get(key)
            if value is None:return default
            if len(value)!=1:raise StoreError('查询字段不得重复：'+key)
            return value[0]
        try:
            pts=one('native_pts')
            return self.server.selection().records(native_pts=int(pts) if pts is not None else None,
                query=one('query',''),curve_id=one('curve_id'),offset=int(one('offset','0')),limit=int(one('limit','64')))
        except (TypeError,ValueError) as exc:raise StoreError('取点查询必须使用有效原生PTS和整数分页') from exc

    def _static(self, relative):
        path = (self.server.ui_dir / relative).resolve()
        try:
            path.relative_to(self.server.ui_dir)
        except ValueError:
            self._json(404, {"error": "not_found", "message": "文件不存在"})
            return
        if not path.is_file():
            self._json(404, {"error": "not_found", "message": "文件不存在"})
            return
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") or mime in {"application/javascript", "application/json"} else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            path = self._path()
            if path == '/api/model/projection':
                query=parse_qs(urlsplit(self.path).query)
                pts=query.get('native_pts')
                if pts is None or len(pts)!=1 or not pts[0].isdigit():
                    raise StoreError('网格预览必须给一个原生整数PTS',422,'native_pts_type')
                image,identity=self.server.projection_preview(int(pts[0]))
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(len(image)));self.send_header('Cache-Control','no-store')
                self.send_header('X-Projection-Identity',json.dumps(identity,ensure_ascii=True,separators=(',',':')))
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers();self.wfile.write(image);return
            if path == '/api/observations/frame':
                query=parse_qs(urlsplit(self.path).query)
                pts=query.get('native_pts')
                if pts is None or len(pts)!=1 or not pts[0].isdigit():raise StoreError('原片取点图必须给一个原生整数PTS')
                image=self.server.native_frame_png(int(pts[0]))
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(len(image)));self.send_header('Cache-Control','private, max-age=3600')
                self.send_header('X-Source-Native-PTS',pts[0]);self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers();self.wfile.write(image);return
            if path == "/":
                return self._static("index.html")
            if path.startswith("/ui/"):
                return self._static(path[4:])
            endpoints = {"/api/architecture": self.server.store.architecture,
                         "/api/contracts": self.server.store.contracts,
                         "/api/constraints": self.server.store.constraints,
                         "/api/data": self.server.store.datasets,
                         "/api/review": self.server.store.review,
                         "/api/observations": self._observations,
                         "/api/model/parameters": lambda: self.server.runtime_model().parameters(),
                         "/api/state": self.server.state,
                         "/api/model": self._model_snapshot}
            if path not in endpoints:
                return self._json(404, {"error": "not_found", "message": "接口不存在"})
            self._json(200, endpoints[path]())
        except StoreError as exc:
            self._error(exc)
        except (OSError, ValueError) as exc:
            LOG.exception("GET failed")
            self._json(500, {"error": "read_failed", "message": str(exc)})

    def _check_origin(self):
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != self.server.server_address[1]:
                raise StoreError("仅允许本机控制页面修改", 403, "invalid_origin")

    def do_PUT(self):
        try:
            self._check_origin()
            path = self._path()
            prefix = "/api/constraints/"
            model_edit = path == "/api/model/parameters"
            selection_edit = path == "/api/observations/selection"
            if not model_edit and not selection_edit and (not path.startswith(prefix) or not path[len(prefix):] or "/" in path[len(prefix):]):
                return self._json(404, {"error": "not_found", "message": "接口不存在"})
            if self.headers.get_content_type() != "application/json":
                raise StoreError("请求必须使用application/json", 415, "invalid_content_type")
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise StoreError("无效Content-Length", 400, "invalid_json") from exc
            if size < 0:
                raise StoreError("缺少Content-Length", 411, "length_required")
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (ValueError, UnicodeError) as exc:
                raise StoreError("请求必须为有效JSON", 400, "invalid_json") from exc
            if (model_edit or selection_edit) and self.server.store.programme_input().get('inspection_model',{}).get('read_only'):
                raise StoreError("这页正在显示原样找回的第一镜头，原参数和原资料保留不改。",409,"restored_original_read_only")
            if model_edit:result = self.server.runtime_model().update(payload)
            elif selection_edit:result = self.server.selection().update(payload)
            else:result = self.server.store.update_constraint(path[len(prefix):], payload)
            self._json(200, result)
        except StoreError as exc:
            self._error(exc)
        except (OSError, ValueError) as exc:
            LOG.exception("PUT failed")
            self._json(500, {"error": "write_failed", "message": str(exc)})

    def do_POST(self):
        self._json(409, {"error": "engine_not_implemented", "message": "此控制接口不启动数值任务；实际进度由当前运行状态报告。S01全段真实验证通过前禁止进入后续镜头。",
                         "run": None, "advance_allowed": False})

    def do_DELETE(self):
        self._json(405, {"error": "method_not_allowed", "message": "不支持删除当前来源或历史记录"})


def make_server(root, data, host="127.0.0.1", port=8787, event="ankara_2026", programme="okada_mei_sp", hardware=None):
    store = CurrentStore(root, data, event=event, programme=programme)
    return ControlServer((host, port), store, Path(root) / "calibration" / "solve2" / "ui", hardware=hardware)
