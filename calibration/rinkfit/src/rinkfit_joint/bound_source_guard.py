"""Scoped immutable source binding for a frozen evaluation or render.

Perform the existing full material/operator guard at both boundaries. During
frames, detect recalls and any change to every dependency actually authenticated
by that guard. No global cache, relaxed source membership, or fitted UV is used.
"""
from pathlib import Path
import sys,json,gzip
from . import io_utils,joint_solver

_MARKERS=('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json')

def _stamp(path):
    try:s=path.stat()
    except OSError as exc:raise ValueError('Frozen source dependency unavailable: '+str(path))from exc
    return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_dev,s.st_ino)

class BoundSourceIdentityGuard:
    def __init__(self,packet,*,packet_sha256=None,extra_refs=()):
        self.packet=Path(packet).resolve();self.expected_packet_sha256=packet_sha256
        self.dependencies={};self.packet_paths={self.packet};self.recall_directories={self.packet.parent};self.extra_refs=list(extra_refs)
        self.full_checks=0;self.metadata_checks=0;self._full_check()

    def _observe(self,path,digest,stamp):
        if not isinstance(digest,str)or len(digest)!=64:raise ValueError('Invalid source dependency digest')
        existing=self.dependencies.get(path)
        if existing is not None and (existing['sha256']!=digest or existing['stamp']!=stamp):
            raise ValueError('Frozen source dependency changed: '+str(path))
        self.dependencies[path]=dict(sha256=digest,stamp=stamp);self._seen.add(path)

    def _hash(self,path,expected=None):
        path=Path(path).resolve();before=_stamp(path);digest=io_utils.file_sha256(path)
        if before!=_stamp(path):raise ValueError('Source dependency changed during authentication: '+str(path))
        if expected is not None and digest!=expected:raise ValueError('Frozen source SHA mismatch: '+str(path))
        self._observe(path,digest,before)

    def _bind_ref(self,ref,base):
        if not isinstance(ref,dict)or 'sha256'not in ref or not('file'in ref or 'path'in ref):return None
        path=Path(ref.get('file',ref.get('path')));path=path if path.is_absolute()else base/path;path=path.resolve()
        if path in self._seen:
            data=self.dependencies[path]
            if data['sha256']!=ref['sha256']or _stamp(path)!=data['stamp']:raise ValueError('Inconsistent source dependency: '+str(path))
        else:self._hash(path,ref['sha256'])
        return path

    def _bind_numeric_dependencies(self,packet):
        base=self.packet.parent
        for ref in packet.get('arrays',{}).values():self._bind_ref(ref,base)
        for group in packet.get('groups',[]):self._bind_ref(group.get('W'),base)
        visited=set()
        def query_refs(value,root):
            if isinstance(value,dict):
                ref=self._bind_ref(value,root)
                if ref is not None:
                    if ref not in visited and (ref.suffix=='.json'or ref.name.endswith('.json.gz')):
                        visited.add(ref);opener=gzip.open if ref.suffix=='.gz'else open
                        with opener(ref,'rt',encoding='utf8')as stream:query_refs(json.load(stream),ref.parent)
                    return
                for child in value.values():query_refs(child,root)
            elif isinstance(value,list):
                for child in value:query_refs(child,root)
        for name in ('queries','query_packet','frozen_query_manifest','frozen_queries'):query_refs(packet.get(name),base)

    def _markers(self):
        for directory in self.recall_directories|{p.parent for p in self.packet_paths}:
            for name in _MARKERS:
                marker=directory/name
                if marker.exists():raise ValueError('Source physical identity certificate recalled: '+str(marker))

    def _full_check(self):
        self._markers();before_packet=_stamp(self.packet);self._seen=set();pending={}
        # The existing checkpoint closure also traces the actual hash functions;
        # capturing the actual accesses avoids guessing a parallel proof schema.
        functions={io_utils.file_sha256.__code__:('path',False),joint_solver.sha.__code__:('p',True)}
        previous=sys.getprofile()
        def trace(frame,event,arg):
            if event=='call'and frame.f_code is Path.exists.__code__:
                marker=Path(frame.f_locals['self']).resolve()
                if marker.name in _MARKERS:self.recall_directories.add(marker.parent)
            if event=='call'and frame.f_code in functions:
                key,ispacket=functions[frame.f_code];path=Path(frame.f_locals[key]).resolve()
                pending[id(frame)]=(path,_stamp(path))
                if ispacket:self.packet_paths.add(path)
            elif event=='return'and id(frame)in pending:
                path,before=pending.pop(id(frame))
                if arg is not None:
                    if before!=_stamp(path):raise ValueError('Dependency changed while validating: '+str(path))
                    self._observe(path,arg,before)
            if callable(previous):previous(frame,event,arg)
        sys.setprofile(trace)
        try:
            packet=joint_solver.assert_source_identity_current(self.packet,return_verified_packet=True)
            self._hash(self.packet,self.expected_packet_sha256)
            self._bind_numeric_dependencies(packet)
            for ref in self.extra_refs:self._hash(ref.get('file',ref.get('path')),ref['sha256'])
            for path,data in list(self.dependencies.items()):
                if path not in self._seen:self._hash(path,data['sha256'])
        finally:sys.setprofile(previous)
        if before_packet!=_stamp(self.packet):raise ValueError('Frozen packet changed while binding source')
        self.full_checks+=1;self.check()

    def check(self):
        self._markers()
        for path,data in self.dependencies.items():
            if _stamp(path)!=data['stamp']:raise ValueError('Frozen source dependency changed: '+str(path))
        self.metadata_checks+=1

    def finish(self):
        self.check();self._full_check()
        return dict(schema='rinkfit.scoped_immutable_source_guard.v1',packet=str(self.packet),
            packet_sha256=self.dependencies[self.packet]['sha256'],authenticated_dependencies=len(self.dependencies),recall_directories=len(self.recall_directories),
            complete_source_guard_calls=self.full_checks,recall_and_dependency_metadata_checks=self.metadata_checks,
            full_SHA_and_material_operator_guard_at_start_and_end=True,
            changed_or_recalled_source_allowed=False,global_cache_used=False)
