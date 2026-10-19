"""Explicit startup-only evidence authentication with immutable open handles.

Only an allowlist of historical evidence references can be locked.  The caller
must end this scope before worker evaluation or solver trials.  Each original
item still executes the source revision's complete semantic validator; an AST
substitution replaces only its byte-hash expression.  No source equation,
noise, membership, producer guard, or file on disk is changed.

Windows FILE_SHARE_READ permits other readers and prevents writers/replacement
for the scope's lifetime.  A conflicting existing writer rejects this opt-in;
it is never interrupted.  No mtime-based identity or cross-run cache is used.
"""
from __future__ import annotations

import ast
import ctypes
from ctypes import wintypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import threading
import time


def reference(path):
    path=Path(path).resolve()
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),sha256=digest)


def collect_evidence_references(*documents):
    """Collect explicit evidence_references only; never recurse into outputs."""
    result={}
    def walk(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key=='evidence_references':
                    if not isinstance(child,list):raise ValueError('Evidence references must be a list')
                    for ref in child:
                        path=str(Path(ref['path']).resolve());expected=str(ref['sha256'])
                        normalized=os.path.normcase(path)
                        if normalized in result and result[normalized]['sha256']!=expected:
                            raise ValueError('One immutable evidence path has conflicting expected SHA')
                        result[normalized]=dict(path=path,sha256=expected)
                else:walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    for document in documents:walk(document)
    return tuple(result.values())


class StartupAuthenticationError(ValueError):
    pass


class _FileInformation(ctypes.Structure):
    _fields_=[('attributes',wintypes.DWORD),('created',wintypes.FILETIME),
        ('accessed',wintypes.FILETIME),('written',wintypes.FILETIME),
        ('volume',wintypes.DWORD),('size_high',wintypes.DWORD),('size_low',wintypes.DWORD),
        ('links',wintypes.DWORD),('index_high',wintypes.DWORD),('index_low',wintypes.DWORD)]


def _compile_evidence(original,authenticated_digest):
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    target=ast.dump(ast.parse('hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()',mode='eval').body,
        include_attributes=False)
    class Replace(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if ast.dump(node,include_attributes=False)==target:
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_startup_authenticated_digest',ctx=ast.Load()),
                    args=[ast.Name(id='ref',ctx=ast.Load())],keywords=[]),node)
            return self.generic_visit(node)
    replace=Replace();tree=replace.visit(tree)
    if replace.count!=1:raise StartupAuthenticationError('Exactly one original evidence hash expression is required')
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__)
    namespace['_startup_authenticated_digest']=authenticated_digest
    exec(compile(tree,original.__code__.co_filename+'::startup_authentication_scope','exec'),namespace)
    return namespace[original.__name__]


class startup_authentication_scope:
    """Use around one startup authentication only, with exact source SHA.

    ``expected_evidence_producer`` binds the original first_shot_revision.py.
    ``evidence_references`` is an explicit immutable allowlist.  References not
    in that list fail instead of locking an activity/output file by accident.
    """
    def __init__(self,*,evidence_references,expected_evidence_producer):
        self.expected_producer=dict(expected_evidence_producer)
        self.allowed={};self.raw_paths={};self.entries={};self.state='new';self.patched=None
        for ref in evidence_references:
            path=str(Path(ref['path']).resolve());key=os.path.normcase(path);sha=str(ref['sha256'])
            if len(sha)!=64 or any(c not in '0123456789abcdefABCDEF' for c in sha):
                raise ValueError('An explicit expected SHA256 is required')
            if key in self.allowed and self.allowed[key]['sha256']!=sha:
                raise ValueError('One immutable evidence path has conflicting expected SHA')
            self.allowed[key]=dict(path=path,sha256=sha)
            self.raw_paths[str(ref['path'])]=key
        self.stats=dict(item_hash_expression_calls=0,real_byte_hashes=0,memo_hits=0,
            real_hash_bytes=0,avoided_hash_bytes=0)
        self.records=[]

    def _windows(self):
        if os.name!='nt':raise StartupAuthenticationError('Read-sharing snapshot locks require Windows')
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
            wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        api.CreateFileW.restype=wintypes.HANDLE
        api.GetFileInformationByHandle.argtypes=[wintypes.HANDLE,ctypes.POINTER(_FileInformation)]
        api.GetFileInformationByHandle.restype=wintypes.BOOL
        api.ReadFile.argtypes=[wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID];api.ReadFile.restype=wintypes.BOOL
        api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
        return api

    def _open(self,path):
        handle=self.api.CreateFileW(path,0x80000000,1,None,3,0x08000000,None)
        if handle==ctypes.c_void_p(-1).value:
            raise StartupAuthenticationError('Immutable evidence read-share lock unavailable: '+path+
                ' [Windows error '+str(ctypes.get_last_error())+']')
        return handle

    def _info(self,handle):
        info=_FileInformation()
        if not self.api.GetFileInformationByHandle(handle,ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if info.attributes&0x10:raise StartupAuthenticationError('Evidence reference must be a file')
        return dict(volume_serial=int(info.volume),file_index=(int(info.index_high)<<32)|info.index_low,
            byte_length=(int(info.size_high)<<32)|info.size_low)

    def __enter__(self):
        if self.state!='new':raise StartupAuthenticationError('Authentication scope cannot be reused')
        from . import first_shot_revision as revision
        self.owner_thread=threading.get_ident();self.api=self._windows();self.module=revision
        self.original=revision._evidence
        actual=reference(revision.__file__)
        if actual!={key:self.expected_producer[key] for key in ('path','sha256')}:
            raise StartupAuthenticationError('Original evidence validator source SHA changed')
        if Path(self.original.__code__.co_filename).resolve()!=Path(revision.__file__).resolve():
            raise StartupAuthenticationError('An unknown evidence validator wrapper is already installed')
        self.patched=_compile_evidence(self.original,self.authenticated_digest)
        self.module._evidence=self.patched;self.state='active';self.started=time.perf_counter()
        return self

    def authenticated_digest(self,ref):
        """Strict checked-ref memo; bytes remain held immutable, never mtime."""
        if self.state!='active' or threading.get_ident()!=self.owner_thread:
            raise StartupAuthenticationError('Evidence certificates belong to one active startup thread')
        self.stats['item_hash_expression_calls']+=1
        try:
            key=self.raw_paths.get(str(ref['path']))
            if key is None:
                key=os.path.normcase(str(Path(ref['path']).resolve()));self.raw_paths[str(ref['path'])]=key
            expected=self.allowed.get(key)
            if expected is None or ref['sha256']!=expected['sha256']:
                raise StartupAuthenticationError('Foreign evidence path or expected SHA is outside this startup scope')
            prior=self.entries.get(key)
            if prior is not None:
                check=self._open(expected['path'])
                try:
                    if self._info(check)!=prior['fingerprint']:
                        raise StartupAuthenticationError('Evidence pathname no longer names the authenticated locked snapshot')
                finally:self.api.CloseHandle(check)
                self.stats['memo_hits']+=1;self.stats['avoided_hash_bytes']+=prior['fingerprint']['byte_length']
                return prior['actual_sha256']
            handle=self._open(expected['path'])
            try:
                before=self._info(handle);digest=hashlib.sha256();size=0
                buffer=ctypes.create_string_buffer(1024*1024);count=wintypes.DWORD()
                while True:
                    if not self.api.ReadFile(handle,buffer,len(buffer),ctypes.byref(count),None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if not count.value:break
                    digest.update(buffer.raw[:count.value]);size+=count.value
                actual=digest.hexdigest()
                if before!=self._info(handle) or size!=before['byte_length'] or actual!=expected['sha256']:
                    raise StartupAuthenticationError('Source revision evidence changed')
                entry=dict(expected_reference=expected,actual_sha256=actual,fingerprint=before,handle=handle)
                self.entries[key]=entry;handle=None
                self.records.append({k:v for k,v in entry.items() if k!='handle'})
                self.stats['real_byte_hashes']+=1;self.stats['real_hash_bytes']+=size
                return actual
            finally:
                if handle is not None:self.api.CloseHandle(handle)
        except BaseException:
            self.close(failed=True)
            raise

    def verify_reference(self,ref):
        self.authenticated_digest(ref)

    def close(self,*,failed=False):
        if self.state!='active':return
        restoration_error=None
        try:
            if self.module._evidence is not self.patched:
                restoration_error=StartupAuthenticationError('Another caller replaced the startup evidence validator')
            else:self.module._evidence=self.original
        finally:
            for entry in self.entries.values():
                handle=entry.pop('handle',None)
                if handle is not None:self.api.CloseHandle(handle)
            self.state='failed' if failed else 'closed';self.elapsed=time.perf_counter()-self.started
        if restoration_error is not None:raise restoration_error

    def __exit__(self,kind,error,traceback):
        self.close(failed=kind is not None)
        return False

    def receipt(self):
        return dict(schema='rinkspace.single-startup-immutable-evidence-authentication.v1',
            producer=reference(__file__),original_evidence_producer=self.expected_producer,
            state=self.state,statistics=dict(self.stats),authenticated_references=list(self.records),
            source_item_semantics='original validator AST; only one hash expression replaced',
            lock='Windows GENERIC_READ with FILE_SHARE_READ only; deny writes and replacement until close',
            expected_SHA_and_real_bytes_authenticated=True,mtime_used_as_byte_identity=False,
            trial_producer_guards_patched=False,source_likelihood_or_membership_changed=False,
            cross_run_cache=False,caller_must_close_before_workers_or_trials=True,
            elapsed_seconds=getattr(self,'elapsed',None))
