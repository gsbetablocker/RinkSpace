"""Explicit policy selection for the unchanged historical owner adapter.

Only its named arm load is configured. Every original arm/hook byte, old
policy copy, new policy and real same-state arc replay is authenticated.
No file SHA is replaced by a claim about different bytes.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import importlib.util
import hashlib
from .first_shot_source_transition import reference,read


@contextmanager
def reviewed_ordered_successor(manifest_reference, *, physical_q_sha256):
    manifest=read(manifest_reference)
    if (manifest.get('schema')!='rinkspace.reviewed-ordered-arc-producer-successor.v1'
            or manifest.get('actual_state_sha256')!=physical_q_sha256
            or manifest.get('new_target_requires_fresh_complete_J')is not True):
        raise ValueError('Explicit same-state reviewed source-policy successor required')
    refs=[manifest_reference,*[manifest[k]for k in ('historical_arm','historical_policy','successor_policy','actual_same_state_replay')]]
    refs.extend(manifest[k]for k in ('material_side_evidence','ordinary_source_quarantine','ordinary_quarantine_producer')if k in manifest)
    def check():
        for ref in refs:
            if reference(ref['path'])!=ref:raise ValueError('Frozen reviewed ordered-policy successor changed')
    check();replay=read(manifest['actual_same_state_replay'])
    if (replay['state_sha']!=physical_q_sha256 or replay['source_pixels_changed']is not False
            or set(replay['counts'])!={'associated','unqualified_single_source_vertex'}
            or replay['counts']['associated']!=manifest['contiguous_arcs']
            or sum(r['source_points']for r in replay['records']if r['status']=='associated')!=manifest['qualified_points']):
        raise ValueError('Actual reviewed arc replay does not justify this successor')
    if manifest.get('association_method')is not None:
        if (replay.get('ordinary_quarantine')!=manifest.get('ordinary_source_quarantine')
                or replay.get('failures')
                or any(r.get('association_method')!=manifest['association_method']for r in replay['records']if r['status']=='associated')):
            raise ValueError('Actual original-material-side replay or explicit quarantine differs')
    original=importlib.util.spec_from_file_location;certificates=[]
    def selected_spec(name,path,*args,**kwargs):
        spec=original(name,path,*args,**kwargs)
        if name!='reviewed_source_and_export_arm' or Path(path).resolve()!=Path(manifest['historical_arm']['path']).resolve():return spec
        loader=spec.loader
        class CheckedLoader:
            def __getattr__(self,key):return getattr(loader,key)
            def create_module(self,spec):return loader.create_module(spec)
            def exec_module(self,module):
                check();loader.exec_module(module)
                if module.POLICY_SHA!=manifest['historical_policy']['sha256']:
                    raise ValueError('Historical arm does not select the declared original ordered policy')
                module.POLICY_SHA=manifest['successor_policy']['sha256']
                certificates.append(dict(schema='rinkspace.explicit-ordered-policy-load.v1',manifest=deepcopy(manifest_reference),
                    historical_arm=manifest['historical_arm'],historical_policy=manifest['historical_policy'],
                    current_policy=manifest['successor_policy'],producer=reference(__file__),
                    source_labels_UV_W_and_native_PTS_not_rewritten=True,old_source_J_H_reuse_forbidden=True))
        spec.loader=CheckedLoader();return spec
    importlib.util.spec_from_file_location=selected_spec
    try:
        yield certificates
        if len(certificates)!=1:raise ValueError('Exactly the declared original owner arm must be loaded')
    finally:
        importlib.util.spec_from_file_location=original
