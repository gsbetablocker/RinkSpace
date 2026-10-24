"""Recover required complete backing files into explicit effective-input storage.

Original logical addresses, original descriptors and their full-file SHA remain
unchanged. The native effective reader validates the copied full backing bytes
before interpreting any slice. This never creates covariance or observations.
"""
from pathlib import Path
import hashlib,importlib.util,json
from .effective_input_audit_v1 import CheckedBytes,resolve_packet,references,require
from .effective_input_prepare_v1 import ref


def recover_required_backings(head,decoder_ref,parts,out):
    store=CheckedBytes();packet,_=resolve_packet(head,store);out=Path(out)
    decoder_path=store.path(decoder_ref,Path(head['file']).parent);store.read(decoder_ref,decoder_path.parent)
    spec=importlib.util.spec_from_file_location('rinkfit_complete_original_bundle',decoder_path)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    bundle=decoder.AuditPackage(list(map(Path,parts)));pool=out/'payload';pool.mkdir(parents=True,exist_ok=True)
    seen=set();entries=[];checked=0
    for field,r in references(packet):
        if 'dtype' not in r:continue
        p=store.path(r,Path(head['file']).parent);key=(str(p.resolve()).casefold(),r['sha256'])
        if key in seen:continue
        seen.add(key);checked+=1
        if p.is_file():
            # Present-but-changed source data is never silently hidden by a copy.
            store.array(r,Path(head['file']).parent)
            continue
        raw=bundle.read({'file':str(p),'sha256':r['sha256']});actual=hashlib.sha256(raw).hexdigest()
        require(actual==r['sha256'],'Restored complete backing SHA differs from original input')
        target=pool/(actual+'.bin')
        if target.exists():require(hashlib.sha256(target.read_bytes()).hexdigest()==actual,'Existing effective payload differs')
        else:
            with target.open('xb') as stream:stream.write(raw)
        entries.append(dict(schema='rinkfit.exact_runtime_payload_file.v1',original=dict(file=str(p),sha256=actual),
            actual=ref(target),origin='complete_original_lossless_bundle',original_required_data_not_historical=True,
            first_packet_field=field))
    return dict(schema='rinkfit.exact_runtime_payload_map.v1',source_packet=head,source_parts=[dict(file=str(Path(p)),bytes=Path(p).stat().st_size) for p in parts],
                decoder=decoder_ref,checked_backing_files=checked,runtime_payload=entries,
                recovered_complete_backing_files=len(entries),all_recovered_full_original_SHA_verified=True,
                source_observations_or_W_changed=False,solver_started=False)
