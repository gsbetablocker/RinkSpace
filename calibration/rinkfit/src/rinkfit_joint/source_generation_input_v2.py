"""Strict, streaming consumption of a versioned S01--S04 source-reader input.

Small bindings share the complete base support ledger; appended source parts
are streamed without duplicating or expanding the base document on disk.
This loads source evidence, not solver admission or an invented covariance.
"""
from pathlib import Path
from .io_utils import read_json,read_jsonl,file_sha256,object_sha256

class QualifiedSourceGenerationInput:
    def __init__(self,binding_path):
        self.path=Path(binding_path).resolve();self.identity=file_sha256(self.path)
        self.binding=read_json(self.path)
        if self.binding.get('schema')!='rinkfit.S01_S04.qualified_source_reader_input.v2':
            raise ValueError('Unsupported qualified source input schema')
        self._refs={};self._pin(self.binding['generation']);self.generation=read_json(self.binding['generation']['path'])
        for key in ('rows','base_reader_qualification','new_reader_qualification','common_model','model_chart_repair','static_ice_exclusions'):
            self._pin(self.binding[key])
        for part in self.binding['source_support_parts']:self._pin(part['ref'])
        for ref in self.binding.get('dependency_refs',[]):self._pin(ref)
        for logical,ref in self.generation['references'].items():
            self._pin(ref)
            if '/observations/frames/' in logical:
                source=read_json(ref['path'])['source']
                self._pin({'path':source['image_path'],'sha256':source['png_sha256']})
                self._pin({'path':source['mask_path'],'sha256':source['mask_sha256']})
        for key in ('common_model_candidate','ad24_source_detail','ad24_handoff'):
            if key in self.generation:self._pin(self.generation[key])
        self._prefix={}
        for proof in self.binding['rebound_packet_prefix_proofs']:
            self._pin(proof['old_packet']);self._pin(proof['new_packet'])
            old=read_json(proof['old_packet']['path']);new=read_json(proof['new_packet']['path']);n=proof['old_feature_count']
            if n!=len(old['features']) or object_sha256(old['features'])!=object_sha256(new['features'][:n]):
                raise ValueError('Reader reuse changes an existing source feature')
            for key in ('source','unbound_static_materials'):
                if object_sha256(old[key])!=object_sha256(new[key]):raise ValueError('Reader reuse changes source RGB/mask/clock or unbound evidence')
            self._prefix[proof['logical_path']]=proof
        registry=self.generation['registry_prefix_proof'];self._pin(registry['parent']);self._pin(registry['new'])
        old=read_json(registry['parent']['path'])['targets'];new=read_json(registry['new']['path'])['targets']
        if any(k not in new or object_sha256(v)!=object_sha256(new[k]) for k,v in old.items()):
            raise ValueError('Appended registry changes an existing material descriptor')
        q=read_json(self.binding['new_reader_qualification']['path'])
        if not all(q.get(k) for k in ('all_new_UV_binary64_exact','all_new_reader_numeric_valid','all_new_windowboundary_zero','all_original_UV_ID_PTS_exact','manual76_preserved')):
            raise ValueError('Frozen reader qualification is incomplete')
        if self.binding.get('consumer_api'):self._pin(self.binding['consumer_api'])
        self.row_count=int(self.binding['total_rows'])

    def _pin(self,ref):
        p=Path(ref['path']).resolve();expected=ref['sha256']
        if p in self._refs and self._refs[p]!=expected:raise ValueError('Conflicting source dependency hashes')
        if p not in self._refs:
            from .exact_dependency_aliases import resolve_dependency
            p=resolve_dependency(ref)
            if 'bytes' in ref and p.stat().st_size!=ref['bytes']:raise ValueError('Changed source input dependency size')
            self._refs[p]=expected

    def assert_dependencies(self):
        """Batch boundary pin-check; no video decode, detector or source re-read."""
        for p,expected in self._refs.items():
            if file_sha256(p)!=expected:raise ValueError('Changed qualified source dependency: '+str(p))

    def rows(self):
        seen=set();n=0
        for row in read_jsonl(self.binding['rows']['path']):
            rid=row['source_row_id']
            if rid in seen:raise ValueError('Duplicate source row ID')
            seen.add(rid)
            expected=self.generation['references'][row['source_packet']]['sha256']
            if row['source_packet_sha256']!=expected:raise ValueError('Current row packet binding differs')
            n+=1;yield row
        if n!=self.row_count:raise ValueError('Full source row denominator changed')

    def supports(self):
        seen=set()
        for part in self.binding['source_support_parts']:
            exclude=set(part.get('exclude_source_packets',[]))
            for source in read_jsonl(part['ref']['path']):
                if source['source_packet'] in exclude:continue
                sid=source['statistical_input_id']
                if sid in seen:raise ValueError('Duplicated support-family input')
                seen.add(sid);proof=self._prefix.get(source['source_packet'])
                if proof:
                    if source['source_packet_sha256']==proof['old_packet']['sha256']:
                        source={**source,'ancestor_source_packet_sha256':source['source_packet_sha256'],
                                'source_packet_sha256':proof['new_packet']['sha256'],
                                'exact_source_feature_prefix_reuse':True}
                    elif source['source_packet_sha256']!=proof['new_packet']['sha256']:
                        raise ValueError('Support prefix source binding mismatch')
                expected=self.generation['references'][source['source_packet']]['sha256']
                if source['source_packet_sha256']!=expected:raise ValueError('Current support packet binding differs')
                yield source
