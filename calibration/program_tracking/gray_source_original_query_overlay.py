"""Read authenticated original gray query inputs without creating observations.

The producer overlay stores appended candidate IDs and full original queries.
The getter binds an ID through its literal source quartiles and exact clock.
An optional frozen producer cache can bind already selected prefix IDs through
their byte-exact raw pixel and source clock. Its roles and flags never approve
identity, and only the authenticated overlay supplies float64 query inputs.
Original point coordinates and source roles stay intact.
"""
from pathlib import Path
import json
import numpy as np
from .strict_model_scope import authenticate_source_reference


class OriginalSourceQueryOverlay:
    def __init__(self, source_pack_reference, overlay_reference=None,*,
            measurement_cache_reference=None,programme_manifest_reference=None):
        self.source_pack_reference=dict(source_pack_reference)
        authenticate_source_reference(self.source_pack_reference)
        with np.load(self.source_pack_reference['path'],allow_pickle=False) as z:
            self.source_metadata=json.loads(str(z['metadata_json'].item()))
            actual=z['source_frame_actual_pts'];native=z['source_frame_native_pts'];digests=z['source_frame_BGR_sha256']
        self.frame_bindings={int(a):dict(actual_pts=int(a),native_pts=int(n),decoded_native_BGR_sha256=
            b.decode('ascii') if isinstance(b,(bytes,np.bytes_)) else str(b)) for a,n,b in zip(actual,native,digests)}
        refs=[dict(r) for r in self.source_metadata['original_references']
              if Path(r['path']).name.endswith('_source_role_overlay.npz')]
        if overlay_reference is None:
            if len(refs)>1:raise ValueError('An explicit authenticated original source overlay is required')
            overlay_reference=refs[0] if refs else None
        elif not any(Path(r['path']).resolve()==Path(overlay_reference['path']).resolve()
                and r['sha256']==overlay_reference['sha256'] for r in refs):
            raise ValueError('Query overlay must already be an authority of this original source pack')
        self.reference=dict(overlay_reference) if overlay_reference is not None else None
        self.arrays={};self.lookup={};self.status_counts={};self.cache=None
        self.measurement_cache_reference=None;self.programme_manifest_reference=None
        if self.reference is None:return
        authenticate_source_reference(self.reference)
        names=('actual_pts','offsets','original_candidate_counts','candidate_query',
            'candidate_side_quantiles_uv','query_offsets','query_curve','query_uv','query_normal','query_interval')
        with np.load(self.reference['path'],allow_pickle=False) as z:
            self.arrays={name:z[name] for name in names}
            self.metadata=json.loads(str(z['metadata_json'].item()))
        for values in self.arrays.values():values.flags.writeable=False
        source=self.source_metadata['source'];packet=self.metadata['source']
        source_sha=source.get('video_sha256',source.get('source_sha256'))
        packet_sha=packet.get('video_sha256',packet.get('source_sha256'))
        if source_sha!=packet_sha or source['time_base']!=packet['time_base']:
            raise ValueError('Original query overlay belongs to a different video clock')
        a=self.arrays;self.lookup={int(pts):i for i,pts in enumerate(a['actual_pts'])}
        if len(self.lookup)!=len(a['actual_pts']) or not set(self.lookup).issubset(self.frame_bindings):
            raise ValueError('Original query overlay exposure clock differs from source pack')
        for offsets,field in [('offsets','candidate_query'),('query_offsets','query_curve')]:
            values=a[offsets]
            if len(values)!=len(a['actual_pts'])+1 or values[0]!=0 or np.any(np.diff(values)<0) or values[-1]!=len(a[field]):
                raise ValueError('Original query overlay offsets are incomplete')
        self.inventory=self.metadata['effective_inventory']
        if measurement_cache_reference is not None:
            if programme_manifest_reference is None:
                raise ValueError('The original frozen producer argv must bind the query-index cache')
            self.measurement_cache_reference=dict(measurement_cache_reference)
            self.programme_manifest_reference=dict(programme_manifest_reference)
            authenticate_source_reference(self.measurement_cache_reference)
            authenticate_source_reference(self.programme_manifest_reference)
            args=json.loads(Path(self.programme_manifest_reference['path']).read_text(encoding='utf8'))['common_arguments']
            if Path(args[args.index('--measurements')+1]).resolve()!=Path(self.measurement_cache_reference['path']).resolve():
                raise ValueError('Frozen producer argv and query-index cache paths differ')
            from rinkspace.programme_source_measurement_cache import ProgrammeSourceMeasurementCache
            self.cache=ProgrammeSourceMeasurementCache(self.measurement_cache_reference['path'],
                source_sha256=source_sha,time_base=source['time_base'],
                expected_actual_pts=self.source_metadata['expected_actual_pts'])

    def _result(self,status,**values):
        self.status_counts[status]=self.status_counts.get(status,0)+1
        return dict(status=status,source_pack_reference=self.source_pack_reference,
            source_query_overlay_reference=self.reference,original_source_values_changed=False,**values)

    def get(self,actual_pts,original_candidate_index,*,quantiles,uv=None,sides=None,raw_candidate_uv=None,
            curve_id=None,source_binding=None):
        actual=int(actual_pts);candidate=int(original_candidate_index)
        base=dict(actual_pts=actual,original_candidate_index=candidate)
        binding=self.frame_bindings.get(actual)
        if binding is None:return self._result('pending_original_query_source_exposure_binding_missing',**base)
        if source_binding is not None and (int(source_binding['actual_pts'])!=actual or
                int(source_binding.get('native_pts',source_binding.get('source_pts',actual)))!=binding['native_pts'] or
                source_binding.get('decoded_native_BGR_sha256',source_binding.get('decoded_BGR_sha256'))!=binding['decoded_native_BGR_sha256']):
            raise ValueError('Point and original query use different source RGB clocks')
        base['source_binding']=binding
        if actual not in self.lookup:return self._result('pending_original_query_overlay_exposure_missing',**base)
        frame=self.lookup[actual];a=self.arrays;prefix=int(a['original_candidate_counts'][frame])
        original=np.asarray(quantiles,float)
        if original.shape!=(2,3,2) or not np.isfinite(original).all():
            return self._result('pending_original_query_literal_quartile_support_missing',**base)
        if sides is not None and not np.array_equal(original[:,1],np.asarray(sides,float)):
            return self._result('pending_original_query_literal_q50_sides_differ',**base)
        if uv is not None and not np.array_equal(original[:,1].mean(0),np.asarray(uv,float)):
            return self._result('pending_original_query_literal_q50_center_differ',**base)
        extra={}
        if candidate<prefix:
            if self.cache is None:
                return self._result('pending_original_candidate_precedes_appended_query_overlay',original_candidate_prefix_count=prefix,**base)
            cached=self.cache.frame(actual)
            cb=cached['source_binding']
            if (int(cb['actual_pts'])!=actual or int(cb['native_pts'])!=binding['native_pts'] or
                    cb['decoded_native_BGR_sha256']!=binding['decoded_native_BGR_sha256']):
                raise ValueError('Query-index cache and immutable source point use different RGB clocks')
            if prefix!=len(cached['candidate_uv']) or candidate<0:
                return self._result('pending_original_query_index_cache_candidate_domain_differs',**base)
            if raw_candidate_uv is None:
                return self._result('pending_original_prefix_raw_candidate_locator_missing',**base)
            if not np.array_equal(np.asarray(raw_candidate_uv,float),np.asarray(cached['candidate_uv'][candidate],float)):
                return self._result('pending_original_prefix_raw_candidate_differs_from_index_cache',**base)
            query=int(cached['candidate_query'][candidate]);index=None
            extra=dict(source_query_index_cache_reference=self.measurement_cache_reference,
                source_query_index_cache_frozen_argv_reference=self.programme_manifest_reference,
                cached_candidate_raw_UV_bitexact_to_immutable_pack=True,
                cached_candidate_values_used_only_to_bind_already_selected_query_index=True,
                cached_roles_flags_or_identity_used_as_approval=False,
                original_query_locator_kind='selected_prefix_raw_candidate_and_original_overlay_query',
                literal_original_quartiles_require_current_own_RGB_production_query_replay=True)
        else:
            lo,hi=map(int,a['offsets'][frame:frame+2]);local=candidate-prefix
            if local>=hi-lo or local<0:return self._result('pending_original_candidate_overlay_locator_missing',**base)
            index=lo+local;q=a['candidate_side_quantiles_uv'][index]
            if q.shape!=(2,3,2) or not np.isfinite(q).all():
                return self._result('pending_original_query_literal_quartile_support_missing',**base)
            if not np.array_equal(q,original):
                return self._result('pending_original_query_literal_quartiles_differ',**base)
            query=int(a['candidate_query'][index])
            extra=dict(original_quartiles_equal=True,
                original_query_locator_kind='original_appended_overlay_literal_quartiles')
        qlo,qhi=map(int,a['query_offsets'][frame:frame+2])
        if query<0 or query>=qhi-qlo:return self._result('pending_original_query_index_outside_exposure',**base)
        qi=qlo+query;definition=self.inventory[int(a['query_curve'][qi])]
        if curve_id is not None and definition['curve_id']!=curve_id:
            return self._result('pending_original_query_named_curve_differs',**base)
        point=a['query_uv'][qi];normal=a['query_normal'][qi];interval=a['query_interval'][qi]
        if not all(np.isfinite(v).all() for v in (point,normal,interval)) or interval[0]>=interval[1]:
            return self._result('pending_original_query_numeric_support_missing',**base)
        return self._result('original_source_query_literal_authority_supported',uv=point,normal=normal,
            interval=interval,query_index=query,curve_id=definition['curve_id'],
            operator=definition['operator'],material_signal=definition['material_signal'],
            material_transition=definition.get('material_transition'),
            original_candidate_prefix_count=prefix,
            source_query_overlay_frame_index=frame,source_query_overlay_candidate_index=index,
            source_query_overlay_query_index=qi,**extra,**base)

    def verify_unchanged(self):
        for reference in (self.source_pack_reference,self.reference,
                self.measurement_cache_reference,self.programme_manifest_reference):
            if reference is not None:authenticate_source_reference(reference)
