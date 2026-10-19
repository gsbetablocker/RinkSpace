"""All requested own-image gray-source qualification; no UV addition or fit."""
from __future__ import annotations
import argparse,hashlib,json,sys,shutil,time,sqlite3,pickle,zlib
from pathlib import Path
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.gray_source_review_pack_headers_v2 import read_gray_source_review_headers
from calibration.program_tracking.gray_source_navigation_profile_review_v3 import review_named_gray_point_with_current_profiles
from calibration.program_tracking.gray_source_contour_review import _material_signature_intervals,_all_sample_pixels_supported
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from calibration.program_tracking.projection import project_fast
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.far_rail_source_components import source_exclusion
from calibration.program_tracking.strict_source_named_identity_filter_with_exclusions import FILTER_SCHEMA
from rinkspace.programme_source_stream import FrozenProgrammeSource
from rinkspace.vision_programme_ice_review import ice_curve_inventory
from rinkspace.vision_source_shapes import _MaterialProfileFrame
from rinkspace.vision_ice_model_context_identity import _actual_paint_colour_context


def ref(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=file_sha(path))


def literal(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',nargs=2,metavar=('PATH','SHA256'),required=True)
    p.add_argument('--material-id',required=True);p.add_argument('--curve-id',required=True)
    p.add_argument('--model-identity',nargs=2,metavar=('PATH','SHA256'),required=True)
    p.add_argument('--programme-manifest',type=Path,required=True)
    p.add_argument('--stem-facts',nargs=2,action='append',default=[],metavar=('PATH','SHA256'))
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--source-correction-database',nargs=2,metavar=('PATH','SHA256'))
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
    receipt_path=dest/'full_window_current_source_profile_review.json'
    if receipt_path.exists()or(dest/'point_review_progress.jsonl').exists():raise FileExistsError('Preserve prior source review')
    source_ref=dict(path=str(Path(args.pack[0]).resolve()),sha256=args.pack[1]);authenticate_source_reference(source_ref)
    model_ref=dict(path=str(Path(args.model_identity[0]).resolve()),sha256=args.model_identity[1]);authenticate_source_reference(model_ref)
    # Bind exact startup Python bytes. Original live files are never authorities
    # of this output; imported relevant bytes are checked again at completion.
    authority=dest/'exact_python_authority';implementation=[]
    for directory in ('src','calibration','scripts'):
        for original in (ROOT/directory).rglob('*.py'):
            target=authority/original.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target)
    for name in ('calibration/program_tracking/gray_source_navigation_profile_review_v3.py',
            'calibration/program_tracking/gray_source_contour_review.py','calibration/program_tracking/gray_source_review_pack_headers_v2.py',
            'src/rinkspace/vision_source_shapes.py','src/rinkspace/vision_ice_model_context_identity.py',
            'src/rinkspace/vision_programme_ice_review.py',
            'scripts/diagnostics/review_gray_source_profiles_full_window_v4.py'):
        implementation.append(ref(authority/name))
    with np.load(source_ref['path'],allow_pickle=False)as z:
        values={k:z[k]for k in ('uv','source_sides_uv','source_side_quantiles_uv','sigma_px',
            'original_point_indices','source_proof_template_index','row_offsets','source_frame_actual_pts',
            'source_frame_native_pts','source_frame_BGR_sha256')}
        metadata=json.loads(str(z['metadata_json'].item()))
    headers,_=read_gray_source_review_headers(source_ref['path']);rows=[];by_actual=defaultdict(list)
    if args.source_correction_database:
        correction_ref=dict(path=str(Path(args.source_correction_database[0]).resolve()),sha256=args.source_correction_database[1])
    elif args.source_dir:
        spill_path=args.source_dir/'lossless_source_state/source_state_spill_receipt.json'
        correction_ref=json.loads(spill_path.read_text(encoding='utf8'))['correction_database']
    else:raise ValueError('Original production query correction database is required')
    authenticate_source_reference(correction_ref)
    correction_db=sqlite3.connect(Path(correction_ref['path']).as_uri()+'?mode=ro',uri=True)
    for ri,h in enumerate(headers):
        if h['physical_material_id']!=args.material_id:continue
        if h['scope']!='ice'or h['model_target'].get('curve_id')!=args.curve_id:
            raise ValueError('Selected material and named ice NAV target differ')
        rows.append(ri);by_actual[h['actual_pts']].append(ri)
    argv=json.loads(args.programme_manifest.read_text(encoding='utf8'))['common_arguments']
    option=lambda name:argv[argv.index(name)+1]
    programme=FrozenProgrammeSource(args.model_identity[0],option('--source'),ROOT/'calibration/program_tracking/source_shots.json')
    if programme.actual_inventory.tolist()!=metadata['expected_actual_pts']:raise ValueError('Programme clock shortened')
    inventory=ice_curve_inventory(programme.scene);curve=next(c for c in inventory if c['curve_id']==args.curve_id)
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    # Source-only scope utilities are the already used public producer code.
    import scripts.admit_programme_source_inputs as public_source
    negative_refs=[ref(argv[i+1])for i,name in enumerate(argv)if name=='--known-source-material-exclusion']
    known_negative=public_source.load_known_source_material_exclusions(negative_refs)
    manual_keep=defaultdict(set);manual_pending=defaultdict(set);palette_lo=[];palette_hi=[];fact_refs=[]
    for path,sha in args.stem_facts:
        fact_ref=dict(path=str(Path(path).resolve()),sha256=sha);authenticate_source_reference(fact_ref);fact_refs.append(fact_ref)
        fact=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        for record in fact.get('records',[]):
            if 'review'in record:
                binding=record['review']['source_binding'];sr=record['source_pack_reference']
                globals_=record['original_global_point_indices'];trusted=[globals_[i]for path in record['review']['source_stem_witness']['trusted_current_stem_paths']for i in path]
                rawref=record['raw_reference']
                known=[globals_[i]for i in record['review']['pending_original_local_point_indices']]if record.get('cut')in ('001','003')else[]
            else:
                if 'trusted_current_global_stem_indices'not in record:continue
                sr=record['source_pack_reference'];trusted=record['trusted_current_global_stem_indices'];known=record.get('current_source_competition_global_indices_for_pending_review',[])
                binding=dict(actual_pts=record['actual_pts'],decoded_native_BGR_sha256=record['decoded_native_BGR_sha256'])
                rawref=next(r for r in record['source_review_references']if Path(r['path']).name.endswith('_EXACT_RAW_NATIVE.png'))
            if not trusted:continue
            authenticate_source_reference(rawref)
            authenticate_source_reference(sr)
            raw=cv2.imread(rawref['path']);rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
            if hashlib.sha256(raw.tobytes()).hexdigest()!=binding['decoded_native_BGR_sha256']:raise ValueError('Stem own RAW hash changed')
            with np.load(sr['path'],allow_pickle=False)as z:ss=z['source_sides_uv'][trusted];uv=z['uv'][trusted]
            width=np.linalg.norm(ss[:,1]-ss[:,0],axis=1);normal=(ss[:,1]-ss[:,0])/width[:,None]
            readable,colours,_,_=_actual_paint_colour_context(rgb,uv,ss,normal,'dark',native_rgb_field=rgb.astype(np.float32))
            lo,hi=_material_signature_intervals(colours);ok=readable&np.isfinite(lo).all((1,2))&np.isfinite(hi).all((1,2))
            palette_lo.extend(lo[ok]);palette_hi.extend(hi[ok])
            if (record.get('trusted_pixels_supply_palette_only')is not True
                    and Path(sr['path']).resolve()==Path(source_ref['path'])and sr['sha256']==source_ref['sha256']):
                manual_keep[binding['actual_pts']].update(trusted);manual_pending[binding['actual_pts']].update(known)
    if not palette_lo:raise ValueError('A current independently witnessed actual gray material palette is required')
    palette=[np.asarray(palette_lo).min(0),np.asarray(palette_hi).max(0)]
    requested=metadata['processed_actual_pts'];requested_set=set(requested);processed=[];pending_entries=[];counts=Counter();started=time.perf_counter()
    frame_lookup={int(a):(int(n),b.decode('ascii')if isinstance(b,(bytes,np.bytes_))else str(b))for a,n,b in zip(values['source_frame_actual_pts'],values['source_frame_native_pts'],values['source_frame_BGR_sha256'])}
    progress_path=dest/'point_review_progress.jsonl'
    with progress_path.open('x',encoding='utf8')as progress:
        for frame in programme.frames(start_pts=min(requested),stop_pts=max(requested)+1,cpu_threads=1):
            actual=int(frame['actual_pts']);native=int(frame['native_pts'])
            if not frame['representative']or actual not in requested_set:continue
            processed.append(actual);raw=frame['raw_bgr'];rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB);binding=dict(frame['source_binding'],native_pts=native)
            if actual in frame_lookup and (native,hashlib.sha256(raw.tobytes()).hexdigest())!=frame_lookup[actual]:raise ValueError('Own actual RGB differs from source pack')
            if actual not in by_actual:continue
            payloads=correction_db.execute('SELECT payload,sha256 FROM corrections WHERE pts=?',(actual,)).fetchall()
            if len(payloads)!=1:raise ValueError('One exact original source query packet is required')
            blob,digest=payloads[0]
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original source query packet bytes changed')
            original_packet=pickle.loads(zlib.decompress(blob))['measurement']
            if (original_packet['source_binding']['actual_pts']!=actual
                or original_packet['decoded_native_RGB_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest()):
                raise ValueError('Original production query belongs to another actual RGB')
            union,_=masks.static_scene_exclusion(native);excluded=source_exclusion(raw,union)
            scope_readout=public_source.measure_board_source_frame(raw,excluded,binding)
            bands=scope_readout.get('ice_scope_boundary_candidates',scope_readout['boundary_candidates'])
            valid=~public_source.conservative_ice_exclusion(excluded,bands)
            valid&=~public_source.exact_source_material_exclusion(binding,known_negative)
            xyz=curve['xyz'];nav,depth=project_fast(frame['camera25'],xyz)
            good=np.isfinite(nav).all(1)&(depth>.05);indices=np.flatnonzero(good);runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            navpaths=[nav[run]for run in runs if len(run)>=2]
            mf=_MaterialProfileFrame(rgb,binding,hashlib.sha256(rgb.tobytes()).hexdigest());native_field=mf.rgb.astype(np.float32)
            frame_pending=0;frame_supported=0
            for ri in by_actual[actual]:
                h=headers[ri];lo,hi=map(int,values['row_offsets'][ri:ri+2]);unc=h.get('source_readout_uncertainty')or{}
                quant=unc.get('source_native_pixel_quantization_step_px',unc.get('quantization_step_px'))
                interp=unc.get('source_interpolation_grid_px',unc.get('interpolation_grid_px'))
                excluded_points=[];categories=[];reasons=[]
                for g in range(lo,hi):
                    original_id=int(values['original_point_indices'][g]);original_query=int(original_packet['candidate_query'][original_id])
                    if not np.array_equal(original_packet['candidate_side_quantiles_uv'][original_id],values['source_side_quantiles_uv'][g]):
                        raise ValueError('Original source q locators differ from the complete saved packet')
                    center_query=dict(uv=original_packet['query_uv'][original_query],
                        normal=original_packet['query_normal'][original_query],interval=original_packet['query_interval'][original_query])
                    if g in manual_pending[actual]:result=dict(status='pending_independently_reviewed_actual_competing_gray_branch',native_quantization_step_px=quant,source_interpolation_grid_px=interp)
                    elif quant is None or interp is None or not np.isfinite([quant,interp]).all()or min(quant,interp)<=0:result=dict(status='pending_original_native_quantization_or_interpolation_authority_missing')
                    elif not navpaths:result=dict(status='pending_current_named_material_NAV_unavailable')
                    else:
                        path=min(navpaths,key=lambda p:float(np.min(np.linalg.norm(p-values['uv'][g],axis=1))))
                        result=review_named_gray_point_with_current_profiles(mf,values['uv'][g],values['source_sides_uv'][g],values['source_side_quantiles_uv'][g],source_valid_mask=valid,navigation_polyline=path,native_quantization_step_px=quant,source_interpolation_grid_px=interp,navigation_sampling_spacing_px=24.,navigation_normal_search_radius_px=24.,source_neighbor_navigation_footprint_px=80.,named_gray_material_signature_bounds=palette,native_rgb_field=native_field,original_center_source_query=center_query)
                    if g in manual_keep[actual]and g not in manual_pending[actual]:
                        ss=values['source_sides_uv'][g:g+1];vv=values['uv'][g:g+1];qq=values['source_side_quantiles_uv'][g:g+1]
                        width=np.linalg.norm(ss[:,1]-ss[:,0],axis=1)
                        normals=np.divide(ss[:,1]-ss[:,0],width[:,None],out=np.zeros((1,2)),where=width[:,None]>0)
                        readable,_,_,positions=_actual_paint_colour_context(mf.rgb,vv,ss,normals,'dark',native_rgb_field=native_field)
                        hard=(quant is not None and interp is not None and np.isfinite(qq).all()
                            and np.array_equal(vv,ss.mean(1))and np.array_equal(ss,qq[:,:,1])
                            and bool(readable.all())and bool(_all_sample_pixels_supported(valid,positions,normals).all()))
                        if hard:
                            result=dict(result,status='independently_witnessed_own_current_named_gray_source_point_preserved',auxiliary_profile_pending_does_not_revoke_exact_current_source_witness=True)
                        else:result=dict(result,status='pending_current_witness_original_q50_native_material_or_hard_source_scope_missing')
                    status=result['status'];counts[status]+=1
                    record=dict(pack_row_index=ri,pack_point_index=g,actual_pts=actual,native_pts=native,
                        original_candidate_index=int(values['original_point_indices'][g]),split=h['split'],source_binding=binding,
                        original_source_query_index=original_query,original_source_query_packet_sha256=digest,
                        own_immutable_RGB_sha256=mf.digest,own_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        native_rgb_field_rebuilt_once_from_this_immutable_native_image=True,**result)
                    progress.write(json.dumps(record,ensure_ascii=False,default=literal)+'\n')
                    if status.startswith('pending_'):
                        frame_pending+=1
                        if h['split']=='fit':excluded_points.append(g);reasons.append(status);categories.append('actual_wrong_branch'if g in manual_pending[actual]else'unresolved_current_source_profile_precision_or_continuation')
                    else:frame_supported+=1
                if excluded_points:
                    local=np.asarray(excluded_points)-lo;entry=dict(pack_row_index=ri,pack_point_indices=excluded_points,
                        physical_material_id=h['physical_material_id'],source_binding=binding,
                        original_candidate_indices=values['original_point_indices'][excluded_points].tolist(),
                        exclude_from_final_named_FIT=True,status='source_material_measured_named_identity_pending',
                        reason='; '.join(sorted(set(reasons))),source_guard_point_statuses=reasons,
                        pending_categories=categories,entity_identity_revoked=False)
                    if 'source_query_indices' in h:
                        entry['source_query_indices']=np.asarray(h['source_query_indices'])[local].tolist()
                    pending_entries.append(entry)
            progress.flush()
            print(json.dumps(dict(actual_pts=actual,processed_actual=len(processed),requested_actual=len(requested),
                current_source_supported=frame_supported,pending=frame_pending,elapsed_seconds=time.perf_counter()-started)),flush=True)
    if processed!=requested:raise ValueError('Requested exact source exposures were skipped')
    for reference in implementation:
        original=ROOT/Path(reference['path']).relative_to(authority)
        if file_sha(original)!=reference['sha256']:raise ValueError('Imported Python changed during review; original snapshot preserved')
    if file_sha(source_ref['path'])!=source_ref['sha256']:raise ValueError('Original source pack changed')
    authenticate_source_reference(correction_ref);correction_db.close()
    receipt=dict(schema='rinkspace.all-requested-current-gray-source-profile-review.v1',source_pack_reference=source_ref,
        model_identity_reference=model_ref,material_id=args.material_id,curve_id=args.curve_id,
        original_production_query_database_reference=correction_ref,
        exact_python_authority_references=implementation,actual_source_stem_fact_references=fact_refs,
        known_actual_source_negative_material_references=negative_refs,
        expected_programme_actual_exposures=len(metadata['expected_actual_pts']),processed_actual_pts=processed,
        selected_material_rows=len(rows),selected_material_points=sum(headers[i]['point_count']for i in rows),
        status_point_counts=dict(counts),pending_named_fit_points=sum(len(e['pack_point_indices'])for e in pending_entries),
        native_quantization_and_interpolation_are_separate=True,point_evidence_reference=ref(progress_path),
        original_UV_sigma_quartiles_sides_proof_locators_and_HELD_unchanged=True,
        all_requested_exposures_checked=True,complete_independent_source_accuracy_holdout_claimed=False,
        new_UV_observations_created=0,source_FIT_admitted=False,elapsed_seconds=time.perf_counter()-started)
    with receipt_path.open('x',encoding='utf8')as output:json.dump(receipt,output,ensure_ascii=False,indent=2,default=literal)
    fact=dict(schema=FILTER_SCHEMA,retain_pending_source_readouts_full_literal_evidence=True,
        original_packs_or_alias_clock_evidence_must_not_be_changed=True,
        source_pack_groups=[dict(source_pack_reference=source_ref,pending_named_point_count=receipt['pending_named_fit_points'],
            source_review_witnesses=[ref(receipt_path),*fact_refs],entries=pending_entries)])
    fp=dest/'current_effective_source_pending_filter.json'
    with fp.open('x',encoding='utf8')as output:json.dump(fact,output,ensure_ascii=False,indent=2,default=literal)
    print(json.dumps(dict(receipt_reference=ref(receipt_path),pending_filter_reference=ref(fp),
        pending_named_fit_points=receipt['pending_named_fit_points'],status_point_counts=dict(counts)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
