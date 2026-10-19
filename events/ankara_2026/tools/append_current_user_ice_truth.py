"""Bind the registered 10 October ice corrections to existing truth factors.

Copies the current authenticated packet prefix exactly. The new UV come only
from the latest user raster, never from a camera, detector, or fitted ellipse.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,math
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[3]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/current_user_ice_corrections/truth_append'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_speed_candidate_20261010.json'
USER=ROOT/'events/ankara_2026/model_constraints/user_ice_vision_supervision_20261010.json'
CAMERA=DATA/'calibration/runs/solver_acceleration_20261003/okada_full9_current_review71_clarabel_20261009_152520_411/complete_physical_camera_model_UNACCEPTED_5e381af28c8fb955.npz'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(name,obj):
    p=OUT/name;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');return p
def rowsha(row):return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    options=read(CONFIG)['source_options']; user=read(USER)
    keys={'ice':'--additional-ice-source','human_authority':'--direct-human-truth-authority',
        'full_guard':'--direct-human-full-stroke-guard','stroke_caps':'--direct-human-stroke-cap-support',
        'location':'--practical-human-location-envelope','confirmed_authority':'--user-confirmed-source-truth-authority'}
    old={k:read(options[v]) for k,v in keys.items()}; packets=deepcopy(old)
    with np.load(old['full_guard']['guard_arrays_reference']['path'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    with np.load(CAMERA,allow_pickle=False) as z:
        aliases=dict(zip(z['pts'].tolist(),z['exposure_representative_pts'].tolist()))
    source_template=next(r for r in old['ice']['rows'] if r['id']=='user_complete_yellow_F2_outer_native229989888')
    ident=source_template['id']; templates={k:next(r for r in old[k]['records'] if r['id']==ident)
        for k in ('human_authority','full_guard','stroke_caps','location')}
    prepared=[]
    # Keep three observed branches. Both F2 branches share one FIT curve budget.
    for revision in user['revisions']:
        pts=revision['source_binding']['source_pts'];actual=int(aliases[pts])
        sample=next(s for s in user['samples'] if s['source_pts']==pts)
        batches=[]
        for annotation in revision['annotations']:
            reg=read(annotation['registration']['path']);assert sha(annotation['registration']['path'])==annotation['registration']['sha256']
            uv=np.asarray(annotation['uv']);normal=np.asarray(annotation['normal']);bounds=np.asarray(annotation['interval_px'])
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
            selected=np.unique(np.r_[0,np.searchsorted(arc,np.arange(31.,arc[-1],31.)),len(uv)-1]).astype(int)
            batches.append((annotation,reg,uv,normal,bounds,arc,selected))
        total_queries=sum(len(b[-1]) for b in batches)
        whole_uv=np.concatenate([b[2] for b in batches])
        all_ink=np.asarray(batches[0][1]['observed_user_ink_native_pixels'],float)
        global_nearest=cKDTree(whole_uv).query(all_ink)[1]
        branch_offset=0
        for annotation,reg,uv,normal,bounds,arc,selected in batches:
            gid=annotation['group_id'];i=len(packets['human_authority']['records']);j=len(packets['ice']['rows']);prefix=f'record_{i:02d}'
            curve=(dict(kind='current_ice_segment',feature_id='lane3',segment_index=2)
                   if annotation['primitive']=='line' else dict(kind='house_ring',row=2,side='F',ring='outer'))
            attachment=reg['user_attachment'];assert sha(attachment['path'])==attachment['sha256']
            native=reg['native_original_RGB'];assert sha(native['path'])==native['sha256']
            registration=annotation['registration'];unc=float(reg['p95_registration_error_native_px'])
            basis=np.asarray(reg['source_view_from_attachment_affine'],float)
            basis[:,2]-=np.asarray(reg['parent_view_from_native_translation'])
            # Registration stores a native-image crop/header translation.
            # Pixel-cell offsets only require the independently fitted linear part.
            belongs=(global_nearest>=branch_offset)&(global_nearest<branch_offset+len(uv))
            ink=all_ink[belongs];nearest=global_nearest[belongs]-branch_offset
            branch_offset+=len(uv)
            cell_offsets=np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])@basis[:,:2].T
            raw_bands=[]
            for vertex in range(len(uv)):
                pixels=ink[nearest==vertex]
                if not len(pixels):raise ValueError('Registered vertex has no original raster pixel cells')
                cells=(pixels[:,None,:]+cell_offsets).reshape(-1,2)
                signed=(cells-uv[vertex])@normal[vertex]
                raw_bands.append([signed.min(),signed.max()])
            raw_bands=np.asarray(raw_bands)
            bounds=raw_bands+np.array([-unc,unc])
            for key,value in {'uv':uv,'normal':normal,'normal_interval_px':bounds,
                'annotation_stroke_normal_interval_px':raw_bands,'source_arc_length_px':arc}.items():
                arrays[prefix+'_'+key]=value.copy()
            alias=dict(ref(CAMERA),fields_read=['pts','exposure_representative_pts'],camera_projection_parameters_read=False,
                requested_native_pts=pts,actual_exposure_pts=actual,presentation_and_representative_RGB_byte_identity_not_claimed=True)
            common=dict(id=gid,pts=pts,actual_pts=actual,curve=curve,
                geometry_role='actual_user_confirmed_current_paint_carrier',source_raw_frame=native['path'],
                source_file_sha256=native['sha256'],source_annotation_registration=registration,
                source_annotation_attachment=attachment,source_exposure_alias_authority=alias,
                source_user_statement=reg['user_statement'])
            row=deepcopy(source_template);row.update(common,feature_id=curve.get('feature_id','F2'),
                uv=uv[selected].tolist(),normal=normal[selected].tolist(),normal_interval_px=bounds[selected].tolist(),
                sigma_px=[6.*math.sqrt(total_queries)]*len(selected),source_working_sigma_before_stroke_correlation_px=[6.]*len(selected),
                source_stroke_correlation_group=annotation['physical_curve_id']+'_'+str(pts),
                source_stroke_correlation_query_count=total_queries,
                sigma_semantics='Existing 6px source working scale times sqrt(total sparse queries across the same physical user curve); not measured drawing precision',
                source_human_authority='Latest direct user registered source drawing',
                source_sampling=dict(method='Existing 31 native-pixel arc coverage; every query is an unchanged registered user vertex',
                    native_optical_window_px=31.,selected_trace_indices=selected.tolist(),selected_sample_count=len(selected),
                    raw_source_trace_sample_count=len(uv),source_arc_length_native_px=float(arc[-1]),point_count_cap_used=False,
                    whole_stroke_pixels_not_assumed_independent=True),
                source_normal_support_authority=annotation['support_origin'])
            row['source_support_summary'].update(registration_error_native_px=unc,
                declared_support_type='Latest registered user raster extent plus actual registration p95; manual error remains unquantified')
            packets['ice']['rows'].append(row)
            auth=deepcopy(templates['human_authority']);auth.update(common,original_row_index=j,
                source_row_canonical_json_sha256=rowsha(row),uv=row['uv'],normal=row['normal'],normal_interval_px=row['normal_interval_px'],
                lower_bound_px=bounds[selected,0].tolist(),upper_bound_px=bounds[selected,1].tolist(),historical_sigma_px=row['sigma_px'],
                annotation_stroke_normal_interval_px=raw_bands[selected].tolist(),registration_uncertainty_native_px=[unc]*len(selected),
                registration_uncertainty_kind='independent_RGB_registration_inlier_p95_native_pixels',
                authority_temporal_scope=dict(native_pts_start_inclusive=pts,native_pts_stop_inclusive=pts,propagation_to_other_exposures_is_GT=False),
                human_semantic_authority=reg['user_statement'],original_sampling=deepcopy(row['source_sampling']),
                complete_human_stroke=dict(reference=registration,native_coordinates_field='per_column_observed_user_branches',
                    sample_count=len(uv),native_bbox_xyxy=[*uv.min(0).tolist(),*uv.max(0).tolist()],
                    all_digitized_pixels_not_independent=True,full_extent_between_support_queries_authority=True))
            auth['envelope_provenance'].update(method=annotation['support_origin'],source_reading_receipt=registration,
                source_support_item_indices=selected.tolist(),exact_annotation_BGR=[[0,255,255]]*len(selected))
            packets['human_authority']['records'].append(auth)
            guard=deepcopy(templates['full_guard']);guard.update(common,authority_record_index=i,
                authority_source_row_canonical_json_sha256=rowsha(row),vertex_count=len(uv),array_prefix=prefix,
                native_from_screenshot_affine=basis.tolist(),registration_uncertainty_native_px=unc,
                registration_uncertainty_kind=auth['registration_uncertainty_kind'],trace_origin=annotation['support_origin'],
                source_normal_recipe='Normals of unchanged registered observed user branch',
                signed_colour_support_recipe=annotation['support_origin'],normal_band_span_min_px=float(np.diff(bounds,axis=1).min()),
                normal_band_span_max_px=float(np.diff(bounds,axis=1).max()),native_bbox_xyxy=auth['complete_human_stroke']['native_bbox_xyxy'])
            packets['full_guard']['records'].append(guard)
            caps=[]
            for end,index in [('first',0),('last',len(uv)-1)]:
                pixels=ink[nearest==index]
                if not len(pixels):raise ValueError('Drawing cap has no original raster cells')
                cells=(pixels[:,None,:]+cell_offsets).reshape(-1,2)
                tangent=np.array([-normal[index,1],normal[index,0]])
                coords=(cells-uv[index])@np.array([normal[index],tangent]).T
                lo=coords.min(0);hi=coords.max(0)
                cap=deepcopy(templates['stroke_caps']);cap.update(common,source_full_guard_record_index=i,
                    source_vertex_index=index,stroke_end=end,uv=uv[index].tolist(),normal=normal[index].tolist(),tangent=tangent.tolist(),
                    normal_cell_support_px=[float(lo[0]-unc),float(hi[0]+unc)],tangent_cell_support_px=[float(lo[1]-unc),float(hi[1]+unc)],
                    unexpanded_coloured_pixel_cell_normal_span_px=[float(lo[0]),float(hi[0])],
                    unexpanded_coloured_pixel_cell_tangent_span_px=[float(lo[1]),float(hi[1])],
                    registration_uncertainty_native_px=unc,registration_uncertainty_kind=auth['registration_uncertainty_kind'],
                    selected_actual_coloured_pixel_count=len(pixels),selection='Original registered raster cells nearest the unchanged observed branch cap; no physical object endpoint',
                    source_colour_connected_component='latest_registered_yellow_ink')
                cap.pop('selected_screenshot_bbox_xyxy',None)
                packets['stroke_caps']['records'].append(cap);caps.append(cap)
            loc=deepcopy(templates['location']);loc.update(id=gid,full_vertices=len(uv),
                actual_initial_normal_outside_raster_px=None,actual_initial_tangent_outside_source_footprint_px=None,
                normal_expansion_px=max(0.,6.-float(np.minimum(-bounds[:,0],bounds[:,1]).min())),
                tangent_expansion_px=max(0.,6.-min(min(-c['tangent_cell_support_px'][0],c['tangent_cell_support_px'][1]) for c in caps)),
                float64_operation_allowance_px=float(128*np.finfo(float).eps*np.max(abs(uv))))
            packets['location']['records'].append(loc)
            prepared.append(dict(id=gid,source_pts=pts,actual_pts=actual,physical_curve_id=annotation['physical_curve_id'],
                sparse_queries=len(selected),complete_user_vertices=len(uv),shared_physical_curve_query_count=total_queries,
                raster_support_is_not_subpixel_truth=True,inherited_practical_location_scale_px=6.))
    ice=packets['ice'];ice.update(observation_count=len(ice['rows']),fit_observation_count=sum(r['split']=='fit' for r in ice['rows']),
        current_fit_scalar_count=sum(2*len(r['uv']) for r in ice['rows'] if r['split']=='fit'))
    # The exclusion manifest authenticates its complete raw container. Appending
    # user rows creates a new immutable container, with the same exclusion facts.
    qualification=read(ice.pop('source_only_review_qualification')['path'])
    rawpath=save('native_ice_raw_parent79_plus_latest_user3.json',ice)
    qualification['original_packet']=ref(rawpath)
    qualpath=save('native_ice_same15_exclusions_parent79_plus_latest_user3.json',qualification)
    ice['source_only_review_qualification']=ref(qualpath)
    icepath=save('native_ice_qualified_parent79_plus_latest_user3.json',ice)
    auth=packets['human_authority']
    for r in auth['records'][len(old['human_authority']['records']):]:r['packet_reference']=ref(icepath)
    auth.update(record_count=len(auth['records']),native_UV_count=sum(len(r['uv']) for r in auth['records']))
    authpath=save('human_parent50_plus_latest_user3.json',auth)
    arraypath=OUT/'human_parent50_plus_latest_user3_full_guard_arrays.npz';np.savez_compressed(arraypath,**arrays)
    guard=packets['full_guard'];guard.update(primary_authority_reference=ref(authpath),guard_arrays_reference=ref(arraypath),
        record_count=len(guard['records']),full_stroke_vertex_count=sum(r['vertex_count'] for r in guard['records']),primary_UV_count=auth['native_UV_count'])
    guardpath=save('human_parent50_plus_latest_user3_full_guard.json',guard)
    caps=packets['stroke_caps'];caps.update(primary_authority_reference=ref(authpath),full_trace_reference=ref(guardpath),
        full_trace_arrays_reference=ref(arraypath),endpoint_record_count=len(caps['records']))
    cappath=save('human_parent50_plus_latest_user3_caps.json',caps)
    packets['location']['authority_references']['native']=ref(authpath)
    locpath=save('human_parent50_plus_latest_user3_location.json',packets['location'])
    confirmed=packets['confirmed_authority']
    for record in confirmed['records']:
        if record['family']=='native_ice_curve':
            prior=read(record['packet_reference']['path'])['rows'][record['original_row_index']]
            assert prior==ice['rows'][record['original_row_index']]
            record['packet_reference']=ref(icepath)
    confirmedpath=save('confirmed_same30_rebound_latest_user3.json',confirmed)
    for name in ('ice','human_authority','full_guard','stroke_caps','location'):
        key='rows' if name=='ice' else 'records'
        assert packets[name][key][:len(old[name][key])]==old[name][key]
    with np.load(old['full_guard']['guard_arrays_reference']['path'],allow_pickle=False) as z:
        assert all(z[k].dtype==arrays[k].dtype and z[k].shape==arrays[k].shape and z[k].tobytes()==arrays[k].tobytes() for k in z.files)
    receipt=save('latest_user_ice_truth_append_receipt.json',dict(schema='rinkspace.current-user-ice-truth-append.v1',
        source_config=ref(CONFIG),registered_user_source=ref(USER),new_groups=prepared,
        paths={k:ref(p) for k,p in dict(ice=icepath,human_authority=authpath,full_guard=guardpath,guard_arrays=arraypath,
            stroke_caps=cappath,location=locpath,confirmed_authority=confirmedpath,board=Path(options['--additional-board-source'])).items()},
        exact_old_prefixes_preserved=True,old_full_guard_arrays_byte_exact=True,new_total_truth_groups=85,
        old_limits_relaxed=False,model_or_network_generated_UV=False,actual_factor_instantiation_pending=True))
    print(json.dumps(dict(receipt=str(receipt),new_groups=prepared)))

if __name__=='__main__':main()
