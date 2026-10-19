"""Fixed group/shot native geometry objective for physical initialization only."""
from pathlib import Path
import json
import numpy as np
from .io_utils import file_sha256,write_json

def compile_geometry_initialization(parent_packet,output,shot_total_weight=1.0):
    src=Path(parent_packet);packet=json.loads(src.read_text(encoding='utf8'));out=Path(output)
    if out.exists():raise ValueError('new initialization generation required')
    out.mkdir(parents=True)
    cache={}
    def locate(ref):
        path=Path(ref['file']);path=path if path.is_absolute() else src.parent/path
        key=(str(path),ref['sha256'])
        if key not in cache:
            if file_sha256(path)!=ref['sha256']:raise ValueError('changed packet dependency')
            cache[key]=True
        return path
    arrays={k:{**v,'file':str(locate(v))} for k,v in packet['arrays'].items()}
    shots=np.fromfile(arrays['shot_index']['file'],arrays['shot_index']['dtype'])
    start=packet['new_source_start'];original=[];new=[];counts={}
    for g in packet['groups']:
        if all(r<start for r in g['rows']):
            original.append({**g,'W':{**g['W'],'file':str(locate(g['W']))}});continue
        if any(r<start for r in g['rows']):raise ValueError('mixed original/new statistical unit')
        shot_values=np.unique(shots[g['rows']])
        if len(shot_values)!=1:raise ValueError('source group crosses shots')
        shot=int(shot_values[0]);counts[shot]=counts.get(shot,0)+1;new.append((g,shot))
    groups=list(original);spectrum=[];rank=packet['current_original_W_rank'];bundle=out/'geometry_W.f64';wstream=bundle.open('wb')
    for i,(g,shot) in enumerate(new):
        ref=g['W'];path=locate(ref);offset=ref.get('offset_bytes',0)
        W=np.memmap(path,dtype=ref['dtype'],mode='r',offset=offset,shape=tuple(ref['shape']))
        s=np.linalg.svd(W,compute_uv=False);cov=1/(s*s)
        spectrum.append({'id':g['id'],'shot_id':packet['shot_ids'][shot],'rows':len(g['rows']),'rank':g['rank'],
            'W_singular_min':float(s.min()),'W_singular_max':float(s.max()),'W_condition':float(s.max()/s.min()),
            'supported_covariance_eigen_min_px2':float(cov.min()),'supported_covariance_eigen_max_px2':float(cov.max()),
            'source_W':{**ref,'file':str(path)}})
        m=len(g['rows']);scale=float(np.sqrt(shot_total_weight/(m*counts[shot])))
        geom=np.eye(2*m,dtype='<f8')*scale;offset=wstream.tell();geom.tofile(wstream)
        groups.append({**g,'rank':2*m,'W':{'file':bundle.name,'sha256':'pending_closed_bundle','offset_bytes':offset,'dtype':'<f8','shape':list(geom.shape)},
            'statistic_kind':'initialization_surrogate','original_numerical_scheme':'fixed_group_and_shot_native_pixel_geometry',
            'source_local_W':{**ref,'file':str(path)},'source_local_W_used':False,
            'geometry_weight':{'formula':'W_g = sqrt(shot_total_weight / (rows_in_group * groups_in_shot)) * I_2m',
                'rows_in_group':m,'groups_in_shot':counts[shot],'shot_total_weight':shot_total_weight,'scale':scale},
            'noise_calibration_status':'not_a_noise_likelihood; physical_camera_initialization_only'})
        rank+=2*m
    wstream.close();bundle_sha=file_sha256(bundle)
    for g in groups[len(original):]:g['W']['sha256']=bundle_sha
    result={**packet,'arrays':arrays,'groups':groups,'joint_statistics_rank':rank,
        'new_source_statistics_rank':rank-packet['current_original_W_rank'],
        'new_source_statistics_status':'fixed group and shot native geometry initialization; original local pixel W retained separately',
        'initialization_objective_only':True,'physical_noise_likelihood_claimed':False,
        'formal_new_source_statistical_optimum_claimed':False,
        'parent_numerical_packet':{'file':str(src),'sha256':file_sha256(src)},
        'source_pixel_W_spectra':{'file':'source_pixel_W_spectrum.json'},
        'source_pixels_admission_or_auxiliaries_changed':False}
    write_json(out/'source_pixel_W_spectrum.json',spectrum)
    result['source_pixel_W_spectra']['sha256']=file_sha256(out/'source_pixel_W_spectrum.json')
    write_json(out/'packet.json',result)
    cv=np.asarray([x['supported_covariance_eigen_min_px2'] for x in spectrum]);wm=np.asarray([x['W_singular_max'] for x in spectrum])
    report={'schema':'rinkfit.fixed_group_shot_native_geometry_initialization.v1','original_s01_W_unchanged':True,
        'source_mean_four_aux_admission_or_role_changed':False,'new_groups':len(new),'new_fit_rows':sum(len(g['rows']) for g,_ in new),
        'group_count_by_shot':{packet['shot_ids'][s]:c for s,c in counts.items()},'shot_total_weight':shot_total_weight,
        'joint_statistics_rank':rank,'cov_min_quantiles_px2':np.quantile(cv,[0,.01,.1,.5,.9,1]).tolist(),
        'W_max_singular_quantiles':np.quantile(wm,[0,.01,.1,.5,.9,1]).tolist(),
        'source_groups_cov_min_below_1e_10_px2':int(np.sum(cv<1e-10)),
        'spectrum_extremes':sorted(spectrum,key=lambda x:x['supported_covariance_eigen_min_px2'])[:20],
        'packet_sha256':file_sha256(out/'packet.json'),'formal_likelihood_claimed':False}
    write_json(out/'report.json',report);return report
