"""Current native profile likelihood and exact discrete pixel influence transport.

Only current per-row material localizations are counted. Historical raw readings
are noise/navigation inputs. The frozen Gaussian/channel/bilinear/peak operator
is linearized in native RGB pixels and correlated within a whole source family.
"""
from __future__ import annotations
import argparse,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_whitening,StatisticsUnavailable

COLOR=np.array([[.2126,.7152,.0722],[1.,-1.,0.],[-.5,-.5,1.]])
OFF=np.arange(-3.,3.0001,.25)
G=np.exp(-np.arange(-3,4,dtype=float)**2/(2*.8**2));G/=G.sum()

def peak_jacobian(profile):
    y=np.asarray(profile,float);gradient=np.gradient(y,.25);values=abs(gradient)
    k=int(values.argmax())
    if k<2 or k>len(y)-3:raise StatisticsUnavailable('native profile peak touches finite search boundary')
    v=values[k-1:k+2];den=v[0]-2*v[1]+v[2];num=.5*(v[0]-v[2])
    if abs(den)<1e-12 or abs(num/den)>=1:
        raise StatisticsUnavailable('native peak interpolation is nonregular')
    dv=.25*(np.array([.5,0.,-.5])*den-num*np.array([1.,-2.,1.]))/(den*den)
    result=np.zeros(len(y))
    for j,a in enumerate(dv):
        at=k-1+j;s=np.sign(gradient[at]);result[at+1]+=a*s/.5;result[at-1]-=a*s/.5
    return result

def profile_shift_variance(intensity,tangent_offsets):
    y=np.asarray(intensity,float);mean=y.mean(axis=0);L=peak_jacobian(mean)
    shifts=(y-mean)@L;t=np.asarray(tangent_offsets,float)
    A=np.polynomial.polynomial.polyvander(t,2);fit=A@np.linalg.lstsq(A,shifts,rcond=None)[0]
    residual=shifts-fit;variance=float(residual@residual/(len(t)-np.linalg.matrix_rank(A)))
    if not np.isfinite(variance) or variance<0:raise StatisticsUnavailable('no measured local profile-shift variance')
    return variance,residual,L

def _reflect(i,n):
    while i<0 or i>=n:i=-i-1 if i<0 else 2*n-i-1
    return i

def influence(uv,normal,channel,L):
    result=defaultdict(float);q=np.asarray(uv,float);normal=np.asarray(normal,float)
    for off,derivative in zip(OFF,L):
      if derivative==0:continue
      xy=q+off*normal;u,v=np.floor(xy).astype(int);du,dv=xy-[u,v]
      for dx,dy,bilinear in [(0,0,(1-du)*(1-dv)),(1,0,du*(1-dv)),(0,1,(1-du)*dv),(1,1,du*dv)]:
        for gy,wy in enumerate(G):
          yy=_reflect(v+dy+gy-3,1080)
          for gx,wx in enumerate(G):
            xx=_reflect(u+dx+gx-3,1920);value=derivative*bilinear*wx*wy
            for color,coefficient in enumerate(COLOR[channel]):
              if coefficient:result[((yy*1920+xx)*3+color)]+=value*coefficient
    return dict(result)

def _sample(channels,uv,normal,channel):
    q=np.asarray(uv)[None,:]+OFF[:,None]*np.asarray(normal)[None,:]
    return map_coordinates(channels[channel],[q[:,1],q[:,0]],order=1,mode='nearest',prefilter=False)

def compile_pixel_statistics(audited_rows,profile_dir,output,*,roles=None):
    out=Path(output)
    if out.exists():raise ValueError('pixel statistics generation must be new')
    out.mkdir(parents=True)
    profile_dir=Path(profile_dir);pm=json.loads((profile_dir/'manifest.json').read_text(encoding='utf8'))
    if file_sha256(profile_dir/'native_profiles.npz')!=pm['arrays_sha256'] or file_sha256(profile_dir/'profile_index.jsonl.gz')!=pm['index_sha256']:
        raise ValueError('native profile noise inputs changed')
    with np.load(profile_dir/'native_profiles.npz',allow_pickle=False) as loaded:
        z={k:loaded[k] for k in loaded.files}
    noise={};family_noise=defaultdict(list);noise_fail=Counter();pending_pairs=defaultdict(list)
    for item in read_jsonl(profile_dir/'profile_index.jsonl.gz'):
        i=item['profile_index']
        try:
            var,residual,L=profile_shift_variance(z['intensity'][i],z['tangent_offsets'])
            rec={**item,'variance':var,'residual':residual,'profile_uv':z['uv'][i],
                 'normal':z['normal'][i],'channel':int(z['channel'][i])}
            pending_pairs[(item['source_family_id'],item['source_group_id'],item['raw_index'])].append(rec)
        except StatisticsUnavailable as e:noise_fail[str(e)]+=1
    for key,items in pending_pairs.items():
        rec=items[0]
        if len(items)==2 and {x['side'] for x in items}=={0,1}:
            residual=.5*(items[0]['residual']+items[1]['residual'])
            rec={**rec,'variance':float(residual@residual/2),'profile_uv':np.mean([x['profile_uv'] for x in items],axis=0),'two_flank_common_noise_preserved':True}
        if rec['variance']>=0:
            noise[key]=rec;family_noise[key[0]].append(rec)
    rows=list(read_jsonl(audited_rows));families=defaultdict(list)
    for r in rows:families[r['source_family_id']].append(r)
    roles_data={}
    if roles:
        with gzip.open(roles,'rt',encoding='utf8') as f:rd=json.load(f)
        rr=rd.get('rows',rd)
        roles_data={r['source_row_id']:r for r in rr} if isinstance(rr,list) else rr
    counts=Counter();decisions=[];registry={};measurements=[]
    cached_path=None;channels=None
    with gzip.open(out/'groups.jsonl.gz','wt',encoding='utf8') as gf:
      for ordinal,(family,members) in enumerate(sorted(families.items())):
        ready=[r for r in members if r.get('current_uv') is not None and not r.get('point_correspondence') and r.get('source_measurement_audit',{}).get('status') in {'current_native_material_interface','two_current_material_edges_midline','manual_native_click_preserved'}]
        source_ready=family_noise.get(family,[]);record={'source_group_id':family,'source_row_ids':[r['source_row_id'] for r in members]}
        if not ready or not source_ready:
            record.update(status='pending',reason='missing actual finite profile measurements or measured noise');decisions.append(record);counts['groups_pending']+=1;continue
        framepaths={r['native_png'] for r in ready}
        if len(framepaths)!=1:raise ValueError('source family spans native frames')
        png=Path(next(iter(framepaths)))
        if cached_path!=png:
            if file_sha256(png)!=ready[0]['native_png_sha256']:raise ValueError('native PNG changed')
            rgb=np.asarray(Image.open(png).convert('RGB'),dtype=np.float32)
            channels=gaussian_filter(np.einsum('hwc,kc->khw',rgb,COLOR),sigma=(0,.8,.8));cached_path=png
        tree=cKDTree(np.asarray([x['profile_uv'] for x in source_ready]));vectors=[];accepted=[];normal_list=[];hashes={};variance_records=[]
        for row in ready:
            rid=row['source_row_id'];measure=row['source_measurement_audit'];uv=np.asarray(row['current_uv'],float)
            nearest=source_ready[int(tree.query(uv)[1])];normal=np.asarray(measure.get('normal') or nearest['normal'],float)
            if normal.shape!=(2,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:counts['invalid_normals']+=1;continue
            normal/=np.linalg.norm(normal)
            role=roles_data.get(rid,{}).get('fit_role',row.get('fit_role','development'))
            if role not in {'development','training','historical_regression'}:counts['nonfit_role_rows']+=1;continue
            try:
                vec=defaultdict(float)
                if row.get('measurement_operator_resolved')=='paint_center':
                    flanks=measure.get('flank_uv',[]);profiles=measure.get('flank_profiles',[])
                    if len(flanks)!=2:raise StatisticsUnavailable('width center has no two current material flanks')
                    parts=[(flanks[j],int(profiles[j].get('channel',nearest['channel'])),.5) for j in range(2)]
                else:parts=[(uv,int(measure.get('profile',{}).get('channel',nearest['channel'])),1.)]
                for q,c,weight in parts:
                    L=peak_jacobian(_sample(channels,q,normal,c));part=influence(q,normal,c,L)
                    for key,value in part.items():vec[key]+=weight*value
                norm=float(np.sqrt(sum(v*v for v in vec.values())))
                if norm<=0:raise StatisticsUnavailable('source pixel operator has zero native influence')
                quantization_variance=norm*norm/12.
                empirical_excess=max(0.,nearest['variance']-quantization_variance)
                scale=np.sqrt(1/12.+empirical_excess/(norm*norm));vec={key:value*scale for key,value in vec.items() if value}
            except StatisticsUnavailable as e:counts['nonregular_peak_rows']+=1;continue
            accepted.append(row);vectors.append(vec);normal_list.append(normal)
            hashes[rid]=object_sha256({'native_pts':row['native_pts'],'current_uv':row['current_uv'],'source_row_id':rid,'measurement_operator':row['measurement_operator_resolved']})
            variance_records.append({'source_row_id':rid,'variance_px2':quantization_variance+empirical_excess,'empirical_variance_px2':nearest['variance'],'quantization_variance_px2':quantization_variance,'empirical_excess_variance_px2':empirical_excess,'noise_profile_index':nearest['profile_index'],
                'noise_source_group':nearest['source_group_id'],'same_source_group':nearest['source_group_id']==row['source_group_id']})
        if not accepted:
            record.update(status='pending',reason='no regular current measurement operator');decisions.append(record);counts['groups_pending']+=1;continue
        cols={key:i for i,key in enumerate(dict.fromkeys(key for vec in vectors for key in vec))}
        ii=[];jj=[];vv=[]
        for i,vec in enumerate(vectors):
            for key,value in vec.items():ii.append(i);jj.append(cols[key]);vv.append(value)
        F=csr_matrix((vv,(ii,jj)),shape=(len(accepted),len(cols)));C=(F@F.T).toarray()
        try:Wn,proof=supported_whitening(C)
        except (ValueError,StatisticsUnavailable) as e:
            record.update(status='pending',reason=str(e));decisions.append(record);counts['groups_pending']+=1;continue
        normals=np.asarray(normal_list);W=np.empty((len(Wn),2*len(accepted)));W[:,0::2]=Wn*normals[:,0];W[:,1::2]=Wn*normals[:,1]
        name=f'g{ordinal:05d}_W.f64';W.astype('<f8').tofile(out/name)
        ridlist=[r['source_row_id'] for r in accepted]
        record.update(status='numeric_whitening_ready',statistic_kind='current_native_profile_pixel_influence',row_ids=ridlist,
            rank=len(W),W={'file':name,'sha256':file_sha256(out/name),'dtype':'<f8','shape':list(W.shape)},
            numeric_whitening_verified=True,current_measurements_sha256=hashes,
            original_source_family_sha256=object_sha256([{'source_row_id':r['source_row_id'],'native_pts':r['native_pts'],'source_uv':r['source_uv']} for r in accepted]),
            source_measurements_sha256={r['source_row_id']:object_sha256({'native_pts':r['native_pts'],'source_uv':r['source_uv'],'source_row_id':r['source_row_id']}) for r in accepted},
            native_png={'file':str(png),'sha256':accepted[0]['native_png_sha256']},proof=proof,
            noise_estimates=variance_records,source_role='development',
            source_representation='only current real per-row native material-profile localizations; raw readings are noise/navigation inputs',
            statistical_assumptions={'native_RGB_noise':'decoded 8-bit RGB quantization Uniform[-0.5,+0.5] LSB, variance 1/12 per actual native sample; independence is an explicit noise hypothesis, not an accuracy certificate',
                'empirical_source_texture':'positive excess of measured five-profile localization variance over transported quantization variance; no negative variance component' ,
                'local_homogeneity':'nearest actually measured same-family profile variance; exact segment match recorded per row',
                'shared_pixel_correlations':'transported with actual discrete native RGB influence; cross-color channel mixing retained',
                'historical_selection':'development; no independent validation claim'},
            raw_and_representatives_both_counted=False,original_W_claimed=False,PSD_floor_or_ridge_used=False)
        gf.write(json.dumps(record,ensure_ascii=False)+'\n');registry[family]=record;decisions.append({k:v for k,v in record.items() if k not in {'proof','noise_estimates','current_measurements_sha256'}})
        counts['groups_ready']+=1;counts['represented_rows']+=len(accepted);counts['statistical_rank']+=len(W)
    write_json(out/'statistics_registry.json',registry);write_json(out/'decisions.json',decisions)
    result={'schema':'rinkfit.joint9.current_pixel_statistics.v1','counts':dict(counts),'profile_noise_failures':dict(noise_fail),
        'audited_rows':{'file':str(audited_rows),'sha256':file_sha256(audited_rows)},
        'source_profiles_manifest':{'file':str(profile_dir/'manifest.json'),'sha256':file_sha256(profile_dir/'manifest.json')},
        'locked_roles':None if roles is None else {'file':str(roles),'sha256':file_sha256(roles)},
        'raw_and_representatives_both_counted':False,'independent_representative_sigma_used':False,
        'PSD_floor_or_ridge_used':False,'formal_statistical_optimum_claimed':False,'fit_admission_claimed':False}
    write_json(out/'manifest.json',result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rows',required=True);ap.add_argument('--profiles',required=True);ap.add_argument('--roles');ap.add_argument('--out',required=True);a=ap.parse_args()
    print(json.dumps(compile_pixel_statistics(a.rows,a.profiles,a.out,roles=a.roles),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
