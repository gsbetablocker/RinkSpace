"""Numerical source statistics from current native readouts.

The operator is fitted in image space only.  Its covariance is propagated to
representatives once; adding representatives cannot add source information.
No world radius ratios, independent representative sigma, ridge or PSD floor
are used.  A missing empirical covariance remains an explicit missing group.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .io_utils import read_jsonl, write_json, file_sha256

class StatisticsUnavailable(ValueError): pass

def supported_whitening(covariance):
    C=np.asarray(covariance,float)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or not np.isfinite(C).all():
        raise ValueError('invalid finite covariance')
    asym=float(np.max(np.abs(C-C.T),initial=0.))
    if asym>1e-10*max(1.,float(np.max(abs(C),initial=0.))):
        raise ValueError('covariance is not symmetric')
    values,vectors=np.linalg.eigh((C+C.T)*.5)
    threshold=np.finfo(float).eps*max(C.shape)*max(float(np.max(abs(values),initial=0.)),np.finfo(float).tiny)*8
    if np.any(values < -threshold): raise ValueError('source covariance is not PSD')
    keep=values>threshold
    if not keep.any(): raise StatisticsUnavailable('no positive empirical source covariance support')
    W=(vectors[:,keep]/np.sqrt(values[keep])).T
    # Pseudoinverse support, never make an unmeasured null mode informative.
    if not np.allclose(W@C@W.T,np.eye(int(keep.sum())),rtol=1e-7,atol=1e-8):
        raise ValueError('whitening failed on measured covariance support')
    return W,{'rank':int(keep.sum()),'dimension':len(C),'support_threshold':threshold,
        'minimum_retained_eigenvalue':float(values[keep].min()),
        'maximum_discarded_eigenvalue':float(values[~keep].max(initial=0.)),
        'covariance_symmetry_error':asym,'PSD_floor_or_ridge_added':False}

def supported_factor_whitening(factor):
    F=np.asarray(factor,float)
    U,s,_=np.linalg.svd(F,full_matrices=False)
    threshold=np.finfo(float).eps*max(F.shape)*s.max(initial=0.)*8
    keep=s>threshold
    if not keep.any():raise StatisticsUnavailable('no positive empirical covariance factor support')
    W=(U[:,keep]/s[keep]).T
    white=W@F
    if not np.allclose(white@white.T,np.eye(int(keep.sum())),rtol=1e-7,atol=1e-8):
        raise ValueError('source factor whitening failed')
    return W,{'rank':int(keep.sum()),'dimension':F.shape[0],
        'support_threshold':threshold**2,'minimum_retained_eigenvalue':float((s[keep]**2).min()),
        'maximum_discarded_eigenvalue':float((s[~keep]**2).max(initial=0.)),
        'PSD_floor_or_ridge_added':False,'thin_covariance_factor_used':True}

def hac_parameter_covariance(J,residual,segments=None):
    J=np.asarray(J,float);e=np.asarray(residual,float)
    if J.ndim!=2 or e.shape!=(len(J),) or len(J)<=J.shape[1]:
        raise StatisticsUnavailable('insufficient current readouts for source fit covariance')
    U,s,V=np.linalg.svd(J,full_matrices=False)
    threshold=np.finfo(float).eps*max(J.shape)*s.max()*8
    keep=s>threshold
    if not keep.any():raise StatisticsUnavailable('unobservable source fit')
    inverse=(V[keep].T/(s[keep]**2))@V[keep]
    scores=J*e[:,None]
    segments=np.zeros(len(J),int) if segments is None else np.asarray(segments)
    meat=np.zeros((J.shape[1],J.shape[1]));lags=[]
    for segment in np.unique(segments):
        z=scores[segments==segment];n=len(z)
        # Fixed data-based bandwidth; contiguous ordered pixels are not IID.
        lag=min(n-1,int(np.floor(4*(n/100.)**(2/9))))
        lags.append(lag);meat+=z.T@z
        for k in range(1,lag+1):
            cross=z[k:].T@z[:-k];meat+=(1-k/(lag+1))*(cross+cross.T)
    C=inverse@meat@inverse*(len(J)/(len(J)-int(keep.sum())))
    C=(C+C.T)*.5
    return C,{'readouts':len(J),'source_parameter_rank':int(keep.sum()),
        'source_parameter_dimension':J.shape[1],'HAC_Bartlett_lags':lags,
        'normal_residual_rms_px':float(np.sqrt(np.mean(e*e))),
        'measurement_model':'empirical native readout normal errors; HAC sandwich',
        'zero_residuals_do_not_create_infinite_information':True}

def _jacobian(function,theta):
    theta=np.asarray(theta,float);result=[]
    for k in range(len(theta)):
        h=np.cbrt(np.finfo(float).eps)*max(1.,abs(float(theta[k])))
        plus=theta.copy();minus=theta.copy();plus[k]+=h;minus[k]-=h
        result.append((function(plus)-function(minus))/(2*h))
    return np.stack(result,axis=1)

def _ellipse_values(theta,uv):
    center=theta[:2];a,b=np.exp(theta[2:4]);c,s=np.cos(theta[4]),np.sin(theta[4])
    R=np.array([[c,-s],[s,c]]);q=(uv-center)@R
    gradient_local=np.stack((2*q[:,0]/a**2,2*q[:,1]/b**2),axis=1)
    gradient=gradient_local@R.T;den=np.linalg.norm(gradient,axis=1)
    if np.any(den<1e-12):raise StatisticsUnavailable('ellipse normal singularity')
    residual=((q[:,0]/a)**2+(q[:,1]/b)**2-1)/den
    return residual,gradient/den[:,None]

def fit_representative_statistics(raw_uv,representative_uv,*,curve_kind='cubic_graph',segments=None):
    raw=np.asarray(raw_uv,float);rep=np.asarray(representative_uv,float)
    if raw.ndim!=2 or raw.shape[1]!=2 or rep.ndim!=2 or rep.shape[1]!=2 or not np.isfinite(raw).all() or not np.isfinite(rep).all():
        raise ValueError('nonfinite or invalid native readout arrays')
    if len(raw)<6 or len(rep)<2:raise StatisticsUnavailable('too few real source pixels for a finite source operator')
    center=raw.mean(axis=0);_,sing,V=np.linalg.svd(raw-center,full_matrices=False)
    if curve_kind=='free_ellipse':
        axes=np.maximum(np.std((raw-center)@V.T,axis=0)*np.sqrt(2),np.finfo(float).eps)
        theta=np.r_[center,np.log(axes),np.arctan2(V[0,1],V[0,0])]
        fun=lambda q:_ellipse_values(q,raw)[0]
        solved=least_squares(fun,theta,method='trf',max_nfev=200)
        if not solved.success:raise StatisticsUnavailable('current-source free ellipse fit did not converge')
        theta=solved.x;error,nraw=_ellipse_values(theta,raw);J=_jacobian(fun,theta)
        shift,nrep=_ellipse_values(theta,rep)
        Jr=_jacobian(lambda q:_ellipse_values(q,rep)[0],theta)
        current=rep-shift[:,None]*nrep
        operator={'kind':'source_only_free_ellipse','parameters':theta.tolist(),
            'world_radius_or_old_family_ratios_used':False}
    elif curve_kind=='cubic_graph':
        tangent=V[0];normal=np.array([-tangent[1],tangent[0]])
        x=(raw-center)@tangent;scale=float(np.ptp(x))
        if scale<=1e-9:raise StatisticsUnavailable('source graph has zero tangent span')
        xx=x/scale;yy=(raw-center)@normal
        degree=min(3,len(raw)-2);A=np.polynomial.polynomial.polyvander(xx,degree)
        theta=np.linalg.lstsq(A,yy,rcond=None)[0]
        derivative=np.polynomial.polynomial.polyval(xx,np.polynomial.polynomial.polyder(theta))/scale
        den=np.sqrt(1+derivative**2);error=(A@theta-yy)/den;J=A/den[:,None]
        xp=(rep-center)@tangent/scale;yp=(rep-center)@normal
        Ap=np.polynomial.polynomial.polyvander(xp,degree)
        dp=np.polynomial.polynomial.polyval(xp,np.polynomial.polynomial.polyder(theta))/scale
        nrep=(normal[None,:]-dp[:,None]*tangent[None,:])/np.sqrt(1+dp**2)[:,None]
        Jr=Ap/np.sqrt(1+dp**2)[:,None]
        shift=(Ap@theta-yp)/np.sqrt(1+dp**2)
        current=rep+shift[:,None]*nrep
        operator={'kind':'source_only_cubic_image_graph','coefficients':theta.tolist(),
            'center':center.tolist(),'tangent':tangent.tolist(),'normal':normal.tolist(),
            'coordinate_scale':scale,'world_scene_used':False}
    else:raise ValueError('unsupported explicit source operator')
    Ctheta,proof=hac_parameter_covariance(J,error,segments)
    B=(-Jr[:,:,None]*nrep[:,None,:]).transpose(0,2,1).reshape(2*len(rep),-1)
    eigen,vectors=np.linalg.eigh(Ctheta)
    threshold=np.finfo(float).eps*max(Ctheta.shape)*np.max(abs(eigen))*8
    if np.any(eigen < -threshold):raise ValueError('source HAC parameter covariance is not PSD')
    good=eigen>threshold
    if not good.any():raise StatisticsUnavailable('no measured parameter covariance support')
    factor=B@(vectors[:,good]*np.sqrt(eigen[good]))
    W,support=supported_factor_whitening(factor)
    C=factor@factor.T
    proof.update(support);proof.update(source_operator=operator,
        representatives=len(rep),raw_and_representatives_both_counted=False,
        source_parameter_covariance=Ctheta.tolist(),
        maximum_source_operator_remeasurement_px=float(np.linalg.norm(current-rep,axis=1).max(initial=0.)))
    return {'current_uv':current,'W':W,'covariance':C,'proof':proof}

def compile_sources(audited_rows,retained_support,output_dir):
    out=Path(output_dir)
    if out.exists():raise ValueError('statistics generation must be new')
    out.mkdir(parents=True)
    rows={r['source_row_id']:r for r in read_jsonl(audited_rows)}
    support=list(read_jsonl(retained_support));families=defaultdict(list)
    for item in support:
        # Different actual boundary layers never share a source ellipse merely
        # because an old navigation fit imposed radius ratios on them.
        rr=[rows[i] for i in item['representative_row_ids'] if i in rows]
        if not rr:continue
        byidentity=defaultdict(list)
        for row in rr:
            if row.get('point_correspondence'):continue
            identity=tuple(row.get(k) for k in ['target_id','physical_surface','measurement_operator_resolved','boundary_role'])
            byidentity[identity].append(row['source_row_id'])
        for identity,ids in byidentity.items():
            selected=dict(item);selected['representative_row_ids']=ids
            families[(item['source_family_id'],identity)].append(selected)
    summaries=[];counts=Counter()
    raw_owners={}
    with gzip.open(out/'groups.jsonl.gz','wt',encoding='utf8') as gf:
      for ordinal,((family,identity),items) in enumerate(sorted(families.items(),key=lambda x:str(x[0]))):
        ids=list(dict.fromkeys(i for item in items for i in item['representative_row_ids'] if i in rows))
        accepted=[i for i in ids if not rows[i].get('point_correspondence') and rows[i].get('current_uv') is not None and rows[i].get('physical_identity_status') in {'confirmed','confirmed_static_material','typed_identity_confirmed','typed_native_material_supported','typed_identity_current_pixel_supported'}]
        # Source worker status vocabulary is retained in the decision ledger;
        # explicit solver_eligible/admission is authoritative when supplied.
        accepted=[i for i in ids if not rows[i].get('point_correspondence') and rows[i].get('current_uv') is not None and (rows[i].get('solver_eligible') is True or rows[i].get('admitted') is True)] or accepted
        raw=[];segments=[];seen=set();reused=[]
        for si,item in enumerate(items):
          for audit in item.get('actual_source_audits',[]):
            uv=audit.get('current_uv')
            if uv is None or audit.get('raw_index') not in item['current_support_indices']:continue
            key=(item['source_packet_sha256'],tuple(audit['raw_uv']))
            if key in seen:continue
            seen.add(key)
            if key in raw_owners:reused.append(raw_owners[key])
            raw.append(uv);segments.append(si)
        record={'source_group_id':family,'physical_identity':list(identity),
            'representative_source_row_ids':accepted,'raw_readouts':len(raw),
            'unique_raw_and_source_packet':True,'source_inputs':[x['statistical_input_id'] for x in items],'identity_admission_required':True,'fit_role':'development','historical_source_selection_not_independent_validation':True}
        try:
            if not accepted:raise StatisticsUnavailable('no source-supported typed representative rows')
            if reused:raise StatisticsUnavailable('actual readout also belongs to another source operator; shared joint source fit required')
            kind='free_ellipse' if any(x.get('source_curve_family_metadata',{}).get('source_circle_fit') for x in items) else 'cubic_graph'
            result=fit_representative_statistics(raw,[rows[i]['current_uv'] for i in accepted],curve_kind=kind,segments=segments)
            W=result['W'];name=f'g{ordinal:05d}_W.f64';W.astype('<f8').tofile(out/name)
            uvname=f'g{ordinal:05d}_uv.f64';result['current_uv'].astype('<f8').tofile(out/uvname)
            record.update(status='numeric_source_covariance_ready',statistic_kind='native_readout_operator_propagation',
                rank=len(W),W={'file':name,'sha256':file_sha256(out/name),'dtype':'<f8','shape':list(W.shape)},
                remeasured_uv={'file':uvname,'sha256':file_sha256(out/uvname),'dtype':'<f8','shape':list(result['current_uv'].shape)},
                proof=result['proof'])
            for raw_key in seen:raw_owners[raw_key]=family+'|'+str(identity)
            gf.write(json.dumps(record,ensure_ascii=False)+'\n');counts['groups_ready']+=1;counts['statistical_rank']+=len(W);counts['represented_rows']+=len(accepted)
        except (StatisticsUnavailable,ValueError,np.linalg.LinAlgError) as error:
            record.update(status='statistics_not_available',reason=str(error));counts['groups_missing']+=1
        summaries.append(record)
    write_json(out/'decisions.json',summaries)
    result={'schema':'rinkfit.joint9.source_statistics.v1','counts':dict(counts),
        'inputs':{'audited_rows':{'path':str(audited_rows),'sha256':file_sha256(audited_rows)},
                  'retained_support':{'path':str(retained_support),'sha256':file_sha256(retained_support)}},
        'full_W_ready':counts['groups_missing']==0,'raw_and_representatives_both_counted':False,
        'independent_representative_sigma_used':False,'PSD_floor_or_ridge_used':False,
        'formal_statistical_optimum_claimed':False}
    write_json(out/'manifest.json',result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rows',required=True)
    ap.add_argument('--support',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    print(json.dumps(compile_sources(a.rows,a.support,a.out),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
