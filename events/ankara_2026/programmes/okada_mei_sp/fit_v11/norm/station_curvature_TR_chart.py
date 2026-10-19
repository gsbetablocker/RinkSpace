"""Temporary sampled-curvature trust for total original S=seed+eta.
This only changes a numerical proposal chart; it does not fix original S in the
physical problem or certify a continuous Taylor remainder.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(r,base=None):
 p=Path(r.get('path',r.get('file','')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=r['sha256']:raise ValueError('Changed curvature trust authority '+str(p))
 return p.resolve()
def apply_station_curvature_TR(chart,manifest,array_key='radius'):
 P=Path(manifest).resolve();d=json.loads(P.read_text(encoding='utf8'));authority=chart.authority
 if not d.get('schema','').startswith('rinkfit.original_station_numerical_TR_native_RS_curvature.'):raise ValueError('Explicit numerical curvature station-trust schema required')
 for name in ['checkpoint','transport','model','packet','semantic']:
  if name not in d:raise ValueError('Missing curvature authority '+name)
  verify(d[name])
  if d[name]['sha256']!=authority[name]['sha256']:raise ValueError('Mixed curvature and W complete state '+name)
 if d.get('complete_auxiliary_sha256',authority['complete_auxiliary_sha256'])!=authority['complete_auxiliary_sha256']:raise ValueError('Changed full raw-N auxiliary bytes')
 desc=d['arrays'][array_key];radius=np.fromfile(verify(desc,P.parent),desc['dtype']).reshape(desc['shape']);N=len(chart.station_seed)
 if radius.shape!=(N,)or not np.isfinite(radius).all()or np.any(radius<0):raise ValueError('Complete finite nonnegative original-S numerical radii required')
 bad=np.flatnonzero(abs(chart.station_seed)>radius)
 if len(bad):raise ValueError('Primary FULL station seed leaves new curvature trust; re-solve/recover primary, never clip seed. rows='+str(bad[:30].tolist()))
 selected=chart.selected;lo=np.maximum(chart.station_lower,-radius[selected]-chart.station_seed[selected]);hi=np.minimum(chart.station_upper,radius[selected]-chart.station_seed[selected])
 if np.any(lo>hi)or np.any(lo>0)or np.any(hi<0):raise ValueError('Working anchor not feasible in original finite cell and numerical station trust')
 keep=lo<hi;dropped=selected[~keep]
 if np.any(lo[~keep]!=0)or np.any(hi[~keep]!=0):raise ValueError('A fixed secondary eta must be its exact seed value')
 out=copy.copy(chart);out.selected=selected[keep].copy();out.station_lower=lo[keep];out.station_upper=hi[keep];out.cell_lower=chart.cell_lower[keep];out.cell_upper=chart.cell_upper[keep]
 for key in ['T','Jstation','sensor_Jstation']:setattr(out,key,getattr(chart,key)[:,keep].copy())
 out.reasons=chart.reasons.copy();out.reasons[np.isin(chart.requested,dropped)]='numerical_curvature_TR_fixed_zero';out.authority=dict(authority,station_TR=dict(manifest=ref(P),array_key=array_key,total_deltaS_is_seed_plus_eta=True,temporary_numerical_chart_only=True))
 metadata=dict(schema='rinkfit.full_seed_plus_selected_S_curvature_TR_application.v1',curvature_manifest=ref(P),radius_array=desc,array_key=array_key,checkpoint=authority['checkpoint'],transport=authority['transport'],physical_source_W_E_unchanged=True,full_seed_bitwise_preserved=True,selected_before=len(selected),selected_after=len(out.selected),dropped_zero_eta_source_rows=dropped.tolist(),unselected_station_seed_count=int(np.count_nonzero(chart.station_seed))-int(np.count_nonzero(chart.station_seed[selected])),unknown_or_zero_radius_is_temporary_fixed_S_not_deleted_source=True,original_finite_cell_intersected=True,original_sensor_unchanged_and_still_audited=True,not_a_new_physical_prior_or_KKT=True)
 return out,metadata
