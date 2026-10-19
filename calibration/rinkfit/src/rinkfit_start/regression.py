"""Full native-row regressions from a complete state; never infers independent evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));__package__='rinkfit_start'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def typed(ref, base):
    path = Path(base) / ref['file']
    if sha(path) != ref['sha256']:
        raise ValueError(f'Asset identity mismatch: {path}')
    return np.fromfile(path, ref['dtype']).reshape(ref['shape'])


def summary(error, indices, valid):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return dict(n=0, valid=0)
    e = error[indices]
    c = e[:, 2]
    return dict(n=len(indices), valid=int(valid[indices].sum()),
                median=float(np.median(c)), p95=float(np.quantile(c,.95)),
                p99=float(np.quantile(c,.99)), rms=float(np.sqrt(np.mean(c*c))),
                max=float(c.max()), mean_du=float(np.mean(e[:,0])),
                mean_dv=float(np.mean(e[:,1])), over2=int((c>2).sum()),
                over10=int((c>10).sum()), over20=int((c>20).sum()))


class NativeRegression:
    def __init__(self, run, semantic):
        self.run, self.path = Path(run), Path(semantic)
        self.sem = s = json.loads(self.path.read_text(encoding='utf-8'))
        base = self.path.parent
        self.n = int(s['observations'])
        self.uv = np.fromfile(self.run/'backend_input/uv.f64','<f8').reshape(self.n,2)
        self.pts = np.fromfile(self.run/'backend_input/pts.i64','<i8')
        self.frames = np.fromfile(self.run/'backend_input/frame.i32','<i4')
        table = json.loads((self.run/'backend_input/string_tables.json').read_text(encoding='utf-8'))
        mids = np.fromfile(self.run/'backend_input/material.u32','<u4')
        self.labels = np.asarray(table['material_labels'])[mids]
        self.active = typed(s['active_source_mask'],base).astype(bool)
        self.kind = typed(s['arrays']['kind'],base)
        self.normal = typed(s['arrays']['profile_normal'],base)
        members = typed(s['arrays']['group_rows'],base)
        weights = typed(s['arrays']['weights'],base)
        self.sigma = typed(s['arrays']['source_sigma'],base)
        ind = typed(s['arrays']['ordinary_rows'],base).astype(int)
        self.independent = ind[self.active[ind]]
        stats_path = base/s['active_statistics']['file']
        if sha(stats_path) != s['active_statistics']['sha256']:
            raise ValueError('Statistics identity mismatch')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        if 'source_audit_revision' in s:
            from .semantic_resolution import load_semantics
            _,_,_,extra=load_semantics(self.path,self.n,expected_sha256=sha(self.path))
            replace={gid:(r['source_rows'],r['W']) for gid,r in extra['source_exclusion']['replacements'].items()}
        else:
            replace = {x['group_id']:(np.asarray(x['source_rows']),typed(x['W'],stats_path.parent))
                       for x in stats['replacements']}
        gpath = base/s['assets']['groups.jsonl']['file']
        if sha(gpath) != s['assets']['groups.jsonl']['sha256']:
            raise ValueError('Group identity mismatch')
        self.audit_revision = extra.get('source_audit_revision') if 'source_audit_revision' in s else None
        self.units = []
        partition = np.zeros(self.n,int)
        rank = 0
        for line in gpath.read_text(encoding='utf-8').splitlines():
            g = json.loads(line)
            rr = members[g['member_start']:g['member_start']+g['member_count']].astype(int)
            W = weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)].reshape(g['rank'],2*len(rr))
            if g['group_id'] in replace:
                rr,W = replace[g['group_id']]
            self.units.append((g['group_id'],rr,W))
            partition[rr]+=1
            rank+=len(W)
        partition[self.independent]+=1
        rank+=2*len(self.independent)
        if not np.array_equal(partition,self.active.astype(int)):
            raise ValueError('Original W partition must count every active source exactly once')
        self.rank = rank

    def evaluate(self, state, *, allow_source_revision=False):
        state = Path(state)
        from .state_integrity import validate_report_files
        validate_report_files(state)
        with np.load(state/'candidate_predictions.npz') as z:
            if not np.array_equal(z['source_rows'],np.arange(self.n)):
                raise ValueError('State source row mismatch')
            if not np.array_equal(z['native_pts'],self.pts):
                raise ValueError('State PTS mismatch')
            stored_active=z['active_source_mask'].astype(bool)
            scope_only=False
            if not np.array_equal(stored_active,self.active):
                if not allow_source_revision or not self.audit_revision:
                    raise ValueError('State source scope mismatch')
                removed=np.flatnonzero(stored_active&~self.active).tolist()
                if (np.any(self.active&~stored_active)
                    or removed!=self.audit_revision['new_excluded_source_rows']):
                    raise ValueError('State scope difference exceeds the verified new audit exclusions')
                # Revalidate the immutable semantic parent, including its historical
                # exclusions, then require this exact parent scope in the state.
                from .semantic_resolution import load_semantics
                parent=self.sem['source_audit_parent_manifest']
                parent_path=self.path.parent/parent.get('file',parent.get('path',''))
                _,_,_,parent_extra=load_semantics(parent_path,self.n,expected_sha256=parent['sha256'])
                if not np.array_equal(stored_active,parent_extra['source_exclusion']['active_mask']):
                    raise ValueError('Stored prediction scope is not the verified revision parent')
                identities=[]
                for name,key in [('report.json','semantic'),('initializer_manifest.json','semantic_manifest')]:
                    p=state/name
                    if p.exists():identities.append(json.loads(p.read_text(encoding='utf-8')).get(key,{}).get('sha256'))
                if parent['sha256'] not in identities:
                    raise ValueError('Stored state lacks the verified parent semantic identity')
                scope_only=True
            e = z['uv']-self.uv
            valid = z['RS_valid'].copy()
            c = np.linalg.norm(e,axis=1)
            profile = self.kind==2
            c[profile]=abs(np.einsum('ij,ij->i',e[profile],self.normal[profile]))
            if not np.all(np.isfinite(c[self.active])):
                raise ValueError('Nonfinite active prediction')
            if not np.allclose(e[self.active],z['error_native_uv'][self.active],atol=1e-12,rtol=0):
                raise ValueError('Stored residual mismatch')
            if not np.allclose(c[self.active],z['coverage_error_px'][self.active],atol=1e-12,rtol=0):
                raise ValueError('Stored engineering error mismatch')
        E = np.column_stack((e,c))
        costs=[]
        for gid,rr,W in self.units:
            v=W@e[rr].reshape(-1)
            costs.append(dict(group_id=gid,half_cost=.5*float(v@v)))
        Wcost=sum(x['half_cost'] for x in costs)+.5*float(np.sum((e[self.independent]/self.sigma[self.independent,None])**2))
        groups={name:np.flatnonzero(self.active&(self.labels==name)) for name in np.unique(self.labels)}
        ranges=[('opening',1,100),('pre_spike',101,134),('spike',135,140),
                ('post_spike',141,172),('full_board',173,305),('later',306,int(self.frames.max()))]
        periods={name:np.flatnonzero(self.active&(self.frames>=lo)&(self.frames<=hi)) for name,lo,hi in ranges}
        ad2=np.flatnonzero(self.active&(self.kind==1)&np.char.startswith(self.labels,'Ad2.'))
        out=dict(state=str(state),model_sha256=sha(state/'model.json'),
                 evaluated_value_report=dict(path=str(state/'report.json'),sha256=sha(state/'report.json')),
                 predictions_sha256=sha(state/'candidate_predictions.npz'),
                 raw_rows=self.n,active_rows=int(self.active.sum()),frames=len(np.unique(self.frames)),
                 stored_scope=dict(active_rows=int(stored_active.sum()),excluded_rows=np.flatnonzero(~stored_active).tolist(),
                     active_mask_bytes_sha256=hashlib.sha256(stored_active.astype('u1').tobytes()).hexdigest()),
                 evaluated_scope=dict(active_rows=int(self.active.sum()),excluded_rows=np.flatnonzero(~self.active).tolist(),
                     semantic_path=str(self.path),semantic_sha256=sha(self.path)),
                 same_physical_state_scope_only=scope_only,
                 stored_RS_predictions_reused_without_new_physical_fit=scope_only,
                 excluded_rows=np.flatnonzero(~self.active).tolist(),statistics_rank=self.rank,
                 profile_rows=int(np.sum(self.active&(self.kind==2))),partition_exactly_once=True,
                 whole_original_W_half_cost=Wcost,all=summary(E,np.flatnonzero(self.active),valid),
                 materials={name:summary(E,rr,valid) for name,rr in groups.items()},
                 periods={name:summary(E,rr,valid) for name,rr in periods.items()},
                 ad2_entities=dict(all=summary(E,ad2,valid),periods={name:summary(E,np.intersect1d(ad2,rr),valid) for name,rr in periods.items()}),
                 rings={label:{name:summary(E,np.intersect1d(groups.get(label,[]),rr).astype(int),valid) for name,rr in periods.items()}
                        for label in ['ice.R3:inner','ice.R3:middle','ice.R3:outer','ice.C0:outer']},
                 validation_role='development regression: source rows previously used for fitting',
                 profile_global_certification_not_inferred=True,
                 measurement_roles={**{f'ice.{name}:outer':'circular_band_centerline' for name in ['C0','A1','A2','A3','A4']},
                     **{f'ice.R3:{part}':'target_material_boundary' for part in ['inner','middle','outer']}},
                 C0_outer_is_historical_identifier_not_exterior_material_edge=True,
                 legacy_rings_report_key_includes_C0_centerline_for_compatibility=True)
        return out,E,valid,costs


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True);ap.add_argument('--semantic',required=True)
    ap.add_argument('--state',required=True);ap.add_argument('--baseline');ap.add_argument('--output',required=True)
    ap.add_argument('--allow-source-revision',action='store_true',help='Reevaluate the exact verified semantic-parent physical state under only the declared source-audit exclusions; never claims a new RS solve.')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reg=NativeRegression(a.run,a.semantic)
    result,E,valid,costs=reg.evaluate(a.state,allow_source_revision=a.allow_source_revision)
    if a.baseline:
        before,B,bvalid,_=reg.evaluate(a.baseline,allow_source_revision=a.allow_source_revision)
        result['baseline']=before
        result['change']=dict(max_px=result['all']['max']-before['all']['max'],
            W_half_cost=result['whole_original_W_half_cost']-before['whole_original_W_half_cost'],
            median_px=result['all']['median']-before['all']['median'],
            worsened_over2_px_rows=int(np.sum(reg.active&(E[:,2]>B[:,2]+2))))
    write_csv(out/'materials.csv',[dict(material=k,**v) for k,v in result['materials'].items()])
    write_csv(out/'frames.csv',[dict(frame=int(f),**summary(E,np.flatnonzero(reg.active&(reg.frames==f)),valid)) for f in np.unique(reg.frames)])
    write_csv(out/'W_groups.csv',costs)
    sel=reg.active&(np.char.startswith(reg.labels,'ice.R3:')|(reg.labels=='ice.C0:outer')|((reg.kind==1)&np.char.startswith(reg.labels,'Ad2.')))
    write_csv(out/'focused_rows.csv',[dict(source_row=int(i),frame=int(reg.frames[i]),pts=int(reg.pts[i]),label=reg.labels[i],raw_u=reg.uv[i,0],raw_v=reg.uv[i,1],du=E[i,0],dv=E[i,1],error_px=E[i,2],RS_valid=bool(valid[i])) for i in np.flatnonzero(sel)])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['raw_rows','active_rows','frames','statistics_rank','whole_original_W_half_cost','all','ad2_entities']},ensure_ascii=False))

if __name__=='__main__':main()
