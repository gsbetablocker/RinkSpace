"""Scoped replay of existing frozen source/grid evidence, without nine-shot coupling.

The numerical ice witnesses retain their exact old world chart. Advertisement
additions and block reindexing do not change that chart; any actual frozen ice
carrier or scene change still refuses this grid comparison. This is not a new
source generation and does not certify the historical 8a28 full-3D baseline.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .source_quality_compare_v1 import (SourceQualityComparison, _read, _ref,
    _sha, _object_sha, _camera, _metric, _delta, _frame_stats, _aggregate, _windows)
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene


def _semantic_geometry(model, labels):
    lookup = {m['label']: m for m in model['materials']}
    def block(i):
        b = model['blocks'][int(i)]
        return dict(name=b['name'], values=b['values'])
    def path(p):
        return {k: (block(v) if k.endswith('_block') and isinstance(v, int)
                    and v >= 0 else v)
                for k, v in p.items()
                if 'provenance' not in k and not k.startswith('source_')}
    return _object_sha(dict(scene=block(model['scene_block']), materials=[
        dict(label=label, paths=[path(p) for p in lookup[label]['paths']])
        for label in sorted(labels)]))


class ScopedSourceQualityComparison(SourceQualityComparison):
    def __init__(self, binding_path, shot_ids=('s01','s02','s03','s04')):
        super().__init__(binding_path)
        self.shot_ids = tuple(shot_ids)
        if not self.shot_ids or len(set(self.shot_ids)) != len(self.shot_ids):
            raise ValueError('Explicit unique shots required')
        if any(s not in self.reference['shots'] for s in self.shot_ids):
            raise ValueError('Shot outside frozen lineage')
        sn = [int(s[1:]) for s in self.shot_ids]
        keep = np.isin(self.shots, sn)
        for attr in ('pts','shots','X','target','mask','mode','inferred','direct',
                     'reference_valid','ref_sse','ref_den','ref_bad'):
            setattr(self, attr, getattr(self, attr)[keep])
        keep = np.isin(self.ash, [self.metadata['shot_ids'].index(s) for s in self.shot_ids])
        for attr in ('rows','raw','uv','apt','ash','mi','pi','entity',
                     'anchor_reference_error','anchor_reference_valid','anchor_mode'):
            setattr(self, attr, getattr(self, attr)[keep])
        self.ice_labels = [m['label'] for m in self.reference['materials']
                           if m['label'].startswith('ice.') and '.tracklet.' not in m['label']]
        self.ice_identity = _semantic_geometry(self.reference, self.ice_labels)

    def _validate_model_identity(self, model):
        if model.get('source_clock') != self.source_clock:
            raise ValueError('Candidate source clock differs')
        if set(model['rigs']) != set(self.reference['rigs']):
            raise ValueError('Shared three-rig identity differs')
        for sid in self.shot_ids:
            if model['shots'][sid]['rig_id'] != self.reference['shots'][sid]['rig_id']:
                raise ValueError('Shot owning rig differs: ' + sid)

    def _anchors(self, model):
        scene = NativeScene(as_event_view(model, self.shot_ids[0]))
        lookup = {}
        for mi, m in enumerate(model['materials']):
            for pi, p in enumerate(m['paths']):
                key = m['label'] + '/' + p['id']
                if key in lookup: raise ValueError('Duplicate physical entity: ' + key)
                lookup[key] = (mi, pi)
        world = np.full((len(self.uv), 3), np.nan)
        entity_ok = np.zeros(len(self.uv), bool)
        failures = {}
        for j, entity in enumerate(self.entity):
            try:
                mi, pi = lookup[entity]; p = scene.path(mi, pi)
                if not p.get('physical_point') or len(p.get('points', [])) != 1:
                    raise ValueError('Fixed point topology differs')
                world[j] = scene.world(mi, pi, 0.); entity_ok[j] = True
            except Exception as e: failures[str(entity)] = type(e).__name__ + ': ' + str(e)
        pred = np.full_like(self.uv, np.nan); valid = np.zeros(len(self.uv), bool)
        exceptions = {}
        for sid in self.shot_ids:
            si = self.metadata['shot_ids'].index(sid); sel = (self.ash == si) & entity_ok
            if not sel.any(): continue
            try:
                cam = _camera(model, sid)
                r = cam.project_rs(world[sel], (self.apt[sel]-model['shots'][sid]['time_origin_pts'])/12800.)
                pred[sel] = r['uv']; valid[sel] = r['valid']
            except Exception as e: exceptions[sid] = type(e).__name__ + ': ' + str(e)
        error = np.linalg.norm(pred-self.uv, axis=1)
        def group(sel): return _delta(_metric(error[sel], valid[sel]),
            _metric(self.anchor_reference_error[sel], self.anchor_reference_valid[sel]))
        byshot = {}
        for sid in self.shot_ids:
            si = self.metadata['shot_ids'].index(sid); sel = self.ash == si
            pts = self.pts[self.shots == int(sid[1:])]
            sse = np.zeros(len(pts)); den = np.zeros(len(pts), int); bad = den.copy()
            for j, p in enumerate(pts):
                a = sel & (self.apt == p); g = a & valid & np.isfinite(error)
                den[j] = int(a.sum()); bad[j] = int(a.sum()-g.sum()); sse[j] = np.sum(error[g]**2)
            byshot[sid] = dict(**group(sel), source_exposures=int((den>0).sum()),
                temporal=_windows(pts,sse,den,bad,den>0))
        return dict(scope='Frozen original fixed-source anchors in explicit shots; semantic entity lookup; actual candidate XYZ; not new-generation packet coverage',
            all=group(np.ones(len(error),bool)), by_shot=byshot,
            by_entity={str(e):group(self.entity==e) for e in np.unique(self.entity)},
            exceptions=exceptions, entity_failures=failures, all_original_scoped_rows_retained=True), dict(
                anchor_error_px=error, anchor_valid=valid, anchor_predicted_uv=pred, anchor_raw_index=self.raw)

    def compare(self, model_path, output_dir):
        self.assert_dependencies(); mp=Path(model_path).resolve(); model=_read(mp)
        self._validate_model_identity(model); out=Path(output_dir).resolve()
        if out.exists(): raise FileExistsError('Use a new immutable comparison output')
        out.mkdir(parents=True)
        try: digest=_semantic_geometry(model,self.ice_labels); same=digest==self.ice_identity
        except KeyError: digest=None; same=False
        source, arrays=self._anchors(model)
        report=dict(schema='rinkfit.scoped_source_quality_comparison.v2',
            parent_binding=_ref(self.path), candidate=_ref(mp),
            reference_model=self.binding['refs']['reference_model'], reference_stage=self.reference_stage,
            shot_ids=list(self.shot_ids), actual_model_shots=list(model['shots']),
            scene_count=1, rig_count=len(model['rigs']), native_frames=len(self.pts),
            original_fixed_source_rows=len(self.uv), quality_accepted=False,
            automatic_acceptance_supported=False, combined_quality_score=None,
            source_anchors=source, same_frozen_ice_geometry=same,
            candidate_ice_identity=digest, frozen_ice_identity=self.ice_identity,
            source_improvement_cannot_cancel_grid_regression=True,
            scope_limits=self.binding['scope_limits']+[
                'Only explicitly selected shots evaluated; inactive shots cannot affect this replay.',
                'Reference v8 is a rejected internal comparator; S01 d78 and compact ice H are separate numerical targets.',
                'This comparison alone does not establish equality to 8a28 full Brown/3D surfaces.',
                'Current source generation and all solver residuals require their separate packet audit.'])
        if not same:
            report['grid']=dict(status='unsupported_changed_frozen_ice_geometry',
                original_denominator=int(self.mask.sum()), failed_denominator_not_dropped=True)
        else:
            error=np.full(self.mask.shape,np.nan); valid=np.zeros(self.mask.shape,bool)
            exceptions={}; domains={}
            for sid in self.shot_ids:
                sel=self.shots==int(sid[1:])
                try:
                    cam=_camera(model,sid); r=cam.project_rs(self.X[sel],
                        (self.pts[sel,None]-model['shots'][sid]['time_origin_pts'])/12800.)
                    error[sel]=np.linalg.norm(r['uv']-self.target[sel],axis=-1); valid[sel]=r['valid']
                    m=self.mask[sel]
                    domains[sid]=dict(bracket_failed=int((m&~r['bracketed']).sum()),
                        certificate_failed=int((m&~r['certificate']['certified']).sum()),
                        returned_invalid=int((m&~r['valid']).sum()), whole_Brown_disk=cam.optical_domain_certificate())
                except Exception as e: exceptions[sid]=type(e).__name__+': '+str(e)
            sse,den,bad=_frame_stats(error,valid,self.mask)
            def group(sel): return _delta(_aggregate(sse[sel],den[sel],bad[sel]),
                _aggregate(self.ref_sse[sel],self.ref_den[sel],self.ref_bad[sel]))
            byshot={}
            for sid in self.shot_ids:
                sel=self.shots==int(sid[1:]); row=group(sel)
                row.update(frames=int(sel.sum()),temporal=_windows(self.pts[sel],sse[sel],den[sel],bad[sel]),
                    new_failed_witnesses=int((self.mask[sel]&self.reference_valid[sel]&~valid[sel]).sum()))
                byshot[sid]=row
            report['grid']=dict(status='evaluated_exact_frozen_ice_chart',
                all=group(np.ones(len(self.pts),bool)), by_shot=byshot, domains=domains,
                exceptions=exceptions, all_failure_denominators_retained=True,
                scope='S01 complete d78 physical Brown/RS; S02-S04 actualDemo ice H; numerical targets, not source UV')
            arrays.update(grid_error_px=error,grid_valid=valid,grid_frame_valid_sse=sse,
                          grid_frame_denominator=den,grid_frame_failed=bad)
        arrays.update(native_pts=self.pts,shot_number=self.shots,bridge_mode=self.mode,
                      grid_original_mask=self.mask,source_anchor_native_pts=self.apt)
        self.assert_dependencies(); np.savez_compressed(out/'comparison_arrays.npz',**arrays)
        report.update(arrays=_ref(out/'comparison_arrays.npz'),producer=dict(path=__file__,sha256=_sha(__file__)))
        (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--binding',required=True)
    p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--shots',nargs='+',default=['s01','s02','s03','s04']); a=p.parse_args(argv)
    r=ScopedSourceQualityComparison(a.binding,a.shots).compare(a.model,a.out)
    print(json.dumps(dict(output=str(Path(a.out).resolve()),frames=r['native_frames'],
        source_rows=r['original_fixed_source_rows'],grid_status=r['grid']['status']),ensure_ascii=False))
if __name__=='__main__': main()
