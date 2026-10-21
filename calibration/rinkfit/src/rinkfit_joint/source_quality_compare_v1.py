"""Read-only source/grid comparison. It never accepts a candidate or runs a fit.

The old fixed-world grid applies only to the frozen template geometry digest.
Changed geometry is refused for that cohort; original fixed-source anchors are
still recomputed from the candidate's actual NativeScene fixed entities.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from .joint_model import as_event_view
from .screen_to_native_seed import _project
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.native_camera import rotation
SCHEMA='rinkfit.source_quality_comparison_binding.v1'

def _read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _ref(p):return dict(path=str(Path(p).resolve()),sha256=_sha(p))
def _object_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def geometry_identity(model):
    """Consumed scene/material definitions and referenced physical block values.

    Motion, optical parameters and unused control mirrors are excluded. Pure
    source provenance is excluded. All actual geometry/ad block values stay in.
    """
    blocks=set([int(model['scene_block'])]); materials=[]
    for material in model['materials']:
        paths=[]
        for path in material['paths']:
            q={k:v for k,v in path.items() if 'provenance' not in k and not k.startswith('source_')}
            for k,v in q.items():
                if k.endswith('_block') and isinstance(v,int) and v>=0:blocks.add(v)
            paths.append(q)
        materials.append(dict(label=material['label'],paths=paths))
    consumed=[dict(index=i,name=model['blocks'][i]['name'],values=model['blocks'][i]['values']) for i in sorted(blocks)]
    return _object_sha(dict(scene_block=model['scene_block'],materials=materials,consumed_geometry_blocks=consumed))
def _camera(model,sid):
    c=camera_from_event(as_event_view(model,sid));a=model['blocks'][model['shots'][sid]['reference_angles_block']]['values'];c.R0=rotation(c.b,a[1])@c.R0@rotation(c.n,a[0]);return c

def _metric(error,valid):
    e=np.asarray(error,float).ravel();v=np.asarray(valid,bool).ravel()&np.isfinite(e);z=e[v];d=len(e);n=int(v.sum())
    return dict(denominator=d,valid=n,failed=d-n,full_rms_px=float(np.sqrt(np.mean(z*z))) if n==d and d else None,available_rms_px=float(np.sqrt(np.mean(z*z))) if n else None,available_max_px=float(z.max()) if n else None)
def _aggregate(sse,den,bad):
    d=int(np.sum(den));b=int(np.sum(bad));n=d-b;s=float(np.sum(sse))
    return dict(denominator=d,valid=n,failed=b,full_rms_px=float(np.sqrt(s/d)) if d and b==0 else None,available_rms_px=float(np.sqrt(s/n)) if n else None)
def _delta(candidate,reference):
    a,b=candidate.get('full_rms_px'),reference.get('full_rms_px');d=None if a is None or b is None else a-b
    return dict(candidate=candidate,reference=reference,full_rms_delta_px=d,numeric_regression=(d>1e-6) if d is not None else None,roundoff_only_comparison_tolerance_px=1e-6,comparison_unresolved_due_to_failed_denominator=d is None)
def _frame_stats(error,valid,mask):
    good=mask&valid&np.isfinite(error);return np.sum(np.where(good,error*error,0.),axis=1),mask.sum(1),(mask&~good).sum(1)
def _windows(pts,sse,den,bad,exposures=None):
    worst=None;maxbad=None;n=0
    for i in range(len(pts)-49):
        j=i+50
        if int(pts[j-1]-pts[i])!=49*256:continue
        q=_aggregate(sse[i:j],den[i:j],bad[i:j]);row=dict(start_pts=int(pts[i]),end_pts=int(pts[j-1]),native_frames=50,**q)
        if exposures is not None:row['source_exposures']=int(exposures[i:j].sum())
        if q['denominator']==0:continue
        n+=1
        if q['full_rms_px'] is not None and (worst is None or q['full_rms_px']>worst['full_rms_px']):worst=row
        if maxbad is None or q['failed']>maxbad['failed']:maxbad=row
    return dict(windows_with_measurements=n,worst_full_valid_50nativeframe_span=worst,max_failure_50nativeframe_span=maxbad)

class SourceQualityComparison:
    """One small shared context, all original clocks, masks and fixed-source UV."""
    def __init__(self,binding_path):
        self.path=Path(binding_path).resolve();self.binding=_read(self.path)
        if self.binding.get('schema')!=SCHEMA:raise ValueError('Unsupported comparison binding')
        self.identity=_sha(self.path);self.assert_dependencies();r=self.binding['refs'];self.reference=_read(r['reference_model']['path'])
        if geometry_identity(self.reference)!=self.binding['geometry_identity']:raise ValueError('Reference geometry identity mismatch')
        self.source_clock=copy.deepcopy(self.reference['source_clock']);self.reference_stage=self.binding['reference_stage']
        seed=np.load(r['compact_arrays']['path'],allow_pickle=False);tr=np.load(r['transfer']['path'],allow_pickle=False);s01=np.load(r['S01_witnesses']['path'],allow_pickle=False);line=np.load(r['native50_bridge']['path'],allow_pickle=False);pr=_read(r['transfer_report']['path'])
        p=tr['native_pts'];sn=tr['shot_number'];H=seed['H_reference_xy_to_native']@np.asarray(pr['template']['current_to_reference_H']);pixels=np.broadcast_to(np.asarray([(u,v) for v in (180.,360.,540.,720.,900.) for u in (180.,440.,700.,960.,1220.,1480.,1740.)]),tr['witness_mask'].shape+(2,));xy=_project(np.linalg.inv(H),pixels);X=np.concatenate([xy,np.zeros(xy.shape[:-1]+(1,))],-1)
        self.pts=np.r_[s01['native_pts'],p];self.shots=np.r_[np.ones(len(s01['native_pts']),dtype=np.uint8),sn];self.X=np.concatenate([s01['world'],X]);self.target=np.concatenate([s01['target_uv'],pixels]);self.mask=np.concatenate([s01['witness_mask'],tr['witness_mask']]);self.mode=line['bridge_mode'];self.inferred=line['inferred'];self.direct=line['direct_mapping']
        if not np.array_equal(self.pts,line['native_pts']) or not np.array_equal(self.shots,line['shot_number']) or not np.all(np.diff(self.pts)==256):raise ValueError('Full9 native PTS/shot lineage mismatch')
        old=np.load(r['reference_metrics']['path'],allow_pickle=False);pref=self.reference_stage+'_';oldvalid=np.unpackbits(old[pref+'valid_packbits'],count=tr['witness_mask'].size).reshape(tr['witness_mask'].shape).astype(bool)
        self.reference_valid=np.concatenate([s01['baseline_valid'],oldvalid]);sa,sb,sc=_frame_stats(s01['baseline_rs_error_px'],s01['baseline_valid'],s01['witness_mask']);self.ref_sse=np.r_[sa,old[pref+'frame_valid_sse']];self.ref_den=np.r_[sb,old[pref+'frame_denominator']];self.ref_bad=np.r_[sc,old[pref+'frame_failed']]
        a=np.load(r['source_arrays']['path'],allow_pickle=False);self.metadata=_read(r['source_metadata']['path']);rows=np.flatnonzero(a['kind']==1)
        if len(rows)!=2316:raise ValueError('All2316 original fixed-source anchors are required')
        self.rows=rows;self.raw=a['raw_index'][rows];self.uv=a['uv'][rows];self.apt=a['native_pts'][rows];self.ash=a['shot_index'][rows];self.mi=a['material_index'][rows];self.pi=a['path'][rows];self.entity=[]
        for mi,pi in zip(self.mi,self.pi):
            material=self.reference['materials'][int(mi)];path=material['paths'][int(pi)];self.entity.append(material['label']+'/'+path['id'])
        self.entity=np.asarray(self.entity);self.anchor_reference_error=old[pref+'anchor_error_px'];self.anchor_reference_valid=np.unpackbits(old[pref+'anchor_valid_packbits'],count=len(rows)).astype(bool)
        ii=np.searchsorted(self.pts,self.apt)
        if np.any(ii==len(self.pts)) or not np.array_equal(self.pts[ii],self.apt):raise ValueError('Original source anchor outside native comparison clock')
        self.anchor_mode=self.mode[ii]
        self.assert_dependencies()
    def assert_dependencies(self):
        for name,ref in self.binding['refs'].items():
            if _sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen comparison dependency changed: '+name)
        for ref in self.binding['producer_refs']:
            if _sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen comparison producer changed: '+ref['path'])
    def _validate_model_identity(self,model):
        if model.get('source_clock')!=self.source_clock:raise ValueError('Candidate source clock differs')
        if set(model['rigs'])!=set(self.reference['rigs']) or set(model['shots'])!=set(self.reference['shots']):raise ValueError('Require one scene/three rigs/nine motions')
        for sid in self.reference['shots']:
            if model['shots'][sid]['rig_id']!=self.reference['shots'][sid]['rig_id']:raise ValueError('Shot owning rig changed: '+sid)
    def _anchors(self,model):
        scene=NativeScene(as_event_view(model,'s01'));world=[]
        for mi,pi,entity in zip(self.mi,self.pi,self.entity):
            mi,pi=int(mi),int(pi);material=model['materials'][mi];path=scene.path(mi,pi)
            if material['label']+'/'+path['id']!=entity or not path.get('physical_point') or len(path.get('points',[]))!=1:raise ValueError('Original fixed-source entity identity changed: '+entity)
            world.append(scene.world(mi,pi,0.))
        world=np.asarray(world);pred=np.full(self.uv.shape,np.nan);valid=np.zeros(len(world),bool);exceptions={}
        for si,sid in enumerate(self.metadata['shot_ids']):
            sel=self.ash==si
            try:
                cam=_camera(model,sid);r=cam.project_rs(world[sel],(self.apt[sel]-model['shots'][sid]['time_origin_pts'])/12800.);pred[sel]=r['uv'];valid[sel]=r['valid']
            except Exception as e:exceptions[sid]=type(e).__name__+': '+str(e)
        error=np.linalg.norm(pred-self.uv,axis=1);byshot={};byentity={};byrole={}
        def group(sel):return _delta(_metric(error[sel],valid[sel]),_metric(self.anchor_reference_error[sel],self.anchor_reference_valid[sel]))
        for si,sid in enumerate(self.metadata['shot_ids']):
            sel=self.ash==si;record=group(sel);fp=self.pts[self.shots==si+1];fs=np.zeros(len(fp));fd=np.zeros(len(fp),int);fb=np.zeros(len(fp),int);ex=np.zeros(len(fp),bool)
            for j,p in enumerate(fp):
                q=sel&(self.apt==p);fd[j]=int(q.sum());ex[j]=bool(fd[j]);good=q&valid&np.isfinite(error);fb[j]=fd[j]-int(good.sum());fs[j]=float(np.sum(error[good]**2))
            record.update(source_exposures=int(ex.sum()),native_frames_in_shot=len(fp),temporal=_windows(fp,fs,fd,fb,ex));byshot[sid]=record
        for entity in np.unique(self.entity):byentity[str(entity)]=group(self.entity==entity)
        for mode in range(5):byrole[str(mode)]=group(self.anchor_mode==mode)
        return dict(scope='all2316 original kind1 fixed physical source anchors; actual candidate NativeScene.world recomputed, UV/ID/PTS unchanged',all=group(np.ones(len(error),bool)),by_shot=byshot,by_entity=byentity,by_bridge_mode=byrole,exceptions=exceptions,all_original_rows_retained=True),dict(anchor_error_px=error,anchor_valid=valid,anchor_predicted_uv=pred,anchor_raw_index=self.raw)
    def compare(self,model_path,output_dir):
        self.assert_dependencies();mp=Path(model_path).resolve();model=_read(mp);self._validate_model_identity(model);out=Path(output_dir).resolve()
        if out.exists():raise FileExistsError('Use a new comparison output generation')
        out.mkdir(parents=True);digest=geometry_identity(model);same=digest==self.binding['geometry_identity'];source,arr=self._anchors(model)
        report=dict(schema='rinkfit.source_quality_comparison.v1',binding=_ref(self.path),candidate=_ref(mp),reference_model=self.binding['refs']['reference_model'],reference_stage=self.reference_stage,quality_accepted=False,automatic_acceptance_supported=False,combined_quality_score=None,source_improvement_cannot_cancel_grid_regression=True,scene_count=1,rig_count=3,shot_count=9,S09_rig=model['shots']['s09']['rig_id'],candidate_geometry_identity=digest,frozen_geometry_identity=self.binding['geometry_identity'],same_template_geometry=same,source_anchors=source,scope_limits=self.binding['scope_limits'],all_native_frames=8434,all_source_rows=2316)
        if not same:
            report['grid']=dict(status='unsupported_changed_geometry',reason='Refuse fixed v4 current-world XY for a changed scene/material geometry; entity carrier/conic operator required',original_frozen_denominator=int(self.mask.sum()),all_frames=8434,full_rms_px=None,source_rows_dropped=0,quality_comparison_unresolved=True)
        else:
            error=np.full(self.mask.shape,np.nan);valid=np.zeros(self.mask.shape,bool);exceptions={};domains={}
            for sn in range(1,10):
                sid=f's{sn:02d}';sel=self.shots==sn
                try:
                    cam=_camera(model,sid);r=cam.project_rs(self.X[sel],(self.pts[sel,None]-model['shots'][sid]['time_origin_pts'])/12800.);error[sel]=np.linalg.norm(r['uv']-self.target[sel],axis=-1);valid[sel]=r['valid'];mm=self.mask[sel];domains[sid]=dict(bracket_failed=int((mm&~r['bracketed']).sum()),certificate_failed=int((mm&~r['certificate']['certified']).sum()),returned_invalid=int((mm&~r['valid']).sum()),whole_Brown_disk=cam.optical_domain_certificate())
                except Exception as e:exceptions[sid]=type(e).__name__+': '+str(e)
            sse,den,bad=_frame_stats(error,valid,self.mask);byshot={};byrole={}
            def group(sel):return _delta(_aggregate(sse[sel],den[sel],bad[sel]),_aggregate(self.ref_sse[sel],self.ref_den[sel],self.ref_bad[sel]))
            for sn in range(1,10):
                sel=self.shots==sn;g=group(sel);g.update(frames=int(sel.sum()),temporal=_windows(self.pts[sel],sse[sel],den[sel],bad[sel]),new_failed_witnesses=int((self.mask[sel]&self.reference_valid[sel]&~valid[sel]).sum()),fixed_failed_witnesses=int((self.mask[sel]&~self.reference_valid[sel]&valid[sel]).sum()));byshot[f's{sn:02d}']=g
            for mode in range(5):byrole[str(mode)]=group(self.mode==mode)
            report['grid']=dict(status='evaluated_same_template_only',scope='S01 preserved d78 Brown/RS owner plus S02-S09 actualDemo planeH numerical transfer; not old8a28 full3D surface truth',all=group(np.ones(len(self.pts),bool)),by_shot=byshot,by_entity=dict(numeric_ice_transfer_grid=group(np.ones(len(self.pts),bool))),by_bridge_mode=byrole,domains=domains,exceptions=exceptions,original_mask_unchanged=True,all_failure_denominators_retained=True,full_3D_visibility_claimed=False)
            arr.update(grid_error_px=error,grid_valid=valid,grid_frame_valid_sse=sse,grid_frame_denominator=den,grid_frame_failed=bad)
        arr.update(native_pts=self.pts,shot_number=self.shots,bridge_mode=self.mode,grid_original_mask=self.mask,source_anchor_bridge_mode=self.anchor_mode)
        self.assert_dependencies();np.savez_compressed(out/'comparison_arrays.npz',**arr);report['arrays']=_ref(out/'comparison_arrays.npz');report['producer']={'path':__file__,'sha256':_sha(__file__)};(out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');return report

def compare_candidate(model_path,binding_path,output_dir):
    return SourceQualityComparison(binding_path).compare(model_path,output_dir)
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--binding',required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);a=ap.parse_args(argv);r=compare_candidate(a.model,a.binding,a.out);print(json.dumps(dict(output=str(Path(a.out).resolve()),quality_accepted=r['quality_accepted'],same_template_geometry=r['same_template_geometry'],grid_status=r['grid']['status'],source_anchors=r['source_anchors']['all'])))
if __name__=='__main__':main()
