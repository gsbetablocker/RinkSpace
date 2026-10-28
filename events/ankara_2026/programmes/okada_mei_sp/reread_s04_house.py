"""Reuse the S01 joint material reader with CURRENT native S04 geometry.

Native geometry and camera navigate the source. Only the legacy reader's
actual current-RGB interface UVs are exported, never representative_uv.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from calibration.program_tracking.native_ice_material_repair import NestedHouseReader
from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
from rinkspace.compat.d78_entities import native_definition
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from review_points_s04 import CurrentNativeGrid,OUT,ADS,read,write
from ice_points_s04 import source_domain,REC
from rinkspace.review.point_connections import polyline_order


class _ExposureLookup:
    def __getitem__(self,pts):return pts


class NativeHouseReader(NestedHouseReader):
    def __init__(self,grid, *, motion_path=None):
        self.grid=grid
        scene=read(grid.manifest['scene_path'])
        model=SceneParameterModel(scene,Path(grid.manifest['scene_path']),optimize_scene=True)
        self.scene=ParameterState(model.initial_state.layout,grid.state['scene_values']).apply_scene(scene)
        self.groups={}
        self.native_paths={}
        self.cameras=_ExposureLookup()
        self.project=self._project
        self.state={}
        self.cache={}
        self.cache_pts=None
        self.motion={r['source_pts']:r for r in read(motion_path or OUT/'evidence/source_navigation.json')['pairs']}
        self.temporal={};self.navigation_record={}
        for path in self.scene['paths']:
            if path['layer']!='ice':continue
            kind,params=native_definition(self.scene,path,grid.core)
            if kind!=1:continue
            fid,role=path['label'].removeprefix('ice.').split(':')
            outline=path['measurement']=='centreline'
            if outline:role='outer'
            group=self.groups.setdefault(fid,dict(center=np.array(params[:2]),radii={},anchor=getattr(grid,'reference',745984),
                authority='current_native_shared_circle_family_and_registered_shot_state',paths=[],outline=outline))
            if not np.array_equal(group['center'],params[:2]):raise ValueError('Native house layers are not concentric')
            group['radii'][role]=params[3]
            group['paths'].append(dict(path,feature_id=fid))
            self.native_paths[(fid,role)]=(kind,params,path)
        for group in self.groups.values():
            group['radii']=dict(sorted(group['radii'].items(),key=lambda x:x[1],reverse=True))

    def initial_navigation(self,fid,index):
        prior=self.temporal.get(fid);motion=self.motion.get(index)
        proof=dict(kind='current_native_source_search',source_pixels_are_navigation_only=True)
        self.navigation_record[fid]=proof
        if prior is None or prior['pts']!=index-256 or not motion or motion['status']!='candidate':return np.zeros(6)
        if motion['diagnostic_median_reference_px']>2. or 2.*motion['diagnostic_p95_reference_px']>6.:return np.zeros(6)
        scale=np.diag([.5,.5,1.]);h=np.linalg.inv(scale)@np.linalg.inv(np.asarray(motion['H_reference_from_current']))@scale
        support=cv2.perspectiveTransform(prior['support'][None],h)[0]
        hull=np.asarray(motion['fit_support_current_xy'],np.float32)*2.
        supported=np.array([cv2.pointPolygonTest(hull,tuple(map(float,q)),True)>=-2. for q in support])
        if supported.sum()<3 or np.linalg.norm(np.ptp(support[supported],axis=0))<35.:return np.zeros(6)
        theta=np.linspace(0,2*np.pi,129)
        base=self.projection(self.groups[fid],index,'outer',theta,None)[0]
        old=self.projection(self.groups[fid],prior['pts'],'outer',theta,prior['parameters'])[0]
        moved=cv2.perspectiveTransform(old[None],h)[0]
        if not np.isfinite(base).all() or not np.isfinite(moved).all():return np.zeros(6)
        design=np.c_[base-[960.,540.],np.ones(len(base))]
        fit=np.linalg.lstsq(design,moved-[960.,540.],rcond=None)[0]
        initial=np.array([fit[2,0],fit[2,1],fit[0,0]-1,fit[1,0],fit[0,1],fit[1,1]-1])
        bound=np.array([90.,60.,.25,.18,.06,.25])if fid=='C0'else np.array([40.,30.,.06,.05,.025,.08])
        if np.any(abs(initial)>=bound):
            proof['temporal_navigation_outside_current_search_chart']=True
            return np.zeros(6)
        proof.update(kind='previous_actual_material_with_same_shot_image_motion',reference_pts=prior['pts'],
            current_supported_prior_readouts=int(supported.sum()),registration_p95_native_px=2.*motion['diagnostic_p95_reference_px'],
            source_motion_file=str(OUT/'evidence/source_navigation.json'))
        return initial

    def record_source_family(self,fid,pts,result):
        if not result:return
        rows=[r for data in result.values()for r in data['rows']]
        if len(rows)<3:return
        self.temporal[fid]=dict(pts=pts,support=np.asarray([r['uv']for r in rows]),
            parameters=np.asarray(next(iter(result.values()))['joint_group_navigation']['shared_source_affine_parameters']))

    def _project(self,pts,xyz):
        g=self.grid
        if hasattr(g,'project_xyz'):
            result=g.project_xyz(pts,xyz)
        else:
            time=(pts-g.shot['reference_pts'])/12800+g.timeline.offset(g.shot)
            result=g.core.project_motion(g.state['rig'],g.state['controls'],g.timeline.first,g.timeline.dt,1080,time,xyz)
        return result['uv'],result['depth']

    def conic(self,curve):
        # The caller's historical method name is retained, but this adapter
        # keeps the exact current native distorted curve, including offscreen
        # samples. No fitted ellipse replaces its visible geometry.
        return np.asarray(curve,float)

    def closest_source_curve_points(self,curve,points):
        order,_=polyline_order(points,curve)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        return np.column_stack([np.interp(order,arc,curve[:,a])for a in (0,1)])

    def transformed_conic_exact_distance(self,curve,points,parameters):
        projected=self.source_transform(curve,parameters)
        order,distance=polyline_order(points,projected)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(projected,axis=0),axis=1))]
        foot=np.column_stack([np.interp(order,arc,projected[:,a])for a in (0,1)])
        segment=np.clip(np.searchsorted(arc,order,side='right')-1,0,len(projected)-2)
        tangent=projected[segment+1]-projected[segment];delta=points-foot
        return distance*np.sign(tangent[:,0]*delta[:,1]-tangent[:,1]*delta[:,0])

    def projection(self,group,index,role,theta,parameters=None):
        if self.cache_pts!=index:self.cache.clear();self.cache_pts=index
        fid=group['paths'][0]['feature_id']
        theta=np.asarray(theta,float)
        key=(fid,role,theta.tobytes())
        if key not in self.cache:
            kind,params,path=self.native_paths[(fid,role)]
            stations=np.mod(theta/(2*np.pi),1.)
            xyz=np.asarray(self.grid.core.sample_path(self.scene['board'],kind,params,[],stations.tolist()))
            self.cache[key]=self._project(index,xyz)
        uv,depth=self.cache[key]
        return self.source_transform(uv,parameters),depth

    def read(self,fid,index,native):
        group=self.groups[fid]
        if fid=='C0' or not group['outline']:
            from calibration.program_tracking.native_ice_short_arc_repair import short_native_outer_arc
            # Occluded, flattened circles cannot reliably determine a new
            # affine shape. Reuse the existing observable translation reader.
            # All physical layers retain the exact current native projection.
            output,diagnostics=short_native_outer_arc(self,fid,index,native,allow_outline=True,preserve_native_projection=True)
            diagnostics['source_navigation']='current_native_circle_shape_with_observable_source_translation_only'
            if output and not group['outline']:
                navigation=output['outer']['joint_group_navigation']
                parameters=np.asarray(navigation['shared_source_affine_parameters'])
                raw,extra=self.measurements(group,index,native,parameters)
                for role in group['radii']:
                    if role=='outer':continue
                    measured=[r for r in raw if r['role']==role]
                    if not measured:continue
                    base=self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0]
                    error=np.abs(self.transformed_conic_exact_distance(base,np.asarray([r['uv']for r in measured]),parameters))
                    measured=[r for r,e in zip(measured,error)if e<=3.]
                    if not measured:continue
                    output[role]=dict(rows=measured,curve=self.source_transform(base,parameters),
                        joint_group_navigation=dict(navigation,current_source_readout_count=len(measured),
                            kind='current_native_nested_layer_with_same_observable_family_translation',
                            maximum_true_euclidean_material_to_fitted_curve_px=float(error[error<=3.].max())))
                diagnostics['nested_current_source_support']=extra
            return output,diagnostics
        output,diagnostics=super().read(fid,index,native)
        group=self.groups[fid]
        diagnostics['temporal_source_navigation']=self.navigation_record.get(fid,{})
        if group['outline']:return output,diagnostics
        if output:
            parameters=np.asarray(next(iter(output.values()))['joint_group_navigation']['shared_source_affine_parameters'])
        else:
            raw,_=self.measurements(group,index,native,self.initial_navigation(fid,index))
            if len(raw)<3:return output,diagnostics
            conics={role:self.conic(self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0])for role in group['radii']}
            # A cropped family with three real reads cannot determine the six
            # affine navigation coordinates. Reuse the same physical family
            # with only a source-image translation, never invent missing UVs.
            def residual(translation):
                z=np.r_[translation,np.zeros(4)]
                return np.concatenate([self.transformed_conic_exact_distance(conics[role],
                    np.array([r['uv']for r in raw if r['role']==role]),z)
                    for role in conics if any(r['role']==role for r in raw)])
            fit=least_squares(residual,np.zeros(2),bounds=([-40.,-30.],[40.,30.]),loss='soft_l1',f_scale=1.5,max_nfev=50)
            parameters=np.r_[fit.x,np.zeros(4)]
        raw,extra=self.measurements(group,index,native,parameters)
        for role in group['radii']:
            if role in output:continue
            measured=[r for r in raw if r['role']==role]
            if not measured:continue
            base=self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0]
            error=np.abs(self.transformed_conic_exact_distance(self.conic(base),np.asarray([r['uv']for r in measured]),parameters))
            measured=[r for r,e in zip(measured,error)if e<=3.]
            if not measured:continue
            output[role]=dict(rows=measured,curve=self.source_transform(base,parameters),
                joint_group_navigation=dict(kind='partial_current_RGB_support_in_one_native_circle_family',
                    shared_source_affine_parameters=parameters.tolist(),native_family=fid,
                    no_missing_curve_samples_filled=True,no_independent_layer_geometry=True,
                    source_support_count=len(measured),whole_circle_reconstruction_claimed=False))
        diagnostics['partial_source_support']=extra
        return output,diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pts',type=int,nargs='+')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--only',nargs='+',help='Reread these circle families, retaining all other saved actual readings')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    grid=CurrentNativeGrid();reader=NativeHouseReader(grid)
    visibility={r['pts']:r for r in read(ADS/'evidence/visibility.json')['frames']}
    dest=args.output_dir or OUT/'evidence/joint_house_RGB'
    dest.mkdir(parents=True,exist_ok=True)
    algorithm=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'calibration/program_tracking/native_ice_material_repair.py').read_bytes()
        +(ROOT/'calibration/program_tracking/native_ice_short_arc_repair.py').read_bytes()).hexdigest()
    requested=sorted(visibility) if args.all else args.pts
    if not requested:parser.error('Use --all or explicit original --pts')
    for frame_index,pts in enumerate(requested):
        start=time.monotonic();vis=visibility[pts]
        im=cv2.imread(vis['image']);mask=cv2.imread(vis['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=vis['source_BGR_sha256']:raise ValueError('Wrong source exposure')
        board=read(REC/f'dense_board_v1/qualified_frames/{pts}.json')
        graphics,border,_,_=source_domain(im,mask,board)
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=excluded)
        features={};diagnostics={};frame_results={};shape_features=[]
        for fid in reader.groups:
            if args.only and fid not in args.only:continue
            # The inherited implementation requires a complete finite projected
            # family to set its search conics. Missing projection is explicit.
            curves=[reader.projection(reader.groups[fid],pts,role,np.linspace(0,2*np.pi,513))[0] for role in reader.groups[fid]['radii']]
            if not all(np.isfinite(v).all() for v in curves):
                diagnostics[fid]=dict(status='native_search_family_outside_projection_domain')
                continue
            cloud=np.vstack(curves)
            if (cloud[:,0].max() < -75 or cloud[:,0].min()>1995 or cloud[:,1].max() < -75 or cloud[:,1].min()>1155):
                diagnostics[fid]=dict(status='native_search_family_outside_source_image')
                continue
            result,diag=reader.read(fid,pts,native)
            diagnostics[fid]=diag
            frame_results[fid]=result
            for role,data in result.items():
                path=reader.native_paths[(fid,role)][2]
                observed=[]
                for row in data['rows']:
                    r={k:v for k,v in row.items() if k not in ('representative_uv','representative_to_actual_material_px')}
                    r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pts,
                        source_row_id=f's04_joint_house_RGB/{pts}/{fid}/{role}/{row["query_index"]}')
                    observed.append(r)
                features[path['label']]=dict(native_path_id=path['id'],rows=observed,
                    navigation=data['joint_group_navigation'])
                ellipse=cv2.fitEllipse(np.asarray(data['curve'],np.float32))
                shape_features.append(dict(target_id=path['label'],measurement_operator='paint_center'if reader.groups[fid]['outline']else'material_interface',
                    segments=[dict(uv=[r['uv']for r in observed],source_evidence=[r['source_evidence']for r in observed])],
                    geometric_interpretation=dict(kind='projected_physical_circle_ellipse',center_uv=ellipse[0],axes_px=ellipse[1],angle_degrees=ellipse[2])))
        material_conflicts=conflicts(shape_features)
        contaminated={r['target_id']for r in material_conflicts}
        rejected={name:features.pop(name)for name in contaminated}
        for fid,result in frame_results.items():
            if any(name.startswith('ice.'+fid+':')for name in contaminated):continue
            reader.record_source_family(fid,pts,result)
        record=dict(source_pts=pts,source_BGR_sha256=vis['source_BGR_sha256'],algorithm_sha256=algorithm,
            features=features,diagnostics=diagnostics,native_authority=grid.receipt,
            material_identity_conflicts=material_conflicts,rejected_source_candidates=rejected,
            original_sources_modified=False,model_or_fitted_curve_pixels_emitted=False,
            implementation='calibration.program_tracking.native_ice_material_repair.NestedHouseReader',
            seconds=time.monotonic()-start)
        target=dest/f'{pts}.json';pending=dest/f'{pts}.pending.json'
        if args.only:
            old=read(target)
            if old['source_BGR_sha256']!=record['source_BGR_sha256']:raise ValueError('Partial repair source mismatch')
            old['features']={k:v for k,v in old['features'].items()if k.removeprefix('ice.').split(':')[0]not in args.only}
            old['features'].update(features);old['diagnostics'].update(diagnostics)
            old.setdefault('incremental_reader_repairs',[]).append(dict(families=args.only,
                algorithm_sha256=algorithm,seconds=record['seconds'],actual_RGB_reread=True))
            record=old
        write(pending,record);pending.replace(target)
        if not args.all or frame_index%50==0 or frame_index==len(requested)-1:
            print(json.dumps(dict(frame=frame_index+1,total=len(requested),pts=pts,
                counts={k:len(v['rows'])for k,v in features.items()},seconds=record['seconds']),ensure_ascii=False),flush=True)
    grid.unchanged()


if __name__=='__main__':main()
