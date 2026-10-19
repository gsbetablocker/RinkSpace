"""Opt-in secondary source factory retaining the confirmed physical near cap.

The default qualified factory and production entries are unchanged. The old
full-perimeter near-visible factor stays intact: actual replay selected the
correct side, so no false wrong-facet claim or retrospective rebinding occurs.
Only the separately user-confirmed lower straight side receives source bands.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from z190_secondary_qualified_core_source import prepare_full_secondary_qualified_branches
from z190_secondary_nonzero_mount_full_source import write,ref
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_source_camera_diagnostic import absolute_value_jac15
from projection import to_state,from_state
from secondary_nearboard_top_source import (
    DEFAULT_REPORT,EDGE_SIGMA_PX,lower_long_side_top,
    nearboard_interval_residual,_read_report)

USER_VISUAL_HELD_PTS=230422016
NEAR_KIND='confirmed_lower_long_side_top_horizontal_support'


def confirmed_nearboard_rows(scene,aliases,*,report_path=DEFAULT_REPORT,
                            visual_held_pts=(USER_VISUAL_HELD_PTS,)):
    """Union confirmed same-exposure bands; source/roles remain auditable."""
    path=Path(report_path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    packet=_read_report(path,digest)
    world,geometry=lower_long_side_top(scene)
    by_actual={};source_frames=packet['frames']
    if len(source_frames)!=82:raise ValueError('Confirmed near source packet changed')
    for frame in source_frames:
        p=int(frame['native_pts'])
        if p not in aliases:raise ValueError('Confirmed near source native PTS missing')
        by_actual.setdefault(int(aliases[p]),[]).append(frame)
    held_actual={int(aliases[int(p)])for p in visual_held_pts}
    fit=[];held=[];unknown=[]
    for actual,frames in sorted(by_actual.items()):
        by_y={};members=sorted(int(f['native_pts'])for f in frames)
        for frame in frames:
            for m in frame['measurements']:
                y=int(m['y']);lo,hi=map(int,m['x_interval_px'])
                if not(0<=lo<hi<500 and 700<=y<=1070):raise ValueError('Bad source near interval')
                by_y.setdefault(y,[]).append((lo,hi,int(frame['native_pts'])))
        if not by_y:
            unknown.append(dict(canonical_pts=actual,native_members=members,
                source_measurements_empty=True,not_fitted=True));continue
        measurements=[dict(y=y,x_interval_px=[min(v[0]for v in vals),max(v[1]for v in vals)],
            source_native_pts=sorted(v[2]for v in vals))for y,vals in sorted(by_y.items())]
        lohi=np.array([m['x_interval_px']for m in measurements],float)
        mid=lohi.mean(axis=1);count=len(measurements)
        original_held=actual in held_actual
        row=dict(id='lower_long_side_board_top_confirmed_source_interval',
            pts=actual,split='held_visual_user_reported'if original_held else'fit',
            source_role='user_reported_visual_held'if original_held else'confirmed_continuous_source_development',
            world=world.copy(),uv=np.column_stack((mid,[m['y']for m in measurements])),
            sigma=np.full(count,EDGE_SIGMA_PX*np.sqrt(count)),point_correspondence=False,
            normal=np.tile([1.,0.],(count,1)),normal_interval=lohi-mid[:,None],
            curve_observation_kind=NEAR_KIND,source_horizontal_intervals=measurements,
            source_native_members=members,source_pixel_sha256={str(f['native_pts']):f['source_pixel_sha256']for f in frames},
            original_source_frame_records=frames,origin=str(path),source_packet_sha256=digest,
            physical_identity='lower_long_side_board_top',physical_identity_authority=packet['authority'],
            world_binding='exact current lower straight top edge; corners excluded; no UV-derived world station',
            sigma_modelling_decision='Original EDGE_SIGMA_PX=4 engineering outside-support sigma times sqrt(unique exposure scan rows); not measured covariance',
            source_interval_semantics=packet['measurement_policy'],source_UV_from_camera=False,
            new_factor_admission=True,actual_admission='held/reportonly'if original_held else'solver_development',
            not_blind=True,prior_read_source_packet=True)
        (held if original_held else fit).append(row)
    metadata=dict(source=ref(path),authority=packet['authority'],physical_identity=packet['physical_identity'],
        original_source_coverage=packet['coverage'],geometry=geometry,original_sigma_px=EDGE_SIGMA_PX,
        source_native_frames=len(source_frames),source_actual_exposures=len(by_actual),
        development_exposure_groups=len(fit),held_exposure_groups=len(held),unknown_exposure_groups=len(unknown),
        development_scan_rows=sum(len(r['uv'])for r in fit),held_scan_rows=sum(len(r['uv'])for r in held),
        held_actual_pts=sorted(held_actual),held_native_members=sorted(p for p,a in aliases.items()if a in held_actual),
        same_exposure_policy='Union original horizontal support by equal source scan Y; one physical group per actual exposure',
        old_full_perimeter_near_factor_UV_world_sigma_role_unchanged=True,
        physical_identity_read_from_source_and_user_confirmation_not_projection=True,
        no_new_world_geometry=True,unknown_groups=unknown,accuracy_accepted=False)
    return fit,held,metadata


def typed_nearboard_absolute_value_jac15(camera,rows,jacobian=True):
    """Original absolute factors plus original near support, physical15 units.

    The physical derivative has the same conditioned coordinates as the
    released absolute_value_jac15: C/100, rvec, logf, PP/1000, Brown4. Finite
    differences replay the checked original continuous-side support operator.
    """
    values=[];blocks=[]
    for row in rows:
        if row.get('curve_observation_kind')!=NEAR_KIND:
            out=absolute_value_jac15(camera,[row],jacobian)
            if jacobian:v,j=out;values.append(v);blocks.append(j)
            else:values.append(out)
            continue
        world=np.asarray(row['world'],float);measurements=row['source_horizontal_intervals']
        if len(measurements)!=len(row['uv']):raise ValueError('Typed near source row count changed')
        source_keys=tuple(map(int,row['source_native_members']))
        def value(cam):
            return nearboard_interval_residual(cam,world,measurements,source_keys=source_keys)
        raw=value(camera);scale=np.repeat(np.asarray(row['sigma'],float),2)
        values.append(raw/scale)
        if jacobian:
            state=np.r_[to_state(camera),np.asarray(camera,float)[11:15]]
            matrix=np.empty((len(raw),15),float)
            for col in range(15):
                width=1e-6*max(1.,abs(float(state[col])));high=state.copy();low=state.copy()
                high[col]+=width;low[col]-=width
                matrix[:,col]=(value(from_state(high,camera))-value(from_state(low,camera)))/(2*width)
            blocks.append(matrix/scale[:,None])
    result=np.concatenate(values)if values else np.empty(0)
    if not jacobian:return result
    return result,np.vstack(blocks)if blocks else np.empty((0,15))


def prepare_full_secondary_confirmed_nearboard(output,candidate_scene_path=None,**kwargs):
    """New opt-in bundle: full original graph plus qualified near development."""
    output=Path(output);prepared=prepare_full_secondary_qualified_branches(output,candidate_scene_path,**kwargs)
    old=prepared['bundle'];fit,held,meta=confirmed_nearboard_rows(prepared['scene'],prepared['aliases'])
    groups={int(p):list(rr)for p,rr in old.abs_groups}
    for row in fit:groups.setdefault(int(row['pts']),[]).append(row)
    abs_groups=[(int(p),groups[int(p)])for p in old.stamps if groups.get(int(p))]
    sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rr in abs_groups for r in rr])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,
        motion_sigma=old.motion_sigma,abs_groups=abs_groups,native_residual_sigma=np.r_[old.motion_sigma,sigma])
    bundle=MovingPivotSourceBundle(base,old.calibration,old.motion_kernel,
        reference_pts=old.reference_pts,initial_reference_optical_height_scene=old.initial_reference_optical_height_scene,
        finite_planes=old.finite_planes.copy(),finite_records=list(old.finite_records),scene=prepared['scene'],
        absolute_evaluator=typed_nearboard_absolute_value_jac15)
    prepared.update(bundle=bundle,confirmed_nearboard_fit_rows=fit,
        confirmed_nearboard_held_rows=held,confirmed_nearboard_metadata=meta)
    write(output/'confirmed_nearboard_admission.json',dict(metadata=meta,development_rows=fit,held_rows=held,
        production_changed=False,old_original_near_factor_not_replaced=True))
    return prepared
