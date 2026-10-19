"""Only the seven existing Ad4 operators: signed values and static geometry J."""
from copy import copy
from collections import OrderedDict
import json,time
import cv2
import joint_absolute_source_context_20261011 as ctx
import audit_low6_private_RF4_causal_response_20261011 as local
from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
from calibration.solver_runtime.source_loss import source_irls_weights
from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
from joint_absolute_subspace_diagnostics_20261011 import analyze_subspaces
a,np=ctx.a,ctx.np

started=time.perf_counter();c=ctx.load_context();ctx.baseline(c,set(local.SIX))
assert len(c.groups)==7 and sum(g['scalar_rows']for g in c.groups)==40
logical=set(c.board.geometry.material_parameter_indices(4).tolist())
for g in c.groups:
    if g['source_kind']=='original':logical.update(np.flatnonzero(c.provider.source_parameter_activity(g['row'],c.world)).tolist())
logical=np.asarray(sorted(logical),int);columns=c.world_indices[logical]
names=list(c.provider.parameter_names)
print(json.dumps(dict(stage='Ad4_only_geometry_scope',columns=[dict(logical=int(k),evaluation=int(col),name=names[k],value=float(c.q[col]))for k,col in zip(logical,columns)],seconds=time.perf_counter()-started)),flush=True)
native=[g for g in c.groups if g['source_kind']=='native'];source=c.board
Js=[]
for multiplier in (1.,.5):
    view=copy(source);view.rows=[g['row']for g in native];view.scales=np.concatenate([g['scales']for g in native])
    view.source_point_qualification=None;view._active_fit_view_cache=None;view._cache=OrderedDict()
    view.camera_steps={int(col):1e-6*multiplier for col in columns}
    cams={g['actual_pts']:ctx.camera(c,c.q,g['actual_pts'])for g in native};assoc=view._nominal_associations(c.q,cams)
    nominal=np.concatenate([g['actual']for g in native])
    def values(q,indices):return view._derivative_residual(q,cameras=cams,associations=assoc,apply_optical_support=True,row_indices=indices)
    np.testing.assert_allclose(values(c.q,np.arange(len(native))),nominal,rtol=0,atol=1e-10)
    J=native_factor_dependency_jacobian(view,c.q,view.rows,columns,columns,nominal,values,np.empty(0,int))
    original=next(g for g in c.groups if g['source_kind']=='original')
    G=evaluate_geometry_group(c.provider,c.world,ctx.camera(c,c.q,original['actual_pts']),[original['row']],c.calls['low'],1e-6*multiplier)
    Js.append(np.vstack((J,G[:,logical])))
    print(json.dumps(dict(stage='Ad4_only_geometry_J',multiplier=multiplier,shape=list(Js[-1].shape),seconds=time.perf_counter()-started)),flush=True)

r=np.concatenate([g['actual']for g in c.groups]);w=np.concatenate([source_irls_weights(g['actual'],g['quadratic'],g['loss'])for g in c.groups])
printcols=np.asarray([i for i,name in enumerate([names[k]for k in logical])if name.startswith('Ad4_')],int)
rows=np.repeat(np.arange(len(c.groups)),[g['scalar_rows']for g in c.groups]);empty=np.empty((len(r),0))
answer={}
for name,select in [('Ad4_print_only',printcols),('Ad4_and_current_course',np.arange(len(columns)))]:
    result=analyze_subspaces(r,empty,Js[0][:,select],empty,empty,Js[1][:,select],empty,w,rows)
    answer[name]=result
    print(json.dumps(dict(stage='Ad4_static_geometry_projection',model=name,summary=result['summary']),default=a.plain),flush=True)

table=[]
for g in c.groups:
    camera=ctx.camera(c,c.q,g['actual_pts']);pixel=(g['actual']*g['scales']).reshape(-1,2)
    normal=g['row'].get('normal');normal=None if normal is None else np.asarray(normal,float)
    signed=None if normal is None else np.einsum('ni,ni->n',pixel,normal)
    table.append(dict(actual_pts=g['actual_pts'],id=g['id'],source_kind=g['source_kind'],uv=np.asarray(g['row']['uv']).tolist(),
        native_model_minus_source_xy_px=pixel.tolist(),signed_source_normal_error_px=None if signed is None else signed.tolist(),
        source_normals=None if normal is None else normal.tolist(),normal_interval=g['row'].get('normal_interval'),
        interval_is_location_dead_zone=bool(g['source_kind']=='native'and source._uses_optical_support(g['row'])),
        sigma_px=np.asarray(g['row']['sigma']).tolist(),focal_px=float(np.exp(camera[6])),Rodrigues_R3=camera[3:6].tolist()))
print(json.dumps(dict(stage='Ad4_signed_pose_table',rows=table),default=a.plain),flush=True)
pairs=[]
for i,at in enumerate(local.SIX):
    ca=ctx.camera(c,c.q,at);ra=cv2.Rodrigues(ca[3:6])[0]
    for bt in local.SIX[i+1:]:
        cb=ctx.camera(c,c.q,bt);rb=cv2.Rodrigues(cb[3:6])[0]
        angle=float(np.degrees(np.arccos(np.clip((np.trace(rb@ra.T)-1)/2,-1,1))))
        pairs.append(dict(source_pts=at,target_pts=bt,seconds=(bt-at)/15360,relative_rotation_degrees=angle,focal_ratio=float(np.exp(cb[6]-ca[6]))))
print(json.dumps(dict(stage='Ad4_all_pose_pairs',pairs=pairs)),flush=True)
arrays=dict(residual=r,weights=w,J_h=Js[0],J_half=Js[1],geometry_logical_indices=logical,geometry_evaluation_columns=columns)
out=a.H/'LOW6_Ad4_only_static_geometry_small_response_20261011.npz';np.savez_compressed(out,**arrays)
report=dict(columns=[dict(logical=int(k),evaluation=int(col),name=names[k])for k,col in zip(logical,columns)],source_table=table,pose_pairs=pairs,
    static_geometry_models={name:result['summary']for name,result in answer.items()},arrays=a.ref(out),no_q_no_direction_no_fit_no_images=True,seconds=time.perf_counter()-started)
outjson=out.with_suffix('.json');outjson.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=a.plain)+'\n',encoding='utf8')
print(json.dumps(dict(stage='Ad4_small_response_complete',path=str(outjson),seconds=time.perf_counter()-started)),flush=True)
