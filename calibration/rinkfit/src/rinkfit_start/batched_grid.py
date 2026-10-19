import json,time,numpy as np
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture,select_certified_upper_bound
BATCH_STATS=dict(grid_task_count=0,grid_batches=0,grid_points=0,task_RS_iterations={},per_task_stopping_preserved=True,batch_points_target=4096)
class _GroupedExactRS:
 def __init__(self,camera,lengths):self.camera=camera;self.lengths=np.asarray(lengths,int);self.starts=np.r_[0,np.cumsum(self.lengths)[:-1]]
 def project_rs(self,world,frame_time,*,tolerance_px=1e-8,max_steps=48):
  camera=self.camera;X=np.asarray(world,float);t=np.asarray(frame_time,float);shape=np.broadcast_shapes(X.shape[:-1],t.shape);assert len(shape)==1;X=np.broadcast_to(X,shape+(3,));t=np.broadcast_to(t,shape);cert=camera.rs_certificate(X,t);left=np.zeros(shape);right=np.full(shape,1079.);pl=camera.at(X,t+camera.tau*(left/1079-.5));pr=camera.at(X,t+camera.tau*(right/1079-.5));fl=left-pl["uv"][...,1];fr=right-pr["uv"][...,1];bracket=(fl<=0)&(fr>=0)&pl["valid"]&pr["valid"];done=np.zeros(len(self.lengths),bool);iterations=np.zeros(len(done),int)
  for _ in range(max_steps):
   v=(left+right)/2;p=camera.at(X,t+camera.tau*(v/1079-.5));f=v-p["uv"][...,1];moving=np.repeat(~done,self.lengths);right=np.where(moving&(f>=0),v,right);left=np.where(moving&(f<0),v,left);iterations[~done]+=1;done|=np.maximum.reduceat(right-left,self.starts)<=tolerance_px
   if done.all():break
  for n in iterations:BATCH_STATS["task_RS_iterations"][str(n)]=BATCH_STATS["task_RS_iterations"].get(str(n),0)+1
  v=(left+right)/2;out=camera.at(X,t+camera.tau*(v/1079-.5));eq=v-out["uv"][...,1];valid=out["valid"]&bracket&cert["certified"]&(np.abs(eq)<=tolerance_px*2);out.update(valid=valid,row=v,root_equality_px=eq,bracketed=bracket,certificate=cert);return out

def _grid_batches(scene,camera,tasks,station,times,grid_count):
 def project(pack):
  lengths=[len(z[-1])for z in pack];mm=np.concatenate([np.full(len(z[-1]),z[2],int)for z in pack]);bb=np.concatenate([np.full(len(z[-1]),z[3],int)for z in pack]);ss=np.concatenate([z[-1]for z in pack]);tt=np.concatenate([np.full(len(z[-1]),times[z[4][0]])for z in pack]);view=_GroupedExactRS(camera,lengths);rr=_certified_candidates(scene,view,mm,bb,ss,tt);BATCH_STATS["grid_batches"]+=1;BATCH_STATS["grid_points"]+=len(ss);offset=0
  for task_id,frame,m,b,ii,grid in pack:
   end=offset+len(grid);yield task_id,frame,m,b,ii,grid,dict(uv=rr["uv"][offset:end],valid=rr["valid"][offset:end]);offset=end
 pack=[];size=0
 for task_id,((frame,m,b),rows)in enumerate(tasks.items()):
  ii=np.asarray(rows,int);path=scene.path(m,b);knots=np.arange(len(path.get("points",[])))/max(1,len(path.get("points",[]))-1);grid=np.unique(np.r_[np.linspace(0.,1.,grid_count),knots,station[ii]]);BATCH_STATS["grid_task_count"]+=1
  if pack and size+len(grid)>BATCH_STATS["batch_points_target"]:
   yield from project(pack);pack=[];size=0
  pack.append((task_id,frame,m,b,ii,grid));size+=len(grid)
 if pack:yield from project(pack)

def _true_rs_grid_recovery(scene,camera,material,frames,times,observed,ordinary,
                           expected_variants,station,branch,result,*,
                           same_branch_only,threshold_px=20.,grid_count=4097,
                           progress=None):
    """Extra computation on difficult rows, never an observation selection.

    Finite stations and every available polyline knot are tested with true RS.
    Visibility transition bisection retains an actually certified endpoint;
    neither the grid nor local refinement proves global nearest distance.
    """
    started=time.monotonic();error=np.linalg.norm(result['uv']-observed,axis=1)
    requested=np.flatnonzero(ordinary&((~result['valid'])|(error>threshold_px)))
    tasks={}
    for i in requested:
        paths=scene.model['materials'][int(material[i])]['paths']
        candidates=[int(branch[i])] if same_branch_only else [b for b,p in enumerate(paths) if p['variant_id']==expected_variants[i]]
        for b in candidates:tasks.setdefault((int(frames[i]),int(material[i]),b),[]).append(int(i))
    seed_s=station.copy();seed_b=branch.copy();seed_cost=np.full(len(station),np.inf)
    boundaries=[];projected=0
    def offer(ii,s,b,uv,valid):
        cost=np.sum((uv-observed[ii])**2,axis=1);take=valid&(cost<seed_cost[ii]);jj=ii[take]
        seed_cost[jj]=cost[take];seed_s[jj]=np.asarray(s)[take];seed_b[jj]=b
    for task_id,frame,m,b,ii,grid,gr in _grid_batches(scene,camera,tasks,station,times,grid_count):
        projected+=len(grid)
        valid_ids=np.flatnonzero(gr['valid'])
        if len(valid_ids):
            d=np.sum((gr['uv'][valid_ids][None,:,:]-observed[ii,None,:])**2,axis=2);jj=valid_ids[np.argmin(d,axis=1)]
            offer(ii,grid[jj],b,gr['uv'][jj],gr['valid'][jj])
        # This also catches camera-domain visibility boundaries, while every
        # retained candidate still needs the normal native RS certificate.
        edges=np.flatnonzero(gr['valid'][1:]!=gr['valid'][:-1])
        for edge in edges:
            boundaries.append((task_id,grid[edge],grid[edge+1],bool(gr['valid'][edge]),m,b,times[ii[0]]))
    if boundaries:
        lo=np.asarray([x[1]for x in boundaries]);hi=np.asarray([x[2]for x in boundaries]);left_valid=np.asarray([x[3]for x in boundaries])
        mm=np.asarray([x[4]for x in boundaries]);bb=np.asarray([x[5]for x in boundaries]);tt=np.asarray([x[6]for x in boundaries])
        for _ in range(36):
            middle=(lo+hi)*.5;rr=_certified_candidates(scene,camera,mm,bb,middle,tt);projected+=len(middle)
            left=rr['valid']==left_valid;lo=np.where(left,middle,lo);hi=np.where(left,hi,middle)
        edge_s=np.where(left_valid,lo,hi);edge_rs=_certified_candidates(scene,camera,mm,bb,edge_s,tt);projected+=len(edge_s)
        task_items=list(tasks.items())
        for j,item in enumerate(boundaries):
            ii=np.asarray(task_items[item[0]][1],int)
            offer(ii,np.full(len(ii),edge_s[j]),int(bb[j]),np.broadcast_to(edge_rs['uv'][j],(len(ii),2)),np.full(len(ii),edge_rs['valid'][j]))
    found=requested[np.isfinite(seed_cost[requested])]
    old_invalid=int(np.count_nonzero(~result['valid'][requested]));improved=0
    if len(found):
        candidate=_certified_candidates(scene,camera,material[found],seed_b[found],seed_s[found],times[found])
        old={k:v[found]for k,v in result.items()if isinstance(v,np.ndarray)and v.shape[:1]==(len(station),)}
        ss,bb,rr,take=select_certified_upper_bound(station[found],branch[found],old,seed_s[found],seed_b[found],candidate,observed[found])
        station[found]=ss;branch[found]=bb
        for k in result:
            if isinstance(result[k],np.ndarray)and k in rr:result[k][found]=rr[k]
        improved+=int(take.sum())
        local=_implicit_capture(scene,camera,material[found],seed_b[found],seed_s[found],times[found],observed[found],iterations=16,initial_row=candidate['row'])
        refined=_certified_candidates(scene,camera,material[found],seed_b[found],local,times[found])
        old={k:v[found]for k,v in result.items()if isinstance(v,np.ndarray)and v.shape[:1]==(len(station),)}
        ss,bb,rr,take=select_certified_upper_bound(station[found],branch[found],old,local,seed_b[found],refined,observed[found])
        station[found]=ss;branch[found]=bb
        for k in result:
            if isinstance(result[k],np.ndarray)and k in rr:result[k][found]=rr[k]
        improved+=int(take.sum())
    report=dict(requested_rows=len(requested),scheduling_threshold_px=threshold_px,
        rows_with_legal_RS_grid_candidate=len(found),rows_without_candidate=requested[~np.isfinite(seed_cost[requested])].tolist(),
        initially_invalid_requested_rows=old_invalid,finally_invalid_requested_rows=int(np.count_nonzero(~result['valid'][requested])),
        frame_material_branch_tasks=len(tasks),grid_and_boundary_RS_projections=projected,
        visibility_transition_brackets=len(boundaries),candidate_improvements=improved,
        same_branch_only=same_branch_only,global_nearest_certified=False,
        source_obligations_removed=0,elapsed_seconds=time.monotonic()-started)
    if progress:progress(json.dumps(dict(stage='difficult_rows_true_RS_grid',**report)),flush=True)
    return report

