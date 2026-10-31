"""Compile CURRENT frozen native geometry for the common resident GPU factors."""
from pathlib import Path
from ..compat.d78_entities import finite_path_support, native_definition
from ..domain.scene_assets import read_scene_assets, scene_content_identity
from ..domain.parameters import SceneParameterModel
from ..domain.print_assets import frozen_print_curves, resolve_instance

from ..domain.camera import complete_camera_rig

def compile_geometry(scene, scene_path, core, *, parameter_state=None):
    if scene.get("schema")!="rinkspace.field_model.v1":
        raise ValueError("GPU solve requires the current native field model, not legacy replay")
    state=parameter_state or SceneParameterModel(scene,scene_path).initial_state
    if state.layout["scene_content_revision"]!=scene_content_identity(scene,scene_path):
        raise ValueError("Parameter state belongs to a different native scene")
    current=state.apply_scene(scene)
    paths={}
    for path in current["paths"]:
        kind,parameters=native_definition(current,path,core)
        paths[path["id"]]={"id":path["id"],"kind":kind,"parameters":parameters,
            "vertices":path.get("vertices",[]),"knots":[],"layer":path["layer"],"label":path["label"],
            "support":None if kind==7 else finite_path_support(path)}
    for ref,assets in read_scene_assets(current,Path(scene_path)):
        for identity in ref.get("instance_ids",[ref.get("instance_id")]):
            instance,template,size,placement=resolve_instance(assets,identity)
            if size["values"] is None or placement["values"] is None:continue
            for path in frozen_print_curves(assets,identity,core,parameter_state=state):
                if path["id"] in paths:raise ValueError("duplicate current native path")
                paths[path["id"]]={**path,"layer":"advertisement","label":identity.rsplit("/",1)[-1]}
    return current,paths


class NativeSceneGPU:
    """One explicitly selected native model. Observation inputs cannot replace it."""
    def __init__(self, scene_path, core, *, parameter_state=None):
        import json
        self.core=core
        self.scene_path=Path(scene_path).resolve()
        scene=json.loads(self.scene_path.read_text(encoding="utf-8-sig"))
        self.parameter_state=parameter_state or SceneParameterModel(scene,self.scene_path).initial_state
        self.scene,self.paths=compile_geometry(scene,self.scene_path,core,parameter_state=self.parameter_state)
        self.scene_revision=scene_content_identity(scene,self.scene_path)
        self.unlocalized=[]
        for ref,assets in read_scene_assets(scene,self.scene_path):
            for identity in ref.get("instance_ids",[ref.get("instance_id")]):
                _,template,size,placement=resolve_instance(assets,identity)
                if size["values"] is None or placement["values"] is None:
                    self.unlocalized.append({"instance_id":identity,"template_id":template["id"],
                        "path_ids":[identity+"::"+p["id"] for p in template["paths"]],
                        "reason":"current_native_model_has_no_numeric_placement_or_size"})
        self.inventory={"path_ids":list(self.paths),"path_count":len(self.paths),
            "layer_counts":{layer:sum(p["layer"]==layer for p in self.paths.values()) for layer in ("ice","board","advertisement")},
            "unlocalized":self.unlocalized}

    def resident(self, *, camera, controls, path_ids, stations, pts, observed, scene_directions=None, scene_lower=None, scene_upper=None, motion_times=None, rigs=None, camera_ids=None):
        """Bind source coordinates to existing paths; geometry has no data-side setter."""
        import numpy as np
        from ..core.gpu import ResidentResidualProblem
        if self.scene.get('shared_geometry_options',0)&64:
            from ..domain.ice_tolerance import scene_physical_bounds
            required_lower,required_upper=scene_physical_bounds(self.parameter_state)
            scene_lower=required_lower if scene_lower is None else np.maximum(required_lower,scene_lower)
            scene_upper=required_upper if scene_upper is None else np.minimum(required_upper,scene_upper)
        ids=list(path_ids)
        missing=sorted(set(ids)-self.paths.keys())
        if missing:raise ValueError("Path is absent from the selected numeric native model: "+str(missing))
        native_pts=np.asarray(pts)
        if native_pts.ndim!=1 or native_pts.dtype.kind not in "iu" or len(native_pts)!=len(ids):
            raise ValueError("One native integer PTS is required per row")
        # Only observation-referenced paths belong to this numerical problem.
        # Compiling the whole display scene lets an unobserved installation
        # veto a shared-template update even though it supplies no evidence.
        # Shared parameter references are retained by the selected records.
        order=list(dict.fromkeys(ids));indices={name:i for i,name in enumerate(order)}
        times=np.array([(int(p)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"] for p in native_pts],np.float64)
        if motion_times is not None:
            # Disjoint shots can occupy separate spline-storage blocks. Their
            # clocks are translated, never rescaled; original PTS stay in data.
            times=np.asarray(motion_times,np.float64)
            if times.shape!=native_pts.shape or not np.isfinite(times).all():
                raise ValueError('One finite motion time is required per original source PTS')
        resident=ResidentResidualProblem(self.core,board=self.scene["board"],layout=self.scene["shared_layout"],
            options=self.scene["shared_geometry_options"],rig=complete_camera_rig(camera) if rigs is None else rigs,controls=controls,
            first_control_time=camera["first_control_time"],control_dt=camera["control_dt"],image_height=camera["image_height"],
            codes=np.array([indices[name] for name in ids],np.int32),stations=stations,times=times,observed=observed,camera_ids=camera_ids)
        try:
            resident.configure_geometry([self.paths[name] for name in order])
            if scene_directions is not None:
                records,data=compile_joint_scene_records(self.scene,self.scene_path,self.parameter_state,order,core=self.core)
                resident.configure_scene_parameters(values=self.parameter_state.values,records=records,data=data,
                    directions=scene_directions,lower=scene_lower,upper=scene_upper,charts=scene_coordinate_charts(self.parameter_state))
        except BaseException:resident.close();raise
        return resident


def bind_fixed_junctions(native_scene, features):
    """Add kind-4 fixed vertices derived only from current native parents/branch."""
    from ..domain.contracts import Feature
    from ..domain.junctions import evaluate_ice_board_junction
    records={f['id']:f for f in features}
    for record in features:
        if record['kind']!='fixed_junction':continue
        wanted=[record['id'],*record['parents']]
        values={}
        for identity in wanted:
            f=records[identity]
            values[identity]=Feature(f['id'],f['kind'],f['surface'],f['identity'],f['measurement'],
                tuple(f.get('parents',())),str(f['branch']) if 'branch' in f else None,
                f.get('has_station_parameter',False),f['label'],junction_relation=f.get('junction_relation'))
        if record.get('junction_relation')=='ice_board_bottom':
            value=evaluate_ice_board_junction(values[record['id']],values,native_scene.scene,native_scene.core,
                path_ids={k:records[k].get('path_id',k) for k in wanted})
        else:
            parents=[native_scene.paths[records[k].get('path_id',k)] for k in record['parents']]
            if any(p['layer']!='ice' for p in parents):
                raise ValueError('Fixed same-plane junction needs native ice parents')
            kinds=[p['kind'] for p in parents]
            def finite_line(p):
                points=native_scene.core.sample_path(native_scene.scene['board'],p['kind'],p['parameters'],
                    p.get('vertices',[]),p['support'])
                return [v for point in points for v in point]
            if kinds==[0,0]:
                if str(record['branch'])!='0':raise ValueError('Line-line junction has only branch 0')
                value=native_scene.core.line_line(*[finite_line(p) for p in parents])
            elif sorted(kinds)==[0,1]:
                line=parents[kinds.index(0)];circle=parents[kinds.index(1)]
                value=native_scene.core.line_circle(finite_line(line),circle['parameters'],int(record['branch']))
            else:raise ValueError('Native fixed junction requires line-line or line-circle parents')
        xyz=value['xyz'] if isinstance(value,dict) else value
        native_scene.paths[record['id']]=dict(id=record['id'],kind=4,parameters=[],vertices=[list(xyz)],knots=[],
            support=[0.,1.],layer=record['surface'],label=record['label'],parents=record['parents'],branch=record['branch'])


def compile_joint_scene_records(scene, scene_path, state, path_ids, *, core):
    """Compile references to existing shared blocks; never derive values from observations."""
    import numpy as np
    from ..domain.print_assets import _print_context, _print_seed
    records={};data=[]
    def append(values):
        offset=len(data);data.extend(np.asarray(values,np.float64).ravel().tolist());return offset
    for path in scene["paths"]:
        records[path["id"]]=[1 if "generated_code" in path else 0,int(path.get("generated_code",0)),0,0,0,0,0,0,0,0,0,0]
    for reference,assets in read_scene_assets(scene,Path(scene_path)):
        for identity in reference.get("instance_ids",[reference.get("instance_id")]):
            instance,template,size,placement=resolve_instance(assets,identity)
            if state[size["id"]] is None or state[placement["id"]] is None:continue
            context=_print_context(assets,identity,state)
            so=state.blocks[size["id"]]["offset"];po=state.blocks[placement["id"]]["offset"]
            internal=template.get("internal_geometry")
            domain=template.get("material_domain")
            if domain is not None and domain.get("kind")!="physical_unit_panel":
                raise ValueError("Unsupported declared print material domain")
            for path in template["paths"]:
                seed=_print_seed(path,context,core);vo=append(seed);family=shape=count=side=reverse=bounds=0
                binding=None if internal is None else internal["path_bindings"][path["id"]]
                if binding is not None:
                    kind=binding["kind"]
                    if kind in ("shape_group","rectangle_side"):
                        group=context["groups"][binding["group_id"]];shape_id=group["parameter_block_id"]
                        family=1 if kind=="shape_group" else 3
                        if family==1:bounds=append(group["reference_bounds"])
                        else:side=int(binding["side"]);reverse=int(binding["reverse"])
                    elif kind=="unfolded_square":family=2;shape_id=binding["parameter_block_id"]
                    elif kind!="panel_chart_gauge":raise ValueError("Unknown native print shape binding")
                    if family:shape=state.blocks[shape_id]["offset"];count=state.blocks[shape_id]["size"]
                records[identity+"::"+path["id"]]=[2,int(domain is not None),so,po,shape,count,family,side,reverse,vo,len(seed),bounds]
    missing=set(path_ids)-records.keys()
    if missing:raise ValueError("Native joint geometry needs explicit dynamic parents for "+str(sorted(missing)))
    return np.asarray([records[k] for k in path_ids],np.int32),np.asarray(data or [0.],np.float64)


def scene_coordinate_charts(state):
    """Native positive-dimension and ordered-radius charts; ambient values stay unchanged."""
    import numpy as np
    charts=np.zeros(len(state.values),np.int32)
    for block in state.blocks.values():
        if not block["available"]:continue
        offset=block["offset"]
        if block["kind"]=="board_shape":
            charts[offset+np.array([1,2,3,5])]=1;charts[offset+4]=2
        elif block["kind"]=="shared_geometry":
            name=block["id"].rsplit("/",1)[-1]
            positive={"curling_shared":[3,4],"lane_shared":[1],"hockey_shared":[1,2],
                "hockey_faceoff_shared":[2,4],"hockey_ends_ellipse":list(range(6)),
                "lane_transverse":[],"ice_tolerance":[]}.get(name)
            if positive is None:raise ValueError("Shared geometry needs a declared native coordinate chart: "+name)
            positive=block.get('positive_local_indices',positive)
            charts[offset+np.asarray(positive,dtype=int)]=1
            if name=="curling_shared":charts[offset+5:offset+8]=[3,4,5]
    return charts


def initialize_curve_stations(native, *, camera, controls, path_ids, stations, pts,
                              observed, kinds, motion_times=None, half_seconds=None, progress=None, backend='cuda'):
    """Seed named-curve nuisance coordinates using current native projections.

    Raw pixels, curve identity, camera and scene parameters are never altered.
    Invalid seed geometry remains an explicit failure, not a point exclusion.
    """
    import collections
    import numpy as np
    if backend not in ('cpu','cuda'):raise ValueError('Unknown native station initialization backend')
    groups=collections.defaultdict(list);samples={};stations=np.asarray(stations).copy()
    for row,(pt,path,kind) in enumerate(zip(pts,path_ids,kinds)):
        if kind==1:groups[(int(pt),path)].append(row)
    for _,name in groups:
        if name in samples:continue
        path=native.paths[name]
        if path['kind']==7:grid=np.linspace(0,native.core.board_perimeter(native.scene['board']),1025)
        elif path['kind']==8:grid=np.unique(np.r_[path['knots'],np.linspace(0,1,129)])
        else:grid=np.linspace(0,1,257 if path['kind'] in (1,6) else 129)
        samples[name]=grid
    unresolved=[];distance=np.full(len(stations),np.nan);done=0
    def process(items):
        nonlocal done
        names=[];ss=[];clock=[];tm=[];ranges=[]
        for (pt,name),rows in items:
            grid=samples[name];begin=len(ss);names.extend([name]*len(grid));ss.extend(grid);clock.extend([pt]*len(grid))
            if motion_times is not None:tm.extend([motion_times[rows[0]]]*len(grid))
            ranges.append((begin,len(ss),rows))
        ss=np.asarray(ss)
        with native.resident(camera=camera,controls=controls,path_ids=names,stations=ss,pts=np.asarray(clock,np.int64),
                observed=np.zeros((len(ss),2)),motion_times=np.asarray(tm) if motion_times is not None else None) as resident:
            if half_seconds is not None:resident.configure_projection_time_support(half_seconds=half_seconds)
            evaluated=resident.evaluate(backend)
        for begin,end,rows in ranges:
            uv=evaluated['values'][begin:end,3:5];status=evaluated['status'][begin:end];grid=ss[begin:end]
            valid=(status[:-1]==0)&(status[1:]==0)
            if not valid.any():unresolved.append(dict(row=rows[0],pts=int(pts[rows[0]]),path=path_ids[rows[0]],count=len(rows)));continue
            left=uv[:-1][valid];delta=np.diff(uv,axis=0)[valid];length=np.einsum('ij,ij->i',delta,delta);lo=grid[:-1][valid];hi=grid[1:][valid]
            for start in range(0,len(rows),128):
                rr=np.asarray(rows[start:start+128]);d=observed[rr,None,:]-left[None,:,:]
                t=np.clip(np.einsum('ijk,jk->ij',d,delta)/np.maximum(length,1e-300),0,1)
                dist=np.sum((d-t[:,:,None]*delta)**2,axis=2);j=np.argmin(dist,axis=1);ii=np.arange(len(rr))
                stations[rr]=lo[j]+t[ii,j]*(hi[j]-lo[j]);distance[rr]=np.sqrt(dist[ii,j])
            done+=1
        if progress:progress(dict(groups_done=done,groups_total=len(groups),unresolved_groups=len(unresolved)))
    batch=[];count=0
    for key,rows in groups.items():
        n=len(samples[key[1]])
        if batch and count+n>250000:process(batch);batch=[];count=0
        batch.append((key,rows));count+=n
    if batch:process(batch)
    return stations,dict(unresolved=unresolved,groups=len(groups),seed_error_percentiles=np.percentile(distance[np.isfinite(distance)],[0,50,95,100]).tolist())
