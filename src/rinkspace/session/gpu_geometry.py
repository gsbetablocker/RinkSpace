"""Compile CURRENT frozen native geometry for the common resident GPU factors."""
from pathlib import Path
from ..compat.d78_entities import finite_path_support, native_definition
from ..domain.scene_assets import read_scene_assets, scene_content_identity
from ..domain.parameters import SceneParameterModel
from ..domain.print_assets import frozen_print_curves, resolve_instance

def complete_camera_rig(camera):
    """Serialize the selected native camera, including its existing C1 wide hinge.

    The knot is a model-basis setting; all four coefficients have solver chart
    directions. An old proof that the hinge was inactive is provenance, not a
    permission to omit it after freely fitted zoom enters that region.
    """
    import numpy as np
    rig=np.asarray(camera['rig'],np.float64)
    if rig.shape==(43,):return rig.copy()
    if rig.shape!=(38,):raise ValueError('Unsupported native physical camera pack')
    extension=camera.get('brown_wide_extension')
    if extension is None:
        proofs=camera.get('inactive_extension_proofs',[])
        if proofs:
            if len(proofs)!=1 or proofs[0].get('kind')!='wide_brown_hinge':raise ValueError('Unsupported native optics extension')
            extension=proofs[0]['original_definition']
    if extension is None:return rig.copy()
    if extension.get('schema')!='rinkfit.source_supported_wide_brown_hinge.v1' or extension.get('power')!=2:raise ValueError('Unsupported native wide optics formula')
    wide=np.asarray([extension['activation_log_f_native'],*extension['coefficients']],np.float64)
    if wide.shape!=(5,) or not np.isfinite(wide).all():raise ValueError('Invalid current native wide optics values')
    return np.r_[rig,wide]


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

    def resident(self, *, camera, controls, path_ids, stations, pts, observed, scene_directions=None, scene_lower=None, scene_upper=None):
        """Bind source coordinates to existing paths; geometry has no data-side setter."""
        import numpy as np
        from ..core.gpu import ResidentResidualProblem
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
        resident=ResidentResidualProblem(self.core,board=self.scene["board"],layout=self.scene["shared_layout"],
            options=self.scene["shared_geometry_options"],rig=complete_camera_rig(camera),controls=controls,
            first_control_time=camera["first_control_time"],control_dt=camera["control_dt"],image_height=camera["image_height"],
            codes=np.array([indices[name] for name in ids],np.int32),stations=stations,times=times,observed=observed)
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
                "hockey_faceoff_shared":[2,4],"hockey_ends_ellipse":list(range(6))}.get(name)
            if positive is None:raise ValueError("Shared geometry needs a declared native coordinate chart: "+name)
            charts[offset+np.asarray(positive,dtype=int)]=1
            if name=="curling_shared":charts[offset+5:offset+8]=[3,4,5]
    return charts
