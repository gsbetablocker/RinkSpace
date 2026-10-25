"""Compile CURRENT frozen native geometry for the common resident GPU factors."""
from pathlib import Path
from ..compat.d78_entities import native_definition
from ..domain.scene_assets import read_scene_assets, scene_content_identity
from ..domain.parameters import SceneParameterModel
from ..domain.print_assets import frozen_print_curves, resolve_instance

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
            "vertices":path.get("vertices",[]),"knots":[],"layer":path["layer"],"label":path["label"]}
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
        self.scene,self.paths=compile_geometry(scene,self.scene_path,core,parameter_state=parameter_state)
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

    def resident(self, *, camera, controls, path_ids, stations, pts, observed):
        """Bind source coordinates to existing paths; geometry has no data-side setter."""
        import numpy as np
        from ..core.gpu import ResidentResidualProblem
        ids=list(path_ids)
        missing=sorted(set(ids)-self.paths.keys())
        if missing:raise ValueError("Path is absent from the selected numeric native model: "+str(missing))
        native_pts=np.asarray(pts)
        if native_pts.ndim!=1 or native_pts.dtype.kind not in "iu" or len(native_pts)!=len(ids):
            raise ValueError("One native integer PTS is required per row")
        order=list(dict.fromkeys(ids));indices={name:i for i,name in enumerate(order)}
        times=np.array([(int(p)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"] for p in native_pts],np.float64)
        resident=ResidentResidualProblem(self.core,board=self.scene["board"],layout=self.scene["shared_layout"],
            options=self.scene["shared_geometry_options"],rig=camera["rig"],controls=controls,
            first_control_time=camera["first_control_time"],control_dt=camera["control_dt"],image_height=camera["image_height"],
            codes=np.array([indices[name] for name in ids],np.int32),stations=stations,times=times,observed=observed)
        try:resident.configure_geometry([self.paths[name] for name in order])
        except BaseException:resident.close();raise
        return resident
