"""Freeze material index, label and path identities independently of numbers."""
from pathlib import Path
from .packet_storage import write_packet
import json
import numpy as np
from .io_utils import file_sha256,write_json,object_sha256

def bind_material_topology(packet_path,model_path,material_map,output):
    src=Path(packet_path);p=json.loads(src.read_text(encoding='utf8'));model_path=Path(model_path);model=json.loads(model_path.read_text(encoding='utf8'));out=Path(output)
    if out.exists():raise ValueError('new binding generation required')
    out.mkdir(parents=True);cache={}
    def ref(s):
        path=Path(s['file']);path=path if path.is_absolute() else src.parent/path;k=(str(path),s['sha256'])
        if k not in cache:
            if file_sha256(path)!=s['sha256']:raise ValueError('packet numeric dependency changed during material binding')
            cache[k]=True
        return {**s,'file':str(path)}
    arrays={k:ref(v) for k,v in p['arrays'].items()};groups=[{**g,'W':ref(g['W'])} for g in p['groups']]
    material=np.fromfile(arrays['material_index']['file'],arrays['material_index']['dtype']);ad=np.fromfile(arrays['admission']['file'],arrays['admission']['dtype']);role=np.fromfile(arrays['fit_role']['file'],arrays['fit_role']['dtype']);active=(ad==1)&(role==1);count=len(model['materials'])
    if np.any(material[active]>=count):raise ValueError('active source material lies outside frozen model prefix')
    table=[{'index':i,'label':m['label'],'paths':[{'index':j,'id':q['id'],'physical_point':bool(q.get('physical_point',False))} for j,q in enumerate(m['paths'])]} for i,m in enumerate(model['materials'])]
    binding={'schema':'rinkfit.joint_material_binding.v1','model_material_prefix_count':count,'materials':table,
        'source_material_map':{'file':str(Path(material_map)),'sha256':file_sha256(material_map)},'model_topology_ref':{'file':str(model_path),'sha256':file_sha256(model_path)},
        'topological_identity_table_sha256':object_sha256(table),'numeric_model_values_are_part_of_identity_binding':False,'allow_append_only_model_material_suffix':True}
    result={**p,'arrays':arrays,'groups':groups,'material_binding':binding,'material_binding_parent_packet':{'file':str(src),'sha256':file_sha256(src)}}
    write_packet(out/'packet.json',result);report={'schema':'rinkfit.material_index_identity_binding_receipt.v1','raw_rows':p['row_count'],'active_rows':int(active.sum()),'model_material_prefix_count':count,'active_materials':len(np.unique(material[active])),
        'source_arrays_W_or_membership_changed':False,'packet_sha256':file_sha256(out/'packet.json'),'topological_identity_table_sha256':binding['topological_identity_table_sha256']}
    write_json(out/'report.json',report);return report
