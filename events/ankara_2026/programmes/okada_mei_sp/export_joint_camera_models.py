"""Keep compact final camera/scene parameters in OneDrive, full run arrays in Data."""
from pathlib import Path
import hashlib,json
import numpy as np
import prepare_native_joint_shots as prepared


def main():
    run=prepared.RUN;report=prepared.read(run/'camera_models/comparison.json');manifest=prepared.read(run/'joint_manifest.json')
    directory=prepared.HERE/'camera_experiments/joint_results_20261027'
    for rec in report['models']:
        path=directory/(rec['model']+'.npz')
        if len(str(path).encode('utf-16-le'))//2>=260 or any(len(p.encode('utf-16-le'))//2>255 for p in path.parts):raise ValueError('OneDrive path exceeds the Windows limit')
    directory.mkdir(exist_ok=True);records=[]
    for rec in report['models']:
        value=prepared.npz(rec['snapshot']);path=directory/(rec['model']+'.npz')
        metadata=dict(schema='rinkspace.joint_camera_parameter_export.v1',model=rec['model'],profile=rec['profile'],
            snapshot_sha256=rec['snapshot_sha256'],complete_checkpoint_in_data=rec['snapshot'],
            per_observation_stations_included=False,point_input_sha256=report['baseline']['point_input_sha256'],
            scene_definition=manifest['scene_path'],scene_file_sha256=manifest['scene_file_sha256'],
            shared_camera_definition=manifest['shared_camera_definition'],shots=manifest['shots'],
            control_dt=manifest['control_dt'],first_control_time=manifest['first_control_time'],
            native_library_provenance_sha256=str(value['native_library_provenance_sha256']),
            conditional_metres_per_L=report['conditional_metres_per_L'],physical_camera_validated=False)
        arrays={k:value[k] for k in ['rig','controls','scene_values','board','layout']};arrays['metadata_json']=np.array(json.dumps(metadata,ensure_ascii=False,sort_keys=True))
        # Reuse existing exports if their numerical snapshot identity is unchanged.
        same=False
        if path.exists():
            existing=prepared.npz(path);same=json.loads(str(existing['metadata_json']))==metadata
        if not same:np.savez(path,**arrays)
        records.append(dict(model=rec['model'],parameters=path.name,sha256=prepared.digest(path),metrics=rec['metrics'],physical=rec['physical'],numerical=rec['numerical']))
    prepared.write(directory/'index.json',dict(schema='rinkspace.joint_camera_parameter_exports.v1',baseline_sha256=report['baseline']['sha256'],
        completed_candidates=report['completed_candidates'],expected_candidates=report['expected_candidates'],
        all_source_rows_retained=True,production_baseline_replaced=False,models=records,
        local_comparison=str(run/'camera_models/comparison.html'),local_visual_comparison=str(run/'camera_models/visual_comparison.html')))
    print('PARAMETER_INDEX',directory/'index.json')

if __name__=='__main__':main()
