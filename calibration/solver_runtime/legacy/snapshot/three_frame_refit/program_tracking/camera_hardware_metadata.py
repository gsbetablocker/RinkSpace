"""Load source-scoped hardware context without changing calibration values.

The registry distinguishes manufacturer capabilities, the team's model-family
reply, per-installation assumptions, and final encoded-video measurements.
It supplies no K, Brown, physical sensor width, frame-rate conversion or pose.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_module_dir as _runtime_data_module_dir
import argparse,copy,hashlib,json
from fractions import Fraction
from pathlib import Path

ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
DEFAULT_REGISTRY=ROOT/'configs/camera_hardware/sony_pxw_z190.json'
DEFAULT_ASSOCIATIONS=ROOT/'configs/camera_hardware/okada_sp_installations_z190_v1.json'
DEFAULT_SHOTS=_runtime_data_module_dir(__file__) / 'source_shots.json'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):value.update(data)
    return value.hexdigest()


def load_metadata(registry_path=DEFAULT_REGISTRY,association_path=DEFAULT_ASSOCIATIONS,shots_path=DEFAULT_SHOTS):
    registry_path,association_path,shots_path=map(Path,(registry_path,association_path,shots_path))
    model=json.loads(registry_path.read_text(encoding='utf8'))
    associations=json.loads(association_path.read_text(encoding='utf8'))
    shots=json.loads(shots_path.read_text(encoding='utf8'))
    if model['schema_version']!='rinkspace.camera-hardware-model.v1' or associations['schema_version']!='rinkspace.programme-camera-hardware-associations.v1':
        raise ValueError('Unknown hardware metadata schema')
    if associations['hardware_registry']['sha256']!=digest(registry_path):raise ValueError('Hardware registry differs from its recorded association')
    if associations['source_shots_input']['sha256']!=digest(shots_path):raise ValueError('Native shot identity changed; hardware association must be reviewed')
    # Device provenance and camera classification are real inputs as well;
    # checking only the registry/shot file would leave stale authority links.
    classification=associations['classification_input']
    classification_path=Path(classification['path'])
    if digest(classification_path)!=classification['sha256']:
        raise ValueError('Camera classification authority changed; review hardware association')
    provenance=model['team_reply_provenance']
    evidence_path=Path(provenance['evidence_path'])
    if digest(evidence_path)!=provenance['evidence_sha256']:
        raise ValueError('User-reported hardware evidence changed; review its registry record')
    latest=associations.get('latest_user_camera_confirmation')
    latest_path=None
    latest_facts=None
    mount_photo_path=None
    if latest is not None:
        latest_path=Path(latest['path'])
        if digest(latest_path)!=latest['sha256']:
            raise ValueError('Latest user-confirmed installation facts changed')
        latest_facts=json.loads(latest_path.read_text(encoding='utf8'))
        if latest_facts['programme_source_sha256']!=shots['source_sha256_from_timeline']:
            raise ValueError('Latest camera confirmation belongs to another programme source')
        if set(latest_facts['camera_groups'])!={g['camera_group_id']for g in associations['installations']}:
            raise ValueError('Latest camera confirmation changes the three-installation scope')
        mount=latest_facts.get('rigid_mount_offset')
        if mount is not None:
            photo=mount['source_photo']
            mount_photo_path=Path(photo['path'])
            if digest(mount_photo_path)!=photo['sha256']:
                raise ValueError('User mechanical-mount reference image changed')
    positions=associations.get('latest_user_tripod_positions')
    position_path=position_image_path=None
    position_facts=None
    if positions is not None:
        position_path=Path(positions['path'])
        if digest(position_path)!=positions['sha256']:
            raise ValueError('Latest user tripod-position evidence changed')
        position_facts=json.loads(position_path.read_text(encoding='utf8'))
        if (position_facts['schema']!='rinkspace.user-tripod-position-evidence.v1'
                or position_facts['source_sha256']!=shots['source_sha256_from_timeline']
                or position_facts['scene_sha256']!=associations['geometry_binding']['active_scene_sha256']):
            raise ValueError('Tripod-position evidence belongs to another source or scene')
        image=position_facts['annotated_image']
        position_image_path=Path(image['path'])
        if digest(position_image_path)!=image['sha256']:
            raise ValueError('User annotated tripod-position image changed')
        for name,slot in [('long_side_main',0),('long_side_secondary',1),('low_end',2)]:
            if position_facts[name]['installation_id']!=slot:
                raise ValueError('User tripod annotation changes installation identity')
    if associations['source']['sha256']!=shots['source_sha256_from_timeline']:raise ValueError('Association is from another source')
    if associations['source']['final_video_resolution_wh']!=shots['source_resolution'] or associations['source']['final_video_rate']!=shots['source_rate']:
        raise ValueError('Association changed native video dimensions or display clock')
    groups=associations['installations'];ids=[g['installation_id']for g in groups]
    if len(ids)!=len(set(ids)) or len({g['camera_group_id']for g in groups})!=len(groups):raise ValueError('Physical installation groups have been merged')
    if any(g['hardware_model_id']!=model['model_id'] for g in groups):raise ValueError('Unknown hardware model association')
    for group in groups:
        for index in group['source_shot_indices']:
            if index not in shots['analysis_scope']['shot_indices'] or shots['shots'][index]['camera_id']!=group['camera_group_id']:
                raise ValueError('Hardware metadata reclassified a source cut')
    if model['sensor']['physical_sensor_width_mm']is not None or model['lens']['brown_distortion_coefficients']is not None:
        raise ValueError('This model-family registry does not carry measured sensor dimensions or published Brown calibration')
    input_paths=[registry_path,association_path,shots_path,classification_path,evidence_path]
    if latest_path is not None:input_paths.append(latest_path)
    if mount_photo_path is not None:input_paths.append(mount_photo_path)
    if position_path is not None:input_paths.extend([position_path,position_image_path])
    return dict(registry=model,associations=associations,source_shots=shots,
                user_camera_facts=latest_facts,
                user_tripod_position_facts=position_facts,
                input_manifest={str(p):digest(p)for p in input_paths},
                numeric_camera_values_modified=False)


def calibration_context(camera_group_id,*,source_sha256=None,metadata=None):
    """Manifest context for one installation; never an intrinsic initializer."""
    metadata=load_metadata()if metadata is None else metadata
    associations=metadata['associations']
    if source_sha256 is not None and source_sha256!=associations['source']['sha256']:
        raise ValueError('Hardware context cannot transfer to an unidentified other source/event')
    match=[g for g in associations['installations']if g['camera_group_id']==camera_group_id]
    if not match:
        return dict(camera_group_id=camera_group_id,hardware_model_id=None,installation_id=None,
                    association_status='unknown_unassociated_group',numeric_camera_values_modified=False)
    group=match[0]
    positions=metadata.get('user_tripod_position_facts')
    camera_facts=metadata.get('user_camera_facts') or {}
    position_name={0:'long_side_main',1:'long_side_secondary',2:'low_end'}[group['calibration_installation_slot']]
    return dict(camera_group_id=camera_group_id,installation_id=group['installation_id'],
                calibration_installation_slot=group['calibration_installation_slot'],
                hardware_model_id=group['hardware_model_id'],association_status=group['model_association_status'],
                manufacturer_capabilities=copy.deepcopy(metadata['registry']),
                field_settings=copy.deepcopy(group['field_settings']),
                latest_user_camera_confirmation=copy.deepcopy(associations.get('latest_user_camera_confirmation')),
                installation_height_policy=copy.deepcopy(camera_facts.get('installation_height_policy')),
                lens_and_pivot_height_estimates=copy.deepcopy(camera_facts.get('height_estimate')),
                working_height_prior=copy.deepcopy(camera_facts.get('working_height_prior')),
                rigid_mount_offset=copy.deepcopy(camera_facts.get('rigid_mount_offset')),
                latest_user_tripod_positions=copy.deepcopy(associations.get('latest_user_tripod_positions')),
                tripod_position_observation=None if positions is None else copy.deepcopy(positions[position_name]),
                final_source_video=copy.deepcopy(associations['source']),
                metadata_input_manifest=copy.deepcopy(metadata['input_manifest']),
                independent_installation_parameters_required=True,numeric_camera_values_modified=False,
                effective_focal_unit='source video pixels; physical-mm conversion unavailable')


def probe_source(metadata):
    """Read the actual final source and one exact native frame per group."""
    try:
        from .runtime.bootstrap import add_geometry_dependencies
    except ImportError:
        from runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()
    import av
    source=Path(metadata['source_shots']['source_path'])
    source_digest=digest(source)
    expected=metadata['associations']['source']['sha256']
    if source_digest!=expected:raise ValueError('Native video SHA differs from the classified source')
    observations=[]
    with av.open(str(source))as container:
        stream=container.streams.video[0]
        properties=dict(width=stream.codec_context.width,height=stream.codec_context.height,
                        average_rate=str(stream.average_rate),time_base=str(stream.time_base),
                        encoded_codec=stream.codec_context.name,start_pts=stream.start_time,
                        final_video_not_camera_capture_format=True)
        for group in metadata['associations']['installations']:
            index=group['source_shot_indices'][0]
            target=int(metadata['source_shots']['shots'][index]['start_pts'])
            container.seek(max(0,target-15360),stream=stream,backward=True)
            frame=next((f for f in container.decode(stream)if f.pts is not None and f.pts>=target),None)
            if frame is None or frame.pts!=target:raise ValueError('Source probe failed to decode exact native frame')
            bgr=frame.to_ndarray(format='bgr24')
            observations.append(dict(camera_group_id=group['camera_group_id'],installation_id=group['installation_id'],
                shot_index=index,requested_pts=target,decoded_pts=frame.pts,resolution_wh=[frame.width,frame.height],
                native_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                hardware_model_id=group['hardware_model_id'],model_association_status=group['model_association_status']))
    final=metadata['associations']['source']
    if [properties['width'],properties['height']]!=final['final_video_resolution_wh'] or Fraction(properties['average_rate'])!=Fraction(final['final_video_rate']) or Fraction(properties['time_base'])!=Fraction(final['final_video_time_base']):
        raise ValueError('Encoded source dimensions or PTS clock differ from the association')
    return dict(source_path=str(source),source_sha256=source_digest,actual_final_video_properties=properties,
                source_frame_associations=observations,source_capture_frame_rate=None,
                camera_capture_or_output_settings_inferred=False,geometry_and_optics_modified=False)


def run(a):
    metadata=load_metadata(a.registry,a.associations,a.shots)
    report=dict(metadata_inputs=metadata['input_manifest'],numeric_camera_values_modified=False,
                installation_contexts=[calibration_context(g['camera_group_id'],metadata=metadata)
                                       for g in metadata['associations']['installations']],
                unassociated_fourth_group=calibration_context('high_oblique_overview',metadata=metadata))
    if a.probe:report['actual_source_probe']=probe_source(metadata)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Use a new report path to retain the previous source probe')
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(report=str(a.output),installations=len(metadata['associations']['installations']),
                         actual_source_probed=a.probe,numeric_camera_values_modified=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY);p.add_argument('--associations',type=Path,default=DEFAULT_ASSOCIATIONS)
    p.add_argument('--shots',type=Path,default=DEFAULT_SHOTS);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',action='store_true');run(p.parse_args())
