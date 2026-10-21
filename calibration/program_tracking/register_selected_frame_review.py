"""Register a completed single-frame candidate without accepting it or fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
from .extract_opening_frame_batch import read, write, sha, MODEL, ROOT
from rinkspace.storage import AssetStore


def register(spec_path: Path):
    spec=read(spec_path);directory=spec_path.parent;index=spec['programme_frame_index_1based']
    packet_path=directory/f'frame_{index:06d}_points.json'
    binding_path=directory/f'frame_{index:06d}_model_binding.json'
    packet=read(packet_path);binding=read(binding_path)
    if packet['user_accepted'] or binding['source_user_accepted']:
        raise ValueError('This registration is only for a pending candidate')
    if binding['source_packet']['sha256']!=sha(packet_path):raise ValueError('Binding is stale')
    if packet['point_count']!=binding['point_count']:raise ValueError('Binding count differs')
    media=Path(spec['media_directory'])
    def ref(path):return dict(path=str(path.resolve()),sha256=sha(path))
    manifest=dict(schema='rinkspace.single-frame-review-delivery.v1',
        programme_frame_index_1based=index,source_pts=packet['source']['native_pts'],
        source_time_base=packet['source']['time_base'],source_time=packet['source']['source_time'],
        frame_count=1,point_count=packet['point_count'],user_accepted=False,
        status='waiting_for_user_review',source_packet=ref(packet_path),
        model_binding=ref(binding_path),camera_seed=ref(directory/'camera_seed.json'),
        navigation_mapping=ref(directory/'navigation_mapping.json'),
        source_image=ref(media/'source.png'),points_image=ref(media/'points_review.png'),
        navigation_image=ref(media/'navigation.png'),review_html=ref(AssetStore(ROOT).review_page(directory)),
        source_specification=ref(spec_path),ice_source_packet=ref(directory/'ice_features.json'),
        model_id=binding['model_id'],optimization_performed=False,
        programme_frames_101_to_199_processed=False,
        accepted_previous_batch_coordinates_modified=False)
    if packet.get('user_correction'):
        manifest['user_correction']=ref(Path(packet['user_correction']['path']))
    manifest['inferred_display_segment_count']=len(packet.get('inferred_segments',[]))
    manifest['display_inference_included_in_point_count']=False
    manifest['diagnostics']={}
    for name in ('accepted100_integrity_validation','candidate_model_entry_validation'):
        diagnostic=media/(name+'.json')
        if diagnostic.is_file():manifest['diagnostics'][name]=ref(diagnostic)
    manifest_path=directory/'review_manifest.json';write(manifest_path,manifest)
    model=read(MODEL/'model.json')
    current=ref(manifest_path)
    candidates=[r for r in model.get('candidate_frame_datasets',[]) if r['path']!=current['path']]
    candidates.append(dict(current,frame_index=index,frame_count=1,point_count=packet['point_count']))
    model.update(candidate_frame_datasets=candidates,
        candidate_source_frame_count=sum(r['frame_count'] for r in candidates),
        candidate_source_point_count=sum(r['point_count'] for r in candidates),
        candidate_user_acceptance_pending=True)
    write(MODEL/'model.json',model)
    programme=directory.parents[2]
    selection_path=programme/'primary_source_selection.json'
    selection=read(selection_path)
    continuation=selection['continuation']
    continuation.update(selected_preview_frame_processed=index,
        selected_preview_manifest=str(manifest_path.resolve()),
        selected_preview_source_pts=packet['source']['native_pts'],
        selected_preview_user_accepted=False,
        source_frame_count_processed=continuation['user_accepted_frame_count']+len(candidates),
        last_contiguous_processed_programme_frame=100,
        last_contiguous_processed_pts=281600,
        next_step='wait_for_user_review_of_frame_200_before_processing_101_to_199')
    write(selection_path,selection)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification',type=Path)
    args=parser.parse_args()
    result=register(args.specification)
    print(f"Registered frame {result['programme_frame_index_1based']}: {result['point_count']} points, pending review")
