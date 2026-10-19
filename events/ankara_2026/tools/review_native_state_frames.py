"""Draw an immutable actual scene/camera on exact native source moments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
from calibration.program_tracking.render_result import SceneRenderer,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE,decode_selected
from calibration.program_tracking.board_grid_overlay import make_board_grid_renderer
from calibration.program_tracking.current_source_points import CurrentSourcePoints
from rinkspace.native_source_cache import AuthenticatedNativeSourceCache
from rinkspace.storage import AssetStore
import cv2
import numpy as np

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def reference(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

STORAGE=AssetStore(ROOT)

def authenticate_current_source_overlay(path,identity,aliases,selected):
    """Bind explicit active pixels to this immutable model and review clock."""
    if any(pts not in aliases for pts in selected):
        raise ValueError('Requested native presentation has no saved exposure alias')
    points=CurrentSourcePoints(path,identity['camera_path'],q_sha256=identity['q_sha256'])
    saved=identity.get('current_source_points')
    if saved is not None and (Path(saved['path']).resolve()!=points.path or
            saved['sha256']!=points.reference['sha256']):
        raise ValueError('Current source points differ from the actual saved-state binding')
    manifest=points.metadata['input_manifest']
    if (manifest['sha256']!=identity['input_manifest_sha256'] or
            sha(manifest['path'])!=identity['input_manifest_sha256']):
        raise ValueError('Current source points belong to different saved fitting inputs')
    if any(aliases[pts] not in points.lookup for pts in selected):
        raise ValueError('Requested actual exposure has no current active source input')
    binding=dict(points.reference,q_sha256=identity['q_sha256'],
        camera_sha256=identity['camera_sha256'],input_manifest=manifest,
        selected_exposures=[dict(native_pts=pts,actual_pts=aliases[pts],
            active_points=len(points.points(aliases[pts])))for pts in selected])
    return points,binding

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state-identity',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pts',required=True,type=int,nargs='+')
    p.add_argument('--unit',type=float,default=None)
    p.add_argument('--body',type=Path,help='Use the exact same-camera source-fit body and its own saved metric scale')
    p.add_argument('--current-source-points',type=Path,
        help='Draw only exact saved-q active source UV with the shared yellow markers; no body is required')
    p.add_argument('--source-cache-manifest',type=Path,action='append',default=[],
        help='Authenticate and reuse existing original PNGs; repeat to combine receipts. Missing exposures fail without decoding or writing a new source cache.')
    p.add_argument('--contact-sheets',action='store_true',help='Write contact pages instead of one model PNG per source presentation')
    p.add_argument('--detail-pts',type=int,nargs='*',default=[],help='With contact pages, also write only these requested full native model details')
    p.add_argument('--contact-sheet-columns',type=int,default=2)
    p.add_argument('--contact-sheet-rows',type=int,default=2)
    p.add_argument('--contact-sheet-scale',type=float,default=1.,help='Default 1 preserves every native model pixel; a smaller value is navigation only')
    p.add_argument('--check-only',action='store_true',help='Authenticate source/model/cache pixels and exit without rendering or writing output')
    p.add_argument('--reuse-existing',action='store_true',help='Reuse a complete same-request output after authenticating its receipts and image hashes')
    a=p.parse_args()
    a.state_identity=STORAGE.resolve(a.state_identity);a.source=STORAGE.source(a.source);a.output=STORAGE.resolve(a.output)
    if not a.output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data').resolve()):
        raise ValueError('Review pixels belong in the existing Data root')
    if (len(set(a.pts))!=len(a.pts) or not set(a.detail_pts).issubset(a.pts) or
            min(a.contact_sheet_columns,a.contact_sheet_rows)<1 or
            not np.isfinite(a.contact_sheet_scale) or not 0<a.contact_sheet_scale<=1):
        raise ValueError('Require distinct requested PTS, requested details and positive contact-page dimensions/scale')
    selected=sorted(a.pts)
    identity=json.loads(a.state_identity.read_text(encoding='utf8'))
    scene_path=a.state_identity.parent/'scene_UNACCEPTED.json'
    if sha(scene_path)!=identity['scene_sha256'] or sha(identity['camera_path'])!=identity['camera_sha256']:
        raise ValueError('Immutable actual state identity mismatch')
    state_path=a.state_identity.parent/'complete_state_UNACCEPTED.npz'
    if sha(state_path)!=identity['state_npz_sha256']:raise ValueError('Complete saved solver state changed')
    with np.load(state_path,allow_pickle=False)as z:
        if (hashlib.sha256(z['q'].tobytes()).hexdigest()!=identity['q_sha256'] or
                z['q'].shape!=(int(identity['parameter_count']),)):
            raise ValueError('Complete q/layout identity mismatch')
    cams=np.load(identity['camera_path'],allow_pickle=False)
    metadata=json.loads(str(cams['metadata_json'].item()))
    aliases=dict(zip(map(int,cams['pts']),map(int,cams['exposure_representative_pts'])))
    indices={int(pts):i for i,pts in enumerate(cams['pts'])}
    if not set(selected).issubset(indices):raise ValueError('Requested native presentation has no saved camera')
    current_points=point_binding=None
    if a.current_source_points:
        current_points,point_binding=authenticate_current_source_overlay(
            STORAGE.resolve(a.current_source_points),identity,aliases,selected)
    cache=None
    if a.source_cache_manifest:
        cache=AuthenticatedNativeSourceCache(a.source_cache_manifest,source_path=a.source,
            source_sha256=metadata['source_sha256'],source_time_base=metadata['source_time_base'],
            aliases=aliases,resolve_source=STORAGE.source)
        for pts in selected:cache.row_for_pts(pts)
    elif sha(a.source)!=metadata['source_sha256']:
        raise ValueError('Specified original video differs from the saved camera source SHA')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    selected_masks=[dict(native_pts=pts,path=str(masks.lookup[pts][0]),
        row_index=int(masks.lookup[pts][1]),status=masks.lookup[pts][2])
        for pts in selected if pts in masks.lookup]
    mask_binding=dict(directory=str(masks.mask_dir.resolve()),
        selected_chunk_files=[reference(path)for path in sorted({Path(row['path'])for row in selected_masks})],
        native_presentation_rows=selected_masks,missing_native_pts=[pts for pts in selected if pts not in masks.lookup])
    request=dict(schema='rinkspace.immutable-native-source-state-frame-review-request.v2',
        state_identity=reference(a.state_identity),complete_state=reference(state_path),
        camera=reference(identity['camera_path']),scene=reference(scene_path),
        source_path=str(a.source.resolve()),source_sha256=metadata['source_sha256'],
        source_time_base=metadata['source_time_base'],pts=selected,unit=a.unit,
        original_source_file_SHA_authenticated_this_run=cache is None,
        body=None if not a.body else reference(a.body),source_cache_manifests=[] if cache is None else cache.manifests,
        contact_sheets=a.contact_sheets,detail_pts=sorted(set(a.detail_pts)),
        contact_sheet_columns=a.contact_sheet_columns,contact_sheet_rows=a.contact_sheet_rows,
        contact_sheet_scale=a.contact_sheet_scale,smoothing=False,all_paths_and_ads_enabled=True,
        masks=mask_binding,people=reference(STORAGE.resolve(DEFAULT_PEOPLE)))
    # Keep the default request byte-for-byte compatible with old receipts.
    if point_binding is not None:request['current_source_points']=point_binding
    receipt_path=a.output/'review_identity.json'
    if a.reuse_existing and receipt_path.is_file():
        old=json.loads(receipt_path.read_text(encoding='utf8'))
        if old.get('request')!=request:raise ValueError('Existing render has different model, source or review scope')
        for row in old['frames']:
            if 'path' in row and sha(row['path'])!=row['sha256']:raise ValueError('Existing native model image changed')
        for page in old['contact_pages']:
            if sha(page['path'])!=page['sha256']:raise ValueError('Existing contact page changed')
        if sorted(row['source_pts']for row in old['frames'])!=selected:raise ValueError('Existing render omits requested presentations')
        if cache:
            for pts in selected:cache.read(pts)
            cache.assert_unchanged()
        print(json.dumps(dict(output=str(a.output),images=len(old['frames']),reused_complete_existing_render=True,
            new_source_images_written=0,new_model_images_written=0)),flush=True)
        return
    if a.check_only:
        if cache:
            for pts in selected:cache.read(pts)
            cache.assert_unchanged()
        print(json.dumps(dict(status='authenticated',native_presentations=len(selected),
            actual_exposures=len({aliases[pts]for pts in selected}),cache_used=cache is not None,
            exact_cached_PNG_and_BGR_checked=cache is not None,output_written=False)),flush=True)
        return
    if receipt_path.exists():raise FileExistsError('Preserve existing review; use --reuse-existing for the exact same request')
    scene=json.loads(scene_path.read_text(encoding='utf8'))
    view=None
    if a.body:
        # The solver installer also loads a historical bare render_result name.
        # This review must use the same current renderer as the complete video.
        from calibration.program_tracking import render_programme_root_trace as root_view_module
        root_view_module.SceneRenderer=SceneRenderer
        view=root_view_module.ProgrammeRootView(identity['camera_path'],a.body,scene_path,display_camera_smoothing=False)
        r=view.renderer
    else:
        Renderer=make_board_grid_renderer(SceneRenderer,scene_unit_m=a.unit)
        r=Renderer(scene,draw_grid=True,draw_advertisements=True,graphics_regions=((28,24,554,253),(1690,944,1865,1000)))
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[];pages=[];tiles=[]
    def save_image(path,image):
        if path.exists():raise FileExistsError('Preserve existing rendered evidence: '+str(path))
        if not cv2.imwrite(str(path),image):raise OSError(path)
        return reference(path)
    def flush_page():
        if not tiles:return
        width,height=tiles[0].shape[1],tiles[0].shape[0]
        page=np.full((height*a.contact_sheet_rows,width*a.contact_sheet_columns,3),20,np.uint8)
        for ordinal,tile in enumerate(tiles):
            y,x=(ordinal//a.contact_sheet_columns)*height,(ordinal%a.contact_sheet_columns)*width
            page[y:y+height,x:x+width]=tile
        path=a.output/f'all_model_contact_page_{len(pages):03d}.png'
        pages.append(dict(save_image(path,page),page_index=len(pages),
            source_pts=[row['source_pts']for row in rows[-len(tiles):]],
            native_model_pixels_not_resized=a.contact_sheet_scale==1.,native_rendered_hw=list(page.shape[:2])))
        tiles.clear()
    def frames():
        if cache:
            for pts in selected:
                raw,source_ref=cache.read(pts)
                yield pts,raw,dict(method='Authenticated existing original native PNG',**source_ref)
        else:
            for pts,raw in decode_selected(a.source,selected):
                yield pts,raw,dict(method='Exact original video frame.pts decoded in memory',native_pts=pts)
    for pts,raw,source_ref in frames():
        try:mask,mask_status=masks.get(pts,require_sam2=True)
        except ValueError:mask=None;mask_status='missing_or_invalid_measured_person_mask'
        if view is None:image,_=r.draw(raw,cams['camera25'][indices[pts]],mask,title=None)
        else:image=view.draw(raw,pts)
        if current_points is not None:current_points.draw(image,aliases[pts])
        row=dict(source_pts=pts,actual_exposure_pts=aliases[pts],native_shape=list(raw.shape),
            source_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),source_reference=source_ref,
            native_model_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),person_mask_status=mask_status)
        if not a.contact_sheets or pts in a.detail_pts:
            row.update(save_image(a.output/f'native_all_model_{pts}.png',image))
        if a.contact_sheets:
            tile=image if a.contact_sheet_scale==1. else cv2.resize(image,None,
                fx=a.contact_sheet_scale,fy=a.contact_sheet_scale,interpolation=cv2.INTER_AREA)
            panel=np.full((tile.shape[0]+36,tile.shape[1],3),245,np.uint8);panel[36:]=tile
            cv2.putText(panel,f'native PTS {pts} | actual {aliases[pts]} | q {identity["q_sha256"][:12]}',
                (8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(15,15,15),1,cv2.LINE_AA)
            row.update(contact_page_index=len(pages),contact_page_row=len(tiles)//a.contact_sheet_columns,
                contact_page_column=len(tiles)%a.contact_sheet_columns,
                contact_model_pixel_xywh=[(len(tiles)%a.contact_sheet_columns)*panel.shape[1],
                    (len(tiles)//a.contact_sheet_columns)*panel.shape[0]+36,tile.shape[1],tile.shape[0]])
            tiles.append(panel)
        rows.append(row)
        if a.contact_sheets and len(tiles)==a.contact_sheet_rows*a.contact_sheet_columns:flush_page()
    flush_page()
    if [row['source_pts']for row in rows]!=selected:raise ValueError('Source decoder omitted or reordered native presentations')
    if cache:cache.assert_unchanged()
    if point_binding is not None and (sha(point_binding['path'])!=point_binding['sha256'] or
            sha(point_binding['input_manifest']['path'])!=point_binding['input_manifest']['sha256']):
        raise ValueError('Current source point binding changed during review')
    body_receipt=None if view is None else dict(path=str(a.body),sha256=sha(a.body),metres_per_scene_unit=view.unit,saved_camera_body_exact_match=True)
    receipt_path.write_text(json.dumps(dict(schema='rinkspace.immutable-native-source-state-frame-review.v2',request=request,
        state_identity=str(a.state_identity),q_sha256=identity['q_sha256'],camera_sha256=identity['camera_sha256'],
        scene_sha256=identity['scene_sha256'],path_counts=r.path_counts,all_paths_and_ads_enabled=True,smoothing=False,
        board_grid=r.board_grid.report,body=body_receipt,frames=rows,contact_pages=pages,
        new_source_images_written=0,original_frames_decoded=0 if cache else len(rows),
        original_cache_frames_read=len(rows)if cache else 0,accuracy_accepted=False,
        isolated_native_source_diagnostics_not_complete_shot_accuracy_validation=True,
        right_side_plan_only_present_when_exact_body_view_requested=view is not None),indent=2),encoding='utf8')
    print(json.dumps(dict(images=len(rows),contact_pages=len(pages),output=str(a.output),new_source_images_written=0)))

if __name__=='__main__':main()
