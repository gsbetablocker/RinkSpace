"""Independent native midpoint print reads; preserve all existing held rows."""
from pathlib import Path
from copy import deepcopy
import json,hashlib,argparse
import av,cv2,numpy as np
from short_board_global_source_20261007 import OUT,DATA,SOURCE
from measure_short_board_source_20261007 import rim,ELLIPSES,SHIELD

TIMES={230094336:230092800,230201856:230200320}
DEST=OUT/"independent_midtime_source"


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def admit():
    source_path=DEST/"independent_midtime_source_candidate_v2.json";source=read(source_path)
    evidence=dict(kind="independent_native_RGB_print_source_read_and_visual_marker_review",source_candidate_path=str(source_path),
        source_candidate_sha256=sha(source_path),source_marker_review_image=source["source_marker_review_image"],
        no_retained_held_pixel_used_as_new_source=True,no_user_exact_pixel_confirmation_claimed=True)
    parent_circle=OUT/"circle_parent_plus_short_board_multitime_20261007.json"
    parent_material=OUT/"material_parent_plus_short_board_Ad25_Ad26_multitime_20261007.json"
    circle=read(parent_circle);material=read(parent_material)
    certificate=read(circle["geometry_contract"]["same_print_source_evidence"]["path"])
    for row in source["circle_rows"]:
        circle["rows"].append(dict(deepcopy(row),solver_admitted=True,source_review_authority=evidence))
        certificate["source_frames"].append(dict(advertisement_number=24,pts=row["pts"],source_raw_frame=row["source_raw_frame"],source_file_sha256=row["source_file_sha256"],additional_native_source_review=evidence))
    material["rows"].extend(dict(deepcopy(row),solver_admitted=True,source_review_authority=evidence) for row in source["shield_rows"])
    cert=DEST/"same_print_Ad11_24_with_independent_midtime_20261007.json";write(cert,certificate)
    circle["geometry_contract"]["same_print_source_evidence"]=dict(path=str(cert),sha256=sha(cert))
    outputs=[]
    for value,parent,name in ((circle,parent_circle,"circle_parent_plus_short_board_independent_midtime_FIT_20261007.json"),
            (material,parent_material,"material_parent_plus_short_board_independent_midtime_FIT_20261007.json")):
        original=read(parent);assert value["rows"][:len(original["rows"])]==original["rows"]
        value.update(status="explicit_new_fit_target_authorized",derived_from_native_source_parent=dict(path=str(parent),sha256=sha(parent)),
            independent_midtime_source=evidence,all_existing_FIT_and_holdout_rows_unchanged=True,source_installed_in_solver=False)
        output=DEST/name
        if output.exists():raise FileExistsError("Preserve immutable admitted packet")
        write(output,value);outputs.append(dict(path=str(output),sha256=sha(output),previous_rows=len(original["rows"]),new_FIT_rows=len(value["rows"])-len(original["rows"])))
    write(DEST/"midtime_source_admission_record.json",dict(outputs=outputs,all_original_holdout_rows_preserved=True,new_exposures=list(TIMES)))
    print(json.dumps(dict(outputs=outputs,all_original_holdout_rows_preserved=True)))


def main():
    DEST.mkdir(parents=True,exist_ok=True);frames={}
    with av.open(str(SOURCE)) as c:
        stream=c.streams.video[0];stream.thread_count=1
        for pts in TIMES:
            c.seek(pts-15360,stream=stream,backward=True)
            for f in c.decode(stream):
                if f.pts>pts:break
                if f.pts==pts:frames[pts]=f.to_ndarray(format="bgr24");break
    if set(frames)!=set(TIMES):raise ValueError("Exact additional native exposure missing")
    frozen=read(DATA/"programmes/okada_mei_sp/results/shared_radius_05db_pelvis_priority_full_programme_20261005/FROZEN_INPUTS_UNACCEPTED.json")
    if sha(frozen["camera"]["path"])!=frozen["camera"]["sha256"]:raise ValueError("Clock archive changed")
    with np.load(frozen["camera"]["path"],allow_pickle=False) as z:aliases=dict(zip(map(int,z["pts"]),map(int,z["exposure_representative_pts"])))
    circles=[];shields=[];panels=[]
    for pts,origin in TIMES.items():
        bgr=frames[pts];old=cv2.imread(str(OUT/f"native_{origin}.png"));file=DEST/f"native_{pts}.png"
        if file.exists():
            if not np.array_equal(cv2.imread(str(file)),bgr):raise ValueError("Existing native source PNG changed")
        else:cv2.imwrite(str(file),bgr)
        sift=cv2.SIFT_create(nfeatures=4000)
        mask=np.zeros((1080,1920),np.uint8);mask[250:450]=255
        ka,da=sift.detectAndCompute(cv2.cvtColor(old,cv2.COLOR_BGR2GRAY),mask)
        kb,db=sift.detectAndCompute(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),mask)
        good=[a for a,b in cv2.BFMatcher().knnMatch(da,db,k=2) if a.distance<.72*b.distance]
        first=np.float32([ka[m.queryIdx].pt for m in good]);second=np.float32([kb[m.trainIdx].pt for m in good])
        H,inlier=cv2.findHomography(first,second,cv2.RANSAC,2.5)
        if H is None or int(inlier.sum())<30:raise ValueError("Actual board navigation insufficient")
        def transfer(xy):return cv2.perspectiveTransform(np.asarray(xy,np.float32).reshape(-1,1,2),H).reshape(-1,2)
        cx,cy,rx,ry=ELLIPSES[origin];center=transfer([[cx,cy]])[0];ends=transfer([[cx+rx,cy],[cx,cy+ry]])
        ellipse=(float(center[0]),float(center[1]),float(np.linalg.norm(ends[0]-center)),float(np.linalg.norm(ends[1]-center)))
        grey=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);draw=bgr.copy();uv=[];normals=[];support=[]
        angles=(-150,-120,-90,-60,-30) if origin==230092800 else (-150,-120,-90,-60,-30,0,30,60,90,120,150,180)
        for angle in angles:
            point,n,proof=rim(grey,ellipse,angle)
            if proof["valid"]:
                uv.append(point.tolist());normals.append(n.tolist());support.append(proof);cv2.circle(draw,tuple(np.rint(point).astype(int)),3,(20,70,230),-1,cv2.LINE_AA)
        if len(uv)<3:raise ValueError("New actual circular rim has insufficient visible support")
        common=dict(pts=pts,actual_pts=aliases[pts],split="fit",source_raw_frame=str(file),source_file_sha256=sha(file),
            source_RGB_actual_measurement=True,model_projection_used_to_generate_or_select_uv=False,
            source_optical_normal_support_authorized=False,source_identity_same_event_print_catalog=True,
            independent_from_retained_held_exposure=True,held_reference_used_only_for_identity_and_read_window_not_pixels=True,
            source_read_window_navigation=dict(original_pts=origin,board_SIFT_inliers=int(inlier.sum()),source_only_H=H.tolist(),
                H_only_for_broad_source_read_windows=True,H_transferred_pixels_never_admitted_as_source=True))
        circles.append(dict(**common,id=f"short_board_Ad24_independent_midtime_outer_stroke_{pts}",feature_id="Ad24",
            curve=dict(kind="board_circle",advertisement_id=24,edge="outermost_visible_circular_print_stroke"),uv=uv,normal=normals,
            sigma_px=[6.]*len(uv),source_search_support=support,endpoint_observation=False,
            source_scope="Independently reread actual outermost printed circular rim at new native exposure; no center or size factor"))
        x0,y0,x1,y1=SHIELD[origin];corner=transfer([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        x0,y0=np.maximum(np.floor(corner.min(axis=0)-5).astype(int),[0,0]);x1,y1=np.minimum(np.ceil(corner.max(axis=0)+5).astype(int),[1920,1080])
        patch=bgr[y0:y1,x0:x1].astype(float);blue=patch[:,:,0]-(patch[:,:,1]+patch[:,:,2])/2
        mask=((blue>13.)&(patch.mean(axis=2)<185)).astype(np.uint8);count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));ys,xs=np.where(labels==label);top=int(ys.min());bottom=int(ys.max())
        # Perspective/antialiasing can place the right top corner two rows
        # below the left. A common min-y row clipped the actual right corner.
        # Read each upper contour corner from its own actual native boundary.
        upper=ys<=top+7;left=int(xs[upper].min());right=int(xs[upper].max())
        lefty=int(ys[(xs==left)&upper].min());righty=int(ys[(xs==right)&upper].min());bx=xs[ys>=bottom-1]
        points=np.array([[x0+left,y0+lefty],[x0+right,y0+righty],[x0+float(np.median(bx)),y0+bottom]])
        for point in points:cv2.drawMarker(draw,tuple(np.rint(point).astype(int)),(230,80,25),cv2.MARKER_CROSS,11,2,cv2.LINE_AA)
        shields.append(dict(**common,id=f"short_board_Ad25_independent_midtime_named_points_{pts}",feature_id="Ad25",
            curve=dict(kind="shared_board_pattern_material_points",advertisement_id=25),point_correspondence=True,
            material_roles=["upper_left_printed_outline_corner","upper_right_printed_outline_corner","bottom_printed_outline_tip"],
            uv=points.tolist(),sigma_px=[4.]*3,
            source_search_support=dict(native_ROI_xyxy=[int(x0),int(y0),int(x1),int(y1)],native_RGB_ROI=patch.astype(np.uint8).tolist(),
                native_blue_component_pixels=int(len(xs)),H_transferred_pixels_not_used_as_source=True)))
        header=np.full((35,1920,3),245,np.uint8);cv2.putText(header,f'New independent native exposure PTS{pts} actual{aliases[pts]} | RED Ad24 raw rim; BLUE Ad25 raw named points; FIT',(10,25),0,.6,(20,20,20),1)
        panels.append(np.vstack((header,draw[230:500])))
    image=DEST/"native_midtime_source_read_markers_v2.png";cv2.imwrite(str(image),np.vstack(panels))
    candidate=dict(schema="rinkspace.short-board-independent-midtime-native-source-candidate.v1",circle_rows=circles,shield_rows=shields,
        source_marker_review_image=str(image),status="requires_native_marker_review_before_admission",all_original_FIT_and_HELD_unchanged=True,
        source_model_camera_or_world_not_used_to_read_pixels=True)
    candidate["reader_correction_history"]=dict(previous_unadmitted_candidate_path=str(DEST/"independent_midtime_source_candidate.json"),
        previous_min_y_common_top_row_failed_to_read_actual_right_upper_corner=True,corrected_each_upper_outline_corner_at_its_actual_native_y=True)
    write(DEST/"independent_midtime_source_candidate_v2.json",candidate)
    print(json.dumps(dict(candidate=str(DEST/"independent_midtime_source_candidate_v2.json"),image=str(image),circles=[dict(pts=r["pts"],uv=r["uv"]) for r in circles],shields=[dict(pts=r["pts"],uv=r["uv"]) for r in shields])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--admit",action="store_true")
    args=p.parse_args();admit() if args.admit else main()
