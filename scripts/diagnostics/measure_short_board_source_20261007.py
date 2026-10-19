"""Source-only printed circles/shield corners across the reviewed short end."""
from pathlib import Path
from copy import deepcopy
import json,hashlib
import cv2,numpy as np
from short_board_global_source_20261007 import OUT,STAMPS,DATA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

# Native source visual ROIs only. These are broad read windows, no projection.
ELLIPSES={230043904:(2.,415.,53.,48.),230092800:(722.,307.,53.,54.),230146560:(797.,306.,54.,55.),230200320:(691.5,344.6,57.,58.3),230254336:(1800.,326.,55.,58.)}
SHIELD={230092800:(310,250,390,360),230146560:(380,250,462,370),230200320:(248,280,335,420),230254336:(1336,260,1440,415)}

def rim(gray,ellipse,angle):
    cx,cy,rx,ry=ellipse;t=np.deg2rad(angle)
    guess=np.array([cx+rx*np.cos(t),cy+ry*np.sin(t)])
    n=np.array([np.cos(t)/rx,np.sin(t)/ry]);n/=np.linalg.norm(n);tangent=np.array([-n[1],n[0]])
    d=np.arange(-11,11.001,.25);a=np.arange(-3,3.01)
    xy=guess[None,None]+d[:,None,None]*n+a[None,:,None]*tangent
    signal=cv2.remap(gray.astype(np.float32),xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    signal=np.median(signal,axis=1);smooth=cv2.GaussianBlur(signal[:,None],(1,9),1.8).reshape(-1)
    background=np.linspace(np.median(smooth[:8]),np.median(smooth[-8:]),len(d));mass=np.maximum(background-smooth,0.)
    j=int(np.argmax(mass));peak=float(mass[j]);lo=hi=j
    while lo>0 and mass[lo-1]>.25*peak:lo-=1
    while hi<len(d)-1 and mass[hi+1]>.25*peak:hi+=1
    weights=mass[lo:hi+1];support=d[lo:hi+1]
    axis=float(np.sum(support*weights)/max(1e-20,weights.sum()))
    uv=guess+axis*n
    return uv,n,dict(native_source_search_center=guess.tolist(),normal=n.tolist(),offsets_px=d.tolist(),
        native_gray_normal_profile=signal.tolist(),source_dark_stroke_mass=mass.tolist(),
        selected_contiguous_mass_interval_px=[float(d[lo]),float(d[hi])],peak_gray_contrast=peak,
        axis_offset_px=axis,at_search_boundary=bool(lo==0 or hi==len(d)-1),
        valid=bool(peak>3. and lo>0 and hi<len(d)-1 and np.all(uv>2) and np.all(uv<[1917,1077])))

def main():
    frozen=read(DATA/"programmes/okada_mei_sp/results/shared_radius_05db_pelvis_priority_full_programme_20261005/FROZEN_INPUTS_UNACCEPTED.json")
    if sha(frozen["camera"]["path"])!=frozen["camera"]["sha256"]:raise ValueError("Original source exposure map changed")
    with np.load(frozen["camera"]["path"],allow_pickle=False) as camera:
        aliases={int(p):int(a) for p,a in zip(camera["pts"],camera["exposure_representative_pts"])}
    images={p:cv2.imread(str(OUT/f"native_{p}.png")) for p in STAMPS}
    circles=[];shields=[];panels=[]
    for p,bgr in images.items():
        grey=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);draw=bgr.copy();path=OUT/f"native_{p}.png"
        # First frame circle clipped at left; no guessed ellipse center factor.
        angles=(-65,-35,0,35,65) if p==STAMPS[0] else ((-150,-120,-90,-60,-30) if p==230092800 else (-150,-120,-90,-60,-30,0,30,60,90,120,150,180))
        uv=[];normal=[];support=[]
        for angle in angles:
            point,n,proof=rim(grey,ELLIPSES[p],angle)
            if not proof["valid"]:continue
            uv.append(point.tolist());normal.append(n.tolist());support.append(proof)
            cv2.circle(draw,tuple(np.rint(point).astype(int)),3,(35,80,230),-1,cv2.LINE_AA)
        circles.append(dict(id=f"short_board_Ad24_actual_outer_stroke_{p}",pts=p,actual_pts=p,feature_id="Ad24",
            curve=dict(kind="board_circle",advertisement_id=24,edge="outermost_visible_circular_print_stroke"),split="holdout" if p in (230092800,230200320) else "fit",
            uv=uv,normal=normal,sigma_px=[6.]*len(uv),source_search_support=support,source_raw_frame=str(path),source_file_sha256=sha(path),
            source_RGB_actual_measurement=True,model_projection_used_to_generate_or_select_uv=False,
            source_scope="actual outer black/gray printed circular stroke; ellipse ROI only for source read, not a center or size world factor",
            sigma_semantics="6px engineering weight for blur and reader scope, not calibrated covariance",
            source_optical_normal_support_authorized=False,endpoint_observation=False))
        if p==230254336:
            points=[];normals=[];proofs=[]
            for angle in (-150,-120,-90,-60,-30,0,30,60,90,120,150,180):
                point,n,proof=rim(grey,(763.,312.,46.,48.),angle)
                if proof["valid"]:
                    points.append(point.tolist());normals.append(n.tolist());proofs.append(proof)
                    cv2.circle(draw,tuple(np.rint(point).astype(int)),3,(40,180,50),-1,cv2.LINE_AA)
            circles.append(dict(id=f"short_board_Ad26_actual_outer_stroke_{p}",pts=p,actual_pts=p,feature_id="Ad26",
                curve=dict(kind="shared_board_pattern_circle_curve",advertisement_id=26,edge="outermost_visible_circular_print_stroke"),
                split="fit",uv=points,normal=normals,sigma_px=[6.]*len(points),source_search_support=proofs,
                source_raw_frame=str(path),source_file_sha256=sha(path),source_RGB_actual_measurement=True,
                model_projection_used_to_generate_or_select_uv=False,source_scope="actual ministry round outermost printed circular stroke; source-only ROI, no center factor",
                source_optical_normal_support_authorized=False,endpoint_observation=False))
        if p in SHIELD:
            x0,y0,x1,y1=SHIELD[p];patch=bgr[y0:y1,x0:x1].astype(float)
            blue=patch[:,:,0]-(patch[:,:,1]+patch[:,:,2])/2
            mask=((blue>13.)&(patch.mean(axis=2)<185)).astype(np.uint8)
            count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
            label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));ys,xs=np.where(labels==label)
            top=int(ys.min());bottom=int(ys.max());topx=xs[ys<=top+1];bottomx=xs[ys>=bottom-1]
            points=np.array([[x0+float(topx.min()),y0+top],[x0+float(topx.max()),y0+top],[x0+float(np.median(bottomx)),y0+bottom]],float)
            for point in points:cv2.drawMarker(draw,tuple(np.rint(point).astype(int)),(230,80,25),cv2.MARKER_CROSS,11,2,cv2.LINE_AA)
            shields.append(dict(id=f"short_board_Ad25_actual_named_shield_points_{p}",pts=p,actual_pts=p,feature_id="Ad25",
                curve=dict(kind="shared_board_pattern_material_points",advertisement_id=25),split="holdout" if p in (230092800,230200320) else "fit",
                material_roles=["upper_left_printed_outline_corner","upper_right_printed_outline_corner","bottom_printed_outline_tip"],
                point_correspondence=True,uv=points.tolist(),sigma_px=[4.]*3,source_raw_frame=str(path),source_file_sha256=sha(path),
                source_RGB_actual_measurement=True,model_projection_used_to_generate_or_select_uv=False,
                source_search_support=dict(native_ROI_xyxy=[x0,y0,x1,y1],blue_ink_score=">13 native units",largest_connected_blue_ink_pixels=int(len(xs)),
                    native_RGB_ROI=patch.astype(np.uint8).tolist(),geometry_prior_or_model_not_used=True),
                source_scope="existing three named shield outline material points; source colour component and native marker review required",
                sigma_semantics="4px existing engineering floor, not calibrated covariance"))
        head=np.full((35,1920,3),245,np.uint8);cv2.putText(head,f"PTS{p} | RED measured outer Ad24 stroke; BLUE measured named Ad25 shield points; source-only",(10,25),0,.6,(20,20,20),1)
        panels.append(np.vstack((head,draw[230:520])))
    image=OUT/"short_board_source_measured_markers.png";cv2.imwrite(str(image),np.vstack(panels))
    common=dict(source_video_sha256="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",time_base="1/15360",image_size_wh=[1920,1080],original_source_unchanged=True,source_installed_in_solver=False)
    for row in circles+shields:
        row["actual_pts"]=aliases[row["pts"]]
        row["source_exposure_clock_authority"]=dict(path=frozen["camera"]["path"],sha256=frozen["camera"]["sha256"],
            clock_fields_only_used=True,projection_or_geometry_values_not_read=True)
    packet=dict(schema="rinkspace.short-board-multitime-source-candidate.v1",**common,status="SOURCE_CANDIDATE_REQUIRES_NATIVE_MARKER_REVIEW",
        circle_rows=circles,shield_rows=shields,source_marker_review_image=str(image),
        pixel_ellipse_centres_not_world_point_observations=True,projected_model_camera_or_world_not_imported=True)
    file=OUT/"short_board_multitime_source_candidate.json";file.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(packet=str(file),image=str(image),circles=[dict(pts=r["pts"],count=len(r["uv"])) for r in circles],shields=[dict(pts=r["pts"],uv=r["uv"]) for r in shields])))

if __name__=="__main__":main()
