"""Render the selected source shots with exact PTS and frozen native geometry."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import islice
import os
import sys
from time import monotonic

import av
import cv2
import numpy as np

from common import (HEIGHT, JOINT, OUTPUT, STEP, TB, WIDTH, frozen_scene, observation_packet,
                    read, scope, sha, source_frames, source_pts, visibility_packet, write)
from ice_evidence import BLUE_AREAS, CONTOUR, MIDPOINT, assert_ice_evidence, ice_roles
from measure import BOTTOM, DT, FIRST, TOP, load_camera, motion_clock
from rinkspace.review.render import validate_video
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.vision_near_board import exterior_mask


EMPTY_UV = np.empty((0, 2))
ICE_FACING_TOP_DECK = "source_ice_facing_edge_of_top_deck"
PRINT_MATERIALS = {
    "white_letter", "signed_logo", "panel_edge", "source_purple_panel_edge", "japanese",
    "meiwa_blue", "meiwa_green", "anniversary", "source_reviewed_purple_boundary",
    "source_reviewed_dark_ring_outer_flank", "source_ad2_named_right_stroke",
    "red_badge_outer_print_stroke", "source_board_print_contour",
    "source_reviewed_ad_panel_blur_step", "source_ad21_signed_blue_tile_edge",
}
BOARD_MATERIALS = {
    TOP: ("source_blue_rail_top", "actual_blue_band_upper_edge", ICE_FACING_TOP_DECK),
    BOTTOM: ("source_yellow_rail_bottom", "actual_yellow_kickplate_lower_edge", "source_signed_kickplate_lower_edge"),
}


@dataclass(frozen=True)
class ExposureObservations:
    uv: np.ndarray
    frame_sha256: str
    ice: int
    board: int
    advertisement: int
    reserved: int
    top_deck: tuple[np.ndarray, np.ndarray] | None = None


@lru_cache(maxsize=1)
def graphics_start_pts():
    source = scope()
    cues = source.get("programme_timer_cues", ())
    for cue in cues:
        if cue["display"] == "0:00":
            return int(cue["short_pts"]) - 3 * STEP
    return round(28.60 * 12800)


def restore_source_graphics(raw,pts,mask):
    """Keep the programme clock and detected broadcast cards above the rink."""
    first = graphics_start_pts()
    if pts<first:
        return mask
    if mask is None:
        mask=np.zeros((HEIGHT,WIDTH),np.uint8)
    if pts>=first+STEP:
        timer=np.array([[1694,949],[1859,949],[1859,990],[1820,993],
                        [1777,995],[1732,994],[1694,991]],np.int32)
        cv2.fillConvexPoly(mask,timer,1)

    def gradient_card(top,bottom):
        cyan=raw[top:top+5,83:389].astype(np.int16)
        magenta=raw[bottom-7:bottom-2,83:389].astype(np.int16)
        upper=np.median(cyan[:,:,0]-cyan[:,:,2],axis=0)>17
        lower=np.median(magenta[:,:,2]-magenta[:,:,1],axis=0)>37
        return bool(np.count_nonzero(upper&lower)>51)

    if gradient_card(31,62):
        cv2.rectangle(mask,(34,30),(75,61),1,-1)
        cv2.rectangle(mask,(34,69),(75,100),1,-1)
        # The two names animate inside the same cards for the first seconds.
        right1=491 if pts>=first+14*STEP else 294
        right2=491 if pts>=first+round(.85*12800) else 346
        cv2.rectangle(mask,(80,30),(right1,62),1,-1)
        cv2.rectangle(mask,(80,69),(right2,100),1,-1)
        rank=raw[33:48,506:537].astype(np.int16)
        if np.median(rank[:,:,2]-rank[:,:,0])>42:
            cv2.rectangle(mask,(497,30),(550,65),1,-1)
    if gradient_card(108,140):
        cv2.rectangle(mask,(80,107),(392,216),1,-1)
        for left in (80,107,134,161,188,215,242):
            chip=raw[227:246,left+1:left+24].astype(np.int16)
            green=np.median(chip[:,:,1]-chip[:,:,2])>27
            white=np.median(chip.mean(2))>213 and np.median(chip.max(2)-chip.min(2))<22
            if green or white:
                cv2.rectangle(mask,(left,224),(left+25,250),1,-1)
    return mask


class Review:
    def __init__(self, record, silhouettes=None, *, hide_board_backfaces=False):
        self.record=record
        self.native=frozen_scene(record,verify=True)
        sys.path.insert(0,str(JOINT))
        import render_okada as draw
        self.draw=draw
        samples=sample_native_review_scene(read(self.native.scene_path),self.native.core,self.native.scene_path,
                       parameter_state=self.native.parameter_state,native=self.native)
        names={r["id"] for r in samples["cuts"]}
        if names!=set(self.native.paths) or len(names)!=922:
            raise ValueError("Full native review paths differ from the fitted scene")
        name=record["source"]["title"].split(" (",1)[0].upper()
        draw.TITLE=f"{name} | ANKARA SHORT PROGRAMME"
        self.world,self.paths,self.overview=draw.draw_config(samples)
        self.hide_board_backfaces = hide_board_backfaces
        if hide_board_backfaces:
            from scipy.spatial import cKDTree

            foot = next(cut for cut in samples["cuts"] if cut["id"] == BOTTOM)
            cap = next(cut for cut in samples["cuts"] if cut["id"] == TOP)
            self.board_foot = slice(foot["begin"], foot["end"])
            self.board_cap = slice(cap["begin"], cap["end"])
            self.board_course = self.world[self.board_foot, :2]
            np.testing.assert_allclose(self.world[self.board_cap, :2], self.board_course, atol=1.e-12)
            course = self.board_course[:-1]
            tangent = np.roll(course, -1, axis=0) - np.roll(course, 1, axis=0)
            inward = np.column_stack((tangent[:, 1], -tangent[:, 0]))
            inward *= np.where(np.sum(-course * inward, axis=1) < 0., -1., 1.)[:, None]
            self.board_inward = np.vstack((inward, inward[0]))
            self.print_vertices = np.concatenate([
                np.arange(cut["begin"], cut["end"]) for cut in samples["cuts"]
                if cut["layer"] == "advertisement"
            ])
            nearest = cKDTree(course).query(self.world[self.print_vertices, :2])[1]
            self.print_wall = course[nearest]
            self.print_normal = inward[nearest]
        self.overview[66:105]=22
        draw.text(self.overview,"ANKARA | SHARED JOINT 1113",(18,84),.47,(60,210,255))
        self.camera=dict(first_control_time=FIRST,control_dt=DT,max_readout=0.)
        self.inventory=dict(Counter(s["layer"] for s in samples["cuts"]))
        self.silhouettes=silhouettes
        self.packed=None
        self.mask_valid=None
        self.label_anchors=None
        self.path_starts=None
        if silhouettes is not None:
            self.label_anchors={}
            labels=defaultdict(list)
            for start,stop,_,_,label in self.paths:
                if label is not None:
                    labels[label].append(np.arange(start,stop))
            for label,segments in labels.items():
                indices=np.concatenate(segments)
                coords=self.world[indices]
                median=np.median(coords,axis=0)
                self.label_anchors[label]=int(indices[np.argmin(np.sum((coords-median)**2,axis=1))])
            starts=np.array([p[0] for p in self.paths],np.int64)
            stops=np.array([p[1] for p in self.paths],np.int64)
            if starts[0]!=0 or stops[-1]!=len(self.world) or np.any(stops[:-1]!=starts[1:]):
                raise ValueError("Native render path slices do not partition the vertices")
            self.path_starts=starts

    def start_shot(self, shot):
        if self.silhouettes is None:
            return
        with np.load(visibility_packet(self.silhouettes,shot["id"]),allow_pickle=False) as z:
            np.testing.assert_array_equal(z["pts"],source_pts(shot))
            if str(z["source_sha256"].item())!=self.record["source"]["sha256"]:
                raise ValueError("Foreground matte belongs to another original source")
            self.mask_valid=z["valid"]
            self.packed=z["packed"]
        expected=(len(self.mask_valid),HEIGHT,WIDTH//8)
        if self.packed.shape!=expected:
            raise ValueError("Foreground matte is outside the registered pixel domain")

    def matte(self,i):
        if self.packed is None or not self.mask_valid[i]:
            return None
        mask=np.unpackbits(self.packed[i],axis=1,count=WIDTH)
        # Keep subpixel antialiasing off the athlete at the physical silhouette.
        return cv2.dilate(mask,np.ones((3,3),np.uint8))

    def source(self, points):
        return source_frames(self.record["source"]["registered_path"],points)

    def board_visibility(self, uv, status, centre):
        behind_print = np.sum((centre[:2] - self.print_wall) * self.print_normal, axis=1) <= 0.
        near = np.sum((centre[:2] - self.board_course) * self.board_inward, axis=1) <= 0.
        cap, foot = uv[self.board_cap], uv[self.board_foot]
        valid = (near & (status[self.board_cap] == 0) & (status[self.board_foot] == 0)
                 & np.isfinite(cap).all(axis=1) & np.isfinite(foot).all(axis=1))
        spans = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
        polys = []
        for start, stop in spans:
            if stop - start < 2:
                continue
            upper, lower = cap[start:stop], foot[start:stop]
            bounds = np.vstack((upper, lower))
            if (bounds.max(axis=0) < 0.).any() or (bounds.min(axis=0) >= [WIDTH, HEIGHT]).any():
                continue
            # The opaque near wall covers the ice and farther facing prints.
            # Inset its matte from the visible top edge by one native pixel.
            down = lower - upper
            inset = upper + down / np.maximum(1., np.linalg.norm(down, axis=1, keepdims=True))
            polygon = np.vstack((inset, lower[::-1]))
            polys.append(np.rint(np.clip(polygon, -1.e5, 1.e5) * 16).astype(np.int32))
        visible = uv.copy()
        visible[self.print_vertices[behind_print]] = np.nan
        visible[self.board_foot][near] = np.nan
        if not polys:
            return visible, None
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        for polygon in polys:
            cv2.fillPoly(mask, [polygon], 1, shift=4)
        return visible, mask

    def image(self,raw,uv,status,points,pts,sid,i,length, *, source_observations=False,
              camera_centre=None, source_top_deck=None):
        board_matte = None
        if self.hide_board_backfaces:
            if camera_centre is None or not np.isfinite(camera_centre).all():
                raise ValueError("Board visibility needs a defined native camera centre")
            uv, board_matte = self.board_visibility(uv, status, camera_centre)
        matte=self.matte(i)
        if self.silhouettes is not None:
            matte=restore_source_graphics(raw,pts,matte)
        if board_matte is not None:
            matte = board_matte if matte is None else cv2.bitwise_or(matte, board_matte)
        if self.hide_board_backfaces and source_top_deck is not None:
            edge, normals = source_top_deck
            source_matte = exterior_mask(edge, normals, np.zeros_like(edge), (HEIGHT, WIDTH)).astype(np.uint8)
            matte = source_matte if matte is None else cv2.bitwise_or(matte, source_matte)
        canvas,visible,state=self.draw.image(raw,uv,status,self.paths,self.overview,points,pts,sid,i,length,None,
                                           subpixel_bits=4,occluded=matte,label_anchors=self.label_anchors,
                                           path_starts=self.path_starts)
        if source_observations and matte is not None:
            # The display matte restores original pixels over the projected grid.
            # Keep the independently qualified original UV visible above it.
            x,y=np.floor(points.T).astype(np.int64)
            hidden=matte[y,x]!=0
            left=canvas[-HEIGHT:,:WIDTH]
            for xy in np.rint(points[hidden]*16).astype(np.int32):
                cv2.circle(left,tuple(xy),32,(0,0,255),-1,cv2.LINE_AA,shift=4)
        return canvas,visible,state

    def project(self, rig, controls, times):
        camera=self.camera | dict(max_readout=2.*(.64*abs(float(rig[37]))+1.e-7))
        return self.draw.project_batch(self.native,self.native.core,rig,controls,camera,times,self.world)

    def cpu_batch(self,rig,controls,times,pool):
        def one(t):
            return self.native.core.project_motion(rig,controls,FIRST,DT,HEIGHT,t,self.world)
        projected=list(pool.map(one,times))
        return np.stack([a["uv"] for a in projected]),np.stack([a["status"] for a in projected])

    def independent_projection_check(self, rig, control, t, uv, status, *, compare_cuda=False):
        # Verify the display uses the same row-exposure native projection as
        # the solver. Some off-screen rows have a stricter GPU solve guard.
        j=np.arange(0,len(self.world),max(1,len(self.world)//480))
        q=self.native.core.project_motion(rig,control,FIRST,DT,HEIGHT,t,self.world[j])
        valid=(q["status"]==0)&(q["uv"][:,0]>=0)&(q["uv"][:,0]<WIDTH)&(q["uv"][:,1]>=0)&(q["uv"][:,1]<HEIGHT)
        if not valid.any():
            return 0.
        if (status[j[valid]]!=0).any():
            raise ValueError("Visible projection lost by rendering")
        if compare_cuda:
            points=self.world[j[valid]]
            camera=self.camera|dict(max_readout=2.*(.64*abs(float(rig[37]))+1.e-7))
            dev,ds=self.draw.project_batch(self.native,self.native.core,rig,control,camera,[t],points)
            if np.any(ds):
                raise ValueError("Rendered visible projections differ from solver projection support")
            check=dev[0]
        else:
            check=q["uv"][valid]
        error=float(np.linalg.norm(check-uv[j[valid]],axis=1).max())
        if error>1.e-6:
            raise ValueError("Overlay projection differs from native CPU")
        return error


def check_observation_binding(arrays, record, context):
    for key, expected in (("source_sha256",record["source"]["sha256"]),
                          ("geometry_sha256",record["geometry_binding"]["value_sha256"])):
        if arrays[key].item()!=expected:
            raise ValueError(f"{context} differs from the registered {key}")


def check_observation_clock(pts, indices, shot):
    expected=source_pts(shot)
    if (pts.dtype.kind not in "iu" or indices.dtype.kind not in "iu" or pts.shape!=indices.shape
            or np.any((indices<0)|(indices>=len(expected))) or not np.array_equal(pts,expected[indices])):
        raise ValueError(f"Invalid original observation PTS for {shot['id']}")


def active_by_pts(candidate, record=None):
    points=defaultdict(list)
    if candidate is not None:
        with np.load(candidate/"active_rows.npz",allow_pickle=False) as z:
            pts,uv=z["pts"],z["uv"]
            if record is not None:
                check_observation_binding(z,record,"Solver training rows")
                shot_ids,indices=z["shot"],z["idx"]
                if not np.isin(shot_ids,[s["id"] for s in record["shots"]]).all():
                    raise ValueError("Solver training rows contain an unregistered shot")
                for shot in record["shots"]:
                    local=shot_ids==shot["id"]
                    check_observation_clock(pts[local],indices[local],shot)
            for p,xy in zip(pts,uv,strict=True):
                points[int(p)].append(xy)
    return {p:np.asarray(v) for p,v in points.items()}


def observations_by_pts(folder, record, native):
    """Read current original-pixel evidence across all declared folds."""
    points={}
    roles=ice_roles(native)
    for shot in record["shots"]:
        sid=shot["id"]
        with np.load(observation_packet(folder,sid),allow_pickle=False) as packet:
            check_observation_binding(packet,record,sid)
            keep=packet["eligible"]
            clock,indices=packet["pts"],packet["idx"]
            check_observation_clock(clock,indices,shot)
            if keep.dtype.kind!="b" or keep.shape!=clock.shape:
                raise ValueError(f"Malformed observation eligibility for {sid}")
            if (packet["excluded_reason"][keep]!="").any():
                raise ValueError(f"Excluded observation marked eligible for {sid}")
            # Validate the selected source rows once. The ice validator reuses
            # these arrays when it checks role, flank geometry and contrast.
            data={key:packet[key][keep] for key in ("path","pts","uv","source_bracket",
                  "source_side_contrast","ice_measurement_role")}
            pts,uv,path=data["pts"],data["uv"],data["path"]
            material,family,fold=[packet[key][keep] for key in ("material","family","fold")]
            if (uv.shape!=(len(pts),2) or not ((uv>=0.)&(uv<[WIDTH,HEIGHT])).all()
                    or fold.dtype.kind not in "iu" or not np.isin(fold,(0,1,2,3)).all()):
                raise ValueError(f"Invalid source pixel or observation fold for {sid}")
            keys,code=np.unique(path,return_inverse=True)
            if any(key not in native.paths for key in keys):
                raise ValueError(f"Observation path outside the frozen scene for {sid}")
            layers=np.array([native.paths[key]["layer"] for key in keys],dtype=str)[code]
            labels=np.array([native.paths[key]["label"] for key in keys],dtype=str)[code]
            role=np.array([roles.get(key,"") for key in keys],dtype=str)[code]
            assert_ice_evidence(data,np.ones(len(pts),bool),roles)
            verified_width = (material == "source_verified_width_centre") | (
                (labels == "ice.C:segment0") & (material == "source_reviewed_red_outer_flanks")
            )
            ring_contour = np.zeros(len(labels), dtype=bool)
            for boundary in ("center", "inner", "middle", "outer"):
                ring_contour |= (
                    np.char.endswith(labels, ":" + boundary)
                    & (material == "paired_annulus_" + boundary)
                )
            outer_edge = material == "source_dark_ice_outer_edge"
            if outer_edge.any():
                bracket = data["source_bracket"][outer_edge]
                contrast = data["source_side_contrast"][outer_edge]
                inside = ((bracket >= 0.) & (bracket < [WIDTH, HEIGHT])).all(axis=(1, 2))
                midpoint = np.linalg.norm(bracket.mean(axis=1) - uv[outer_edge], axis=1)
                support = np.linalg.norm(bracket[:, 1] - bracket[:, 0], axis=1)
                witness = inside & (midpoint < 1.e-6) & (support > 0.)
                witness &= np.isfinite(contrast).all(axis=1) & (contrast > 0.).all(axis=1)
                if not witness.all():
                    raise ValueError(f"Invalid observed outer pigment edge for {sid}")
                ring_contour |= outer_edge & np.char.endswith(labels, ":outer")
            match=((role==MIDPOINT)&verified_width
                   | (role==CONTOUR)&(ring_contour
                                    |(np.isin(labels,list(BLUE_AREAS))
                                      &(material=="source_blue_crease_curved_boundary"))))
            for board_path,materials in BOARD_MATERIALS.items():
                match|=(path==board_path)&np.isin(material,materials)
            ad=layers=="advertisement"
            instances=np.array([key.split("::",1)[0].rsplit("/",1)[-1] for key in keys],dtype=str)[code]
            match|=ad&(family==instances)&np.isin(material,list(PRINT_MATERIALS))
            if not match.all():
                i=np.flatnonzero(~match)[0]
                raise ValueError(f"No qualified direct material for {sid} at PTS {pts[i]}: {path[i]} {material[i]}")
            deck = material == ICE_FACING_TOP_DECK
            deck_normals = packet["normal"][keep][deck] if deck.any() else EMPTY_UV
            if (deck_normals.shape != (np.count_nonzero(deck), 2)
                    or not np.isfinite(deck_normals).all()
                    or (np.linalg.norm(deck_normals, axis=1) < .8).any()):
                raise ValueError(f"Missing source ice-facing normals for {sid}")
            deck_row = np.full(len(pts), -1, dtype=np.int64)
            deck_row[deck] = np.arange(len(deck_normals))
            frame_pts=packet["source_frame_pts"]
            frame_hashes=packet["source_frame_BGR_sha256"]

        if (frame_pts.dtype.kind not in "iu" or frame_pts.ndim!=1 or frame_hashes.shape!=frame_pts.shape
                or frame_hashes.dtype.kind!="U"
                or np.any(frame_pts[1:]<=frame_pts[:-1]) or not np.isin(frame_pts,source_pts(shot)).all()):
            raise ValueError(f"Invalid original frame hash index for {sid}")
        digests=dict(zip(frame_pts,frame_hashes,strict=True))
        order=np.argsort(pts,kind="stable")
        for current in np.split(order,np.flatnonzero(np.diff(pts[order]))+1):
            if not len(current):
                continue
            p=int(pts[current[0]])
            digest=digests.get(p,"")
            if len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"Missing decoded source identity for {sid} at PTS {p}")
            layer=layers[current]
            deck_at_pts = current[deck[current]]
            top_deck = (uv[deck_at_pts], deck_normals[deck_row[deck_at_pts]]) if len(deck_at_pts) >= 2 else None
            points[p]=ExposureObservations(uv[current],digest,int(np.count_nonzero(layer=="ice")),
                      int(np.count_nonzero(layer=="board")),int(np.count_nonzero(layer=="advertisement")),
                      int(np.count_nonzero(fold[current])),top_deck)
    return points


def source_uv(p,raw,active,observed):
    if observed is None:
        return active.get(p,EMPTY_UV)
    entry=observed.get(p)
    if entry is None:
        return EMPTY_UV
    if sha256(np.ascontiguousarray(raw)).hexdigest()!=entry.frame_sha256:
        raise ValueError(f"Decoded original pixels differ at observation PTS {p}")
    return entry.uv


def projected_batches(review, pool, source, rig, controls, times, cpu=None):
    """Overlap the next native projection with drawing and encoding this batch."""
    def submit(batch, offset):
        t=times[offset:offset+len(batch)]
        return pool.submit(review.project,rig,controls,t) if cpu is None else pool.submit(review.cpu_batch,rig,controls,t,cpu)

    i=0
    batch=list(islice(source,24))
    if not batch:
        return
    pending=submit(batch,i)
    while batch:
        uv,status=pending.result()
        following=list(islice(source,24))
        if following:
            pending=submit(following,i+len(batch))
        yield batch,uv,status
        i+=len(batch)
        batch=following


def preview(review, candidate, folder, *, times_from=None, observations=None):
    folder.mkdir(exist_ok=True)
    observed=None
    if observations is not None:
        source=review.record["source"]
        if sha(source["registered_path"])!=source["sha256"]:
            raise ValueError("Original video differs from the registered source")
        observed=observations_by_pts(observations,review.record,review.native)
    active=active_by_pts(candidate) if observed is None else {}
    qa=read(times_from/"comparison.json")["shots"] if times_from else {}
    rows=[]
    for shot in review.record["shots"]:
        sid=shot["id"]
        review.start_shot(shot)
        pts,c,rig=load_camera(shot,candidate)
        requested=[int(shot["representative_pts"])]
        worst=qa.get(sid,{}).get("worst",[])
        if worst:
            requested.append(int(worst[0]["pts"]))
        for p,frame in review.source(sorted(set(requested))):
            j=int(np.searchsorted(pts,p))
            t=motion_clock(shot,candidate,pts)[j]
            z=review.native.core.project_motion(rig,c,FIRST,DT,HEIGHT,t,review.world)
            raw=frame.to_ndarray(format="bgr24")
            centre = (review.native.core.evaluate_motion_camera(rig,c,FIRST,DT,[t])["centre"][0]
                      if review.hide_board_backfaces else None)
            entry = observed.get(p) if observed is not None else None
            im,n,_=review.image(raw,z["uv"],z["status"],source_uv(p,raw,active,observed),
                       p,sid,j,len(pts),source_observations=observed is not None,camera_centre=centre,
                       source_top_deck=entry.top_deck if entry is not None else None)
            if not n:
                raise ValueError(f"No model visible at {sid} original PTS {p}")
            cv2.imwrite(str(folder/f"{sid}_{p}.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,92])
            small=cv2.resize(im[:,:1920],(480,286),interpolation=cv2.INTER_AREA)
            rows.append(small)
    for start in range(0,len(rows),16):
        subset=rows[start:start+16]
        while len(subset)%4:
            subset.append(np.full_like(rows[0],22))
        page=np.vstack([np.hstack(subset[i:i+4]) for i in range(0,len(subset),4)])
        cv2.imwrite(str(folder/f"overview_{start//16+1}.jpg"),page,[cv2.IMWRITE_JPEG_QUALITY,92])


def render(review, candidate, folder, *, start_pts=None, stop_before_pts=None, parallel_cpu=False, observations=None):
    started=monotonic()
    expected=np.concatenate([source_pts(s) for s in review.record["shots"]])
    for flag,value in (("--start-pts",start_pts),("--stop-before-pts",stop_before_pts)):
        if value is not None and (not isinstance(value,(int,np.integer)) or value not in expected):
            raise ValueError(f"{flag} must be an integer PTS of an original programme frame")
    window_start=review.record["processing_scope"]["first_pts"] if start_pts is None else int(start_pts)
    if stop_before_pts is not None and stop_before_pts<=window_start:
        raise ValueError("--stop-before-pts must follow the first frame of the render window")
    if start_pts is not None:
        expected=expected[expected>=window_start]
    if stop_before_pts is not None:
        expected=expected[expected<stop_before_pts]

    folder.mkdir(exist_ok=True)
    source=review.record["source"]
    if sha(source["registered_path"])!=source["sha256"]:
        raise ValueError("Original video differs from the registered source")
    active=active_by_pts(candidate,review.record if observations is not None else None)
    observed=observations_by_pts(observations,review.record,review.native) if observations is not None else None
    name=review.record["programme_id"].removesuffix("_sp")
    raw_path=folder/f"{name}_grid.writing.mp4"
    final=folder/f"{name}_ankara_grid_1080p50.mp4"
    if final.exists():
        raise ValueError("Review video already exists")
    path=folder/"coverage.csv"
    counts={}
    all_pts=[]
    with av.open(str(raw_path),"w",options={"movflags":"+faststart","video_track_timescale":"12800"}) as mux, \
            path.open("x",encoding="utf-8",newline="") as ledger, \
            ThreadPoolExecutor(max_workers=1) as pool, \
            ThreadPoolExecutor(max_workers=min(8,max(1,(os.cpu_count() or 4)-4))) as cpu:
        from csv import writer
        csv=writer(ledger)
        fields=("shot","source_pts","visible_vertices","active_original_uv_count","camera_status",
                "athlete_matte_available")
        if observed is not None:
            fields+=("shown_original_uv_count","shown_ice_uv_count","shown_board_uv_count",
                     "shown_advertisement_uv_count","shown_reserved_uv_count")
        csv.writerow(fields)
        stream=mux.add_stream("libx264",rate=50)
        stream.width=2560
        stream.height=1144
        stream.pix_fmt="yuv420p"
        stream.time_base=stream.codec_context.time_base=TB
        stream.codec_context.thread_count=7
        stream.options={"crf":"19","preset":"veryfast","threads":"7"}
        for shot in review.record["shots"]:
            sid=shot["id"]
            start=monotonic()
            shot_pts=source_pts(shot)
            first_index=0 if start_pts is None else int(np.searchsorted(shot_pts,window_start))
            stop_index=len(shot_pts) if stop_before_pts is None else int(np.searchsorted(shot_pts,stop_before_pts))
            if not stop_index:
                break
            if first_index==stop_index:
                continue
            pts,c,rig=load_camera(shot,candidate)
            np.testing.assert_array_equal(pts,shot_pts)
            shot_length=len(pts)
            window=slice(first_index,stop_index)
            t=motion_clock(shot,candidate,pts)[window]
            pts=pts[window]
            review.start_shot(shot)
            camera=review.native.core.evaluate_motion_camera(rig,c,FIRST,DT,t)
            if np.any(camera["status"]):
                raise ValueError("Native camera undefined on required frame: "+sid)
            sample=review.source(pts)
            i=0
            visible, training_rows, frames_with_training, maximum = 0,0,0,0.
            shown_rows, frames_with_shown, shown_ice, shown_board, shown_ads, shown_reserved = 0,0,0,0,0,0
            for batch,uv,status in projected_batches(review,pool,sample,rig,c,t,cpu if parallel_cpu else None):
                for j,(p,raw) in enumerate(batch):
                    position=i+j
                    k=first_index+position
                    pixels=raw.to_ndarray(format="bgr24")
                    dot=source_uv(p,pixels,active,observed)
                    entry=observed.get(p) if observed is not None else None
                    canvas,n,_=review.image(pixels,uv[j],status[j],dot,p,sid,k,shot_length,
                                           source_observations=observed is not None,
                                           camera_centre=camera["centre"][position],
                                           source_top_deck=entry.top_deck if entry is not None else None)
                    if not n:
                        raise ValueError(f"No visible native grid for {sid} PTS {p}")
                    if position in (0,len(pts)//2,len(pts)-1):
                        maximum=max(maximum,review.independent_projection_check(rig,c,t[position],uv[j],status[j],
                                    compare_cuda=parallel_cpu))
                    movie=av.VideoFrame.from_ndarray(canvas,format="bgr24")
                    movie.pts=int(p)-window_start;movie.time_base=TB
                    for packet in stream.encode(movie):
                        mux.mux(packet)
                    matte_valid=bool(review.mask_valid[k]) if review.mask_valid is not None else None
                    train_count=len(active.get(p,EMPTY_UV))
                    row=(sid,p,n,train_count,int(camera["status"][position]),matte_valid)
                    if observed is not None:
                        ice,board,ads,reserved=(entry.ice,entry.board,entry.advertisement,entry.reserved
                                                ) if entry is not None else (0,0,0,0)
                        row+=(len(dot),ice,board,ads,reserved)
                        shown_rows+=len(dot)
                        frames_with_shown+=bool(len(dot))
                        shown_ice+=ice
                        shown_board+=board
                        shown_ads+=ads
                        shown_reserved+=reserved
                    csv.writerow(row)
                    training_rows+=train_count
                    frames_with_training+=bool(train_count)
                    visible+=bool(n)
                    all_pts.append(p)
                i+=len(batch)
                if i%240==0:
                    print(sid,"rendered",i,"of",len(pts),f"{monotonic()-start:.1f}s",flush=True)
            counts[sid]=dict(frames=i,visible_grid_frames=visible,active_original_uv=training_rows,
                        frames_with_uv=frames_with_training,cpu_gpu_max_px=maximum,
                        native_foreground_mattes=int(review.mask_valid[window].sum()) if review.mask_valid is not None else 0)
            if observed is not None:
                counts[sid].update(shown_original_uv=shown_rows,frames_with_shown_uv=frames_with_shown,
                                  shown_ice_uv=shown_ice,shown_board_uv=shown_board,
                                  shown_advertisement_uv=shown_ads,shown_reserved_uv=shown_reserved)
            print(sid,"render complete",counts[sid],f"{monotonic()-start:.1f}s",flush=True)
        for packet in stream.encode():
            mux.mux(packet)
    np.testing.assert_array_equal(all_pts,expected)
    validation=validate_video(raw_path,expected,TB)
    if not validation.get("native_PTS_equal",False):
        raise ValueError("Review video lost the source PTS")
    raw_path.replace(final)
    report=dict(path=str(final),sha256=sha(final),bytes=final.stat().st_size,source_clock="1/12800",
           native_frame_count=len(all_pts),first_pts=int(all_pts[0]),last_pts=int(all_pts[-1]),candidate=str(candidate),
           stop_before_pts=stop_before_pts,
           projection_backend="native_cpu_parallel" if parallel_cpu else "native_cuda",
           drawing_subpixel_bits=4,
           hides_backfacing_board_prints=review.hide_board_backfaces,
           occludes_geometry_behind_near_board=review.hide_board_backfaces,
           uses_measured_top_deck_occlusion=review.hide_board_backfaces and observed is not None,
           source_bound_foreground=str(review.silhouettes) if review.silhouettes is not None else None,
           inventory=review.inventory,encoded_clock_validation=validation,coverage=str(path),shots=counts,
           source_sha256=review.record["source"]["sha256"],
           geometry_value_sha256=review.record["geometry_binding"]["value_sha256"],seconds=monotonic()-started)
    if observations is not None:
        report["source_observations"]=str(observations)
    write(folder/"review.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate")
    p.add_argument("--out",required=True)
    p.add_argument("--preview",action="store_true")
    p.add_argument("--worst")
    p.add_argument("--start-pts",type=int,help="First original frame PTS to render, inclusive")
    p.add_argument("--stop-before-pts",type=int,help="Original frame PTS to stop before")
    p.add_argument("--parallel-cpu",action="store_true")
    p.add_argument("--silhouettes")
    p.add_argument("--observations",help="Current original-pixel observation packets, including reserved folds")
    p.add_argument("--hide-board-backfaces", action="store_true",
                   help="Respect the opaque near wall and the facing side of the board prints")
    a=p.parse_args()
    if a.observations and not (a.candidate or a.preview):
        p.error("--observations requires --candidate or --preview")
    cv2.setNumThreads(2)
    obj=Review(scope(),OUTPUT/a.silhouettes if a.silhouettes else None,
               hide_board_backfaces=a.hide_board_backfaces)
    path=OUTPUT/a.out
    chosen=OUTPUT/a.candidate if a.candidate else None
    observations=OUTPUT/a.observations if a.observations else None
    if a.preview:
        preview(obj,chosen,path,times_from=OUTPUT/a.worst if a.worst else None,observations=observations)
    elif chosen:
        render(obj,chosen,path,start_pts=a.start_pts,stop_before_pts=a.stop_before_pts,parallel_cpu=a.parallel_cpu,
               observations=observations)


if __name__=="__main__":
    main()
