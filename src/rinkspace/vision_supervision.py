"""Source supervision roles and whole-shot splits; never manufacture labels.

The inventory points back to immutable inputs. Human positions train directly;
accepted re-read RGB intervals train with their source uncertainty. Unreviewed
profiles and propagated search tracks remain candidates. Printed component edges,
letter holes and physical panel borders retain different geometric roles even
when an existing semantic head calls all of them ``ad_boundary``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import re
from numbers import Integral
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_time(pts, time_base):
    milliseconds = round(int(pts) * Fraction(time_base) * 1000)
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def source_sample_key(sample):
    """Bind a native frame to its source content and normalized native clock.

    ``source_hash`` is the existing sample protocol; ``source_sha256`` is its
    explicit source-binding alias. Neither a URL nor a bare PTS is identity.
    The supplied record remains unchanged, including any clock spelling.
    """
    digest = sample.get("source_hash", sample.get("source_sha256"))
    if not isinstance(digest, str) or not digest:
        raise ValueError("A source sample key requires the original source hash")
    if (sample.get("source_sha256") is not None and
            sample.get("source_hash") is not None and
            sample["source_sha256"] != sample["source_hash"]):
        raise ValueError("Source hash aliases disagree")
    try:
        clock = Fraction(sample["time_base"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("A source sample key requires a valid native clock") from error
    pts = sample.get("source_pts")
    if clock <= 0 or isinstance(pts, bool) or not isinstance(pts, Integral) or pts < 0:
        raise ValueError("A source sample key requires positive clock and nonnegative integer native PTS")
    return digest, str(clock), int(pts)


def source_updates_by_key(samples, updates):
    """Resolve reviewed updates without changing source records or evidence.

    New updates supply source_hash/source_sha256 and time_base. Legacy rows
    can omit both only when (native PTS, original frame SHA) identifies exactly
    one existing sample. Partial identities, missing/changed images, duplicate
    updates or ambiguous legacy rows fail instead of selecting another video.
    Returned values are the original update records, with no UV/role edits.
    """
    sample_by_key = {}
    for sample in samples:
        key = source_sample_key(sample)
        if key in sample_by_key:
            raise ValueError("Duplicate source sample key in update targets")
        sample_by_key[key] = sample
    resolved = {}
    for update in updates:
        has_source = any(update.get(field) is not None for field in ("source_hash", "source_sha256"))
        has_clock = update.get("time_base") is not None
        if has_source != has_clock:
            raise ValueError("A reviewed update needs both source hash and native clock")
        frame_sha = update.get("frame_sha256")
        if not isinstance(frame_sha, str) or not frame_sha:
            raise ValueError("A reviewed update requires the original frame SHA")
        if has_source:
            key = source_sample_key(update)
            target = sample_by_key.get(key)
            if target is None:
                raise ValueError("Reviewed update source key has no original sample")
            if target.get("frame_sha256") != frame_sha:
                raise ValueError("Reviewed update original frame SHA differs")
        else:
            pts = update.get("source_pts")
            if isinstance(pts, bool) or not isinstance(pts, Integral) or pts < 0:
                raise ValueError("Legacy reviewed update requires integer native PTS")
            candidates = [key for key, sample in sample_by_key.items()
                          if key[2] == int(pts) and sample.get("frame_sha256") == frame_sha]
            if len(candidates) != 1:
                raise ValueError(f"Legacy reviewed update PTS/frame SHA needs one source sample, found {len(candidates)}")
            key = candidates[0]
        if key in resolved:
            raise ValueError("Duplicate reviewed update for one source sample key")
        resolved[key] = update
    return resolved


def annotation_revision_sha256(annotation):
    """Bind a superseded source record by all original fields, not its name."""
    return hashlib.sha256(json.dumps(annotation,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()


def active_source_supervision_mask(annotation):
    """Return source-point authority selection, independently of RGB visibility.

    Later direct user revisions may retire named original positions while all
    UV, normals, intervals and original authority remain intact. The source
    visibility mask still describes visible RGB and is never changed here.
    Consumers calculate normals on the complete source curve before selecting
    this mask, and retain original indices to avoid joining retired gaps.
    """
    count=len(annotation.get("uv",[]))
    if "source_active_supervision_mask" in annotation:
        mask=annotation["source_active_supervision_mask"]
        if (not isinstance(mask,list) or len(mask)!=count or
                any(type(value) is not bool for value in mask)):
            raise ValueError("Active source supervision needs one explicit boolean per original UV")
        mask=list(mask)
    else:
        mask=[True]*count
    if annotation.get("superseded_by_user_boundary_revision"):
        return [False]*count  # Existing whole-group revision protocol.
    return mask


def source_supervision_point_counts(annotation):
    """Distinguish unchanged positions, current authority and retired history."""
    mask=active_source_supervision_mask(annotation)
    retired=len(mask)-sum(mask)
    enabled=(annotation.get("state") not in ("ignored","occluded","unreviewed") and
             annotation.get("supervision_enabled",True) and
             annotation.get("vision_supervision",{}).get("semantic_supervision_enabled",True))
    active=sum(mask) if enabled else 0
    return dict(original_source_position_count=len(mask),active_source_supervision_position_count=active,
                superseded_source_position_count=retired,
                other_inactive_source_position_count=len(mask)-active-retired)


def training_supervision_consumption_record(*, step, sample_index, sample, selected_annotations,
        source_loss_diagnostics, classes, shot_key, crop_origin_xy, crop_hw, full_frame,
        previous_source_observation, optimizer_step_succeeded, reviewed_region_counts=(),
        background_counts=None, localization_diagnostics=()):
    """Describe one successful optimizer step's actual target consumers.

    Call after ``optimizer.step`` with that step's existing crop annotations,
    interval-loss diagnostics and actual mask/loss counts. This helper executes
    no model and draws no random numbers. It never replaces actual counts with
    manifest sizes. The caller appends its result to a run-bound JSONL ledger;
    it must retain a separate header containing model/checkpoint/input hashes.

    Each reviewed-region row must provide ``region_index`` (or its existing
    unique ``region_id``), ``supervised_pixels``,
    ``protected_human_pixels`` and ``human_protection_policy`` from the actual
    mask consumer. Zero protected pixels with ``fail_on_human_overlap`` means
    a conflict would fail the step; it does not imply a mask was subtracted.
    Keep selected indices distinct from semantic indices whose source interval
    has at least one in-bounds support pixel. The actual diagnostic count must
    agree with this weight-independent selection.
    """
    import numpy as np

    def integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"Consumption {name} must be an integer >= {minimum}")
        return int(value)

    def exact_indices(values):
        values=np.asarray(values)
        if values.ndim!=1 or values.dtype.kind not in "iu" or not len(values):
            raise ValueError("Consumption source indices must be nonempty integer original positions")
        indices=values.astype(np.int64)
        if np.any(indices<0) or np.any(np.diff(indices)<=0):
            raise ValueError("Consumption source indices must be unique ordered original positions")
        runs=[]
        for value in indices.tolist():
            if runs and value==runs[-1][0]+runs[-1][1]:
                runs[-1][1]+=1
            else:
                runs.append([value,1])
        return indices,dict(count=len(indices),runs_start_length=runs,
            ordered_original_indices_int64_le_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())

    if optimizer_step_succeeded is not True:
        raise ValueError("Consumption rows require an actually successful optimizer step")
    step=integer(step,"step",1);sample_index=integer(sample_index,"sample index")
    if type(full_frame) is not bool or not isinstance(shot_key,str) or not shot_key:
        raise ValueError("Consumption needs explicit full-frame state and native shot identity")
    if len(crop_origin_xy)!=2 or len(crop_hw)!=2:
        raise ValueError("Consumption needs native crop origin and height/width")
    origin=[integer(v,"crop origin") for v in crop_origin_xy]
    height,width=[integer(v,"crop dimension",1) for v in crop_hw]
    source_hash,clock,pts=source_sample_key(sample)
    scope=sample.get("source_shot_scope")
    if scope:
        if (scope.get("source_sha256")!=source_hash or str(Fraction(scope["time_base"]))!=clock or
                not scope["start_pts"]<=pts<scope["end_pts_exclusive"] or
                f"{source_hash}:{scope['start_pts']}:{scope['end_pts_exclusive']}"!=shot_key):
            raise ValueError("Consumption native shot/source scope disagrees")
    names=list(classes)
    if not names or len(names)!=len(set(names)):
        raise ValueError("Consumption requires the actual distinct semantic classes")
    originals={a["group_id"]:a for a in sample["annotations"]}
    if len(originals)!=len(sample["annotations"]):
        raise ValueError("Consumption source has duplicate annotation groups")
    diagnostics={d["group_id"]:d for d in source_loss_diagnostics}
    if len(diagnostics)!=len(source_loss_diagnostics):
        raise ValueError("Consumption contains duplicate actual semantic diagnostics")
    groups=[];seen=set()
    for selected in selected_annotations:
        identity=selected["group_id"]
        if identity in seen or identity not in originals:
            raise ValueError("Consumption selected group is duplicated or has no source authority")
        seen.add(identity);original=originals[identity]
        indices,selected_encoding=exact_indices(selected["source_point_indices"])
        full_uv=np.asarray(original["uv"],np.float64)
        if indices[-1]>=len(full_uv):
            raise ValueError("Consumption source index is outside the original annotation")
        active=np.asarray(active_source_supervision_mask(original),bool)
        visible=np.asarray(original.get("source_visible_mask",[True]*len(full_uv)))
        if visible.dtype!=np.bool_ or visible.shape!=(len(full_uv),):
            raise ValueError("Consumption visibility needs one explicit boolean per original UV")
        if not np.all(active[indices]&visible[indices]):
            raise ValueError("Consumption includes retired or source-invisible authoritative positions")
        uv=np.asarray(selected["uv"],np.float64)
        if (uv.shape!=(len(indices),2) or not np.isfinite(uv).all() or
                not np.allclose(uv+np.asarray(origin),full_uv[indices],rtol=0,atol=1e-8) or
                np.any(uv[:,0]<0) or np.any(uv[:,0]>=width) or np.any(uv[:,1]<0) or np.any(uv[:,1]>=height)):
            raise ValueError("Consumption crop UVs do not match the selected original source positions")
        if (original.get("state") in ("ignored","occluded","unreviewed") or
                selected.get("state")!=original.get("state") or selected["kind"]!=original["kind"]):
            raise ValueError("Consumption changed source state/role or selected an ignored authority")
        semantic_keep=np.zeros(len(indices),bool)
        if original["kind"] in names:
            if original.get("state")=="user_confirmed_visual_supervision":
                semantic_keep[:]=True
            else:
                normals=np.asarray(selected["normal"],np.float32)
                interval=np.asarray(selected["interval_px"],np.float32)
                if normals.shape!=(len(indices),2) or interval.shape!=(len(indices),2):
                    raise ValueError("Consumption interval/normal shape disagrees with selected source positions")
                offsets=np.arange(-24,25,dtype=np.float32)
                pixels=uv.astype(np.float32)[:,None]+normals[:,None]*offsets[None,:,None]
                inside=((pixels[...,0]>=0)&(pixels[...,0]<width)&(pixels[...,1]>=0)&(pixels[...,1]<height))
                support=(offsets[None]>=interval[:,:1])&(offsets[None]<=interval[:,1:])&inside
                semantic_keep=support.any(axis=1)
        expected_count=int(semantic_keep.sum());diagnostic=diagnostics.get(identity)
        actual_count=integer(diagnostic["sample_count"],"actual semantic sample count") if diagnostic else 0
        if (actual_count!=expected_count or
                (diagnostic is not None and diagnostic.get("kind")!=original["kind"])):
            raise ValueError("Consumption actual loss diagnostic disagrees with source target selection")
        consumed=indices[semantic_keep]
        semantic_encoding=exact_indices(consumed)[1] if len(consumed) else dict(count=0,runs_start_length=[],
            ordered_original_indices_int64_le_sha256=hashlib.sha256(b"").hexdigest())
        groups.append(dict(group_id=identity,kind=original["kind"],state=original.get("state"),
            authority_role=original.get("vision_supervision",{}).get("authority_role"),
            original_annotation_sha256=annotation_revision_sha256(original),
            original_position_count=len(full_uv),selected_source_positions=selected_encoding,
            actual_semantic_consumed_source_positions=semantic_encoding,
            actual_source_loss_diagnostic=deepcopy(diagnostic),
            semantic_indices_basis="original crop indices plus actual interval support/inbounds selection; actual loss diagnostic count verified",
            retired_source_positions_consumed=0,source_invisible_positions_consumed=0,
            model_confidence_used_to_select_human=False))
    if set(diagnostics)-seen:
        raise ValueError("Consumption semantic diagnostic has no actual selected annotation")
    localization_rows=[];localization_seen=set()
    selected_by_id={a["group_id"]:a for a in selected_annotations}
    for diagnostic in localization_diagnostics:
        identity=diagnostic["group_id"]
        if identity in localization_seen or identity not in selected_by_id:
            raise ValueError("Localization consumption needs one actual selected source group")
        localization_seen.add(identity)
        count=integer(diagnostic["sample_count"],"actual localization sample count")
        values=diagnostic["source_point_indices"]
        indices,encoding=(exact_indices(values) if count else (np.empty(0,np.int64),dict(count=0,
            runs_start_length=[],ordered_original_indices_int64_le_sha256=hashlib.sha256(b"").hexdigest())))
        if len(values)!=count or len(indices)!=count or not set(indices.tolist()).issubset(
                set(np.asarray(selected_by_id[identity]["source_point_indices"]).tolist())):
            raise ValueError("Localization consumption indices/count differ from actual crop targets")
        integer(diagnostic["actual_positive_support_sample_count"],"actual localization positive support samples")
        if diagnostic.get("retired_source_positions_used_for_support")!=0:
            raise ValueError("Localization consumption cannot include retired source support")
        localization_record=dict(group_id=identity,actual_localization_source_positions=encoding,
            actual_localization_diagnostic=deepcopy(diagnostic))
        if "rank_sample_count" in diagnostic:
            rank_count=integer(diagnostic["rank_sample_count"],"actual rank sample count")
            rank_values=diagnostic["rank_source_point_indices"]
            rank_indices,rank_encoding=(exact_indices(rank_values) if rank_count else (np.empty(0,np.int64),dict(count=0,
                runs_start_length=[],ordered_original_indices_int64_le_sha256=hashlib.sha256(b"").hexdigest())))
            if (len(rank_values)!=rank_count or not set(rank_indices.tolist()).issubset(set(indices.tolist())) or
                    diagnostic.get("rank_weight",0)<=0 and rank_count):
                raise ValueError("Rank consumption differs from actual positively weighted localization targets")
            localization_record["actual_rank_source_positions"]=rank_encoding
        localization_rows.append(localization_record)
    source_regions=sample.get("regions",[])
    region_by_id={r["region_id"]:i for i,r in enumerate(source_regions) if r.get("region_id") is not None}
    if len(region_by_id)!=sum(r.get("region_id") is not None for r in source_regions):
        raise ValueError("Consumption source has duplicate reviewed region identities")
    region_rows=[];region_seen=set()
    for counts in reviewed_region_counts:
        if counts.get("region_index") is not None:
            index=integer(counts["region_index"],"actual source region index")
        elif counts.get("region_id") in region_by_id:
            index=region_by_id[counts["region_id"]]
        else:
            raise ValueError("Consumption region count needs its original source index or existing ID")
        if index>=len(source_regions) or index in region_seen:
            raise ValueError("Consumption region count lacks a unique original source region")
        region_seen.add(index);region=source_regions[index];identity=region.get("region_id")
        if counts.get("region_id") is not None and counts["region_id"]!=identity:
            raise ValueError("Consumption source region index and identity disagree")
        supervised=integer(counts["supervised_pixels"],"actual reviewed-region supervised pixels")
        protected=integer(counts["protected_human_pixels"],"actual reviewed-region protected pixels")
        policy=counts.get("human_protection_policy")
        if policy not in ("fail_on_human_overlap","actual_mask_subtraction","no_human_mask_subtraction"):
            raise ValueError("Consumption requires the actual human protection policy")
        if policy!="actual_mask_subtraction" and protected:
            raise ValueError("Consumption cannot claim protected pixels without actual mask subtraction")
        x0,y0,x1,y1=region["rectangle_xyxy"]
        region_width=max(0,min(width,int(x1-origin[0]))-max(0,int(x0-origin[0])))
        region_height=max(0,min(height,int(y1-origin[1]))-max(0,int(y0-origin[1])))
        if supervised+protected>region_width*region_height:
            raise ValueError("Consumption reviewed-region counts exceed the actual source region/crop overlap")
        if counts.get("excluded_kinds") is not None and counts["excluded_kinds"]!=region.get("excluded_kinds"):
            raise ValueError("Consumption actual exclusion roles differ from their source authority")
        region_rows.append(dict(region_id=identity,region_index=index,
            merged_source_manifest_pointer=f"/samples/{sample_index}/regions/{index}",positive_kind=region.get("kind"),
            excluded_kinds=deepcopy(region.get("excluded_kinds")),
            actual_source_region_sha256=annotation_revision_sha256(region),actual_loss_counts=deepcopy(counts)))
    if region_seen!=set(range(len(source_regions))):
        raise ValueError("Consumption must record every actual reviewed-region consumer, including zero masks")
    if background_counts is not None:
        if integer(background_counts["supervised_pixels"],"actual background-mask supervised pixels")>height*width:
            raise ValueError("Consumption background counts exceed the actual crop pixels")
    return dict(schema="rinkspace.optimizer-success-source-target-consumption.v1",step=step,
        optimizer_step_succeeded=True,sample_index=sample_index,
        source_binding=dict(source_sha256=source_hash,time_base=clock,source_pts=pts),
        frame_sha256=sample.get("frame_sha256"),native_RGB_sha256=sample.get("decoded_native_RGB_sha256",sample.get("native_RGB_sha256")),
        shot_key=shot_key,crop_origin_native_xy=origin,native_training_hw=[height,width],full_frame=full_frame,
        previous_source_observation=deepcopy(previous_source_observation),groups=groups,
        actual_localization_groups=localization_rows,
        reviewed_regions=region_rows,actual_background_loss_counts=deepcopy(background_counts),
        reconstructed_prior_batches=False,manifest_sizes_are_not_consumption_counts=True,
        per_group_loss_weighting="existing actual interval_supervision implementation; record does not reweight targets",
        pixel_accuracy_or_independent_ability_accepted=False)


def _confirmed_position_support_segments(sample, kind, origin):
    """Manual polyline uncertainty supports, never new RGB measurements."""
    import numpy as np
    segments=[];source_counts=[]
    for original in sample["annotations"]:
        if original.get("kind")!=kind or original.get("state") not in (
                "user_confirmed_visual_supervision","user_trace_weak"):
            continue
        full=np.asarray(original["uv"],float);count=len(full)
        active=np.asarray(active_source_supervision_mask(original),bool)
        visible=np.asarray(original.get("source_visible_mask",[True]*count))
        if visible.dtype!=np.bool_ or visible.shape!=(count,):
            raise ValueError("Localization support requires explicit source visibility per original UV")
        if original.get("interval_px") is None:
            continue  # CE remains valid; no original position uncertainty was supplied.
        interval=np.asarray(original["interval_px"],float)
        if interval.shape==(2,):interval=np.tile(interval,(count,1))
        elif interval.ndim==0:interval=np.tile([-float(interval),float(interval)],(count,1))
        if interval.shape!=(count,2) or not np.isfinite(interval).all() or np.any(interval[:,0]>interval[:,1]):
            raise ValueError("Localization support must preserve valid original position intervals")
        normal=original.get("normal",original.get("normals"))
        if normal is None:
            tangent=np.gradient(full,axis=0) if count>1 else np.zeros_like(full)
            normal=np.column_stack((-tangent[:,1],tangent[:,0]))
        normal=np.asarray(normal,float)
        if normal.shape!=full.shape or not np.isfinite(normal).all():
            raise ValueError("Localization normals must belong to the complete original curve")
        lengths=np.linalg.norm(normal,axis=1);normal=normal/np.maximum(lengths[:,None],1e-12)
        points=full-np.asarray(origin);ids=np.asarray(original.get("segment_ids",np.zeros(count,int)))
        if ids.shape!=(count,):raise ValueError("Localization segment IDs must match original source positions")
        eligible=active&(lengths>1e-12)
        group=original["group_id"]
        used=np.flatnonzero(eligible)
        source_counts.append(dict(group_id=group,original_count=count,active_original_indices=used.tolist(),
            visible_count=int((eligible&visible).sum()),known_nonvisible_count=int((eligible&~visible).sum()),
            retired_original_count=int((~active).sum()),retired_source_positions_used_for_support=0))
        for index in used:
            segments.append((points[index],points[index],normal[index],normal[index],
                interval[index],interval[index],bool(visible[index])))
        for a,b in zip(range(count-1),range(1,count)):
            if eligible[a] and eligible[b] and ids[a]==ids[b]:
                segments.append((points[a],points[b],normal[a],normal[b],interval[a],interval[b],
                    bool(visible[a]&visible[b])))
        if (count>2 and original.get("annotation_closed",original.get("closed",False)) and
                eligible[0] and eligible[-1] and ids[0]==ids[-1]):
            segments.append((points[-1],points[0],normal[-1],normal[0],interval[-1],interval[0],
                bool(visible[-1]&visible[0])))
    return segments,source_counts


def _manual_profile_support(query, segments, *, tangent_tolerance_px):
    """Protect full manual paths without bridging inactive or missing segments."""
    import numpy as np
    positive=np.zeros(len(query),bool);nonvisible=np.zeros(len(query),bool)
    if not segments or not len(query):return positive,nonvisible
    arrays=[np.asarray([s[i] for s in segments]) for i in range(7)]
    a,b,na,nb,ia,ib,visibility=arrays
    radius=np.maximum(np.abs(ia).max(1),np.abs(ib).max(1))+tangent_tolerance_px
    lo=np.floor((np.minimum(a,b)-radius[:,None])/32).astype(int)
    hi=np.floor((np.maximum(a,b)+radius[:,None])/32).astype(int)
    buckets={}
    for index in range(len(a)):
        for x in range(lo[index,0],hi[index,0]+1):
            for y in range(lo[index,1],hi[index,1]+1):buckets.setdefault((x,y),[]).append(index)
    tiles,owners=np.unique(np.floor(query/32).astype(int),axis=0,return_inverse=True)
    for tile_id,tile in enumerate(tiles):
        choices=buckets.get(tuple(tile),[])
        if not choices:continue
        indices=np.flatnonzero(owners==tile_id);start=a[choices];end=b[choices];delta=end-start
        displacement=query[indices,None]-start[None]
        parameter=np.clip((displacement*delta[None]).sum(2)/np.maximum((delta*delta).sum(1)[None],1e-20),0,1)
        nearest=start[None]+parameter[...,None]*delta[None]
        normal=na[choices][None]+parameter[...,None]*(nb[choices]-na[choices])[None]
        normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-12)
        relative=query[indices,None]-nearest;signed=(relative*normal).sum(2)
        tangent=np.linalg.norm(relative-signed[...,None]*normal,axis=2)
        lower=ia[choices,0][None]+parameter*(ib[choices,0]-ia[choices,0])[None]
        upper=ia[choices,1][None]+parameter*(ib[choices,1]-ia[choices,1])[None]
        supported=(tangent<=tangent_tolerance_px)&(signed>=lower)&(signed<=upper)
        positive[indices]=(supported&visibility[choices][None]).any(1)
        nonvisible[indices]=(supported&~visibility[choices][None]).any(1)
    return positive,nonvisible


def confirmed_position_interval_localization_loss(logits, selected_annotations, classes, *,
        original_sample, crop_origin_xy, profile_radius_px=24, profile_step_px=1.,
        temperature=1., tangent_tolerance_px=1., rank_weight=0., rank_margin_log_odds=1.):
    """Auxiliary spatial likelihood; keep the existing human UV class CE.

    Original normals and position intervals define source-profile supports.
    All active confirmed same-kind manual paths may contribute positive support;
    other circle sides never compete as negatives. Full-source support is built
    before crop selection, with no bridge over retired or explicit segment gaps.
    Known nonvisible supports are removed from comparisons, never background.

    Spatial scores are class-vs-rest log odds from raw logits. They avoid both
    FP32 class-P=1 saturation and cheating via a shared per-pixel logit offset.
    Outside support is a comparative localization constraint, not a background
    or sibling-class label. Unknown pixels retain no explicit semantic target.
    This function adds no head and emits no predicted UV or RGB ground truth.
    Optional max ranking keeps the strongest own-interval sample above every
    sampled outside-union candidate by a positive log-odds margin. Other
    confirmed same-kind supports never become negative competitors. Retain
    both mass and direct class CE: bilinear pixel sharing can couple gradients
    at nearby authoritative positions. Finite profile sampling does not prove
    the location of a continuous peak between samples or an actual RGB edge.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    if (logits.ndim!=4 or logits.shape[0]!=1 or logits.shape[1]!=len(classes) or len(classes)<2 or
            len(set(classes))!=len(classes) or min(logits.shape[-2:])<2):
        raise ValueError("Localization needs actual single-source class logits [1,C,H,W]")
    if (isinstance(profile_radius_px,bool) or not isinstance(profile_radius_px,Integral) or profile_radius_px<1 or
            not all(np.isfinite(v) and v>0 for v in (profile_step_px,temperature,tangent_tolerance_px))):
        raise ValueError("Localization requires positive native profile/temperature parameters")
    if (isinstance(rank_weight,bool) or not np.isfinite(rank_weight) or rank_weight<0 or
            isinstance(rank_margin_log_odds,bool) or not np.isfinite(rank_margin_log_odds) or rank_margin_log_odds<=0):
        raise ValueError("Localization ranking needs nonnegative weight and a positive log-odds margin")
    source_sample_key(original_sample)
    height,width=logits.shape[-2:];origin=np.asarray(crop_origin_xy,float)
    if origin.shape!=(2,) or not np.isfinite(origin).all():raise ValueError("Localization needs native crop origin")
    originals={a["group_id"]:a for a in original_sample["annotations"]}
    if len(originals)!=len(original_sample["annotations"]):
        raise ValueError("Localization needs unique original source group identities")
    if len({a["group_id"] for a in selected_annotations})!=len(selected_annotations):
        raise ValueError("Localization cannot consume the same selected source group twice")
    offsets=np.arange(-profile_radius_px,profile_radius_px+profile_step_px*.5,profile_step_px,dtype=np.float32)
    losses=[];diagnostics=[];support_cache={}
    for annotation in selected_annotations:
        if annotation.get("state")!="user_confirmed_visual_supervision" or annotation.get("kind") not in classes:
            continue
        group=annotation["group_id"];original=originals.get(group)
        if original is None or original["kind"]!=annotation["kind"]:
            raise ValueError("Localization crop group lacks its original source authority")
        indices=np.asarray(annotation["source_point_indices"])
        if (indices.ndim!=1 or indices.dtype.kind not in "iu" or not len(indices) or
                np.any(indices<0) or indices.max()>=len(original["uv"]) or np.any(np.diff(indices)<=0)):
            raise ValueError("Localization requires exact ordered original source point indices")
        active=np.asarray(active_source_supervision_mask(original),bool)
        visible=np.asarray(original.get("source_visible_mask",[True]*len(active)))
        if visible.dtype!=np.bool_ or visible.shape!=active.shape or not (active[indices]&visible[indices]).all():
            raise ValueError("Localization cannot consume retired or source-invisible human positions")
        points=np.asarray(annotation["uv"],float)
        if not np.allclose(points+origin,np.asarray(original["uv"])[indices],atol=1e-8,rtol=0):
            raise ValueError("Localization crop UV differs from unchanged original human positions")
        normal=np.asarray(annotation["normal"],float);interval=np.asarray(annotation["interval_px"],float)
        if (normal.shape!=points.shape or interval.shape!=(len(points),2) or
                not np.isfinite(normal).all() or not np.isfinite(interval).all() or
                np.any(interval[:,0]>interval[:,1])):
            raise ValueError("Localization needs the selected original normals and intervals")
        original_interval=original.get("interval_px")
        if original_interval is None:
            diagnostics.append(dict(group_id=group,kind=annotation["kind"],sample_count=0,
                source_point_indices=[],actual_positive_support_sample_count=0,
                retired_source_positions_used_for_support=0,
                skipped_reason="no original position uncertainty interval; direct human CE retained"))
            continue
        original_interval=np.asarray(original_interval,float)
        if original_interval.ndim==0:
            original_interval=np.tile([-float(original_interval),float(original_interval)],(len(original["uv"]),1))
        elif original_interval.shape==(2,):
            original_interval=np.tile(original_interval,(len(original["uv"]),1))
        if original_interval.shape!=(len(original["uv"]),2) or not np.array_equal(interval,original_interval[indices]):
            raise ValueError("Localization must preserve unchanged original position intervals")
        original_normal=original.get("normal",original.get("normals"))
        if original_normal is None:
            full_uv=np.asarray(original["uv"],float)
            tangent=np.gradient(full_uv,axis=0) if len(full_uv)>1 else np.zeros_like(full_uv)
            original_normal=np.column_stack((-tangent[:,1],tangent[:,0]))
        original_normal=np.array(original_normal,dtype=float,copy=True)
        if original_normal.shape!=np.asarray(original["uv"]).shape:
            raise ValueError("Localization needs complete original curve normals")
        original_normal/=np.maximum(np.linalg.norm(original_normal,axis=1,keepdims=True),1e-6)
        if not np.allclose(normal,original_normal[indices],atol=1e-8,rtol=0):
            raise ValueError("Localization cannot recompute normals on the cropped or active subset")
        norm=np.linalg.norm(normal,axis=1);normal=normal/np.maximum(norm[:,None],1e-12)
        query=points[:,None]+normal[:,None]*offsets[None,:,None]
        inside=((query[...,0]>=0)&(query[...,0]<width)&(query[...,1]>=0)&(query[...,1]<height))
        own=(offsets[None]>=interval[:,:1])&(offsets[None]<=interval[:,1:])
        kind=annotation["kind"]
        if kind not in support_cache:support_cache[kind]=_confirmed_position_support_segments(original_sample,kind,origin)
        segments,support_sources=support_cache[kind]
        other_positive,nonvisible=_manual_profile_support(query.reshape(-1,2),segments,tangent_tolerance_px=tangent_tolerance_px)
        positive=own|other_positive.reshape(inside.shape)
        valid=inside&(~nonvisible.reshape(inside.shape)|positive)
        positive &= valid
        rows=(norm>1e-12)&positive.any(1)&(valid&~positive).any(1)
        selected_indices=indices[rows]
        row_diagnostic=dict(group_id=group,kind=kind,sample_count=int(rows.sum()),source_point_indices=selected_indices.tolist(),
            selected_human_position_count=len(indices),actual_positive_support_sample_count=int(positive[rows].sum()),
            own_interval_support_sample_count=int((own&valid)[rows].sum()),
            added_same_kind_confirmed_support_sample_count=int((positive&~own)[rows].sum()),
            actual_comparative_profile_sample_count=int((valid&~positive)[rows].sum()),
            known_nonvisible_support_samples_excluded=int((inside&~valid).sum()),
            full_source_confirmed_support=support_sources,retired_source_positions_used_for_support=0,
            position_intervals_changed=False,normals_recomputed_after_crop=False,
            class_CE_replaced=False,unlabelled_pixels_assigned_background_or_other_positive_class=False,
            human_confidence_filter_used=False,profile_radius_native_px=int(profile_radius_px),
            profile_step_native_px=float(profile_step_px),temperature=float(temperature),
            spatial_score="raw class logit minus logsumexp other-class logits",
            same_kind_manual_polyline_support_is_not_new_RGB_measurement=True,
            positive_profile_support_bool_sha256=hashlib.sha256(positive.tobytes()).hexdigest(),
            localization_loss=0.,mass_loss=0.,rank_loss=0.,rank_weight=float(rank_weight),
            rank_margin_log_odds=float(rank_margin_log_odds),rank_sample_count=0,rank_source_point_indices=[],
            rank_inside_support="unchanged own position interval",
            rank_outside_support="inbounds visible comparisons excluding full confirmed same-kind union",
            other_confirmed_same_kind_support_used_as_rank_negative=False,
            sampled_rank_condition_is_not_continuous_peak_or_RGB_precision=True)
        if rows.any():
            grid=torch.as_tensor(query[rows],device=logits.device,dtype=torch.float32)
            grid=grid*grid.new_tensor([2/(width-1),2/(height-1)])-1
            scores=F.grid_sample(logits.float(),grid[None],align_corners=True)[0]
            category=list(classes).index(kind);others=[i for i in range(len(classes)) if i!=category]
            odds=(scores[category]-torch.logsumexp(scores[others],dim=0))/temperature
            valid_tensor=torch.as_tensor(valid[rows],device=logits.device)
            positive_tensor=torch.as_tensor(positive[rows],device=logits.device)
            mass_values=(torch.logsumexp(odds.masked_fill(~valid_tensor,-torch.inf),dim=1)-
                  torch.logsumexp(odds.masked_fill(~positive_tensor,-torch.inf),dim=1))
            mass_loss, source_mass = source_weighted_mean(mass_values, annotation, rows)
            own_tensor=torch.as_tensor((own&valid)[rows],device=logits.device)
            outside_tensor=valid_tensor&~positive_tensor
            rank_rows=own_tensor.any(1)&outside_tensor.any(1)
            rank_loss=odds.sum()*0
            if rank_weight>0 and rank_rows.any():
                inside_max=odds[rank_rows].masked_fill(~own_tensor[rank_rows],-torch.inf).max(1).values
                outside_max=odds[rank_rows].masked_fill(~outside_tensor[rank_rows],-torch.inf).max(1).values
                rank_mask=rows.copy()
                rank_mask[rows]=rank_rows.detach().cpu().numpy()
                rank_loss,_=source_weighted_mean((outside_max-inside_max+rank_margin_log_odds).relu(), annotation, rank_mask)
            loss=mass_loss+rank_weight*rank_loss
            rank_indices=selected_indices[rank_rows.detach().cpu().numpy()] if rank_weight>0 else np.empty(0,np.int64)
            row_diagnostic.update(mass_loss=float(mass_loss.detach()),rank_loss=float(rank_loss.detach()),
                rank_sample_count=len(rank_indices),rank_source_point_indices=rank_indices.tolist())
            losses.append((annotation,loss,source_mass));row_diagnostic["localization_loss"]=float(loss.detach())
        diagnostics.append(row_diagnostic)
    return reduce_source_group_losses(losses,logits.sum()*0),diagnostics


def confirmed_source_point_coverage(sample, consumption_records):
    """Count actual successful semantic consumers by untouched original index."""
    import numpy as np
    key=source_sample_key(sample)
    originals={a["group_id"]:a for a in sample["annotations"]}
    if len(originals)!=len(sample["annotations"]):
        raise ValueError("Point coverage requires unique original source groups")
    counts={a["group_id"]:np.zeros(len(a["uv"]),np.int64) for a in sample["annotations"]
            if a.get("state")=="user_confirmed_visual_supervision"}
    for record in consumption_records:
        if record.get("optimizer_step_succeeded") is not True:
            raise ValueError("Point coverage needs actual successful optimizer consumption records")
        if source_sample_key(record["source_binding"])!=key:
            continue
        seen=set()
        for group in record["groups"]:
            identity=group["group_id"]
            if identity in seen:
                raise ValueError("Coverage cannot count one consumed source group twice in a step")
            seen.add(identity)
            if identity not in counts:continue
            original=originals[identity]
            if (group.get("original_annotation_sha256")!=annotation_revision_sha256(original) or
                    group.get("original_position_count")!=len(original["uv"])):
                raise ValueError("Coverage consumption belongs to another original annotation revision")
            encoded=group["actual_semantic_consumed_source_positions"]
            values=[]
            for start,length in encoded["runs_start_length"]:
                if (isinstance(start,bool) or isinstance(length,bool) or not isinstance(start,Integral) or
                        not isinstance(length,Integral) or start<0 or length<1):
                    raise ValueError("Coverage requires original integer source-index runs")
                values.extend(range(start,start+length))
            indices=np.asarray(values,np.int64)
            if (len(indices)!=encoded["count"] or len(set(values))!=len(values) or
                    len(indices) and indices.max()>=len(counts[identity]) or
                    hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()!=encoded["ordered_original_indices_int64_le_sha256"]):
                raise ValueError("Coverage source-index count, range or SHA changed")
            active=np.asarray(active_source_supervision_mask(original),bool)
            visible=np.asarray(original.get("source_visible_mask",[True]*len(active)))
            if visible.dtype!=np.bool_ or visible.shape!=active.shape or not (active[indices]&visible[indices]).all():
                raise ValueError("Coverage cannot consume retired or invisible human positions")
            counts[identity][indices]+=1
    return dict(schema="rinkspace.confirmed-source-point-coverage.v1",source_key=list(key),
        groups={name:values.tolist() for name,values in counts.items()},
        original_annotation_sha256={a["group_id"]:annotation_revision_sha256(a) for a in sample["annotations"] if a["group_id"] in counts},
        only_actual_successful_semantic_source_consumption_counted=True)


def confirmed_source_crop_target(sample, coverage, rng, *, crop_size=512, bin_size=32):
    """Choose a least-consumed existing human point bin in this selected source.

    This sampler cannot create supervision, change shot weights, or promote
    model/assistant pixels. The caller uses the returned original UV to center
    an RGB crop, then its existing crop selector computes all actual targets.
    """
    import numpy as np
    if (isinstance(crop_size,bool) or not isinstance(crop_size,Integral) or crop_size<1 or
            isinstance(bin_size,bool) or not isinstance(bin_size,Integral) or bin_size<1):
        raise ValueError("Human coverage sampling requires positive crop and original-index bin sizes")
    key=source_sample_key(sample)
    if tuple(coverage.get("source_key",()))!=key:
        raise ValueError("Human crop coverage belongs to another original source frame")
    width,height=sample["frame_size_wh"]
    if crop_size>min(width,height):raise ValueError("Crop must remain in the original native source canvas")
    candidates=[]
    for original in sample["annotations"]:
        if original.get("state")!="user_confirmed_visual_supervision":continue
        uv=np.asarray(original["uv"],float);active=np.asarray(active_source_supervision_mask(original),bool)
        visible=np.asarray(original.get("source_visible_mask",[True]*len(active)))
        if visible.dtype!=np.bool_ or visible.shape!=active.shape:
            raise ValueError("Human coverage sampler needs the original explicit visibility mask")
        recorded=np.asarray(coverage.get("groups",{}).get(original["group_id"]))
        if coverage.get("original_annotation_sha256",{}).get(original["group_id"])!=annotation_revision_sha256(original):
            raise ValueError("Human crop coverage cannot silently use another annotation revision")
        if recorded.shape!=active.shape or recorded.dtype.kind not in "iu" or (recorded<0).any():
            raise ValueError("Coverage must count every unchanged original human position")
        eligible=active&visible&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        for first in range(0,len(uv),bin_size):
            indices=np.flatnonzero(eligible[first:first+bin_size])+first
            if len(indices):candidates.append((float(recorded[indices].mean()),original,indices,recorded))
    if not candidates:return None
    minimum=min(c[0] for c in candidates)
    _,original,indices,recorded=rng.choice([c for c in candidates if c[0]==minimum])
    minimum_point_count=recorded[indices].min()
    index=int(rng.choice(indices[recorded[indices]==minimum_point_count].tolist()))
    point=np.asarray(original["uv"])[index]
    origin=[int(np.clip(point[0]-crop_size//2,0,width-crop_size)),
            int(np.clip(point[1]-crop_size//2,0,height-crop_size))]
    return dict(source_key=list(key),group_id=original["group_id"],original_source_point_index=index,
        original_source_uv=point.tolist(),crop_origin_native_xy=origin,crop_hw=[int(crop_size)]*2,
        candidate_bin_original_indices=indices.tolist(),bin_mean_actual_consumption=minimum,
        selected_point_actual_consumption=int(recorded[index]),
        only_active_visible_direct_confirmed_source_positions=True,no_new_supervision_created=True,
        original_UV_authority_interval_and_normal_modified=False,shot_sampling_weight_modified=False)


def apply_user_boundary_revisions(manifest, revisions):
    """Append explicitly source-bound human corrections to a runtime copy.

    Source files and original UV/authority/intervals remain intact. A new group
    ID identifies the new version; a specifically named older group can leave
    active supervision only under this later user revision. Its exact original
    record survives in the append-only ledger. This does not propagate one
    frame's red line to other frames or infer a new position for an old drawing.
    The caller explicitly appends a separately bound sample before revising a
    new source frame; this helper never invents its image or clock identity.
    """
    result=deepcopy(manifest)
    samples={}
    for sample in result["samples"]:
        key=source_sample_key(sample)
        if key in samples:
            raise ValueError("Boundary revisions need unique source samples")
        samples[key]=sample
    ledger=result.setdefault("boundary_supervision_revision_ledger",[])
    prior={row["revision_id"]:row for row in ledger}
    for revision in revisions:
        if revision.get("schema")!="rinkspace.user-source-boundary-supervision-revision.v1":
            raise ValueError("Boundary revision needs its explicit source-bound schema")
        identity=revision.get("revision_id")
        version=revision.get("version")
        if (not isinstance(identity,str) or not identity or isinstance(version,bool)
                or not isinstance(version,Integral) or version<1):
            raise ValueError("Boundary revision needs a named positive integer version")
        authority=revision.get("authority",{})
        if (not isinstance(authority,dict) or not (
                authority.get("kind")=="direct_user_confirmation" or
                authority.get("kind")=="direct_user_drawing" and authority.get("human_authority") is True)):
            raise ValueError("A source boundary replacement requires direct user confirmation")
        revision_sha=annotation_revision_sha256(revision)
        if identity in prior:
            if prior[identity]["revision_record_sha256"]!=revision_sha:
                raise ValueError("An appended user boundary revision cannot change under the same ID")
            continue
        key=source_sample_key(revision["source_binding"])
        sample=samples.get(key)
        if sample is None or sample.get("frame_sha256")!=revision.get("frame_sha256"):
            raise ValueError("Boundary revision does not match its original source image")
        declared_rgb=revision.get("decoded_native_RGB_sha256")
        if declared_rgb is not None:
            if not sample.get("frame_path") or native_pixel_sha256(sample["frame_path"],sample.get("frame_key"))!=declared_rgb:
                raise ValueError("Boundary revision differs from the actual native RGB pixels")
            if sample.get("decoded_native_RGB_sha256") not in (None,declared_rgb):
                raise ValueError("Boundary revision cannot change an existing native RGB identity")
            # Enrich this runtime copy of a legacy sample; its original file is
            # untouched and no source coordinate or annotation is rewritten.
            sample["decoded_native_RGB_sha256"]=declared_rgb
        versions=[row["version"] for row in ledger if tuple(row["source_sample_key"])==key]
        if versions and version<=max(versions):
            raise ValueError("Later user boundary revision versions must increase for this source frame")
        groups={row["group_id"]:row for row in sample["annotations"]}
        additions=deepcopy(revision.get("annotations",[]))
        new_ids=[row.get("group_id") for row in additions]
        if not additions or any(not isinstance(value,str) or not value or value in groups for value in new_ids) or len(set(new_ids))!=len(new_ids):
            raise ValueError("A boundary correction must append distinct new version group IDs")
        for row in additions:
            if (row.get("state")!="user_confirmed_visual_supervision" or
                    annotation_role(row,sample)["authority_role"]!="direct_user_confirmed_native_UV"):
                raise ValueError("New source boundary coordinates require direct human supervision")
            role=row.get("boundary_role")
            if row.get("kind")=="ice_contact" and role!="lower_board_base_strip_to_ice":
                raise ValueError("Ice contact is the lower board base-strip edge, not its internal upper edge")
            board_roles = ("lower_board_base_strip_to_ice", "upper_base_strip_internal_board_edge")
            ice_roles = ("paint_outer_boundary", "paint_inner_boundary", "paint_long_edge", "paint_centerline")
            if not (role in board_roles or row.get("kind") == "ice_paint" and role in ice_roles
                    and row.get("feature_id") and row.get("primitive") in ("line", "circle")):
                raise ValueError("A source boundary correction requires its explicitly confirmed physical role")
            active_source_supervision_mask(row)
            if "source_visible_mask" in row:
                validate_source_visibility(row,sample)
            row["user_boundary_revision_id"]=identity
            row["user_boundary_revision_version"]=int(version)
        original_records=[]
        partial_records=[]
        superseded=revision.get("supersedes",[])
        if len({row["group_id"] for row in superseded})!=len(superseded):
            raise ValueError("A boundary revision cannot supersede one group twice")
        for requested in superseded:
            group=groups.get(requested["group_id"])
            if group is None or annotation_revision_sha256(group)!=requested["original_annotation_sha256"]:
                raise ValueError("Boundary revision references a different original group record")
            original_records.append(deepcopy(group))
            if "source_point_indices" in requested:
                indices=requested["source_point_indices"]
                if (not isinstance(indices,list) or not indices or
                        any(isinstance(index,bool) or not isinstance(index,Integral) for index in indices) or
                        len(set(indices))!=len(indices) or
                        any(index<0 or index>=len(group.get("uv",[])) for index in indices)):
                    raise ValueError("Partial boundary retirement needs nonempty unique in-range source point indices")
                mask=active_source_supervision_mask(group)
                if any(not mask[index] for index in indices):
                    raise ValueError("A source boundary point cannot be retired twice")
                for index in indices:
                    mask[index]=False
                partial=dict(revision_id=identity,version=int(version),source_point_indices=list(indices),
                             original_annotation_sha256=requested["original_annotation_sha256"])
                group["source_active_supervision_mask"]=mask
                group.setdefault("partial_source_boundary_revisions",[]).append(deepcopy(partial))
                partial_records.append(dict(group_id=group["group_id"],**partial))
            else:
                group["original_state_before_user_boundary_revision"]=group.get("state")
                group["state"]="ignored"
                group["superseded_by_user_boundary_revision"]=identity
                group["superseded_reason"]="Later direct user source-boundary correction; original positions are preserved in the ledger"
        sample["annotations"].extend(additions)
        entry=dict(revision_id=identity,version=int(version),revision_record_sha256=revision_sha,
                   source_sample_key=list(key),frame_sha256=revision["frame_sha256"],
                   authority=deepcopy(authority),original_superseded_annotations=original_records,
                   partial_superseded_source_positions=partial_records,
                   added_group_ids=new_ids,original_source_files_modified=False)
        ledger.append(entry)
        prior[identity]=entry
    return result


def validate_source_visibility(annotation, sample):
    """Separate authoritative boundary positions from visible original RGB.

    Visibility is an explicit review of the bound native image, not a model
    probability or a rejection of a human position behind a foreground object.
    The pointwise mask stays parallel to the complete, unchanged drawing.
    """
    mask=annotation.get("source_visible_mask")
    evidence=annotation.get("source_visibility_evidence",{})
    if (not isinstance(mask,list) or len(mask)!=len(annotation.get("uv",[])) or
            any(type(value) is not bool for value in mask)):
        raise ValueError("Source visibility needs one explicit boolean per original boundary position")
    if (not isinstance(evidence,dict) or evidence.get("model_confidence_used") is not False or
            evidence.get("method")!="assistant_original_source_visibility_review" or
            source_sample_key(evidence)!=source_sample_key(sample) or
            evidence.get("frame_sha256")!=sample.get("frame_sha256") or
            evidence.get("decoded_native_RGB_sha256")!=sample.get("decoded_native_RGB_sha256")):
        raise ValueError("Source visibility must bind the actual native RGB independently of model confidence")
    return dict(authoritative_position_count=len(mask),visible_source_position_count=sum(mask),
                nonvisible_or_uncertain_position_count=len(mask)-sum(mask),
                human_positions_discarded=0,model_confidence_used=False)


def merge_user_boundary_samples(manifest, packet):
    """Add source templates and apply append-only, versioned human revisions."""
    if packet.get("schema")!="rinkspace.user-boundary-supervision-samples.v1":
        raise ValueError("User boundary samples need their versioned source-bound packet schema")
    for evidence in packet.get("source_evidence_files",[]):
        if file_sha256(evidence["path"])!=evidence["sha256"]:
            raise ValueError("An explicit user boundary source evidence file changed")
    result=deepcopy(manifest)
    known={source_sample_key(sample):sample for sample in result["samples"]}
    if len(known)!=len(result["samples"]):
        raise ValueError("User boundary merge needs distinct original source samples")
    for template in packet.get("samples",[]):
        key=source_sample_key(template)
        if template.get("annotations")!=[] or not template.get("frame_path") or not template.get("decoded_native_RGB_sha256"):
            raise ValueError("A new boundary sample needs an explicit original RGB template, with revisions separate")
        if key in known:
            if any(known[key].get(field)!=template.get(field) for field in ("frame_sha256","decoded_native_RGB_sha256")):
                raise ValueError("A repeated boundary template cannot change its original image")
        else:
            known[key]=deepcopy(template)
            result["samples"].append(known[key])
    return apply_user_boundary_revisions(result,packet.get("revisions",[]))


def merge_source_reviewed_regions(manifest, packet):
    """Append independently source-reviewed material regions by full source key."""
    if packet.get("schema")!="rinkspace.source-reviewed-material-regions.v1":
        raise ValueError("Material regions need their explicit source-review packet schema")
    for evidence in packet.get("source_evidence_files",[]):
        if file_sha256(evidence["path"])!=evidence["sha256"]:
            raise ValueError("A reviewed material-region source evidence file changed")
    result=deepcopy(manifest)
    updates=source_updates_by_key(result["samples"],packet.get("sample_updates",[]))
    for sample in result["samples"]:
        for region in updates.get(source_sample_key(sample),{}).get("regions",[]):
            if (not region.get("region_id") or region.get("state")!="source_reviewed_region" or
                    region.get("model_projection_used") is not False or
                    region.get("boundary_pixel_GT") is not False or
                    region.get("decoded_native_RGB_sha256")!=sample.get("decoded_native_RGB_sha256")):
                raise ValueError("Material regions require named original RGB review, separate from precise boundary labels")
            if "excluded_kinds" in region:
                excluded=region["excluded_kinds"]
                if ("kind" in region or not isinstance(excluded,list) or not excluded or
                        any(not isinstance(kind,str) or not kind for kind in excluded) or
                        len(set(excluded))!=len(excluded)):
                    raise ValueError("A reject-only material region needs distinct excluded kinds and no positive kind")
                if (region.get("reject_only_authority")!="actual_visual_role_absence" or
                        region.get("confidence_used_as_rejection_evidence") is not False or
                        region.get("model_probability_used_as_rejection_evidence") is not False):
                    raise ValueError("A semantic rejection needs actual source-role absence, not confidence or model probabilities")
            regions=sample.setdefault("regions",[])
            prior=[row for row in regions if row.get("region_id")==region["region_id"]]
            if prior:
                if len(prior)!=1 or prior[0]!=region:
                    raise ValueError("A reviewed material region cannot change under its existing ID")
            else:
                regions.append(deepcopy(region))
    return result


def native_boundary_overlay_reference(config, boundary_packet=None, *, root=None):
    """Select the current configured scope; retain packet references as history.

    Each digest belongs only to the path declared beside it. A later configured
    overlay must never inherit a frozen user packet's older overlay digest.
    Full source/timeline/adjacent-row checks remain the native overlay adapter's
    responsibility after this reference is selected and fingerprinted.
    """
    packet=boundary_packet or {}
    if config.get("native_boundary_overlay"):
        path=config["native_boundary_overlay"]
        expected=config.get("native_boundary_overlay_sha256")
        origin="configured_native_boundary_overlay"
    elif packet.get("source_boundary_overlay_path"):
        path=packet["source_boundary_overlay_path"]
        expected=packet.get("source_boundary_overlay_sha256")
        origin="legacy_user_boundary_packet_reference"
    elif config.get("source_boundary_overlay"):
        path=config["source_boundary_overlay"]
        expected=config.get("source_boundary_overlay_sha256")
        origin="legacy_configured_source_boundary_overlay"
    else:
        return None
    path=Path(path)
    if not path.is_absolute() and root is not None:
        path=Path(root)/path
    path=path.resolve()
    actual=file_sha256(path)
    if expected is not None and actual!=expected:
        raise ValueError("Selected native boundary overlay differs from its declared SHA256")
    return dict(path=str(path),sha256=actual,declared_sha256=expected,reference_origin=origin)


def timeline_with_native_boundary_overlay(timeline_path, source_sha256, overlay_path):
    """Apply a fully validated cut overlay to a copy, preserving classifications."""
    from .vision_review import source_shots
    timeline=json.loads(Path(timeline_path).read_text(encoding="utf-8-sig"))
    validated=source_shots(timeline_path,source_sha256,boundary_overlay_path=overlay_path)
    # source_shots validates every override before filtering ignored intervals.
    # Apply that entire validated packet atomically, including ignored neighbors.
    overlay=json.loads(Path(overlay_path).read_text(encoding="utf-8-sig"))
    corrected={row["shot_id"]:row for row in overlay["shot_overrides"]}
    overlay_meta=validated[0]["source_boundary_overlay"]
    for index,row in enumerate(timeline["intervals"]):
        correction=corrected.get(str(row.get("id",row.get("shot_id"))))
        if correction is not None:
            actual=deepcopy(correction["corrected_row"])
            actual["source_boundary_overlay"]=deepcopy(overlay_meta)
            actual["original_classified_start_pts"]=correction["original_start_pts"]
            actual["original_classified_end_pts_exclusive"]=correction["original_end_pts_exclusive"]
            timeline["intervals"][index]=actual
    return timeline


def native_pixel_sha256(path, frame_key=None):
    """Use the existing PNG/NPZ native-RGB reader; container hashes differ."""
    from .vision import _image
    return hashlib.sha256(_image(str(path), frame_key).tobytes()).hexdigest()


def annotation_role(annotation, sample=None):
    """Classify evidence authority without changing its UV, kind or uncertainty.

    A physical advertisement identity is not human confirmation of every ink
    edge's position. Existing reviewed print supervision remains enabled; its
    geometric meaning is not silently expanded to a rectangular panel border.
    """
    sample = sample or {}
    authority = annotation.get("authority", {})
    human = (annotation.get("state") == "user_confirmed_visual_supervision"
             or isinstance(authority, dict) and authority.get("human_authority") is True
             or sample.get("authority") == "direct_user_drawing")
    projected = (annotation.get("model_projection_used_to_generate_or_select_uv") is True
                 or annotation.get("world_or_camera_projection_used") is True
                 or annotation.get("original_quality", {}).get(
                     "model_projection_used_to_generate_or_select_uv") is True)
    state = annotation.get("state", "unreviewed")
    explicit_role = annotation.get("boundary_role", "")
    target_role = annotation.get("target_role")
    declared_target = target_role.split(";", 1)[0].strip() if isinstance(target_role, str) else None
    if annotation.get("kind") == "ad_boundary":
        if explicit_role == "internal_ink_hole":
            geometry_role = "internal_ink_hole"
        elif not explicit_role and declared_target == "outermost_visible_circular_print_stroke":
            # This is an existing source target, not inferred from a circle
            # shape or an advertisement ID. Fragment and hole roles remain
            # separate when the annotation already declares one explicitly.
            geometry_role = declared_target
        elif "printed" in explicit_role or "ink" in explicit_role or "native_print_boundary" in annotation.get("group_id", ""):
            geometry_role = explicit_role or "printed_component_edge"
        elif "external" in annotation.get("group_id", ""):
            geometry_role = "reviewed_external_advertisement_edge"
        else:
            geometry_role = explicit_role or "advertisement_edge_role_not_declared"
    else:
        geometry_role = annotation.get("ring_role") or explicit_role or annotation.get("kind")
    no_support = state in {"ignored", "occluded", "unreviewed"}
    candidate_only = (annotation.get("is_training_annotation") is False
                      or no_support
                      or "candidate" in state and not human
                      or annotation.get("supervision_enabled") is False and not human)
    superseded=annotation.get("superseded_by_user_boundary_revision")
    mode = ("forbidden_model_projection" if projected else
            "superseded_by_later_direct_user_boundary_revision" if superseded else
            "no_visible_source_support" if no_support else
            "direct_user_confirmed_native_UV" if human else
            "candidate_requires_source_or_human_review" if candidate_only else
            "assistant_original_source_interval")
    printed = "ink" in str(geometry_role) or "print" in str(geometry_role)
    if geometry_role == "external_printed_ink_component":
        semantic_role = "actual_printed_pattern_external_contour"
    elif geometry_role == "internal_ink_hole":
        semantic_role = "internal_print_hole_separate_from_external_contour"
    elif geometry_role == "print_background_outer_boundary":
        semantic_role = "printed_background_external_contour"
    else:
        semantic_role = geometry_role
    return dict(
        authority_role=mode, original_state=state,
        semantic_kind=annotation.get("kind"), geometry_role=geometry_role,
        semantic_boundary_role=semantic_role,
        original_target_role=target_role,
        printed_boundary_definition_authority=("direct_user_confirmation_20261006"
                                               if printed else None),
        semantic_supervision_enabled=not projected and not superseded and not no_support and (human or not candidate_only),
        superseded_by_user_boundary_revision=superseded,
        base_strip_belongs_to_board=(True if geometry_role in ("lower_board_base_strip_to_ice","upper_base_strip_internal_board_edge") else None),
        ice_contact_role_consistent=(geometry_role=="lower_board_base_strip_to_ice"
                                     if annotation.get("kind")=="ice_contact" and geometry_role in
                                     ("lower_board_base_strip_to_ice","upper_base_strip_internal_board_edge") else None),
        user_positions_filtered_by_model_confidence=False,
        geometry_as_panel_border=annotation.get("kind") == "ad_boundary" and not printed
        and geometry_role == "reviewed_external_advertisement_edge",
        original_pixel_ground_truth_claim=annotation.get("pixel_ground_truth", False),
        new_pixel_ground_truth_created=False,
        current_semantic_kind_preserved=True,
        overall_occupied_extent_is_separate=True if printed else None,
        internal_holes_are_separate=True if printed else None,
    )


def apply_supervision_roles(manifest):
    """Return a copy for existing training; all original information survives.

    Only an already authoritative human drawing's old weak state is promoted.
    Printed contours are annotated by role, not deleted, retyped or disabled.
    Consumers must inspect the role before treating semantics as geometry.
    """
    result = deepcopy(manifest)
    for sample in result["samples"]:
        for annotation in sample.get("annotations", []):
            role = annotation_role(annotation, sample)
            annotation["vision_supervision"] = role
            if role["authority_role"] == "forbidden_model_projection":
                raise ValueError("A model projection cannot become source supervision")
            if role["authority_role"] == "direct_user_confirmed_native_UV":
                annotation["original_supervision_state"] = annotation.get("state")
                annotation["state"] = "user_confirmed_visual_supervision"
    return result


def attach_source_support_integration(manifest):
    """Share one measured-path integral across declared same-curve branches.

    Without authenticated physical identity, retain the original annotation
    group instead of guessing which lines coincide. Occluded/inactive samples
    split paths and receive no visual supervision mass.
    """
    import numpy as np
    from .source_sampling import source_support_integration_weights
    result = deepcopy(manifest)
    for sample in result["samples"]:
        groups = {}
        for annotation in sample.get("annotations", []):
            if not annotation_role(annotation, sample)["semantic_supervision_enabled"]:
                continue
            identity = annotation.get("physical_curve_id") or annotation["group_id"]
            groups.setdefault((annotation["kind"], identity), []).append(annotation)
        for (kind, identity), annotations in groups.items():
            uv, paths, measured, spans = [], [], [], []
            for annotation in annotations:
                points = np.asarray(annotation["uv"], float)
                start = len(uv); count = len(points)
                active = np.asarray(active_source_supervision_mask(annotation), bool)
                visible = np.asarray(annotation.get("source_visible_mask", [True]*count))
                ids = np.asarray(annotation.get("segment_ids", np.zeros(count, int)))
                if visible.dtype != bool or visible.shape != active.shape or ids.shape != active.shape:
                    raise ValueError("Source integration needs explicit parallel visibility and segment IDs")
                keep = active & visible
                path = []
                for i in range(count):
                    if path and (not keep[i] or ids[i] != ids[i-1]):
                        paths.append(path); path = []
                    if keep[i]: path.append(start+i)
                if path: paths.append(path)
                uv.extend(points.tolist()); measured.extend(keep.tolist())
                spans.append((annotation, start, start+count))
            if not any(measured):
                continue
            weights = source_support_integration_weights(uv, paths, measured_mask=np.asarray(measured, bool))
            for annotation, start, end in spans:
                annotation["source_integration_weights"] = weights[start:end].tolist()
                annotation["source_integration_group"] = f"{kind}:{identity}"
                annotation["source_integration_basis"] = "visible_original_path_length_duplicate_UV_shared_one_physical_curve_budget"
    return result


def source_weighted_mean(values, annotation, mask=None):
    """Return loss mean plus its source measure for same-curve reduction."""
    import torch
    if annotation.get("source_integration_weights") is None:
        return values.mean(), 1.
    weights = torch.as_tensor(annotation["source_integration_weights"], device=values.device, dtype=values.dtype)
    if mask is not None:
        weights = weights[torch.as_tensor(mask, device=values.device, dtype=torch.bool)]
    if weights.shape != values.shape or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError("Loss integration weights must follow actual original source positions")
    mass = weights.sum()
    if not bool(mass > 0):
        raise ValueError("Active source loss must have positive source support measure")
    return (values*weights).sum()/mass, float(mass.detach())


def reduce_source_group_losses(terms, zero):
    """Terms are (annotation, mean, mass); one budget per declared curve."""
    import torch
    groups = {}
    for annotation, value, mass in terms:
        identity = annotation.get("source_integration_group", annotation["group_id"])
        groups.setdefault(identity, []).append((value, mass))
    reduced = [sum(value*mass for value,mass in rows)/sum(mass for _,mass in rows) for rows in groups.values()]
    return torch.stack(reduced).mean() if reduced else zero


def select_training_authorities(manifest, allowed):
    """Explicit curriculum selection; keep labels and their authority intact.

    This does not declare excluded readings wrong or delete them. In particular,
    an unvalidated propagation cannot outvote direct human source corrections.
    Reviewed material regions have their own authority and remain separate.
    """
    choices = {"direct_user_confirmed_native_UV", "assistant_original_source_interval"}
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
            or not set(allowed) <= choices or "direct_user_confirmed_native_UV" not in allowed):
        raise ValueError("Training authority selection must preserve direct user supervision")
    result = deepcopy(manifest)
    excluded = []
    for sample in result["samples"]:
        for annotation in sample.get("annotations", []):
            role = annotation_role(annotation, sample)
            if role["semantic_supervision_enabled"] and role["authority_role"] not in allowed:
                excluded.append(dict(source_sample_key=list(source_sample_key(sample)),
                    group_id=annotation["group_id"], original_annotation_sha256=annotation_revision_sha256(annotation),
                    original_state=annotation.get("state"), authority_role=role["authority_role"]))
                annotation["original_state_before_authority_curriculum"] = annotation["state"]
                annotation["state"] = "unreviewed"
                annotation["training_curriculum_excluded"] = True
                annotation["vision_supervision"] = annotation_role(annotation, sample)
    result["training_authority_selection"] = dict(allowed_authorities=allowed,
        excluded_groups=excluded, excluded_readings_declared_wrong=False,
        original_files_or_UV_modified=False, reviewed_material_regions_retained=True)
    return result


def validate_boundary_definition(definition):
    """Check the user's distinction, without confirming assistant pixel UVs."""
    roles = definition["definition"]
    if (roles.get("primary_visual_boundary") != "actual_printed_graphic_outer_contours"
            or roles.get("visible_rectangle_required") is not False
            or definition["authority"].get("kind") != "direct_user_confirmation"):
        raise ValueError("Printed-boundary roles need the actual human confirmation record")
    if definition["authority_scope"].get("assistant_extracted_pixel_precision_confirmed_by_this_reply") is not False:
        raise ValueError("Semantic role confirmation cannot automatically approve extracted pixel precision")


def _containing_shot(intervals, pts):
    rows = [row for row in intervals if row.get("interval_type", "shot") == "shot"
            and int(row["start_pts"]) <= int(pts) < int(row["end_pts"])]
    if len(rows) != 1:
        raise ValueError(f"Source PTS {pts} needs one complete shot, found {len(rows)}")
    return rows[0]


def build_shot_plan(samples, timeline, *, evaluation_pts=(262901760,),
                    development_pts=(228372480, 36188160)):
    """Whole native shots and actual source SHA define splits, never URLs.

    The held development drawing was already used to diagnose training errors.
    It stays a development split, separate from complete independent shots.
    """
    digest = timeline["source_sha256"]
    intervals = timeline["intervals"]
    independent = {(_containing_shot(intervals, pts)["start_pts"],
                    _containing_shot(intervals, pts)["end_pts"]) for pts in evaluation_pts}
    development = {(_containing_shot(intervals, pts)["start_pts"],
                    _containing_shot(intervals, pts)["end_pts"]) for pts in development_pts}
    if independent & development:
        raise ValueError("A whole shot cannot be both development and independent evaluation")
    selected = independent | development
    for sample in samples:
        if sample["source_hash"] != digest:
            continue  # Other source hashes require their own native timeline.
        shot = _containing_shot(intervals, sample["source_pts"])
        selected.add((shot["start_pts"], shot["end_pts"]))
    shots = []
    for start, end in sorted(selected):
        row = deepcopy(_containing_shot(intervals, start))
        split = ("independent_evaluation" if (start, end) in independent else
                 "development" if (start, end) in development else "train")
        row.update(source_sha256=digest, time_base=timeline["time_base"], split=split,
                   content_group_key=f"{digest}:{start}:{end}",
                   start_time=source_time(start, timeline["time_base"]),
                   end_time_exclusive=source_time(end, timeline["time_base"]),
                   whole_shot=True, propagation_across_cuts_allowed=False)
        shots.append(row)
    return dict(source_sha256=digest, shots=shots, input_url_used_for_split=False,
                exact_RGB_duplicates_across_splits_allowed=False)


def split_indices(samples, plan):
    """Reject raw RGB duplication across splits, including alternate URLs."""
    indices = dict(train=[], development=[], independent_evaluation=[], pending_source_timeline=[])
    content_splits = {}
    for index, sample in enumerate(samples):
        matching = [row for row in plan["shots"] if row["source_sha256"] == sample["source_hash"]
                    and row["start_pts"] <= sample["source_pts"] < row["end_pts"]]
        if len(matching) > 1:
            raise ValueError("Native shot split intervals overlap")
        if matching and sample.get("time_base") and Fraction(sample["time_base"]) != Fraction(matching[0]["time_base"]):
            raise ValueError("Source label and shot split native clocks differ")
        split = matching[0]["split"] if matching else "pending_source_timeline"
        indices[split].append(index)
        pixel_digest = sample.get("decoded_native_RGB_sha256") or sample.get("native_RGB_sha256")
        if pixel_digest:
            previous = content_splits.setdefault(pixel_digest, split)
            if previous != split:
                raise ValueError("Identical native RGB content appears across different splits")
    return indices


def shot_sampling_groups(samples, plan, training_indices=None):
    """One sampling unit per real source/shot, regardless of nearby-frame count.

    The caller draws a shot uniformly, then an available human/RGB pool
    uniformly, then a frame uniformly. Mixed frames may occur in both pools;
    their authoritative human labels always survive. No model confidence or
    amount of propagated data determines the shot's sampling probability.
    """
    if training_indices is None:
        training_indices = split_indices(samples, plan)["train"]
    grouped = {}
    for index in training_indices:
        sample = samples[index]
        matching = [row for row in plan["shots"] if row["split"] == "train"
                    and row["source_sha256"] == sample["source_hash"]
                    and row["start_pts"] <= sample["source_pts"] < row["end_pts"]]
        if len(matching) != 1:
            raise ValueError("Training sampling requires one verified training shot")
        shot = matching[0]
        key = shot["content_group_key"]
        group = grouped.setdefault(key, dict(
            shot_key=key, source_sha256=shot["source_sha256"], start_pts=shot["start_pts"], end_pts=shot["end_pts"],
            all_indices=[], human_indices=[], source_RGB_indices=[]))
        group["all_indices"].append(index)
        roles = [annotation.get("vision_supervision") or annotation_role(annotation, sample)
                 for annotation in sample.get("annotations", [])]
        if any(role["authority_role"] == "direct_user_confirmed_native_UV"
               and role["semantic_supervision_enabled"] for role in roles):
            group["human_indices"].append(index)
        if (any(role["authority_role"] == "assistant_original_source_interval"
                and role["semantic_supervision_enabled"] for role in roles)
                or any(region.get("state") in (None, "source_reviewed_region") and region.get("authority")
                       for region in sample.get("regions", []))):
            group["source_RGB_indices"].append(index)
    for group in grouped.values():
        group["sampling_pools"] = [pool for pool in (group["human_indices"], group["source_RGB_indices"]) if pool]
        if not group["sampling_pools"]:
            raise ValueError("Training shot has no authoritative human or source-RGB supervision")
        group["uniform_shot_weight"] = 1.0 / len(grouped)
    return list(grouped.values())


def balanced_source_selection(groups, state, rng):
    """Exact long-run uniform shot/pool/frame policy with bounded starvation.

    Counts refer only to this sampler's actual selections. Point density and
    model residuals cannot alter the allocation. State is checkpointed with RNG.
    """
    layout=[dict(shot_key=g['shot_key'],pools=g['sampling_pools']) for g in groups]
    digest=annotation_revision_sha256(layout)
    if not state:
        state.update(layout_sha256=digest,shots={},pools={},frames={})
    if state.get('layout_sha256')!=digest:
        raise ValueError('Balanced source sampling cannot silently change its shot/pool layout')
    def choose(items,counts):
        least=min(counts.get(key,0) for key,_ in items)
        selected=rng.choice([(key,value) for key,value in items if counts.get(key,0)==least])
        counts[selected[0]]=counts.get(selected[0],0)+1
        return selected
    key,group=choose([(g['shot_key'],g) for g in groups],state['shots'])
    pool_key,pool=choose([(f'{key}|{i}',p) for i,p in enumerate(group['sampling_pools'])],state['pools'])
    _,index=choose([(f'{pool_key}|{i}',i) for i in pool],state['frames'])
    return group,index


def native_shot_scope(sample, plan):
    """Explicit half-open native PTS bounds for neighbour decoding at any FPS."""
    rows = [row for row in plan["shots"] if row["source_sha256"] == sample["source_hash"]
            and row["start_pts"] <= int(sample["source_pts"]) < row["end_pts"]]
    if len(rows) != 1:
        raise ValueError("Source neighbour decoding needs one verified native shot")
    shot = rows[0]
    if Fraction(sample["time_base"]) != Fraction(shot["time_base"]):
        raise ValueError("Source label and neighbour shot scope use different native clocks")
    return dict(source_sha256=shot["source_sha256"], time_base=shot["time_base"],
                start_pts=int(shot["start_pts"]), end_pts_exclusive=int(shot["end_pts"]),
                shot_id=shot["id"], split=shot["split"], content_group_key=shot["content_group_key"],
                classification_authority=shot.get("authority"), review_status=shot.get("review_status"),
                boundary_precision_pts=shot.get("boundary_precision_pts"),
                exposure_alias_proof_is_separate=True)


def attach_native_shot_scopes(manifest, plan):
    """Return source-scoped samples for the existing sequence-preparation path."""
    result = deepcopy(manifest)
    for sample in result["samples"]:
        scope = native_shot_scope(sample, plan)
        if sample.get("source_shot_id") not in (None, scope["shot_id"]):
            raise ValueError("Existing sample shot identity conflicts with native classification")
        sample["source_shot_scope"] = scope
    return result


def validate_sequence_shot_scope(sample, sequence_row, plan=None):
    """Reject cross-cut/cache/source leakage; never replace a neighbour by guess."""
    scope = sample.get("source_shot_scope")
    if scope is None:
        if plan is None:
            raise ValueError("Source sequence lacks its explicit native shot scope")
        scope = native_shot_scope(sample, plan)
    if (sample["source_hash"] != scope["source_sha256"]
            or Fraction(sample["time_base"]) != Fraction(scope["time_base"])):
        raise ValueError("Cached neighbour shot scope differs from the annotated source/clock")
    if (sequence_row.get("source_video_sha256") != scope["source_sha256"]
            or Fraction(sequence_row["time_base"]) != Fraction(scope["time_base"])):
        raise ValueError("Neighbour sequence and shot scope have different source/clock identities")
    pts = sequence_row["native_pts"]
    if len(pts) != 3 or int(pts[1]) != int(sample["source_pts"]):
        raise ValueError("Source context must retain the exact annotated middle native PTS")
    if not all(scope["start_pts"] <= int(stamp) < scope["end_pts_exclusive"] for stamp in pts):
        raise ValueError("Source context contains a neighbour across its native shot cut")
    if not int(pts[0]) <= int(pts[1]) <= int(pts[2]):
        raise ValueError("Native neighbour PTS is reversed")
    return scope


def measured_interval_annotation(guide, reading, fb_valid, *, source_pts, authority):
    """Convert actual RGB read locations to source intervals, never flow UVs.

    ``reading`` uses the existing reader's accepted locations and the original
    profile point indices. Human UV remains in the untouched anchor file. RGB
    support is translated from guide-relative to measured-location-relative
    coordinates; this is an engineering support, not calibrated pixel truth.
    """
    import numpy as np
    indices = np.asarray(reading["source_indices"], dtype=int)
    actual = np.asarray(reading["actual_source_uv"], dtype=float).reshape(-1, 2)
    transported = np.asarray(reading["guide"]["uv"], dtype=float)
    normals = np.asarray(reading["guide"]["normals"], dtype=float)
    supports = np.asarray(reading["support"]["normal_interval_px"], dtype=float)
    fb_valid = np.asarray(fb_valid, dtype=bool)
    if len(indices) != len(actual) or len(transported) != len(fb_valid):
        raise ValueError("Temporal evidence must retain its original profile point axis")
    keep = fb_valid[indices] & np.isfinite(actual).all(1)
    indices, actual = indices[keep], actual[keep]
    if not len(indices):
        return None
    selected_normals = normals[indices]
    offset = ((actual - transported[indices]) * selected_normals).sum(1)
    relative = supports[indices] - offset[:, None]
    if not np.isfinite(relative).all() or np.any(relative[:, 0] > relative[:, 1]):
        raise ValueError("Measured RGB location support is not finite and ordered")
    return dict(
        group_id=f"temporal_RGB_{guide['source_group_id']}_pts{source_pts}",
        feature_id=guide["id"], kind=guide["kind"], primitive=guide["primitive"],
        physical_role=guide["physical_role"], uv=actual.tolist(), normal=selected_normals.tolist(),
        interval_px=relative.tolist(), state="source_reviewed_interval", authority=authority,
        identity_authority=guide["identity_authority"], source_identity_verified=True,
        source_anchor_group_id=guide["source_group_id"], source_anchor_point_indices=indices.tolist(),
        measurement_method=reading["support"]["method"],
        propagated_search_UV_used_as_labels=False, model_projection_used_to_generate_or_select_uv=False,
        actual_source_RGB_locations_used=True, pixel_ground_truth=False,
        is_calibrated_uncertainty=False, new_human_positions_created=False,
        propagation_FB_valid_at_measured_indices=fb_valid[indices].tolist(),
        source_semantic_identity="inherited_from_confirmed_human_anchor_within_same_native_shot",
    )


def export_temporal_supervision(config_path, source_packet_path=None, output_path=None):
    """Reuse saved same-shot flow hints; reread every available original RGB.

    GPU only samples native profiles; no network, flow model or scene solver is
    loaded. Readable observed source intervals are training supervision. Failed
    profiles remain candidates with explicit reasons and representative images.
    A streamed Data NPZ stores lossless RGB; existing ``_image(path, frame_key)``
    and ``frame_neighbours(source_video_path, sample)`` consume these samples.
    """
    import av
    import numpy as np
    import torch
    import zipfile
    from PIL import Image
    from .storage import AssetStore
    from .vision import _source_json
    from .vision_geometry import guided_source_reading, source_annotations_to_guides

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    store = AssetStore(config_path.parents[1])
    data = Path(config["data_directory"])
    packet_path = Path(source_packet_path) if source_packet_path else data / "source_reliable_exposures.json"
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(packet_bytes.decode("utf-8-sig"))
    packet_sha = hashlib.sha256(packet_bytes).hexdigest()
    anchor_path = data / "user_complete_frame_source.json"
    anchor_bytes = anchor_path.read_bytes()
    anchor = json.loads(anchor_bytes.decode("utf-8-sig"))["sample_confirmed"]
    anchor_sha = hashlib.sha256(anchor_bytes).hexdigest()
    if anchor["source_pts"] != packet["reference_native_pts"]:
        raise ValueError("Saved temporal propagation belongs to another human anchor")
    video = store.source(packet["source_video_path"])
    digest = file_sha256(video)
    if digest != packet["source_video_sha256"] or digest != anchor["source_hash"]:
        raise ValueError("Saved temporal RGB and human anchor source identities differ")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native temporal source profiling requires the configured CUDA device")
    shot = packet["classification"]["interval"]
    frame_records = {int(row["source_pts"]): row for row in packet["frames"]}
    frame_packet_indices = {int(row["source_pts"]): index for index, row in enumerate(packet["frames"])}
    if any(not shot["start_pts"] <= pts < shot["end_pts"] for pts in frame_records):
        raise ValueError("Temporal supervision cannot cross the classified source cut")
    clock = packet["time_base"]
    h, w = anchor["frame_size_wh"][1], anchor["frame_size_wh"][0]
    classes = tuple(config["classes"])
    # Uniform normalized probabilities have no visual or geometric semantics.
    # Existing reader treats type values as diagnostics, never RGB vetoes.
    probabilities = torch.full((1, len(classes), h, w), 1.0 / len(classes), device=device)
    definition_path = data / "paint_role_contract_current.json"
    definitions = json.loads(definition_path.read_text(encoding="utf-8-sig"))
    role_map = {row["curve_id"]: row for row in definitions["curve_records"]}
    # This named gray-disc material transition is visible in the unannotated
    # anchor image; the source pattern is a filled disk, not a painted axis.
    if "F1:outer" in role_map:
        role_map["F1:outer"] = {**role_map["F1:outer"],
                                "source_material_transition": "gray_disc_vs_white_ice",
                                "source_material_authority": "original_native_anchor_RGB_review_20261006"}
    known_annotations = {row["group_id"]: row for row in anchor["annotations"]}
    original_anchor_indices = {row["group_id"]: index for index, row in enumerate(anchor["annotations"])}
    output = Path(output_path) if output_path else data / "temporal_supervision_samples.json"
    bundle = data / "temporal_source_RGB.npz"
    bundle_temporary = bundle.with_suffix(".npz.tmp")
    samples, candidates, representatives, uncertainties = [], [], {}, {}
    authority = dict(path=str(packet_path), sha256=packet_sha,
                     source_anchor_path=str(anchor_path), source_anchor_sha256=anchor_sha,
                     scope="Human-confirmed physical identity plus readable original RGB transitions in the same shot")
    decoded, measured, accepted_points = 0, 0, 0
    with zipfile.ZipFile(bundle_temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            if str(stream.time_base) != clock:
                raise ValueError("Native source clock changed")
            container.seek(min(frame_records), stream=stream, backward=True)
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None or pts < min(frame_records):
                    continue
                if pts > max(frame_records):
                    break
                if pts not in frame_records:
                    continue
                decoded += 1
                record = frame_records[pts]
                rgb = frame.to_ndarray(format="rgb24")
                rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgb_sha != record["decoded_native_RGB_sha256"] or rgb.shape != (h, w, 3):
                    raise ValueError(f"Original RGB differs from saved temporal evidence at PTS {pts}")
                if pts == anchor["source_pts"]:
                    continue  # Preserve the existing human coordinates and labels as the anchor.
                binding = dict(source_sha256=digest, source_pts=pts, time_base=clock, native_hw=[h, w],
                               shot_id=shot["id"], image_domain="native_brown_pixels")
                source_annotations = []
                groups = {}
                for group in record["groups"]:
                    parent_id = group["group_id"]
                    if parent_id not in known_annotations:
                        raise ValueError("Propagation group has no confirmed human source identity")
                    original = known_annotations[parent_id]
                    current = {**deepcopy(original), "uv":group["transported_search_uv_only"],
                               "normal":group["transported_search_normal_only"],
                               "state":"source_annotation_search_hint", "source_identity_verified":True,
                               "authority":authority}
                    source_annotations.append(current)
                    groups[parent_id] = group
                adapted = source_annotations_to_guides(source_annotations, source_binding=binding,
                    annotation_source_binding=binding, role_definitions=role_map,
                    verified_identity_group_ids=tuple(groups), classes=classes)
                guides = adapted["guides"]
                for guide in guides:
                    guide.update(origin="temporal_source_search_hint", identity_confirmed=True,
                                 identity_authority=known_annotations[guide["source_group_id"]]["authority"])
                tensor = torch.from_numpy(rgb.copy()).to(device=device).permute(2, 0, 1)[None].float() / 255
                with torch.inference_mode():
                    result = guided_source_reading(tensor, probabilities, guides, classes=classes,
                        rgb_source_binding=binding, probability_source_binding=binding,
                        guide_source_binding=binding, probabilities_conditioned_on_geometry=False,
                        search_radius_px=packet["profile"]["search_radius_native_px"],
                        sample_step_px=packet["profile"]["sample_step_native_px"])
                annotations, frame_groups = [], []
                for guide, reading in zip(guides, result["readings"]):
                    clean = _source_json(reading)
                    group = groups[guide["source_group_id"]]
                    annotation = measured_interval_annotation(guide, clean, group["pointwise"]["FB_valid"],
                                                              source_pts=pts, authority=authority)
                    accepted = len(annotation["uv"]) if annotation else 0
                    if annotation:
                        annotation.update(propagation_source_pointer=f"/frames/{frame_packet_indices[pts]}/groups/{list(groups).index(guide['source_group_id'])}",
                                          original_human_annotation_pointer=f"/sample_confirmed/annotations/{original_anchor_indices[guide['source_group_id']]}",
                                          propagation_FB_error_summary=group["FB_error_native_px"],
                                          individual_FB_errors_available=False)
                        annotation["source_RGB_quality"] = dict(
                            engineering_weights=[clean["source_quality_weights"][i]
                                                 for i in annotation["source_anchor_point_indices"]],
                            weights_are_model_confidence=False, quality_is_calibrated_accuracy=False)
                        annotations.append(annotation)
                    flags = {name:int(np.asarray(value, dtype=bool).sum()) for name, value in clean["flags"].items()}
                    frame_groups.append(dict(group_id=guide["source_group_id"], physical_id=guide["id"],
                        readable_training_points=accepted, search_point_count=len(guide["points_uv"]),
                        FB_valid_points=int(np.asarray(group["pointwise"]["FB_valid"]).sum()),
                        source_RGB_flags=flags, original_flow_search_is_label=False,
                        RGB_measurement_method=clean["support"]["method"], pixel_ground_truth=False))
                    if accepted > representatives.get(guide["source_group_id"], {}).get("readable_points", -1):
                        representatives[guide["source_group_id"]] = dict(source_pts=pts, readable_points=accepted,
                                                                      source_time=source_time(pts, clock), rgb=rgb.copy())
                    ambiguity_flags = {name:value for name, value in clean["flags"].items()
                                       if "competing" in name or "multi_edge" in name or "wide" in name or "blur" in name}
                    affected = sorted({i for values in ambiguity_flags.values()
                                       for i, bad in enumerate(values) if bad and group["pointwise"]["FB_valid"][i]})
                    if len(affected) > uncertainties.get(guide["source_group_id"], {}).get("unreadable_points", 0):
                        uncertainties[guide["source_group_id"]] = dict(source_pts=pts,
                            source_time=source_time(pts, clock), physical_id=guide["id"], unreadable_points=len(affected),
                            source_anchor_point_indices=affected, actual_RGB_flags=ambiguity_flags,
                            question_candidate="These known-identity locations have ambiguous/wide RGB evidence; inspect original and temporal neighbours before asking the user which visible border belongs to the feature.",
                            automatic_training_target_created_at_these_points=False, rgb=rgb.copy())
                candidates.append(dict(source_pts=pts, source_time=source_time(pts, clock),
                                       groups=frame_groups, unresolved_roles=adapted["unresolved"]))
                measured += 1
                if annotations:
                    key = f"source_pts_{pts}"
                    with archive.open(f"{key}.npy", "w") as member:
                        np.lib.format.write_array(member, rgb, allow_pickle=False)
                    samples.append(dict(source_pts=pts, time_base=clock, source_hash=digest,
                        source_video_path=str(video), source_time_seconds=float(pts * Fraction(clock)),
                        frame_path=str(bundle), frame_key=key, decoded_native_RGB_sha256=rgb_sha,
                        frame_size_wh=[w, h], annotations=annotations, ignored_regions=[], pixel_ground_truth=False,
                        source_shot_id=shot["id"], source_split="train", authority="confirmed_human_identity_and_actual_source_RGB_interval"))
                    accepted_points += sum(len(row["uv"]) for row in annotations)
                if measured % 30 == 0:
                    print(json.dumps(dict(temporal_frames_measured=measured, training_samples=len(samples),
                                          actual_RGB_training_points=accepted_points)), flush=True)
    if decoded != len(frame_records):
        raise ValueError("Some existing same-shot native frames were not decoded")
    if not samples:
        raise ValueError("No readable source intervals were measured; candidates cannot count as module progress")
    if file_sha256(packet_path) != packet_sha or file_sha256(anchor_path) != anchor_sha:
        raise ValueError("Source inputs changed during temporal export")
    bundle_temporary.replace(bundle)
    bundle_sha = file_sha256(bundle)
    for sample in samples:
        sample["frame_sha256"] = bundle_sha
    representative_images = []
    for group_id, representative in representatives.items():
        path = data / f"temporal_source_{representative['source_pts']}.png"
        if not path.exists():
            Image.fromarray(representative.pop("rgb")).save(path)
        else:
            representative.pop("rgb")
        representative_images.append(dict(group_id=group_id, original_frame_path=str(path), **representative))
    representative_uncertainties = []
    for group_id, representative in uncertainties.items():
        path = data / f"temporal_source_{representative['source_pts']}.png"
        if not path.exists():
            Image.fromarray(representative.pop("rgb")).save(path)
        else:
            representative.pop("rgb")
        representative_uncertainties.append(dict(group_id=group_id, original_frame_path=str(path), **representative))
    report = dict(schema="rinkspace.source-temporal-supervision.v1", samples=samples, sample_updates=[],
        source_video_path=str(video), source_sha256=digest, source_packet_path=str(packet_path),
        source_packet_sha256=packet_sha, human_anchor_path=str(anchor_path), human_anchor_sha256=anchor_sha,
        decoded_same_shot_native_frames=decoded, measured_non_anchor_frames=measured,
        actual_RGB_training_points=accepted_points, representative_original_images=representative_images,
        representative_uncertainties=representative_uncertainties,
        candidates=candidates, role_definition_path=str(definition_path), role_definition_sha256=file_sha256(definition_path),
        protocol="frame_path=lossless_RGB_npz; frame_key=native_member; frame_sha256=container_SHA; decoded_native_RGB_sha256=member_RGB_SHA",
        three_frame_context="Existing frame_neighbours reads original video around source_pts; native clock/alias proof remains source-specific",
        original_human_coordinates_modified=False, flow_UV_used_as_labels=False, model_or_camera_projection_used=False,
        identity_inherited_from_human=True, source_RGB_support_is_calibrated_uncertainty=False,
        pixel_ground_truth=False, full_shot_accuracy_accepted=False, all_unknown_counted_as_success=False,
        required_new_user_questions=[])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def source_sequence_intervals(manifest):
    """Validate an exact native sequence without deriving camera identities.

    Programme completion is the requested source range, including all its
    supplied intervals. A completed two-frame diagnostic is not completion of
    this range. Camera authority and native cuts come from the manifest's
    immutable source packet, rather than a different event timeline overlay.
    """
    digest, clock, start = source_sample_key(dict(
        source_sha256=manifest.get("source_sha256"), time_base=manifest.get("time_base"),
        source_pts=manifest.get("start_pts")))
    end, step = manifest.get("end_pts_exclusive"), manifest.get("native_step_pts")
    if (isinstance(end, bool) or not isinstance(end, Integral) or end <= start or
            isinstance(step, bool) or not isinstance(step, Integral) or step <= 0 or
            (end - start) % step):
        raise ValueError("A source sequence requires an exact positive native range and step")
    if manifest.get("frame_count") != (end - start) // step:
        raise ValueError("The requested source sequence frame count differs from its native range")
    intervals = manifest.get("source_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("A complete source sequence requires all classified intervals")
    cursor, identities = start, set()
    for interval in intervals:
        identity = interval.get("id")
        first, last = interval.get("start_pts"), interval.get("end_pts_exclusive")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Source sequence interval identities must be present and unique")
        if (isinstance(first, bool) or not isinstance(first, Integral) or first != cursor or
                isinstance(last, bool) or not isinstance(last, Integral) or last <= first or
                last > end or (last - first) % step):
            raise ValueError("Source sequence intervals must partition every requested native frame")
        if interval.get("native_frame_count") != (last - first) // step:
            raise ValueError("A source interval frame count differs from its native range")
        if (interval.get("source_sha256", digest) != digest or
                str(Fraction(interval.get("time_base", clock))) != clock):
            raise ValueError("A source interval has a different original source or native clock")
        if not interval.get("camera_authority"):
            raise ValueError("Source sequence camera authority must be retained, including uncertainty")
        identities.add(identity)
        cursor = last
    if cursor != end:
        raise ValueError("Source sequence intervals leave part of the requested range uncovered")
    return intervals


def source_sequence_coverage_audit(manifest, review_summary, *, observations=None,
                                 exposure_groups=None, installation_aliases=None,
                                 source_seed_policy=None):
    """Check one complete programme run, consuming source rows exactly once.

    Rows may be hydrated by ``vision_storage`` by the caller for subsequent
    quality checks; this coverage audit needs only their original metadata.
    Passing a summary alone cannot establish actual row coverage. Counts and
    completion never establish pixel accuracy or complete physical identity.
    """
    intervals = source_sequence_intervals(manifest)
    digest, clock, start = source_sample_key(dict(source_sha256=manifest["source_sha256"],
        time_base=manifest["time_base"], source_pts=manifest["start_pts"]))
    end, step = manifest["end_pts_exclusive"], manifest["native_step_pts"]
    failures, examples = Counter(), []
    allowed_seed_keys = set()
    if source_seed_policy is not None:
        if (source_seed_policy.get("schema") != "rinkspace.explicit-assisted-source-seed-policy.v1" or
                source_seed_policy.get("mode") != "assisted" or
                source_seed_policy.get("source_sha256") != digest or
                str(Fraction(source_seed_policy.get("time_base", "0"))) != clock or
                source_seed_policy.get("human_coordinates_use") != "search_and_identity_only_not_output" or
                source_seed_policy.get("assisted_calibration_not_independent_validation") is not True):
            raise ValueError("Assisted source seeds require an explicit same-source calibration policy")
        seed_sha = source_seed_policy.get("seed_packet_sha256")
        if not isinstance(seed_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", seed_sha):
            raise ValueError("Assisted source seed policy needs the authenticated packet SHA")
        for binding in source_seed_policy.get("allowed_source_bindings", ()):
            key = source_sample_key(binding)
            if (key[:2] != (digest, clock) or not start <= key[2] < end or
                    (key[2] - start) % step or key in allowed_seed_keys):
                raise ValueError("Assisted seed policy has duplicate or non-native source positions")
            allowed_seed_keys.add(key)
        if not allowed_seed_keys:
            raise ValueError("Assisted seed policy needs explicit native source positions")
    installation_names = {}
    if installation_aliases is not None:
        if (installation_aliases.get("source_sha256") != digest or
                str(Fraction(installation_aliases.get("time_base", "0"))) != clock):
            raise ValueError("Installation aliases belong to another source or native clock")
        interval_by_id = {interval["id"]: interval for interval in intervals}
        for alias in installation_aliases.get("mappings", []):
            shot = interval_by_id.get(alias.get("shot_id"))
            if (shot is None or alias.get("camera_id") != shot.get("camera_id") or
                    not alias.get("installation_id") or not alias.get("authority_reference")):
                raise ValueError("An installation alias must explicitly retain its source shot, taxonomy and authority")
            installation_names.setdefault(shot["id"], set()).add(alias["installation_id"])

    def fail(reason, **evidence):
        failures[reason] += 1
        if len(examples) < 32:
            examples.append(dict(reason=reason, **evidence))

    if review_summary.get("source_sha256") != digest:
        fail("review_source_SHA_differs")
    if (review_summary.get("classification_sha256") !=
            manifest.get("primary_source_intervals", {}).get("sha256")):
        fail("review_primary_source_intervals_SHA_differs")
    reported = review_summary.get("shots", [])
    if len(reported) != len(intervals):
        fail("review_does_not_include_all_requested_intervals", expected=len(intervals), actual=len(reported))
    for expected, actual in zip(intervals, reported):
        fields = dict(id=expected["id"], start_pts=expected["start_pts"],
            end_pts_exclusive=expected["end_pts_exclusive"], camera_id=expected.get("camera_id"),
            camera_authority=expected["camera_authority"])
        if any(actual.get(key) != value for key, value in fields.items()):
            fail("review_interval_scope_or_camera_authority_differs", expected=fields)
        if (actual.get("processed_frames") != expected["native_frame_count"] or
                actual.get("first_pts") != expected["start_pts"] or
                actual.get("last_pts") != expected["end_pts_exclusive"] - step or
                actual.get("source_pts_complete") is not True):
            fail("review_interval_native_completion_differs", interval_id=expected["id"])
    if (review_summary.get("frame_count") != manifest["frame_count"] or
            review_summary.get("source_pts_complete") is not True or
            review_summary.get("all_decoded_original_frames_processed") is not True or
            review_summary.get("status") != "source_range_complete"):
        fail("review_summary_is_not_requested_whole_sequence")
    before, after = (review_summary.get("source_code_sha256_at_run_start"),
                     review_summary.get("source_code_sha256_at_run_end"))
    if not before or before != after:
        fail("runtime_code_not_verified_unchanged")
    input_policy_matches = (review_summary.get("independent_mode_has_no_test_annotation_injection") is True
        if source_seed_policy is None else
        review_summary.get("mode") == "assisted" and
        review_summary.get("source_seed_sha256") == source_seed_policy["seed_packet_sha256"] and
        review_summary.get("independent_mode_has_no_test_annotation_injection") is False and
        review_summary.get("actual_RGB_is_only_output_geometry") is True)
    if (review_summary.get("prediction_is_pixel_truth") is not False or
            review_summary.get("accuracy_accepted") is not False or
            review_summary.get("source_board_foot_annotation_pixels_used_as_runtime_input") is not False or
            not input_policy_matches):
        fail("runtime_input_or_accuracy_contract_not_verified")
    model_sha = review_summary.get("model_sha256")
    expected_model = manifest.get("runtime_model", {}).get("sha256")
    if expected_model and model_sha != expected_model:
        fail("runtime_model_differs_from_requested_frozen_weights")

    exposure_map = {}
    if exposure_groups is not None:
        for group in exposure_groups:
            positions = group["full_pts"]
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("Exposure proof group has missing or repeated native positions")
            canonical = min(positions)
            for pts in positions:
                if pts in exposure_map:
                    raise ValueError("Exposure proof assigns a native frame to multiple groups")
                exposure_map[pts] = canonical
        if any(pts not in exposure_map for pts in range(start, end, step)):
            raise ValueError("Provided exposure proof does not cover the complete source sequence")
        for interval in intervals:
            for pts in range(interval["start_pts"], interval["end_pts_exclusive"], step):
                if not interval["start_pts"] <= exposure_map[pts] < interval["end_pts_exclusive"]:
                    raise ValueError("An exposure group crosses the supplied native source cut")

    native_count, interval_index, previous_canonical = 0, 0, None
    interval_counts, new_observations, issue_counts = Counter(), Counter(), Counter()
    reset_count, instance_reset_evidence_count = 0, 0
    if observations is not None:
        for row in observations:
            expected_pts = start + native_count * step
            binding = row.get("source_binding", {})
            try:
                row_source, row_clock, pts = source_sample_key(binding)
            except ValueError:
                fail("invalid_row_source_binding", row=native_count)
                native_count += 1
                continue
            if (row_source, row_clock, pts) != (digest, clock, expected_pts) or pts >= end:
                fail("native_row_outside_exact_ordered_sequence", row=native_count,
                     expected_pts=expected_pts, actual_source_pts=pts)
            while interval_index < len(intervals) - 1 and pts >= intervals[interval_index]["end_pts_exclusive"]:
                interval_index += 1
                previous_canonical = None
            interval = intervals[interval_index]
            interval_counts[interval["id"]] += 1
            first = pts == interval["start_pts"]
            allowed_installations = {(interval.get("camera_id") or "unclassified_installation")}
            allowed_installations.update(installation_names.get(interval["id"], ()))
            if (binding.get("shot_id") != interval["id"] or
                    binding.get("installation_id") not in allowed_installations):
                fail("native_row_cut_or_camera_binding_differs", source_pts=pts)
            if row.get("model_sha256") != model_sha:
                fail("native_row_model_weights_differ", source_pts=pts)
            if source_seed_policy is None:
                if row.get("source_seed_injected") is not False:
                    fail("native_row_human_seed_not_excluded", source_pts=pts)
            elif row.get("source_seed_injected") is not ((row_source, row_clock, pts) in allowed_seed_keys):
                fail("native_row_source_seed_differs_from_explicit_calibration_policy", source_pts=pts)
            if first:
                reset_count += 1
                if row.get("memory_update") != "shot_reset":
                    fail("native_first_frame_has_no_source_memory_reset", source_pts=pts)
            elif row.get("memory_update") == "shot_reset":
                fail("unexpected_memory_reset_inside_source_interval", source_pts=pts)
            if row.get("instance_source_shot_reset_requested") is not None:
                instance_reset_evidence_count += 1
                if row["instance_source_shot_reset_requested"] is not first:
                    fail("instance_memory_reset_differs_from_native_cut", source_pts=pts)
            rgb_sha = row.get("RGB_sha256")
            if not isinstance(rgb_sha, str) or len(rgb_sha) != 64:
                fail("native_row_original_RGB_hash_missing", source_pts=pts)
            canonical = binding.get("canonical_exposure_pts", pts)
            if exposure_groups is not None:
                proof_sha = manifest.get("source_clock_evidence", {}).get("exposure_proof", {}).get("sha256")
                if (canonical != exposure_map.get(pts) or
                        binding.get("exposure_proof_sha256") != proof_sha):
                    fail("native_row_exposure_identity_differs_from_registered_proof", source_pts=pts)
            elif canonical != pts:
                fail("native_row_exposure_alias_not_checked_against_proof", source_pts=pts)
            expected_new = first or canonical != previous_canonical
            if row.get("new_observation") is not expected_new:
                fail("source_observation_advance_differs_from_verified_exposure", source_pts=pts)
            if row.get("new_observation") is True:
                new_observations[interval["id"]] += 1
            previous_canonical = canonical
            for issue in row.get("issues", []):
                issue_counts.update(issue.get("reasons", []))
            native_count += 1
        if native_count != manifest["frame_count"]:
            fail("actual_native_rows_do_not_cover_requested_sequence", expected=manifest["frame_count"], actual=native_count)
    rows_verified = observations is not None
    return dict(schema="rinkspace.complete-source-sequence-coverage.v1",
        source_sha256=digest, time_base=clock, start_pts=start, end_pts_exclusive=end,
        expected_native_frame_count=manifest["frame_count"], actual_native_row_count=native_count,
        source_intervals_count=len(intervals), actual_interval_row_counts=dict(interval_counts),
        cut_reset_count=reset_count, instance_reset_evidence_rows=instance_reset_evidence_count,
        source_exposure_proof_checked=exposure_groups is not None,
        explicit_source_installation_aliases_checked=installation_aliases is not None,
        runtime_input_policy="explicit_assisted_calibration" if source_seed_policy else "independent_no_annotation_input",
        source_seed_policy_checked=source_seed_policy is not None,
        assisted_source_seed_positions=sorted(key[2] for key in allowed_seed_keys),
        actual_new_observation_counts=dict(new_observations), actual_issue_counts=dict(issue_counts),
        source_rows_checked=rows_verified, failure_counts=dict(failures), failure_examples=examples,
        failure_examples_are_truncated=sum(failures.values()) > len(examples),
        complete_native_source_sequence_verified=rows_verified and not failures,
        accuracy_accepted=False, independent_accuracy=False,
        physical_identity_and_pixel_accuracy_require_original_RGB_quality_review=True)


def crosscheck_source_observation_groups(reference_groups, measured_groups, *, source_binding):
    """Compare actual same-source identities/roles after visual output exists.

    Callers explicitly normalize legacy identities and physical roles while
    keeping pointers to their unchanged observations. No fuzzy ID matching,
    frame alias inference, projection, or promotion of old assistant points to
    ground truth is performed here. Distances are posterior diagnostics, not
    eligibility thresholds or a replacement solver.
    """
    import numpy as np

    key = source_sample_key(source_binding)
    pool = {}
    unresolved = []
    for index, group in enumerate(measured_groups):
        if source_sample_key(group["source_binding"]) != key:
            raise ValueError("Measured source group belongs to another source frame or clock")
        if (group.get("actual_RGB_measurement") is not True or
                group.get("model_or_world_projection_used_as_UV") is not False):
            raise ValueError("Crosscheck requires original RGB measurements, never projection UV")
        uv = np.asarray(group["uv"])
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError("Measured source UV must be a finite Nx2 native array")
        feature, role = group.get("feature_id"), group.get("geometry_role")
        if not feature or not role:
            unresolved.append(dict(measured_group_index=index, source_point_count=len(uv),
                reason="source_physical_identity_or_boundary_role_unresolved"))
            continue
        pool.setdefault((feature, role), []).append((index, uv))

    comparisons = []
    for reference in reference_groups:
        if source_sample_key(reference["source_binding"]) != key:
            raise ValueError("Legacy reference belongs to another original source frame or clock")
        uv = np.asarray(reference["uv"], dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError("Legacy UV must retain a finite Nx2 source array")
        identity = reference.get("feature_id"), reference.get("geometry_role")
        matches = pool.get(identity, []) if all(identity) else []
        row = dict(reference_group_id=reference["group_id"],
            reference_authority_role=reference["authority_role"],
            reference_source_pointer=reference.get("source_pointer"),
            feature_id=identity[0], geometry_role=identity[1], reference_position_count=len(uv),
            reference_UV_float64_sha256=hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest(),
            original_reference_is_not_automatically_pixel_truth=True,
            identity_and_physical_role_matched=bool(matches),
            measured_group_indices=[index for index, _ in matches])
        if not matches or not any(len(points) for _, points in matches):
            row.update(status="missing_same_identity_and_role_source_measurement",
                       original_positions_modified=False)
            comparisons.append(row)
            continue
        measured = np.concatenate([points for _, points in matches], axis=0).astype(np.float64, copy=False)
        nearest_indices, distances = [], []
        # Bound transient memory independently of total programme length; all
        # original reference points still contribute once to the comparison.
        for begin in range(0, len(uv), 128):
            delta = measured[None] - uv[begin:begin + 128, None]
            squared = np.sum(delta * delta, axis=2)
            indices = squared.argmin(axis=1)
            nearest_indices.extend(indices.tolist())
            distances.extend(np.sqrt(squared[np.arange(len(indices)), indices]).tolist())
        chosen = measured[np.asarray(nearest_indices, dtype=np.int64)]
        row.update(status="posterior_same_source_identity_role_distance",
            matched_reference_point_count=len(uv), nearest_measured_UV=chosen.tolist(),
            nearest_Euclidean_native_pixel_distance=distances,
            original_positions_modified=False, source_positions_used_to_query_network_or_reader=False)
        normal, interval = reference.get("normal"), reference.get("normal_interval")
        if normal is not None or interval is not None:
            if normal is None or interval is None:
                raise ValueError("A legacy normal-interval comparison requires both original fields")
            normal, interval = np.asarray(normal, dtype=np.float64), np.asarray(interval, dtype=np.float64)
            if (normal.shape != uv.shape or interval.shape != uv.shape or
                    not np.isfinite(normal).all() or not np.isfinite(interval).all() or
                    (interval[:, 0] > interval[:, 1]).any() or
                    not np.allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-5, rtol=0)):
                raise ValueError("Legacy normal and interval arrays must preserve exact valid native support")
            signed = np.sum((chosen - uv) * normal, axis=1)
            outside = np.maximum(interval[:, 0] - signed, 0) + np.maximum(signed - interval[:, 1], 0)
            row.update(signed_original_normal_displacement_px=signed.tolist(),
                       original_normal_interval_outside_px=outside.tolist(),
                       normal_interval_is_not_new_GT_or_background=True)
        comparisons.append(row)
    return dict(schema="rinkspace.source-measurement-legacy-crosscheck.v1",
        source_binding=dict(source_binding), comparisons=comparisons,
        unresolved_measured_groups=unresolved, current_original_RGB_required_for_conflict_adjudication=True,
        missing_or_role_conflicting_points_not_fabricated=True, old_source_rows_not_promoted_to_new_FIT=True,
        comparison_performed_after_source_output=True, accuracy_accepted=False)


def build_supervision_index(config_path):
    """Inventory existing files and propose expansion; decode/train nothing."""
    from .storage import AssetStore

    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    store = AssetStore(root)
    fingerprints = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        fingerprints[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode("utf-8-sig"))

    config = read(config_path)
    clock_registry_path = (store.resolve(config["source_clock_registry"], must_exist=True)
                           if config.get("source_clock_registry") else None)
    clock_registry = read(clock_registry_path) if clock_registry_path else None
    data = Path(config["data_directory"])
    original = read(data / "manifest.json")
    supplementary_path = data / "supplementary_samples.json"
    additional = read(supplementary_path) if supplementary_path.exists() else {"samples": []}
    sample_origins = [("manifest.json", index) for index in range(len(original["samples"]))]
    sample_origins += [("supplementary_samples.json", index) for index in range(len(additional.get("samples", [])))]
    samples = deepcopy(original["samples"] + additional.get("samples", []))
    temporal_path = data / "temporal_supervision_samples.json"
    if temporal_path.exists():
        temporal = read(temporal_path)
        sample_origins += [(temporal_path.name, index) for index in range(len(temporal.get("samples", [])))]
        samples.extend(deepcopy(temporal.get("samples", [])))
    boundary_path=data / "user_boundary_supervision_samples.json"
    boundary_packet=read(boundary_path) if boundary_path.exists() else None
    boundary_ledger=[]
    if boundary_packet is not None:
        original_keys={source_sample_key(sample) for sample in samples}
        for index,template in enumerate(boundary_packet.get("samples",[])):
            if source_sample_key(template) not in original_keys:
                sample_origins.append((boundary_path.name,index))
                original_keys.add(source_sample_key(template))
        merged=merge_user_boundary_samples(dict(samples=samples),boundary_packet)
        samples=merged["samples"]
        boundary_ledger=merged["boundary_supervision_revision_ledger"]
        for evidence in boundary_packet.get("source_evidence_files",[]):
            fingerprints[str(Path(evidence["path"]))]=file_sha256(evidence["path"])
    material_path=data / "user_contact_material_region_supervision.json"
    material_packet=read(material_path) if material_path.exists() else None
    if material_packet is not None:
        samples=merge_source_reviewed_regions(dict(samples=samples),material_packet)["samples"]
        for evidence in material_packet.get("source_evidence_files",[]):
            fingerprints[str(Path(evidence["path"]))]=file_sha256(evidence["path"])
    rejection_path=(store.resolve(config["source_semantic_rejections"],must_exist=True)
                    if config.get("source_semantic_rejections") else data / "source_material_semantic_rejections.json")
    rejection_packet=read(rejection_path) if rejection_path.exists() else None
    if rejection_packet is not None:
        samples=merge_source_reviewed_regions(dict(samples=samples),rejection_packet)["samples"]
        for evidence in rejection_packet.get("source_evidence_files",[]):
            fingerprints[str(Path(evidence["path"]))]=file_sha256(evidence["path"])
    # Updates retain their unmodified source roles/interval reviews in the
    # inventory. Numerical interval merging remains the existing trainer's job.
    updates = source_updates_by_key(original["samples"], additional.get("sample_updates", []))
    for sample in samples:
        update = updates.get(source_sample_key(sample), {})
        sample["regions"] = sample.get("regions", []) + update.get("regions", [])
        sample["source_updates"] = update
    samples = apply_supervision_roles({"samples": samples})["samples"]
    video = store.source(config["source_video"])
    digest = file_sha256(video)
    fingerprints[str(video)] = digest
    timeline_path = store.event / "history/legacy_outputs/camera_classification/women_sp_20260923/timeline.json"
    timeline = read(timeline_path)
    if timeline["source_sha256"] != digest:
        raise ValueError("Native timeline differs from the actual training video")
    overlay_reference=native_boundary_overlay_reference(config,boundary_packet,root=root)
    overlay_path=overlay_reference["path"] if overlay_reference else None
    if overlay_reference:
        overlay=read(overlay_path)
        if fingerprints[overlay_path]!=overlay_reference["sha256"]:
            raise ValueError("Selected native boundary overlay changed while building the index")
        timeline=timeline_with_native_boundary_overlay(timeline_path,digest,overlay_path)
    plan = build_shot_plan(samples, timeline,
                          evaluation_pts=config.get("independent_evaluation_source_pts", [262901760]),
                          development_pts=config.get("development_source_pts", [228372480, 36188160]))
    inventory = []
    roles = Counter()
    source_position_counts=Counter()
    for index, sample in enumerate(samples):
        if sample["source_hash"] != digest:
            raise ValueError("Additional source needs its own verified video binding")
        path = store.source(sample["frame_path"])
        actual_frame_sha = fingerprints.get(str(path)) or file_sha256(path)
        if actual_frame_sha != sample["frame_sha256"]:
            raise ValueError(f"Source image changed at PTS {sample['source_pts']}")
        pixel_sha = native_pixel_sha256(path, sample.get("frame_key"))
        if sample.get("decoded_native_RGB_sha256") and sample["decoded_native_RGB_sha256"] != pixel_sha:
            raise ValueError("Native bundle member differs from its bound original RGB")
        sample["native_RGB_sha256"] = pixel_sha
        fingerprints[str(path)] = actual_frame_sha
        source_name, original_index = sample_origins[index]
        source_file = data / source_name
        annotations = []
        for number, annotation in enumerate(sample["annotations"]):
            role = annotation["vision_supervision"]
            roles[role["authority_role"]] += 1
            position_counts=source_supervision_point_counts(annotation)
            source_position_counts.update(position_counts)
            annotations.append(dict(group_id=annotation.get("group_id"), feature_id=annotation.get("feature_id"),
                                    point_count=len(annotation.get("uv", [])), role=role,
                                    **position_counts,
                                    source_pointer=annotation.get("source_revision_pointer",f"/samples/{original_index}/annotations/{number}"),
                                    source_visibility=(validate_source_visibility(annotation,sample)
                                                       if "source_visible_mask" in annotation else None)))
        inventory.append(dict(source_pts=sample["source_pts"], time_base=sample["time_base"],
                              source_time=source_time(sample["source_pts"], sample["time_base"]),
                              source_sha256=sample["source_hash"], frame_path=str(path),
                              frame_key=sample.get("frame_key"),
                              frame_sha256=actual_frame_sha, native_RGB_sha256=pixel_sha,
                              original_annotation_file=str(source_file), annotations=annotations,
                              reviewed_region_count=len(sample.get("regions", [])),
                              reject_only_region_count=sum("excluded_kinds" in region for region in sample.get("regions",[])),
                              source_updates_preserved=bool(sample["source_updates"])))
    indices = split_indices(samples, plan)
    for split, members in indices.items():
        for member in members:
            inventory[member]["split"] = split
    # The two unused source images permit immediate complete-shot review. Their
    # raw RGB hashes also guard against mislabeled aliases of training frames.
    evaluation_images, development_images = [], []
    eval_pts = config.get("independent_evaluation_source_pts", [262901760])
    dev_pts = config.get("development_source_pts", [228372480, 36188160])
    for pts in (*eval_pts, *dev_pts):
        existing = next((row for row in inventory if row["source_pts"] == pts), None)
        path = Path(existing["frame_path"]) if existing else store.event / f"evidence/photo_review_20260923/holdout/women_sp_lower_right_{pts}.png"
        rgb_sha = native_pixel_sha256(path)
        if rgb_sha in {row["native_RGB_sha256"] for row in inventory if row["split"] == "train"}:
            raise ValueError("Evaluation original RGB is already training content")
        fingerprints[str(path)] = file_sha256(path)
        destination = evaluation_images if pts in eval_pts else development_images
        destination.append(dict(source_pts=pts, source_sha256=digest, frame_path=str(path),
                                source_time=source_time(pts, timeline["time_base"]), native_RGB_sha256=rgb_sha,
                                use="whole_shot_representative_not_frame_only_validation"))
    ko = store.event / "media/originals/gnskF9dQwJo.mp4"
    ko_sha = file_sha256(ko)
    fingerprints[str(ko)] = ko_sha
    ko_meta_path = store.event / "history/legacy_outputs/workspace/ko/metadata.json"
    ko_meta = read(ko_meta_path)
    if ko_meta["source_sha256"] != ko_sha or ko_sha == digest:
        raise ValueError("Cross-programme source identity is not verified")
    definition_path = root / "events/ankara_2026/model_constraints/visual_boundary_roles_user_confirmed_20261006.json"
    definition = read(definition_path) if definition_path.exists() else None
    if definition is not None:
        validate_boundary_definition(definition)
    ad_path = data / "source_ads_complete_frame.json"
    ad_packet = read(ad_path)
    reliable_path = data / "source_reliable_exposures.json"
    reliable = read(reliable_path) if reliable_path.exists() else None
    existing_expansion = None
    if reliable is not None:
        if reliable["source_video_sha256"] != digest:
            raise ValueError("Existing temporal RGB candidates belong to another source")
        existing_expansion = dict(
            path=str(reliable_path), reference_native_pts=reliable["reference_native_pts"],
            native_frame_count=reliable["same_shot_measured_native_frame_count"],
            frames=[{key: frame.get(key) for key in (
                "source_pts", "time_base", "decoded_native_RGB_sha256", "source_sha256",
                "actual_exposure_pts", "official_exposure_identity")} for frame in reliable["frames"]],
            clearer_source_candidates={feature: [
                {**deepcopy(row), "source_time": source_time(row["source_pts"], row["time_base"]),
                 "is_training_annotation": False, "pixel_accuracy_confirmed": False}
                for row in ranking[:3]]
                for feature, ranking in reliable["feature_specific_candidate_ranking"].items()},
            pixel_profiles_are="candidate_evidence_not_precise_supervision",
            original_human_reference_frame=next(row["frame_path"] for row in inventory
                                               if row["source_pts"] == reliable["reference_native_pts"]))
    return dict(
        schema="rinkspace.vision-supervision-index.v1", input_fingerprints=fingerprints,
        input_files_modified=False, source_video=str(video), source_sha256=digest,
        source_clock_registry=(dict(path=str(clock_registry_path),
                                    sha256=fingerprints[str(clock_registry_path)],
                                    schema=clock_registry.get("schema"))
                               if clock_registry_path else None),
        samples=inventory, authority_group_counts=dict(roles),
        source_supervision_position_counts=dict(source_position_counts),split_indices=indices, shot_plan=plan,
        user_boundary_revision_packet=(dict(path=str(boundary_path),sha256=fingerprints[str(boundary_path)],
                                             original_records_modified=False,revision_ledger=boundary_ledger)
                                       if boundary_packet is not None else None),
        user_contact_material_region_packet=(dict(path=str(material_path),sha256=fingerprints[str(material_path)],
                                                   original_records_modified=False)
                                            if material_packet is not None else None),
        source_semantic_rejection_packet=(dict(path=str(rejection_path),sha256=fingerprints[str(rejection_path)],
                                              specified_other_positive_kinds=False,original_records_modified=False)
                                         if rejection_packet is not None else None),
        source_boundary_overlay=(dict(**overlay_reference,
                                      camera_taxonomy_and_authority_preserved=True)
                                 if overlay_path else None),
        shot_sampling_groups=shot_sampling_groups(samples, plan, indices["train"]),
        independent_evaluation_images=evaluation_images,
        development_evaluation_images=development_images,
        cross_programme_evaluation=dict(source_path=str(ko), source_sha256=ko_sha, split="independent_evaluation",
                                       duration_seconds=ko_meta["duration_s"], frame_count=ko_meta["frame_count"],
                                       native_time_base=ko_meta["stream_time_base"],
                                       shot_index_path=str(ko_meta_path.parent / "shots.automatic.json"),
                                       shot_boundaries_are="automatic_candidates_require_visual_review",
                                       all_frames_reserved=True, independent_by_URL=False,
                                       exact_RGB_overlap_with_training_checked=False,
                                       cross_event_evaluation=False),
        source_scope="Current local event is Ankara only; no cross-event accuracy claim",
        temporal_expansion=dict(method="same-shot GPU image tracking then actual RGB rereading",
                                target_shots=[row["content_group_key"] for row in plan["shots"] if row["split"] == "train"],
                                all_native_exposures=True, no_camera_projection_labels=True,
                                tracked_pixel_profiles_are="candidates_not_new_ground_truth",
                                occluded_or_unreadable_positions="missing_not_completed"),
        existing_same_shot_extension_index=existing_expansion,
        user_confirmed_printed_boundary_definition=dict(
            authority="direct_user_confirmation_20261006", source_file=str(definition_path),
            source_file_present=definition is not None, original_record=definition,
            scope="Ad21, Ad23 and advertisements with the same unframed-print structure",
            external_contour="actual visible printed pattern contour; source UV unchanged",
            occupied_extent="record separately; never draw a suggested bbox as a source boundary",
            internal_holes="record separately; never merge into the external contour",
            existing_source_supervision_retained=True, pixel_accuracy_automatically_promoted=False),
        required_new_user_questions=[],
        existing_ad_uncertainties=[{**row, "source_pts":ad_packet["source"]["source_pts"],
                                  "source_time":source_time(ad_packet["source"]["source_pts"], ad_packet["source"]["time_base"]),
                                  "original_frame_path":ad_packet["source"]["source_frame_path"],
                                  "action":"seek clearer same-shot evidence; preserve missing pixels"}
                                 for row in ad_packet.get("uncertain_boundary_regions", [])],
        new_training_annotations_created=0, accuracy_accepted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/rink_vision.json"))
    parser.add_argument("--output", help="Defaults to the existing Data vision_training directory")
    parser.add_argument("--export-temporal", action="store_true", help="Reread saved same-shot hints on CUDA and export source intervals")
    parser.add_argument("--source-packet", help="Existing source_reliable_exposures.json, without recomputing flow")
    args = parser.parse_args()
    if args.export_temporal:
        result = export_temporal_supervision(args.config, args.source_packet, args.output)
        print(json.dumps(dict(training_samples=len(result["samples"]),
                              actual_RGB_training_points=result["actual_RGB_training_points"],
                              same_shot_native_frames=result["decoded_same_shot_native_frames"],
                              representative_images=result["representative_original_images"]), ensure_ascii=False))
        return
    result = build_supervision_index(args.config)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    output = Path(args.output) if args.output else Path(config["data_directory"]) / "supervision_index.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(dict(output=str(output), samples=len(result["samples"]),
                          roles=result["authority_group_counts"], split_indices=result["split_indices"],
                          new_training_annotations_created=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
