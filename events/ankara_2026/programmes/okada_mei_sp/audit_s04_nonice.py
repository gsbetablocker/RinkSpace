"""Replay source visibility/identity checks for every retained S04 ad and board.

Writes diagnostics only inside the S04 ice delivery run. Existing source,
advertisement package, native files and the peer fit input are read-only.
"""
from pathlib import Path
import argparse
import collections
import gzip
import itertools
import json
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from review_points_s04 import OUT,ADS,read,write,rows,CurrentNativeGrid
from ice_points_s04 import source_domain,REC
from rinkspace.vision_board_reader import _unexcluded_pixels
from ad_points_s04 import configure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repaired',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(2);grid=CurrentNativeGrid()
    z=dict(np.load(OUT/'points_repaired/point_observations.npz'))
    active=set(z['observation_ids'].tolist());r,t,m,o,s,ats,_=configure()
    ats+=read(ADS/'source_review/additional_source_atlases.json')['atlases']
    doc=read(ADS/'source_review/source_identity_charts.json')
    refs={int(ai):r.reference(ats[int(ai)])for ai in doc['records']if doc['records'][ai]}
    board=collections.defaultdict(list)
    if args.repaired:
        current=OUT/'evidence/nonice_repaired_records.jsonl.gz'
        for row in rows(current):
            if row['family']=='board':board[row['native_pts']].append(row)
        source=(row for row in rows(current)if row['family']=='advertisement')
    else:
        for row in rows(OUT/'evidence/selected_board_records.jsonl.gz'):board[row['native_pts']].append(row)
        source=rows(ADS/'evidence/selected_point_records.jsonl.gz')
    totals=collections.Counter();failures=collections.Counter();frames=[];start=time.monotonic()
    prefix='nonice_repaired'if args.repaired else'nonice'
    destination=OUT/f'evidence/{prefix}_checks.jsonl.gz'
    with gzip.open(destination,'wt',encoding='utf8')as out:
        for index,(pt,iterator)in enumerate(itertools.groupby(source,key=lambda c:c['pts'])):
            measured=[c for c in iterator if args.repaired or c['stable_id']in active]
            original={c['stable_id']:dict(c)for c in measured}
            eligible=[c for c in measured if 'atlas_id'in c['identity']]
            checked,cache=o.frame_check(pt,eligible,doc,refs)
            checked+=[c for c in measured if 'atlas_id'not in c['identity']]
            im,gray,mask=r.current(pt);bp=read(REC/f'dense_board_v1/qualified_frames/{pt}.json')
            graphics,border,_,_=source_domain(im,mask,bp);exclude=(mask>0)|graphics
            top=np.asarray(bp['roles']['board.top']['trace_uv']);top=top[np.argsort(top[:,0])]
            frame_bad=collections.Counter()
            for c in checked:
                flags=[]
                if c['decision']=='not_admitted':flags.append(c['reason'])
                profile=c['source_profile'];normal=np.asarray(profile['normal']);base=np.asarray(profile.get('search_uv',c['uv']))
                support=np.vstack([c['uv'],base+np.asarray(profile['offsets'])[:,None]*normal])
                if not _unexcluded_pixels(exclude,support).all():flags.append('actual_material_transition_support_intersects_current_exclusion')
                x,y=c['uv'];upper=float(np.interp(x,top[:,0],top[:,1]));lower=float(np.interp(x,np.arange(1920),border))
                if not upper<=y<=lower:flags.append('outside_actual_visible_vertical_board_face')
                result=dict(observation_id=c['stable_id'],pts=pt,uv=c['uv'],native_path_id=c['path_id'],family='advertisement',
                    source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],issues=flags,
                    current_observability=c.get('current_observability'),source_identity=original[c['stable_id']]['identity'])
                out.write(json.dumps(result,ensure_ascii=False)+'\n');totals['advertisement']+=1;failures.update(flags);frame_bad.update(flags)
            for c in board[pt]:
                flags=[];ps=c['pixel_support'];support=np.vstack([c['uv'],ps.get('position_interval_endpoints_uv',[c['uv']])])
                if not _unexcluded_pixels(exclude,support).all():flags.append('board_material_support_intersects_current_exclusion')
                if not ps.get('current_pixel_profile_reread'):flags.append('board_missing_current_RGB_profile')
                if not(bp['paired_blue_cap_and_white_board_body_semantics']and bp['yellow_contact_from_current_global_band_not_accumulated_extrapolation']):flags.append('board_material_pair_identity_unresolved')
                result=dict(observation_id='board:'+c['source_row_id'],pts=pt,uv=c['uv'],native_path_id=c['native_path_id'],family='board',issues=flags,
                    source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],source_profile=ps)
                out.write(json.dumps(result,ensure_ascii=False)+'\n');totals['board']+=1;failures.update(flags);frame_bad.update(flags)
            frames.append(dict(pts=pt,ads=len(checked),board=len(board[pt]),issues=dict(frame_bad)))
            if index%50==0 or index==792:print(json.dumps(dict(frames=index+1,total=793,issues=dict(failures),seconds=time.monotonic()-start)),flush=True)
    grid.unchanged()
    write(OUT/f'evidence/{prefix}_check_summary.json',dict(frames=frames,totals=dict(totals),issues=dict(failures),source_package_changed=False,
        native_authority=grid.receipt,human_acceptance=False,findings_require_specific_repair_not_automatic_blanket_deletion=True))


if __name__=='__main__':main()
