"""Read-only S02 source audit against failure classes found in S04.

Candidates are never automatic exclusions or reassignments. Coordinates in
this audit come from the delivered input and this exposure's source image.
"""
from pathlib import Path
import argparse, collections, functools, gzip, hashlib, json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(r'D:/CodexData/RinkSpace_Data/runs/native_points_s02_20261027')
REPAIR = BASE / 'source_repair'
OUT = REPAIR / 'peer_review'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@functools.lru_cache(maxsize=1)
def load():
    z = dict(np.load(REPAIR/'points/point_observations.npz'))
    mapping = read(REPAIR/'points/mapping.json')
    names = np.asarray(mapping['path_ids'])[z['path_indices']]
    entities = {v['id']: v for v in read(BASE/'evidence/native_entities.json')['entities']}
    return z, names, entities

def hull(rows):
    uv = np.asarray([r['uv'] for r in rows], np.float32)
    if len(uv) < 5:
        return None
    q = cv2.convexHull(uv)
    return q if cv2.contourArea(q) > 100 else None

def scan():
    z, names, entities = load()
    hits, duplicates, summaries = [], [], []
    for pt in np.unique(z['pts']):
        frame = np.flatnonzero(z['pts'] == pt)
        packet = read(REPAIR/f'ice/{pt}.json')
        features = packet['features']
        linear = [i for i in frame if entities[names[i]]['layer'] == 'ice'
                  and entities[names[i]]['kind'] == 'ice_segment']
        centrelines = [i for i in frame if entities[names[i]]['layer'] == 'ice'
                      and entities[names[i]].get('measurement') == 'centreline']
        for house in ['F1','F2','F3','F4','R1','R2','R3','R4']:
            contours = {layer: hull(features.get(f'ice.{house}:{layer}', {}).get('rows', []))
                        for layer in ['outer','middle','inner','center']}
            for outer, inner in [('outer','middle'),('inner','center')]:
                a, b = contours[outer], contours[inner]
                if a is None or b is None:
                    continue
                for i in centrelines:
                    xy = tuple(map(float, z['uv'][i]))
                    if cv2.pointPolygonTest(a, xy, True) > 2 and cv2.pointPolygonTest(b, xy, True) < -2:
                        hits.append(dict(row=int(i), id=str(z['observation_ids'][i]), pts=int(pt),
                            uv=list(xy), path=str(names[i]), label=entities[names[i]]['label'],
                            house=house, band=f'{outer}-{inner}', candidate_only=True))
        # Nearby reads of differently named straight entities can expose a
        # duplicated physical stripe; proximity itself cannot decide identity.
        for j, i in enumerate(linear):
            for k in linear[j+1:]:
                if names[i] == names[k]:
                    continue
                dist = float(np.linalg.norm(z['uv'][i]-z['uv'][k]))
                if dist < 4:
                    duplicates.append(dict(pts=int(pt), rows=[int(i),int(k)],
                        labels=[entities[names[n]]['label'] for n in [i,k]], distance=dist))
        summaries.append(dict(pts=int(pt), rows=len(frame), ice_rows=sum(entities[names[i]]['layer']=='ice' for i in frame),
            line_rows=len(linear), actual_ice_paths=list(features)))
    write(OUT/'ring_band_candidates.json', hits)
    write(OUT/'duplicate_line_candidates.json', duplicates)
    write(OUT/'scan.json', dict(input_sha256=sha(REPAIR/'points/point_observations.npz'),
        frames=len(summaries), rows=len(names), ring_band_candidate_rows=len({r['row'] for r in hits}),
        candidate_frames=len({r['pts'] for r in hits}), duplicate_line_pairs=len(duplicates),
        no_input_changes=True, candidates_are_not_accuracy_verdicts=True, coverage=summaries))
    print(json.dumps(dict(ring_rows=len({r['row'] for r in hits}), frames=len({r['pts'] for r in hits}),
        labels=dict(collections.Counter(h['label'] for h in hits)), duplicate_pairs=len(duplicates)), ensure_ascii=False))

def colour(label):
    seed = hashlib.sha256(label.encode()).digest()
    return tuple(80+int(v)%176 for v in seed[:3])

def annotate(pt, layer='ice', selected=None):
    z,names,entities=load()
    im=cv2.imread(str(BASE/f'source_frames/{pt}.png'))
    ix=np.flatnonzero(z['pts']==pt)
    groups=collections.defaultdict(list)
    for i in ix:
        if entities[names[i]]['layer'] != layer:
            continue
        groups[str(names[i])].append(i)
    for path, ii in groups.items():
        label=entities[path]['label']; c=colour(label)
        xy=z['uv'][ii]
        for i,q in zip(ii,xy):
            cv2.circle(im,tuple(np.rint(q).astype(int)),3,c,-1,cv2.LINE_AA)
            if selected is not None and i in selected:
                cv2.circle(im,tuple(np.rint(q).astype(int)),10,(0,0,255),2,cv2.LINE_AA)
        anchor=tuple(np.rint(xy[len(xy)//2]).astype(int)+[3,-7])
        short=label.replace('ice.','')
        if layer=='advertisement':short=path.split('::')[0].split('/')[-1]+':'+path.split('::')[-1].split('.print.')[-1].replace('/path0','')
        cv2.putText(im,short,anchor,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(im,short,anchor,cv2.FONT_HERSHEY_SIMPLEX,.45,c,1,cv2.LINE_AA)
    return im

def overview():
    pts=list(range(575744,720641,256*25))
    if pts[-1]!=720640:pts.append(720640)
    for layer in ['ice','advertisement']:
        for page in range((len(pts)+5)//6):
            tiles=[]
            for pt in pts[page*6:(page+1)*6]:
                im=annotate(pt,layer)
                crop=im[450:1080] if layer=='ice' else im[:850]
                tile=cv2.resize(crop,(960,340 if layer=='ice' else 425))
                tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(tile,f'{pt/12800:.3f}s  PTS {pt}',(10,21),0,.65,(255,255,255),1,cv2.LINE_AA)
                tiles.append(tile)
            while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
            canvas=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
            OUT.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(OUT/f'{layer}_overview_{page}.jpg'),canvas)
    print('overview frames',pts)

def candidate_pages():
    z,names,entities=load();hits=read(OUT/'ring_band_candidates.json')
    groups=collections.defaultdict(list)
    for h in hits:groups[h['pts'],h['house']].append(h)
    index=[];tiles=[];page=0
    for (pt,house), hh in sorted(groups.items()):
        features=read(REPAIR/f'ice/{pt}.json')['features']
        xy=np.array([v['uv'] for layer in ['outer','middle','inner','center']
                     for v in features.get(f'ice.{house}:{layer}',{}).get('rows',[])])
        lo=np.maximum(np.floor(xy.min(0)-[60,55]).astype(int),[0,0]);hi=np.minimum(np.ceil(xy.max(0)+[60,55]).astype(int),[1920,1080])
        im=annotate(pt,selected={h['row'] for h in hh});crop=im[lo[1]:hi[1],lo[0]:hi[0]]
        scale=min(950/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
        tile=np.zeros((340,960,3),np.uint8);tile[35:35+crop.shape[0],:crop.shape[1]]=crop
        cv2.putText(tile,f'{len(index)}: {pt/12800:.3f}s {house} n={len(hh)}',(6,23),0,.65,(255,255,255),1,cv2.LINE_AA)
        index.append(dict(index=len(index),pts=pt,house=house,rows=[h['row']for h in hh],crop=[*lo.tolist(),*hi.tolist()],page=page))
        tiles.append(tile)
        if len(tiles)==8:
            cv2.imwrite(str(OUT/f'ring_candidates_{page:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,8,2)]));tiles=[];page+=1
    if tiles:
        while len(tiles)<8:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT/f'ring_candidates_{page:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,8,2)]))
    write(OUT/'ring_candidate_index.json',index);print('groups',len(index),'pages',page+bool(tiles))


def finalize():
    """Save this visual review, without editing or filtering solver inputs.

    These scopes describe images already inspected by the assistant. They are
    deliberately tied to one input digest: candidate discovery on another
    input must never inherit these accuracy decisions automatically.
    """
    expected = '5eef9ad919afbd4c169c2392e40bf7ecbbe447126a0e45ab68ece832352cbf05'
    if sha(REPAIR/'points/point_observations.npz') != expected:
        raise ValueError('Visual decisions belong to the reviewed input only')
    z, names, entities = load()
    ring_candidates = read(OUT/'ring_band_candidates.json')
    board_candidates = read(OUT/'ad_top_board_candidates.json')
    ring_rows = {v['row'] for v in ring_candidates
                 if v['house'] in ('R3','R4') and v['pts'] < 610000
                 and v['label'] != 'ice.A1:outer'}
    ad5_rows = {v['row'] for v in board_candidates
                if v['path'].startswith('print_instance/Ad5::')}
    representative_top_rows = {4574,11533,13401,13402,13403,13404,13405,
                               19668,19669,19693}
    glyph_rows = {94887,94888,94890,94891}
    assert len(ring_rows) == 354 and len(ad5_rows) == 71
    scopes = dict(ice_house_band_misidentified_as_straight=ring_rows,
                  ad5_logo_top_misidentified_as_board_top=ad5_rows,
                  other_ad_top_misidentified_as_board_top=representative_top_rows,
                  ad2_I_background_misidentified_as_letter=glyph_rows)
    selected = set().union(*scopes.values())
    assert sum(map(len,scopes.values())) == len(selected) == 439
    records = {}
    with gzip.open(REPAIR/'points/point_records.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['source_row'] in selected:
                records[row['source_row']] = row
    assert set(records) == selected
    point_findings = []
    for issue, indices in scopes.items():
        for i in sorted(indices):
            row = records[i]
            assert not row['human_acceptance']
            assert row['observation_id'] == str(z['observation_ids'][i])
            assert row['pts'] == int(z['pts'][i])
            assert np.array_equal(np.asarray(row['uv']),z['uv'][i])
            point_findings.append(dict(issue=issue,source_row=i,
                observation_id=row['observation_id'],pts=row['pts'],uv=row['uv'],
                native_path_id=row['native_path_id'],origin=row['origin'],
                source_group_id=row.get('source_group_id'),
                source_BGR_sha256=row['source_BGR_sha256'],
                source_image=str(BASE/f"source_frames/{row['pts']}.png"),
                review_authority='assistant_visual_review_not_user_acceptance',
                finding='confirmed_wrong_material_identity',applied_to_input=False))
    write(OUT/'confirmed_point_findings.json',point_findings)
    programme = Path(__file__).resolve().parent
    peer_files = ['s04_ad2_identity_review_20261027.json',
                  's04_lane_identity_review_20261027.json',
                  's04_remaining_identity_review_20261027.json',
                  's04_ad4_source_review_20261027.json',
                  's04_ad4_source_review_followup_20261027.json',
                  's04_ad4_glyph_review_20261027.json',
                  's04_ad4_context_review_20261027.json',
                  's04_tail_identity_review_20261027.json']
    peer = [];peer_exclusions=set()
    for name in peer_files:
        p = programme/name;doc = read(p)
        peer.append(dict(path=str(p),sha256=sha(p),
            excluded_rows=doc.get('excluded_rows',0),
            rebind_operation_count=doc.get('rebound_rows',len(doc.get('changes',[]))),
            note='Rebind operations include revisions and must not be summed as unique points.'))
        peer_exclusions.update(v['observation_id'] for v in doc.get('operations',[])
                               if v.get('action')=='exclude')
    assert sum(v['excluded_rows'] for v in peer) == len(peer_exclusions) == 1735
    issue_summary=[]
    for issue,indices in scopes.items():
        pts=sorted({int(z['pts'][i])for i in indices})
        issue_summary.append(dict(issue=issue,confirmed_rows=len(indices),
            frames=len(pts),source_pts_range=[pts[0],pts[-1]],
            source_seconds_range=[pts[0]/12800,pts[-1]/12800],
            labels=dict(collections.Counter(entities[names[i]]['label']for i in indices))))
    protections=[(REPAIR/'points/point_observations.npz',expected),
        (BASE/'points/point_observations.npz','5c8f8331b8a1a3db40270007846c3f0d25aa259643a3f999aec0b1f4bb6556b2'),
        (BASE/'camera_fit/geometry_seed.npz','5738c5195eb4a93bfac36231fd322e486b1f9caf43b20e132f5874ad3c31014c'),
        (ROOT/'events/ankara_2026/scene/field/model.json','fcebdfd0a4a4c85e590be72daa75bdc7bc8c47e28434e51c0fbd5621e41ef2d5')]
    hashes=[]
    for p,expected_hash in protections:
        current=sha(p);assert current==expected_hash,str(p)
        hashes.append(dict(path=str(p),sha256=current,unchanged=True))
    review=dict(schema='rinkspace.source_identity_audit.v1',shot='S02',
        input_sha256=expected,delivered_input_rows=len(names),delivered_input_frames=len(np.unique(z['pts'])),
        source_pts_range=[int(z['pts'].min()),int(z['pts'].max())],time_base=[1,12800],
        peer_review_source='local_camera_fitting_review',
        peer_thread_title='确认相机模型并拟合冈田芽衣短节目',
        peer_conversation_read=True,peer_review_files=peer,
        peer_unique_excluded_points_after_replaced_s04_package=len(peer_exclusions),
        peer_count_correction='1735 includes 239 Ad4 exclusions; the interim 1496 count omitted those.',
        confirmed_wrong_identity_rows=len(selected),confirmed_is_lower_bound=True,
        confirmed_point_findings=str(OUT/'confirmed_point_findings.json'),issues=issue_summary,
        causes=[
            dict(issue='ice_house_band_misidentified_as_straight',
                code=str(programme/'repair_points_s02.py'),lines=[145,162],
                evidence='Line reads never enter the shared circle/material conflict check. A broad filled house band passes local dark-band measurement and a nearly straight fit.',
                required_repair='Resolve the shared visible material and line connectivity before emitting a named line; remeasure the intended line from this RGB exposure. The band midpoint cannot become a ring boundary by renaming it.'),
            dict(issue='ad_top_misidentified_as_board_top',
                code=str(programme/'repair_points_s02.py'),lines=[260,296],
                evidence='The colored-to-white edge predicate admits the blue board cap, although its current native identity is not the advertisement.',
                required_repair='Use the actual board surface and the native advertisement boundary meaning to constrain source search, then measure the actual material transition.'),
            dict(issue='ad5_native_boundary_semantics_ignored',
                native_authority=str(ROOT/'events/ankara_2026/scene/ad5/assets_r2.json'),
                native_role='logo_outer_boundary',
                native_meaning='MEIWA blue/green rectangular graphic edge, not the entire white advertisement panel',
                evidence='The generic top_edge suffix routes this graphic boundary to a whole-panel reader; correct green bottom-edge reads coexist with wrong blue board-top reads.',
                required_repair='Read the existing native asset role, preserving the curated native rectangle; do not delete the complete Ad5 group.'),
            dict(issue='ad2_I_background_misidentified_as_letter',
                code=str(programme/'ad_entity_observability_s01.py'),lines=[60,64],
                evidence='The false reads pass local RGB profiles and current_structure_resolved=True. This group takes the fine_detail=False early return without a whole-glyph source check.',
                required_repair='Verify each current glyph with full-letter context and neighboring exposures; restrict a profile to its real material. Do not discard every blurred letter or tighten a residual threshold.')],
        evidence_images=dict(rings=[str(OUT/'rings_598528.jpg'),str(OUT/'rings_599552.jpg')],
            ad5_all_frames=[str(OUT/f'ad5_all_{p}.jpg')for p in range(5)],
            other_ad_top=[str(OUT/f'ad_top_{pt}.jpg')for pt in [579328,589056,595200,611840,612096]],
            glyph=str(OUT/'I_710144_detail.jpg')),
        coverage=dict(automated_frames=567,overview_frames=24,
            targeted_visual_review='All early R3/R4 candidate crops and the 43 Ad5 exposures; representative other ad-top frames and current Ad2 glyph sequences were visually inspected.',
            incomplete_visual_scope='Not every point of the 567 frames has been visually certified. Remaining ring overlap, other ad-top candidates, cross-letter/inner-hole assignments and straight-to-curve identities remain unconfirmed.',
            ring_candidates=len({v['row']for v in ring_candidates}),
            unconfirmed_ring_candidate_rows=len({v['row']for v in ring_candidates}-ring_rows),
            board_top_candidates=len({v['row']for v in board_candidates}),
            unconfirmed_board_top_candidate_rows=len({v['row']for v in board_candidates}-ad5_rows-representative_top_rows),
            candidate_proximity_is_not_a_disposition_rule=True),
        preserved_after_review=dict(ad5_other_edges='Correct green graphic-bottom reads remain in scope; whole Ad5 deletion is not justified.',
            blurred_letters='Several parts of Ad2 A/K/S/O remain visible despite motion blur. No whole-frame or whole-glyph rejection follows from these findings.',
            ring_and_ad_remaining_candidates='Candidate locations not confirmed by source inspection are not labeled as errors.',
            manual_acceptance='All 439 listed rows are machine observations. This audit creates no user acceptance.'),
        input_changed=False,native_changed=False,solver_started=False,gpu_used=False,
        peer_messages_sent=False,review_only_not_a_repaired_package=True,protected_files=hashes)
    write(OUT/'review.json',review)
    print(json.dumps(dict(confirmed_wrong_identity_rows=len(selected),
        peer_unique_excluded_points=len(peer_exclusions),issues=issue_summary,
        input_unchanged=True,review=str(OUT/'review.json')),ensure_ascii=False))

if __name__=='__main__':
    cv2.setNumThreads(2)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['scan','overview','candidate_pages','finalize'])
    globals()[parser.parse_args().command]()
