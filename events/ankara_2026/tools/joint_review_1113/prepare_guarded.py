"""Prepare conservative second-pass evidence on source intervals withheld in advance."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
import json
import shutil

import joint_inputs as ji
from joint_inputs import OUT,np,cv2,TB,STEP
from audit_residuals import localize,cap_group
from qualify import cover_indices

BASE=OUT/'prepared_joint'
TARGET=OUT/'prepared_guarded'
EXTRA=OUT/'supplement'
S01_A4={457216,461568,465920,470272,483328}
S05_A4={1005312}
OKADA_WRONG_A4={1434880,1439488,1444096,1440768,1441024}
FOCUS=('jin_s03','jin_s04','jin_s06','jin_s10','jin_s12','jin_s13','jin_s14','jin_s15','jin_s16',
       'jin_s18','jin_s19','jin_s20','jin_s21','jin_s22','jin_s24','okada_s04','okada_s05','okada_s08')
PAINT_QUALITY={'direct_coloured_paint_flanks','paired_blue_annulus_outer_source','direct_red_badge_ink',
               'original_s04_confirmed_red_center','red_hockey_arc_two_flanks'}
EDGE_QUALITY={'blue_band_upper_boundary_direct_source','yellow_kickplate_lower_edge_direct_source',
               'top_face_blue_band_ice_side_direct_source','blue_cap_ice_side_long_edge'}
KEY_FAMILIES={'B1','B2','C','lane1','lane2','lane3','lane4','goal_line_left','goal_line_right',
              'A4','F1','F2','R1','R2','Ad4','Ad13','Ad14','Ad17','Ad18','Ad19','Ad26'}
STOP_RAIL={'s03','s06','s10','s12','s13','s16','s19','s20','s21','s22'}
RECAP={'board.top':9,'board.bottom':9,'B1':7,'B2':7,'C':7,'goal_line_left':7,'goal_line_right':7,
       'R1':8,'R2':8,'F1':8,'F2':8}


def role(shot,pts):
    idx=(pts-shot.pts[0])//STEP
    q=(idx+19*(int(shot.id[1:])%6))%174
    # Freeze different source intervals for the second pass. The first
    # trial's audit intervals become development evidence.
    if (70<=q<77 or 129<=q<136):return 2
    if 57<=q<70 or 117<=q<129:return 1
    if 45<=q<84 or 106<=q<143:return 3
    return 0


def raw_rule(shot,g,native):
    f=ji.family(g['path'],native);m=g['material']
    # The residual is consulted later for numerical leverage only.
    # Decisions here describe confusion already verified on original frames.
    if shot.program=='jin':
        if (f.startswith('goal_line_') or f.startswith('trapezoid_')) and shot.id in STOP_RAIL:
            if m not in PAINT_QUALITY and m not in {'two_sided_original_paint_band'}:
                return 'unverified_goal_near_yellow_rail'
        if shot.id=='s19' and f=='F1' and m=='measured_paired_annulus':
            return 'reflection_confused_with_F1'
        if shot.id=='s14' and f in ('A1','A2','A3','C','lane3','lane4') and g['origin']=='fresh':
            if m=='measured_paired_paint_edges':
                return 'wide_view_identity_unresolved'
        if shot.id in ('s21','s22') and f=='board.top' and m=='blue_cap_far' and g['origin']=='fresh':
            return 'uncertain_edge_next_to_judges_or_wide_turn'
    return None


def admit_source(shot,native):
    groups=[]
    keys={}
    rows=[]
    p=EXTRA/(shot.name+'.jsonl')
    if not p.exists():return keys,rows
    for v in ji.lines(p):
        idx=int(shot.at(v['pts']))
        gs=[]
        for g in v['groups']:
            xy=cap_group(g,10)
            if xy is None:continue
            gs.append(dict(k=g['path'],uv=xy,origin='fresh',material=g['material']))
        if gs:
            keys[v['pts']]=v['source_BGR_sha256']
            groups.append((idx,v['pts'],shot.times[idx],gs))
    if groups:
        a=localize(shot,native,groups)
        for g in a:
            u=(~g['endpoint'])|(g['error']<=5.)
            if not u.any():continue
            rows.append(dict(pts=g['pts'],k=g['k'],origin='fresh',material=g['material'],
                uv=g['uv'][u],station=g['station'][u]))
    return keys,rows


def select_family(shot,pts,f,a):
    # Density and uncertainty are independent decisions. Explicit source
    # flanks replace ambiguous near-duplicates, and spatial extremes are
    # retained with one budget for an entire visible feature.
    n=RECAP.get(f,6)
    prior=np.asarray([m in PAINT_QUALITY or m in EDGE_QUALITY for m in a['material']])
    if f in ('C','B1','B2','goal_line_left','goal_line_right','R1','F1','F2') and prior.sum()>=4:
        may=np.flatnonzero(prior)
    else:
        may=np.arange(len(prior))
    if f.startswith('lane') or f in ('F1','F2','R1','R2','board.top','board.bottom'):
        groups=defaultdict(list)
        for i in may:
            sub=a['path'][i]
            if f.startswith('board.'):
                sub='near' if a['uv'][i][1]>=620 else 'far'
            groups[sub].append(i)
        chosen=[]
        for k,v in groups.items():
            idx=cover_indices(a['uv'][v],max(2,int(np.ceil(n/len(groups)))),14.)
            chosen.extend(np.asarray(v)[idx].tolist())
        picked=np.asarray(chosen,int)
    else:
        picked=may[cover_indices(a['uv'][may],n,8. if f.startswith('Ad') else 14.)]
    mass=1.
    if f.startswith('board.'):mass=1.95
    elif f in ('B1','B2','C','goal_line_left','goal_line_right'):mass=1.37
    elif f in KEY_FAMILIES:mass=1.18
    if np.any(a['origin'][picked]=='fresh') and not np.any(prior[picked]):
        mass*=.77
    return picked,np.sqrt(mass/len(picked))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.parse_args()
    t0=monotonic()
    _,native,shots,_=ji.inputs()
    TARGET.mkdir(exist_ok=False)
    shutil.copyfile(BASE/'setup.json',TARGET/'setup.json')
    shutil.copyfile(BASE/'baseline.npz',TARGET/'baseline.npz')
    info={}
    for name,shot in shots.items():
        with np.load(BASE/(name+'.npz'),allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        uv_map={(int(p),str(k),tuple(u)):float(s) for p,k,u,s in zip(a['pts'],a['path'],a['uv'],a['station'])}
        material={}
        history={}
        removed=Counter()
        fresh=Counter()
        recs=defaultdict(list)
        for old in ji.lines(BASE/(name+'_decisions.jsonl')):
            p=old['pts']
            material[p]=old.get('source_BGR_sha256')
            history[p]=list(old['excluded'])
            for g in old['groups']:
                f=ji.family(g['path'],native)
                if shot.program=='okada' and f=='A4' and p in OKADA_WRONG_A4:
                    reason='wrong_A4_association_verified_in_original'
                else:
                    reason=raw_rule(shot,g,native)
                if reason:
                    removed[reason]+=len(g['uv'])
                    history[p].append(dict(g,pts=p,reason=reason,family=f))
                else:
                    station=[uv_map[(p,g['path'],tuple(uv))] for uv in g['uv']]
                    recs[p].append(dict(g,station=station))
        sha,new=admit_source(shot,native)
        for p,h in sha.items():
            if p in material and material[p] and material[p]!=h:raise ValueError('Supplement source mismatch')
            material[p]=h
        for g in new:
            p=g['pts']
            recs[p].append(dict(path=g['k'],uv=g['uv'].tolist(),station=g['station'].tolist(),
                                material=g['material'],origin='fresh'))
        observations=defaultdict(list)
        with (TARGET/(name+'_decisions.jsonl')).open('x',encoding='utf8',buffering=1<<16) as f:
            for p in sorted(material):
                idx=int(shot.at(p));fold=role(shot,p)
                by=defaultdict(list)
                for g in recs[p]:
                    by[ji.family(g['path'],native)].append(g)
                log=[]
                for fam,gs in by.items():
                    raw=defaultdict(list)
                    for g in gs:
                        n=len(g['uv'])
                        for k in ('uv','station'):raw[k].extend(g[k])
                        for k in ('path','material','origin'):raw[k].extend([g[k]]*n)
                    rows={k:np.asarray(v) for k,v in raw.items()}
                    ii,w=select_family(shot,p,fam,rows)
                    for k in ('uv','station','path'):observations[k].extend(rows[k][ii])
                    observations['weight'].extend([w]*len(ii))
                    observations['pts'].extend([p]*len(ii))
                    observations['idx'].extend([idx]*len(ii))
                    observations['fold'].extend([fold]*len(ii))
                    observations['family'].extend([fam]*len(ii))
                    observations['fresh'].extend(rows['origin'][ii]=='fresh')
                    q=defaultdict(list)
                    for i in ii:
                        k=tuple(str(rows[x][i]) for x in ('path','origin','material'))
                        q[k].append(rows['uv'][i].tolist())
                        if rows['origin'][i]=='fresh':
                            fresh[fam]+=1
                    for (k,orig,m),uv in q.items():log.append(dict(path=k,origin=orig,material=m,uv=uv))
                f.write(json.dumps(dict(pts=p,source_BGR_sha256=material[p],fold=fold,groups=log,
                                   excluded=history.get(p,[])),separators=(',',':'))+'\n')
        np.savez_compressed(TARGET/(name+'.npz'),**{k:np.asarray(v) for k,v in observations.items()})
        n=len(observations['uv'])
        info[name]=dict(frames=len(material),rows=n,removed=dict(removed),fresh=dict(fresh),
                        fold={str(k):v for k,v in Counter(observations['fold']).items()})
        print(name,n,sum(removed.values()),sum(fresh.values()),round(monotonic()-t0,1),flush=True)
    ji.write(TARGET/'input_summary.json',info)
if __name__=='__main__':main()
