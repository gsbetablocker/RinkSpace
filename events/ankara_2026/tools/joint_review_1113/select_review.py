"""Compose a conservative candidate and compare the same material on the original clock."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import monotonic
import json

import verify_joint as v
from verify_joint import ji,sj,np,OUT
from audit_residuals import localize
from solve_joint import atomic_arrays

BASE=OUT/'prepared_guarded'
SOURCE=OUT/'guarded'
DEST=OUT/'selection'
ADS=(13,14,17,18,19)
PLACEMENT_FRACTION={18:.35,19:.72}
COARSE={
'jin_s03':('motion',1.),
'jin_s13':('motion',.55),
'jin_s15':('joint',.85),
'jin_s21':('joint',.65),
'okada_s04':('motion',.38),
'okada_s05':('motion',.40),
}
WINDOWS={
'jin_s03':(38.72,42.16),'jin_s04':(46.92,47.42),'jin_s06':(52.82,57.30),'jin_s10':(78.26,89.14),
'jin_s13':(112.30,113.60),'jin_s14':(120.32,123.98),'jin_s15':(124.32,127.58),
'jin_s18':(136.88,138.16),'jin_s19':(142.82,143.60),'jin_s21':(147.30,149.04),
'jin_s22':(149.10,155.06),'jin_s24':(172.06,196.10),
'okada_s04':(59.2,72.9),'okada_s05':(77.74,81.42),'okada_s08':(141.80,142.82)}
FINAL_FOCUS=('jin_s03','jin_s04','jin_s10','jin_s13','jin_s14','jin_s15','jin_s18','jin_s19',
             'jin_s21','jin_s22','jin_s24','okada_s04','okada_s05','okada_s08')
KEY=('board.top','board.bottom','lane1','lane2','lane3','C','B2','F1','F2','R1','R2','A4',
     'goal_line_left','goal_line_right','Ad1','Ad4','Ad8','Ad9','Ad13','Ad14','Ad17','Ad18','Ad19','Ad26')


def candidate(base,other,layout,installations,motion):
    current={k:z.copy() for k,z in base.items()}
    for d in layout['blocks']:
        identity=d['id']
        if not identity.startswith('installation/') or int(identity.split('/')[1][2:]) not in installations:
            continue
        idx=slice(d['offset'],d['offset']+d['size'])
        n=int(identity.split('/')[1][2:]);alpha=PLACEMENT_FRACTION.get(n,1.)
        current['scene_values'][idx]=base['scene_values'][idx]+alpha*(other['joint']['scene_values'][idx]-base['scene_values'][idx])
    for name,(k,alpha) in motion.items():
        j=name+'_controls'
        current[j]=base[j]+alpha*(other[k][j]-base[j])
    return current


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='selection')
    ap.add_argument('--ads',nargs='+',type=int,default=ADS)
    ap.add_argument('--test',action='store_true')
    a=ap.parse_args()
    sj.INPUT=BASE
    m,core,b,series=sj.load()
    changed={k:v.load_checkpoint(SOURCE/(k+'.npz')) for k in ('motion','joint')}
    root=OUT/a.out;root.mkdir(exist_ok=False)
    mix=candidate(b,changed,m['scene_layout'],a.ads,COARSE)
    atomic_arrays(root/'review.npz',**mix)
    test_geometry=candidate(b,changed,m['scene_layout'],a.ads,{})
    n={k:sj.NativeSceneGPU(m['scene_path'],core,parameter_state=sj.ParameterState(m['scene_layout'],
                np.ascontiguousarray(z['scene_values']))) for k,z in
           (('baseline',b),('scene',test_geometry),('review',mix))}
    results={};groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    start=monotonic()
    for name,r in series.items():
        shot=v.shot_from_inputs(name,m,b)
        use=(r['fold']==1)|(r['fold']==2)
        if name in WINDOWS:
            a0,a1=WINDOWS[name]
            local=(r['pts']>=round(12800*a0))&(r['pts']<=round(12800*a1))
            use|=local
        else:local=np.zeros(len(use),bool)
        record={}
        for label,z in (('baseline',b),('scene',test_geometry),('review',mix)):
            err,sign=v.eval_one(n[label],shot,r,z,use)
            record[label]={}
            for fold,sel in [('audit',r['fold']==2),('development',r['fold']==1),('local',local)]:
                table=v.collect(r,err,sign,sel)
                record[label][fold]=v.summarise(table)
                if fold!='local':
                    for k,values in table.items():groups[label][shot.program+'_'+fold][k].extend(values)
        results[name]=record
        print(name,'audit',[(k,round(x['audit'].get('all',{}).get('balanced_rms',0.),3)) for k,x in record.items()],
               'local',[(k,round(x['local'].get('all',{}).get('balanced_rms',0.),3)) for k,x in record.items()],
               round(monotonic()-start,1),flush=True)
    totals={label:{prog:v.summarise(rows) for prog,rows in data.items()} for label,data in groups.items()}
    ji.write(root/'comparison.json',dict(scene_delta_instances=a.ads,camera_motion=COARSE,installation_fraction=PLACEMENT_FRACTION,
              audit_definition='Separate native source intervals from guarded fit; consulted for candidate selection',
              shot=results,aggregate=totals,source_windows=WINDOWS,geometry_sha256=ji.numeric_hash(mix['scene_values'])))
    for k,a in totals.items():
        print(k,{prog:round(g['all']['balanced_rms'],3) for prog,g in a.items()},flush=True)

if __name__=='__main__':main()
