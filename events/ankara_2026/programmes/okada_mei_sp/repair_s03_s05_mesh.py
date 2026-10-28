"""Original RGB repair navigated by the frozen current native review mesh.

Candidate files are separate from admitted observations. A camera, transported
guide or fitted shape cannot supply an output UV.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, collections, json, time
import cv2
import numpy as np
import audit_s03_s05_projection as audit
from ice_points_s04 import read, write, source_domain
from reread_s04_lines import NativeLineReader
from rinkspace.review.scene import sample_review_scene
from rinkspace.vision_native import file_ref


class ReviewGrid(audit.source.common.shared.NativeGrid):
    def __init__(self):
        m,z,mapping,g,self.core,native=audit.fit.load()
        self.parameter_state=native.parameter_state
        self.scene=self.parameter_state.apply_scene(read(m['scene_path']))
        camera=read(audit.fit.OUT/'camera_seed.json')
        state=audit.fit.OUT/'navigation_pose/accepted_camera_latest.npz'
        self.state=dict(np.load(state,allow_pickle=False))
        assert np.array_equal(self.state['scene_values'],g['scene_values'])
        self.first=camera['first_control_time'];self.dt=camera['control_dt'];self.reference=camera['reference_pts']
        self.manifest=m
        self.samples=sample_review_scene(read(m['scene_path']),self.core,Path(m['scene_path']),parameter_state=self.parameter_state,observed_path_ids=set())
        self.cuts={r['id']:r for r in self.samples['cuts']}
        self.receipt=dict(camera=file_ref(state),geometry=m['geometry_origin'],native_geometry_changed=False)


class NativeSourceGuides:
    """Use current finite geometry, retaining qualified adjacent image motion.

Old named image traces are deliberately not given unconditional authority to
override the finite path. The old traces stay in the immutable source ledger.
"""
    def __init__(self,folder,first):
        self.transforms={first:(np.eye(3),0)};component=0;prior=np.eye(3)
        scale=np.diag([.5,.5,1.])
        for r in read(folder/'evidence/source_navigation.json')['pairs']:
            pt=r['source_pts']
            if r['status']=='candidate' and r['diagnostic_median_reference_px']<=2. and 2*r['diagnostic_p95_reference_px']<=6.:
                h=np.linalg.inv(scale)@np.linalg.inv(np.asarray(r['H_reference_from_current']))@scale
                prior=h@prior
                prior/=np.linalg.norm(prior)
            else:component+=1;prior=np.eye(3)
            self.transforms[pt]=(prior.copy(),component)

    def at(self,name,pt):return None


def reread(pts,labels,schedule=None):
    src=audit.source;grid=ReviewGrid();folder=src.OUT/'evidence/native_overlay_review/reread';folder.mkdir(exist_ok=True)
    guides=NativeSourceGuides(src.OUT,src.FIRST)
    line=NativeLineReader(grid,photographic=guides,motion_path=src.OUT/'evidence/source_navigation.json',
        source_prefix=src.SHOT+'_mesh_review',native_crossline_partitions=True,recover_partial_blue=True,temporal_achromatic_blue=True)
    allpaths=line.paths
    for p in line.paths:p.update(_source_query_spacing_px=35,_separate_profile_support=True)
    house=src.common.shared.SourceHouseReader(grid,motion_path=src.OUT/'evidence/source_navigation.json')
    vis=read(src.OUT/'evidence/visibility.json');count=0
    for v in vis['frames']:
        pt=v['pts']
        if pt not in pts:continue
        wanted=schedule[pt]if schedule else labels
        # Existing same-frame lane profiles supply the historical thin-paint
        # width reference before connections are read. A connection-only call
        # otherwise silently loses that material discriminator.
        need_width=any('house_connection' in s for s in wanted)
        line.paths=[p for p in allpaths if p['label']in wanted or need_width and p['feature_id'].startswith('lane')]
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)>0
        graphics,border,_,_=source_domain(im,mask.astype('uint8')*255,read(v['board_source']))
        mask|=graphics|(np.arange(1080)[:,None]<=border[None])|(cv2.imread(v['near_outside_ref']['file'],0)>0)
        features,diagnostics=line.read(pt,im,mask,vis['source_video_sha256'])if line.paths else ({},{})
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=mask)
        features={k:v for k,v in features.items()if k in wanted}
        for fid in sorted({name[4:].split(':')[0]for name in wanted} & house.groups.keys()):
            result,diag=house.read(fid,pt,native);diagnostics[fid]=diag
            for role,data in result.items():
                path=house.native_paths[(fid,role)][2]
                if path['label']not in wanted:continue
                rr=[]
                for i,row in enumerate(data['rows']):
                    r={k:v for k,v in row.items()if k not in('representative_uv','representative_to_actual_material_px')}
                    r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,
                        source_row_id=f'{src.SHOT}_mesh_review_RGB/{pt}/{fid}/{role}/{i}')
                    rr.append(r)
                features[path['label']]=dict(native_path_id=path['id'],rows=rr,navigation=data['joint_group_navigation'])
            house.record_source_family(fid,pt,result)
        # Explicit empty results are review evidence, not silent fallback to
        # the same wrong historical guide on the next assembly.
        write(folder/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],
            features=features,requested_labels=wanted,diagnostics=diagnostics,review_grid=grid.receipt,
            original_RGB_reread=True,model_or_flow_pixels_emitted=False,human_acceptance=False))
        count+=1
        print(json.dumps(dict(event='mesh_RGB_reread',shot=src.SHOT,frames=count,pts=pt,
            rows={k:len(v['rows'])for k,v in features.items()})),flush=True)


def candidate_schedule():
    src=audit.source;diag=read(src.OUT/'evidence/native_overlay_review/diagnostic.json');schedule=collections.defaultdict(set);seeds=[]
    for r in diag['records']:
        if r['layer']!='ice':continue
        candidate=(r['kind']=='ice_segment'and(r['median']>40 or r['maximum']>100))or(r['label']=='ice.C0:centreline'and r['median']>40)or(r['kind']!='ice_segment'and r['median']>100)
        if not candidate:continue
        seeds.append(r)
        for pt in range(max(src.FIRST,r['pts']-512),min(src.LAST,r['pts']+512)+1,256):schedule[pt].add(r['label'])
    if src.SHOT=='s03':
        for pt in range(src.FIRST,src.LAST+1,256):schedule[pt].update(['ice.lane3:edge1','ice.lane4:edge0','ice.lane4:edge1'])
    result={pt:sorted(v)for pt,v in sorted(schedule.items())}
    write(src.OUT/'evidence/native_overlay_review/reread_schedule.json',dict(
        diagnostic_selection_only=True,no_distance_threshold_used_to_admit_or_delete_source=True,seeds=seeds,
        frames=[dict(pts=pt,labels=v)for pt,v in result.items()],adjacent_exposures_included=True))
    return result


def comparison(pt,label):
    src=audit.source;grid=ReviewGrid();folder=src.OUT/'evidence/native_overlay_review'
    z=np.load(src.OUT/'points/point_observations.npz');mapping=read(src.OUT/'points/mapping.json')
    entities={r['label']:r for r in read(src.OUT/'evidence/native_entities.json')['entities']};pid=entities[label]['id']
    pi=mapping['path_ids'].index(pid);old=z['uv'][(z['pts']==pt)&(z['path_indices']==pi)]
    f=read(folder/f'reread/{pt}.json')['features'].get(label,{'rows':[]});new=np.array([r['uv']for r in f['rows']]).reshape(-1,2)
    projection=grid.project(pt)[pid];q=projection['uv'];ok=projection['valid']&np.isfinite(q).all(1)&(abs(q).max(1)<1e5)
    im=cv2.imread(str(src.OUT/f'source_frames/{pt}.png'));tiles=[]
    both=np.vstack([old,new]);lo=np.maximum(np.floor(both.min(0)-[100,100]).astype(int),[0,0]);hi=np.minimum(np.ceil(both.max(0)+[100,100]).astype(int),[1920,1080])
    for title,uv,color in [('BEFORE',old,(30,70,255)),('CURRENT RGB REREAD',new,(255,230,20))]:
        tile=im.copy()
        for run in np.split(np.flatnonzero(ok),np.flatnonzero(np.diff(np.flatnonzero(ok))>1)+1):
            if len(run)>1:cv2.polylines(tile,[np.rint(q[run]).astype('int32')],False,(90,220,90),1,cv2.LINE_AA)
        for xy in uv:cv2.circle(tile,tuple(np.rint(xy).astype(int)),3,color,-1,cv2.LINE_AA)
        tile=tile[lo[1]:hi[1],lo[0]:hi[0]];s=min(1920/tile.shape[1],600/tile.shape[0]);tile=cv2.resize(tile,None,fx=s,fy=s)
        canvas=np.zeros((tile.shape[0]+45,1920,3),np.uint8);canvas[45:,:tile.shape[1]]=tile
        cv2.putText(canvas,f'{title} {label} {pt/12800:.3f}s PTS{pt} {len(uv)} points',(15,30),0,.75,(255,255,255),1,cv2.LINE_AA);tiles.append(canvas)
    cv2.imwrite(str(folder/f'comparison_{pt}_{pid.replace("/","_").replace(":","_")}.jpg'),np.vstack(tiles))


def main():
    p=argparse.ArgumentParser();p.add_argument('shot',choices=['s03','s05']);p.add_argument('--pts',nargs='+',type=int)
    p.add_argument('--labels',nargs='+');p.add_argument('--scheduled',action='store_true');args=p.parse_args()
    cv2.setNumThreads(2);audit.configure(args.shot)
    if args.scheduled:
        schedule=candidate_schedule();reread(set(schedule),None,schedule)
    else:reread(set(args.pts),args.labels)


if __name__=='__main__':main()
