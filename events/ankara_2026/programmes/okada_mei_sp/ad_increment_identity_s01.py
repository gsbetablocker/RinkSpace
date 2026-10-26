"""Resolve historical hint conflicts before freezing the additive increment."""
from pathlib import Path
import sys,cv2,numpy as np,json,collections
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
OUT=b.OUT/'increments/001'

def audit():
    z=dict(np.load(OUT/'additions.npz'));mapping=b.read(OUT/'mapping.json');paths=mapping['path_ids'];lookup={p:i for i,p in enumerate(paths)};data=b.load();assets=b.assets();changes=[]
    def bind(i,target,proof):
        before=paths[z['path_indices'][i]];after=target if target.startswith('print_instance/') else 'print_instance/'+target.split('.')[0]+'::'+target
        if before==after:return
        z['path_indices'][i]=lookup[after];changes.append(dict(increment_row=int(i),source_row=int(z['source_rows'][i]),stable_id=str(z['observation_ids'][i]),pts=int(z['pts'][i]),uv=z['uv'][i].tolist(),before=before,after=after,proof=proof,raw_UV_PTS_ID_unchanged=True,baseline_untouched=True))
    for r in b.read(OUT/'identity_conflicts.json'):
        bind(r['increment_row'],r['verified'],dict(kind='later_actual_source_entity_correspondence_overrides_historical_hint',file=r['proof'],sha256=b.sha(Path(r['proof'])),record=r['record'],registration=r['registration']))
    for i,row in enumerate(z['source_rows']):
        if row>=len(data['pts']):continue
        label=str(data['labels'][row]);current=paths[z['path_indices'][i]]
        if label=='Ad7.top_edge' and '.top_edge' not in current:
            bind(i,assets[7]['aliases'][label][0],dict(kind='direct_actual_source_blue_panel_top_edge',image=str(OUT/'identity_439296.png'),same_qualified_source_edge_label=label,reason='Source point lies on the blue panel upper boundary, above all letters; degenerate glyph registration is contradicted by actual source'))
    atlas=next(a for a in b.read(b.OUT/'atlases.json')['atlases'] if a['id']==6);vis={f['pts']:f for f in b.read(b.OUT/'visibility.json')['frames']};cache={}
    for i,row in enumerate(z['source_rows']):
        current=paths[z['path_indices'][i]]
        if not ('Ad4.print.shot4_' in current):continue
        pt=int(z['pts'][i]);uv=z['uv'][i];x,y=np.rint(uv).astype(int)
        if pt not in cache:
            f=vis[pt];im=cv2.imread(f['image']);B,G,R=cv2.split(im.astype(float));purple=((R-G>18)&(B-G>15)&(R-B>-70)).astype(np.uint8);purple[:180]=0
            cs,_=cv2.findContours(purple,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cs=[c for c in cs if cv2.contourArea(c)>250];hull=np.zeros(im.shape[:2],np.uint8)
            if cs:cv2.fillConvexPoly(hull,cv2.convexHull(np.concatenate(cs)),1)
            proof=b.OUT/'verified_final'/f'ad4_{pt}.json';doc=b.read(proof);refs=[]
            for reg in doc['registrations']:
                if reg['atlas_id']!=6:continue
                for path in atlas['paths']:
                    curve=cv2.perspectiveTransform(np.asarray(path['pixels'],np.float32)[None],np.asarray(reg['H']))[0];refs.append((path['path_id'],curve,reg))
            cache[pt]=(hull,refs,proof)
        hull,refs,proof=cache[pt]
        if not hull[y,x] or not refs:continue
        distances=[float(np.min(np.linalg.norm(curve-uv,axis=1))) for _,curve,_ in refs];j=int(np.argmin(distances))
        if distances[j]<4:
            target,curve,reg=refs[j];bind(i,target,dict(kind='actual_purple_ANKARA_panel_and_registered_source_contour',file=str(proof),sha256=b.sha(proof),source_atlas_id=6,source_contour_distance=distances[j],registration=reg,native_geometry_used=False))
    np.savez_compressed(OUT/'additions.npz',**z);b.write(OUT/'identity_audit.json',dict(corrected_increment_bindings=len(changes),changes=changes,observation_count_unchanged=True,raw_pixels_unchanged=True,baseline_untouched=True,unreviewed_global_accuracy_not_claimed=True));print(json.dumps(dict(corrected_increment_bindings=len(changes),by_kind=dict(collections.Counter(r['proof']['kind'] for r in changes)))),flush=True)
def apply_reviewed():
    z=dict(np.load(OUT/'additions.npz'));mapping=b.read(OUT/'mapping.json');lookup={p:i for i,p in enumerate(mapping['path_ids'])};rows={int(r):i for i,r in enumerate(z['source_rows'])};review=b.read(OUT/'identity_overrides_final.json');changed=False
    for record in review['records']:
        i=rows[record['source_row']]
        if str(z['observation_ids'][i])!=record['stable_id'] or int(z['pts'][i])!=record['pts'] or not np.array_equal(z['uv'][i],record['original_uv']):raise ValueError('Reviewed identity belongs to different source pixels')
        for key,value in [('path_indices',lookup[record['path_id']]),('kind',record['kind']),('stations',record['station'])]:
            changed|=bool(z[key][i]!=value);z[key][i]=value
    if changed:
        if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten; prepare a new increment')
        np.savez_compressed(OUT/'additions.npz',**z)
    print(json.dumps(dict(reviewed_identity_overrides=len(review['records']),changes_required=changed)),flush=True)

if __name__=='__main__':
    if '--rebuild-candidates' in sys.argv:
        if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten')
        audit()
    else:apply_reviewed()
