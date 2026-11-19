import csv, hashlib, json, os, shutil
from pathlib import Path
import numpy as np

work=Path(os.environ['RS_CIRCLE_AUDIT']); root=work.parents[1]; name='jin_s14';
packet=root/'pan_inputs'/f'{name}.npz'; errors=root/'point_review'/f'{name}.npz'; state_before=root.parent/'lane4_side_1118/verified/state.npz'; state_after=root/'reviewed/state.npz'
setup=json.loads((root/'pan_inputs/setup.json').read_bytes())['shots'][name]
a=dict(np.load(packet)); c=dict(np.load(work/'measure.npz')); m=dict(np.load(work/'material_properties.npz')); q=dict(np.load(setup['source_packets']['8']['path']));
rows=c['rows']; sr=c['source_rows']; brackets=q['bracket'][sr]; u=a['uv'][rows]; dist=m['distance_lines']; sign=c['signed']; width=np.linalg.norm(brackets[:,1]-brackets[:,0],axis=1); chroma=m['chroma_delta']; contrast=m['darkness_delta']
# The ranges below were inspected on the source-only bands, then checked against
# the entire neighboring physical markings and their projected paths. Exclude
# crossings, reflections/ice texture, and uncertain width pairs from scored C0.
classes={
 'C_centre_straight': [(72510,72516),(72648,72662),(72703,72719),(72774,72791),(72841,72863),
     (72908,72926),(72978,72982),(73040,73065),(73118,73128),(73167,73194),(73236,73262),
     (73312,73332),(73399,73411),(73468,73483),(73545,73551),(73603,73604)],
 'B1_blue_straight': [(73424,73430),(73492,73501),(73552,73562),(73605,73625),(73677,73709),
     (73752,73764),(73771,73778),(73820,73831),(73836,73840),(73884,73886),(73951,73959),
     (73995,74002),(74029,74034)],
 'C0_verified_pair': [(72534,72539),(72541,72541),(72571,72573),(72575,72577),
     (72728,72730),(72799,72802),(72868,72869),(72927,72930),(72932,72934),
     (72987,72995),(73066,73070),(73341,73341),(73414,73414),(73422,73422)],
 'C0_arc_pair_uncertain': [(72533,72533),(72540,72540),(72570,72570),(72574,72574),
     (72578,72578),(72727,72727),(72931,72931),(72983,72986),(73129,73134),
     (73342,73342),(73412,73413),(73415,73421),(73423,73423)],
}
status=np.full(len(rows),'unqualified',dtype='U30'); reason=np.full(len(rows),'C0 material or both source edges unresolved',dtype='U95')
rationale={
  'C_centre_straight':'paired dark band follows C across the C0 intersection, continues straight beyond the arc',
  'B1_blue_straight':'paired dark/blue band follows B1 and stays apart from the near curved C0 branch',
  'C0_verified_pair':'paired grey source band follows the bent tip into near C0 arc, separate from C/B1',
  'C0_arc_pair_uncertain':'on visible near C0 arc, but oblique/too-wide brackets leave true midpoint uncertain',
}
for val, segments in classes.items():
 mask=np.logical_or.reduce([(rows>=start)&(rows<=stop) for start,stop in segments]); assert (status[mask]=='unqualified').all(); status[mask]=val; reason[mask]=rationale[val]
why=np.select([status=='C0_verified_pair',status=='C0_arc_pair_uncertain',np.isin(status,['C_centre_straight','B1_blue_straight'])],['retain_C0_check','exclude_pending_paired_reread','exclude_from_C0_check'],default='exclude_pending_source_identity')
by_frame={int(p):i//3+1 for i,p in enumerate(c['pts'])}; old=json.loads((root/'point_review'/f'{name}.json').read_bytes())
header=['packet_row_zero_based','source_stripe_row_zero_based','source_pts','time_base_num','time_base_den','source_seconds','idx','source_u','source_v','bracket1_u','bracket1_v','bracket2_u','bracket2_v','bracket_width_px','signed_C0_prev_px','signed_C0_reviewed_px','signed_approx_C_straight_y_px','signed_approx_B1_y_px','source_mid_minus_flanks_RminusB','source_darkness_against_flank1_dn','source_darkness_against_flank2_dn','classification','reason','C0_check_use','source_evidence']
with (work/'held_rows.csv').open('w',encoding='utf-8',newline='') as fo:
 wr=csv.writer(fo); wr.writerow(header)
 for i,r in enumerate(rows):
  pt=int(a['pts'][r]); wr.writerow((r,sr[i],pt,1,12800,f'{pt/12800:.3f}',a['idx'][r],*u[i],*brackets[i,0],*brackets[i,1],width[i],*sign[:,i],dist[i,0],dist[i,1],chroma[i,-1],*contrast[i,:2],status[i],reason[i],why[i],f'row_context_{by_frame[pt]}.png'))

source_read=np.genfromtxt(work/'normal_reread_candidates.csv',delimiter=',',names=True)
read_path=work/'c0_paired_source.csv'; shutil.copyfile(work/'normal_reread_candidates.csv',read_path)
assert len(set(zip(source_read['pts'],source_read['u'],source_read['v'])))==len(source_read)
sh=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert old['source_sha256']==setup['source_sha256']; assert old['packet_sha256']==sh(packet)
uv=np.c_[source_read['u'],source_read['v']]; edges=np.stack((np.c_[source_read['edge_inner_u'],source_read['edge_inner_v']],np.c_[source_read['edge_outer_u'],source_read['edge_outer_v']]),axis=1); normal=np.c_[source_read['outward_u'],source_read['outward_v']]; signed=np.c_[source_read['signed_previous_px'],source_read['signed_reviewed_px']]
assert np.allclose(uv,edges.mean(1),atol=1e-10,rtol=0) and np.allclose(np.linalg.norm(normal,axis=1),1.0,atol=1e-10,rtol=0)
length=np.linalg.norm(edges[:,1]-edges[:,0],axis=1); assert np.allclose((edges[:,1]-edges[:,0])/length[:,None],normal,rtol=0,atol=1e-10)
np.savez_compressed(work/'c0_paired_source.npz', pts=source_read['pts'].astype(np.int64), idx=source_read['s14_index'].astype(np.int64),
    uv=uv, bracket=edges, normal=normal, signed_previous_px=signed[:,0], signed_reviewed_px=signed[:,1],
    contrast_min_dn=source_read['contrast_min_dn'], source_tangent_samples=source_read['source_tangent_samples'].astype(np.int64),
    source_tangent_residual_px=source_read['source_tangent_residual_px'],
    time_base_num=np.array(1),time_base_den=np.array(12800), source_sha256=np.array(setup['source_sha256']),
    path=np.full(len(uv),'d78_material_144/path0'), role=np.full(len(uv),'source_review_only'),
    original_controls=np.isin(source_read['s14_index'],[70,168,182]))
cut=lambda q:np.percentile(np.asarray(q),[10,50,90]).round(3).tolist()
groups={}
for key in np.unique(status):
 mask=status==key; groups[key]=dict(rows=int(mask.sum()),exposures=int(len(np.unique(a['pts'][rows[mask]]))),signed_previous_p10_p50_p90=cut(sign[0,mask]),signed_reviewed_p10_p50_p90=cut(sign[1,mask]),abs_previous_p50=float(np.median(abs(sign[0,mask]))),abs_reviewed_p50=float(np.median(abs(sign[1,mask]))))
measures=dict(rows=len(source_read),exposures=len(np.unique(source_read['pts'])),first_s=float(source_read['pts'][0])/12800,last_s=float(source_read['pts'][-1])/12800,
    signed_previous_p10_p50_p90=cut(signed[:,0]),signed_reviewed_p10_p50_p90=cut(signed[:,1]),
    paired_growth_p10_p50_p90=cut(signed[:,1]-signed[:,0]),
    edge_pair_width_p10_p50_p90=cut(length),tangent_residual_px_p10_p50_p90=cut(source_read['source_tangent_residual_px']))
summary=dict(source={'path':setup['source'],'sha256':setup['source_sha256'],'time_base':[1,12800],'size':[1920,1080]},input_packet={'path':str(packet),'sha256':sh(packet)},
    row_errors={'path':str(errors),'sha256':sh(errors)},source_stripes={'path':setup['source_packets']['8']['path'],'sha256':sh(setup['source_packets']['8']['path'])},
    previous={'path':str(state_before),'sha256':sh(state_before)}, reviewed={'path':str(state_after),'sha256':sh(state_after)},
    reviewed_geometry_path='d78_material_144/path0', row_groups=groups, rereads=measures)
(work/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'groups':groups,'measures':measures},indent=2))
