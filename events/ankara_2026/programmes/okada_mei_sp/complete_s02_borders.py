"""Recover visible S02 panel borders through existing RGB profile readers."""
import collections
import functools
import gzip,json
import cv2
import numpy as np
import points_s02 as p
from ice_points_s04 import select_representatives


@functools.lru_cache(maxsize=600)
def motion(pt):
    return p.read(p.REC/f'dense_print_v1/source_anchor_motion/{pt}.json')


def transport(q,first,last):
    """Adjacent actual board features navigate the search; no transported UV exits."""
    q=np.asarray(q,float).copy();evidence=[];step=256 if last>first else -256
    for previous in range(first,last,step):
        pt=previous+step;r=motion(max(previous,pt))
        if r.get('cross_cut')or r.get('previous_native_pts')!=min(previous,pt):return None
        old=np.asarray(r.get('source_uv',[]),np.float32);new=np.asarray(r.get('current_uv',[]),np.float32)
        if old.ndim!=2 or old.shape!=new.shape or len(old)<12:return None
        if step<0:old,new=new,old
        lo=q.min(0)-[220,60];hi=q.max(0)+[220,60]
        keep=(old>=lo).all(1)&(old<=hi).all(1)&(np.asarray(r['forward_backward_error_px'])<.8)
        old,new=old[keep],new[keep]
        if len(old)<12:return None
        A,inlier=cv2.estimateAffine2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=2500,confidence=.999)
        if A is None or inlier.sum()<10 or inlier.mean()<.65:return None
        error=np.linalg.norm(cv2.transform(old[None],A)[0]-new,axis=1)
        q=cv2.transform(q[None],A)[0]
        evidence.append(dict(from_pts=previous,to_pts=pt,inliers=int(inlier.sum()),pairs=len(old),p95=float(np.percentile(error[inlier[:,0]>0],95))))
    return q,evidence


def run():
    from calibration.program_tracking.extract_selected_shot_board import reread_auxiliary_board_interfaces,actual_strip_interface_exclusion
    r,t,m,o,s,ats,_=p.ad_configure();cv2.setNumThreads(2);p.verify_authority()
    z=np.load(p.OUT/'ads/source_review/verified_ads.npz');mapping=p.read(p.OUT/'ads/source_review/verified_mapping.json')
    counts=collections.Counter((int(pt),mapping['path_ids'][i])for pt,i in zip(z['pts'],z['path_indices']))
    guides=collections.defaultdict(list)
    for atlas in ats:
        if not p.FIRST<=int(atlas.get('source_pts')or 0)<=p.LAST:continue
        groups=collections.defaultdict(list)
        for path in atlas['paths']:
            proof=path.get('proof',{})
            if proof.get('kind')!='earlier_actual_S02_readout_with_unique_current_asset_alias':continue
            if not any(x in path['path_id']for x in ['top_edge','bottom_edge']):continue
            pid='print_instance/'+proof['physical_instance']+'::'+path['path_id']
            if pid not in mapping['path_ids']:continue
            groups[pid].append(np.asarray(path['pixels'],float))
        for pid,parts in groups.items():
            guides[pid].append(dict(pts=atlas['source_pts'],q=np.vstack(parts),atlas_id=atlas['id']))
    folder=p.OUT/'evidence/reread_ad_borders';folder.mkdir(exist_ok=True);frames=[]
    history=p.OUT/'evidence/panel_gap_prior_review.jsonl.gz'
    if not history.exists():
        with gzip.open(history,'wt',encoding='utf8')as dest:
            for file in sorted(folder.glob('*.json')):
                if file.stem.isdigit() and (old:=p.read(file)).get('features'):dest.write(json.dumps(old,ensure_ascii=False)+'\n')
    @functools.lru_cache(maxsize=2)
    def body_interfaces(pt):
        board=p.read(p.REC/f'dense_board_v1/qualified_frames/{pt}.json');features=[]
        for role in ['top','contact']:
            features.append(dict(target_id=f'board.{role}.s02',segments=[dict(uv=board['roles']['board.'+role]['trace_uv'])]))
        im,_,mask=r.current(pt)
        return reread_auxiliary_board_interfaces(im,mask>0,features,dict(read_auxiliary_board_interfaces=['blue_cap_lower_body_interface','yellow_strip_upper_body_interface']))
    # Every missing named border is considered. A failed temporal identity is
    # explicit; a visible letter is not required to measure the panel itself.
    for frame,pt in enumerate(range(p.FIRST,p.LAST+1,256)):
        features={};failures={};im,gray,mask=r.current(pt);rejected=[]
        for pid,gg in guides.items():
            if counts[pt,pid]>=2:continue
            gg=sorted(gg,key=lambda v:abs(v['pts']-pt));guide=gg[0]
            carried=transport(guide['q'],guide['pts'],pt)
            if carried is None and pid.startswith('print_instance/Ad4::') and 592384<pt<600320:
                # Both exact S02 anchors name this same large panel. Its
                # blurred interior loses texture tracking before its perimeter
                # disappears. Interpolate search support only, then reread RGB.
                ends=[next(g for g in gg if g['pts']==stamp)for stamp in [592384,600320]]
                qs=[]
                for end in ends:
                    qq=end['q'];fit=np.polynomial.Polynomial.fit(qq[:,0],qq[:,1],1).convert()
                    xx=np.linspace(qq[:,0].min(),qq[:,0].max(),80);qs.append(np.c_[xx,fit(xx)])
                alpha=(pt-592384)/(600320-592384)
                carried=(qs[0]*(1-alpha)+qs[1]*alpha,[dict(source_bracket_pts=[592384,600320],source_atlases=[g['atlas_id']for g in ends],search_interpolation_only=True)])
            if carried is None and pid.startswith('print_instance/Ad2::') and pt in [608256,608512]:
                # The current image was visually inspected: the same purple
                # ISU panel lies to the right of Ad3. These are rough search
                # locations, not clicks, acceptance, or new geometry.
                yy=248. if 'top_edge' in pid else 371.
                carried=(np.array([[800.,yy],[1450.,yy]]),[dict(assistant_visual_navigation_pts=608256,
                    source_BGR_sha256=r.VIS[608256]['source_BGR_sha256'],approximate_search_only=True,user_accepted=False)])
            if carried is None:failures[pid]='adjacent_board_identity_support_not_established';continue
            support,history=carried
            if np.ptp(support[:,0])<30:continue
            line=np.polynomial.Polynomial.fit(support[:,0],support[:,1],1).convert().coef
            line=np.pad(line,(0,max(0,2-len(line))))
            xx=np.arange(max(6.,support[:,0].min()),min(1914.,support[:,0].max()),12.)
            if len(xx)<2:continue
            q=np.column_stack([xx,line[0]+line[1]*xx]);normal=np.array([-line[1],1.]);normal/=np.linalg.norm(normal)
            if 'top_edge' in pid:normal*=-1
            aux=body_interfaces(pt);top='top_edge'in pid
            identity='blue_cap_lower_body_interface'if top else'yellow_strip_upper_body_interface'
            off=np.arange(-24.,24.5,.5);v=r.sample(gray,q[:,None,:]+off[None,:,None]*normal)
            v=cv2.GaussianBlur(v.astype(np.float32),(5,1),.8);grad=np.diff(v,axis=1)
            positions=q[:,None,:]+((off[:-1]+off[1:])*.5)[None,:,None]*normal
            outside=actual_strip_interface_exclusion(aux,positions.reshape(-1,2),identity=identity,above=top).reshape(grad.shape)
            grad[outside]=-np.inf;peak=grad.argmax(1)
            search=q+((off[peak]+off[peak+1])*.5)[:,None]*normal
            readings=m.read_profiles(im,r.VIS[pt],search,np.tile(normal,(len(q),1)));records=[]
            for j,row in enumerate(readings):
                if m.profile_reason(row):continue
                uv=np.asarray(row['actual_source_uv']);offsets=np.asarray(row['source_q10_q50_q90_normal_offsets_px'])
                if actual_strip_interface_exclusion(aux,[uv],identity=identity,above=top)[0]:continue
                probe=np.array([offsets[0]-3,offsets[0]-1,offsets[2]+1,offsets[2]+3]);rgb=r.sample(im,(search[j]+probe[:,None]*normal)[None])[0]
                sides=np.stack([rgb[:2].mean(0),rgb[2:].mean(0)]);chroma=np.ptp(sides,axis=1)
                if not(chroma.max()>20 and chroma.min()<max(15.,chroma.max()*.60)):continue
                records.append(dict(uv=uv.tolist(),native_path_id=pid,pts=pt,source_row_id=f's02_panel_RGB/{pt}/{pid}/{j}',
                    source_evidence=dict(query_uv=search[j].tolist(),profile_normal=normal.tolist(),query_index=j,
                        source_q10_q50_q90_normal_offsets_px=offsets.tolist(),source_flags=row['source_flags'],
                        source_search_radius_px=row['source_search_radius_px'],actual_colored_panel_to_background_transition=True),
                    pixel_support=dict(normal=normal.tolist(),query_index=j)))
            if len(records)>=4:
                # A long panel border must also be a coherent material trace.
                # Fit only selects real measured support; no fitted UV exits.
                uv=np.asarray([v['uv']for v in records],np.float32)
                vx,vy,x,y=cv2.fitLine(uv,cv2.DIST_HUBER,0,.01,.01).ravel()
                error=np.abs((uv-[x,y])@[-vy,vx]);good=error<=4.
                rejected.extend(dict(v,reason='panel_boundary_branch_disagrees_with_whole_current_source_trace',normal_departure_px=float(e))for v,e,ok in zip(records,error,good)if not ok)
                records=[v for v,ok in zip(records,good)if ok]
            if records:
                keep,why=select_representatives(records,dict(kind='ice_segment'))
                features[pid]=dict(rows=[dict(records[j],selection_reasons=sorted(why[j]))for j in keep],
                    navigation=dict(original_source_atlas=guide['atlas_id'],source_pts=guide['pts'],adjacent_source_tracks=history,
                        only_search_locations=True,actual_RGB_material_required=True),dense_actual_readings=len(records))
        packet=dict(pts=pt,features=features,identity_unavailable=failures,rejected_source_candidates=rejected,
            strip_interface_exclusion='existing actual blue-lower and yellow-upper source material reader; no new world geometry',
            human_acceptance=False,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'])
        p.write(folder/f'{pt}.json',packet);frames.append(dict(pts=pt,rows=sum(len(v['rows'])for v in features.values()),paths=list(features)))
        if frame%100==0:p.emit('panel_gap_read',frames=frame+1,new_rows=sum(v['rows']for v in frames))
    p.write(folder/'coverage.json',dict(frames=frames));p.verify_authority();p.emit('panel_gap_complete',rows=sum(v['rows']for v in frames),frames=sum(v['rows']>0 for v in frames))


if __name__=='__main__':run()
