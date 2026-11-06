"""Explicit current fit admission for source skate structures and support XY."""
from pathlib import Path
import argparse,hashlib,json
import copy,itertools
import numpy as np

FEATURE_WEIGHTS={'runner_lower_edge':.35,'runner_upper_edge':.15,'blade_assembly_outline':.15,
 'toe_pick':.10,'rear_extension':.10,'boot_sole':.05,'heel_block':.05,'front_stanchion':.025,'rear_stanchion':.025}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def collect(paths,output):
    """Preserve original records; new task admission never edits old reviews."""
    docs=[read(p)for p in paths];base=docs[0];frames={};unassigned=[];skipped=[]
    for path,doc in zip(paths,docs):
        for raw in doc['frames']:
            row=dict(raw);pts=int(row['pts']);foot=row.get('foot')
            row['annotation_provenance']=dict(path=str(path),sha256=sha(path))
            if foot not in (0,1):unassigned.append(row);continue
            key=(pts,foot)
            if key in frames:raise ValueError('Do not overwrite source structures: '+str(key))
            for feature in row['features'].values():
                points=np.asarray(feature.get('points_uv',[]),float)
                if points.size and(points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all()):raise ValueError('Nonfinite source points')
            support=row.get('source_support')
            if support is None:
                # These22 were directly read with +/- one native exposure in
                # the preceding source-only task; never infer other supports.
                if path!=paths[0] or not(81.62<=pts/12800.<=92.44)or foot!=1:raise ValueError('New support needs source sequence evidence')
                support=dict(supported=True,evidence_status='source_sequence_review',context_source_pts=row['context_source_pts'],description_zh='既有22帧完整源图及前后原生曝光：右脚在落冰后连续支撑旋转，钢刃/冰面交界另保留有限像素范围。')
            row['source_support']=support;row['historical_role']=row.get('role');row['role']='fit';row['optimization_enabled']=True
            row['current_admission']='User requested these complete source observations enter the continuous S05 fit; finite source ranges and ambiguous local boundaries retained'
            frames[key]=row
        skipped.extend(doc.get('ambiguity_frames',[]))
    result={k:base[k]for k in ['schema','source_sha256','source_video','time_base','native_size']}
    result.update(frames=[frames[k]for k in sorted(frames)],unassigned_identity_observations=unassigned,source_only=True,model_projection_used=False,
        current_fit_admission=True,fit_run=False,accuracy_accepted=False,annotation_sources=[dict(path=str(p),sha256=sha(p))for p in paths],ambiguity_frames=skipped)
    write(output,result)
    return result


def integration(structures,contacts,output,*,frame_weight_budget=3.2,location_budget_fraction=.25):
    source=read(structures);mapped=read(contacts);rows=[];admissions=[];excluded=[]
    if source['source_sha256']!=mapped['source_sha256']:raise ValueError('Source identity mismatch')
    mapped_keys={(r['pts'],r['foot'])for r in mapped['frames']if 'ice_interface'in r['features']}
    for frame in source['frames']:
        pts=int(frame['pts']);foot=int(frame['foot']);support=frame['source_support'];supported=support.get('supported')is True and'ice_interface'in frame['features']
        if supported:
            context=support.get('context_source_pts',frame.get('context_source_pts',[]))
            if len(context)<3 or not min(context)<pts<max(context):raise ValueError('Need actually reviewed surrounding support source PTS: '+str((pts,foot)))
            if(pts,foot)not in mapped_keys:raise ValueError('Admitted support mapping missing')
            admissions.append(dict(pts=pts,foot=foot,role='fit',optimization_enabled=True,phase='source_supported',context_source_pts=context,support_identity_evidence=support['description_zh']))
        valid={}
        for name,feature in frame['features'].items():
            points=np.asarray(feature.get('points_uv',[]),float)
            if name not in FEATURE_WEIGHTS or feature.get('status')not in('observed','ambiguous')or len(points)<2:
                excluded.append(dict(pts=pts,foot=foot,feature=name,reason='no corresponding bound geometry or no localized visible source boundary'));continue
            # Drop only repeated/subpixel-neighbour readings, retaining the
            # untouched full source record and indices of all retained UVs.
            ids=[0]
            for i in range(1,len(points)):
                if np.linalg.norm(points[i]-points[ids[-1]])>=1.:ids.append(i)
            if len(ids)<2:continue
            points=points[ids]
            if feature.get('closed')and np.linalg.norm(points[-1]-points[0])>=1.:points=np.r_[points,points[:1]]
            valid[name]=(feature,points,ids)
        total=sum(FEATURE_WEIGHTS[k]for k in valid)
        for name,(feature,points,ids)in valid.items():
            rows.append(dict(id=f'full_source_{pts}_foot{foot}_{name}',pts=pts,foot=foot,role='fit',optimization_enabled=True,evidence_status='observed',
                source_status=feature['status'],structure_feature=name,polyline_uv=points.tolist(),pixel_uncertainty_radius_px=feature['pixel_uncertainty_radius_px'],
                contour_frame_budget=frame_weight_budget*(1-location_budget_fraction if supported else 1.),feature_budget_fraction=FEATURE_WEIGHTS[name]/total,
                source_annotation=feature,source_retained_point_indices=ids,source_crop_sha256=frame['source_crop_sha256'],
                source_admission='finite directly localized source region; not fixed template material correspondence',bounded_ambiguous_boundary=feature['status']=='ambiguous'))
    unassigned=[]
    grouped={}
    for frame in source.get('unassigned_identity_observations',[]):grouped.setdefault(int(frame['pts']),[]).append(frame)
    for pts,frames in sorted(grouped.items()):
        if not 1<=len(frames)<=2:
            excluded.append(dict(pts=pts,feature='unassigned_source_set',reason='Visible source objects exceed the two physical skates'));continue
        branches=[]
        for assignment in itertools.permutations((0,1),len(frames)):
            branch=[]
            for frame,foot in zip(frames,assignment):
                valid={k:f for k,f in frame['features'].items()if k in FEATURE_WEIGHTS and f.get('status')in('observed','ambiguous')and len(f.get('points_uv',[]))>=2}
                total=sum(FEATURE_WEIGHTS[k]for k in valid)
                for name,feature in valid.items():
                    points=np.asarray(feature['points_uv'],float);ids=[0]
                    for i in range(1,len(points)):
                        if np.linalg.norm(points[i]-points[ids[-1]])>=1.:ids.append(i)
                    points=points[ids]
                    if len(points)<2:continue
                    if feature.get('closed')and np.linalg.norm(points[-1]-points[0])>=1.:points=np.r_[points,points[:1]]
                    branch.append(dict(id=f'unassigned_{pts}_{frame.get("source_track_id",frame.get("foot_proposal"))}_{name}',pts=pts,foot=foot,role='fit',optimization_enabled=True,evidence_status='observed',source_status=feature['status'],structure_feature=name,polyline_uv=points.tolist(),pixel_uncertainty_radius_px=feature['pixel_uncertainty_radius_px'],contour_frame_budget=frame_weight_budget,feature_budget_fraction=FEATURE_WEIGHTS[name]/total,source_track_id=frame.get('source_track_id'),source_anatomical_foot=None,source_assignment='latent_one_to_one_set_match_not_source_identity'))
            branches.append(dict(rows=branch))
        if branches[0]['rows']and len(branches[0]['rows'])==len(branches[1]['rows']):unassigned.append(dict(pts=pts,branches=branches))
    extent=[]
    complete_keys={(int(r['pts']),int(r['foot']))for r in source['frames']if Path(r['annotation_provenance']['path']).name=='annotations_combined.json'}
    for row in rows:
        if row['structure_feature']=='runner_lower_edge'and(row['pts'],row['foot'])in complete_keys and row['source_status']=='observed':
            row['feature_budget_fraction']*=.5
            reverse=copy.deepcopy(row);reverse.update(id=row['id']+'_complete_extent',complete_visible_runner=True,visibility='complete',bidirectional_budget_share=.5)
            extent.append(reverse)
    d=dict(schema='rinkspace.full_source_structure_integration.v1',source_sha256=source['source_sha256'],time_base=source['time_base'],
        full_source_skate_structures=dict(rows=rows,source_document_path=str(structures),source_document_sha256=sha(structures)),
        source_support_locations=mapped,source_support_evidence=admissions,
        source_structure_budget=dict(frame_weight_budget=frame_weight_budget,location_budget_fraction=location_budget_fraction),
        unassigned_source_skate_sets=unassigned,
        complete_runner_extent=dict(rows=extent,scope='original22 complete visible lower runners only',same_original_feature_budget_split=True),
        excluded_features=excluded,unassigned_identity_observations=source.get('unassigned_identity_observations',[]),
        observation_semantics='Complete2D source contours and camera-conditional XY share a finite per-foot/per-exposure budget; not independent3D observations',
        old_review_source_unchanged=True,acceptance='not visually accepted')
    write(output,d);return d


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','integrate']);p.add_argument('--structures',nargs='+',type=Path,required=True);p.add_argument('--contacts',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=collect(a.structures,a.output)if a.action=='collect'else integration(a.structures[0],a.contacts,a.output)
    print(json.dumps(dict(output=str(a.output),frames=len(d.get('frames',[])),fit_feature_rows=len(d.get('full_source_skate_structures',{}).get('rows',[])),support_locations=len(d.get('source_support_evidence',[])))))
if __name__=='__main__':main()
