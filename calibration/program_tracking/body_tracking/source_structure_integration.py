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


def _load_contracts(path, source):
    """Match the reviewed declarations to unchanged native source readings."""
    if path is None:return {},None
    raw=Path(path).read_bytes();document=json.loads(raw.decode('utf-8-sig'))
    if document.get('schema')!='rinkspace.known_foot_source_contour_contract_review.v1':
        raise ValueError('Unsupported explicit contour contract schema')
    if document.get('source_video',{}).get('sha256')!=source['source_sha256']:
        raise ValueError('Contour contract source identity mismatch')
    if document.get('source_time_base')!=source['time_base'] or document.get('native_size')!=source.get('native_size'):
        raise ValueError('Contour contract native clock/pixel domain mismatch')
    frames={(int(f['pts']),int(f['foot'])):f for f in source['frames']}
    index={}
    for contract in document['rows']:
        key=(int(contract['pts']),int(contract['foot']),contract['feature'])
        if key in index:raise ValueError('Duplicate contour contract: '+str(key))
        frame=frames.get(key[:2]);feature=None if frame is None else frame['features'].get(key[2])
        if feature is None:raise ValueError('Contour contract has no original feature: '+str(key))
        region=contract.get('source_range',{})
        original_range=[0,len(feature['points_uv'])-1]if feature['points_uv']else None
        if (region.get('index_base')!=0 or region.get('original_index_range_inclusive')!=original_range
                or region.get('points_uv_original')!=feature['points_uv']
                or region.get('pixel_uncertainty_radius_px_original')!=feature['pixel_uncertainty_radius_px']
                or region.get('original_status')!=feature['status']
                or region.get('source_closed')!=bool(feature.get('closed',False))):
            raise ValueError('Contour contract changed original points/radius/status/range: '+str(key))
        if contract.get('evidence',{}).get('source_crop_sha256')!=frame['source_crop_sha256']:
            raise ValueError('Contour contract crop identity mismatch: '+str(key))
        if contract.get('visibility')not in('visible','partial','occluded','unknown')or contract.get('completeness')not in('complete','partial','unknown'):
            raise ValueError('Explicit contour visibility/completeness required: '+str(key))
        for note in contract.get('local_structural_notes',[]):
            bounds=note.get('index_range_inclusive')
            if bounds is not None and(not isinstance(bounds,list)or len(bounds)!=2 or any(type(v)is not int for v in bounds)
                    or not 0<=bounds[0]<=bounds[1]<len(feature['points_uv'])):
                raise ValueError('Invalid explicit local structural indices: '+str(key))
        index[key]=contract
    return index,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),schema=document.get('schema'),
        source_file=document.get('source_file'),source_file_sha256=document.get('source_file_sha256'),human_confirmed=document.get('human_confirmed',False))


def _attach_contract(row,frame,feature,name,contract,provenance,track_contract=None):
    """Keep raw evidence and finite appearance ranges; never invent material ends."""
    row.setdefault('source_annotation',copy.deepcopy(feature))
    row.setdefault('source_crop_sha256',frame.get('source_crop_sha256'))
    identity=contract or track_contract or {}
    track=frame.get('source_track_id',identity.get('source_track_id'))
    row.update(source_structure_feature=name,source_track_id=track,
        source_track_id_provenance=identity.get('source_track_id_provenance'),
        source_identity_provenance=identity.get('identity_provenance'),
        source_anatomical_foot=frame.get('foot'),
        visibility=(contract or {}).get('visibility','unknown'),completeness=(contract or {}).get('completeness','unknown'),
        source_order=(contract or {}).get('source_order','unknown'),
        source_range=copy.deepcopy((contract or {}).get('source_range',dict(index_base=0,
            original_index_range_inclusive=[0,len(feature['points_uv'])-1],points_uv_original=feature['points_uv'],
            pixel_uncertainty_radius_px_original=feature['pixel_uncertainty_radius_px'],original_status=feature['status'],source_closed=bool(feature.get('closed',False))))),
        source_regions=copy.deepcopy((contract or {}).get('source_regions',dict(original_uncertainty_radius_px=feature['pixel_uncertainty_radius_px'],silhouette_terminals=None))),
        source_endpoints=copy.deepcopy((contract or {}).get('endpoints',dict(material_front=None,material_rear=None))),
        source_visible_side=copy.deepcopy((contract or {}).get('visible_side',dict(value='unknown'))),
        local_structural_notes=copy.deepcopy((contract or {}).get('local_structural_notes',[])),
        source_contour_contract=copy.deepcopy(contract),source_contract_matched=contract is not None,source_contour_contract_provenance=copy.deepcopy(provenance),
        source_contract_sha256=None if provenance is None else provenance['sha256'],
        points_uv_original=copy.deepcopy(feature['points_uv']),pixel_uncertainty_radius_px_original=feature['pixel_uncertainty_radius_px'],
        annotation_provenance=copy.deepcopy(frame.get('annotation_provenance')),
        source_frame_metadata=copy.deepcopy({k:v for k,v in frame.items()if k!='features'}),
        source_budget_group_id=row['id'],source_feature_budget_fraction_original=row['feature_budget_fraction'])
    return row


def _note_feature(note):
    """Only explicit component labels, including the reviewed Chinese labels.

    Mixed post/tail or opening notes remain assembly; no geometric guessing.
    The free text is retained verbatim alongside its original index range.
    """
    explicit=note.get('structure_feature')
    if explicit is not None:return explicit if explicit in('boot_sole','heel_block','runner_lower_edge')else None
    # Exact reviewed declarations only; unseen prose stays an assembly candidate.
    return {
        '与原 boot_sole 鞋底轮廓重合；不能按钢刃外缘匹配。':'boot_sole',
        '鞋跟块及其与后支架连接候选；不是连续钢片下缘。':'heel_block',
        '鞋跟暗块及其与支架接界；不是钢片外缘。':'heel_block',
        '刀体外部下缘的连续外观范围。':'runner_lower_edge',
        '刀带外部下缘的连续外观范围。':'runner_lower_edge',
    }.get(note.get('meaning_zh',''))


def _split_assembly_row(row):
    """Partition already-retained source edges, conserving the assembly budget."""
    if row['source_structure_feature']!='blade_assembly_outline':return [row]
    notes=[(i,n,_note_feature(n))for i,n in enumerate(row['local_structural_notes'])
           if n.get('index_range_inclusive')is not None and _note_feature(n)is not None]
    if not notes:return [row]
    uv=np.asarray(row['polyline_uv'],float);ids=list(row['source_retained_point_indices'])
    if len(uv)==len(ids)+1:ids.append(ids[0])
    if len(uv)!=len(ids):raise ValueError('Assembly source index path mismatch')
    labels=[]
    for a,b in zip(ids,ids[1:]):
        matches=[(i,label)for i,n,label in notes
                 if a<b and n['index_range_inclusive'][0]<=a<b<=n['index_range_inclusive'][1]]
        if len({label for _,label in matches})>1:raise ValueError('Conflicting structural component ranges')
        labels.append((matches[0][1],tuple(i for i,_ in matches))if matches else('blade_assembly_outline',()))
    lengths=np.linalg.norm(np.diff(uv,axis=0),axis=1);total=float(lengths.sum())
    rows=[];start=0
    for end in range(1,len(labels)+1):
        if end<len(labels)and labels[end]==labels[start]:continue
        label,note_ids=labels[start];piece=copy.deepcopy(row);share=float(lengths[start:end].sum()/total)
        piece.update(id=row['id']+f'_component{len(rows)}_{label}',structure_feature=label,
            polyline_uv=uv[start:end+1].tolist(),source_retained_point_indices=ids[start:end+1],
            source_component_note_indices=list(note_ids),source_component_budget_share=share,
            feature_budget_fraction=row['feature_budget_fraction']*share)
        rows.append(piece);start=end
    return rows


def integration(structures,contacts,output=None,*,frame_weight_budget=3.2,location_budget_fraction=.25,contracts=None):
    """Build a new document; reverse admission requires an explicit contract path.

    Existing calls remain valid. output=None returns only an in-memory document.
    Appearance terminals are finite source regions, never fixed material points.
    """
    source=read(structures);mapped=read(contacts);rows=[];admissions=[];excluded=[]
    if source['source_sha256']!=mapped['source_sha256']:raise ValueError('Source identity mismatch')
    contract_index,contract_provenance=_load_contracts(contracts,source)
    track_contracts={}
    for (pts,foot,_),contract in contract_index.items():track_contracts.setdefault((pts,foot),contract)
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
            row=dict(id=f'full_source_{pts}_foot{foot}_{name}',pts=pts,foot=foot,role='fit',optimization_enabled=True,evidence_status='observed',
                source_status=feature['status'],structure_feature=name,polyline_uv=points.tolist(),pixel_uncertainty_radius_px=feature['pixel_uncertainty_radius_px'],
                contour_frame_budget=frame_weight_budget*(1-location_budget_fraction if supported else 1.),feature_budget_fraction=FEATURE_WEIGHTS[name]/total,
                source_annotation=feature,source_retained_point_indices=ids,source_crop_sha256=frame['source_crop_sha256'],
                source_admission='finite directly localized source region; not fixed template material correspondence',bounded_ambiguous_boundary=feature['status']=='ambiguous')
            _attach_contract(row,frame,feature,name,contract_index.get((pts,foot,name)),contract_provenance,track_contracts.get((pts,foot)))
            rows.extend(_split_assembly_row(row))
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
                    row=dict(id=f'unassigned_{pts}_{frame.get("source_track_id",frame.get("foot_proposal"))}_{name}',pts=pts,foot=foot,role='fit',optimization_enabled=True,evidence_status='observed',source_status=feature['status'],structure_feature=name,polyline_uv=points.tolist(),pixel_uncertainty_radius_px=feature['pixel_uncertainty_radius_px'],contour_frame_budget=frame_weight_budget,feature_budget_fraction=FEATURE_WEIGHTS[name]/total,source_track_id=frame.get('source_track_id'),source_anatomical_foot=None,source_assignment='latent_one_to_one_set_match_not_source_identity')
                    _attach_contract(row,frame,feature,name,None,contract_provenance)
                    row['source_anatomical_foot']=None
                    branch.append(row)
            branches.append(dict(rows=branch))
        if branches[0]['rows']and len(branches[0]['rows'])==len(branches[1]['rows']):unassigned.append(dict(pts=pts,branches=branches))
    extent=[]
    for row in rows:
        contract=row.get('source_contour_contract')or{}
        if(row['source_structure_feature']=='runner_lower_edge'and row['visibility']=='visible'
                and row['completeness']=='complete'and contract.get('role')=='fit'
                and contract.get('optimization_enabled',True)):
            row['feature_budget_fraction']*=.5;row['bidirectional_budget_share']=.5
            reverse=copy.deepcopy(row);reverse.update(id=row['id']+'_complete_extent',complete_visible_runner=True)
            extent.append(reverse)
    d=dict(schema='rinkspace.full_source_structure_integration.v1',source_sha256=source['source_sha256'],time_base=source['time_base'],
        full_source_skate_structures=dict(rows=rows,source_document_path=str(structures),source_document_sha256=sha(structures)),
        source_support_locations=mapped,source_support_evidence=admissions,
        source_structure_budget=dict(frame_weight_budget=frame_weight_budget,location_budget_fraction=location_budget_fraction),
        unassigned_source_skate_sets=unassigned,
        complete_runner_extent=dict(rows=extent,scope='explicit visible+complete lower-runner contracts only; never filename/status admission',same_original_feature_budget_split=True),
        source_contour_contract_provenance=contract_provenance,
        excluded_features=excluded,unassigned_identity_observations=source.get('unassigned_identity_observations',[]),
        observation_semantics='Complete2D source contours and camera-conditional XY share a finite per-foot/per-exposure budget; not independent3D observations',
        old_review_source_unchanged=True,acceptance='not visually accepted')
    if contract_index:
        d['full_source_skate_structures']['association_policy']=dict(model_visibility='view_visible_exterior',freeze_inner=True,arc_length_measure=True)
        d['complete_runner_extent']['association_policy']=dict(model_visibility='near_visible_runner_branch',freeze_inner=True,arc_length_measure=True)
        # Known-foot contracts do not authorize identities or visibility for
        # unassigned source sets; leave their branch policies unchanged.
    if output is not None:write(output,d)
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','integrate']);p.add_argument('--structures',nargs='+',type=Path,required=True);p.add_argument('--contacts',type=Path);p.add_argument('--contracts',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=collect(a.structures,a.output)if a.action=='collect'else integration(a.structures[0],a.contacts,a.output,contracts=a.contracts)
    print(json.dumps(dict(output=str(a.output),frames=len(d.get('frames',[])),fit_feature_rows=len(d.get('full_source_skate_structures',{}).get('rows',[])),support_locations=len(d.get('source_support_evidence',[])))))
if __name__=='__main__':main()
