"""Admit existing source-reviewed free-shoe regions without inventing contact.

Reads the current support document and its immutable source review, writes a
NEW document. It never edits the inputs or changes the old support rows.
No network, FK, pose fitting or tracking is run. All reliability values and
the anatomical association are explicit caller decisions, not probabilities.

Adapted from relations/admit_free_shoe_regions.py in the supplied
RinkSpace_3d6f51f_Audit_Integration package. Its source hashes, evidence-role
preservation and admission semantics are retained. Registered input/output
paths are resolved through the shared AssetStore.
"""
from __future__ import annotations
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_document(document,review,*,free_foot,association_provenance,
                   partial_reliability,blurred_reliability,objective_weight,
                   source_frame_directory):
    if isinstance(free_foot,bool) or free_foot not in (0,1):raise ValueError('Explicit free foot 0/1 required')
    if not association_provenance:raise ValueError('Source association provenance is required')
    for x in (partial_reliability,blurred_reliability):
        if not np.isfinite(x) or not 0<x<=1:raise ValueError('Explicit finite reliability in (0,1] required')
    if not np.isfinite(objective_weight) or objective_weight<=0:raise ValueError('Positive objective weight required')
    if document['source_sha256']!=review['source_sha256'] or document['shot_id']!=review['shot_id'] or document['time_base']!=review['time_base']:
        raise ValueError('Source/shot/time base mismatch')
    if document.get('support_foot')==free_foot:raise ValueError('Free and support track cannot silently share a foot')
    out=copy.deepcopy(document);existing={r['id'] for r in out['observations']}
    before=len(out['observations']);added=[];skipped=[]
    source_frame_directory=Path(source_frame_directory)
    for row in review['rows']:
        if not isinstance(row['pts'],int) or isinstance(row['pts'],bool):raise ValueError('Integer source PTS required')
        free=row.get('free',{});vis=free.get('visibility','unknown')
        if vis not in ('visible','partially_visible','visible_blurred'):
            skipped.append({'pts':row['pts'],'reason':'free track visibility '+vis});continue
        if free.get('contact_state')!='airborne':
            raise ValueError('This importer requires explicitly source-labelled airborne free-shoe rows; do not invent or overwrite contact state')
        if row['role'] not in ('fit','review'):raise ValueError('Unknown source role')
        frame=source_frame_directory/(str(row['pts'])+'.png')
        if not frame.is_file():raise FileNotFoundError(frame)
        frame_sha=digest(frame)
        old=[x for x in document['observations'] if x['pts']==row['pts']]
        if not old:raise ValueError('This source exposure is not in the existing support document')
        if any(x['role']!=row['role'] for x in old):raise ValueError('Review/fit role conflict')
        if any(x['image_footprint']['source_frame_sha256']!=frame_sha for x in old):raise ValueError('Source frame SHA mismatch')
        for feature,ids in [('shoe_body',list(range(14))),('toe',[1,8]),('heel',[0,7])]:
            region=free.get(feature)
            if not region:continue
            rv=region.get('visibility',vis)
            if rv not in ('visible','partially_visible','visible_blurred','coarse'):
                skipped.append({'pts':row['pts'],'feature':feature,'reason':rv});continue
            c=np.asarray(region['center_uv'],float);half=np.asarray(region['half_extent_uv'],float)
            if c.shape!=(2,) or half.shape!=(2,) or not np.isfinite(c).all() or not np.isfinite(half).all() or np.any(half<=0):raise ValueError('Invalid source rectangle')
            if region.get('rotation_deg',0)!=0:raise ValueError('Adapter currently accepts axis-aligned reviewed rectangles')
            rid=f"spin_{row['pts']}_free_{feature}"
            if rid in existing:raise ValueError('Free-shoe rows already exist; do not duplicate their evidence')
            reliability=blurred_reliability if vis=='visible_blurred' else partial_reliability
            new=dict(id=rid,source_id=document['source_id'],source_sha256=document['source_sha256'],
                shot_id=document['shot_id'],pts=row['pts'],time_base=copy.deepcopy(document['time_base']),
                foot=free_foot,foot_track='free_track',kind='boot_marker_region',
                centre_uv=c.tolist(),region_shape_px2=np.diag(2*half**2).tolist(),
                visibility='visible',source_feature_visibility=rv,
                contact_state='airborne',role=row['role'],reliability=float(reliability),objective_weight=float(objective_weight),
                provenance='existing source-reviewed free shoe; '+association_provenance,
                image_footprint=dict(source_frame=str(frame),source_frame_sha256=frame_sha,
                    region_rectangle_half_extent_uv=half.tolist(),native_dimensions=[1920,1080]),
                feature=feature,adapter_point_ids=ids,
                source_uncertainty=region.get('reason','coarse/partially visible source shoe region; no exact material point'),
                association=association_provenance,human_confirmed=False)
            out['observations'].append(new);added.append(rid);existing.add(rid)
    if not added:raise ValueError('No actual source-reviewed free-shoe regions admitted')
    out['free_shoe_admission']=dict(added_rows=len(added),added_ids=added,free_foot=free_foot,
        association_provenance=association_provenance,skipped=skipped,
        role_preserved=True,new_ice_support_rows=0,new_grip_truth_rows=0,
        original_support_count=len(document.get('supports',[])),original_observation_count=before,
        note='Free-shoe pixels are distinct from a held hand/shoe relation; no contact point or interval is inferred here.')
    assert out.get('supports',[])==document.get('supports',[])
    assert out['observations'][:before]==document['observations']
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support-document',type=Path,required=True)
    p.add_argument('--source-review',type=Path,required=True)
    p.add_argument('--source-frame-directory',type=Path,required=True)
    p.add_argument('--free-foot',type=int,choices=[0,1],required=True)
    p.add_argument('--association-provenance',required=True)
    p.add_argument('--partial-reliability',type=float,required=True)
    p.add_argument('--blurred-reliability',type=float,required=True)
    p.add_argument('--objective-weight',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--data-root',type=Path)
    a=p.parse_args(argv)
    store=AssetStore(a.project,data_root=a.data_root)
    if not store.configured:raise ValueError('A registered Data root is required')
    a.support_document=store.resolve(a.support_document,must_exist=True)
    a.source_review=store.resolve(a.source_review,base=store.data,must_exist=True)
    a.source_frame_directory=store.resolve(a.source_frame_directory,base=store.data,must_exist=True)
    a.output=guard_project_write(store.resolve(a.output,base=store.data),project_root=store.project)
    if not a.output.is_relative_to(store.data):raise ValueError('Output must stay in registered Data')
    if a.output.exists():raise FileExistsError('Use a new output path; preserve old evidence')
    doc=json.loads(a.support_document.read_text(encoding='utf-8'))
    review=json.loads(a.source_review.read_text(encoding='utf-8'))
    result=build_document(doc,review,free_foot=a.free_foot,association_provenance=a.association_provenance,
        partial_reliability=a.partial_reliability,blurred_reliability=a.blurred_reliability,
        objective_weight=a.objective_weight,source_frame_directory=a.source_frame_directory)
    result['free_shoe_admission']['input_sha256']={str(a.support_document):digest(a.support_document),str(a.source_review):digest(a.source_review)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['free_shoe_admission'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
