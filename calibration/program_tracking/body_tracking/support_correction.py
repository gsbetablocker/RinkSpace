"""Build a current-only whole-S05 source correction document and run/deliver."""
from pathlib import Path
import argparse,copy,json
from .rebuild_prepare import digest
from .fullbody_round import write_json
from .source_guided_refit import fit_phase

def document(stage,review,output):
    stage,review,output=map(Path,(stage,review,output));output.mkdir(parents=True,exist_ok=False)
    source=json.loads((stage/'source_evidence.json').read_text(encoding='utf-8'));source['body_proxy_review']=json.loads((review/'body_observation_review_1106.json').read_text(encoding='utf-8'))
    source['support_contact_source_review']=json.loads((review/'source_contact_review_1106.json').read_text(encoding='utf-8'))
    source['support_location_policy']=dict(mode='native_image_candidate',reason='source review: steel lower contours are not known Z=0 contact material; preserve candidate native band and finite episode ice, keep old planar footprints diagnostic',source_review_sha256=digest(review/'source_contact_review_1106.json'))
    source['root_prior_policy']=dict(camera_ray_depth_confidence=.04,reason='fixed finite anisotropic old-initial prior; FS complete-trial ray-depth ambiguity and current source direction conflict; transverse and actual-motion priors retained',camera_or_scale_changed=False)
    groups={}
    for row in source['full_source_skate_structures']['rows']:groups.setdefault((row['pts'],row['foot'],row['source_structure_feature']),[]).append(row)
    reverse=[]
    for key,rows in groups.items():
        row=rows[0]
        if key[2]not in ('blade_assembly_outline','rear_extension')or row['completeness']!='complete'or row['visibility']!='visible':continue
        item=copy.deepcopy(row);item['id']='complete_structure_'+str(key[0])+'_'+str(key[1])+'_'+key[2];item['structure_feature']=key[2];item['polyline_uv']=copy.deepcopy(row['source_annotation']['points_uv'])
        if row['source_annotation'].get('closed')and item['polyline_uv'][0]!=item['polyline_uv'][-1]:item['polyline_uv'].append(item['polyline_uv'][0])
        item['feature_budget_fraction']=sum(r['feature_budget_fraction']for r in rows)*.5
        for forward in rows:forward['feature_budget_fraction']*=.5
        reverse.append(item)
    source['complete_visible_structure_extent']=dict(rows=reverse,samples_per_complete_feature=64,source_role='reverse share of same original complete feature, never independent evidence')
    source['support_correction_revision']=dict(whole_seconds=[76.,95.],whole_exposures=951,new_independent_source_exposures=0,no_source_radius_changed=True,no_shared_shape_or_camera_change=True,complete_structure_reverse_rows=len(reverse),body_review_sha256=digest(review/'body_observation_review_1106.json'))
    path=output/'source_evidence.json';write_json(path,source);return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--parameter',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--wall-seconds',type=float,default=1800.);ap.add_argument('--max-accepted',type=int,default=60);a=ap.parse_args()
    path=document(a.stage,a.root/'source_review',a.root/'inputs')
    fit_phase(a.project,a.stage,a.parameter,a.root/'joint',json.loads(path.read_text(encoding='utf-8')),'support_joint',a.wall_seconds,a.max_accepted,parameter_stem='spin_support_correction_1106',compare_parameters=[a.parameter])

if __name__=='__main__':main()
