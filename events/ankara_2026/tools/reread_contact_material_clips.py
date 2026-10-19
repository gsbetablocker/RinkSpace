"""Stream two clock-bound RGB material clips; no NN, camera or source changes."""
from __future__ import annotations
import argparse,gzip,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rinkspace.vision import confirmed_exposure_groups,load_source_clock_registry
from events.ankara_2026.tools.read_registered_human_source_shots import NativeDecoder,plain,sha,SOURCE_SHA,sparse_original_points
from events.ankara_2026.tools.reread_old_boundary_material import contact_material

def write(path,data):Path(path).write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def emit(stream,data):stream.write(json.dumps(plain(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-row-index',type=Path,required=True);p.add_argument('--authority',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args()
    if o.output.exists()or not o.output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data')):raise ValueError('New Data output required')
    o.output.mkdir(parents=True);torch.set_num_threads(2)
    if sha(o.old_row_index)!='aef268b0f9b046c7644f665eff32dc769fbf6237a798ab91a6bd5dc46833000f':raise ValueError('Old index differs')
    if sha(o.authority)!='1c17877c4746c910c3e25353cc02ce9c53714ec49b2eb48525146a04a2e795f7':raise ValueError('Human authority differs')
    old=json.loads(o.old_row_index.read_text(encoding='utf8'));authority=json.loads(o.authority.read_text(encoding='utf8'))
    full=json.loads((o.authority.parent/'human_registered_full_stroke_hard_guard_v1.json').read_text(encoding='utf8'))
    ref=next(r for r in authority['records']if r['curve']['kind']=='ice_board_contact');frow=next(r for r in full['records']if r['id']==ref['id'])
    if sha(full['guard_arrays_reference']['path'])!='5abd423e11a6e1d28f0d69ac41498c8336300b1bad20fd6ef885354d6add6e51':raise ValueError('Human full arrays differ')
    with np.load(full['guard_arrays_reference']['path'],allow_pickle=False)as z:human,indices=sparse_original_points(z[frow['array_prefix']+'_uv'])
    mapping,clock=confirmed_exposure_groups(SOURCE_SHA,'1/15360',registry=load_source_clock_registry(ROOT/'events/ankara_2026/model_constraints/source_clock_registry.json'))
    allactual=sorted(set(int(row['canonical'])for row in mapping.values()))
    firstold=next(r for r in old['records']if r['shot']=='shot2'and r['group_actual_pts']==229497600 and r['semantic_role']=='ice_board_contact')
    secondseed=np.asarray(firstold['original_uv'],float);secondseed=secondseed[np.argsort(secondseed[:,0],kind='stable')]
    ranges=[dict(name='shot5_existing_nearest_contact_bracket',shot='shot5',start=229992704,stop=next(p for p in allactual if p>230076416),
        seed=human,seed_ref_pts=230024192,reference_scope='direct_human_source_hint; only its own exact exposure is human truth'),
        dict(name='shot2_complete_source_shot',shot='shot2',start=229497600,stop=229671680,seed=secondseed,
        seed_ref_pts=229497600,reference_scope='frozen historical UV navigation only; not human truth')]
    report=dict(status='running',source_video_sha256=SOURCE_SHA,clock=clock,old_source_index_sha256=sha(o.old_row_index),
        human_authority_sha256=sha(o.authority),human_arrays_sha256=sha(full['guard_arrays_reference']['path']),
        no_NN_forward_or_training=True,no_camera_or_world_projection=True,
        source_column_curve_search_not_point_correspondence_tracking=True,
        every_actual_exposure_has_fresh_RGB_material_read=True,coarse_or_previous_UV_is_navigation_only=True,
        material_hypotheses_not_promoted_to_GT=True,world_facet_association_pending=True,source_packets_or_active_inputs_changed=False,
        code_sha256={str(q):sha(q)for q in (Path(__file__),ROOT/'events/ankara_2026/tools/reread_old_boundary_material.py',ROOT/'src/rinkspace/vision_source_topology.py')},clips=[])
    decoder=NativeDecoder('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4');start=time.perf_counter()
    try:
        for clip in ranges:
            displays=sorted(p for p in mapping if clip['start']<=p<clip['stop']);aliases=defaultdict(list)
            for pts in displays:aliases[int(mapping[pts]['canonical'])].append(pts)
            actual=sorted(aliases);hint=clip['seed'].copy();counts=Counter();done=[]
            binding=lambda pts:dict(source_sha256=SOURCE_SHA,source_pts=pts,time_base='1/15360',native_hw=[1080,1920],shot_id=clip['shot'])
            with gzip.open(o.output/(clip['name']+'_material.jsonl.gz'),'wt',encoding='utf8')as stream:
                for pts in actual:
                    rgb=decoder.exact(pts)
                    if clip['shot']=='shot5'and pts==ref['actual_pts']:
                        expected=np.asarray(Image.open(ref['source_raw_frame']).convert('RGB'))
                        if not np.array_equal(rgb,expected):raise ValueError('Native reference RGB differs')
                        hint=human.copy()
                    result=contact_material(rgb,binding(pts),hint)
                    # Coordinates are native-column curve searches. They are
                    # never asserted to follow one physical world point/facet.
                    candidate=result['fresh_material_lower_uv'];qualified=result['unique_mask']
                    emit(stream,dict(actual_pts=pts,display_pts=aliases[pts],source_binding=binding(pts),
                        decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        navigation_uv=hint.copy(),navigation_reference_pts=clip['seed_ref_pts'],navigation_scope=clip['reference_scope'],
                        direct_human_reference=clip['shot']=='shot5'and pts==ref['actual_pts'],
                        fresh_first_yellow_gray_fall_uv=candidate,local_material_role_supported=qualified,
                        material_evidence=result['first_fall_evidence'],
                        graphic_or_material_ambiguity=[dict(index=r['original_uv_index'],reason=r['reason'],outframe=r['outframe'],graphic=r['actual_RGB_graphic_role_conflict'])for r in result['classifications']if not r['unique_actual_material_hypothesis']],
                        physical_world_facet_or_point_association_verified=False,automatic_source_retirement=False))
                    hint[qualified]=candidate[qualified]
                    done.append(pts);counts['queries']+=len(hint);counts['supported_local_material_hypotheses']+=int(qualified.sum())
                    counts['unsupported_or_competing']+=int((~qualified).sum())
                    if len(done)%40==0:print(json.dumps(dict(stage='fresh_source_RGB_material_clip',clip=clip['name'],actual=len(done),expected=len(actual),pts=pts)),flush=True)
            decoded=[]
            with gzip.open(o.output/(clip['name']+'_native_clock.jsonl.gz'),'wt',encoding='utf8')as stream:
                for pts,rgb in decoder.presentations(clip['start'],clip['stop']):
                    decoded.append(pts);emit(stream,dict(source_pts=pts,actual_pts=int(mapping[pts]['canonical']),
                        RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),proof_sha256=clock['proof_sha256'],independent_new_point=False))
            if decoded!=displays or done!=actual:raise ValueError('Incomplete source clock coverage')
            report['clips'].append(dict(name=clip['name'],start_pts=clip['start'],stop_exclusive=clip['stop'],
                actual_exposures=len(actual),native_presentations=len(displays),complete_actual_RGB_and_native_clock=True,counts=dict(counts)))
            write(o.output/'receipt.json',report)
        report.update(status='complete_fresh_RGB_material_clips',wall_seconds=time.perf_counter()-start)
        write(o.output/'receipt.json',report);print(json.dumps(dict(status=report['status'],clips=report['clips'],seconds=report['wall_seconds'])),flush=True)
    finally:decoder.close()

if __name__=='__main__':main()
