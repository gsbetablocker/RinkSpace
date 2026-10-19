"""Read actual reference RGB profiles through the existing foot material API."""
from pathlib import Path
import argparse,gzip,json,sys
import numpy as np
from PIL import Image,ImageDraw
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from rinkspace.vision_source_topology import _original_RGB_profiles_on_CPU,_ordered_board_foot_profile_batches

def normal(points,prior):
    t=np.gradient(points,axis=0);n=np.c_[-t[:,1],t[:,0]];n/=np.maximum(np.linalg.norm(n,axis=1),1e-12)[:,None]
    n[(n*prior).sum(1)<0]*=-1;return n

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args()
    o.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    meta=json.loads((o.run/'run_manifest.json').read_text(encoding='utf8'))
    authority=json.loads(Path(meta['authority_path']).read_text(encoding='utf8'))
    full=json.loads((Path(meta['authority_path']).parent/'human_registered_full_stroke_hard_guard_v1.json').read_text(encoding='utf8'))
    record=next(r for r in authority['records']if r['curve']['kind']=='ice_board_contact')
    frow=next(r for r in full['records']if r['id']==record['id'])
    with np.load(full['guard_arrays_reference']['path'],allow_pickle=False)as z:
        alluv=z[frow['array_prefix']+'_uv'].copy();allnorm=z[frow['array_prefix']+'_normal'].copy()
    with gzip.open(o.run/'shot5_measurements.jsonl.gz','rt',encoding='utf8')as f:
        for line in f:
            frame=json.loads(line)
            if frame['actual_pts']==record['actual_pts']and frame['phase']!='later_reference_backward_navigation_preparation':
                reading=next(r for r in frame['features']if r['id']==record['id']);break
    selected=np.asarray(reading['original_query_indices'],int);uv=alluv[selected];n=normal(uv,allnorm[selected])
    rgb=np.asarray(Image.open(record['source_raw_frame']).convert('RGB'));binding=frame['source_binding']
    profiles=_original_RGB_profiles_on_CPU(rgb,uv,n,48,binding)
    original=reading['yellow_material_order'];old=np.asarray(original['lower_exit_uv']);oldunique=np.asarray(original['unique_lower_role'],bool)&~np.asarray(original['missing_same_band'],bool)
    ordered=_ordered_board_foot_profile_batches(profiles,torch.zeros(rgb.shape[:2],dtype=torch.bool),np.full(len(uv),rgb.shape[0]*.25),None)
    offsets=profiles['rgb_gradient_offsets_native_px'];newuv=profiles['points_native_uv']+profiles['normals_unit']*ordered.get('selected_exit_offset_native_px',offsets[ordered['exit_index']])[:,None]
    unique=ordered['usable'].numpy()&~ordered['missing_same_band'].numpy()
    olddelta=((old-uv)*n).sum(1);newdelta=((newuv.numpy()-uv)*n).sum(1)
    summary=dict(reference_pts=record['actual_pts'],reference_source=record['source_raw_frame'],human_role=record['geometry_role'],
        development_reference_not_independent_validation=True,original_semantic_ranked_unique=int(oldunique.sum()),
        original_semantic_ranked_median_distance=float(np.median(np.abs(olddelta[oldunique]))),
        original_semantic_ranked_maximum_distance=float(np.max(np.abs(olddelta[oldunique]))),
        no_semantic_unique=int(unique.sum()),no_semantic_median_distance=float(np.median(np.abs(newdelta[unique]))),
        no_semantic_maximum_distance=float(np.max(np.abs(newdelta[unique]))),
        original_query_indices=selected.tolist(),original_signed_normal_difference_px=olddelta.tolist(),
        no_semantic_signed_normal_difference_px=newdelta.tolist(),original_unique_mask=oldunique.tolist(),no_semantic_unique_mask=unique.tolist())
    (o.output/'actual_reference_profiles_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    np.savez_compressed(o.output/'actual_reference_profiles.npz',source_uv=uv,normal=n,RGB_profile=profiles['rgb_profile'].numpy(),
        offsets=profiles['offsets_native_px'].numpy(),warm_gamma=ordered['gamma'].numpy(),luminance=ordered['native_RGB_luminance'].numpy(),
        original_semantic_selected_uv=old,no_semantic_selected_uv=newuv.numpy(),no_semantic_entry_index=ordered['entry_index'].numpy(),
        no_semantic_exit_index=ordered['exit_index'].numpy(),all_pair_indices=ordered['all_ordered_pair_indices_profile_entry_exit'].numpy(),
        all_pair_scores=ordered['all_ordered_pair_scores'].numpy())
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,list)},ensure_ascii=False));return
    indices=[int(np.argmin(abs(uv[:,0]-x)))for x in (1320,1570,1830)]
    figure,axes=plt.subplots(2,3,figsize=(15,7),sharex=True)
    off=profiles['offsets_native_px'].numpy();colors=profiles['rgb_profile'].numpy()
    for column,i in enumerate(indices):
        for channel,color in enumerate(('red','green','blue')):axes[0,column].plot(off,colors[i,:,channel],color=color,label=color)
        axes[1,column].plot(off,ordered['gamma'][i].numpy(),label='yellow chroma')
        axes[1,column].plot(off,ordered['native_RGB_luminance'][i].numpy(),label='gray luminance')
        for ax in axes[:,column]:
            ax.axvline(0,color='black',label='human definition');ax.axvline(olddelta[i],color='magenta',label='old semantic ranking')
            ax.axvline(newdelta[i],color='cyan',label='RGB-only existing score');ax.set_xlim(-20,25);ax.grid();ax.legend(fontsize=7)
        axes[0,column].set_title(f'actual source x={uv[i,0]:.1f}, y={uv[i,1]:.1f}')
    figure.tight_layout();figure.savefig(o.output/'actual_reference_RGB_profiles.png',dpi=130);plt.close(figure)
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    for points,color in ((uv,'yellow'),(old,'magenta'),(newuv.numpy(),'cyan')):
        for x,y in points:draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
    image.crop((1200,410,1920,750)).save(o.output/'actual_reference_original_RGB_crop.png')
    print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,list)},ensure_ascii=False))

if __name__=='__main__':main()
