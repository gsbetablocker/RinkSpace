"""All-frame sequential native contact/top strips for source-side visual QC."""
from __future__ import annotations
import cv2
import numpy as np
from .extract_first_shot_board_source import DATA,read,write,sha


def render(selected_pages=None):
    destination=DATA/'board_boundary_sequential_pages';destination.mkdir(exist_ok=True)
    pages=[];canvases={role:np.zeros((440,1920,3),np.uint8)for role in ('contact','top')}
    manifest={int(r['index']):r for r in read(DATA/'frame_manifest.json')['frames']}
    entries=[]
    for position,index in enumerate(range(201,1249)):
        if selected_pages is not None and position//16+1 not in selected_pages:continue
        record=manifest[index];raw=cv2.imread(record['image_path']);packet=read(DATA/f'board_candidates/{index:06d}.json')
        for role in ('contact','top'):
            features=[f for f in packet['features']if f['target_id'].startswith('board.'+role+'.')]
            points=np.vstack([s['uv']for f in features for s in f['segments']]);y0=max(0,int(np.floor(points[:,1].min()))-42);y1=min(1080,int(np.ceil(points[:,1].max()))+43)
            crop=raw[y0:y1].copy();color=(255,230,0)if role=='contact'else(0,230,255)
            for f in features:
                for s in f['segments']:
                    p=np.asarray(s['uv'])-[0,y0];cv2.polylines(crop,[np.rint(p).astype('int32')],False,color,2,cv2.LINE_AA)
                    for xy in p:cv2.circle(crop,tuple(np.rint(xy).astype(int)),5,color,-1,cv2.LINE_AA)
            reduced=cv2.resize(crop,(480,max(1,int(round(crop.shape[0]*.25)))),interpolation=cv2.INTER_AREA)
            tile=np.zeros((110,480,3),np.uint8);cv2.putText(tile,f'{index:04d}  {role}  native crop y={y0}:{y1}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            if reduced.shape[0]>83:reduced=cv2.resize(reduced,(480,83),interpolation=cv2.INTER_AREA)
            tile[27:27+reduced.shape[0]]=reduced
            row,column=divmod(position%16,4);canvases[role][row*110:(row+1)*110,column*480:(column+1)*480]=tile
        entries.append(index)
        if len(entries)==16 or index==1248:
            page=(position//16)+1;paths={}
            for role,canvas in canvases.items():
                path=destination/f'{role}_{page:03d}_{entries[0]:04d}_{entries[-1]:04d}.jpg';cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,94]);paths[role]=str(path);canvases[role].fill(0)
            pages.append(dict(page=page,indices=entries,paths=paths));entries=[]
    if selected_pages is None:
        write(destination/'manifest.json',dict(schema='rinkspace.sequential-native-board-boundary-pages.v1',frames=1048,programme_frame_range_1based=[201,1248],page_count=len(pages),native_source_pixels=True,full_native_width_retained=True,horizontal_scale=.25,crops_preserve_actual_boundary_slope=True,pages=pages))
    print('Sequential board boundary QC pages',len(pages),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pages')
    args=parser.parse_args();render(set(map(int,args.pages.split(',')))if args.pages else None)
