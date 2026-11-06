"""Exact source-only crops for full-sequence skate structure reading."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from rinkspace.viewer3d.source import SourceVideo


def prepare(sequence,output,stride=20):
    sequence=Path(sequence);output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    doc=json.loads((sequence/'source_sequence.json').read_text(encoding='utf-8'))
    clock=doc['clock'];selected=list(range(0,len(clock),stride))
    if selected[-1]!=len(clock)-1:selected.append(len(clock)-1)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    sha=lambda a:hashlib.sha256(a).hexdigest()
    reader=SourceVideo(doc['source_video']['path'],(1,12800));rows=[];tiles=[]
    try:
        for index in selected:
            row=clock[index];pts=int(row['pts']);image=reader.read(pts)
            if sha(image.tobytes())!=row['decoded_source_rgb_sha256']:raise ValueError('Source pixels differ')
            body=Image.fromarray(image).crop(row['body_navigation_bounds_clipped_xyxy']);body.thumbnail((180,270))
            tile=Image.new('RGB',(500,310),'#eeeeee');pen=ImageDraw.Draw(tile)
            pen.text((8,4),f'{pts/12800:.2f}s / PTS {pts}',fill='black',font=font);tile.paste(body,(0,35))
            context=Image.new('RGB',(950,820),'#eeeeee');cd=ImageDraw.Draw(context)
            whole=Image.fromarray(image);whole.thumbnail((950,420));context.paste(whole,(0,0))
            for foot in (0,1):
                box=np.asarray(row['feet_navigation_bounds_clipped_xyxy'][foot],int)+[-16,-16,16,16]
                box=np.r_[np.maximum(box[:2],0),np.minimum(box[2:],[1920,1080])].astype(int)
                crop=Image.fromarray(image).crop(tuple(box));name=f'{pts}_f{foot}'
                path=output/'native'/f'{name}.png';path.parent.mkdir(exist_ok=True);crop.save(path)
                zoom=crop.resize((crop.width*3,crop.height*3),Image.Resampling.NEAREST)
                ruler=Image.new('RGB',(zoom.width+80,zoom.height+100),'white');rd=ImageDraw.Draw(ruler);ruler.paste(zoom,(60,60))
                rd.text((5,4),f'{pts/12800:.2f}s / foot proposal {foot} / origin {box[:2].tolist()}',font=font,fill='black')
                for u in range(0,crop.width,10):rd.text((60+u*3,37),str(u),font=font,fill='black')
                for v in range(0,crop.height,10):rd.text((8,60+v*3),str(v),font=font,fill='black')
                rp=output/'rulers'/f'{name}.png';rp.parent.mkdir(exist_ok=True);ruler.save(rp)
                thumb=crop.resize((150,150));tile.paste(thumb,(190+foot*155,78));pen.text((190+foot*155,48),f'proposal f{foot}',font=font,fill='black')
                z=crop.resize((390,390));context.paste(z,(foot*460,420));cd.text((foot*460,795),f'foot proposal {foot}',font=font,fill='black')
                rows.append(dict(index=index,pts=pts,source_seconds=pts/12800.,foot_proposal=foot,native_bounds_xyxy=box.tolist(),native_crop=str(path),native_crop_sha256=sha(path.read_bytes()),native_frame_rgb_sha256=row['decoded_source_rgb_sha256'],ruler_image=str(rp),ruler_scale=3,ruler_image_origin_xy=[60,60],context_image=str(output/'context'/f'{pts}.png'),context_pts=[pts],foot_identity_is_navigation_proposal=True))
            cp=output/'context'/f'{pts}.png';cp.parent.mkdir(exist_ok=True);context.save(cp);tiles.append(tile)
    finally:reader.close()
    sheets=[]
    for start in range(0,len(tiles),8):
        sheet=Image.new('RGB',(2000,620),'white')
        for j,tile in enumerate(tiles[start:start+8]):sheet.paste(tile,((j%4)*500,(j//4)*310))
        path=output/f'survey_{start:02d}.png';sheet.save(path);sheets.append(dict(path=str(path),pts=[clock[i]['pts']for i in selected[start:start+8]]))
    result=dict(schema='rinkspace.source_structure_inputs.v1',source_video=doc['source_video'],source_native_size=[1920,1080],time_base=[1,12800],native_coordinate_domain='original source native pixels',no_model_projection_used=True,source_sampling_stride_frames=stride,frames=rows,survey_sheets=sheets)
    (output/'inputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=20);a=p.parse_args()
    d=prepare(a.sequence,a.output,a.stride);print(json.dumps(dict(frames=len(d['frames']),sheets=d['survey_sheets'])))
if __name__=='__main__':main()
