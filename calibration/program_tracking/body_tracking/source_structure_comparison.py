"""Exact native source/old/current runner overlays, without opening a viewer."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .blade_projection import SavedBladeProjection
from .source_skate_structure_review import draw_path,COLORS


def compare(project,parent_stage,parent_parameter,stage,parameter,structures,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path(structures).read_text(encoding='utf-8'));rows=source['frames']+source.get('unassigned_identity_observations',[])
    before=SavedBladeProjection(project,parent_stage,parent_parameter);after=SavedBladeProjection(project,stage,parameter)
    times=[76.8,80.4,81.62,87.64,89.6,92.8,94.4,95.]
    selected=[r for t in times for r in rows if r['pts']==round(t*12800)and(r.get('foot')==1 or r.get('foot')is None)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);records=[]
    for number,row in enumerate(selected):
        pts=int(row['pts']);box=np.asarray(row['native_bounds_xyxy']);crop=Image.open(row['source_crop_path']).convert('RGB');scale=3
        zoom=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST);W,H=zoom.size
        panels=[zoom.copy(),zoom.copy(),zoom.copy()];metrics=[]
        for panel in panels[1:]:
            d=ImageDraw.Draw(panel)
            for key in ['blade_assembly_outline','runner_lower_edge','runner_upper_edge','toe_pick','rear_extension']:
                f=row['features'].get(key,{})
                if len(f.get('points_uv',[]))>=2:draw_path(d,(np.asarray(f['points_uv'])-box[:2])*scale,COLORS[key],2,f.get('closed',False),f.get('status','unknown'))
        for index,query in enumerate([before,after],1):
            packet=query.binding.packet();edge=packet['edges'];upper=edge+[0,0,packet['parameters'].blade_web_height_m]
            for foot in([row['foot']]if row.get('foot')is not None else[0,1]):
                curves=np.r_[edge,upper];uv=query.project_points([pts],foot,curves.reshape(-1,3))['uv'][0].reshape(4,-1,2)
                for curve in uv:draw_path(ImageDraw.Draw(panels[index]),(curve-box[:2])*scale,(245,38,166),2)
                metrics.append(dict(candidate='before'if index==1 else'after',model_foot=foot,runner_uv=uv.tolist()))
        canvas=Image.new('RGB',(W*3+48,H+100),'white');d=ImageDraw.Draw(canvas)
        d.text((12,5),f'原片 {pts/12800.:.2f}s / PTS {pts} / 源foot={row.get("foot")}；粉色为真实模型钢刃回投，绿色/紫色等为源描线',font=font,fill='black')
        for index,label in enumerate(['原片','原片＋源描线＋旧候选','原片＋源描线＋本次联合候选']):
            x=12+index*(W+12);d.text((x,38),label,font=font,fill='black');canvas.paste(panels[index],(x,70))
        path=output/f'comparison_{pts}_{number}.png';canvas.save(path);records.append(dict(pts=pts,source_foot=row.get('foot'),source_track_id=row.get('source_track_id'),path=str(path),predictions=metrics))
    (output/'comparison.json').write_text(json.dumps(dict(parent_parameter=str(parent_parameter),candidate_parameter=str(parameter),native_projection=True,no_gui=True,accuracy_accepted=False,frames=records),ensure_ascii=False,indent=2),encoding='utf-8')
    return records


def main():
    p=argparse.ArgumentParser()
    for key in ['project','parent-stage','parent-parameter','stage','parameter','structures','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=compare(**vars(a));print(json.dumps(dict(comparison_cards=len(r),output=str(a.output))))
if __name__=='__main__':main()
