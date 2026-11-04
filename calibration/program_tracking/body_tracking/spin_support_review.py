"""Native-camera visual checks for the source-supported joint body candidate."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from rinkspace.viewer3d.source import SourceVideo
from .spin_support_joint import prepare
from .support_root_placement import restore_parent
from .skate_integration import restore_checkpoint_layers
from .skate_physics import fk_state
from .skate_template import profile_height
from .footwear_binding import transform
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import load
from .fullbody_round import write_json
from .native_mesh_overlay import paint


def project_mesh(ref,camera,frames):
    t=ref.times[frames];times=(t[:,None]+np.linspace(-.5,.5,5)*float(camera.p['rig'][37])).ravel();r=ref.at(times);s=fk_state(ref,r,np.zeros((len(times),69)));kw={}if 'hand_local_R'not in s else {'hand_local_R':s['hand_local_R']}
    vertices=ref.binding.surface.sample_vertices(s['native'],s['frames'],double_precision=True,**kw);world=np.einsum('nij,nvj->nvi',s['Q'],vertices)+s['P'][:,None]
    full=moving_projection(camera,world.reshape(len(frames),5,-1,3),frames)['uv']
    packet=ref.binding.packet();lower=np.asarray(packet['template']['vertices']);upper=ref.binding.deform(packet['upper'],ref.binding.deformation(s['native'],s['frames']));shoe=np.concatenate([transform(s['R'],s['O'],lower),transform(s['R'],s['O'],upper)],axis=2)
    uv=moving_projection(camera,shoe.reshape(len(frames),5,-1,3),frames)['uv'].reshape(len(frames),2,len(lower)+upper.shape[2],2)
    p=packet['parameters'];x=.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m);q=np.array([x,0,float(profile_height(x,p))]);middle=s['O']+np.einsum('nfij,j->nfi',s['R'],q);mid_uv=moving_projection(camera,middle.reshape(len(frames),5,2,3),frames)['uv']
    spec=ref.grasp_geometry;ids=spec['palm_patches']['left']['vertex_ids'];weights=spec['palm_patches']['left']['weights'];H=np.einsum('v,nvc->nc',weights,world[:,ids]);seg=np.asarray(spec['segment_local_solver']);G=s['O'][:,0,None]+np.einsum('nij,kj->nki',s['R'][:,0],seg);points=np.concatenate([H[:,None],G],axis=1);grip=moving_projection(camera,points.reshape(len(frames),5,3,3),frames)['uv']
    return full,uv,mid_uv,grip,packet,len(lower)


def snapshots(project,base,old,output,final=False,checkpoint=None,tag=None):
    target=output/(tag or ('native_final_frames'if final else'native_seed_frames'));target.mkdir(parents=True,exist_ok=True)
    new,camera,info,doc,relations=prepare(project,base,old,output);active=json.loads((old/'active_candidate.json').read_text(encoding='utf8'));parent,_,_,_=restore_parent(project,base,old,target/'parent_geometry',Path(active['parameter']))
    if checkpoint is not None:restore_checkpoint_layers(new,load(checkpoint),include_primary=True)
    elif final:
        summary=json.loads((output/'run_summary.json').read_text(encoding='utf8'));restore_checkpoint_layers(new,load(summary['parameter']),include_primary=True)
    seconds=[84.4,89.56,92.46];pts=np.array([round(v*12800)for v in seconds]);frames=np.searchsorted(new.p['pts'],pts)
    views=[project_mesh(parent,camera,frames),project_mesh(new,camera,frames)]
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))['source']['path'];video=SourceVideo(source,(1,12800));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);saved=[]
    try:
        for j,(frame,pt)in enumerate(zip(frames,pts)):
            raw=Image.fromarray(video.read(int(pt)));panels=[raw]
            for r,view,label in zip([parent,new],views,['原联合4步','新支撑联合候选'if (final or checkpoint is not None)else'新支撑初值']):
                mesh,shoe,mids,grip,packet,nlower=view;su=shoe[j];parts=[su[0,:nlower],su[0,nlower:],su[1,:nlower],su[1,nlower:]];faces=np.asarray(packet['template']['faces'],int)
                im=paint(raw,mesh[j],r.binding.surface.faces,parts,[faces,r.binding.faces,faces,r.binding.faces],label,font,int(pt));draw=ImageDraw.Draw(im)
                for f,color in [(0,(255,155,40)),(1,(50,150,255))]:
                    q=mids[j,f];draw.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),outline=color,width=3);draw.text((q[0]+8,q[1]),'foot'+str(f)+(' 支撑'if f==1 else' 自由'),font=font,fill=color)
                H,a,b=grip[j];draw.line([tuple(a),tuple(b)],fill=(255,90,240),width=4);draw.ellipse((H[0]-6,H[1]-6,H[0]+6,H[1]+6),outline=(255,90,240),width=3);panels.append(im)
            native=target/('native_'+str(int(pt))+'.png');panels[-1].save(native)
            box=np.asarray(new.p['source_bbox'][frame]);cx=(box[0]+box[2])/2;crop=(round(cx-360),0,round(cx+360),1080);canvas=Image.new('RGB',(1800,980),(12,16,23))
            for k,im in enumerate(panels):canvas.paste(im.crop(crop).resize((600,900)),(k*600,45))
            draw=ImageDraw.Draw(canvas)
            for k,text in enumerate(['原片','原联合4步','新联合候选'if (final or checkpoint is not None)else'新支撑初值']):draw.text((k*600+10,8),text,font=font,fill='white')
            draw.text((10,949),'原片 '+format(pt/12800.,'.2f')+'s；同一原相机／native畸变／5曝光节点；蓝色foot1，紫色掌面与有限握持段',font=font,fill='white');comparison=target/('compare_'+str(int(pt))+'.png');canvas.save(comparison);saved.append(dict(pts=int(pt),native=str(native),comparison=str(comparison)))
    finally:video.close()
    write_json(target/'frames.json',dict(source=source,time_base=[1,12800],frames=saved,final=final,checkpoint=None if checkpoint is None else str(checkpoint),world_alignment=False,source_camera_unchanged=True));print(json.dumps(dict(event='support_native_snapshots',path=str(target),final=final)),flush=True)
    return saved
