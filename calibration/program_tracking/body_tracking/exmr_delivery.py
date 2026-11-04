"""Source movie and unaligned world diagnostics for the actual comparison states."""
from pathlib import Path
from fractions import Fraction
import argparse,json
import numpy as np
from .exmr_prior import load,write_json,digest
from .exmr_review import EDGES
from .stage_quality import body_groups


def cases(output):
    root=Path(output)/'common'
    definitions=[('gvhmr_fixed','gvhmr',root/'gvhmr/initial/fitted/result.npz','fixed_common_initial'),
        ('exmr_prefoot','exmr_prefoot',root/'exmr_prefoot/initial/fitted/result.npz','diagnostic_unoptimized'),
        ('exmr_final','exmr_final',root/'exmr_final/initial/fitted/result.npz','attempt_rejected_unknown'),
        ('gvhmr_joint1','gvhmr',root/'gvhmr/fitted/result.npz','joint_fit_1_step')]
    result={}
    for label,case,path,status in definitions:
        if not path.exists():raise FileNotFoundError(path)
        d=load(path);d['_case_id']=case
        result[label]=(path,d,status)
    return result


def diagnostics(output):
    output=Path(output);all_cases=cases(output);result={}
    for label,(path,d,status) in all_cases.items():
        pts=d['pts'];core=(pts>=88*12800)&(pts<=92*12800)
        obs=load(output/'common'/d['_case_id']/'observations.npz')
        np.testing.assert_array_equal(obs['pts'],pts)
        P=d['P_world_mid'];foot=d['skate_reference_world_mid'][:,1]
        def track(x):
            return dict(xy_span_m=np.ptp(x[core,:2],axis=0).tolist(),
                xy_path_m=float(np.linalg.norm(np.diff(x[core,:2],axis=0),axis=1).sum()))
        gap=d['skate_support_gap_m_mid'];rows=[]
        for i in np.flatnonzero(np.isfinite(gap)):
            feet=np.flatnonzero(d['source_support_mode_mid'][i]!='unknown').tolist()
            rows.append(dict(source_pts=int(pts[i]),source_seconds=float(pts[i]/12800),
                source_support_candidate_feet=feet,raw_reference_gap_m=float(gap[i]),
                interpretation='minimum over admitted source support candidate feet; not one invented anatomical-foot measurement'))
        run_path=output/'common'/d['_case_id']/'run_summary.json'
        run=json.loads(run_path.read_text(encoding='utf8')) if status=='joint_fit_1_step' and run_path.exists() else None
        result[label]=dict(result=str(path),result_sha256=digest(path),stage=status,
            source_frames=len(pts),source_first_last_pts=[int(pts[0]),int(pts[-1])],
            accepted_steps=0 if run is None else run['accepted_steps'],
            retained_step=None if run is None else run['retained_step'],
            core_body_groups=body_groups(d['COCO_uv'][core],obs['uv'][core],obs['use'][core],pts[core]),
            core_pelvis_projection_G_equals_Pxy=track(P),core_fixed_steel_material_reference_xy=track(foot),
            raw_nominal_plane_penetration_depth_max_m=float(np.maximum(-d['skate_clearance_m_mid'],0).max()),
            model_foot_cavity_breach_max_m=float(d['foot_cavity_max_violation_m_mid'].max()),
            structural_constraint_max_m=None if run is None else run['run'].get('max_violation_m'),
            source_supported_raw_gaps=rows,unknown_contact_foot_frames=int((d['skate_contact_mode_mid']=='unknown').sum()),
            ice_is_delivery_gate=False,user_accepted=False)
    report=dict(schema='rinkspace.spin-model-delivery.v1',source_time_base=[1,12800],
        whole_continuous_seconds=[76,95],diagnostic_core_seconds=[88,92],cases=result,
        interpretations=dict(G='vertical projection of model pelvis P; not CoM or a measured support point',
            steel_reference='same attached material point on steel web; not contact minimum',
            heights='raw model/reference-plane diagnostics, not independent physical ice truth',
            support='only source-admitted exposures; network static probabilities never admitted',
            image_error='source anatomical body-keypoint residual; not shoe/blade or contact acceptance'),
        no_trajectory_alignment=True,no_more_optimization=True)
    collar=output/'collar_diagnostic.json'
    if collar.exists():report['collar_diagnostic']=json.loads(collar.read_text(encoding='utf8'))
    write_json(output/'delivery_comparison.json',report)
    return report


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);all_cases=cases(output)
    fig,ax=plt.subplots(1,3,figsize=(15,4.6));colors={'gvhmr_fixed':'#cf5c4e','exmr_prefoot':'#248abc','exmr_final':'#71a845','gvhmr_joint1':'#9864b5'}
    for label,(_,d,status) in all_cases.items():
        sec=d['pts']/12800.;core=(sec>=88)&(sec<=92);P=d['P_world_mid'][core];F=d['skate_reference_world_mid'][core,1]
        name=label+' ('+status+')';color=colors[label]
        ax[0].plot(P[:,0],P[:,1],color=color,label=name)
        ax[1].plot(F[:,0],F[:,1],color=color,label=name)
        ax[2].plot(sec[core],d['skate_clearance_m_mid'][core,1],color=color,label=name)
        for point in (0,-1):
            for axis,x in zip(ax[:2],[P,F]):axis.scatter(x[point,0],x[point,1],color=color,s=22,marker='o' if point==0 else 'x')
    for axis in ax[:2]:axis.set_aspect('equal',adjustable='datalim');axis.set_xlabel('world X / conditional m');axis.set_ylabel('world Y / conditional m');axis.grid(alpha=.25)
    ax[0].set_title('Pelvis projection G = Pxy (88-92s)')
    ax[1].set_title('Same steel material point, foot 1')
    ax[2].set_title('Raw model / nominal-plane clearance')
    ax[2].set_xlabel('source seconds');ax[2].set_ylabel('conditional m');ax[2].axhline(0,color='black',lw=.8);ax[2].grid(alpha=.25)
    ax[2].legend(loc='best',fontsize=8)
    fig.suptitle('Same source, camera, fixed S01 body scale and skate. No XY alignment. Heights are diagnostics, not contact truth.',fontsize=10)
    fig.tight_layout();fig.savefig(output/'world_comparison.png',dpi=150);fig.savefig(output/'world_comparison.svg');plt.close(fig)


def movie(output,base):
    import av
    from PIL import Image,ImageDraw,ImageFont
    from .action_clip_input import _official_frames
    output,base=Path(output),Path(base);all_cases=cases(output)
    first=next(iter(all_cases.values()))[1];pts=first['pts'];indices={int(p):i for i,p in enumerate(pts)}
    original=load(base/'action_integration_1103/spin_s05/prepared.npz')
    rect=np.stack([np.interp(pts,original['source_pts'],original['roi_native_xyxy'][:,j]) for j in range(4)],axis=1)
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))['source']['path']
    target=output/'source_GVHMR_ExMR_76_95.mp4'
    if target.exists():raise FileExistsError('Preserve existing video')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    record=[];out=av.open(str(target),'w');stream=out.add_stream('libx264',rate=50)
    stream.width=1260;stream.height=600;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'};stream.codec_context.thread_count=4
    colors={'gvhmr_fixed':(245,100,85),'exmr_prefoot':(50,190,245)}
    try:
        for p,bgr in _official_frames(source,pts):
            i=indices[p];box=rect[i];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            height=max(540.,(box[3]-box[1])*1.20);width=max(height*.7,(box[2]-box[0])*1.20);height=max(height,width/.7)
            crop=(int(cx-width/2),int(cy-height/2),int(cx+width/2),int(cy+height/2))
            image=Image.fromarray(bgr[:,:,::-1]);canvas=Image.new('RGB',(1260,600))
            for column,label in enumerate(['source','gvhmr_fixed','exmr_prefoot']):
                panel=image.crop(crop).resize((420,600));draw=ImageDraw.Draw(panel)
                if label!='source':
                    d=all_cases[label][1];uv=(d['COCO_uv'][i]-np.array(crop[:2]))*np.array([420/(crop[2]-crop[0]),600/(crop[3]-crop[1])])
                    for a,b in EDGES:
                        if np.isfinite(uv[[a,b]]).all():draw.line([tuple(uv[a]),tuple(uv[b])],fill=colors[label],width=3)
                    for x,y in uv:
                        if np.isfinite([x,y]).all():draw.ellipse((x-3,y-3,x+3,y+3),fill=colors[label])
                draw.rectangle((0,0,420,29),fill='#181818');draw.text((6,5),f'{label} | source {p/12800:.2f}s',fill='white',font=font)
                draw.rectangle((0,578,420,600),fill='#181818');draw.text((6,580),'Body joints only; not shoe/blade points',fill='white',font=font)
                canvas.paste(panel,(420*column,0))
            frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');frame.pts=i;frame.time_base=Fraction(1,50)
            for packet in stream.encode(frame):out.mux(packet)
            record.append(int(p))
        for packet in stream.encode():out.mux(packet)
    finally:out.close()
    if record!=pts.tolist():raise ValueError('Video/source PTS mismatch')
    write_json(output/'source_movie_clock.json',dict(video=str(target),source_video=source,source_seconds=[76,95],
        source_time_base=[1,12800],source_pts=record,encoded_rate_hz=50,encoded_frames=len(record),
        timeline='one output frame per original native source exposure, inclusive endpoints',
        viewport='common tracked source ROI for readability only; not world trajectory or an observation',
        geometry='actual common-state native exposure/row projection saved in result; not raw network camera overlay'))
    return target


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base-root',required=True)
    p.add_argument('--video',action='store_true');p.add_argument('--plot',action='store_true')
    a=p.parse_args(argv);diagnostics(a.output)
    if a.plot:plot(a.output)
    if a.video:movie(a.output,a.base_root)

if __name__=='__main__':main()