"""Summarize a frozen native camera's zoom-domain compensation without fitting."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rinkspace.core.native import NativeCore
from rinkspace.path_policy import guard_project_write
import prepare_native_multicamera as prepared
from export_native_six_review import _verify_native_library_binding


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main(args):
    out=guard_project_write(args.output)
    for suffix in ('.json','.png'):
        if out.with_suffix(suffix).exists():raise FileExistsError(out.with_suffix(suffix))
    physical=prepared.read(args.physical);motion=prepared.read(args.motion)
    m=prepared.read(args.run_root/'joint_manifest.json');v=prepared.npz(args.state)
    _verify_native_library_binding(m,v);core=NativeCore(m['native_library'])
    sha=digest(args.state)
    if physical['identities'][str(args.state)]!=sha or motion['states']['candidate']['sha256']!=sha:raise ValueError('Mixed diagnostic states')
    scale=physical['metric_scale'];fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    records=[]
    for i,rig in enumerate(v['rig']):
        entry=physical['rigs'][i];supported=np.asarray(entry['observed_focal_intervals_pixels'])
        lo,hi=float(supported.min()),float(supported.max());f=np.geomspace(lo,hi,501);z=np.log(f/rig[24])
        pupil=(rig[22]*z+rig[23]*z*z)*scale;forward=rig[20]*scale+pupil
        controls=np.tile([0.,0.,0.],(4*len(f),1))
        # Independent four-control constant-focal cells share no exposure clock.
        controls[:,2]=np.repeat(np.log(f),4)
        times=4*np.arange(len(f))+1.
        opt=core.evaluate_motion_camera(rig,controls,0.,1.,times)
        if np.any(opt['status']):raise ValueError('Invalid constant-focal native probe')
        x=959.5/opt['fx'];y=539.5/opt['fy'];radius=x*x+y*y
        first=959.5*opt['k1']*radius;second=959.5*opt['k2']*radius*radius
        tangent=(2*opt['p1']*x*y+opt['p2']*(radius+2*x*x))*opt['fx']
        net=first+second+tangent
        ax=axes[i,0];ax.plot(f,forward*100,label='Complete Q-to-C forward offset',color='#1e638c',lw=2)
        for a,b in supported:ax.axvspan(a,b,color='#b4d7e8',alpha=.35)
        ax.axhline(-10,color='#bb4b45',ls='--');ax.axhline(30,color='#bb4b45',ls='--')
        ax.set_xscale('log');ax.set_ylabel('Conditional cm');ax.set_title(entry['id']+' | pupil / mount')
        ax.grid(alpha=.2);ax.set_xlabel('Delivered focal length (px)');ax.legend(fontsize=8)
        ax=axes[i,1];ax.plot(f,first,label='k1 component');ax.plot(f,second,label='k2 component');ax.plot(f,net,label='Net incl. tangent',color='#20272d',lw=2)
        ax.set_xscale('log');ax.set_ylabel('Corner horizontal shift (px)');ax.set_xlabel('Delivered focal length (px)')
        ax.set_title(entry['id']+' | opposing radial terms');ax.grid(alpha=.2);ax.legend(fontsize=8)
        vertex=None
        if rig[23]!=0:
            vz=-rig[22]/(2*rig[23]);vf=float(rig[24]*np.exp(vz))
            if lo<=vf<=hi:vertex=dict(focal_pixels=vf,pupil_shift_conditional_m=float((rig[22]*vz+rig[23]*vz*vz)*scale),complete_forward_conditional_m=float((rig[20]+rig[22]*vz+rig[23]*vz*vz)*scale),observed_zoom_support=bool(np.any((supported[:,0]<=vf)&(vf<=supported[:,1]))))
        records.append(dict(id=entry['id'],focal_support_pixels=supported.tolist(),focal_span_ratio=hi/lo,
            ideal_horizontal_field_degrees_at_widest=float(np.degrees(2*np.arctan(959.5/lo))),
            readout_ms=float(rig[37]*1000),readout_over_delivered_frame_period=float(rig[37]/.02),
            complete_forward_over_joined_zoom_range_conditional_m=entry['prediction_across_unobserved_zoom_gaps']['complete_forward_m'],quadratic_vertex=vertex,
            widest_native_optics={key:float(opt[key][0]) for key in ('fx','fy','cx','cy','k1','k2','p1','p2')},
            widest_corner_horizontal_terms_pixels=dict(k1=float(first[0]),k2=float(second[0]),tangential=float(tangent[0]),net=float(net[0])),
            maximum_absolute_corner_term_pixels=dict(k1=float(abs(first).max()),k2=float(abs(second).max()),net=float(abs(net).max()))))
    fig.suptitle('Frozen joint_mount_flex | shaded focal intervals are observed; all metres conditional',fontsize=13)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out.with_suffix('.png'),dpi=145);plt.close(fig)
    report=dict(schema='rinkspace.native_zoom_compensation_details.v1',state=str(args.state),state_sha256=sha,
        physical=str(args.physical),motion=str(args.motion),rigs=records,optimizer_started=False,state_modified=False,
        source_uv_modified=False,hardware_mode_confirmed=False,
        mode_note='The fixed main/secondary 2:1 magnification is a coupling hypothesis. Catalog 35-mm-equivalent focal values are not a measurement of the delivered sensor crop or event settings.',
        plots_note='Native effective optics are queried in isolated constant-focal cells; component arithmetic was already verified against full native source projections by physical_state.json.')
    if args.source_motion:
        flow=prepared.read(args.source_motion)
        if flow['identities'][str(args.state)]!=sha:raise ValueError('Source-motion state differs')
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        windows=[('s08',136.16,136.40),('s05',101.40,101.64),('s02',56.06,56.30)]
        for ax,(sid,a,b) in zip(axes,windows):
            all_rows=[r for r in flow['records'] if r['shot']==sid and a-1e-8<=r['source_seconds']<b-1e-8]
            valid=[r for r in all_rows if r['status']=='compared'];tt=[r['source_seconds'] for r in valid]
            ax.plot(tt,[r['source_dx_median_px'] for r in valid],marker='o',label='Measured source texture',color='#247c65')
            ax.plot(tt,[r['native_dx_median_px'] for r in valid],marker='o',label='Current native camera',color='#c45143')
            for r in all_rows:
                if r['status']!='compared':ax.axvline(r['source_seconds'],color='.6',ls=':',alpha=.4)
            ax.set_title(sid+' | consecutive 20 ms frames');ax.set_xlabel('Source seconds');ax.set_ylabel('Median horizontal displacement (px)');ax.grid(alpha=.2);ax.ticklabel_format(useOffset=False);ax.legend(fontsize=8)
        fig.suptitle('Raw source motion vs unchanged native camera | missing source support marked dotted')
        fig.savefig(out.parent/'source_motion_comparison.png',dpi=145);plt.close(fig)
    if digest(args.state)!=sha:raise ValueError('Input state changed')
    out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
    print(str(out.with_suffix('.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','physical','motion','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-motion',type=Path)
    main(p.parse_args())
