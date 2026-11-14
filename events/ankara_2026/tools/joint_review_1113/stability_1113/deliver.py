"""Render revised Jin shots and deliver one exact-clock full programme video."""
from argparse import ArgumentParser
import json
import subprocess
from time import monotonic
from concurrent.futures import ProcessPoolExecutor,as_completed

from camera_audit import HERE, PREP, ji, sj, vj, setup, native_scene
import assemble_full as af
import render_joint as rj
from register_review import install_json,install_npz
from rinkspace.path_policy import guard_project_write
import numpy as np

BASE=af.FILES['jin']
OUTPUT=HERE/'delivery'
STAGED=HERE/'selected.npz'
FILE=OUTPUT/'jin_shuxian_ankara_stability_full_50fps.mp4'
FOLDER=BASE.parent
START=413440
END=2569216


def get_state():
    meta,core,hist,old=setup()
    new=vj.load_checkpoint(STAGED)
    native=native_scene(meta,core,new)
    if ji.numeric_hash(old['scene_values'])!=ji.numeric_hash(new['scene_values']):
        raise ValueError('Shared geometry changed during stability repair')
    names=[k for k in meta['shots'] if k.startswith('jin_')]
    changed=[k for k in names if not np.array_equal(old[k+'_controls'],new[k+'_controls'])
             or not np.array_equal(old[k+'_rig'],new[k+'_rig'])]
    return meta,hist,old,new,native,names,changed


def render_shot(key):
    rj.cv2.setNumThreads(1)
    m,base,old,state,native,names,changed=get_state()
    d=vj.shot_from_inputs(key,m,base)
    dest=OUTPUT/'shots'/key
    dest.mkdir(parents=True,exist_ok=False)
    rj.draw.TITLE='S. JIN | ANKARA | STABILITY REVIEW'
    world,paths,overview=rj.labels_and_world(native)
    points=rj.latest_points(d,key,ji.context())
    with (dest/'progress.jsonl').open('x',encoding='utf8',buffering=1) as progress:
        rj.clip(d,native,state,dest,world,paths,overview,points,None,progress)
    ji.write(dest/'review.json',dict(geometry_sha256=ji.numeric_hash(state['scene_values']),
        shots=[key],numeric_state=ji.pin(STAGED),path_count=len(paths),unlocalized=len(native.unlocalized)))
    return key


def changed_frames(core,shot,old,new,native):
    # A centre-time equality check by itself cannot certify the output after a
    # change to the rolling-row halo. Render the full changed shot.
    return len(shot.pts)


def bindings(meta,hist,old,state,native,names,changed):
    initial=ji.read(BASE)
    bound=ji.check(initial['camera'])
    z=vj.load_checkpoint(bound['path'])
    assert z['shot_ids'].tolist()==[k[4:] for k in names]
    controls=z['controls'].copy()
    rigs=z['rig_by_source_pts'].copy()
    for i,k in enumerate(names):
        a,b=z['control_offsets'][i:i+2]
        c,d=z['source_offsets'][i:i+2]
        np.testing.assert_array_equal(z['source_pts'][c:d],hist[k+'_pts'])
        np.testing.assert_array_equal(z['local_exposure_seconds'][c:d],hist[k+'_times'])
        np.testing.assert_array_equal(controls[a:b],old[k+'_controls'])
        np.testing.assert_array_equal(rigs[c:d],old[k+'_rig'])
        controls[a:b]=state[k+'_controls']
        rigs[c:d]=state[k+'_rig']
    np.testing.assert_array_equal(z['source_pts'],np.arange(START,END+256,256))
    geometry=ji.numeric_hash(state['scene_values'])
    assert z['scene_values_sha256'].item()==geometry==initial['geometry']['value_sha256']
    z['controls']=controls
    z['rig_by_source_pts']=rigs
    ref=install_npz(FOLDER/'stability_review_1113_camera.npz',z)
    c=ji.check(ref)
    previous=ji.read(HERE.parent/'full_review/jin_shuxian_receipt.json')
    old_clips={x['shot']:x for x in previous['shots']}
    cache={}
    clips=[]
    for i,k in enumerate(names):
        sid=k[4:]
        if k in changed:
            d=OUTPUT/'shots'/k
            clip=ji.read(d/(k+'_receipt.json'))
            chosen={'file':clip['video'],'exact_frames':clip['decoded_frames']}
        else:
            clip=old_clips[sid]
            chosen={'file':clip['file'],'exact_frames':clip['frames']}
        clips.append(af.verify_clip('jin',sid,hist[k+'_pts'],state[k+'_controls'],state[k+'_rig'],
                                   geometry,chosen,cache))
    return initial,z['source_pts'],clips,ref


def join(initial,pts,clips):
    a=OUTPUT/'jin_shuxian.ffconcat'
    body=['ffconcat version 1.0']
    for c in clips:
        body.extend(['file '+af.windows_escape(ji.Path(c['file']['path'])),
                     'duration '+af.seconds(256*c['frames'])])
    with a.open('x',encoding='utf8') as f:
        f.write('\n'.join(body)+'\n')
    assert not af.audio_info(initial['source']['path'])
    temp=FILE.with_suffix('.part.mp4')
    print('Joining',len(clips),len(pts),flush=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
        '-f','concat','-safe','0','-i',str(a),'-map','0:v:0','-an','-c:v','copy',
        '-video_track_timescale','12800','-movflags','+faststart',str(temp)],check=True)
    proof=af.verify_join(temp,pts,clips)
    assert proof['frame_pts_exact'] and proof['decoded_frames']==8422
    assert not af.audio_info(temp)
    temp.replace(FILE)
    return proof


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--render-only',action='store_true')
    p.add_argument('--shots',nargs='+')
    a=p.parse_args()
    t=monotonic()
    meta,hist,old,new,native,names,changed=get_state()
    print('Changed shots',changed,flush=True)
    if not changed:
        raise ValueError('No actual repair was selected')
    OUTPUT.mkdir(exist_ok=True)
    todo=[k for k in changed if (not a.shots or k[4:] in a.shots) and
          not (OUTPUT/'shots'/k/'review.json').is_file()]
    with ProcessPoolExecutor(max_workers=a.workers) as workers:
        for f in as_completed([workers.submit(render_shot,k) for k in todo]):
            print('Rendered',f.result(),round(monotonic()-t),flush=True)
    if a.render_only:
        return
    initial,pts,clips,ref=bindings(meta,hist,old,new,native,names,changed)
    proof=join(initial,pts,clips)
    receipt=dict(programme=initial['programme_id'],scene=initial['geometry'],
        video=ji.pin(FILE),camera=ref,source=initial['source'],source_first_pts=int(pts[0]),source_last_pts=int(pts[-1]),
        source_first_seconds=pts[0]/12800.,source_last_seconds=pts[-1]/12800.,source_time_base='1/12800',
        frame_rate=50,duration_seconds=len(pts)/50.,source_has_audio=False,output_audio=[],
        validation=proof,shots=clips,changed_shots=[k[4:] for k in changed],
        prior_binding=ji.pin(BASE),selection=ji.pin(HERE/'selection.json'))
    q=OUTPUT/'receipt.json'
    ji.write(q,receipt)
    binding=dict(initial)
    binding['review_status']='camera_stability_review_candidate'
    binding['user_visual_acceptance']=False
    binding['previous_binding']=ji.pin(BASE)
    binding['camera']=ref
    binding['stability_validation']=ji.pin(HERE/'selection.json')
    binding['clips']={r['shot']:{'file':r['file'],'exact_frames':r['frames']} for r in clips}
    binding['full_review']={'video':receipt['video'],'receipt':ji.pin(q),
        'original_source_seconds':[receipt['source_first_seconds'],receipt['source_last_seconds']],
        'frames':8422,'frame_rate':50,'exact_source_PTS':True,'audio_preserved_from_source':False}
    f=install_json(guard_project_write(FOLDER/'stability_review_1113.json'),binding)
    print('Verified and registered',f,proof,FILE,round(monotonic()-t),flush=True)


if __name__=='__main__':
    main()
