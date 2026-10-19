"""One portable launch entry. Run `python launch.py --help`."""
from pathlib import Path
import argparse
import json
import sys

from .pipeline import PACK_ROOT, load_context, prepare, register, diagnose, build_graphs, metric_initializers, read_run
from .runtime import json_value, atomic_json, now


def parser():
    p=argparse.ArgumentParser(description='Okada shot 1 v1.1 architecture: immutable inputs, S1 registration, S2 graph, Ceres handoff')
    sub=p.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect',help='Validate ALL source arrays and show initialization coverage')
    inspect.add_argument('--data-dir',type=Path,default=PACK_ROOT/'data')
    pre=sub.add_parser('prepare',help='Freeze all-shot ledger, immutable input hashes and run options')
    pre.add_argument('--data-dir',type=Path,default=PACK_ROOT/'data');pre.add_argument('--output',type=Path,required=True)
    pre.add_argument('--config',type=Path)
    start=sub.add_parser('start',help='Prepare and start all-shot S1 jobs; does not run or certify a physical event fit')
    start.add_argument('--data-dir',type=Path,default=PACK_ROOT/'data');start.add_argument('--output',type=Path,required=True)
    start.add_argument('--config',type=Path);start.add_argument('--workers',type=int)
    reg=sub.add_parser('register',help='Run/resume all scheduled S1 jobs, keeping all source obligations')
    reg.add_argument('--run-dir',type=Path,required=True);reg.add_argument('--workers',type=int)
    diag=sub.add_parser('diagnose',help='Explicit small initialization diagnostic; never shrinks full-shot scope')
    diag.add_argument('--run-dir',type=Path,required=True);diag.add_argument('--frame',type=int,required=True)
    diag.add_argument('--plane',required=True)
    for name,help_text in [('graph','Build real-corner pairwise graphs over all 1248 frames'),
                           ('bootstrap','Convert all historical camera seeds to named fixed-head/cubic initialization state'),
                           ('upgrade','Build conditional shared-P initializers from same-frame rank-8 planar results'),
                           ('export-backend','Export every source row to portable Ceres event packet'),
                           ('status','Read last progress record'),('stop','Request cooperative pause after current S1 jobs'),
                           ('clear-stop','Clear only this run\'s cooperative pause marker')]:
        a=sub.add_parser(name,help=help_text);a.add_argument('--run-dir',type=Path,required=True)
    fit=sub.add_parser('fit',help='Launch a built EVENT adapter; rejects the supplied synthetic Ceres demo')
    fit.add_argument('--run-dir',type=Path,required=True);fit.add_argument('--backend',type=Path,required=True)
    fit.add_argument('--resume',type=Path)
    fit.add_argument('--packet-manifest',type=Path)
    fit.add_argument('--physical-run-dir',type=Path)
    fit.add_argument('--runtime',type=Path)
    fit.add_argument('--extend-resource-from',type=Path)
    check=sub.add_parser('verify-native',help='Independent all-source identity and raw pixel arithmetic check')
    check.add_argument('--run-dir',type=Path,required=True);check.add_argument('--predictions',type=Path,required=True)
    curves=sub.add_parser('curve-graph',help='Direct original two-frame line/conic graph; full source scope retained')
    curves.add_argument('--run-dir',type=Path,required=True);curves.add_argument('--workers',type=int,default=8)
    atlas=sub.add_parser('atlas-upgrade',help='Estimate unknown component Q and conditional common camera candidates')
    atlas.add_argument('--run-dir',type=Path,required=True)
    replay=sub.add_parser('replay-physical',help='Independent all-source scene, fixed-head, RS/profile and grouped-W replay')
    replay.add_argument('--run-dir',type=Path,required=True);replay.add_argument('--model',type=Path,required=True)
    replay.add_argument('--state',type=Path,required=True);replay.add_argument('--output-dir',type=Path)
    replay.add_argument('--semantic-manifest',type=Path)
    replay.add_argument('--packet-manifest',type=Path)
    finalize=sub.add_parser('finalize-packet',help='Freeze full semantic/model/station inputs for the physical event backend')
    finalize.add_argument('--run-dir',type=Path,required=True)
    finalize.add_argument('--model',type=Path)
    finalize.add_argument('--initial-state-dir',type=Path)
    finalize.add_argument('--manifest',type=Path)
    finalize.add_argument('--semantic-manifest',type=Path)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.command=='inspect':
            obs,seeds,supports,pending=load_context(a.data_dir)
            result={'source':obs.summary(),'S1_supported_materials':len(supports),'S1_pending_materials':pending,
                    'coverage':seeds.coverage(obs),'physical_event_adapter_included':True,
                    'event_adapter_source_present_is_not_a_fit_certificate':True}
        elif a.command=='prepare':result=prepare(a.data_dir,a.output,a.config)
        elif a.command=='start':
            prepare(a.data_dir,a.output,a.config);result=register(a.output,a.workers)
        elif a.command=='register':result=register(a.run_dir,a.workers)
        elif a.command=='diagnose':result=diagnose(a.run_dir,a.frame,a.plane)
        elif a.command=='graph':result=build_graphs(a.run_dir)
        elif a.command=='bootstrap':
            from .bootstrap import bootstrap_state
            from .runtime import run_lock
            root,run=read_run(a.run_dir);data=Path(run['data_dir'])
            with run_lock(root):
                state=bootstrap_state(data/'seeds.json',data/'camera_seeds.npz')
                atomic_json(root/'bootstrap_state.json',state)
            result={'output':str(root/'bootstrap_state.json'),'status':state['status'],
                    'blockers':state['blockers'],'physical_fit_completed':False}
        elif a.command=='upgrade':result=metric_initializers(a.run_dir)
        elif a.command=='export-backend':
            from .event_packet import export_packet
            result=export_packet(a.run_dir)
        elif a.command=='fit':
            from .event_packet import start_physical_backend
            result=start_physical_backend(a.run_dir,a.backend,a.resume,a.packet_manifest,
                a.physical_run_dir,a.runtime,a.extend_resource_from)
        elif a.command=='verify-native':
            from .event_packet import verify_native_export
            result=verify_native_export(a.run_dir,a.predictions)
        elif a.command=='curve-graph':
            from .curve_graph import build_curve_graph
            result=build_curve_graph(a.run_dir,a.workers)
        elif a.command=='atlas-upgrade':
            from .atlas_upgrade import upgrade_component_atlases,common_camera_from_atlases
            result={'charts':upgrade_component_atlases(a.run_dir),'common_camera':common_camera_from_atlases(a.run_dir)}
        elif a.command=='replay-physical':
            from .native_replay import replay_native
            result=replay_native(a.run_dir,a.model,a.state,a.output_dir,
                semantic_manifest_path=a.semantic_manifest,packet_manifest_path=a.packet_manifest)
        elif a.command=='finalize-packet':
            from .event_packet import finalize_event_packet
            result=finalize_event_packet(a.run_dir,a.model,a.initial_state_dir,a.manifest,a.semantic_manifest)
        elif a.command=='status':
            root=Path(a.run_dir).resolve();path=root/'status.json'
            result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'not_started'}
        elif a.command=='stop':
            root,run=read_run(a.run_dir);atomic_json(root/'STOP_REQUESTED',{'utc':now(),'scope':'S1 cooperative pause'})
            result={'status':'pause_requested','active_jobs_finish_with_complete_local_checkpoints':True}
        elif a.command=='clear-stop':
            root,run=read_run(a.run_dir);(root/'STOP_REQUESTED').unlink(missing_ok=True)
            result={'status':'pause_marker_cleared','resume_command':'register'}
        else:raise AssertionError(a.command)
        print(json.dumps(json_value(result),ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed job manifests can be resumed; incomplete jobs are not accepted.',file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({'error':str(exc),'type':type(exc).__name__,'physical_fit_accepted':False},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
