"""Close archived camera-computation inputs without traversing original media."""
from pathlib import Path

from .io import load_json, sha256


def collect_continuity_dependencies(root, run_paths, *, dependency_runs=(), reverse_evidence=(),implementation_snapshots=()):
    """Collect exact JSON/NPZ numerical inputs, including nested input runs.

    Reverse passes historically recorded input hashes without paths. Their
    directories must be explicitly supplied (or reached by a graph input link),
    then the hashes are resolved and verified. Never search unrelated projects
    or claim that archived image observations can regenerate themselves without
    the original source video.
    """
    root=Path(root).resolve();files={};records={};queue=[];bindings=[];reverse_roots=[];code_bindings=[]
    def safe(path):
        path=(root/path).resolve(strict=True)
        if not path.is_relative_to(root):raise ValueError(f'Numerical dependency leaves project: {path}')
        return path
    def add(path,reason,expected=None):
        path=safe(path)
        if not path.is_file() or path.suffix.lower() not in {'.json','.npz','.py'}:
            raise ValueError(f'Unsupported numerical archive input: {path}')
        actual=sha256(path)
        if expected is not None and expected!=actual:raise ValueError(f'Numerical input SHA mismatch: {path}')
        files[path]={'path':path.relative_to(root).as_posix(),'sha256':actual,'bytes':path.stat().st_size,'reason':reason}
        return path
    def schedule(path,expected=None):
        path=safe(path)
        file=path/'localization.json' if path.is_dir() else path
        add(file,'archived numerical run needed to replay camera graph or merge',expected)
        if file not in records and file not in queue:queue.append(file)
        return file
    code_candidates=[]
    for relative in implementation_snapshots:
        directory=safe(relative)
        code_candidates.extend(p for p in directory.rglob('*.py') if p.is_file())
    code_digest_cache={}
    def bind_code(owner,value,default=None):
        if not value:return
        declarations=value if isinstance(value,dict) else {default:value}
        aliases={'edge_refit':'src/rinkspace/continuity_edge_refit.py','camera_graph':'src/rinkspace/continuity_graph.py',
                 'experiment_runner':'scripts/refit_continuity_edges.py','visual_checker':'scripts/check_continuity_visual.py'}
        for name,expected in declarations.items():
            if not name:raise ValueError(f'Implementation path is missing for {owner}')
            relative=aliases.get(name,name)
            guesses=[root/relative,root/'src/rinkspace'/relative,*code_candidates]
            found=None
            for path in guesses:
                if not path.is_file():continue
                if path not in code_digest_cache:code_digest_cache[path]=sha256(path)
                if code_digest_cache[path]==expected:found=path;break
            if found is None:raise ValueError(f'Missing exact implementation {name} SHA {expected}; supply its --implementation-snapshot')
            add(found,'exact implementation of archived numerical graph or reverse merge',expected)
            code_bindings.append({'owner':str(owner.relative_to(root)),'declared_name':name,
                                  'path':found.relative_to(root).as_posix(),'sha256':expected})
    for path in [*run_paths,*dependency_runs]:schedule(path)
    for relative in reverse_evidence:
        directory=safe(relative)
        if not directory.is_dir():raise ValueError('Reverse evidence must be a directory')
        for name in ('reverse_raw_observations.json','reverse_raw_observations.npz','camera_graph.json','merge_diagnostics.json'):
            add(directory/name,'archived independently reobserved reverse ice pairs and merge policy')
        raw=load_json(directory/'reverse_raw_observations.json')
        bind_code(directory/'reverse_raw_observations.json',raw.get('implementation_sha256'))
        declared=raw.get('storage',{}).get('file')
        if declared and declared!='reverse_raw_observations.npz':
            add(directory/declared,'explicit reverse point storage archive')
        for child in ('reverse_graph','merged'):schedule(directory/child)
        reverse_roots.append(directory)
    while queue:
        file=queue.pop(0)
        if file in records:continue
        record=load_json(file);record.pop('frames',None);records[file]=record;directory=file.parent
        context=record.get('solver_context') or {}
        point_store=record.get('local_correspondence_storage') or {}
        for key,default in ((context,'solver_context.json'),(point_store,'local_observations.npz')):
            # Older runs used a prose storage description; it is not a path.
            if isinstance(key,str) and Path(key).suffix.lower() not in {'.json','.npz'}:key={}
            name=key if isinstance(key,str) else key.get('file',default)
            if not isinstance(name,str) or Path(name).name!=name:raise ValueError('Numerical sidecar name must stay in its run directory')
            if key or (directory/name).is_file():add(directory/name,'complete context or real point-pair archive for replay')
        for name in ('camera_graph.json','metadata_revision.json','binding_revision.json','athlete_continuity.json','edge_comparisons.json','visual_checks.json'):
            if (directory/name).is_file():add(directory/name,'numerical provenance, disclosed metadata revision or posthoc checks')
        graph=record.get('joint_graph') or {}
        bind_code(file,graph.get('implementation_sha256'),'src/rinkspace/continuity_graph.py')
        bind_code(file,graph.get('merge_implementation_sha256'),'scripts/track_reverse_continuity.py')
        if graph.get('file'):add(directory/graph['file'],'actual camera-graph diagnostics')
        for direction,entry in (graph.get('input_records') or {}).items():
            if not isinstance(entry,dict) or not entry.get('path'):raise ValueError('Graph input record needs its exact archived path')
            expected=entry.get('localization_sha256')
            if not expected:raise ValueError('Graph input record needs its localization SHA')
            target=schedule(entry['path'],expected)
            if entry.get('observations_sha256'):
                add(target.parent/'local_observations.npz','frozen forward/reverse graph observations',entry['observations_sha256'])
            bindings.append({'output':file.relative_to(root).as_posix(),'input':target.relative_to(root).as_posix(),
                             'kind':f'joint_graph_input_{direction}','exact_input_sha256_checked':True})
        if graph.get('input_run'):
            expected=graph.get('input_sha256',graph.get('input_localization_sha256'))
            target=schedule(graph['input_run'],expected)
            if graph.get('input_observations_sha256'):
                add(target.parent/'local_observations.npz','frozen graph input observations',graph['input_observations_sha256'])
            bindings.append({'output':file.relative_to(root).as_posix(),'input':target.relative_to(root).as_posix(),
                             'kind':'joint_graph_input','exact_input_sha256_checked':expected is not None})
        refit=record.get('edge_refit_experiment') or {}
        bind_code(file,refit.get('implementation_sha256'))
        if refit.get('source_run'):
            target=schedule(refit['source_run'],refit.get('source_localization_sha256'))
            for name,key in [('solver_context.json','source_solver_context_sha256'),('local_observations.npz','source_observation_npz_sha256')]:
                add(target.parent/name,'frozen input for edge refit comparison',refit.get(key))
            bindings.append({'output':file.relative_to(root).as_posix(),'input':target.relative_to(root).as_posix(),
                             'kind':'edge_refit_input','exact_input_sha256_checked':True})
        gate=record.get('conflict_publication_gate') or {}
        bind_code(file,gate.get('implementation_sha256'),'src/rinkspace/continuity_conflict_gate.py')
        if gate.get('diagnostics_file'):add(directory/gate['diagnostics_file'],'disclosed publication-conflict gate diagnostics')
        if gate.get('input_path'):
            target=schedule(gate['input_path'],gate.get('input_localization_sha256'))
            for name,key in [('solver_context.json','input_context_sha256'),('local_observations.npz','input_observations_sha256'),
                             ('camera_graph.json','input_camera_graph_sha256')]:
                if gate.get(key):add(target.parent/name,'exact input to publication conflict gate',gate[key])
            bindings.append({'output':file.relative_to(root).as_posix(),'input':target.relative_to(root).as_posix(),
                             'kind':'conflict_publication_gate_input','exact_input_sha256_checked':bool(gate.get('input_localization_sha256'))})
        reverse=record.get('reverse_pass_evidence') or {}
        bind_code(file,reverse.get('implementation_sha256'))
        for key in ('forward_input_run','forward_graph_input_run'):
            if reverse.get(key):schedule(reverse[key],reverse.get(key.replace('_run','_sha256')))
    digest_paths={files[path]['sha256']:path for path in records}
    for file,record in records.items():
        reverse=record.get('reverse_pass_evidence') or {}
        if not reverse:continue
        for key in ('forward_input_sha256','forward_graph_input_sha256'):
            expected=reverse.get(key)
            if not expected:raise ValueError(f'Reverse archive omits {key}: {file}')
            target=digest_paths.get(expected)
            if target is None:
                raise ValueError(f'Missing reverse input {key}={expected}; supply its exact --dependency-run directory')
            bindings.append({'output':file.relative_to(root).as_posix(),'input':target.relative_to(root).as_posix(),
                             'kind':key,'exact_input_sha256_checked':True})
        if not reverse_roots:raise ValueError('Reverse run requires --reverse-evidence with raw observations and merge diagnostics')
        # Bind this reverse result to the explicitly supplied archived source,
        # not merely to a similarly named directory.
        candidates=[]
        for directory in reverse_roots:
            raw=load_json(directory/'reverse_raw_observations.json')
            if raw.get('source_sha256')==record.get('source_sha256') and raw.get('implementation_sha256')==reverse.get('implementation_sha256'):
                candidates.append(directory)
        if len(candidates)!=1:raise ValueError('Reverse raw observation identity must match exactly one declared archive')
        bindings.append({'output':file.relative_to(root).as_posix(),
                         'input':(candidates[0]/'reverse_raw_observations.json').relative_to(root).as_posix(),
                         'kind':'reverse_raw_observation_identity','source_and_implementation_identity_checked':True})
    return {'schema_version':'rinkspace.archived-continuity-dependencies.v1','files':list(files.values()),
        'bindings':bindings,'implementation_bindings':code_bindings,'run_count':len(records),'source_video_included':False,
        'replay_scope':'Archived real image observations to camera graph and merge; generating new image observations still needs originals.'}
