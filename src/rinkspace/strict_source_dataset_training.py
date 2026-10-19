"""Train the existing vision head from the current immutable source FIT view.

Original source packets stay untouched. Explicit whole-shot and per-row ML
roles, measured support, current point masks and calibration HELD protection
precede supervision formatting. Unknown material is never a negative label.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack, _exact_binding
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_fit_dataset import finite_source_row, _named_pending_indices
from .strict_source_supervision import _positive_annotations_from_authenticated_pack, StrictSourceStreamingTrainer
from .strict_source_review_schedule import reviewed_source_crop_schedule


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def _identity(ref):
    return str(Path(ref['path']).resolve()).casefold(), ref['sha256']


def _source_json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported source supervision value: {type(value).__name__}')


class DatasetSourceTrainingInputs:
    """Keep supervision records after releasing each full decoded source graph."""
    def __init__(self, dataset_reference, *, shot_plan, classes, human_samples=(), pending_filters=(), training_admissions=(), supervision_cache_directory=None, source_video_reference=None, cache_compatibility_reference=None):
        authenticate_source_reference(dataset_reference)
        self.authority = dataset_reference
        self.manifest = json.loads(Path(dataset_reference['path']).read_text(encoding='utf8'))
        if self.manifest.get('schema') != 'rinkspace.immutable-reviewed-finite-source-fit-dataset.v1':
            raise ValueError('Current immutable finite-support source FIT dataset required')
        self.source = self.manifest['source']
        self.source_video_reference = source_video_reference
        if source_video_reference:
            authenticate_source_reference(source_video_reference)
            if source_video_reference['sha256'] != self.source['video_sha256']:
                raise ValueError('Training clock media must be the dataset original video')
        self.classes = classes
        self.annotations_by_native = {}
        self.frame_bindings = {}
        self.counts = Counter()
        self.packet_records = []
        self.extra_pending = {}
        self.extra_literal_record_checks = {}
        self.extra_standard_filters = {}
        self.pending_filter_references = list(pending_filters)
        self.training_admission_references = list(training_admissions)
        self.admitted_points = {}
        self.admitted_entries = {}
        dataset_source_keys = {_identity(packet['source_pack']) for packet in self.manifest['cuts']}
        for ref in training_admissions:
            authenticate_source_reference(ref)
            admission = json.loads(Path(ref['path']).read_text(encoding='utf8'))
            if (admission.get('schema') != 'rinkspace.current-reviewed-exact-source-vision-training-admission.v1'
                    or admission.get('vision_training_role') != 'train'
                    or admission.get('source_pack_roles_unchanged') is not True
                    or admission.get('model_projection_used_as_labels') is not False
                    or admission.get('unknown_pending_used_as_positive_or_negative') is not False
                    or admission.get('other_unlisted_source_roles_promoted') is not False
                    or admission.get('fit_dataset_reference', {}).get('sha256') != dataset_reference['sha256']):
                raise ValueError('Training admission must explicitly bind reviewed source points without rewriting source roles')
            source_ref = admission['source_pack_reference']
            authenticate_source_reference(source_ref)
            key = _identity(source_ref)
            if key not in dataset_source_keys:
                raise ValueError('Training admission source is outside this immutable dataset')
            for witness in admission['source_review_witnesses']:
                authenticate_source_reference(witness)
            points = set()
            for entry in admission['entries']:
                if type(entry['pack_row_index']) is not int or entry['pack_row_index'] < 0:
                    raise ValueError('Training admission needs exact original nonnegative row indices')
                ids = entry['pack_point_indices']
                if not ids or any(type(i) is not int for i in ids) or ids != sorted(set(ids)) or points.intersection(ids):
                    raise ValueError('Training admission exact point indices changed')
                points.update(ids)
            if len(points) != admission['points'] or self.admitted_points.get(key):
                raise ValueError('Training admission must be one unique exact source point receipt')
            self.admitted_points[key] = points
            self.admitted_entries[key] = admission['entries']
        for ref in pending_filters:
            authenticate_source_reference(ref)
            fact = json.loads(Path(ref['path']).read_text(encoding='utf-8-sig'))
            if fact.get('schema') == 'rinkspace.exact-original-source-material-identity-pending.v1':
                if (fact.get('negative_label_created') is not False or
                        fact['frozen_dataset_reference']['sha256'] != dataset_reference['sha256']):
                    raise ValueError('Current source scope pending must preserve dataset and unknown semantics')
                source_ref = fact['source_pack_reference']
                self.extra_literal_record_checks.setdefault(_identity(source_ref),[]).extend(fact['records'])
                groups = [dict(source_pack_reference=source_ref,pending_named_point_count=fact['pending_points'],
                    entries=[dict(pack_point_indices=fact['original_pack_global_point_indices'])])]
            else:
                if fact.get('schema') != 'rinkspace.exact-source-named-identity-pending-filter.v1':
                    raise ValueError('Additional source masks need exact named identity-pending authority')
                groups = fact['source_pack_groups']
                for group in groups:
                    for witness in group['source_review_witnesses']:
                        authenticate_source_reference(witness)
                    self.extra_standard_filters.setdefault(_identity(group['source_pack_reference']), []).append(ref)
            for group in groups:
                source_ref = group['source_pack_reference']
                authenticate_source_reference(source_ref)
                if _identity(source_ref) not in dataset_source_keys:
                    raise ValueError('An additional current point-mask belongs outside this immutable dataset')
                indices = {i for entry in group['entries'] for i in entry['pack_point_indices']}
                if len(indices) != group['pending_named_point_count']:
                    raise ValueError('Current additional identity-pending indices changed')
                self.extra_pending.setdefault(_identity(source_ref), set()).update(indices)
        train_shots = {r['id']:r for r in shot_plan['shots'] if r['split']=='train' and r.get('whole_shot') is True
            and r['source_sha256']==self.source['video_sha256'] and Fraction(r['time_base'])==Fraction(self.source['time_base'])}
        human_by_native = {int(s['source_pts']):s for s in human_samples if s['source_hash']==self.source['video_sha256']}
        protected_exposures, protected_materials, protected_features = set(), set(), set()
        for packet in self.manifest['cuts']:
            authenticate_source_reference(packet['source_pack'])
            rows, offsets = read_minimal_compact_source_headers(packet['source_pack']['path'])
            authenticate_source_reference(packet['support_index'])
            with np.load(packet['support_index']['path'], allow_pickle=False) as z:
                held = z['held_point_indices']
            for ri,row in enumerate(rows):
                lo,hi = map(int, offsets[ri:ri+2])
                a,b = np.searchsorted(held, [lo,hi])
                if b>a or row['split']=='held':
                    protected_exposures.add(row['actual_pts'])
                    protected_materials.add((row['shot_id'],row['physical_material_id']))
                    protected_features.add((row['shot_id'],row['feature_id']))
        self.protected_actual_exposures = protected_exposures
        self.counts['HELD_protected_actual_exposures'] = len(protected_exposures)
        self.counts['HELD_protected_same_shot_materials'] = len(protected_materials)
        self.counts['HELD_protected_same_shot_features'] = len(protected_features)
        self.counts['additional_current_named_pending_unique_points'] = sum(map(len, self.extra_pending.values()))
        self.counts['new_exact_review_training_admitted_points'] = sum(map(len, self.admitted_points.values()))
        cache_dir = Path(supervision_cache_directory).resolve() if supervision_cache_directory else None
        if cache_dir:
            cache_dir.mkdir(parents=True,exist_ok=True)
        cache_environment = dict(source=self.source,classes=classes,train_shots=train_shots,
            human_annotations=[dict(native=int(s['source_pts']),annotations=s['annotations']) for s in human_samples
                if s['source_hash']==self.source['video_sha256']],
            protected_exposures=sorted(protected_exposures),
            protected_materials=sorted(protected_materials,key=repr),protected_features=sorted(protected_features,key=repr),
            formatter_code={str(p):file_sha(p) for p in (Path(__file__),
                Path(_positive_annotations_from_authenticated_pack.__code__.co_filename),
                Path(finite_source_row.__code__.co_filename),Path(read_minimal_compact_source_headers.__code__.co_filename),
                Path(StrictAbsoluteSourcePack.__init__.__code__.co_filename))})
        if cache_compatibility_reference:
            authenticate_source_reference(cache_compatibility_reference)
            compatibility=json.loads(Path(cache_compatibility_reference['path']).read_text(encoding='utf8'))
            adapter_path=str(Path(__file__))
            if (compatibility.get('schema') != 'rinkspace.source-supervision-cache-adapter-compatibility.v1'
                    or compatibility.get('current_adapter_sha256') != cache_environment['formatter_code'][adapter_path]
                    or compatibility.get('source_UV_roles_normal_intervals_and_label_formatter_unchanged') is not True
                    or compatibility.get('change') != 'canonical_original_video_clock_path_instead_of_source_evidence_PNG_path'):
                raise ValueError('Cache compatibility must bind exactly this source clock path-only repair')
            cache_environment['formatter_code'][adapter_path]=compatibility['previous_adapter_sha256']
        environment_digest = hashlib.sha256(json.dumps(cache_environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for packet_number, packet in enumerate(self.manifest['cuts']):
            ref = packet['source_pack']
            minimal,offsets = read_minimal_compact_source_headers(ref['path'])
            admitted = self.admitted_points.get(_identity(ref), set())
            admitted_rows = {e['pack_row_index'] for e in self.admitted_entries.get(_identity(ref), ())}
            if any(ri >= len(minimal) for ri in admitted_rows):
                raise ValueError('Training admission original row is outside the source packet')
            eligible = [i for i,h in enumerate(minimal) if
                (h.get('vision_training_role')=='train' or i in admitted_rows)
                and h['split']=='fit'
                and h['shot_id'] in train_shots]
            if not eligible:
                self.counts['source_packets_without_explicit_train_role'] += 1
                continue
            cache_gate = dict(environment_sha256=environment_digest,source_reference=ref,
                support_index_sha256=packet['support_index']['sha256'],packet_number=packet_number,
                additional_pending_indices=sorted(self.extra_pending.get(_identity(ref),())),
                exact_training_admission_entries=self.admitted_entries.get(_identity(ref),[]))
            cache_key=hashlib.sha256(json.dumps(cache_gate,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            cache_path=cache_dir/f'{cache_key}.json.gz' if cache_dir else None
            receipt_path=cache_dir/f'{cache_key}.receipt.json' if cache_dir else None
            if cache_path and cache_path.exists() and receipt_path.exists():
                cached_receipt=json.loads(receipt_path.read_text(encoding='utf8'))
                authenticate_source_reference(cached_receipt['cache_reference'])
                if cached_receipt['gate'] != cache_gate:
                    raise ValueError('Source supervision cache no longer binds this exact authority and protection gate')
                with gzip.open(cache_path,'rt',encoding='utf8') as cache_file:
                    cached=json.load(cache_file)
                for native,annotations in cached['annotations_by_native'].items():
                    native=int(native);binding=cached['frame_bindings'][str(native)]
                    if source_video_reference:
                        binding['source_path']=source_video_reference['path']
                    if native in self.frame_bindings and self.frame_bindings[native]!=binding:
                        raise ValueError('Cached source supervision changed native frame identity')
                    self.annotations_by_native.setdefault(native,[]).extend(annotations)
                    self.frame_bindings[native]=binding
                self.counts.update(cached['counts'])
                self.counts['supervision_cache_packets_reused']+=1
                self.packet_records.append(dict(source_packet_index=packet_number,source_reference=ref,
                    cut=packet.get('cut'),source_kind=packet.get('source_kind'),support_index_reference=packet['support_index'],
                    supervision_points=cached['supervision_points'],supervision_cache_receipt=reference(receipt_path)))
                print(json.dumps(dict(source_packet=packet_number,supervision_points=cached['supervision_points'],
                    supervision_cache_reused=True)),flush=True)
                continue
            counts_before=self.counts.copy();packet_annotations={};packet_bindings={}
            pack = StrictAbsoluteSourcePack(ref['path'], expected_sha256=ref['sha256'])
            for entry in self.admitted_entries.get(_identity(ref), ()):
                ri = entry['pack_row_index'];h = pack.rows[ri]
                lo,hi = map(int,pack.values['row_offsets'][ri:ri+2]);indices = entry['pack_point_indices']
                if (h.get('vision_training_role') is not None or h['split'] != 'fit'
                        or h['shot_id'] not in train_shots
                        or not train_shots[h['shot_id']]['start_pts'] <= h['native_pts'] < train_shots[h['shot_id']]['end_pts']
                        or any(not lo <= i < hi for i in indices)
                        or any(h[k] != entry[k] for k in ('physical_material_id','feature_id','actual_pts','native_pts','decoded_BGR_sha256'))
                        or pack.values['original_point_indices'][indices].tolist() != entry['original_candidate_indices']):
                    raise ValueError('Exact current training admission differs from original source identity or whole-train role')
            if self.extra_standard_filters.get(_identity(ref)):
                bindings = {int(p):(int(n),bytes(h).decode('ascii')) for p,n,h in zip(
                    pack.values['source_frame_actual_pts'], pack.values['source_frame_native_pts'],
                    pack.values['source_frame_BGR_sha256']) if n>=0 and h}
                checked = _named_pending_indices(pack.rows,pack.values['row_offsets'],ref,
                    self.source,bindings,self.extra_standard_filters[_identity(ref)])
                if not set(map(int,checked)).issubset(self.extra_pending[_identity(ref)]):
                    raise ValueError('Additional source masks lost exact public source-filter indices')
            if any(type(i) is not int or not 0 <= i < len(pack.values['uv']) for i in self.extra_pending.get(_identity(ref),())):
                raise ValueError('Current extra source point-mask lies outside its original pixel array')
            for check in self.extra_literal_record_checks.get(_identity(ref),()):
                ri = check['original_source_row_index']
                h = pack.rows[ri]
                lo,hi = map(int,pack.values['row_offsets'][ri:ri+2])
                indices = check['point_indices']
                if (h['actual_pts'] != check['actual_pts'] or h['physical_material_id'] != check['physical_material_id']
                        or any(not lo <= i < hi for i in indices)
                        or not np.array_equal(pack.values['uv'][indices],np.asarray(check['uv'],float))
                        or pack.values['original_point_indices'][indices].tolist() != check['original_source_point_indices']):
                    raise ValueError('Current scope review lost exact original UV/material/candidate identity')
            authenticate_source_reference(packet['support_index'])
            with np.load(packet['support_index']['path'], allow_pickle=False) as z:
                selection = z['fit_point_indices']
                if not np.array_equal(z['row_offsets'], pack.values['row_offsets']):
                    raise ValueError('Training index changed original source row boundaries')
            produced = 0
            for ri in eligible:
                header = pack.rows[ri]
                lo,hi = map(int, pack.values['row_offsets'][ri:ri+2])
                a,b = np.searchsorted(selection, [lo,hi])
                if b==a:
                    continue
                shot = train_shots[header['shot_id']]
                if not shot['start_pts'] <= header['native_pts'] < shot['end_pts']:
                    self.counts['outside_original_whole_train_shot_points'] += b-a
                    continue
                if (header['actual_pts'] in protected_exposures or
                        (header['shot_id'],header['physical_material_id']) in protected_materials or
                        (header['shot_id'],header['feature_id']) in protected_features):
                    self.counts['HELD_exposure_or_same_shot_entity_protected_points'] += b-a
                    continue
                local = selection[a:b]-lo
                if header.get('vision_training_role') != 'train':
                    local = np.asarray([int(i) for i in local if int(i+lo) in admitted],np.int64)
                extra = self.extra_pending.get(_identity(ref), set())
                keep = np.asarray([int(i) for i in local if int(i+lo) not in extra], np.int64)
                self.counts['additional_current_named_pending_points_excluded'] += len(local)-len(keep)
                if not len(keep):
                    continue
                row = dict(header, **{name:pack.values[name][lo:hi] for name in
                    ('uv','sigma_px','source_sides_uv','raw_candidate_uv','source_side_quantiles_uv','original_point_indices')})
                row = finite_source_row(row, keep, ref, ri, output_split='fit')
                human = human_by_native.get(int(row['native_pts']))
                annotations = _positive_annotations_from_authenticated_pack([row], authority=ref, classes=classes,
                    human_annotations=human['annotations'] if human else ())
                if not annotations:
                    self.counts['missing_actual_normal_or_interval_or_semantic_support_points'] += len(keep)
                    continue
                for annotation in annotations:
                    annotation['group_id'] = f'logical_source_{packet_number}_row_{ri}'
                    annotation['source_dataset_locator'] = row['source_dataset_locator']
                    annotation['vision_training_role'] = 'train'
                    self.counts['supervision_points_' + annotation['kind']] += len(annotation['uv'])
                native = int(row['native_pts'])
                self.annotations_by_native.setdefault(native, []).extend(annotations)
                binding = dict(actual_pts=row['actual_pts'], native_pts=native, decoded_BGR_sha256=row['decoded_BGR_sha256'],
                    shot_id=row['shot_id'], source_path=source_video_reference['path'] if source_video_reference else
                        pack.proof_templates[0]['source_image_evidence']['path'])
                if native in self.frame_bindings and binding != self.frame_bindings[native]:
                    raise ValueError('Current supervised packets disagree on original native frame identity')
                self.frame_bindings[native] = binding
                packet_annotations.setdefault(native,[]).extend(annotations)
                packet_bindings[native]=binding
                produced += len(keep)
                self.counts['supervision_points'] += len(keep)
                self.counts['supervision_groups'] += len(annotations)
            self.packet_records.append(dict(source_packet_index=packet_number, source_reference=ref,
                cut=packet.get('cut'), source_kind=packet.get('source_kind'),
                support_index_reference=packet['support_index'], supervision_points=produced))
            if cache_path:
                counts_delta={k:self.counts[k]-counts_before[k] for k in self.counts if self.counts[k]!=counts_before[k]}
                with gzip.open(cache_path.with_suffix('.temporary'),'wt',encoding='utf8') as cache_file:
                    json.dump(dict(annotations_by_native=packet_annotations,frame_bindings=packet_bindings,
                        counts=counts_delta,supervision_points=produced),cache_file,separators=(',',':'),
                        default=_source_json_default)
                cache_path.with_suffix('.temporary').replace(cache_path)
                receipt_path.write_text(json.dumps(dict(gate=cache_gate,cache_reference=reference(cache_path)),
                    sort_keys=True,separators=(',',':'))+'\n',encoding='utf8')
                self.packet_records[-1]['supervision_cache_receipt']=reference(receipt_path)
                self.counts['supervision_cache_packets_created']+=1
            print(json.dumps(dict(source_packet=packet_number, supervision_points=produced)), flush=True)
            del pack, minimal
            gc.collect()
        authenticate_source_reference(dataset_reference)
        self.counts['supervision_native_frames'] = len(self.frame_bindings)
        self.counts['supervision_actual_exposures'] = len({v['actual_pts'] for v in self.frame_bindings.values()})
        if not self.frame_bindings:
            raise ValueError('No current measured labels have explicit whole-shot and row training authority')

    def exact_frame_binding(self, native):
        return dict(self.frame_bindings[int(native)], source_sha256=self.source['video_sha256'],time_base=self.source['time_base'])

    def supervision_catalog(self, native, *, human_annotations=()):
        return dict(annotations=list(human_annotations)+self.annotations_by_native.get(int(native),[]), negatives=[], semantic_annotations=[])

    def frame_supervision(self, bgr, binding, *, human_annotations=()):
        current = _exact_binding(binding)
        item = self.frame_bindings[current['native_pts']]
        digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
        if digest != item['decoded_BGR_sha256'] or current['actual_pts'] != item['actual_pts']:
            raise ValueError('Training decoded original source differs from its literal point proof')
        return self.supervision_catalog(current['native_pts'], human_annotations=human_annotations)


def train(config_path):
    import av
    import random
    import torch
    from .vision import configure,load_training_manifest,validate_training_sources,_image,_tensor,_annotations_in_crop,source_reviewed_region_masks,write_json,_atomic_checkpoint
    from .vision_network import RinkVisionNet
    from .vision_retention import source_retention_snapshot,source_retention_regressions
    from .vision_head_guard import build_head_support_cache
    config,data = configure(config_path)
    spec = config['strict_source_dataset_training']
    if config.get('current_named_pending_review_filters_required_before_launch') and not spec.get('additional_current_pending_filters'):
        raise ValueError('Current source identity review masks must be bound before any GPU training')
    torch.set_num_threads(int(spec.get('cpu_threads',1)))
    if not torch.cuda.is_available():
        raise RuntimeError('The configured original vision GPU is required')
    checkpoint = Path(spec['resume_checkpoint']).resolve()
    checkpoint_ref = reference(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('optimizer') is None or saved.get('classes') != config['classes']:
        raise ValueError('Existing complete model and optimizer with unchanged classes are required')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    if saved.get('torch_rng_state') is not None:
        torch.set_rng_state(saved['torch_rng_state'])
    if saved.get('cuda_rng_states') is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
    if saved.get('numpy_rng_state') is not None:
        np.random.set_state(saved['numpy_rng_state'])
    if saved.get('python_rng_state') is not None:
        random.setstate(saved['python_rng_state'])
    manifest,plan = load_training_manifest(config,data)
    validate_training_sources(manifest)
    print(json.dumps(dict(status='current_dataset_training_preparation_started',
        resume_checkpoint=checkpoint_ref, fit_dataset=spec['fit_dataset'],
        current_pending_filters=spec.get('additional_current_pending_filters', []))), flush=True)
    inputs = DatasetSourceTrainingInputs(spec['fit_dataset'],shot_plan=plan,classes=config['classes'],
        human_samples=manifest['samples'],pending_filters=spec.get('additional_current_pending_filters',()),
        training_admissions=spec.get('exact_current_review_training_admissions',()),
        supervision_cache_directory=data/'current_review_source_supervision_cache',
        source_video_reference=dict(path=str(Path(config['source_video']).resolve()),
            sha256=json.loads(Path(spec['fit_dataset']['path']).read_text(encoding='utf8'))['source']['video_sha256']),
        cache_compatibility_reference=spec.get('label_cache_adapter_compatibility'))
    run_data = Path(config['training_run_directory']).resolve()
    if not run_data.is_relative_to(data):
        raise ValueError('Training checkpoints, logs and index bundles must stay in Data')
    run_data.mkdir(parents=True, exist_ok=True)
    if spec.get('prepare_supervision_only') is True:
        result=dict(schema='rinkspace.current-source-vision-supervision-preparation.v1',
            status='source_supervision_prepared_optimizer_not_started',fit_dataset=spec['fit_dataset'],
            counts={key:int(value) for key,value in inputs.counts.items()},packet_records=inputs.packet_records,
            pending_filter_references=inputs.pending_filter_references,
            exact_review_training_admission_references=inputs.training_admission_references,
            formal_model_replaced=False,optimizer_updates=0)
        write_json(run_data/f'source_supervision_preparation_{spec["fit_dataset"]["sha256"]}.json',result)
        return result
    human_by_native = {int(s['source_pts']):s for s in manifest['samples'] if s['source_hash']==inputs.source['video_sha256']}
    catalog = []
    for native in sorted(inputs.frame_bindings):
        human = human_by_native.get(native)
        supervision = inputs.supervision_catalog(native,human_annotations=human['annotations'] if human else ())
        catalog.append((native,supervision['annotations'],supervision['negatives']))
    schedule, schedule_record = reviewed_source_crop_schedule(catalog,crop_size=config['crop_size'],native_hw=(1080,1920),
        optimizer_steps=config['training_steps'],reviewed_source_sha256s=[key[1] for key in inputs.admitted_points])
    bundle = dict(schema='rinkspace.current-logical-source-vision-training-bundle.v1',fit_dataset=spec['fit_dataset'],
        source_video_reference=inputs.source_video_reference,
        label_cache_adapter_compatibility=spec.get('label_cache_adapter_compatibility'),
        pending_filter_references=inputs.pending_filter_references,source=inputs.source,
        exact_review_training_admission_references=inputs.training_admission_references,
        packet_records=inputs.packet_records,counts={key:int(value) for key,value in inputs.counts.items()},crop_schedule=schedule_record,
        crop_origins_by_native={str(k):v for k,v in schedule.items()},
        existing_whole_shot_plan_unchanged=True,explicit_row_train_role_or_exact_new_review_receipt_required=True,
        all_HELD_source_exposures_and_same_shot_entities_protected=True,
        unknown_pending_missing_used_as_positive_or_negative=False,model_projection_used_as_labels=False,
        original_sigma_and_UV_unchanged=True,source_numeric_arrays_copied_to_bundle=False,
        user_stopped_reflection_experiment_resumed=False,confirmed_negative_archives_preserved_not_relabelled=True)
    bundle_path = run_data/'current_source_training_bundle.json'
    if bundle_path.exists():
        raise FileExistsError('Preserve earlier source training bundle')
    write_json(bundle_path,bundle)
    network = RinkVisionNet(**saved['model_config']).to(config['device'])
    network.load_state_dict(saved['state_dict'],strict=True)
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = config['learning_rate']
    if config['training_update_scope'] != 'semantic_head':
        raise ValueError('This guarded source bundle uses the existing semantic-head correction configuration')
    for name,parameter in network.named_parameters():
        parameter.requires_grad_(name.startswith('semantic_head.'))
    trainer = StrictSourceStreamingTrainer(network,optimizer,inputs,classes=config['classes'],device=config['device'],
        crop_size=config['crop_size'],human_position_weight=config.get('human_position_localization_weight',1.),
        primitive_weight=config.get('primitive_supervision_weight',.2),position_rank_weight=config.get('human_position_rank_weight',0.))
    guards = [s for s in manifest['samples'] if s.get('regions') or any(
        a.get('state')=='user_confirmed_visual_supervision' for a in s['annotations'])]
    if not guards:
        raise ValueError('Current correction must retain original confirmed source supervision')
    head_support = []
    def snapshots(capture=False):
        network.eval()
        values=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for sample in guards:
                rgb=_image(sample['frame_path'],sample.get('frame_key'));h,w=rgb.shape[:2]
                refined=[]
                hook=network.semantic_head.register_forward_pre_hook(lambda module,args:refined.append(args[0].detach())) if capture else None
                output=network(_tensor(rgb[None],config['device']))
                if hook:hook.remove()
                annotations=_annotations_in_crop(sample,(0,0),(h,w))
                regions=[(dict(r,region_id=r.get('region_id',str(i))),mask) for i,(r,mask) in enumerate(
                    source_reviewed_region_masks(sample,(0,0),(h,w),config['device']))]
                binding=dict(source_sha256=sample['source_hash'],source_pts=sample['source_pts'],time_base=sample['time_base'],
                    shot_id=sample['source_shot_scope']['shot_id'],native_hw=[h,w])
                snap=source_retention_snapshot(output['logits'][:,0].float().softmax(1),
                    output['primitive_logits'][:,0].float().softmax(1),source_binding=binding,
                    annotations=annotations,classes=config['classes'],reviewed_regions=regions)
                values.append(snap)
                if capture:
                    support=build_head_support_cache(refined[0],output['logits'][:,0].float().softmax(1),sample=sample,
                        source_binding=binding,classes=config['classes'],reference_weight=network.semantic_head.weight,
                        reference_bias=network.semantic_head.bias,reference_snapshot=snap,reviewed_regions=regions,
                        retain_native_cuda_features=True)
                    if not support['native_GPU_baseline_decisions_verified']:
                        raise ValueError('Original native retention guard does not reproduce the trained model')
                    head_support.append(support)
                del output
        network.train()
        return values
    baseline = snapshots(capture=True)
    trainer.guarded_head_support=head_support
    run_id=time.time_ns()
    ledger=run_data/f'current_source_training_consumption_{run_id}.jsonl'
    comparisons=[];failed=False;last_guard=0;frames_consumed=0
    source_path=Path(config['source_video']).resolve()
    authenticate_source_reference(dict(path=str(source_path),sha256=inputs.source['video_sha256']))
    print(json.dumps(dict(status='current_dataset_GPU_training_started',source_native_frames=len(schedule),
        source_range_pts=[min(schedule),max(schedule)],crop_schedule=schedule_record,checkpoint=checkpoint_ref,
        bundle=reference(bundle_path))),flush=True)
    with av.open(str(source_path)) as container:
        stream=container.streams.video[0];stream.thread_count=int(spec.get('cpu_threads',1))
        if Fraction(stream.time_base)!=Fraction(inputs.source['time_base']):
            raise ValueError('Actual training video clock differs from original source evidence')
        container.seek(min(schedule),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or int(frame.pts)<min(schedule):continue
            if int(frame.pts)>max(schedule):break
            native=int(frame.pts)
            if native not in schedule:continue
            bgr=frame.to_ndarray(format='bgr24');binding=inputs.exact_frame_binding(native)
            human=human_by_native.get(native)
            result=trainer.consume_frame(bgr,binding,shot_role='train',human_annotations=human['annotations'] if human else (),
                crop_origins=schedule[native],max_optimizer_steps=config['training_steps']-trainer.successful_steps)
            frames_consumed+=1
            with ledger.open('a',encoding='utf8') as record:
                record.write(json.dumps(result,separators=(',',':'))+'\n')
            print(json.dumps(dict(native_pts=native,optimizer_steps=trainer.successful_steps,source_crops=trainer.source_crops_consumed)),flush=True)
            if trainer.successful_steps-last_guard>=config.get('source_retention_guard_every',20):
                comparisons=[source_retention_regressions(current,prior) for current,prior in zip(snapshots(),baseline)]
                last_guard=trainer.successful_steps
                failed=any(not r['retained_without_new_classification_errors'] for r in comparisons)
                if failed:break
            if trainer.successful_steps>=config['training_steps']:break
    comparisons=[source_retention_regressions(current,prior) for current,prior in zip(snapshots(),baseline)]
    failed=failed or any(not r['retained_without_new_classification_errors'] for r in comparisons)
    candidate=run_data/f'checkpoint_current_source_candidate_{run_id}.pt'
    _atomic_checkpoint(candidate,dict(saved,model_config=network.model_config,state_dict=network.state_dict(),
        optimizer=optimizer.state_dict(),step=int(saved['step'])+trainer.successful_steps,
        torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),
        numpy_rng_state=np.random.get_state(),python_rng_state=random.getstate(),
        current_source_training_bundle=reference(bundle_path),resume_checkpoint=checkpoint_ref))
    authenticate_source_reference(checkpoint_ref)
    authenticate_source_reference(inputs.authority)
    report=dict(schema='rinkspace.current-logical-source-vision-training-consumption.v1',
        status='source_retention_failed' if failed else 'current_source_candidate_trained',candidate_checkpoint=reference(candidate),
        successful_optimizer_steps=trainer.successful_steps,source_crops_consumed=trainer.source_crops_consumed,
        scheduled_source_crops=schedule_record['scheduled_crop_count'],frames_consumed=frames_consumed,
        bundle_reference=reference(bundle_path),consumption_reference=reference(ledger),resume_checkpoint=checkpoint_ref,
        original_native_retention_comparisons=comparisons,original_historical_RNG_available=saved.get('torch_rng_state') is not None,
        original_model_and_optimizer_restored=True,formal_model_replaced=False,user_stopped_reflection_experiment_resumed=False,
        independent_whole_shot_accuracy_accepted=False)
    # The public inference loader deliberately uses weights_only=True. Keep
    # Python/numpy RNG in the complete checkpoint and publish a compatible
    # inference candidate separately, without replacing the formal model.
    candidate_model = run_data/f'rink_vision_current_source_candidate_{run_id}.pt'
    inference_scope = dict(report, latest_step=int(saved['step'])+trainer.successful_steps,
        accuracy_accepted=False, source_retention_guard=dict(failed=failed))
    _atomic_checkpoint(candidate_model,dict(model_config=network.model_config,state_dict=network.state_dict(),
        classes=config['classes'],training_report=inference_scope,accuracy_accepted=False))
    report['candidate_model'] = reference(candidate_model)
    write_json(run_data/f'current_source_training_consumption_{run_id}.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(train(args.config),ensure_ascii=False),flush=True)
