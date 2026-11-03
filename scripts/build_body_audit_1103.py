"""Build the current S01 private audit supplement without fitting or inference.

Reuses the existing review Bundle; public sources are pinned, not copied.
Every array in the five current output NPZs survives lossless row replacement.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import io
import json
import subprocess
import sys
import zipfile
import zlib

import numpy as np
from build_body_review import Bundle, digest, load, norm
from audit_body_audit_1103 import Store, metrics


COMMIT = '78dba301449d49c429b7c3fee7717aea1a61cf7a'
GITHUB = 'https://github.com/gsbetablocker/RinkSpace'
PROGRAMME = 'events/ankara_2026/programmes/okada_mei_sp'


def npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


class Arrays:
    def __init__(self, bundle):
        self.bundle = bundle
        self.datasets = {}
        self.loaded = {}
        self.known = {}
        self.blobs = {}
        self.blob_encodings = {}

    @staticmethod
    def identity(value):
        return value.dtype.str, tuple(value.shape), digest(value.tobytes(order='C'))

    @staticmethod
    def encoded(value):
        raw = npy(value)
        metadata = None
        cost = len(zlib.compress(raw, 9))
        if value.dtype.kind in 'fiu' and value.dtype.itemsize > 1 and value.size > 64:
            planes = np.frombuffer(value.tobytes(order='C'), np.uint8).reshape(-1, value.dtype.itemsize).T.copy()
            alternative = npy(planes)
            other_cost = len(zlib.compress(alternative, 9))
            if other_cost + 120 < cost:
                raw, cost = alternative, other_cost
                metadata = dict(encoding='byte_planes', dtype=value.dtype.str, shape=list(value.shape))
        return raw, cost, metadata

    def blob(self, value):
        raw, _, metadata = self.encoded(value)
        sha = digest(raw)
        if sha not in self.blobs:
            self.blobs[sha] = self.bundle.add('arrays/'+sha+'.npy', raw, kind='lossless numeric array storage')
        if metadata:
            self.blob_encodings[self.blobs[sha]] = metadata
        return self.blobs[sha]

    def add(self, name, path, parent=None):
        path = Path(path)
        raw = path.read_bytes()
        packet = load(path)
        base = self.loaded.get(parent, {})
        fields = {}
        rows = len(packet['pts']) if 'pts' in packet else None
        # Register the gear subarrays so duplicate boot/blade arrays become slices.
        keys = sorted(packet, key=lambda k: (k != 'gear_world_mid', k))
        for key in keys:
            value = packet[key]
            identity = self.identity(value)
            descriptor = dict(dtype=value.dtype.str, shape=list(value.shape),
                              raw_sha256=identity[2], uncompressed_bytes=value.nbytes)
            if identity in self.known:
                descriptor.update(self.known[identity])
            elif (key in base and value.dtype == base[key].dtype and value.shape == base[key].shape
                  and value.ndim and rows and len(value) == rows):
                left = np.ascontiguousarray(value).view(np.uint8).reshape(rows, -1)
                right = np.ascontiguousarray(base[key]).view(np.uint8).reshape(rows, -1)
                changed = np.flatnonzero(np.any(left != right, axis=1))
                replace = value[changed]
                full_cost = self.encoded(value)[1]
                row_cost = self.encoded(replace)[1]+self.encoded(changed)[1]+200
                # XOR operates on bytes, preserving floating-point bit patterns exactly.
                xor = np.bitwise_xor(left.reshape(-1, value.dtype.itemsize).T,
                                     right.reshape(-1, value.dtype.itemsize).T).copy()
                xor_cost = self.encoded(xor)[1]+200
                if xor_cost < min(full_cost, row_cost):
                    descriptor.update(storage='xor_bytes', base=[parent, key], mask=self.blob(xor))
                elif row_cost < full_cost:
                    descriptor.update(storage='rows', base=[parent, key], indices=self.blob(changed),
                                      values=self.blob(replace), changed_rows=len(changed), total_rows=rows)
                else:
                    descriptor.update(storage='npy', member=self.blob(value))
            else:
                descriptor.update(storage='npy', member=self.blob(value))
            fields[key] = descriptor
            self.known.setdefault(identity, dict(storage='alias', target=[name, key]))
            if key == 'gear_world_mid' and value.shape[1:] == (2, 48, 3):
                for start, stop in ((0, 14), (14, 48)):
                    sample = value[:, :, start:stop, :]
                    self.known.setdefault(self.identity(sample), dict(storage='slice', target=[name, key],
                        slices=[[None, None, None], [None, None, None], [start, stop, None], [None, None, None]]))
        self.loaded[name] = packet
        self.datasets[name] = dict(original_path=str(path.resolve()), original_npz_bytes=len(raw),
                                   original_npz_sha256=digest(raw), parent_for_deduplication=parent,
                                   all_original_keys_retained=True, arrays=fields)
        self.bundle.resources[norm(path.resolve())] = dict(array_dataset=name, restore_command='python AUDIT.py --restore NEW_DIRECTORY')

    def finish(self):
        result = dict(schema='rinkspace.lossless-array-overlays.v1', datasets=self.datasets, blob_encodings=self.blob_encodings,
                      semantics='aliases, exact slices, original changed rows, byte-plane shuffle, or bytewise XOR; no floating-point arithmetic, rounding, downcast, time resampling or omitted keys',
                      raw_hash_definition='SHA256 of C-order array bytes plus independently checked dtype.str and shape',
                      container_identity='Restored NPZ container bytes differ; original container SHA records provenance, raw array SHA verifies exact numerical identity')
        self.bundle.json('ARRAY_INDEX.json', result)
        return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def public_index(project):
    paths = ['README.md', 'LICENSE.md', 'scripts/build_body_review.py', 'docs/LICENSING_REVIEW.md', 'docs/GITHUB_REVIEW.md',
             'docs/VALIDATION.md', 'docs/SOLVER.md', 'docs/REBUILD.md', 'environments/README.md',
             PROGRAMME+'/README.md', PROGRAMME+'/body_fullbody_1102.json', PROGRAMME+'/body_fullbody_1102.npz',
             PROGRAMME+'/boot_integration_1102.json', PROGRAMME+'/boot_integration_1102.npz',
             PROGRAMME+'/opening_repair_1102.json', PROGRAMME+'/opening_repair_1102.npz',
             PROGRAMME+'/fit_v11/camera_solver_refined_1102.json', PROGRAMME+'/fit_v11/camera_solver_refined_1102.npz',
             PROGRAMME+'/viewer3d_fullbody_1102.json', PROGRAMME+'/viewer3d_boot_1102.json',
             PROGRAMME+'/body_verification_sources_1103.json', 'events/ankara_2026/scene/README.md']
    source_dir = project/'calibration/program_tracking/body_tracking'
    source_names = ('fullbody_', 'boot_', 'opening_', 'footwear_binding', 'footwear_observations', 'footwear_pixel_factors')
    paths += [p.relative_to(project).as_posix() for p in source_dir.glob('*.py') if p.name.startswith(source_names)]
    paths += [p.relative_to(project).as_posix() for p in (project/'scripts').glob('run_*1102*.py')]
    rows = []
    for name in sorted(set(paths)):
        frozen = subprocess.check_output(['git', '-C', str(project), 'show', COMMIT+':'+name])
        if frozen != (project/name).read_bytes():
            raise ValueError('Public dependency has local edits: '+name)
        rows.append(dict(path=name, bytes=len(frozen), sha256=digest(frozen), url=GITHUB+'/blob/'+COMMIT+'/'+name))
    return dict(repository=GITHUB, commit=COMMIT, branch='rebuild', files=rows,
                note='Public source, licenses and small parameters are already published and not duplicated. Newly supplied AUDIT.py and builder are packaging utilities, not historical solver source.')


def build(project, data, target):
    project, data, target = project.resolve(), data.resolve(), target.resolve()
    if target.is_relative_to(project) or not target.is_relative_to(data):
        raise ValueError('Private package must stay in the Data directory')
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix('.pending')
    b = Bundle(pending)
    root = data/'runs/body_s01_20261029'
    f, boot, opening = [root/name for name in ('fullbody_1102', 'boot_integration_1102', 'opening_repair_1102')]
    a = Arrays(b)
    a.add('body_input', root/'input.npz')
    a.add('observations_fullbody', f/'observations.npz')
    a.add('observations_boot', boot/'observations.npz', 'observations_fullbody')
    a.add('observations_opening', opening/'observations.npz', 'observations_boot')
    a.add('reference', f/'reference/result.npz')
    a.add('legacy', f/'legacy/result.npz', 'reference')
    a.add('full', f/'full/result.npz', 'reference')
    a.add('boot', boot/'fitted/result.npz', 'full')
    a.add('opening', opening/'fitted/result.npz', 'boot')
    a.add('boot_baseline_diagnostics', boot/'baseline_diagnostics.npz')
    a.add('historical_observations_v1', f/'history/observations_v1/observations.npz', 'observations_fullbody')
    arrays = a.finish()

    def record(member, path, kind='original run record'):
        return b.file(member, path, kind=kind, deduplicate=True)

    # Reports retain original bytes. Public copies are linked, not treated as new facts.
    record('observations/input.json', root/'input.json')
    record('observations/source_shoe_review.json', root/'next_round_20261029/J1_s2/source_review.json')
    record('observations/shoe_admission_all_candidates.json', root/'footwear_v2/pose_transfer/admission.json')
    record('observations/footwear_semantics.json', boot/'observation_semantics.json')
    phase_files = {
        'fullbody': (f, ['ledger.json', 'metrics.json', 'results.txt', 'state_checks.json', 'derivative_checks.json',
                        'validation_summary.json', 'visual_review.json', 'camera_binding.json', 'metric_binding.json']),
        'boot': (boot, ['ledger.json', 'boot_report.json', 'results.txt', 'inputs.json', 'binding.json', 'registration.json',
                       'registration.npz', 'joint_result.json', 'joint_result.npz', 'invariants_final.json', 'precision_check.json',
                       'derivative_checks.json', 'verification.json', 'visual_review.json', 'source_shoes.json']),
        'opening': (opening, ['ledger.json', 'opening_report.json', 'results.txt', 'inputs.json', 'initialization.json',
                             'before_audit.json', 'joint_result.json', 'derivative_checks.json', 'verification.json',
                             'opening_visual_review.json', 'source_opening.json', 'metrics_plot.json',
                             'preparation_source_match.json', 'preparation_observation_review.json', 'preparation_ice_review.json',
                             'preparation_camera.json', 'preparation_camera.npz', 'preparation_pose.json', 'preparation_pose.npz',
                             'preparation_geometry_check.json'])}
    for phase, (folder, names) in phase_files.items():
        for name in names:
            record('records/'+phase+'/'+name, folder/name)
        b.events('logs/'+phase+'_accepted.jsonl', folder/'fit_progress.jsonl')
    for name in ('camera_pose.npz', 'camera_snapshot.npz', 'review_scene.npz', 'review_scene.json'):
        record('scene/'+name, f/name, 'frozen camera or evaluated native scene')
    for mode in ('legacy', 'full'):
        for i in range(1, 6):
            folder = f/mode/f'w{i}'
            for p in sorted(folder.iterdir()):
                if p.name in ('config.json', 'input_signature.json', 'reuse_validation.json', 'result.json', 'result.npz'):
                    record(f'solver/fullbody/{mode}/w{i}/'+p.name, p)
    for phase, folder in (('opening', opening/'fit'),):
        for name in ('config.json', 'result.json', 'result.npz'):
            record('solver/'+phase+'/'+name, folder/name)
    for i in (1, 2):
        record(f'solver/boot/w{i}.npz', boot/f'w{i}/result.npz')
    for mode in ('reference', 'legacy', 'full'):
        record(f'records/fullbody/{mode}_export.json', f/mode/'export.json')
    history = f/'history/observations_v1'
    for name in ('decisions.json', 'ledger.json', 'metrics.json', 'results.txt', 'reference_export.json', 'legacy_export.json', 'full_export.json'):
        record('history/observations_v1/'+name, history/name)
    for mode in ('legacy', 'full'):
        for i in (1, 3, 5):
            for name in ('config.json', 'result.json', 'result.npz'):
                record(f'history/observations_v1/{mode}/w{i}/'+name, history/mode/f'w{i}'/name)
    for name, names in {
        'condition_sigma_0p5': ['rejection.json', 'inputs.json', 'fit/config.json', 'fit/last_accepted.npz'],
        'preparation_not_yet_used': ['inputs.json', 'initialization.json', 'joint_result.json', 'fit/config.json', 'fit/result.json', 'fit/result.npz']}.items():
        folder = opening/'history'/name
        for filename in names:
            record('history/'+name+'/'+filename, folder/filename)
        b.events('logs/history_'+name+'_accepted.jsonl', folder/'fit_progress.jsonl')

    figures = []
    def figure(member, source, role):
        actual = record(member, source, 'selected existing image evidence')
        figures.append(dict(path=actual, role=role, original_path=str(source)))
    for i in range(1, 5):
        figure(f'figures/fullbody/source_{i}.png', f/f'compare_source_{i}.png', '原始/采用点与旧状态、legacy、full同源图比较；PTS见metrics.static_figures')
    for i in (1, 2):
        figure(f'figures/fullbody/geometry_{i}.png', f/f'compare_3d_{i}.png', '保存几何与冰面关系；不是独立三维真值')
    figure('figures/boot/source_shoes.png', boot/'source_shoes.png', '26.260、36.020、40.760秒原鞋刀像素与open/fitted')
    for folder, stamp, pts in [('clean_player_check', '1793625425795149700', (336128, 461056, 521728)),
                               ('clean_section_check', '1793625427705344400', (461056, 521728))]:
        record('records/boot/'+folder+'.json', boot/folder/'checks.json')
        for p in pts:
            figure(f'figures/boot/{folder}_{p}.png', boot/folder/f'compare_{p}_{stamp}.png',
                   f'{p/12800:.3f}秒同播放器closed/open/fitted；'+('完整表面' if folder == 'clean_player_check' else '侧视剖面'))
    for name, role in [('preparation_matched_source.png', '19.960秒准备姿势原像素'),
                       ('preparation_sequence.png', '前镜头准备姿势到S01裁切的源帧序列'),
                       ('preparation_geometry_overlay.png', '冻结长边副机位和当前native几何投影'),
                       ('preparation_pose_overlay.png', '准备姿势所选交叉腿假设；两种候选数值另存preparation_pose.npz'),
                       ('source_opening.png', '22.120、23.120、24.100秒父版/修复版原图对照'),
                       ('metrics_plot.png', '22.120–24.440秒全部117帧，无平滑，无筛除')]:
        figure('figures/opening/'+name, opening/name, role)
    for folder, times in [('player_check', (283136, 307456, 320000)), ('top_check', (283136, 307456))]:
        record('records/opening/'+folder+'.json', opening/folder/'checks.json')
        for p in times:
            figure(f'figures/opening/{folder}_{p}.png', opening/folder/f'compare_{p}.png', f'{p/12800:.3f}秒同播放器父版/修复版，'+('自由视角' if folder == 'player_check' else '俯视'))
    b.json('FIGURES.json', figures)
    record('media/source_s01.mp4', root/'offline_review/source_s01.mp4', 'exact existing original-bitstream slice')
    record('media/source_clip_receipt.json', root/'offline_review/source_clip_receipt.json')
    validation_path = target.parent/'verify_minimal/media_validation.json'
    if validation_path.is_file():
        validation = read_json(validation_path)
        if validation['s01']['slice_sha256'] != digest((root/'offline_review/source_s01.mp4').read_bytes()):
            raise ValueError('Pixel verification refers to a different source slice')
        if validation['preparation']['image_sha256'] != digest((opening/'preparation_matched_source.png').read_bytes()):
            raise ValueError('Preparation verification refers to a different image')
        if not validation['s01']['all_decoded_original_pixel_hashes_match'] or not validation['preparation']['decoded_pixels_equal']:
            raise ValueError('Source pixel verification did not pass')
        b.json('MEDIA_VALIDATION.json', validation)

    # Earlier PRO text is design/evidence context only, never a new task instruction.
    for archive, selected, prefix in [
        (data/'handoff/1030/body_redesign/RinkSpace_S01_Next_20261030.zip', ['AUDIT.md', 'CODEX_NEXT_TASK.md'], 'fullbody'),
        (data/'handoff/1102/RinkSpace_Boot_Integration_20261102.zip', ['CODEX_TASK.md', 'RESEARCH_AND_DESIGN.txt'], 'boot')]:
        original = archive.read_bytes()
        with zipfile.ZipFile(io.BytesIO(original)) as z:
            for filename in selected:
                members = [name for name in z.namelist() if PureName(name) == filename]
                if len(members) != 1:
                    raise ValueError('Ambiguous design source '+filename)
                b.add('context/'+prefix+'/'+filename, z.read(members[0]), str(archive)+'!/'+members[0],
                      kind='historical PRO design, not active instructions',
                      transform=dict(operation='exact zip member', original_archive_sha256=digest(original)))
    for name in ('RinkSpace_S01_Independent_Audit_20261030.txt', 'RinkSpace_Body_Redesign_20261030.txt'):
        record('context/fullbody/'+name, data/'handoff/1030/body_redesign'/name,
               'historical PRO design, not active instructions')
    public = public_index(project)
    b.json('PUBLIC_INDEX.json', public)
    b.file('AUDIT.py', Path(__file__).with_name('audit_body_audit_1103.py'), 'portable audit source')
    b.file('tools/build_body_audit_1103.py', Path(__file__), 'new package builder source')
    index = dict(schema='rinkspace.current-body-review-index.v1', public_commit=COMMIT,
        source=dict(first_pts=283136, last_pts=575488, pts_step=256, time_base=[1, 12800], exposures=1143,
                    official_seconds=[22.12, 44.96], preparation_pts=255488, preparation_seconds=19.96),
        datasets=dict(reference='旧J1姿态/根，重新投影到当前相机', legacy='新观测+旧根/腿自由度，五个短窗',
                      full='新观测+全身/根朝向，五个短窗', boot='full父版+全S01足鞋共享配准+两个人体短窗',
                      opening='boot父版+22.12–25.00秒开头修复，当前最终导出'),
        windows=dict(fullbody=[[24.5,27],[32.2,33.8],[35.2,36.8],[37.4,39],[40,41.5]],
                     boot_body=[[35.2,36.8],[40,41.5]], opening=[22.12,25], stationary_prior=[22.12,24.44]),
        reports=dict(fullbody='records/fullbody/metrics.json', boot='records/boot/boot_report.json', opening='records/opening/opening_report.json'),
        figures='FIGURES.json', array_index='ARRAY_INDEX.json', public_index='PUBLIC_INDEX.json',
        exclusions=['complete SMPL-X model and compact templates', 'complete body mesh caches', 'GPU environments/weights/native DLL',
                    'whole official programme video', 'duplicate old full exports, except all current comparator arrays', 'finite-difference evaluation log rows'],
        audit_scope='saved states, exact source evidence, original/adopted observation semantics, before/after metrics and failure history; not full optimizer rerun or complete-body-surface regeneration')
    b.json('REVIEW_INDEX.json', index)
    b.json('RESOURCE_MAP.json', dict(normalization='slashes normalized, casefold', original_to_package=b.resources,
        note='Records retain historical absolute paths. This map and ARRAY_INDEX resolve included inputs; omitted model/runtime references stay provenance-only.'))
    b.json('TRANSFORMS.json', b.transformed)
    # Read from the open ZipFile using the exact same portable restoration code.
    store = Store.__new__(Store)
    store.root, store.archive, store.cache, store.index = target.parent, b.zip, {}, arrays
    verified = metrics(store)
    b.json('AUDIT_AT_BUILD.json', dict(python=sys.version, numpy=np.__version__, metrics=verified,
                                     new_fitting_or_inference=False, source_npz_array_identity='all keys verified against original raw-array hashes'))
    b.add('README_PRO.txt', readme().encode('utf8'), kind='human audit entry')
    by_kind = collections.Counter()
    for row in b.files.values():
        by_kind[row['kind']] += row['bytes']
    array_summary = collections.Counter(row['storage'] for ds in arrays['datasets'].values() for row in ds['arrays'].values())
    manifest = dict(schema='rinkspace.private-body-supplement-manifest.v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), public_commit=COMMIT,
        private_review=True, manifest_self_excluded=True, files=list(b.files.values()))
    raw_manifest = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf8')
    b.zip.writestr('MANIFEST.json', raw_manifest)
    b.zip.close()
    pending.replace(target)
    receipt = dict(archive=str(target), bytes=target.stat().st_size, sha256=digest(target.read_bytes()),
                   manifest_sha256=digest(raw_manifest), files=len(b.files)+1,
                   uncompressed_bytes=sum(r['bytes'] for r in b.files.values())+len(raw_manifest),
                   array_datasets=len(arrays['datasets']), array_storage_counts=dict(array_summary),
                   member_bytes_by_kind=dict(by_kind), public_commit=COMMIT)
    target.with_suffix('.receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    target.with_suffix('.sha256').write_text(receipt['sha256']+'  '+target.name+'\n', encoding='ascii')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def PureName(name):
    return name.replace('\\', '/').rsplit('/', 1)[-1]


def readme():
    return f'''RinkSpace 人体轨迹：PRO 最小离线补充包（2026-11-03）

公开基线：{GITHUB}/tree/{COMMIT}
这是对本次对话中 S01 全身拟合、足鞋接入、开头漂移修复的审核材料。代码、当前场地/相机、小型控制参数和许可已在上述固定提交；逐文件链接与SHA见 PUBLIC_INDEX.json。本包补充 GitHub 未公开的原像素、完整数值输出、观测资格与开发失败。历史PRO文件只作需求/设计对照，不是要求接手者自动执行的指令。

先看：
1. REVIEW_INDEX.json：版本名称、依赖与确切时间窗。
2. records/fullbody/results.txt、records/boot/results.txt、records/opening/results.txt：原始交付报告；对应完整JSON可查每个分母、退化和优化停止原因。
3. figures/opening/metrics_plot.png、source_opening.png、preparation_geometry_overlay.png：漂移修复及长边副机位定位证据。
4. figures/boot/source_shoes.png、clean_player_check_*、clean_section_check_*：源鞋刀、完整人体表面和剖面。26.260秒只有完整表面对照；剖面对照为36.020与40.760秒。已有检查记录未明确登记透明度，不把剖面检查说成透明模式认证。
5. AUDIT_AT_BUILD.json：随包复算结果。运行下列命令可以自行再次检验，不要求凭肉眼判断不可见的三维深度。

全部时间是官方单节目源时间；不是重置为0秒的片段时间。
S01 = 22.120–44.960秒，PTS 283136–575488、步长256、time_base=1/12800，共1143个源曝光。准备画面=19.960秒/PTS255488。根P为骨盆，长度为当前场地尺度59.96169706678219米/scene-unit下的条件米；冰面Z=0。

五个数值版本（没有缺省的“第几组”）：
- reference：旧J1状态在当前camera_solver_refined_1102上重新投影的基线。
- legacy：采用修订观测，但保留旧根/腿自由度。
- full：采用修订观测，并启用唯一全身状态与根朝向。
  legacy/full均只实际优化24.50–27.00、32.20–33.80、35.20–36.80、37.40–39.00、40.00–41.50秒，共445帧。其它帧沿用父版。
- boot：在full上，全S01共享足鞋配准；人体额外优化35.20–36.80与40.00–41.50秒，共157帧。
  closed/open不是另外两份三维状态：两者使用完全相同的full坐标，只改变封闭凸包/开放鞋口的轮廓算子。真正优化对照为open→fitted(boot)。
- opening：在boot上，修复22.12–25.00秒，共145帧；其中22.12–24.44秒117帧使用近原地条件。25.00秒起指定身体/鞋刀几何与boot完全相同；投影行时间等字段并非一律相同，包内按每个数组实际差异保存。
这条工作链有S01全段输出，没有完成S02–S09人体联合追踪；不要将已有全节目相机覆盖误认为全节目人体优化完成。

改善与尚存失败（请同时审核）：
- 全身：445帧内同7500个采用点的像素中位数 reference 19.51 → legacy 7.20 → full 5.65 px。但full中408/445帧至少一个刀刃样点低于冰面1cm，最深约17.9cm；不能据二维改善判定三维轨迹已准确。
- 足鞋：近似内腔越界P95约43.53→3.26mm，但原图外鞋heel/toe误差中位3.74→16.47px，轮廓6.47→8.81px，刀刃3.67→6.24px；共享鞋长/宽增加约18.56%/24.74%，鞋舌模式325/2286脚曝光触顶，穿冰仍在。
- 开头：117帧骨盆XY净位移6.338→0.181m、最大跨度6.338→0.224m；躯干倾角P95 82.34→18.53度；同805个头肩点像素中位46.57→6.16px。但22.98秒仍有约2.81cm穿冰，24.32秒双刀最低样点仍高于冰面约9.11cm，根朝向投影解包变化约470度。125帧不可见下肢仍标为推断。
- 准备镜头：冻结长边副机位/场地，仅拟合该曝光pan/tilt/logf，源MEIWA边缘拟合约0.5px；未进入相机拟合的10个粗冰线点中位约7.99px。B2×F2–R2及交叉腿姿态是有源证据的条件约束，不等于实测支撑点/物理左右已确定。
- 主要人体优化均未达到一阶梯度收敛；ftol停止、程序检查通过、连续播放正常各自保留，不等价于动作/接触准确。

观测与失败链：
observations_fullbody/boot/opening完整保留采用UV、use、sigma、source_id、semantic_index、原COCO/RTMW133、crop affine、inferred标志。足鞋阶段仅将所有1143帧j15/j16的sigma由7改成24px；原UV和use不变。opening观测与boot逐项相同。原236条鞋候选资格、采用206条轮廓/7条刀曲线/11条鞋头跟记录、外鞋语义和局部点修正均保留。
history保留首次错误观测关联、约0.706m漂移被拒绝的条件版本、尚未采用准备镜头的版本，包含报告/控制与停止记录。旧历史全导出不重复装入，原身份仍在export JSON；日志只删逐次有限差分evaluation，保留其它事件。TRANSFORMS.json列出实际删除行数、原字节数和SHA。

离线核验（解压后，在本目录运行）：
    python AUDIT.py
    python AUDIT.py --metrics
    python AUDIT.py --pixels
第一条只需Python标准库，检查全部成员SHA/大小。第二条仅加NumPy，恢复全部数组并验证原dtype/shape/数值字节SHA；独立重算身体三种观测分母、鞋刀源点距离、开头位移/倾角/像素/刀高和25秒几何衔接。第三条另需PyAV，检查1169个原码流包与全部1143帧原解码像素哈希。也可加 --zip 本包.zip 从ZIP直接审核。
已实测环境：Python 3.11、NumPy 2.4.6用于数值审核；S01的严格解码哈希使用PyAV 18.1.0。准备姿势PNG原先使用PyAV 13.0.0，本次也已与该版本解码的原片19.960秒逐像素核对一致。两项历史素材来自不同解码环境，均保留原件；MEDIA_VALIDATION.json登记各自FFmpeg库版本、源身份和跨版本差异。PyAV 13检查S01原码流/PTS通过，但与S01既有BGR回执不同；PyAV 18检查准备图也有像素转换差异。严格像素复验须使用各自记录的版本，普通读图和数组/码流身份检查不受此限制。
若要在自己的分析工具中使用普通NPZ：
    python AUDIT.py --restore 新目录
将得到reference.npz、legacy.npz、full.npz、boot.npz、opening.npz及全部输入观测/诊断。五份结果的所有原字段均保留，包括root P/Q、21个局部旋转、22关节/COCO、鞋刀/内腔/形变、滚动快门行时与投影。为了减小包体，使用重复数组引用、变化行原值、字节重排或按位XOR，不做浮点差值相加。这些都是可逆的字节压缩。恢复的NPZ容器SHA会变化，各数组的数值字节SHA必须与原件一致；原件NPZ SHA同时登记，不能互换。

media/source_s01.mp4复用既有AV1原码流片段，无重编码/缩放；为解码保留21.600秒起的26个前置帧，审核目标仍是22.120–44.960秒。receipt含原视频身份、包PTS和目标帧像素SHA。用本地PotPlayer查看，准备姿势原图与序列在figures/opening。FIGURES.json列出所选现有图片及角色，未装入数百张重复播放器截图。

PRO可据此审查真实源像素、观测语义/分母、全段连续数值、拟合前后对照、退化和失败。鞋轮廓距离复算使用保存的有限可见边选择；近似内腔项复算为保存诊断数组的统计，并非重新碰撞完整人体表面。native场地采样、实际相机快照/姿态和准备镜头参数也在包内，可结合公开投影代码追查。
完整优化器重跑或任意姿势下的完整SMPL-X表面重建仍需相符授权模型/紧凑模板、运行环境与native构建。这些不属于本次数值/图像审核的最小依赖，不随包提供；已有完整表面剖面截图用于查看当时实际检查结果。
本包含非公开比赛像素与研究输出，仅作本次离线审核，不推送GitHub；各素材和SMPL-X衍生结果的权利仍按原来源及公开LICENSE/LICENSING_REVIEW处理。包内没有预训练权重、SMPL-X原模型/紧凑模板、完整人体网格缓存、GPU环境或Windows DLL。
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.project, args.data, args.output)
