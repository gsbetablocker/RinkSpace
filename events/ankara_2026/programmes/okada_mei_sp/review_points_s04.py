"""S04 actual-pixel review with source-run connections and native entity labels.

Read-only with respect to observations, original media, native and fit runs.
Photographic contours supply display order only. Native curves are a separate
review layer and never substitute for measured source pixels.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import collections
import gzip
import hashlib
import json
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from rinkspace.review.point_connections import entity_color, polyline_order, whole_entity_connections
from rinkspace.review.render import source_frames, validate_video
from rinkspace.vision_native import file_ref

DATA = Path('D:/CodexData/RinkSpace_Data')
OUT = DATA/'runs/native_ice_points_s04_20261027'
ADS = DATA/'runs/native_ad_points_s04_20261026'
FIRST, LAST, STEP = 745984, 948736, 256
SHOT_LABEL = 'S04'
FONT = 'C:/Windows/Fonts/msyh.ttc'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def short_label(path_id, entity):
    label = entity['label']
    if label.startswith('ice.'):
        label = label[4:].replace('house_connection_', '').replace('_', '-')
        label = label.replace(':segment0', '').replace(':centreline', ' 中线')
        return label
    if label.startswith('board.') or path_id.startswith('board/'):
        return '围挡上沿' if path_id.endswith('/top') else '围挡下沿'
    ad, rest = path_id.split('::', 1)
    ad = ad.split('/')[-1]
    rest = rest.removeprefix(ad+'.').replace('print.', '').replace('native440.', '')
    rest = rest.replace('boundary_', '轮廓').replace('shot4_', '').replace('alpha_', '')
    rest = rest.replace('/path0', '').replace('official_svg.path', '图案').replace('.subpath', '.')
    rest = rest.replace('_inner', '(内)').replace('_part', '.笔')
    rest = rest.replace('top_edge', '上边').replace('bottom_edge', '下边')
    rest = rest.replace('right_vertical', '右边').replace('left_vertical', '左边')
    return ad+'/'+rest


class CurrentNativeGrid:
    """Replay existing native geometry and S04 camera; never optimize or use GPU."""
    def __init__(self, *, full_model=False, observed_path_ids=None):
        from rinkspace.core.native import NativeCore
        from rinkspace.domain.parameters import SceneParameterModel, ParameterState
        from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
        from rinkspace.compat.d78_entities import sample_entities
        folder = DATA/'runs/native_joint_s01_s04_20261027'
        manifest_path = folder/'joint_manifest.json'
        state_path = folder/'source_review_initial.npz'
        self.manifest = read(manifest_path)
        self.state = dict(np.load(state_path,allow_pickle=False))
        if str(self.state['point_input_sha256']) != self.manifest['point_input_sha256']:
            raise ValueError('Native source-review state differs from the current fitting input')
        scene_path = Path(self.manifest['scene_path'])
        if file_ref(scene_path)['sha256'] != self.manifest['scene_file_sha256']:
            raise ValueError('Native scene differs from the selected current state')
        self.refs = [file_ref(p) for p in [manifest_path,state_path,scene_path,self.manifest['native_library']]]
        self.core = NativeCore(self.manifest['native_library'])
        scene = read(scene_path)
        model = SceneParameterModel(scene,scene_path,optimize_scene=True)
        parameters = ParameterState(model.initial_state.layout,self.state['scene_values'])
        self.parameter_state = parameters
        self.scene = parameters.apply_scene(scene)
        if full_model:
            from rinkspace.review.scene import sample_review_scene
            self.samples = sample_review_scene(scene,self.core,scene_path,
                parameter_state=parameters,observed_path_ids=observed_path_ids)
        else:
            self.samples = sample_entities(self.scene,self.core,samples=257)
        self.timeline = ShotMotionTimeline(self.manifest['shots'],self.manifest['control_dt'],self.manifest['first_control_time'])
        self.shot = next(s for s in self.manifest['shots'] if s['id']=='s04')
        self.cuts = {r['id']:r for r in self.samples['cuts']}
        self.receipt = dict(source_files=self.refs,all_current_paths=len(self.cuts),
            ice_paths=sum(r['layer']=='ice' for r in self.cuts.values()),
            board_paths=sum(r['layer']=='board' for r in self.cuts.values()),
            advertisement_paths=sum(r['layer']=='advertisement' for r in self.cuts.values()),
            missing_entities=self.samples['missing_entities'],
            geometry_numeric_source=str(state_path),camera_and_motion_source=str(state_path),
            native_geometry_or_parameters_changed=False,GPU_used=False,
            projection_is_navigation_and_review_not_a_measurement=True)

    def project(self,pt):
        time = (pt-self.shot['reference_pts'])/12800+self.timeline.offset(self.shot)
        result = self.core.project_motion(self.state['rig'],self.state['controls'],
            self.timeline.first,self.timeline.dt,1080,time,self.samples['xyz'])
        return {key:dict(cut,uv=result['uv'][cut['begin']:cut['end']],
            valid=result['status'][cut['begin']:cut['end']]==0)
            for key,cut in self.cuts.items()}

    @staticmethod
    def order(points,path):
        p=path['uv'];good=path['valid'][:-1]&path['valid'][1:]
        indices=np.flatnonzero(good)
        if not len(indices):return None
        left=p[:-1][good];delta=np.diff(p,axis=0)[good];length=np.sum(delta*delta,axis=1)
        u=np.clip(np.einsum('ijk,jk->ij',points[:,None]-left[None],delta)/np.maximum(length,1e-12),0.,1.)
        dist=np.sum((points[:,None]-left[None]-u[:,:,None]*delta[None])**2,axis=2)
        k=dist.argmin(1)
        return indices[k]+u[np.arange(len(points)),k]

    def unchanged(self):
        for ref in self.refs:
            if file_ref(ref['path']) != ref:raise ValueError('Native replay source changed during review')


class SourceReview:
    def __init__(self, input_dir, *, show_native_grid=True):
        self.show_native_grid = show_native_grid
        self.mapping = read(input_dir/'mapping.json')
        self.point_file = input_dir/'point_observations.npz'
        self.before = file_ref(self.point_file)
        if self.before['sha256'] != self.mapping['point_input_sha256']:
            raise ValueError('Point file and mapping disagree')
        self.z = dict(np.load(self.point_file, allow_pickle=False))
        self.entities = {r['native_path_id']: r for r in read(OUT/'evidence/native_entities.json')['entities']}
        self.paths = self.mapping['path_ids']
        observed={self.paths[int(i)] for i in np.unique(self.z['path_indices'])}
        self.grid = CurrentNativeGrid(full_model=True,observed_path_ids=observed)
        for key,cut in self.grid.cuts.items():
            self.entities.setdefault(key,dict(native_path_id=key,kind=cut['kind'],label=cut['label']))
        from rinkspace.review.draw import top_down
        overview=top_down(self.grid.samples['xyz'],list(self.grid.cuts.values()),height=640,
            missing_count=len(self.grid.samples['missing_entities']))
        # Keep the existing common-model renderer; crop only its display footer,
        # whose default red-point legend does not describe this review.
        self.overview=cv2.resize(overview[:570],(620,330),interpolation=cv2.INTER_AREA)
        self.visibility = {r['pts']: r for r in read(ADS/'evidence/visibility.json')['frames']}
        self.meta = {}
        self.ids = {str(s): i for i, s in enumerate(self.z['observation_ids'])}
        central=input_dir/'point_records.jsonl.gz'
        if central.exists():
            for r in rows(central):
                self.add(r['observation_id'],r['uv'],r['pts'],r['native_path_id'],r['review_order'])
        else:
            self.load_original_evidence()
        self.finish_setup()

    def load_original_evidence(self):
        for r in rows(OUT/'evidence/selected_ice_records.jsonl.gz'):
            key = r['observation_key']
            p = r.get('pixel_support', {})
            self.add(key, r['uv'], r['native_pts'], r['native_path_id'], dict(
                run=r['visible_span_id']+'@'+r['source_ref']['path'],
                order=float(p.get('query_index', 0)), family='ice'))
        for r in rows(OUT/'evidence/selected_board_records.jsonl.gz'):
            self.add('board:'+r['source_row_id'], r['uv'], r['native_pts'], r['native_path_id'], dict(
                run=r['visible_span_id'], order=float(r['uv'][0]), family='board'))
        for r in rows(ADS/'evidence/selected_point_records.jsonl.gz'):
            ident = r['identity']
            if 'atlas_id' in ident:
                metadata = dict(atlas=int(ident['atlas_id']), source_path=int(ident['source_path_index']), family='ad')
            elif ident.get('method') == 'same_Ad4_named_outer_material_edge_from_actual_source_guides_and_current_RGB':
                metadata = dict(run=r['path_id']+'@'+str(ident['source_guide_pts']),
                    order=float(r['stable_id'].rsplit('_',1)[1]), family='ad_source_edge')
            else:
                raise ValueError('Source ordering not yet supported: '+str(ident))
            self.add(r['stable_id'], r['uv'], r['pts'], r['path_id'], metadata)

    def finish_setup(self):
        if len(self.meta) != len(self.z['pts']):
            missing = [str(self.z['observation_ids'][i]) for i in range(len(self.z['pts'])) if i not in self.meta]
            raise ValueError('Review cannot silently omit source records: '+str(missing[:5]))
        self.atlases = read(ADS/'atlases.json')['atlases']+read(ADS/'source_review/additional_source_atlases.json')['atlases']
        self.charts = read(ADS/'source_review/source_identity_charts.json')['records']
        self.by_pts = collections.defaultdict(list)
        for i, pt in enumerate(self.z['pts']):
            self.by_pts[int(pt)].append(i)
        self.known = {}
        peer = DATA/'runs/native_joint_s01_s04_20261027/s04_source_repairs.jsonl.gz'
        if peer.exists():
            for r in rows(peer):
                i = self.ids.get(r['observation_id'])
                if i is not None and np.array_equal(self.z['uv'][i], r['uv']) and self.z['pts'][i] == r['pts']:
                    self.known[i] = r['reason']
        self.protected = []
        for ref in read(OUT/'evidence/native_authority.json')['files']:
            # Capture the CURRENT files. A peer has edited native implementation
            # since the old delivery; rendering neither restores nor builds it.
            self.protected.append(file_ref(ref['path']))
        self.font = ImageFont.truetype(FONT, 19)
        self.small = ImageFont.truetype(FONT, 16)
        self.title = ImageFont.truetype(FONT, 27)

    def add(self, key, uv, pts, path, metadata):
        i = self.ids.get(key)
        if i is None:
            return  # Preserved dense/history records may not be current representatives.
        if i in self.meta or not np.array_equal(self.z['uv'][i], uv) or int(self.z['pts'][i]) != pts or self.paths[int(self.z['path_indices'][i])] != path:
            raise ValueError('Source evidence changed or duplicated a delivered observation: '+key)
        self.meta[i] = metadata

    def group_edges(self, ids, mask, pt, grid):
        uv = self.z['uv'][ids]
        records = [self.meta[i] for i in ids]
        pid = self.paths[int(self.z['path_indices'][ids[0]])]
        if all(r['family'] != 'ad' for r in records):
            path = grid.get(pid)
            # Flattened rings can be closer to the opposite projected side
            # when the camera is imperfect. Keep actual reader material order.
            circle=path is not None and path['kind'] in ('ice_circle','ice_ellipse')
            order=np.asarray([r['order']for r in records])if circle else self.grid.order(uv,path)if path is not None else None
            if order is None:
                centre = np.mean(uv,axis=0)
                direction = np.linalg.svd(uv-centre,full_matrices=False)[2][0]
                order = (uv-centre)@direction
            closed = bool(path is not None and path['closed'] and path['layer']=='ice')
            return whole_entity_connections(uv,[r['run'] for r in records],order,mask,closed=closed)
        # Choose one photographic ordering for the entire native entity,
        # including observations contributed by other source references.
        references = []
        for ai,pi in sorted({(r['atlas'],r['source_path']) for r in records if r['family']=='ad'}):
            chart = self.charts.get(str(ai),{}).get(str(pt))
            if chart is None:raise ValueError(f'Missing source contour order: {pt}/{ai}/{pi}')
            source = self.atlases[ai]['paths'][pi]
            p = cv2.perspectiveTransform(np.asarray(source['pixels'],float)[None],np.asarray(chart['H'],float))[0]
            order,distance = polyline_order(uv,p)
            references.append((float(np.median(distance)),order,np.linalg.norm(p[0]-p[-1])<2.))
        _,order,closed = min(references,key=lambda x:x[0])
        runs = [str((r['atlas'],r['source_path'])) if r['family']=='ad' else r['run'] for r in records]
        return whole_entity_connections(uv,runs,order,mask,closed=closed)

    def frame(self, pt, source):
        vis = self.visibility[pt]
        if hashlib.sha256(source.tobytes()).hexdigest() != vis['source_BGR_sha256']:
            raise ValueError('Decoded source exposure differs from actual readout source')
        mask = cv2.imread(vis['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != source.shape[:2]:
            raise ValueError('Exact source occlusion mask missing')
        groups = collections.defaultdict(list)
        for i in self.by_pts[pt]:
            groups[int(self.z['path_indices'][i])].append(i)
        canvas = np.full((1160, 2560, 3), 22, np.uint8)
        canvas[:1080, :1920] = source
        if self.show_native_grid:
            canvas[125:455,1930:2550]=self.overview
        grid = self.grid.project(pt)
        grid_visible = []
        native_runs = []
        for path in grid.values():
            uv=path['uv'];valid=path['valid']
            inside=valid&(uv>=0).all(1)&(uv<[1920,1080]).all(1)
            if inside.any():grid_visible.append(dict(path_id=path['id'],label=path['label']))
            indices=np.flatnonzero(valid)
            for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                if len(run)>1:
                    native_runs.append(np.rint(np.clip(uv[run],-1e7,1e7)).astype(np.int32))
        if self.show_native_grid:
            cv2.polylines(canvas[:1080,:1920],native_runs,False,(30,30,30),3,cv2.LINE_AA)
            cv2.polylines(canvas[:1080,:1920],native_runs,False,(245,245,245),1,cv2.LINE_AA)
        ledger, labels = [], []
        ordered = sorted(groups, key=lambda k: (0 if self.entities[self.paths[k]]['kind'].startswith('ice') else 1, self.paths[k]))
        for path_index in ordered:
            path = self.paths[path_index]
            ids = groups[path_index]
            label = short_label(path, self.entities[path])
            rgb = entity_color(path)
            color = rgb[::-1]
            edges, breaks = self.group_edges(ids, mask, pt, grid)
            uv = self.z['uv'][ids]
            dashed = {(b['a'],b['b']) for b in breaks}
            for a, b in edges:
                if (a,b) in dashed:
                    length=float(np.linalg.norm(uv[b]-uv[a]))
                    for start in np.arange(0.,length,12.):
                        p=uv[a]+(uv[b]-uv[a])*start/max(length,1e-12)
                        q=uv[a]+(uv[b]-uv[a])*min(length,start+7.)/max(length,1e-12)
                        cv2.line(canvas,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),color,1,cv2.LINE_AA)
                else:
                    cv2.line(canvas, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 1, cv2.LINE_AA)
            for i, q in zip(ids, uv):
                xy = tuple(np.rint(q).astype(int))
                cv2.circle(canvas, xy, 3, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(canvas, xy, 2, color, -1, cv2.LINE_AA)
                if self.z['kind'][i] == 0:
                    cv2.rectangle(canvas, (xy[0]-4, xy[1]-4), (xy[0]+4, xy[1]+4), color, 1)
                if i in self.known:
                    cv2.drawMarker(canvas, xy, (30, 30, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
            anchor = uv[np.argmin(np.linalg.norm(uv-np.median(uv, axis=0), axis=1))]
            labels.append((label, rgb, anchor, len(ids), sum(i in self.known for i in ids)))
            ledger.append(dict(path_id=path, label=label, observations=len(ids),
                connection_rows=[[int(ids[a]), int(ids[b])] for a, b in edges],
                breaks=[dict(a=int(ids[b['a']]), b=int(ids[b['b']]), reason=b['reason']) for b in breaks]))
        picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(picture)
        occupied = []
        for label, rgb, q, count, known in labels:
            compact = label.replace('lane', 'L').replace(':edge', '.e')
            compact = compact.replace(':outer', '/外').replace(':middle', '/中').replace(':inner', '/内').replace(':center', '/芯')
            text = compact+(' ×' if known else '')
            box = draw.textbbox((0, 0), text, font=self.small)
            w, h = box[2]+8, 24
            candidates = []
            for dx, dy in [(10,-28), (10,8), (-w-10,-28), (-w-10,8), (12,-52), (12,34)]:
                x = int(np.clip(q[0]+dx, 0, 1920-w))
                y = int(np.clip(q[1]+dy, 104, 1080-h))
                r = (x, y, x+w, y+h)
                overlap = sum(max(0,min(r[2],b[2])-max(r[0],b[0]))*max(0,min(r[3],b[3])-max(r[1],b[1])) for b in occupied)
                candidates.append((overlap, r))
            _, r = min(candidates, key=lambda v: v[0])
            occupied.append(r)
            draw.line((int(q[0]), int(q[1]), r[0] if q[0]<r[0] else r[2], r[1]+12), fill=rgb, width=1)
            draw.rectangle(r, fill=(22,22,22))
            draw.text((r[0]+4,r[1]), text, font=self.small, fill=rgb)
        title = ' · 全连取点 + native 网格' if self.show_native_grid else ' · 原片取点与实体连线'
        draw.text((1940, 18), SHOT_LABEL+title, font=self.title, fill=(245,245,245))
        draw.text((1940, 60), f'原片 {pt/12800:.3f} 秒   PTS {pt}', font=self.font, fill=(230,230,230))
        draw.text((1940, 93), f'当前 {len(self.by_pts[pt])} 点 / {len(labels)} 个实体', font=self.font, fill=(210,210,210))
        step = 16
        legend_top = 465 if self.show_native_grid else 125
        per_col = 38 if self.show_native_grid else 57
        columns=max(2,int(np.ceil(len(labels)/per_col)))
        column_width=620/columns
        for j, (label, rgb, _, count, known) in enumerate(labels):
            col, row = divmod(j, per_col)
            x, y = int(1940+col*column_width), legend_top+row*step
            draw.line((x, y+12, x+17, y+12), fill=rgb, width=2)
            display = f'{label}  {count}点'+(' ×' if known else '')
            # Wrap a rare long source name by using a smaller font, not by
            # truncating the identity or conflating separate contours.
            font = ImageFont.truetype(FONT,14)
            if draw.textlength(display, font=font) > column_width-32:
                font = ImageFont.truetype(FONT, max(9, int(14*(column_width-32)/draw.textlength(display,font=font))))
            draw.text((x+22, y), display, font=font, fill=rgb)
        grid_note = '白线：当前 native 全模型（冰面、围挡、广告）   ' if self.show_native_grid else ''
        draw.text((20,1092), grid_note+'彩色点及连线：本帧实际读数，同一实体全部连接   方框：固定角点', font=self.font, fill=(240,240,240))
        note = '红 ×：已有来源错误记录，待修。' if self.known else ''
        drawn = '网格与连接线均不是新读点。' if self.show_native_grid else '连接线不是新读点。'
        draw.text((20,1126), '虚线：跨遮挡／缺段及未观测的闭合连接，仍完整显示点间关系。原始点不移动；'+drawn+note, font=self.font, fill=(225,225,225))
        return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR), dict(pts=pt, observations=len(self.by_pts[pt]),
            known_source_errors=sum(i in self.known for i in self.by_pts[pt]), entities=ledger,
            source_BGR_sha256=vis['source_BGR_sha256'],native_grid_displayed=self.show_native_grid,
            native_grid_visible=grid_visible if self.show_native_grid else [],
            native_grid_paths_evaluated=len(grid),native_invalid_samples=sum(int((~p['valid']).sum())for p in grid.values()))

    def unchanged(self):
        self.grid.unchanged()
        if file_ref(self.point_file) != self.before:
            raise ValueError('Point input changed during read-only rendering')
        for ref in self.protected:
            if file_ref(ref['path']) != ref:
                raise ValueError('Concurrent native change during review: '+ref['path'])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-pts', type=int, nargs='+')
    parser.add_argument('--no-native-grid', action='store_true', help='Show source observations and connections without the native projection or top view')
    parser.add_argument('--input-dir', type=Path, default=OUT/('points_repaired' if (OUT/'points_repaired/source_review_application.json').exists() else 'points'))
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)
    review = SourceReview(args.input_dir, show_native_grid=not args.no_native_grid)
    suffix = "_no_projection" if args.no_native_grid else ""
    folder = OUT/'review'
    folder.mkdir(exist_ok=True)
    if args.preview_pts:
        for pt in args.preview_pts:
            source = cv2.imread(review.visibility[pt]['image'])
            frame, record = review.frame(pt, source)
            path = folder/f'connected{suffix}_{pt}.jpg'
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            print(json.dumps(dict(path=str(path), entities=len(record['entities']), points=record['observations'])), flush=True)
        review.unchanged()
        return
    video = folder/(SHOT_LABEL+'_points_connected_entities'+suffix+'.mp4')
    temporary = folder/(SHOT_LABEL+'_points_connected_entities'+suffix+'.partial.mp4')
    tb = Fraction(1,12800)
    expected = np.arange(FIRST,LAST+1,STEP)
    start = time.monotonic()
    connection_count = broken_count = points = 0
    with av.open(str(temporary), 'w') as container, gzip.open(folder/f'connected_frame_index{suffix}.jsonl.gz','wt',encoding='utf8') as ledger:
        stream = container.add_stream('libx264', rate=50)
        stream.width, stream.height, stream.pix_fmt = 2560,1160,'yuv420p'
        stream.time_base = tb
        stream.options = {'crf':'17','preset':'fast'}
        for index, (pt, im) in enumerate(source_frames(Path(review.mapping['source_video']),
                dict(time_base_num=1,time_base_den=12800,width=1920,height=1080), expected)):
            frame, record = review.frame(pt, im)
            points += record['observations']
            connection_count += sum(len(r['connection_rows']) for r in record['entities'])
            broken_count += sum(len(r['breaks']) for r in record['entities'])
            ledger.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
            vf = av.VideoFrame.from_ndarray(frame,format='bgr24')
            vf.pts, vf.time_base = pt-FIRST,tb
            for packet in stream.encode(vf):container.mux(packet)
            if index%100 == 0:
                print(json.dumps(dict(stage='connected_review',frames=index+1,total=len(expected),seconds=round(time.monotonic()-start,1))),flush=True)
        for packet in stream.encode():container.mux(packet)
    check = validate_video(temporary, expected, tb)
    review.unchanged()
    if points != len(review.z['pts']):raise ValueError('Some input points were not shown')
    temporary.replace(video)
    result = dict(**file_ref(video), **check, point_input=review.before,
        point_input_sha256=review.before['sha256'], source_time_base='1/12800',
        source_pts=[FIRST,LAST], source_seconds=[FIRST/12800,LAST/12800],
        points_displayed=points, original_UV_PTS_ID_path_unchanged=True,
        point_connections=connection_count, visible_dashed_connections=broken_count,
        known_source_errors_marked=len(review.known), human_acceptance=False,
        native_geometry_displayed=review.show_native_grid, fitted_source_curve_pixels_emitted=False,
        native_grid=review.grid.receipt, native_or_fit_files_written=False,
        protected_geometry_files_unchanged=review.protected,
        every_entity_fully_connected=True, missing_or_occluded_links_drawn_dashed=True,
        connection_order='native material order for ice/board; actual photographic contour for advertisements',
        same_complete_native_model_top_view=review.show_native_grid,seconds=time.monotonic()-start)
    write(folder/f'connected_video{suffix}.json',result)
    old_video_manifest = folder/f'video{suffix}.json'
    if old_video_manifest.exists() and 'point_connections' not in read(old_video_manifest):
        old_video_manifest.replace(folder/'points_only_video.json')
    write(old_video_manifest,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
