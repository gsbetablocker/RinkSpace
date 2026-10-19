"""Render an immutable solver scene in XY using the production review paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime import install
install()
from calibration.program_tracking.render_result import fixed_scene_paths, advertisement_review_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(identity_path, output):
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    scene_path = identity_path.parent / 'scene_UNACCEPTED.json'
    if sha(scene_path) != identity['scene_sha256']:
        raise ValueError('Immutable solver scene differs from its state identity')
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    paths = fixed_scene_paths(scene, draw_grid=True)
    ads = advertisement_review_paths(scene)
    counts = {k: sum(kind == k for kind, _ in paths) for k in {kind for kind, _ in paths}}
    width, height = 3800, 2500
    image = Image.new('RGB', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(image)
    fonts = {n: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', n) for n in (24, 26, 30, 34, 56)}
    boundary = np.asarray(scene['ice_boundary']['points_xyz'], float)[:, :2]
    low, high = boundary.min(0), boundary.max(0)
    centre = (low + high) / 2
    scale = min(3260 / (high[0] - low[0]), 1570 / (high[1] - low[1]))
    origin = np.array([width / 2, 1115.])

    def xy(points):
        points = (np.asarray(points, float)[:, :2] - centre) * scale
        points[:, 1] *= -1
        return points + origin

    def line(points, colour, weight):
        draw.line([tuple(p) for p in xy(points)], fill=colour, width=weight, joint='curve')

    def label(position, text, colour='#233547', font=30):
        draw.text(tuple(position), text, font=fonts[font], fill=colour, anchor='mm',
                  stroke_width=4, stroke_fill='#f8fafc')

    token = identity['q_sha256'][:16]
    draw.text((120, 68), '共同模型 · 完整俯视平面图', font=fonts[56], fill='#172c42')
    draw.text((120, 150), f'实际求解场景 q={token}… · 待原画面验收', font=fonts[34], fill='#536779')
    draw.text((120, 195), '世界 XY 等比例投影；广告立面压到地面位置，保留实际有限边界。',
              font=fonts[26], fill='#536779')
    colours = {'grid': ('#e2ebee', 1), 'contact': ('#26394a', 7),
               'board_top': ('#6c8793', 3), 'marking': ('#735ea8', 5)}
    for kind, points in paths:
        colour, weight = colours.get(kind, ('#735ea8', 5))
        line(points, colour, weight)

    names = {'goal_line_left': '左球门底线', 'goal_line_right': '右球门底线',
             'goal_crease_left': '左球门区', 'goal_crease_right': '右球门区',
             'goal_trapezoid_left_lower_outer': '左梯形 下沿',
             'goal_trapezoid_left_upper_outer': '左梯形 上沿',
             'goal_trapezoid_right_lower_outer': '右梯形 下沿',
             'goal_trapezoid_right_upper_outer': '右梯形 上沿',
             'curling_centerline_row3': 'curling row3',
             **{f'house_connection_F{i}_R{i}': f'F{i}–R{i}' for i in range(1, 5)},
             **{f'lane{i}': f'Lane{i}' for i in range(1, 5)}}
    for feature in scene['ice_features']:
        name, kind = feature['id'], feature['kind']
        colour = '#296898' if name in ('B1', 'B2') else (
            '#c04e58' if name in ('C', 'C0', 'A1', 'A2', 'A3', 'A4') else
            '#49649e' if name.startswith('lane') else '#0b8d90')
        for path_kind, points in fixed_scene_paths(dict(scene, ice_features=[feature]), draw_grid=False):
            if path_kind == 'marking':
                line(points, colour, 5)
        if kind == 'circle':
            centre3 = np.asarray(feature['center_xyz'], float)
            label(xy(centre3[None])[0], name, '#155d72', 34)
        elif kind == 'polyline':
            points = np.concatenate([np.asarray(v, float) for v in feature['segments_xyz']])
            shift = np.array([0., -25.])
            if name.startswith('house_connection'):
                shift = np.array([260., -24.])
            elif name.startswith('goal_trapezoid'):
                shift = np.array([0., 28. if 'lower' in name else -28.])
            label(xy(points.mean(0)[None])[0] + shift, names.get(name, name), '#4d566c', 26)
        elif kind == 'polygon':
            points = np.asarray(feature['polygon_xyz'], float)
            label(xy(points.mean(0)[None])[0] + [0., 36.], names.get(name, name), font=26)

    grouped = {}
    for ad in ads:
        points = np.asarray(ad['xyz'], float)
        projected = xy(points)
        if ad.get('primitive') == 'points':
            for x, y in projected:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='#b76a18')
        else:
            draw.line([tuple(p) for p in projected], fill='#b76a18', width=9, joint='curve')
        grouped.setdefault(int(ad['number']), []).append(points)
    if sorted(grouped) != list(range(1, 27)):
        raise ValueError('Expected all existing Ad1–26 production paths')
    anchors = {number: xy(np.concatenate(points).mean(0)[None])[0] for number, points in grouped.items()}
    sides = {side: [] for side in ('top', 'bottom', 'left', 'right')}
    extent = (high - low) * scale / 2
    for number, anchor in anchors.items():
        direction = anchor - origin
        relative = np.abs(direction) / extent
        side = ('left' if direction[0] < 0 else 'right') if relative[0] > relative[1] else (
            'top' if direction[1] < 0 else 'bottom')
        sides[side].append((number, anchor))
    for side, items in sides.items():
        axis = 0 if side in ('top', 'bottom') else 1
        items.sort(key=lambda item: item[1][axis])
        previous = -np.inf
        for number, anchor in items:
            position = anchor.copy()
            if side in ('top', 'bottom'):
                position[1] = origin[1] + (-1 if side == 'top' else 1) * (extent[1] + 88)
            else:
                position[0] = origin[0] + (-1 if side == 'left' else 1) * (extent[0] + 95)
            position[axis] = max(position[axis], previous + (112 if axis == 0 else 56))
            previous = position[axis]
            draw.line([tuple(anchor), tuple(position)], fill='#b58a59', width=2)
            label(position, f'Ad{number}', '#925014', 34)

    draw.line([(120, 2080), (3680, 2080)], fill='#d1dde3', width=3)
    footer = [
        '冰面标记：F1–F4 / R1–R4 全部靶圈（外圈／中圈／内圈／圆心） · C0 / A1–A4 · Lane1–4 · B1 / B2 / C',
        '左右球门底线与球门区 · 四条梯形外沿 · curling row3 · F1–R1 至 F4–R4 连接线',
        '广告：Ad1–Ad26 全部实际边界／印刷轮廓位置；立面在俯视中重合为沿围挡的有限轨迹。',
        f'实际模型 {len(scene["ice_features"])} 个标记 / {counts.get("marking", 0)} 条冰面路径 · 全部启用 · 尚未通过完整镜头验收',
    ]
    for index, text in enumerate(footer):
        draw.text((120, 2120 + index * 55), text, font=fonts[30], fill='#354f63')
    metadata = PngImagePlugin.PngInfo()
    for key, value in {'state_identity': str(identity_path), 'scene_path': str(scene_path),
                       'scene_sha256': identity['scene_sha256'], 'q_sha256': identity['q_sha256'],
                       'path_counts': counts, 'advertisement_numbers': sorted(grouped),
                       'accuracy_status': 'unaccepted actual solver scene'}.items():
        metadata.add_text(key, value if isinstance(value, str) else json.dumps(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, pnginfo=metadata)
    return dict(path=str(output), q_sha256=identity['q_sha256'], path_counts=counts,
                advertisements=sorted(grouped), feature_ids=[f['id'] for f in scene['ice_features']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.state_identity, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
