"""Use an installed or explicitly supplied font; never download at render time."""
from __future__ import annotations

from functools import lru_cache
import os
from PIL import ImageFont


def review_font(size, *, cjk=True):
    return _load(int(size), cjk, os.environ.get('RINKSPACE_CJK_FONT', ''))


@lru_cache(maxsize=64)
def _load(size, cjk, configured):
    if configured:
        try:
            return ImageFont.truetype(configured, size)
        except OSError as exc:
            raise OSError('Cannot load RINKSPACE_CJK_FONT: '+configured) from exc
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ] if cjk else ['C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    raise OSError('No suitable review font is installed; set RINKSPACE_CJK_FONT to an existing CJK .ttf/.ttc/.otf file')
