"""
图片过滤工具
-----------
位图三层过滤 + 矢量图区域检测 + 渲染
"""

import io, re
from typing import List, Optional


# ═══════════════════════════════════════════════════════════
# Layer 0: 页面级合成关键词
# ═══════════════════════════════════════════════════════════

_SCHEME_PAGE_KW = [
    r"\bScheme\b", r"\bSynthesis\s+of\b",
    r"\bPreparation\s+of\b", r"\bGeneral\s+Procedure\b",
]

def page_has_scheme(page_text: str) -> bool:
    return any(re.search(kw, page_text, re.IGNORECASE) for kw in _SCHEME_PAGE_KW)


# ═══════════════════════════════════════════════════════════
# Layer 1: 图注文本黑名单
# ═══════════════════════════════════════════════════════════

_CAPTION_BLACKLIST = [
    r"\bnmr\b", r"\bdept\b", r"\bcosy\b", r"\bhsqc\b", r"\bhmbc\b",
    r"\bnoesy\b", r"\broesy\b", r"\btocsy\b",
    r"\bmass\s*spect", r"\besi-ms\b", r"\bhr-esi-ms\b", r"\bm/z\b",
    r"\bhplc\b", r"\bchromatogram\b", r"\bretention\s*time\b",
    r"\bxrd\b", r"\bx-ray\s*diffraction\b", r"\bpowder\s*diffraction\b",
    r"\btem\b", r"\bsem\b", r"\belectron\s*microscopy\b",
    r"\bcv\b", r"\bcyclic\s*voltammetry\b", r"\bvoltammogram\b",
    r"\btg\b", r"\btga\b", r"\bthermogravimetric\b",
    r"\bcrystal\b", r"\bsingle\s*crystal\b", r"\bcrystallography\b", r"\bpacking\b",
    r"\bppm\b", r"\bchemical\s*shift\b", r"\babsorbance\b", r"\bwavelength\b",
    r"\bspectra\b", r"\bspectrum\b", r"\bchart\b", r"\bplot\b",
    r"\bfigure\s*s\d", r"\bfig\.?\s*s\d", r"\bfig\s*s\d",
]

_CAPTION_WHITELIST = [
    r"\bscheme\b", r"\bsynthesis\b", r"\breaction\b",
    r"\bpreparation\b", r"\bscheme\s*s\d",
]

def check_caption(text: str) -> str:
    """return: 'keep' | 'skip' | 'unknown'"""
    if not text or len(text) < 3:
        return "unknown"
    t = text.lower()
    for pat in _CAPTION_WHITELIST:
        if re.search(pat, t):
            return "keep"
    for pat in _CAPTION_BLACKLIST:
        if re.search(pat, t):
            return "skip"
    return "unknown"


# ═══════════════════════════════════════════════════════════
# Layer 2: 尺寸 / 比例 / 位置
# ═══════════════════════════════════════════════════════════

def check_raster_props(w: int, h: int, page_y: float, page_h: float) -> bool:
    """True=保留"""
    if w < 200 or h < 200:
        return False
    # 竖长图: h/w > 1.3
    if h / w > 1.3:
        return False
    # 超细长: w/h > 6
    if w / h > 6:
        return False
    # 页面底部30% → NM R谱图
    if page_h > 0 and page_y / page_h > 0.7:
        return False
    return True


# ═══════════════════════════════════════════════════════════
# Layer 3: 像素内容
# ═══════════════════════════════════════════════════════════

def check_pixel_content(img_bytes: bytes) -> bool:
    """True=保留"""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return True
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        total = arr.size

        # 空白图
        if np.sum(arr > 250) / total > 0.9:
            return False

        # 灰度方差 < 35 → 谱图渐变
        if np.var(arr) < 35:
            return False

        # 坐标轴: 底部 1/5, 周期性刻度
        bottom = arr[int(h * 0.8):, :]
        col_means = np.mean(bottom, axis=0)
        diffs = np.abs(np.diff(col_means))
        if np.sum(diffs > 20) > w * 0.05:
            return False

        # 连续长曲线: 行暗像素 > 40%
        binary = (arr < 128).astype(np.uint8)
        row_dark = np.sum(binary, axis=1) / w
        if np.sum(row_dark > 0.4) > h * 0.03:
            return False

        # 密集点阵 (>200 独立连通域)
        try:
            from scipy import ndimage
            labeled, n = ndimage.label(binary)
            if n > 200:
                return False
        except ImportError:
            pass

        return True
    except Exception:
        return True


# ═══════════════════════════════════════════════════════════
# 矢量结构图: 搜索关键词位置 → 整页渲染 → 多区域裁剪
# ═══════════════════════════════════════════════════════════

_STRUCT_KEYWORDS = ["Synthesis of", "Scheme", "Preparation of"]

def find_structure_regions(page) -> list:
    """
    搜索页面中 Synthesis of / Scheme 关键词位置，
    返回裁剪区域列表 (PDF 坐标)。
    每个区域: (x0, y0, x1, y1) — 关键词周围 300pt 范围。
    """
    regions = []
    page_w = page.rect.width
    page_h = page.rect.height
    found_rects = []

    for kw in _STRUCT_KEYWORDS:
        try:
            found_rects.extend(page.search_for(kw))
        except Exception:
            pass

    if not found_rects:
        return []

    # 合并重叠的关键词位置
    merged = _merge_nearby_rects(found_rects, gap=50)

    # 每个关键词区域扩展为 300pt 窗口
    for r in merged:
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        half_w = 180  # 左右各 180pt
        half_h = 200  # 上下各 200pt (结构式通常在标题下方)
        x0 = max(0, cx - half_w)
        y0 = max(0, cy - 30)   # 标题上方留 30pt
        y1 = min(page_h, cy + half_h)
        x1 = min(page_w, cx + half_w)
        if x1 > x0 and y1 > y0:
            regions.append((x0, y0, x1, y1))

    return regions


def _merge_nearby_rects(rects: list, gap: float = 50) -> list:
    """合并水平或垂直距离 < gap 的矩形"""
    if not rects:
        return []
    sorted_r = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged = [sorted_r[0]]
    for r in sorted_r[1:]:
        last = merged[-1]
        dx = abs(r.x0 - last.x0)
        dy = abs(r.y0 - last.y0)
        if dx < gap and dy < gap:
            merged[-1] = type(last)(
                min(last.x0, r.x0), min(last.y0, r.y0),
                max(last.x1, r.x1), max(last.y1, r.y1))
        else:
            merged.append(r)
    return merged


def render_page_hires(page, scale: float = 3.0) -> Optional[bytes]:
    """整页渲染为高清 PNG"""
    try:
        import fitz
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    except Exception:
        return None


def crop_rendered_page(png_bytes: bytes, pdf_bbox: tuple,
                       page_w: float, page_h: float, scale: float = 3.0) -> Optional[bytes]:
    """从整页渲染图中裁出 pdf_bbox 对应区域"""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))

        x0 = int(pdf_bbox[0] * scale)
        x1 = int(pdf_bbox[2] * scale)
        y0 = int((page_h - pdf_bbox[3]) * scale)
        y1 = int((page_h - pdf_bbox[1]) * scale)

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(img.size[0], x1); y1 = min(img.size[1], y1)
        if x1 <= x0 or y1 <= y0:
            return None

        cropped = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
