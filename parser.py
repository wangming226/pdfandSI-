"""
PDF 解析 + 文本清洗 + 实验段落筛选
===================================
第一阶段: PyMuPDF 提取文本 + 图片
第二阶段: 清洗 + 匹配实验标题截取段落
输出: 正文合成文本 + SI 详细步骤 (分别给千问)
"""

import os, re, zipfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pdf_matcher import MatchedPaper

# 合成页面关键词
_SCHEME_PAGE_KW = [
    r"\bScheme\b",
    r"\bSynthesis\s+of\b",
    r"\bPreparation\s+of\b",
    r"\bGeneral\s+Procedure\b",
]

# 谱图/非结构式关键词 — 检查每张图附近的文字
_IMAGE_SPECTRA_KW = [
    r"\bNMR\b", r"\bNOESY\b", r"\bCOSY\b", r"\bHSQC\b", r"\bHMBC\b",
    r"\bHPLC\b", r"\bGC-MS\b", r"\bLC-MS\b",
    r"\bSpectra\b", r"\bChart\b", r"\bCrystal\s*data\b", r"\bXRD\b",
    r"\bPXRD\b", r"\bSEM\b", r"\bTEM\b", r"\bTGA\b", r"\bthermogravim\b",
    r"\bIR\s*(spectrum|spectra)\b", r"\bwavenumber\b", r"\btransmittance\b",
    r"\bdiffraction\b", r"\bORTEP\b", r"\bthermal\s*ellipsoid",
]


def _get_image_nearby_text(page, bbox) -> str:
    """提取图片 bbox 周围扩展区域的文本 (取图下方图注)"""
    try:
        # bbox 是 fitz.Rect
        import fitz
        # 图片下方扩展 60pt 取图注文字
        below = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, min(bbox.y1 + 60, page.rect.height))
        return page.get_text("text", clip=below).strip()
    except Exception:
        return ""


def _image_nearby_is_spectra(nearby_text: str) -> bool:
    """图片附近文字是否含谱图/晶体图关键词"""
    if not nearby_text or len(nearby_text) < 5:
        return False
    return any(re.search(kw, nearby_text, re.IGNORECASE) for kw in _IMAGE_SPECTRA_KW)


def _page_has_scheme(page_text: str) -> bool:
    """页面是否包含反应图或合成步骤"""
    return any(re.search(kw, page_text, re.IGNORECASE) for kw in _SCHEME_PAGE_KW)


def _is_structure_image(img_bytes: bytes) -> bool:
    """
    PIL 四规则过滤: 判断图片是否为分子结构式。
    全部条件同时校验，任一条不通过即剔除。
    """
    try:
        from PIL import Image
        import io
        import numpy as np
    except ImportError:
        # 无 PIL 时回退: < 200KB 就保留
        return len(img_bytes) < 200_000

    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size

        # ── 规则1: 尺寸 ≥ 200px ──
        if w < 200 or h < 200:
            return False

        # ── 规则2: 长宽比 1.2~4.5 (横向长方形) ──
        ratio = w / h if h > 0 else 0
        if ratio < 1.2 or ratio > 4.5:
            return False

        # ── 规则3: 排除纯色占位图 ──
        gray = img.convert("L")
        arr = np.array(gray)
        total = arr.size
        if np.sum(arr > 250) / total > 0.99:   # 几乎全白=占位图
            return False
        if np.sum(arr < 10) / total > 0.99:    # 几乎全黑=占位图
            return False

        return True

    except Exception:
        return len(img_bytes) < 200_000


@dataclass
class ParsedPaper:
    """解析后的论文"""
    doi: str
    # 正文
    has_main: bool = False
    main_text: Optional[str] = None         # 正文全文
    main_experimental: Optional[str] = None  # 正文中的合成段落
    main_pages: int = 0
    # SI
    has_si: bool = False
    si_text: Optional[str] = None            # SI 全文
    si_experimental: Optional[str] = None    # SI 中的合成步骤
    si_pages: int = 0
    # 图片
    images: List[dict] = field(default_factory=list)       # 位图
    vector_images: List[dict] = field(default_factory=list)  # 矢量渲染图
    # 元信息
    quality: str = "none"  # full | si_only | body_only | none


# ═══════════════════════════════════════════════════════════
# 第一阶段: PDF 解析
# ═══════════════════════════════════════════════════════════

def _parse_pdf(pdf_path: str, extract_images: bool = True) -> Tuple[str, int, List[dict], List[dict]]:
    """
    解析单个 PDF: 提取文本 + 图片。
    返回 (full_text, page_count, raster_images, vector_images)
    """
    import fitz
    from imtools import (page_has_scheme, check_caption, check_raster_props,
                         check_pixel_content, render_page_hires)

    text_pages = []
    raster_images = []
    vector_images = []
    doc = None

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                text_pages.append(page_text)

            if not extract_images:
                continue

            page_h = page.rect.height
            on_scheme_page = page_has_scheme(page_text)

            # ── 矢量图: 合成页面 + 有矢量绘图 → 整页 3x 渲染 ──
            if on_scheme_page and page.get_drawings():
                png_bytes = render_page_hires(page)
                if png_bytes and len(png_bytes) > 10000:
                    vector_images.append({
                        "page": page_num + 1,
                        "context": page_text[:400],
                        "ext": "png",
                        "data": png_bytes,
                        "source": "vector",
                    })

            # ── 位图: 三层过滤 ──
            if not on_scheme_page:
                continue

            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue
                    img_bytes = base_image["image"]
                    if len(img_bytes) < 5000:
                        continue

                    # Layer 1: 图下文字检测 (Figure SX 格式 = 谱图, 其他/无 = 结构式)
                    bbox = page.get_image_bbox(img_info)
                    caption = ""
                    img_y = 0
                    if bbox:
                        img_y = bbox.y0
                        cap_rect = fitz.Rect(bbox.x0, bbox.y1, bbox.x1,
                                             min(bbox.y1 + 50, page_h))
                        caption = page.get_text("text", clip=cap_rect).strip()
                    # 只跳过明确的谱图: 图下文字含 Figure S\d+ NMR/IR/MS/Spectra 等
                    if caption and check_caption(caption) == "skip":
                        continue

                    # Layer 2: 尺寸
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)
                    if not check_raster_props(w, h, img_y, page_h):
                        continue

                    # Layer 3: 像素 (无图注时才做像素分析)
                    if not has_caption:
                        if not check_pixel_content(img_bytes):
                            continue

                    raster_images.append({
                        "page": page_num + 1,
                        "context": caption[:300] or page_text[:400],
                        "ext": base_image.get("ext", "png"),
                        "data": img_bytes,
                        "source": "raster",
                    })
                except Exception:
                    pass

    except Exception as e:
        return f"[解析失败: {e}]", 0, [], []
    finally:
        if doc:
            doc.close()

    full_text = "\n".join(text_pages)
    full_text = re.sub(r"\x00", "", full_text)
    return full_text, page_count, raster_images, vector_images


def _parse_docx(docx_path: str) -> str:
    """解析 DOCX 为纯文本"""
    try:
        with zipfile.ZipFile(docx_path) as z:
            if "word/document.xml" in z.namelist():
                xml = z.read("word/document.xml").decode("utf-8")
                text = re.sub(r"<[^>]+>", " ", xml)
                text = re.sub(r"\s+", " ", text)
                return text.strip()
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════
# 第二阶段: 文本清洗
# ═══════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """基础清洗: 去掉无关内容和噪音"""
    if not text:
        return ""

    # 1. 去掉参考文献行 (以 [1] [2] 等开头的行)
    text = re.sub(r"^\s*\[\d+\].*$", "", text, flags=re.MULTILINE)

    # 2. 去掉 DOI 行
    text = re.sub(r"\b10\.\d{4,}/[^\s]{5,}\b", "", text)

    # 3. 去掉 Scheme X / Figure X / Table X 图注 (整行)
    text = re.sub(
        r"^\s*(Scheme|Figure|Fig\.?|Table)\s+\d+.*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 4. 去掉纯表格数据行 (全是数字/空格/±/小数点的短行)
    text = re.sub(r"^\s*[\d\s\.\,\+\-\±\±]+\s*$", "", text, flags=re.MULTILINE)

    # 5. 去掉页眉页脚类短行 (如页码、期刊名缩写)
    text = re.sub(r"^\s*(S\d+|Page\s+\d+|\d+)\s*$", "", text, flags=re.MULTILINE)

    # 6. 去掉多余换行 (3个以上 → 2个)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. 去掉行内多余空格
    text = re.sub(r" {2,}", " ", text)

    # 8. 清理乱码字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════
# 第二阶段: 实验段落筛选
# ═══════════════════════════════════════════════════════════

# 实验标题关键词 (分两组: 优先合成 > 回退表征)
_SYNTHESIS_HEADINGS = [
    r"\bGeneral\s+(Experimental\s+)?Procedures?\b",
    r"\bSynthesis\s+(of|and|for)\b",
    r"\bSyntheses\b",
    r"\bPreparation\s+of\b",
    r"合成步骤",
    r"合成方法",
    r"合成与表征",
]

_EXPERIMENTAL_HEADINGS = [
    r"\bExperimental\s+Section\b",
    r"\bExperimental\s+(Procedures?|Details?|Methods?)\b",
    r"\bMaterials?\s+and\s+Methods?\b",
    r"\bMethods?\b",
    r"\bSynthesis\b",
    r"\bSyntheses\b",
    r"\bPreparation\b",
    r"\bCharacterization\b",
    r"\bPhysical\s+Measurements?\b",
    r"实验部分",
    r"实验方法",
    r"材料与方法",
    r"仪器与试剂",
]

# 下一章节标题 (截断点)
_NEXT_SECTION_HEADINGS = [
    r"\bResults?\s+and\s+Discussion\b",
    r"\bResults?\b",
    r"\bConclusions?\b",
    r"\bReferences?\b",
    r"\bBibliography\b",
    r"\bAcknowledgments?\b",
    r"\bSupplementary\s+(Material|Information|Data)\b",
    r"\bSupporting\s+Information\b",
    r"\bAppendix\b",
    r"\bAuthor\s+Contributions?\b",
    r"\bConflicts?\s+of\s+Interest\b",
    r"\bData\s+Availability\b",
    # 中文
    r"结果与讨论",
    r"结论",
    r"参考文献",
    r"致谢",
    r"附录",
    r"补充材料",
]


def _extract_experimental_section(text: str) -> Optional[str]:
    """
    从全文中截取实验段落:
    1. 找到实验标题第一次出现的位置
    2. 从该位置开始，到下一个大章节标题前结束
    """
    if not text or len(text) < 200:
        return None

    # 找实验标题
    start_pos = None
    for pat in _EXPERIMENTAL_HEADINGS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if start_pos is None or m.start() < start_pos:
                start_pos = m.start()

    if start_pos is None:
        return None

    # 从标题所在行首开始
    line_start = text.rfind("\n", 0, start_pos)
    start = line_start + 1 if line_start >= 0 else start_pos

    # 从 start 往后搜索最近的结束标题
    after = text[start:]
    end_pos = None
    for pat in _NEXT_SECTION_HEADINGS:
        m = re.search(r"\n\s*" + pat, after, re.IGNORECASE)
        if m:
            if end_pos is None or m.start() < end_pos:
                end_pos = m.start()

    if end_pos is not None:
        section = after[:end_pos]
    else:
        # 没有明确结束标志，取 15000 字符
        section = after[:15000]

    return section.strip() if len(section.strip()) > 100 else None


def _extract_si_experimental(si_text: str) -> Optional[str]:
    """
    从 SI 中提取合成步骤。
    SI 结构: 封面 → 目录 → 实验步骤 → 谱图
    策略:
      1. 跳过目录
      2. 优先找 "General Procedures" / "Synthesis of"
      3. 找不到才回退 "Experimental Section" / "Characterization"
      4. 截到谱图部分之前
    """
    if not si_text or len(si_text) < 200:
        return None

    # 跳过封面和目录
    toc_lines = re.findall(r"\.{4,}\s*\d+", si_text[:5000])
    search_start = 0
    if len(toc_lines) > 3:
        last_toc = 0
        for m in re.finditer(r"\.{4,}\s*\d+", si_text[:5000]):
            last_toc = m.end()
        search_start = max(last_toc, 500)

    si_after_toc = si_text[search_start:]

    # 优先找 Synthesis 标题
    start_pos = None
    for pat in _SYNTHESIS_HEADINGS:
        m = re.search(pat, si_after_toc, re.IGNORECASE)
        if m:
            start_pos = m.start()
            break

    # 回退: 找任何实验标题
    if start_pos is None:
        for pat in _EXPERIMENTAL_HEADINGS:
            m = re.search(pat, si_after_toc, re.IGNORECASE)
            if m:
                start_pos = m.start()
                break

    if start_pos is None:
        # SI 整个就是实验，取前半 (最多 20000c)
        section = si_after_toc[:20000]
        return section.strip() if len(section.strip()) > 200 else None

    line_start = si_after_toc.rfind("\n", 0, start_pos)
    start = line_start + 1 if line_start >= 0 else start_pos

    after = si_after_toc[start:]

    # 截断点: 只在 References 或明确的谱图数据章节标题处停止
    # 不截仪器描述段落 (NMR/Mass/TEM 方法描述属于实验部分)
    si_end_patterns = [
        r"^\s*(References?)\s*$",
        r"^\s*(Copies?\s+of\s+Spectra)\s*$",
        r"^\s*(NMR\s+Spectra)\s*$",         # 谱图数据章节 (非方法)
    ]
    end_pos = None
    for pat in si_end_patterns:
        m = re.search(pat, after, re.MULTILINE | re.IGNORECASE)
        if m:
            if end_pos is None or m.start() < end_pos:
                end_pos = m.start()

    section = after[:end_pos] if end_pos else after[:30000]
    return section.strip() if len(section.strip()) > 200 else None


# ═══════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════

def parse_one(mp: MatchedPaper, img_dir: Optional[str] = None) -> ParsedPaper:
    """解析一篇论文"""
    result = ParsedPaper(doi=mp.record.doi)

    # ── 正文 PDF ──
    if mp.main_pdf and mp.main_pdf.lower().endswith(".pdf"):
        text, pages, images, vec_imgs = _parse_pdf(mp.main_pdf)
        result.vector_images = vec_imgs
        result.main_text = _clean_text(text)
        result.main_pages = pages
        result.has_main = len(result.main_text or "") > 300
        result.images = images + vec_imgs

        # 保存正文图片
        if img_dir and (images or vec_imgs):
            os.makedirs(img_dir, exist_ok=True)
            safe_doi = mp.record.doi.replace("/", "_")[:50]
            for i, img in enumerate(images):
                fname = f"{safe_doi}_p{img['page']}_img{i}.{img['ext']}"
                fpath = os.path.join(img_dir, fname)
                try:
                    with open(fpath, "wb") as f:
                        f.write(img["data"])
                    img["path"] = fpath
                    del img["data"]
                except Exception:
                    pass

    # ── SI ──
    si_parts: list = []
    si_images: list = []
    for si_path in mp.si_files:
        if si_path.lower().endswith(".pdf"):
            t, p, imgs, vec_si = _parse_pdf(si_path, extract_images=True)
            for v in vec_si:
                v["source"] = "si_vector"
            result.vector_images.extend(vec_si)
            result.si_pages += p
            if t and len(t) > 100:
                si_parts.append(t)
            for img in imgs:
                img["source"] = "si"
            si_images.extend(imgs)
        elif si_path.lower().endswith(".docx"):
            t = _parse_docx(si_path)
            if t and len(t) > 100:
                si_parts.append(t)

    if si_parts:
        result.si_text = "\n\n".join(si_parts)
        result.has_si = True

    # ── SI 图片保存 (位图+矢量) ──
    si_all = list(si_images) + list(result.vector_images)
    if si_all and img_dir:
        safe_doi = mp.record.doi.replace("/", "_")[:50]
        for i, img in enumerate(si_all):
            src_tag = "V" if "vector" in str(img.get("source","")) else "R"
            fname = f"{safe_doi}_SI_{src_tag}_p{img['page']}_i{i}.{img['ext']}"
            fpath = os.path.join(img_dir, fname)
            try:
                with open(fpath, "wb") as f:
                    f.write(img["data"])
                img["path"] = fpath
                del img["data"]
            except Exception:
                pass
        result.images.extend(si_all)

    # ── 实验段落提取 ──
    # 正文实验部分
    if result.has_main and result.main_text:
        result.main_experimental = _extract_experimental_section(result.main_text)

    # SI 实验部分
    if result.has_si and result.si_text:
        result.si_experimental = _extract_si_experimental(_clean_text(result.si_text))

    # ── 质量标记 ──
    if result.has_si and result.si_experimental:
        result.quality = "full"
    elif result.main_experimental:
        result.quality = "body_only"
    elif result.has_si:
        result.quality = "si_only"
    else:
        result.quality = "none"

    return result


def parse_all(
    matched: List[MatchedPaper],
    img_dir: Optional[str] = None,
    concurrency: int = 3,
) -> List[ParsedPaper]:
    """批量解析"""
    results = []

    print(f"\n📄 PDF解析 + 实验段落筛选 (并发: {concurrency})")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(parse_one, mp, img_dir): mp
            for mp in matched if mp.has_main
        }

        with tqdm(total=len(futures), desc="解析进度") as pbar:
            for future in as_completed(futures):
                mp = futures[future]
                try:
                    pp = future.result()
                    results.append(pp)
                    body_ok = "B" if pp.main_experimental else "-"
                    si_ok = "S" if pp.si_experimental else "-"
                    imgs = f" {len(pp.images)}图" if pp.images else ""
                    status = f"{pp.quality:10s} | 正文{body_ok} SI{si_ok}{imgs}"
                    pbar.set_postfix_str(status[:55])
                except Exception as e:
                    results.append(ParsedPaper(doi=mp.record.doi))
                    pbar.set_postfix_str(f"ERR: {str(e)[:30]}")
                pbar.update(1)

    # 统计
    full = sum(1 for r in results if r.quality == "full")
    body = sum(1 for r in results if r.quality == "body_only")
    si = sum(1 for r in results if r.quality == "si_only")
    none_ = sum(1 for r in results if r.quality == "none")
    imgs = sum(1 for r in results if r.images)

    print(f"\n📊 full={full} body={body} si={si} none={none_} | 有图={imgs}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ris_parser import parse_ris
    from pdf_matcher import match_pdfs

    records = parse_ris("x-mol.ris")
    matched = match_pdfs(records)
    test = [mp for mp in matched if mp.has_main][:3]
    parsed = parse_all(test)

    for pp in parsed:
        print(f"\n{'='*60}")
        print(f"DOI: {pp.doi}")
        print(f"质量: {pp.quality}")
        print(f"正文实验: {'YES' if pp.main_experimental else 'NO'} "
              f"({len(pp.main_experimental or '')}c)")
        print(f"SI实验:   {'YES' if pp.si_experimental else 'NO'} "
              f"({len(pp.si_experimental or '')}c)")
        print(f"图片: {len(pp.images)} 张")
        if pp.si_experimental:
            print(f"SI前300: {pp.si_experimental[:300]}")
