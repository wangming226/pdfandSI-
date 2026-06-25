"""
PDF 匹配器
----------
扫描 pdfandSI/ 和 晶体/ 文件夹，按 DOI 匹配 RIS 记录。
输出: 每篇论文的 PDF 路径映射 (正文 / SI / 晶体)
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict

from ris_parser import PaperRecord


@dataclass
class MatchedPaper:
    """RIS 记录 + 匹配到的文件"""
    record: PaperRecord
    main_pdf: Optional[str] = None      # 正文 PDF
    si_files: List[str] = field(default_factory=list)   # SI 文件 (PDF/DOCX)
    crystal_files: List[str] = field(default_factory=list)  # CIF/晶体文件

    @property
    def has_main(self) -> bool:
        return self.main_pdf is not None

    @property
    def has_si(self) -> bool:
        return len(self.si_files) > 0

    @property
    def has_crystal(self) -> bool:
        return len(self.crystal_files) > 0

    @property
    def doi(self) -> str:
        return self.record.doi


# ═══════════════════════════════════════════════════════════
# DOI 提取
# ═══════════════════════════════════════════════════════════

DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/[^\s\]\)]+)", re.IGNORECASE)


def _extract_doi_from_pdf(pdf_path: str) -> Optional[str]:
    """从 PDF 第一页提取 DOI"""
    try:
        import fitz
    except ImportError:
        raise ImportError("请安装 pymupdf: pip install pymupdf")

    try:
        doc = fitz.open(pdf_path)
        # 只读前 3 页
        for page_num in range(min(3, len(doc))):
            text = doc[page_num].get_text("text")
            m = DOI_PATTERN.search(text)
            if m:
                doi = m.group(1)
                # 清理尾部标点
                doi = doi.rstrip(".,;:)]}'\"")
                doc.close()
                return doi
        doc.close()
    except Exception:
        pass
    return None


def _extract_doi_from_docx(docx_path: str) -> Optional[str]:
    """从 DOCX 文件中提取 DOI (纯文本搜索)"""
    try:
        import zipfile
        from xml.etree import ElementTree as ET
    except ImportError:
        return None

    try:
        with zipfile.ZipFile(docx_path) as z:
            if "word/document.xml" in z.namelist():
                xml_content = z.read("word/document.xml").decode("utf-8")
                # 去掉 XML 标签
                text = re.sub(r"<[^>]+>", " ", xml_content)
                m = DOI_PATTERN.search(text)
                if m:
                    return m.group(1).rstrip(".,;:)]}'\"")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════
# 文件分类
# ═══════════════════════════════════════════════════════════

# SI 文件名模式
SI_PATTERNS = [
    r"_si_\d+",                    # ja6c00821_si_001.pdf
    r"-sup-\d+-suppmat",           # anie72582-sup-0001-suppmat.pdf
    r"-mmc\d+",                    # 1-s2.0-xxx-mmc1.pdf, xxx-mmc1.docx
    r"supp(mat|lementary)",        # supplementary material
    r"supporting.information",
    r"supporting",
]

# 晶体文件扩展名
CRYSTAL_EXTENSIONS = {".cif", ".res", ".ins", ".fcf", ".pdb"}


def _is_si_file(filename: str) -> bool:
    """通过文件名判断是否为 SI"""
    lower = filename.lower()
    for pat in SI_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def _is_crystal_file(filepath: str) -> bool:
    """通过扩展名判断是否为晶体文件"""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in CRYSTAL_EXTENSIONS


def _is_docx(filepath: str) -> bool:
    return filepath.lower().endswith(".docx")


# ═══════════════════════════════════════════════════════════
# 匹配逻辑
# ═══════════════════════════════════════════════════════════

def match_pdfs(
    records: List[PaperRecord],
    pdf_dir: str = "pdfandSI",
    crystal_dir: str = "晶体",
) -> List[MatchedPaper]:
    """
    将 PDF 文件按 DOI 匹配到 RIS 记录。

    Parameters
    ----------
    records : List[PaperRecord]
        RIS 记录列表
    pdf_dir : str
        正文+SI 文件夹
    crystal_dir : str
        晶体数据文件夹

    Returns
    -------
    List[MatchedPaper]
    """
    # ── 收集所有文件 ──
    all_files: List[str] = []

    if os.path.isdir(pdf_dir):
        for f in os.listdir(pdf_dir):
            path = os.path.join(pdf_dir, f)
            if os.path.isfile(path):
                all_files.append(path)

    if os.path.isdir(crystal_dir):
        for f in os.listdir(crystal_dir):
            path = os.path.join(crystal_dir, f)
            if os.path.isfile(path):
                all_files.append(path)

    # ── 索引: DOI → MatchedPaper ──
    doi_index: Dict[str, MatchedPaper] = {}
    for rec in records:
        # 用标准化的 DOI (小写) 作 key
        key = rec.doi.lower().strip()
        doi_index[key] = MatchedPaper(record=rec)

    # 也建一个 DOI suffix 索引 (如 jacs.6c00821)
    suffix_index: Dict[str, MatchedPaper] = {}
    for rec in records:
        suffix = rec.doi_suffix.lower()
        if suffix:
            suffix_index[suffix] = doi_index[rec.doi.lower().strip()]

    # ── 逐文件匹配 ──
    unmatched: List[str] = []
    crystal_files: List[str] = []
    main_doi_map: Dict[str, str] = {}  # 主文件路径 → DOI (用于后续SI关联)

    for filepath in all_files:
        filename = os.path.basename(filepath)

        # 晶体文件
        if _is_crystal_file(filepath):
            crystal_files.append(filepath)
            continue

        # 跳过非论文文件
        if filepath.lower().endswith(".ris"):
            continue  # RIS 引用文件不是 PDF

        matched = False

        # 方法1: 从文件内容提取 DOI (PDF 和 DOCX)
        file_doi = None
        if filepath.lower().endswith(".pdf"):
            file_doi = _extract_doi_from_pdf(filepath)
        elif filepath.lower().endswith(".docx"):
            file_doi = _extract_doi_from_docx(filepath)

        if file_doi:
            key = file_doi.lower().strip()
            if key in doi_index:
                mp = doi_index[key]
                _add_file(mp, filepath)
                matched = True
                if mp.main_pdf == filepath:
                    main_doi_map[filepath] = mp.doi

        # 方法2: 文件名数字匹配 (处理 ACS/Wiley 缩写)
        if not matched:
            filename_clean = filename.lower().replace(".", "").replace("-", "").replace("_", "")
            for suffix, mp in suffix_index.items():
                if _match_by_number_fragment(suffix, filename_clean):
                    _add_file(mp, filepath)
                    matched = True
                    break

        if not matched:
            unmatched.append(filepath)

    # ── 方法3: 关联匹配 ──
    # SI 文件通常与正文共享 PII 或 basename
    # 如 1-s2.0-S1001841726002160-mmc1.docx ↔ 1-s2.0-S1001841726002160-main.pdf
    still_unmatched = []
    for filepath in unmatched:
        filename = os.path.basename(filepath)
        matched = False

        # 尝试从文件名中提取 PII/S-number (Elsevier)
        pii_match = re.match(r"(1-s2\.0-S\d+)-", filename)
        if pii_match:
            pii = pii_match.group(1)
            # 在已匹配的主文件中找同样 PII 的
            for mp in doi_index.values():
                if mp.main_pdf and pii in os.path.basename(mp.main_pdf):
                    _add_file(mp, filepath)
                    matched = True
                    break

        # Wiley: anieXXXXX-sup-... 找对应的 anie main
        if not matched:
            wiley_match = re.match(r"(anie\d+)-", filename)
            if wiley_match:
                anie_id = wiley_match.group(1)
                for mp in doi_index.values():
                    if mp.main_pdf and anie_id in os.path.basename(mp.main_pdf).lower():
                        _add_file(mp, filepath)
                        matched = True
                        break

        # ACS: ja6cXXXXX_si → 找对应的主文件 (主文件用标题命名不太好关联)
        if not matched:
            alpha_num = re.findall(r"\d+[a-z]?\d*", filename.lower())
            if alpha_num:
                main_id = max(alpha_num, key=len)
                if len(main_id) >= 5:
                    for mp in doi_index.values():
                        if mp.has_main and not mp.has_si and mp.record.doi_suffix:
                            if main_id in mp.record.doi_suffix.replace(".", "").lower():
                                _add_file(mp, filepath)
                                matched = True
                                break

        # 作者匹配: ≥3 位作者命中
        if not matched and _is_si_file(filename):
            si_authors = _extract_authors_from_file(filepath)
            if si_authors and len(si_authors) >= 2:
                best_mp = None
                best_hits = 0
                for mp in doi_index.values():
                    if mp.has_main and not mp.has_si:
                        ris_last_names = set(
                            a.split(",")[0].strip().lower()
                            for a in mp.record.authors
                        )
                        hits = sum(
                            1 for a in si_authors
                            if len(a) > 2 and a in ris_last_names
                        )
                        if hits > best_hits:
                            best_hits = hits
                            best_mp = mp
                if best_mp and best_hits >= 3:
                    _add_file(best_mp, filepath)
                    matched = True

        # 同前缀匹配: d6ta00338a1.pdf → 主文件 d6ta00338a.pdf 的 SI
        if not matched:
            basename_no_ext = os.path.splitext(filename)[0].lower()
            for mp in doi_index.values():
                if mp.main_pdf:
                    main_base = os.path.splitext(os.path.basename(mp.main_pdf))[0].lower()
                    # 如果一个文件名以另一个为前缀，就是 SI
                    if basename_no_ext.startswith(main_base) and len(basename_no_ext) > len(main_base):
                        _add_file(mp, filepath)
                        matched = True
                        break

        if not matched:
            still_unmatched.append(filepath)

    unmatched = still_unmatched

    # ── 晶体文件匹配 (按 DOI 推断) ──
    for cfile in crystal_files:
        cfname = os.path.basename(cfile).lower()
        for suffix, mp in suffix_index.items():
            clean_suffix = suffix.replace(".", "").lower()
            digits = re.findall(r"\d+", clean_suffix)
            main_digits = max(digits, key=len) if digits else ""
            if len(main_digits) > 4 and main_digits in cfname.replace(".", "").replace("-", "").replace("_", ""):
                mp.crystal_files.append(cfile)
                break

    # ── 统计 ──
    matched = [mp for mp in doi_index.values()]
    main_count = sum(1 for mp in matched if mp.has_main)
    si_count = sum(1 for mp in matched if mp.has_si)
    crystal_count = sum(1 for mp in matched if mp.has_crystal)

    print(f"\n📁 文件匹配结果:")
    print(f"  总文件数: {len(all_files)}")
    print(f"  RIS 记录: {len(records)}")
    print(f"  匹配正文: {main_count}")
    print(f"  匹配 SI:  {si_count}")
    print(f"  匹配晶体: {crystal_count}")
    print(f"  未匹配文件: {len(unmatched)}")
    if unmatched:
        print(f"  未匹配列表:")
        for u in unmatched:
            print(f"    - {os.path.basename(u)}")

    return matched


def _extract_authors_from_file(filepath: str) -> list:
    """从 PDF/DOCX 文件前几页提取可能的作者姓氏"""
    text = ""
    try:
        if filepath.lower().endswith(".pdf"):
            import fitz
            doc = fitz.open(filepath)
            for page in doc[:3]:
                text += page.get_text("text")
            doc.close()
        elif filepath.lower().endswith(".docx"):
            import zipfile
            with zipfile.ZipFile(filepath) as z:
                if "word/document.xml" in z.namelist():
                    xml = z.read("word/document.xml").decode("utf-8")
                    text = re.sub(r"<[^>]+>", " ", xml)
    except Exception:
        return []

    # 取前 3000 字符，提取大写开头的词作为候选姓氏
    text = text[:3000]
    names = re.findall(r"\b([A-Z][a-z]{2,20})\b", text)
    return list(set(n.lower() for n in names))


def _match_by_number_fragment(doi_suffix: str, filename_clean: str) -> bool:
    """
    通过 DOI suffix 中的数字片段匹配文件名。
    处理 ACS 缩写问题: suffix="jacs.6c00821" → filename="ja6c00821si001"
    """
    # 提取 suffix 中的 journal 缩写部分和数字
    # "jacs.6c00821" → journal_part = "jacs", article_id = "6c00821"
    parts = re.split(r"[.\-]", doi_suffix)
    # ACS 文章编号含字母 (如 6c00821), 用 alphanumeric 匹配
    article_ids = [p for p in parts if re.match(r"^[a-z]?\d+[a-z]?\d*$", p, re.IGNORECASE)]

    if not article_ids:
        return False

    # 取最长的文章编号
    article_id = max(article_ids, key=len)

    # 检查这个文章编号是否在文件名中
    if article_id.lower() in filename_clean:
        return True

    # ACS 特殊处理: "jacs" → "ja"
    if len(parts) > 1 and parts[0] in ("jacs", "jacsat", "jacsau"):
        short_journal = parts[0][:2] + article_id  # "ja" + "6c00821"
        if short_journal.lower() in filename_clean:
            return True

    return False


def _add_file(mp: MatchedPaper, filepath: str):
    """将文件添加到 MatchedPaper (分类: 正文 / SI)"""
    filename = os.path.basename(filepath)

    if _is_si_file(filename) or _is_docx(filepath):
        mp.si_files.append(filepath)
    elif mp.main_pdf is None:
        mp.main_pdf = filepath
    else:
        # 已有正文，这个可能是另一个版本 (放 SI)
        mp.si_files.append(filepath)


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ris_parser import parse_ris

    records = parse_ris("x-mol.ris")
    matched = match_pdfs(records)

    print(f"\n===== 详细匹配结果 =====")
    for mp in matched:
        main = os.path.basename(mp.main_pdf) if mp.main_pdf else "❌ 无"
        si = [os.path.basename(s) for s in mp.si_files]
        cry = [os.path.basename(c) for c in mp.crystal_files]
        status = []
        if mp.has_main:
            status.append("正文")
        if mp.has_si:
            status.append(f"SI×{len(si)}")
        if mp.has_crystal:
            status.append(f"晶体×{len(cry)}")
        print(f"  DOI: {mp.doi}")
        print(f"    文件: {', '.join(status) if status else '❌ 无匹配'}")
        print(f"    正文: {main}")
        if si:
            print(f"    SI:   {si}")
        if cry:
            print(f"    晶体: {cry}")
        print()
