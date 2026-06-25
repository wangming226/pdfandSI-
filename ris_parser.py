"""
RIS 文件解析器
--------------
解析 X-MOL 导出的 RIS 文件，提取论文元数据。
输出: List[PaperRecord] — DOI、标题、作者、摘要、期刊、年份
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PaperRecord:
    """RIS 中的一条论文记录"""
    doi: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    journal_full: str = ""
    journal_abbr: str = ""
    year: Optional[int] = None
    url: str = ""
    ris_raw: dict = field(default_factory=dict)  # 原始RIS字段

    @property
    def first_author(self) -> str:
        return self.authors[0] if self.authors else ""

    @property
    def doi_suffix(self) -> str:
        """DOI 最后一段，如 jacs.6c00821"""
        parts = self.doi.split("/") if self.doi else []
        return parts[-1] if parts else ""

    @property
    def doi_short(self) -> str:
        """DOI 不带前缀的简短形式，如 10.1002_anie.8127392"""
        return self.doi.replace("/", "_") if self.doi else ""


# ═══════════════════════════════════════════════════════════
# HTML 标签清洗
# ═══════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """去掉 RIS 摘要中的 HTML/XML 标签"""
    # 自闭合标签 <ce:inf loc="post"> → 去掉
    text = re.sub(r"<[^>]+>", "", text)
    # HTML 实体
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&amp;", "&").replace("&quot;", '"')
    text = re.sub(r"&#?\w+;", "", text)
    # 多余空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════
# RIS 解析
# ═══════════════════════════════════════════════════════════

def parse_ris(filepath: str) -> List[PaperRecord]:
    """
    解析 RIS 文件，返回 PaperRecord 列表。

    Parameters
    ----------
    filepath : str
        RIS 文件路径 (如 x-mol.ris)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RIS 文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 ER  -  分割记录
    # 先标准化换行
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    records_text = re.split(r"\nER\s*-\s*\n", content)

    records: List[PaperRecord] = []

    for block in records_text:
        block = block.strip()
        if not block or not block.startswith("TY"):
            continue

        # 解析字段: TAG  - VALUE (可能跨行)
        raw = _parse_ris_block(block)

        # 提取关键字段
        doi = _get_str(raw, "DO")
        title = _get_str(raw, "T1")

        if not doi or not title:
            continue  # 跳过无效记录

        # 作者
        authors = _get_list(raw, "AU")

        # 年份
        year = None
        py_val = _get_str(raw, "PY")
        if py_val:
            try:
                year = int(py_val.strip())
            except ValueError:
                pass

        # 摘要 (去HTML标签)
        abstract = _strip_html(_get_str(raw, "AB"))

        records.append(PaperRecord(
            doi=doi.strip(),
            title=title.strip(),
            authors=[a.strip() for a in authors if a.strip()],
            abstract=abstract,
            journal_full=_get_str(raw, "JF").strip(),
            journal_abbr=_get_str(raw, "JO").strip(),
            year=year,
            url=_get_str(raw, "UR").strip(),
            ris_raw=raw,
        ))

    return records


def _parse_ris_block(block: str) -> dict:
    """解析单条 RIS 记录的字段，处理多行值和重复标签"""
    raw: dict = {}
    lines = block.split("\n")

    current_tag = None
    current_value = ""

    for line in lines:
        # RIS 字段格式: TAG  - VALUE
        match = re.match(r"^([A-Z0-9]{2})\s{0,4}[-]\s{0,4}(.*)", line)
        if match:
            # 保存上一个 tag
            if current_tag:
                _add_value(raw, current_tag, current_value.strip())

            current_tag = match.group(1)
            current_value = match.group(2)
        else:
            # 续行 (值跨多行)
            current_value += " " + line.strip()

    # 最后一个 tag
    if current_tag:
        _add_value(raw, current_tag, current_value.strip())

    return raw


def _get_str(raw: dict, key: str, default: str = "") -> str:
    """安全获取字符串值（处理重复标签导致的 list）"""
    val = raw.get(key, default)
    if isinstance(val, list):
        return val[0] if val else default
    return val if val else default


def _get_list(raw: dict, key: str) -> list:
    """安全获取列表值"""
    val = raw.get(key, [])
    if isinstance(val, str):
        return [val] if val else []
    return val


def _add_value(raw: dict, tag: str, value: str):
    """将值添加到 raw dict，重复的 tag 转为 list"""
    if tag in raw:
        existing = raw[tag]
        if isinstance(existing, list):
            existing.append(value)
        else:
            raw[tag] = [existing, value]
    else:
        raw[tag] = value


# ═══════════════════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    ris_path = sys.argv[1] if len(sys.argv) > 1 else "x-mol.ris"
    records = parse_ris(ris_path)

    print(f"📋 解析到 {len(records)} 条 RIS 记录:\n")
    for i, r in enumerate(records, 1):
        print(f"  [{i}] DOI: {r.doi}")
        print(f"      标题: {r.title[:100]}...")
        print(f"      第一作者: {r.first_author}")
        print(f"      期刊: {r.journal_abbr} ({r.year})")
        print(f"      摘要: {len(r.abstract)} chars")
        print()
