"""
千问提取器
----------
三阶段:
  1. 多模态 VL 仅提取图片中的 SMILES (独立任务)
  2. 纯文本模型提取完整反应结构化数据
  3. 代码层按标签精确匹配合并 SMILES
"""

import os, re, json, time, base64
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from parser import ParsedPaper
import config


# ═══════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class ExtractedReaction:
    reactant_smiles: List[str] = field(default_factory=list)
    reactant_names: List[str] = field(default_factory=list)
    metal_salts: List[str] = field(default_factory=list)
    metals: List[str] = field(default_factory=list)
    anions: List[str] = field(default_factory=list)
    temperature_celsius: Optional[float] = None
    time_hours: Optional[float] = None
    solvents: List[str] = field(default_factory=list)
    atmosphere: Optional[str] = None
    products: List[dict] = field(default_factory=list)


@dataclass
class ExtractionResult:
    doi: str
    success: bool = False
    reactions: List[ExtractedReaction] = field(default_factory=list)
    smiles_dict: Dict[str, str] = field(default_factory=dict)  # {label: smiles}
    raw_json: str = ""
    model_used: str = ""
    source: str = ""
    chunk_count: int = 0
    error: str = ""


# ═══════════════════════════════════════════════════════════
# API 客户端 (重试 + 限流)
# ═══════════════════════════════════════════════════════════

class _QwenClient:
    def __init__(self):
        self._client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            timeout=config.QWEN_TIMEOUT,
        )
        self._last_call = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < config.REQUEST_DELAY:
            time.sleep(config.REQUEST_DELAY - elapsed)
        self._last_call = time.time()

    def chat(self, messages: list, model: str = None) -> str:
        model = model or config.QWEN_MODEL
        for attempt in range(config.RETRY_MAX + 1):
            self._rate_limit()
            try:
                resp = self._client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=config.QWEN_TEMPERATURE,
                    max_tokens=config.QWEN_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if attempt < config.RETRY_MAX:
                    wait = config.RETRY_BACKOFF[min(attempt, len(config.RETRY_BACKOFF)-1)]
                    if "429" in msg or "rate" in msg.lower():
                        wait = max(wait, 5)
                    print(f"  [重试 {attempt+1}/{config.RETRY_MAX}] 等待{wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API失败(已重试{config.RETRY_MAX}次): {msg}")
        return ""


_client = None

def _get_client() -> _QwenClient:
    global _client
    if _client is None:
        _client = _QwenClient()
    return _client


# ═══════════════════════════════════════════════════════════
# JSON 解析
# ═══════════════════════════════════════════════════════════

def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


# ═══════════════════════════════════════════════════════════
# 文本切片
# ═══════════════════════════════════════════════════════════

def _chunk_text(text: str, max_size: int = None) -> List[str]:
    if max_size is None:
        max_size = config.TEXT_CHUNK_SIZE
    if len(text) <= max_size:
        return [text]

    chunks = []
    overlap = config.TEXT_CHUNK_OVERLAP
    pos = 0
    while pos < len(text):
        end = pos + max_size
        if end >= len(text):
            chunks.append(text[pos:])
            break
        search_region = text[max(pos, end - max_size // 3):end + max_size // 3]
        break_point = None
        for m in re.finditer(r"\n(?=\s*(?:Synthesi[sz]|Preparation|General\s+Procedure)\b)", search_region):
            bp = max(pos, end - max_size // 3) + m.start()
            if bp > pos:
                break_point = bp
                break
        if break_point is None:
            for m in re.finditer(r"\n\s*\n\s*\n", search_region):
                bp = max(pos, end - max_size // 3) + m.start()
                if bp > pos + max_size // 2:
                    break_point = bp
                    break
        if break_point is None:
            for m in re.finditer(r"\n\s*\n", search_region):
                bp = max(pos, end - max_size // 3) + m.start()
                if bp > pos + max_size // 2:
                    break_point = bp
                    break
        if break_point:
            chunks.append(text[pos:break_point].strip())
            pos = break_point - overlap if break_point - overlap > pos else break_point
        else:
            chunks.append(text[pos:end].strip())
            pos = end - overlap if end - overlap > pos else end
    return chunks


# ═══════════════════════════════════════════════════════════
# 图片过滤: 跳过谱图/显微图，只保留可能含分子结构的图片
# ═══════════════════════════════════════════════════════════

_SPECTRA_KEYWORDS = [
    r"\bNMR\b", r"\bIR\s*(spectrum|spectra)\b", r"\bMass\s*spect", r"\bESI-MS\b",
    r"\bTGA\b", r"\bPXRD\b", r"\bXRD\s*(pattern|data)", r"\bSEM\b", r"\bTEM\b", r"\bAFM\b",
    r"\bUV.vis\b", r"\bfluorescence\b", r"\belemental\s*analysis\b",
    r"\bchromatogram\b", r"\bthermogravim", r"\btransmittance\b",
    r"\bNOESY\b", r"\bCOSY\b", r"\bHSQC\b", r"\bHMBC\b",  # 2D NMR 技术
    r"\bwavenumber\b", r"\bchemical\s*shift\b",
    r"\bPXRD\b", r"\bdiffraction\b",
]

def _is_likely_structure_image(img: dict) -> bool:
    """判断图片是否可能是分子结构/反应方程式"""
    ctx = (img.get("context") or "").lower()
    # 谱图关键词 ≥2 个 → 整页跳过
    hits = sum(1 for kw in _SPECTRA_KEYWORDS if re.search(kw, ctx))
    if hits >= 2:
        return False
    return True


# ═══════════════════════════════════════════════════════════
# 阶段1: 多模态 VL — 仅提取 SMILES
# ═══════════════════════════════════════════════════════════

_SMILES_SYSTEM_PROMPT = """你是有机化学结构鉴定专家。你的唯一任务是从图片中提取化合物的SMILES结构式。

规则:
1. 只标注图片中能看到的化合物编号/名称 (如 "5AAA", "L1", "cage AAA")
2. 每个化合物给出对应的 SMILES
3. 如果图片是谱图(NMR/IR/MS)、显微图(TEM/SEM)、表格或不含化学结构，返回空列表
4. 不要提取反应条件、产率等文字信息
5. SMILES 从图片中逐字读取，不要修改

返回严格 JSON:
{
  "compounds": [
    {"label": "5AAA", "smiles": "C1=CC=C(C=C1)C=NCCN=CC2=CC=CC=C2"},
    {"label": "L1", "smiles": "O=CC1=CC=C(C=O)C=C1"}
  ]
}"""


def _extract_smiles_from_images(images: List[dict]) -> Dict[str, str]:
    """
    阶段1: 仅从图片提取 SMILES。
    返回 {label: smiles} 字典。
    """
    if not images:
        return {}

    # 过滤：只保留可能的分子结构图
    structure_imgs = [img for img in images if _is_likely_structure_image(img)]
    if not structure_imgs:
        return {}

    # 最多 15 张
    structure_imgs = structure_imgs[:15]

    # 构建 content array
    content = [
        {"type": "text", "text": "请提取以下图片中所有化合物的SMILES结构式。"}
    ]
    for img in structure_imgs:
        if img.get("path") and os.path.exists(img["path"]):
            try:
                with open(img["path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = img.get("ext", "png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                })
            except Exception:
                pass

    messages = [
        {"role": "system", "content": _SMILES_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    # 用 qwen-vl-plus
    vl_model = "qwen-vl-plus"
    client = _get_client()
    response = client.chat(messages, model=vl_model)

    # 解析 SMILES 字典
    smiles_dict = {}
    data = _parse_json(response)
    if data and "compounds" in data:
        for c in data["compounds"]:
            label = (c.get("label") or "").strip()
            smiles = (c.get("smiles") or "").strip()
            # 过滤垃圾 SMILES: 空/太短/太长/重复字符过多
            if not label or not smiles or len(smiles) < 4:
                continue
            if len(smiles) > 500:  # 超长 = VL 幻觉
                continue
            if any(smiles.count(ch) / len(smiles) > 0.6 for ch in 'C123456789'):
                continue  # 单一字符过多 = 聚合物幻觉
            smiles_dict[label.lower()] = smiles
            smiles_dict[label] = smiles

    return smiles_dict


# ═══════════════════════════════════════════════════════════
# 阶段2: 纯文本 — 完整反应数据
# ═══════════════════════════════════════════════════════════

def _extract_text_reactions(doi: str, title: str, text: str, source: str) -> tuple:
    """
    阶段2: 从文本中提取完整反应结构化数据。
    返回 (reactions_list, raw_json)。
    """
    chunks = _chunk_text(text)
    client = _get_client()
    all_reactions = []
    all_raw = []

    for i, chunk in enumerate(chunks):
        user_msg = config.USER_MESSAGE_TEMPLATE.format(
            doi=doi, title=title or doi,
            source=f"{source} (第{i+1}/{len(chunks)}部分)", text=chunk
        )
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        response = client.chat(messages)
        all_raw.append(response)
        data = _parse_json(response)
        if data and "reactions" in data:
            for r in data["reactions"]:
                all_reactions.append(ExtractedReaction(
                    reactant_smiles=r.get("reactant_smiles", []) or [],
                    reactant_names=r.get("reactant_names", []) or [],
                    metal_salts=r.get("metal_salts", []) or [],
                    metals=r.get("metals", []) or [],
                    anions=r.get("anions", []) or [],
                    temperature_celsius=r.get("temperature_celsius"),
                    time_hours=r.get("time_hours"),
                    solvents=r.get("solvents", []) or [],
                    atmosphere=r.get("atmosphere"),
                    products=r.get("products", []) or [],
                ))

    return all_reactions, json.dumps(all_raw, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 阶段3: 代码层 SMILES 注入
# ═══════════════════════════════════════════════════════════

def _best_smiles_match(label: str, smiles_dict: Dict[str, str]) -> Optional[str]:
    """
    为标签查找最佳 SMILES 匹配。
    1. 精确匹配 (大小写不敏感)
    2. 包含匹配 (label 包含 dict_key 或 dict_key 包含 label)
       选重叠部分最长的那一个
    """
    key = label.strip().lower()
    # 精确匹配
    if key in smiles_dict:
        return smiles_dict[key]
    if label.strip() in smiles_dict:
        return smiles_dict[label.strip()]

    # 包含匹配: 找重叠最长的
    best_smi = None
    best_overlap = 0
    for dk in smiles_dict:
        if key and dk and (key in dk or dk in key):
            overlap = min(len(key), len(dk))
            if overlap > best_overlap:
                best_overlap = overlap
                best_smi = smiles_dict[dk]
    return best_smi


def _inject_smiles(reactions: List[ExtractedReaction],
                   smiles_dict: Dict[str, str]) -> List[ExtractedReaction]:
    """阶段3: 包含匹配，将 SMILES 注入反应数据。"""
    if not smiles_dict:
        return reactions

    for rx in reactions:
        resolved = []
        for item in rx.reactant_smiles:
            smi = _best_smiles_match(item, smiles_dict)
            if smi:
                resolved.append(smi)
            else:
                resolved.append(item)  # 已是 SMILES 则保留
        rx.reactant_smiles = resolved

        for name in rx.reactant_names:
            smi = _best_smiles_match(name, smiles_dict)
            if smi and smi not in rx.reactant_smiles:
                rx.reactant_smiles.append(smi)

        for prod in rx.products:
            pname = (prod.get("product_name") or "").strip()
            smi = _best_smiles_match(pname, smiles_dict)
            if smi:
                prod["product_smiles"] = smi

    return reactions


# ═══════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════

def extract_one(
    pp: ParsedPaper,
    mode: str = "text",
    force: bool = False,
) -> Optional[ExtractionResult]:
    """
    mode: "text" | "multimodal" | "combined"
    """
    doi = pp.doi

    # 缓存
    cache_path = _cache_path(doi)
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            reactions = [ExtractedReaction(**r) if isinstance(r, dict) else r
                         for r in cached.get("reactions", [])]
            return ExtractionResult(
                doi=cached["doi"], success=cached.get("success", False),
                reactions=reactions,
                smiles_dict=cached.get("smiles_dict", {}),
                raw_json=cached.get("raw_json", ""),
                model_used=cached.get("model_used", ""),
                source=cached.get("source", ""),
                chunk_count=cached.get("chunk_count", 0),
                error=cached.get("error", ""),
            )
        except Exception:
            pass

    text = pp.si_experimental or pp.main_experimental or pp.main_text
    if not text or len(text) < 100:
        return ExtractionResult(doi=doi, error="无有效文本")

    source = "SI" if pp.si_experimental else ("正文实验部分" if pp.main_experimental else "正文")
    has_images = pp.images and len(pp.images) > 0
    title = ""

    try:
        if mode == "combined" and has_images:
            # 阶段1: 图片 → SMILES 字典
            smiles_dict = _extract_smiles_from_images(pp.images)

            # 阶段2: 文本 → 反应数据
            reactions, raw = _extract_text_reactions(doi, title, text, source)

            # 阶段3: 代码注入
            reactions = _inject_smiles(reactions, smiles_dict)

            smi_count = len(smiles_dict)
            result = ExtractionResult(
                doi=doi, success=len(reactions) > 0, reactions=reactions,
                smiles_dict=smiles_dict,
                raw_json=json.dumps({"text_reactions": raw, "smiles_from_images": smiles_dict},
                                    ensure_ascii=False),
                model_used=f"{config.QWEN_MODEL} + qwen-vl-plus",
                source=source, chunk_count=len(_chunk_text(text)),
            )

        elif mode == "multimodal" and has_images:
            smiles_dict = _extract_smiles_from_images(pp.images)
            result = ExtractionResult(
                doi=doi, success=len(smiles_dict) > 0,
                smiles_dict=smiles_dict,
                raw_json=json.dumps(smiles_dict, ensure_ascii=False),
                model_used="qwen-vl-plus", source=source,
            )

        else:
            reactions, raw = _extract_text_reactions(doi, title, text, source)
            result = ExtractionResult(
                doi=doi, success=len(reactions) > 0, reactions=reactions,
                raw_json=raw, model_used=config.QWEN_MODEL,
                source=source, chunk_count=len(_chunk_text(text)),
            )

    except Exception as e:
        result = ExtractionResult(doi=doi, error=str(e))

    if result:
        _save_cache(doi, result)

    return result


def extract_batch(
    parsed: List[ParsedPaper],
    mode: str = "text",
    force: bool = False,
) -> List[ExtractionResult]:
    results = []
    if not config.DASHSCOPE_API_KEY:
        print("❌ 未配置 DASHSCOPE_API_KEY，请在 config.py 中填写")
        return results

    names = {"text": "纯文本", "multimodal": "仅SMILES", "combined": "图片SMILES+文本反应"}
    print(f"\n🤖 千问提取 ({names.get(mode, mode)}, 并发={config.CONCURRENCY})")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as executor:
        futures = {executor.submit(extract_one, pp, mode, force): pp for pp in parsed}
        with tqdm(total=len(futures), desc="提取进度") as pbar:
            for future in as_completed(futures):
                pp = futures[future]
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                        status = f"{pp.doi[:30]} | "
                        if mode in ("combined", "text"):
                            status += f"{len(r.reactions)}反应"
                            if r.smiles_dict:
                                status += f" {len(r.smiles_dict)}SMILES"
                        else:
                            status += f"{len(r.smiles_dict)}SMILES"
                        pbar.set_postfix_str(status[:50])
                except Exception as e:
                    results.append(ExtractionResult(doi=pp.doi, error=str(e)))
                pbar.update(1)

    ok = sum(1 for r in results if r.success)
    total_rxn = sum(len(r.reactions) for r in results)
    total_smi = sum(len(r.smiles_dict) for r in results)
    print(f"\n📊 成功={ok}/{len(results)} | 反应={total_rxn} | SMILES条目={total_smi}")
    return results


# ═══════════════════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════════════════

def _cache_path(doi: str) -> str:
    safe = doi.replace("/", "_").replace("\\", "_")[:80]
    return os.path.join(config.CACHE_DIR, f"{safe}.json")


def _save_cache(doi: str, result: ExtractionResult):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    d = {
        "doi": result.doi, "success": result.success,
        "reactions": [
            {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
            for r in result.reactions
        ],
        "smiles_dict": result.smiles_dict,
        "raw_json": result.raw_json, "model_used": result.model_used,
        "source": result.source, "chunk_count": result.chunk_count,
        "error": result.error,
    }
    with open(_cache_path(doi), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ris_parser import parse_ris
    from pdf_matcher import match_pdfs
    from parser import parse_all as parse_papers

    records = parse_ris("x-mol.ris")
    matched = match_pdfs(records)
    valid = [mp for mp in matched if mp.has_main][:2]
    parsed = parse_papers(valid, img_dir=config.IMAGE_DIR)

    results = extract_batch(parsed, mode="combined")

    for r in results:
        print(f"\n{'='*60}")
        print(f"DOI: {r.doi}")
        print(f"反应: {len(r.reactions)} | SMILES库: {len(r.smiles_dict)} 条")
        for label, smi in list(r.smiles_dict.items())[:5]:
            print(f"  {label} → {smi[:60]}...")
        for i, rx in enumerate(r.reactions[:3]):
            has_smi = len(rx.reactant_smiles) > 0
            print(f"  反应{i+1}: SMILES={'✅' if has_smi else '❌'} metals={rx.metals}")
