"""
统一配置
-------
API 密钥、模型、批量参数、Prompt 模板，所有模块共用。
"""

# ═══════════════════════════════════════════════════════════
# 千问 API (DashScope OpenAI 兼容模式)
# ═══════════════════════════════════════════════════════════
DASHSCOPE_API_KEY = ""   # https://dashscope.console.aliyun.com
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型选择
QWEN_MODEL = "qwen-plus"       # qwen-turbo | qwen-plus | qwen-max | qwen3-235b-a22b
QWEN_TEMPERATURE = 0.1         # 结构化提取用低温
QWEN_MAX_TOKENS = 4096         # 输出上限
QWEN_TIMEOUT = 120             # 单次请求超时 (秒)

# ═══════════════════════════════════════════════════════════
# 分批提取参数
# ═══════════════════════════════════════════════════════════
TEXT_CHUNK_SIZE = 5000          # 每片最大字符数 (约 3000-4000 token)
TEXT_CHUNK_OVERLAP = 300        # 片间重叠字符数
CONCURRENCY = 3                 # 并发 API 请求数
RETRY_MAX = 3                   # 失败重试次数
RETRY_BACKOFF = [1, 3, 8]       # 重试等待秒数 (指数递增)
REQUEST_DELAY = 0.3             # 请求间隔 (秒, 避免限流)

# ═══════════════════════════════════════════════════════════
# 输出路径
# ═══════════════════════════════════════════════════════════
CACHE_DIR = "extraction_cache"  # 提取结果缓存 (按 DOI 存 JSON)
IMAGE_DIR = "extracted_images"  # 从 PDF 提取的图片

# ═══════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是合成化学家助手，精读有机金属笼（metal-organic cage, coordination cage, metallacage）论文的实验部分。

从提供的文本中提取所有笼状产物的合成信息。只提取文中明确呈现的数据，绝不推测。缺失的字段写 null。

返回严格 JSON，格式如下：
{
  "reactions": [
    {
      "reactant_smiles": ["C1=CC=..."],
      "reactant_names": ["subcomponent A", "subcomponent B"],
      "metal_salts": ["Pd(NO3)2"],
      "metals": ["Pd(II)"],
      "anions": ["NO3-"],
      "temperature_celsius": 80.0,
      "time_hours": 24.0,
      "solvents": ["DMF", "H2O"],
      "atmosphere": "N2",
      "products": [
        {
          "product_name": "[Pd2L4](NO3)4",
          "product_formed": true,
          "product_formula": "C120H96N24O12Pd2",
          "crystal_appearance": "pale yellow block",
          "csd_number": "CCDC 2345678",
          "crystal_system": "monoclinic",
          "space_group": "P21/c",
          "unit_cell_params": "a=15.2 b=22.4 c=18.7 α=90 β=105 γ=90"
        }
      ]
    }
  ],
  "extraction_notes": "提取了1个反应，CSD数据完整"
}

字段说明：
- reactant_smiles: 反应物的 SMILES 字符串。副组件/前体/配体的SMILES。从文中逐字复制，不要修改。
- reactant_names: 反应物的化学名或编号 (如 "subcomponent A", "ligand L1")。
- metal_salts: 使用的金属盐全称 (如 "Pd(NO3)2", "Zn(OTf)2")。
- metals: 提取的金属离子及氧化态 (如 "Pd(II)", "Zn(II)")。
- anions: 阴离子 (如 "NO3-", "OTf-", "BF4-")。
- temperature_celsius: 反应温度，单位 °C。如文中写 "80°C" 填 80.0。室温填 25.0。
- time_hours: 反应时间，单位小时。如 "30 min" 填 0.5，"overnight" 填 16.0。
- solvents: 溶剂列表 (如 ["DMF", "H2O"])。
- atmosphere: 反应气氛 ("N2", "Ar", "air", "O2")。
- products: 产物列表。一个反应可生成多个笼产物。
- product_name: 产物名称/编号 (如 "Pd2L4 cage 1a", "[Zn6L4](OTf)12")。
- product_formed: true=笼子成功生成, false=反应失败/未生成目标笼。
- product_formula: 分子式 (如 "C120H96N24O12Pd2")。
- crystal_appearance: 晶体外观描述 (如 "pale yellow block crystals")。
- csd_number: CCDC 沉积号 (如 "CCDC 2345678", "CCDC-2345678", 或纯数字)。
- crystal_system: 晶系 (triclinic/monoclinic/orthorhombic/tetragonal/cubic/hexagonal/trigonal)。
- space_group: 空间群 (如 "P21/c", "Fm-3m")。
- unit_cell_params: 晶胞参数简写。
- extraction_notes: 提取情况简述。
"""

# 多模态 Prompt — 有图片时使用
SYSTEM_PROMPT_MULTIMODAL = SYSTEM_PROMPT + """

如果文本中引用了图片编号 (如 Figure S1, Scheme 1)，且图片内容为反应方程式或分子结构，
请一并参考图片提取 SMILES 和反应信息。图片内容优先于文本描述。"""

# 用户消息模板
USER_MESSAGE_TEMPLATE = """论文 DOI: {doi}
论文标题: {title}

以下为论文{source}文本：

{text}

请提取其中的有机金属笼合成信息。"""

USER_MESSAGE_MULTIMODAL = """论文 DOI: {doi}
论文标题: {title}

以下为论文{source}文本和实验相关图片。
请综合文本和图片，提取有机金属笼合成信息。"""
