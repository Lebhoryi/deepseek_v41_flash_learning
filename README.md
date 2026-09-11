# DeepSeek-V4.1-Flash 源码研读仓库

> 官方 HuggingFace 仓库 [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的本地镜像 + 学习笔记。
> 不含模型权重（全量约 510 GB / 48 分片，仅保留 `index.json` 索引），用于**静态读源码**研习 V4.1-Flash 的网络结构与创新点。

- **模型**：DeepSeek-V4.1-Flash（2026-09 发布，MIT License）
- **架构**：552B 主干 + 196B Engram；激活 8B(prefill) / 16B(decode)；40 层 CED（20 编码器 + 20 解码器）
- **创新点**：CED 因果编解码器 / CSA2 压缩稀疏注意力 2 / 分层稀疏索引器 / Single-Pass mHC / Engram 条件记忆 / DSpark 投机解码 / FP4 主 KV 缓存（890 B/token）
- **官方技术报告**：`DeepSeek_V41_Tech_Report.pdf`（51 页，文本版见 `docs/Tech_Report.txt`）

## 仓库结构

目录结构已按官方仓库还原（下载时被打平，导致内部导入失效；2026-09-11 还原修复）。

```
dsv41-flash-learning/
├── README_hf.md                    # 官方模型卡（英文原版，含评测对比表与采样参数）
├── config.json                     # HF 权重配置（architectures=DeepseekV41ForCausalLM）
├── index.json                      # 48 片 safetensors 权重索引（96085 张量 / 510 GB）
├── DeepSeek_V41_Tech_Report.pdf    # 官方技术报告
├── encoding/                       # Prompt 编码参考实现（无推理依赖）
│   ├── encoding.py                 #   编解码：encode_messages / parse_message_from_completion_text
│   ├── test_encoding.py            #   21 个测试 = V4.1 格式契约（DSML 前导空格 / effort 1-100 / 中途 System 消息）
│   └── README.md
├── inference/                      # 最小可读推理实现（可读参考，非生产引擎）
│   ├── model.py                    #   核心前向 1309 行：CED / CSA2 三模式 / Engram / mHC / DSpark
│   ├── kernel.py                   #   TileLang kernel：dense-fp8 / MoE-fp4 GEMM、sparse_attn、hc_split_sinkhorn
│   ├── convert.py                  #   HF 权重 → TP 分片转换（含 FP4 反量化 cast_e2m1fn_to_e4m3fn）
│   ├── engram.py                   #   n-gram 哈希查表（素数表 / 压缩 token map / NgramHashState）
│   ├── image_processor.py          #   图像网格规划（resize 比例 / token 约束）
│   ├── vision.py                   #   DeepSeek-ViT + Aligner
│   ├── generate.py                 #   生成入口（torchrun TP 并行）
│   ├── config.json                 #   推理侧配置（generate.py 实际读取）
│   ├── run.sh / requirements.txt / README.md
└── docs/                           # 学习笔记
    ├── Tech_Report.txt             #   技术报告文本提取版（2879 行，可全文检索）
    └── ch2-architecture-mindmap.html  # 第二章 Architecture 交互式思维导图
```

## 快速开始

```bash
# 参考实现自测（小模型，走真实 dense-fp8 / MoE-fp4 kernel，验证形状与管道，无需权重）
python inference/model.py

# Prompt 编码契约测试（无需 torch）
pytest encoding/test_encoding.py

# 完整推理流程：先转权重，再生成（需要 510GB 权重 + 多卡）
export HF_CKPT_PATH=<权重目录> SAVE_PATH=<输出目录> MP=8
python inference/convert.py --hf-ckpt-path $HF_CKPT_PATH --save-path $SAVE_PATH \
  --model-parallel 8 --expert-dtype fp4 --tokenizer-path $HF_CKPT_PATH
./inference/run.sh $SAVE_PATH        # 默认 MP=8，可 MP=4 覆盖
```

> 注意：`inference/generate.py` 通过 `sys.path.insert(../encoding)` 引用仓库根的 `encoding/` 目录，**目录结构不可再扁平化**。

## 学习笔记

- **架构全景**：`docs/ch2-architecture-mindmap.html`（第二章 Architecture 思维导图，可交互折叠）
- **创新点 → 源码坐标**（报告章节 → 代码位置）：

| 创新点 | 报告 | 源码 |
|---|---|---|
| CED 因果编解码器 | 2.2 | `config.json` compress_ratios / kv_source_layers；`inference/model.py` `Transformer.forward` |
| CSA2 三模式（Full/Reindex/Reuse） | 2.3.1 | `inference/model.py` `Attention`(613) / `Indexer`(488) / `Compressor`(429) |
| 分层稀疏索引器 | 2.3.2 | `inference/model.py` `select_candidate_blocks`(583)；config candidate_* |
| Single-Pass mHC | 2.4.1 | `inference/model.py` `Block`(907) hc_mixes/pre/post；`kernel.py` hc_split_sinkhorn(465) |
| Engram 条件记忆（196B） | 2.4.2 | `inference/engram.py` + `model.py` `Engram`(328) |
| DSpark 投机解码 | 2.4.3 | `inference/model.py` DSpark*（1021-1158）/ forward_spec |
| FP4 主 KV 缓存 | 2.4.4 | `inference/convert.py` cast_e2m1fn_to_e4m3fn(18)；`kernel.py` fp4 系列 |
| head-wise Muon / Sinkhorn 优化 | 2.5 | 报告 Algorithm 1（训练侧，代码不在仓库） |

## 与官方仓库的差异

| 缺失项 | 说明 |
|---|---|
| `examples/` | 示例输入（example.txt / example_harmony.json），可从 HF 补 |
| `evaluation/` | DeepSWE v1.1 复现说明（dsh-minimal + Pier patch），可从 HF 补 |
| `LICENSE` | 官方为 MIT License |
| 模型权重 | 全量 510 GB / 48 分片，需自行下载 |
| `assets/` | README 中的两张对比图（README_hf.md 引用） |

其他说明：官方仓库有 `Tech_Report.pdf` 与 `DeepSeek_V41_Tech_Report.pdf` 两份同名内容（已校验文本一致），本仓库仅保留官方命名的后者。

## 相关链接

- 模型仓库：[deepseek-ai/DeepSeek-V4.1-Flash · HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- Prompt 生产级编码库（Rust）：[deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)
- 官方模型卡（英文）：[`README_hf.md`](./README_hf.md)