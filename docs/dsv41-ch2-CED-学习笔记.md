# DeepSeek-V4.1-Flash · 2.2 CED 学习记录（自测 + 澄清）

> 日期：2026-09-15
> 来源：技术报告 2.2 Causal Encoder-Decoder（`docs/Tech_Report.txt` L379–413，公式 (1)）+ 3.2.2 SWA Bounded Replay（L990–1026）
> 说明：本笔记是一次 CED 自测的记录，含 8 道题（题目 + 作答 + 批改）与两次专题澄清（replay 概念、CED vs YoCo 的两项改进）。
>
> ⚠️ **重要提醒**：参考实现 `inference/model.py` 的 `Transformer.forward`（40 层统一顺序跑）**并未真正实现 CED** —— `kv_source_layers=[2,8,14,20]` 是 CSA2 Full Mode 的层间 KV 复用，不是 CED 的 `H_{L/2}→decoder KV` 投影；权重 `index.json` 里也没有 `W_l^KV / W_l^Z` 投影参数。CED 机制目前**仅存于技术报告 2.2**，README 里"把 CED 指到 Transformer.forward"的映射表是误导，别照那个学。

---

## 一、自测 8 题（题目 + 作答 + 批改）

> 作答为当时凭记忆写出，未翻报告；批改按报告原文逐条校正。

### Q1 · 动机

- **题目**：CED 要解决什么瓶颈？为什么在 agentic workflow 场景下会被放大？
- **作答**：解决 prefill 计算瓶颈、减少 TTFT；agent 里工具调用多，KV miss 时 prefill 要重新计算。
- **批改**：✓。报告原话"频繁 tool call 产生大量 prefill 请求，KV miss 时重算开销严重"。TTFT 的推理也对（prefill 减半即压 TTFT）。

### Q2 · 溯源（相比 YoCo 的两项改进）

- **题目**：CED 借鉴了哪个工作？该工作核心做法？CED 在此基础上做了哪两个方向的结构性改进？
- **作答**：借鉴 YoCo；核心是上半层直接复用下半层生成的 KV。改进 1 = 复用 hidden 经投影重新生成 KV、每层 KV 不一样；改进 2 = 不知道。
- **批改**：△。YoCo 核心对。但两项改进的正式表述是：
  1. **增强整体 KV cache 容量**（overall KV cache capacity）；
  2. **增强 KV 生成的计算深度**（computational depth of KV generation）。
  "投影重生成、每层不一样"是②的机制体现，①（容量）漏了。详见下文"专题澄清 2"。

### Q3 · 架构划分

- **题目**：L 层分成哪两部分，各自叫什么、边界在哪？
- **作答**：encoder 和 decoder，0–19 encoder，20–39 decoder。
- **批改**：✓。40 层 = 20 层 causal encoder + 20 层 decoder。报告 `l > L/2` 是 1-indexed（21–40），对应 0-index 的 20–39；`H_{L/2}` 是 encoder 最后一层（第 20 层）的输出。

### Q4 · 核心公式 (1)

- **题目**：写出公式 (1)，说明 `C_l`、`Z_l` 各是什么，`W_l^KV`、`W_l^Z` 的 layer-dependent 意味着什么。
- **作答**：`C_l` = decoder 全局 KV cache；`Z_l` = 对应缩放；`W_l^KV`、`W_l^Z` 表示 prefill 时 decoder 不用再经过 attn+FFN，直接投影计算、省算力。
- **批改**：△。三处校正：
  - `C_l` = decoder 全局注意力的 **KV entries** ✓；
  - `Z_l` = **compression weights（压缩权重）**，不是"缩放/scale"，它配合 KV 压缩（呼应 FP4 主 KV 缓存）；
  - layer-dependent 的关键点是**每一层 decoder 有一套自己独立的投影矩阵** —— 输入都是同一个 `H_{L/2}`，但每层投影出的 KV 各不相同。"不用过 attn+FFN、直接投影省算力"方向对（prefill 阶段 decoder 层不产生自己的 hidden state，这正是省算力来源）。

### Q5 · 全局 vs 滑动窗口（重点辨析）

- **题目**：全局注意力的 KV 和 SWA 的 KV 分别从哪一层 hidden state 推导？为什么 SWA 要逐层计算、不也去投影？
- **作答**：全局从 encoder 最后一层 `H_{L/2}`；SWA 需要当前层的计算缓存、不能投影。
- **批改**：△。前半对（全局从 `H_{L/2}` 投影；SWA 从每层自己的 `H_l` 逐层推导）。但"为什么逐层"的正式理由是**增加局部 KV 生成的计算深度**。"需要当前层缓存"是现象不是原因，因果反了。

### Q6 · SWA Replay 的开销（最大缺口）

- **题目**：prefill 阶段算 decoder SWA KV 为何必须额外 replay？额外开销是多少 tokens？"有效感受野更小"引出什么优化？
- **作答**：额外开销是 `n_win`（默认 128）；其余两问不知道，"什么是 replay？"
- **批改**：✗。四处校正：
  1. 额外开销不是 `n_win`，是 **`n_win × L/2`** tokens（漏了 × L/2，即 decoder 层数）；
  2. `n_win = 128` 答对（`config.json sliding_window: 128`，报告 L1111）；
  3. **replay = 重放**（见"专题澄清 1"）；
  4. "有效感受野更小"（Chen et al. 2025）→ **Decoder SWA Bounded Replay**：只 prefill 最后 `n_win` 个 token + 截断窗口、接受近似，显著省算力。

### Q7 · 复杂度

- **题目**：`N >> n_win` 时 prefill 复杂度从多少降到多少？
- **作答**：`O(NL)` → `O(NL/2)`。
- **批改**：✓（终态对）。精确中间式是 `O(NL/2 + n_win×L/2)`，`N >> n_win` 时 ≈ `O(NL/2)`。中间那一项正是 Q6 replay 开销的来源。

### Q8 · 辨析（拔高）

- **题目**：CED 的"上半层共享 KV"和 CSA2 的"层维度 KV 复用"本质区别？
- **作答**：CSA2 还没看，回答不了；CED 上半层共享的是全局 KV。
- **批改**：△。CED 半句对（共享全局 KV，且从单一 `H_{L/2}` 逐层投影**重新生成**，不是直接复用）。CSA2 待看 2.3 后回来补：CSA2 是"层维度**直接复用**其它层的 cache/索引"，与 CED 的"投影重新生成"机制本质不同。

---

## 二、专题澄清 1：什么是 replay（重放）

**一句话**：replay（重放）= 把已经算过一遍的 token，再喂进 decoder 层补跑一遍，专门补出 CED 跳过的 decoder SWA KV。

**为什么要 replay（缺了什么）**：
- CED 的 prefill 只跑到 encoder 第 20 层就停（得到 `H_{L/2}`）。
- decoder 层的全局 KV 可直接从 `H_{L/2}` 投影，所以不用跑 decoder 层。
- 但 decoder 层还有一套 **SWA KV**，按设计**必须从每一层自己的 hidden state 算出来**（不能投影）；prefill 没跑 decoder 层 → 这套 SWA KV 是空的。
- decode 第一步立刻要用它（局部注意力看最近 128 个 token 的 K/V），缺口必须补。

**具体重放什么（三段式）**：
1. 对象：prompt 的**最后 `n_win = 128` 个 token**（不是全部）；
2. 路径：**decoder 层 20–39**（prefill 跳过的层），只走 SWA 分支；
3. 产出：decoder 各层的 **SWA KV**，仅供 decode 用，**不写进前缀缓存**。

> 关键细节：replay **不是从零重算**。prefill 时所有 token 已过完 encoder，最后 128 个 token 的 `H_{L/2}` 早算好留着，replay 就是把这 128 个 `H_{L/2}` 再喂进 decoder 层跑一遍，encoder 部分复用、不重算。

**为什么叫"重放"**：这 128 个 token 在 prefill 时已完整"演"过 encoder 那半场，现在又为补 SWA 在 decoder 上"再走第二遍"。

**1000-token 时间线**：

| 阶段 | 发生了什么 |
|---|---|
| ① prefill | token 1…1000 过 encoder 层 0–19 → `H_{L/2}` → 投影出 decoder 全局 KV，**停**（decoder 层未运行） |
| ② replay | 最后 128 个 token（873…1000）的 `H_{L/2}` 重新喂进 decoder 层 20–39，算 SWA KV |
| ③ decode | decoder 全局 KV（投影）+ SWA KV（重放）齐了，开始生成第一个 token |

**"Bounded" 的含义**：理论上精确重建 decoder SWA KV 更贵（SWA 依赖逐层累积、感受野逐层扩大），但实际有效感受野远小于理论值 → 只重放最后 128 个 token + 截断窗口、接受近似状态，质量几乎不降。所以叫 **Bounded Replay（有界重放）**——重放范围被限制在最后 `n_win` 个 token。

> 报告里有两个 Bounded Replay：**Decoder**（上述，2.2/3.2.2）和 **Encoder**（前缀缓存命中时 encoder SWA KV 未持久化，也需重放最后 `n_win` 个 token 来补）。

---

## 三、专题澄清 2：CED 相比 YoCo 的两项改进

**YoCo 的基准做法**：L 层分上下两半，上半层**不生成自己的 KV**，直接复用下半层已算好的同一套 KV（cache once）。代价 = 上半层所有层用的都是**同一份 KV**，层 21 和层 39 看到的 K/V 一模一样。

**改进 ① 增强 KV cache 容量（overall KV cache capacity）**：
- YoCo：记忆库里只有 **1 份通用 KV**，上半 20 层全取同一份 → 容量小。
- CED：每层用自己的投影权重 `W_l^KV` 从 `H_{L/2}` 提炼出**自己专属的一份 KV**，20 层 = 20 份逐层不同的 KV → 容量大。
- 本质：**YoCo 是"共享一份"，CED 是"每层一份"**。配套的 `Z_l`（compression weights）给这些"变多变丰富"的 KV 配压缩权重，容量涨上去的同时存储压得下去（呼应 FP4 主 KV 缓存）。⚠️ 报告对 `Z_l` 只给了"压缩权重"一句话，具体压缩机制未展开，属原文留白。

**改进 ② 增强 KV 生成的计算深度（computational depth of KV generation）**：
- YoCo：KV 是下半层一次算出来、原样复用，生成过程扁平、浅层。
- CED：KV 逐层重新投影生成（`C_l = H_{L/2} · W_l^KV`，每层一套权重），且 SWA KV 逐层从每层自己的 hidden state 算 → 生成经过层层加工。

**对照表**：

| | YoCo | CED |
|---|---|---|
| 上半层 KV 来源 | 直接复用下半层同一份 KV | 从 `H_{L/2}` 逐层投影，**每层独立一份** |
| KV 容量 | 1 份共享，扁平 | L/2 份逐层特化，容量大 |
| KV 生成深度 | 浅（一次性复用） | 深（逐层投影 + SWA 逐层算） |
| 结果 | 省 prefill，但表达能力受限 | 省 prefill **且性能不掉** |

---

## 四、待补缺口（下次自测前要补齐）

1. **`Z_l` 压缩权重的具体机制**：报告只给"compression weights"一句话，未展开如何压缩，需查后续论文/代码。
2. **CSA2 层维度 KV 复用 vs CED 的区别**：还没看 2.3，看完回来补 Q8。
3. **Decoder SWA Bounded Replay 的"截断/近似"细节**：截断后 query 的注意力范围 `[max(s, i−W+1), i]` 的精确语义（报告 L994–996）。

---

## 附：核心公式与原文定位

- **公式 (1)**（报告 L394–396）：`C_l = H_{L/2} · W_l^KV`，`Z_l = H_{L/2} · W_l^Z`，`l > L/2`
  - `C` = KV entries；`Z` = compression weights；`H_{L/2}` = encoder 最后一层 hidden state。
- **复杂度**（报告 L412–413）：`O(NL)` → `O(NL/2 + n_win·L/2) ≈ O(NL/2)`。
- **关键参数**：`num_hidden_layers=40`、`sliding_window=128`（`config.json`）。
