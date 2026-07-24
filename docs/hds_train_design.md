# HDS 统一训练设计草案

## 1. 文档状态

本文档描述当前计划采用的 HDS 简化训练方案，属于可继续修改的工作草案。

本方案保留 `docs/hds_design.md` 中定义的 canonical HDS 格式，重点重新划分哪些内容由模型学习、哪些内容由确定性程序构造。当前 E2 实现只完成 CSP；de novo generation 保留为 E4 后续实验。

核心原则是：

> 训练时用完整 HDS 自回归目标学习序列语义，并用 global/local code 辅助分类损失强化离散 code；生成时只采样不确定变量，由程序强制构造能够唯一确定的 HDS 骨架。

## 2. 三个模型的职责

### 2.1 Attn-RVQ

Attn-RVQ 只负责把晶体结构编码为离散的全局和局部 code：

```text
crystal structure
-> global RVQ indices
-> local RVQ indices for every site
```

本方案不要求在 Attn-RVQ 内增加独立的 formula-conditioned global prior。Attn-RVQ 保持结构 tokenizer 的职责，避免与 HDS 生成任务耦合。

### 2.2 HDS 模型

HDS 模型负责：

```text
CSP:
已知 composition 和 atom skeleton
-> 预测 global/local codes

de novo（E4 后续，不属于当前 E2 实现）：
从 BOS 生成 composition
-> 程序构造 atom skeleton
-> 预测 global/local codes
```

当前 E2 checkpoint 训练完整 HDS causal AR 语义，但正式推理入口只启用已知 composition 的 constrained CSP code generation。是否复用同一 checkpoint 从 BOS 生成 composition，留到 E4 验证后决定，不额外训练独立 composition generator。

### 2.3 条件扩散模型

条件扩散模型读取完整 HDS 中的 global/local code，查询对应 Attn-RVQ codebook，并使用所得连续条件生成晶格和分数坐标。

测试集真实结构提取的 direct HDS 仅用于：

- HDS 一致性评估；
- 扩散模型 oracle 上界评估；
- HDS 模型监督目标。

正式 CSP 生成不得使用测试结构提取的 direct HDS。

## 3. 统一 HDS 格式

最终输出仍是完整 HDS：

```text
<BOS> <CRYSTAL>

<FORMULA>
  <ATOM_A> <COUNT_nA>
  <ATOM_B> <COUNT_nB>
</FORMULA>

<GLOBAL>
  <NUM_SITES_N>
  <GLOBAL_STAGE_0> <GLOBAL_CODE_g0>
  ...
  <GLOBAL_STAGE_Qg-1> <GLOBAL_CODE_gQg-1>
</GLOBAL>

<SITES>
  <SITE_1> <ATOM_X1>
    <LOCAL_STAGE_0> <LOCAL_CODE_l10>
    ...
    <LOCAL_STAGE_Ql-1> <LOCAL_CODE_l1,Ql-1>
  </SITE_1>
  ...
</SITES>

</CRYSTAL> <EOS>
```

其中：

```text
Qg = Attn-RVQ 的 global quantizer 数量
Ql = Attn-RVQ 的 local quantizer 数量
N  = 完整 primitive cell 中的原子数量
```

当前常用配置为 `Qg=2`、`Ql=3`，但模型和数据程序必须从 Attn-RVQ 配置或 checkpoint 读取，不允许在代码中固定为五个分类位置。

## 4. AR 训练目标与生成约束

训练阶段保留完整 HDS 的自回归 next-token objective，使 Transformer 学习 formula、全局语义、site 语义和 RVQ code 之间的顺序关系。

“确定性 token”是指正式生成时由程序强制插入，而不是指训练时完全不学习这些 token。

| HDS 内容 | 完整 AR 训练 | CSP 生成 | De novo 生成 |
| --- | --- | --- | --- |
| composition 内容 | next-token 监督 | 由用户条件给定 | 从 BOS 自回归采样 |
| `NUM_SITES` | next-token 监督 | 由 composition 计算并强制插入 | formula 完成后计算并插入 |
| `SITE_i` | next-token 监督 | 强制插入 | 强制插入 |
| site `ATOM_X` | next-token 监督 | composition 展开后强制插入 | composition 展开后强制插入 |
| global/local stage marker | next-token 监督 | 按 Attn-RVQ 配置强制插入 | 按 Attn-RVQ 配置强制插入 |
| global code | AR 主任务 + auxiliary code loss | 受限采样 | 受限采样 |
| local code | AR 主任务 + auxiliary code loss | 受限采样 | 受限采样 |
| HDS 结构标签 | next-token 监督 | 由语法状态机强制插入 | 由语法状态机强制插入 |

CSP 的正式生成流程为：

```text
known full-cell composition_counts
-> compiler builds the complete deterministic skeleton
-> AR model consumes the ordered semantic prefix
-> only global/local code positions are sampled
```

de novo 的候选设计如下，但属于 E4，当前生成脚本会明确拒绝 constrained de novo：

```text
<BOS>
-> AR generates FORMULA content
-> compiler parses composition_counts
-> NUM_SITES and atom skeleton are determined
-> generation switches to the same constrained AR code path as CSP
```

上述 de novo 路径是后续目标，不代表当前 E2 已实现。当前只将已知 FORMULA/full-cell composition 编译为确定性骨架，再受限生成 global/local codes。

## 5. Deterministic HDS Compiler

新增一个无可学习参数的 HDS compiler。当前负责 CSP 骨架构造与 direct-HDS 严格重编译；E4 再扩展到 de novo 骨架构造。

输入：

```text
composition_counts
Qg
Ql
```

输出：

```text
canonical FORMULA block
NUM_SITES
ordered atom skeleton
SITE_i blocks
GLOBAL_STAGE_q markers
LOCAL_STAGE_q markers
global/local code slots
all closing tags
```

compiler 必须执行以下约束：

```text
所有元素 count 为正整数
元素按原子序数排序
同一元素不能重复出现
NUM_SITES = sum(composition counts)
NUM_SITES 不超过数据集最大限制
site atom sequence 等于 composition 展开结果
global stage 从 0 到 Qg-1
每个 site 的 local stage 从 0 到 Ql-1
```

compiler 不允许执行：

```text
首元素补齐
静默截断
从 formula 和 site blocks 两个独立来源推断 atom types
发现冲突后继续采样
```

## 6. 推荐模型结构

主模型采用一个 decoder-only causal Transformer，对完整 canonical HDS 进行自回归建模：

```text
p(HDS) = product_t p(token_t | token_<t)
```

在共享 AR backbone 上增加动态 stage-specific code classification heads：

```text
shared causal Transformer
shared vocabulary LM head
Qg global auxiliary classification heads
Ql local auxiliary classification heads
```

辅助 head 的数量和类别数来自 Attn-RVQ 配置，不固定为五个 head。

### 6.1 完整序列 AR 主任务

共享 LM head 预测所有 HDS token，包括 formula、syntax、stage marker 和 global/local code，从而保留 HDS 作为语义序列的整体建模能力。

### 6.2 Stage-specific code 辅助任务

当下一个 target 是 code token 时，使用 code 前一位置的 causal hidden state 进行额外分类：

```text
<GLOBAL_STAGE_q> hidden
-> global auxiliary head q
-> codebook index

<LOCAL_STAGE_q> hidden
-> local auxiliary head q
-> codebook index
```

同一个 code 位置同时受到共享 AR vocabulary loss 和 stage-specific auxiliary classification loss 的监督。

### 6.3 受限 AR 解码

模型训练学习完整序列，但生成时使用 HDS compiler 和词表约束：

```text
deterministic skeleton position -> 强制写入唯一合法 token
global code position            -> 只允许当前 global stage code
local code position             -> 只允许当前 local stage code
formula position                -> 只允许合法 ATOM/COUNT/closing token
```

因此保留 AR 语义建模并不会重新引入 formula/site 数量不一致。

### 6.4 解码时融合辅助 head

在 code 位置可以融合共享 AR logits 与辅助分类 logits：

```text
code_logits = restricted_AR_logits + beta_aux * auxiliary_logits
```

首轮同时报告：

```text
beta_aux = 0.0：只使用 AR head
beta_aux = 1.0：AR 与 auxiliary head 融合
```

这样可以区分性能提升来自训练期表示正则化，还是来自推理阶段的分类校准。

## 7. 单次训练流程与 Curriculum

模型从头进行一次完整训练，不再维护 Local Masked Pretrain 和 Hierarchical Fine-tune 两个独立 checkpoint。

### 7.1 标准 AR teacher forcing

每条 canonical direct HDS 使用普通 causal shift：

```text
input_ids  = full_hds_ids[:-1]
target_ids = full_hds_ids[1:]
```

同一条序列同时提供：

```text
BOS -> FORMULA 的 de novo 学习信号（仅保留在完整 AR 目标中，当前不启用对应推理）
FORMULA -> GLOBAL 的 CSP 条件学习信号
GLOBAL -> LOCAL 的 hierarchical 学习信号
```

因此不需要人为构造 50% CSP view 和 50% de-novo view。

### 7.2 Global self-conditioned Curriculum

普通 teacher forcing 会让 local code 在训练时始终看到真实 global code，而生成时只能看到预测 global code。为减少这个差异，采用一次训练内的 curriculum：

```text
Warmup：
  所有 local target 使用 true global context

Self-conditioned phase：
  第一遍前向预测 global codes
  按 predicted_global_fraction 将部分 true global tokens 替换为 detached predicted tokens
  第二遍前向计算 local AR loss 和 local auxiliary loss
```

首轮建议：

```text
warmup_steps = 10% of total steps
predicted_global_fraction: 0.00 -> 0.30 linearly
predicted global strategy: greedy, stop-gradient
```

第一版不使用高温随机 global token，以免过早引入过强噪声。

## 8. 损失函数

最终目标为：

```text
L_total = lambda_ar     * L_AR
        + lambda_global * L_global_code
        + lambda_local  * L_local_code
```

其中：

```text
L_AR:
  完整 HDS 所有非 PAD target token 的 next-token CE

L_global_code:
  Qg 个 stage-specific global auxiliary heads 的平均分类 CE

L_local_code:
  所有 site、所有 local stage 的平均辅助分类 CE
```

三个 loss 分别在自己的有效位置求平均，避免 local slot 数量随晶胞大小增长而完全主导总损失。

首轮建议：

```text
lambda_ar     = 1.0
lambda_global = 0.5
lambda_local  = 0.5
```

同时进行以下权重消融：

```text
A: 1.0 / 0.0 / 0.0   AR-only baseline
B: 1.0 / 0.5 / 0.5   recommended auxiliary setting
C: 1.0 / 1.0 / 1.0   stronger code supervision
```

如果 `L_global_code` 和 `L_local_code` 直接复用共享 LM logits，它们在数学上只是 code-position reweighting。主方案使用独立的动态 stage-specific auxiliary heads，才构成真正的辅助分类任务。

codebook-aware KL、embedding cosine 和 residual-sum loss 暂不进入第一轮主目标，保留为后续消融。

## 9. CSP 与 De Novo 的统一训练信号

同一个完整 HDS AR 样本天然覆盖 CSP 和 de novo 两种 prefix 的训练信号：

```text
de novo inference prefix: <BOS>
CSP inference prefix:     known FORMULA block
```

因此第一版不增加 task mixing probability，也不复制两份数据。当前 E2 推理只开放 CSP prefix；de novo prefix 留到 E4。训练日志分别统计：

```text
full AR loss
formula token accuracy / exact match
global AR accuracy
global auxiliary accuracy
local AR accuracy
local auxiliary accuracy
true-global local accuracy
predicted-global local accuracy
```

## 10. De Novo Formula 约束解码

从 BOS 生成 composition 时必须使用语法约束，不能依赖生成结束后的任意修补。

允许的状态转换为：

```text
<FORMULA>
-> <ATOM_X>
-> <COUNT_n>
-> next <ATOM_Y> or </FORMULA>
```

约束包括：

```text
元素不得重复
元素必须按原子序数递增
COUNT 必须为正整数
总原子数不得超过 max_num_sites
至少包含一个元素
只允许在完整 ATOM/COUNT pair 后生成 </FORMULA>
```

可选的化学约束，例如电荷中性、元素组合过滤和训练分布先验，应放在独立筛选层中，不能与基本 HDS 语法校验混为一谈。

## 11. Primitive Cell 定义（当前暂不实施）

> 当前 E2 实现阶段暂不处理 primitive/conventional cell 的额外统一或重构；本节保留为后续数据治理说明。

以下字段是后续统一目标；当前 E2 代码与四份配置不新增或强制该字段：

```yaml
cell_representation: primitive
```

当前 E2 的完整 composition 指**输入 CIF 当前晶胞的完整整数原子计数**，不在本轮额外转换 primitive/conventional cell。未来启用上述字段后，才固定解释为标准化 primitive cell。

训练集、direct HDS、predicted HDS、扩散数据和评估数据必须使用同一个 cell representation。不能将 primitive cell 和 conventional cell 的 composition 混入同一次实验，否则同一物理结构可能对应：

```text
GaTe
Ga2Te2
Ga4Te4
```

等不同 full-cell 标签。

若后续切换到 conventional cell，必须重新生成全部 direct HDS、训练数据和扩散条件，不能只修改配置文件。

## 12. 同元素 Site Canonicalization

训练 direct HDS 时，site 按以下规则确定性排序：

```text
atomic number
-> local-code tuple lexicographic order
-> original site index
```

预测完成后，predicted local-code tuples 使用相同规则重新排序并保存 permutation。

这一步只解决等价原子顺序歧义，不会强迫同元素 site 使用相同 local code。

## 13. 多候选生成

### 13.1 CSP

对于一个已知 composition：

```text
1 composition
-> K global/local HDS candidates
-> each HDS produces R diffusion samples
```

当前可继续使用：

```text
K = 5 HDS candidates
R = 4 diffusion samples per HDS
```

即每个 CSP 样本得到 20 个晶体候选。

### 13.2 De novo（E4 后续）

计划中的 de novo generation 使用树状采样；当前 E2 不执行该流程：

```text
<BOS>
-> M composition candidates
-> K HDS candidates per composition
-> R diffusion samples per HDS
```

其中 `M`、`K`、`R` 独立设置。第一轮不建议同时设得过大，应先分别评估 composition 多样性、HDS 多样性和扩散采样收益。

正式候选选择不得读取测试结构或 direct HDS。

## 14. 建议删除或停止作为主线的内容

按照当前简化方案，以下内容不进入第一版主训练：

- Attn-RVQ 内的 formula-conditioned global prior。
- 独立 composition generator。
- 将 HDS code 预测完全拆成独立的 masked global/local 模型。
- Local Pretrain 与 Hierarchical Fine-tune 两次独立训练任务。
- SG/CS/coord HDS token。
- alpha token。
- 无效 HDS 的首元素补齐和静默截断。
- codebook-aware loss 作为默认必选项。

这些组件可以保留在历史实验目录或作为后续消融，但不应与简化主线混在同一个配置中。

## 15. 必须保留的内容

- 输入 CIF 当前晶胞的完整整数 composition（primitive/conventional 统一留待后续）。
- `NUM_SITES`、formula、site 数量和 atom skeleton 的单一来源。
- 完整 canonical HDS 的 causal next-token 主任务。
- 显式 global/local stage marker。
- stage 数量动态读取 Attn-RVQ 配置。
- 动态 global/local stage-specific auxiliary heads。
- 受限 AR 解码，只在不确定位置进行采样。
- true/predicted global 混合 curriculum。
- 同元素 site canonicalization。
- MP ID 对齐。
- direct/predicted/oracle HDS 来源标记。
- 多 HDS 候选生成。
- Attn-RVQ 聚合模式 provenance 校验。

## 16. Attn-RVQ 连续条件边界

HDS 只保存 global/local code indices，不保存样本级 `alpha_q`。

因此训练和评估必须记录：

```text
attn_rvq_attention_output_mode
attn_rvq_index_reconstructible_from_hds
vq_checkpoint
vq_config
```

如果某个 alpha-weighted Attn-RVQ 的连续输出不能仅根据 indices 精确恢复，则 predicted HDS 与 direct HDS 都只能重建 code-index 对应的近似连续条件。该问题属于 Attn-RVQ 到扩散条件的接口问题，不能通过 HDS 语法本身解决。

## 17. 训练与评估实验

### E0：保留当前两阶段模型作为参考

```text
Local Masked Pretrain
-> Hierarchical Fine-tune
```

只保存已有结果，不重新训练，当前不实施。

### E1：AR-only 基线

```text
full canonical HDS
loss = L_AR
constrained decoding
```

目的：建立同一 HDS、同一模型和同一 curriculum 下的纯 AR 对照。该实验已经完成。

### E2：AR + code auxiliary loss

```text
L = L_AR + 0.5 L_global_code + 0.5 L_local_code
dynamic stage-specific auxiliary heads
global self-conditioned curriculum
```

目的：验证辅助分类是否提高 exact code match 和 VQ cosine，同时保留完整序列生成能力。该项是当前代码实现与训练主线。

形成能 10 分箱只作为独立条件实验分支：

```text
[-4.5, -4.0) -> 1
[-4.0, -3.5) -> 2
...
[0.0, 0.5)   -> 10
```

超出 `[-4.5, 0.5)` 的样本直接报告错误，不截断、不钳制、不删除。无条件 E2 主实验不启用该属性分支，二者使用不同 cache 标识。

### E3：辅助 head 融合与强权重消融

```text
decode beta_aux: 0.0 vs 1.0
loss weights: 0.5/0.5 vs 1.0/1.0
```

目的：区分训练期辅助监督和推理期 logits 融合的作用。当前不实施；E2 主设置固定使用 `beta_aux=1.0`。

### E4：CSP 与 de novo 多候选评估

```text
CSP: 5 HDS x 4 diffusion
de novo: M formula x K HDS x R diffusion
```

该项仅保留为后续实验表，当前不启动正式多候选扩散评估。

## 18. 分析评估指标

以下指标只用于分析模型行为和定位问题，不作为自动验收门槛。当前实现与实验优先关注 18.1 HDS schema 和 18.2 CSP code prediction。

### 18.1 HDS schema

```text
valid HDS rate = 1.0
formula/site composition consistency = 1.0
site_count_match_rate = 1.0
无 UNK、内部 PAD 和非法 stage
```

### 18.2 CSP code prediction

```text
global_code_match_rate
local_code_match_rate
global_vq_cosine
local_vq_cosine
hds_vq_cosine
candidate top-K coverage
```

### 18.3 De novo composition

```text
formula syntax validity
composition validity
unique composition rate
novel composition rate
element/count distribution
NUM_SITES distribution
```

### 18.4 下游扩散

```text
match rate
RMS distance
validity
coverage
oracle-direct 与 predicted-HDS gap
```

## 19. 预期代码修改位置

本方案不新增生产 Python 脚本，而是在原有入口和模型上扩展：

| 文件 | E2 职责 |
| --- | --- |
| `scripts/tokenizer.py` | 统一 HDS compiler、动态 stage schema、canonical 编译和严格校验 |
| `scripts/cif2hds.py` | 从 Attn-RVQ indices 直接输出完整组分、Q-tagged canonical HDS；形成能独立分箱分支 |
| `scripts/make_prompts_from_csv.py` | 从 CIF 构造完整晶胞组分 CSP prompt，不再使用 reduced formula |
| `model/crystal_token_gpt.py` | 保留 causal backbone/shared LM head，增加动态 global/local auxiliary heads |
| `train_ar_model.py` | cache v2、完整 AR loss、辅助 code loss、self-conditioned curriculum 和分项日志 |
| `scripts/generate_hds.py` | compiler 强制 CSP 骨架、受限 code 采样、auxiliary logits 融合和 canonical 多候选 |
| `scripts/gpt_tokens_to_vq_indices.py` | 同时解析 legacy/Q-tagged code 并保留 quantizer id |
| `scripts/sample_from_hds.py` | 删除 atom type 补齐/截断，强制 canonical parser |

其中 `scripts/cif2hds.py` 生成的 legacy direct HDS 也可由 `train_ar_model.py` 的 cache v2 一次性重构，不要求重新运行 Attn-RVQ code 提取。

## 20. 当前建议

首轮实现应并行保留 E1 和完成 E2：

```text
完整 canonical HDS causal AR
+ dynamic global/local auxiliary classification heads
+ one-run global self-conditioned curriculum
+ constrained skeleton decoding
```

当前 E2 从已知 FORMULA/full-cell composition prefix 执行 CSP。辅助 head 只校准 code 位置，不取代 HDS 的完整自回归语义建模；从 BOS 开始的 de novo generation 属于 E4，当前不启用。

## 21. 缓存训练运行顺序

E2 配置默认 `use_hds_cache: true`，必须先准备缓存，再从同一配置启动训练。训练入口不会在缓存缺失时静默执行昂贵的数据准备。

```bash
python train_ar_model.py --config <E2_CONFIG.yaml> --prepare_hds_cache
python train_ar_model.py --config <E2_CONFIG.yaml>
```

形成能 10 分箱使用独立配置和独立 cache。缺失值或超出 `[-4.5, 0.5)` 的样本会在缓存准备阶段直接报错。
