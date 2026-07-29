# HDS 条件扩散改进设计（Trick Design）

本文记录当前 MP-20 HDS 条件扩散实验中三个相互关联的问题及改进方案：CFG condition dropout、Global/Local 条件融合，以及 HDS 多候选选择。本文是设计与实现依据，不把分析指标作为硬性验收条件。

## 1. CFG condition dropout

### 1.1 当前问题

原实现分别采样：

- 每个晶体一个 Global keep/drop mask；
- 每个原子一个独立 Local keep/drop mask。

对于含有 $N$ 个原子的晶体，设 `cond_drop_prob=p`，Global 和全部 Local 同时为 null 的概率为：

$$
P(\text{all-null})=p^{N+1}.
$$

当 $p=0.1$ 时，$N=2,5,10,20$ 分别对应 $10^{-3},10^{-6},10^{-11},10^{-21}$。随着原子数增加，训练几乎不可能看到完整 all-null HDS。

与此同时，完整条件被保留的概率为：

$$
P(\text{full})=(1-p)^{N+1}.
$$

例如 $N=10$ 时仅约为 0.314；至少一个 Local site 被清空的概率为 $1-0.9^{10}\approx0.651$。因此原实现主要训练的是随机缺失 Local site 的部分条件，而不是 CFG 所需的完整 conditional/all-null 两个端点。

### 1.2 与 CFG 推理的冲突

推理时模型显式构造：

- `cond_drop_prob=0.0`：完整 Global + 完整 Local；
- `cond_drop_prob=1.0`：null Global + 全部 null Local。

当前 CFG 形式为：

$$
\epsilon_{\mathrm{cfg}}
=(1-s)\epsilon_{\mathrm{null}}+s\epsilon_{\mathrm{cond}}
=\epsilon_{\mathrm{null}}+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{null}}),
$$

其中 $s$ 为 `cond_scale`。当 $s>1$ 时，训练不足的 null 分支误差会进入并影响 conditional-null 差值，且可能被 guidance 放大。因此较小 `cond_scale` 在部分失败样本上更稳定，并不说明弱条件理论上更好，也可能只是减小了错误 null endpoint 的影响。

这里的“无条件”仅表示不使用 HDS；元素类型、组分与 site 数仍由 CSP 输入提供。

### 1.3 本次实现：graph-level shared dropout

每个晶体只采样一次：

$$
m_b\sim\operatorname{Bernoulli}(1-p),
$$

然后把 $m_b$ 同时用于该晶体的 Global condition 和全部 Local conditions。这样：

| 条件状态 | 概率 |
|---|---:|
| 完整 Global + 完整 Local | $1-p$ |
| null Global + 全部 null Local | $p$ |

all-null 概率不再依赖原子数。实现通过 `node2graph` 将 graph mask 广播到每个 local/site condition，并在训练和推理入口使用相同的数据关系。

本次不实现互斥的类别 dropout。后续可独立实验：

| 类别 | 示例概率 |
|---|---:|
| Full HDS | 0.80 |
| All-null（CFG） | 0.10 |
| Global-only | 0.05 |
| Local-only | 0.05 |

其中 all-null 必须保留明确的 0.10 概率；Global-only/Local-only 是分支鲁棒性训练，不能代替 CFG all-null。

形成能等属性条件属于特定条件实验分支。启用该分支时，应进一步决定属性条件是否与 HDS 共用 CFG graph mask；当前无属性 CSP 主实验不受该接口影响。

## 2. Global/Local 融合

### 2.1 当前数据流

当前 `multiscale_film` 对 6 层 CSPNet 的默认行为是：

1. Local condition 在早期第 0、1 层之前以 site-level FiLM 注入；
2. 中间通过 CSP message passing 更新 node features；
3. Global condition 在后期第 3、4、5 层中，将 node features mean-pool 为 graph feature；
4. 对 graph feature 做 Global FiLM；
5. 把同一个 graph feature broadcast-add 回该图全部节点。

这属于按层顺序注入，不是显式的 Global/Local 联合建模。

### 2.2 主要问题及原因

1. **缺少显式交互。** 网络没有学习哪个 Global mode 应作用于哪个 site，也没有在 Global/Local 冲突时学习分支可信度。
2. **Global 是统一 graph bias。** broadcast-add 给同一晶体的所有节点相同增量，无法产生 site-dependent Global influence，且可能被 LayerNorm、消息传递或后续残差削弱。
3. **Local shortcut。** Local code 与 site 一一对应，并直接靠近坐标预测路径；Local encoder 本身也可能携带全局环境。lattice head 又对 Local-conditioned nodes 做 mean pooling，因此网络可以绕过独立 Global code。
4. **没有分支尺度控制。** 两个分支缺少独立归一化、可学习有界 gate、梯度平衡和一致性约束。重复 Global residual 可能过弱、被忽略，也可能过强而干扰 Local representation。
5. **错误 CFG dropout 加剧分支绕行。** 随机缺 site 的条件训练鼓励模型依赖更稳定的分支，而不是学习两个完整条件之间的联合关系。

坐标损失虽然对应多个原子，但代码使用 mean-reduced MSE，因此不能简单断言 Local 梯度严格是 Global 的 $N$ 倍。Local 更强主要来自 site-level 对齐、更直接的网络路径和更丰富的条件内容。

### 2.3 建议实现：双流、双向门控融合

后续在现有 `CSPNet` 内实现，不新增独立模型脚本。推荐保留三类状态：

- node stream：$h_i$，负责 site/coordinate 表示；
- local semantic：$l_i$，来自每个 site 的 Local HDS；
- explicit global token：$g$，来自 Global HDS，不再只依赖 node mean pooling。

#### A. 独立投影与归一化

$$
\hat g=\operatorname{LN}_g(W_g g),\qquad
\hat l_i=\operatorname{LN}_l(W_l l_i).
$$

Global 和 Local 使用不同的 LayerNorm 与 projection，避免两个分支的数值尺度直接竞争。

#### B. Global 到 Local/node 的门控注入

$$
a_i=\sigma\left(\operatorname{MLP}_{g\rightarrow l}
([h_i,\hat l_i,\hat g,t])\right),
$$

$$
h_i' = h_i
+\gamma_l W_l'\hat l_i
+\gamma_g a_i W_g'\hat g.
$$

$a_i$ 让同一个 Global mode 对不同 site 产生不同作用；$\gamma_g,\gamma_l$ 是有界或可学习残差 gate，建议接近零初始化，避免新条件路径在训练初期破坏原 CSP backbone。

#### C. Local/node 到 Global 的反向聚合

使用以 Global token 为 query、Local/node tokens 为 key/value 的 attention pooling：

$$
u=\operatorname{AttnPool}(q=\hat g,\,kv=\{[h_i',\hat l_i]\}_{i=1}^{N}),
$$

$$
g'=\operatorname{LN}(\hat g+\gamma_{l\rightarrow g}u).
$$

这样 Global 不只是单向广播，而能根据真实 site 集合更新，从而显式检查 Global/Local correspondence。

#### D. 分头路由后再联合

- lattice/coarse path 优先读取 $g'$ 和 attention-pooled node feature；
- coordinate/fine path 优先读取 $h_i'$ 和 $l_i$；
- 最后保留若干 joint CSP layers，让 lattice 与 coordinates 相互协调。

建议先用一个 fusion block 插入中层，而不是每层重复注入。若一个 block 有收益，再消融比较 early/middle/late placement 和 block 数量。

#### E. 训练与诊断

先修复 graph-level CFG，再训练融合模型。应同时记录：

- Global/Local gate 的均值、方差及按 diffusion timestep 的变化；
- 两个条件投影层的梯度范数；
- Full、Global-only、Local-only、mismatched-global、mismatched-local 的 MR/RMSE；
- 将某一分支置 null 后预测变化的 sensitivity。

完整模型应至少表现为：正确双分支优于或不劣于单分支，并且打乱任一有效分支都会产生可解释的性能下降。

## 3. Alpha-weighted HDS 候选选择

### 3.1 当前距离的问题

当前候选距离将 2 个 Global codes 与 $3N$ 个 Local codes 展平，计算等权 Hamming distance：

$$
D_{\mathrm{raw}}
=\frac{\text{mismatched codes}}{2+3N}.
$$

当 $N=10$ 时，Global 仅占 6.25%，Local 占 93.75%。此外，不论两个 codebook vectors 在连续空间中相近还是相远，只要 index 不同，距离都记为 1；所有 RVQ stages 也被赋予相同权重。

### 3.2 Alpha 的可用范围

Attn-RVQ 使用：

$$
r_{q+1}=r_q-\alpha_q e_{q,c_q}.
$$

精确 $\alpha_q$ 依赖 encoder residual 和 quantized vector。生成 HDS 只保留 code indices，因此不能从生成序列恢复 sample-specific alpha，也不应伪造它。

可实践的方案是在 Attn-RVQ 训练或 direct-HDS 提取期间统计：

$$
A_{b,q}=\operatorname{EMA}[\alpha_{b,q}\|e_{b,q,c}\|],
$$

或更细粒度地统计：

$$
A_{b,q,c}=\operatorname{EMA}[\alpha_{b,q}\mid c_q=c].
$$

第一版使用 branch-stage EMA；统计充分后再使用 branch-stage-code lookup。

### 3.3 分层、归一化距离

stage codebook 距离采用：

$$
\delta_{b,q}(c,c')
=\frac{1-\cos(e_{b,q,c},e_{b,q,c'})}{2}.
$$

Global 距离：

$$
d_g=\sum_q\bar w_{g,q}\delta_{g,q}.
$$

Local 距离：

$$
d_l=\frac{1}{N}\sum_{i=1}^{N}
\sum_q\bar w_{l,i,q}\delta_{l,i,q}.
$$

最终距离：

$$
D=\lambda_g d_g+\lambda_l d_l,
\qquad \lambda_g+\lambda_l=1.
$$

Global/Local 必须分别归一化后再混合，alpha weighting 本身不能解决 2 对 $3N$ 的数量失衡。Local sites 在比较前必须采用 canonical order；必要时按元素分组后进行最小代价匹配。

初始验证可比较 $(\lambda_g,\lambda_l)=(0.5,0.5),(0.4,0.6),(0.3,0.7)$。结合 correspondence 实验，$(0.35,0.65)$ 可作为偏向 Local 的实践起点，但最终应按验证集 diffusion MR/RMSE 确定。

### 3.4 MMR top-5 选择

先 oversample、canonicalize 和去重，再选择：

$$
S(H)=\eta\,\operatorname{Normalize}(\log p(H\mid prompt))
+(1-\eta)\min_{H_j\in\mathcal S}D(H,H_j).
$$

建议从 $\eta=0.7\sim0.8$ 开始。`mmr_lambda=1.0` 会完全关闭 diversity，只保留 likelihood 排序。候选不足 5 个时，应在完成上述选择后再复制 rank-0。

完整方案需要同时具备：

1. Global/Local 分支归一化；
2. alpha/effective-contribution stage 权重；
3. codebook 连续距离；
4. canonical site alignment；
5. likelihood-diversity MMR。

仅继续使用 alpha-weighted Attn-RVQ，不会自动使 HDS 候选选择具备上述性质；生成阶段必须显式加载 codebook 与 alpha 统计。
