# VLA-JEPA Latent World Model：数据、评估、审计与结论最终报告

日期：2026-07-19

实验范围：LIBERO 130 tasks，冻结 V-JEPA2 encoder 与 VLA-JEPA predictor，不训练、不微调

报告定位：可独立阅读和展示的总报告；数据采集、实验设计、正式结果、审计修正和最终结论均包含在本文中

## 1. 执行摘要

本项目将 VLA-JEPA 中的视觉编码器和 latent world-model predictor 提取、封装为可在其他项目中独立调用的 `latent_world_model`，随后在 LIBERO 上采集带双视角视频、实际执行动作和 VLA-JEPA latent action 的交互数据，对冻结模型进行了两类必须严格区分的评估：

1. **strict-causal 任务**：只使用现在或过去的视觉 latent 与对应 latent action，预测编码时尚未被看到的未来 latent。这是本项目真正关心的“过去预测未来”。
2. **原生 joint-C3 任务**：按 VLA-JEPA 训练方式联合编码连续 8 帧，以 `z0,z1,z2` 和 `g0,g1,g2` 预测同一次 joint encoding 中的 `z1,z2,z3`。这是对 checkpoint 原生目标是否学成的测试，但不是严格因果未来预测。

最终结论如下：

| 核心问题 | 最接近的完整正式实验 | 主要结果 | 结论 |
|---|---|---|---|
| 只看现在 latent 与 latent action，能否准确预测未观察的下一 latent？ | F0 strict C1，1950 windows | MSE `6.2735`；persistence MSE `4.4658` | **尚不能。直接误差较大，且比保持当前 latent 更差。** |
| 加入过去三步状态/action 是否改善下一步？ | F1 strict C3，1950 windows | MSE `6.5445`，比 F0 高 `0.2709` | **没有改善。** |
| strict 条件下 predictor 是否利用样本特定 latent action？ | F4 vs F0 | 配对 MSE 差 `-0.0001`，95% CI 跨 0 | **未观察到可检测的样本特定增益。** |
| predictor 是否学会 VLA-JEPA 原生 joint-C3 目标？ | J0 joint C3，1950 windows | MSE `3.5046`，L1 `1.3041`，cosine `0.7777` | **是，学到了稳定的 joint-representation 映射。** |
| joint-C3 下是否利用样本特定 latent action？ | J1 vs J0 | 配对 MSE 差 `+0.0000016` | **未观察到可检测差异。** |
| 当前是否已有可靠的 strict-causal 多步结论？ | 历史 F2 | H2/H3 时间排程经审计发现错位 | **没有；历史多步数值只保留用于溯源。** |

因此，当前模型能力最准确的一句话表述是：

> **冻结 predictor 能够完成 VLA-JEPA 的 joint-C3 shifted-latent 目标，但尚未证明、并且在本次完整单步实验中未能完成用户需要的 strict-causal“过去 latent + latent action → 未观察未来 latent”任务。**

这两个结论并不冲突。joint encoder 同时处理完整 8 帧，名义上的 context latent 可以包含未来帧信息；而 strict-causal latent 通过独立因果构造排除了这种可见性。两种协议的 latent 和 target 定义不同，不能把 `3.50` 与 `6.27/6.54` 当作同一表示空间内的模型改进量。

## 2. 工作目标、范围与最终产出

### 2.1 工作目标

本项目完成了以下工作：

- 从 VLA-JEPA 工程中提取、封装视觉编码和 latent prediction 模块；
- 保持上游 checkpoint 的参数、输入 shape 和 latent-action 接口，验证独立模块与源实现完全一致；
- 在五个 LIBERO suite 的全部 130 个任务上运行 VLA-JEPA policy；
- 每个任务采集 10 条 rollout，保存视频、状态、执行动作、query 时间和 latent action；
- 对 1300 条 rollout 做完整性、视频、动作、重放、latent 重算和模型 parity 审计；
- 从每个任务确定性选择 5 条 rollout，设计并运行 strict-causal、action control 和 joint-C3 正式实验；
- 对结果进行配对统计、层级 bootstrap、suite/stage/success/motion/action 分层和实现审计；
- 修正实验解释边界并补做完整 joint-C3 正式实验；
- 将代码、实验定义、摘要结果、报告与哈希 manifest 纳入 Git 版本管理。

### 2.2 明确未做的事项

- 没有重新训练或微调 V-JEPA2 encoder；
- 没有重新训练或微调 VLA-JEPA predictor；
- 没有把 latent action 替换成实际动作编码；
- 没有使用像素 decoder 重建未来帧；V-JEPA2 没有与当前 checkpoint 配套的像素 decoder，本次指标全部在 latent 空间计算；
- 没有因为 LIBERO rollout 失败而删除其动作—视觉序列；成功状态只作为描述性分层。

## 3. 模型、接口与 latent 定义

### 3.1 独立模块

当前 `latent_world_model` 是从 VLA-JEPA 项目提取的独立编码与预测模块，便于被其他工程调用。评估使用：

- V-JEPA2 ViT-L encoder，输入分辨率 256 px，tubelet size 2；
- VLA-JEPA LIBERO checkpoint `VLA-JEPA-LIBERO.pt`；
- predictor 权重前缀 `vj_predictor.*`；
- predictor 参数量 `161,647,616`；
- strict load 无 missing keys、无 unexpected keys；
- 双视角 feature 在同一样本内部拼接，最终 feature dim 为 2048。

独立模块与 VLA-JEPA 源 predictor 的 parity 测试为：

```text
context: [1, 768, 2048]
action:  [1,  24, 2048]
output:  [1, 768, 2048]
max absolute difference  = 0
mean absolute difference = 0
allclose                 = true
```

这证明模块提取本身没有改变 predictor 数值行为。

### 3.2 视觉 latent 与时间块

连续 8 帧经 tubelet size 2 后形成四个时间块：

```text
z0 = frames 0,1
z1 = frames 2,3
z2 = frames 4,5
z3 = frames 6,7
```

每个时间块包含两个相机视角融合后的 256 个 patch token，feature dim 为 2048。默认 C3 predictor 输入为：

```text
context [z0,z1,z2] : [B, 768, 2048]
action  [g0,g1,g2] : [B,  24, 2048]
target  [z1,z2,z3] : [B, 768, 2048]
```

其中每个 `g` 含 8 个 token，三个 `g` 合计 24 个 latent-action tokens。

### 3.3 上游训练目标

VLA-JEPA world-model predictor 的高层训练目标是 joint-C3 teacher forcing：

```text
一次联合编码 8 帧得到 z0,z1,z2,z3
context = [z0,z1,z2]
action  = [g0,g1,g2]
target  = [z1,z2,z3]
loss    = L1(prediction, target)
```

因此，C3 是 checkpoint 的训练内调用；C1 虽然在独立模块的 shape 上可运行，但 checkpoint 没有接受过 C1 训练，应解释为冻结模型在新接口上的迁移测试。

### 3.4 latent action 的语义

本项目所称 latent action 是 VLA-JEPA Qwen 输出中的 24 个 `<|action_i|>` hidden states。上游训练 `vj_predictor` 时输入的是这组 tokens，本项目采集、保存和重新输入的也是同一来源、同一顺序、同一 shape 的 tokens，因此训练—评估接口一致。

它们不需要是实际 7 步动作的可逆或确定性编码，才能用于测试既有 predictor。在线重复同一个 observation 时：

- 24 个 latent tokens 完全不变，最大差异为 0；
- 随机 flow/diffusion action head 产生的 action chunk 会变化，重复推理 MSE 约为 `1.0e-4` 和 `2.5e-3`。

因此二者是同一次 policy query 的共同输出，但不是一一映射。正确的评估问题是 predictor 是否利用了 latent tokens 中与当前样本相关的差异；本项目用 correct-vs-shuffled action 对照直接检验这一点。

## 4. LIBERO 数据集采集

### 4.1 任务覆盖与规模

最终数据集 `vla_jepa_libero130_v3` 覆盖五个标准 suite：

| suite | tasks | rollouts/task | rollouts | successful | success rate | mean frames | mean queries | video MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LIBERO-SPATIAL | 10 | 10 | 100 | 100 | 1.0000 | 100.1600 | 14.7000 | 8.0352 |
| LIBERO-OBJECT | 10 | 10 | 100 | 100 | 1.0000 | 132.1300 | 19.3200 | 8.9626 |
| LIBERO-GOAL | 10 | 10 | 100 | 100 | 1.0000 | 107.6100 | 15.8600 | 7.9739 |
| LIBERO-90 | 90 | 10 | 900 | 182 | 0.2022 | 356.1178 | 51.6500 | 207.8545 |
| LIBERO-10 | 10 | 10 | 100 | 98 | 0.9800 | 248.6500 | 36.0000 | 19.2547 |
| **总计** | **130** | **10** | **1300** | **580** | **0.4462** | — | — | **约 252** |

数据规模：

| 项目 | 数量 |
|---|---:|
| HDF5 rollout | 1300 |
| MP4 rollout | 1300 |
| 控制帧 | 379,361 |
| policy queries | 55,073 |
| 执行动作行 | 379,361 |
| latent-action 数值 | 2,706,948,096 |
| HDF5 总大小 | 约 57.252 GB |
| MP4 总大小 | 约 252 MB |

LIBERO-90 的成功率较低是 policy 表现而不是数据损坏。失败 rollout 仍包含真实、连续且可重放的状态—动作—视觉数据，因此保留用于表示预测评估；是否成功只作为描述性变量。

### 4.2 每条记录保存的内容

每个 HDF5 记录至少包含：

- agent view RGB 序列；
- wrist view RGB 序列；
- 机器人状态；
- 每个控制步实际执行的动作；
- policy query 对应的精确 frame index；
- 每次 query 的 latent-action tokens，shape `[N,24,2048]`，float16；
- 每次 query 对应的 unnormalized action chunk，shape `[N,7,7]`；
- suite、task、episode、语言指令、成功状态等元数据。

HDF5 与 MP4 使用同一个 `suite/task/episode` identity 配对。视频用于人工检查和展示；模型评估直接读取无有损压缩的 HDF5 RGB。

### 4.3 采集时序

每个控制步遵循以下顺序：

```text
读取当前 observation
→ 构造双视角 policy 输入
→ 每 7 个控制步执行一次 policy query
→ 保存该 query 的 24 latent tokens 与 7×7 action chunk
→ 保存当前 RGB、状态和即将执行的 action
→ env.step(action)
```

审计确认 `query_frame_index` 指向发起该次 policy query 的当前 observation，没有 `+1/-1` 偏移。采集过程产生的数据可以同时用于视频轨迹展示、执行动作复查和 latent-world-model 条件评估。

### 4.4 正式评估子集

按照后续“使用一半测试集”的要求，每个 task 确定性选择 episode `0,2,4,6,8`：

```text
130 tasks × 5 rollouts = 650 rollouts
650 rollouts × early/middle/late = 1950 windows
```

| suite | formal rollouts | successful | success rate | mean frames | mean queries | windows |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-SPATIAL | 50 | 50 | 1.0000 | 100.1400 | 14.7400 | 150 |
| LIBERO-OBJECT | 50 | 50 | 1.0000 | 131.4800 | 19.2000 | 150 |
| LIBERO-GOAL | 50 | 50 | 1.0000 | 107.8400 | 15.8600 | 150 |
| LIBERO-90 | 450 | 96 | 0.2133 | 354.0400 | 51.3400 | 1350 |
| LIBERO-10 | 50 | 50 | 1.0000 | 245.4200 | 35.5200 | 150 |
| **总计** | **650** | **296** | **0.4554** | — | — | **1950** |

第一次 strict 正式实验和第二次 joint-C3 正式实验使用完全相同的 1950 个窗口，使不同协议的覆盖范围一致；但由于 latent 构造不同，两者的绝对 MSE 仍不能作纯 predictor 的同分布配对比较。

## 5. 数据与实现全链路验证

### 5.1 完整性与数值检查

| 审计项 | 结果 |
|---|---:|
| 唯一 rollout identity | 1300/1300 |
| 重复记录 | 0 |
| 缺失 HDF5/MP4 配对 | 0 |
| 孤立视频 | 0 |
| action chunk 与逐帧执行 action 不匹配 | 0/379,361 |
| 连续动作最大误差 | 0 |
| gripper 动作不匹配 | 0 |
| latent NaN/Inf | 0/2,706,948,096 values |
| 全零 latent query | 0 |
| MP4/HDF5 帧数不匹配 | 0/1300 |

在每条视频抽样 early/middle/late 共 3900 帧时，MP4 与 HDF5 RGB 的像素 MAE 平均为 `1.84`、最大为 `2.36`，符合 MP4 有损压缩预期；正式模型输入使用 HDF5 RGB，不受该压缩误差影响。

### 5.2 LIBERO 动作重放

从五个 suite 同时抽取成功和失败样本，在 LIBERO 环境重放 990 个动作：

- HDF5 中两个相机画面与 LIBERO 重渲染逐像素一致；
- robot state 最大绝对误差约 `1.2e-7`；
- 成功/失败 outcome 的发生时刻一致；
- task 语言指令一致。

这排除了主要的 observation-action 时序保存错误和任务身份错配。

### 5.3 latent action 在线复算

使用原始 VLA-JEPA policy server 对已保存 observation 重新计算 latent action：

- query 0：五个 suite 各一条，5/5 与 HDF5 逐元素一致；
- middle query：五个 suite 各一条，5/5 与 HDF5 逐元素一致；
- fresh tokens 与 HDF5 tokens 的最大绝对误差为 0。

因此用于 world model 的 latent-action token 没有在采集、序列化或读取过程中发生变化。

### 5.4 predictor 提取一致性

独立 predictor 与 VLA-JEPA 源实现的同输入输出比较中，max/mean absolute difference 均为 0。第一次 strict 误差不能归因于 checkpoint 漏载、key 映射或模块提取错误。

## 6. 评估问题、实验漏斗与统计方法

### 6.1 被控制的问题维度

实验没有对所有变量做完全排列组合，而是采用漏斗式设计保留最有信息量的方向：

- 单帧状态预测单步未来：C1→H1；
- 多帧历史预测单步未来：C3→H1；
- 单帧状态自回归预测多步：C1→H3；
- 正确、置零和同任务错配 latent action；
- 双视角、仅 agent view、仅 wrist view；
- strict-causal 与 original joint 表示协议；
- suite、阶段、成功状态、latent motion、action scale 分层。

### 6.2 阶段 0：最小验证

在正式大规模计算前验证：

- 数据路径和 index 可读；
- encoder、predictor checkpoint 可加载；
- shape、dtype 和数值 finite；
- 分片、断点续跑和结果合并；
- source predictor 与独立 predictor parity；
- 最小真实 GPU forward。

### 6.3 阶段 1：390-window 筛选

每个 task 取一条 rollout，每条 early/middle/late，共 390 windows，运行 S0–S9：

| 条件 | 设计 | MSE | persistence ratio | token cosine |
|---|---|---:|---:|---:|
| S0 | strict C1→H1，正确 action | 6.2601 | 1.4235 | 0.5700 |
| S1 | strict C3→H1，正确 action | 6.5248 | 1.4843 | 0.5596 |
| S2 | strict C1→AR-H3，最终位置 | 6.5351 | 1.1716 | 0.5524 |
| S3 | strict C3→AR-H3，历史实现 | 5.5962 | 1.0013 | 0.6198 |
| S4 | strict C1，zero action | 6.2102 | 1.4123 | 0.5734 |
| S5 | strict C1，same-task shuffle | 6.2607 | 1.4236 | 0.5700 |
| S6 | strict C1，offset-next action | 6.2604 | 1.4235 | 0.5700 |
| S7 | strict C1，只复制 agent view | 7.0369 | 2.7752 | 0.5444 |
| S8 | strict C1，只复制 wrist view | 5.5576 | 0.9049 | 0.5977 |
| S9 | original joint C1→strict target | 5.1309 | 0.5391 | 0.6537 |

筛选显示 C3 历史、action 替换和多视角/编码协议是最值得正式确认的方向。补充 X0 使用 original joint C3→H1、390 windows，得到 MSE `3.3299`；相对 strict S1 配对下降 `3.1948`，95% CI `[-3.2419,-3.1498]`。这一结果随后促使全链路审计和完整 joint-C3 实验。

S2/S3 的历史多步实现后来发现时间错位，因此不用于最终多步能力判断；这里保留其数值只是为了完整记录实验漏斗。

### 6.4 阶段 2：两组完整正式实验

第一次正式实验：

```text
1950 windows × F0–F5 = 11,700 condition rows
```

第二次正式实验：

```text
1950 windows × J0–J2 = 5,850 condition rows
```

所有正式 result key 唯一，主要数值 finite，runner 无未处理错误。

### 6.5 指标

主指标是 prediction 与真实 target latent 的直接误差：

- MSE：主报告指标，越小越好；
- L1：与上游训练 loss 直接对应；
- RMSE：与 latent 原值同尺度；
- normalized MSE：用 target 能量归一化；
- token cosine：预测 token 与对应目标 token 的方向一致性；
- delta cosine：预测变化量与真实变化量的方向一致性；
- retrieval top-1/top-5：预测能否从同批候选中找回其目标；
- prediction/target variance：检查退化输出；
- persistence MSE 与 ratio：辅助判断 predictor 是否超过“未来不变”的简单参照。

`persistence` 定义为：

```text
预测未来 latent = 当前 latent
```

它不替代直接 MSE，也不是模型任务的一部分。报告它的原因是给绝对误差一个当前 latent 动态尺度下的参照：若一个复杂 predictor 的误差高于直接复制当前状态，就没有展示相对于最简单动态假设的预测增益。

### 6.6 统计口径

- 所有条件差异在完全相同的 window 上配对；
- 95% CI 使用 task→rollout 层级 bootstrap，1000 次；
- 同一 episode 的 early/middle/late 作为同一 cluster 保留；
- 只对预先注册的主要比较使用 Holm 多重比较校正；
- 分层和相关性分析是描述性结果，不据此宣称因果关系。

## 7. 第一次正式实验：strict-causal 与控制条件

### 7.1 条件定义

| 条件 | 视觉输入 | latent action | 目标 | 主要目的 |
|---|---|---|---|---|
| F0 | strict causal，双视角，C1 当前 `z2` | 当前窗口正确 `g2` | 当前窗口 strict future `z3` | 严格因果单状态单步主结果 |
| F1 | strict causal，双视角，C3 `z0,z1,z2` | 当前窗口正确 `g0,g1,g2` | 当前窗口 strict future `z3` | 过去多帧是否改善单步预测 |
| F2 | strict causal，C1，自回归 H3 | 历史跨 query action 排程 | strict 连续 future latent | 多步探索；审计后仅 H1 有效 |
| F3 | 与 F0 相同 | 全零 token group | 与 F0 相同 | 是否依赖正常条件激活 |
| F4 | 与 F0 相同 | 同 task、同 stage、其他 episode 的 action | 与 F0 相同 | 是否利用样本特定 action 信息 |
| F5 | original joint，双视角，C1 | 当前窗口正确 `g2` | 与 F0 相同的 strict future `z3` | 非因果联合编码泄漏对照 |

`strict causal` 为每个 latent block 独立构造不包含未来帧的 clip，并在 episode 起点左填充。`original joint` 将连续 8 帧一起编码；V-JEPA2 的非因果时空注意力允许早期 token 受后续帧影响。

F4 与 F0 只改变 action，视觉、目标、stage 和 horizon 全部相同。F5 则只用于量化非因果联合表示让任务变容易的程度，不是原生 joint-C3 目标，也不能作为严格预测结果。

### 7.2 1950-window 主结果

| 条件 | n | MSE | 95% CI | median [Q25,Q75] | L1 | RMSE | normalized MSE | token cosine | delta cosine | persistence MSE/ratio |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| F0 strict C1 | 1950 | **6.2735** | [6.2364,6.3104] | 6.2424 [6.0553,6.4608] | 1.7532 | 2.5039 | 0.7133 | 0.5694 | 0.4120 | 4.4658 / 1.4282 |
| F1 strict C3 | 1950 | **6.5445** | [6.5050,6.5857] | 6.5320 [6.3145,6.7607] | 1.7683 | 2.5574 | 0.7442 | 0.5585 | 0.4058 | 4.4658 / 1.4909 |
| F2 historical H3 | 1950 | 6.5476 | [6.5209,6.5752] | 6.5481 [6.3900,6.6984] | — | — | 0.7436 | 0.5519 | 0.4486 | — / 1.1768 |
| F3 zero action | 1950 | 6.2227 | [6.1846,6.2593] | 6.1872 [6.0004,6.4094] | 1.7488 | 2.4937 | 0.7075 | 0.5728 | 0.4133 | 4.4658 / 1.4168 |
| F4 shuffled action | 1950 | 6.2734 | [6.2402,6.3117] | 6.2442 [6.0525,6.4619] | 1.7532 | 2.5039 | 0.7133 | 0.5694 | 0.4120 | 4.4658 / 1.4282 |
| F5 joint C1→strict | 1950 | 5.1333 | [5.0969,5.1719] | 5.0927 [4.9263,5.3077] | 1.5869 | 2.2647 | 0.5835 | 0.6539 | 0.6900 | 9.5341 / 0.5385 |

F2 数值对应历史最终 horizon，并且 H2/H3 在审计后被确认时间不严格对齐；不能作为严格多步预测结论。F5 的 persistence 使用 joint context，因此其基线数值与 F0/F1 不同。

retrieval 结果方面，F0 top-1 为 `0.0010`、top-5 为 `0.0026`；随机参考分别为 `1/1950≈0.00051` 和 `5/1950≈0.00256`，说明 strict 预测的实例级辨识接近随机。

### 7.3 预注册配对比较

| 比较（left−right，负值表示 left 更好） | mean diff | 95% CI | median diff | left better | paired d | Holm p |
|---|---:|---|---:|---:|---:|---:|
| F0 − persistence | **+1.8078** | [1.7407,1.8746] | 1.7662 | 0.0026 | 2.6901 | <0.001 |
| F1 − F0 | **+0.2709** | [0.2531,0.2888] | 0.2690 | 0.0451 | 1.7081 | <0.001 |
| F3 − F0 | **−0.0509** | [-0.0552,-0.0463] | -0.0501 | 0.8513 | -1.0113 | <0.001 |
| F4 − F0 | **−0.0001** | [-0.0012,0.0009] | -0.0003 | 0.5077 | -0.0076 | 0.846 |
| F5 − F0 | **−1.1402** | [-1.1519,-1.1274] | -1.1386 | 1.0000 | -8.3319 | <0.001 |

解释：

- F0 的 MSE 比 persistence 高 `1.8078`，即约为 persistence 的 `1.428` 倍；当前冻结模型没有在 strict C1 上超过“未来不变”。
- F1 比 F0 更差，说明增加历史 C3 没有弥补 strict 表示偏移，也没有改善下一步预测。
- F3 的下降只有约 `0.051 MSE`，虽然样本多使其统计稳定，但效果小，而且 zero token 是远离 Qwen hidden-state 分布的异常输入，不能解释为“模型正确使用动作”。
- F4 与 F0 无可检测差异，是“样本特定 latent action 未带来增益”的直接证据。
- F5 明显更低，但它使用可接触未来的 joint context；这证明信息泄漏能降低误差，不证明 causal 预测成功。

### 7.4 suite 分层

下表把 strict 主条件、历史上下文、泄漏对照和完整 joint-C3 主结果并列，便于展示不同 suite 上的总体一致性：

| suite | windows | F0 strict C1 | F1 strict C3 | F5 joint C1→strict | J0 joint C3→joint |
|---|---:|---:|---:|---:|---:|
| LIBERO-10 | 150 | 6.0440 | 6.2720 | 4.9172 | 3.4016 |
| LIBERO-90 | 1350 | 6.3210 | 6.5639 | 5.1924 | 3.6130 |
| LIBERO-GOAL | 150 | 6.2221 | 6.4980 | 5.0377 | 3.2583 |
| LIBERO-OBJECT | 150 | 6.2541 | 6.7504 | 5.0592 | 3.1332 |
| LIBERO-SPATIAL | 150 | 6.1468 | 6.4829 | 4.9876 | 3.2498 |

F0 在各 suite 为 `6.04–6.32`，strict 失败不是单一 suite 造成。需要注意 checkpoint 的 `libero_all` 训练 mix 包含 SPATIAL、OBJECT、GOAL 和 LIBERO-10，不包含 LIBERO-90；而正式数据的 1950 windows 中有 1350 个来自 LIBERO-90。因此总体平均主要反映 LIBERO-90，正式汇报必须同时保留 suite 分层。

### 7.5 轨迹阶段与成功状态

| 分层 | F0 strict C1 | F1 strict C3 | F5 joint C1→strict | J0 joint C3→joint |
|---|---:|---:|---:|---:|
| early | 6.1618 | 6.4560 | 5.0248 | 3.4311 |
| middle | 6.3005 | 6.5365 | 5.1511 | 3.5356 |
| late | 6.3583 | 6.6410 | 5.2240 | 3.5472 |
| successful rollout | 6.1536 | 6.4562 | 4.9949 | 3.3350 |
| failed rollout | 6.3739 | 6.6183 | 5.2490 | 3.6464 |

后期和失败 rollout 的平均误差略高，但这些差异混合了 suite、任务长度、状态分布和 policy 性能，属于描述性现象，不证明成功本身导致更低误差。

### 7.6 motion、action scale 与误差

F0 按真实 target latent 变化量四分位：

| motion quartile | windows | target Δ RMS | F0 MSE |
|---|---:|---:|---:|
| Q1 | 488 | 1.9420 | 6.3744 |
| Q2 | 487 | 2.0731 | 6.2608 |
| Q3 | 487 | 2.1542 | 6.2312 |
| Q4 | 488 | 2.2678 | 6.2277 |

F0 按 action scale 四分位：

| action quartile | windows | mean action scale | F0 MSE |
|---|---:|---:|---:|
| Q1 | 488 | 0.1334 | 6.3086 |
| Q2 | 487 | 0.2696 | 6.2902 |
| Q3 | 487 | 0.3469 | 6.2899 |
| Q4 | 488 | 0.4475 | 6.2056 |

误差没有随 latent motion 或 action scale 单调增加，说明 strict 高误差不能简单归因于“动作太大”或“未来变化太快”。F0 的描述性相关性为：

| 变量 | Pearson r | Spearman ρ |
|---|---:|---:|
| target delta RMS | -0.1729 | -0.1683 |
| action norm | -0.1253 | -0.1436 |
| query frame index | +0.3946 | +0.3776 |
| target variance | **+0.8442** | **+0.8154** |

target variance 与 MSE 的相关最强，表明 latent 本身的尺度/方差对绝对 MSE 有显著影响；这也是同时报告 normalized MSE 和 cosine 的原因。

### 7.7 历史多步结果及其失效边界

历史 F2 曾得到：

| nominal horizon | MSE |
|---|---:|
| H1 | 6.2735 |
| H2 | 5.9020 |
| H3 | 6.5476 |

但 policy 每 7 个控制帧 query 一次，而 encoder tubelet stride 为 2。历史排程从当前 `z2(q+4,q+5)` 开始：

```text
g2(current query) → target (q+6,q+7)       # 对齐
g0(next query q+7) → target (q+8,q+9)      # 相差一帧
g1(next query q+7) → target (q+10,q+11)    # 相差一帧
```

因此只有 H1 严格对齐，H2/H3 不能回答正式多步问题。runner 已新增同一 query 内 `z0 --g0→ z1 --g1→ z2 --g2→ z3` 的保护性排程，并默认拒绝无法严格构造的 C3 跨-query H3。要真正评估当前窗口之外的 causal H3，需要按每 2 个控制帧重新采集条件 token，或直接使用时间对齐的实际 action 序列。

## 8. 审计如何修正实验解释

### 8.1 哪些 strict 结论保持不变

数据、时序、latent token、checkpoint 加载和独立模块 parity 均已验证，因此：

- F0 strict C1 的 `MSE=6.2735` 是有效的冻结 checkpoint 迁移结果；
- F1 strict C3 的 `MSE=6.5445` 是有效的严格过去上下文结果；
- F0/F1 没有超过 persistence 的结论不变；
- F4 与 F0 无差异的 action-specificity 结论不变。

这些结果直接回答用户真正要求的 strict-causal 任务。

### 8.2 哪个旧说法需要收回

第一次实验的 F0/F1/F5 没有一个完整匹配上游训练协议：

| 条件 | context | context encoder | target encoder | 是否匹配训练 |
|---|---|---|---|---|
| F0 | C1 | strict causal | strict causal | 否：C1 且 encoder 分布变化 |
| F1 | C3 | strict causal | strict causal | 否：encoder 分布变化 |
| F5 | C1 | original joint | strict causal | 否：C1 且 target 分布变化 |

所以不能从第一次实验推出“predictor 连自己的原生训练目标也没有学会”。审计在每个 task 取一个窗口、共 130 windows 补测：

| 协议 | MSE | L1 | cosine |
|---|---:|---:|---:|
| joint C3→same-joint shifted target | 3.4357 | 1.2879 | 0.7808 |
| strict C3→strict target | 6.3488 | 1.7505 | 0.5667 |

joint 与 strict context latent 的 MSE 为 `8.2880`，target latent MSE 为 `5.5129`。这说明表示协议发生了大幅变化；随后完整 1950-window joint-C3 实验确认了该趋势。

审计后的准确修正是：

> 旧协议确实不能衡量 checkpoint 的原生训练内性能，因此“模型整体没有学成”是过强结论；但它仍能衡量用户关心的 strict-causal transfer，而该方向的负面结果没有被推翻。

### 8.3 correct、zero 和 shuffled action 的含义

130-window joint-C3 诊断得到：

| action 条件 | target MSE |
|---|---:|
| correct | 3.4357 |
| zero | 4.1229 |
| cross-task shuffled | 3.4356 |

correct 与 shuffled tokens 确实不同：相对 RMS 差约 `25.2%`，平均 token cosine 约 `0.823`；但两者 prediction 的 MSE 只有 `1.12e-4`，target MSE 平均差约 `2.0e-5`。

zero token 完全离开正常 Qwen hidden-state 分布，所以 zero 变差只说明 predictor 依赖“正常范围的条件激活”。只有 correct-vs-shuffled 才检验样本特定信息，而 strict 和 joint 两组完整实验都未观察到相应增益。

## 9. 上游多视角 batch 融合审计

这是审计中最重要的上游训练实现问题。它不改变本项目数据正确性的结论，但限制了 released predictor checkpoint 的能力上限和解释范围。

### 9.1 错误来源

上游训练输入最初为：

```text
[B,V,T,C,H,W]
```

展平后是 sample-major：

```text
b0v0,b0v1,b1v0,b1v1,...,b31v0,b31v1
```

源码随后执行：

```python
torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
```

`torch.chunk(..., 2, dim=0)` 并不会恢复 view 维，而是把前 32 行和后 32 行配对。在每设备 `B=32,V=2` 时，实际映射为：

| predictor row | 实际视觉 feature | latent-action row |
|---:|---|---:|
| 0 | sample 0 view 0 + sample 16 view 0 | sample 0 |
| 1 | sample 0 view 1 + sample 16 view 1 | sample 1 |
| 2 | sample 1 view 0 + sample 17 view 0 | sample 2 |
| 3 | sample 1 view 1 + sample 17 view 1 | sample 3 |

后续行以同样方式错位。这造成：

- 两个不同样本被拼成一行；
- 通常拼接的是同一种视角而不是同一样本的两个视角；
- latent action 行没有同步按这个错误关系重排；
- context 与 target 都基于 batch-dependent 的混合视觉身份；
- 同一个样本的输入会随 batch 同伴和排列变化。

只有 `B=1` 时两个 chunk 恰好是 `b0v0` 和 `b0v1`，该写法才偶然正确。released checkpoint 的每设备 batch size 为 32，因此该问题进入了 predictor 的训练目标。

正确的 sample 内融合应为：

```python
video_embeddings = (
    video_embeddings.reshape(B, V, N, D)
    .permute(0, 2, 1, 3)
    .reshape(B, N, V * D)
)
```

### 9.2 数值复现

在 32 个窗口上精确复现两种融合：

| 融合方式 | MSE | L1 | cosine |
|---|---:|---:|---:|
| 正确 sample 内双视角 | 3.3022 | 1.2601 | 0.7897 |
| legacy 错误融合 | 3.2221 | 1.2398 | 0.7950 |
| legacy 错误融合 + shuffled action | 3.2223 | 1.2398 | 0.7949 |

正确与 legacy 输入 latent 的 MSE 为 `10.9984`。legacy loss 略低不表示它物理上更正确，只说明 checkpoint 更接近自己训练时见过的错误输入/target 分布；该 target 混合不同 episode，不能解释为一条真实轨迹的未来预测。

### 9.3 对本报告的影响

必须区分：

| 评估方式 | 能回答什么 | 不能回答什么 |
|---|---|---|
| 正确 sample 内双视角 | checkpoint 在真实单 episode 输入上的可用迁移性能 | 不是严格的训练分布内复现 |
| 精确 legacy batch-32 融合 | checkpoint 是否接近其实际错误训练目标 | 不能代表单条物理轨迹预测，结果依赖 batch 同伴 |

本报告的 strict 与完整 joint-C3 正式实验均使用独立模块中正确、确定性的 sample 内双视角融合，因为实际使用 latent world model 时必须保持一个样本的物理身份。第二次 joint-C3 实验复现的是 intended protocol，不是把 legacy batch 错位扩大到 1950 windows。

当前约束不允许重新训练，因此无法从推理侧消除已经写入 checkpoint 的训练历史。这个问题应被表述为 released predictor 的上游训练限制，而不是本项目评估数据或模块提取错误。

### 9.4 已加入的修复与保护

本项目没有修改既有 rollout 或伪造新 checkpoint，而是加入：

1. 全量 collection/action/video 审计器；
2. LIBERO 动作重放和逐帧状态/图像验证器；
3. 在线 latent-action 复算与重复推理验证器；
4. 训练匹配 joint 与 strict 协议对照器；
5. batch-32 legacy 双视角融合复现器；
6. 独立模块正确多视角融合函数与回归测试；
7. runner 的 Git provenance 路径修复；
8. 同-query C1-H3 时间排程与 legacy misalignment 拒绝保护。

修正后的 F2 已使用真实 encoder/checkpoint 完成单窗口 GPU smoke，`processed=1`、`errors=[]`，输出记录 `h3_schedule=within_query`。这验证了修复后执行路径可运行，但不构成新的完整多步实验。

## 10. 第二次正式实验：完整 joint-C3 teacher forcing

### 10.1 实验目的和协议

第二次实验补齐第一次实验没有覆盖的 checkpoint 原生目标：

```text
同一次 joint 8-frame encoding → z0,z1,z2,z3
context = [z0,z1,z2]
action  = [g0,g1,g2]，共 24 tokens
target  = [z1,z2,z3]，来自同一次 joint encoding
```

J0/J1/J2 使用与第一次正式实验完全相同的 1950 windows：

| 条件 | context/target | latent action | 目的 |
|---|---|---|---|
| J0 | joint C3→same-joint shifted target | 当前窗口正确 tokens | 原生目标主结果 |
| J1 | 与 J0 完全相同 | 同 task、同 stage、其他 episode tokens | 样本特定 action 对照 |
| J2 | 与 J0 完全相同 | 全零 tokens | 正常 token 激活对照 |

J* 输出的 H1/H2/H3 是同一次 teacher-forcing forward 中三个 transition 位置，不是把预测输出递归送回模型的自回归 rollout。

### 10.2 完整结果

| 条件 | n | MSE | 95% CI | L1 | RMSE | normalized MSE | token cosine | persistence ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| J0 correct | 1950 | **3.5046** | [3.4604,3.5450] | **1.3041** | 1.8707 | 0.3943 | **0.7777** | 0.3656 |
| J1 shuffled | 1950 | 3.5046 | [3.4629,3.5434] | 1.3041 | 1.8707 | 0.3943 | 0.7777 | 0.3656 |
| J2 zero | 1950 | 4.1833 | [4.1495,4.2215] | 1.4358 | 2.0444 | 0.4708 | 0.7292 | 0.4365 |

上游直接优化 L1，因此 J0 的 `L1=1.3041` 是最直接的训练目标复现指标。J0 retrieval top-1/top-5 为 `0.9995/1.0000`，说明在 joint 表示下几乎总能匹配同一次编码产生的目标；但由于 context 与 target 高度重叠且含未来信息，这个高 retrieval 是 joint 表示身份保持的证据，不是严格因果能力证明。

### 10.3 三个 transition 位置

| 位置 | 映射 | MSE | 95% CI | L1 | RMSE | token cosine |
|---|---|---:|---|---:|---:|---:|
| 1 | `z0,g0→z1` | 3.6799 | [3.6384,3.7171] | 1.3326 | 1.9170 | 0.7650 |
| 2 | `z1,g1→z2` | 3.5033 | [3.4594,3.5452] | 1.3033 | 1.8703 | 0.7806 |
| 3 | `z2,g2→z3` | 3.3306 | [3.2854,3.3743] | 1.2764 | 1.8230 | 0.7873 |

三个位置均稳定，越靠后的 transition 平均误差反而略低。因为三步是一次 teacher-forcing 调用同时预测，这不是“自回归越走越准”，也不能用来推断窗口外长期 rollout。

### 10.4 action 对照

| 比较（left−J0） | MSE 差 | 95% CI | Holm p | 解释 |
|---|---:|---|---:|---|
| J1 shuffled − J0 | +0.0000016 | [-0.0000244,+0.0000268] | 0.904 | 无可检测差异 |
| J2 zero − J0 | +0.6787 | [0.6719,0.6858] | <0.001 | 全零分布外条件显著恶化 |

因此 predictor 依赖正常 latent-token 激活，但在本测试中没有显示出对同 task/stage 其他 episode token 的敏感性。这个结论不表示 latent action 不含动作或意图信息，只表示当前 predictor 的输出没有利用到足以改变预测误差的样本特定差异。

### 10.5 分层结果

suite、stage 和 success 分层已在第 7.4、7.5 节与 strict 结果并列。joint-C3 主结果在五个 suite 上为 `3.1332–3.6130`；early/middle/late 为 `3.4311/3.5356/3.5472`；成功/失败为 `3.3350/3.6464`。这些结果说明 joint mapping 的优势不是由单一 suite 或单一阶段产生，但仍属于描述性覆盖，不推断任务难度的因果来源。

## 11. 综合解释：模型究竟学到了什么

### 11.1 得到支持的能力

1. **模块可用性得到验证。** encoder/predictor 可以独立加载与调用，source parity 为逐元素一致。
2. **数据链路可信。** 视频、动作、query index、latent action、task identity 和 checkpoint 输入均通过交叉验证。
3. **joint-C3 原生目标已学到。** 完整 130-task、1950-window J0 为 MSE `3.5046`、L1 `1.3041`、cosine `0.7777`，三个 transition 均稳定。
4. **正常 latent-token 激活对 predictor 有作用。** zero token 在 strict 和 joint 条件下均导致变化，joint 下明显恶化。

### 11.2 没有得到支持的能力

1. **strict C1 过去预测未来未成功。** F0 MSE `6.2735`，高于 persistence `4.4658`。
2. **strict C3 历史没有改善 H1。** F1 比 F0 高 `0.2709`。
3. **样本特定 latent action 增益未被观察到。** F4≈F0，J1≈J0。
4. **strict-causal 长期自回归能力尚未得到有效完整测试。** 历史 H3 时间错位，修正后只做过 smoke。

### 11.3 为什么 joint-C3 成功不等于 strict-causal 成功

存在两层未来信息可见性：

- V-JEPA2 joint encoder 用非因果时空注意力同时编码完整 8 帧，所以 `z0/z1/z2` 可受到后续帧影响；
- shifted target `[z1,z2,z3]` 中的 `z1/z2` 已经作为 block 出现在 C3 context 中。

因此 joint-C3 更接近“在包含完整窗口信息的联合表示中完成 shifted mapping”，而不是部署时只看过去预测未知未来。J0 的低误差证明 predictor 对自己的 intended representation 学得稳定，但不满足用户对严格 world model 的定义。

### 11.4 strict 结果与 joint 结果不能直接相减

strict 和 joint 使用不同的 encoder 可见性、context latent 与 target latent。审计已测得 joint/strict context latent MSE `8.2880`、target latent MSE `5.5129`，因此 `6.54→3.50` 不能全部归因于 predictor 变得更准。它首先说明 joint protocol 本身提供了更容易、更接近训练分布的表示。

### 11.5 最终能力矩阵

| 能力声明 | 支持程度 | 证据和边界 |
|---|---|---|
| 独立模块忠实复现源 predictor | 支持 | max/mean diff=0 |
| joint-C3 shifted latent mapping | 支持 | J0 全量 MSE 3.5046、L1 1.3041 |
| strict 单状态预测未观察未来 | 不支持 | F0 MSE 6.2735 > persistence 4.4658 |
| strict 多历史改善单步未来 | 不支持 | F1−F0=+0.2709 |
| predictor 使用样本特定 latent action | 未观察到 | F4≈F0；J1≈J0 |
| predictor 需要正常 token 激活 | 支持 | J2−J0=+0.6787 |
| strict 多步 rollout 能力 | 未确定 | 历史排程错位；修正后未做全量 |
| joint-C3 等价于因果 world model | 否 | joint context 可见未来，且 target/context 重叠 |
| 当前数值代表无训练 bug 的模型上限 | 否 | released checkpoint 受上游 view-fusion 错位影响 |

## 12. 局限、使用建议与后续工作

### 12.1 当前结论的边界

- 结论针对现有 released checkpoint，不外推到修复训练后的 predictor；
- strict C1 是 checkpoint 分布外接口，但正是用户要测试的可迁移能力；
- strict C3 消除了 C1 长度差异，仍未成功，因此不能只把失败归因于 C1；
- LIBERO-90 占正式窗口 69.2%，但 suite 分层显示其他 suite 方向一致；
- MSE 受 target variance 影响，已用 normalized MSE、cosine 和分层补充；
- joint-C3 结果采用正确 sample 内视角融合，属于 intended protocol，而不是 legacy batch 错位的训练目标精确复现；
- 没有像素 decoder，所以不能把 latent MSE 直接换算为人眼可见的视频质量。

### 12.2 在不重新训练条件下的合理用途

当前 predictor 可用于：

- 验证 VLA-JEPA joint-C3 表示和 checkpoint 接口；
- 做 joint representation 的条件映射或表征研究；
- 作为 strict-causal transfer 的负面基线；
- 研究 encoder 协议、action conditioning 和上游实现对表示误差的影响。

不应直接用于宣称：

- 已能在部署时只看过去准确预测未来；
- 已能依据具体 action 区分不同未来；
- 已经验证长期自回归 rollout；
- joint-C3 的低误差就是因果 dynamics 学习成功。

### 12.3 下一步优先级

如果继续保持“不训练模型”的限制：

1. 用修正后的同-query 排程完成严格对齐 C1→H3，明确区分 teacher forcing 与 autoregressive rollout；
2. 增加 cross-task、跨 stage 和语义差异更大的 action shuffle，确认 action-insensitivity 的范围；
3. 在平衡的 suite macro 统计下重复核心指标，降低 LIBERO-90 权重影响；
4. 对 legacy batch-32 融合做多 permutation 审计，量化同一窗口对 batch 同伴的敏感性；
5. 研究 strict latent 的尺度校准或只读后处理，但不能把后处理误称为 predictor 训练能力。

如果未来允许修复模型，最小根治方案是不重训 encoder：冻结 V-JEPA2 和 latent-action 生成器，修正双视角融合，并在严格因果目标上重新训练/微调 predictor；随后用完全相同的 1950 windows、persistence 和 action shuffle 协议重新评估。这一步不属于本次工作。

## 13. 可复现性、产物与版本管理

### 13.1 运行环境

```bash
conda activate VLA_JEPA
cd /home/embodied/users/jlb/proj0/latent_world_model
```

LIBERO 环境用于动作重放，GPU/CUDA 用于 encoder 和 predictor forward。数据、checkpoint 和大体积缓存保留在本地；源代码、实验配置、聚合结果、报告和哈希 manifest 进入 Git。

### 13.2 核心复现入口

完整参数说明保存在 [`EVALUATION.md`](EVALUATION.md)。核心正式流程为：

```bash
# 第一次 strict/control 正式实验：可按 shard 运行后合并
python -m latent_world_model.evaluation.runner --help

# 第一次实验统计与图表
python -m latent_world_model.evaluation.deep_analysis --help

# 数据、协议与上游融合审计
python -m latent_world_model.evaluation.audit_collection --help
python -m latent_world_model.evaluation.audit_model_protocol --help
python -m latent_world_model.evaluation.audit_training_view_fusion --help

# 第二次完整 joint-C3 正式实验与汇总
python -m latent_world_model.evaluation.runner --help
python -m latent_world_model.evaluation.report --help

# 归档完整性
PYTHONPATH=$PWD conda run --no-capture-output -n VLA_JEPA \
  python -m latent_world_model.evaluation.archive_manifest \
  --root . --output reports/ARTIFACT_MANIFEST.json --strict
```

### 13.3 关键产物

| 路径 | 内容 |
|---|---|
| `datasets/vla_jepa_libero130_v3/README.md` | 数据集 schema、目录和使用说明 |
| `evaluation_outputs/formal_half/` | F0–F5 的 11,700 条正式结果及 embedding cache |
| `evaluation_outputs/deep_analysis/` | 第一次实验聚合 CSV、统计与图表 |
| `evaluation_outputs/audit/` | collection、replay、latent、protocol、parity、fusion 审计证据 |
| `evaluation_outputs/joint_c3_full/` | J0–J2 的 5,850 条结果、summary 与图表 |
| `reports/ARTIFACT_MANIFEST.json` | 关键文件大小、行数与 SHA-256 |
| `FINAL_REPORT.md` | 本文，唯一需要对外展示的完整主报告 |

`evaluation_outputs/` 本地约 417 MB，包含逐窗口 JSONL 和 NumPy embedding cache。大体积 HDF5、MP4、checkpoint、逐窗口结果和 cache 不提交 Git，避免仓库膨胀；manifest 使本地证据仍可校验。

### 13.4 图表

现有图表覆盖：

- 正式条件 MSE 与置信区间；
- 配对效应 forest plot；
- suite MSE heatmap；
- stage/success 分层；
- motion quartile 与 MSE；
- MSE vs motion 散点；
- persistence ratio 分布；
- historical horizon error；
- joint-C3 condition 和 transition-position error。

它们分别位于 `evaluation_outputs/deep_analysis/`、`evaluation_outputs/formal_half/` 和 `evaluation_outputs/joint_c3_full/`。本文已经包含理解结论所需的关键数字，图表只用于展示，不要求读者再阅读其他报告。

### 13.5 证据报告的定位

以下旧文档继续保留，作用是审计追溯，而不是理解本文的前置材料：

- `COMPREHENSIVE_REPORT.md`：第一次实验形成时的完整历史分析；
- `EXPERIMENT_AUDIT_REPORT.md`：审计过程和实现证据；
- `SECOND_EXPERIMENT_REPORT.md`：joint-C3 单项实验记录；
- `reports/archive/FIRST_EXPERIMENT_SUMMARY.md`：历史摘要。

如旧文档中的解释与本文冲突，以本文为准；原始数值则以 `metrics.jsonl`、`summary.json` 和 manifest 为证据源。

### 13.6 验证命令

```bash
conda run -n VLA_JEPA python -m unittest discover -s tests -v
git diff --check
```

## 14. 最终结论

本项目已经完成从数据采集到模型审计的闭环：覆盖全部 130 个 LIBERO task，形成 1300 条视频/HDF5 配对 rollout；验证了动作、query、latent action、视频、环境重放、checkpoint 和独立 predictor；在确定性的 130×5 子集上完成 11,700 条 strict/control 结果和 5,850 条 joint-C3 结果；并定位了协议差异、历史 H3 时间错位和上游多视角 batch 融合错误。

在这一证据基础上，结论应分成两句：

> **对用户需要的 strict-causal 任务，当前冻结 predictor 尚未成功：C1/C3 单步误差均较高，未超过 persistence，也没有观察到样本特定 latent-action 增益。**

> **对 VLA-JEPA 原生 intended joint-C3 目标，当前 predictor 已学到稳定映射：完整 1950-window 结果为 MSE 3.5046、L1 1.3041、cosine 0.7777；但其 context 含未来信息，不能等价为过去预测未来。**

这就是当前 checkpoint 的完整能力边界：**joint representation mapping 成功，strict-causal world-model prediction 未成功。**
