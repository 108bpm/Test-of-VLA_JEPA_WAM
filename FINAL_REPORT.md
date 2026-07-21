# VLA-JEPA Latent World Model 在 LIBERO 上的最终评估报告

日期：2026-07-21

评估对象：从 VLA-JEPA 提取的冻结 V-JEPA2 encoder 与冻结 latent predictor

评估范围：LIBERO 全部 130 个任务；正式结果使用每个任务 5 条 rollout，共 650 条轨迹、1950 个评估窗口

训练策略：本项目只评估现有 checkpoint，没有重新训练或微调 encoder、predictor 或 latent-action 生成模块

数据集：[Monita108/VLA_JEPA-on-libero](https://huggingface.co/datasets/Monita108/VLA_JEPA-on-libero)

## 1. 报告摘要

### 1.1 要回答的问题

本项目希望判断现有 VLA-JEPA latent predictor 能否作为 latent world model 使用。具体问题是：

> 给定机器人当前或过去的视觉 latent 状态，以及同一时刻的 VLA-JEPA latent action，模型能否准确预测尚未观察到的未来视觉 latent 状态？

为了避免把不同能力混为一谈，最终评估分为两个任务：

| 任务 | 模型能看到的信息 | 评估目的 |
|---|---|---|
| **严格过去预测未来（strict-causal）** | 只包含当前及过去画面的 latent，加上对应 latent action | 回答模型能否真正从过去预测未知未来 |
| **VLA-JEPA 原生 joint-C3 映射** | 由完整 8 帧联合编码得到的 context latent，加上 24 个 latent-action tokens | 回答 checkpoint 是否学会了其原生训练目标 |

### 1.2 最终答案

| 能力问题 | 结果 | 判断 |
|---|---:|---|
| 当前状态预测下一状态，strict C1 | MSE `6.2735` | **未达到准确预测要求** |
| 过去三个状态预测下一状态，strict C3 | MSE `6.5445` | **增加历史没有改善预测** |
| 直接保持当前 latent 不变 | MSE `4.4658` | 比 strict C1/C3 更准确 |
| strict 条件下更换为其他 episode 的 latent action | 与正确 action 的 MSE 差约 `-0.0001` | **未观察到样本特定 action 增益** |
| 原生 joint-C3 映射 | MSE `3.5046`，L1 `1.3041`，cosine `0.7777` | **checkpoint 学会了稳定的 joint 表示映射** |
| joint-C3 下更换为其他 episode 的 latent action | 与正确 action 的 MSE 差约 `+0.0000016` | **同样未观察到样本特定 action 增益** |

最重要的结论是：

> **当前 predictor 能够完成 VLA-JEPA 的 joint-C3 表示映射，但不能据此称为有效的严格因果 world model。在真正只给现在/过去信息的条件下，它预测未观察未来 latent 的误差高于直接保持当前 latent。**

joint-C3 的结果更好并不与 strict-causal 结果矛盾。joint encoder 在生成 context 时已经同时看到完整 8 帧，因此 context 中可以包含目标时刻的信息；它衡量的是模型对原生联合表示目标的拟合能力，不是只看过去预测未知未来的能力。

## 2. 被评估的模型

### 2.1 模块来源

`latent_world_model` 是从 VLA-JEPA 中提取并独立封装的视觉编码与 latent prediction 模块，目的是让其他项目可以直接使用这两个能力，而不必加载完整的机器人 policy 工程。

评估使用：

- V-JEPA2 ViT-L encoder；
- 输入分辨率 256 px；
- tubelet size 2；
- agent view 和 wrist view 两个相机；
- VLA-JEPA LIBERO checkpoint 中的 `vj_predictor.*`；
- predictor 参数量 `161,647,616`；
- 双视角融合后的 feature dimension 为 2048。

encoder 和 predictor 均使用原有权重并保持冻结。独立 predictor 与 VLA-JEPA 源 predictor 在相同输入下输出逐元素一致，最大绝对差和平均绝对差均为 0。

### 2.2 输入和输出

V-JEPA2 把 8 帧视频编码为四个 latent 时间块：

```text
z0, z1, z2, z3
```

每个时间块对应两个连续视频帧，并包含 256 个空间 patch tokens。双视角融合后，每个 token 的 feature dimension 为 2048。

默认 C3 接口为：

```text
视觉 context [z0,z1,z2] : [B,768,2048]
latent action [g0,g1,g2] : [B, 24,2048]
预测输出                    : [B,768,2048]
训练目标 [z1,z2,z3]       : [B,768,2048]
```

每个 `g` 含 8 个 action tokens，因此三个时间位置共 24 个 tokens。

### 2.3 C1、C3 和 H1 的含义

为方便第一次接触项目的读者，本文使用以下记号：

| 记号 | 含义 |
|---|---|
| C1 | predictor 只接收一个当前 latent 状态和一组 action tokens |
| C3 | predictor 接收连续三个 latent 状态和三组 action tokens |
| H1 | 预测下一 latent 状态，即向前一个 latent 时间步 |

由于 tubelet size 为 2，一个 latent 时间步相当于向前推进两个视频帧。strict-causal 状态本身由截至该时刻的 8 帧历史 clip 编码，因此 C1 并不是只看一个 RGB frame，而是只向 predictor 提供一个已经由过去画面形成的状态表示；C3 则额外提供两个更早的状态表示。

### 2.4 latent action 是什么

这里的 latent action 是 VLA-JEPA 自身用于 world-model predictor 的 24 个 Qwen hidden-state tokens。它们来自 `<|action_i|>` token slots，并不是把实际执行的 7 维机器人动作直接编码成 24 个 tokens。

这不影响本次评估的有效性，因为：

- VLA-JEPA 训练 predictor 时输入的就是这 24 个 tokens；
- 本项目保存并输入的是相同来源、相同顺序和相同 shape 的 tokens；
- 它们可以包含动作、任务意图和策略状态等信息，而不必与实际 action chunk 一一对应。

本报告进一步通过 correct-vs-shuffled 对照判断 predictor 是否真正利用了这些 tokens 中与当前样本有关的信息。

## 3. LIBERO 评估数据集

### 3.1 数据集规模

数据由 VLA-JEPA policy 在 LIBERO 环境中交互产生，覆盖五个标准 suite 的全部 130 个任务，每个任务采集 10 条 rollout。

| Suite | Tasks | Rollouts | Successful | Success rate | Mean frames | Mean policy queries |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-SPATIAL | 10 | 100 | 100 | 100.00% | 100.16 | 14.70 |
| LIBERO-OBJECT | 10 | 100 | 100 | 100.00% | 132.13 | 19.32 |
| LIBERO-GOAL | 10 | 100 | 100 | 100.00% | 107.61 | 15.86 |
| LIBERO-90 | 90 | 900 | 182 | 20.22% | 356.12 | 51.65 |
| LIBERO-10 | 10 | 100 | 98 | 98.00% | 248.65 | 36.00 |
| **总计** | **130** | **1300** | **580** | **44.62%** | — | — |

完整数据包含：

| 数据项 | 数量或大小 |
|---|---:|
| HDF5 rollout records | 1300 |
| 对应 MP4 videos | 1300 |
| 控制帧 | 379,361 |
| policy queries | 55,073 |
| 执行动作 | 379,361 行 |
| latent-action values | 2,706,948,096 |
| HDF5 总大小 | 约 57.25 GB |
| 视频总大小 | 约 252 MB |

### 3.2 每条 rollout 包含什么

每个 HDF5 record 包含：

- agent view RGB，shape `[T,256,256,3]`；
- wrist view RGB，shape `[T,256,256,3]`；
- robot states，shape `[T,8]`；
- 实际执行动作，shape `[T,7]`；
- policy query 对应的 frame index，shape `[N]`；
- latent-action tokens，shape `[N,24,2048]`，float16；
- unnormalized action chunks，shape `[N,7,7]`；
- suite、task、episode、语言指令和成功状态。

每个 HDF5 都有一个按 `suite/task/episode` 对应的 MP4 视频。视频用于展示和人工查看；模型计算直接使用 HDF5 中的原始 RGB，避免视频压缩影响。

### 3.3 数据质量

最终数据集满足：

- 1300 个 rollout identity 全部唯一；
- 1300 个 HDF5 与 1300 个 MP4 一一匹配；
- 没有缺失 record、孤立视频或重复 episode；
- 379,361 行执行动作与保存的 action chunks 对齐；
- 2,706,948,096 个 latent-action 数值中没有 NaN 或 Inf；
- 没有全零 latent query；
- HDF5 和 MP4 帧数全部匹配；
- 环境重放可以复现双视角画面、机器人状态和任务结果；
- 重新在线计算的 latent-action tokens 与已保存 tokens 一致。

失败 rollout 被保留。它们仍然是合法的机器人动作—视觉轨迹，可以用于 latent dynamics 评估；success 只作为结果分层变量，不作为是否保留数据的标准。

### 3.4 正式评估子集

正式评估从每个 task 确定性选择 episode `0,2,4,6,8`：

```text
130 tasks × 5 rollouts = 650 rollouts
650 rollouts × early/middle/late = 1950 windows
```

| Suite | Formal rollouts | Successful | Windows |
|---|---:|---:|---:|
| LIBERO-SPATIAL | 50 | 50 | 150 |
| LIBERO-OBJECT | 50 | 50 | 150 |
| LIBERO-GOAL | 50 | 50 | 150 |
| LIBERO-90 | 450 | 96 | 1350 |
| LIBERO-10 | 50 | 50 | 150 |
| **总计** | **650** | **296** | **1950** |

每条 rollout 使用 early、middle 和 late 三个时间窗口，保证评估覆盖轨迹不同阶段。strict-causal 和 joint-C3 两类实验使用同一组 1950 windows。

## 4. 两种评估协议

### 4.1 Strict-causal：真正的过去预测未来

V-JEPA2 encoder 需要 8 帧输入。为了保证某个 latent 状态不包含未来信息，strict-causal 协议对每个时刻构造一个只截至该时刻的 8 帧 clip，并只取该 clip 的最后一个 latent block 作为状态。episode 开始处不足 8 帧时只使用左侧填充。

因此：

```text
strict state s_t = encoder(frames up to time t)
```

模型在预测 `s_(t+1)` 时无法看到目标时刻之后的画面。

最终 strict 单步实验包括：

| 展示名称 | Artifact ID | 视觉输入 | latent action | 目标 | 目的 |
|---|---|---|---|---|---|
| Strict-C1 | F0 | 当前状态 | 当前正确 action group | 下一真实状态 | 核心 strict 结果 |
| Strict-C3 | F1 | 过去三个状态 | 三组正确 action | 下一真实状态 | 检验增加历史是否有效 |
| Zero-action | F3 | 与 Strict-C1 相同 | 全零 tokens | 与 Strict-C1 相同 | 检验正常条件激活的影响 |
| Shuffled-action | F4 | 与 Strict-C1 相同 | 同 task、同阶段、其他 episode 的 tokens | 与 Strict-C1 相同 | 检验样本特定 action 信息 |
| Joint-input control | F5 | 8 帧联合编码得到的当前 block | 当前正确 action group | 与 Strict-C1 相同 | 展示额外视觉信息对误差的影响 |

Joint-input control 不是严格因果结果。它只用于说明：如果 context 表示可以接触完整 8 帧，任务会变得多容易。

### 4.2 Joint-C3：VLA-JEPA 原生目标

joint-C3 一次性把连续 8 帧送入 encoder：

```text
8 frames --joint encoder--> z0,z1,z2,z3

context = [z0,z1,z2]
action  = [g0,g1,g2]
target  = [z1,z2,z3]
```

三个正式条件为：

| 展示名称 | Artifact ID | latent action | 目的 |
|---|---|---|---|
| Joint-C3 correct | J0 | 当前窗口正确 tokens | 原生目标主结果 |
| Joint-C3 shuffled | J1 | 同 task、同阶段、其他 episode tokens | 样本特定 action 对照 |
| Joint-C3 zero | J2 | 全零 tokens | 正常 token 激活对照 |

joint encoder 同时看到完整 8 帧，所以 `z0,z1,z2` 可以包含后续帧信息；此外 target 中的 `z1,z2` 已经出现在 context block 序列中。该协议适合判断 checkpoint 是否学会自己的训练目标，不适合证明在线因果预测能力。

## 5. 指标和统计方法

### 5.1 直接误差指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| MSE | prediction 与真实 future latent 的逐元素均方误差 | 越低越好；本文主指标 |
| L1 | 逐元素绝对误差；也是 VLA-JEPA predictor 的训练 loss | 越低越好 |
| RMSE | MSE 的平方根，与 latent 原值同尺度 | 越低越好 |
| Normalized MSE | 按 target 能量归一化的 MSE | 越低越好 |
| Token cosine | 对应预测 token 与目标 token 的方向相似度 | 越高越好 |
| Delta cosine | 预测变化量与真实变化量的方向相似度 | 越高越好 |
| Retrieval top-k | 能否在候选中找回与 prediction 对应的真实 target | 越高越好 |

MSE 是对“预测结果和真实结果相差多大”的直接回答，不依赖任何基线。

### 5.2 为什么还报告 persistence

`persistence` 是最简单的动态预测：

```text
预测未来状态 = 当前状态
```

它不是模型目标，也不替代直接 MSE。它提供一个易解释的参照：如果环境在很短时间内变化不大，直接保持当前 latent 本身就可能得到一定精度；world model 至少需要超过这个简单方案，才能表明它学到了有用的短期动态。

```text
persistence ratio = predictor MSE / persistence MSE
```

- ratio `< 1`：predictor 比保持当前状态更好；
- ratio `= 1`：两者相当；
- ratio `> 1`：predictor 比保持当前状态更差。

### 5.3 统计方式

- 所有条件差异都在同一个 window 上配对；
- 置信区间使用 task→rollout 层级 bootstrap，1000 次；
- 同一 rollout 的 early/middle/late 三个窗口保持为同一 cluster；
- 主要配对比较使用 Holm 多重比较校正；
- suite、stage 和 success 结果用于描述覆盖范围，不用于声称因果关系。

## 6. Strict-causal 最终结果

### 6.1 主结果

| 条件 | n | MSE | 95% CI | L1 | RMSE | Normalized MSE | Token cosine | Delta cosine | Persistence MSE | Ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Strict-C1 (F0)** | 1950 | **6.2735** | [6.2364,6.3104] | 1.7532 | 2.5039 | 0.7133 | 0.5694 | 0.4120 | 4.4658 | **1.4282** |
| **Strict-C3 (F1)** | 1950 | **6.5445** | [6.5050,6.5857] | 1.7683 | 2.5574 | 0.7442 | 0.5585 | 0.4058 | 4.4658 | **1.4909** |
| Zero-action (F3) | 1950 | 6.2227 | [6.1846,6.2593] | 1.7488 | 2.4937 | 0.7075 | 0.5728 | 0.4133 | 4.4658 | 1.4168 |
| Shuffled-action (F4) | 1950 | 6.2734 | [6.2402,6.3117] | 1.7532 | 2.5039 | 0.7133 | 0.5694 | 0.4120 | 4.4658 | 1.4282 |
| Joint-input control (F5) | 1950 | 5.1333 | [5.0969,5.1719] | 1.5869 | 2.2647 | 0.5835 | 0.6539 | 0.6900 | 9.5341 | 0.5385 |

Strict-C1 的直接 MSE 为 `6.2735`。这已经直接回答了 prediction 与真实 future latent 的误差大小。其 persistence MSE 为 `4.4658`，意味着 predictor 的误差约为保持当前状态的 `1.428` 倍，即高约 42.8%。

Strict-C3 使用了更多过去状态和更多 action groups，但 MSE 上升到 `6.5445`。在当前 checkpoint 上，增加历史没有改善单步预测。

### 6.2 配对比较

| 比较（左减右；正值表示左侧更差） | Mean MSE difference | 95% CI | Holm p | 解释 |
|---|---:|---|---:|---|
| Strict-C1 − persistence | **+1.8078** | [1.7407,1.8746] | <0.001 | predictor 明显差于保持当前状态 |
| Strict-C3 − Strict-C1 | **+0.2709** | [0.2531,0.2888] | <0.001 | 三状态历史没有改善 H1 |
| Zero-action − Strict-C1 | **−0.0509** | [-0.0552,-0.0463] | <0.001 | 数值差很小，zero 甚至略低 |
| Shuffled-action − Strict-C1 | **−0.0001** | [-0.0012,0.0009] | 0.846 | 正确和错配 action 无可检测差异 |
| Joint-input control − Strict-C1 | **−1.1402** | [-1.1519,-1.1274] | <0.001 | context 接触完整窗口后误差显著下降 |

### 6.3 latent action 的作用

Shuffled-action 只替换 latent action，视觉 context、target、task 和轨迹阶段全部保持不变。它使用同一 task、同一阶段、其他 episode 的正常 tokens，因此比全零 tokens 更适合判断模型是否使用了样本特定的 action 信息。

F4 与 F0 的平均差只有 `-0.0001`，置信区间跨 0。由此可以得出：

> 在 strict-C1 测试中，没有观察到 predictor 利用当前样本 latent action 差异来改善 future-latent prediction。

这不等于 latent action 不包含动作信息；它只表示当前 predictor 的输出对本测试所替换的样本特定差异不敏感。

### 6.4 不同 suite 上的结果

| Suite | Windows | Strict-C1 MSE | Strict-C3 MSE | Joint-input control MSE |
|---|---:|---:|---:|---:|
| LIBERO-10 | 150 | 6.0440 | 6.2720 | 4.9172 |
| LIBERO-90 | 1350 | 6.3210 | 6.5639 | 5.1924 |
| LIBERO-GOAL | 150 | 6.2221 | 6.4980 | 5.0377 |
| LIBERO-OBJECT | 150 | 6.2541 | 6.7504 | 5.0592 |
| LIBERO-SPATIAL | 150 | 6.1468 | 6.4829 | 4.9876 |

Strict-C1 在五个 suite 上均处于约 `6.04–6.32`，说明总体结论不是由某一个 suite 单独造成的。LIBERO-90 包含 90 个任务，因此在按 task 等权的总体结果中占 1350/1950 个 windows；报告保留 suite 分层，避免只看总体平均。

### 6.5 轨迹阶段和成功状态

| 分层 | Strict-C1 MSE | Strict-C3 MSE | Joint-input control MSE |
|---|---:|---:|---:|
| Early | 6.1618 | 6.4560 | 5.0248 |
| Middle | 6.3005 | 6.5365 | 5.1511 |
| Late | 6.3583 | 6.6410 | 5.2240 |
| Successful rollout | 6.1536 | 6.4562 | 4.9949 |
| Failed rollout | 6.3739 | 6.6183 | 5.2490 |

late 和 failed rollout 的平均误差略高，但 strict-C1 在所有分层中都没有表现出足以改变总体判断的低误差区域。

### 6.6 运动量与误差

按真实 latent 变化量从低到高划分四分位时，Strict-C1 MSE 分别为：

| Target-motion quartile | Mean target Δ RMS | Strict-C1 MSE |
|---|---:|---:|
| Q1 | 1.9420 | 6.3744 |
| Q2 | 2.0731 | 6.2608 |
| Q3 | 2.1542 | 6.2312 |
| Q4 | 2.2678 | 6.2277 |

误差没有随真实变化量单调上升。因此，strict 结果不能简单解释为“未来运动太大所以预测困难”。target latent variance 与 MSE 的相关更强（Pearson `0.8442`，Spearman `0.8154`），说明 latent 本身的数值尺度会显著影响绝对 MSE；这也是同时报告 normalized MSE 和 cosine 的原因。

### 6.7 Strict-causal 结论

综合直接误差、persistence、历史输入和 action 对照：

> **当前冻结 predictor 尚不能比较准确地完成“现在/过去 latent + latent action → 未观察的下一 latent”任务。C1 和 C3 均未超过 persistence；增加历史没有改善结果；样本特定 latent action 也没有带来可检测增益。**

## 7. Joint-C3 最终结果

### 7.1 主结果

| 条件 | n | MSE | 95% CI | L1 | RMSE | Normalized MSE | Token cosine | Persistence ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **Joint-C3 correct (J0)** | 1950 | **3.5046** | [3.4604,3.5450] | **1.3041** | 1.8707 | 0.3943 | **0.7777** | 0.3656 |
| Joint-C3 shuffled (J1) | 1950 | 3.5046 | [3.4629,3.5434] | 1.3041 | 1.8707 | 0.3943 | 0.7777 | 0.3656 |
| Joint-C3 zero (J2) | 1950 | 4.1833 | [4.1495,4.2215] | 1.4358 | 2.0444 | 0.4708 | 0.7292 | 0.4365 |

VLA-JEPA 训练 predictor 时直接优化 L1，因此 `L1=1.3041` 是原生目标复现最直接的指标。MSE `3.5046` 和 token cosine `0.7777` 进一步说明 predictor 在 joint 表示下可以稳定输出与 target 接近的表示。

J0 retrieval top-1/top-5 为 `0.9995/1.0000`，表示 prediction 几乎总能在候选中匹配到同一次 joint encoding 产生的 target。这说明 joint 表示保持了很强的样本身份信息，但不能单独证明因果预测，因为 context 已经接触完整窗口。

### 7.2 三个 transition 位置

Joint-C3 在一次 predictor forward 中同时输出三个位置：

| Position | Mapping | MSE | 95% CI | L1 | Token cosine |
|---|---|---:|---|---:|---:|
| 1 | `z0,g0 → z1` | 3.6799 | [3.6384,3.7171] | 1.3326 | 0.7650 |
| 2 | `z1,g1 → z2` | 3.5033 | [3.4594,3.5452] | 1.3033 | 0.7806 |
| 3 | `z2,g2 → z3` | 3.3306 | [3.2854,3.3743] | 1.2764 | 0.7873 |

三个位置都保持稳定，没有某一位置出现明显误差失控。它们是同一次 teacher-forcing forward 的三个输出位置，不是把预测结果递归输入模型的三步 rollout。

### 7.3 latent action 对照

| 比较 | Mean MSE difference | 95% CI | Holm p | 解释 |
|---|---:|---|---:|---|
| J1 shuffled − J0 correct | **+0.0000016** | [-0.0000244,+0.0000268] | 0.904 | 无可检测差异 |
| J2 zero − J0 correct | **+0.6787** | [0.6719,0.6858] | <0.001 | 全零 tokens 明显恶化 |

全零 tokens 远离正常 Qwen hidden-state 分布。J2 变差说明 predictor 需要正常范围的 token 激活；J1 与 J0 完全相近则说明，在正常 token 分布内部，模型没有表现出对当前 episode 特定 token 差异的利用。

### 7.4 不同 suite、阶段和成功状态

| Suite | Windows | Joint-C3 correct MSE |
|---|---:|---:|
| LIBERO-OBJECT | 150 | 3.1332 |
| LIBERO-SPATIAL | 150 | 3.2498 |
| LIBERO-GOAL | 150 | 3.2583 |
| LIBERO-10 | 150 | 3.4016 |
| LIBERO-90 | 1350 | 3.6130 |

| 分层 | Windows | Joint-C3 correct MSE |
|---|---:|---:|
| Early | 650 | 3.4311 |
| Middle | 650 | 3.5356 |
| Late | 650 | 3.5472 |
| Successful rollout | 888 | 3.3350 |
| Failed rollout | 1062 | 3.6464 |

五个 suite 和三个轨迹阶段均呈现稳定 joint mapping，结果不是由单一数据分组产生。

### 7.5 Joint-C3 结论

> **当前 predictor 已经学会 VLA-JEPA 原生 joint-C3 shifted-latent mapping。该能力在全部 130 个任务、1950 个窗口和三个 transition 位置上均稳定存在。**

但该结论必须与 strict-causal 能力分开：joint context 可以包含未来帧信息，且 target 的 `z1,z2` 与 context block 重叠，所以它不能证明模型能够在部署时只看过去预测未知未来。

## 8. 如何综合理解两组结果

### 8.1 为什么 joint MSE 更低

strict 和 joint 使用不同的 latent 构造方式：

```text
strict context：每个状态只由截至该时刻的画面编码
joint context：完整 8 帧一起编码，早期表示可以受后续帧影响
```

因此 joint context 比 strict context 拥有更多与 target 相关的信息。两种协议的 context latent 和 target latent 都不相同，不能把 `6.54 → 3.50` 简单解释成同一个任务上 predictor 提升了约 45%。

正确解释是：

- `6.2735/6.5445` 回答严格过去预测未来的实际能力；
- `3.5046` 回答 predictor 对 VLA-JEPA 原生 joint 表示目标的拟合能力。

### 8.2 能力矩阵

| 能力 | 是否得到支持 | 证据 |
|---|---|---|
| 独立模块忠实复现 VLA-JEPA predictor | 是 | 同输入下输出最大/平均绝对差均为 0 |
| 只用当前 causal latent 预测下一 latent | 否 | Strict-C1 MSE 6.2735，高于 persistence 4.4658 |
| 使用三个过去状态改善下一步预测 | 否 | Strict-C3 比 Strict-C1 高 0.2709 MSE |
| 利用样本特定 latent action 改善预测 | 未观察到 | F4≈F0，J1≈J0 |
| 依赖正常范围的 latent-token 激活 | 是 | Joint zero 比 correct 高 0.6787 MSE |
| 学会 VLA-JEPA joint-C3 原生目标 | 是 | MSE 3.5046，L1 1.3041，cosine 0.7777 |
| joint-C3 证明严格因果预测 | 否 | joint encoder 可见完整窗口，context/target 有重叠 |
| 严格因果的窗口外多步 rollout | 本报告不作结论 | 最终正式结论限定为 H1 |

### 8.3 对核心需求的直接回答

用户的目标是：

```text
输入现在或过去的 latent 状态
+ 对应 latent action
→ 预测尚未观察到的未来 latent 状态
```

在现有 checkpoint、当前 strict-causal latent 定义和 LIBERO 130-task 数据上，答案是：

> **目前不能比较准确地完成。预测结果与真实 future latent 的平均 MSE 为 6.2735（C1）或 6.5445（C3），均高于直接保持当前 latent 的 4.4658；同时没有发现样本特定 latent action 带来的预测增益。**

如果问题改为“模型有没有学会 VLA-JEPA 原本的 joint-C3 目标”，答案则是肯定的：

> **模型能够稳定完成 joint-C3 mapping，MSE 为 3.5046、L1 为 1.3041、token cosine 为 0.7777；但这是一种包含未来信息的联合表示映射，不是严格的过去预测未来。**

## 9. 结论边界

### 9.1 本报告能够证明什么

- 结论覆盖 LIBERO 的全部 130 个任务，而不是少量演示任务；
- strict 和 joint 两组正式结果各使用同一批 1950 个窗口；
- 模型、数据和 latent-action 接口保持冻结且一致；
- 结果同时包含直接误差、相对基线、action controls 和数据分层；
- joint-C3 学习成功与 strict-causal 预测失败可以同时成立。

### 9.2 本报告不声称什么

- 不声称 latent MSE 可以直接换算成人眼可见的视频质量；
- 不声称已经验证严格因果的长期自回归 rollout；
- 不声称 zero-action 变差就证明模型使用了具体动作语义；
- 不声称 joint-C3 的低误差等价于 causal dynamics 学习成功；
- 不把当前 checkpoint 的结果外推到重新训练或不同 latent 定义的模型。

### 9.3 关于像素视频预测

V-JEPA2 没有与当前 encoder 配套的 latent-to-RGB decoder。本项目的 prediction 和 target 都是 patch-level latent features，因此 MSE、L1、cosine 和 retrieval 均在表示空间中计算。

如果需要生成未来 RGB 视频并计算 PSNR、SSIM 或 LPIPS，需要额外训练与当前 V-JEPA2 配置严格匹配的 pixel decoder。该 decoder 不包含在现有 encoder 或 predictor checkpoint 中。

## 10. 项目产物与复现入口

### 10.1 公开入口

| 内容 | 地址或路径 |
|---|---|
| GitHub 仓库 | `git@github.com:108bpm/Test-of-VLA_JEPA_WAM.git` |
| Hugging Face 数据集 | [Monita108/VLA_JEPA-on-libero](https://huggingface.co/datasets/Monita108/VLA_JEPA-on-libero) |
| 英文项目说明 | `README.md` |
| 中文项目说明 | `README_CN.md` |
| 评估命令 | `EVALUATION.md` |
| 无结果实验框架 | `EXPERIMENT_FRAMEWORK.md` |
| 数据 schema | `datasets/vla_jepa_libero130_v3/README.md` |
| 最终报告 | `FINAL_REPORT.md` |

### 10.2 本地结果

| 路径 | 内容 |
|---|---|
| `evaluation_outputs/formal_half/` | strict/control 的逐窗口结果、summary 和图表 |
| `evaluation_outputs/deep_analysis/` | suite/stage/success/motion 分层、统计和图表 |
| `evaluation_outputs/joint_c3_full/` | joint-C3 correct/shuffled/zero 结果、summary 和图表 |
| `reports/ARTIFACT_MANIFEST.json` | 关键文件大小、行数与 SHA-256 |

大体积 HDF5、MP4、checkpoint、逐窗口 JSONL 和 NumPy caches 不提交 Git。公开数据由 Hugging Face 托管，本地正式结果由 artifact manifest 校验。

### 10.3 环境与验证

```bash
conda activate VLA_JEPA
cd /home/embodied/users/jlb/proj0/latent_world_model

python -m unittest discover -s tests -v

PYTHONPATH=$PWD python -m latent_world_model.evaluation.archive_manifest \
  --root . \
  --output reports/ARTIFACT_MANIFEST.json \
  --strict
```

## 11. 汇报结论

本项目完成了一个从 VLA-JEPA 提取的 latent world model 模块，并建立了覆盖 LIBERO 全部 130 个任务的 1300-rollout 数据集。正式评估在每个任务 5 条 rollout、共 1950 个 early/middle/late 窗口上进行，没有重新训练任何模型权重。

最终应向外部听众汇报以下三点：

1. **严格过去预测未来没有成功。** Strict-C1 MSE 为 `6.2735`，Strict-C3 为 `6.5445`，均高于 persistence 的 `4.4658`。
2. **VLA-JEPA 原生 joint-C3 目标学习成功。** Joint-C3 MSE 为 `3.5046`、L1 为 `1.3041`、token cosine 为 `0.7777`。
3. **joint-C3 能力不能替代 strict-causal 能力。** joint context 在编码时可以接触完整窗口中的未来信息；当前 checkpoint 因此可以完成 joint representation mapping，但还不能作为已经验证有效的严格因果 latent world model。

一句话总结：

> **当前模型的 joint representation mapping 有效，但用户真正需要的“现在/过去 latent + latent action → 未观察未来 latent”能力尚未实现。**
