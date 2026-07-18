# VLA-JEPA Latent World Model 最终评估报告

日期：2026-07-18
范围：LIBERO 130 tasks、冻结 V-JEPA2 encoder、冻结 VLA-JEPA predictor、无训练或微调

## 1. 最终结论

本项目要回答两个必须分开的能力问题：

| 问题 | 最接近的正式条件 | 结果 | 最终判断 |
|---|---|---|---|
| 只看现在/过去 latent 和 latent action，能否准确预测未观察的未来 latent？ | strict C1/C3 | C1 MSE 6.2735；C3 MSE 6.5445 | **尚不能。预测误差高于直接保持当前 latent，且未显示 action-specific 增益。** |
| 按 VLA-JEPA 原生 joint-C3 teacher-forcing 目标，predictor 是否学到了条件映射？ | joint C3 → same-joint shifted target | MSE 3.5046；L1 1.3041；cosine 0.7777 | **学到了稳定的 joint-representation 映射。** |

这两个结论不矛盾：

> predictor 在 joint-C3 表示下能够较好地完成自身目标，但这种 joint representation 在编码 context 时已经可以接触完整 8 帧中的未来信息，因此不能证明严格的过去到未来预测能力。

latent action 的使用方式在训练与评估之间保持一致：上游训练和本项目都向 predictor 输入同一来源、相同形状和相同顺序的 24 个 Qwen hidden-state tokens。它们无需与实际执行动作形成确定性一一编码，现有评估仍然有效；需要单独限定的是 predictor 对样本特定 token 差异几乎不敏感。

## 2. 数据、模型与实验覆盖

### 2.1 数据集

完整数据集 `vla_jepa_libero130_v3` 包含：

| 项目 | 数量 |
|---|---:|
| LIBERO tasks | 130 |
| rollout HDF5 | 1300 |
| 对应 MP4 | 1300 |
| 控制帧 | 379,361 |
| policy queries | 55,073 |
| latent-action token shape | 24 × 2048 |

正式实验使用每个 task 的 episode `0,2,4,6,8`：

```text
130 tasks × 5 rollouts = 650 rollouts
650 rollouts × early/middle/late = 1950 windows
```

第一次 strict-causal 实验和第二次 joint-C3 实验使用完全相同的 1950 个窗口。

### 2.2 冻结模型

- V-JEPA2 ViT-L encoder，256 px，tubelet size 2；
- VLA-JEPA LIBERO checkpoint：`VLA-JEPA-LIBERO.pt`；
- 只加载 `vj_predictor.*`，共 161,647,616 个参数；
- strict load 无 missing/unexpected keys；
- encoder、predictor 和 latent-action 生成器均未训练或微调。

8 帧经过 tubelet=2 后得到四个 latent blocks：

```text
z0 = frames 0,1
z1 = frames 2,3
z2 = frames 4,5
z3 = frames 6,7
```

## 3. 第一次实验：strict-causal 过去预测未来

第一次正式实验的完整历史分析见 [`COMPREHENSIVE_REPORT.md`](COMPREHENSIVE_REPORT.md)。这里保留回答核心问题所需的主要结果。

### 3.1 正式条件

| 条件 | 视觉输入 | latent action | 目标 | 状态 |
|---|---|---|---|---|
| F0 | strict C1，当前 z2 | 当前 g2 | strict future z3 | 有效主结果 |
| F1 | strict C3，z0,z1,z2 | g0,g1,g2 | strict future z3 | 有效历史上下文结果 |
| F2 | 历史 C1 AR-H3 | 跨 query action | 连续未来三步 | 旧时间排程不严格对齐，不进入最终多步结论 |
| F3 | 与 F0 相同 | zero tokens | 与 F0 相同 | 条件激活对照 |
| F4 | 与 F0 相同 | 同 task/stage 其他 episode tokens | 与 F0 相同 | action specificity 对照 |
| F5 | joint C1 | 当前 g2 | strict future z3 | 非因果信息对照，不是原生 joint-C3 目标 |

### 3.2 1950-window 正式结果

| 条件 | MSE | L1 | RMSE | token cosine | persistence MSE/ratio |
|---|---:|---:|---:|---:|---:|
| F0 strict C1 | **6.2735** | 1.7532 | 2.5039 | 0.5694 | 4.4658 / 1.4282 |
| F1 strict C3 | **6.5445** | 1.7683 | 2.5574 | 0.5585 | 4.4658 / 1.4909 |
| F3 zero action | 6.2227 | 1.7488 | 2.4937 | 0.5728 | 4.4658 / 1.4168 |
| F4 shuffled action | 6.2734 | 1.7532 | 2.5039 | 0.5694 | 4.4658 / 1.4282 |
| F5 joint C1→strict target | 5.1333 | 1.5869 | 2.2647 | 0.6539 | 9.5341 / 0.5385 |

直接使用当前 latent 预测未来的 persistence 为：

\[
\hat z_{t+1}=z_t
\]

它与 predictor 使用完全相同的 1950 个真实未来目标。F0 相对 persistence 的配对 MSE 差为：

```text
F0 − persistence = +1.8078
95% CI            = [1.7407, 1.8746]
```

即 F0 的 MSE 比保持当前 latent 高约 42.8%。F1 又比 F0 高 `0.2709`，说明增加过去三步没有改善单步预测。F4 与 F0 的配对差只有 `−0.0001`，置信区间跨 0，说明更换为同 task/stage 其他 episode 的 latent action 后预测基本不变。

### 3.3 第一次实验结论

对用户要求的 strict-causal H1 任务，结论已经由完整 `130×5` 正式集确定：

> 当前 predictor 能产生与真实未来 latent 有中等相关性的输出，但没有超过简单 persistence；C3 历史和样本特定 latent action 均未带来可检测的正向增益。因此当前模型尚不能被称为准确的严格因果 latent world model。

## 4. 全链路审计的关键结果

完整审计证据见 [`EXPERIMENT_AUDIT_REPORT.md`](EXPERIMENT_AUDIT_REPORT.md)。最终报告只保留影响结论可信度的重点。

### 4.1 数据和模块没有错位

| 审计项 | 结果 |
|---|---:|
| 唯一 rollout identity | 1300/1300 |
| action chunk 与逐帧执行动作不匹配 | 0/379,361 |
| latent NaN/Inf | 0/2,706,948,096 values |
| MP4/HDF5 帧数不匹配 | 0/1300 |
| 五 suite LIBERO 动作重放 | 990 actions，状态和双视角画面复现 |
| 在线复算 latent action | 10/10 与 HDF5 逐元素一致 |
| 独立 predictor 对上游 predictor | max/mean absolute difference = 0 |

因此，第一次实验的 strict-causal 数值不能归因于数据保存、action 时间索引、checkpoint 加载或 predictor 提取错误。

### 4.2 latent action 的正确解释

24 个 world-model tokens 是 VLA-JEPA 原生的 learned action-conditioning representation。上游训练 `vj_predictor` 时输入的就是这些 tokens，本项目记录并重新输入的也是同一 tokens，所以训练—评估接口一致。

它们不是实际 7 步 action chunk 的确定性编码，这一点不使实验失效。正确的科学限定是：

- 可以用它们测试既有 predictor；
- correct-vs-shuffle 检验 predictor 是否利用其中的样本特定差异；
- strict 和 joint 两组实验都没有观察到这种样本特定增益。

### 4.3 joint encoder 的未来信息可见性

V-JEPA2 encoder 在一次联合编码中使用非因果时空注意力。将连续 8 帧一起编码时，名义上的 `z0/z1/z2` 可以受到后续帧影响。因此：

```text
joint context z0,z1,z2  !=  只由过去信息构造的 causal states
```

此外，原生 shifted target 是 `[z1,z2,z3]`，其中 `z1/z2` 已经作为 block 出现在 C3 context 中。joint-C3 的低误差可以证明目标拟合，但不能证明在线未来预测。

### 4.4 上游双视角 batch 融合错位

这是审计中最重要的上游实现问题。训练输入先从：

```text
[B,V,T,C,H,W]
```

展平为 sample-major：

```text
b0v0,b0v1,b1v0,b1v1,...,b31v0,b31v1
```

源码随后却执行：

```python
torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
```

在每设备 `B=32,V=2` 时，实际 predictor 行为是：

| predictor row | 实际视觉 feature | latent-action row |
|---:|---|---:|
| 0 | sample 0 view 0 + sample 16 view 0 | sample 0 |
| 1 | sample 0 view 1 + sample 16 view 1 | sample 1 |
| 2 | sample 1 view 0 + sample 17 view 0 | sample 2 |
| 3 | sample 1 view 1 + sample 17 view 1 | sample 3 |

所以它混合了相隔半个 batch 的不同样本、通常还是同一视角；latent-action 行并未同步重排。context 和 target 都从这个错误混合行切分，模型输入还会随 batch 同伴和排列变化。只有 `B=1` 时该写法偶然正确。

正确融合应恢复 `[B,V]` 再拼接同一样本的视角：

```python
video_embeddings = (
    video_embeddings.reshape(B, V, N, D)
    .permute(0, 2, 1, 3)
    .reshape(B, N, V * D)
)
```

32-window 数值复现为：

| 融合 | MSE | L1 | cosine |
|---|---:|---:|---:|
| 正确 sample 内双视角 | 3.3022 | 1.2601 | 0.7897 |
| legacy 错误融合 | 3.2221 | 1.2398 | 0.7950 |
| legacy 错误融合 + shuffled action | 3.2223 | 1.2398 | 0.7949 |

正确与 legacy 输入 latent 的 MSE 为 `10.9984`。legacy loss 略低只说明 checkpoint 更接近自己见过的训练实现；其 target 混合不同 episode，不能当作单条真实轨迹的预测质量。

第二次实验使用独立模块中正确、确定性的 sample 内双视角融合，测试的是有物理样本身份的 intended joint-C3 协议。它不是对 legacy batch 混合行为的扩大复现；后者只保留为实现审计。

### 4.5 审计如何修正旧解释

审计修正的是“模型在原生目标上完全失效”这一过强说法，而不是推翻 strict-causal 正式数字：

- strict C1/C3 的 1950-window 结果仍然有效地回答用户的过去到未来任务；
- 旧 F0/F1/F5 没有完整复现 `joint C3 → same-joint shifted target`；
- 130-window 审计补测得到 joint-C3 MSE `3.4357`，提示原生目标明显更容易；
- 历史 F2 跨 query H3 存在 7-frame policy query 与 2-frame tubelet 的时间错位，不进入最终多步结论。

## 5. 第二次实验：完整 joint-C3 目标复现

第二次实验的独立报告见 [`SECOND_EXPERIMENT_REPORT.md`](SECOND_EXPERIMENT_REPORT.md)。它使用与第一次实验完全相同的 1950 windows。

### 5.1 条件定义

```text
同一次 joint 8-frame encoding -> z0,z1,z2,z3
context                         = z0,z1,z2
target                          = z1,z2,z3
latent action                   = g0,g1,g2，共24 tokens
```

| 条件 | latent action | 目的 |
|---|---|---|
| J0 | 正确 tokens | joint-C3 主结果 |
| J1 | 同 task/stage 其他 episode tokens | 样本特定 action 对照 |
| J2 | zero tokens | 正常条件激活对照 |

### 5.2 完整结果

| 条件 | n | MSE | 95% CI | L1 | RMSE | cosine |
|---|---:|---:|---|---:|---:|---:|
| J0 | 1950 | **3.5046** | [3.4604, 3.5450] | **1.3041** | 1.8707 | **0.7777** |
| J1 shuffled | 1950 | 3.5046 | [3.4629, 3.5434] | 1.3041 | 1.8707 | 0.7777 |
| J2 zero | 1950 | 4.1833 | [4.1495, 4.2215] | 1.4358 | 2.0444 | 0.7292 |

J0 的三个 teacher-forcing transition 位置：

| 位置 | 目标 | MSE | L1 | cosine |
|---|---|---:|---:|---:|
| 1 | z1 | 3.6799 | 1.3326 | 0.7650 |
| 2 | z2 | 3.5033 | 1.3033 | 0.7806 |
| 3 | z3 | 3.3306 | 1.2764 | 0.7873 |

配对 action 对照：

| 比较 | MSE 差 | 95% CI | 结论 |
|---|---:|---|---|
| J1 − J0 | +0.0000016 | [−0.0000244, +0.0000268] | 无可检测差异 |
| J2 − J0 | +0.6787 | [0.6719, 0.6858] | zero 显著恶化 |

这证明 predictor 在正常 latent-token 条件下稳定完成了 joint-C3 mapping；但同 task/stage 内替换具体 tokens 几乎不影响结果。

### 5.3 第二次实验结论

> 在 intended joint-C3 方向下，当前 predictor 确实具有明确的 latent 条件映射能力，完整 130×5 结果确认了审计小样本趋势。这个结果证明原生高层目标被学到，但不证明严格因果预测，因为 joint context 已经接触完整窗口中的未来信息。

## 6. 综合回答：当前模型到底能做什么

| 能力 | 是否得到支持 | 证据 |
|---|---|---|
| strict：只用现在 latent 预测下一 latent | 否 | F0 MSE 6.2735，高于 persistence 4.4658 |
| strict：过去三步改善下一步预测 | 否 | F1 比 F0 再高 0.2709 MSE |
| strict：利用样本特定 latent action | 未观察到 | F4 与 F0 几乎相同 |
| joint C3：复现 same-joint shifted latents | 是 | J0 MSE 3.5046、L1 1.3041、cosine 0.7777 |
| joint C3：利用样本特定 latent action | 未观察到 | J1 与 J0 配对差约 0 |
| 严格因果多步预测 | 尚无有效正式结论 | 历史 F2 时间排程不严格对齐 |

最终应这样表述模型能力：

> 当前冻结 predictor 是一个能够在 VLA-JEPA joint-C3 representation 上稳定完成 shifted-latent mapping 的模型，但还不是一个已经验证有效的 strict-causal latent world model。对用户要求的“现在/过去 latent + latent action → 未观察的未来 latent”，其单步正式结果没有超过保持当前状态；joint-C3 的成功不能替代这个结论。

## 7. 报告、数据与归档

### 7.1 报告层级

| 文档 | 作用 |
|---|---|
| 本文 `FINAL_REPORT.md` | 最终统一解释，优先级最高 |
| [`COMPREHENSIVE_REPORT.md`](COMPREHENSIVE_REPORT.md) | 第一次正式实验的完整历史结果、图表和分层分析 |
| [`EXPERIMENT_AUDIT_REPORT.md`](EXPERIMENT_AUDIT_REPORT.md) | 全链路审计证据和详细实现分析 |
| [`SECOND_EXPERIMENT_REPORT.md`](SECOND_EXPERIMENT_REPORT.md) | 第二次完整 joint-C3 实验 |
| [`reports/README.md`](reports/README.md) | 归档索引和保留策略 |

### 7.2 本地结果目录

| 目录 | 内容 |
|---|---|
| `evaluation_outputs/formal_half/` | 第一次正式实验，11700 条 F0–F5 结果 |
| `evaluation_outputs/deep_analysis/` | 第一次实验 CSV、分层结果和图表 |
| `evaluation_outputs/audit/` | collection/replay/parity/protocol/fusion 审计 JSON |
| `evaluation_outputs/joint_c3_full/` | 第二次实验，5850 条 J0–J2 结果、summary 和图表 |

关键本地文件的行数、字节数和 SHA-256 统一记录在 [`reports/ARTIFACT_MANIFEST.json`](reports/ARTIFACT_MANIFEST.json)。大体积 HDF5、MP4、checkpoint、逐窗口 JSONL 和 NumPy cache 不提交 Git；代码、实验定义、报告和 manifest 提交 Git。

### 7.3 复现与验证

- 完整命令：[`EVALUATION.md`](EVALUATION.md)；
- 数据集 schema：[`datasets/vla_jepa_libero130_v3/README.md`](datasets/vla_jepa_libero130_v3/README.md)；
- 代码测试：`conda run -n VLA_JEPA python -m unittest discover -s tests -v`；
- 归档验证：

```bash
PYTHONPATH=$PWD conda run -n VLA_JEPA \
  python -m latent_world_model.evaluation.archive_manifest \
  --root . --output reports/ARTIFACT_MANIFEST.json --strict
```

## 8. 最终一句话

> **strict-causal 方向：当前 predictor 尚未成功；joint-C3 方向：predictor 学到了稳定映射，但该能力依赖包含未来信息的 joint representation，因而不是用户需要的过去预测未来能力。**
