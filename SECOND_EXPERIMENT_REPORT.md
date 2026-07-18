# 第二次实验：完整 joint-C3 teacher-forcing 评估

## 1. 实验目的

第一次正式实验已经在 `130×5` 数据上完成 strict-causal C1/C3，但没有完整测试 predictor 原生的 joint-C3 目标。本实验补齐这一缺口，回答：

> 在使用原始 joint-C3 协议、给定 24 个 VLA-JEPA latent-action tokens 时，冻结 predictor 能否准确复现同一次 joint encoding 的 shifted future latents？

本实验不重新训练或微调 encoder、predictor 或 latent-action 生成器。

## 2. 数据与协议

使用与第一次正式实验完全相同的确定性半量子集：

- 130 个 LIBERO task；
- 每个 task 选择 episode `0,2,4,6,8`，共 650 条 rollout；
- 每条 rollout 使用 early/middle/late 三个窗口，共 1950 windows；
- J0/J1/J2 各 1950 条结果，总计 5850 条；
- 所有结果 key 唯一，主要数值全部 finite，runner `errors=[]`。

每个窗口只进行一次连续 8 帧 joint encoding：

```text
8 frames --joint encoder--> z0,z1,z2,z3

context = [z0,z1,z2]
action  = [g0,g1,g2] = 24 latent-action tokens
target  = [z1,z2,z3]，来自同一次 joint encoder call
```

三个条件为：

| 条件 | context/target | latent action | 目的 |
|---|---|---|---|
| J0 | joint C3 → same-joint shifted target | 当前窗口正确 tokens | 原生目标主结果 |
| J1 | 与 J0 完全相同 | 同 task、同 stage、其他 episode 的 tokens | 检查样本相关 action specificity |
| J2 | 与 J0 完全相同 | 全零 tokens | 检查 predictor 是否依赖正常范围的条件激活 |

J* 输出中的 `h1/h2/h3` 表示同一次 teacher-forcing 调用的第一、第二、第三个 transition 位置，而不是从当前状态向窗口外自回归三步。

## 3. 主要结果

task→rollout 层级 bootstrap 1000 次；同一 episode 的 early/middle/late 三个窗口作为同一 cluster 保留：

| 条件 | n | MSE | 95% CI | L1 | RMSE | token cosine | normalized MSE |
|---|---:|---:|---|---:|---:|---:|---:|
| J0 correct | 1950 | **3.5046** | [3.4604, 3.5450] | **1.3041** | 1.8707 | **0.7777** | 0.3943 |
| J1 shuffled | 1950 | 3.5046 | [3.4629, 3.5434] | 1.3041 | 1.8707 | 0.7777 | 0.3943 |
| J2 zero | 1950 | 4.1833 | [4.1495, 4.2215] | 1.4358 | 2.0444 | 0.7292 | 0.4708 |

上游训练直接优化 L1；因此 J0 的 `L1=1.3041` 是最直接的训练目标复现指标，MSE 和 cosine 提供补充尺度。

### 3.1 三个 transition 位置

| 目标位置 | 映射 | MSE | 95% CI | L1 | RMSE | token cosine |
|---|---|---:|---|---:|---:|---:|
| 1 | z0,g0 → z1 | 3.6799 | [3.6384, 3.7171] | 1.3326 | 1.9170 | 0.7650 |
| 2 | z1,g1 → z2 | 3.5033 | [3.4594, 3.5452] | 1.3033 | 1.8703 | 0.7806 |
| 3 | z2,g2 → z3 | 3.3306 | [3.2854, 3.3743] | 1.2764 | 1.8230 | 0.7873 |

三个位置都表现稳定，第三个位置没有出现误差爆炸。这里不是自回归 rollout：三步由一次 C3 predictor forward 同时产生。

### 3.2 latent-action 对照

完全相同 window 上的配对差异（左减 J0，正值表示更差）：

| 比较 | MSE 差 | 95% CI | Holm p | 解释 |
|---|---:|---|---:|---|
| J1 shuffled − J0 | +0.0000016 | [−0.0000244, +0.0000268] | 0.904 | 与正确 tokens 无可检测差异 |
| J2 zero − J0 | +0.6787 | [0.6719, 0.6858] | <0.001 | 全零分布外条件显著恶化 |

因此，predictor 明显需要“正常范围的 latent-token 激活”，但在本测试中没有显示出对同 task/stage 不同 episode 的样本特定 token 差异的利用。这个结果不否定 latent action 含有动作或意图信息；它只限定当前 predictor 的经验敏感性。

## 4. 分层结果

### 4.1 Suite

| Suite | windows | J0 MSE |
|---|---:|---:|
| LIBERO-OBJECT | 150 | 3.1332 |
| LIBERO-SPATIAL | 150 | 3.2498 |
| LIBERO-GOAL | 150 | 3.2583 |
| LIBERO-10 | 150 | 3.4016 |
| LIBERO-90 | 1350 | 3.6130 |

### 4.2 轨迹阶段与成功状态

| 分层 | windows | J0 MSE |
|---|---:|---:|
| early | 650 | 3.4311 |
| middle | 650 | 3.5356 |
| late | 650 | 3.5472 |
| success | 888 | 3.3350 |
| failure | 1062 | 3.6464 |

这些是描述性分层，不用于推断 task 难度的因果来源。LIBERO-90 有 90 个 task，因此总体 task 等权结果自然包含较多 LIBERO-90 window；最终报告同时保留 suite 分层。

## 5. 与第一次实验和审计诊断的关系

- 第一次严格因果正式结果：F0 C1 MSE `6.2735`，F1 C3 MSE `6.5445`；
- 审计阶段 130-window joint-C3 诊断：MSE `3.4357`；
- 本次完整 1950-window J0：MSE `3.5046`。

本次结果复现并强化了审计小样本趋势。不过 strict 与 joint 使用不同的 latent 构造和 target，`6.27/6.54` 与 `3.50` 不是同一表示空间协议下的纯 predictor 配对差，不能把数值下降全部解释成模型结构提升。

## 6. 结论

本次完整正式实验支持以下结论：

1. **冻结 predictor 成功学习了 joint-C3 teacher-forcing 条件映射。** 在全部 130 tasks、1950 windows 上，J0 的 MSE 为 `3.5046`、L1 为 `1.3041`、token cosine 为 `0.7777`，三个 transition 位置均稳定。
2. **这种能力不等价于严格的过去预测未来。** joint encoder 同时看到完整 8 帧，context latent 可以包含目标时刻的信息；此外 `z1/z2` 本身也出现在 C3 context 中。
3. **样本特定 latent-action 增益仍未观察到。** J1 与 J0 几乎完全相同；zero-token 恶化只证明正常 token 激活重要，不能证明具体 action 差异被使用。

因此，准确表述是：

> 当前 VLA-JEPA predictor 在 joint-C3 表示下具有明确、稳定的原生目标拟合能力；但这是一种带未来信息的 joint-representation 条件映射能力，不能用来证明用户所需的 strict-causal world-model 预测能力。

## 7. 结果产物

- 逐窗口结果：`evaluation_outputs/joint_c3_full/metrics.jsonl`；
- 配置与权重加载：`config.json`、`model_load.json`、`run_summary.json`；
- 聚合统计：`summary.json`、`report.md`；
- 图表：`mse_by_condition.png`、`horizon_error.png`；
- 复现命令：[`EVALUATION.md`](EVALUATION.md)；
- 完整性哈希：[`reports/ARTIFACT_MANIFEST.json`](reports/ARTIFACT_MANIFEST.json)。
