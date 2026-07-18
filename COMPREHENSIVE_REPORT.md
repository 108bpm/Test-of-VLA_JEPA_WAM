# latent_world_model × VLA-JEPA × LIBERO 汇报版综合报告

> **后续审计更正：** 本报告保留原 F0–F5 数值用于结果溯源。全链路复核
> 后确认了训练匹配条件缺失、strict/joint 表示分布偏移、旧 H3 时间错位
> 和上游双视角训练拼接错误。涉及实验有效性与模型能力的解释，请以
> [`EXPERIMENT_AUDIT_REPORT.md`](EXPERIMENT_AUDIT_REPORT.md) 为准。

> 本报告基于已采集的 v3 数据集、阶段 0/1 筛选结果、X0 补充对照和正式半量（130×5）结果生成。所有模型权重被冻结，没有重新训练 encoder 或 predictor；分析脚本只读取输入文件。

## 1. 摘要与结论

本次正式评估使用 **650 条 rollout、1950 条窗口**（每条 rollout 的 early/middle/late 三个时段）和六个条件，共 11700 条条件结果。直接的 future-latent MSE 是主要指标；persistence 只作为辅助参照，不替代直接误差。

核心结论：

- 严格因果、正确 latent action 的 F0 C1→H1 直接 MSE 为约 6.27；在本数据和当前冻结 checkpoint 上，它没有显示出比简单保持当前 latent 更准确的能力。
- 增加过去三步上下文（F1）反而使 H1 误差增加；把 action 置零（F3）有很小但统计稳定的下降，same-task/stage shuffle（F4）与正确 action 基本相同。这说明当前实验中没有观测到可辨识的 action-conditioned 增益。
- 原始联合 8 帧输入（F5）显著更低，但该输入包含被预测时刻之后的帧；它是未来信息泄漏对照，不能被解释为严格因果能力。X0 补充结果同样支持这一点。
- F2 的自回归多步误差不是单调上升：H1→H2 先下降，H3 又上升。它反映的是滚动 predictor 与目标窗口的共同统计，不应直接解释为“每一步都累积同样的误差”。
- LIBERO-90 失败 rollout 被保留，因为失败轨迹仍然包含合法的动作—视觉序列；成功/失败差异只作描述性分层，不当作因果结论。

## 2. 数据采集与质量控制

数据来自五个标准 LIBERO suite 的 VLA-JEPA 推理 rollout。每条记录保存双视角 RGB、状态、执行动作、policy query 帧、24×2048 latent action tokens 和对应的 7×7 unnormalized action chunk；视频与 HDF5 以 suite/task/episode 三元组配对。

| suite | rollouts | successful | success rate | mean frames | mean queries | video MB |
|---|---|---|---|---|---|---|
| libero_spatial | 100 | 100 | 1.0000 | 100.1600 | 14.7000 | 8.0352 |
| libero_object | 100 | 100 | 1.0000 | 132.1300 | 19.3200 | 8.9626 |
| libero_goal | 100 | 100 | 1.0000 | 107.6100 | 15.8600 | 7.9739 |
| libero_90 | 900 | 182 | 0.2022 | 356.1178 | 51.6500 | 207.8545 |
| libero_10 | 100 | 98 | 0.9800 | 248.6500 | 36.0000 | 19.2547 |

总计：1300 条 rollout，580 条成功（44.62%），1300 个匹配视频。manifest 记录 HDF5 约 57.25 GB、视频约 252 MB；完整性检查为重复 0、无效 0、缺失配对 0、孤立视频 0。

LIBERO-90 的成功率明显较低（约 20%），但这是采集结果而不是数据损坏；其失败轨迹仍按预先约定保留。正式集从每个 task 确定性选择 episode 0、2、4、6、8，因而每个 task 有 5 条 rollout，避免因为先后采集顺序或随机抽样改变实验集。

正式子集按 suite：

| suite | formal rollouts | successful | success rate | mean frames | mean queries |
|---|---|---|---|---|---|
| libero_spatial | 50 | 50 | 1.0000 | 100.1400 | 14.7400 |
| libero_object | 50 | 50 | 1.0000 | 131.4800 | 19.2000 |
| libero_goal | 50 | 50 | 1.0000 | 107.8400 | 15.8600 |
| libero_90 | 450 | 96 | 0.2133 | 354.0400 | 51.3400 |
| libero_10 | 50 | 50 | 1.0000 | 245.4200 | 35.5200 |

## 3. 实验设计与可复现性

评估以 query 帧为时间锚点。8 帧经过 tubelet=2 编码为 z0…z3；当前状态为 z2，目标为 z3。strict_causal 为每个 latent block 单独构造不包含未来帧的 clip，并在 episode 起点左填充；original_joint 使用原始连续 8 帧编码，仅用作泄漏对照。H3 是冻结 predictor 的自回归滚动，不是重新训练的多步 head。

正式条件：

| 条件 | 视觉输入 | latent action | 目标 | 主要目的 |
|---|---|---|---|---|
| F0 | strict causal，双视角，C1（当前 z2） | 当前窗口正确 action（当前 query 的 g2） | strict causal 当前窗口真实未来 latent（z3） | 严格因果主结果 |
| F1 | strict causal，双视角，C3（z0,z1,z2） | 当前窗口正确 action（g0,g1,g2） | 当前窗口真实未来 latent（z3） | 检验过去多帧上下文是否改善 H1 |
| F2 | strict causal，双视角，C1（当前 z2） | 正确 action 的自回归序列（g2→下一 query 的 g0→g1） | strict causal 的连续未来 latent（z3,z4,z5） | 检验单帧上下文的多步滚动预测 |
| F3 | strict causal，双视角，C1（当前 z2） | zero action（替换为全零 action group） | 与 F0 相同的 strict-causal future latent（z3） | 检查预测是否依赖当前 action |
| F4 | strict causal，双视角，C1（当前 z2） | 同 task、同 stage、其他 episode 的 action（确定性循环配对） | 与 F0 相同的 strict-causal future latent（z3） | 检查 action 是否提供样本相关的 transition 信息 |
| F5 | original joint，双视角，连续 8 帧共同编码；取 C1 的 z2 | 当前窗口正确 action（当前 query 的 g2） | 与 F0 相同的 strict-causal future latent（z3） | 检查非因果联合编码带来的未来信息泄漏影响 |

F4 与 F0 只替换 action，视觉输入、当前 latent、目标和 H1 预测步长保持一致。F4 的 action 来自完整索引中同 suite、同 task、同 stage 的其他 episode，并按 episode id 做确定性循环配对；因此保留了任务和阶段的大致分布，但破坏了 action 与当前视觉状态之间的对应关系。若 F0 明显优于 F4，说明 action 可能包含样本相关的 transition 信息；若两者接近，则说明当前模型没有表现出可检测的 action-specific 增益，或者 action-token 时间对齐仍需检查。

F5 与 F0 使用相同的 C1、正确 action 和 strict-causal 目标，区别只在视觉编码协议。original_joint 将连续 8 帧同时送入时序 encoder；即使 predictor 最后只取名义上的 z2，联合 encoder 的时空建模仍可能让 z2 表示接触到后续帧信息。因此 F5 是非因果信息泄漏对照，而不是严格推理性能；F5 的误差下降只能说明未来视觉信息会让预测更容易，不能作为 causal world model 能力结论。

阶段漏斗：阶段 0 完成 smoke、shape/dtype/finite、断点续跑和 predictor parity；阶段 1 在每个 task 取一条 rollout，运行 S0–S9 控制；筛选后运行 X0（original joint+C3→H1）作为定向补充；最后运行四路 shard 合并的正式半量。正式 JSONL 共有 11,700 条结果（1,950 windows×6），所有条件和行 ID 完整且无重复。

独立 predictor 与 VLA-JEPA source 的 parity artifact 显示输入 `[1,768,2048]`、action `[1,24,2048]`、输出 `[1,768,2048]`，max absolute difference=0、mean absolute difference=0、allclose=true。

## 4. 正式结果：直接误差为主

| condition | n | mean MSE | CI low | CI high | median | Q25 | Q75 | normalized MSE | persistence ratio | token cosine | delta cosine | retrieval top1 | retrieval top5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 1950 | 6.2735 | 6.2364 | 6.3104 | 6.2424 | 6.0553 | 6.4608 | 0.7133 | 1.4282 | 0.5694 | 0.4120 | 0.0010 | 0.0026 |
| F1 | 1950 | 6.5445 | 6.5050 | 6.5857 | 6.5320 | 6.3145 | 6.7607 | 0.7442 | 1.4909 | 0.5585 | 0.4058 | 0.0005 | 0.0036 |
| F2 | 1950 | 6.5476 | 6.5209 | 6.5752 | 6.5481 | 6.3900 | 6.6984 | 0.7436 | 1.1768 | 0.5519 | 0.4486 | 0.0005 | 0.0026 |
| F3 | 1950 | 6.2227 | 6.1846 | 6.2593 | 6.1872 | 6.0004 | 6.4094 | 0.7075 | 1.4168 | 0.5728 | 0.4133 | 0.0005 | 0.0026 |
| F4 | 1950 | 6.2734 | 6.2402 | 6.3117 | 6.2442 | 6.0525 | 6.4619 | 0.7133 | 1.4282 | 0.5694 | 0.4120 | 0.0005 | 0.0021 |
| F5 | 1950 | 5.1333 | 5.0969 | 5.1719 | 5.0927 | 4.9263 | 5.3077 | 0.5835 | 0.5385 | 0.6539 | 0.6900 | 0.0000 | 0.0031 |

F0 的直接误差是本报告的首要答案：模型输出与真实未来 latent 的平均平方差约为 6.27。它不依赖 persistence 定义，直接回答“预测 latent 与真实未来 latent 相差多大”。L1、RMSE、normalized MSE、token cosine、delta cosine 和 retrieval 是同一误差的补充视角；retrieval 的随机 top-1/top-5 参考分别是 1/1950≈0.00051 和 5/1950≈0.00256。

### 4.1 注册的配对差异

| comparison (left−right) | mean diff | CI low | CI high | median diff | left better | paired d | Holm p |
|---|---|---|---|---|---|---|---|
| F0-persistence | 1.8078 | 1.7407 | 1.8746 | 1.7662 | 0.0026 | 2.6901 | 0.0000 |
| F1-F0 | 0.2709 | 0.2531 | 0.2888 | 0.2690 | 0.0451 | 1.7081 | 0.0000 |
| F3-F0 | -0.0509 | -0.0552 | -0.0463 | -0.0501 | 0.8513 | -1.0113 | 0.0000 |
| F4-F0 | -0.0001 | -0.0012 | 0.0009 | -0.0003 | 0.5077 | -0.0076 | 0.8460 |
| F5-F0 | -1.1402 | -1.1519 | -1.1274 | -1.1386 | 1.0000 | -8.3319 | 0.0000 |

差异按同一 window 配对，再做 task→rollout/window 的层级 bootstrap；负值表示左侧误差更小。F4 的置信区间跨过 0，说明 shuffle 与正确 action 几乎无法区分；F3 的差异很小，虽统计稳定但实际效应有限；F5 的差异很大却不具备因果解释，因为协议不同且含未来帧。

### 4.2 persistence 的正确定位

persistence 是把当前 z2 原样当作未来 latent 的参照。它不是评估目标，也不是要求 world model 必须超过的唯一标准；直接 MSE 才是主要结果。persistence ratio=MSE/persistence MSE 只回答一个附加问题：模型是否比“未来不变”的简单预测更好。ratio<1 表示相对该简单参照更好，ratio>1 表示更差。报告同时保留 persistence，是为了判断 6.27 这个绝对误差在当前 latent 变化尺度下是否有实际增益，而不是把它混入模型定义。

F0 ratio 约 1.43；这意味着在严格 causal 条件下，模型误差约为保持 z2 的 1.43 倍。但无论 ratio 如何，F0 的 6.27 仍然是与真实未来 latent 的直接误差。F5 ratio<1 主要对应未来信息泄漏对照，不能据此宣称 causal predictor 已经成功。

## 5. 时间、任务和运动量分层

### 5.1 自回归 horizon

| horizon | n | mean MSE | median | Q25 | Q75 | min | max |
|---|---|---|---|---|---|---|---|
| 1 | 1950 | 6.2735 | 6.2424 | 6.0553 | 6.4608 | 5.3069 | 7.3115 |
| 2 | 1950 | 5.9020 | 5.8906 | 5.7241 | 6.0800 | 5.1918 | 6.8443 |
| 3 | 1950 | 6.5476 | 6.5481 | 6.3900 | 6.6984 | 5.7796 | 7.7880 |

F2 使用 H1/H2/H3 的同一条滚动轨迹。H2 低于 H1 不应被解读为模型在未来更远处一定更准：目标帧和动作窗口的运动分布不同，且这里只抽取 early/middle/late 三个窗口。真正稳定的结论是 H3 最后一阶段仍处于约 6.55 的误差水平，未呈现可靠的多步预测优势。

### 5.2 suite 分层

| condition | suite | n | mean MSE | median | persistence ratio |
|---|---|---|---|---|---|
| F0 | libero_10 | 150 | 6.0440 | 6.0330 | 1.3513 |
| F0 | libero_90 | 1350 | 6.3210 | 6.2981 | 1.4505 |
| F0 | libero_goal | 150 | 6.2221 | 6.2134 | 1.3940 |
| F0 | libero_object | 150 | 6.2541 | 6.2541 | 1.4514 |
| F0 | libero_spatial | 150 | 6.1468 | 6.1374 | 1.3158 |
| F1 | libero_10 | 150 | 6.2720 | 6.2513 | 1.4032 |
| F1 | libero_90 | 1350 | 6.5639 | 6.5464 | 1.5073 |
| F1 | libero_goal | 150 | 6.4980 | 6.4933 | 1.4569 |
| F1 | libero_object | 150 | 6.7504 | 6.7757 | 1.5677 |
| F1 | libero_spatial | 150 | 6.4829 | 6.4927 | 1.3883 |
| F2 | libero_10 | 150 | 6.4439 | 6.4189 | 1.0996 |
| F2 | libero_90 | 1350 | 6.5477 | 6.5407 | 1.2042 |
| F2 | libero_goal | 150 | 6.6585 | 6.6562 | 1.1169 |
| F2 | libero_object | 150 | 6.5118 | 6.5156 | 1.1676 |
| F2 | libero_spatial | 150 | 6.5757 | 6.5792 | 1.0763 |
| F3 | libero_10 | 150 | 5.9888 | 5.9728 | 1.3391 |
| F3 | libero_90 | 1350 | 6.2754 | 6.2450 | 1.4401 |
| F3 | libero_goal | 150 | 6.1604 | 6.1506 | 1.3806 |
| F3 | libero_object | 150 | 6.1894 | 6.1956 | 1.4367 |
| F3 | libero_spatial | 150 | 6.0781 | 6.0678 | 1.3012 |
| F4 | libero_10 | 150 | 6.0434 | 6.0354 | 1.3512 |
| F4 | libero_90 | 1350 | 6.3210 | 6.2952 | 1.4504 |
| F4 | libero_goal | 150 | 6.2209 | 6.2039 | 1.3938 |
| F4 | libero_object | 150 | 6.2540 | 6.2545 | 1.4514 |
| F4 | libero_spatial | 150 | 6.1472 | 6.1389 | 1.3159 |
| F5 | libero_10 | 150 | 4.9172 | 4.9061 | 0.5215 |
| F5 | libero_90 | 1350 | 5.1924 | 5.1733 | 0.5402 |
| F5 | libero_goal | 150 | 5.0377 | 5.0295 | 0.5393 |
| F5 | libero_object | 150 | 5.0592 | 5.0581 | 0.5428 |
| F5 | libero_spatial | 150 | 4.9876 | 4.9899 | 0.5350 |

F0 的 suite 间差异约在 6.04–6.32，量级小于 F5 泄漏对照相对于 F0 的下降；这支持“协议/信息可见性”是首要影响因素，但 suite 分层仍是描述性结果，不能据此归因于某类任务机制。

### 5.3 时间阶段、成功状态、latent motion 和 action scale

| condition | stage | n | mean MSE | median |
|---|---|---|---|---|
| F0 | early | 650 | 6.1618 | 6.1306 |
| F0 | late | 650 | 6.3583 | 6.3423 |
| F0 | middle | 650 | 6.3005 | 6.2903 |
| F1 | early | 650 | 6.4560 | 6.4559 |
| F1 | late | 650 | 6.6410 | 6.6475 |
| F1 | middle | 650 | 6.5365 | 6.5396 |
| F2 | early | 650 | 6.5294 | 6.5327 |
| F2 | late | 650 | 6.5602 | 6.5550 |
| F2 | middle | 650 | 6.5533 | 6.5574 |
| F3 | early | 650 | 6.1100 | 6.0747 |
| F3 | late | 650 | 6.3050 | 6.2852 |
| F3 | middle | 650 | 6.2531 | 6.2442 |
| F4 | early | 650 | 6.1623 | 6.1247 |
| F4 | late | 650 | 6.3578 | 6.3341 |
| F4 | middle | 650 | 6.3003 | 6.2888 |
| F5 | early | 650 | 5.0248 | 5.0136 |
| F5 | late | 650 | 5.2240 | 5.1943 |
| F5 | middle | 650 | 5.1511 | 5.1204 |

| condition | success | n | mean MSE | median |
|---|---|---|---|---|
| F0 | False | 1062 | 6.3739 | 6.3566 |
| F0 | True | 888 | 6.1536 | 6.1585 |
| F1 | False | 1062 | 6.6183 | 6.6028 |
| F1 | True | 888 | 6.4562 | 6.4717 |
| F2 | False | 1062 | 6.5566 | 6.5424 |
| F2 | True | 888 | 6.5369 | 6.5530 |
| F3 | False | 1062 | 6.3312 | 6.3139 |
| F3 | True | 888 | 6.0929 | 6.0877 |
| F4 | False | 1062 | 6.3735 | 6.3535 |
| F4 | True | 888 | 6.1537 | 6.1561 |
| F5 | False | 1062 | 5.2490 | 5.2321 |
| F5 | True | 888 | 4.9949 | 4.9948 |

| condition | motion quartile | n | mean MSE | mean target Δ RMS |
|---|---|---|---|---|
| F0 | Q1 | 488 | 6.3744 | 1.9420 |
| F0 | Q2 | 487 | 6.2608 | 2.0731 |
| F0 | Q3 | 487 | 6.2312 | 2.1542 |
| F0 | Q4 | 488 | 6.2277 | 2.2678 |
| F1 | Q1 | 488 | 6.6876 | 1.9420 |
| F1 | Q2 | 487 | 6.5565 | 2.0731 |
| F1 | Q3 | 487 | 6.5005 | 2.1542 |
| F1 | Q4 | 488 | 6.4332 | 2.2678 |
| F2 | Q1 | 488 | 6.4905 | 1.9420 |
| F2 | Q2 | 487 | 6.5234 | 2.0731 |
| F2 | Q3 | 487 | 6.5799 | 2.1542 |
| F2 | Q4 | 488 | 6.5968 | 2.2678 |
| F3 | Q1 | 488 | 6.3355 | 1.9420 |
| F3 | Q2 | 487 | 6.2106 | 2.0731 |
| F3 | Q3 | 487 | 6.1728 | 2.1542 |
| F3 | Q4 | 488 | 6.1717 | 2.2678 |
| F4 | Q1 | 488 | 6.3736 | 1.9420 |
| F4 | Q2 | 487 | 6.2604 | 2.0731 |
| F4 | Q3 | 487 | 6.2316 | 2.1542 |
| F4 | Q4 | 488 | 6.2279 | 2.2678 |
| F5 | Q1 | 488 | 5.2295 | 1.9420 |
| F5 | Q2 | 487 | 5.1193 | 2.0731 |
| F5 | Q3 | 487 | 5.0850 | 2.1542 |
| F5 | Q4 | 488 | 5.0992 | 2.2678 |

| condition | action quartile | n | mean MSE | mean action scale |
|---|---|---|---|---|
| F0 | Q1 | 488 | 6.3086 | 0.1334 |
| F0 | Q2 | 487 | 6.2902 | 0.2696 |
| F0 | Q3 | 487 | 6.2899 | 0.3469 |
| F0 | Q4 | 488 | 6.2056 | 0.4475 |
| F1 | Q1 | 488 | 6.5431 | 0.1334 |
| F1 | Q2 | 487 | 6.5653 | 0.2696 |
| F1 | Q3 | 487 | 6.5311 | 0.3469 |
| F1 | Q4 | 488 | 6.5384 | 0.4475 |
| F2 | Q1 | 488 | 6.5213 | 0.1334 |
| F2 | Q2 | 487 | 6.6123 | 0.2696 |
| F2 | Q3 | 487 | 6.5319 | 0.3469 |
| F2 | Q4 | 488 | 6.5251 | 0.4475 |
| F3 | Q1 | 488 | 6.2576 | 0.1334 |
| F3 | Q2 | 487 | 6.2260 | 0.2696 |
| F3 | Q3 | 487 | 6.2443 | 0.3469 |
| F3 | Q4 | 488 | 6.1629 | 0.4475 |
| F4 | Q1 | 488 | 6.3076 | 0.1334 |
| F4 | Q2 | 487 | 6.2918 | 0.2696 |
| F4 | Q3 | 487 | 6.2893 | 0.3469 |
| F4 | Q4 | 488 | 6.2052 | 0.4475 |
| F5 | Q1 | 488 | 5.1943 | 0.1334 |
| F5 | Q2 | 487 | 5.1327 | 0.2696 |
| F5 | Q3 | 487 | 5.1293 | 0.3469 |
| F5 | Q4 | 488 | 5.0770 | 0.4475 |

F0 的 late 阶段误差略高；但 latent-change RMS 分箱本身并不单调（F0 的最高 motion 分箱反而略低），而 target variance 与误差的相关性更强。分层只是定位误差集中在哪里，不能证明“运动量导致误差”。

### 5.4 相关性

| predictor | n | Pearson r | Spearman r | mean x | mean MSE |
|---|---|---|---|---|---|
| target_delta_rms | 1950 | -0.1729 | -0.1683 | 2.1093 | 6.2735 |
| action_norm | 1950 | -0.1253 | -0.1436 | 0.2993 | 6.2735 |
| query_frame | 1950 | 0.3946 | 0.3776 | 143.9344 | 6.2735 |
| target_variance | 1950 | 0.8442 | 0.8154 | 8.7913 | 6.2735 |

相关系数是 F0 window-level 的描述性诊断；它不替代配对比较，也不控制 task、stage 等混杂因素。

## 6. 阶段 1 控制与泄漏诊断

| condition | n | mean MSE | persistence ratio | token cosine |
|---|---|---|---|---|
| S0 | 390 | 6.2601 | 1.4235 | 0.5700 |
| S1 | 390 | 6.5248 | 1.4843 | 0.5596 |
| S2 | 390 | 6.5351 | 1.1716 | 0.5524 |
| S3 | 390 | 5.5962 | 1.0013 | 0.6198 |
| S4 | 390 | 6.2102 | 1.4123 | 0.5734 |
| S5 | 390 | 6.2607 | 1.4236 | 0.5700 |
| S6 | 390 | 6.2604 | 1.4235 | 0.5700 |
| S7 | 390 | 7.0369 | 2.7752 | 0.5444 |
| S8 | 390 | 5.5576 | 0.9049 | 0.5977 |
| S9 | 390 | 5.1309 | 0.5391 | 0.6537 |

阶段 1 的 S0–S9 只用于筛选和发现值得补充的方向，不与正式 F0–F5 混合做主要显著性检验。X0（original joint+C3→H1）MSE 约 3.33，相对 strict C3 的筛选结果下降约 49%，与 F5 的显著下降一起说明联合编码能看到未来帧，不能用于证明 causal world model 能力。

## 7. 图像与数据表

图像由本分析脚本在同一输出目录生成：

- [condition_mse_ci.png](evaluation_outputs/deep_analysis/condition_mse_ci.png)
- [persistence_ratio_box.png](evaluation_outputs/deep_analysis/persistence_ratio_box.png)
- [paired_effects_forest.png](evaluation_outputs/deep_analysis/paired_effects_forest.png)
- [horizon_growth.png](evaluation_outputs/deep_analysis/horizon_growth.png)
- [suite_mse_heatmap.png](evaluation_outputs/deep_analysis/suite_mse_heatmap.png)
- [stage_success_mse.png](evaluation_outputs/deep_analysis/stage_success_mse.png)
- [motion_quartile_mse.png](evaluation_outputs/deep_analysis/motion_quartile_mse.png)
- [screening_controls_mse.png](evaluation_outputs/deep_analysis/screening_controls_mse.png)
- [collection_suite_summary.png](evaluation_outputs/deep_analysis/collection_suite_summary.png)
- [mse_vs_motion.png](evaluation_outputs/deep_analysis/mse_vs_motion.png)

CSV 表：`condition_summary.csv`、`paired_comparisons.csv`、`horizon_summary.csv`、`suite_strata.csv`、`stage_success_strata.csv`、`motion_strata.csv`、`action_strata.csv`、`correlations.csv`、`collection_suite_summary.csv`、`formal_collection_suite_summary.csv`、`screening_summary.csv`。完整 JSON 在 `deep_summary.json`。

## 8. 局限性与下一步

1. 评估是在冻结 checkpoint 上进行的；没有训练/微调，因此结果回答的是“现有模型在该数据和对齐方式上的能力”，不是可达到的上限。
2. 预测目标是 V-JEPA2 latent，不是 RGB；没有公开兼容 decoder，不能直接报告像素级视频质量。
3. formal half 是 130×5 的确定性子集，结果应推广到同一采集分布而非任意 LIBERO 数据。完整 1300 rollout 数据仍保留，可在资源允许时扩大正式评估。
4. Causal clip、query 帧和 action-token 分组的时间对齐是关键风险点。下一步应优先做人工/可视化对齐审计、用 action permutation 检验 token 信息是否可识别、以及按 task 做更严格的 held-out 分析。
5. 若后续要改进模型，应先保留本报告的 F0/F3/F4/F5 作为回归基线，再单独设计训练实验；不要用 original_joint 结果作 causal 训练目标或能力结论。

## 9. 复现入口与版本

- 数据集：`datasets/vla_jepa_libero130_v3/README.md`、`manifest.json`。
- 评估协议和命令：`EVALUATION.md`。
- 正式结果：`evaluation_outputs/formal_half/metrics.jsonl`、`summary.json`。
- 本报告脚本：`latent_world_model/evaluation/deep_analysis.py`。
- 运行示例见脚本 docstring；输出目录默认 `evaluation_outputs/deep_analysis`。
- 代码版本：当前工作区 Git 历史中的评估实现提交；HDF5、MP4、checkpoint 和大型 memmap 均不纳入 Git。
