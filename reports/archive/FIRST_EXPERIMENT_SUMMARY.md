# 第一次正式实验摘要（归档）

> 本文件归档补充 joint-C3 实验之前的 `FINAL_REPORT.md` 核心内容。完整原始分析仍以仓库根目录的 [`COMPREHENSIVE_REPORT.md`](../../COMPREHENSIVE_REPORT.md) 为准；后续协议审计见 [`EXPERIMENT_AUDIT_REPORT.md`](../../EXPERIMENT_AUDIT_REPORT.md)。

## 结论

在冻结的 VLA-JEPA LIBERO predictor 和冻结的 V-JEPA2 encoder 上，严格因果协议的正式结果没有优于 persistence：F0 的平均 `persistence_ratio=1.4282`、MSE 为 `6.2735`。same-task/stage shuffle 与正确 latent action 几乎相同，C3 历史上下文也没有改善 H1，反而使 MSE 增加约 `0.2709`。

## 数据与协议

- 数据源：`datasets/vla_jepa_libero130_v3`，1300 个已验证 HDF5/video 对；
- 正式子集：每 task 的 episode `0,2,4,6,8`，共 130 tasks × 5 rollouts = 650 rollouts；
- 每条 rollout 选择 early/middle/late 三个窗口，共 1950 windows；
- 六个正式条件，共 11700 条结果；
- checkpoint 和 encoder 全程冻结。

## 正式结果

| 条件 | 定义 | MSE | persistence ratio | token cosine |
|---|---|---:|---:|---:|
| F0 | strict C1→H1，正确 action | 6.2735 | 1.4282 | 0.5694 |
| F1 | strict C3→H1，正确 action | 6.5445 | 1.4909 | 0.5585 |
| F2 | 历史 strict C1→AR-H3 | 6.5476 | 1.1768 | 0.5519 |
| F3 | strict C1→H1，zero action | 6.2227 | 1.4168 | 0.5728 |
| F4 | strict C1→H1，同 task/stage shuffle | 6.2734 | 1.4282 | 0.5694 |
| F5 | original joint C1→strict H1 | 5.1333 | 0.5385 | 0.6539 |

配对比较：

| 比较 | 均值差 | 95% CI |
|---|---:|---|
| F0 − persistence | +1.8078 | [1.7434, 1.8728] |
| F1 − F0 | +0.2709 | [0.2536, 0.2891] |
| F3 − F0 | −0.0509 | [−0.0552, −0.0463] |
| F4 − F0 | −0.0001 | [−0.0012, 0.0009] |
| F5 − F0 | −1.1402 | [−1.1535, −1.1274] |

## 后续审计限定

- F0/F1 的 strict-causal H1 数值仍是用户目标的完整正式结果；
- 历史 F2 的跨 query H3 时间排程不严格对齐，不用于最终多步结论；
- F5 是 joint C1 输入对 strict target，不是完整的原生 joint-C3 teacher-forcing 测试；
- 第一次实验没有完整覆盖 `joint C3 → same-joint z1,z2,z3`，该缺口由第二次实验补齐。
