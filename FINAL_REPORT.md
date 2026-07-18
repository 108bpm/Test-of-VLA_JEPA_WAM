# latent_world_model 评估最终报告

## 结论先行

在冻结的 VLA-JEPA LIBERO predictor 和冻结的 V-JEPA2 encoder 上，严格因果协议的正式结果没有优于 persistence baseline：F0 的平均 `persistence_ratio=1.428`，MSE 为 6.274，而保持当前 latent 的对应误差更低。当前 checkpoint 的 action 条件在本数据上没有可检测的正向贡献；zero-action 反而小幅降低误差，same-task/stage shuffle 与正确 action 几乎相同。C3 历史上下文也没有改善 H1，反而使 MSE 增加约 0.271。原始 8 帧联合编码 F5 的 MSE 降到 5.133、`persistence_ratio=0.538`，但该协议明确包含未来帧信息，不能作为无泄漏能力结论。

因此，目前实现更像是“可复现、无泄漏的评估基线”，还不能证明冻结 world model 在严格因果 LIBERO 数据上有可靠的动作条件未来预测能力。

## 数据与协议

- 数据源：`datasets/vla_jepa_libero130_v3`，HDF5/video 对齐检查 1300/1300，索引 3900 个早/中/晚窗口。
- 按最新指令，正式集缩减为每 task 的 episode `0,2,4,6,8`：130 tasks × 5 rollouts = 650 rollouts，3 阶段窗口共 1950；六条件共 11700 条 per-window 结果。
- checkpoint：`VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`，只提取 `vj_predictor.*`，未训练或微调任何权重。
- `strict_causal` 为每个 latent block 单独构造不含未来帧的 8 帧 clip；`original_joint` 是原始连续 8 帧编码，仅用于泄漏对照。
- 8 帧、tubelet=2 产生 `z0...z3`；C1 使用当前 `z2` 和最后 action group 预测 `z3`，C3 使用 `z0,z1,z2`。H3 是冻结 predictor 的自回归滚动，不是新训练的 multi-horizon head。

## 正式结果（650 rollouts / 1950 windows）

| 条件 | 定义 | MSE | persistence ratio | token cosine | H1/H2/H3 MSE |
|---|---|---:|---:|---:|---|
| F0 | strict C1→H1，正确 action | 6.2735 | 1.4282 | 0.5694 | 6.2735 |
| F1 | strict C3→H1，正确 action | 6.5445 | 1.4909 | 0.5585 | 6.5445 |
| F2 | strict C1→AR-H3 | 6.5476 | 1.1768 | 0.5519 | 6.2735 / 5.9020 / 6.5476 |
| F3 | strict C1→H1，zero action | 6.2227 | 1.4168 | 0.5728 | 6.2227 |
| F4 | strict C1→H1，同 task/stage shuffle | 6.2734 | 1.4282 | 0.5694 | 6.2734 |
| F5 | original joint C1→H1 | 5.1333 | 0.5385 | 0.6539 | 5.1333 |

task→rollout hierarchical bootstrap 1000 次，95% CI 的预注册配对比较（左减右，负值表示左更好）：

| 比较 | 均值差 | 95% CI | Holm p |
|---|---:|---|---:|
| F0 − persistence | +1.8078 | [1.7434, 1.8728] | <0.001 |
| F1 − F0 | +0.2709 | [0.2536, 0.2891] | <0.001 |
| F3 − F0 | −0.0509 | [−0.0552, −0.0463] | <0.001 |
| F4 − F0 | −0.0001 | [−0.0012, 0.0009] | 0.774 |
| F5 − F0 | −1.1402 | [−1.1535, −1.1269] | <0.001 |

F0/F1/F2/F3/F4/F5 每个条件均为 1950 条、130 个 task；所有 MSE 和 latent 指标均 finite。检索 top-1/top-5 接近随机水平，说明当前预测表示没有稳定地识别对应 future sample。

## 阶段 0、筛选与补充

- smoke：20/20 条条件记录完成，覆盖 C1/C3、H1/AR-H3、zero/shuffle/offset、两视角/单视角、joint/strict 和断点续跑。
- predictor parity：独立 predictor 与 VLA-JEPA source 使用相同 checkpoint、相同输入，输出 shape `[1,768,2048]`，`max_abs=0`、`mean_abs=0`、`allclose=true`；见 `evaluation_outputs/stage0/predictor_parity.json`。
- 阶段 1：390 windows、S0–S9 全部 3900 条完成。C3、视角消融和 original-joint 的相对变化达到预注册阈值，因此只运行一个定向补充 `X0=original joint+C3→H1`。
- X0：390 条完成，MSE 3.3299；相对筛选 S1（strict C3）降低约 49.0%，配对 bootstrap 差 −3.1948，95% CI [−3.2419,−3.1498]。这进一步支持 original joint 的显著优势主要来自未来帧泄漏，而非严格因果历史能力。

## 分层观察

严格 F0 的 suite MSE 约为：LIBERO-10 6.044、LIBERO-SPATIAL 6.147、LIBERO-GOAL 6.222、LIBERO-OBJECT 6.254、LIBERO-90 6.321；成功 rollout 的 F0 MSE 6.154，失败 rollout 6.374。早/中/晚阶段 F0 分别为 6.162/6.301/6.358，后期略难。所有分层均是同一预测结果的描述性统计，没有扩展组合检验。

## 产物与复现

- 评估设计与命令：[`EVALUATION.md`](EVALUATION.md)
- 确定性索引：`evaluation_outputs/stage0/index.jsonl` 及 `.summary.json`
- 筛选结果：`evaluation_outputs/stage1/metrics.jsonl`、`report.md`
- 补充结果：`evaluation_outputs/stage1_supplemental/metrics.jsonl`
- 正式合并结果：`evaluation_outputs/formal_half/metrics.jsonl`、`summary.json`、`report.md`、`mse_by_condition.png`、`horizon_error.png`
- 四个原始正式 shard：`evaluation_outputs/formal_half_shard{0,1,2,3}`；合并检查无重复，11700/11700 条完整。
- 代码版本：Git commit `16d05e9`（评估代码、文档、测试和轻量数据 manifest）；HDF5/video、checkpoint、缓存和评估大文件均未纳入 Git。

所有 GPU 进程已停止；没有后台 LIBERO/server/runner 进程。后续若要提升严格因果结果，应优先检查 action-token 与 latent transition 的时间对齐、checkpoint 训练域差异和 predictor 权重是否适合当前数据，再考虑任何训练实验；本次任务没有进行训练。
