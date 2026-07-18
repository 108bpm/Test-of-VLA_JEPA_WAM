# VLA-JEPA × LIBERO latent_world_model 实验框架

本文用于说明实验如何设计、如何组织变量以及如何复现研究逻辑。它不包含具体代码、目录、命令、运行结果或结论；实现者可根据本框架自行编写采集器、数据加载器和评估程序。

## 1. 研究目标

研究 action-conditioned latent world model 在以下任务中的能力：

> 给定当前视觉 latent 和与当前时刻对应的 latent action，预测同一轨迹中未来时刻的视觉 latent。

评估对象是 latent 空间中的未来状态预测，不是 RGB 重建，也不是 LIBERO 任务成功率。encoder、latent action source 和 predictor 在本实验中均保持冻结，不重新训练或微调。

## 2. 数据采集框架

### 2.1 任务与 rollout

使用五个标准 LIBERO suite：

- LIBERO-SPATIAL；
- LIBERO-OBJECT；
- LIBERO-GOAL；
- LIBERO-90；
- LIBERO-10。

每个 task 运行固定数量的独立 rollout，并使用固定随机种子。正式评估从每个 task 的完整 rollout 集中确定性选择五条 episode，采用均匀间隔的 episode 编号，以避免抽样顺序影响实验集。

失败 rollout 也保留，只要它产生了完整的视觉—动作时间序列。成功状态用于后续分层统计，不作为数据是否有效的唯一标准。

### 2.2 每条 rollout 的信息

每条轨迹至少包含：

1. 双视角视频帧；
2. 机器人状态和实际执行动作；
3. policy query 的时间索引；
4. 每个 query 对应的 latent action tokens；
5. 与 latent action 对应的 action chunk；
6. task、episode、suite、seed 和 success 等元数据。

关键原则是：视频帧、query 时间、latent action 和执行动作必须共享同一个时间轴。不能只保存视频而事后猜测 action 的对应时刻，也不能用自然语言 instruction 代替 task identity。

“共享时间轴”不自动等于“latent action 是实际执行 action chunk 的确定性编码”。复现者必须检查具体 policy 架构：如果 world-model token 与 action head 使用不同的 hidden-token 槽位或随机采样头，应将二者描述为同一次 query 的共同输出，而不能宣称一一可逆对应。

### 2.3 数据配对与质量要求

视频和结构化记录按 `(suite, task, episode)` 一一对应。每个 query 必须同时存在：

```text
观察帧索引 ↔ latent action ↔ action chunk
```

在进入 latent 评估前，须检查：

- 视频和结构化记录一一对应；
- 帧数、query 数、latent action 数和 action chunk 数一致；
- 视觉帧、动作和 latent 中没有非法值；
- task/episode identity 唯一；
- 所有选中的 query 都有足够的未来帧用于 H1 和 H3。

## 3. 时间建模与变量定义

### 3.1 视觉 latent

固定使用 8 帧视频窗口和 tubelet size 2。encoder 将窗口表示为四个 latent block：

```text
z0, z1, z2, z3
```

其中 `z2` 定义为当前状态，`z3` 是一步未来目标。latent action `[24, 2048]` 按时间分为三个 action group：

```text
g0, g1, g2
```

每个 group 对应一个 latent transition。

这里的“对应”是 predictor 训练目标所规定的时间位置语义。若研究问题要求条件化于真实执行动作，还必须证明这些 group 明确接受了真实 action 监督或由真实 action 确定性编码得到。

### 3.2 上下文长度

- C1：只提供当前状态 `z2` 和 `g2`；
- C3：提供历史状态 `[z0,z1,z2]` 和动作 `[g0,g1,g2]`。

C1 与 C3 用于控制“单帧当前状态”和“过去多帧历史上下文”之间的差异。

### 3.3 预测 horizon

- H1：预测下一个 latent block；
- AR-H3：在同一个 policy query 的 8 帧时间网格内，从 `z0` 开始，将 predictor 连续调用三次，每一步把上一步预测结果放回上下文。

C1 的有效 H3 action sequence 为：

```text
当前 query 的 g0 → 当前 query 的 g1 → 当前 query 的 g2
```

其目标依次为 `z1→z2→z3`。H3 是冻结 predictor 的自回归滚动，不是单独训练的多步 prediction head。

不能在 policy query 周期与 latent stride 不整除时，直接用下一 query 的 action group 补到连续 latent 时间轴。例如 query 每 7 帧一次、tubelet stride 为 2 时，下一 query 的 `g0` 与当前窗口之后的第一个 2 帧 block 会错开一帧。若要预测窗口之外的 H3，必须重新采集与 latent stride 对齐的条件 token，或直接输入逐控制步的实际 action。

## 4. 因果视觉编码协议

### 4.1 Strict causal

每个被表示的 latent block 都从独立的 8 帧 clip 编码，clip 的结束位置不超过该 block 所代表的时间。episode 开始处不足 8 帧时使用左侧填充。这样 encoder 在生成当前状态 latent 时不会看到当前状态之后的帧。

### 4.2 Original joint

将连续 8 帧一次性送入时序 encoder，再取其中的当前 block 作为 predictor 输入。由于 encoder 在整个连续窗口上进行时空建模，当前 block 的表示可能受到后续帧影响。因此 original joint 只能作为非因果信息对照，不能与 strict causal 结果合并为同一种能力。

## 5. 正式实验条件矩阵

| 条件 | 视觉输入 | latent action | 目标 | 主要目的 |
|---|---|---|---|---|
| F0 | strict causal，双视角，C1（当前 z2） | 当前窗口正确 action（当前 query 的 g2） | strict causal 当前窗口真实未来 latent（z3） | 严格因果主结果 |
| F1 | strict causal，双视角，C3（z0,z1,z2） | 当前窗口正确 action（g0,g1,g2） | 当前窗口真实未来 latent（z3） | 检验过去多帧上下文的影响 |
| F2 | strict causal，双视角，C1（窗口起点 z0） | 当前 query 内正确 action 的自回归序列（g0→g1→g2） | strict causal 连续未来 latent（z1,z2,z3） | 检验时间对齐的多步滚动预测 |
| F3 | strict causal，双视角，C1（当前 z2） | zero action | 与 F0 相同的 strict-causal future latent（z3） | 控制 action 是否被使用 |
| F4 | strict causal，双视角，C1（当前 z2） | 同 suite、同 task、同 stage、其他 episode 的 action，确定性循环配对 | 与 F0 相同的 strict-causal future latent（z3） | 检查 action 是否包含样本相关 transition 信息 |
| F5 | original joint，双视角，连续 8 帧共同编码；取 C1 的 z2 | 当前窗口正确 action（当前 query 的 g2） | 与 F0 相同的 strict-causal future latent（z3） | 检查非因果联合编码造成的信息泄漏影响 |

### 5.1 F4 的配对规则

F4 与 F0 必须共享同一个视觉输入、当前状态、目标和预测 horizon。只替换 latent action。partner action 从同 suite、同 task、同 stage 的其他 episode 中选择，并使用固定排序和循环规则，使实验可复现。

这样可以尽量保持 task 和 stage 的动作分布，同时破坏 action 与当前视觉状态之间的样本级对应关系。F4 是 action-specific 信息的控制条件，不是随机改变所有数据因素的条件。

### 5.2 F5 的控制关系

F5 与 F0 使用相同的 C1、正确 action 和 strict-causal target，唯一主要变量是视觉编码协议：

```text
F0: 当前 latent 使用 strict causal encoder
F5: 当前 latent 使用 original joint encoder
```

因此 F5 用于测量“联合编码允许未来视觉信息参与表示”这一因素，不能被当作严格 causal predictor 的正式性能条件。

## 6. 分阶段实验流程

### Stage 0：接口验证

目的：验证模型加载、encoder/predictor shape、dtype/device、latent-action shape、strict/original 两条编码路径、所有控制条件和断点续跑逻辑。Stage 0 只使用极小样本，不进行统计推断。

### Stage 1：筛选矩阵

每个 task 选择一条确定性 rollout，并覆盖三个时间 stage。运行以下方向：

- C1 与 C3；
- H1 与同一 query 内时间对齐的 AR-H3；
- 正确 action、zero action、错位 action、shuffle action；
- agentview-only 与 wrist-only；
- strict causal 与 original joint。

筛选阶段用于确认正式实验应保留的变量和对照；它与正式半量数据分开保存、分开统计。

### 定向补充

当筛选阶段需要单独验证“上下文长度”和“联合编码”是否相互作用时，增加 original joint + C3 + H1 的定向条件。定向补充只回答预先指定的问题，不改变正式 F0–F5 矩阵。

### Stage 2：正式半量

每个 task 确定性选择五条 episode，使用 F0–F5 六个正式条件。每条 rollout 选择 early、middle、late 三个 query window，所有条件共享同一个 window identity，以便配对比较。

## 7. 指标和统计框架

### 7.1 直接预测误差

主要指标是预测 latent 与真实未来 latent 的直接差异：

- MSE；
- L1；
- RMSE；
- normalized MSE。

### 7.2 表示和运动诊断

辅助记录：

- token cosine；
- delta cosine；
- prediction/target variance；
- delta norm ratio；
- action scale；
- target latent change scale；
- gripper category；
- retrieval top-1/top-5。

### 7.3 Persistence 的定位

将当前 latent 原样作为未来 latent 的简单参照，计算 `persistence_mse` 和 `persistence_ratio`。它是解释当前 latent 变化尺度的辅助诊断，不替代直接 future-latent MSE，也不改变主要实验目标。

### 7.4 聚合和比较

- 观察单位是一个 stage window；
- F0/F4 等条件使用相同 window identity 做 paired comparison；
- 任务和 rollout identity 必须保留到统计阶段；
- 置信区间采用 task→rollout/window hierarchical bootstrap；
- 正式注册比较为 F0-persistence、F1-F0、F3-F0、F4-F0、F5-F0；
- 多重比较只对注册比较做 Holm correction；
- 分层只使用 suite、success、stage、latent-change quartile、action scale 和 gripper category，不进行未注册的全组合排列。

## 8. 复现时必须固定的因素

复现者应固定并记录：

1. LIBERO suite、task 列表、每 task rollout 数和 episode 选择规则；
2. simulator/policy 随机种子；
3. VLA-JEPA policy checkpoint、V-JEPA2 encoder checkpoint 和 predictor checkpoint；
4. query-frame 与 latent-action 的时间对齐方式；
5. 8 帧窗口长度、tubelet size、C1/C3 定义和 H1/H3 定义；
6. strict causal/original joint 的编码协议；
7. F4 partner action 的分组、排序和循环规则；
8. 正式条件列表和阶段划分；
9. bootstrap seed、重复次数和注册比较；
10. 模型是否冻结、device/dtype 和 batch/concurrency 策略。

任何实现都应输出一份运行配置，至少包含上述因素和代码版本，以便第三方判断两次实验是否真正使用了同一协议。

## 9. 复现边界

本框架只规定研究问题、数据时间轴、控制变量和统计规则。具体的 LIBERO collector、HDF5 reader、模型调用、并行策略、日志格式和可视化实现由复现者自行编写，但必须保持本协议中定义的：

- rollout identity；
- query/action/frame 对齐；
- strict causal 和 original joint 的区别；
- F0–F5 条件矩阵；
- 观察单位和配对关系；
- 统计与多重比较规则。
