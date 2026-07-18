# VLA-JEPA latent world model 全链路审计报告

## 1. 审计目的与结论摘要

本次审计针对“为什么现有结果显示 latent world model 几乎不起作用”这一问题，逐层检查了：

1. LIBERO 数据采集与文件完整性；
2. policy query、latent action、action chunk、执行动作和视频的时间对应；
3. 在 LIBERO 中重新执行已记录动作后的状态与画面；
4. 用原始 VLA-JEPA 在线重新计算 latent action；
5. 独立 predictor 的权重加载和源码数值一致性；
6. VLA-JEPA 的训练数据构造、双视角融合和 teacher-forcing 目标；
7. F0–F5 的上下文、目标、action 和 horizon 实现。

审计得到的核心判断是：

- **采集数据本身是正确的。** 文件、查询索引、动作块、实际执行动作、LIBERO 状态、双视角画面和 latent action 均建立了直接的数值证据链，没有发现错位或重复数据导致的评估错误。
- **旧正式评估显著高估了模型在训练分布内的误差。** 覆盖全部 130 个 task 的诊断中，训练完全匹配协议的 MSE 约为 3.44，而 strict-causal C3 的 MSE 约为 6.35。旧 F0/F1/F5 没有任何一项完整复现 checkpoint 的训练输入与训练目标。
- **这并不能说明模型具备所需的严格因果 world-model 能力。** 原始 V-JEPA2 encoder 是双向时空注意力；训练时联合编码的 context latent 已经受到目标未来帧影响。strict-causal latent 又与训练 latent 存在很大的分布差异。
- **现有 latent-action 条件在训练与评估之间是一致的。** 上游训练和本次评估都向 predictor 输入同一来源、相同形状和相同语义的 24 个 Qwen hidden-state tokens；在线复算也证明保存值与原始 VLA-JEPA 输出逐元素一致。因此，它们是否是实际执行动作的确定性编码，不影响对既有 checkpoint 的同协议评价。
- **但 checkpoint 对样本相关 latent action 的变化几乎不敏感。** 将有明显数值差异的 action tokens 换成其他任务/样本的 tokens 后，预测几乎不变。这个结果不否定 latent tokens 含有动作或意图信息，只说明当前 predictor 在被审计协议下没有表现出对这些样本差异的明显利用。zero action 会改变预测，但 zero 是严重的分布外输入，不能替代 shuffle 对照。
- **原 VLA-JEPA 训练代码存在已确认的双视角 batch 拼接错误。** batch size 大于 1 时，它会将不同样本或错误视角的 encoder feature 拼接在一起。当前 checkpoint 的训练配置为每设备 batch size 32，因此该错误直接影响了 checkpoint 的 world-model 训练。
- **保存的 24 个 latent action tokens 应解释为上游定义的 learned action-conditioning representation。** 它们不是实际执行 7 步 action chunk 的一一编码，但本项目并不要求这种一一对应；只要保持与上游训练相同的 token 生成方式，它们就是合法的 predictor 条件。实际 action chunk 仍用于记录环境交互和时间对齐，而不是用来替换这 24 个训练条件 tokens。
- **旧 AR-H3 存在不可消除的时间错位。** policy 每 7 个控制帧查询一次，而一个 V-JEPA latent step 对应 2 帧；下一次 query 的 `g0/g1` 无法精确对应旧实现要求的 `q+8...q+11` 目标 latent。

因此，原先“模型完全不起作用”的说法需要拆开理解：旧评估协议确实有问题，latent-action 接口本身则保持了训练—评估一致性；但在修正协议后，checkpoint 仍未表现出对样本相关 latent-action 变化的明显敏感性。当前最值得优先处理的根因是上游 world-model 训练路径中的多视角 batch 融合错误，而不是本次 rollout 数据保存或 latent-action 接口替换错误。

## 2. 审计对象

| 对象 | 审计内容 |
|---|---|
| rollout 数据 | 1300 个 HDF5、1300 个 MP4、schema、identity、帧数、query 数 |
| 采集器 | 当前观测、query、保存、执行动作之间的调用先后顺序 |
| LIBERO | 固定 task、episode、seed、初始状态和 warm-up 后重放动作 |
| VLA policy | 原始 checkpoint 在线复算 `[24,2048]` world-model tokens |
| 独立模块 | encoder 多视角融合、predictor 权重加载、与上游 predictor 的数值 parity |
| 训练代码 | 8 帧采样、tubelet、C3 teacher forcing、双视角 batch 融合、action token 来源 |
| 评估代码 | strict/original、C1/C3、H1/H3、正确/zero/shuffle action 和目标定义 |

使用的冻结权重为：

- VLA-JEPA LIBERO checkpoint：`VLA-JEPA-LIBERO.pt`；
- V-JEPA2 encoder：ViT-L、256 px、tubelet size 2；
- predictor 前缀：`vj_predictor.*`，共 161,647,616 个参数，strict load 无 missing/unexpected keys。

## 3. 数据采集与处理审计

### 3.1 采集器调用顺序

每个 policy 控制步的实际顺序是：

```text
当前 LIBERO observation
  → 旋转并构造双视角 policy 输入
  → 每 7 步进行一次 policy query
  → 保存该 query 返回的 24 个 latent tokens 和 7 步 action chunk
  → 保存当前 observation/state 和本步将执行的 action
  → env.step(executed_action)
```

`query_frame_index` 在当前 observation 写入前取当前 `frame_count`；紧接着写入的 observation 正好位于该索引。因此 query frame 没有前移或后移一帧。

### 3.2 全量数据不变量

使用 `audit_collection` 对所有 1300 条 rollout 进行了逐元素检查：

| 检查项 | 结果 |
|---|---:|
| HDF5 记录 | 1300 |
| 唯一 `(suite, task, episode)` | 1300 |
| 总控制帧 | 379,361 |
| 总 policy query | 55,073 |
| 检查的 action 行 | 379,361 |
| action chunk 与执行动作不匹配 | 0 |
| 连续动作维度最大绝对误差 | 0.0 |
| gripper 不匹配 | 0 |
| 检查的 latent 数值 | 2,706,948,096 |
| NaN/Inf | 0 |
| 全零 query | 0 |
| MP4 帧数不匹配 | 0/1300 |
| 抽检视频帧 | 3900 |
| MP4 对 HDF5 像素 MAE | 平均 1.84，最大 2.36 |

MP4 像素差来自有损视频压缩；world-model 使用的是 HDF5 中无损保存的 RGB，而不是重新解码的 MP4。

### 3.3 LIBERO 环境级动作重放

从五个 suite 各选一条 rollout，包括成功和失败样本。重新构造 task、episode、seed 和初始 simulator state，执行与采集完全相同的 10 次 stabilization action，再逐步输入 HDF5 的 `executed_actions`。

共重放 990 个控制动作，结果为：

- 两个相机的每个 HDF5 帧与重新渲染帧逐像素完全相同；
- robot state 最大绝对误差约 `1.2e-7`；
- 成功轨迹在相同最后一步成功，失败轨迹保持失败；
- task instruction 全部一致。

这项验证直接排除了“视频不是这些 action 实际产生的”这一可能性。

### 3.4 在线复算 latent action

启动原始 VLA-JEPA policy server，使用 HDF5 中 query frame 的双视角图像、instruction 和 state，按采集器相同的 resize 与 RPC payload 重新推理。

- 五个 suite 的 query 0：5/5 逐元素完全一致；
- 五个 suite 的中段 query：5/5 逐元素完全一致；
- fresh token 与 HDF5 float16 token 的最大绝对误差为 0。

因此，HDF5 中的 latent tokens 确实来自所标记的 observation 和原始 checkpoint。

## 4. 模型提取与预测实现审计

### 4.1 predictor 权重与源码 parity

独立 predictor 和 VLA-JEPA 源 predictor 同时 strict-load 相同的 `vj_predictor.*` 权重，并输入同一组 `[1,768,2048]` context 与 `[1,24,2048]` actions：

```text
output max_abs = 0.0
output mean_abs = 0.0
allclose        = true（逐元素相等）
```

所以 predictor 提取、参数名转换或 checkpoint 加载不是误差来源。

### 4.2 正确的独立模块多视角融合

encoder 输入先按 sample-major 顺序展平：

```text
b0v0, b0v1, b1v0, b1v1, ...
```

独立模块恢复 `[B,V]` 两个轴后，再在 feature 维拼接，因此得到：

```text
sample 0 = b0v0 + b0v1
sample 1 = b1v0 + b1v1
```

本次增加了回归测试，固定验证 batch 大于 1 时不会混合样本。

## 5. 上游训练代码审计

### 5.1 实际 teacher-forcing 目标

训练数据为连续 8 帧双视角视频。tubelet size 2 后得到：

```text
z0 = frames q+0,q+1
z1 = frames q+2,q+3
z2 = frames q+4,q+5
z3 = frames q+6,q+7
```

训练代码固定使用：

```text
context = [z0,z1,z2]
actions = [g0,g1,g2]（24 tokens）
target  = [z1,z2,z3]
loss    = L1(prediction, target)
```

因此 released checkpoint 只在 C3、24 tokens、joint encoding 上训练过。C1 并不是其训练上下文。

### 5.2 原始 joint encoder 不是严格因果 encoder

当前 Transformers V-JEPA2 encoder attention 明确设置 `is_causal=False`，并且 forward 不传入时间 attention mask。一次联合编码 8 帧时，`z0,z1,z2` 都可注意到包括 `z3` 在内的整个 clip。

所以完全匹配训练目标的评估适合检查 checkpoint 是否复现其训练任务，但不能作为“只知道现在预测未来”的严格因果证据。

### 5.3 已确认的双视角 batch 融合错误

#### 5.3.1 错误发生在哪一步

VLA-JEPA 的一个训练 batch 首先具有：

```text
batch_videos: [B, V, T, C, H, W]
```

其中 `B` 是每设备 batch size，`V=2` 是相机视角数。源码随后执行：

```python
batch_videos = batch_videos.reshape(B * V, T, C, H, W)
```

PyTorch 的连续 reshape 保留 sample-major 顺序，所以送入 encoder 的样本依次是：

```text
b0v0, b0v1, b1v0, b1v1, ..., b31v0, b31v1
```

encoder 返回：

```text
video_embeddings: [B * V, N, D]
```

其中 `N` 是时空 token 数，`D` 是单视角 feature 维度。此时正确做法应当恢复 `[B,V]` 两个轴，再拼接同一样本的两个视角：

```python
video_embeddings = (
    video_embeddings.reshape(B, V, N, D)
    .permute(0, 2, 1, 3)
    .reshape(B, N, V * D)
)
```

但是上游训练源码实际执行：

```python
torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
```

`torch.chunk(..., chunks=2, dim=0)` 把长度为 `64` 的展平轴切成前后各 `32` 行，然后按相同行号沿 feature 维拼接。这个操作只有在展平顺序为
`[b0v0,b1v0,...,b31v0,b0v1,...,b31v1]` 的 view-major 布局下才正确；实际输入却是上述 sample-major 布局。

#### 5.3.2 batch size 32 时实际拼接了什么

训练配置的每设备 batch size 为 `B=32`。两个 chunk 分别包含：

```text
chunk 0 = b0v0,b0v1,...,b15v0,b15v1
chunk 1 = b16v0,b16v1,...,b31v0,b31v1
```

因此 predictor 各行实际接收到：

| predictor row | 拼接的 encoder features | 对应 action token row |
|---:|---|---:|
| 0 | sample 0 view 0 + sample 16 view 0 | sample 0 |
| 1 | sample 0 view 1 + sample 16 view 1 | sample 1 |
| 2 | sample 1 view 0 + sample 17 view 0 | sample 2 |
| 3 | sample 1 view 1 + sample 17 view 1 | sample 3 |
| ... | ... | ... |
| 31 | sample 15 view 1 + sample 31 view 1 | sample 31 |

这不是简单的“两个相机交换顺序”，而是：

1. 每一行拼接的是相隔半个 batch 的两个不同样本；
2. 拼接的通常是两个样本的同一视角，而不是同一样本的两个视角；
3. Qwen 产生的 latent-action tokens 没有做同样重排，仍然保持原始 sample row；
4. context 和 target 都在错误融合之后从同一行 video embedding 中切分，因此二者共享错误的混合身份，但 action 条件通常与该混合视觉行不匹配；
5. 模型看到的视觉条件取决于同一个 batch 中还有哪些样本以及它们的排列顺序。

只有 `B=1` 时，两个 chunk 恰好分别是 `b0v0` 和 `b0v1`，该写法才偶然正确。released checkpoint 使用的每设备 batch size 是 32，因此不能用 batch-size-1 的正确行为排除其训练影响。

#### 5.3.3 为什么这个错误会写入 checkpoint

错误融合后的 `video_embeddings` 同时用于构造：

```text
context = 每行混合 embedding 的 z0,z1,z2
target  = 同一混合 embedding 的 z1,z2,z3
action  = 原始 batch 行的 24 个 latent-action tokens
```

随后直接计算 world-model teacher-forcing loss。因此 predictor 的优化目标从训练开始就是这个 batch-dependent 映射；这不是评估阶段单独出现的格式问题，也不能只靠在推理代码中改一行来修复已经训练好的权重。

该问题直接影响 world-model predictor 训练路径。它不等价于“LIBERO policy 的实际 action head 一定错误”，因为 action head 是另一条路径；不过 world-model loss 若向共享 Qwen 部分回传梯度，也可能产生间接影响。当前审计能够确定的是 predictor 所接受的视觉/target/action 配对已经被破坏。

#### 5.3.4 数值证据

在 32 个窗口上直接复现两种融合：

| 融合方式 | MSE | L1 | cosine |
|---|---:|---:|---:|
| 正确 sample 内双视角 | 3.3022 | 1.2601 | 0.7897 |
| 原训练错误融合 | 3.2221 | 1.2398 | 0.7950 |
| 原训练错误融合 + shuffled action | 3.2223 | 1.2398 | 0.7949 |

正确融合与错误融合的 latent MSE 为 10.9984。checkpoint 在错误融合输入上略低的误差与训练源码行为一致；同时 shuffled action 仍几乎不改变结果。

这里“错误融合的损失略低”不能说明错误融合更合理。它只能说明 released checkpoint 更接近自己实际见过的错误训练分布。错误融合的 target 本身也混合了不同 episode，所以该损失不再对应某一条真实轨迹的未来 latent 预测质量。

#### 5.3.5 对现有评估应该怎样解释

需要同时区分两个问题：

| 评估方式 | 能回答什么 | 不能回答什么 |
|---|---|---|
| 正确的 sample 内双视角融合 | 当前 checkpoint 在真实单 episode、双视角输入上的可用性能 | 无法视为严格的训练分布内结果，因为 checkpoint 训练时没有稳定看到这种融合 |
| 精确复现 batch-32 错误融合 | checkpoint 是否复现其实际训练目标；错误对 loss 和 action sensitivity 的影响 | 结果依赖 batch 同伴和排列，不能解释为单条轨迹的物理未来预测 |

因此，面向 latent world model 实际能力的主评估仍应使用正确的 sample 内融合，但必须把它标记为“受上游训练 bug 影响的 checkpoint transfer evaluation”；错误融合评估只作为训练审计和 checkpoint sanity check，不能取代主结果。

### 5.4 latent-action 条件的语义与训练—评估一致性

Qwen 输出中存在两套不同槽位：

- 24 个 `<|action_i|>` hidden states 输入 `vj_predictor`；
- 32 个 `<|embodied_action|>` hidden states 输入 diffusion/flow-matching action head。

训练时，真实 action label 不会被编码后写入前一组 world-model tokens；它只用于 action head 的监督。推理时 action head 从随机高斯噪声开始生成 action chunk。

对同一 observation 连续进行三次在线推理：

- 24 个 latent action tokens 的重复推理最大差异为 0；
- normalized action chunk 的重复推理 MSE 分别达到约 `1.0e-4` 和 `2.5e-3`。

因此数据中的 latent tokens 与 action chunk 是“同一次 policy query 的共同输出”，但不是可逆或一一决定关系。这只是表示语义的边界，并不会使现有评估失效：上游训练 `vj_predictor` 时输入的就是这 24 个 tokens，本次评估保存并输入的也是相同 tokens，形状、顺序、来源和生成过程均一致。

本项目可以将它们定义为 VLA-JEPA 自身的 latent action 或 learned action-conditioning representation，而不要求它们等于实际执行动作的编码。需要单独检验的是 predictor 是否利用了这种表示中的样本相关信息；correct-vs-shuffled 对照回答的是这个经验问题，不能反过来把“不是实际动作的一一编码”当作评估无效的理由。

## 6. 旧评估协议审计

### 6.1 训练匹配条件缺失

旧条件的实际关系为：

| 条件 | context | context encoder | target encoder | 是否匹配训练 |
|---|---|---|---|---|
| F0 | C1 | strict causal | strict causal | 否：C1 + encoder 分布变化 |
| F1 | C3 | strict causal | strict causal | 否：encoder 分布变化 |
| F5 | C1 | original joint | strict causal | 否：C1 + target 分布变化 |

没有一个条件是 `joint C3 → joint shifted targets`。

在每个 LIBERO task 选择一个确定性窗口、共 130 个窗口的诊断中：

| 协议 | MSE | L1 | cosine |
|---|---:|---:|---:|
| 训练匹配：joint C3 → joint target | 3.4357 | 1.2879 | 0.7808 |
| strict causal C3 → strict target | 6.3488 | 1.7505 | 0.5667 |

joint 与 strict 的 context latent MSE 为 8.2880，target latent MSE 为 5.5129。这说明误差差异首先是 encoder 表示协议发生了大幅变化，不能全部归因于 predictor 的 transition 误差。

### 6.2 C1 是 checkpoint 的分布外调用

C1 将单独的 `z2,g2` 放到 predictor 的第 0 个时间位置；训练时该网络始终看到三个时间位置和 `[g0,g1,g2]`。虽然独立模块支持可变 context shape，released checkpoint 并未接受过 C1 训练。F0 可作为“冻结模型在新接口上的泛化测试”，但不应称为 checkpoint 的主训练内性能。

### 6.3 zero action 不能证明样本相关 action 被使用

训练匹配协议下：

| action 条件 | target MSE |
|---|---:|
| correct | 3.4357 |
| zero | 4.1229 |
| 跨 task shuffled | 3.4356 |

correct 与 shuffled action 本身并不相同：二者相对 RMS 差约 25.2%，平均 token cosine 约 0.823；但是预测之间的 MSE 只有 `1.12e-4`，target MSE 平均差仅约 `2.0e-5`。

zero token 远离 Qwen hidden-state 分布，所以 zero 导致性能下降只说明 predictor 对“是否存在正常范围的 token 激活”敏感。shuffled 控制才检查样本特定信息；该控制没有观察到实质作用。

### 6.4 AR-H3 的时间对齐错误

旧 F2 从 `z2(q+4,q+5)` 开始：

```text
g2(current query) → target (q+6,q+7)        # 对齐
g0(next query q+7) → target (q+8,q+9)       # 不对齐
g1(next query q+7) → target (q+10,q+11)     # 不对齐
```

下一 query 从 `q+7` 开始训练语义对应的 tubelet 为 `(q+7,q+8)→(q+9,q+10)`，与旧目标整体错开一帧。原因是 7 不能被 tubelet stride 2 整除。

runner 现已把 C1/F2 的默认排程改为同一 query 内的 `z0 --g0→ z1 --g1→ z2 --g2→ z3`，并默认拒绝无法严格构造的 C3 跨 query H3。只有显式传入 `--allow-legacy-misaligned-h3` 才会复现旧排程，并且该输出不得用于严格时间对齐结论。要预测当前 8 帧窗口之外的 H3，需要二选一：

1. 在同一 query 内从 `z0` 开始，用 `g0→g1→g2` 预测 `z1→z2→z3`；
2. 重新采集每 2 个控制帧对应的 world-model tokens，或直接条件化实际 action 序列。

### 6.5 LIBERO-90 是训练外 suite 且主导聚合

checkpoint 的 `libero_all` 训练 mix 包含 spatial、object、goal 和 LIBERO-10，不包含 LIBERO-90。正式半量数据的 650 条 rollout 中有 450 条来自 LIBERO-90，对应 1950 个 stage window 中的 1350 个。

因此总平均值主要反映训练外 LIBERO-90。suite 分层结果仍然有价值，但必须同时报告：

- 四个训练内 suite 的 macro average；
- LIBERO-90 单独结果；
- 五 suite 按 task 等权的 macro average；
- 不把按 rollout 数加权的总平均作为唯一结论。

## 7. 已实施的修复与保护

本次没有修改任何 rollout 或旧评估输出。已增加：

1. 全量 collection/action/video 审计器；
2. LIBERO 动作重放与逐帧状态/图像验证器；
3. 在线 latent-action 复算和重复 action 随机性验证器；
4. 训练匹配与 strict-causal 协议对照器；
5. batch-32 原训练双视角融合复现器；
6. 独立模块正确多视角融合的显式函数和回归测试；
7. runner 中 latent-world-model Git provenance 路径修复；
8. 时间对齐的同-query C1-H3，以及对历史跨-query/C3-H3 的拒绝保护。

这些修复不会重新训练 encoder 或 predictor，也不会伪造一个新 checkpoint 的能力。

修正后的 F2 已使用真实 encoder/checkpoint 完成 GPU smoke：1 个真实窗口、1 个 condition 正常写出，`processed=1`、`errors=[]`，输出明确记录 `h3_schedule=within_query`。修复后的 provenance 也能记录当前 `latent_world_model` commit，而不再返回空值。

## 8. 修正后的评估建议

今后应将结果分为三类，不再混为一个“主 MSE”：

| 类别 | 输入与目标 | 能回答的问题 |
|---|---|---|
| checkpoint sanity | joint C3 + 24 tokens → 同一次 joint encoding 的 z1..z3 | checkpoint 是否复现其训练目标 |
| strict causal transfer | 独立 causal C3 → strict future | 已训练 checkpoint 能否迁移到严格因果 latent 定义 |
| action specificity | correct vs same-task shuffle vs cross-task shuffle | 预测是否真正使用样本相关 action 信息 |

这里的 `action specificity` 是对 checkpoint 如何使用既有 latent action 的控制实验，而不是要求 latent action 必须重建实际执行动作。即使 shuffled 对照显示 predictor 利用较弱，同协议的 correct-action 预测误差仍然是有效测量；只是不能把误差改善自动归因于样本特定的 action 信息。

### 8.1 下一步：优先量化多视角训练错误

在不重新训练 encoder 或 latent world model 的当前约束下，建议按以下顺序推进：

1. **扩展 batch-32 审计。** 从当前 32-window smoke 扩展到按 suite/task 平衡的大样本，固定每设备等价 batch size 32，同时计算正确融合与精确 legacy 融合的 MSE、L1、cosine。
2. **做 batch 排列敏感性实验。** 对完全相同的窗口集合使用多个确定性 permutation 重新组成 batch；正确融合的每条样本输出应保持不变，而 legacy 融合的输入、目标和输出会随 batch 同伴改变。报告每条样本预测的跨排列方差。
3. **在两种融合下都做 latent-action 控制。** 分别比较 correct、same-task shuffle、cross-task shuffle，区分“action 本身利用较弱”和“视觉/action 配对被融合错误破坏”两种来源。
4. **按用途分开汇报。** 正确融合结果作为真实单 episode 的主评估，并标注 checkpoint 受到上游训练 bug 影响；legacy 结果只作为训练复现诊断，不与正常 world-model 的物理预测能力混为一谈。
5. **检查是否存在未受影响的可用权重。** 优先寻找使用 `B=1`、已修复融合代码或重新训练 predictor 的 checkpoint；如果不存在，则当前 released checkpoint 无法通过纯推理代码修复其训练历史。

如果以后允许修复模型，最小的根治方案是不重训 V-JEPA encoder：冻结 encoder 和 latent-action 生成器，改正多视角融合后只重新训练/微调 predictor，并重新执行上述三组对照。当前“不重新训练 latent world model”的约束下，这一步不执行。

## 9. 可复现验证命令

在仓库根目录、`VLA_JEPA` 环境中运行数据与模型审计：

```bash
python -m latent_world_model.evaluation.audit_collection \
  --dataset-root datasets/vla_jepa_libero130_v3 \
  --output evaluation_outputs/audit/collection_integrity.json

python -m latent_world_model.evaluation.audit_model_protocol \
  --index evaluation_outputs/stage0/index.jsonl \
  --encoder checkpoints/vjepa2-vitl-fpc64-256 \
  --checkpoint ../VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
  --one-per-task \
  --output evaluation_outputs/audit/model_protocol_130_tasks.json

python -m latent_world_model.evaluation.audit_training_view_fusion \
  --index evaluation_outputs/stage0/index.jsonl \
  --encoder checkpoints/vjepa2-vitl-fpc64-256 \
  --checkpoint ../VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
  --windows 32 \
  --output evaluation_outputs/audit/training_view_fusion_b32.json
```

LIBERO replay 需要 `libero` 环境和 EGL；在线 latent-action 复算需要临时启动原始 policy server。两者的 CLI 帮助中列出了完整参数。验证结束后应关闭 policy server，并确认 GPU 上没有遗留推理进程。

## 10. 能力边界

本次工作能够修正评估、证明数据对齐、揭示训练实现错误并给出更准确的解释，但不能在“不重新训练 latent world model 和 encoder”的约束下修复已经写入 checkpoint 的多视角训练错误。24 个 latent-action tokens 与上游训练条件保持一致，无需事后替换为实际 action 编码；对它们的限制应通过 action-specificity 对照来描述。

因此当前最严谨的结论是：

> 数据采集、latent-action 接口与独立模块提取没有导致“模型失效”；旧协议确实夸大了误差。当前最严重且已由源码和数值实验共同确认的问题，是 released checkpoint 的 world-model 训练使用了跨样本错位的双视角 batch 融合。现有 checkpoint 可继续按原 latent-action 条件评估，但真实单 episode 的结果必须被解释为受该上游训练错误影响的迁移性能，而不是无缺陷训练下的模型能力上限。
