# V-JEPA Latent World Model

[English](README.md) | [中文](README_CN.md)

这是一个从 VLA-JEPA 中提取并独立封装的、由 latent action 条件化的
latent world model。该模块本身不依赖 Qwen、机器人数据集、训练器或
action policy：冻结的 V-JEPA2 encoder 将多视角视频编码为 latent patch
tokens，predictor 根据历史视觉 latent 和外部 latent action，输出时间对齐
的下一时刻 latent 状态序列。

## 数据集

本项目采集并验证的 VLA-JEPA × LIBERO rollout 数据集已经发布到
Hugging Face：

**[Monita108/VLA_JEPA-on-libero](https://huggingface.co/datasets/Monita108/VLA_JEPA-on-libero)**

数据集覆盖 LIBERO-SPATIAL、LIBERO-OBJECT、LIBERO-GOAL、LIBERO-90 和
LIBERO-10 的全部 130 个任务，共 1300 条双视角 rollout。每条记录包含：

- agent view 与 wrist view RGB；
- 机器人状态；
- 实际执行动作；
- policy query 的精确帧索引；
- `[N,24,2048]` VLA-JEPA latent-action tokens；
- `[N,7,7]` unnormalized action chunks；
- 对应的 MP4 视频和任务元数据。

本地目录结构、HDF5 schema、时间对齐方式和完整统计见
[`datasets/vla_jepa_libero130_v3/README.md`](datasets/vla_jepa_libero130_v3/README.md)。

## 安装

```bash
git clone git@github.com:108bpm/Test-of-VLA_JEPA_WAM.git latent_world_model
cd latent_world_model
pip install -e .
```

仓库不提交 encoder 权重。可以把本地 Hugging Face V-JEPA2 checkpoint
软链接到 `checkpoints/vjepa2-vitl-fpc64-256`，也可以在构造
`LatentWorldModel` 时直接传入 Hugging Face 仓库 ID 或本地路径。checkpoint
目录需要包含 Hugging Face 模型文件和 video processor 配置。

```bash
ln -s /absolute/path/to/vjepa2-vitl-fpc64-256 \
  checkpoints/vjepa2-vitl-fpc64-256
python example.py
```

## 基本 API

```python
from latent_world_model import LatentWorldModel, LatentWorldModelConfig

model = LatentWorldModel(
    encoder_path="checkpoints/vjepa2-vitl-fpc64-256",
    config=LatentWorldModelConfig(
        num_video_frames=8,
        num_views=2,
        latent_action_dim=2048,
        num_action_tokens_per_timestep=8,
    ),
)

# uint8/raw RGB video:
# [batch, views, frames, channels, height, width]
# 8 帧、tubelet=2 时得到 z0...z3；predictor 输入 z0...z2，预测 z1...z3。
predicted, target = model(videos, latent_actions)
loss = (predicted - target).abs().mean()
```

`latent_actions` 是模块面向其他项目的主要扩展接口：

```text
[B, context_steps * num_action_tokens_per_timestep, latent_action_dim]
```

默认 VLA-JEPA 兼容配置为 `[B,24,2048]`：三个 context latent steps，每个
step 使用 8 个 action tokens。第 `i` 组 tokens 条件化从 `z_i` 到
`z_(i+1)` 的 transition。外部 tokens 可以来自语言模型、policy network、
learned action tokenizer 或其他编码模块，但 shape 和时间顺序必须与配置一致。

对本项目使用的 VLA-JEPA checkpoint，24 个 world-model tokens 是 Qwen
在 `<|action_i|>` 槽位上的 hidden states。它们正是源训练和本项目评估共同
使用的 learned action-conditioning representation；不要求它们与 policy
随机生成的 7 步 action chunk 形成可逆的一一映射。相关语义边界和
correct/shuffled/zero 控制实验见 [`FINAL_REPORT.md`](FINAL_REPORT.md)。

如果其他项目已经生成 V-JEPA latent，可以跳过视频编码：

```python
predicted_next_latents = model.predict_from_latents(
    context_latents,
    latent_actions,
)
```

`context_latents` 的 shape 为：

```text
[B, context_steps * patches_per_frame, num_views * encoder_hidden]
```

在本项目的 V-JEPA2 ViT-L、256 px、双视角配置下，默认 context 与输出均为
`[B,768,2048]`，对应三个 256-token 时间块。

## 加载 VLA-JEPA predictor

可以只加载 checkpoint 中的 `vj_predictor.*`，而不实例化不相关的
Qwen/action-head 权重：

```python
model.load_predictor_checkpoint(
    "../VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/"
    "VLA-JEPA-LIBERO.pt"
)
```

V-JEPA2 encoder 保持冻结。独立 predictor 已与 VLA-JEPA 源实现做过数值
parity 验证，同一输入下 max/mean absolute difference 均为 0。

## Future-latent 评估

完整评估入口和可恢复运行命令见 [`EVALUATION.md`](EVALUATION.md)，覆盖：

- strict-causal 与 original-joint latent 构造；
- C1/C3 视觉历史；
- H1 与多步排程；
- correct、shuffled、zero latent-action 控制；
- 分片运行、合并、bootstrap 统计和图表生成；
- 数据、模型协议和多视角融合审计。

唯一的最终结果报告是 [`FINAL_REPORT.md`](FINAL_REPORT.md)。该报告已经
完整整合数据采集与验证、模型接口、实验设计、strict-causal 正式结果、
上游实现审计、joint-C3 正式结果、局限和最终结论，不需要再配合其他
历史报告阅读。

也可以直接调用表示空间指标：

```python
from latent_world_model import evaluate_latent_prediction

predicted, target = model(videos, latent_actions)
metrics = evaluate_latent_prediction(predicted, target)
# l1, mse, mean_token_cosine, retrieval_accuracy
```

`retrieval_accuracy` 检查每个 predicted future representation 是否最接近
batch 中自己的真实 future，可辅助发现表征塌缩。使用该指标时 batch 至少
需要两个样本。

## 实验框架与产物

[`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) 提供不包含结果的实验
框架，描述采集逻辑、控制变量、正式条件、阶段漏斗和统计规则。

大体积 HDF5、MP4、checkpoint、逐窗口 JSONL 和 NumPy cache 不提交 Git。
本地实验产物由 [`reports/ARTIFACT_MANIFEST.json`](reports/ARTIFACT_MANIFEST.json)
记录文件大小、行数和 SHA-256，以便验证数据与正式结果完整性。

## 关于像素重建

V-JEPA2 没有与当前 encoder 配套的像素 decoder。已检查的上游 `vjepa2`
实现只包含 encoder 和 latent predictor，其训练目标是冻结 teacher encoder
的 patch features，而不是 RGB 像素。因此现有 V-JEPA2 encoder checkpoint
无法直接把 latent 解码为视频帧。

本项目首先在 representation space 中使用 MSE、L1、cosine 和 retrieval
等指标。若需要 PSNR、SSIM 或 LPIPS，必须额外训练 latent-to-pixel decoder，
并严格匹配以下配置：

- `facebook/vjepa2-vitl-fpc64-256`；
- 256 px 输入；
- patch size 16；
- tubelet size 2；
- 本项目的多视角融合约定。

该 decoder 需要独立 checkpoint，不能从 V-JEPA2 encoder 权重推导得到。

## 哪些参数可训练

`model.encoder` 默认被冻结、保持 evaluation mode，并始终在
`torch.no_grad()` 下运行。除非接入项目明确改变这一行为，否则只训练
`model.predictor`。`model.loss(...)` 使用与 VLA-JEPA 一致的 L1 latent
prediction objective。

本仓库的正式评估没有重新训练或微调 encoder、predictor 或 latent-action
生成模块。

## 来源与许可说明

`predictor.py`、`vj2_modules.py` 和 `vj2_tensors.py` 提取自 VLA-JEPA 的
action-conditioned V-JEPA predictor；适用的文件保留了 Meta/V-JEPA 上游
许可声明。
