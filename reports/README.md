# 实验报告与数据归档索引

本目录是 `latent_world_model` 实验产物的归档入口。根目录报告保持易于查阅；大体积数据和可再生成结果保留在本机，由哈希 manifest 建立完整性索引，不直接提交 Git。

## 报告层级

| 层级 | 文档 | 用途 |
|---|---|---|
| 最终 | [`FINAL_REPORT.md`](../FINAL_REPORT.md) | 第一次 strict-causal 实验、审计重点和第二次 joint-C3 实验的统一结论 |
| 实验一 | [`COMPREHENSIVE_REPORT.md`](../COMPREHENSIVE_REPORT.md) | 130×5 strict-causal 正式实验的完整历史结果 |
| 审计 | [`EXPERIMENT_AUDIT_REPORT.md`](../EXPERIMENT_AUDIT_REPORT.md) | 数据链路、协议、predictor parity 和上游实现审计 |
| 实验二 | [`SECOND_EXPERIMENT_REPORT.md`](../SECOND_EXPERIMENT_REPORT.md) | 130×5 原生 joint-C3 teacher-forcing 补充实验 |
| 旧摘要 | [`archive/FIRST_EXPERIMENT_SUMMARY.md`](archive/FIRST_EXPERIMENT_SUMMARY.md) | 第二次实验前的旧最终摘要，保留用于结果溯源 |

最终报告优先级最高；历史报告中的数值保留不改，但遇到协议解释冲突时，以审计报告和最终报告为准。

## 本地实验数据

| 数据类别 | 路径 | 管理方式 |
|---|---|---|
| rollout HDF5/MP4 | `datasets/vla_jepa_libero130_v3/` | 本地保留；schema 和统计由 tracked dataset manifest 记录 |
| 确定性窗口索引 | `evaluation_outputs/stage0/index.jsonl` | 本地保留；3900 rows；SHA-256 写入归档 manifest |
| 第一次正式结果 | `evaluation_outputs/formal_half/` | 11700 条 F0–F5 结果及 summary/plots |
| 深入分析 | `evaluation_outputs/deep_analysis/` | 第一次实验的 CSV、图表和分层统计 |
| 审计证据 | `evaluation_outputs/audit/` | collection/replay/parity/protocol/fusion JSON 证据 |
| 第二次正式结果 | `evaluation_outputs/joint_c3_full/` | 5850 条 J0–J2 结果及 summary/plots |

`formal_half_shard*`、`formal_shard*`、`stage0`、`stage1`、`stage1_supplemental` 和 `joint_c3_smoke` 是可恢复/溯源的中间运行目录，继续本地保留；最终统计以合并后的 `formal_half` 和 `joint_c3_full` 为准。归档 manifest 对所有本地结果目录记录文件数和逻辑字节数，并对两个正式结果目录、深入分析和审计证据逐文件记录 SHA-256。

## 保留策略

1. Git 保存代码、README、实验定义、人工解释报告和轻量 manifest。
2. 本地保存 HDF5、MP4、checkpoint、逐窗口 JSONL、NumPy embedding cache 和生成图表。
3. 不删除第一次实验或审计原始结果；第二次实验使用独立目录，避免覆盖。
4. `reports/ARTIFACT_MANIFEST.json` 对关键本地结果和报告记录路径、字节数及 SHA-256。
5. 任何归档移动或恢复后，都应重新运行严格验证：

```bash
PYTHONPATH=$PWD conda run -n VLA_JEPA \
  python -m latent_world_model.evaluation.archive_manifest \
  --root . --output reports/ARTIFACT_MANIFEST.json --strict
```

严格验证要求：数据集 1300 rollouts、索引 3900 windows、第一次实验每个 F 条件 1950 条、第二次实验每个 J 条件 1950 条、无重复 key，且主要误差指标全部 finite。
