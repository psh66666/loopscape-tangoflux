# 实验矩阵与记录模板

以下是要执行的实验，**不是已取得的结果**。所有组使用同一个数据清单、TangoFlux 权重 revision、10 秒片段、LoRA rank 8、优化器更新步数和生成提示/种子。正式评测的提示集应大于 README 中的 8 条演示样例，并在调参前固定。

| 组别 | 训练 | 运行方式 | 回答的问题 |
|---|---|---|---|
| Z | 无 | `loopscape.generate` 不传 `--run-dir` | 原检查点的接缝现状 |
| B | 普通 LoRA | `configs/baseline.yaml` | 同数据微调本身的收益 |
| B+D | 与 B 相同 | 生成时加 `--declick` | 固定长度推理去点击能达到什么程度 |
| B+X | 与 B 相同 | 生成时加 `--crossfade` | 固定长度双侧交叉淡化的收益和音质代价 |
| C | 循环模块 + 边界损失 | `configs/circular.yaml` | 完整方法是否改善听感 |
| C-L | 循环模块，无边界损失 | `configs/circular.yaml --boundary-weight 0 --output-dir runs/circular_no_loss` | 边界损失贡献 |
| B+L | 普通 LoRA + 边界损失 | `configs/baseline.yaml --boundary-weight 0.05 --output-dir runs/baseline_with_loss` | 仅损失能否解释收益 |
| P+L | 普通填充模块 + 边界损失 | `configs/plain_adapter.yaml` | 相同额外参数量下，跨接缝上下文是否有价值 |

训练命令例：

```bash
python -m loopscape.train --config configs/circular.yaml --boundary-weight 0 --output-dir runs/circular_no_loss
python -m loopscape.train --config configs/baseline.yaml --boundary-weight 0.05 --output-dir runs/baseline_with_loss
python -m loopscape.train --config configs/plain_adapter.yaml
```

每组应记录：源码提交、上游与权重 revision、数据清单 SHA-256、许可审查日期、训练配置、随机种子、GPU 型号、峰值显存、每步耗时、总 GPU 小时、模型差量与固定提示 WAV。评价按同文件名成对比较接缝指标，报告质量 CLAP/FAD、RMS、峰值和边缘/中心响度。FAD 样本过少时不作强结论。双循环盲听需随机打乱模型标签并记录听者判断。

若 C 的自动接缝指标更好但双循环听感无收益，或 CLAP/FAD/响度显著恶化，应报告为未达到预期，不选择性隐藏失败样本。
