# Loopscape 实现计划

## 目标

交付可在单张租用 GPU 上执行的 TangoFlux 循环环境声实验代码库及 GitHub 仓库。基线与改进版都能从同一上游检查点微调、生成、评测。此次只执行语法和静态验证，不代替用户运行训练实验。

## 架构与技术栈

Python 3.11、PyTorch 2.4、Diffusers 0.30、Transformers 4.44、Accelerate 0.34。固定上游 TangoFlux 提交；本仓库的 `loopscape` Python 包调用上游模型与 VAE。输入数据是 JSONL 清单和本地 WAV/FLAC，输出为 adapter 权重、配置、日志、WAV 与指标 JSON。用 `argparse` 提供命令行，`yaml` 定义实验配置。

## 执行步骤

1. 固定上游源码：以 `164420faa2b3f8a28bb013ccde7006422a616fc7` 为初始候选 revision，将原始代码和许可证放入 `third_party/TangoFlux/`，在 `docs/upstream.md` 记录下载来源与校验值。命令：`git rev-parse HEAD` 与 `python3 -m compileall -q third_party/TangoFlux`。预期：固定来源可查，Python 编译退出码 0。
2. 建立包和配置：新增 `pyproject.toml`、`loopscape/__init__.py`、`loopscape/config.py`、`configs/baseline.yaml`、`configs/circular.yaml`、`.gitignore`。检查配置中的目标长度、checkpoint revision 和数据路径均显式给出；命令：`python3 -m compileall -q loopscape`。
3. 建立数据链：新增 `loopscape/manifest.py`、`loopscape/prepare.py`、`loopscape/audio.py`，读取带来源/许可证/训练偏好的原始清单，筛选并制作可循环短片段，按源 ID 分组切分，输出可追踪的 JSONL。命令：`python3 -m compileall -q loopscape`。
4. 建立 VAE 回环探针：新增 `loopscape/probe.py`，读取可循环 WAV，经上游 VAE 重建，输出首尾指标和重建文件。程序应在数据或依赖缺失时给出明确错误。命令：`python3 -m compileall -q loopscape`。
5. 建立模型改动：新增 `loopscape/adapters.py` 和 `loopscape/losses.py`，实现 Transformer 线性层 LoRA、可循环的局部潜变量模块，以及估计干净潜变量的边界损失。普通基线和改进版共享 LoRA 位置与秩。命令：`python3 -m compileall -q loopscape`。
6. 建立训练与检查点：新增 `loopscape/train.py`、`loopscape/checkpoint.py`，对两版使用同一数据清单和更新步数，保存 adapter、配置、来源 SHA、随机种子和进度。命令：`python3 -m compileall -q loopscape`。
7. 建立生成与评测：新增 `loopscape/generate.py`、`loopscape/evaluate.py`，固定提示/种子生成同输入样例，计算接缝、RMS 和分布/文本指标可用时的结果；`configs/eval_prompts.jsonl` 存固定示例提示。命令：`python3 -m compileall -q loopscape`。
8. 写运行文档：`README.md`、`docs/data.md`、`docs/rental-gpu.md`、`docs/experiments.md`；明确上游与数据许可、从安装到训练/推理/评测的命令、GPU 显存和计时探针，以及尚未实测的限制。
9. 静态验证后提交并推送：运行 `python3 -m compileall -q loopscape third_party/TangoFlux`、`git diff --check`、`git status --short`；核验 `gh api user --jq .login` 为 `psh66666`，创建并推送该账户下的新仓库，再核验远端默认分支与文件。

以上步骤不包含训练结果验收；训练、生成音质和运行时兼容性由用户取得租机后验证。
