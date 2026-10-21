# Loopscape：可循环环境声景生成

基于 [TangoFlux（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f475c4cf43826411ae61c94bab360a0d-Abstract-Conference.html) 的课程研究代码库。目标是微调文本到环境音效模型，使雨、风、水流等连续背景声在首尾拼接并重复播放时更自然。本仓库包含普通 LoRA 微调、循环上下文模块、边界损失、参数量匹配的控制组、数据处理、生成和评测入口。

**当前状态：已部署到 OneThingAI 的 RTX 4090 实例，并预取固定版本权重；尚无经许可审核的训练数据，因此没有开始训练，也没有实验结果。** 按用户约定，代码由用户运行测试；部署阶段只做语法与环境检查。

## 研究问题

TangoFlux 本身使用 flow matching 预测音频潜变量。改进版在加噪潜变量上通过 circular padding 交换首尾局部上下文，同时对预测的干净潜变量施加边界连续性损失。比较组使用相同的主模型检查点、数据、LoRA 配置、随机种子、更新步数和采样提示。`plain_adapter` 与改进版有相同的卷积参数量，但不跨接缝交换上下文。方法细节见[设计规格](docs/superpowers/specs/2026-10-20-loopscape-design.md)。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `third_party/TangoFlux/` | 固定到上游提交的源码快照与原许可证 |
| `loopscape/prepare.py` | 经许可审计的环境录音清单、循环片段构造与数据切分 |
| `loopscape/curate_wavcaps.py` | 从 WavCaps Freesound 元数据生成候选与逐条许可核查表 |
| `loopscape/probe.py` | 已知循环样本的 VAE 编码—解码回环诊断 |
| `loopscape/train.py` | 单卡 LoRA 基线、循环改进与消融训练 |
| `loopscape/generate.py` | 相同提示和种子的 WAV 生成；可选推理后去点击或双侧交叉淡化控制 |
| `loopscape/evaluate.py` | 接缝、响度、CLAP 和 FAD 评测 |
| `configs/` | 固定训练配置及演示提示 |
| `docs/` | 数据、上游来源、租机要求及实验记录 |

## 安装（Linux + NVIDIA CUDA）

建议 Python 3.11、PyTorch 2.4 系列与兼容的 CUDA 驱动。命令在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
uv pip install -r requirements-training.txt
uv pip install --no-deps -e third_party/TangoFlux -e .
uv pip install -r requirements-quality.txt
```

训练所需依赖独立列出，避免安装上游演示界面的 `gradio` 等非训练组件。上游源码及其声明的版本保持原样。

首次使用会从 Hugging Face 下载固定 revision 的 TangoFlux、VAE 和 FLAN-T5-Large 权重。具体版本见[上游来源](docs/upstream.md)。训练前确认各模型和数据许可证允许你的用途；本项目限非商业研究。

OneThingAI 实例的实际挂载路径、环境激活和重建步骤见[部署记录](docs/onethingai.md)。

## 数据准备

可先从 [WavCaps 官方数据集](https://huggingface.co/datasets/cvssp/WavCaps)取得 `json_files/FreeSound/fsd_final.json`，放在 `data/raw/`。已核对的文件 SHA-256 为 `4499ada1a7a76666695d880feb298f557efb3aa8af3b57fd9b713c75319ff53f`。候选工具依据描述和标签挑出时长至少 10.5 秒的环境录音，并按 Freesound ID 匹配本地 `123456.wav` 或 `123456__*.wav` 等文件。宽筛选约有 6.4 万条，不能逐条直接用于训练。`data/curated/review_shortlist.csv` 是从中选出的 120 条雨、风、流水、海浪元数据候选；它们尚未通过许可或听音审查。工具不会推断许可：

```bash
python -m loopscape.curate_wavcaps --metadata data/raw/fsd_final.json --audio-root data/raw/wavcaps_freesound --output-dir data/curated
cp data/curated/review_shortlist.csv data/curated/review.csv
```

逐条查看 Freesound 原页面、许可条款和上传者的生成式 AI 训练偏好后，在 `review.csv` 填写 `license`、`genai_preference`、`permission_evidence_url`、`ai_training_allowed`、`permission_checked_at`（ISO 日期）；来源链接、作者和描述也要核对。当前实验只接纳经证实为 `no_additional_preferences` 的上传者，且必须有可追溯的 HTTPS 核查证据。WavCaps 旧元数据不含这项偏好。若需批量获取，按 [Freesound 官方说明](https://freesound.org/help/faq/#i-want-to-download-large-amounts-of-sounds-from-freesound-to-train-an-artificial-intelligence-ai-model-can-i-do-that)申请 Data Packs，并使用其中的偏好字段过滤；不要把 WavCaps 的可下载性当作训练许可。只会将明确填 `true`、许可证在项目允许列表且本地音频存在的行导入：

```bash
python -m loopscape.curate_wavcaps --metadata data/raw/fsd_final.json --audio-root data/raw/wavcaps_freesound --output-dir data/curated --review-csv data/curated/review.csv
python -m loopscape.prepare --input data/curated/approved_manifest.jsonl --output-dir data/processed --duration 10 --overlap 0.5
```

也可自行将**已人工审查训练许可**的原始录音及文字描述写入 `data/raw/manifest.jsonl`，每行示例：


```json
{"source_id":"freesound-123456","group_id":"uploader-name","source_url":"https://freesound.org/s/123456/","author":"uploader-name","license":"cc-by-4.0","genai_preference":"no_additional_preferences","permission_evidence_url":"https://example.org/verified-permission-record-123456","ai_training_allowed":true,"permission_checked_at":"2026-10-21","path":"data/raw/123456.wav","caption":"Steady rain falling on a roof"}
```

示例只展示清单格式，不代表该声音已获准用于训练。`group_id` 建议使用上传者或同一原始录音的共同标识，以免同源片段跨训练/测试集。仅接受明确标记许可与检查日期的条目。原录音至少需 10.5 秒；预处理从连续源录音中构造 10 秒循环片段，并输出接受/拒绝清单。自建清单对应命令：

```bash
python -m loopscape.prepare --input data/raw/manifest.jsonl --output-dir data/processed --duration 10 --overlap 0.5
```

正式训练前先人工听取样本、核对描述和许可证，再检查 VAE 是否保持接缝：

```bash
python -m loopscape.probe --config configs/baseline.yaml --audio data/processed/audio/你的样本.wav --output-dir outputs/vae_probe
```

详见[数据规范](docs/data.md)。

## 显存/速度探针和训练

先在租用机器上运行两步，记录日志中的峰值显存和耗时；这是真正租机规格和总成本估算的依据：

```bash
python -m loopscape.train --config configs/baseline.yaml --max-steps 2 --output-dir runs/probe_baseline
python -m loopscape.train --config configs/circular.yaml --max-steps 2 --output-dir runs/probe_circular
```

诊断通过后，分别训练相同的 1000 个优化器更新步：

```bash
python -m loopscape.train --config configs/baseline.yaml
python -m loopscape.train --config configs/circular.yaml
```

中断后加 `--resume`，从 `run.json`、`adapter.safetensors` 和 `state.pt` 恢复。训练日志 `train.jsonl` 记录 loss、耗时、峰值显存；检查点保存上游版本、配置和数据清单 SHA-256。基线和改进均实际更新 LoRA 参数；改进版另更新循环模块。消融命令见[实验矩阵](docs/experiments.md)。

## 生成与评测

`configs/eval_prompts.jsonl` 只有 8 个演示提示；正式报告须建立更大的固定、未参与调参的提示集。下面的命令在所有组使用同一提示、种子和采样设置：

```bash
python -m loopscape.generate --config configs/baseline.yaml --run-dir runs/baseline --output-dir outputs/baseline --prompts configs/eval_prompts.jsonl --seed 1200
python -m loopscape.generate --config configs/circular.yaml --run-dir runs/circular --output-dir outputs/circular --prompts configs/eval_prompts.jsonl --seed 1200
python -m loopscape.generate --config configs/baseline.yaml --run-dir runs/baseline --output-dir outputs/baseline_declick --prompts configs/eval_prompts.jsonl --seed 1200 --declick
python -m loopscape.generate --config configs/baseline.yaml --run-dir runs/baseline --output-dir outputs/baseline_crossfade --prompts configs/eval_prompts.jsonl --seed 1200 --crossfade
python -m loopscape.evaluate --generated-dir outputs/circular --compare-dir outputs/baseline
python -m loopscape.evaluate --generated-dir outputs/circular --reference-manifest data/processed/manifest.jsonl --quality
```

`--declick` 是轻量去点击，`--crossfade` 在首尾两侧做固定长度的交叉淡化；它们都是推理后处理，不能视为训练改进。自动指标必须结合双循环盲听与质量/响度检查；不能只报告首尾一个采样点的跳变。[实验规范](docs/experiments.md)列出所需对照与失败判据。

## 许可和来源

上游 TangoFlux 源码、模型及训练数据有各自限制，详见[上游来源](docs/upstream.md)及 `third_party/TangoFlux/` 中的原始许可证。WavCaps 仅允许学术用途，Freesound 的逐条 Creative Commons 许可及上传者生成式 AI 训练偏好必须另行检查。仓库不包含训练音频、模型权重或实验结果。
