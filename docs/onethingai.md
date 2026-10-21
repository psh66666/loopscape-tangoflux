# OneThingAI 实例部署记录

2026-10-21 检查：实例配备 RTX 4090（24 GiB 显存）、96 GiB 系统内存，系统为 Ubuntu 22.04。已挂载 220 GiB 的云存储 `/root/onethingai-fs`。代码路径为 `/root/onethingai-fs/projects/loopscape-tangoflux`；Hugging Face、Torch 和通用模型缓存放在 `/root/onethingai-fs/cache/`。云存储适合保存代码、权重和训练输出，但小文件 I/O 较慢。[OneThingAI 云存储说明](https://docs.onethingai.com/a0d48/ba014/1de27)确认它跨实例保留，且超过免费额度会计费。

TangoFlux 与 FLAN-T5-Large 的固定 revision 已预取到 Hugging Face 持久缓存，约占 6.8 GiB。当前实例已安装 Python 3.11、PyTorch 2.4.0、torchaudio 2.4.0 及 CLAP/FAD 评测依赖。已在持久盘 `data/raw/sensing_forest/` 开始提取 Zenodo CC0 森林声景子集；查看 `subset.log` 和 `manifest.jsonl` 可确认进度。数据仍需听音与描述核查。

Python 3.11 环境放在实例本地 `/root/miniconda3/envs/loopscape-py311` 以避免在云存储中写入大量小文件。**实例释放或更换后，环境可能需要重建；代码、缓存、数据和输出应留在云存储。**当前仓库的 `data/`、`runs/` 和 `outputs/` 位于云存储路径下，训练时不要把这些目录改到 `/tmp` 或容器临时层。

登录实例后激活环境：

```bash
source /root/onethingai-fs/projects/loopscape-tangoflux/scripts/activate-onethingai.sh
python --version
```

若环境不存在，先重建：

```bash
conda create -y -p /root/miniconda3/envs/loopscape-py311 python=3.11 pip
source /root/onethingai-fs/projects/loopscape-tangoflux/scripts/activate-onethingai.sh
python -m pip install uv
uv pip install --system-certs --python /root/miniconda3/envs/loopscape-py311/bin/python -r requirements-training.txt
uv pip install --system-certs --python /root/miniconda3/envs/loopscape-py311/bin/python --no-deps -e third_party/TangoFlux -e .
uv pip install --system-certs --python /root/miniconda3/envs/loopscape-py311/bin/python -r requirements-quality.txt
```

实例的阿里云 PyPI 镜像未提供上游演示界面的 `gradio`，因此按仓库列出的训练依赖安装；`uv --system-certs` 用于适配实例代理的系统证书。上述安装从 PyPI 官方索引解析固定版本。

正式训练前依照 [README](../README.md) 放入已审查的数据，先运行 VAE 回环诊断和两步显存探针。4090 有 24 GiB 显存，但本项目尚无实测峰值，不能据此保证完整训练不会 OOM。
