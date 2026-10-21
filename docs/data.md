# 数据清单与审计

## Zenodo CC0 森林声景子集

[Sensing the Forest Natural Soundscape Dataset Part 2](https://zenodo.org/records/18911328) 是项目方发布的英国 Surrey 林地长录音档案。随档 `README.txt` 逐条列出 Freesound 原始声音链接，并标记为 Creative Commons 0；本次取得的 README MD5 为 `3e17209751cb59dbf10d2cadd78f00bb`，与发布页一致。完整 ZIP 的发布 MD5 为 `01feafa071d8dd2e5f1a60df362c6cc3`，大小为 20,698,813,016 字节。

`scripts/download_sensing_forest_subset.py` 用 HTTP Range 读取 ZIP 目录和选中成员的开头部分，以种子 `20261021` 从 742 条匹配的五分钟 WAV 中选 240 条。每条从第 5 秒起保存 10.5 秒 WAV；清单记录 Zenodo 页面、Freesound 原声 ID、作者、录制日期、许可证据页、文件 SHA-256 和截取区间。只获取子集，**未核对完整 ZIP 的 MD5 或每个原始五分钟录音的 CRC**。中断后重跑会跳过已存在且写入清单的片段。保存路径在 OneThingAI 的持久盘内，数据不进入 Git。

这批文件覆盖森林的不同时段，却都来自同一个地点与同一采集系统。自动描述仅标注森林和大致时段，不能臆测每条具体包含的鸟鸣、雨声或风声。它适合先做林地氛围的可循环生成实验；若要声称模型对其他环境泛化，需要补充场景不同且许可清晰的数据。正式训练前应随机听取声音，确认描述和音质，并排除人声、强瞬态、静音及设备故障。清单按录制日期分组，避免同一天的相近录音跨训练和测试集。

## Freesound / WavCaps 候选

环境录音并不自动构成可循环训练真值。优先选择稳定的雨、风、水流、机械底噪等连续声景，排除人声、旋律音乐、爆炸等强起止事件。候选可来自 [WavCaps Freesound 子集](https://github.com/XinhaoMei/WavCaps)；[freefield1010](https://arxiv.org/abs/1309.5275)适合声景筛查，但其标准片段为 10 秒，通常不足以满足本项目默认 10 秒加 0.5 秒重叠区的构造要求。

`loopscape.curate_wavcaps` 读取 WavCaps 的 `fsd_final.json`，按描述和标签中的环境声音关键词、元数据时长筛选，并查找本地以 Freesound 数字 ID 命名的音频。默认关键词只是候选召回，不能替代实际听音与挑选。输出 `review_template.csv`、`approved_manifest.jsonl` 和 `review_rejected.jsonl`。第一次运行时批准清单为空；将模板复制为独立的 `review.csv` 后逐条核查并填写。后续运行通过 `--review-csv` 导入已核查条目。模板文件每次都会重建，因此不要直接在模板里保存审核结论。若音频命名不符合 ID 规则，可在审核 CSV 的 `audio_path` 填写实际路径。

2026-10-21 从 WavCaps 官方 Hugging Face 文件取得 Freesound 元数据，SHA-256 为 `4499ada1a7a76666695d880feb298f557efb3aa8af3b57fd9b713c75319ff53f`，共 262,300 条。按现有关键词和时长宽筛约有 63,750 条；另在本地 `data/curated/review_shortlist.csv` 留下 120 条优先人工核查的候选，每类雨、风、流水、海浪各 30 条。短名单只用文字描述、30–600 秒时长、作者去重和明显非连续声排除规则筛出，未核验音频、许可证或上传者偏好，不能作为训练集。`data/curated/` 不进 Git；远程持久盘保留副本。

每条原始清单必须包含 `source_id`、`source_url`、`author`、`license`、`ai_training_allowed`、`permission_checked_at`、`path`、`caption`；`group_id` 推荐用上传者或原始录音标识。Freesound 来源还须写 `genai_preference=no_additional_preferences` 和 `permission_evidence_url`，并在审核日核实。证据链接应指向能够证实许可和上传者偏好的记录；只填音频页面而无法证实偏好，不算完成审核。`ai_training_allowed` 必须由人工核实后写入布尔值 `true`。目前预处理仅接受 `cc0`、`cc-by-4.0`、`cc-by-nc-4.0`；这只是项目数据过滤规则，不能替代对具体条款和上传者偏好的判断。[Freesound 官方 FAQ](https://freesound.org/help/faq/#can-my-sounds-be-used-to-train-generative-artificial-intelligence-gen-ai-models)说明了额外的生成式 AI 偏好及 Data Packs 的获取方式。CC-BY / CC-BY-NC 还涉及训练集披露，后续报告应附已用声音的来源及署名清单。

`loopscape.prepare` 将长源录音裁取 10.5 秒连续区间，用最后 0.5 秒平滑修正开头，得到 10 秒片段。这样在接缝处使用原录音里相邻的两段声音。处理后的清单保留来源和处理参数；拒绝原因写入 `data/processed/rejected.jsonl`。数据按 `group_id` 的 SHA-256 确定性分配到训练、验证和测试集。

预处理后需要人工抽查：

1. 在循环播放两次时有无明显重复、静音或淡化。
2. 文字描述与实际声音是否匹配。
3. 上传者和许可证信息是否在检查日期仍允许该学术用途。
4. 同一原始录音、改名副本和同一作者高度相似声音是否跨分组泄漏。

训练数据、生成音频、权重和缓存默认被 `.gitignore` 排除。若需要交付实验数据，应由用户另行通过符合许可证的方式共享数据清单及获取步骤。
